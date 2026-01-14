"""Self-play training loop for Connect-4 AlphaZero-style model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from connect4.az.game import apply_move, initial_state, legal_moves, terminal_value
from connect4.az.mcts import PUCTMCTS
from connect4.az.model import ModelConfig, PolicyValueNet, pick_device, save_model
from connect4.engine import Connect4Config


@dataclass
class Example:
    board: np.ndarray  # canonical board, shape (H, W)
    pi: np.ndarray  # policy target, shape (W,)
    ply: int
    z: float = 0.0


def _winner_from_start(ply: int, tv: float) -> int:
    """
    Return winner sign (+1/-1/0) relative to the player who moved at ply=0.

    The terminal value tv is from the perspective of the player to move at the
    terminal state. If tv == -1, the player who just moved (ply-1) won.
    """

    if tv == 0.0:
        return 0
    # Last move was by ply-1. If ply is odd, starting player just moved.
    return +1 if (ply % 2 == 1) else -1


def _player_to_move_sign_from_start(ply: int) -> int:
    """At even ply, it is the starting player's turn; at odd ply, the second player's."""

    return +1 if (ply % 2 == 0) else -1


def _temperature_for_ply(ply: int, *, temp_moves: int) -> float:
    return 1.0 if ply < temp_moves else 1e-8


def play_self_game(
    *,
    cfg: Connect4Config,
    model: PolicyValueNet,
    device: torch.device,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temp_moves: int,
    seed: int,
) -> Tuple[List[Example], int]:
    rng = np.random.default_rng(seed)
    s = initial_state(cfg)
    examples: List[Example] = []

    while True:
        tv = terminal_value(cfg, s)
        if tv is not None:
            # Convert the final winner into z targets for every recorded state.
            winner = _winner_from_start(s.ply, tv)
            for ex in examples:
                z = float(winner * _player_to_move_sign_from_start(ex.ply))
                ex.z = z
            return examples, winner

        mcts = PUCTMCTS(
            cfg=cfg,
            model=model,
            device=device,
            sims=sims,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            seed=seed,
        )
        mcts.run(s)

        temperature = _temperature_for_ply(s.ply, temp_moves=temp_moves)
        pi = mcts.root_policy(s, temperature=temperature)
        if float(pi.sum()) <= 0.0:
            legal = legal_moves(cfg, s)
            pi = np.zeros((cfg.width,), dtype=np.float32)
            pi[legal] = 1.0 / len(legal)

        # Store the canonical board and MCTS-derived policy target.
        examples.append(Example(board=s.board.copy(), pi=pi, ply=s.ply))

        # Sample a move from pi during self-play to encourage exploration.
        col = int(rng.choice(cfg.width, p=pi))
        s = apply_move(cfg, s, col)


def _prepare_batch(cfg: Connect4Config, batch: List[Example]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = np.stack([ex.board for ex in batch], axis=0)
    cur = (boards == 1).astype(np.float32)
    opp = (boards == -1).astype(np.float32)
    x = np.stack([cur, opp], axis=1)  # (B, 2, H, W)
    pi = np.stack([ex.pi for ex in batch], axis=0).astype(np.float32)
    z = np.array([ex.z for ex in batch], dtype=np.float32)

    return torch.from_numpy(x), torch.from_numpy(pi), torch.from_numpy(z)


def train(
    *,
    cfg: Connect4Config,
    model: PolicyValueNet,
    device: torch.device,
    iters: int,
    games_per_iter: int,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temp_moves: int,
    replay_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    seed: int,
    out_dir: Path,
    checkpoint_every_games: int,
) -> None:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay: List[Example] = []
    total_games = 0

    for it in range(iters):
        new_examples: List[Example] = []
        wins = {+1: 0, 0: 0, -1: 0}

        # Self-play uses the network in inference mode.
        model.eval()
        for g in trange(games_per_iter, desc=f"self-play iter {it}", leave=False):
            examples, winner = play_self_game(
                cfg=cfg,
                model=model,
                device=device,
                sims=sims,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                temp_moves=temp_moves,
                seed=seed + g + it * games_per_iter,
            )
            new_examples.extend(examples)
            wins[winner] += 1

            total_games += 1
            if checkpoint_every_games > 0 and total_games % checkpoint_every_games == 0:
                ckpt = out_dir / f"connect4_az_games_{total_games}.pt"
                save_model(ckpt, model)

        replay.extend(new_examples)
        if len(replay) > replay_size:
            replay = replay[-replay_size:]

        if len(replay) < batch_size:
            continue

        # Training uses standard train mode.
        model.train()
        # Train on random mini-batches sampled from the replay buffer.
        for _ in trange(train_steps, desc=f"train iter {it}", leave=False):
            batch = [replay[int(rng.integers(len(replay)))] for _ in range(batch_size)]
            x, pi, z = _prepare_batch(cfg, batch)
            x = x.to(device)
            pi = pi.to(device)
            z = z.to(device)

            logits, values = model(x)
            log_probs = F.log_softmax(logits, dim=-1)

            # Cross-entropy between target policy and predicted policy.
            policy_loss = -(pi * log_probs).sum(dim=1).mean()

            # MSE between predicted value and game outcome.
            value_loss = F.mse_loss(values, z)

            loss = policy_loss + value_loss

            # Standard SGD step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"iter={it} games={games_per_iter} wins(+1/0/-1)={wins[+1]}/{wins[0]}/{wins[-1]} "
            f"replay={len(replay)}"
        )

    save_model(out_dir / "connect4_az_latest.pt", model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect-4 AlphaZero-style self-play training")
    parser.add_argument("--iters", type=int, default=10, help="training iterations")
    parser.add_argument("--games-per-iter", type=int, default=10, help="self-play games per iter")
    parser.add_argument("--sims", type=int, default=200, help="MCTS sims per move")
    parser.add_argument("--cpuct", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha")
    parser.add_argument("--dirichlet-eps", type=float, default=0.25, help="Dirichlet epsilon")
    parser.add_argument("--temp-moves", type=int, default=10, help="plies with temperature=1")
    parser.add_argument("--replay-size", type=int, default=5000, help="replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
    parser.add_argument("--train-steps", type=int, default=50, help="gradient steps per iter")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--out-dir", type=Path, default=Path("connect4_az_models"), help="output dir")
    parser.add_argument(
        "--checkpoint-every-games",
        type=int,
        default=50,
        help="save a checkpoint every N games (0 disables)",
    )
    parser.add_argument("--channels", type=int, default=64, help="model trunk channels")
    args = parser.parse_args()

    cfg = Connect4Config()
    cfg.validate()
    device = pick_device(args.device)

    model_cfg = ModelConfig(channels=args.channels)
    model = PolicyValueNet(cfg=cfg, model_cfg=model_cfg).to(device)

    train(
        cfg=cfg,
        model=model,
        device=device,
        iters=args.iters,
        games_per_iter=args.games_per_iter,
        sims=args.sims,
        c_puct=args.cpuct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        temp_moves=args.temp_moves,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out_dir,
        checkpoint_every_games=args.checkpoint_every_games,
    )


if __name__ == "__main__":
    main()
