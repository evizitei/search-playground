"""Self-play training loop for Connect-4 AlphaZero-style model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from connect4.az.game import apply_move, initial_state, legal_moves, terminal_value
from connect4.az.bridge import to_canonical_state
from connect4.az.mcts import PUCTMCTS
from connect4.az.model import ModelConfig, PolicyValueNet, load_model, pick_device, save_model
from connect4.agents.alphabeta import AlphaBetaAgent
from connect4.agents.random_agent import RandomAgent
from connect4.engine import (
    Connect4Config,
    GameState,
    apply_move as engine_apply_move,
    initial_state as engine_initial_state,
    terminal_result,
)
from connect4.cli import render_board


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


def _policy_entropy(pi: np.ndarray) -> float:
    # Shannon entropy of the policy distribution (higher = more diffuse).
    p = np.clip(pi.astype(np.float32), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _select_move_from_pi(pi: np.ndarray, *, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-8:
        return int(pi.argmax())
    probs = np.power(pi, 1.0 / max(temperature, 1e-8))
    s = float(probs.sum())
    if s <= 0.0:
        return int(pi.argmax())
    probs = probs / s
    return int(rng.choice(len(probs), p=probs))


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
) -> Tuple[List[Example], int, float, float]:
    rng = np.random.default_rng(seed)
    s = initial_state(cfg)
    examples: List[Example] = []
    entropies: List[float] = []
    temp_entropies: List[float] = []

    while True:
        tv = terminal_value(cfg, s)
        if tv is not None:
            # Convert the final winner into z targets for every recorded state.
            winner = _winner_from_start(s.ply, tv)
            for ex in examples:
                z = float(winner * _player_to_move_sign_from_start(ex.ply))
                ex.z = z
            avg_entropy = float(np.mean(entropies)) if entropies else 0.0
            avg_temp_entropy = float(np.mean(temp_entropies)) if temp_entropies else 0.0
            return examples, winner, avg_entropy, avg_temp_entropy

        mcts = PUCTMCTS(
            cfg=cfg,
            model=model,
            device=device,
            sims=sims,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            seed=int(rng.integers(1_000_000)),
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
        ent = _policy_entropy(pi)
        entropies.append(ent)
        if s.ply < temp_moves:
            temp_entropies.append(ent)

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


def _az_select_move(
    *,
    cfg: Connect4Config,
    model: PolicyValueNet,
    device: torch.device,
    s: GameState,
    sims: int,
    c_puct: float,
    seed: int,
    temperature: float,
    debug: bool,
) -> int:
    # Convert engine state to canonical form for MCTS.
    canonical = to_canonical_state(s)
    mcts = PUCTMCTS(
        cfg=cfg,
        model=model,
        device=device,
        sims=sims,
        c_puct=c_puct,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
        seed=seed,
    )
    mcts.run(canonical)
    pi = mcts.root_policy(canonical, temperature=1e-8)
    if debug:
        prior = mcts.root_prior(canonical)
        stats = mcts.root_stats(canonical)
        if stats is not None and int(stats.N.sum()) > 0:
            n = stats.N.astype(np.float32)
            pi_counts = n / float(n.sum())
        else:
            pi_counts = np.zeros_like(prior)
        print(
            f"eval_debug prior={np.round(prior, 3)} entP={_policy_entropy(prior):.3f} "
            f"N={stats.N.tolist() if stats is not None else []} "
            f"piN={np.round(pi_counts, 3)} entN={_policy_entropy(pi_counts):.3f} "
            f"pi={np.round(pi, 3)} ent={_policy_entropy(pi):.3f}"
        )
    rng = np.random.default_rng(seed)
    return _select_move_from_pi(pi, temperature=temperature, rng=rng)


def evaluate_vs_alphabeta(
    *,
    cfg: Connect4Config,
    model: PolicyValueNet,
    device: torch.device,
    games: int,
    ab_depth: int,
    ab_nodes: Optional[int],
    sims: int,
    c_puct: float,
    seed: int,
    temperature: float,
    debug: bool,
) -> Dict[str, int]:
    """
    Evaluate the current model against a low-budget alpha-beta opponent.

    We alternate who starts each game to reduce first-player bias and
    report results from the model's perspective (wins/draw/loss).
    """

    rng = np.random.default_rng(seed)
    ab_agent = AlphaBetaAgent("AlphaBeta Eval", max_depth=ab_depth, max_nodes=ab_nodes)

    wins = 0
    draws = 0
    losses = 0

    model.eval()
    for g in range(games):
        az_starts = (g % 2 == 0)
        s = engine_initial_state(cfg)

        while True:
            tr = terminal_result(cfg, s)
            if tr.is_terminal:
                if tr.winner == 0:
                    draws += 1
                else:
                    az_winner = +1 if az_starts else -1
                    if tr.winner == az_winner:
                        wins += 1
                    else:
                        losses += 1
                break

            # Decide who moves based on the current player and who started.
            az_turn = (s.current_player == +1 and az_starts) or (s.current_player == -1 and not az_starts)
            if az_turn:
                col = _az_select_move(
                    cfg=cfg,
                    model=model,
                    device=device,
                    s=s,
                    sims=sims,
                    c_puct=c_puct,
                    seed=int(rng.integers(1_000_000)),
                    temperature=temperature,
                    debug=debug and (s.ply == 0),
                )
            else:
                col = ab_agent.select_move(cfg, s)

            s = engine_apply_move(cfg, s, col)

    return {"wins": wins, "draws": draws, "losses": losses}


def evaluate_vs_random(
    *,
    cfg: Connect4Config,
    model: PolicyValueNet,
    device: torch.device,
    games: int,
    sims: int,
    c_puct: float,
    seed: int,
    temperature: float,
    debug: bool,
) -> Dict[str, int]:
    """
    Evaluate the current model against a random opponent.

    We alternate who starts each game to reduce first-player bias and
    report results from the model's perspective (wins/draw/loss).
    """

    rng = np.random.default_rng(seed)
    rand_agent = RandomAgent("Random Eval", seed=seed)

    wins = 0
    draws = 0
    losses = 0

    model.eval()
    for g in range(games):
        az_starts = (g % 2 == 0)
        s = engine_initial_state(cfg)

        while True:
            tr = terminal_result(cfg, s)
            if tr.is_terminal:
                if tr.winner == 0:
                    draws += 1
                else:
                    az_winner = +1 if az_starts else -1
                    if tr.winner == az_winner:
                        wins += 1
                    else:
                        losses += 1
                break

            az_turn = (s.current_player == +1 and az_starts) or (s.current_player == -1 and not az_starts)
            if az_turn:
                col = _az_select_move(
                    cfg=cfg,
                    model=model,
                    device=device,
                    s=s,
                    sims=sims,
                    c_puct=c_puct,
                    seed=int(rng.integers(1_000_000)),
                    temperature=temperature,
                    debug=debug and (s.ply == 0),
                )
            else:
                col = rand_agent.select_move(cfg, s)

            s = engine_apply_move(cfg, s, col)

    return {"wins": wins, "draws": draws, "losses": losses}


def greedy_selfplay_game(
    *,
    cfg: Connect4Config,
    model: PolicyValueNet,
    device: torch.device,
    sims: int,
    c_puct: float,
    seed: int,
    temperature: float,
) -> Tuple[GameState, int]:
    """
    Play a single greedy self-play game (both sides use MCTS argmax).

    Returns the final state and the winner (+1/-1/0).
    """

    rng = np.random.default_rng(seed)
    s = engine_initial_state(cfg)
    model.eval()

    while True:
        tr = terminal_result(cfg, s)
        if tr.is_terminal:
            return s, tr.winner

        col = _az_select_move(
            cfg=cfg,
            model=model,
            device=device,
            s=s,
            sims=sims,
            c_puct=c_puct,
            seed=int(rng.integers(1_000_000)),
            temperature=temperature,
            debug=False,
        )
        s = engine_apply_move(cfg, s, col)


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
    eval_games: int,
    eval_ab_depth: int,
    eval_ab_nodes: Optional[int],
    eval_sims: int,
    eval_cpuct: float,
    eval_seed: int,
    eval_rand_games: int,
    eval_debug: bool,
    eval_greedy: bool,
    eval_temperature: float,
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
        entropy_sum = 0.0
        temp_entropy_sum = 0.0
        for g in trange(games_per_iter, desc=f"self-play iter {it}", leave=False):
            examples, winner, avg_entropy, avg_temp_entropy = play_self_game(
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
            entropy_sum += avg_entropy
            temp_entropy_sum += avg_temp_entropy

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
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
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

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())

        steps = max(train_steps, 1)
        avg_loss = total_loss / steps
        avg_policy = total_policy / steps
        avg_value = total_value / steps
        eff_epochs = (train_steps * batch_size) / max(len(replay), 1)

        avg_entropy = entropy_sum / max(games_per_iter, 1)
        avg_temp_entropy = temp_entropy_sum / max(games_per_iter, 1)
        print(
            f"iter={it} games={games_per_iter} total_games={total_games} "
            f"wins(+1/0/-1)={wins[+1]}/{wins[0]}/{wins[-1]} replay={len(replay)} "
            f"loss={avg_loss:.3f} pi={avg_policy:.3f} v={avg_value:.3f} "
            f"eff_epochs={eff_epochs:.2f} ent={avg_entropy:.2f} entT={avg_temp_entropy:.2f}"
        )

        if eval_games > 0:
            eval_result = evaluate_vs_alphabeta(
                cfg=cfg,
                model=model,
                device=device,
                games=eval_games,
                ab_depth=eval_ab_depth,
                ab_nodes=eval_ab_nodes,
                sims=eval_sims,
                c_puct=eval_cpuct,
                seed=eval_seed + it,
                temperature=eval_temperature,
                debug=eval_debug,
            )
            print(
                f"eval_vs_ab games={eval_games} "
                f"wins/draw/loss={eval_result['wins']}/{eval_result['draws']}/{eval_result['losses']}"
            )

        if eval_rand_games > 0:
            eval_rand = evaluate_vs_random(
                cfg=cfg,
                model=model,
                device=device,
                games=eval_rand_games,
                sims=eval_sims,
                c_puct=eval_cpuct,
                seed=eval_seed + 1000 + it,
                temperature=eval_temperature,
                debug=eval_debug,
            )
            print(
                f"eval_vs_random games={eval_rand_games} "
                f"wins/draw/loss={eval_rand['wins']}/{eval_rand['draws']}/{eval_rand['losses']}"
            )

        if eval_greedy:
            final_state, winner = greedy_selfplay_game(
                cfg=cfg,
                model=model,
                device=device,
                sims=eval_sims,
                c_puct=eval_cpuct,
                seed=eval_seed + 2000 + it,
                temperature=eval_temperature,
            )
            outcome = "draw" if winner == 0 else ("X" if winner == 1 else "O")
            print("eval_greedy_selfplay result:", outcome)
            print(render_board(cfg, final_state))

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
    parser.add_argument("--load-model", type=Path, default=None, help="initialize from a saved model")
    parser.add_argument("--eval-games", type=int, default=4, help="eval games vs alpha-beta per iter (0 disables)")
    parser.add_argument("--eval-ab-depth", type=int, default=2, help="alpha-beta depth for eval opponent")
    parser.add_argument("--eval-ab-nodes", type=int, default=200, help="alpha-beta node budget for eval opponent")
    parser.add_argument("--eval-sims", type=int, default=100, help="MCTS sims per move for eval agent")
    parser.add_argument("--eval-cpuct", type=float, default=1.5, help="PUCT constant for eval agent")
    parser.add_argument("--eval-seed", type=int, default=123, help="base seed for eval games")
    parser.add_argument("--eval-rand-games", type=int, default=8, help="eval games vs random per iter (0 disables)")
    parser.add_argument("--eval-debug", action="store_true", help="print eval root policy and entropy")
    parser.add_argument("--eval-greedy", action="store_true", help="print a greedy self-play final board each iter")
    parser.add_argument("--eval-temperature", type=float, default=1e-8, help="temperature for eval move selection")
    args = parser.parse_args()

    cfg = Connect4Config()
    cfg.validate()
    device = pick_device(args.device)

    if args.load_model is not None:
        model = load_model(args.load_model, device=device)
        if model.cfg != cfg:
            raise ValueError(f"loaded model cfg {model.cfg} does not match {cfg}")
    else:
        model_cfg = ModelConfig(channels=args.channels)
        model = PolicyValueNet(cfg=cfg, model_cfg=model_cfg).to(device)

    if args.checkpoint_every_games > 0:
        save_model(args.out_dir / "connect4_az_games_0.pt", model)

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
        eval_games=args.eval_games,
        eval_ab_depth=args.eval_ab_depth,
        eval_ab_nodes=args.eval_ab_nodes,
        eval_sims=args.eval_sims,
        eval_cpuct=args.eval_cpuct,
        eval_seed=args.eval_seed,
        eval_rand_games=args.eval_rand_games,
        eval_debug=args.eval_debug,
        eval_greedy=args.eval_greedy,
        eval_temperature=args.eval_temperature,
    )


if __name__ == "__main__":
    main()
