"""Interactive AlphaZero visualization (AlphaZero vs other agents)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from connect4.agents.alphabeta import AlphaBetaAgent, evaluate_moves
from connect4.agents.random_agent import RandomAgent
from connect4.az.bridge import to_canonical_state
from connect4.az.game import apply_move as az_apply_move
from connect4.az.game import encode_state, legal_moves as az_legal_moves, terminal_value
from connect4.az.mcts import NodeStats, PUCTMCTS
from connect4.az.model import load_model, pick_device
from connect4.cli import render_board
from connect4.engine import Connect4Config, GameState, apply_move, initial_state, terminal_result


def _policy_entropy(pi: np.ndarray) -> float:
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


def _compute_root_outputs(
    *,
    cfg: Connect4Config,
    model: torch.nn.Module,
    device: torch.device,
    s: GameState,
    sims: int,
    c_puct: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[NodeStats], "CanonicalState"]:
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

    prior = mcts.root_prior(canonical)
    pi = mcts.root_policy(canonical, temperature=1e-8)
    stats = mcts.root_stats(canonical)
    if stats is not None and int(stats.N.sum()) > 0:
        n = stats.N.astype(np.float32)
        pi_counts = n / float(n.sum())
    else:
        pi_counts = np.zeros_like(prior)

    with torch.no_grad():
        x = encode_state(cfg, canonical).unsqueeze(0).to(device)
        _, value = model(x)
        v = float(value.item())

    return prior, pi, pi_counts, v, stats, canonical


def _print_step(
    *,
    cfg: Connect4Config,
    s: GameState,
    prior: np.ndarray,
    pi: np.ndarray,
    pi_counts: np.ndarray,
    value: float,
    ab_scores: Optional[dict[int, float]] = None,
    q_values: Optional[list[Optional[float]]] = None,
    child_values: Optional[list[Optional[float]]] = None,
) -> None:
    player = "X" if s.current_player == 1 else "O"
    print(render_board(cfg, s))
    print(f"player_to_move: {player} (az)")
    print(f"prior: {np.round(prior, 3)} entP={_policy_entropy(prior):.3f}")
    print(f"piN:   {np.round(pi_counts, 3)} entN={_policy_entropy(pi_counts):.3f}")
    print(f"pi:    {np.round(pi, 3)} ent={_policy_entropy(pi):.3f}")
    print(f"value (for player to move): {value:.3f}")
    if ab_scores:
        ordered = [ab_scores.get(i) for i in range(cfg.width)]
        rounded = [None if v is None else round(v, 3) for v in ordered]
        print(f"ab:    {rounded}")
    if q_values:
        rounded = [None if v is None else round(v, 3) for v in q_values]
        print(f"q:     {rounded}")
    if child_values:
        rounded = [None if v is None else round(v, 3) for v in child_values]
        print(f"v1:    {rounded}")


def _print_non_az_step(*, cfg: Connect4Config, s: GameState, agent_label: str) -> None:
    player = "X" if s.current_player == 1 else "O"
    print(render_board(cfg, s))
    print(f"player_to_move: {player} ({agent_label})")


def play_interactive(
    *,
    cfg: Connect4Config,
    model_x: Optional[Path],
    model_o: Optional[Path],
    x_agent: str,
    o_agent: str,
    ab_depth: int,
    ab_nodes: Optional[int],
    ab_debug: bool,
    az_debug: bool,
    sims: int,
    c_puct: float,
    temperature: float,
    device: str,
    seed: int,
    auto: bool,
) -> None:
    dev = pick_device(device)
    model_x_net = load_model(model_x, device=dev) if x_agent == "az" else None
    if o_agent == "az":
        if model_o is None:
            model_o = model_x
        model_o_net = load_model(model_o, device=dev)
    else:
        model_o_net = None

    for net in (model_x_net, model_o_net):
        if net is not None and net.cfg != cfg:
            raise ValueError("model config does not match game config")

    rng = np.random.default_rng(seed)
    rand_seed = int(rng.integers(1_000_000))
    ab_seed = int(rng.integers(1_000_000))
    random_x = RandomAgent("random_x", seed=rand_seed) if x_agent == "random" else None
    random_o = RandomAgent("random_o", seed=rand_seed + 1) if o_agent == "random" else None
    ab_x = AlphaBetaAgent("ab_x", max_depth=ab_depth, max_nodes=ab_nodes) if x_agent == "ab" else None
    ab_o = AlphaBetaAgent("ab_o", max_depth=ab_depth, max_nodes=ab_nodes) if o_agent == "ab" else None
    s = initial_state(cfg)
    auto_play = auto

    while True:
        tr = terminal_result(cfg, s)
        if tr.is_terminal:
            if tr.winner == 0:
                print("Result: draw")
            else:
                winner = "X" if tr.winner == 1 else "O"
                print(f"Result: {winner} wins")
            print(render_board(cfg, s))
            return

        agent_kind = x_agent if s.current_player == 1 else o_agent
        if agent_kind == "az":
            model = model_x_net if s.current_player == 1 else model_o_net
            if model is None:
                raise ValueError("missing model for AlphaZero agent")
            prior, pi, pi_counts, value, stats, canonical = _compute_root_outputs(
                cfg=cfg,
                model=model,
                device=dev,
                s=s,
                sims=sims,
                c_puct=c_puct,
                seed=int(rng.integers(1_000_000)),
            )
            scores = None
            q_values = None
            child_values = None
            if ab_debug:
                scores = evaluate_moves(cfg, s, max_depth=ab_depth, max_nodes=ab_nodes)
            if az_debug and stats is not None:
                q_values = [None] * cfg.width
                for a in range(cfg.width):
                    if not stats.legal[a]:
                        continue
                    n = float(stats.N[a])
                    q_values[a] = float(stats.W[a] / n) if n > 0 else 0.0

                child_values = [None] * cfg.width
                legal = az_legal_moves(cfg, canonical)
                for col in legal:
                    child = az_apply_move(cfg, canonical, int(col))
                    tv = terminal_value(cfg, child)
                    if tv is not None:
                        # tv is from the child player-to-move perspective.
                        v_child = -float(tv)
                    else:
                        with torch.no_grad():
                            x = encode_state(cfg, child).unsqueeze(0).to(dev)
                            _, v = model(x)
                            # Flip to current player perspective.
                            v_child = -float(v.item())
                    child_values[int(col)] = v_child
            _print_step(
                cfg=cfg,
                s=s,
                prior=prior,
                pi=pi,
                pi_counts=pi_counts,
                value=value,
                ab_scores=scores,
                q_values=q_values,
                child_values=child_values,
            )
            col = _select_move_from_pi(pi, temperature=temperature, rng=rng)
        elif agent_kind == "random":
            agent = random_x if s.current_player == 1 else random_o
            if agent is None:
                raise ValueError("random agent not initialized")
            _print_non_az_step(cfg=cfg, s=s, agent_label="random")
            col = agent.select_move(cfg, s)
        elif agent_kind == "ab":
            agent = ab_x if s.current_player == 1 else ab_o
            if agent is None:
                raise ValueError("alpha-beta agent not initialized")
            _print_non_az_step(cfg=cfg, s=s, agent_label="alpha-beta")
            col = agent.select_move(cfg, s)
        else:
            raise ValueError(f"unknown agent kind: {agent_kind}")

        print(f"chosen_move: col {col}")

        if not auto_play:
            cmd = input("Enter=next, a=auto, q=quit: ").strip().lower()
            if cmd == "q":
                return
            if cmd == "a":
                auto_play = True

        s = apply_move(cfg, s, col)
        print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive AlphaZero visualization")
    parser.add_argument("--model-x", type=Path, default=None, help="model path for X (required if X is az)")
    parser.add_argument("--model-o", type=Path, default=None, help="model path for O (defaults to X)")
    parser.add_argument("--x-agent", type=str, choices=["az", "random", "ab"], default="az", help="agent for X")
    parser.add_argument("--o-agent", type=str, choices=["az", "random", "ab"], default="az", help="agent for O")
    parser.add_argument("--ab-depth", type=int, default=4, help="alpha-beta depth (if used)")
    parser.add_argument("--ab-nodes", type=int, default=None, help="alpha-beta node budget (if used)")
    parser.add_argument("--sims", type=int, default=400, help="MCTS sims per move")
    parser.add_argument("--cpuct", type=float, default=1.5, help="PUCT constant")
    parser.add_argument("--temperature", type=float, default=1e-8, help="move selection temperature")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--auto", action="store_true", help="run without interactive prompts")
    parser.add_argument("--ab-debug", action="store_true", help="print alpha-beta scores each AZ move")
    parser.add_argument("--az-debug", action="store_true", help="print MCTS Q and child value per move")
    args = parser.parse_args()

    cfg = Connect4Config()
    cfg.validate()

    if args.x_agent == "az" and args.model_x is None:
        parser.error("--model-x is required when X uses the az agent")
    if args.o_agent == "az" and args.model_o is None and args.model_x is None:
        parser.error("--model-o (or --model-x) is required when O uses the az agent")

    play_interactive(
        cfg=cfg,
        model_x=args.model_x,
        model_o=args.model_o,
        x_agent=args.x_agent,
        o_agent=args.o_agent,
        ab_depth=args.ab_depth,
        ab_nodes=args.ab_nodes,
        ab_debug=args.ab_debug,
        az_debug=args.az_debug,
        sims=args.sims,
        c_puct=args.cpuct,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
        auto=args.auto,
    )


if __name__ == "__main__":
    main()
