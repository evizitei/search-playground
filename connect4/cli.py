"""CLI rendering and input helpers for Connect-4."""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from connect4.agents import Agent, AlphaBetaAgent, HumanAgent, RandomAgent
from connect4.engine import Connect4Config, GameState, Move, apply_move, initial_state, legal_moves, terminal_result


def render_board(cfg: Connect4Config, s: GameState) -> str:
    sym = {+1: "X", -1: "O", 0: "."}
    lines: List[str] = []
    for r in range(cfg.height - 1, -1, -1):
        lines.append(" ".join(sym[int(s.board[r, c])] for c in range(cfg.width)))
    lines.append("-" * (2 * cfg.width - 1))
    lines.append(" ".join(str(c) for c in range(cfg.width)))
    return "\n".join(lines)


def format_move_history(moves: Sequence[Move]) -> str:
    parts = []
    for m in moves:
        player = "X" if m.player == 1 else "O"
        parts.append(f"{m.ply}:{player}@{m.col}")
    return " ".join(parts)


def _parse_column(raw: str, width: int) -> Optional[int]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        col = int(raw)
    except ValueError:
        return None

    if 0 <= col < width:
        return col
    if 1 <= col <= width:
        return col - 1
    return None


def prompt_for_human_move(cfg: Connect4Config, s: GameState, name: str) -> int:
    legal = legal_moves(cfg, s).tolist()
    prompt = f"{name} ({'X' if s.current_player == 1 else 'O'}) to move. Column {legal}: "

    while True:
        raw = input(prompt)
        col = _parse_column(raw, cfg.width)
        if col is None:
            print("Enter a column index (0-based or 1-based).")
            continue
        if col not in legal:
            print("Illegal move: column full or out of range.")
            continue
        return col


def play_game(cfg: Connect4Config, x_agent: Agent, o_agent: Agent) -> None:
    s = initial_state(cfg)

    while True:
        print(render_board(cfg, s))
        tr = terminal_result(cfg, s)
        if tr.is_terminal:
            if tr.winner == 0:
                print("Result: draw")
            else:
                winner = "X" if tr.winner == 1 else "O"
                print(f"Result: {winner} wins ({tr.reason})")
            if s.move_history:
                print(f"Moves: {format_move_history(s.move_history)}")
            return

        agent = x_agent if s.current_player == 1 else o_agent
        col = agent.select_move(cfg, s)

        s = apply_move(cfg, s, col)
        move = s.move_history[-1]
        print(f"Move: {'X' if move.player == 1 else 'O'} -> col {move.col}, row {s.last_row}")
        print("")


def _pick_seed(base: Optional[int], override: Optional[int], *, offset: int = 0) -> Optional[int]:
    if override is not None:
        return override
    if base is not None:
        return base + offset
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect-4 (6x6) CLI")
    parser.add_argument("--x", choices=["human", "random", "alphabeta"], default="human", help="agent for X")
    parser.add_argument("--o", choices=["human", "random", "alphabeta"], default="human", help="agent for O")

    parser.add_argument("--seed", type=int, default=None, help="base random seed (for random agents)")
    parser.add_argument("--seed-x", type=int, default=None, help="seed for X random agent")
    parser.add_argument("--seed-o", type=int, default=None, help="seed for O random agent")

    parser.add_argument("--ab-depth", type=int, default=4, help="alpha-beta depth (plies)")
    parser.add_argument("--ab-depth-x", type=int, default=None, help="alpha-beta depth for X")
    parser.add_argument("--ab-depth-o", type=int, default=None, help="alpha-beta depth for O")

    parser.add_argument("--ab-nodes", type=int, default=None, help="alpha-beta node budget")
    parser.add_argument("--ab-nodes-x", type=int, default=None, help="alpha-beta node budget for X")
    parser.add_argument("--ab-nodes-o", type=int, default=None, help="alpha-beta node budget for O")

    args = parser.parse_args()

    cfg = Connect4Config()

    def build_agent(side: str) -> Agent:
        if side == "x":
            choice = args.x
            depth = args.ab_depth_x if args.ab_depth_x is not None else args.ab_depth
            nodes = args.ab_nodes_x if args.ab_nodes_x is not None else args.ab_nodes
            seed = _pick_seed(args.seed, args.seed_x, offset=0)
            name = "Player X"
        else:
            choice = args.o
            depth = args.ab_depth_o if args.ab_depth_o is not None else args.ab_depth
            nodes = args.ab_nodes_o if args.ab_nodes_o is not None else args.ab_nodes
            seed = _pick_seed(args.seed, args.seed_o, offset=1)
            name = "Player O"

        if choice == "human":
            return HumanAgent(name, prompt_for_human_move)
        if choice == "random":
            return RandomAgent(f"Random {name[-1]}", seed=seed)
        if choice == "alphabeta":
            return AlphaBetaAgent(f"AlphaBeta {name[-1]}", max_depth=depth, max_nodes=nodes)

        raise ValueError(f"unsupported agent choice: {choice}")

    x_agent = build_agent("x")
    o_agent = build_agent("o")

    play_game(cfg, x_agent, o_agent)


if __name__ == "__main__":
    main()
