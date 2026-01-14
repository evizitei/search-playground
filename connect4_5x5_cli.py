#!/usr/bin/env python3
"""
Connect-4 on a 5x5 grid with gravity, plus a simple CLI for 2 humans.

Designed to be MCTS-friendly:
- explicit GameState dataclass
- functional apply_move that returns a new state
- move history with (ply, player, row, col)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Connect4Config:
    width: int = 5
    height: int = 5
    k: int = 4

    def validate(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("width/height must be >= 1")
        if self.k < 2:
            raise ValueError("k must be >= 2")
        if self.k > max(self.width, self.height):
            raise ValueError("k must be <= max(width, height)")


@dataclass(frozen=True)
class Move:
    ply: int
    player: int  # +1 or -1
    row: int
    col: int


@dataclass(frozen=True)
class GameState:
    board: np.ndarray  # shape (H, W), dtype=int8
    heights: np.ndarray  # shape (W,), dtype=int16
    current_player: int  # +1 or -1
    last_row: int
    last_col: int
    ply: int
    move_history: Tuple[Move, ...]


@dataclass(frozen=True)
class TerminalResult:
    is_terminal: bool
    winner: int  # +1 / -1 / 0 (draw) / 2 (not terminal)
    reason: str


def initial_state(cfg: Connect4Config) -> GameState:
    cfg.validate()
    board = np.zeros((cfg.height, cfg.width), dtype=np.int8)
    heights = np.zeros((cfg.width,), dtype=np.int16)
    return GameState(
        board=board,
        heights=heights,
        current_player=+1,
        last_row=-1,
        last_col=-1,
        ply=0,
        move_history=(),
    )


def legal_moves(cfg: Connect4Config, s: GameState) -> np.ndarray:
    return np.nonzero(s.heights < cfg.height)[0]


def apply_move(cfg: Connect4Config, s: GameState, col: int) -> GameState:
    if col < 0 or col >= cfg.width:
        raise ValueError("col out of range")
    if s.heights[col] >= cfg.height:
        raise ValueError("illegal move: column full")

    row = int(s.heights[col])
    board = s.board.copy()
    board[row, col] = s.current_player
    heights = s.heights.copy()
    heights[col] += 1

    move = Move(ply=s.ply, player=s.current_player, row=row, col=col)
    history = s.move_history + (move,)

    return GameState(
        board=board,
        heights=heights,
        current_player=-s.current_player,
        last_row=row,
        last_col=col,
        ply=s.ply + 1,
        move_history=history,
    )


def _count_dir(cfg: Connect4Config, board: np.ndarray, row: int, col: int, dr: int, dc: int) -> int:
    start = int(board[row, col])
    if start == 0:
        return 0
    r, c = row + dr, col + dc
    count = 0
    while 0 <= r < cfg.height and 0 <= c < cfg.width:
        if int(board[r, c]) != start:
            break
        count += 1
        r += dr
        c += dc
    return count


def is_win_from_last_move(cfg: Connect4Config, s: GameState) -> bool:
    if s.last_row < 0:
        return False
    r, c = s.last_row, s.last_col
    if int(s.board[r, c]) == 0:
        return False

    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        total = 1 + _count_dir(cfg, s.board, r, c, dr, dc) + _count_dir(cfg, s.board, r, c, -dr, -dc)
        if total >= cfg.k:
            return True
    return False


def terminal_result(cfg: Connect4Config, s: GameState) -> TerminalResult:
    if is_win_from_last_move(cfg, s):
        winner = -s.current_player
        return TerminalResult(True, winner, "connect-k")
    if s.ply >= cfg.width * cfg.height:
        return TerminalResult(True, 0, "draw")
    return TerminalResult(False, 2, "in-progress")


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
        parts.append(f"{m.ply}:{player}@({m.row},{m.col})")
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


def play_cli(cfg: Connect4Config) -> None:
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

        player = "X" if s.current_player == 1 else "O"
        legal = legal_moves(cfg, s).tolist()
        prompt = f"Player {player} to move. Column {legal}: "

        while True:
            raw = input(prompt)
            col = _parse_column(raw, cfg.width)
            if col is None:
                print("Enter a column index (0-based or 1-based).")
                continue
            if col not in legal:
                print("Illegal move: column full or out of range.")
                continue
            break

        s = apply_move(cfg, s, col)
        move = s.move_history[-1]
        print(f"Move: {'X' if move.player == 1 else 'O'} -> col {move.col}, row {move.row}")
        print("")


def main() -> None:
    cfg = Connect4Config()
    play_cli(cfg)


if __name__ == "__main__":
    main()
