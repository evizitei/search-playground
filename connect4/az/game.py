"""Canonicalized Connect-4 game logic for AlphaZero-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from connect4.engine import Connect4Config


@dataclass(frozen=True)
class CanonicalState:
    """
    Canonical state where the player to move is always +1.

    board values:
      +1 = current player
      -1 = opponent
      0  = empty
    """

    board: np.ndarray  # shape (H, W), dtype=int8
    heights: np.ndarray  # shape (W,), dtype=int16
    last_row: int
    last_col: int
    ply: int


def initial_state(cfg: Connect4Config) -> CanonicalState:
    cfg.validate()
    board = np.zeros((cfg.height, cfg.width), dtype=np.int8)
    heights = np.zeros((cfg.width,), dtype=np.int16)
    return CanonicalState(board=board, heights=heights, last_row=-1, last_col=-1, ply=0)


def legal_moves(cfg: Connect4Config, s: CanonicalState) -> np.ndarray:
    return np.nonzero(s.heights < cfg.height)[0]


def apply_move(cfg: Connect4Config, s: CanonicalState, col: int) -> CanonicalState:
    """
    Apply a move for the canonical current player (+1), then flip perspective.

    1) Place a +1 stone at (row=heights[col], col)
    2) Increment heights[col]
    3) Multiply board by -1 so the next player becomes +1
    """

    if col < 0 or col >= cfg.width:
        raise ValueError("col out of range")
    if s.heights[col] >= cfg.height:
        raise ValueError("illegal move: column full")

    row = int(s.heights[col])
    board = s.board.copy()
    board[row, col] = +1
    heights = s.heights.copy()
    heights[col] += 1

    board *= -1
    return CanonicalState(board=board, heights=heights, last_row=row, last_col=col, ply=s.ply + 1)


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


def is_win_from_last_move(cfg: Connect4Config, s: CanonicalState) -> bool:
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


def terminal_value(cfg: Connect4Config, s: CanonicalState) -> Optional[float]:
    """
    Return terminal value from the perspective of the canonical player to move.

    - -1.0 if the opponent just won
    -  0.0 if draw
    - None if game continues
    """

    if is_win_from_last_move(cfg, s):
        return -1.0
    if s.ply >= cfg.width * cfg.height:
        return 0.0
    return None


def encode_state(cfg: Connect4Config, s: CanonicalState) -> torch.Tensor:
    """
    Encode the canonical board into 2 planes.

    plane 0: current player stones (board == +1)
    plane 1: opponent stones       (board == -1)
    """

    cur = (s.board == 1).astype(np.float32)
    opp = (s.board == -1).astype(np.float32)
    x = np.stack([cur, opp], axis=0)
    return torch.from_numpy(x)
