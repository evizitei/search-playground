"""Helpers for converting engine GameState into canonical AlphaZero states."""

from __future__ import annotations

from connect4.az.game import CanonicalState
from connect4.engine import GameState


def to_canonical_state(s: GameState) -> CanonicalState:
    """
    Convert an engine GameState to a canonical state (player to move is +1).

    We multiply the board by current_player so that the player to move becomes +1.
    """

    board = s.board * s.current_player
    return CanonicalState(
        board=board,
        heights=s.heights.copy(),
        last_row=s.last_row,
        last_col=s.last_col,
        ply=s.ply,
    )
