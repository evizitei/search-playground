"""Abstract base class for Connect-4 agents."""

from __future__ import annotations

import abc

from connect4.engine import Connect4Config, GameState


class Agent(abc.ABC):
    name: str

    @abc.abstractmethod
    def select_move(self, cfg: Connect4Config, s: GameState) -> int:
        raise NotImplementedError
