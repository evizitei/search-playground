"""Random baseline agent."""

from __future__ import annotations

import random
from typing import Optional

from connect4.agents.base import Agent
from connect4.engine import Connect4Config, GameState, legal_moves


class RandomAgent(Agent):
    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name = name
        self.rng = random.Random(seed)

    def select_move(self, cfg: Connect4Config, s: GameState) -> int:
        legal = legal_moves(cfg, s).tolist()
        return self.rng.choice(legal)
