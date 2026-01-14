"""Human-in-the-loop agent that defers input handling to a CLI prompt function."""

from __future__ import annotations

from typing import Callable

from connect4.agents.base import Agent
from connect4.engine import Connect4Config, GameState

PromptFn = Callable[[Connect4Config, GameState, str], int]


class HumanAgent(Agent):
    def __init__(self, name: str, prompt_fn: PromptFn) -> None:
        self.name = name
        self.prompt_fn = prompt_fn

    def select_move(self, cfg: Connect4Config, s: GameState) -> int:
        return self.prompt_fn(cfg, s, self.name)
