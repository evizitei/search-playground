"""AlphaZero-style agent backed by a policy/value network and PUCT MCTS."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from connect4.agents.base import Agent
from connect4.az.bridge import to_canonical_state
from connect4.az.mcts import PUCTMCTS
from connect4.az.model import load_model, pick_device
from connect4.engine import Connect4Config, GameState


class AlphaZeroAgent(Agent):
    def __init__(
        self,
        name: str,
        *,
        model_path: Path,
        sims: int = 400,
        c_puct: float = 1.5,
        device: str = "auto",
        temperature: float = 1e-8,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.sims = sims
        self.c_puct = c_puct
        self.temperature = temperature
        self.seed = seed
        self.device = pick_device(device)
        self.model = load_model(model_path, device=self.device)

    def select_move(self, cfg: Connect4Config, s: GameState) -> int:
        if (cfg.width, cfg.height, cfg.k) != (
            self.model.cfg.width,
            self.model.cfg.height,
            self.model.cfg.k,
        ):
            raise ValueError("model config does not match game config")

        canonical = to_canonical_state(s)
        mcts = PUCTMCTS(
            cfg=cfg,
            model=self.model,
            device=self.device,
            sims=self.sims,
            c_puct=self.c_puct,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.0,
            seed=self.seed,
        )
        mcts.run(canonical)

        pi = mcts.root_policy(canonical, temperature=self.temperature)
        if float(pi.sum()) <= 0.0:
            legal = np.nonzero(canonical.heights < cfg.height)[0]
            pi = np.zeros((cfg.width,), dtype=np.float32)
            pi[legal] = 1.0 / len(legal)

        return int(pi.argmax())
