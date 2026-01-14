"""Policy/value network for Connect-4 AlphaZero-style training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from connect4.engine import Connect4Config


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 64


def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class PolicyValueNet(nn.Module):
    """
    Small convolutional network with separate policy and value heads.

    Policy head outputs logits over columns (width actions).
    Value head outputs a scalar in [-1, 1] via tanh.
    """

    def __init__(self, *, cfg: Connect4Config, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg

        channels = model_cfg.channels
        self.trunk = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        trunk_out = channels * cfg.height * cfg.width
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(trunk_out, cfg.width),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(trunk_out, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 2, H, W)
        returns:
          - policy_logits: (B, W)
          - value: (B,) in [-1, 1]
        """

        h = self.trunk(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def save_model(path: Path, model: PolicyValueNet) -> None:
    payload = {
        "cfg": {"width": model.cfg.width, "height": model.cfg.height, "k": model.cfg.k},
        "model_cfg": {"channels": model.model_cfg.channels},
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(path: Path, *, device: torch.device) -> PolicyValueNet:
    payload = torch.load(path, map_location=device)
    cfg = Connect4Config(**payload["cfg"])
    cfg.validate()
    model_cfg = ModelConfig(**payload["model_cfg"])
    model = PolicyValueNet(cfg=cfg, model_cfg=model_cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
