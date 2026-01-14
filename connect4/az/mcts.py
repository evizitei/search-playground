"""PUCT MCTS for canonical Connect-4 states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from connect4.az.game import CanonicalState, apply_move, encode_state, legal_moves, terminal_value
from connect4.az.model import PolicyValueNet
from connect4.engine import Connect4Config


@dataclass
class NodeStats:
    """
    Per-node edge statistics, stored as arrays indexed by action (column).

    P[a] : prior from the policy net
    N[a] : visit count
    W[a] : total backed-up value
    """

    P: np.ndarray  # float32, shape (W,)
    N: np.ndarray  # int32,   shape (W,)
    W: np.ndarray  # float32, shape (W,)
    legal: np.ndarray  # bool, shape (W,)


def _dirichlet_noise(rng: np.random.Generator, alpha: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    x = rng.gamma(shape=alpha, scale=1.0, size=(n,))
    s = float(x.sum())
    if s == 0.0:
        return np.full((n,), 1.0 / n, dtype=np.float32)
    return (x / s).astype(np.float32)


class PUCTMCTS:
    """
    Neural-guided MCTS with PUCT selection and value network leaf evaluation.

    The canonicalization convention means the value head always represents the
    outcome for the player to move at that state. During backup, we simply flip
    the value sign on each ply.
    """

    def __init__(
        self,
        *,
        cfg: Connect4Config,
        model: PolicyValueNet,
        device: torch.device,
        sims: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_eps: float,
        seed: int,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng(seed)

        # Tree keyed by canonical board bytes. Player-to-move is implicit.
        self.tree: Dict[bytes, NodeStats] = {}

    def run(self, root: CanonicalState) -> None:
        for _ in range(self.sims):
            self._simulate(root)

    def root_policy(self, root: CanonicalState, *, temperature: float) -> np.ndarray:
        key = root.board.tobytes()
        stats = self.tree.get(key)
        if stats is None:
            return np.zeros((self.cfg.width,), dtype=np.float32)

        legal_idx = np.nonzero(stats.legal)[0]
        if len(legal_idx) == 0:
            return np.zeros((self.cfg.width,), dtype=np.float32)

        if temperature <= 1e-8:
            a = int(legal_idx[np.argmax(stats.N[legal_idx])])
            pi = np.zeros((self.cfg.width,), dtype=np.float32)
            pi[a] = 1.0
            return pi

        # pi(a) proportional to N(a)^(1/T)
        inv_t = 1.0 / float(temperature)
        w = np.zeros((self.cfg.width,), dtype=np.float32)
        n = stats.N.astype(np.float32)
        w[legal_idx] = np.power(n[legal_idx], inv_t)
        s = float(w.sum())
        if s == 0.0:
            w[legal_idx] = 1.0 / len(legal_idx)
            return w
        return w / s

    def _simulate(self, root: CanonicalState) -> None:
        path: List[Tuple[NodeStats, int]] = []
        s = root

        while True:
            # Each loop is one step down the tree. We either:
            #  - hit a terminal node, or
            #  - expand a new leaf, or
            #  - select an action and continue.
            key = s.board.tobytes()
            stats = self.tree.get(key)

            tv = terminal_value(self.cfg, s)
            if tv is not None:
                # Terminal value is already from the player-to-move perspective.
                self._backup(path, tv)
                return

            if stats is None:
                # Leaf expansion evaluates the network and initializes priors.
                value = self._expand(s, is_root=(len(path) == 0))
                self._backup(path, value)
                return

            # PUCT selection.
            a = self._select(stats)
            path.append((stats, a))
            s = apply_move(self.cfg, s, a)

    def _expand(self, s: CanonicalState, *, is_root: bool) -> float:
        legal = legal_moves(self.cfg, s)
        legal_mask = np.zeros((self.cfg.width,), dtype=bool)
        legal_mask[legal] = True

        with torch.no_grad():
            x = encode_state(self.cfg, s).unsqueeze(0).to(self.device)
            logits, value = self.model(x)
            policy = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            v = float(value.item())

        # Mask illegal actions and renormalize.
        policy = policy.astype(np.float32)
        policy[~legal_mask] = 0.0
        s_policy = float(policy.sum())
        if s_policy == 0.0:
            policy[legal_mask] = 1.0 / int(legal_mask.sum())
        else:
            policy /= s_policy

        # Add Dirichlet noise at the root for exploration in self-play.
        if is_root and self.dirichlet_eps > 0.0:
            noise = _dirichlet_noise(self.rng, self.dirichlet_alpha, int(legal_mask.sum()))
            noisy = policy.copy()
            noisy[legal_mask] = (
                (1.0 - self.dirichlet_eps) * policy[legal_mask]
                + self.dirichlet_eps * noise
            )
            policy = noisy

        # Store stats so future simulations can select among actions.
        self.tree[s.board.tobytes()] = NodeStats(
            P=policy,
            N=np.zeros((self.cfg.width,), dtype=np.int32),
            W=np.zeros((self.cfg.width,), dtype=np.float32),
            legal=legal_mask,
        )

        return v

    def _select(self, stats: NodeStats) -> int:
        legal_idx = np.nonzero(stats.legal)[0]
        if len(legal_idx) == 0:
            return 0

        n_sum = float(np.sum(stats.N))
        if n_sum == 0.0:
            n_sum = 1.0

        best_score = -1e9
        best_a = int(legal_idx[0])

        for a in legal_idx:
            n = float(stats.N[a])
            q = float(stats.W[a] / n) if n > 0 else 0.0
            u = self.c_puct * float(stats.P[a]) * (np.sqrt(n_sum) / (1.0 + n))
            score = q + u
            if score > best_score:
                best_score = score
                best_a = int(a)

        return best_a

    def _backup(self, path: Iterable[Tuple[NodeStats, int]], value: float) -> None:
        v = float(value)
        for stats, a in reversed(list(path)):
            # Update edge statistics for the chosen action.
            stats.N[a] += 1
            stats.W[a] += v
            # Flip perspective as we move back up the tree.
            v = -v
