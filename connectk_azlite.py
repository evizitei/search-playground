#!/usr/bin/env python3
"""
Connect-K "AlphaZero-lite" demo (PyTorch + PUCT MCTS).

This is intentionally written for reading/learning:
  - lots of comments in the key theoretical places (canonicalization, PUCT, backup, targets)
  - minimal abstraction (single file, explicit data flow)
  - not optimized for speed

Core ideas you should be able to "see" in code:

1) Canonical state representation (AlphaZero trick)
   ------------------------------------------------
   We store the board so that the player to move is always +1 ("current player").
   After making a move we multiply the board by -1, so the next player again becomes +1.

   Benefits:
     - The network never needs a "side to move" input.
     - The value head V(s) always means "expected outcome for the player to move".
     - During MCTS backup, values flip sign at each ply (v := -v) naturally.

2) Neural-guided MCTS with PUCT
   ----------------------------
   AlphaZero replaces random rollouts with a policy+value network.
   It also replaces classic UCT with PUCT (prior-guided exploration):

      a* = argmax_a [ Q(s,a) + U(s,a) ]

      Q(s,a) = W(s,a) / N(s,a)  (mean backed-up value)

      U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

   Intuition:
     - Q prefers moves that empirically lead to wins.
     - U prefers moves that are:
         * high prior probability under the policy net P(s,a)
         * under-explored (small N)
         * at a heavily-visited node (sqrt(sum N) grows)

3) Self-play training targets (π, z)
   ---------------------------------
   For each position s encountered in self-play:
     - π is the normalized MCTS visit counts at the root.
     - z is the final game outcome from the perspective of the player to move at s.

   Training loss:
     - policy loss: cross-entropy between π and softmax(policy_logits)
     - value loss: MSE between z and V(s)

CLI:
  - train: self-play + training loop
  - play:  play vs the current model
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from rich.console import Console
from rich.table import Table
from tqdm import trange


console = Console()
app = typer.Typer(no_args_is_help=True)


# -------------------------
# Game: Connect-K
# -------------------------


@dataclass(frozen=True)
class ConnectKConfig:
    width: int = 7
    height: int = 6
    k: int = 4

    def validate(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("width/height must be >= 1")
        if self.k < 2:
            raise ValueError("k must be >= 2")
        if self.k > max(self.width, self.height):
            raise ValueError("k must be <= max(width, height)")


@dataclass(frozen=True)
class GameState:
    """
    Canonical state: player to move is always +1.

    board: int8 array (H, W) with values in {-1, 0, +1}
      +1 = stones of the player to move (current player)
      -1 = stones of the opponent

    heights: number of stones in each column (for fast move application)
    last_move: (row, col) of the most recent move BEFORE the perspective flip
               stored as the location in the *current* board representation.
               After apply_move(), the just-played stone becomes -1 (opponent),
               but the location remains valid for win checking.
    ply: number of moves made since start (used only for labeling winners in self-play)
    """

    board: np.ndarray  # shape (H, W), dtype=int8
    heights: np.ndarray  # shape (W,), dtype=int16
    last_row: int
    last_col: int
    ply: int


def initial_state(cfg: ConnectKConfig) -> GameState:
    cfg.validate()
    board = np.zeros((cfg.height, cfg.width), dtype=np.int8)
    heights = np.zeros((cfg.width,), dtype=np.int16)
    return GameState(board=board, heights=heights, last_row=-1, last_col=-1, ply=0)


def legal_mask(cfg: ConnectKConfig, s: GameState) -> np.ndarray:
    return (s.heights < cfg.height)


def apply_move(cfg: ConnectKConfig, s: GameState, col: int) -> GameState:
    """
    Apply a move for the canonical current player (+1), then flip perspective.

    Step-by-step:
      1) Place a +1 stone at (row=heights[col], col)
      2) Increment heights[col]
      3) Multiply board by -1 so the next player becomes +1 ("canonicalize")

    This is the "canonical form" trick. After the flip, the stone we just placed
    becomes -1 because it now belongs to the opponent (from the next player's view).
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
    return GameState(board=board, heights=heights, last_row=row, last_col=col, ply=s.ply + 1)


def _count_dir(cfg: ConnectKConfig, board: np.ndarray, row: int, col: int, dr: int, dc: int) -> int:
    """Count consecutive stones equal to board[row,col] in direction (dr,dc), excluding start cell."""
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


def is_win_from_last_move(cfg: ConnectKConfig, s: GameState) -> bool:
    """
    Only the last move can create a new connect-K, so we check lines through last_row/last_col.

    Note: because we canonicalize by flipping after each move, the last move was made by the
    *opponent* in the current representation, so board[last_row,last_col] is typically -1.
    That is fine: we check for a K-line of whatever sign is at that cell.
    """
    if s.last_row < 0:
        return False
    r, c = s.last_row, s.last_col
    if int(s.board[r, c]) == 0:
        return False

    # Horizontal, vertical, 2 diagonals.
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        total = 1 + _count_dir(cfg, s.board, r, c, dr, dc) + _count_dir(cfg, s.board, r, c, -dr, -dc)
        if total >= cfg.k:
            return True
    return False


def terminal_value(cfg: ConnectKConfig, s: GameState) -> Optional[float]:
    """
    Return terminal value from the perspective of the player to move (canonical current player):
      - -1.0 if the opponent just won with the last move
      -  0.0 if draw (board full)
      - None if game continues
    """
    if is_win_from_last_move(cfg, s):
        return -1.0
    if s.ply >= cfg.width * cfg.height:
        return 0.0
    return None


def render_board(cfg: ConnectKConfig, s: GameState) -> str:
    """
    Print from canonical perspective:
      X = current player (+1)
      O = opponent (-1)
    """
    sym = {+1: "X", -1: "O", 0: "."}
    lines: List[str] = []
    for r in range(cfg.height - 1, -1, -1):
        lines.append(" ".join(sym[int(s.board[r, c])] for c in range(cfg.width)))
    lines.append("-" * (2 * cfg.width - 1))
    lines.append(" ".join(str(c) for c in range(cfg.width)))
    return "\n".join(lines)


def encode_state(cfg: ConnectKConfig, s: GameState) -> torch.Tensor:
    """
    Encode canonical board into 2 planes, as commonly used in AlphaZero-style code.

    shape: (2, H, W)
      plane 0: current player's stones (board == +1)
      plane 1: opponent's stones      (board == -1)
    """
    cur = (s.board == 1).astype(np.float32)
    opp = (s.board == -1).astype(np.float32)
    x = np.stack([cur, opp], axis=0)
    return torch.from_numpy(x)


# -------------------------
# Network: policy + value
# -------------------------


class PolicyValueNet(nn.Module):
    """
    Small ConvNet for variable board sizes.

    Policy head outputs logits over columns (width actions).
    Value head outputs a scalar in [-1, 1] via tanh.
    """

    def __init__(self, *, cfg: ConnectKConfig, channels: int = 64):
        super().__init__()
        self.cfg = cfg

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
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(path: Path, *, device: torch.device) -> PolicyValueNet:
    payload = torch.load(path, map_location=device)
    cfg = ConnectKConfig(**payload["cfg"])
    cfg.validate()
    model = PolicyValueNet(cfg=cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


# -------------------------
# MCTS (PUCT) data structure
# -------------------------


def _dirichlet_noise(rng: np.random.Generator, alpha: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    x = rng.gamma(shape=alpha, scale=1.0, size=(n,))
    s = float(x.sum())
    if s == 0.0:
        return np.full((n,), 1.0 / n, dtype=np.float32)
    return (x / s).astype(np.float32)


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


class PUCTMCTS:
    """
    Neural-guided MCTS with PUCT selection and value network leaf evaluation.
    """

    def __init__(
        self,
        *,
        cfg: ConnectKConfig,
        model: PolicyValueNet,
        device: torch.device,
        sims: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_eps: float,
        seed: int,
    ):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng(seed)

        # Tree keyed by canonical board bytes. (Player-to-move is implicit via canonicalization.)
        self.tree: Dict[bytes, NodeStats] = {}

    def run(self, root: GameState) -> None:
        for _ in range(self.sims):
            self._simulate(root)

    def root_policy(self, root: GameState, *, temperature: float) -> np.ndarray:
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

        # π(a) ∝ N(a)^(1/T)
        inv_t = 1.0 / float(temperature)
        w = np.zeros((self.cfg.width,), dtype=np.float32)
        n = stats.N.astype(np.float32)
        w[legal_idx] = np.power(n[legal_idx], inv_t)
        s = float(w.sum())
        if s == 0.0:
            w[legal_idx] = 1.0 / len(legal_idx)
            return w
        return w / s

    def _simulate(self, root: GameState) -> None:
        cfg = self.cfg
        path: List[Tuple[bytes, int]] = []
        s = root

        while True:
            tv = terminal_value(cfg, s)
            if tv is not None:
                v = float(tv)
                break

            key = s.board.tobytes()
            stats = self.tree.get(key)
            if stats is None:
                stats, v = self._expand_and_eval(s, is_root=(s is root))
                self.tree[key] = stats
                break

            a = self._select_action(stats)
            path.append((key, a))
            s = apply_move(cfg, s, a)

        # Backup:
        # Each edge (s,a) is from the perspective of the player to move at s (canonical).
        # When we move to the next state, the "player to move" swaps; in canonical form
        # this is represented by flipping the board, so the value flips sign.
        for key, a in reversed(path):
            st = self.tree[key]
            st.N[a] += 1
            st.W[a] += v
            v = -v

    def _expand_and_eval(self, s: GameState, *, is_root: bool) -> Tuple[NodeStats, float]:
        cfg = self.cfg
        legal = legal_mask(cfg, s).astype(bool)

        with torch.no_grad():
            x = encode_state(cfg, s).unsqueeze(0).to(self.device)
            logits, value = self.model(x)
            logits = logits.squeeze(0)

            # Mask illegal actions by setting logits to -inf before softmax.
            mask = torch.tensor(legal, device=self.device)
            logits = logits.masked_fill(~mask, float("-inf"))
            priors = F.softmax(logits, dim=0)

        p = priors.detach().cpu().numpy().astype(np.float32)
        v = float(value.item())

        if is_root and self.dirichlet_eps > 0.0:
            legal_idx = np.nonzero(legal)[0]
            noise = _dirichlet_noise(self.rng, self.dirichlet_alpha, len(legal_idx))
            p2 = p.copy()
            for i, a in enumerate(legal_idx):
                p2[a] = (1.0 - self.dirichlet_eps) * p[a] + self.dirichlet_eps * noise[i]
            p = p2

        return NodeStats(
            P=p,
            N=np.zeros((cfg.width,), dtype=np.int32),
            W=np.zeros((cfg.width,), dtype=np.float32),
            legal=legal,
        ), v

    def _select_action(self, stats: NodeStats) -> int:
        # PUCT: maximize Q + U.
        n = stats.N.astype(np.float32)
        w = stats.W.astype(np.float32)
        q = np.zeros_like(w)
        nonzero = n > 0
        q[nonzero] = w[nonzero] / n[nonzero]

        sum_n = float(n.sum())
        u = self.c_puct * stats.P * math.sqrt(sum_n + 1e-8) / (1.0 + n)
        score = q + u
        score = np.where(stats.legal, score, -1e30)
        return int(score.argmax())


# -------------------------
# Self-play + training
# -------------------------


@dataclass
class Example:
    x: np.ndarray  # float32 (2,H,W)
    pi: np.ndarray  # float32 (W,)
    z: float


def _winner_from_start(s: GameState, tv: float) -> int:
    """
    Return winner sign (+1/-1/0) relative to the player who moved at ply=0.

    If terminal_value(tv) != 0, the game ended because the player who just moved won.
    Since s.ply counts moves already made:
      - if ply is odd: last move was by the starting player => winner = +1
      - if ply is even: last move was by the second player  => winner = -1
    """
    if tv == 0.0:
        return 0
    return +1 if (s.ply % 2 == 1) else -1


def _player_to_move_sign_from_start(ply: int) -> int:
    """At even ply, it's the starting player's turn; at odd ply, the second player's."""
    return +1 if (ply % 2 == 0) else -1


def play_self_game(
    *,
    cfg: ConnectKConfig,
    model: PolicyValueNet,
    device: torch.device,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temperature_moves: int,
    seed: int,
) -> Tuple[List[Example], int]:
    rng = random.Random(seed)
    s = initial_state(cfg)

    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []

    while True:
        tv = terminal_value(cfg, s)
        if tv is not None:
            winner = _winner_from_start(s, tv)
            examples: List[Example] = []
            for x, pi, ply in trajectory:
                if winner == 0:
                    z = 0.0
                else:
                    # z is from the perspective of the player to move at that state.
                    # If the starting player wins (+1), then at even ply z=+1, at odd ply z=-1.
                    z = float(winner * _player_to_move_sign_from_start(ply))
                examples.append(Example(x=x, pi=pi, z=z))
            return examples, winner

        mcts = PUCTMCTS(
            cfg=cfg,
            model=model,
            device=device,
            sims=sims,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            seed=rng.randrange(1 << 30),
        )
        mcts.run(s)

        temp = 1.0 if s.ply < temperature_moves else 1e-9
        pi = mcts.root_policy(s, temperature=temp)
        x = np.stack([(s.board == 1).astype(np.float32), (s.board == -1).astype(np.float32)], axis=0)
        trajectory.append((x, pi, s.ply))

        if temp <= 1e-8:
            a = int(pi.argmax())
        else:
            a = rng.choices(range(cfg.width), weights=pi.tolist(), k=1)[0]
        s = apply_move(cfg, s, a)


def train_step(
    *,
    model: PolicyValueNet,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[Example],
    value_loss_weight: float,
) -> Dict[str, float]:
    model.train()

    x = torch.tensor(np.stack([ex.x for ex in batch], axis=0), dtype=torch.float32, device=device)
    pi = torch.tensor(np.stack([ex.pi for ex in batch], axis=0), dtype=torch.float32, device=device)
    z = torch.tensor([ex.z for ex in batch], dtype=torch.float32, device=device)

    logits, v = model(x)
    logp = F.log_softmax(logits, dim=1)

    # Policy loss: cross-entropy for a *distributional* target π (from visit counts).
    policy_loss = -(pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z)
    loss = policy_loss + value_loss_weight * value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }


def pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@app.command()
def train(
    width: int = 7,
    height: int = 6,
    k: int = 4,
    model_out: Path = Path("connectk_model.pt"),
    seed: int = 0,
    device: str = "auto",
    channels: int = 64,
    sims: int = 200,
    cpuct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    temperature_moves: int = 10,
    iters: int = 10,
    games_per_iter: int = 10,
    replay_size: int = 10_000,
    train_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    value_loss_weight: float = 1.0,
    save_every: int = 1,
) -> None:
    """
    Self-play + training loop.

    A useful workflow is to start small (fewer sims, small board) then scale up.
    """
    cfg = ConnectKConfig(width=width, height=height, k=k)
    cfg.validate()
    dev = pick_device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    model = PolicyValueNet(cfg=cfg, channels=channels).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    replay: List[Example] = []

    for it in range(1, iters + 1):
        t0 = time.time()
        wins = {+1: 0, 0: 0, -1: 0}
        new_ex: List[Example] = []

        for _ in trange(games_per_iter, desc=f"self-play iter {it}", leave=False):
            examples, winner = play_self_game(
                cfg=cfg,
                model=model,
                device=dev,
                sims=sims,
                c_puct=cpuct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                temperature_moves=temperature_moves,
                seed=rng.randrange(1 << 30),
            )
            wins[winner] += 1
            new_ex.extend(examples)

        replay.extend(new_ex)
        if len(replay) > replay_size:
            replay = replay[-replay_size:]

        metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        steps = 0
        for _ in range(train_steps):
            if len(replay) < batch_size:
                break
            batch = [replay[rng.randrange(len(replay))] for _ in range(batch_size)]
            m = train_step(
                model=model,
                device=dev,
                optimizer=optimizer,
                batch=batch,
                value_loss_weight=value_loss_weight,
            )
            for k2 in metrics:
                metrics[k2] += m[k2]
            steps += 1
        if steps:
            for k2 in metrics:
                metrics[k2] /= steps

        dt = time.time() - t0
        console.print(
            f"iter={it} dt={dt:.1f}s wins(+1/0/-1)={wins[+1]}/{wins[0]}/{wins[-1]} "
            f"replay={len(replay)} loss={metrics['loss']:.3f} "
            f"pi={metrics['policy_loss']:.3f} v={metrics['value_loss']:.3f}"
        )

        if save_every and (it % save_every == 0):
            save_model(model_out, model)
            console.print(f"saved: {model_out}")

    save_model(model_out, model)
    console.print(f"saved: {model_out}")


def _print_root_stats(cfg: ConnectKConfig, stats: NodeStats) -> None:
    table = Table(title="Root stats (PUCT MCTS)")
    table.add_column("a(col)", justify="right")
    table.add_column("P(a)", justify="right")
    table.add_column("N(a)", justify="right")
    table.add_column("Q(a)", justify="right")
    for a in range(cfg.width):
        if not bool(stats.legal[a]):
            continue
        n = int(stats.N[a])
        q = float(stats.W[a] / n) if n > 0 else 0.0
        table.add_row(str(a), f"{float(stats.P[a]):.3f}", str(n), f"{q:.3f}")
    console.print(table)


@app.command()
def play(
    model_path: Path = Path("connectk_model.pt"),
    width: int = 7,
    height: int = 6,
    k: int = 4,
    device: str = "auto",
    sims: int = 400,
    cpuct: float = 1.5,
    human_first: bool = True,
    show_root_stats: bool = True,
) -> None:
    """
    Play against MCTS(model). Root Dirichlet noise is disabled for play.
    """
    cfg = ConnectKConfig(width=width, height=height, k=k)
    cfg.validate()
    dev = pick_device(device)

    if not model_path.exists():
        raise typer.BadParameter(f"model not found: {model_path}")
    model = load_model(model_path, device=dev)
    if model.cfg != cfg:
        raise typer.BadParameter(f"model cfg {model.cfg} does not match requested cfg {cfg}")

    s = initial_state(cfg)
    human_is_start = human_first

    def is_human_turn(ply: int) -> bool:
        # starting player moves at even ply
        return (ply % 2 == 0) if human_is_start else (ply % 2 == 1)

    while True:
        console.print(render_board(cfg, s))
        tv = terminal_value(cfg, s)
        if tv is not None:
            if tv == 0.0:
                console.print("draw")
                return
            winner = _winner_from_start(s, tv)
            human_sign = +1 if human_is_start else -1
            console.print("you win" if winner == human_sign else "you lose")
            return

        if is_human_turn(s.ply):
            legal = np.nonzero(legal_mask(cfg, s))[0].tolist()
            while True:
                raw = typer.prompt(f"your move {legal}")
                try:
                    col = int(raw)
                except ValueError:
                    continue
                if col in legal:
                    break
            s = apply_move(cfg, s, col)
            continue

        mcts = PUCTMCTS(
            cfg=cfg,
            model=model,
            device=dev,
            sims=sims,
            c_puct=cpuct,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.0,  # disable noise for play
            seed=0,
        )
        mcts.run(s)

        key = s.board.tobytes()
        root_stats = mcts.tree.get(key)
        if show_root_stats and root_stats is not None:
            _print_root_stats(cfg, root_stats)

        pi = mcts.root_policy(s, temperature=1e-9)
        a = int(pi.argmax())
        console.print(f"ai move: {a}")
        s = apply_move(cfg, s, a)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

