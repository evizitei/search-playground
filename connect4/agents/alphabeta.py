"""Alpha-beta (minimax) agent with a simple heuristic evaluation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np

from connect4.agents.base import Agent
from connect4.engine import Connect4Config, GameState, apply_move, legal_moves, terminal_result


class AlphaBetaAgent(Agent):
    """
    Educational alpha-beta agent.

    The key ideas are intentionally spelled out in comments so the search can be
    followed with pencil-and-paper:
      - maximize for the root player, minimize for the opponent
      - prune with alpha/beta when a branch cannot change the final decision
      - stop at a depth/node budget and fall back to a heuristic evaluation
    """

    def __init__(
        self,
        name: str,
        *,
        max_depth: int = 4,
        max_nodes: Optional[int] = None,
    ) -> None:
        self.name = name
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._nodes = 0

    def select_move(self, cfg: Connect4Config, s: GameState) -> int:
        legal = legal_moves(cfg, s).tolist()
        if not legal:
            raise ValueError("no legal moves available")

        root_player = s.current_player
        best_score = -math.inf
        best_move = legal[0]
        self._nodes = 0

        # Move ordering improves alpha-beta pruning. We try center columns first.
        for col in _ordered_moves(cfg, legal):
            child = apply_move(cfg, s, col)
            score = self._search(
                cfg,
                child,
                depth=1,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
            )
            if score > best_score:
                best_score = score
                best_move = col

            if self._budget_exhausted():
                break

        return best_move

    def _search(
        self,
        cfg: Connect4Config,
        s: GameState,
        *,
        depth: int,
        alpha: float,
        beta: float,
        root_player: int,
    ) -> float:
        # Count every visited node so the caller can enforce a hard node budget.
        self._nodes += 1

        # If the node budget is exhausted, return a static evaluation immediately.
        if self._budget_exhausted():
            return _evaluate(cfg, s, root_player)

        tr = terminal_result(cfg, s)
        if tr.is_terminal:
            if tr.winner == 0:
                return 0.0
            return 1_000_000.0 if tr.winner == root_player else -1_000_000.0

        # Depth-limited search: use heuristic evaluation at the frontier.
        if depth >= self.max_depth:
            return _evaluate(cfg, s, root_player)

        legal = legal_moves(cfg, s).tolist()
        if not legal:
            return 0.0

        maximizing = s.current_player == root_player
        if maximizing:
            value = -math.inf
            for col in _ordered_moves(cfg, legal):
                child = apply_move(cfg, s, col)
                value = max(
                    value,
                    self._search(cfg, child, depth=depth + 1, alpha=alpha, beta=beta, root_player=root_player),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # beta cut-off
            return value

        value = math.inf
        for col in _ordered_moves(cfg, legal):
            child = apply_move(cfg, s, col)
            value = min(
                value,
                self._search(cfg, child, depth=depth + 1, alpha=alpha, beta=beta, root_player=root_player),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break  # alpha cut-off
        return value

    def _budget_exhausted(self) -> bool:
        return self.max_nodes is not None and self._nodes >= self.max_nodes


def _ordered_moves(cfg: Connect4Config, legal: List[int]) -> Iterable[int]:
    center = (cfg.width - 1) / 2.0
    return sorted(legal, key=lambda c: abs(c - center))


def _evaluate(cfg: Connect4Config, s: GameState, root_player: int) -> float:
    """
    Heuristic evaluation for a non-terminal position.

    The score is computed by scanning every length-K "window" on the board.
    - Windows containing both players are neutral (blocked).
    - Windows with only root player's stones are positive.
    - Windows with only opponent stones are negative.

    This keeps the evaluation symmetric and makes the sign easy to reason about.
    """

    weights: Dict[int, int] = {1: 1, 2: 5, 3: 25}
    score = 0.0

    # Small center-column bias: strong lines tend to go through the middle.
    center_col = cfg.width // 2
    center = s.board[:, center_col]
    score += 2.0 * int(np.sum(center == root_player))
    score -= 2.0 * int(np.sum(center == -root_player))

    for window in _iter_windows(cfg, s.board):
        score += _score_window(window, root_player, weights)

    return score


def _iter_windows(cfg: Connect4Config, board: np.ndarray) -> Iterable[np.ndarray]:
    k = cfg.k

    # Horizontal windows.
    for r in range(cfg.height):
        for c in range(cfg.width - k + 1):
            yield board[r, c : c + k]

    # Vertical windows.
    for c in range(cfg.width):
        for r in range(cfg.height - k + 1):
            yield board[r : r + k, c]

    # Diagonal (bottom-left to top-right).
    for r in range(cfg.height - k + 1):
        for c in range(cfg.width - k + 1):
            yield np.array([board[r + i, c + i] for i in range(k)])

    # Diagonal (top-left to bottom-right).
    for r in range(k - 1, cfg.height):
        for c in range(cfg.width - k + 1):
            yield np.array([board[r - i, c + i] for i in range(k)])


def _score_window(window: np.ndarray, root_player: int, weights: Dict[int, int]) -> float:
    root_count = int(np.sum(window == root_player))
    opp_count = int(np.sum(window == -root_player))
    if root_count > 0 and opp_count > 0:
        return 0.0
    if root_count > 0:
        return float(weights.get(root_count, 0))
    if opp_count > 0:
        return float(-weights.get(opp_count, 0))
    return 0.0
