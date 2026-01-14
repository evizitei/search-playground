# Search playgrounds: MCTS + neural-guided MCTS

This repo contains small, educational implementations of:
- Monte Carlo Tree Search (MCTS) with UCT on a constraint problem (map coloring)
- Neural-guided MCTS (AlphaZero-lite: policy+value net + PUCT) on Connect‑K

## Setup (pyenv + uv)

This repo pins a stable Python via `pyenv` and uses `uv` for dependency management.

```bash
pyenv install -s 3.12.12
pyenv local 3.12.12
uv sync
```

## Map coloring (MCTS / UCT)

The “map” is a planar graph generated from an `N x N` grid with one random diagonal per cell (a planar triangulation-ish graph). The solver tries to assign colors so that adjacent vertices never share a color.

Run:

```bash
uv run python mcts_map_coloring.py --size 8 --seed 0 --iterations 50000 --rollout random
```

To make the search meaningfully harder (and watch MCTS struggle), try fewer colors:

```bash
uv run python mcts_map_coloring.py --size 8 --seed 0 --colors 3 --iterations 200000 --inspect-root --inspect-pv 10 --no-render
```

More options:

```bash
uv run python mcts_map_coloring.py --help
```

Useful tweaks:
- `--size N`: makes the graph larger (`N*N` vertices).
- `--iterations K`: more MCTS iterations (more search).
- `--c 1.4`: exploration constant in UCT.
- `--rollout random|greedy`: simulation policy.
- `--verbose-every 1000`: prints MCTS progress; set `0` to disable.
- `--inspect-root`: prints root child visit/Q stats (helps “see” UCT).
- `--inspect-pv DEPTH`: prints the most-visited path through the tree.
- `--continue-after-solve`: keeps searching after success for inspection.
- `--print-assignment list|json`: prints the vertex→color mapping.

## What to look at

The implementation is intentionally “vanilla”:
- **State**: a prefix assignment of colors in a fixed vertex order.
- **Actions**: pick a legal color for the next vertex.
- **Rollout**: extend the partial assignment with a simple policy until success or dead-end.
- **Reward**: `1.0` if a full coloring is found; otherwise fraction of vertices colored before failing.
- **UCT**: selects children by `Q/N + c * sqrt(log(N_parent)/N_child)`.

Main entry points:
- `mcts_search()` in `mcts_map_coloring.py`
- `uct_select_child()` in `mcts_map_coloring.py`
- `rollout()` in `mcts_map_coloring.py`

## Connect‑K AlphaZero‑lite (neural-guided MCTS)

`connectk_azlite.py` is an educational implementation of:
- a policy+value network in PyTorch
- PUCT MCTS guided by that network
- a minimal self-play training loop that learns from `(π, z)` targets (visit-count policy + final outcome)

Train (small and fast-ish defaults):

```bash
uv run python connectk_azlite.py train --width 5 --height 4 --k 4 --iters 5 --games-per-iter 5 --sims 50
```

Play against the latest saved model:

```bash
uv run python connectk_azlite.py play --width 5 --height 4 --k 4 --model-path connectk_model.pt
```
