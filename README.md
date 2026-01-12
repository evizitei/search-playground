# MCTS / UCT playground: 4-coloring a planar graph

This repo contains a small, dependency-free implementation of Monte Carlo Tree Search (MCTS) with the UCT selection rule, applied to a simple constraint problem: **4-coloring a planar graph**.

The “map” is a planar graph generated from an `N x N` grid with one random diagonal per cell (a planar triangulation-ish graph). The solver tries to assign one of 4 colors to each vertex so that adjacent vertices never share a color.

## Run

```bash
python3 mcts_map_coloring.py --size 8 --seed 0 --iterations 50000 --rollout random
```

To make the search meaningfully harder (and watch MCTS struggle), try fewer colors:

```bash
python3 mcts_map_coloring.py --size 8 --seed 0 --colors 3 --iterations 200000 --inspect-root --inspect-pv 10 --no-render
```

More options:

```bash
python3 mcts_map_coloring.py --help
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

## Connect-K AlphaZero-lite (neural-guided MCTS)

`connectk_azlite.py` is a dependency-free, educational implementation of:
- a tiny policy+value network (manual backprop + Adam)
- PUCT MCTS guided by that network
- a minimal self-play training loop that learns from `(π, z)` targets (visit-count policy + final outcome)

Train (small and fast-ish defaults):

```bash
python3 connectk_azlite.py train --width 5 --height 4 --k 4 --iters 5 --games-per-iter 5 --sims 50
```

Play against the latest saved model:

```bash
python3 connectk_azlite.py play --width 5 --height 4 --k 4 --model connectk_model.json --no-dirichlet
```
