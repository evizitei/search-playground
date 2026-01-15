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

## Connect‑4 (6x6) CLI + agents

The classic gravity‑bound Connect‑4 on a 6x6 board, with pluggable agents:
`human`, `random`, `alphabeta` (minimax with alpha‑beta pruning), and `alphazero`
(policy/value network + PUCT MCTS).

Run (human vs alpha‑beta):

```bash
uv run python connect4_cli.py --x human --o alphabeta --ab-depth 5
```

Random vs random (different seeds by default):

```bash
uv run python connect4_cli.py --x random --o random --seed 42
```

Alpha‑beta vs alpha‑beta with different budgets:

```bash
uv run python connect4_cli.py --x alphabeta --o alphabeta --ab-depth-x 4 --ab-depth-o 6 --ab-nodes 2000
```

Agent flags:
- `--x` / `--o`: `human | random | alphabeta | alphazero`
- `--seed`, `--seed-x`, `--seed-o`: RNG control for random agents
- `--ab-depth`, `--ab-depth-x`, `--ab-depth-o`: depth in plies
- `--ab-nodes`, `--ab-nodes-x`, `--ab-nodes-o`: node budget limit
- `--az-model`, `--az-model-x`, `--az-model-o`: AlphaZero model path(s)
- `--az-sims`, `--az-cpuct`: AlphaZero MCTS parameters

## Connect‑4 AlphaZero-style self-play training

Train a policy/value network via self-play with PUCT MCTS. Checkpoints are saved
every N self-play games so you can load different training stages later.

Example (small, fast-ish defaults):

```bash
uv run python connect4_az_train.py --iters 5 --games-per-iter 5 --sims 100 --checkpoint-every-games 10
```

Useful knobs:
- `--sims`: MCTS simulations per move
- `--cpuct`: exploration constant in PUCT
- `--dirichlet-alpha`, `--dirichlet-eps`: exploration noise at root
- `--temp-moves`: plies using temperature=1 before switching to greedy
- `--replay-size`: replay buffer size
- `--train-steps`: gradient steps per iteration
- `--batch-size`: training batch size
- `--checkpoint-every-games`: save model every N self-play games
- `--eval-games`: evaluation games per iter vs low-budget alpha-beta
- `--eval-ab-depth`, `--eval-ab-nodes`: alpha-beta strength for evaluation
- `--eval-sims`: MCTS sims per move for the eval agent

Play against a trained model:

```bash
uv run python connect4_cli.py --x human --o alphazero --az-model connect4_az_models/connect4_az_latest.pt
```

Quick diagnostics to sanity-check training:
- Run a tiny loop (`--iters 1 --games-per-iter 1 --sims 10`) and verify the run completes.
- Check that policies sum to 1 for legal moves by printing `pi.sum()` in `connect4/az/train.py`.
- Confirm the CLI can load a checkpoint and play a full game without errors.
