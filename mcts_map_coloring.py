#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import typer
from rich.console import Console


app = typer.Typer(no_args_is_help=True)
console = Console()


Color = int  # 0..3
Vertex = int
Prefix = Tuple[Color, ...]


def build_planar_grid_graph(n: int, *, seed: int) -> List[List[Vertex]]:
    """
    Build a simple planar graph on an n x n grid of vertices, with random diagonals
    added to each cell (triangulation-ish). This is always planar and thus 4-colorable.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    rng = random.Random(seed)
    vcount = n * n
    adj: List[set[int]] = [set() for _ in range(vcount)]

    def vid(r: int, c: int) -> int:
        return r * n + c

    def add_edge(a: int, b: int) -> None:
        if a == b:
            return
        adj[a].add(b)
        adj[b].add(a)

    # 4-neighborhood grid edges.
    for r in range(n):
        for c in range(n):
            if r + 1 < n:
                add_edge(vid(r, c), vid(r + 1, c))
            if c + 1 < n:
                add_edge(vid(r, c), vid(r, c + 1))

    # Add exactly one diagonal per cell to avoid crossings.
    for r in range(n - 1):
        for c in range(n - 1):
            if rng.random() < 0.5:
                add_edge(vid(r, c), vid(r + 1, c + 1))
            else:
                add_edge(vid(r + 1, c), vid(r, c + 1))

    return [sorted(neis) for neis in adj]


def order_vertices_by_degree(adj: Sequence[Sequence[Vertex]]) -> List[Vertex]:
    return sorted(range(len(adj)), key=lambda v: len(adj[v]), reverse=True)


def compute_pos_in_order(order: Sequence[Vertex]) -> List[int]:
    pos = [-1] * len(order)
    for i, v in enumerate(order):
        pos[v] = i
    return pos


def legal_colors_for_next_vertex(
    *,
    adj: Sequence[Sequence[Vertex]],
    order: Sequence[Vertex],
    pos_in_order: Sequence[int],
    prefix: Prefix,
    n_colors: int,
) -> List[Color]:
    depth = len(prefix)
    v = order[depth]
    used = set()
    for nei in adj[v]:
        p = pos_in_order[nei]
        if 0 <= p < depth:
            used.add(prefix[p])
    return [c for c in range(n_colors) if c not in used]


def rollout(
    *,
    adj: Sequence[Sequence[Vertex]],
    order: Sequence[Vertex],
    pos_in_order: Sequence[int],
    prefix: Prefix,
    n_colors: int,
    rng: random.Random,
    policy: str,
) -> Tuple[float, Prefix]:
    """
    Returns (reward, completed_prefix).
    reward is in [0, 1]. 1.0 means a full proper coloring was completed.
    """
    colors: List[Color] = list(prefix)
    vcount = len(order)

    while len(colors) < vcount:
        depth = len(colors)
        v = order[depth]

        legal = legal_colors_for_next_vertex(
            adj=adj,
            order=order,
            pos_in_order=pos_in_order,
            prefix=tuple(colors),
            n_colors=n_colors,
        )
        if not legal:
            return len(colors) / vcount, tuple(colors)

        if policy == "random":
            choice = rng.choice(legal)
        elif policy == "greedy":
            # Prefer colors that remove the fewest options from future neighbors.
            # Cheap approximation: for each future neighbor u, see if choosing `color`
            # would newly forbid `color` (i.e., `color` isn't already used among u's
            # assigned neighbors).
            cur_prefix = tuple(colors)

            def score(color: Color) -> int:
                newly_forbidden = 0
                for u in adj[v]:
                    pu = pos_in_order[u]
                    if pu <= depth:
                        continue  # already assigned or current vertex
                    used_u = set()
                    for w in adj[u]:
                        pw = pos_in_order[w]
                        if 0 <= pw < depth:
                            used_u.add(cur_prefix[pw])
                    if color not in used_u:
                        newly_forbidden += 1
                return newly_forbidden

            choice = min(legal, key=score)
        else:
            raise ValueError(f"unknown rollout policy: {policy}")

        colors.append(choice)

    return 1.0, tuple(colors)


@dataclass
class Node:
    prefix: Prefix
    parent: Optional["Node"] = None
    action_from_parent: Optional[Color] = None
    children: Dict[Color, "Node"] = field(default_factory=dict)
    untried_actions: Optional[List[Color]] = None
    visits: int = 0
    value_sum: float = 0.0

    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


def uct_select_child(node: Node, *, exploration_c: float) -> Node:
    assert node.visits > 0
    log_parent = math.log(node.visits)

    def uct(child: Node) -> float:
        if child.visits == 0:
            return float("inf")
        return child.q() + exploration_c * math.sqrt(log_parent / child.visits)

    return max(node.children.values(), key=uct)


def mcts_search(
    *,
    adj: Sequence[Sequence[Vertex]],
    n_colors: int,
    iterations: int,
    exploration_c: float,
    seed: int,
    rollout_policy: str,
    order: Sequence[Vertex],
    verbose_every: int,
    continue_after_solve: bool,
) -> Tuple[float, Prefix, Node]:
    rng = random.Random(seed)
    pos_in_order = compute_pos_in_order(order)
    root = Node(prefix=tuple())

    best_reward = 0.0
    best_prefix: Prefix = tuple()

    vcount = len(order)
    start = time.time()

    for it in range(1, iterations + 1):
        node = root

        # Selection
        while True:
            if len(node.prefix) >= vcount:
                break
            if node.untried_actions is None:
                node.untried_actions = legal_colors_for_next_vertex(
                    adj=adj,
                    order=order,
                    pos_in_order=pos_in_order,
                    prefix=node.prefix,
                    n_colors=n_colors,
                )
                rng.shuffle(node.untried_actions)
            if node.untried_actions:
                break
            if not node.children:
                break
            node = uct_select_child(node, exploration_c=exploration_c)

        # Expansion (one step)
        if len(node.prefix) < vcount:
            if node.untried_actions is None:
                node.untried_actions = legal_colors_for_next_vertex(
                    adj=adj,
                    order=order,
                    pos_in_order=pos_in_order,
                    prefix=node.prefix,
                    n_colors=n_colors,
                )
                rng.shuffle(node.untried_actions)
            if node.untried_actions:
                action = node.untried_actions.pop()
                child_prefix = node.prefix + (action,)
                child = node.children.get(action)
                if child is None:
                    child = Node(prefix=child_prefix, parent=node, action_from_parent=action)
                    node.children[action] = child
                node = child

        # Simulation
        reward, completed_prefix = rollout(
            adj=adj,
            order=order,
            pos_in_order=pos_in_order,
            prefix=node.prefix,
            n_colors=n_colors,
            rng=rng,
            policy=rollout_policy,
        )
        if reward > best_reward:
            best_reward = reward
            best_prefix = completed_prefix

        # Backpropagation
        cur: Optional[Node] = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            cur = cur.parent

        if verbose_every and (it % verbose_every == 0 or reward == 1.0):
            elapsed = time.time() - start
            console.print(
                f"iter={it} best={best_reward:.3f} root_q={root.q():.3f} "
                f"root_children={len(root.children)} elapsed={elapsed:.2f}s"
            )
        if reward == 1.0 and not continue_after_solve:
            break

    return best_reward, best_prefix, root


def prefix_to_assignment(prefix: Prefix, order: Sequence[Vertex], vcount: int) -> List[int]:
    colors = [-1] * vcount
    for i, c in enumerate(prefix):
        colors[order[i]] = c
    return colors


def verify_coloring(adj: Sequence[Sequence[Vertex]], colors: Sequence[int], n_colors: int) -> None:
    for v, cv in enumerate(colors):
        if cv < 0 or cv >= n_colors:
            raise ValueError("uncolored/invalid vertex present")
        for nei in adj[v]:
            if colors[nei] == cv:
                raise ValueError(f"invalid coloring: {v} and {nei} share color {cv}")


def render_grid(colors: Sequence[int], n: int) -> str:
    # Map 0..3 to simple glyphs.
    glyph = {0: "A", 1: "B", 2: "C", 3: "D"}
    lines: List[str] = []
    for r in range(n):
        row = []
        for c in range(n):
            v = r * n + c
            row.append(glyph.get(colors[v], "?"))
        lines.append(" ".join(row))
    return "\n".join(lines)


def format_vertex(v: int, n: int) -> str:
    return f"{v}({v // n},{v % n})"


def print_principal_variation(root: Node, *, order: Sequence[Vertex], n: int, max_depth: int) -> None:
    node = root
    depth = 0
    while node.children and depth < max_depth:
        child = max(node.children.values(), key=lambda ch: ch.visits)
        v = order[depth]
        color = child.action_from_parent
        console.print(
            f"pv depth={depth:02d} v={format_vertex(v, n)} color={color} "
            f"visits={child.visits} q={child.q():.3f}"
        )
        node = child
        depth += 1


def print_root_children(root: Node) -> None:
    if not root.children:
        console.print("root has no children (no legal moves at depth 0)")
        return
    items = sorted(root.children.items(), key=lambda kv: kv[1].visits, reverse=True)
    console.print("root children (color -> visits, q):")
    for color, child in items:
        console.print(f"  {color} -> {child.visits}, {child.q():.3f}")

@app.command()
def solve(
    size: int = typer.Option(6, help="Grid size N (graph has N*N vertices)."),
    seed: int = typer.Option(0, help="Random seed (graph + MCTS)."),
    iterations: int = typer.Option(50_000, help="Max MCTS iterations."),
    c: float = typer.Option(1.4, help="UCT exploration constant."),
    colors: int = typer.Option(4, help="Number of available colors."),
    order: str = typer.Option("degree", help="Vertex ordering: degree|natural."),
    rollout: str = typer.Option("random", help="Rollout policy: random|greedy."),
    verbose_every: int = typer.Option(1_000, help="Print progress every N iterations (0 disables)."),
    continue_after_solve: bool = typer.Option(
        False,
        "--continue-after-solve",
        help="Keep searching even after a valid coloring is found (useful for inspection).",
        is_flag=True,
    ),
    inspect_root: bool = typer.Option(
        False, "--inspect-root", help="Print root child stats after the search.", is_flag=True
    ),
    inspect_pv: int = typer.Option(0, help="Print the principal variation up to DEPTH steps."),
    print_assignment: Optional[str] = typer.Option(
        None, help="Print vertex->color assignment: list|json (in addition to the grid, if enabled)."
    ),
    no_render: bool = typer.Option(False, "--no-render", help="Do not print the final grid coloring.", is_flag=True),
) -> None:
    if order not in {"degree", "natural"}:
        raise typer.BadParameter("order must be 'degree' or 'natural'")
    if rollout not in {"random", "greedy"}:
        raise typer.BadParameter("rollout must be 'random' or 'greedy'")
    if print_assignment not in {None, "list", "json"}:
        raise typer.BadParameter("print-assignment must be 'list' or 'json'")

    adj = build_planar_grid_graph(size, seed=seed)
    vertex_order = order_vertices_by_degree(adj) if order == "degree" else list(range(len(adj)))

    best_reward, best_prefix, root = mcts_search(
        adj=adj,
        n_colors=colors,
        iterations=iterations,
        exploration_c=c,
        seed=seed,
        rollout_policy=rollout,
        order=vertex_order,
        verbose_every=verbose_every,
        continue_after_solve=continue_after_solve,
    )

    vcount = size * size
    if inspect_root:
        print_root_children(root)
    if inspect_pv:
        print_principal_variation(root, order=vertex_order, n=size, max_depth=inspect_pv)

    if best_reward == 1.0:
        assignment = prefix_to_assignment(best_prefix, vertex_order, vcount)
        verify_coloring(adj, assignment, colors)
        console.print(f"success: {colors}-coloring found for {size}x{size} in <= {iterations} iterations")
        if print_assignment == "list":
            for v, col in enumerate(assignment):
                console.print(f"v={format_vertex(v, size)} color={col}")
        elif print_assignment == "json":
            payload = {"size": size, "colors": colors, "seed": seed, "assignment": {str(v): int(col) for v, col in enumerate(assignment)}}
            console.print(json.dumps(payload, indent=2, sort_keys=True))
        if not no_render:
            console.print(render_grid(assignment, size))
        return

    console.print(
        f"failed: best partial coloring reached {best_reward:.3f} of vertices "
        f"({int(best_reward * vcount)}/{vcount})"
    )
    raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
