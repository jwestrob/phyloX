from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .tree import PhyloTree


@dataclass(frozen=True)
class NNIMove:
    edge: tuple[int, int]
    swap: tuple[int, int]
    delta_log_likelihood: float
    new_log_likelihood: float


@dataclass(frozen=True)
class SearchIteration:
    step: str
    iteration: int
    log_likelihood: float


@dataclass(frozen=True)
class TopologySearchResult:
    tree: PhyloTree
    log_likelihood: float
    history: tuple[SearchIteration, ...]


def enumerate_nni_moves(
    tree: PhyloTree,
    score_fn: Callable[[PhyloTree], float],
    current_score: float | None = None,
) -> list[NNIMove]:
    """
    Evaluate best NNI alternative for each internal binary edge.
    """
    base_score = float(score_fn(tree)) if current_score is None else float(current_score)
    moves: list[NNIMove] = []
    for edge_idx in tree.internal_edge_indices():
        u, v, _ = tree.edges[edge_idx]
        alts = nni_alternatives(tree, u, v)
        best_delta = -np.inf
        best_tree: PhyloTree | None = None
        best_swap: tuple[int, int] | None = None
        best_score = base_score
        for swap_u, swap_v, t_alt in alts:
            s = float(score_fn(t_alt))
            d = s - base_score
            if d > best_delta:
                best_delta = d
                best_tree = t_alt
                best_swap = (swap_u, swap_v)
                best_score = s
        if best_tree is not None and best_swap is not None and best_delta > 0:
            moves.append(
                NNIMove(
                    edge=(u, v),
                    swap=best_swap,
                    delta_log_likelihood=float(best_delta),
                    new_log_likelihood=float(best_score),
                )
            )
    moves.sort(key=lambda x: x.delta_log_likelihood, reverse=True)
    return moves


def nni_hillclimb(
    tree: PhyloTree,
    score_fn: Callable[[PhyloTree], float],
    max_rounds: int = 20,
    edge_disjoint_batch: bool = True,
    min_delta: float = 1e-9,
) -> TopologySearchResult:
    cur_tree = tree
    cur_score = float(score_fn(cur_tree))
    history = [SearchIteration(step="init", iteration=0, log_likelihood=cur_score)]

    for it in range(1, max_rounds + 1):
        moves = enumerate_nni_moves(cur_tree, score_fn, current_score=cur_score)
        moves = [m for m in moves if m.delta_log_likelihood > min_delta]
        if not moves:
            break

        selected = _select_edge_disjoint_moves(moves) if edge_disjoint_batch else [moves[0]]
        applied = False
        trial_tree = cur_tree
        for mv in selected:
            u, v = mv.edge
            su, sv = mv.swap
            try:
                trial_tree = apply_nni_swap(trial_tree, u, v, su, sv)
                applied = True
            except ValueError:
                continue

        if not applied:
            break
        trial_score = float(score_fn(trial_tree))
        if trial_score <= cur_score + min_delta:
            break
        cur_tree = trial_tree
        cur_score = trial_score
        history.append(SearchIteration(step="nni", iteration=it, log_likelihood=cur_score))

    return TopologySearchResult(tree=cur_tree, log_likelihood=cur_score, history=tuple(history))


def spr_escape_move(
    tree: PhyloTree,
    score_fn: Callable[[PhyloTree], float],
    current_score: float | None = None,
    max_radius: int = 6,
    max_candidates_per_prune: int = 20,
) -> tuple[PhyloTree, float]:
    """
    Limited-radius SPR: evaluate candidate regrafts for oriented prune edges.
    """
    base_score = float(score_fn(tree)) if current_score is None else float(current_score)
    best_tree = tree
    best_score = base_score

    deg = tree.degrees()
    node_dist = _all_pairs_node_distances(tree)

    for a in range(tree.num_nodes):
        if deg[a] < 3:
            continue
        neigh = [n for n, _ in tree.neighbors(a)]
        for s in neigh:
            subtree_nodes = _subtree_nodes_from_directed_edge(tree, parent=a, child=s)
            candidates = []
            for u, v, _ in tree.edges:
                if u == a or v == a:
                    continue
                if (u in subtree_nodes) or (v in subtree_nodes):
                    continue
                radius = min(node_dist[a, u], node_dist[a, v])
                if radius <= max_radius:
                    candidates.append((radius, u, v))
            candidates.sort(key=lambda x: x[0])
            for _, u, v in candidates[:max_candidates_per_prune]:
                t_new = apply_spr_move(tree, attach_node=a, subtree_root=s, target_edge=(u, v))
                s_new = float(score_fn(t_new))
                if s_new > best_score:
                    best_tree = t_new
                    best_score = s_new

    return best_tree, best_score


def nni_with_optional_spr(
    tree: PhyloTree,
    score_fn: Callable[[PhyloTree], float],
    outer_rounds: int = 5,
    nni_rounds: int = 8,
    use_spr: bool = True,
) -> TopologySearchResult:
    cur_tree = tree
    cur_score = float(score_fn(cur_tree))
    history = [SearchIteration(step="init", iteration=0, log_likelihood=cur_score)]

    for i in range(1, outer_rounds + 1):
        nni = nni_hillclimb(cur_tree, score_fn, max_rounds=nni_rounds, edge_disjoint_batch=True)
        cur_tree = nni.tree
        cur_score = nni.log_likelihood
        history.append(SearchIteration(step="nni", iteration=i, log_likelihood=cur_score))

        if use_spr:
            spr_tree, spr_score = spr_escape_move(cur_tree, score_fn, current_score=cur_score)
            if spr_score > cur_score:
                cur_tree = spr_tree
                cur_score = spr_score
                history.append(SearchIteration(step="spr", iteration=i, log_likelihood=cur_score))
            else:
                break
        else:
            if i > 1 and abs(history[-1].log_likelihood - history[-2].log_likelihood) < 1e-9:
                break

    return TopologySearchResult(tree=cur_tree, log_likelihood=cur_score, history=tuple(history))


def nni_alternatives(tree: PhyloTree, u: int, v: int) -> list[tuple[int, int, PhyloTree]]:
    u_n = [x for x, _ in tree.neighbors(u) if x != v]
    v_n = [x for x, _ in tree.neighbors(v) if x != u]
    if len(u_n) < 2 or len(v_n) < 2:
        return []

    # For bifurcating trees this yields the two unique alternatives.
    s0 = u_n[0]
    out: list[tuple[int, int, PhyloTree]] = []
    for sv in v_n:
        t_alt = apply_nni_swap(tree, u, v, s0, sv)
        out.append((s0, sv, t_alt))
    # Deduplicate possible duplicates in non-binary cases.
    uniq: list[tuple[int, int, PhyloTree]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for su, sv, t in out:
        key = tuple(sorted((min(a, b), max(a, b)) for a, b, _ in t.edges))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((su, sv, t))
    return uniq


def apply_nni_swap(tree: PhyloTree, u: int, v: int, swap_u: int, swap_v: int) -> PhyloTree:
    """
    Swap the subtrees attached by edges (u, swap_u) and (v, swap_v).
    """
    key_uv = _edge_key(u, swap_u)
    key_vv = _edge_key(v, swap_v)
    edge_lengths = {_edge_key(a, b): float(t) for a, b, t in tree.edges}
    if key_uv not in edge_lengths or key_vv not in edge_lengths:
        raise ValueError("invalid NNI swap edges")
    l_u = edge_lengths[key_uv]
    l_v = edge_lengths[key_vv]

    new_edges: list[tuple[int, int, float]] = []
    for a, b, t in tree.edges:
        k = _edge_key(a, b)
        if k == key_uv or k == key_vv:
            continue
        new_edges.append((a, b, float(t)))
    new_edges.append((u, swap_v, l_v))
    new_edges.append((v, swap_u, l_u))
    return PhyloTree(num_nodes=tree.num_nodes, edges=new_edges, leaf_count=tree.leaf_count)


def apply_spr_move(
    tree: PhyloTree,
    attach_node: int,
    subtree_root: int,
    target_edge: tuple[int, int],
) -> PhyloTree:
    """
    Perform a single SPR move while preserving node count:
    - cut oriented edge attach_node -> subtree_root
    - suppress attach_node in backbone
    - regraft at target_edge by reusing attach_node as subdivision vertex
    """
    u, v = target_edge
    if attach_node == subtree_root:
        raise ValueError("attach_node and subtree_root must differ")
    if attach_node == u or attach_node == v:
        raise ValueError("target edge must not touch attach_node")
    if subtree_root == u or subtree_root == v:
        raise ValueError("target edge must not touch subtree_root")

    neigh = tree.neighbors(attach_node)
    others = [n for n, _ in neigh if n != subtree_root]
    if len(others) != 2:
        raise ValueError("attach_node must have exactly two other neighbors for SPR")
    x, y = others

    lengths = {_edge_key(a, b): float(t) for a, b, t in tree.edges}
    k_as = _edge_key(attach_node, subtree_root)
    k_ax = _edge_key(attach_node, x)
    k_ay = _edge_key(attach_node, y)
    k_uv = _edge_key(u, v)
    for key in (k_as, k_ax, k_ay, k_uv):
        if key not in lengths:
            raise ValueError("required edge missing")

    t_as = lengths[k_as]
    t_ax = lengths[k_ax]
    t_ay = lengths[k_ay]
    t_uv = lengths[k_uv]

    remove_keys = {k_as, k_ax, k_ay, k_uv}
    new_edges: list[tuple[int, int, float]] = []
    for a, b, t in tree.edges:
        if _edge_key(a, b) in remove_keys:
            continue
        new_edges.append((a, b, float(t)))

    new_edges.append((x, y, t_ax + t_ay))
    new_edges.append((attach_node, u, t_uv * 0.5))
    new_edges.append((attach_node, v, t_uv * 0.5))
    new_edges.append((attach_node, subtree_root, t_as))

    return PhyloTree(num_nodes=tree.num_nodes, edges=new_edges, leaf_count=tree.leaf_count)


def _select_edge_disjoint_moves(moves: list[NNIMove]) -> list[NNIMove]:
    selected: list[NNIMove] = []
    used_nodes: set[int] = set()
    for mv in moves:
        u, v = mv.edge
        su, sv = mv.swap
        touched = {u, v, su, sv}
        if any(x in used_nodes for x in touched):
            continue
        selected.append(mv)
        used_nodes.update(touched)
    return selected


def _subtree_nodes_from_directed_edge(tree: PhyloTree, parent: int, child: int) -> set[int]:
    stack = [child]
    seen = {parent}
    out: set[int] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        out.add(node)
        for nxt, _ in tree.neighbors(node):
            if nxt not in seen:
                stack.append(nxt)
    return out


def _all_pairs_node_distances(tree: PhyloTree) -> np.ndarray:
    n = tree.num_nodes
    out = np.full((n, n), np.inf, dtype=np.float64)
    adj = tree.adjacency()
    for src in range(n):
        out[src, src] = 0.0
        q = [src]
        head = 0
        while head < len(q):
            node = q[head]
            head += 1
            for nxt, _ in adj[node]:
                if np.isfinite(out[src, nxt]):
                    continue
                out[src, nxt] = out[src, node] + 1.0
                q.append(nxt)
    return out


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)
