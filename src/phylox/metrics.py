from __future__ import annotations

import itertools

import numpy as np

from .tree import PhyloTree, pairwise_leaf_distances


def bipartition_splits(tree: PhyloTree) -> set[frozenset[int]]:
    """
    Set of canonical non-trivial leaf bipartitions induced by edges.
    """
    n = tree.leaf_count
    splits: set[frozenset[int]] = set()
    for u, v, _ in tree.edges:
        side = _leaf_component(tree, start=u, blocked=v)
        if len(side) == 0 or len(side) == n:
            continue
        if len(side) == 1 or len(side) == n - 1:
            continue
        comp = set(range(n)) - side
        canon = _canonical_split(side, comp)
        splits.add(frozenset(canon))
    return splits


def rf_distance(tree_a: PhyloTree, tree_b: PhyloTree, normalize: bool = False) -> float:
    if tree_a.leaf_count != tree_b.leaf_count:
        raise ValueError("trees must have same number of leaves")
    sa = bipartition_splits(tree_a)
    sb = bipartition_splits(tree_b)
    raw = float(len(sa - sb) + len(sb - sa))
    if not normalize:
        return raw
    denom = float(max(len(sa) + len(sb), 1))
    return raw / denom


def sampled_quartet_agreement(
    tree_a: PhyloTree,
    tree_b: PhyloTree,
    n_samples: int = 1000,
    seed: int | None = None,
) -> float:
    if tree_a.leaf_count != tree_b.leaf_count:
        raise ValueError("trees must have same number of leaves")
    if tree_a.leaf_count < 4:
        return 1.0
    rng = np.random.default_rng(seed)
    n = tree_a.leaf_count
    d_a = pairwise_leaf_distances(tree_a)
    d_b = pairwise_leaf_distances(tree_b)

    agree = 0
    total = 0
    for _ in range(n_samples):
        q = rng.choice(n, size=4, replace=False)
        qa = _quartet_topology_from_distances(d_a, q)
        qb = _quartet_topology_from_distances(d_b, q)
        if qa is None or qb is None:
            continue
        total += 1
        if qa == qb:
            agree += 1
    if total == 0:
        return 0.0
    return float(agree) / float(total)


def all_quartet_agreement(tree_a: PhyloTree, tree_b: PhyloTree, max_quartets: int = 200_000) -> float:
    n = tree_a.leaf_count
    total_quartets = n * (n - 1) * (n - 2) * (n - 3) // 24
    if total_quartets > max_quartets:
        return sampled_quartet_agreement(tree_a, tree_b, n_samples=max_quartets)
    d_a = pairwise_leaf_distances(tree_a)
    d_b = pairwise_leaf_distances(tree_b)
    agree = 0
    total = 0
    for q in itertools.combinations(range(n), 4):
        qa = _quartet_topology_from_distances(d_a, np.asarray(q))
        qb = _quartet_topology_from_distances(d_b, np.asarray(q))
        if qa is None or qb is None:
            continue
        total += 1
        if qa == qb:
            agree += 1
    if total == 0:
        return 0.0
    return float(agree) / float(total)


def _leaf_component(tree: PhyloTree, start: int, blocked: int) -> set[int]:
    out: set[int] = set()
    stack = [start]
    seen = {blocked}
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node < tree.leaf_count:
            out.add(node)
        for nxt, _ in tree.neighbors(node):
            if nxt not in seen:
                stack.append(nxt)
    return out


def _canonical_split(a: set[int], b: set[int]) -> set[int]:
    if len(a) < len(b):
        return a
    if len(b) < len(a):
        return b
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    return a if a_sorted < b_sorted else b


def _quartet_topology_from_distances(d: np.ndarray, q: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b, c, e = map(int, q)
    s1 = d[a, b] + d[c, e]
    s2 = d[a, c] + d[b, e]
    s3 = d[a, e] + d[b, c]
    vals = np.asarray([s1, s2, s3], dtype=np.float64)
    i = int(np.argmin(vals))
    # treat near ties as unresolved
    sorted_vals = np.sort(vals)
    if sorted_vals[1] - sorted_vals[0] < 1e-10:
        return None
    if i == 0:
        p1, p2 = (a, b), (c, e)
    elif i == 1:
        p1, p2 = (a, c), (b, e)
    else:
        p1, p2 = (a, e), (b, c)
    s_p1 = tuple(sorted(p1))
    s_p2 = tuple(sorted(p2))
    return (s_p1, s_p2) if s_p1 < s_p2 else (s_p2, s_p1)
