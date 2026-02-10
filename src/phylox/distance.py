from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tree import PhyloTree


@dataclass(frozen=True)
class NeighborJoiningResult:
    tree: PhyloTree
    labels: list[str] | None = None


def masked_euclidean_distance_matrix(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray | None = None,
    partition_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Pairwise masked Euclidean distance:

      d(i,j) = sqrt( sum_k m_ik m_jk w_k (z_ik-z_jk)^2 / sum_k m_ik m_jk w_k )

    If two taxa have no overlapping observed dimensions, distance is +inf.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if z.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if m.shape != z.shape:
        raise ValueError("mask must match embeddings shape")
    n_taxa, d_total = z.shape

    if dim_to_partition is None:
        dim_weights = np.ones(d_total, dtype=np.float64)
    else:
        dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
        if dim_to_partition.shape != (d_total,):
            raise ValueError("dim_to_partition must have shape (d_total,)")
        if np.any(dim_to_partition < 0):
            raise ValueError("dim_to_partition must be non-negative")
        n_partitions = int(dim_to_partition.max()) + 1 if d_total > 0 else 0
        if partition_weights is None:
            p_weights = np.ones(n_partitions, dtype=np.float64)
        else:
            p_weights = np.asarray(partition_weights, dtype=np.float64)
            if p_weights.shape != (n_partitions,):
                raise ValueError("partition_weights length mismatch")
        dim_weights = p_weights[dim_to_partition]

    dmat = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    for i in range(n_taxa):
        dmat[i, i] = 0.0
        zi = z[i]
        mi = m[i]
        for j in range(i + 1, n_taxa):
            overlap = mi & m[j]
            if not np.any(overlap):
                dij = np.inf
            else:
                w = dim_weights[overlap]
                diff = zi[overlap] - z[j, overlap]
                denom = float(np.sum(w))
                num = float(np.sum(w * diff * diff))
                dij = np.sqrt(num / denom) if denom > 0 else np.inf
            dmat[i, j] = dij
            dmat[j, i] = dij
    return dmat


def neighbor_joining(
    distance_matrix: np.ndarray,
    labels: Sequence[str] | None = None,
    min_branch_length: float = 1e-8,
) -> NeighborJoiningResult:
    """
    Build an unrooted tree using the classic neighbor-joining algorithm.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix must be square")
    if np.any(np.isnan(D)):
        raise ValueError("distance_matrix contains NaN")
    if np.any(D < 0):
        raise ValueError("distance_matrix must be non-negative")
    if not np.allclose(D, D.T):
        raise ValueError("distance_matrix must be symmetric")

    n = D.shape[0]
    if n < 2:
        raise ValueError("need at least two taxa for NJ")

    if labels is not None and len(labels) != n:
        raise ValueError("labels length mismatch")

    # Use a dense matrix indexed by node id; allocate enough room for all internal nodes.
    max_nodes = 2 * n - 2
    work = np.full((max_nodes, max_nodes), np.inf, dtype=np.float64)
    work[:n, :n] = D
    np.fill_diagonal(work, 0.0)

    active = list(range(n))
    next_node = n
    edges: list[tuple[int, int, float]] = []

    while len(active) > 2:
        m = len(active)
        sub = work[np.ix_(active, active)]
        if not np.all(np.isfinite(sub)):
            raise ValueError("distance_matrix contains disconnected taxa pairs (+inf)")

        row_sum = np.sum(sub, axis=1)
        q = (m - 2) * sub - row_sum[:, None] - row_sum[None, :]
        np.fill_diagonal(q, np.inf)
        min_idx = np.argmin(q)
        ai, aj = divmod(min_idx, m)
        i = active[ai]
        j = active[aj]

        dij = work[i, j]
        delta = (row_sum[ai] - row_sum[aj]) / (m - 2)
        li = 0.5 * (dij + delta)
        lj = dij - li
        li = max(float(li), min_branch_length)
        lj = max(float(lj), min_branch_length)

        u = next_node
        next_node += 1
        edges.append((u, i, li))
        edges.append((u, j, lj))

        for k in active:
            if k == i or k == j:
                continue
            dik = work[i, k]
            djk = work[j, k]
            duk = 0.5 * (dik + djk - dij)
            work[u, k] = duk
            work[k, u] = duk
        work[u, u] = 0.0

        active = [x for x in active if x != i and x != j]
        active.append(u)

    i, j = active
    final_len = max(float(work[i, j]), min_branch_length)
    edges.append((i, j, final_len))

    tree = PhyloTree(num_nodes=next_node, edges=edges, leaf_count=n)
    return NeighborJoiningResult(tree=tree, labels=list(labels) if labels is not None else None)
