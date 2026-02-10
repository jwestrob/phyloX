from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import PhyloTree


@dataclass(frozen=True)
class SimulatedDataset:
    tree: PhyloTree
    embeddings: np.ndarray
    mask: np.ndarray
    dim_to_partition: np.ndarray
    covariates: np.ndarray | None = None


def simulate_ou_embeddings(
    tree: PhyloTree,
    dim_to_partition: np.ndarray,
    alpha_by_partition: np.ndarray,
    sigma2_by_partition: np.ndarray,
    n_taxa: int | None = None,
    mask: np.ndarray | None = None,
    coverage_scale: np.ndarray | None = None,
    rate_by_dim: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate OU latent states on tree with stationary variance fixed to 1 and noisy leaf observations.
    """
    rng = np.random.default_rng(seed)
    dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
    alpha = np.asarray(alpha_by_partition, dtype=np.float64)
    sigma2 = np.asarray(sigma2_by_partition, dtype=np.float64)

    if n_taxa is None:
        n_taxa = tree.leaf_count
    if n_taxa != tree.leaf_count:
        raise ValueError("n_taxa must match tree.leaf_count")
    d_total = int(dim_to_partition.size)

    if d_total == 0:
        raise ValueError("dim_to_partition must be non-empty")
    n_partitions = int(dim_to_partition.max()) + 1
    if alpha.shape != (n_partitions,):
        raise ValueError("alpha_by_partition length mismatch")
    if sigma2.shape != (n_partitions,):
        raise ValueError("sigma2_by_partition length mismatch")
    if np.any(alpha <= 0):
        raise ValueError("alpha_by_partition must be > 0")
    if np.any(sigma2 <= 0):
        raise ValueError("sigma2_by_partition must be > 0")

    if rate_by_dim is None:
        dim_rate = np.ones(d_total, dtype=np.float64)
    else:
        dim_rate = np.asarray(rate_by_dim, dtype=np.float64)
        if dim_rate.shape != (d_total,):
            raise ValueError("rate_by_dim shape mismatch")
        if np.any(dim_rate <= 0):
            raise ValueError("rate_by_dim must be > 0")

    if mask is None:
        m = np.ones((n_taxa, d_total), dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (n_taxa, d_total):
            raise ValueError("mask shape mismatch")

    if coverage_scale is None:
        cov_scale = np.ones((n_taxa, n_partitions), dtype=np.float64)
    else:
        cov_scale = np.asarray(coverage_scale, dtype=np.float64)
        if cov_scale.shape != (n_taxa, n_partitions):
            raise ValueError("coverage_scale shape mismatch")
        if np.any(cov_scale <= 0):
            raise ValueError("coverage_scale must be > 0")

    rooted = tree.rooted()
    latent = np.zeros((tree.num_nodes, d_total), dtype=np.float64)
    latent[rooted.root] = rng.normal(size=d_total)

    for node in rooted.preorder:
        if node == rooted.root:
            continue
        parent = int(rooted.parent[node])
        t = float(rooted.branch_length_to_parent[node])
        alpha_dim = alpha[dim_to_partition] * dim_rate
        a = np.exp(-alpha_dim * t)
        q = 1.0 - a * a
        eps = rng.normal(size=d_total)
        latent[node] = a * latent[parent] + np.sqrt(np.maximum(q, 0.0)) * eps

    leaves = latent[:n_taxa]
    obs = leaves.copy()
    for p in range(n_partitions):
        idx = np.flatnonzero(dim_to_partition == p)
        if idx.size == 0:
            continue
        noise = rng.normal(size=(n_taxa, idx.size))
        obs[:, idx] += np.sqrt(sigma2[p] * cov_scale[:, [p]]) * noise

    obs = np.where(m, obs, 0.0)
    return obs, m


def simulate_random_binary_tree(
    n_taxa: int,
    branch_scale: float = 0.2,
    seed: int | None = None,
) -> PhyloTree:
    """
    Generate a random unrooted binary tree with leaves 0..n_taxa-1.
    """
    if n_taxa < 3:
        raise ValueError("n_taxa must be >= 3")
    rng = np.random.default_rng(seed)

    # Start with 3-leaf star using internal node n_taxa.
    next_node = n_taxa + 1
    root_internal = n_taxa
    edges: list[tuple[int, int, float]] = [
        (root_internal, 0, float(rng.exponential(branch_scale))),
        (root_internal, 1, float(rng.exponential(branch_scale))),
        (root_internal, 2, float(rng.exponential(branch_scale))),
    ]

    for leaf in range(3, n_taxa):
        # Choose an existing edge to split.
        e_idx = int(rng.integers(0, len(edges)))
        a, b, t = edges.pop(e_idx)
        new_internal = next_node
        next_node += 1
        t_split = max(t * 0.5, 1e-6)

        edges.append((a, new_internal, t_split))
        edges.append((b, new_internal, t_split))
        edges.append((new_internal, leaf, float(rng.exponential(branch_scale))))

    num_nodes = next_node
    return PhyloTree(num_nodes=num_nodes, edges=edges, leaf_count=n_taxa)


def simulate_partitioned_dataset(
    n_taxa: int,
    dims_per_partition: list[int],
    alpha_by_partition: np.ndarray,
    sigma2_by_partition: np.ndarray,
    missing_prob_by_partition: np.ndarray | None = None,
    confounder_dim: int = 5,
    nuisance_strength: float = 0.0,
    seed: int | None = None,
) -> SimulatedDataset:
    """
    Convenience generator for end-to-end pipeline tests.
    """
    rng = np.random.default_rng(seed)
    n_partitions = len(dims_per_partition)
    if alpha_by_partition.shape != (n_partitions,):
        raise ValueError("alpha_by_partition length mismatch")
    if sigma2_by_partition.shape != (n_partitions,):
        raise ValueError("sigma2_by_partition length mismatch")
    if missing_prob_by_partition is None:
        missing_prob_by_partition = np.zeros(n_partitions, dtype=np.float64)
    else:
        missing_prob_by_partition = np.asarray(missing_prob_by_partition, dtype=np.float64)
        if missing_prob_by_partition.shape != (n_partitions,):
            raise ValueError("missing_prob_by_partition length mismatch")

    dim_to_partition = np.concatenate(
        [np.full(d, p, dtype=np.int64) for p, d in enumerate(dims_per_partition)]
    )
    tree = simulate_random_binary_tree(n_taxa=n_taxa, seed=seed)
    d_total = int(np.sum(dims_per_partition))

    mask = np.ones((n_taxa, d_total), dtype=bool)
    start = 0
    for p, d in enumerate(dims_per_partition):
        stop = start + d
        missing_taxa = rng.random(n_taxa) < missing_prob_by_partition[p]
        mask[missing_taxa, start:stop] = False
        start = stop

    z, m = simulate_ou_embeddings(
        tree=tree,
        dim_to_partition=dim_to_partition,
        alpha_by_partition=alpha_by_partition,
        sigma2_by_partition=sigma2_by_partition,
        mask=mask,
        seed=seed,
    )

    covariates = None
    if confounder_dim > 0:
        covariates = rng.normal(size=(n_taxa, confounder_dim))
        if nuisance_strength > 0:
            # Inject nuisance structure partition-wise.
            start = 0
            for p, d in enumerate(dims_per_partition):
                stop = start + d
                beta = rng.normal(scale=nuisance_strength, size=(confounder_dim, d))
                z[:, start:stop] += covariates @ beta
                start = stop
            z = np.where(m, z, 0.0)

    return SimulatedDataset(
        tree=tree,
        embeddings=z,
        mask=m,
        dim_to_partition=dim_to_partition,
        covariates=covariates,
    )
