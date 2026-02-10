from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tree import PhyloTree

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class OULikelihoodResult:
    total_log_likelihood: float
    partition_log_likelihoods: np.ndarray


def ou_log_likelihood(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    alpha_by_partition: Sequence[float],
    sigma2_by_partition: Sequence[float],
    partition_weights: Sequence[float] | None = None,
    coverage_scale: np.ndarray | None = None,
    root: int | None = None,
    return_details: bool = False,
) -> float | OULikelihoodResult:
    """
    Compute masked OU log-likelihood with stationary variance fixed to 1.

    Model per partition g and dimension k in g:
      x_child | x_parent ~ N(exp(-alpha_g * t) * x_parent, 1 - exp(-2 alpha_g * t))
      y_leaf | x_leaf     ~ N(x_leaf, sigma_g^2 * coverage_scale[i, g])

    Inputs:
    - embeddings: shape (n_taxa, d_total)
    - mask: same shape as embeddings; True where observation is present
    - dim_to_partition: shape (d_total,), integer partition ids in [0, n_partitions)
    - alpha_by_partition, sigma2_by_partition: length n_partitions
    - partition_weights: optional length n_partitions, default 1
    - coverage_scale: optional shape (n_taxa, n_partitions), default 1
    """
    embeddings, mask, dim_to_partition, alpha, sigma2, weights, coverage = _validate_and_prepare(
        tree=tree,
        embeddings=embeddings,
        mask=mask,
        dim_to_partition=dim_to_partition,
        alpha_by_partition=alpha_by_partition,
        sigma2_by_partition=sigma2_by_partition,
        partition_weights=partition_weights,
        coverage_scale=coverage_scale,
    )

    rooted = tree.rooted(root=root)
    n_partitions = alpha.shape[0]
    partition_terms = np.zeros(n_partitions, dtype=np.float64)

    for p in range(n_partitions):
        dim_idx = np.flatnonzero(dim_to_partition == p)
        if dim_idx.size == 0:
            continue
        partition_terms[p] = _partition_log_likelihood(
            rooted=rooted,
            embeddings=embeddings,
            mask=mask,
            dim_idx=dim_idx,
            alpha=float(alpha[p]),
            sigma2=float(sigma2[p]),
            coverage=coverage[:, p],
        )

    total = float(np.dot(weights, partition_terms))
    if return_details:
        return OULikelihoodResult(
            total_log_likelihood=total,
            partition_log_likelihoods=partition_terms,
        )
    return total


def _partition_log_likelihood(
    rooted,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_idx: np.ndarray,
    alpha: float,
    sigma2: float,
    coverage: np.ndarray,
) -> float:
    m = dim_idx.size
    n_nodes = rooted.num_nodes
    n_taxa = embeddings.shape[0]

    # Canonical Gaussian terms per node, per dim:
    # exp(-0.5 * J * x^2 + h * x + c)
    J = np.zeros((n_nodes, m), dtype=np.float64)
    h = np.zeros((n_nodes, m), dtype=np.float64)
    c = np.zeros((n_nodes, m), dtype=np.float64)

    obs = mask[:, dim_idx]
    y = embeddings[:, dim_idx]
    var = sigma2 * coverage[:, None]
    inv_var = np.where(obs, 1.0 / var, 0.0)

    J[:n_taxa] = inv_var
    h[:n_taxa] = y * inv_var
    c[:n_taxa] = np.where(
        obs,
        -0.5 * (y * y * inv_var + np.log(_TWO_PI * var)),
        0.0,
    )

    for node in rooted.postorder:
        if node == rooted.root:
            continue
        parent = int(rooted.parent[node])
        t = float(rooted.branch_length_to_parent[node])
        a = np.exp(-alpha * t)
        q = 1.0 - a * a

        Jc = J[node]
        hc = h[node]
        cc = c[node]

        if q <= 1e-14:
            Jmsg = (a * a) * Jc
            hmsg = a * hc
            cmsg = cc
        else:
            denom = 1.0 + q * Jc
            Jmsg = (a * a) * Jc / denom
            hmsg = a * hc / denom
            cmsg = cc - 0.5 * np.log(denom) + 0.5 * (hc * hc) * q / denom

        J[parent] += Jmsg
        h[parent] += hmsg
        c[parent] += cmsg

    root = rooted.root
    Jr = J[root]
    hr = h[root]
    cr = c[root]

    # Integrate root with stationary prior x_root ~ N(0, 1).
    log_norm = -0.5 * np.log(1.0 + Jr)
    quad = 0.5 * (hr * hr) / (1.0 + Jr)
    return float(np.sum(cr + log_norm + quad))


def _validate_and_prepare(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    alpha_by_partition: Sequence[float],
    sigma2_by_partition: Sequence[float],
    partition_weights: Sequence[float] | None,
    coverage_scale: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
    alpha = np.asarray(alpha_by_partition, dtype=np.float64)
    sigma2 = np.asarray(sigma2_by_partition, dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if mask.shape != embeddings.shape:
        raise ValueError("mask must have the same shape as embeddings")
    n_taxa, d_total = embeddings.shape

    if n_taxa != tree.leaf_count:
        raise ValueError("number of embedding rows must match tree.leaf_count")
    if dim_to_partition.shape != (d_total,):
        raise ValueError("dim_to_partition must have shape (d_total,)")
    if np.any(dim_to_partition < 0):
        raise ValueError("dim_to_partition must be non-negative")

    n_partitions = int(dim_to_partition.max()) + 1 if d_total > 0 else len(alpha)
    if alpha.shape != (n_partitions,):
        raise ValueError("alpha_by_partition length mismatch")
    if sigma2.shape != (n_partitions,):
        raise ValueError("sigma2_by_partition length mismatch")
    if np.any(alpha <= 0):
        raise ValueError("all alpha values must be > 0")
    if np.any(sigma2 <= 0):
        raise ValueError("all sigma2 values must be > 0")

    if partition_weights is None:
        weights = np.ones(n_partitions, dtype=np.float64)
    else:
        weights = np.asarray(partition_weights, dtype=np.float64)
        if weights.shape != (n_partitions,):
            raise ValueError("partition_weights length mismatch")

    if coverage_scale is None:
        coverage = np.ones((n_taxa, n_partitions), dtype=np.float64)
    else:
        coverage = np.asarray(coverage_scale, dtype=np.float64)
        if coverage.shape != (n_taxa, n_partitions):
            raise ValueError("coverage_scale must have shape (n_taxa, n_partitions)")
        if np.any(coverage <= 0):
            raise ValueError("coverage_scale must be > 0")

    return embeddings, mask, dim_to_partition, alpha, sigma2, weights, coverage
