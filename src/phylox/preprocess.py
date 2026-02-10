from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import partition_indices, partition_presence_from_mask


@dataclass(frozen=True)
class PartitionWhitening:
    mean: np.ndarray
    zca_matrix: np.ndarray
    effective_rank: int


@dataclass(frozen=True)
class WhiteningResult:
    embeddings: np.ndarray
    transforms: dict[int, PartitionWhitening]


def regress_out_confounders(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    covariates: np.ndarray,
    ridge_lambda: float = 1.0,
    fit_intercept: bool = True,
    min_taxa: int = 8,
) -> np.ndarray:
    """
    Per-partition ridge residualization: Z_g <- Z_g - X beta_g.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    x = np.asarray(covariates, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if m.shape != z.shape:
        raise ValueError("mask shape mismatch")
    if x.ndim != 2 or x.shape[0] != z.shape[0]:
        raise ValueError("covariates must be shape (n_taxa, n_covariates)")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0")

    out = z.copy()
    part_idx = partition_indices(dim_to_partition)
    finite_cov = np.all(np.isfinite(x), axis=1)

    for p, idx in part_idx.items():
        if idx.size == 0:
            continue
        present = np.any(m[:, idx], axis=1)
        use = present & finite_cov
        if np.count_nonzero(use) < max(min_taxa, x.shape[1] + 1):
            continue

        X = x[use]
        if fit_intercept:
            X = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])

        Y = out[np.ix_(use, idx)]
        XtX = X.T @ X
        reg = ridge_lambda * np.eye(XtX.shape[0], dtype=np.float64)
        if fit_intercept:
            reg[0, 0] = 0.0
        beta = np.linalg.solve(XtX + reg, X.T @ Y)
        fitted = X @ beta
        out[np.ix_(use, idx)] = Y - fitted

    return np.where(m, out, 0.0)


def zca_whiten_partitions(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    d_eff_by_partition: Mapping[int, int] | None = None,
    eps: float = 1e-6,
    min_taxa: int = 6,
) -> WhiteningResult:
    """
    ZCA whitening per partition with optional truncation rank per partition.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if z.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if m.shape != z.shape:
        raise ValueError("mask shape mismatch")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    out = z.copy()
    transforms: dict[int, PartitionWhitening] = {}
    part_idx = partition_indices(dim_to_partition)

    for p, idx in part_idx.items():
        if idx.size == 0:
            continue
        block = out[:, idx]
        block_mask = m[:, idx]
        full_rows = np.all(block_mask, axis=1)
        if np.count_nonzero(full_rows) < min_taxa:
            continue

        Y = block[full_rows]
        mean = np.mean(Y, axis=0)
        centered = Y - mean
        denom = max(centered.shape[0] - 1, 1)
        cov = (centered.T @ centered) / float(denom)
        cov += eps * np.eye(cov.shape[0], dtype=np.float64)

        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        target_rank = idx.size
        if d_eff_by_partition is not None and p in d_eff_by_partition:
            target_rank = int(d_eff_by_partition[p])
        rank = max(1, min(target_rank, idx.size))

        U = evecs[:, :rank]
        S = evals[:rank]
        inv_sqrt = np.diag(1.0 / np.sqrt(S + eps))
        zca = U @ inv_sqrt @ U.T

        block_centered = block - mean
        block_whitened = block_centered @ zca
        out[:, idx] = block_whitened
        transforms[p] = PartitionWhitening(mean=mean, zca_matrix=zca, effective_rank=rank)

    return WhiteningResult(embeddings=np.where(m, out, 0.0), transforms=transforms)


def zca_whiten_global(
    embeddings: np.ndarray,
    mask: np.ndarray,
    d_eff: int | None = None,
    eps: float = 1e-6,
    min_taxa: int = 6,
) -> tuple[np.ndarray, PartitionWhitening | None]:
    """
    Optional global whitening across the full concatenated embedding.
    Uses only taxa with complete observations across all dimensions.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if z.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if m.shape != z.shape:
        raise ValueError("mask shape mismatch")

    full_rows = np.all(m, axis=1)
    if np.count_nonzero(full_rows) < min_taxa:
        return np.where(m, z, 0.0), None

    Y = z[full_rows]
    mean = np.mean(Y, axis=0)
    centered = Y - mean
    cov = (centered.T @ centered) / float(max(centered.shape[0] - 1, 1))
    cov += eps * np.eye(cov.shape[0], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    rank = z.shape[1] if d_eff is None else max(1, min(int(d_eff), z.shape[1]))
    U = evecs[:, :rank]
    S = evals[:rank]
    zca = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    out = (z - mean) @ zca
    return np.where(m, out, 0.0), PartitionWhitening(mean=mean, zca_matrix=zca, effective_rank=rank)


def coverage_noise_scale(
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    mode: str = "partition_fraction_inverse",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-(taxon, partition) noise multipliers h_{i,g}.

    Modes:
    - partition_fraction_inverse: h_{i,g} = 1 / coverage(i,g)
    - global_partition_count: h_{i,g} = G_total / G_present(i)
    """
    m = np.asarray(mask, dtype=bool)
    dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
    if m.ndim != 2:
        raise ValueError("mask must be 2D")
    if dim_to_partition.shape != (m.shape[1],):
        raise ValueError("dim_to_partition length mismatch")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    n_taxa = m.shape[0]
    n_partitions = int(dim_to_partition.max()) + 1 if dim_to_partition.size else 0
    if n_partitions == 0:
        return np.zeros((n_taxa, 0), dtype=np.float64)

    if mode == "global_partition_count":
        presence = partition_presence_from_mask(m, dim_to_partition)
        g_present = np.count_nonzero(presence, axis=1)
        g_total = float(n_partitions)
        scale_row = g_total / np.maximum(g_present.astype(np.float64), eps)
        return np.repeat(scale_row[:, None], n_partitions, axis=1)

    if mode == "partition_fraction_inverse":
        out = np.ones((n_taxa, n_partitions), dtype=np.float64)
        for p in range(n_partitions):
            idx = np.flatnonzero(dim_to_partition == p)
            if idx.size == 0:
                continue
            frac = np.mean(m[:, idx], axis=1)
            out[:, p] = 1.0 / np.maximum(frac, eps)
        return out

    raise ValueError(f"unsupported coverage mode '{mode}'")


def preprocess_embeddings(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    covariates: np.ndarray | None = None,
    ridge_lambda: float = 1.0,
    whitening: str = "partition",
    d_eff_by_partition: Mapping[int, int] | None = None,
    d_eff_global: int | None = None,
) -> tuple[np.ndarray, dict[int, PartitionWhitening]]:
    """
    Combined preprocessing path:
    1) optional confounder regression
    2) partition or global ZCA whitening
    """
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)

    if covariates is not None:
        z = regress_out_confounders(
            embeddings=z,
            mask=m,
            dim_to_partition=dim_to_partition,
            covariates=covariates,
            ridge_lambda=ridge_lambda,
        )

    if whitening == "partition":
        white = zca_whiten_partitions(
            embeddings=z,
            mask=m,
            dim_to_partition=dim_to_partition,
            d_eff_by_partition=d_eff_by_partition,
        )
        return white.embeddings, white.transforms

    if whitening == "global":
        z_wh, transform = zca_whiten_global(
            embeddings=z,
            mask=m,
            d_eff=d_eff_global,
        )
        transforms = {} if transform is None else {-1: transform}
        return z_wh, transforms

    if whitening == "none":
        return np.where(m, z, 0.0), {}

    raise ValueError(f"unsupported whitening mode '{whitening}'")
