from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi

import numpy as np

from .optimize import (
    OUModelParameters,
    optimize_branch_lengths_coordinate,
    optimize_partition_parameters,
    score_ou_model,
)
from .tree import PhyloTree


@dataclass(frozen=True)
class StudentTEMRecord:
    iteration: int
    log_likelihood: float


@dataclass(frozen=True)
class StudentTEMResult:
    tree: PhyloTree
    params: OUModelParameters
    log_likelihood: float
    history: tuple[StudentTEMRecord, ...]


def compute_student_t_precisions(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    sigma2_by_partition: np.ndarray,
    coverage_scale: np.ndarray | None,
    dof_by_partition: np.ndarray,
    min_precision: float = 1e-3,
    max_precision: float = 1e3,
) -> np.ndarray:
    """
    E-step approximation for Student-t observation noise:
      lambda_{ik} = (nu_g + 1) / (nu_g + r_{ik}^2)

    with r_{ik} standardized by observation variance + stationary latent variance.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    dim_map = np.asarray(dim_to_partition, dtype=np.int64)
    sigma2 = np.asarray(sigma2_by_partition, dtype=np.float64)
    dof = np.asarray(dof_by_partition, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if m.shape != z.shape:
        raise ValueError("mask must match embeddings shape")
    n_taxa, d_total = z.shape
    if dim_map.shape != (d_total,):
        raise ValueError("dim_to_partition shape mismatch")
    n_partitions = int(dim_map.max()) + 1 if d_total > 0 else 0
    if sigma2.shape != (n_partitions,):
        raise ValueError("sigma2_by_partition length mismatch")
    if dof.shape != (n_partitions,):
        raise ValueError("dof_by_partition length mismatch")
    if np.any(dof <= 2.0):
        raise ValueError("all Student-t dof values must be > 2")

    if coverage_scale is None:
        coverage = np.ones((n_taxa, n_partitions), dtype=np.float64)
    else:
        coverage = np.asarray(coverage_scale, dtype=np.float64)
        if coverage.shape != (n_taxa, n_partitions):
            raise ValueError("coverage_scale shape mismatch")
        if np.any(coverage <= 0):
            raise ValueError("coverage_scale must be > 0")

    precision = np.ones((n_taxa, d_total), dtype=np.float64)
    for p in range(n_partitions):
        idx = np.flatnonzero(dim_map == p)
        if idx.size == 0:
            continue
        # Standardize by expected latent+observation variance.
        var = 1.0 + sigma2[p] * coverage[:, [p]]
        r2 = (z[:, idx] * z[:, idx]) / np.maximum(var, 1e-12)
        lam = (dof[p] + 1.0) / (dof[p] + r2)
        precision[:, idx] = np.clip(lam, min_precision, max_precision)

    precision = np.where(m, precision, 1.0)
    return precision


def fit_student_t_em(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    rounds: int = 3,
    dof_init: float = 4.0,
    optimize_dof: bool = True,
    dof_bounds: tuple[float, float] = (2.1, 50.0),
    dof_grid_size: int = 48,
    optimize_branch_lengths: bool = True,
    optimize_alpha: bool = True,
    optimize_sigma2: bool = True,
) -> StudentTEMResult:
    """
    Approximate EM for robust Student-t observation noise using latent precisions.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    dim_map = np.asarray(dim_to_partition, dtype=np.int64)
    n_partitions = int(dim_map.max()) + 1 if dim_map.size else 0

    if params.student_t_dof_by_partition is None:
        dof = np.full(n_partitions, float(dof_init), dtype=np.float64)
    else:
        dof = np.asarray(params.student_t_dof_by_partition, dtype=np.float64).copy()
    dof = np.clip(dof, dof_bounds[0], dof_bounds[1])

    if params.precision_weights is None:
        precision = np.ones_like(z, dtype=np.float64)
    else:
        precision = np.asarray(params.precision_weights, dtype=np.float64).copy()

    cur_tree = tree
    cur_params = params.with_precision_weights(precision).with_student_t_dof(dof)
    cur_ll = score_ou_model(cur_tree, z, m, dim_map, cur_params)
    history = [StudentTEMRecord(iteration=0, log_likelihood=cur_ll)]

    for it in range(1, rounds + 1):
        # E-step: update latent precisions.
        precision = compute_student_t_precisions(
            embeddings=z,
            mask=m,
            dim_to_partition=dim_map,
            sigma2_by_partition=cur_params.sigma2_by_partition,
            coverage_scale=cur_params.coverage_scale,
            dof_by_partition=dof,
        )
        cur_params = cur_params.with_precision_weights(precision)

        if optimize_dof:
            dof = _optimize_student_t_dof_by_grid(
                embeddings=z,
                mask=m,
                dim_to_partition=dim_map,
                sigma2_by_partition=cur_params.sigma2_by_partition,
                coverage_scale=cur_params.coverage_scale,
                bounds=dof_bounds,
                grid_size=dof_grid_size,
            )
            cur_params = cur_params.with_student_t_dof(dof)

        if optimize_branch_lengths:
            cur_tree, _ = optimize_branch_lengths_coordinate(
                tree=cur_tree,
                embeddings=z,
                mask=m,
                dim_to_partition=dim_map,
                params=cur_params,
                rounds=1,
            )
        cur_params, cur_ll = optimize_partition_parameters(
            tree=cur_tree,
            embeddings=z,
            mask=m,
            dim_to_partition=dim_map,
            params=cur_params,
            optimize_alpha=optimize_alpha,
            optimize_sigma2=optimize_sigma2,
            rounds=1,
        )
        cur_ll = score_ou_model(cur_tree, z, m, dim_map, cur_params)
        history.append(StudentTEMRecord(iteration=it, log_likelihood=cur_ll))

    return StudentTEMResult(
        tree=cur_tree,
        params=cur_params,
        log_likelihood=cur_ll,
        history=tuple(history),
    )


def _optimize_student_t_dof_by_grid(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    sigma2_by_partition: np.ndarray,
    coverage_scale: np.ndarray | None,
    bounds: tuple[float, float] = (2.1, 50.0),
    grid_size: int = 48,
) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    dim_map = np.asarray(dim_to_partition, dtype=np.int64)
    sigma2 = np.asarray(sigma2_by_partition, dtype=np.float64)
    n_taxa, d_total = z.shape
    n_partitions = int(dim_map.max()) + 1 if d_total > 0 else 0

    if coverage_scale is None:
        coverage = np.ones((n_taxa, n_partitions), dtype=np.float64)
    else:
        coverage = np.asarray(coverage_scale, dtype=np.float64)

    lo, hi = bounds
    grid = np.linspace(lo, hi, int(max(grid_size, 8)))
    out = np.full(n_partitions, 4.0, dtype=np.float64)

    for p in range(n_partitions):
        idx = np.flatnonzero(dim_map == p)
        if idx.size == 0:
            continue
        var = 1.0 + sigma2[p] * coverage[:, [p]]
        r2 = (z[:, idx] * z[:, idx]) / np.maximum(var, 1e-12)
        obs = m[:, idx]
        r2_obs = r2[obs]
        if r2_obs.size == 0:
            continue

        best_nu = grid[0]
        best_ll = -np.inf
        for nu in grid:
            ll = _student_t_std_loglike(r2_obs, float(nu))
            if ll > best_ll:
                best_ll = ll
                best_nu = nu
        out[p] = best_nu
    return out


def _student_t_std_loglike(r2_values: np.ndarray, nu: float) -> float:
    # Standardized residual likelihood under univariate Student-t(nu).
    c = lgamma((nu + 1.0) * 0.5) - lgamma(nu * 0.5) - 0.5 * (log(pi * nu))
    return float(np.sum(c - 0.5 * (nu + 1.0) * np.log1p(r2_values / nu)))
