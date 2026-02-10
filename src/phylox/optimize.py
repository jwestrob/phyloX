from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .ou_likelihood import ou_log_likelihood
from .tree import PhyloTree


@dataclass(frozen=True)
class OUModelParameters:
    alpha_by_partition: np.ndarray
    sigma2_by_partition: np.ndarray
    partition_weights: np.ndarray | None = None
    coverage_scale: np.ndarray | None = None
    rate_by_dim: np.ndarray | None = None
    precision_weights: np.ndarray | None = None
    student_t_dof_by_partition: np.ndarray | None = None
    gamma_rates_by_partition: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None

    def with_alpha(self, alpha: np.ndarray) -> "OUModelParameters":
        return OUModelParameters(
            alpha_by_partition=np.asarray(alpha, dtype=np.float64),
            sigma2_by_partition=self.sigma2_by_partition,
            partition_weights=self.partition_weights,
            coverage_scale=self.coverage_scale,
            rate_by_dim=self.rate_by_dim,
            precision_weights=self.precision_weights,
            student_t_dof_by_partition=self.student_t_dof_by_partition,
            gamma_rates_by_partition=self.gamma_rates_by_partition,
        )

    def with_sigma2(self, sigma2: np.ndarray) -> "OUModelParameters":
        return OUModelParameters(
            alpha_by_partition=self.alpha_by_partition,
            sigma2_by_partition=np.asarray(sigma2, dtype=np.float64),
            partition_weights=self.partition_weights,
            coverage_scale=self.coverage_scale,
            rate_by_dim=self.rate_by_dim,
            precision_weights=self.precision_weights,
            student_t_dof_by_partition=self.student_t_dof_by_partition,
            gamma_rates_by_partition=self.gamma_rates_by_partition,
        )

    def with_rate_by_dim(self, rate_by_dim: np.ndarray) -> "OUModelParameters":
        return OUModelParameters(
            alpha_by_partition=self.alpha_by_partition,
            sigma2_by_partition=self.sigma2_by_partition,
            partition_weights=self.partition_weights,
            coverage_scale=self.coverage_scale,
            rate_by_dim=np.asarray(rate_by_dim, dtype=np.float64),
            precision_weights=self.precision_weights,
            student_t_dof_by_partition=self.student_t_dof_by_partition,
            gamma_rates_by_partition=self.gamma_rates_by_partition,
        )

    def with_precision_weights(self, precision_weights: np.ndarray | None) -> "OUModelParameters":
        return OUModelParameters(
            alpha_by_partition=self.alpha_by_partition,
            sigma2_by_partition=self.sigma2_by_partition,
            partition_weights=self.partition_weights,
            coverage_scale=self.coverage_scale,
            rate_by_dim=self.rate_by_dim,
            precision_weights=(
                None if precision_weights is None else np.asarray(precision_weights, dtype=np.float64)
            ),
            student_t_dof_by_partition=self.student_t_dof_by_partition,
            gamma_rates_by_partition=self.gamma_rates_by_partition,
        )

    def with_student_t_dof(self, dof: np.ndarray | None) -> "OUModelParameters":
        return OUModelParameters(
            alpha_by_partition=self.alpha_by_partition,
            sigma2_by_partition=self.sigma2_by_partition,
            partition_weights=self.partition_weights,
            coverage_scale=self.coverage_scale,
            rate_by_dim=self.rate_by_dim,
            precision_weights=self.precision_weights,
            student_t_dof_by_partition=None if dof is None else np.asarray(dof, dtype=np.float64),
            gamma_rates_by_partition=self.gamma_rates_by_partition,
        )


@dataclass(frozen=True)
class OptimizationRecord:
    step: str
    iteration: int
    log_likelihood: float


@dataclass(frozen=True)
class RefinementResult:
    tree: PhyloTree
    params: OUModelParameters
    log_likelihood: float
    history: tuple[OptimizationRecord, ...]


def score_ou_model(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
) -> float:
    return float(
        ou_log_likelihood(
            tree=tree,
            embeddings=embeddings,
            mask=mask,
            dim_to_partition=dim_to_partition,
            alpha_by_partition=params.alpha_by_partition,
            sigma2_by_partition=params.sigma2_by_partition,
            partition_weights=params.partition_weights,
            coverage_scale=params.coverage_scale,
            rate_by_dim=params.rate_by_dim,
            precision_weights=params.precision_weights,
            gamma_rates_by_partition=params.gamma_rates_by_partition,
        )
    )


def optimize_branch_lengths_coordinate(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    rounds: int = 2,
    min_length: float = 1e-6,
    max_length: float = 10.0,
    tol: float = 1e-6,
) -> tuple[PhyloTree, float]:
    """
    Coordinate ascent on branch lengths (log-scale golden search per edge).
    """
    cur_tree = tree
    cur_score = score_ou_model(cur_tree, embeddings, mask, dim_to_partition, params)

    for _ in range(rounds):
        improved = False
        for edge_idx, (_, _, t_cur) in enumerate(cur_tree.edges):
            low = max(min_length, t_cur * 0.25 if t_cur > 0 else min_length)
            high = min(max_length, max(t_cur * 4.0, min_length * 10.0))
            if high <= low:
                continue

            def objective(log_t: float) -> float:
                t = float(np.exp(log_t))
                trial = cur_tree.with_edge_length(edge_idx, t)
                return score_ou_model(trial, embeddings, mask, dim_to_partition, params)

            x_opt, f_opt = _golden_max(np.log(low), np.log(high), objective, max_iter=24)
            if f_opt > cur_score + tol:
                cur_tree = cur_tree.with_edge_length(edge_idx, float(np.exp(x_opt)))
                cur_score = f_opt
                improved = True
        if not improved:
            break
    return cur_tree, cur_score


def optimize_partition_parameters(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    optimize_alpha: bool = True,
    optimize_sigma2: bool = True,
    rounds: int = 2,
    alpha_bounds: tuple[float, float] = (1e-3, 20.0),
    sigma2_bounds: tuple[float, float] = (1e-6, 5.0),
    tol: float = 1e-6,
) -> tuple[OUModelParameters, float]:
    cur_params = params
    cur_score = score_ou_model(tree, embeddings, mask, dim_to_partition, cur_params)

    lo_a, hi_a = alpha_bounds
    lo_s, hi_s = sigma2_bounds

    for _ in range(rounds):
        improved = False
        alpha = cur_params.alpha_by_partition.copy()
        sigma2 = cur_params.sigma2_by_partition.copy()

        for p in range(alpha.shape[0]):
            if optimize_alpha:
                cur_ap = alpha[p]

                def objective_alpha(log_a: float) -> float:
                    alpha_try = alpha.copy()
                    alpha_try[p] = np.exp(log_a)
                    params_try = cur_params.with_alpha(alpha_try)
                    return score_ou_model(tree, embeddings, mask, dim_to_partition, params_try)

                x_opt, f_opt = _golden_max(np.log(lo_a), np.log(hi_a), objective_alpha, max_iter=24)
                if f_opt > cur_score + tol:
                    alpha[p] = float(np.exp(x_opt))
                    cur_params = cur_params.with_alpha(alpha)
                    cur_score = f_opt
                    improved = True

            if optimize_sigma2:
                def objective_sigma(log_s: float) -> float:
                    sigma_try = sigma2.copy()
                    sigma_try[p] = np.exp(log_s)
                    params_try = cur_params.with_sigma2(sigma_try)
                    return score_ou_model(tree, embeddings, mask, dim_to_partition, params_try)

                x_opt, f_opt = _golden_max(np.log(lo_s), np.log(hi_s), objective_sigma, max_iter=24)
                if f_opt > cur_score + tol:
                    sigma2[p] = float(np.exp(x_opt))
                    cur_params = cur_params.with_sigma2(sigma2)
                    cur_score = f_opt
                    improved = True

        if not improved:
            break
    return cur_params, cur_score


def optimize_block_rates(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    blocks: Mapping[int, Sequence[np.ndarray]],
    l2_log_rate: float = 0.1,
    bounds: tuple[float, float] = (0.25, 4.0),
    rounds: int = 1,
    tol: float = 1e-6,
) -> tuple[OUModelParameters, float]:
    """
    Optimize tier-1 blockwise rates with L2 regularization on log(rate).
    `blocks[p]` is a sequence of global dim-index arrays for partition p.
    """
    if params.rate_by_dim is None:
        d_total = int(mask.shape[1])
        cur_rate = np.ones(d_total, dtype=np.float64)
        cur_params = params.with_rate_by_dim(cur_rate)
    else:
        cur_rate = params.rate_by_dim.copy()
        cur_params = params

    def score_with_penalty(rate_by_dim: np.ndarray) -> float:
        raw = score_ou_model(
            tree=tree,
            embeddings=embeddings,
            mask=mask,
            dim_to_partition=dim_to_partition,
            params=cur_params.with_rate_by_dim(rate_by_dim),
        )
        penalty = 0.5 * l2_log_rate * float(np.sum(np.log(rate_by_dim) ** 2))
        return raw - penalty

    cur_score = score_with_penalty(cur_rate)
    lo, hi = bounds

    for _ in range(rounds):
        improved = False
        for p, p_blocks in blocks.items():
            _ = p
            for dims in p_blocks:
                dims = np.asarray(dims, dtype=np.int64)
                if dims.size == 0:
                    continue
                base = float(np.median(cur_rate[dims]))

                def objective(log_r: float) -> float:
                    r = float(np.exp(log_r))
                    trial_rate = cur_rate.copy()
                    trial_rate[dims] = r
                    return score_with_penalty(trial_rate)

                low = np.log(max(lo, base * 0.25))
                high = np.log(min(hi, base * 4.0))
                if high <= low:
                    low = np.log(lo)
                    high = np.log(hi)
                x_opt, f_opt = _golden_max(low, high, objective, max_iter=24)
                if f_opt > cur_score + tol:
                    cur_rate[dims] = float(np.exp(x_opt))
                    cur_score = f_opt
                    improved = True
        if not improved:
            break

    final_params = cur_params.with_rate_by_dim(cur_rate)
    raw_final = score_ou_model(tree, embeddings, mask, dim_to_partition, final_params)
    return final_params, raw_final


def alternating_refinement(
    tree: PhyloTree,
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    rounds: int = 3,
    fit_block_rates: bool = False,
    blocks: Mapping[int, Sequence[np.ndarray]] | None = None,
) -> RefinementResult:
    """
    Phase-B style alternating optimization:
    branch lengths <-> partition parameters (and optional block rates).
    """
    cur_tree = tree
    cur_params = params
    history: list[OptimizationRecord] = []

    cur_ll = score_ou_model(cur_tree, embeddings, mask, dim_to_partition, cur_params)
    history.append(OptimizationRecord(step="init", iteration=0, log_likelihood=cur_ll))

    for i in range(1, rounds + 1):
        cur_tree, cur_ll = optimize_branch_lengths_coordinate(
            tree=cur_tree,
            embeddings=embeddings,
            mask=mask,
            dim_to_partition=dim_to_partition,
            params=cur_params,
        )
        history.append(OptimizationRecord(step="branch", iteration=i, log_likelihood=cur_ll))

        cur_params, cur_ll = optimize_partition_parameters(
            tree=cur_tree,
            embeddings=embeddings,
            mask=mask,
            dim_to_partition=dim_to_partition,
            params=cur_params,
        )
        history.append(OptimizationRecord(step="partition", iteration=i, log_likelihood=cur_ll))

        if fit_block_rates and blocks is not None:
            cur_params, cur_ll = optimize_block_rates(
                tree=cur_tree,
                embeddings=embeddings,
                mask=mask,
                dim_to_partition=dim_to_partition,
                params=cur_params,
                blocks=blocks,
            )
            history.append(OptimizationRecord(step="block-rate", iteration=i, log_likelihood=cur_ll))

    return RefinementResult(
        tree=cur_tree,
        params=cur_params,
        log_likelihood=cur_ll,
        history=tuple(history),
    )


def build_contiguous_blocks(
    dim_to_partition: np.ndarray,
    n_blocks_by_partition: Mapping[int, int] | Sequence[int] | int,
) -> dict[int, list[np.ndarray]]:
    dim_to_partition = np.asarray(dim_to_partition, dtype=np.int64)
    if dim_to_partition.ndim != 1:
        raise ValueError("dim_to_partition must be 1D")
    if np.any(dim_to_partition < 0):
        raise ValueError("dim_to_partition must be non-negative")

    if dim_to_partition.size == 0:
        return {}
    n_partitions = int(dim_to_partition.max()) + 1

    def get_blocks(p: int) -> int:
        if isinstance(n_blocks_by_partition, int):
            return int(n_blocks_by_partition)
        if isinstance(n_blocks_by_partition, Sequence):
            if p >= len(n_blocks_by_partition):
                raise ValueError("n_blocks_by_partition too short")
            return int(n_blocks_by_partition[p])
        if p not in n_blocks_by_partition:
            raise ValueError(f"missing block count for partition {p}")
        return int(n_blocks_by_partition[p])

    out: dict[int, list[np.ndarray]] = {}
    for p in range(n_partitions):
        idx = np.flatnonzero(dim_to_partition == p)
        if idx.size == 0:
            out[p] = []
            continue
        n_blocks = max(1, min(get_blocks(p), idx.size))
        splits = np.array_split(np.arange(idx.size), n_blocks)
        out[p] = [idx[s] for s in splits if s.size > 0]
    return out


def _golden_max(
    lo: float,
    hi: float,
    fn,
    max_iter: int = 32,
) -> tuple[float, float]:
    gr = 0.6180339887498949
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = fn(c)
    fd = fn(d)

    for _ in range(max_iter):
        if fc > fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = fn(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = fn(d)

    if fc > fd:
        return c, fc
    return d, fd
