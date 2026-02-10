from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .distance import masked_euclidean_distance_matrix, neighbor_joining, repair_infinite_distances
from .gpu import build_torch_score_bundle, torch_available
from .optimize import (
    OUModelParameters,
    RefinementResult,
    alternating_refinement,
    build_contiguous_blocks,
    score_ou_model,
)
from .preprocess import coverage_noise_scale, preprocess_embeddings
from .robust import fit_student_t_em
from .search import nni_hillclimb, nni_with_optional_spr
from .tree import PhyloTree


@dataclass(frozen=True)
class InferenceConfig:
    do_confounder_regression: bool = True
    whitening: str = "partition"
    ridge_lambda: float = 1.0
    coverage_mode: str = "partition_fraction_inverse"

    n_starts: int = 4
    start_jitter: float = 0.01

    phase_a_alpha: float = 1.0
    phase_a_sigma2: float = 0.1
    phase_a_nni_rounds: int = 8

    bc_rounds: int = 3
    refine_rounds_per_bc: int = 1
    phase_c_nni_rounds: int = 8
    use_spr: bool = True

    fit_block_rates: bool = False
    n_blocks_per_partition: int = 1

    robust_student_t: bool = False
    robust_em_rounds: int = 2
    robust_dof_init: float = 4.0
    robust_optimize_dof: bool = True
    robust_optimize_alpha: bool = True
    robust_optimize_sigma2: bool = True
    robust_optimize_branch_lengths: bool = True

    use_gpu: bool = False
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    gpu_require_cuda: bool = False
    nni_batch_scoring: bool = True


@dataclass(frozen=True)
class StartResult:
    start_index: int
    tree: PhyloTree
    params: OUModelParameters
    log_likelihood: float
    history: tuple[str, ...]


@dataclass(frozen=True)
class SpeciesTreeResult:
    tree: PhyloTree
    params: OUModelParameters
    log_likelihood: float
    starts: tuple[StartResult, ...]
    preprocessed_embeddings: np.ndarray
    coverage_scale: np.ndarray


@dataclass(frozen=True)
class GeneTreeResult:
    partition_id: int
    taxa_indices: np.ndarray
    tree: PhyloTree
    params: OUModelParameters
    log_likelihood: float


def infer_species_tree_ml(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    covariates: np.ndarray | None = None,
    partition_weights: Sequence[float] | None = None,
    alpha_init: Sequence[float] | None = None,
    sigma2_init: Sequence[float] | None = None,
    config: InferenceConfig | None = None,
    seed: int | None = None,
) -> SpeciesTreeResult:
    cfg = InferenceConfig() if config is None else config
    rng = np.random.default_rng(seed)

    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    dim_map = np.asarray(dim_to_partition, dtype=np.int64)
    if z.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if m.shape != z.shape:
        raise ValueError("mask shape mismatch")
    if dim_map.shape != (z.shape[1],):
        raise ValueError("dim_to_partition shape mismatch")
    if np.any(dim_map < 0):
        raise ValueError("dim_to_partition must be non-negative")

    n_partitions = int(dim_map.max()) + 1 if dim_map.size else 0
    if n_partitions == 0:
        raise ValueError("at least one partition is required")

    cov = covariates if cfg.do_confounder_regression else None
    z_proc, _ = preprocess_embeddings(
        embeddings=z,
        mask=m,
        dim_to_partition=dim_map,
        covariates=cov,
        ridge_lambda=cfg.ridge_lambda,
        whitening=cfg.whitening,
    )
    coverage = coverage_noise_scale(m, dim_map, mode=cfg.coverage_mode)

    if alpha_init is None:
        alpha = np.full(n_partitions, 1.0, dtype=np.float64)
    else:
        alpha = np.asarray(alpha_init, dtype=np.float64)
    if sigma2_init is None:
        sigma2 = np.full(n_partitions, 0.1, dtype=np.float64)
    else:
        sigma2 = np.asarray(sigma2_init, dtype=np.float64)
    if alpha.shape != (n_partitions,) or sigma2.shape != (n_partitions,):
        raise ValueError("alpha_init/sigma2_init length mismatch")

    p_weights = None if partition_weights is None else np.asarray(partition_weights, dtype=np.float64)
    if p_weights is not None and p_weights.shape != (n_partitions,):
        raise ValueError("partition_weights length mismatch")

    base_params = OUModelParameters(
        alpha_by_partition=alpha,
        sigma2_by_partition=sigma2,
        partition_weights=p_weights,
        coverage_scale=coverage,
    )

    starts: list[StartResult] = []
    best: StartResult | None = None

    for s in range(cfg.n_starts):
        dmat = masked_euclidean_distance_matrix(
            embeddings=z_proc,
            mask=m,
            dim_to_partition=dim_map,
            partition_weights=p_weights,
        )
        dmat = repair_infinite_distances(dmat)
        if s > 0 and cfg.start_jitter > 0:
            dmat = _jitter_distances(dmat, jitter=cfg.start_jitter, rng=rng)

        tree = neighbor_joining(dmat).tree
        history: list[str] = [f"start-{s}:nj"]

        # Phase A: topology stabilization under conservative fixed parameters.
        phase_a_params = OUModelParameters(
            alpha_by_partition=np.full(n_partitions, cfg.phase_a_alpha, dtype=np.float64),
            sigma2_by_partition=np.full(n_partitions, cfg.phase_a_sigma2, dtype=np.float64),
            partition_weights=p_weights,
            coverage_scale=coverage,
        )
        score_a, batch_score_a = _build_score_functions(
            embeddings=z_proc,
            mask=m,
            dim_to_partition=dim_map,
            params=phase_a_params,
            config=cfg,
        )
        nni_a = nni_hillclimb(
            tree,
            score_a,
            batch_score_fn=batch_score_a,
            max_rounds=cfg.phase_a_nni_rounds,
            edge_disjoint_batch=True,
        )
        tree = nni_a.tree
        history.append("phase-a-nni")

        # Phase B/C: parameter refinement + topology refinement.
        cur_params = base_params
        cur_ll = score_ou_model(tree, z_proc, m, dim_map, cur_params)

        blocks = None
        if cfg.fit_block_rates:
            blocks = build_contiguous_blocks(dim_map, cfg.n_blocks_per_partition)

        for _ in range(cfg.bc_rounds):
            refined: RefinementResult = alternating_refinement(
                tree=tree,
                embeddings=z_proc,
                mask=m,
                dim_to_partition=dim_map,
                params=cur_params,
                rounds=cfg.refine_rounds_per_bc,
                fit_block_rates=cfg.fit_block_rates,
                blocks=blocks,
            )
            tree = refined.tree
            cur_params = refined.params
            cur_ll = refined.log_likelihood
            history.append("phase-b-refine")

            if cfg.robust_student_t:
                robust = fit_student_t_em(
                    tree=tree,
                    embeddings=z_proc,
                    mask=m,
                    dim_to_partition=dim_map,
                    params=cur_params,
                    rounds=cfg.robust_em_rounds,
                    dof_init=cfg.robust_dof_init,
                    optimize_dof=cfg.robust_optimize_dof,
                    optimize_branch_lengths=cfg.robust_optimize_branch_lengths,
                    optimize_alpha=cfg.robust_optimize_alpha,
                    optimize_sigma2=cfg.robust_optimize_sigma2,
                )
                tree = robust.tree
                cur_params = robust.params
                cur_ll = robust.log_likelihood
                history.append("phase-b-robust-em")

            score_c, batch_score_c = _build_score_functions(
                embeddings=z_proc,
                mask=m,
                dim_to_partition=dim_map,
                params=cur_params,
                config=cfg,
            )
            topo = nni_with_optional_spr(
                tree=tree,
                score_fn=score_c,
                batch_score_fn=batch_score_c,
                outer_rounds=1,
                nni_rounds=cfg.phase_c_nni_rounds,
                use_spr=cfg.use_spr,
            )
            tree = topo.tree
            cur_ll = score_c(tree)
            history.append("phase-c-topology")

        result = StartResult(
            start_index=s,
            tree=tree,
            params=cur_params,
            log_likelihood=cur_ll,
            history=tuple(history),
        )
        starts.append(result)
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result

    assert best is not None
    return SpeciesTreeResult(
        tree=best.tree,
        params=best.params,
        log_likelihood=best.log_likelihood,
        starts=tuple(starts),
        preprocessed_embeddings=z_proc,
        coverage_scale=coverage,
    )


def infer_gene_trees_ml(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    covariates: np.ndarray | None = None,
    partition_weights: Sequence[float] | None = None,
    config: InferenceConfig | None = None,
    seed: int | None = None,
) -> list[GeneTreeResult]:
    cfg = InferenceConfig() if config is None else config
    z = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    dim_map = np.asarray(dim_to_partition, dtype=np.int64)
    n_partitions = int(dim_map.max()) + 1 if dim_map.size else 0

    out: list[GeneTreeResult] = []
    for p in range(n_partitions):
        idx = np.flatnonzero(dim_map == p)
        if idx.size == 0:
            continue
        taxa = np.flatnonzero(np.any(m[:, idx], axis=1))
        if taxa.size < 4:
            continue

        z_p = z[np.ix_(taxa, idx)]
        m_p = m[np.ix_(taxa, idx)]
        dim_p = np.zeros(idx.size, dtype=np.int64)
        cov_p = None if covariates is None else np.asarray(covariates)[taxa]
        w_p = None
        if partition_weights is not None:
            wp = float(np.asarray(partition_weights, dtype=np.float64)[p])
            w_p = np.asarray([wp], dtype=np.float64)

        result = infer_species_tree_ml(
            embeddings=z_p,
            mask=m_p,
            dim_to_partition=dim_p,
            covariates=cov_p,
            partition_weights=w_p,
            config=cfg,
            seed=seed,
        )
        out.append(
            GeneTreeResult(
                partition_id=p,
                taxa_indices=taxa,
                tree=result.tree,
                params=result.params,
                log_likelihood=result.log_likelihood,
            )
        )
    return out


def _jitter_distances(dmat: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    D = np.asarray(dmat, dtype=np.float64).copy()
    n = D.shape[0]
    finite = np.isfinite(D)
    # Add symmetric relative jitter to finite off-diagonal entries.
    for i in range(n):
        for j in range(i + 1, n):
            if not finite[i, j]:
                continue
            scale = max(D[i, j], 1e-8)
            noise = rng.normal(loc=0.0, scale=jitter * scale)
            val = max(D[i, j] + noise, 1e-8)
            D[i, j] = val
            D[j, i] = val
    np.fill_diagonal(D, 0.0)
    return D


def _build_score_functions(
    embeddings: np.ndarray,
    mask: np.ndarray,
    dim_to_partition: np.ndarray,
    params: OUModelParameters,
    config: InferenceConfig,
) -> tuple:
    if config.use_gpu and torch_available(require_cuda=config.gpu_require_cuda):
        try:
            bundle = build_torch_score_bundle(
                embeddings=embeddings,
                mask=mask,
                dim_to_partition=dim_to_partition,
                params=params,
                device=config.gpu_device,
                dtype=config.gpu_dtype,
            )
            batch = bundle.batch_score_fn if config.nni_batch_scoring else None
            return bundle.score_fn, batch
        except Exception:
            pass
    score = lambda t: score_ou_model(t, embeddings, mask, dim_to_partition, params)
    return score, None
