from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .metrics import rf_distance
from .pipeline import InferenceConfig, infer_species_tree_ml
from .simulate import simulate_partitioned_dataset


@dataclass(frozen=True)
class ScaleBenchmarkRow:
    n_taxa: int
    d_total: int
    wall_time_sec: float
    rss_mb: float
    log_likelihood: float
    rf_to_true: float
    normalized_rf_to_true: float


def run_scale_benchmark(
    taxa_sizes: list[int],
    dims_per_partition: list[int],
    missing_prob_by_partition: list[float],
    confounder_dim: int,
    nuisance_strength: float,
    config: InferenceConfig,
    seed: int = 0,
) -> list[ScaleBenchmarkRow]:
    if len(missing_prob_by_partition) != len(dims_per_partition):
        raise ValueError("missing_prob_by_partition length must match dims_per_partition")
    rng = np.random.default_rng(seed)
    n_partitions = len(dims_per_partition)
    alpha = np.linspace(0.8, 1.2, n_partitions, dtype=np.float64)
    sigma2 = np.full(n_partitions, 0.1, dtype=np.float64)

    rows: list[ScaleBenchmarkRow] = []
    for n_taxa in taxa_sizes:
        sim_seed = int(rng.integers(0, 1_000_000_000))
        sim = simulate_partitioned_dataset(
            n_taxa=n_taxa,
            dims_per_partition=dims_per_partition,
            alpha_by_partition=alpha,
            sigma2_by_partition=sigma2,
            missing_prob_by_partition=np.asarray(missing_prob_by_partition, dtype=np.float64),
            confounder_dim=confounder_dim,
            nuisance_strength=nuisance_strength,
            seed=sim_seed,
        )

        t0 = time.perf_counter()
        result = infer_species_tree_ml(
            embeddings=sim.embeddings,
            mask=sim.mask,
            dim_to_partition=sim.dim_to_partition,
            covariates=sim.covariates,
            config=config,
            seed=sim_seed,
        )
        dt = time.perf_counter() - t0

        rf = rf_distance(result.tree, sim.tree, normalize=False)
        nrf = rf_distance(result.tree, sim.tree, normalize=True)

        rows.append(
            ScaleBenchmarkRow(
                n_taxa=n_taxa,
                d_total=int(sim.embeddings.shape[1]),
                wall_time_sec=float(dt),
                rss_mb=float(_get_rss_mb()),
                log_likelihood=float(result.log_likelihood),
                rf_to_true=float(rf),
                normalized_rf_to_true=float(nrf),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phylox-benchmark-scale",
        description="Run scaling benchmarks for PhyloGP-ML on synthetic partitioned datasets.",
    )
    parser.add_argument("--taxa-sizes", default="500,1000,5000")
    parser.add_argument("--dims-per-partition", default="64,64,64")
    parser.add_argument("--missing-prob-by-partition", default="0.1,0.2,0.1")
    parser.add_argument("--confounder-dim", type=int, default=5)
    parser.add_argument("--nuisance-strength", type=float, default=0.25)
    parser.add_argument("--n-starts", type=int, default=2)
    parser.add_argument("--bc-rounds", type=int, default=2)
    parser.add_argument("--phase-a-nni-rounds", type=int, default=4)
    parser.add_argument("--phase-c-nni-rounds", type=int, default=4)
    parser.add_argument("--use-spr", action="store_true")
    parser.add_argument("--fit-block-rates", action="store_true")
    parser.add_argument("--robust-student-t", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--quick", action="store_true", help="Use a reduced benchmark profile")
    args = parser.parse_args()

    taxa = [int(x.strip()) for x in args.taxa_sizes.split(",") if x.strip()]
    dims = [int(x.strip()) for x in args.dims_per_partition.split(",") if x.strip()]
    miss = [float(x.strip()) for x in args.missing_prob_by_partition.split(",") if x.strip()]

    if args.quick:
        # Quick mode for smoke-testing benchmark pipeline.
        taxa = [min(200, t) for t in taxa]

    cfg = InferenceConfig(
        n_starts=args.n_starts,
        bc_rounds=args.bc_rounds,
        phase_a_nni_rounds=args.phase_a_nni_rounds,
        phase_c_nni_rounds=args.phase_c_nni_rounds,
        use_spr=args.use_spr,
        fit_block_rates=args.fit_block_rates,
        robust_student_t=args.robust_student_t,
        use_gpu=args.use_gpu,
        gpu_device=args.gpu_device,
    )

    rows = run_scale_benchmark(
        taxa_sizes=taxa,
        dims_per_partition=dims,
        missing_prob_by_partition=miss,
        confounder_dim=args.confounder_dim,
        nuisance_strength=args.nuisance_strength,
        config=cfg,
        seed=args.seed,
    )
    payload = [asdict(r) for r in rows]
    print(json.dumps(payload, indent=2))
    if args.out_json is not None:
        Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _get_rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
