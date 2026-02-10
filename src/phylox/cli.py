from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import InferenceConfig, infer_species_tree_ml
from .tree import to_newick


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phylox-fit",
        description="Infer an ML phylogeny from partitioned embedding matrices.",
    )
    parser.add_argument("--input", required=True, help="Input .npz with embeddings/mask/dim_to_partition")
    parser.add_argument("--out-newick", required=True, help="Output Newick file path")
    parser.add_argument("--out-npz", default=None, help="Optional output .npz for fitted parameters")
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--phase-a-nni-rounds", type=int, default=8)
    parser.add_argument("--bc-rounds", type=int, default=3)
    parser.add_argument("--phase-c-nni-rounds", type=int, default=8)
    parser.add_argument("--no-spr", action="store_true", help="Disable SPR escape moves")
    parser.add_argument("--whitening", choices=["partition", "global", "none"], default="partition")
    parser.add_argument("--no-confounder-regression", action="store_true")
    parser.add_argument("--robust-student-t", action="store_true", help="Enable Student-t EM noise refinement")
    parser.add_argument("--robust-em-rounds", type=int, default=2)
    parser.add_argument("--robust-dof-init", type=float, default=4.0)
    parser.add_argument("--no-robust-optimize-dof", action="store_true")
    parser.add_argument("--use-gpu", action="store_true", help="Use torch-based scoring when available")
    parser.add_argument("--gpu-device", default="auto", help="auto|cuda|cpu|mps")
    parser.add_argument("--gpu-dtype", default="float32", help="float32|float64|float16|bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float64)
    mask = np.asarray(data["mask"], dtype=bool)
    dim_to_partition = np.asarray(data["dim_to_partition"], dtype=np.int64)
    covariates = np.asarray(data["covariates"], dtype=np.float64) if "covariates" in data else None
    partition_weights = (
        np.asarray(data["partition_weights"], dtype=np.float64) if "partition_weights" in data else None
    )
    taxa = data["taxa"].tolist() if "taxa" in data else [f"T{i}" for i in range(embeddings.shape[0])]

    cfg = InferenceConfig(
        do_confounder_regression=not args.no_confounder_regression,
        whitening=args.whitening,
        n_starts=args.n_starts,
        phase_a_nni_rounds=args.phase_a_nni_rounds,
        bc_rounds=args.bc_rounds,
        phase_c_nni_rounds=args.phase_c_nni_rounds,
        use_spr=not args.no_spr,
        robust_student_t=args.robust_student_t,
        robust_em_rounds=args.robust_em_rounds,
        robust_dof_init=args.robust_dof_init,
        robust_optimize_dof=not args.no_robust_optimize_dof,
        use_gpu=args.use_gpu,
        gpu_device=args.gpu_device,
        gpu_dtype=args.gpu_dtype,
    )

    result = infer_species_tree_ml(
        embeddings=embeddings,
        mask=mask,
        dim_to_partition=dim_to_partition,
        covariates=covariates,
        partition_weights=partition_weights,
        config=cfg,
        seed=args.seed,
    )

    newick = to_newick(result.tree, taxon_labels=list(map(str, taxa)))
    Path(args.out_newick).write_text(newick + "\n", encoding="utf-8")

    if args.out_npz is not None:
        edges = np.asarray(result.tree.edges, dtype=np.float64)
        np.savez_compressed(
            args.out_npz,
            edges=edges,
            leaf_count=np.asarray([result.tree.leaf_count], dtype=np.int64),
            alpha_by_partition=result.params.alpha_by_partition,
            sigma2_by_partition=result.params.sigma2_by_partition,
            partition_weights=(
                result.params.partition_weights
                if result.params.partition_weights is not None
                else np.ones_like(result.params.alpha_by_partition)
            ),
            log_likelihood=np.asarray([result.log_likelihood], dtype=np.float64),
        )

    summary = {
        "n_taxa": int(embeddings.shape[0]),
        "d_total": int(embeddings.shape[1]),
        "n_partitions": int(np.max(dim_to_partition) + 1) if dim_to_partition.size else 0,
        "log_likelihood": float(result.log_likelihood),
        "n_starts": int(len(result.starts)),
        "robust_student_t": bool(args.robust_student_t),
        "use_gpu": bool(args.use_gpu),
        "out_newick": str(args.out_newick),
        "out_npz": str(args.out_npz) if args.out_npz is not None else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
