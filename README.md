# phyloX

PhyloGP-ML prototype for maximum-likelihood tree inference from partitioned embedding data.

Current implementation focus:
- M0: core OU masked likelihood engine on fixed topologies.
- M1: masked distances + NJ initialization + branch/parameter refinement.
- M2: NNI hillclimb with edge-disjoint batching and optional SPR escape.
- M3: partitioned species-tree and gene-tree orchestration.
- M4: tier-1 blockwise rates and multi-start search.
- M5: robust Student-t noise refinement (EM-style latent precision updates).
- GPU-capable scoring path for likelihood and batched NNI candidate scoring (PyTorch optional).
- Scale benchmark runner for 500/1k/5k taxa experiments.

## Quick Start

Install (editable):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .[dev]
```

Optional GPU install:

```bash
.venv/bin/python -m pip install -e '.[dev,gpu]'
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

## CLI

Input `.npz` must contain:
- `embeddings`: `(n_taxa, d_total)` float
- `mask`: `(n_taxa, d_total)` bool/int
- `dim_to_partition`: `(d_total,)` int

Optional arrays:
- `covariates`: `(n_taxa, n_covariates)` float
- `partition_weights`: `(n_partitions,)` float
- `taxa`: `(n_taxa,)` string labels

Example:

```bash
.venv/bin/phylox-fit \
  --input data/input_embeddings.npz \
  --out-newick results/tree.nwk \
  --out-npz results/model_fit.npz \
  --n-starts 4 \
  --bc-rounds 3
```

Robust + GPU example:

```bash
.venv/bin/phylox-fit \
  --input data/input_embeddings.npz \
  --out-newick results/tree_robust.nwk \
  --out-npz results/model_fit_robust.npz \
  --robust-student-t \
  --robust-em-rounds 3 \
  --use-gpu \
  --gpu-device cuda
```

Apple Silicon (MPS) example:

```bash
.venv/bin/phylox-fit \
  --input data/input_embeddings.npz \
  --out-newick results/tree_mps.nwk \
  --out-npz results/model_fit_mps.npz \
  --use-gpu \
  --gpu-device mps \
  --gpu-dtype float32
```

Notes:
- `--gpu-device auto` (default) now picks `cuda`, then `mps`, then `cpu`.
- On MPS, unsupported ops are allowed to fall back to CPU.

## Scaling Benchmarks

Run the default 500/1000/5000 benchmark sweep:

```bash
.venv/bin/phylox-benchmark-scale --out-json results/scale_benchmark.json
```

Quick smoke benchmark:

```bash
.venv/bin/phylox-benchmark-scale --quick --taxa-sizes 200,200,200
```
