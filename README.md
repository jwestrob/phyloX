# phyloX

PhyloGP-ML prototype for maximum-likelihood tree inference from partitioned embedding data.

Current implementation focus:
- M0: core OU masked likelihood engine on fixed topologies.
- M1: masked distances + NJ initialization + branch/parameter refinement.
- M2: NNI hillclimb with edge-disjoint batching and optional SPR escape.
- M3: partitioned species-tree and gene-tree orchestration.
- M4: tier-1 blockwise rates and multi-start search.

## Quick Start

Install (editable):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .[dev]
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
