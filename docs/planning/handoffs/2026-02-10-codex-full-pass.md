# Handoff

## Task
- ID: PX-M0-02, PX-M0-03, PX-M1-02, PX-M1-03, PX-M2-01, PX-M2-02, PX-M2-03, PX-M3-01, PX-M3-02, PX-M4-01, PX-M4-02, PX-M5-01
- Title: End-to-end PhyloGP-ML CPU reference implementation
- Owner: Codex

## Scope Completed
- Implemented partitioned data and preprocessing stack (confounder regression, ZCA whitening, coverage scaling).
- Extended OU likelihood with tier-1 blockwise rates and optional discrete-Gamma integration.
- Added branch-length and parameter refinement optimizers with alternating schedule.
- Added NNI hillclimb with edge-disjoint batching and limited-radius SPR escape moves.
- Added simulation harness, topology metrics, species-tree pipeline, and gene-tree mode.
- Added Newick export and CLI entrypoint (`phylox-fit`).
- Added comprehensive tests for new modules.

## Files Touched
- `src/phylox/data.py`
- `src/phylox/preprocess.py`
- `src/phylox/ou_likelihood.py`
- `src/phylox/optimize.py`
- `src/phylox/search.py`
- `src/phylox/simulate.py`
- `src/phylox/metrics.py`
- `src/phylox/pipeline.py`
- `src/phylox/cli.py`
- `src/phylox/tree.py`
- `src/phylox/__init__.py`
- `tests/test_data_and_preprocess.py`
- `tests/test_ou_rates_gamma.py`
- `tests/test_optimize_and_search.py`
- `tests/test_pipeline_and_metrics.py`
- `README.md`
- `pyproject.toml`
- `docs/planning/TASK_BOARD.md`

## Validation
- Commands run:
  - `.venv/bin/python -m pip install -e '.[dev]'`
  - `.venv/bin/python -m pytest -q`
  - `.venv/bin/phylox-fit --input <simulated.npz> --out-newick <tree.nwk> --out-npz <fit.npz> ...`
- Results:
  - `21 passed`
  - CLI smoke test succeeded and produced Newick + fit artifact.

## Risks / Open Issues
- Current runtime path is CPU/numpy only; no CUDA/Triton kernels yet.
- NNI scoring is batched logically but not GPU-accelerated.
- Scale benchmarks for 500/1k/5k taxa are not yet implemented.
- Robust Student-t EM noise path is not yet implemented.

## Next Recommended Step
- Suggested next owner: Performance Agent
- Suggested follow-up task ID: PX-M3-03
