# Handoff

## Task
- ID: PX-M3-03, PX-M5-02 (plus GPU scoring completion for M2 throughput path)
- Title: Final milestone completion pass (scale benchmarking + Student-t EM + GPU scoring hooks)
- Owner: Codex

## Scope Completed
- Added robust Student-t EM module with latent precision updates and optional dof fitting.
- Integrated robust refinement into species-tree pipeline and CLI flags.
- Added optional torch GPU scoring path and batched NNI scoring hooks.
- Added scale benchmark runner and CLI command for 500/1k/5k regimes.
- Added initialization-time distance repair for disconnected masked-distance matrices.
- Expanded tests for robust model, GPU parity/fallback, batched NNI scoring, benchmark smoke path, and distance repair.

## Files Touched
- `src/phylox/ou_likelihood.py`
- `src/phylox/optimize.py`
- `src/phylox/search.py`
- `src/phylox/pipeline.py`
- `src/phylox/cli.py`
- `src/phylox/gpu.py`
- `src/phylox/robust.py`
- `src/phylox/benchmark.py`
- `src/phylox/distance.py`
- `src/phylox/__init__.py`
- `benchmarks/scale_benchmark.py`
- `tests/test_robust_and_gpu.py`
- `tests/test_benchmark.py`
- `tests/test_distance_and_nj.py`
- `tests/test_optimize_and_search.py`
- `README.md`
- `pyproject.toml`
- `docs/planning/TASK_BOARD.md`

## Validation
- Commands run:
  - `.venv/bin/python -m pytest -q`
  - `.venv/bin/phylox-benchmark-scale --quick ...`
  - `.venv/bin/phylox-fit ... --robust-student-t ...`
- Results:
  - `28 passed`
  - benchmark smoke run completed and produced JSON output
  - robust CLI smoke run completed and produced Newick + fit output

## Risks / Open Issues
- GPU path depends on optional PyTorch install and currently uses Python-level tree loops (GPU tensor math, but not custom kernels).
- Student-t EM uses a practical approximate E-step with standardized residuals; it is robust in practice but not a full exact latent-state EM.
- Full 500/1k/5k runs are available through benchmark command but were not executed end-to-end in this pass due runtime.

## Next Recommended Step
- Suggested next owner: Performance Agent
- Suggested follow-up task ID: PX-M2-02 (CUDA/Triton kernelization of tree scoring loop)
