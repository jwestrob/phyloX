# Handoff

## Task
- ID: PX-M0-01, PX-M1-01
- Title: OU masked likelihood core; masked distance + NJ initialization
- Owner: Codex

## Scope Completed
- Added a first-pass OU log-likelihood engine with partition-specific parameters.
- Added masked pairwise distance computation and Neighbor-Joining tree initialization.
- Added tests validating likelihood correctness and core distance/NJ behavior.
- Established Python project scaffold and package exports.

## Files Touched
- `pyproject.toml`
- `README.md`
- `.gitignore`
- `src/phylox/__init__.py`
- `src/phylox/tree.py`
- `src/phylox/ou_likelihood.py`
- `src/phylox/distance.py`
- `tests/test_ou_likelihood.py`
- `tests/test_distance_and_nj.py`
- `docs/planning/TASK_BOARD.md`

## Validation
- Commands run: `.venv/bin/python -m pytest -q`
- Results: `6 passed`

## Risks / Open Issues
- Current implementation is CPU/numpy only; GPU pruning and batched NNI are not yet implemented.
- NJ currently fails fast on disconnected taxa pairs (`+inf` distances) instead of repairing/smoothing.
- No CLI or pipeline orchestration yet.

## Next Recommended Step
- Suggested next owner: Modeling Agent
- Suggested follow-up task ID: PX-M0-02
