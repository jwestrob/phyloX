# Task Board

| ID | Milestone | Task | Owner | Status | Depends On | Scope |
|---|---|---|---|---|---|---|
| PX-M0-01 | M0 | Build masked partitioned OU likelihood on fixed tree | Codex | Review | - | Core likelihood kernel + API |
| PX-M0-02 | M0 | Add OU simulation and recovery validation fixtures | Unassigned | Backlog | PX-M0-01 | Simulation harness + correctness tests |
| PX-M0-03 | M0 | Root invariance and numerical stability checks | Unassigned | Backlog | PX-M0-01 | Invariance tests + precision checks |
| PX-M1-01 | M1 | Implement masked pairwise distances and NJ init | Codex | Review | PX-M0-01 | Distance builder + NJ topology init |
| PX-M1-02 | M1 | Optimize branch lengths on fixed topology | Unassigned | Backlog | PX-M1-01 | Branch-length optimizer |
| PX-M1-03 | M1 | Fit partition-specific `alpha_g` and `sigma_g^2` | Unassigned | Backlog | PX-M1-02 | Parameter refinement loop |
| PX-M2-01 | M2 | Implement directional message pass caches | Unassigned | Backlog | PX-M1-03 | Postorder/preorder cache infra |
| PX-M2-02 | M2 | Batch NNI scoring for all internal edges | Unassigned | Backlog | PX-M2-01 | GPU/torch batched scoring |
| PX-M2-03 | M2 | Edge-disjoint NNI application loop | Unassigned | Backlog | PX-M2-02 | Move selection + iterative hillclimb |
| PX-M3-01 | M3 | Enable partitioned concatenation species-tree mode | Unassigned | Backlog | PX-M2-03 | Multi-partition shared-topology path |
| PX-M3-02 | M3 | Add coverage-aware noise scaling for missingness | Unassigned | Backlog | PX-M3-01 | Missing-data robustness |
| PX-M3-03 | M3 | Scale tests for `n_taxa` 500/1k/5k | Unassigned | Backlog | PX-M3-02 | Runtime + memory benchmarks |
| PX-M4-01 | M4 | Add blockwise rate heterogeneity | Unassigned | Backlog | PX-M3-02 | Tier-1 ASRV analogue |
| PX-M4-02 | M4 | Add multi-start orchestration and best-of-run selection | Unassigned | Backlog | PX-M2-03 | Search robustness tooling |
| PX-M5-01 | M5 | Add limited-radius SPR escape moves | Unassigned | Backlog | PX-M4-02 | Optional topology escape |
| PX-M5-02 | M5 | Add robust Student-t noise (EM scale-mixture) | Unassigned | Backlog | PX-M4-01 | Optional robustness model |

## Notes
- Claim a task by setting `Owner` and moving `Status` to `In Progress`.
- Keep one active task per owner to reduce context switching.
- If blocked, set `Status` to `Blocked` and describe the blocker in the commit/PR/handoff.
