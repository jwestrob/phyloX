# PhyloGP-ML Roadmap

This roadmap translates the project plan into executable milestones.

## M0: Core Likelihood Correctness
Goal: OU pruning likelihood on fixed topology with masking.

Deliverables:
- Partition-aware masked OU likelihood engine.
- Root invariance checks.
- OU simulation harness with known-tree recovery tests.

Exit Criteria:
- Likelihood stable and reproducible on simulation fixtures.
- Root placement does not change final log-likelihood.

## M1: NJ Init + Parameter Refinement
Goal: Build a strong fixed-topology optimization baseline.

Deliverables:
- Masked-distance matrix and NJ tree initialization.
- Branch-length optimization on fixed topology.
- Partition-specific `alpha_g` and `sigma_g^2` fitting under identifiability constraints.

Exit Criteria:
- Refined likelihood improves over raw NJ initialization.
- Parameter optimization converges robustly on benchmark subsets.

## M2: GPU-Batched NNI Hillclimb
Goal: ML topology search at scale.

Deliverables:
- Directional message passes (postorder + preorder).
- Batched scoring of internal-edge NNI alternatives.
- Edge-disjoint parallel move application.

Exit Criteria:
- Topology likelihood improves beyond M1 baseline.
- Throughput scales with taxa and embedding dimensionality.

## M3: Partitioned Concatenation Species Trees
Goal: Multi-gene species-tree inference as default mode.

Deliverables:
- Shared topology/branch lengths with per-partition OU params.
- Missing-data masking and coverage-aware noise scaling.
- Benchmarks at `n_taxa` up to 5k.

Exit Criteria:
- Improved topology metrics vs distance-only baselines.
- Stable performance at high missingness regimes.

## M4: Rate Heterogeneity Tier 1 + Multi-Start
Goal: Increase robustness to embedding idiosyncrasies.

Deliverables:
- Blockwise per-partition rate scalers.
- Regularized optimization for rate parameters.
- Multi-start search orchestration.

Exit Criteria:
- Better or more stable topologies across replicated runs.

## M5: Optional Accuracy Extensions
Goal: Deploy only if M4 plateaus on accuracy.

Deliverables:
- Limited SPR escape moves.
- Robust observation noise option (scale-mixture Student-t).

Exit Criteria:
- Clear accuracy gains on target evaluation regimes.
