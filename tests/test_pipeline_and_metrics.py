import numpy as np

from phylox.distance import masked_euclidean_distance_matrix, neighbor_joining
from phylox.metrics import rf_distance
from phylox.optimize import score_ou_model
from phylox.pipeline import InferenceConfig, infer_gene_trees_ml, infer_species_tree_ml
from phylox.simulate import simulate_partitioned_dataset, simulate_random_binary_tree


def test_rf_distance_identity_is_zero():
    t = simulate_random_binary_tree(n_taxa=8, seed=1)
    assert rf_distance(t, t) == 0.0
    assert rf_distance(t, t, normalize=True) == 0.0


def test_species_pipeline_runs_and_improves_over_nj_under_final_params():
    sim = simulate_partitioned_dataset(
        n_taxa=10,
        dims_per_partition=[12, 12, 12],
        alpha_by_partition=np.asarray([0.8, 1.0, 1.2]),
        sigma2_by_partition=np.asarray([0.08, 0.08, 0.08]),
        missing_prob_by_partition=np.asarray([0.1, 0.2, 0.1]),
        confounder_dim=3,
        nuisance_strength=0.3,
        seed=2026,
    )
    cfg = InferenceConfig(
        do_confounder_regression=True,
        whitening="partition",
        n_starts=2,
        bc_rounds=2,
        phase_a_nni_rounds=4,
        phase_c_nni_rounds=4,
        use_spr=True,
    )
    result = infer_species_tree_ml(
        embeddings=sim.embeddings,
        mask=sim.mask,
        dim_to_partition=sim.dim_to_partition,
        covariates=sim.covariates,
        config=cfg,
        seed=7,
    )
    assert np.isfinite(result.log_likelihood)
    assert len(result.starts) == 2

    dmat = masked_euclidean_distance_matrix(
        embeddings=result.preprocessed_embeddings,
        mask=sim.mask,
        dim_to_partition=sim.dim_to_partition,
        partition_weights=result.params.partition_weights,
    )
    nj = neighbor_joining(dmat).tree
    ll_nj = score_ou_model(
        nj,
        result.preprocessed_embeddings,
        sim.mask,
        sim.dim_to_partition,
        result.params,
    )
    ll_final = score_ou_model(
        result.tree,
        result.preprocessed_embeddings,
        sim.mask,
        sim.dim_to_partition,
        result.params,
    )
    assert ll_final >= ll_nj - 1e-8


def test_gene_tree_mode_runs():
    sim = simulate_partitioned_dataset(
        n_taxa=9,
        dims_per_partition=[6, 8, 10],
        alpha_by_partition=np.asarray([0.9, 1.1, 1.3]),
        sigma2_by_partition=np.asarray([0.1, 0.1, 0.1]),
        missing_prob_by_partition=np.asarray([0.0, 0.1, 0.2]),
        confounder_dim=2,
        nuisance_strength=0.1,
        seed=19,
    )
    cfg = InferenceConfig(n_starts=1, bc_rounds=1, phase_a_nni_rounds=2, phase_c_nni_rounds=2, use_spr=False)
    results = infer_gene_trees_ml(
        embeddings=sim.embeddings,
        mask=sim.mask,
        dim_to_partition=sim.dim_to_partition,
        covariates=sim.covariates,
        config=cfg,
        seed=4,
    )
    assert len(results) >= 1
    for gr in results:
        assert np.isfinite(gr.log_likelihood)
