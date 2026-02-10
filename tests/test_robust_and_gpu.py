import numpy as np

from phylox.gpu import ou_log_likelihood_torch, torch_available
from phylox.optimize import OUModelParameters, score_ou_model
from phylox.ou_likelihood import ou_log_likelihood
from phylox.pipeline import InferenceConfig, infer_species_tree_ml
from phylox.robust import fit_student_t_em
from phylox.simulate import simulate_ou_embeddings, simulate_partitioned_dataset
from phylox.tree import PhyloTree


def _tree() -> PhyloTree:
    edges = [
        (4, 0, 0.2),
        (4, 1, 0.3),
        (4, 5, 0.1),
        (5, 2, 0.25),
        (5, 3, 0.35),
    ]
    return PhyloTree(num_nodes=6, edges=edges, leaf_count=4)


def test_precision_weights_are_supported():
    rng = np.random.default_rng(3)
    tree = _tree()
    z = rng.normal(size=(4, 5))
    m = np.ones_like(z, dtype=bool)
    dim_map = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    alpha = np.asarray([0.9, 1.1], dtype=np.float64)
    sigma2 = np.asarray([0.1, 0.1], dtype=np.float64)

    ll_base = ou_log_likelihood(tree, z, m, dim_map, alpha, sigma2)
    prec = np.ones_like(z, dtype=np.float64)
    prec[0, 0] = 0.2
    ll_weighted = ou_log_likelihood(
        tree,
        z,
        m,
        dim_map,
        alpha,
        sigma2,
        precision_weights=prec,
    )
    assert np.isfinite(ll_weighted)
    assert not np.isclose(ll_base, ll_weighted)


def test_student_t_em_downweights_outlier():
    tree = _tree()
    dim_map = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    alpha = np.asarray([0.9, 1.2], dtype=np.float64)
    sigma2 = np.asarray([0.08, 0.12], dtype=np.float64)
    z, m = simulate_ou_embeddings(tree, dim_map, alpha, sigma2, seed=12)
    z = z.copy()
    out_i, out_k = 1, 4
    z[out_i, out_k] += 8.0

    params = OUModelParameters(alpha_by_partition=alpha, sigma2_by_partition=sigma2)
    ll0 = score_ou_model(tree, z, m, dim_map, params)
    robust = fit_student_t_em(
        tree=tree,
        embeddings=z,
        mask=m,
        dim_to_partition=dim_map,
        params=params,
        rounds=2,
        optimize_branch_lengths=False,
        optimize_alpha=False,
        optimize_sigma2=True,
    )

    assert robust.params.precision_weights is not None
    assert robust.params.precision_weights[out_i, out_k] < 1.0
    assert robust.log_likelihood >= ll0 - 1e-8


def test_pipeline_gpu_flag_runs_even_without_torch_cuda():
    sim = simulate_partitioned_dataset(
        n_taxa=8,
        dims_per_partition=[6, 6],
        alpha_by_partition=np.asarray([0.8, 1.1]),
        sigma2_by_partition=np.asarray([0.1, 0.1]),
        seed=111,
    )
    cfg = InferenceConfig(
        n_starts=1,
        bc_rounds=1,
        phase_a_nni_rounds=2,
        phase_c_nni_rounds=2,
        use_spr=False,
        use_gpu=True,
        gpu_device="cuda",
    )
    out = infer_species_tree_ml(
        embeddings=sim.embeddings,
        mask=sim.mask,
        dim_to_partition=sim.dim_to_partition,
        covariates=sim.covariates,
        config=cfg,
        seed=2,
    )
    assert np.isfinite(out.log_likelihood)


def test_torch_likelihood_matches_numpy_when_available():
    if not torch_available():
        return
    rng = np.random.default_rng(123)
    tree = _tree()
    z = rng.normal(size=(4, 5))
    m = rng.random(size=(4, 5)) > 0.1
    dim_map = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    alpha = np.asarray([0.9, 1.1], dtype=np.float64)
    sigma2 = np.asarray([0.1, 0.2], dtype=np.float64)

    ll_np = ou_log_likelihood(tree, z, m, dim_map, alpha, sigma2)
    ll_t = ou_log_likelihood_torch(
        tree=tree,
        embeddings=z,
        mask=m,
        dim_to_partition=dim_map,
        alpha_by_partition=alpha,
        sigma2_by_partition=sigma2,
        device="cpu",
        dtype="float64",
    )
    assert np.isclose(ll_np, ll_t, atol=1e-7, rtol=1e-7)
