import numpy as np

from phylox.ou_likelihood import make_blockwise_rate_by_dim, ou_log_likelihood
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


def test_gamma_single_category_matches_base_likelihood():
    rng = np.random.default_rng(31)
    tree = _tree()
    z = rng.normal(size=(4, 6))
    m = rng.random(size=(4, 6)) > 0.1
    dim_map = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    alpha = np.asarray([0.8, 1.2], dtype=np.float64)
    sigma2 = np.asarray([0.1, 0.15], dtype=np.float64)

    ll_base = ou_log_likelihood(tree, z, m, dim_map, alpha, sigma2)
    ll_gamma = ou_log_likelihood(
        tree,
        z,
        m,
        dim_map,
        alpha,
        sigma2,
        gamma_rates_by_partition={1: (np.asarray([1.0]), np.asarray([1.0]))},
    )
    assert np.isclose(ll_base, ll_gamma, atol=1e-10, rtol=1e-10)


def test_make_blockwise_rate_by_dim():
    dim_map = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    rates = make_blockwise_rate_by_dim(
        dim_map,
        blocks_per_partition={0: 2, 1: 2},
        rates_by_partition_block={0: [0.5, 2.0], 1: [1.0, 1.5]},
    )
    assert rates.shape == (7,)
    assert np.all(rates > 0)
    assert np.isclose(rates[0], 0.5)
    assert np.isclose(rates[2], 2.0)
