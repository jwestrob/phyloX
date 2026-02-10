import numpy as np

from phylox.ou_likelihood import ou_log_likelihood
from phylox.tree import PhyloTree, pairwise_leaf_distances


def _example_tree() -> PhyloTree:
    # Leaves: 0,1,2,3. Internal: 4,5.
    edges = [
        (4, 0, 0.20),
        (4, 1, 0.30),
        (4, 5, 0.10),
        (5, 2, 0.25),
        (5, 3, 0.35),
    ]
    return PhyloTree(num_nodes=6, edges=edges, leaf_count=4)


def _dense_ou_ll_per_dim(
    tree: PhyloTree,
    y: np.ndarray,
    obs: np.ndarray,
    alpha: float,
    sigma2: float,
    coverage: np.ndarray,
) -> float:
    idx = np.flatnonzero(obs)
    if idx.size == 0:
        return 0.0

    dist = pairwise_leaf_distances(tree)
    cov = np.exp(-alpha * dist[np.ix_(idx, idx)])
    cov[np.diag_indices_from(cov)] += sigma2 * coverage[idx]

    y_obs = y[idx]
    chol = np.linalg.cholesky(cov)
    solve = np.linalg.solve(chol, y_obs)
    quad = float(np.dot(solve, solve))
    logdet = float(2.0 * np.sum(np.log(np.diag(chol))))
    n = idx.size
    return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)


def test_ou_matches_dense_gaussian_baseline():
    rng = np.random.default_rng(42)
    tree = _example_tree()
    n_taxa = tree.leaf_count
    d_total = 7

    z = rng.normal(size=(n_taxa, d_total))
    m = rng.random(size=(n_taxa, d_total)) > 0.25
    # Ensure every dim has at least one observation.
    for k in range(d_total):
        if not np.any(m[:, k]):
            m[rng.integers(0, n_taxa), k] = True

    dim_to_partition = np.asarray([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    alpha = np.asarray([0.7, 1.3, 0.9], dtype=np.float64)
    sigma2 = np.asarray([0.10, 0.20, 0.15], dtype=np.float64)
    weights = np.asarray([1.0, 0.5, 1.2], dtype=np.float64)
    coverage = np.asarray(
        [
            [1.0, 1.1, 0.9],
            [1.2, 1.0, 1.0],
            [0.9, 1.3, 1.1],
            [1.1, 0.8, 1.0],
        ],
        dtype=np.float64,
    )

    ll = ou_log_likelihood(
        tree=tree,
        embeddings=z,
        mask=m,
        dim_to_partition=dim_to_partition,
        alpha_by_partition=alpha,
        sigma2_by_partition=sigma2,
        partition_weights=weights,
        coverage_scale=coverage,
    )

    dense = 0.0
    for k in range(d_total):
        p = dim_to_partition[k]
        dense += weights[p] * _dense_ou_ll_per_dim(
            tree=tree,
            y=z[:, k],
            obs=m[:, k],
            alpha=float(alpha[p]),
            sigma2=float(sigma2[p]),
            coverage=coverage[:, p],
        )

    assert np.isclose(ll, dense, atol=1e-8, rtol=1e-8)


def test_root_invariance():
    rng = np.random.default_rng(7)
    tree = _example_tree()

    z = rng.normal(size=(tree.leaf_count, 5))
    m = rng.random(size=(tree.leaf_count, 5)) > 0.2
    dim_to_partition = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    alpha = np.asarray([0.9, 1.1], dtype=np.float64)
    sigma2 = np.asarray([0.1, 0.2], dtype=np.float64)

    ll_r4 = ou_log_likelihood(tree, z, m, dim_to_partition, alpha, sigma2, root=4)
    ll_r5 = ou_log_likelihood(tree, z, m, dim_to_partition, alpha, sigma2, root=5)
    ll_r0 = ou_log_likelihood(tree, z, m, dim_to_partition, alpha, sigma2, root=0)

    assert np.isclose(ll_r4, ll_r5, atol=1e-10, rtol=1e-10)
    assert np.isclose(ll_r4, ll_r0, atol=1e-10, rtol=1e-10)


def test_all_missing_dimension_contributes_zero():
    tree = _example_tree()
    z = np.zeros((tree.leaf_count, 3), dtype=np.float64)
    m = np.asarray(
        [
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
        ]
    )
    dim_to_partition = np.asarray([0, 0, 1], dtype=np.int64)
    alpha = np.asarray([1.0, 1.0], dtype=np.float64)
    sigma2 = np.asarray([0.1, 0.1], dtype=np.float64)

    ll_with_missing_dim = ou_log_likelihood(tree, z, m, dim_to_partition, alpha, sigma2)
    ll_without_missing_dim = ou_log_likelihood(
        tree,
        z[:, :2],
        m[:, :2],
        dim_to_partition=np.asarray([0, 0], dtype=np.int64),
        alpha_by_partition=np.asarray([1.0], dtype=np.float64),
        sigma2_by_partition=np.asarray([0.1], dtype=np.float64),
    )
    assert np.isclose(ll_with_missing_dim, ll_without_missing_dim, atol=1e-12, rtol=1e-12)
