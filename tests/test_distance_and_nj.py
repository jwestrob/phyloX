import numpy as np

from phylox.distance import masked_euclidean_distance_matrix, neighbor_joining, repair_infinite_distances


def test_masked_distance_matrix_basic():
    z = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            [3.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    m = np.asarray(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]
    )

    d = masked_euclidean_distance_matrix(z, m)
    assert d.shape == (3, 3)
    assert np.isclose(d[0, 1], np.sqrt((0.0 + 1.0 + 4.0) / 3.0))
    assert np.isclose(d[0, 2], np.sqrt((9.0 + 0.0) / 2.0))
    assert np.isclose(d[1, 2], np.sqrt((9.0 + 1.0) / 2.0))


def test_masked_distance_with_no_overlap_is_inf():
    z = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    m = np.asarray(
        [
            [True, False],
            [False, True],
        ]
    )
    d = masked_euclidean_distance_matrix(z, m)
    assert np.isinf(d[0, 1])


def test_repair_infinite_distances_makes_matrix_finite():
    d = np.asarray(
        [
            [0.0, np.inf, 2.0],
            [np.inf, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    fixed = repair_infinite_distances(d)
    assert np.all(np.isfinite(fixed))
    assert np.allclose(fixed, fixed.T)
    assert np.allclose(np.diag(fixed), 0.0)


def test_neighbor_joining_builds_valid_tree():
    d = np.asarray(
        [
            [0.0, 5.0, 9.0, 9.0],
            [5.0, 0.0, 10.0, 10.0],
            [9.0, 10.0, 0.0, 8.0],
            [9.0, 10.0, 8.0, 0.0],
        ],
        dtype=np.float64,
    )
    result = neighbor_joining(d)
    tree = result.tree
    n = d.shape[0]

    assert tree.num_nodes == 2 * n - 2
    assert len(tree.edges) == 2 * n - 3

    degree = np.zeros(tree.num_nodes, dtype=int)
    for u, v, t in tree.edges:
        degree[u] += 1
        degree[v] += 1
        assert t > 0

    for leaf in range(n):
        assert degree[leaf] == 1
