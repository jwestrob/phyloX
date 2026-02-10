import numpy as np

from phylox.data import (
    concatenate_partition_embeddings,
    compute_partition_weights,
    partition_presence_from_mask,
)
from phylox.preprocess import (
    coverage_noise_scale,
    regress_out_confounders,
    zca_whiten_partitions,
)


def test_concatenate_partitions_and_weights():
    p1 = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [np.nan, np.nan],
        ]
    )
    p2 = np.asarray(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    data = concatenate_partition_embeddings({"g1": p1, "g2": p2}, weight_scheme="inverse_dims")
    assert data.embeddings.shape == (3, 3)
    assert data.mask.shape == (3, 3)
    assert data.partition_names == ("g1", "g2")
    assert np.all(data.dim_to_partition == np.asarray([0, 0, 1]))
    assert np.allclose(data.partition_weights, np.asarray([0.5, 1.0]))


def test_partition_presence_from_mask():
    mask = np.asarray(
        [
            [True, True, False, False],
            [False, False, True, True],
            [True, True, True, True],
        ]
    )
    dim_to_partition = np.asarray([0, 0, 1, 1], dtype=np.int64)
    presence = partition_presence_from_mask(mask, dim_to_partition)
    expected = np.asarray(
        [
            [True, False],
            [False, True],
            [True, True],
        ]
    )
    assert np.array_equal(presence, expected)


def test_regress_out_confounders_reduces_signal():
    rng = np.random.default_rng(11)
    n = 80
    d = 6
    cov = rng.normal(size=(n, 2))
    beta = rng.normal(size=(2, d))
    z = cov @ beta + 0.05 * rng.normal(size=(n, d))
    mask = np.ones_like(z, dtype=bool)
    dim_to_partition = np.zeros(d, dtype=np.int64)

    before = np.mean(np.abs(np.corrcoef(cov[:, 0], z[:, 0])[0, 1]))
    residual = regress_out_confounders(
        embeddings=z,
        mask=mask,
        dim_to_partition=dim_to_partition,
        covariates=cov,
        ridge_lambda=1.0,
    )
    after = np.mean(np.abs(np.corrcoef(cov[:, 0], residual[:, 0])[0, 1]))
    assert after < before


def test_zca_whiten_partitions_approximately_unit_covariance():
    rng = np.random.default_rng(5)
    n = 100
    z = rng.normal(size=(n, 4))
    z[:, 1] = z[:, 0] * 0.8 + 0.2 * rng.normal(size=n)
    mask = np.ones_like(z, dtype=bool)
    dim_to_partition = np.asarray([0, 0, 0, 0], dtype=np.int64)

    result = zca_whiten_partitions(z, mask, dim_to_partition)
    out = result.embeddings
    cov = np.cov(out.T)
    assert np.allclose(cov, np.eye(4), atol=0.15)


def test_coverage_noise_scale_modes():
    mask = np.asarray(
        [
            [True, True, False, False],
            [True, False, True, False],
            [True, True, True, True],
        ]
    )
    dim_to_partition = np.asarray([0, 0, 1, 1], dtype=np.int64)

    h_part = coverage_noise_scale(mask, dim_to_partition, mode="partition_fraction_inverse")
    assert h_part.shape == (3, 2)
    assert np.isclose(h_part[0, 0], 1.0)
    assert h_part[0, 1] > 1.0

    h_global = coverage_noise_scale(mask, dim_to_partition, mode="global_partition_count")
    assert h_global.shape == (3, 2)
    assert np.all(h_global >= 1.0)


def test_compute_partition_weights():
    dim_to_partition = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    w_u = compute_partition_weights(dim_to_partition, scheme="uniform")
    w_i = compute_partition_weights(dim_to_partition, scheme="inverse_dims")
    assert np.allclose(w_u, np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(w_i, np.asarray([0.5, 1.0, 1.0 / 3.0]))
