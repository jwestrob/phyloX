import numpy as np

from phylox.benchmark import run_scale_benchmark
from phylox.pipeline import InferenceConfig


def test_scale_benchmark_smoke():
    cfg = InferenceConfig(
        n_starts=1,
        bc_rounds=1,
        phase_a_nni_rounds=1,
        phase_c_nni_rounds=1,
        use_spr=False,
    )
    rows = run_scale_benchmark(
        taxa_sizes=[12],
        dims_per_partition=[4, 4],
        missing_prob_by_partition=[0.1, 0.2],
        confounder_dim=2,
        nuisance_strength=0.1,
        config=cfg,
        seed=5,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.n_taxa == 12
    assert row.d_total == 8
    assert np.isfinite(row.wall_time_sec)
    assert np.isfinite(row.log_likelihood)
    assert 0.0 <= row.normalized_rf_to_true <= 1.0
