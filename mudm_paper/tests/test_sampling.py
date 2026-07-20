import numpy as np
from mudm_paper.sampling import sample_to_budget
from mudm_paper.tests.fixtures import make_sphere_mesh

def test_all_methods_return_budget():
    # onthefly_qem is exercised separately (test_onthefly_qem_never_exceeds_budget):
    # its edge-collapse decimation lands on a discrete staircase of achievable
    # vertex counts (qem.py) and can undershoot budget even when the raw mesh
    # has far more vertices than budget -- Fix B / I2 forbids padding that
    # shortfall with duplicates, so it does not share this exact-budget guarantee.
    pos, idx = make_sphere_mesh(3)
    for m in ("random", "fps", "precomputed"):
        out = sample_to_budget(pos, idx, budget=128, method=m, seed=0)
        assert out.shape == (128, 3) and out.dtype == np.float32

def test_onthefly_qem_never_exceeds_budget():
    """QEM's edge-collapse decimation lands on a discrete staircase of
    achievable vertex counts (see qem.py) -- it can UNDERSHOOT a budget even
    when the raw input mesh has far more vertices than budget (642 verts here,
    budget 128; empirically lands on a ~111-vertex rung). Fix B / I2 forbids
    padding that shortfall with duplicates, so the output must be <= budget,
    never more, and never duplicate-padded up to exactly budget."""
    pos, idx = make_sphere_mesh(3)                  # 642 verts >> budget
    out = sample_to_budget(pos, idx, budget=128, method="onthefly_qem", seed=0)
    assert out.dtype == np.float32
    assert out.shape[1] == 3
    assert 0 < out.shape[0] <= 128

def test_fps_more_spread_than_random():
    pos, idx = make_sphere_mesh(4)
    def spread(x): return float(np.mean(np.var(x, axis=0)))
    s_fps = spread(sample_to_budget(pos, idx, 64, "fps", 0))
    s_rnd = spread(sample_to_budget(pos, idx, 64, "random", 0))
    assert s_fps >= 0.5 * s_rnd

def test_budget_exceeds_vertices_returns_all():
    pos, idx = make_sphere_mesh(1)                 # 42 verts
    n_available = len(pos)
    pos_rows = [tuple(row) for row in pos.tolist()]
    for m in ("random", "fps", "onthefly_qem", "precomputed"):
        out = sample_to_budget(pos, idx, budget=200, method=m, seed=1)
        # Fix B / I2: right-censored -- ALL real points, no duplicate padding
        assert out.shape == (n_available, 3)
        assert out.shape[0] < 200
        assert out.dtype == np.float32
        out_rows = [tuple(row) for row in out.tolist()]
        assert len(set(out_rows)) == len(out_rows)          # no duplicate rows introduced
        assert all(row in pos_rows for row in out_rows)     # every row is a real input vertex

def test_unknown_method_raises():
    pos, idx = make_sphere_mesh(1)
    import pytest
    with pytest.raises((ValueError, KeyError)):
        sample_to_budget(pos, idx, 10, "nope", 0)
