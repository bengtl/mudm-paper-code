import numpy as np
from mudm_paper.fps import farthest_point_sample

def test_fps_count_and_determinism():
    pts = np.random.RandomState(0).randn(200, 3).astype(np.float32)
    a = farthest_point_sample(pts, 32, seed=1)
    b = farthest_point_sample(pts, 32, seed=1)
    assert a.shape == (32,) and np.array_equal(a, b)
    assert len(set(a.tolist())) == 32

def test_fps_spreads_to_extremes():
    pts = np.array([[0,0,0],[10,0,0],[0,10,0],[0,0,10]] + [[0.1*i,0,0] for i in range(50)], np.float32)
    idx = set(farthest_point_sample(pts, 4, seed=0).tolist())
    assert {0,1,2,3}.issubset(idx) or len(idx & {1,2,3}) >= 2

def test_fps_k_ge_n():
    pts = np.zeros((5,3), np.float32)
    assert np.array_equal(farthest_point_sample(pts, 10, seed=0), np.arange(5))
