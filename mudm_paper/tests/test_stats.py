import numpy as np
from mudm_paper.stats import aggregate, paired_t, tost_equivalence

def test_aggregate_se():
    r = aggregate([0.9, 0.9, 0.9, 0.9])
    assert r["se"] == 0.0 and r["n"] == 4 and abs(r["mean"] - 0.9) < 1e-9

def test_tost_declares_equivalent_when_close():
    rng = np.random.RandomState(0)
    a = 0.90 + 0.01 * rng.randn(20); b = 0.895 + 0.01 * rng.randn(20)
    assert tost_equivalence(list(a), list(b), margin=0.03)["equivalent"] is True

def test_tost_rejects_when_far():
    rng = np.random.RandomState(0)
    a = 0.90 + 0.01 * rng.randn(20); b = 0.70 + 0.01 * rng.randn(20)
    assert tost_equivalence(list(a), list(b), margin=0.03)["equivalent"] is False

def test_paired_t_basic():
    a = [0.9, 0.91, 0.89, 0.90]; b = [0.5, 0.52, 0.48, 0.51]
    r = paired_t(a, b)
    assert r["diff"] > 0.3 and r["p"] < 0.05
