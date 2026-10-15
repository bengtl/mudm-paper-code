import numpy as np
import pytest

from mudm_paper.stats import aggregate, holm, paired_t, tost_equivalence

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


# --- Holm step-down adjustment ---------------------------------------------
# Reference values hand-computed and cross-checked against
# statsmodels.stats.multitest.multipletests(p, method="holm").


@pytest.mark.parametrize(
    "p, expected",
    [
        ([0.01, 0.04, 0.03, 0.005], [0.03, 0.06, 0.06, 0.02]),
        ([0.2, 0.6, 0.01, 0.9, 0.03], [0.6, 1.0, 0.05, 1.0, 0.12]),
        ([0.04, 0.04, 0.04], [0.12, 0.12, 0.12]),
        # five-level family shaped like the hemilineage fusion row
        ([0.013, 0.025, 0.028, 4.8e-4, 0.19], [0.052, 0.075, 0.075, 2.4e-3, 0.19]),
    ],
)
def test_holm_known_values(p, expected):
    assert holm(p) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_holm_monotone_and_capped():
    rng = np.random.default_rng(0)
    p = list(rng.uniform(0, 1, 50))
    adj = holm(p)
    order = np.argsort(p)
    a_sorted = np.asarray(adj)[order]
    assert np.all(np.diff(a_sorted) >= 0)  # monotone in raw-p order
    assert max(adj) <= 1.0
    assert all(a >= r for a, r in zip(adj, p))  # never below the raw p


def test_holm_single_and_empty():
    assert holm([0.3]) == [0.3]
    assert holm([]) == []


def test_holm_rejects_invalid():
    with pytest.raises(ValueError):
        holm([0.1, float("nan")])
    with pytest.raises(ValueError):
        holm([1.2])
