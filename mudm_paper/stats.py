import numpy as np
from scipy import stats


def aggregate(per_seed: list[float]) -> dict:
    """Aggregate list of floats into mean, se, and n.

    Returns:
        dict with keys: "mean" (float), "se" (float), "n" (int)
        se = std(ddof=1)/sqrt(n), or 0.0 if all values equal or n<2.
    """
    arr = np.asarray(per_seed, dtype=float)
    n = len(arr)
    mean = float(arr.mean())

    if n < 2:
        se = 0.0
    else:
        std = arr.std(ddof=1)
        if std == 0.0:
            se = 0.0
        else:
            se = float(std / np.sqrt(n))

    return {"mean": mean, "se": se, "n": n}


def paired_t(a: list[float], b: list[float]) -> dict:
    """Paired t-test between two lists.

    Returns:
        dict with keys: "diff" (mean(a) - mean(b)), "t" (t-statistic), "p" (two-sided p-value)
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    diff = float(a_arr.mean() - b_arr.mean())

    return {"diff": diff, "t": float(t_stat), "p": float(p_val)}


def tost_equivalence(a: list[float], b: list[float], margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests (TOST) for equivalence.

    Tests whether a and b are statistically indistinguishable within margin.

    Args:
        a: First list of values
        b: Second list of values
        margin: Equivalence margin (absolute difference threshold)
        alpha: Significance level (default 0.05)

    Returns:
        dict with keys: "equivalent" (bool), "p_lower" (float), "p_upper" (float),
                        "mean_diff" (float)
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean_d = float(d.mean())
    se = d.std(ddof=1) / np.sqrt(n)

    if se == 0:  # degenerate: equivalent iff mean within margin
        eq = abs(mean_d) < margin
        return {
            "equivalent": bool(eq),
            "p_lower": 0.0 if eq else 1.0,
            "p_upper": 0.0 if eq else 1.0,
            "mean_diff": mean_d
        }

    df = n - 1
    t_lower = (mean_d - (-margin)) / se  # H0: mean_d <= -margin ; reject if mean_d >> -margin
    p_lower = stats.t.sf(t_lower, df)    # upper-tail
    t_upper = (mean_d - margin) / se     # H0: mean_d >= +margin ; reject if mean_d << +margin
    p_upper = stats.t.cdf(t_upper, df)   # lower-tail
    equivalent = (p_lower < alpha) and (p_upper < alpha)

    return {
        "equivalent": bool(equivalent),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "mean_diff": mean_d
    }
