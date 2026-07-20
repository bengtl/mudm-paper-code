"""Budget-matched sampling dispatch — turns a mesh into AT MOST `budget` 3D points.

Fix B / I2 (pre-registered methodology change): all methods return shape
(n, 3) float32 where n = min(budget, len(positions)) — NEVER padded with
duplicate/replacement points. A neuron whose mesh has fewer real vertices
than `budget` (e.g. a coarse LOD) is per-sample right-censored: it
contributes fewer, but all REAL, points rather than being topped up with
repeated ones. This means experiment arms differ only in sampling strategy
/ LOD source, not in whether duplicate geometry was manufactured to hit a
point count. Callers that batch multiple (variable-length) samples together
must pad to the batch max with a validity mask instead of assuming a fixed
per-sample point count — see sweep.py's collate in `_build_xy`.
"""

import numpy as np
from mudm_paper.fps import farthest_point_sample
from mudm_paper.qem import onthefly_qem


def sample_to_budget(
    positions: np.ndarray, indices: np.ndarray, budget: int, method: str, seed: int
) -> np.ndarray:
    """Sample a mesh to AT MOST `budget` 3D points via one of four methods.

    Args:
        positions: (n, 3) float32 array of vertex coordinates.
        indices: (m, 3) uint32 array of triangle indices.
        budget: Target (maximum) number of points to return.
        method: One of "random", "fps", "onthefly_qem", or "precomputed".
        seed: Random seed for reproducibility.

    Returns:
        (n, 3) float32 array of sampled points, n = min(budget, len(positions)).
        NEVER padded with duplicate/replacement points (Fix B / I2): a mesh
        with fewer vertices than `budget` contributes ALL of its (real)
        vertices, right-censored rather than topped up.

    Raises:
        ValueError or KeyError: If method is not recognized.
    """
    if method == "random":
        return _sample_random(positions, budget, seed)
    elif method == "fps":
        return _sample_fps(positions, budget, seed)
    elif method == "onthefly_qem":
        return _sample_onthefly_qem(positions, indices, budget, seed)
    elif method == "precomputed":
        # Precomputed is an alias of random (the distinction is the caller's input LOD)
        return _sample_random(positions, budget, seed)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def _sample_random(positions: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Uniform random subsample to budget, WITHOUT duplicate padding
    (Fix B / I2): a mesh with n <= budget vertices returns all n of them."""
    n = len(positions)
    if n <= budget:
        return positions.astype(np.float32)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n, size=budget, replace=False)
    return positions[indices].astype(np.float32)


def _sample_fps(positions: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Farthest point sampling, WITHOUT duplicate padding (Fix B / I2).

    `farthest_point_sample` already returns `arange(n)` when `budget >= n`,
    so the result is naturally right-censored to n real points -- no pad
    step needed.
    """
    idx = farthest_point_sample(positions, budget, seed)
    return positions[idx].astype(np.float32)


def _sample_onthefly_qem(
    positions: np.ndarray, indices: np.ndarray, budget: int, seed: int
) -> np.ndarray:
    """On-the-fly QEM decimation, then subsample WITHOUT duplicate padding
    (Fix B / I2).

    QEM lands on a discrete staircase of achievable vertex counts. If that
    count is already <= budget, return all of it (right-censored);
    otherwise subsample without replacement down to exactly `budget`.
    """
    pos_qem, _ = onthefly_qem(positions, indices, budget)
    n_qem = len(pos_qem)

    if n_qem <= budget:
        return pos_qem.astype(np.float32)
    rng = np.random.RandomState(seed)
    indices_subsample = rng.choice(n_qem, size=budget, replace=False)
    return pos_qem[indices_subsample].astype(np.float32)
