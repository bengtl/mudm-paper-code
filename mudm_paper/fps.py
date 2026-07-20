import numpy as np

def farthest_point_sample(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = len(points)
    if k >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = rng.randint(n)
    dist = np.full(n, np.inf)
    for i in range(1, k):
        d = np.sum((points - points[sel[i - 1]]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        sel[i] = int(np.argmax(dist))
    return sel
