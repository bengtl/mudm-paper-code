"""On-the-fly QEM decimation — a fair fine-mesh baseline.

Wraps muDM's real Rust QEM simplifier (`mudm_tools._rs.decimate_tin`) so it
can be driven by a target *vertex* count instead of its native face-keep
ratio. Used to produce a fair "decimate-on-demand" comparison point against
FPS sampling and precomputed LODs — same underlying QEM engine muDM uses for
its tiling pipeline, just invoked at query time instead of build time.

`decimate_tin`'s I/O contract (confirmed empirically, see
paper/harness/tests/test_qem.py and the Task 2 report):
  - Input/output "coordinates": a list of faces.
  - Each face is a 1-element list containing one ring: `[ring]`.
  - Each ring is 4 **coordinate** points (not indices): `[p0, p1, p2, p0]`,
    the last point closing the triangle (duplicate of the first).
  - `target_ratio` is the fraction of *faces* (not vertices) to keep;
    `>= 1.0` (or <= 4 input faces) passes the input through unchanged.
  - The QEM internals collapse edges in coarse per-iteration passes, so the
    achievable output vertex count for a given ratio is a staircase, not a
    continuous function — bisection lands on the nearest achievable rung.
"""

import numpy as np
from mudm_tools._rs import decimate_tin

_MAX_BISECT_STEPS = 30


def _to_rings(positions: np.ndarray, indices: np.ndarray) -> list:
    """Convert (positions, indices) into the TIN coordinate rings
    `decimate_tin` expects: a list of faces, each `[[p0, p1, p2, p0]]`.
    """
    rings = []
    for tri in indices:
        p0 = positions[int(tri[0])].tolist()
        p1 = positions[int(tri[1])].tolist()
        p2 = positions[int(tri[2])].tolist()
        rings.append([[p0, p1, p2, p0]])
    return rings


def _rings_to_arrays(rings: list) -> tuple[np.ndarray, np.ndarray]:
    """Parse TIN coordinate rings back into deduplicated (positions, indices)
    arrays. Vertices are deduplicated by exact coordinate match — the ring
    format carries geometry, not indices, so shared vertices across faces
    appear as bit-identical repeated float tuples.
    """
    vert_map: dict[tuple, int] = {}
    positions: list[list[float]] = []
    indices: list[list[int]] = []
    for face in rings:
        ring = face[0]
        tri = []
        for p in ring[:3]:  # skip the closing duplicate of p0
            key = tuple(p)
            vid = vert_map.get(key)
            if vid is None:
                vid = len(positions)
                vert_map[key] = vid
                positions.append(p)
            tri.append(vid)
        indices.append(tri)
    pos_arr = np.asarray(positions, dtype=np.float32)
    idx_arr = np.asarray(indices, dtype=np.uint32)
    return pos_arr, idx_arr


def onthefly_qem(
    positions: np.ndarray, indices: np.ndarray, target_verts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate a mesh to ~target_verts vertices using muDM's real QEM.

    Passthrough (returns the inputs unchanged) when
    ``target_verts >= len(positions)``. Otherwise bisects on
    `decimate_tin`'s face-keep ratio until the resulting vertex count lands
    within ~10% of `target_verts`, returning the closest candidate seen if
    the QEM's coarse collapse granularity never lands inside that band.
    """
    n_verts = len(positions)
    if target_verts >= n_verts:
        return positions, indices

    rings = _to_rings(positions, indices)
    tol = max(1, round(0.10 * target_verts))

    lo, hi = 0.0, 1.0
    best_err = None
    best_pos = None
    best_idx = None
    for _ in range(_MAX_BISECT_STEPS):
        mid = (lo + hi) / 2.0
        out_rings = decimate_tin(rings, mid)
        p2, i2 = _rings_to_arrays(out_rings)
        v = len(p2)
        err = abs(v - target_verts)
        if best_err is None or err < best_err:
            best_err, best_pos, best_idx = err, p2, i2
        if err <= tol:
            break
        if v < target_verts:
            lo = mid  # need more vertices -> keep more faces -> raise ratio
        else:
            hi = mid  # need fewer vertices -> keep fewer faces -> lower ratio

    return best_pos, best_idx
