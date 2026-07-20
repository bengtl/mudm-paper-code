import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def make_sphere_mesh(n_subdiv: int):
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = [np.array(v, float) for v in
             [[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
              [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]]]
    verts = [v/np.linalg.norm(v) for v in verts]
    faces = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
             [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
             [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    for _ in range(n_subdiv):
        mid, new = {}, []
        def midpoint(a, b):
            key = (min(a,b), max(a,b))
            if key not in mid:
                m = verts[a] + verts[b]; m = m/np.linalg.norm(m)
                verts.append(m); mid[key] = len(verts)-1
            return mid[key]
        for a,b,c in faces:
            ab,bc,ca = midpoint(a,b),midpoint(b,c),midpoint(c,a)
            new += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        faces = new
    return np.asarray(verts, np.float32), np.asarray(faces, np.uint32)


def write_obj(path: str, positions: np.ndarray, indices: np.ndarray) -> None:
    """Write a mesh as a Wavefront OBJ file: `v x y z` vertex lines followed by
    `f i j k` face lines with 1-BASED indices (the OBJ convention, and how
    `download_hemibrain.py`'s `_dl_one` writes real neuron meshes). Used to
    feed tiny synthetic meshes to the REAL muDM Rust tiler in tests.
    """
    with open(path, "w") as f:
        np.savetxt(f, positions, fmt="v %.7g %.7g %.7g")
        np.savetxt(f, indices.astype(np.int64) + 1, fmt="f %d %d %d")


# Schema mirrors the real muDM tiling pipeline's parquet output. `zoom` is
# NOT a column here -- it is the Hive partition directory (`zoom=Z/`), added
# back automatically by pyarrow.dataset when reading with a HivePartitioning.
_ROW_SCHEMA = pa.schema([
    ("tile_x", pa.uint16()),
    ("tile_y", pa.uint16()),
    ("tile_d", pa.uint16()),
    ("feature_id", pa.uint32()),
    ("geom_type", pa.uint8()),
    ("positions", pa.large_binary()),  # raw little-endian f32 [x,y,z, x,y,z, ...]
    ("indices", pa.large_binary()),    # raw little-endian u32 triangle indices, tile-local
    ("tags", pa.map_(pa.string(), pa.string())),
])

_GEOM_TYPE_MESH = 3


def _pack_positions(pos: np.ndarray) -> bytes:
    return np.ascontiguousarray(pos, dtype="<f4").tobytes()


def _pack_indices(idx: np.ndarray) -> bytes:
    return np.ascontiguousarray(idx, dtype="<u4").tobytes()


def _subset_mesh(pos: np.ndarray, faces: np.ndarray) -> tuple:
    """Restrict a mesh to the vertices referenced by `faces`, remapping
    indices to a compact tile-local space (0..k-1). Used to carve a full
    sphere mesh into independent per-tile fragments, the way the real
    tiler splits one neuron's geometry across several tiles.
    """
    used = np.unique(faces)
    lut = np.full(int(pos.shape[0]), -1, dtype=np.int64)
    lut[used] = np.arange(len(used))
    local_pos = pos[used].astype(np.float32)
    local_idx = lut[faces].astype(np.uint32)
    return local_pos, local_idx


def _tags_for(
    body_id: str,
    i: int,
    omit_field_for: set | None = None,
    placeholder_ids: set | None = None,
    omit_instance_for: set | None = None,
) -> list:
    """Tag list for one neuron. `omit_field_for` and `placeholder_ids` are
    optional sets of body_ids used to simulate two real-muDM sparsity
    patterns, both applied to the `"subclass"` field only:

    - `omit_field_for`: the `"subclass"` tag is dropped entirely (real
      tags are sparse -- many neurons lack a given field).
    - `placeholder_ids`: the `"subclass"` value is replaced with the
      placeholder bucket `"SNxxxx"` (a provisional/unassigned label, spec
      Sec 3.2). Ignored for a body_id also in `omit_field_for` (omission
      wins).

    `"instance"` is a bilateral-homolog tag independent of the above two:
    consecutive neurons (2k, 2k+1) form pair `k`, tagged `f"type{k}_L"` /
    `f"type{k}_R"` -- mirroring real muDM's `instance` field, which
    `sweep.py`'s `group_field` support strips the trailing `_[LR]` off of
    to recover the shared homolog group key. `omit_instance_for` is a
    third, independent sparsity knob (additive -- does not affect
    `omit_field_for`/`placeholder_ids`, which only ever touch
    `"subclass"`): the optional set of body_ids whose `"instance"` tag is
    dropped entirely, simulating a neuron missing group_field coverage
    (e.g. many real sensory neurons lack an `instance` assignment).
    """
    pair, side = i // 2, ("L" if i % 2 == 0 else "R")
    tags = [
        ("body_id", body_id),
        ("neuron_class", "t1" if i % 2 == 0 else "t2"),
        ("subclass", f"s{i % 3}"),
        ("soma_side", side),
        ("instance", f"type{pair}_{side}"),
    ]
    if omit_field_for and body_id in omit_field_for:
        tags = [(k, v) for k, v in tags if k != "subclass"]
    elif placeholder_ids and body_id in placeholder_ids:
        tags = [(k, "SNxxxx" if k == "subclass" else v) for k, v in tags]
    if omit_instance_for and body_id in omit_instance_for:
        tags = [(k, v) for k, v in tags if k != "instance"]
    return tags


def _row(tile_x: int, feature_id: int, pos: np.ndarray, idx: np.ndarray, tags: list) -> dict:
    return {
        "tile_x": tile_x,
        "tile_y": 0,
        "tile_d": 0,
        "feature_id": feature_id,
        "geom_type": _GEOM_TYPE_MESH,
        "positions": _pack_positions(pos),
        "indices": _pack_indices(idx),
        "tags": tags,
    }


def write_synthetic_pyramid(
    path: str,
    n_neurons: int = 6,
    zooms: tuple = (0, 1, 2),
    omit_field_for: set | None = None,
    placeholder_ids: set | None = None,
    omit_instance_for: set | None = None,
) -> None:
    """Write a synthetic parquet pyramid at `path`, one `zoom=Z/part_000.parquet`
    per zoom level, matching the real muDM tiling pipeline's row schema.

    Each neuron body_id = f"N{i}" gets a sphere mesh written as a single
    row (tile) per zoom -- except body_id "N0" at the coarsest zoom, which
    is split into two rows (two tiles), each carrying only the vertices its
    half of the faces references with tile-local indices. This exercises
    the loader's multi-tile concatenation + index-offset path.

    `omit_field_for` / `placeholder_ids`: optional sets of body_ids (e.g.
    `{"N0", "N1"}`) forwarded to `_tags_for` to simulate sparse tags
    (field dropped) / placeholder labels (`"subclass"` -> `"SNxxxx"`)
    respectively. `omit_instance_for` is the analogous, independent knob
    for the bilateral-homolog `"instance"` tag (see `_tags_for`). All
    three default to `None`, which omits/overrides nothing -- identical
    tags to the prior fixture behavior, so existing callers and tests are
    unaffected.
    """
    os.makedirs(path, exist_ok=True)
    coarsest = min(zooms)

    for z in zooms:
        rows = []
        for i in range(n_neurons):
            body_id = f"N{i}"
            tags = _tags_for(
                body_id, i,
                omit_field_for=omit_field_for,
                placeholder_ids=placeholder_ids,
                omit_instance_for=omit_instance_for,
            )
            n_subdiv = 1 if z == coarsest else 2  # vary mesh detail slightly per zoom
            pos, faces = make_sphere_mesh(n_subdiv)

            if body_id == "N0" and z == coarsest:
                half = len(faces) // 2
                for tile_x, face_chunk in enumerate((faces[:half], faces[half:])):
                    local_pos, local_idx = _subset_mesh(pos, face_chunk)
                    rows.append(_row(tile_x, i, local_pos, local_idx, tags))
            else:
                rows.append(_row(0, i, pos, faces, tags))

        table = pa.Table.from_pylist(rows, schema=_ROW_SCHEMA)
        zdir = os.path.join(path, f"zoom={z}")
        os.makedirs(zdir, exist_ok=True)
        pq.write_table(table, os.path.join(zdir, "part_000.parquet"))
