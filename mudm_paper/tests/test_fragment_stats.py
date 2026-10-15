"""Tests for `mudm_paper.fragment_stats` (per-feature vertex/face/area
statistics of a muDM mesh pyramid, fragment-duplication accounting, and the
finest-level fidelity comparison against source OBJ meshes).

All reference numbers are computed independently in the test (set-based
uniqueness on float32 tuples, Heron's formula for triangle areas) so that an
indexing or offset bug in the module cannot cancel out against itself.
"""

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mudm_paper import fragment_stats as fs
from mudm_paper.tests.fixtures import (
    _ROW_SCHEMA,
    _row,
    _subset_mesh,
    make_sphere_mesh,
    write_obj,
    write_synthetic_pyramid,
)


# ----------------------------------------------------------------------------
# Independent reference helpers
# ----------------------------------------------------------------------------

def _heron_area(pos: np.ndarray, idx: np.ndarray) -> float:
    """Total triangle area via Heron's formula (deliberately NOT the cross
    product the module uses)."""
    p = np.asarray(pos, dtype=np.float64)
    total = 0.0
    for i, j, k in np.asarray(idx, dtype=np.int64):
        a = np.linalg.norm(p[j] - p[i])
        b = np.linalg.norm(p[k] - p[j])
        c = np.linalg.norm(p[i] - p[k])
        s = 0.5 * (a + b + c)
        total += float(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0)))
    return total


def _vertex_set(pos: np.ndarray) -> set:
    return {tuple(v) for v in np.asarray(pos, dtype=np.float32).tolist()}


def _write_rows(root: str, zoom: int, rows: list, name: str = "part_000.parquet",
                row_group_size: int | None = None) -> None:
    zdir = os.path.join(root, f"zoom={zoom}")
    os.makedirs(zdir, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_ROW_SCHEMA)
    kwargs = {} if row_group_size is None else {"row_group_size": row_group_size}
    pq.write_table(table, os.path.join(zdir, name), **kwargs)


def _tags(body_id: str) -> list:
    return [("body_id", body_id), ("neuron_class", "t1")]


# ----------------------------------------------------------------------------
# (a) single-row feature: no duplication
# ----------------------------------------------------------------------------

def test_single_row_feature_has_no_duplicates(tmp_path):
    root = str(tmp_path / "pyr")
    write_synthetic_pyramid(root, n_neurons=3, zooms=(0, 1))
    stats = fs.per_feature_zoom_stats(root, [1])
    assert set(stats) == {1}
    z1 = stats[1]
    assert set(z1) == {0, 1, 2}
    pos, faces = make_sphere_mesh(2)  # zoom 1 is the finer level in the fixture
    for fid, st in z1.items():
        assert st["n_rows"] == 1
        assert st["n_verts"] == len(pos) == 162
        assert st["n_unique_verts"] == st["n_verts"]
        assert st["n_tris"] == len(faces) == 320
        assert st["n_unique_tris"] == st["n_tris"]
        assert st["body_id"] == f"N{fid}"


def test_fixture_split_feature_dedups_to_source_vertex_count(tmp_path):
    """The fixture splits N0 into two tiles at the coarsest zoom; the
    concatenated row count exceeds the sphere's 42 vertices while the unique
    count recovers exactly 42, and area/tris equal the whole sphere."""
    root = str(tmp_path / "pyr")
    write_synthetic_pyramid(root, n_neurons=2, zooms=(0, 1))
    st = fs.per_feature_zoom_stats(root, [0])[0][0]
    pos, faces = make_sphere_mesh(1)
    assert st["n_rows"] == 2
    assert st["n_verts"] > 42
    assert st["n_unique_verts"] == 42
    assert st["n_tris"] == 80 and st["n_unique_tris"] == 80
    assert st["area"] == pytest.approx(_heron_area(pos, faces), rel=1e-9)
    assert st["area_unique"] == pytest.approx(_heron_area(pos, faces), rel=1e-9)


# ----------------------------------------------------------------------------
# (b) two rows sharing k identical vertices (and one duplicated face)
# ----------------------------------------------------------------------------

def _split_sphere_rows(fid: int, body_id: str, dup_face: bool):
    pos, faces = make_sphere_mesh(1)
    fa = faces[:40]
    fb = np.vstack([faces[40:], faces[:1]]) if dup_face else faces[40:]
    pa_, ia = _subset_mesh(pos, fa)
    pb_, ib = _subset_mesh(pos, fb)
    rows = [_row(0, fid, pa_, ia, _tags(body_id)), _row(1, fid, pb_, ib, _tags(body_id))]
    k = len(_vertex_set(pa_) & _vertex_set(pb_))
    assert k > 0
    return rows, k, (pos, faces, fa, fb, pa_, ia, pb_, ib)


def test_shared_vertices_across_rows_are_counted_once(tmp_path):
    root = str(tmp_path / "pyr")
    rows, k, _ = _split_sphere_rows(5, "B5", dup_face=False)
    _write_rows(root, 0, rows)
    st = fs.per_feature_zoom_stats(root, [0])[0][5]
    n_verts = sum(len(np.frombuffer(r["positions"], "<f4")) // 3 for r in rows)
    assert st["n_verts"] == n_verts
    assert st["n_unique_verts"] == n_verts - k
    assert st["n_unique_verts"] == 42
    assert st["n_tris"] == 80 and st["n_unique_tris"] == 80


def test_duplicated_face_is_counted_once_in_unique_tris_and_area(tmp_path):
    root = str(tmp_path / "pyr")
    rows, k, (pos, faces, fa, fb, pa_, ia, pb_, ib) = _split_sphere_rows(9, "B9", dup_face=True)
    _write_rows(root, 0, rows)
    st = fs.per_feature_zoom_stats(root, [0])[0][9]
    assert st["n_tris"] == 81
    assert st["n_unique_tris"] == 80
    assert st["n_unique_verts"] == 42
    # Raw area double-counts the duplicated face; the dedup'd area does not.
    assert st["area"] == pytest.approx(_heron_area(pa_, ia) + _heron_area(pb_, ib), rel=1e-9)
    assert st["area_unique"] == pytest.approx(_heron_area(pos, faces), rel=1e-9)
    assert st["area"] > st["area_unique"]


def test_face_dedup_can_be_disabled(tmp_path):
    root = str(tmp_path / "pyr")
    rows, _, _ = _split_sphere_rows(9, "B9", dup_face=True)
    _write_rows(root, 0, rows)
    st = fs.per_feature_zoom_stats(root, [0], dedup_faces=False)[0][9]
    assert "n_unique_tris" not in st and "area_unique" not in st
    assert st["n_unique_verts"] == 42 and st["n_tris"] == 81


# ----------------------------------------------------------------------------
# (c) surface area of a single sphere row
# ----------------------------------------------------------------------------

def test_single_row_area_matches_heron(tmp_path):
    root = str(tmp_path / "pyr")
    pos, faces = make_sphere_mesh(2)
    _write_rows(root, 3, [_row(0, 1, pos, faces, _tags("B1"))])
    st = fs.per_feature_zoom_stats(root, [3])[3][1]
    ref = _heron_area(pos, faces)
    assert st["area"] == pytest.approx(ref, rel=1e-9)
    assert st["area_unique"] == pytest.approx(ref, rel=1e-9)
    # Sanity: a subdivided icosphere approaches 4*pi from below.
    assert 0.95 * 4 * np.pi < ref < 4 * np.pi


# ----------------------------------------------------------------------------
# Streaming across part files / row groups / batches; feature filter; tags
# ----------------------------------------------------------------------------

def test_feature_spread_over_two_part_files_and_tiny_batches(tmp_path):
    """Rows of one feature living in `part_000.parquet` and
    `part_001.mu.parquet` (both real naming conventions), read with
    row_group_size=1 and batch_rows=1 (forces sliced Arrow arrays with a
    non-zero offset), give the same numbers as a single-file read."""
    rows, k, _ = _split_sphere_rows(3, "B3", dup_face=True)
    extra = [_row(0, 4, *make_sphere_mesh(1), _tags("B4"))]

    one = str(tmp_path / "one")
    _write_rows(one, 0, rows + extra)
    two = str(tmp_path / "two")
    _write_rows(two, 0, [rows[0], extra[0]], name="part_000.parquet", row_group_size=1)
    _write_rows(two, 0, [rows[1]], name="part_001.mu.parquet", row_group_size=1)

    ref = fs.per_feature_zoom_stats(one, [0])[0]
    got = fs.per_feature_zoom_stats(two, [0], batch_rows=1)[0]
    assert set(ref) == set(got) == {3, 4}
    for fid in ref:
        for key in ("n_rows", "n_verts", "n_unique_verts", "n_tris", "n_unique_tris", "body_id"):
            assert ref[fid][key] == got[fid][key], (fid, key)
        assert ref[fid]["area"] == pytest.approx(got[fid]["area"], rel=1e-12)
        assert ref[fid]["area_unique"] == pytest.approx(got[fid]["area_unique"], rel=1e-12)
    assert got[3]["n_unique_verts"] == 42 and got[3]["n_unique_tris"] == 80


def test_compaction_path_merges_pending_hashes(tmp_path, monkeypatch):
    """Force the incremental compaction threshold to 1 so every row triggers a
    merge; results must be identical to the no-compaction path."""
    rows, _, _ = _split_sphere_rows(3, "B3", dup_face=True)
    root = str(tmp_path / "pyr")
    _write_rows(root, 0, rows)
    ref = fs.per_feature_zoom_stats(root, [0])[0][3]
    monkeypatch.setattr(fs, "_PENDING_LIMIT", 1)
    got = fs.per_feature_zoom_stats(root, [0])[0][3]
    assert got == ref


def test_feature_filter_and_select_feature_ids(tmp_path):
    root = str(tmp_path / "pyr")
    write_synthetic_pyramid(root, n_neurons=4, zooms=(0, 1))
    assert fs.select_feature_ids(root, 1, 2) == [0, 1]
    assert fs.select_feature_ids(root, 1, 100) == [0, 1, 2, 3]
    stats = fs.per_feature_zoom_stats(root, [0, 1], feature_ids=[1, 3])
    assert set(stats[0]) == {1, 3} and set(stats[1]) == {1, 3}


def test_body_id_falls_back_to_feature_id_without_tag(tmp_path):
    root = str(tmp_path / "pyr")
    pos, faces = make_sphere_mesh(1)
    _write_rows(root, 0, [_row(0, 77, pos, faces, [("neuron_class", "t1")])])
    st = fs.per_feature_zoom_stats(root, [0])[0][77]
    assert st["body_id"] == "77"


def test_missing_zoom_directory_raises(tmp_path):
    root = str(tmp_path / "pyr")
    write_synthetic_pyramid(root, n_neurons=1, zooms=(0,))
    with pytest.raises(FileNotFoundError):
        fs.per_feature_zoom_stats(root, [0, 5])


# ----------------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------------

def test_vertex_hash_is_exact_on_bit_patterns():
    pos, _ = make_sphere_mesh(3)  # 642 distinct vertices
    h = fs.hash_vertices(pos)
    assert h.dtype == np.uint64 and h.shape == (len(pos),)
    assert len(np.unique(h)) == len(pos)
    # Identical bit patterns hash identically, including across a copy.
    assert np.array_equal(fs.hash_vertices(pos.copy()), h)
    # +0.0 and -0.0 are different float32 bit patterns and stay distinct
    # (uniqueness is bit-exact, as documented).
    a = fs.hash_vertices(np.array([[0.0, 1.0, 2.0]], np.float32))
    b = fs.hash_vertices(np.array([[-0.0, 1.0, 2.0]], np.float32))
    assert a[0] != b[0]


def test_face_hash_is_winding_independent():
    pos, faces = make_sphere_mesh(1)
    vh = fs.hash_vertices(pos)
    tri = faces.astype(np.int64)
    h0 = fs.hash_faces(vh[tri])
    h1 = fs.hash_faces(vh[tri[:, [1, 2, 0]]])  # rotated
    h2 = fs.hash_faces(vh[tri[:, [0, 2, 1]]])  # reversed winding
    assert np.array_equal(h0, h1) and np.array_equal(h0, h2)
    assert len(np.unique(h0)) == len(faces)


# ----------------------------------------------------------------------------
# OBJ readers
# ----------------------------------------------------------------------------

def test_numpy_obj_parser_handles_slash_forms_quads_and_comments(tmp_path):
    p = tmp_path / "m.obj"
    p.write_text(
        "# comment\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "vt 0 0\n"
        "f 1/1/1 2/2/1 3/3/1\n"
        "f 1//1 3//1 4//1\n"
        "f 1 2 3 4\n"  # quad -> 2 triangles by fan
        "\n"
    )
    verts, faces = fs.parse_obj_numpy(str(p))
    assert verts.shape == (4, 3) and verts.dtype == np.float64
    assert faces.shape == (4, 3) and faces.dtype == np.uint32
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3], [0, 1, 2], [0, 2, 3]]


def test_numpy_obj_parser_fast_path_matches_write_obj(tmp_path):
    pos, faces = make_sphere_mesh(2)
    p = tmp_path / "s.obj"
    write_obj(str(p), pos, faces)
    verts, idx = fs.parse_obj_numpy(str(p))
    assert idx.tolist() == faces.tolist()
    assert np.allclose(verts, pos, rtol=1e-6, atol=0)


@pytest.mark.parametrize("reader", ["numpy", "auto"])
def test_read_obj_stats(tmp_path, reader):
    pos, faces = make_sphere_mesh(2)
    p = tmp_path / "s.obj"
    write_obj(str(p), pos, faces)
    n_verts, n_tris, area = fs.obj_mesh_stats(str(p), reader=reader)
    assert (n_verts, n_tris) == (162, 320)
    assert area == pytest.approx(_heron_area(pos, faces), rel=1e-6)


# ----------------------------------------------------------------------------
# (d) compare_to_obj
# ----------------------------------------------------------------------------

def _pyramid_with_objs(tmp_path, n=3):
    root = str(tmp_path / "pyr")
    objd = str(tmp_path / "obj")
    os.makedirs(objd)
    rows = []
    for i in range(n):
        pos, faces = make_sphere_mesh(1 + (i % 2))
        rows.append(_row(0, 10 + i, pos, faces, _tags(f"B{i}")))
        write_obj(os.path.join(objd, f"B{i}.obj"), pos, faces)
    _write_rows(root, 2, rows)
    return root, objd


@pytest.mark.parametrize("workers", [1, 2])
def test_compare_to_obj_reports_unit_ratios(tmp_path, workers):
    root, objd = _pyramid_with_objs(tmp_path, n=3)
    finest = fs.per_feature_zoom_stats(root, [2])[2]
    cmp = fs.compare_to_obj(finest, objd, workers=workers, obj_reader="numpy")
    assert cmp["n_compared"] == 3 and cmp["n_missing_obj"] == 0
    # write_obj prints %.7g, so the OBJ round trip is not bit-exact: area
    # ratios sit within ~1e-6 of 1; counts are exact.
    for key in ("median", "p5", "p95", "min", "max"):
        assert cmp["area_ratio"][key] == pytest.approx(1.0, abs=1e-5)
        assert cmp["area_unique_ratio"][key] == pytest.approx(1.0, abs=1e-5)
        assert cmp["tris_ratio"][key] == 1.0
        assert cmp["unique_tris_ratio"][key] == 1.0
        assert cmp["unique_verts_ratio"][key] == 1.0
    assert cmp["n_area_ratio_gt_tol"] == 0 and cmp["tol"] == 1e-3
    assert len(cmp["worst"]) == 3 and {w["body_id"] for w in cmp["worst"]} == {"B0", "B1", "B2"}


def test_compare_to_obj_flags_duplication_and_missing_files(tmp_path):
    root = str(tmp_path / "pyr")
    objd = str(tmp_path / "obj")
    os.makedirs(objd)
    rows, _, (pos, faces, *_rest) = _split_sphere_rows(1, "B1", dup_face=True)
    rows.append(_row(0, 2, pos, faces, _tags("B2")))  # no OBJ for B2
    _write_rows(root, 0, rows)
    write_obj(os.path.join(objd, "B1.obj"), pos, faces)
    finest = fs.per_feature_zoom_stats(root, [0])[0]
    cmp = fs.compare_to_obj(finest, objd, workers=1, obj_reader="numpy")
    assert cmp["n_compared"] == 1 and cmp["n_missing_obj"] == 1
    assert cmp["tris_ratio"]["median"] == pytest.approx(81 / 80)
    assert cmp["unique_tris_ratio"]["median"] == 1.0
    assert cmp["unique_verts_ratio"]["median"] == 1.0
    assert cmp["area_ratio"]["median"] > 1.0 + 1e-3
    assert cmp["n_area_ratio_gt_tol"] == 1
    assert cmp["area_unique_ratio"]["median"] == pytest.approx(1.0, abs=1e-5)
    assert cmp["n_area_unique_ratio_gt_tol"] == 0


def test_compare_to_obj_with_nothing_to_compare(tmp_path):
    objd = str(tmp_path / "obj")
    os.makedirs(objd)
    cmp = fs.compare_to_obj({5: {"n_verts": 1, "n_unique_verts": 1, "n_tris": 0, "area": 0.0,
                                  "n_rows": 1, "body_id": "nope"}}, objd, workers=1)
    assert cmp["n_compared"] == 0 and cmp["n_missing_obj"] == 1
    assert cmp["area_ratio"]["median"] is None
    json.dumps(cmp, allow_nan=False)


# ----------------------------------------------------------------------------
# (e) summarize
# ----------------------------------------------------------------------------

def test_summarize_on_hand_built_stats():
    stats = {
        4: {
            1: {"n_rows": 1, "n_verts": 100, "n_unique_verts": 100, "n_tris": 200, "area": 1.0},
            2: {"n_rows": 2, "n_verts": 600, "n_unique_verts": 300, "n_tris": 1200, "area": 2.0},
            3: {"n_rows": 1, "n_verts": 3000, "n_unique_verts": 3000, "n_tris": 6000, "area": 3.0},
            4: {"n_rows": 1, "n_verts": 5000, "n_unique_verts": 5000, "n_tris": 10000, "area": 4.0},
        }
    }
    s = fs.summarize(stats)[4]
    nv = np.array([100, 600, 3000, 5000], float)
    assert s["n_features"] == 4
    assert s["verts"]["mean"] == pytest.approx(nv.mean())
    assert s["verts"]["median"] == pytest.approx(np.median(nv))
    assert s["verts"]["p10"] == pytest.approx(np.percentile(nv, 10))
    assert s["verts"]["p90"] == pytest.approx(np.percentile(nv, 90))
    assert s["verts"]["min"] == 100 and s["verts"]["max"] == 5000
    assert s["frac_features_below_512"] == pytest.approx(0.25)
    assert s["frac_features_below_2048"] == pytest.approx(0.5)
    assert s["frac_features_below_4096"] == pytest.approx(0.75)
    assert s["frac_features_unique_below_512"] == pytest.approx(0.5)
    assert s["dup_fraction_mean"] == pytest.approx((0 + 0.5 + 0 + 0) / 4)
    assert s["dup_fraction_weighted"] == pytest.approx(1 - 8400 / 8700)
    assert s["rows_per_feature_mean"] == pytest.approx(1.25)
    assert s["n_rows"] == 5
    assert s["n_verts"] == 8700 and s["n_unique_verts"] == 8400 and s["n_tris"] == 17400
    assert s["area"] == pytest.approx(10.0)
    # No face-dedup fields were supplied, so the corresponding outputs are None.
    assert s["n_unique_tris"] is None and s["area_unique"] is None
    json.dumps(s, allow_nan=False)


def test_summarize_empty_zoom_is_json_safe():
    s = fs.summarize({0: {}})[0]
    assert s["n_features"] == 0 and s["verts"]["median"] is None
    assert s["dup_fraction_weighted"] is None
    json.dumps(s, allow_nan=False)


# ----------------------------------------------------------------------------
# (f) CLI end to end
# ----------------------------------------------------------------------------

def test_cli_writes_validator_style_json(tmp_path):
    root, objd = _pyramid_with_objs(tmp_path, n=2)
    # Add a coarser zoom so --zooms exercises more than one level.
    pos, faces = make_sphere_mesh(1)
    _write_rows(root, 0, [_row(0, 10, pos, faces, _tags("B0")), _row(0, 11, pos, faces, _tags("B1"))])
    out = tmp_path / "manc_fragment_stats.json"
    fs.main([
        "--parquet", root,
        "--zooms", "0,2",
        "--out", str(out),
        "--obj-dir", objd,
        "--workers", "1",
        "--obj-reader", "numpy",
    ])
    d = json.loads(out.read_text())
    assert d["benchmark_id"] == "manc_fragment_stats"
    assert d["timestamp"].endswith("Z") and "T" in d["timestamp"]
    assert isinstance(d["hardware"]["cpu_cores"], int)
    assert d["params"]["zooms"] == [0, 2] and d["params"]["finest_zoom"] == 2
    assert set(d["per_zoom"]) == {"0", "2"}
    assert d["per_zoom"]["2"]["n_features"] == 2
    assert d["obj_comparison"]["n_compared"] == 2
    assert d["runtime_s"]["per_zoom"]["2"] >= 0


def test_cli_without_obj_dir_and_with_max_features(tmp_path):
    root = str(tmp_path / "pyr")
    write_synthetic_pyramid(root, n_neurons=4, zooms=(0, 1))
    out = tmp_path / "probe.json"
    side = tmp_path / "per_feature.parquet"
    fs.main(["--parquet", root, "--zooms", "0,1", "--out", str(out), "--max-features", "2",
             "--per-feature-out", str(side), "--no-face-dedup"])
    d = json.loads(out.read_text())
    assert d["obj_comparison"] is None
    assert d["params"]["max_features"] == 2 and d["params"]["feature_ids"] == [0, 1]
    assert d["params"]["dedup_faces"] is False
    assert d["per_zoom"]["0"]["n_features"] == 2 and d["per_zoom"]["1"]["n_features"] == 2
    assert d["per_zoom"]["0"]["n_unique_tris"] is None
    t = pq.read_table(str(side)).to_pydict()
    assert sorted(zip(t["zoom"], t["feature_id"])) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert t["body_id"][:2] == ["N0", "N1"] or set(t["body_id"]) == {"N0", "N1"}
    assert all(v is None for v in t["n_unique_tris"])


# ----------------------------------------------------------------------------
# Integration: output of the REAL muDM Rust tiler (not the fixture mirror)
# ----------------------------------------------------------------------------

def test_real_tiler_pyramid_finest_level_is_lossless_after_dedup(tmp_path):
    """Tile three spheres with the real tiler (as `test_tile_parquet` does) and
    check the reviewer-facing claims end to end: the real writer's part-file
    naming, `large_binary` layout and `tags` map decode; at the finest zoom
    the dedup'd vertex/face counts and area match the source OBJ exactly
    (area to float32 quantisation), while the raw concatenated counts exceed
    the source because boundary faces are kept whole in every cell."""
    pytest.importorskip("mudm_tools._rs")
    from mudm_paper.tile_parquet import tile_to_parquet

    md = tmp_path / "obj"
    md.mkdir()
    out = str(tmp_path / "pyr")
    src = {}
    for i, bid in enumerate(["100", "200", "300"]):
        pos, faces = make_sphere_mesh(3 + (i % 2))
        pos = pos * 50 + np.array([i * 40.0, 0, 0], np.float32)  # spread across cells
        write_obj(str(md / f"{bid}.obj"), pos, faces)
        src[bid] = (pos, faces)
    meta = md / "meta.json"
    meta.write_text(json.dumps({b: {"neuron_class": "t"} for b in src}))
    tile_to_parquet(str(md), str(meta), out, max_zoom=2, base_cells=10)

    stats = fs.per_feature_zoom_stats(out, [0, 1, 2], log=False)
    finest = stats[2]
    assert {st["body_id"] for st in finest.values()} == set(src)
    for st in finest.values():
        pos, faces = src[st["body_id"]]
        assert st["n_rows"] > 1  # actually fragmented
        assert st["n_unique_verts"] == len(pos)
        assert st["n_unique_tris"] == len(faces)
        assert st["n_verts"] > st["n_unique_verts"] and st["n_tris"] > st["n_unique_tris"]
        assert st["area_unique"] == pytest.approx(_heron_area(pos, faces), rel=1e-6)
        assert st["area"] > st["area_unique"]

    cmp = fs.compare_to_obj(finest, str(md), workers=1)
    assert cmp["n_compared"] == 3 and cmp["n_missing_obj"] == 0
    for key in ("median", "min", "max"):
        assert cmp["unique_verts_ratio"][key] == 1.0
        assert cmp["unique_tris_ratio"][key] == 1.0
        assert cmp["area_unique_ratio"][key] == pytest.approx(1.0, abs=1e-6)
        assert cmp["tris_ratio"][key] > 1.0 and cmp["area_ratio"][key] > 1.0
    assert cmp["n_area_unique_ratio_gt_tol"] == 0
    assert cmp["n_area_ratio_gt_tol"] == 3
    assert cmp["totals"]["tris_excess_fraction"] > 0

    # Coarser zooms are QEM-simplified: never more unique vertices than source.
    for z in (0, 1):
        for st in stats[z].values():
            assert st["n_unique_verts"] <= len(src[st["body_id"]][0])
    summary = fs.summarize(stats)
    assert summary[2]["dup_fraction_weighted"] > 0
    json.dumps(summary, allow_nan=False)
