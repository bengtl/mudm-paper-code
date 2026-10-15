"""Unit tests for the PURE helpers in `mudm_paper.bench_formats`.

These exercise only the stdlib+numpy helpers that run without the Rust
extension or any decoder (the module imports its pipeline/decoder deps lazily,
so importing it here is safe on a machine without them). The integration path
(encode_all / the per-format decoders / run_spec_sheet) runs on the benchmark
host and is deliberately NOT tested here.
"""

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mudm_paper import bench_formats as bf


# ---------------------------------------------------------------------------
# compute_l2_error (index-aligned; copied from benchmark_draco.py)
# ---------------------------------------------------------------------------


def test_l2_identical_is_zero():
    v = np.array([[0.0, 0, 0], [1, 2, 3], [4, 5, 6]])
    m = bf.compute_l2_error(v, v.copy())
    assert m["metric"] == "index_aligned"
    assert m["mean"] == 0.0 and m["max"] == 0.0
    assert m["rel_mean"] == 0.0 and m["rel_max"] == 0.0


def test_l2_known_shift():
    # Shift every vertex by the same vector -> every per-vertex error == |d|.
    v = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
    d = np.array([3.0, 4.0, 0.0])  # |d| = 5
    m = bf.compute_l2_error(v, v + d)
    assert m["mean"] == pytest.approx(5.0)
    assert m["max"] == pytest.approx(5.0)
    # bbox diagonal of v is sqrt(10^2+10^2+10^2) = 17.32...
    assert m["rel_mean"] == pytest.approx(5.0 / float(np.linalg.norm([10, 10, 10])))


def test_l2_shape_mismatch_raises():
    a = np.zeros((5, 3))
    b = np.zeros((4, 3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        bf.compute_l2_error(a, b)


def test_l2_explicit_bbox_diagonal():
    v = np.array([[0.0, 0, 0], [1, 0, 0]])
    m = bf.compute_l2_error(v, v + np.array([2.0, 0, 0]), bbox_diagonal=4.0)
    assert m["mean"] == pytest.approx(2.0)
    assert m["rel_mean"] == pytest.approx(0.5)  # 2 / 4


# ---------------------------------------------------------------------------
# set_l2_error (order-independent; for meshopt / Draco-merge reordering)
# ---------------------------------------------------------------------------


def test_set_l2_identical_is_zero():
    v = np.array([[0.0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m = bf.set_l2_error(v, v.copy())
    assert m["metric"] == "set_based"
    assert m["mean"] == pytest.approx(0.0)
    assert m["max"] == pytest.approx(0.0)


def test_set_l2_is_order_independent():
    # A permuted decode must score ~0 set-based (whereas index-aligned would not).
    v = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10], [5, 5, 5]])
    perm = v[[3, 1, 4, 0, 2]]
    m = bf.set_l2_error(v, perm)
    assert m["mean"] == pytest.approx(0.0)
    assert m["max"] == pytest.approx(0.0)


def test_set_l2_nearest_neighbor_distance():
    # Each decoded point sits 1.0 from its nearest original point.
    orig = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
    dec = orig + np.array([1.0, 0, 0])
    m = bf.set_l2_error(orig, dec)
    assert m["mean"] == pytest.approx(1.0, abs=1e-9)
    assert m["n_decoded"] == 4 and m["n_original"] == 4


def test_set_l2_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        bf.set_l2_error(np.zeros((0, 3)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# dir_size / parquet_partition_sizes
# ---------------------------------------------------------------------------


def test_dir_size_sums_tree(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 250)
    (sub / "c.bin").write_bytes(b"z" * 50)
    deep = sub / "deep"
    deep.mkdir()
    (deep / "d.bin").write_bytes(b"w" * 7)
    assert bf.dir_size(tmp_path) == 100 + 250 + 50 + 7


def test_dir_size_single_file(tmp_path: Path):
    f = tmp_path / "only.bin"
    f.write_bytes(b"q" * 42)
    assert bf.dir_size(f) == 42


def test_parquet_partition_sizes(tmp_path: Path):
    for z, nbytes in [(0, 30), (1, 60), (2, 90)]:
        zdir = tmp_path / f"zoom={z}"
        zdir.mkdir()
        (zdir / "part_000.parquet").write_bytes(b"p" * nbytes)
    # A non-zoom directory must be ignored.
    (tmp_path / "segment_properties").mkdir()
    (tmp_path / "segment_properties" / "info").write_bytes(b"i" * 999)
    sizes = bf.parquet_partition_sizes(tmp_path)
    assert sizes == {0: 30, 1: 60, 2: 90}


# ---------------------------------------------------------------------------
# timed / peak_rss / percentile helpers
# ---------------------------------------------------------------------------


def test_timed_returns_result_and_positive_float():
    result, seconds = bf.timed(lambda x, y: x + y, 21, y=21)
    assert result == 42
    assert isinstance(seconds, float) and seconds >= 0.0


def test_peak_rss_bytes_positive():
    assert bf.peak_rss_bytes() > 0


def test_rss_platform_records_units():
    info = bf.rss_platform()
    assert info["ru_maxrss_units"] in ("bytes", "kilobytes")
    assert info["ru_maxrss_bytes"] > 0
    assert "platform" in info


def test_median_and_percentile():
    assert bf._median([]) == 0.0
    assert bf._median([3.0, 1.0, 2.0]) == 2.0
    assert bf._median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert bf._percentile([], 0.95) == 0.0
    assert bf._percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert bf._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


# ---------------------------------------------------------------------------
# glb_json_bin (pure GLB chunk splitter, copied from verify_meshopt_q.py)
# ---------------------------------------------------------------------------


def _make_glb(json_obj: dict, bin_bytes: bytes) -> bytes:
    json_bytes = json.dumps(json_obj).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)  # pad to 4 bytes
    bin_pad = bin_bytes + b"\x00" * ((4 - len(bin_bytes) % 4) % 4)
    body = (
        struct.pack("<II", len(json_bytes), bf._GLB_JSON_CHUNK) + json_bytes
        + struct.pack("<II", len(bin_pad), bf._GLB_BIN_CHUNK) + bin_pad
    )
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(body))  # "glTF", v2, total len
    return header + body


def test_glb_json_bin_splits_chunks():
    J = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{}]}]}
    BN = b"\x01\x02\x03\x04\x05"
    glb = _make_glb(J, BN)
    j, bn = bf.glb_json_bin(glb)
    assert j["asset"]["version"] == "2.0"
    assert j["meshes"] == [{"primitives": [{}]}]
    assert bn[:5] == BN  # bin chunk recovered (trailing zero-padding allowed)


def test_glb_json_bin_no_json_raises():
    # A GLB header followed by only a BIN chunk -> no JSON chunk -> error.
    bin_pad = b"\x00\x00\x00\x00"
    body = struct.pack("<II", len(bin_pad), bf._GLB_BIN_CHUNK) + bin_pad
    glb = struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body
    with pytest.raises(ValueError, match="no JSON chunk"):
        bf.glb_json_bin(glb)


# ---------------------------------------------------------------------------
# _run_header / assemble_report (schema-valid JSON header)
# ---------------------------------------------------------------------------


def test_run_header_iso8601_and_hardware():
    ts, hw = bf._run_header(hardware={"cpu_cores": 8})
    assert ts.endswith("Z") and "T" in ts
    # ISO-8601 UTC, parseable.
    from datetime import datetime

    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    assert hw == {"cpu_cores": 8}


def test_run_header_default_hardware_has_cpu_cores():
    _ts, hw = bf._run_header()
    assert "cpu_cores" in hw and isinstance(hw["cpu_cores"], int)


def test_assemble_report_shape():
    rep = bf.assemble_report(
        "B7_format_specsheet",
        {"encode": {"parquet": {"encode_s": 1.0}}},
        config={"n": 200},
        hardware={"cpu_cores": 80},
        git_sha_mudm_tools="04370c9",
    )
    assert rep["benchmark_id"] == "B7_format_specsheet"
    assert rep["timestamp"].endswith("Z")
    assert rep["hardware"] == {"cpu_cores": 80}
    assert rep["config"] == {"n": 200}
    assert rep["git_sha_mudm_tools"] == "04370c9"
    assert "encode" in rep["results"]


def _load_schema_validator():
    """Load results/_schema_validator.py by absolute path."""
    here = Path(__file__).resolve()
    # <repo>/mudm_paper/tests/test_bench_formats.py -> <repo>/results/
    repo_dir = here.parents[2]
    validator_path = repo_dir / "results" / "_schema_validator.py"
    spec = importlib.util.spec_from_file_location("_bench_schema_validator", validator_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assembled_report_passes_schema_validator(tmp_path: Path):
    """A written report (named <benchmark_id>.json) must satisfy the committed
    benchmark-results schema validator: benchmark_id == filename stem, ISO-8601
    UTC timestamp, hardware with cpu_cores."""
    validator = _load_schema_validator()
    stem = "B7_format_specsheet"
    rep = bf.assemble_report(
        stem,
        {"decode": {"parquet": {"median_total_s": 0.01}}},
        config={"n": 10},
        hardware={"cpu_cores": 80},
    )
    dest = tmp_path / f"{stem}.json"
    dest.write_text(json.dumps(rep))
    validator.validate_file(dest)  # raises SchemaError on any violation


def test_report_id_must_match_stem_negative(tmp_path: Path):
    """Sanity check the validator is actually enforcing the stem match through
    our header (a mismatched filename is rejected)."""
    validator = _load_schema_validator()
    rep = bf.assemble_report("right_id", {"status": "passed"}, hardware={"cpu_cores": 1})
    dest = tmp_path / "wrong_id.json"
    dest.write_text(json.dumps(rep))
    with pytest.raises(validator.SchemaError, match="does not match filename stem"):
        validator.validate_file(dest)


def test_join_records_counts_matches_only():
    """join_records joins returned feature ids to per-body records, counting
    only features whose body has a record; duplicates collapse to one."""
    from mudm_paper.bench_formats import join_records

    feature_to_body = {1: "10001", 2: "10002", 3: "10003"}
    records = {"10001": [0.1, 0.2], "10003": [0.3, 0.4], "99999": [0.5]}
    # fid 2 has no record; fid 4 has no body mapping; fid 1 repeated.
    assert join_records([1, 1, 2, 3, 4], feature_to_body, records) == 2
    assert join_records([], feature_to_body, records) == 0


def test_io_rchar_int_or_none():
    """_io_rchar returns a non-negative int on Linux and None elsewhere."""
    from mudm_paper.bench_formats import _io_rchar

    v = _io_rchar()
    assert v is None or (isinstance(v, int) and v >= 0)


# ---------------------------------------------------------------------------
# HTTP access-condition PURE helpers (no server; see test_bench_http.py for
# the served end-to-end path)
# ---------------------------------------------------------------------------


def test_parse_ng_multilod_manifest_lod0_layout():
    """The multilod manifest parser must expose LOD-0 fragment (offset, size)
    pairs at the head of the data file: LOD 0 is the finest level and the
    writer concatenates fragments LOD-major (streaming.rs)."""
    num_lods = 2
    m = struct.pack("<3f", 1.0, 2.0, 3.0) + struct.pack("<3f", 10.0, 20.0, 30.0)
    m += struct.pack("<I", num_lods)
    m += struct.pack("<2f", 1.0, 2.0)                # lod_scales
    m += struct.pack("<6f", *([0.0] * 6))            # vertex_offsets [2,3]
    m += struct.pack("<2I", 3, 1)                    # num_fragments_per_lod
    m += struct.pack("<9I", *([0] * 9))              # lod0 fragment_positions [3,3]
    m += struct.pack("<3I", 100, 200, 300)           # lod0 fragment sizes
    m += struct.pack("<3I", 0, 0, 0)                 # lod1 fragment_positions [3,1]
    m += struct.pack("<1I", 40)                      # lod1 fragment size
    parsed = bf.parse_ng_multilod_manifest(m)
    assert parsed["num_lods"] == 2
    assert parsed["num_fragments_per_lod"] == [3, 1]
    assert parsed["lod0_fragments"] == [(0, 100), (100, 200), (300, 300)]
    assert parsed["lod0_bytes"] == 600
    assert parsed["total_bytes"] == 640
    assert parsed["chunk_shape"] == [1.0, 2.0, 3.0]


def test_http_counters_snapshot_delta():
    c = bf.HttpCounters()
    c._on_start("HEAD")
    c._on_end("HEAD", 200, {"Content-Length": "999"}, ctx=bf.HttpCounters.new_ctx())
    a = c.snapshot()
    ctx = bf.HttpCounters.new_ctx()
    c._on_start("GET")
    c._on_end("GET", 206, {"Content-Length": "10"}, ctx=ctx)
    c._on_chunk(b"0123456789", ctx=ctx)
    ctx2 = bf.HttpCounters.new_ctx()
    c._on_start("GET")
    c._on_end("GET", 200, {"Content-Encoding": "br"}, ctx=ctx2)   # no Content-Length
    c._on_chunk(b"x" * 50, ctx=ctx2)
    d = bf.HttpCounters.delta(a, c.snapshot())
    assert a["http_requests"] == 1 and a["head_requests"] == 1
    assert a["bytes_transferred"] == 0                 # HEAD bodies never count
    assert d["http_requests"] == 2 and d["get_requests"] == 2
    assert d["bytes_transferred"] == 10 + 50           # header when present, else body
    assert d["bytes_body"] == 60
    assert c.snapshot()["status_counts"] == {"200": 2, "206": 1}


def test_tiles3d_leaf_uris_picks_deepest_level():
    tileset = {
        "root": {
            "content": {"uri": "0/0/0/0.glb"},
            "children": [
                {"content": {"uri": "1/0/0/0.glb"},
                 "children": [{"content": {"uri": "2/0/0/0.glb"}}, {"content": {"uri": "2/1/0/0.glb"}}]},
                {"content": {"uri": "1/1/0/0.glb"}},
            ],
        }
    }
    depth, uris = bf.tiles3d_finest_uris(tileset)
    assert depth == 2
    assert uris == ["2/0/0/0.glb", "2/1/0/0.glb"]
    assert bf.tiles3d_finest_uris({"root": {}}) == (0, [])


def test_measure_random_access_cold_warm_split():
    """`cold_*` summarise the FIRST fetch of each key only; `warm_median_s` the
    repeats. A fetch that is slower on first sight of a key must show
    cold_median_s > warm_median_s, while the all-sample keys are unchanged."""
    import time

    seen: set = set()

    def fetch(key):
        if key not in seen:
            seen.add(key)
            time.sleep(0.005)  # cold: 5 ms on first sight
        return key

    keys = list(range(6))
    ra = bf.measure_random_access(fetch, keys, iters=3)
    assert ra["n_keys"] == 6 and ra["iters_per_key"] == 3 and ra["n_samples"] == 18
    assert set(ra) >= {"median_s", "p95_s", "cold_median_s", "cold_p95_s", "warm_median_s"}
    assert ra["cold_median_s"] >= 0.005
    assert ra["warm_median_s"] < 0.005
    assert ra["cold_median_s"] > ra["warm_median_s"]
    assert ra["cold_p95_s"] >= ra["cold_median_s"]
    # The all-sample median is dominated by the 12 warm repeats.
    assert ra["median_s"] < ra["cold_median_s"]

    single = bf.measure_random_access(fetch, [100, 101], iters=1)
    assert single["warm_median_s"] is None
    assert single["cold_median_s"] == single["median_s"]
