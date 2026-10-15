"""HTTP / object-storage access-condition tests for `mudm_paper.bench_formats`.

Every test here runs against a LOCAL threaded HTTP server (no public internet)
that mimics the salient behaviour of the muDM catalog origin (S3 + CloudFront):

* `Range: bytes=a-b` honoured with 206 + `Content-Range` (Python's stock
  `SimpleHTTPRequestHandler` ignores Range, so a small handler is written here);
* `Accept-Ranges: bytes` advertised; HEAD supported;
* JSON-like bodies (`*.json`, `info`, legacy NG manifests `<id>:0`) are served
  brotli-encoded with `Content-Encoding: br` REGARDLESS of `Accept-Encoding`,
  exactly as the live host does; Parquet / mesh bytes are served identity;
* missing objects answer 404 (the live host answers 403 for missing keys; the
  discovery code must treat both as "absent", which is exercised separately).

The synthetic pyramid comes from `fixtures.write_synthetic_pyramid` and is then
re-laid-out into the live naming (`part_NNN.mu.parquet`) with one row group per
row and the features spread over a tile grid, so that tile_x/tile_y predicate
pushdown has row-group statistics to prune against and a 3x3 window read can be
shown to transfer fewer bytes than a whole-level read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

# The HTTP client stack lives in the `bench` extra; skip cleanly without it.
brotli = pytest.importorskip("brotli")
pytest.importorskip("aiohttp")
pytest.importorskip("fsspec")
pytest.importorskip("requests")

from mudm_paper import bench_formats as bf  # noqa: E402
from mudm_paper.tests.fixtures import write_synthetic_pyramid  # noqa: E402

# ---------------------------------------------------------------------------
# Range-capable test server
# ---------------------------------------------------------------------------

_BR_SUFFIXES = (".json", ":0")
_BR_NAMES = {"info"}


class _RangeHandler(BaseHTTPRequestHandler):
    """Static-file handler with Range support and CloudFront-like brotli JSON."""

    protocol_version = "HTTP/1.1"
    root: Path
    log: list

    def log_message(self, *_args):  # silence stderr noise
        pass

    def _resolve(self) -> Path | None:
        rel = unquote(urlsplit(self.path).path).lstrip("/")
        fp = (self.root / rel).resolve()
        try:
            fp.relative_to(self.root.resolve())
        except ValueError:
            return None
        return fp if fp.is_file() else None

    def do_HEAD(self):  # noqa: N802
        self._serve(head=True)

    def do_GET(self):  # noqa: N802
        self._serve(head=False)

    def _serve(self, *, head: bool) -> None:
        self.log.append((self.command, urlsplit(self.path).path, self.headers.get("Range")))
        fp = self._resolve()
        if fp is None:
            self.send_response(404)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        data = fp.read_bytes()
        encoding = None
        if fp.name.endswith(_BR_SUFFIXES) or fp.name in _BR_NAMES:
            data = brotli.compress(data)
            encoding = "br"
            ctype = "application/json"
        elif fp.suffix == ".parquet":
            ctype = "application/vnd.apache.parquet"
        else:
            ctype = "application/octet-stream"

        status, start, end = 200, 0, len(data) - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a == "":
                start = max(0, len(data) - int(b))
            else:
                start = int(a)
                if b:
                    end = min(int(b), len(data) - 1)
            if start > end or start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        body = data[start : end + 1]

        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.end_headers()
        if not head:
            self.wfile.write(body)


def _serve(root: Path, log: list) -> tuple[ThreadingHTTPServer, str]:
    handler = type("_H", (_RangeHandler,), {"root": root, "log": log})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------------------
# Synthetic site: pyramid (live naming, pushdown-friendly layout) + NG + 3D Tiles
# ---------------------------------------------------------------------------

GRID_W = 4


def _relayout_for_pushdown(parquet_dir: Path, n_parts: int = 2) -> None:
    """Rewrite each `zoom=Z/part_000.parquet` from the fixture into
    `part_000.mu.parquet .. part_{n_parts-1}` with ONE ROW GROUP PER ROW and the
    rows spread over a GRID_W-wide tile grid (tile_x = i % GRID_W, tile_y = i //
    GRID_W). Row-group statistics on tile_x/tile_y then let a 3x3 window prune
    most row groups, which is the property the HTTP byte-accounting test checks.
    """
    for zdir in sorted(parquet_dir.glob("zoom=*")):
        src = zdir / "part_000.parquet"
        table = pq.read_table(src)
        n = table.num_rows
        tx = pa.array([i % GRID_W for i in range(n)], pa.uint16())
        ty = pa.array([i // GRID_W for i in range(n)], pa.uint16())
        table = table.set_column(table.schema.get_field_index("tile_x"), "tile_x", tx)
        table = table.set_column(table.schema.get_field_index("tile_y"), "tile_y", ty)
        src.unlink()
        per = -(-n // n_parts)
        for k in range(n_parts):
            chunk = table.slice(k * per, per)
            if chunk.num_rows == 0:
                break
            pq.write_table(chunk, zdir / f"part_{k:03d}.mu.parquet", row_group_size=1)


def _write_ng_multilod_loose(ng_dir: Path, feature_id: int, lod_sizes: list[list[int]]) -> bytes:
    """Write a loose multilod NG object (`<id>.index` + `<id>`) whose manifest
    follows the streaming.rs byte layout documented on
    `bench_formats._ng_read_dracopy_loose`. Fragment payloads are synthetic bytes
    (no Draco needed: the HTTP comparator measures fetch, decode is optional).
    Returns the expected LOD-0 payload bytes."""
    num_lods = len(lod_sizes)
    m = struct.pack("<3f", 1.0, 1.0, 1.0) + struct.pack("<3f", 0.0, 0.0, 0.0)
    m += struct.pack("<I", num_lods)
    m += struct.pack(f"<{num_lods}f", *[float(2 ** l) for l in range(num_lods)])
    m += struct.pack(f"<{num_lods * 3}f", *([0.0] * (num_lods * 3)))
    m += struct.pack(f"<{num_lods}I", *[len(s) for s in lod_sizes])
    data = b""
    lod0 = b""
    for lod, sizes in enumerate(lod_sizes):
        nf = len(sizes)
        m += struct.pack(f"<{3 * nf}I", *([0] * (3 * nf)))
        m += struct.pack(f"<{nf}I", *sizes)
        for fi, size in enumerate(sizes):
            frag = bytes([(lod * 16 + fi) % 256]) * size
            data += frag
            if lod == 0:
                lod0 += frag
    ng_dir.mkdir(parents=True, exist_ok=True)
    (ng_dir / f"{feature_id}.index").write_bytes(m)
    (ng_dir / f"{feature_id}").write_bytes(data)
    return lod0


def _make_glb(json_obj: dict, bin_bytes: bytes) -> bytes:
    json_bytes = json.dumps(json_obj).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_pad = bin_bytes + b"\x00" * ((4 - len(bin_bytes) % 4) % 4)
    body = (
        struct.pack("<II", len(json_bytes), bf._GLB_JSON_CHUNK) + json_bytes
        + struct.pack("<II", len(bin_pad), bf._GLB_BIN_CHUNK) + bin_pad
    )
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


@pytest.fixture
def site(tmp_path: Path):
    """A served synthetic muDM dataset root at `<base>/tiles/synthetic`."""
    root = tmp_path / "site"
    dataset_dir = root / "tiles" / "synthetic"
    parquet_dir = dataset_dir / "features.parquet"
    write_synthetic_pyramid(str(parquet_dir), n_neurons=16, zooms=(0, 1, 2))
    _relayout_for_pushdown(parquet_dir, n_parts=2)

    # Loose multilod Neuroglancer store: info (br-served) + objects for a few ids.
    ng_dir = dataset_dir / "neuroglancer"
    ng_dir.mkdir(parents=True)
    (ng_dir / "info").write_text(json.dumps({
        "@type": "neuroglancer_multilod_draco", "vertex_quantization_bits": 10,
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "lod_scale_multiplier": 1.0,
    }))
    lod0_expected = {}
    for fid in range(16):
        lod0_expected[fid] = _write_ng_multilod_loose(
            ng_dir, fid, lod_sizes=[[100 + fid, 50], [30], [10]]
        )

    # Legacy (single-resolution) precomputed mesh store: <id>:0 manifest + fragments.
    legacy_dir = dataset_dir / "ng_legacy"
    legacy_dir.mkdir()
    (legacy_dir / "info").write_text(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
    (legacy_dir / "7:0").write_text(json.dumps({"fragments": ["7:0:a", "7:0:b"]}))
    (legacy_dir / "7:0:a").write_bytes(b"A" * 300)
    (legacy_dir / "7:0:b").write_bytes(b"B" * 200)

    # Sharded store: only an info with a sharding spec.
    sharded_dir = dataset_dir / "ng_sharded"
    sharded_dir.mkdir()
    (sharded_dir / "info").write_text(json.dumps({
        "@type": "neuroglancer_multilod_draco",
        "sharding": {"@type": "neuroglancer_uint64_sharded_v1", "hash": "murmurhash3_x86_128",
                     "preshift_bits": 0, "minishard_bits": 4, "shard_bits": 4},
    }))

    # 3D Tiles: tileset.json (br-served) with a two-level tree; GLBs at z/x/y/d.glb.
    t3d = dataset_dir / "3dtiles"
    for z, x, y, d, nodes in [(0, 0, 0, 0, 1), (1, 0, 0, 0, 2), (1, 1, 0, 0, 3)]:
        gp = t3d / str(z) / str(x) / str(y) / f"{d}.glb"
        gp.parent.mkdir(parents=True, exist_ok=True)
        J = {"asset": {"version": "2.0"},
             "nodes": [{"extras": {"body_id": f"N{i}"}} for i in range(nodes)]}
        gp.write_bytes(_make_glb(J, b"\x01\x02\x03\x04" * 64))
    tileset = {
        "asset": {"version": "1.1"},
        "root": {
            "content": {"uri": "0/0/0/0.glb"},
            "children": [
                {"content": {"uri": "1/0/0/0.glb"}},
                {"content": {"uri": "1/1/0/0.glb"}},
            ],
        },
    }
    (t3d / "tileset.json").write_text(json.dumps(tileset))

    log: list = []
    server, origin = _serve(root, log)
    try:
        yield SimpleNamespace(
            root=root,
            origin=origin,
            base_url=f"{origin}/tiles/synthetic",
            ng_url=f"{origin}/tiles/synthetic/neuroglancer",
            legacy_url=f"{origin}/tiles/synthetic/ng_legacy",
            sharded_url=f"{origin}/tiles/synthetic/ng_sharded",
            tiles3d_url=f"{origin}/tiles/synthetic/3dtiles",
            parquet_dir=parquet_dir,
            log=log,
            lod0_expected=lod0_expected,
        )
    finally:
        server.shutdown()
        server.server_close()


def _local_dataset(parquet_dir: Path):
    return ds.dataset(
        str(parquet_dir),
        partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())])),
    )


# ---------------------------------------------------------------------------
# probe + discovery
# ---------------------------------------------------------------------------


def test_probe_catalog_reports_status_and_headers(site):
    out = bf.probe_catalog(f"{site.origin}/tiles", ["synthetic", "missing-ds"])
    syn = out["synthetic"]
    pq_probe = syn["features.parquet/zoom=0/part_000.mu.parquet"]
    assert pq_probe["status"] == 200 and pq_probe["exists"] is True
    assert pq_probe["content_length"] == os.path.getsize(
        site.parquet_dir / "zoom=0" / "part_000.mu.parquet"
    )
    assert pq_probe["content_encoding"] in (None, "identity")
    assert pq_probe["accept_ranges"] == "bytes"
    assert syn["neuroglancer/info"]["status"] == 200
    assert syn["neuroglancer/info"]["content_encoding"] == "br"
    # tileset.json lives under 3dtiles/ in this fixture; the root path is absent.
    assert syn["tileset.json"]["status"] == 404 and syn["tileset.json"]["exists"] is False
    assert syn["3dtiles/tileset.json"]["status"] == 200
    missing = out["missing-ds"]
    assert all(v["exists"] is False for v in missing.values())
    assert all(v["status"] == 404 for v in missing.values())


def test_discover_parts_stops_at_first_missing(site):
    fs = bf.CountingHTTPFileSystem()
    parts = bf.discover_parts(site.base_url, [0, 2], fs)
    assert parts == {0: ["part_000.mu.parquet", "part_001.mu.parquet"],
                     2: ["part_000.mu.parquet", "part_001.mu.parquet"]}
    # 2 hits + 1 miss per zoom, HEAD only (no GET fallback on the miss).
    snap = fs.counters.snapshot()
    assert snap["http_requests"] == 6
    assert snap["head_requests"] == 6
    assert snap["bytes_transferred"] == 0
    heads = [e for e in site.log if e[0] == "HEAD"]
    gets = [e for e in site.log if e[0] == "GET"]
    assert len(heads) == 6 and gets == []


def test_discover_parts_treats_403_as_absent(site):
    """S3 without ListBucket answers 403 for missing keys: discovery must stop
    there too, not raise."""
    import aiohttp

    class Forbidden(bf.CountingHTTPFileSystem):
        async def _head_info(self, url, **kwargs):
            if "part_001" in url:
                raise aiohttp.ClientResponseError(
                    request_info=None, history=(), status=403, message="Forbidden"
                )
            return await super()._head_info(url, **kwargs)

    parts = bf.discover_parts(site.base_url, [1], Forbidden())
    assert parts == {1: ["part_000.mu.parquet"]}


# ---------------------------------------------------------------------------
# Parquet over HTTP: bulk + window vs local, byte accounting
# ---------------------------------------------------------------------------


def test_http_bulk_matches_local_decode(site):
    zoom = 2
    fs = bf.CountingHTTPFileSystem()
    parts = bf.discover_parts(site.base_url, [zoom], fs)
    dataset = bf.http_parquet_dataset(site.base_url, parts, fs)
    before = fs.counters.snapshot()
    remote = bf.bulk_positions(dataset, zoom)
    delta = bf.HttpCounters.delta(before, fs.counters.snapshot())

    local = bf.decode_parquet_bulk(str(site.parquet_dir), zoom)
    assert set(remote) == set(local)
    assert {f: v.shape[0] for f, v in remote.items()} == {f: v.shape[0] for f, v in local.items()}
    for fid in local:
        np.testing.assert_array_equal(remote[fid], local[fid])
    assert delta["http_requests"] > 0
    assert delta["bytes_transferred"] > 0


def test_http_window_matches_local_and_transfers_fewer_bytes(site):
    zoom = 2
    fs = bf.CountingHTTPFileSystem()
    parts = bf.discover_parts(site.base_url, [zoom], fs)
    dataset = bf.http_parquet_dataset(site.base_url, parts, fs)
    scan = bf.http_scan_options(hole_size_limit=0)

    # Warm the footers so both arms are measured on equal terms, then bulk.
    mx, my = bf.pick_window_origin(dataset, zoom, scan_options=scan)
    before = fs.counters.snapshot()
    bf.bulk_positions(dataset, zoom, scan_options=scan)
    bulk = bf.HttpCounters.delta(before, fs.counters.snapshot())

    site.log.clear()
    before = fs.counters.snapshot()
    n_rows = bf.window_row_count(dataset, zoom, mx, my, scan_options=scan)
    win = bf.HttpCounters.delta(before, fs.counters.snapshot())

    # Same window, same rows as the local predicate on the same files.
    local = _local_dataset(site.parquet_dir)
    assert (mx, my) == bf.pick_window_origin(local, zoom)
    assert n_rows == bf.window_row_count(local, zoom, mx, my)
    assert 0 < n_rows < bf.window_row_count(local, zoom, 0, 0) + 100  # sanity: bounded

    # Pushdown => range reads of only the surviving row groups.
    assert win["http_requests"] > 0
    assert win["bytes_transferred"] < bulk["bytes_transferred"]
    gets = [e for e in site.log if e[0] == "GET" and e[1].endswith(".parquet")]
    assert gets, "window read issued no parquet GETs"
    assert all(rng is not None and rng.startswith("bytes=") for _m, _p, rng in gets), (
        "window read must be served by Range requests, never a whole-file GET"
    )
    # bytes_transferred is the wire count: the server's Content-Length of each
    # ranged body; for identity-encoded Parquet it equals the decoded body.
    assert win["bytes_transferred"] == win["bytes_body"]


# ---------------------------------------------------------------------------
# Neuroglancer + 3D Tiles comparators over HTTP
# ---------------------------------------------------------------------------


def test_ng_http_multilod_loose_fetch(site):
    fs = bf.CountingHTTPFileSystem()
    info = bf.fetch_ng_info(site.ng_url, fs)
    assert info["@type"] == "neuroglancer_multilod_draco"
    before = fs.counters.snapshot()
    got = bf.fetch_ng_object_http(site.ng_url, 3, fs, info=info)
    delta = bf.HttpCounters.delta(before, fs.counters.snapshot())
    assert got["layout"] == "multilod_loose"
    assert got["lod0_bytes"] == len(site.lod0_expected[3]) == 103 + 50
    assert got["n_lod0_fragments"] == 2
    # index (full GET) + one Range GET spanning the contiguous LOD-0 fragments.
    assert delta["http_requests"] == 2
    ranged = [e for e in site.log if e[0] == "GET" and e[1].endswith("/neuroglancer/3")]
    assert ranged and ranged[-1][2] == f"bytes=0-{103 + 50 - 1}"
    assert delta["bytes_transferred"] == got["lod0_bytes"] + got["index_bytes"]


def test_ng_http_legacy_manifest_fetch(site):
    fs = bf.CountingHTTPFileSystem()
    info = bf.fetch_ng_info(site.legacy_url, fs)
    before = fs.counters.snapshot()
    got = bf.fetch_ng_object_http(site.legacy_url, 7, fs, info=info)
    delta = bf.HttpCounters.delta(before, fs.counters.snapshot())
    assert got["layout"] == "legacy_manifest"
    assert got["n_fragments"] == 2
    assert got["fragment_bytes"] == 500
    assert delta["http_requests"] == 3  # manifest + 2 fragments
    # The manifest is served brotli-encoded: bytes_transferred counts the WIRE
    # (compressed) size from Content-Length, bytes_body the decoded manifest.
    manifest_raw = (site.root / "tiles" / "synthetic" / "ng_legacy" / "7:0").read_bytes()
    assert delta["bytes_body"] == len(manifest_raw) + 500
    assert delta["bytes_transferred"] == len(brotli.compress(manifest_raw)) + 500
    assert delta["bytes_transferred"] != delta["bytes_body"]


def test_ng_http_sharded_records_note(site):
    fs = bf.CountingHTTPFileSystem()
    info = bf.fetch_ng_info(site.sharded_url, fs)
    got = bf.fetch_ng_object_http(site.sharded_url, 1, fs, info=info)
    assert got["layout"] == "sharded"
    assert got["note"] == bf.NG_SHARDED_NOTE


def test_tiles3d_http_fetches_finest_glb(site):
    fs = bf.CountingHTTPFileSystem()
    before = fs.counters.snapshot()
    res = bf.measure_tiles3d_http(site.tiles3d_url, fs, n=8, iters=1)
    delta = bf.HttpCounters.delta(before, fs.counters.snapshot())
    assert res["finest_depth"] == 1
    assert res["n_leaf_glb"] == 2 and res["n_fetched"] == 2
    assert res["glb_nodes_total"] == 2 + 3
    assert res["http_requests"] == delta["http_requests"] == 1 + 2  # tileset.json + 2 GLBs
    assert res["bytes_transferred"] > 0


# ---------------------------------------------------------------------------
# End-to-end report + CLI wiring
# ---------------------------------------------------------------------------


def _load_schema_validator():
    here = Path(__file__).resolve()
    # <repo>/mudm_paper/tests/test_bench_http.py -> <repo>/results/
    validator_path = here.parents[2] / "results" / "_schema_validator.py"
    spec = importlib.util.spec_from_file_location("_bench_schema_validator_http", validator_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_access_http_writes_schema_valid_report(site, tmp_path: Path):
    out = tmp_path / "B9_access_http_synthetic.json"
    rep = bf.run_access_http(
        site.base_url, str(out), zoom=2, n_ids=4, iters=2, bulk_n=3,
        ng_url=site.ng_url, tiles3d_url=site.tiles3d_url, hardware={"cpu_cores": 2},
    )
    _load_schema_validator().validate_file(out)
    r = rep["results"]
    assert rep["benchmark_id"] == "B9_access_http_synthetic"
    assert rep["config"]["base_url"] == site.base_url
    assert r["transport"]["parquet"]["accept_ranges"] == "bytes"
    assert r["discovery"]["parts"] == {"2": ["part_000.mu.parquet", "part_001.mu.parquet"]}
    for arm in ("bulk_read", "spatial_window", "by_id_default"):
        assert r[arm]["http_requests"] > 0, arm
        assert r[arm]["bytes_transferred"] > 0, arm
    assert r["bulk_read"]["n_total"] == 16
    assert r["bulk_read"]["parquet_whole_level_s"] > 0
    assert r["spatial_window"]["window"] == "3x3 tiles"
    assert r["spatial_window"]["rows_returned"] > 0
    assert r["by_id_default"]["n_keys"] == 4 and r["by_id_default"]["n_samples"] == 8
    assert r["by_id_neuroglancer"]["layout"] == "multilod_loose"
    assert r["by_id_neuroglancer"]["n_samples"] == 8
    assert r["bulk_read"]["neuroglancer_per_neuron_ms"] > 0
    assert r["tiles3d_http"]["n_fetched"] == 2
    assert json.loads(out.read_text())["results"]["by_id_default"]["scope"].startswith("zoom=2")


def test_run_access_http_with_parts_json_skips_discovery(site, tmp_path: Path):
    parts_json = tmp_path / "parts.json"
    parts_json.write_text(json.dumps({"2": ["part_000.mu.parquet"]}))
    out = tmp_path / "B9_partial.json"
    rep = bf.run_access_http(
        site.base_url, str(out), zoom=2, n_ids=2, iters=1, bulk_n=0,
        parts_json=str(parts_json), hardware={"cpu_cores": 2},
    )
    r = rep["results"]
    assert r["discovery"]["parts"] == {"2": ["part_000.mu.parquet"]}
    assert r["discovery"]["source"] == "parts_json"
    assert r["bulk_read"]["n_total"] == 8  # half the neurons live in part_000
    assert "by_id_neuroglancer" not in r and "tiles3d_http" not in r


def test_cli_probe_http_and_access_http(site, tmp_path: Path, capsys):
    probe_out = tmp_path / "probe.json"
    bf.main([
        "--mode", "probe-http", "--base-url", f"{site.origin}/tiles",
        "--datasets", "synthetic,nope", "--out", str(probe_out),
    ])
    captured = capsys.readouterr().out
    assert "synthetic" in captured and "nope" in captured
    probe = json.loads(probe_out.read_text())
    assert probe["synthetic"]["features.parquet/zoom=0/part_000.mu.parquet"]["status"] == 200
    assert probe["nope"]["features.parquet/zoom=0/part_000.mu.parquet"]["status"] == 404

    out = tmp_path / "B9_cli.json"
    bf.main([
        "--mode", "access-http", "--base-url", site.base_url, "--zoom", "1",
        "--n", "2", "--iters", "1", "--bulk-n", "2", "--out", str(out),
        "--ng-url", site.ng_url, "--hole-size-limit", "0",
    ])
    rep = json.loads(out.read_text())
    assert rep["benchmark_id"] == "B9_cli"
    assert rep["results"]["zoom"] == 1
    assert rep["config"]["hole_size_limit"] == 0
    assert "wrote" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Discovery at scale: galloping search + parallel HEAD prefetch
# ---------------------------------------------------------------------------


def _write_dummy_parts(site, zoom: int, n: int) -> None:
    zdir = site.parquet_dir / f"zoom={zoom}"
    zdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (zdir / f"part_{i:03d}.mu.parquet").write_bytes(b"PAR1" + bytes(16) + b"PAR1")


def test_discover_parts_gallops_with_fewer_heads_than_linear(site):
    # 12 contiguous parts at a synthetic zoom 7 (HEAD only; bodies irrelevant).
    _write_dummy_parts(site, 7, 12)
    fs = bf.CountingHTTPFileSystem()
    parts = bf.discover_parts(site.base_url, [7], fs)
    assert parts[7] == [f"part_{i:03d}.mu.parquet" for i in range(12)]
    heads = fs.counters.snapshot()["head_requests"]
    assert heads < 12 + 1  # linear probing would need n + 1
    assert fs.counters.snapshot()["get_requests"] == 0


def test_discover_parts_respects_cap(site):
    _write_dummy_parts(site, 8, 12)
    fs = bf.CountingHTTPFileSystem()
    parts = bf.discover_parts(site.base_url, [8], fs, cap=5)
    assert len(parts[8]) == 5
    assert bf.discover_parts(site.base_url, [9], fs) == {9: []}  # no part_000 at all


def test_prefetch_info_serves_dataset_open_from_memo(site):
    fs = bf.CountingHTTPFileSystem()
    parts = bf.discover_parts(site.base_url, [2], fs)
    urls = [bf._part_url(site.base_url, 2, n) for n in parts[2]]
    n = bf.prefetch_info(fs, urls + [site.base_url + "/features.parquet/zoom=2/nope.parquet"])
    assert n == 3
    heads_before = fs.counters.snapshot()["head_requests"]
    bf.http_parquet_dataset(site.base_url, parts, fs)  # pyarrow's per-path info() hits the memo
    assert fs.counters.snapshot()["head_requests"] == heads_before


def test_cli_legacy_modes_still_require_obj_dir_and_out(tmp_path: Path):
    """--obj-dir/--out are no longer argparse-required (probe-http needs
    neither) but spec-sheet / access must still refuse to run without them."""
    with pytest.raises(SystemExit):
        bf.main(["--mode", "access", "--out", str(tmp_path / "x.json"), "--out-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        bf.main(["--mode", "spec-sheet", "--obj-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        bf.main(["--mode", "access-http", "--out", str(tmp_path / "x.json")])  # no --base-url
    with pytest.raises(SystemExit):
        bf.main(["--mode", "probe-http", "--base-url", "http://127.0.0.1:9"])  # no --datasets
