"""Format spec-sheet benchmark: muDM output formats on reviewer-familiar measures.

This module builds ONE shared muDM tile index from a directory of `<body_id>.obj`
neuron meshes, then emits the three muDM output formats from that single index
(the "one index -> many formats" property) and measures each on the standard axes
a reviewer expects from a storage/serialization comparison:

    1. encode time      -- wall-clock around each emitter call.
    2. stored size       -- bytes on disk (whole tree; per-zoom for parquet).
    3. decode time       -- bytes -> (positions, indices) in memory, NATIVE decoders.
    4. random-access     -- fetch ONE neuron by body_id from each format.
    5. peak memory       -- ru_maxrss high-water around encode and decode-all.
    6. fidelity          -- L2 vertex error vs the source OBJ, per format.

Formats compared:
    obj            -- source / baseline (Wavefront OBJ, parsed by the Rust `parse_obj`).
    parquet        -- muDM Parquet + ZSTD, full float32 (lossless geometry).
    tiles3d        -- muDM 3D Tiles, meshopt-q14 (u16-quantized, KHR_mesh_quantization).
    neuroglancer   -- muDM Neuroglancer multi-LOD Draco, optionally sharded.

Design note -- pure vs integration split
-----------------------------------------
The pure, unit-testable helpers at the top of this module import ONLY the stdlib
and numpy, so this file imports (and the helpers run) on a machine without the
Rust extension, DracoPy, meshoptimizer, cloudvolume, or trimesh. Every function
that touches the real pipeline or a decoder imports it LAZILY, inside the
function body. The integration path runs on the benchmark host; the committed
unit tests cover only the pure helpers.

Reference implementations mirrored here (the first three live in mudm-tools):
    scripts/benchmark_draco.py                -- `compute_l2_error`, encode/decode timing shape.
    rust/tools/mmc/verify_meshopt_q.py        -- `glb_json_bin`, `decode_positions` (meshopt-q14).
    tests/test_draco_inmemory.py              -- DracoPy.decode -> mesh.points / mesh.faces.
    mudm_paper/data.py                        -- parquet HivePartitioning + frombuffer decode.
    mudm_paper/tile_parquet.py                -- shared-index encode (scan_obj_bounds + gen).
    mudm_paper/run_manc.py                    -- `_run_header` schema-valid JSON header.

Access-pattern modes: `--mode access` measures bulk / spatial-window / by-id on
LOCAL disk; `--mode access-http` repeats the Parquet patterns over HTTP range
reads from the deployed object store (plus NG / 3D Tiles per-object GETs) with
per-arm request and byte accounting; `--mode probe-http` reports which artefacts
a deployed dataset exposes. See the "HTTP / object-storage" section below.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urljoin

import numpy as np

# GLB chunk-type magic (little-endian ASCII), copied from verify_meshopt_q.py.
_GLB_JSON_CHUNK = 0x4E4F534A  # b"JSON"
_GLB_BIN_CHUNK = 0x004E4942   # b"BIN\0"


# ===========================================================================
# Pure helpers (stdlib + numpy only; importable and testable on any machine)
# ===========================================================================


def dir_size(path: str | Path) -> int:
    """Total size in bytes of every regular file under `path` (recursive).

    Follows the directory tree with `os.walk`; symlinks are not followed and
    unreadable entries (races, broken links) are skipped rather than raising,
    so a live output directory can be measured safely.
    """
    total = 0
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def parquet_partition_sizes(parquet_dir: str | Path) -> dict[int, int]:
    """Per-zoom-level byte size of a partitioned parquet pyramid.

    Returns ``{zoom: bytes}`` by summing the file sizes under each
    ``zoom=<N>/`` Hive-partition directory. This is a pure filesystem walk (no
    pyarrow), so it captures the on-disk cost of each level of detail exactly as
    stored (positions + indices + tags + parquet framing, ZSTD-compressed).
    """
    sizes: dict[int, int] = {}
    root = Path(parquet_dir)
    if not root.is_dir():
        return sizes
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("zoom="):
            try:
                zoom = int(child.name.split("=", 1)[1])
            except ValueError:
                continue
            sizes[zoom] = dir_size(child)
    return sizes


def compute_l2_error(
    original: np.ndarray,
    decoded: np.ndarray,
    bbox_diagonal: float | None = None,
) -> dict[str, float]:
    """Index-aligned per-vertex L2 error between original and decoded vertices.

    Copied verbatim (behaviour-wise) from ``scripts/benchmark_draco.py``. This
    metric assumes vertex `i` of `decoded` corresponds to vertex `i` of
    `original` -- valid for order-preserving decoders (DracoPy) when the vertex
    counts match, and raises on a shape mismatch. For decoders that permute or
    change the vertex count (meshopt, cloudvolume's multi-fragment merge), use
    `set_l2_error` instead.

    Args:
        original: Nx3 original vertex positions.
        decoded: Nx3 decoded vertex positions.
        bbox_diagonal: Bounding-box diagonal for relative error; computed from
            `original` if None.

    Returns:
        Dict with mean, max, p95, p99, and relative-to-diagonal error metrics.
    """
    orig = np.asarray(original, dtype=np.float64)
    dec = np.asarray(decoded, dtype=np.float64)

    if orig.shape != dec.shape:
        raise ValueError(
            f"Shape mismatch: original {orig.shape} vs decoded {dec.shape}"
        )

    diffs = np.linalg.norm(orig - dec, axis=1)

    if bbox_diagonal is None:
        bbox_min = orig.min(axis=0)
        bbox_max = orig.max(axis=0)
        bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    diag = max(bbox_diagonal, 1e-10)
    sorted_diffs = np.sort(diffs)
    n = len(sorted_diffs)

    return {
        "metric": "index_aligned",
        "n": int(n),
        "mean": float(np.mean(diffs)) if n else 0.0,
        "max": float(np.max(diffs)) if n else 0.0,
        "p95": float(sorted_diffs[int(n * 0.95)]) if n > 0 else 0.0,
        "p99": float(sorted_diffs[int(n * 0.99)]) if n > 0 else 0.0,
        "rel_mean": (float(np.mean(diffs)) / diag) if n else 0.0,
        "rel_max": (float(np.max(diffs)) / diag) if n else 0.0,
    }


def set_l2_error(
    original: np.ndarray,
    decoded: np.ndarray,
    bbox_diagonal: float | None = None,
) -> dict[str, float]:
    """Order-independent (set-based) L2 error: each decoded vertex to its
    nearest original vertex.

    Used where the decoder does NOT preserve vertex order or count -- meshopt
    permutes the vertex buffer, and cloudvolume/Draco multi-fragment merges
    change both order and count -- so an index-aligned comparison is invalid.
    This measures the displacement of each decoded vertex from the original
    point set (a one-directional nearest-neighbour distance), which is the
    quantity that a quantizing format perturbs.

    Uses `scipy.spatial.cKDTree` when available (lazy import); falls back to a
    numpy brute-force nearest-neighbour for small meshes / no-scipy hosts. The
    result carries ``"metric": "set_based"`` so downstream reporting never
    conflates it with the index-aligned metric.
    """
    orig = np.asarray(original, dtype=np.float64).reshape(-1, 3)
    dec = np.asarray(decoded, dtype=np.float64).reshape(-1, 3)
    if orig.shape[0] == 0 or dec.shape[0] == 0:
        raise ValueError(f"empty input: original {orig.shape}, decoded {dec.shape}")

    try:
        from scipy.spatial import cKDTree  # lazy: keeps the helper importable w/o scipy

        tree = cKDTree(orig)
        dists, _ = tree.query(dec, k=1)
        diffs = np.asarray(dists, dtype=np.float64)
    except Exception:
        # Brute force: nearest original vertex for each decoded vertex. O(V*N);
        # fine for the small meshes exercised by the unit tests and an adequate
        # fallback when scipy is unavailable.
        diffs = np.empty(dec.shape[0], dtype=np.float64)
        for i in range(dec.shape[0]):
            diffs[i] = np.min(np.linalg.norm(orig - dec[i], axis=1))

    if bbox_diagonal is None:
        bbox_min = orig.min(axis=0)
        bbox_max = orig.max(axis=0)
        bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    diag = max(bbox_diagonal, 1e-10)

    sorted_diffs = np.sort(diffs)
    n = len(sorted_diffs)
    return {
        "metric": "set_based",
        "n_decoded": int(dec.shape[0]),
        "n_original": int(orig.shape[0]),
        "mean": float(np.mean(diffs)),
        "max": float(np.max(diffs)),
        "p95": float(sorted_diffs[int(n * 0.95)]) if n > 0 else 0.0,
        "p99": float(sorted_diffs[int(n * 0.99)]) if n > 0 else 0.0,
        "rel_mean": float(np.mean(diffs)) / diag,
        "rel_max": float(np.max(diffs)) / diag,
    }


def timed(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call `fn(*args, **kwargs)` and return `(result, elapsed_seconds)`.

    Uses `time.perf_counter` (monotonic, highest available resolution), matching
    the timing loop in `scripts/benchmark_draco.py`.
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def peak_rss_bytes() -> int:
    """Current process peak resident set size (high-water mark), in BYTES.

    `resource.getrusage(...).ru_maxrss` is reported in bytes on macOS (Darwin)
    and in kilobytes on Linux; this normalises both to bytes. Note that
    ru_maxrss is monotonic non-decreasing over the process lifetime -- it never
    reports a *drop* after memory is freed -- so a delta across an operation
    only reflects growth if that operation set a new all-time high. Callers
    should treat inter-phase deltas as a lower bound on that phase's marginal
    peak and rely on `rss_platform()` to record the raw units.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def rss_platform() -> dict[str, Any]:
    """Record the raw ru_maxrss reading, its units, and the platform.

    Kept alongside the normalised `peak_rss_bytes()` so the JSON preserves the
    exact OS-reported number (kB on Linux, bytes on macOS) for audit.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "platform": sys.platform,
        "ru_maxrss_raw": int(raw),
        "ru_maxrss_units": "bytes" if sys.platform == "darwin" else "kilobytes",
        "ru_maxrss_bytes": peak_rss_bytes(),
    }


def _run_header(hardware: dict | None = None) -> tuple[str, dict]:
    """Return `(timestamp, hardware)` for a validator-ready result header.

    Mirrors `mudm_paper/run_manc.py::_run_header`: an ISO-8601 UTC timestamp
    (``...Z``) plus a hardware dict defaulting to the host CPU count (and any
    visible CUDA devices, if torch happens to be importable -- not required
    here). The schema validator requires `hardware` to carry at least
    `cpu_cores` or `cpu_model`.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if hardware is None:
        hardware = {"cpu_cores": os.cpu_count() or 0}
        try:
            import torch

            if torch.cuda.is_available():
                hardware["gpu"] = ", ".join(
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                )
        except Exception:
            pass
    return timestamp, hardware


def assemble_report(
    benchmark_id: str,
    results: dict,
    *,
    config: dict | None = None,
    hardware: dict | None = None,
    git_sha_mudm_tools: str | None = None,
) -> dict:
    """Assemble a schema-valid benchmark JSON dict.

    The output satisfies `results/_schema_validator.py`:
    ``benchmark_id`` (must equal the output filename stem), an ISO-8601 UTC
    ``timestamp``, a ``hardware`` dict with at least ``cpu_cores``, and the
    domain payload under ``results``. ``config`` and ``git_sha_mudm_tools`` are
    optional-but-expected metadata.
    """
    timestamp, hw = _run_header(hardware)
    report: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "timestamp": timestamp,
        "hardware": hw,
        "results": results,
    }
    if config is not None:
        report["config"] = config
    if git_sha_mudm_tools is not None:
        report["git_sha_mudm_tools"] = git_sha_mudm_tools
    return report


def _median(xs: list[float]) -> float:
    """Median of a list, 0.0 on empty (statistics.median raises on empty)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0)


def _percentile(xs: list[float], q: float) -> float:
    """Simple nearest-rank percentile (q in [0, 1]); 0.0 on empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(len(s) * q))])


# ===========================================================================
# GLB / meshopt decode helpers (mirror verify_meshopt_q.py exactly)
# ===========================================================================


def glb_json_bin(b: bytes) -> tuple[dict, bytes | None]:
    """Split a GLB binary into its JSON and BIN chunks.

    Copied from `rust/tools/mmc/verify_meshopt_q.py`. Returns
    ``(json_dict, bin_bytes)``; `bin_bytes` is None if the GLB has no BIN chunk.
    """
    off = 12  # skip the 12-byte GLB header (magic, version, length)
    J: dict | None = None
    BN: bytes | None = None
    while off < len(b):
        clen, ct = struct.unpack("<II", b[off:off + 8])
        off += 8
        c = b[off:off + clen]
        off += clen
        if ct == _GLB_JSON_CHUNK:
            J = json.loads(c)
        elif ct == _GLB_BIN_CHUNK:
            BN = c
    if J is None:
        raise ValueError("GLB has no JSON chunk")
    return J, BN


def decode_positions(J: dict, BN: bytes, prim: dict, node: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decode a meshopt-compressed primitive's POSITION to world coordinates.

    Copied from `rust/tools/mmc/verify_meshopt_q.py::decode_positions`. Reads the
    `EXT_meshopt_compression` bufferView, calls the stock meshoptimizer vertex
    decoder, then applies the node TRS (translation + scale) per
    KHR_mesh_quantization. Handles the u16-quantized (componentType 5123, stride
    8 = 4x u16, keep xyz) and raw-f32 (stride 12) cases.

    Returns ``(world_positions Nx3 float64, scale [3])``. Requires the
    `meshoptimizer` package (lazy import).
    """
    a = J["accessors"][prim["attributes"]["POSITION"]]
    bv = J["bufferViews"][a["bufferView"]]
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    comp = a.get("componentType", 5126)  # 5123 = u16 (q14), 5126 = f32
    count = int(a["count"])
    ext = bv.get("extensions", {}).get("EXT_meshopt_compression")
    if ext is not None:
        # meshopt-compressed bufferView -> stock meshopt vertex decoder.
        import meshoptimizer as mo  # lazy

        raw = BN[ext["byteOffset"]:ext["byteOffset"] + ext["byteLength"]]
        if comp == 5123:  # u16 quantized (meshopt-q14 wire format), stride 8 = 4x u16
            dec = np.asarray(
                mo.decode_vertex_buffer(ext["count"], 8, raw, np.dtype((np.uint16, 4)))
            ).reshape(-1, 4)[:, :3].astype(np.float64)
        else:  # f32
            dec = np.asarray(
                mo.decode_vertex_buffer(ext["count"], 12, raw, np.dtype((np.float32, 3)))
            ).reshape(-1, 3).astype(np.float64)
    else:
        # Uncompressed bufferView: the encoder leaves small meshes raw (still
        # KHR_mesh_quantization u16 in a byteStride-8 buffer, or plain f32).
        # Read directly at bufferView.byteOffset + accessor.byteOffset, honouring
        # byteStride, and keep xyz.
        base = int(bv.get("byteOffset", 0)) + int(a.get("byteOffset", 0))
        if comp == 5123:
            stride = int(bv.get("byteStride", 8)) or 8
            raw = BN[base:base + count * stride]
            dec = np.frombuffer(raw, dtype="<u2").reshape(count, stride // 2)[:, :3].astype(np.float64)
        else:
            stride = int(bv.get("byteStride", 12)) or 12
            raw = BN[base:base + count * stride]
            dec = np.frombuffer(raw, dtype="<f4").reshape(count, stride // 4)[:, :3].astype(np.float64)
    return t + s * dec, s


def _glb_triangle_indices(J: dict, BN: bytes, prim: dict) -> np.ndarray:
    """Decode a primitive's triangle indices (meshopt-compressed index buffer).

    Returns an Mx3 uint32 array, or an empty (0,3) array if the primitive has no
    indices. Uses `meshoptimizer.decode_index_buffer`. Best-effort: index
    decoding is not required for the vertex-fidelity metric, so any failure
    yields an empty index array rather than raising.
    """
    if "indices" not in prim:
        return np.empty((0, 3), dtype=np.uint32)
    try:
        acc = J["accessors"][prim["indices"]]
        bv = J["bufferViews"][acc["bufferView"]]
        comp = acc.get("componentType", 5125)  # 5125=u32, 5123=u16, 5121=u8
        width = 4 if comp == 5125 else (2 if comp == 5123 else 1)
        np_dtype = {4: "<u4", 2: "<u2", 1: "u1"}[width]
        ext = bv.get("extensions", {}).get("EXT_meshopt_compression")
        if ext is not None:
            import meshoptimizer as mo  # lazy

            raw = BN[ext["byteOffset"]:ext["byteOffset"] + ext["byteLength"]]
            idx = np.asarray(
                mo.decode_index_buffer(
                    ext["count"], width, raw, np.dtype(np.uint32 if width == 4 else np.uint16)
                )
            )
        else:
            # Uncompressed index bufferView (small meshes): tightly packed.
            count = int(acc["count"])
            base = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
            raw = BN[base:base + count * width]
            idx = np.frombuffer(raw, dtype=np_dtype)
        return idx.astype(np.uint32).reshape(-1, 3)
    except Exception:
        return np.empty((0, 3), dtype=np.uint32)


# ===========================================================================
# Integration: shared-index encode (mirrors tile_parquet.py)
# ===========================================================================


def scan_bounds(obj_paths: list[str]) -> tuple:
    """World bounds for a set of OBJ paths via the Rust parallel bounds scan."""
    from mudm_tools._rs import scan_obj_bounds  # lazy

    return scan_obj_bounds(list(obj_paths))


def encode_all(
    obj_paths: list[str],
    out_root: str,
    bounds: tuple | None = None,
    *,
    max_zoom: int = 4,
    base_cells: int = 100,
    sharded: bool = True,
    vertex_quantization_bits: int = 10,
    io_threads: int = 12,
    tags_list: list[dict] | None = None,
) -> dict:
    """Build ONE shared tile index and emit all three muDM formats from it.

    This is the "one index -> many formats" property: a single
    `StreamingTileGenerator` is ingested once with `add_obj_files`, then each
    emitter reads the SAME flushed fragment store. (`verify_meshopt_q.py` relies
    on the same pattern -- two `generate_3dtiles` calls on one generator.)

    For each format, wall-clock encode time, on-disk size, and the ru_maxrss
    high-water mark immediately after the emitter are recorded. The parquet
    section additionally records per-zoom-level bytes.

    Args:
        obj_paths: Sorted list of `<body_id>.obj` mesh paths.
        out_root: Root directory; formats land in `parquet/`, `tiles3d/`,
            `neuroglancer/` subdirectories.
        bounds: Precomputed world bounds; scanned if None.
        max_zoom / base_cells: Pyramid depth and base grid resolution.
        sharded: Emit the Neuroglancer sharded (`neuroglancer_uint64_sharded_v1`)
            layout when True, else loose per-segment files.
        vertex_quantization_bits: NG Draco quantization (default 10).
        io_threads: Parquet parallel I/O concurrency.
        tags_list: Optional per-file tag dicts (must include `body_id`);
            defaults to `{"body_id": <stem>}` per file, matching tile_parquet.py.

    Returns:
        Dict with `dirs` (per-format output paths), `bounds`, `n_neurons`, and a
        per-format `encode` block (`encode_s`, `stored_bytes`,
        `peak_rss_after` / `peak_rss_bytes_after`, and for parquet
        `size_per_zoom`).
    """
    from mudm_tools._rs import StreamingTileGenerator  # lazy
    from mudm_tools.tiling3d.parquet_writer import generate_parquet  # lazy

    obj_paths = list(obj_paths)
    if bounds is None:
        bounds = scan_bounds(obj_paths)
    if tags_list is None:
        tags_list = [
            {"body_id": os.path.splitext(os.path.basename(p))[0]} for p in obj_paths
        ]

    dirs = {
        "parquet": os.path.join(out_root, "parquet"),
        "tiles3d": os.path.join(out_root, "tiles3d"),
        "neuroglancer": os.path.join(out_root, "neuroglancer"),
    }

    # One generator, one ingest -- the shared index every emitter reads from.
    # TMPDIR is passed through explicitly (point it at a volume with room for
    # the tiler's spill files) exactly as tile_parquet.py does.
    gen = StreamingTileGenerator(
        min_zoom=0,
        max_zoom=max_zoom,
        base_cells=base_cells,
        temp_dir=os.environ.get("TMPDIR"),
    )
    # Time the shared index build (ingest + QEM simplification across all zooms).
    # This is the once-per-dataset cost amortised across every emitter below.
    _, shared_index_build_s = timed(
        gen.add_obj_files, obj_paths, bounds, tags_list, ingest_threads=0, simplify=True
    )

    encode: dict[str, Any] = {}

    # --- parquet (muDM Parquet + ZSTD, full float32) ---
    _, t = timed(
        generate_parquet,
        gen,
        dirs["parquet"],
        bounds,
        partitioned=True,
        parallel=True,
        io_threads=io_threads,
    )
    encode["parquet"] = {
        "encode_s": t,
        "stored_bytes": dir_size(dirs["parquet"]),
        "size_per_zoom": parquet_partition_sizes(dirs["parquet"]),
        "peak_rss_bytes_after": peak_rss_bytes(),
        "peak_rss_after": rss_platform(),
    }

    # --- tiles3d (muDM 3D Tiles, meshopt-q14) ---
    _, t = timed(gen.generate_3dtiles, dirs["tiles3d"], bounds, compression="meshopt-q14")
    encode["tiles3d"] = {
        "encode_s": t,
        "stored_bytes": dir_size(dirs["tiles3d"]),
        "peak_rss_bytes_after": peak_rss_bytes(),
        "peak_rss_after": rss_platform(),
    }

    # --- neuroglancer (muDM Neuroglancer multi-LOD Draco, sharded) ---
    _, t = timed(
        gen.generate_neuroglancer_multilod,
        dirs["neuroglancer"],
        bounds,
        vertex_quantization_bits=vertex_quantization_bits,
        sharded=sharded,
    )
    encode["neuroglancer"] = {
        "encode_s": t,
        "stored_bytes": dir_size(dirs["neuroglancer"]),
        "peak_rss_bytes_after": peak_rss_bytes(),
        "peak_rss_after": rss_platform(),
        "sharded": bool(sharded),
        "vertex_quantization_bits": int(vertex_quantization_bits),
    }

    return {
        "dirs": dirs,
        "bounds": list(bounds),
        "n_neurons": len(obj_paths),
        "max_zoom": max_zoom,
        "shared_index_build_s": shared_index_build_s,
        "encode": encode,
    }


# ===========================================================================
# Integration: body_id <-> feature_id resolution + per-format decoders
# ===========================================================================


def build_body_feature_map(parquet_dir: str) -> dict[str, int]:
    """Map `body_id -> feature_id` by scanning the parquet `feature_id` column
    and `tags`.

    `feature_id` is the tiler's internal segment key (the Neuroglancer segment
    id and the natural parquet predicate-pushdown key); `body_id` is a tag. The
    formats key their features by `feature_id`, so callers that address neurons
    by the human-facing `body_id` need this map. Built once and reused.
    """
    import pyarrow as pa  # lazy
    import pyarrow.dataset as ds  # lazy

    dataset = ds.dataset(
        parquet_dir,
        partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())])),
    )
    table = dataset.to_table(columns=["feature_id", "tags"])
    out: dict[str, int] = {}
    for fid, tags in zip(
        table.column("feature_id").to_pylist(),
        table.column("tags").to_pylist(),
    ):
        body_id = dict(tags).get("body_id")
        if body_id is not None and body_id not in out:
            out[body_id] = int(fid)
    return out


def decode_obj(obj_paths: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse OBJ meshes to `{body_id: (positions Nx3 f64, faces Mx3 u32)}`.

    Primary path is the Rust `parse_obj` (as in `benchmark_draco.py`); falls back
    to `trimesh` if the extension is unavailable. body_id is the filename stem.
    """
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        from mudm_tools._rs import parse_obj  # lazy, primary

        for p in obj_paths:
            verts, faces = parse_obj(str(p))
            verts = np.array(verts, dtype=np.float64).reshape(-1, 3)
            faces = np.array(faces, dtype=np.uint32).reshape(-1, 3)
            body_id = os.path.splitext(os.path.basename(p))[0]
            meshes[body_id] = (verts, faces)
        return meshes
    except Exception:
        import trimesh  # lazy, fallback

        for p in obj_paths:
            m = trimesh.load(str(p), process=False)
            verts = np.asarray(m.vertices, dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(m.faces, dtype=np.uint32).reshape(-1, 3)
            body_id = os.path.splitext(os.path.basename(p))[0]
            meshes[body_id] = (verts, faces)
        return meshes


def decode_parquet(
    parquet_dir: str,
    feature_ids: list[int],
    *,
    zoom: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Decode the finest-level (given `zoom`) geometry of each `feature_id` from
    the parquet pyramid.

    Mirrors `mudm_paper/data.py`: opens the dataset with a
    `HivePartitioning([("zoom", int32)])`, then materialises only the requested
    zoom partition and the requested features via a pushed-down predicate on the
    `zoom` and `feature_id` columns. A neuron may span several tiles at a zoom,
    so per-feature rows are concatenated with the running-vertex-count offset
    applied to each tile's (tile-local) indices -- the same concatenation the
    loader does.

    Returns `{feature_id: (positions Nx3 f32, indices Mx3 u32)}`.
    """
    import pyarrow as pa  # lazy
    import pyarrow.dataset as ds  # lazy

    dataset = ds.dataset(
        parquet_dir,
        partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())])),
    )
    wanted = [int(f) for f in feature_ids]
    table = dataset.to_table(
        columns=["feature_id", "positions", "indices"],
        filter=(ds.field("zoom") == int(zoom)) & ds.field("feature_id").isin(wanted),
    )
    accum: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for fid, pos_bytes, idx_bytes in zip(
        table.column("feature_id").to_pylist(),
        table.column("positions").to_pylist(),
        table.column("indices").to_pylist(),
    ):
        pos = np.frombuffer(pos_bytes, dtype="<f4").reshape(-1, 3)
        idx = np.frombuffer(idx_bytes, dtype="<u4").reshape(-1, 3)
        accum.setdefault(int(fid), []).append((pos, idx))

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fid, chunks in accum.items():
        positions, indices, offset = [], [], 0
        for pos, idx in chunks:
            positions.append(pos)
            indices.append(idx.astype(np.int64) + offset)
            offset += len(pos)
        all_pos = np.concatenate(positions, axis=0).astype(np.float32)
        all_idx = np.concatenate(indices, axis=0).astype(np.uint32)
        out[fid] = (all_pos, all_idx)
    return out


def _build_tiles3d_index(tiles3d_dir: str, max_zoom: int) -> dict[str, list[tuple[str, int]]]:
    """Index the finest-zoom 3D Tiles GLBs: `body_id -> [(glb_path, node_idx), ...]`.

    3D Tiles is tile-centric (one GLB per `output_dir/{z}/{x}/{y}/{d}.glb`, one
    node per feature per tile) with NO native feature index, so addressing a
    neuron by id requires scanning GLB node `extras` (which carry the feature
    tags, incl. `body_id`; see encoder_glb.rs). This scans only the finest zoom
    level. Building this index is itself the tile-centric random-access penalty
    and is timed separately by the caller.
    """
    index: dict[str, list[tuple[str, int]]] = {}
    finest = os.path.join(tiles3d_dir, str(max_zoom))
    pattern = os.path.join(finest, "**", "*.glb")
    for gp in glob.glob(pattern, recursive=True):
        try:
            J, _BN = glb_json_bin(Path(gp).read_bytes())
        except Exception:
            continue
        for ni, node in enumerate(J.get("nodes", [])):
            extras = node.get("extras") or {}
            body_id = extras.get("body_id")
            if body_id is not None:
                index.setdefault(str(body_id), []).append((gp, ni))
    return index


def decode_tiles3d_meshopt(
    tiles3d_dir: str,
    body_ids: list[str],
    *,
    max_zoom: int,
    index: dict[str, list[tuple[str, int]]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Decode each neuron's finest-zoom geometry from the meshopt-q14 3D Tiles.

    For each body_id, locate its finest-zoom nodes (via `index`, built if not
    supplied), decode each node's POSITION with the stock meshopt decoder, apply
    the node TRS (KHR_mesh_quantization) -- exactly `verify_meshopt_q.py`'s
    `decode_positions` -- and concatenate the per-tile vertices/indices into one
    per-neuron mesh. meshopt permutes vertex order, so fidelity on this output
    must use `set_l2_error`, never `compute_l2_error`.

    Returns `{body_id: (positions Nx3 f64, indices Mx3 u32)}` (indices best-effort;
    may be empty (0,3) if index decoding fails -- vertex fidelity does not need them).
    """
    if index is None:
        index = _build_tiles3d_index(tiles3d_dir, max_zoom)

    # Cache parsed GLBs across features so a shared tile is read once.
    glb_cache: dict[str, tuple[dict, bytes]] = {}

    def _load(gp: str) -> tuple[dict, bytes]:
        if gp not in glb_cache:
            glb_cache[gp] = glb_json_bin(Path(gp).read_bytes())
        return glb_cache[gp]

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body_id in body_ids:
        verts_parts: list[np.ndarray] = []
        idx_parts: list[np.ndarray] = []
        offset = 0
        for gp, ni in index.get(str(body_id), []):
            J, BN = _load(gp)
            node = J["nodes"][ni]
            if "mesh" not in node:
                continue
            prim = J["meshes"][node["mesh"]]["primitives"][0]
            if prim.get("attributes", {}).get("POSITION") is None:
                continue
            world, _scale = decode_positions(J, BN, prim, node)
            tri = _glb_triangle_indices(J, BN, prim)
            verts_parts.append(world)
            if tri.shape[0]:
                idx_parts.append(tri.astype(np.int64) + offset)
            offset += world.shape[0]
        if verts_parts:
            all_pos = np.concatenate(verts_parts, axis=0).astype(np.float64)
            all_idx = (
                np.concatenate(idx_parts, axis=0).astype(np.uint32)
                if idx_parts
                else np.empty((0, 3), dtype=np.uint32)
            )
            out[str(body_id)] = (all_pos, all_idx)
    return out


def _ng_read_dracopy_loose(ng_dir: str, feature_id: int) -> np.ndarray:
    """Decode a single segment's FINEST-LOD vertices from the LOOSE NG output
    using DracoPy.

    The loose multilod layout writes, per feature: a data file named
    ``<feature_id>`` (all LOD fragments' Draco bytes concatenated) and a binary
    manifest ``<feature_id>.index``. The manifest byte layout (from
    streaming.rs) is little-endian::

        chunk_shape            3  x f32
        grid_origin            3  x f32
        num_lods               1  x u32
        lod_scales             L  x f32
        vertex_offsets         L*3 x f32      (C-order [L, 3])
        num_fragments_per_lod  L  x u32
        per LOD l:
            fragment_positions [3, nf_l] u32  (all x, then all y, then all z)
            fragment_offsets   [nf_l]    u32  (BYTE SIZE of each fragment)

    LOD 0 is the FINEST level (the writer walks zooms high->low, `lod =
    max_zoom - zoom`), and `all_draco_data` concatenates fragments LOD-major, so
    LOD 0's fragments occupy the first `sum(fragment_offsets[0])` bytes. This
    decodes each LOD-0 fragment with `DracoPy.decode` and concatenates the
    per-fragment vertices.

    Returns an Nx3 float64 array. Raises FileNotFoundError if the loose files are
    absent (e.g. sharded output) -- use the cloudvolume reader for sharded data.
    """
    import DracoPy  # lazy

    data_path = os.path.join(ng_dir, str(feature_id))
    index_path = os.path.join(ng_dir, f"{feature_id}.index")
    if not (os.path.exists(data_path) and os.path.exists(index_path)):
        raise FileNotFoundError(
            f"loose NG files for feature {feature_id} not found under {ng_dir} "
            f"(sharded output? use the cloudvolume reader)"
        )

    # vertex_quantization_bits comes from the mesh `info`; DracoPy returns the
    # quantized integers in [0, 2^bits - 1], and the world transform below
    # re-expands them (the job the Neuroglancer viewer normally does).
    qbits = 10
    try:
        info = json.loads((Path(ng_dir) / "info").read_text())
        qbits = int(info.get("vertex_quantization_bits", 10))
    except Exception:
        pass
    qmax = float((1 << qbits) - 1)

    manifest = Path(index_path).read_bytes()
    data = Path(data_path).read_bytes()

    # The manifest walk (all LODs' fragment tables, LOD-0 byte offsets into
    # `data`) is the pure `parse_ng_multilod_manifest`, shared with the HTTP
    # per-object fetch; only LOD 0 (finest) is decoded here.
    parsed = parse_ng_multilod_manifest(manifest)
    if parsed["num_lods"] == 0:
        return np.empty((0, 3), dtype=np.float64)
    chunk_shape = np.array(parsed["chunk_shape"], dtype=np.float64)
    grid_origin = np.array(parsed["grid_origin"], dtype=np.float64)
    lod_scale0 = float(parsed["lod_scales"][0])
    vertex_offset0 = np.array(parsed["vertex_offsets"][0], dtype=np.float64)
    fpos0 = np.array(parsed["fragment_positions"][0], dtype=np.float64).reshape(-1, 3)

    verts_parts: list[np.ndarray] = []
    for fi, (off_b, size) in enumerate(parsed["lod0_fragments"]):
        if size <= 0:
            continue
        mesh = DracoPy.decode(data[off_b:off_b + size])
        q = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
        # Neuroglancer multi-resolution dequant: re-expand the per-fragment
        # quantized integers to world coordinates.
        world = grid_origin + vertex_offset0 + chunk_shape * lod_scale0 * (fpos0[fi] + q / qmax)
        verts_parts.append(world)
    if not verts_parts:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(verts_parts, axis=0)


def _ng_read_cloudvolume(ng_dir: str, feature_id: int) -> np.ndarray:
    """Decode a single segment via cloudvolume (the "standard reader").

    cloudvolume is the reviewer-familiar Neuroglancer-precomputed reader and
    transparently handles BOTH the sharded and loose multilod Draco layouts plus
    the multi-fragment Draco merge. Returns Nx3 float64 vertices.

    NOTE: muDM writes a *bare mesh* directory
    (root `info` of @type `neuroglancer_multilod_draco`), not a full precomputed
    *volume*. The exact cloudvolume construction that binds a mesh source to a
    bare mesh dir may vary across cloudvolume releases; this attempts the precomputed
    file source and is wrapped by `decode_neuroglancer` so any failure falls back
    to the DracoPy loose reader.
    """
    from cloudvolume import CloudVolume  # lazy

    cv = CloudVolume(f"precomputed://file://{ng_dir}", mip=0, fill_missing=True)
    mesh = cv.mesh.get(int(feature_id), lod=0)
    # cv.mesh.get may return a Mesh or a {segid: Mesh} dict depending on version.
    if isinstance(mesh, dict):
        mesh = next(iter(mesh.values()))
    return np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)


def decode_neuroglancer(
    ng_dir: str,
    feature_ids: list[int],
    *,
    prefer_cloudvolume: bool = True,
) -> dict[int, np.ndarray]:
    """Decode the finest-LOD vertices of each feature from the NG multilod output.

    Prefers the cloudvolume "standard reader" (handles sharded + loose); on any
    failure per feature, falls back to the DracoPy loose reader. Returns
    `{feature_id: positions Nx3 f64}`. Draco/cloudvolume reorder and (for the
    multi-fragment merge) change vertex count, so fidelity here must use
    `set_l2_error`.
    """
    out: dict[int, np.ndarray] = {}
    cloudvolume_ok = prefer_cloudvolume
    for fid in feature_ids:
        verts: np.ndarray | None = None
        if cloudvolume_ok:
            try:
                verts = _ng_read_cloudvolume(ng_dir, fid)
            except Exception:
                # Disable further cloudvolume attempts once it fails to construct
                # (usually a layout/import issue, not per-feature), then fall back.
                cloudvolume_ok = False
        if verts is None:
            verts = _ng_read_dracopy_loose(ng_dir, fid)
        out[int(fid)] = verts
    return out


# ===========================================================================
# Integration: measurement drivers (decode timing, random access, fidelity)
# ===========================================================================


def measure_decode(
    decode_fn: Callable[[], dict],
    *,
    iters: int,
) -> dict:
    """Time a full-sample decode `iters` times and summarise.

    `decode_fn` decodes the whole sample and returns `{key: mesh}`. The peak
    ru_maxrss high-water is captured after the first (warm) decode as the
    decode-all peak. Returns median / p95 total seconds, per-neuron median, and
    the decoded meshes from the final iteration (for fidelity).
    """
    totals: list[float] = []
    decoded: dict = {}
    peak_after = None
    for i in range(max(1, iters)):
        decoded, t = timed(decode_fn)
        totals.append(t)
        if i == 0:
            peak_after = rss_platform()
    n = max(1, len(decoded))
    return {
        "iters": iters,
        "n_neurons": len(decoded),
        "median_total_s": _median(totals),
        "p95_total_s": _percentile(totals, 0.95),
        "median_per_neuron_s": _median(totals) / n,
        "peak_rss_after_decode": peak_after,
        "_decoded": decoded,  # consumed by the fidelity pass, stripped before JSON
    }


def measure_random_access(
    fetch_fn: Callable[[Any], Any],
    keys: list[Any],
    *,
    iters: int,
) -> dict:
    """Time fetching ONE neuron at a time, `iters` fetches per key, and summarise.

    `fetch_fn(key)` fetches+decodes a single neuron. Reports median and p95
    single-fetch latency across all (key, iter) samples, plus a COLD/WARM split:
    `cold_median_s` / `cold_p95_s` over the FIRST fetch of each key only (the
    page cache / CDN edge has not seen the object yet) and `warm_median_s` over
    iterations 2..n (None when iters == 1). With iters > 1 the all-sample
    median is dominated by warm repeats, so the cold numbers are the ones to
    quote for a first-touch access cost.
    """
    latencies: list[float] = []
    cold: list[float] = []
    warm: list[float] = []
    for key in keys:
        for i in range(max(1, iters)):
            _, t = timed(fetch_fn, key)
            latencies.append(t)
            (cold if i == 0 else warm).append(t)
    return {
        "n_keys": len(keys),
        "iters_per_key": iters,
        "n_samples": len(latencies),
        "median_s": _median(latencies),
        "p95_s": _percentile(latencies, 0.95),
        "cold_median_s": _median(cold),
        "cold_p95_s": _percentile(cold, 0.95),
        "warm_median_s": _median(warm) if warm else None,
    }


def compute_fidelity(
    source: dict[str, np.ndarray],
    decoded: dict[str, np.ndarray],
    *,
    order_preserving: bool,
) -> dict:
    """Aggregate per-neuron L2 fidelity of `decoded` against `source` vertices.

    `source[body_id]` and `decoded[body_id]` are Nx3 arrays. When
    `order_preserving` is True and the vertex counts match, the index-aligned
    `compute_l2_error` is reported; otherwise (meshopt permutation, Draco
    multi-fragment merge, or tiling reorder) the set-based `set_l2_error` is used.
    The chosen metric is recorded per neuron and the report notes whether every
    neuron used the same metric.
    """
    per_neuron: list[dict] = []
    metrics_used: set[str] = set()
    for body_id, dec in decoded.items():
        src = source.get(body_id)
        if src is None or len(src) == 0 or len(dec) == 0:
            continue
        if order_preserving and src.shape == dec.shape:
            m = compute_l2_error(src, dec)
        else:
            m = set_l2_error(src, dec)
        metrics_used.add(m["metric"])
        per_neuron.append({"body_id": body_id, **m})

    def _avg(key: str) -> float:
        vals = [d[key] for d in per_neuron if key in d]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "n_neurons": len(per_neuron),
        "metrics_used": sorted(metrics_used),
        "mean_l2": _avg("mean"),
        "max_l2": _avg("max"),
        "p95_l2": _avg("p95"),
        "rel_mean_l2": _avg("rel_mean"),
        "rel_max_l2": _avg("rel_max"),
    }


# ===========================================================================
# CLI
# ===========================================================================


def _sample_body_ids(obj_paths: list[str], n: int, seed: int = 0) -> list[str]:
    """Deterministically sample up to `n` body_ids (OBJ stems)."""
    body_ids = [os.path.splitext(os.path.basename(p))[0] for p in obj_paths]
    if n <= 0 or n >= len(body_ids):
        return body_ids
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(body_ids), size=n, replace=False)
    return [body_ids[i] for i in sorted(idx)]


def run_spec_sheet(
    obj_dir: str,
    out_json: str,
    *,
    n: int = 200,
    max_zoom: int = 4,
    iters: int = 10,
    sharded: bool = True,
    out_root: str | None = None,
    hardware: dict | None = None,
) -> dict:
    """Full spec-sheet run: encode all formats, then decode / random-access /
    fidelity a sample, and assemble a schema-valid report.

    This is the integration entry point run on the benchmark host; it
    touches the pipeline and every decoder. `out_root` (encode outputs) defaults
    to a `formats_out/` directory beside `out_json`.
    """
    obj_paths = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
    if not obj_paths:
        raise FileNotFoundError(f"no .obj files under {obj_dir}")

    if out_root is None:
        out_root = os.path.join(os.path.dirname(os.path.abspath(out_json)), "formats_out")
    os.makedirs(out_root, exist_ok=True)

    # --- encode all three formats from one shared index ---
    enc = encode_all(
        obj_paths,
        out_root,
        max_zoom=max_zoom,
        sharded=sharded,
    )
    dirs = enc["dirs"]

    # --- resolve the sample + body_id <-> feature_id ---
    sample_bodies = _sample_body_ids(obj_paths, n)
    body_to_feature = build_body_feature_map(dirs["parquet"])
    sample_features = [body_to_feature[b] for b in sample_bodies if b in body_to_feature]
    sample_paths = [os.path.join(obj_dir, f"{b}.obj") for b in sample_bodies]

    # --- source geometry (baseline decode + fidelity reference) ---
    source_meshes = decode_obj(sample_paths)
    source_verts = {b: v for b, (v, _f) in source_meshes.items()}

    # --- decode timing (median over iters), per format ---
    decode: dict[str, Any] = {}

    decode["obj"] = measure_decode(lambda: decode_obj(sample_paths), iters=iters)
    decode["parquet"] = measure_decode(
        lambda: decode_parquet(dirs["parquet"], sample_features, zoom=max_zoom),
        iters=iters,
    )
    t3d_index = _build_tiles3d_index(dirs["tiles3d"], max_zoom)
    decode["tiles3d"] = measure_decode(
        lambda: decode_tiles3d_meshopt(
            dirs["tiles3d"], sample_bodies, max_zoom=max_zoom, index=t3d_index
        ),
        iters=iters,
    )
    decode["neuroglancer"] = measure_decode(
        lambda: decode_neuroglancer(dirs["neuroglancer"], sample_features),
        iters=iters,
    )

    # --- fidelity vs source OBJ (pop the decoded meshes off the decode blocks) ---
    parquet_decoded = {
        b: decode["parquet"]["_decoded"][body_to_feature[b]][0]  # [0] = positions (decode_parquet returns (pos, idx))
        for b in sample_bodies
        if body_to_feature.get(b) in decode["parquet"]["_decoded"]
    }
    ng_decoded = {
        b: decode["neuroglancer"]["_decoded"].get(body_to_feature.get(b))
        for b in sample_bodies
        if body_to_feature.get(b) in decode["neuroglancer"]["_decoded"]
    }
    t3d_decoded = {b: m[0] for b, m in decode["tiles3d"]["_decoded"].items()}

    fidelity = {
        # parquet f32 is lossless but tiling concatenation reorders vs the OBJ,
        # so it is compared set-based (expected ~0). tiles3d/NG are quantized +
        # reordered -> set-based by construction.
        "parquet": compute_fidelity(source_verts, parquet_decoded, order_preserving=False),
        "tiles3d": compute_fidelity(source_verts, t3d_decoded, order_preserving=False),
        "neuroglancer": compute_fidelity(source_verts, ng_decoded, order_preserving=False),
    }

    # --- random-access latency: fetch ONE neuron by body_id ---
    ra_bodies = sample_bodies[: min(len(sample_bodies), max(1, n // 4 or 1))]
    ra_features = [body_to_feature[b] for b in ra_bodies if b in body_to_feature]
    random_access = {
        "parquet": measure_random_access(
            lambda fid: decode_parquet(dirs["parquet"], [fid], zoom=max_zoom),
            ra_features,
            iters=iters,
        ),
        "tiles3d": {
            # 3D Tiles has no native feature index; building one is the penalty.
            "index_build_s": timed(_build_tiles3d_index, dirs["tiles3d"], max_zoom)[1],
            **measure_random_access(
                lambda b: decode_tiles3d_meshopt(
                    dirs["tiles3d"], [b], max_zoom=max_zoom, index=t3d_index
                ),
                ra_bodies,
                iters=iters,
            ),
        },
        "neuroglancer": measure_random_access(
            lambda fid: decode_neuroglancer(dirs["neuroglancer"], [fid]),
            ra_features,
            iters=iters,
        ),
    }

    # --- strip the heavy decoded-mesh payloads before serialising ---
    for fmt in decode.values():
        fmt.pop("_decoded", None)

    results = {
        "n_neurons_total": enc["n_neurons"],
        "n_sampled": len(sample_bodies),
        "n_sampled_features": len(sample_features),
        "shared_index_build_s": enc["shared_index_build_s"],
        "encode": enc["encode"],
        "decode": decode,
        "random_access": random_access,
        "fidelity": fidelity,
    }
    config = {
        "obj_dir": obj_dir,
        "n": n,
        "max_zoom": max_zoom,
        "iters": iters,
        "sharded": sharded,
        "out_root": out_root,
    }
    benchmark_id = Path(out_json).stem
    report = assemble_report(
        benchmark_id, results, config=config, hardware=hardware
    )

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    return report


# ===========================================================================
# Access-pattern measures (bulk / spatial-window / by-id / by-id-sorted).
# Run against an ALREADY-ENCODED out_root -- no re-encode.
# ===========================================================================


def decode_parquet_bulk(parquet_dir: str, zoom: int) -> dict[int, np.ndarray]:
    """Read the ENTIRE level partition in ONE columnar scan and reconstruct every
    feature's positions -- the bulk / ML-training access pattern (no per-feature
    filter, so it is a single sequential read of the whole level)."""
    import pyarrow as pa  # lazy
    import pyarrow.dataset as ds  # lazy

    dataset = ds.dataset(
        parquet_dir, partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())]))
    )
    return bulk_positions(dataset, zoom)


def write_feature_sorted_parquet(
    src_parquet_dir: str, dst_dir: str, zoom: int, row_group_size: int = 2000
) -> tuple[str, float]:
    """Re-lay-out one level partition SORTED by feature_id, with small row groups.

    A neuron's rows become contiguous and each row group carries a tight
    feature_id min/max, so a by-id predicate skips all non-matching row groups
    instead of scanning the level. Returns (output_path, relayout_seconds). This
    trades some spatial-tile locality for id-locality -- a tunable layout choice,
    not a free fix (the point is the columnar store SUPPORTS either)."""
    import pyarrow as pa  # lazy
    import pyarrow.dataset as ds  # lazy
    import pyarrow.parquet as pq  # lazy

    def _do() -> str:
        dataset = ds.dataset(
            src_parquet_dir,
            partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())])),
        )
        table = dataset.to_table(filter=ds.field("zoom") == int(zoom)).sort_by("feature_id")
        os.makedirs(dst_dir, exist_ok=True)
        out = os.path.join(dst_dir, f"zoom{zoom}_by_feature.parquet")
        pq.write_table(
            table, out, row_group_size=row_group_size, compression="zstd", compression_level=3
        )
        return out

    return timed(_do)


def _parquet_count_by_id(dataset, fid: int) -> int:
    """Fetch one feature's positions by feature_id; returns row count (forces read)."""
    import pyarrow.dataset as ds  # lazy

    return dataset.to_table(
        columns=["feature_id", "positions"], filter=ds.field("feature_id") == int(fid)
    ).num_rows


def _io_rchar() -> int | None:
    """Bytes read by this process so far (Linux /proc/self/io rchar; None elsewhere).

    rchar counts all read() traffic including page-cache hits, so deltas are an
    upper-bound proxy for bytes touched per access pattern, comparable across
    arms measured in the same process.
    """
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("rchar:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def join_records(
    fids: list[int], feature_to_body: dict[int, str], records: dict[str, Any]
) -> int:
    """Join returned feature_ids to an external per-body record table.

    Mirrors the fusion experiment's data path (run_manc --extra-features):
    features come back from a query, each is joined to its linked record by
    body id. Returns the number of features with a matching record.
    """
    joined = 0
    for fid in set(fids):
        body = feature_to_body.get(fid)
        if body is not None and body in records:
            joined += 1
    return joined


def run_access_patterns(
    out_root: str,
    obj_dir: str,
    out_json: str,
    *,
    zoom: int = 4,
    bulk_n: int = 0,
    n_ids: int = 200,
    iters: int = 10,
    records_json: str | None = None,
    join_n: int = 200,
    hardware: dict | None = None,
) -> dict:
    """Measure the access patterns on the ALREADY-ENCODED formats under out_root.

    1. bulk read a whole level (parquet: one scan of all n_total; per-item formats
       measured on the first `bulk_n` neurons -> per-neuron rate -> extrapolated to
       the whole level, keeping the comparison fair and the run bounded).
    2. spatial-window read (parquet tile_x/tile_y predicate pushdown vs its own
       whole-level scan).
    3. by-id single fetch on the default tile-ordered layout (a scan).
    4. by-id single fetch on a feature_id-sorted re-layout (row-group skipping),
       with the one-time relayout cost reported.
    5. (with records_json) feature->record join: read a spatial window's geometry
       and join each feature to its linked per-body record (the fusion
       experiment's access pattern), against a per-feature render-path fetch of
       the same window (bounded at join_n, extrapolated).
    Neuroglancer single-fetch is reported alongside as the per-neuron-file
    contrast. Where /proc/self/io is available, per-arm rchar deltas (bytes
    read incl. cache) are reported as bytes_read_* keys.
    """
    import pyarrow as pa  # lazy
    import pyarrow.dataset as ds  # lazy

    dirs = {f: os.path.join(out_root, f) for f in ("parquet", "tiles3d", "neuroglancer")}
    obj_by_body = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(obj_dir, "*.obj"))
    }
    body_to_feature = build_body_feature_map(dirs["parquet"])
    all_bodies = [b for b in body_to_feature if b in obj_by_body]
    all_features = [body_to_feature[b] for b in all_bodies]
    N = len(all_features)
    results: dict[str, Any] = {"zoom": zoom, "n_neurons_total": N}

    # --- 1. BULK read whole level ---
    bn = min(bulk_n if bulk_n > 0 else N, N)
    io0 = _io_rchar()
    _, t_pq = timed(decode_parquet_bulk, dirs["parquet"], zoom)  # whole level, all N
    io1 = _io_rchar()
    bulk_bytes = (io1 - io0) if (io0 is not None and io1 is not None) else None
    sub_bodies = all_bodies[:bn]
    sub_features = [body_to_feature[b] for b in sub_bodies]
    sub_paths = [obj_by_body[b] for b in sub_bodies]
    _, t_ng = timed(decode_neuroglancer, dirs["neuroglancer"], sub_features)
    _, t_t3 = timed(decode_tiles3d_meshopt, dirs["tiles3d"], sub_bodies, max_zoom=zoom)
    _, t_obj = timed(decode_obj, sub_paths)
    results["bulk_read"] = {
        "n_total": N,
        "bulk_n_measured": bn,
        "parquet_whole_level_s": t_pq,
        "parquet_per_neuron_ms": t_pq / max(N, 1) * 1000,
        "neuroglancer_per_neuron_ms": t_ng / max(bn, 1) * 1000,
        "neuroglancer_whole_level_extrap_s": t_ng / max(bn, 1) * N,
        "tiles3d_per_neuron_ms": t_t3 / max(bn, 1) * 1000,
        "tiles3d_whole_level_extrap_s": t_t3 / max(bn, 1) * N,
        "obj_per_neuron_ms": t_obj / max(bn, 1) * 1000,
        "obj_whole_level_extrap_s": t_obj / max(bn, 1) * N,
    }
    if bulk_bytes is not None:
        results["bulk_read"]["bytes_read_parquet_whole_level"] = bulk_bytes

    # --- 2. SPATIAL WINDOW (parquet tile pushdown vs whole-level scan) ---
    dataset = ds.dataset(
        dirs["parquet"], partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())]))
    )
    tt = dataset.to_table(columns=["tile_x", "tile_y"], filter=ds.field("zoom") == int(zoom))
    txy = sorted(set(zip(tt.column("tile_x").to_pylist(), tt.column("tile_y").to_pylist())))
    mx, my = txy[len(txy) // 2]

    def _win() -> int:
        return dataset.to_table(
            columns=["feature_id", "positions"],
            filter=(ds.field("zoom") == int(zoom))
            & (ds.field("tile_x") >= mx) & (ds.field("tile_x") <= mx + 2)
            & (ds.field("tile_y") >= my) & (ds.field("tile_y") <= my + 2),
        ).num_rows

    io0 = _io_rchar()
    nrows, t_win = timed(_win)
    io1 = _io_rchar()
    results["spatial_window"] = {
        "window": "3x3 tiles",
        "rows_returned": nrows,
        "parquet_window_s": t_win,
        "parquet_whole_level_s": t_pq,
        "speedup_vs_whole_level": (t_pq / t_win) if t_win > 0 else 0.0,
    }
    if io0 is not None and io1 is not None:
        results["spatial_window"]["bytes_read_parquet_window"] = io1 - io0

    # --- 3 + 4. BY-ID single fetch: default (scan) vs feature-sorted (row-group skip) ---
    sample_features = all_features[:n_ids]
    results["by_id_default"] = measure_random_access(
        lambda fid: _parquet_count_by_id(dataset, fid), sample_features, iters=iters
    )
    sorted_path, relayout_s = write_feature_sorted_parquet(
        dirs["parquet"], os.path.join(out_root, "parquet_by_feature"), zoom
    )
    sdataset = ds.dataset(sorted_path)
    ra_sorted = measure_random_access(
        lambda fid: _parquet_count_by_id(sdataset, fid), sample_features, iters=iters
    )
    ra_sorted["relayout_s"] = relayout_s
    ra_sorted["relayout_bytes"] = (
        dir_size(sorted_path) if os.path.isdir(sorted_path) else os.path.getsize(sorted_path)
    )
    results["by_id_feature_sorted"] = ra_sorted
    results["by_id_neuroglancer"] = measure_random_access(
        lambda fid: decode_neuroglancer(dirs["neuroglancer"], [fid]), sample_features, iters=iters
    )

    # --- 5. FEATURE -> RECORD JOIN (the fusion experiment's access pattern) ---
    if records_json:
        with open(records_json) as f:
            records = json.load(f)
        feature_to_body = {v: k for k, v in body_to_feature.items()}

        win_filter = (
            (ds.field("zoom") == int(zoom))
            & (ds.field("tile_x") >= mx) & (ds.field("tile_x") <= mx + 2)
            & (ds.field("tile_y") >= my) & (ds.field("tile_y") <= my + 2)
        )

        # muDM path: ONE pushdown read of the window's geometry, then a dict
        # join of every returned feature to its record by body id.
        def _mudm_join() -> tuple[int, int]:
            t = dataset.to_table(columns=["feature_id", "positions"], filter=win_filter)
            fids = t.column("feature_id").to_pylist()
            return join_records(fids, feature_to_body, records), len(set(fids))

        io0 = _io_rchar()
        (n_joined, n_win_features), t_join_pq = timed(_mudm_join)
        io1 = _io_rchar()

        # Render-path baseline: the SAME window's features fetched one file at
        # a time from Neuroglancer, then the same dict join. The baseline gets
        # the window's feature list for free (in its favour); bounded at
        # join_n and extrapolated, like the bulk arm.
        win_ids = sorted(
            {f for f in dataset.to_table(columns=["feature_id"], filter=win_filter)
             .column("feature_id").to_pylist()}
        )
        sample_ids = win_ids[: min(join_n, len(win_ids))]

        def _ng_join() -> int:
            for fid in sample_ids:
                decode_neuroglancer(dirs["neuroglancer"], [fid])
            return join_records(sample_ids, feature_to_body, records)

        io2 = _io_rchar()
        _, t_join_ng = timed(_ng_join)
        io3 = _io_rchar()

        results["feature_record_join"] = {
            "window": "3x3 tiles",
            "records_file": os.path.basename(records_json),
            "n_records": len(records),
            "n_window_features": n_win_features,
            "n_joined": n_joined,
            "parquet_join_s": t_join_pq,
            "ng_sample_n": len(sample_ids),
            "ng_fetch_join_s": t_join_ng,
            "ng_fetch_join_extrap_s": (
                t_join_ng / max(len(sample_ids), 1) * max(n_win_features, 1)
            ),
        }
        if io0 is not None and io1 is not None:
            results["feature_record_join"]["bytes_read_parquet_join"] = io1 - io0
        if io2 is not None and io3 is not None:
            results["feature_record_join"]["bytes_read_ng_sample"] = io3 - io2

    results["peak_rss_bytes"] = peak_rss_bytes()

    report = assemble_report(
        Path(out_json).stem,
        results,
        config={
            "out_root": out_root, "zoom": zoom, "bulk_n": bulk_n, "n_ids": n_ids,
            "iters": iters, "records_json": records_json, "join_n": join_n,
        },
        hardware=hardware,
    )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    return report



# ===========================================================================
# HTTP / object-storage access condition (serverless range reads)
# ===========================================================================
#
# The paper's deployment story is static object storage (S3 behind CloudFront):
#
#     https://ngrf.mudm.io/tiles/<id>/features.parquet/zoom=<Z>/part_NNN.mu.parquet
#     https://ngrf.mudm.io/tiles/<id>/tileset.json  (+ 3dtiles/{z}/{x}/{y}/{d}.glb)
#     https://ngrf.mudm.io/tiles/<id>/neuroglancer/info (+ per-segment objects)
#
# Parquet parts are served identity-encoded with `Accept-Ranges: bytes`; JSON
# (tilejson, tileset.json, NG `info`) is served brotli-encoded (`Content-Encoding:
# br`) regardless of `Accept-Encoding`; a missing key answers 403 (S3 without
# ListBucket), not 404. `run_access_http` re-runs the SAME Parquet access
# patterns as `run_access_patterns` (bulk whole level, 3x3 spatial window, by-id
# single fetch) through pyarrow.dataset over an fsspec HTTP filesystem, so that
# predicate pushdown turns into HTTP Range reads of only the surviving row
# groups, and adds per-object Neuroglancer / 3D Tiles GETs as the render-path
# contrast.
#
# Request / byte accounting. Every HTTP call fsspec makes (HEAD for size, Range
# GET for a block, full GET for a small object) goes through ONE aiohttp
# ClientSession, and an aiohttp TraceConfig on that session tallies:
#     http_requests      on_request_start, per request (HEAD + GET)
#     bytes_transferred  WIRE bytes: the response Content-Length when the server
#                        sends one (206 range length; brotli size for br JSON),
#                        else the decoded body length; HEAD never counts
#     bytes_body         decoded body bytes handed to the reader
# Bytes are charged when the body is actually read (on_response_chunk_received,
# which aiohttp fires once per `read()`), so an aborted/unread GET costs a
# request but no bytes. Counters are thread-safe (pyarrow scans fragments from
# its IO thread pool). Nothing here is imported at module load: fsspec/aiohttp/
# pyarrow are imported lazily and the counting filesystem class is built on
# first use (`bench_formats.CountingHTTPFileSystem` resolves via __getattr__).

NG_SHARDED_NOTE = "sharded store, per-object lookup not implemented"
_DEFAULT_PART_PATTERN = "part_{:03d}.mu.parquet"
_DISCOVERY_CAP = 4096
_PROBE_PATHS = (
    "features.parquet/zoom=0/part_000.mu.parquet",
    "neuroglancer/info",
    "tileset.json",
    "3dtiles/tileset.json",
)


class HttpCounters:
    """Thread-safe request / byte tallies for one HTTP session (see the
    accounting note at the top of this section). Pure: stdlib only."""

    _COUNTER_KEYS = (
        "http_requests", "head_requests", "get_requests", "bytes_transferred", "bytes_body",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.http_requests = 0
        self.head_requests = 0
        self.get_requests = 0
        self.bytes_transferred = 0
        self.bytes_body = 0
        self.status_counts: dict[str, int] = {}

    @staticmethod
    def new_ctx() -> SimpleNamespace:
        """Per-request context carried from the response headers to the body read."""
        return SimpleNamespace(method=None, status=None, content_length=None)

    def _on_start(self, method: str) -> None:
        with self._lock:
            self.http_requests += 1
            if str(method).upper() == "HEAD":
                self.head_requests += 1
            else:
                self.get_requests += 1

    def _on_end(self, method: str, status: int, headers: Any, *, ctx: SimpleNamespace) -> None:
        ctx.method = str(method).upper()
        ctx.status = int(status)
        cl = headers.get("Content-Length") if headers is not None else None
        try:
            ctx.content_length = int(cl) if cl is not None else None
        except (TypeError, ValueError):
            ctx.content_length = None
        with self._lock:
            self.status_counts[str(status)] = self.status_counts.get(str(status), 0) + 1

    def _on_chunk(self, chunk: bytes, *, ctx: SimpleNamespace) -> None:
        if getattr(ctx, "method", None) == "HEAD":
            return
        n = len(chunk)
        wire = ctx.content_length if getattr(ctx, "content_length", None) is not None else n
        with self._lock:
            self.bytes_body += n
            self.bytes_transferred += int(wire)

    def snapshot(self) -> dict:
        with self._lock:
            out = {k: int(getattr(self, k)) for k in self._COUNTER_KEYS}
            out["status_counts"] = dict(self.status_counts)
        return out

    @staticmethod
    def delta(before: dict, after: dict) -> dict:
        return {k: after[k] - before[k] for k in HttpCounters._COUNTER_KEYS}

    def measure(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float, dict]:
        """`(result, seconds, counter_delta)` around one call."""
        before = self.snapshot()
        result, seconds = timed(fn, *args, **kwargs)
        return result, seconds, self.delta(before, self.snapshot())


def _make_trace_config(counters: HttpCounters):
    """aiohttp TraceConfig wiring the session's request lifecycle to `counters`."""
    import aiohttp  # lazy

    tc = aiohttp.TraceConfig(
        trace_config_ctx_factory=lambda trace_request_ctx=None: HttpCounters.new_ctx()
    )

    async def on_start(_session, _ctx, params):
        counters._on_start(params.method)

    async def on_end(_session, ctx, params):
        counters._on_end(params.method, params.response.status, params.response.headers, ctx=ctx)

    async def on_chunk(_session, ctx, params):
        counters._on_chunk(params.chunk, ctx=ctx)

    tc.on_request_start.append(on_start)
    tc.on_request_end.append(on_end)
    tc.on_response_chunk_received.append(on_chunk)
    return tc


_counting_fs_cls: type | None = None


def _counting_fs_class() -> type:
    """Build (once) and return the CountingHTTPFileSystem class; lazy so the
    module stays importable without fsspec/aiohttp."""
    global _counting_fs_cls
    if _counting_fs_cls is not None:
        return _counting_fs_cls

    import aiohttp  # lazy
    from fsspec.implementations.http import HTTPFileSystem  # lazy

    class CountingHTTPFileSystem(HTTPFileSystem):
        """fsspec HTTP filesystem whose every request is tallied in `.counters`.

        Differences from the stock HTTPFileSystem, all in service of exact,
        like-for-like accounting against object storage:

        * ``cache_type="none"`` by default: each read maps to exactly ONE Range
          GET of the requested bytes (the stock 5 MB read-ahead would drag most
          of a Parquet part along with every footer / column-chunk read);
        * ``info`` is HEAD-only (no GET fallback) and memoised per URL, so a
          Parquet part costs one HEAD plus its Range reads (pyarrow's
          FSSpecHandler calls ``info`` twice per file) and a missing part costs
          one HEAD; 403 and 404 both map to FileNotFoundError (S3 answers 403
          for missing keys when ListBucket is not granted);
        * ``exists`` / ``isfile`` go through that memoised HEAD instead of
          fsspec's un-read full GET.
        """

        cachable = False  # fsspec caches instances by init args; never share counters

        def __init__(self, counters: HttpCounters | None = None, *, timeout_s: float = 120.0, **kwargs):
            self.counters = counters if counters is not None else HttpCounters()
            client_kwargs = dict(kwargs.pop("client_kwargs", None) or {})
            client_kwargs.setdefault("timeout", aiohttp.ClientTimeout(total=timeout_s))
            client_kwargs["trace_configs"] = list(client_kwargs.get("trace_configs") or []) + [
                _make_trace_config(self.counters)
            ]
            kwargs.setdefault("cache_type", "none")
            super().__init__(client_kwargs=client_kwargs, **kwargs)
            self._info_memo: dict[str, dict] = {}

        async def _head_info(self, url, **kwargs):
            session = await self.set_session()
            kw = dict(self.kwargs)
            kw.update(kwargs)
            headers = dict(kw.pop("headers", None) or {})
            headers["Accept-Encoding"] = "identity"
            r = await session.head(self.encode_url(url), allow_redirects=True, headers=headers, **kw)
            async with r:
                r.raise_for_status()
                enc = r.headers.get("Content-Encoding")
                info: dict[str, Any] = {"url": str(r.url), "content_encoding": enc}
                cl = r.headers.get("Content-Length")
                if cl is not None and (not enc or enc == "identity"):
                    info["size"] = int(cl)
                info["mimetype"] = r.headers.get("Content-Type", "").partition(";")[0]
                if r.headers.get("Accept-Ranges") == "none":
                    info["partial"] = False
                for h in ("ETag", "Last-Modified"):
                    if r.headers.get(h):
                        info[h] = r.headers[h]
            return info

        async def _info(self, url, **kwargs):
            memo = self._info_memo.get(url)
            if memo is not None:
                return dict(memo)
            try:
                info = await self._head_info(url, **kwargs)
            except aiohttp.ClientResponseError as exc:
                if exc.status in (403, 404):
                    raise FileNotFoundError(url) from exc
                raise
            out = {"name": url, "size": None, **info, "type": "file"}
            self._info_memo[url] = out
            return dict(out)

        async def _exists(self, path, **kwargs):
            try:
                await self._info(path, **kwargs)
                return True
            except FileNotFoundError:
                return False

        async def _isfile(self, path, **kwargs):
            return await self._exists(path, **kwargs)

    _counting_fs_cls = CountingHTTPFileSystem
    return _counting_fs_cls


def __getattr__(name: str):
    if name == "CountingHTTPFileSystem":
        return _counting_fs_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _part_url(base_url: str, zoom: int, name: str) -> str:
    if name.startswith(("http://", "https://")):
        return name
    return f"{base_url.rstrip('/')}/features.parquet/zoom={int(zoom)}/{name}"


def discover_parts(
    base_url: str,
    zooms: list[int],
    fs,
    *,
    part_pattern: str = _DEFAULT_PART_PATTERN,
    cap: int = _DISCOVERY_CAP,
) -> dict[int, list[str]]:
    """Find how many `<base>/features.parquet/zoom=<Z>/part_NNN` objects exist
    per zoom with HEAD probes (403 / 404 = absent), capped at `cap` per zoom.

    Object stores cannot list, so this is the only dataset-agnostic discovery.
    The writer numbers parts contiguously from 000, so existence is monotone in
    the index and the count is found by GALLOPING (0, 1, 2, 4, 8, ...) then
    binary search: O(log n) HEADs instead of n + 1 (hemibrain's finest level
    has thousands of parts; a linear probe took minutes at ~65 ms per HEAD).
    Returns ``{zoom: [part names]}``; a level with `cap` parts is reported at
    the cap (raise `cap` if the last probe exists)."""

    def _exists(z: int, i: int) -> bool:
        try:
            fs.info(_part_url(base_url, z, part_pattern.format(i)))
            return True
        except FileNotFoundError:
            return False

    parts: dict[int, list[str]] = {}
    for z in zooms:
        n = 0
        if cap > 0 and _exists(z, 0):
            lo, hi = 0, 1  # part[lo] exists; find the first missing index hi
            while hi < cap and _exists(z, hi):
                lo, hi = hi, hi * 2
            hi = min(hi, cap)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _exists(z, mid):
                    lo = mid
                else:
                    hi = mid
            n = hi
        parts[int(z)] = [part_pattern.format(i) for i in range(n)]
    return parts


def prefetch_info(fs, urls: list[str], *, concurrency: int = 32) -> int:
    """HEAD many URLs concurrently on the filesystem's event loop so that the
    memoised `info` (and therefore pyarrow's per-path `get_file_info` / `open`)
    is served from memory. Missing URLs are ignored. Returns the URL count."""
    import asyncio  # lazy (stdlib, but only needed here)
    from fsspec.asyn import sync  # lazy

    async def _run() -> None:
        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def one(u: str) -> None:
            async with sem:
                try:
                    await fs._info(u)
                except FileNotFoundError:
                    pass

        await asyncio.gather(*(one(u) for u in urls))

    sync(fs.loop, _run)
    return len(urls)


def http_parquet_dataset(base_url: str, parts: dict[int, list[str]], fs):
    """Open the listed parts as ONE pyarrow dataset over HTTP, Hive-partitioned
    on `zoom` exactly like the local reader, so the same filters apply.

    The fsspec filesystem is wrapped in `pyarrow.fs.PyFileSystem(FSSpecHandler)`;
    pyarrow then reads the footer and the surviving row groups' column chunks
    through `seek`+`read` on an fsspec HTTPFile, i.e. HTTP Range requests.
    Opening the dataset inspects the first file's schema (footer reads) and
    HEADs every listed part (prefetched concurrently via `prefetch_info`)."""
    import pyarrow as pa  # lazy
    import pyarrow.dataset as ds  # lazy
    import pyarrow.fs as pafs  # lazy

    paths = [_part_url(base_url, z, n) for z, names in sorted(parts.items()) for n in names]
    if not paths:
        raise FileNotFoundError(f"no parquet parts listed under {base_url}")
    # pyarrow HEADs every path sequentially at construction; warm the memo in parallel.
    prefetch_info(fs, paths)
    pyfs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
    return ds.dataset(
        paths,
        filesystem=pyfs,
        format="parquet",
        partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())])),
        partition_base_dir=f"{base_url.rstrip('/')}/features.parquet",
    )


def http_scan_options(
    *,
    pre_buffer: bool = True,
    hole_size_limit: int | None = None,
    range_size_limit: int | None = None,
):
    """Parquet scan options for the HTTP condition. `pre_buffer=True` (pyarrow's
    default) coalesces the column-chunk reads of one row group into few Range
    requests; `hole_size_limit` (bytes; pyarrow default 8 KiB) is the largest
    gap that is bridged by reading THROUGH it -- set 0 to never fetch bytes a
    filter has already pruned (only contiguous ranges are merged)."""
    import pyarrow as pa  # lazy
    import pyarrow.dataset as ds  # lazy

    kw: dict[str, Any] = {"pre_buffer": bool(pre_buffer)}
    if hole_size_limit is not None or range_size_limit is not None:
        ckw: dict[str, int] = {}
        if hole_size_limit is not None:
            ckw["hole_size_limit"] = int(hole_size_limit)
        if range_size_limit is not None:
            ckw["range_size_limit"] = int(range_size_limit)
        kw["cache_options"] = pa.CacheOptions(**ckw)
    return ds.ParquetFragmentScanOptions(**kw)


def _positions_by_feature(table) -> dict[int, np.ndarray]:
    """`{feature_id: positions Nx3 f32}` from a (feature_id, positions) table,
    concatenating a feature's tiles in scan order."""
    accum: dict[int, list] = {}
    for fid, pb in zip(
        table.column("feature_id").to_pylist(), table.column("positions").to_pylist()
    ):
        accum.setdefault(int(fid), []).append(np.frombuffer(pb, "<f4").reshape(-1, 3))
    return {fid: np.concatenate(parts, axis=0) for fid, parts in accum.items()}


def _scan_kwargs(scan_options, fragment_readahead: int | None) -> dict:
    """`to_table` kwargs shared by the readers: parquet scan options plus the
    number of fragments (parts) opened concurrently (pyarrow default 4; raise
    it over HTTP so footer reads of many parts overlap the request latency)."""
    kw: dict[str, Any] = {"fragment_scan_options": scan_options}
    if fragment_readahead is not None:
        kw["fragment_readahead"] = int(fragment_readahead)
    return kw


def bulk_positions(
    dataset, zoom: int, *, scan_options=None, fragment_readahead: int | None = None
) -> dict[int, np.ndarray]:
    """Whole-level read of (feature_id, positions) on any pyarrow dataset (local
    or HTTP) -- the bulk / ML-training access pattern of `decode_parquet_bulk`."""
    import pyarrow.dataset as ds  # lazy

    table = dataset.to_table(
        columns=["feature_id", "positions"],
        filter=ds.field("zoom") == int(zoom),
        **_scan_kwargs(scan_options, fragment_readahead),
    )
    return _positions_by_feature(table)


def pick_window_origin(
    dataset, zoom: int, *, scan_options=None, fragment_readahead: int | None = None
) -> tuple[int, int]:
    """The median occupied (tile_x, tile_y) of a level -- the 3x3 window origin
    used by both the local and the HTTP spatial-window arm (deterministic)."""
    import pyarrow.dataset as ds  # lazy

    tt = dataset.to_table(
        columns=["tile_x", "tile_y"],
        filter=ds.field("zoom") == int(zoom),
        **_scan_kwargs(scan_options, fragment_readahead),
    )
    txy = sorted(set(zip(tt.column("tile_x").to_pylist(), tt.column("tile_y").to_pylist())))
    if not txy:
        raise ValueError(f"zoom={zoom} has no tiles")
    mx, my = txy[len(txy) // 2]
    return int(mx), int(my)


def _window_filter(zoom: int, mx: int, my: int):
    import pyarrow.dataset as ds  # lazy

    return (
        (ds.field("zoom") == int(zoom))
        & (ds.field("tile_x") >= mx) & (ds.field("tile_x") <= mx + 2)
        & (ds.field("tile_y") >= my) & (ds.field("tile_y") <= my + 2)
    )


def window_row_count(
    dataset, zoom: int, mx: int, my: int, *, scan_options=None, fragment_readahead: int | None = None
) -> int:
    """Rows of (feature_id, positions) inside the 3x3-tile window at (mx, my)
    via tile_x/tile_y predicate pushdown (row-group statistics pruning)."""
    return dataset.to_table(
        columns=["feature_id", "positions"],
        filter=_window_filter(zoom, mx, my),
        **_scan_kwargs(scan_options, fragment_readahead),
    ).num_rows


def _parquet_count_by_id_scoped(
    dataset, fid: int, *, zoom: int | None, scan_options=None, fragment_readahead: int | None = None
) -> int:
    """`_parquet_count_by_id` with an optional zoom scope (HTTP arm default:
    one level, see `run_access_http`)."""
    import pyarrow.dataset as ds  # lazy

    flt = ds.field("feature_id") == int(fid)
    if zoom is not None:
        flt = (ds.field("zoom") == int(zoom)) & flt
    return dataset.to_table(
        columns=["feature_id", "positions"], filter=flt,
        **_scan_kwargs(scan_options, fragment_readahead),
    ).num_rows


# --- Neuroglancer per-object comparator over HTTP ---------------------------


def parse_ng_multilod_manifest(manifest: bytes) -> dict:
    """Parse a multilod-draco loose manifest (`<feature_id>.index`, byte layout
    documented on `_ng_read_dracopy_loose`). Pure (struct only).

    Returns chunk_shape / grid_origin / lod_scales / vertex_offsets, the
    per-LOD fragment positions and sizes, and -- what a per-object HTTP fetch
    needs -- `lod0_fragments` as `(byte_offset, size)` pairs into the data
    file (LOD 0 is the finest level and is written FIRST, so its fragments are
    one contiguous prefix of `lod0_bytes` bytes) plus `total_bytes`."""
    off = 0
    chunk_shape = list(struct.unpack_from("<3f", manifest, off))
    off += 12
    grid_origin = list(struct.unpack_from("<3f", manifest, off))
    off += 12
    (num_lods,) = struct.unpack_from("<I", manifest, off)
    off += 4
    lod_scales = list(struct.unpack_from(f"<{num_lods}f", manifest, off))
    off += num_lods * 4
    flat = struct.unpack_from(f"<{num_lods * 3}f", manifest, off)
    vertex_offsets = [list(flat[3 * i : 3 * i + 3]) for i in range(num_lods)]
    off += num_lods * 3 * 4
    num_frags = list(struct.unpack_from(f"<{num_lods}I", manifest, off))
    off += num_lods * 4

    positions: list[list[list[int]]] = []
    sizes: list[list[int]] = []
    lod0: list[tuple[int, int]] = []
    cursor = 0
    for lod in range(num_lods):
        nf = num_frags[lod]
        fp = struct.unpack_from(f"<{3 * nf}I", manifest, off)  # all x, all y, all z
        off += 3 * nf * 4
        positions.append([[fp[i], fp[nf + i], fp[2 * nf + i]] for i in range(nf)])
        sz = list(struct.unpack_from(f"<{nf}I", manifest, off))
        off += nf * 4
        sizes.append(sz)
        for size in sz:
            if lod == 0:
                lod0.append((cursor, int(size)))
            cursor += int(size)
    return {
        "chunk_shape": chunk_shape,
        "grid_origin": grid_origin,
        "num_lods": int(num_lods),
        "lod_scales": lod_scales,
        "vertex_offsets": vertex_offsets,
        "num_fragments_per_lod": num_frags,
        "fragment_positions": positions,
        "fragment_sizes": sizes,
        "lod0_fragments": lod0,
        "lod0_bytes": int(sum(s for _o, s in lod0)),
        "total_bytes": int(cursor),
    }


def fetch_ng_info(ng_url: str, fs) -> dict:
    """GET `<ng>/info` (brotli-encoded on the live host; decoded by aiohttp)."""
    return json.loads(fs.cat_file(f"{ng_url.rstrip('/')}/info"))


def fetch_ng_object_http(ng_url: str, feature_id: int, fs, *, info: dict | None = None, decode: bool = False) -> dict:
    """Fetch ONE segment's finest-LOD bytes from a Neuroglancer precomputed mesh
    store over HTTP, the way the viewer does:

    * multilod draco, loose (`@type` neuroglancer_multilod_draco, no sharding):
      GET `<id>.index`, then ONE Range GET of the contiguous LOD-0 prefix of
      `<id>` (2 requests);
    * legacy single-resolution (`neuroglancer_legacy_mesh`): GET the `<id>:0`
      JSON manifest, then GET each listed fragment (1 + n_fragments requests);
    * sharded (`info` has `sharding`): NOT implemented -- returns
      ``{"layout": "sharded", "note": NG_SHARDED_NOTE}`` without fetching.

    `decode=True` additionally runs DracoPy on the LOD-0 fragments (lazy,
    best-effort; vertex count only). Raises FileNotFoundError if the object is
    absent."""
    base = ng_url.rstrip("/")
    if info is None:
        info = fetch_ng_info(ng_url, fs)
    fid = int(feature_id)
    if info.get("sharding"):
        return {"feature_id": fid, "layout": "sharded", "note": NG_SHARDED_NOTE}

    if "multilod" in str(info.get("@type", "")):
        manifest = fs.cat_file(f"{base}/{fid}.index")
        parsed = parse_ng_multilod_manifest(manifest)
        lod0 = b""
        if parsed["lod0_bytes"] > 0:
            lod0 = fs.cat_file(f"{base}/{fid}", start=0, end=parsed["lod0_bytes"])
        out = {
            "feature_id": fid,
            "layout": "multilod_loose",
            "index_bytes": len(manifest),
            "lod0_bytes": len(lod0),
            "n_lod0_fragments": len(parsed["lod0_fragments"]),
            "object_total_bytes": parsed["total_bytes"],
        }
        if decode:
            try:
                import DracoPy  # lazy

                nv = 0
                for off_b, size in parsed["lod0_fragments"]:
                    if size:
                        nv += int(np.asarray(DracoPy.decode(lod0[off_b:off_b + size]).points).reshape(-1, 3).shape[0])
                out["n_vertices"] = nv
            except Exception as exc:  # decode is optional for the fetch measure
                out["decode_error"] = f"{type(exc).__name__}: {exc}"
        return out

    manifest_json = json.loads(fs.cat_file(f"{base}/{fid}:0"))
    frags = list(manifest_json.get("fragments", []))
    nbytes = 0
    for fr in frags:
        nbytes += len(fs.cat_file(f"{base}/{fr}"))
    return {
        "feature_id": fid,
        "layout": "legacy_manifest",
        "n_fragments": len(frags),
        "fragment_bytes": nbytes,
    }


# --- 3D Tiles per-tile GLB comparator over HTTP ----------------------------


def tiles3d_finest_uris(tileset: dict) -> tuple[int, list[str]]:
    """Walk a 3D Tiles tileset tree and return `(depth, content URIs)` of the
    DEEPEST level that carries content (the finest zoom). Pure."""
    best = -1
    uris: list[str] = []

    def walk(node: Any, depth: int) -> None:
        nonlocal best, uris
        if not isinstance(node, dict):
            return
        content = node.get("content") or {}
        uri = content.get("uri") or content.get("url")
        if uri:
            if depth > best:
                best, uris = depth, [uri]
            elif depth == best:
                uris.append(uri)
        for child in node.get("children") or []:
            walk(child, depth + 1)

    walk(tileset.get("root") or {}, 0)
    return (max(best, 0), uris)


def measure_tiles3d_http(tiles3d_url: str, fs, *, n: int = 20, iters: int = 1) -> dict:
    """GET `<tiles3d>/tileset.json` (br on the live host), pick up to `n`
    finest-level GLBs (evenly strided, deterministic) and GET each `iters`
    times; parse each GLB's JSON chunk to count nodes (= features per tile)."""
    base = tiles3d_url.rstrip("/")
    counters = fs.counters
    before = counters.snapshot()
    tileset, t_tileset = timed(lambda: json.loads(fs.cat_file(f"{base}/tileset.json")))
    depth, uris = tiles3d_finest_uris(tileset)
    if n > 0 and len(uris) > n:
        stride = len(uris) / n
        sample = [uris[int(i * stride)] for i in range(n)]
    else:
        sample = list(uris)

    nodes_total = 0
    bytes_total = 0
    latencies: list[float] = []
    for it in range(max(1, iters)):
        for uri in sample:
            url = urljoin(base + "/", uri)
            raw, t = timed(fs.cat_file, url)
            latencies.append(t)
            if it == 0:
                bytes_total += len(raw)
                try:
                    J, _bn = glb_json_bin(raw)
                    nodes_total += len(J.get("nodes", []))
                except Exception:
                    pass
    delta = HttpCounters.delta(before, counters.snapshot())
    return {
        "tiles3d_url": base,
        "tileset_json_s": t_tileset,
        "finest_depth": depth,
        "n_leaf_glb": len(uris),
        "n_fetched": len(sample),
        "iters": iters,
        "n_samples": len(latencies),
        "median_s": _median(latencies),
        "p95_s": _percentile(latencies, 0.95),
        "glb_nodes_total": nodes_total,
        "glb_bytes_total": bytes_total,
        "http_requests": delta["http_requests"],
        "bytes_transferred": delta["bytes_transferred"],
        "bytes_body": delta["bytes_body"],
    }


# --- catalog probe (requests; uninstrumented, discovery only) --------------


def probe_url(url: str, *, session=None, timeout_s: float = 30.0) -> dict:
    """HEAD one URL and report status + the transport headers that matter for
    range access (content-length, content-encoding, accept-ranges)."""
    import requests  # lazy

    s = session or requests.Session()
    try:
        r = s.head(url, allow_redirects=True, timeout=timeout_s)
    except requests.RequestException as exc:
        return {"url": url, "status": None, "exists": False, "error": f"{type(exc).__name__}: {exc}"}
    h = r.headers
    cl = h.get("Content-Length")
    return {
        "url": url,
        "status": int(r.status_code),
        "exists": r.status_code == 200,
        "content_length": int(cl) if cl is not None and cl.isdigit() else None,
        "content_encoding": h.get("Content-Encoding"),
        "content_type": h.get("Content-Type"),
        "accept_ranges": h.get("Accept-Ranges"),
        "server": h.get("Server"),
        "x_cache": h.get("X-Cache"),
    }


def probe_catalog(
    base_url: str,
    dataset_ids: list[str],
    *,
    paths: tuple[str, ...] = _PROBE_PATHS,
    timeout_s: float = 30.0,
) -> dict[str, dict[str, dict]]:
    """For each dataset id, HEAD `<base>/<id>/<path>` for the muDM artefacts a
    benchmark needs (finest-zoom parquet part, NG `info`, 3D Tiles tileset) and
    report status / content-length / content-encoding / accept-ranges. On the
    live host missing keys answer 403 (reported as-is, `exists: False`)."""
    import requests  # lazy

    out: dict[str, dict[str, dict]] = {}
    with requests.Session() as s:
        for did in dataset_ids:
            out[did] = {
                p: probe_url(f"{base_url.rstrip('/')}/{did}/{p}", session=s, timeout_s=timeout_s)
                for p in paths
            }
    return out


# --- driver -----------------------------------------------------------------


def run_access_http(
    base_url: str,
    out_json: str,
    *,
    zoom: int = 4,
    n_ids: int = 20,
    iters: int = 3,
    bulk_n: int = 0,
    parts: dict[int, list[str]] | None = None,
    parts_json: str | None = None,
    ng_url: str | None = None,
    tiles3d_url: str | None = None,
    hardware: dict | None = None,
    pre_buffer: bool = True,
    hole_size_limit: int | None = None,
    part_pattern: str = _DEFAULT_PART_PATTERN,
    by_id_all_zooms: bool = False,
    timeout_s: float = 120.0,
    discovery_cap: int = _DISCOVERY_CAP,
    fragment_readahead: int | None = 16,
) -> dict:
    """The `run_access_patterns` measures re-run over HTTP against a deployed
    dataset root (`<base_url>` = the directory holding `features.parquet/`).

    1. BULK whole-level read of `zoom` (one columnar scan; Range reads of every
       row group's feature_id + positions chunks).
    2. 3x3 SPATIAL WINDOW via tile_x/tile_y pushdown (Range reads of the
       surviving row groups only), vs its own whole-level scan.
    3. BY-ID single fetch on the default tile-ordered layout. Unlike the local
       arm this is scoped to ONE level by default (`by_id_all_zooms=False`): the
       local by-id scans every zoom, which over HTTP would be `pyramid_bytes x
       n x iters`; the scope is recorded in the result.
    4. (with `ng_url`) Neuroglancer per-object fetch of the same ids (by-id
       latency + a per-neuron bulk arm extrapolated to the level, like local).
    5. (with `tiles3d_url`) one-GLB-per-tile GETs at the finest level.

    Every arm records wall time, `http_requests`, `bytes_transferred` (wire) and
    `bytes_body` (decoded) from the shared session counters. Output JSON has the
    same header as `run_access_patterns` (schema-valid via `assemble_report`).
    """
    base = base_url.rstrip("/")
    fs = _counting_fs_class()(timeout_s=timeout_s)
    counters: HttpCounters = fs.counters
    scan = http_scan_options(pre_buffer=pre_buffer, hole_size_limit=hole_size_limit)
    results: dict[str, Any] = {"zoom": int(zoom), "base_url": base}

    # --- discovery: which parts exist per zoom (HEAD probes, or a listing) ---
    zooms = list(range(0, int(zoom) + 1)) if by_id_all_zooms else [int(zoom)]
    t_probe = 0.0
    if parts_json:
        with open(parts_json) as f:
            listed = json.load(f)
        parts = {int(k): list(v) for k, v in listed.items()}
        source = "parts_json"
    elif parts is not None:
        parts = {int(k): list(v) for k, v in parts.items()}
        source = "argument"
    else:
        parts, t_probe = timed(
            discover_parts, base, zooms, fs, part_pattern=part_pattern, cap=discovery_cap
        )
        source = "probe"
    parts = {z: names for z, names in parts.items() if z in zooms}
    if not parts.get(int(zoom)):
        raise FileNotFoundError(f"no parquet parts for zoom={zoom} under {base}")
    d0 = counters.snapshot()
    dataset, t_open = timed(http_parquet_dataset, base, parts, fs)
    d_disc = HttpCounters.delta(d0, counters.snapshot())
    print(
        f"[access-http] {base}: {sum(len(v) for v in parts.values())} part(s) for zooms {zooms} "
        f"({source}); dataset open {t_open:.1f}s"
    )
    results["discovery"] = {
        "source": source,
        "zooms": zooms,
        "parts": {str(z): names for z, names in sorted(parts.items())},
        "n_parts": sum(len(v) for v in parts.values()),
        "capped": any(len(v) >= discovery_cap for v in parts.values()) if source == "probe" else False,
        "discovery_cap": discovery_cap,
        "probe_s": t_probe,
        "dataset_open_s": t_open,
        "http_requests": d_disc["http_requests"],
        "bytes_transferred": d_disc["bytes_transferred"],
    }
    results["transport"] = {
        "parquet": probe_url(_part_url(base, zoom, parts[int(zoom)][0]), timeout_s=timeout_s),
        "accounting": (
            "bytes_transferred = response Content-Length per read body (wire bytes); "
            "bytes_body = decoded bytes; HEAD counted as a request only. Parquet footers "
            "are read once per part (charged to discovery/bulk) and cached on the dataset "
            "(pyarrow cache_metadata), so window/by-id arms pay row-group reads only."
        ),
    }

    # --- 1. BULK whole level ---
    feats, t_pq, d_bulk = counters.measure(
        bulk_positions, dataset, zoom, scan_options=scan, fragment_readahead=fragment_readahead
    )
    all_features = list(feats)
    N = len(all_features)
    print(
        f"[access-http] bulk zoom={zoom}: {N} features, {t_pq:.1f}s, "
        f"{d_bulk['http_requests']} requests, {d_bulk['bytes_transferred'] / 1e6:.1f} MB"
    )
    results["bulk_read"] = {
        "n_total": N,
        "parquet_whole_level_s": t_pq,
        "parquet_per_neuron_ms": t_pq / max(N, 1) * 1000,
        "http_requests": d_bulk["http_requests"],
        "bytes_transferred": d_bulk["bytes_transferred"],
        "bytes_body": d_bulk["bytes_body"],
    }

    # --- 2. SPATIAL WINDOW ---
    (mx, my), t_pick, d_pick = counters.measure(
        pick_window_origin, dataset, zoom, scan_options=scan, fragment_readahead=fragment_readahead
    )
    nrows, t_win, d_win = counters.measure(
        window_row_count, dataset, zoom, mx, my, scan_options=scan, fragment_readahead=fragment_readahead
    )
    results["spatial_window"] = {
        "window": "3x3 tiles",
        "window_origin": [mx, my],
        "rows_returned": nrows,
        "parquet_window_s": t_win,
        "parquet_whole_level_s": t_pq,
        "speedup_vs_whole_level": (t_pq / t_win) if t_win > 0 else 0.0,
        "http_requests": d_win["http_requests"],
        "bytes_transferred": d_win["bytes_transferred"],
        "bytes_body": d_win["bytes_body"],
        "bytes_ratio_vs_whole_level": (
            d_win["bytes_transferred"] / d_bulk["bytes_transferred"]
            if d_bulk["bytes_transferred"] else None
        ),
        "tile_index": {  # locating the window (tile_x/tile_y column scan), not timed into the arm
            "seconds": t_pick,
            "http_requests": d_pick["http_requests"],
            "bytes_transferred": d_pick["bytes_transferred"],
        },
    }

    print(
        f"[access-http] window {mx},{my}: {nrows} rows, {t_win:.2f}s, "
        f"{d_win['http_requests']} requests, {d_win['bytes_transferred'] / 1e6:.1f} MB"
    )

    # --- 3. BY-ID single fetch, default (tile-ordered) layout ---
    sample = all_features[:n_ids] if n_ids > 0 else all_features
    id_zoom = None if by_id_all_zooms else int(zoom)
    # On a tile-ordered level feature_id statistics prune little, so one by-id
    # fetch can cost a whole-level read; warn with the worst case before starting.
    print(
        f"[access-http] by-id: {len(sample)} ids x {iters} iters; worst case "
        f"{len(sample) * max(1, iters) * d_bulk['bytes_transferred'] / 1e6:.0f} MB "
        f"(level scan per fetch) -- lower --n/--iters if this is too much"
    )
    b0 = counters.snapshot()
    ra = measure_random_access(
        lambda fid: _parquet_count_by_id_scoped(
            dataset, fid, zoom=id_zoom, scan_options=scan, fragment_readahead=fragment_readahead
        ),
        sample,
        iters=iters,
    )
    d_id = HttpCounters.delta(b0, counters.snapshot())
    ra.update({
        "scope": f"zoom={zoom} only" if id_zoom is not None else f"zooms {zooms}",
        "layout": "tile-ordered (default)",
        "http_requests": d_id["http_requests"],
        "bytes_transferred": d_id["bytes_transferred"],
        "bytes_body": d_id["bytes_body"],
        "requests_per_fetch": d_id["http_requests"] / max(ra["n_samples"], 1),
        "bytes_per_fetch": d_id["bytes_transferred"] / max(ra["n_samples"], 1),
    })
    results["by_id_default"] = ra

    # --- 4. Neuroglancer per-object comparator ---
    if ng_url:
        ngb = ng_url.rstrip("/")
        try:
            info, t_info, _d_info = counters.measure(fetch_ng_info, ngb, fs)
        except Exception as exc:
            results["by_id_neuroglancer"] = {
                "ng_url": ngb, "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            if info.get("sharding"):
                results["by_id_neuroglancer"] = {
                    "ng_url": ngb, "layout": "sharded", "note": NG_SHARDED_NOTE,
                    "info_type": info.get("@type"), "info_s": t_info,
                }
            else:
                missing = 0

                def _fetch(fid: int):
                    nonlocal missing
                    try:
                        return fetch_ng_object_http(ngb, fid, fs, info=info)
                    except FileNotFoundError:
                        missing += 1
                        return None

                layout = "multilod_loose" if "multilod" in str(info.get("@type", "")) else "legacy_manifest"
                b0 = counters.snapshot()
                ra_ng = measure_random_access(_fetch, sample, iters=iters)
                d_ng = HttpCounters.delta(b0, counters.snapshot())
                ra_ng.update({
                    "ng_url": ngb,
                    "layout": layout,
                    "info_type": info.get("@type"),
                    "info_s": t_info,
                    "n_missing_fetches": missing,
                    "http_requests": d_ng["http_requests"],
                    "bytes_transferred": d_ng["bytes_transferred"],
                    "bytes_body": d_ng["bytes_body"],
                    "requests_per_fetch": d_ng["http_requests"] / max(ra_ng["n_samples"], 1),
                    "bytes_per_fetch": d_ng["bytes_transferred"] / max(ra_ng["n_samples"], 1),
                })
                results["by_id_neuroglancer"] = ra_ng

                # Bulk arm: per-object fetch of the first bulk_n features, per-neuron
                # rate extrapolated to the whole level (as the local arm does).
                bn = min(bulk_n if bulk_n > 0 else N, N)
                sub = all_features[:bn]
                _, t_ng, d_ngb = counters.measure(lambda: [_fetch(f) for f in sub])
                results["bulk_read"].update({
                    "neuroglancer_bulk_n": bn,
                    "neuroglancer_per_neuron_ms": t_ng / max(bn, 1) * 1000,
                    "neuroglancer_whole_level_extrap_s": t_ng / max(bn, 1) * N,
                    "neuroglancer_http_requests": d_ngb["http_requests"],
                    "neuroglancer_bytes_transferred": d_ngb["bytes_transferred"],
                    "neuroglancer_bytes_whole_level_extrap": d_ngb["bytes_transferred"] / max(bn, 1) * N,
                })

    # --- 5. 3D Tiles per-tile GLB comparator ---
    if tiles3d_url:
        try:
            results["tiles3d_http"] = measure_tiles3d_http(tiles3d_url, fs, n=n_ids, iters=iters)
        except Exception as exc:
            results["tiles3d_http"] = {
                "tiles3d_url": tiles3d_url.rstrip("/"), "error": f"{type(exc).__name__}: {exc}",
            }

    results["http_totals"] = counters.snapshot()
    results["peak_rss_bytes"] = peak_rss_bytes()

    report = assemble_report(
        Path(out_json).stem,
        results,
        config={
            "mode": "access-http", "base_url": base, "zoom": int(zoom), "n_ids": n_ids,
            "iters": iters, "bulk_n": bulk_n, "parts_json": parts_json, "ng_url": ng_url,
            "tiles3d_url": tiles3d_url, "pre_buffer": pre_buffer,
            "hole_size_limit": hole_size_limit, "part_pattern": part_pattern,
            "by_id_all_zooms": by_id_all_zooms, "timeout_s": timeout_s,
            "discovery_cap": discovery_cap, "fragment_readahead": fragment_readahead,
        },
        hardware=hardware,
    )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the format spec-sheet benchmark."""
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Format spec-sheet: compare muDM output formats (parquet / 3D Tiles "
            "meshopt-q14 / Neuroglancer multilod Draco) against the source OBJ on "
            "encode time, stored size, decode time, random-access latency, peak "
            "memory, and vertex fidelity -- all from ONE shared tile index."
        )
    )
    parser.add_argument(
        "--obj-dir", default=None,
        help="Directory of <body_id>.obj meshes (spec-sheet / access modes)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (benchmark_id = stem); optional for probe-http",
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Sample size for decode / random-access / fidelity (default 200; 0=all)",
    )
    parser.add_argument("--max-zoom", type=int, default=4, help="Pyramid depth (default 4)")
    parser.add_argument(
        "--iters", type=int, default=10, help="Decode/random-access iterations (default 10)"
    )
    sharded = parser.add_mutually_exclusive_group()
    sharded.add_argument(
        "--sharded", dest="sharded", action="store_true", default=True,
        help="Emit Neuroglancer sharded layout (default)",
    )
    sharded.add_argument(
        "--no-sharded", dest="sharded", action="store_false",
        help="Emit Neuroglancer loose per-segment files",
    )
    parser.add_argument(
        "--out-root", default=None,
        help="Encode output root (default: beside --out); for --mode access this is the EXISTING encoded formats root",
    )
    parser.add_argument(
        "--mode", choices=["spec-sheet", "access", "access-http", "probe-http"], default="spec-sheet",
        help=(
            "spec-sheet (encode+decode+fidelity); access (bulk/spatial/by-id patterns on an "
            "existing --out-root); access-http (the same Parquet patterns over HTTP range reads "
            "from a deployed dataset --base-url, plus NG / 3D Tiles per-object GETs); "
            "probe-http (which artefacts each --datasets id exposes under --base-url)"
        ),
    )
    parser.add_argument(
        "--bulk-n", type=int, default=0,
        help="access mode: per-item bulk sample size (0=all); parquet always reads the whole level",
    )
    parser.add_argument(
        "--records", default=None,
        help=(
            "access mode: JSON file mapping body_id -> record (e.g. the fusion "
            "feature table); enables the feature->record join arm"
        ),
    )
    parser.add_argument(
        "--join-n", type=int, default=200,
        help="access mode: per-feature baseline sample size for the join arm (default 200)",
    )
    http = parser.add_argument_group("access-http / probe-http")
    http.add_argument(
        "--base-url", default=None,
        help=(
            "access-http: dataset root holding features.parquet/ (e.g. "
            "https://ngrf.mudm.io/tiles/manc-full); probe-http: the parent of the dataset "
            "ids (e.g. https://ngrf.mudm.io/tiles)"
        ),
    )
    http.add_argument(
        "--zoom", type=int, default=None,
        help="access-http: level to measure (default: --max-zoom)",
    )
    http.add_argument(
        "--parts-json", default=None,
        help='access-http: JSON {"<zoom>": ["part_000.mu.parquet", ...]} listing the parts per zoom (skips HEAD discovery)',
    )
    http.add_argument(
        "--part-pattern", default=_DEFAULT_PART_PATTERN,
        help=f"access-http: part-name pattern for discovery (default {_DEFAULT_PART_PATTERN!r})",
    )
    http.add_argument("--ng-url", default=None, help="access-http: Neuroglancer precomputed mesh dir URL (has /info)")
    http.add_argument("--tiles3d-url", default=None, help="access-http: 3D Tiles dir URL (has /tileset.json)")
    http.add_argument(
        "--no-pre-buffer", dest="pre_buffer", action="store_false", default=True,
        help="access-http: disable pyarrow read coalescing (one Range GET per column chunk)",
    )
    http.add_argument(
        "--hole-size-limit", type=int, default=None,
        help="access-http: largest gap (bytes) pyarrow reads THROUGH when coalescing ranges (pyarrow default 8192; 0 = never fetch pruned bytes)",
    )
    http.add_argument(
        "--by-id-all-zooms", action="store_true", default=False,
        help="access-http: by-id fetch scans every zoom 0..--zoom like the local arm (default: --zoom only)",
    )
    http.add_argument("--http-timeout", type=float, default=120.0, help="access-http: per-request timeout seconds")
    http.add_argument(
        "--discovery-cap", type=int, default=_DISCOVERY_CAP,
        help=f"access-http: max parts probed per zoom (default {_DISCOVERY_CAP}; hemibrain zoom 4 exceeds it)",
    )
    http.add_argument(
        "--fragment-readahead", type=int, default=16,
        help="access-http: parts opened concurrently per scan (pyarrow default 4; default 16 here)",
    )
    http.add_argument("--datasets", default=None, help="probe-http: comma-separated dataset ids")
    args = parser.parse_args(argv)

    if args.mode == "probe-http":
        if not args.base_url or not args.datasets:
            parser.error("--mode probe-http requires --base-url and --datasets")
        ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
        probe = probe_catalog(args.base_url, ids, timeout_s=args.http_timeout)
        for did, paths in probe.items():
            print(did)
            for rel, r in paths.items():
                print(
                    f"  {r.get('status')!s:>4}  {rel:<45} len={r.get('content_length')} "
                    f"enc={r.get('content_encoding')} ranges={r.get('accept_ranges')}"
                )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(probe, f, indent=2)
            print(f"wrote {args.out} (mode=probe-http)")
        return

    if args.mode == "access-http":
        if not args.base_url or not args.out:
            parser.error("--mode access-http requires --base-url and --out")
        report = run_access_http(
            args.base_url,
            args.out,
            zoom=args.zoom if args.zoom is not None else args.max_zoom,
            n_ids=args.n,
            iters=args.iters,
            bulk_n=args.bulk_n,
            parts_json=args.parts_json,
            ng_url=args.ng_url,
            tiles3d_url=args.tiles3d_url,
            pre_buffer=args.pre_buffer,
            hole_size_limit=args.hole_size_limit,
            part_pattern=args.part_pattern,
            by_id_all_zooms=args.by_id_all_zooms,
            timeout_s=args.http_timeout,
            discovery_cap=args.discovery_cap,
            fragment_readahead=args.fragment_readahead,
        )
        print(f"wrote {args.out} (benchmark_id={report['benchmark_id']}, mode=access-http)")
        return

    if not args.obj_dir or not args.out:
        parser.error(f"--mode {args.mode} requires --obj-dir and --out")

    if args.mode == "access":
        if not args.out_root:
            parser.error("--mode access requires --out-root (the existing encoded formats root)")
        report = run_access_patterns(
            args.out_root,
            args.obj_dir,
            args.out,
            zoom=args.max_zoom,
            bulk_n=args.bulk_n,
            n_ids=args.n,
            iters=args.iters,
            records_json=args.records,
            join_n=args.join_n,
        )
        print(f"wrote {args.out} (benchmark_id={report['benchmark_id']}, mode=access)")
        return

    report = run_spec_sheet(
        args.obj_dir,
        args.out,
        n=args.n,
        max_zoom=args.max_zoom,
        iters=args.iters,
        sharded=args.sharded,
        out_root=args.out_root,
    )
    print(f"wrote {args.out} (benchmark_id={report['benchmark_id']})")


if __name__ == "__main__":
    main()
