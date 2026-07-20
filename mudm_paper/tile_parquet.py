"""Fresh-tiling driver: OBJ meshes + JSON labels -> f32 parquet mesh pyramid.

Phase-0 data-prep tool for the muDM economy harness (the bridge to the real
oden data-prep run, Phase E0). Given a directory of `<body_id>.obj` neuron
meshes and a JSON file mapping body_id -> per-neuron metadata, this drives the
REAL muDM Rust tiler (`mudm_tools._rs.StreamingTileGenerator`) to produce a
partitioned parquet mesh pyramid at `out_dir/zoom={0..max_zoom}/part_*.parquet`,
with per-neuron metadata baked into each row's `tags` map -- the exact input
`mudm_paper.data.load_neuron_lods`/`load_labels` consume. No synthetic
fixture is involved: this is genuine pipeline output.

Mirrors `mudm-tools/scripts/download_hemibrain.py`'s `tile_streaming(...)`
(emit-all path), restricted to the parquet output (no 3D Tiles):
  1. `scan_obj_bounds(paths)` -- parallel (rayon) world-bounds scan.
  2. One `StreamingTileGenerator`, ingested via a single `add_obj_files(...)`
     batch call (parallel rayon ingest; tags extracted per-file).
  3. `mudm_tools.tiling3d.parquet_writer.generate_parquet(..., partitioned=True)`
     -- the fully-native Rust consolidation path when available.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds
from mudm_tools.tiling3d.parquet_writer import generate_parquet


def tile_to_parquet(
    obj_dir: str,
    meta_json: str,
    out_dir: str,
    *,
    max_zoom: int = 4,
    base_cells: int = 100,
    io_threads: int = 12,
) -> None:
    """Tile `obj_dir/*.obj` meshes into an f32 parquet pyramid at `out_dir`.

    Each `<body_id>.obj` (body_id = filename stem) becomes one feature.
    `meta_json` (`{body_id: {field: value, ...}}`) supplies per-neuron labels
    that are baked into every row's `tags` map alongside `body_id` itself
    (values coerced to `str`, matching the map<string,string> tags column the
    real tiler writes), so downstream code recovers both geometry
    (`data.load_neuron_lods`) and labels (`data.load_labels`) straight from
    the parquet pyramid -- no side-channel metadata file needed.
    """
    paths = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
    if not paths:
        raise FileNotFoundError(f"No .obj files found in {obj_dir}")

    with open(meta_json) as f:
        meta = json.load(f)

    body_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    tags_list = [
        {"body_id": bid, **{k: str(v) for k, v in meta.get(bid, {}).items()}}
        for bid in body_ids
    ]

    # Parallel (rayon) per-file bounds scan, vertex-only -- see scan_obj_bounds.
    bounds = scan_obj_bounds(paths)

    # Pass TMPDIR through explicitly: fragment shards can be large for real
    # neuron meshes, and the box's root partition is often too small (see
    # oden runbook). None (unset) falls back to the Rust default (std::env::temp_dir()).
    temp_dir = os.environ.get("TMPDIR")

    gen = StreamingTileGenerator(
        min_zoom=0,
        max_zoom=max_zoom,
        base_cells=base_cells,
        temp_dir=temp_dir,
    )
    gen.add_obj_files(paths, bounds, tags_list, ingest_threads=0, simplify=True)

    # partitioned=True -> `out_dir/zoom=Z/part_*.parquet` (native Rust
    # consolidation, bounded memory). parallel=True + io_threads matches the
    # measured-optimal knee used by the real hemibrain/manc runs.
    generate_parquet(
        gen,
        out_dir,
        bounds,
        partitioned=True,
        parallel=True,
        io_threads=io_threads,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tile OBJ neuron meshes + a JSON label map into an f32 parquet "
            "mesh pyramid using the real muDM Rust tiler."
        ),
    )
    parser.add_argument("--obj-dir", required=True, help="Directory of <body_id>.obj meshes")
    parser.add_argument("--meta", required=True, help="JSON file: {body_id: {field: value, ...}}")
    parser.add_argument("--out", required=True, help="Output directory for the parquet pyramid")
    parser.add_argument("--max-zoom", type=int, default=4)
    parser.add_argument("--base-cells", type=int, default=100)
    parser.add_argument("--io-threads", type=int, default=12)
    args = parser.parse_args()

    tile_to_parquet(
        args.obj_dir,
        args.meta,
        args.out,
        max_zoom=args.max_zoom,
        base_cells=args.base_cells,
        io_threads=args.io_threads,
    )


if __name__ == "__main__":
    main()
