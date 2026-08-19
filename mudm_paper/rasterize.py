"""Temporary vector-LOD rasterizer for the Nyxus feature-robustness experiment.

Turns muDM 2D polygon features (pixel coordinates) into per-level instance
label masks: each feature is simplified with the pipeline's reduce-to-floor
Douglas--Peucker at a set of tolerances, and every level is rasterized on the
ORIGINAL pixel grid with a stable per-instance label. Nyxus features computed
per (image, label) pair therefore join across levels, which is what the
robustness-across-scales comparison needs.

This is experiment scaffolding, not pipeline code -- the pipeline never
rasterizes. The simplifier mirrors the tiling pipeline's implementation
(mudm_tools/mudm2vt/simplify.py and rust/src/simplify2d.rs) so mask levels
correspond to what the pipeline would store.

Input: a muDM/GeoJSON FeatureCollection per image patch (Polygon or
MultiPolygon geometry, pixel coordinates). Instance label = feature order + 1,
identical at every level; a labels.json sidecar maps labels back to feature
ids when present.

Usage:
    uv run python -m mudm_paper.rasterize <features.json | dir> \
        --out masks/ --eps 0.5 1 2 4 8 --shape 256 256

    Output: masks/tau_0/<stem>.tif, masks/tau_0.5/<stem>.tif, ...
    (uint32, one file per patch per level; tau_0 = full detail).

    Nyxus then runs once per level, e.g.:
        nyxus --features=*ALL* --intDir=<images> --segDir=masks/tau_2 \
              --outDir=<results>/tau_2 --outputType=singlecsv
    and per-(patch, label) rows are compared across tau levels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
MIN_RING_VERTS = 4  # same floor as the fidelity measurement (min_edges=4)


# --- reduce-to-floor Douglas-Peucker (parity with the tiling pipeline) ------

def _dp_importance(pts, imp):
    n = len(pts)
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        ax, ay = pts[s][0], pts[s][1]
        bx, by = pts[e][0], pts[e][1]
        dx, dy = bx - ax, by - ay
        ls = dx * dx + dy * dy
        md, mi = 0.0, s + 1
        if ls < 1e-30:
            for i in range(s + 1, e):
                ex, ey = pts[i][0] - ax, pts[i][1] - ay
                d = math.sqrt(ex * ex + ey * ey)
                if d > md:
                    md, mi = d, i
        else:
            sl = math.sqrt(ls)
            for i in range(s + 1, e):
                d = abs((pts[i][1] - ay) * dx - (pts[i][0] - ax) * dy) / sl
                if d > md:
                    md, mi = d, i
        imp[mi] = md
        stack.append((s, mi))
        stack.append((mi, e))


def simplify_ring(ring, eps: float, floor: int = MIN_RING_VERTS) -> list:
    """Simplify one ring; always keeps at least ``floor`` vertices.

    Endpoints are anchored, matching the pipeline. Closed rings (first ==
    last) stay closed. ``eps == 0`` returns the ring unchanged.
    """
    if eps <= 0 or len(ring) <= max(floor, 2):
        return [[p[0], p[1]] for p in ring]
    n = len(ring)
    fl = max(floor, 2)
    imp = [0.0] * n
    imp[0] = math.inf
    imp[n - 1] = math.inf
    _dp_importance(ring, imp)
    keep = [d > eps for d in imp]
    kept = sum(keep)
    if kept < fl:
        order = [i for i in range(n) if not keep[i]]
        order.sort(key=lambda i: imp[i], reverse=True)
        for i in order:
            if kept >= fl:
                break
            keep[i] = True
            kept += 1
    return [[ring[i][0], ring[i][1]] for i in range(n) if keep[i]]


# --- rasterization -----------------------------------------------------------

def rasterize_instances(shapes: list, shape: tuple) -> np.ndarray:
    """Rasterize labelled polygons into a uint16 instance mask.

    ``shapes`` is a list of (label, polygons) where polygons is a list of
    rings groups ``[exterior, hole1, ...]`` in pixel coordinates. Later
    entries overwrite earlier ones where they overlap (painter's order).
    Holes are carved back to 0.

    uint16 output is a Nyxus-compatibility requirement: Nyxus's TIFF reader
    misparses uint32 label images into garbage ROIs (observed: an 8-label
    mask read back as 39 ROIs, segfaulting at directory scale). Labels past
    65535 raise rather than wrap.
    """
    max_label = max((label for label, _ in shapes), default=0)
    if max_label > 65535:
        raise ValueError(
            f"label {max_label} exceeds the uint16 range required for "
            "Nyxus-readable masks (65535)"
        )
    img = Image.new("I", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for label, polygons in shapes:
        for rings in polygons:
            exterior = rings[0]
            if len(exterior) < 3:
                continue
            draw.polygon([(p[0], p[1]) for p in exterior], fill=int(label))
            for hole in rings[1:]:
                if len(hole) >= 3:
                    draw.polygon([(p[0], p[1]) for p in hole], fill=0)
    return np.asarray(img, dtype=np.int32).astype(np.uint16)


def simplify_shapes(shapes: list, eps: float) -> list:
    """Apply simplify_ring to every ring of every polygon, keeping labels."""
    out = []
    for label, polygons in shapes:
        out.append(
            (label, [[simplify_ring(r, eps) for r in rings] for rings in polygons])
        )
    return out


# --- muDM / GeoJSON input ----------------------------------------------------

def load_feature_collection(path: Path) -> tuple:
    """Read a FeatureCollection; return (shapes, id_map).

    shapes: [(label, polygons)] with label = feature order + 1;
    id_map: {label: feature id} for features that carry an ``id``.
    Polygon and MultiPolygon geometries are supported; anything else is
    skipped (counted in the id_map under the key "skipped").
    """
    fc = json.loads(Path(path).read_text())
    shapes, id_map, skipped = [], {}, 0
    for i, feat in enumerate(fc.get("features", [])):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "Polygon":
            polys = [geom["coordinates"]]
        elif gtype == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            skipped += 1
            continue
        label = len(shapes) + 1
        shapes.append((label, polys))
        fid = feat.get("id", feat.get("properties", {}).get("id"))
        if fid is not None:
            id_map[str(label)] = fid
    if skipped:
        id_map["skipped"] = skipped
    return shapes, id_map


# --- driver ------------------------------------------------------------------

def write_level_masks(
    inputs: list,
    out_dir: Path,
    eps_levels: list,
    shape: tuple,
) -> dict:
    """Rasterize every input FeatureCollection at tau=0 plus each eps level.

    Returns a manifest dict (also written to ``out_dir/manifest.json``).
    """
    import tifffile

    out_dir = Path(out_dir)
    levels = [0.0] + [e for e in eps_levels if e > 0]
    manifest = {
        "shape": list(shape),
        "levels": [f"tau_{e:g}" for e in levels],
        "min_ring_verts": MIN_RING_VERTS,
        "patches": [],
    }
    for src in inputs:
        src = Path(src)
        shapes, id_map = load_feature_collection(src)
        for eps in levels:
            level_dir = out_dir / f"tau_{eps:g}"
            level_dir.mkdir(parents=True, exist_ok=True)
            mask = rasterize_instances(simplify_shapes(shapes, eps), shape)
            tifffile.imwrite(level_dir / f"{src.stem}.tif", mask)
        if id_map:
            ids_dir = out_dir / "labels"
            ids_dir.mkdir(parents=True, exist_ok=True)
            (ids_dir / f"{src.stem}.json").write_text(json.dumps(id_map))
        manifest["patches"].append({"source": str(src), "n_features": len(shapes)})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="FeatureCollection .json file, or a directory of them")
    ap.add_argument("--out", required=True, help="output directory for level masks")
    ap.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS,
                    help=f"simplification tolerances in px (default {DEFAULT_EPS})")
    ap.add_argument("--shape", nargs=2, type=int, default=[256, 256],
                    metavar=("H", "W"), help="mask height width (default 256 256)")
    args = ap.parse_args(argv)

    src = Path(args.input)
    inputs = sorted([*src.glob("*.json"), *src.glob("*.geojson")]) if src.is_dir() else [src]
    if not inputs:
        ap.error(f"no .json/.geojson feature collections under {src}")
    manifest = write_level_masks(inputs, Path(args.out), args.eps, tuple(args.shape))
    n = len(manifest["patches"])
    print(f"wrote {len(manifest['levels'])} levels x {n} patches under {args.out}")
    print("next: run Nyxus once per level, e.g.")
    print(f"  nyxus --features=*ALL* --intDir=<images> "
          f"--segDir={args.out}/tau_2 --outDir=<results>/tau_2 --outputType=singlecsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
