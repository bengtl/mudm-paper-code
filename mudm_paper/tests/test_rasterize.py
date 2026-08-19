"""Tests for the temporary Nyxus rasterizer (mudm_paper.rasterize)."""

import json
import math

import numpy as np
import pytest

from mudm_paper.rasterize import (
    MIN_RING_VERTS,
    load_feature_collection,
    main,
    rasterize_instances,
    simplify_ring,
    simplify_shapes,
    write_level_masks,
)


def _circle(cx, cy, r, n=100, closed=True):
    pts = [
        [cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)]
        for i in range(n)
    ]
    if closed:
        pts.append(list(pts[0]))
    return pts


def _square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


# --- simplify_ring -----------------------------------------------------------

def test_simplify_noop_at_zero_eps():
    ring = _circle(16, 16, 10)
    assert simplify_ring(ring, 0.0) == [[p[0], p[1]] for p in ring]


def test_simplify_hits_floor_at_huge_eps():
    ring = _circle(16, 16, 10)
    simp = simplify_ring(ring, 1e9)
    assert len(simp) == MIN_RING_VERTS


def test_simplify_keeps_closure_and_fidelity():
    shapely = pytest.importorskip("shapely.geometry")
    ring = _circle(32, 32, 20, n=200)
    simp = simplify_ring(ring, 0.5)
    assert simp[0] == simp[-1]  # closed ring stays closed
    full = shapely.Polygon(ring)
    red = shapely.Polygon(simp)
    iou = full.intersection(red).area / full.union(red).area
    assert iou > 0.95
    assert len(simp) < len(ring)


# --- rasterize_instances -----------------------------------------------------

def test_square_mask_label_and_area():
    shapes = [(1, [[_square(5, 5, 15, 15)]])]
    mask = rasterize_instances(shapes, (32, 32))
    assert mask.dtype == np.uint16
    assert set(np.unique(mask)) == {0, 1}
    area = int((mask == 1).sum())
    assert 100 <= area <= 145  # 10px square; PIL includes boundary pixels


def test_hole_is_carved():
    outer = _square(2, 2, 29, 29)
    hole = _square(10, 10, 20, 20)
    mask = rasterize_instances([(1, [[outer, hole]])], (32, 32))
    assert mask[15, 15] == 0  # inside the hole
    assert mask[5, 5] == 1  # in the rim


def test_overlap_painters_order():
    a = _square(2, 2, 20, 20)
    b = _square(10, 10, 28, 28)
    mask = rasterize_instances([(1, [[a]]), (2, [[b]])], (32, 32))
    assert mask[15, 15] == 2  # later label wins in the overlap
    assert mask[5, 5] == 1


def test_degenerate_ring_skipped():
    mask = rasterize_instances([(1, [[[[3, 3], [4, 4]]]]), (2, [[_square(8, 8, 12, 12)]])], (16, 16))
    assert set(np.unique(mask)) == {0, 2}


def test_label_overflow_rejected():
    # uint16 output is a Nyxus-compat requirement (its TIFF reader misparses
    # uint32 label images into garbage ROIs); labels past 65535 must fail
    # loudly rather than wrap.
    shapes = [(70000, [[_square(1, 1, 5, 5)]])]
    with pytest.raises(ValueError, match="uint16"):
        rasterize_instances(shapes, (16, 16))


# --- level stability ---------------------------------------------------------

def test_labels_and_frame_stable_across_levels():
    shapes = [(1, [[_circle(10, 10, 6)]]), (2, [[_circle(22, 22, 6)]])]
    full = rasterize_instances(shapes, (32, 32))
    coarse = rasterize_instances(simplify_shapes(shapes, 4.0), (32, 32))
    assert full.shape == coarse.shape
    assert set(np.unique(full)) == set(np.unique(coarse)) == {0, 1, 2}
    # instance 1 stays in place: mask overlap between levels is high
    inter = int(((full == 1) & (coarse == 1)).sum())
    union = int(((full == 1) | (coarse == 1)).sum())
    assert inter / union > 0.6


# --- loader ------------------------------------------------------------------

def _fc(features):
    return {"type": "FeatureCollection", "features": features}


def test_loader_polygon_multipolygon_and_ids(tmp_path):
    fc = _fc(
        [
            {
                "type": "Feature",
                "id": "nuc-7",
                "geometry": {"type": "Polygon", "coordinates": [_square(1, 1, 5, 5)]},
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[_square(8, 8, 12, 12)], [_square(20, 20, 24, 24)]],
                },
                "properties": {"id": 42},
            },
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 1]}},
        ]
    )
    p = tmp_path / "patch.json"
    p.write_text(json.dumps(fc))
    shapes, id_map = load_feature_collection(p)
    assert [label for label, _ in shapes] == [1, 2]
    assert len(shapes[1][1]) == 2  # MultiPolygon keeps both parts
    assert id_map["1"] == "nuc-7"
    assert id_map["2"] == 42
    assert id_map["skipped"] == 1


# --- end-to-end CLI ----------------------------------------------------------

def test_cli_writes_levels_and_manifest(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    src = tmp_path / "features"
    src.mkdir()
    for name, offset in [("p0", 0), ("p1", 4)]:
        fc = _fc(
            [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_circle(12 + offset, 12, 8, n=80)],
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_circle(40, 40 + offset, 8, n=80)],
                    },
                },
            ]
        )
        (src / f"{name}.json").write_text(json.dumps(fc))

    out = tmp_path / "masks"
    rc = main([str(src), "--out", str(out), "--eps", "2", "8", "--shape", "56", "56"])
    assert rc == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["levels"] == ["tau_0", "tau_2", "tau_8"]
    for level in manifest["levels"]:
        for stem in ["p0", "p1"]:
            mask = tifffile.imread(out / level / f"{stem}.tif")
            assert mask.shape == (56, 56)
            assert mask.dtype == np.uint16
            assert set(np.unique(mask)) == {0, 1, 2}
