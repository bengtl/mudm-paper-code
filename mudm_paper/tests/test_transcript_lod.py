"""Tests for mudm_paper.transcript_lod (2D point-LOD fidelity evaluation).

The fixture writes a tiny synthetic 2D muDM pyramid with the EXACT row schema
the Rust 2D tiler emits (`streaming2d.rs`):

    tile_x u16, tile_y u16, feature_id u32, geom_type u8,
    positions large_binary, indices large_binary,
    ring_lengths list<u32>, tags map<string,string>

laid out as `<root>/zoom=<Z>/part_NNN.mu.parquet` (zoom only in the directory
name). Finest-zoom points are generated at random tiles/genes; coarser zooms
are derived with the tiler's rule (`feature_id % 2**(z_max - z) == 0`, tile
coords shifted right by `z_max - z`). Polygon rows (geom_type 3,
layer_type=cells) and a non-transcript point layer are mixed in to prove the
filtering. Individual tests then override one zoom to inject rule violations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mudm_paper
from mudm_paper import transcript_lod
from mudm_paper.transcript_lod import evaluate_point_lod

_SCHEMA = pa.schema([
    ("tile_x", pa.uint16()),
    ("tile_y", pa.uint16()),
    ("feature_id", pa.uint32()),
    ("geom_type", pa.uint8()),
    ("positions", pa.large_binary()),
    ("indices", pa.large_binary()),
    ("ring_lengths", pa.list_(pa.uint32())),
    ("tags", pa.map_(pa.string(), pa.string())),
])

POINT = 1
POLYGON = 3

Z_MIN, Z_MAX = 2, 5
GRID = 8  # finest tiles drawn from [0, GRID) in x and y (GRID <= 2**Z_MAX)
N_POINTS = 20000
GENE_WEIGHTS = {"G0": 30, "G1": 20, "G2": 14, "G3": 10,
                "G4": 8, "G5": 7, "G6": 6, "G7": 5}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_finest(rng, n=N_POINTS, weights=GENE_WEIGHTS, grid=GRID):
    """Finest-zoom points: contiguous ids, random tiles, weighted random genes."""
    fids = np.arange(n, dtype=np.uint32)
    tx = rng.integers(0, grid, n).astype(np.uint16)
    ty = rng.integers(0, grid, n).astype(np.uint16)
    names = list(weights)
    p = np.asarray([weights[g] for g in names], float)
    genes = rng.choice(np.asarray(names, dtype=object), size=n, p=p / p.sum())
    return fids, tx, ty, genes


def _derive_coarse(level, shift):
    """Apply the tiler's rule: keep fid % 2**shift == 0, shift tile coords."""
    fids, tx, ty, genes = level
    keep = (fids % np.uint32(1 << shift)) == 0
    return (fids[keep], (tx[keep] >> shift).astype(np.uint16),
            (ty[keep] >> shift).astype(np.uint16), genes[keep])


def _point_rows(level, layer="transcripts"):
    fids, tx, ty, genes = level
    n = len(fids)
    tags = [[("layer_type", layer), ("gene_name", str(g))] for g in genes]
    return {
        "tile_x": tx.astype(np.uint16),
        "tile_y": ty.astype(np.uint16),
        "feature_id": fids.astype(np.uint32),
        "geom_type": np.full(n, POINT, np.uint8),
        "positions": [b""] * n,
        "indices": [b""] * n,
        "ring_lengths": [[] for _ in range(n)],
        "tags": tags,
    }


def _polygon_rows(z, n=5, fid_base=10_000_000):
    """A few cell polygons at zoom z (no gene_name tag, layer_type=cells)."""
    return {
        "tile_x": np.arange(n, dtype=np.uint16) % np.uint16(1 << z),
        "tile_y": np.zeros(n, np.uint16),
        "feature_id": (fid_base + np.arange(n)).astype(np.uint32),
        "geom_type": np.full(n, POLYGON, np.uint8),
        "positions": [b"\x00" * 32] * n,
        "indices": [b""] * n,
        "ring_lengths": [[4]] * n,
        "tags": [[("layer_type", "cells"), ("cell_id", f"c{i}")] for i in range(n)],
    }


def _other_point_rows(z, n=3, fid_base=20_000_000):
    """Point rows of a DIFFERENT layer (must be excluded by the layer filter)."""
    return {
        "tile_x": np.zeros(n, np.uint16),
        "tile_y": np.zeros(n, np.uint16),
        "feature_id": (fid_base + np.arange(n)).astype(np.uint32),
        "geom_type": np.full(n, POINT, np.uint8),
        "positions": [b""] * n,
        "indices": [b""] * n,
        "ring_lengths": [[] for _ in range(n)],
        "tags": [[("layer_type", "nuclei"), ("gene_name", "NOT_A_TRANSCRIPT")]] * n,
    }


def _concat_rows(*parts):
    out = {}
    for k in _SCHEMA.names:
        vals = [p[k] for p in parts]
        if isinstance(vals[0], np.ndarray):
            out[k] = np.concatenate(vals)
        else:
            out[k] = sum(vals, [])
    return out


def _write_zoom(root, z, rows, n_parts=1, rng=None):
    zdir = os.path.join(root, f"zoom={z}")
    os.makedirs(zdir, exist_ok=True)
    n = len(rows["feature_id"])
    order = np.arange(n)
    if rng is not None:
        rng.shuffle(order)  # file order must not matter
    bounds = np.linspace(0, n, n_parts + 1).astype(int)
    for i in range(n_parts):
        sel = order[bounds[i]:bounds[i + 1]]
        cols = []
        for name in _SCHEMA.names:
            col = rows[name]
            if isinstance(col, np.ndarray):
                col = col[sel]
            else:
                col = [col[j] for j in sel]
            cols.append(pa.array(col, type=_SCHEMA.field(name).type))
        table = pa.Table.from_arrays(cols, schema=_SCHEMA)
        pq.write_table(table, os.path.join(zdir, f"part_{i:03d}.mu.parquet"))


def write_pyramid(root, finest, *, z_min=Z_MIN, z_max=Z_MAX, overrides=None,
                  rng=None):
    """Write zooms 0..z_max: polygons everywhere, points only in z_min..z_max.

    `overrides` maps zoom -> (fids, tx, ty, genes) to replace the rule-derived
    level (used to inject rule violations).
    """
    overrides = overrides or {}
    for z in range(0, z_max + 1):
        parts = [_polygon_rows(z)]
        if z >= z_min:
            level = overrides.get(z)
            if level is None:
                level = finest if z == z_max else _derive_coarse(finest, z_max - z)
            parts.append(_point_rows(level))
            parts.append(_other_point_rows(z))
        _write_zoom(root, z, _concat_rows(*parts),
                    n_parts=2 if z == z_max else 1, rng=rng)


@pytest.fixture
def uniform_pyramid():
    rng = np.random.default_rng(1234)
    finest = _make_finest(rng)
    root = tempfile.mkdtemp(prefix="lod_uniform_")
    write_pyramid(root, finest, rng=rng)
    return root, finest


def _evaluate(root, **kw):
    kw.setdefault("min_gene_count", 500)
    return evaluate_point_lod(root, **kw)


# ---------------------------------------------------------------------------
# (a) retention + determinism on a rule-conforming pyramid
# ---------------------------------------------------------------------------

def test_retention_and_id_rule_exact(uniform_pyramid):
    root, finest = uniform_pyramid
    res = _evaluate(root)

    assert res["z_max"] == Z_MAX
    assert res["point_zooms"] == list(range(Z_MIN, Z_MAX + 1))
    assert res["pyramid_zooms"] == list(range(0, Z_MAX + 1))
    assert res["finest"]["n_points"] == N_POINTS
    assert res["finest"]["n_genes"] == len(GENE_WEIGHTS)
    assert res["n_genes_total"] == len(GENE_WEIGHTS)
    assert set(res["zooms"]) == {str(z) for z in range(Z_MIN, Z_MAX)}

    for z in range(Z_MIN, Z_MAX):
        r = res["zooms"][str(z)]
        s = Z_MAX - z
        assert r["shift"] == s
        assert r["expected_fraction"] == pytest.approx(2.0 ** -s)
        assert r["n_points"] == len(_derive_coarse(finest, s)[0])
        assert r["retained_fraction"] == pytest.approx(r["expected_fraction"], abs=1e-3)
        assert r["id_rule_exact"] is True
        assert r["n_missing"] == 0
        assert r["n_extra"] == 0
        assert r["n_duplicate_ids"] == 0
        assert r["tile_rule_exact"] is True
        assert r["n_tile_mismatch"] == 0
        assert r["n_cells_extra"] == 0
        # polygons + other-layer points are excluded from the point population
        assert r["n_rows_total"] > r["n_points"]
        assert r["bytes_on_disk"] > 0
        assert r["n_files"] == 1
    assert res["finest"]["n_files"] == 2


# ---------------------------------------------------------------------------
# (b) per-gene and per-cell fidelity are high when genes are id-independent
# ---------------------------------------------------------------------------

def test_gene_and_cell_fidelity_uniform(uniform_pyramid):
    root, _ = uniform_pyramid
    res = _evaluate(root, min_cell_count=20, dropout_threshold=8)

    for z in range(Z_MIN, Z_MAX):
        r = res["zooms"][str(z)]
        assert r["n_genes"] == len(GENE_WEIGHTS)
        assert r["n_genes_eval"] == len(GENE_WEIGHTS)  # all >= 500 at z_max
        assert r["n_genes_lost"] == 0
        assert r["n_genes_new"] == 0
        assert r["median_rel_err"] < 0.15
        assert r["max_rel_err"] < 0.5
        assert r["spearman_genes"] is not None and r["spearman_genes"] > 0.9
        assert r["n_cells"] > 0
        assert r["spearman_cells"] is not None and r["spearman_cells"] > 0.8
        assert r["median_abs_rel_err_cells"] < 0.3
        assert r["dropout_fraction"] == 0.0

    # Halving (z_max-1) must be tight: 10k points remain.
    r = res["zooms"][str(Z_MAX - 1)]
    assert r["max_rel_err"] < 0.2
    assert r["median_abs_rel_err_cells"] < 0.2
    assert r["n_cells"] == pytest.approx(16 * len(GENE_WEIGHTS), abs=0)
    # Coarsest zoom (shift 3) collapses the 8x8 finest grid to a single tile.
    assert res["zooms"][str(Z_MIN)]["n_cells"] == len(GENE_WEIGHTS)


# ---------------------------------------------------------------------------
# (c) biased ids: a gene carried only by odd ids vanishes at every coarse zoom
# ---------------------------------------------------------------------------

def test_biased_gene_is_lost():
    rng = np.random.default_rng(99)
    fids, tx, ty, genes = _make_finest(rng, weights={k: v for k, v in GENE_WEIGHTS.items() if k != "G7"})
    odd = np.flatnonzero(fids % 2 == 1)
    genes = genes.copy()
    genes[rng.choice(odd, 2000, replace=False)] = "ODD"
    finest = (fids, tx, ty, genes)
    root = tempfile.mkdtemp(prefix="lod_biased_")
    write_pyramid(root, finest, rng=rng)

    res = _evaluate(root, dropout_threshold=8)
    assert res["n_genes_total"] == 8
    for z in range(Z_MIN, Z_MAX):
        r = res["zooms"][str(z)]
        # The thinning rule itself is still honoured ...
        assert r["id_rule_exact"] is True
        assert r["retained_fraction"] == pytest.approx(r["expected_fraction"], abs=1e-3)
        # ... but the composition is not preserved.
        assert r["n_genes_lost"] == 1
        assert r["max_rel_err"] == pytest.approx(1.0)
        assert r["max_rel_err_gene"] == "ODD"
        assert r["dropout_fraction"] > 0.0
        assert r["spearman_genes"] is not None

    # At z_max-1 all 16 coarse tiles hold >= 8 ODD points and all drop out.
    r = res["zooms"][str(Z_MAX - 1)]
    assert r["dropout_fraction"] == pytest.approx(16 / r["n_cells_dropout_eval"])


# ---------------------------------------------------------------------------
# (d) rule violation: one expected id dropped, one unexpected id added
# ---------------------------------------------------------------------------

def test_id_rule_violation_detected(uniform_pyramid):
    root, finest = uniform_pyramid
    rng = np.random.default_rng(7)
    fids, tx, ty, genes = _derive_coarse(finest, 1)
    # drop one expected (even) id
    fids, tx, ty, genes = fids[1:], tx[1:], ty[1:], genes[1:]
    # add one unexpected (odd) id with its correctly shifted tile
    f0, tx0, ty0, g0 = finest
    j = int(np.flatnonzero(f0 % 2 == 1)[0])
    fids = np.append(fids, f0[j])
    tx = np.append(tx, tx0[j] >> 1).astype(np.uint16)
    ty = np.append(ty, ty0[j] >> 1).astype(np.uint16)
    genes = np.append(genes, g0[j])
    broken = tempfile.mkdtemp(prefix="lod_broken_")
    write_pyramid(broken, finest, overrides={Z_MAX - 1: (fids, tx, ty, genes)}, rng=rng)

    res = _evaluate(broken)
    r = res["zooms"][str(Z_MAX - 1)]
    assert r["id_rule_exact"] is False
    assert r["n_missing"] == 1
    assert r["n_extra"] == 1
    assert r["n_points"] == N_POINTS // 2
    assert r["tile_rule_exact"] is True  # the tiles of matched ids are still right
    for z in range(Z_MIN, Z_MAX - 1):
        assert res["zooms"][str(z)]["id_rule_exact"] is True


def test_tile_mismatch_detected(uniform_pyramid):
    root, finest = uniform_pyramid
    fids, tx, ty, genes = _derive_coarse(finest, 1)
    tx = tx.copy()
    tx[0] = (int(tx[0]) + 1) % (1 << (Z_MAX - 1))  # move one kept point to a wrong tile
    broken = tempfile.mkdtemp(prefix="lod_tilebroken_")
    write_pyramid(broken, finest, overrides={Z_MAX - 1: (fids, tx, ty, genes)})

    r = _evaluate(broken)["zooms"][str(Z_MAX - 1)]
    assert r["id_rule_exact"] is True
    assert r["tile_rule_exact"] is False
    assert r["n_tile_mismatch"] == 1
    assert r["n_cells_extra"] >= 1


# ---------------------------------------------------------------------------
# degenerate statistics must serialise (no NaN in JSON)
# ---------------------------------------------------------------------------

def test_single_gene_degenerate_spearman():
    rng = np.random.default_rng(3)
    finest = _make_finest(rng, n=2000, weights={"ONLY": 1})
    root = tempfile.mkdtemp(prefix="lod_single_")
    write_pyramid(root, finest, rng=rng)
    res = _evaluate(root, min_gene_count=1)
    r = res["zooms"][str(Z_MAX - 1)]
    assert r["n_genes"] == 1
    assert r["spearman_genes"] is None
    assert r["median_rel_err"] == pytest.approx(0.0)
    text = json.dumps(res, allow_nan=False)  # raises on NaN/inf
    assert "NaN" not in text


# ---------------------------------------------------------------------------
# (e) CLI writes the JSON
# ---------------------------------------------------------------------------

def test_cli_writes_json(uniform_pyramid):
    root, _ = uniform_pyramid
    out = os.path.join(tempfile.mkdtemp(prefix="lod_cli_"), "sub", "lod.json")
    # <repo>/mudm_paper/__init__.py -> <repo>
    repo_root = Path(mudm_paper.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    proc = subprocess.run(
        [sys.executable, "-m", "mudm_paper.transcript_lod",
         "--parquet", root, "--out", out,
         "--min-gene-count", "500", "--min-cell-count", "10",
         "--dropout-threshold", "4", "--batch-size", "4096"],
        capture_output=True, text=True, env=env, cwd=str(repo_root),
    )
    assert proc.returncode == 0, proc.stderr
    with open(out) as f:
        text = f.read()
    assert "NaN" not in text
    res = json.loads(text)
    assert res["z_max"] == Z_MAX
    assert res["params"]["min_gene_count"] == 500
    assert res["params"]["min_cell_count"] == 10
    assert res["params"]["dropout_threshold"] == 4
    assert res["parquet_dir"] == os.path.abspath(root)
    assert str(Z_MAX - 1) in res["zooms"]
    assert "bytes_note" in res
    # in-process entry point agrees with the subprocess run
    out2 = os.path.join(os.path.dirname(out), "lod2.json")
    transcript_lod.main(["--parquet", root, "--out", out2, "--min-gene-count", "500"])
    with open(out2) as f:
        res2 = json.load(f)
    assert res2["zooms"][str(Z_MAX - 1)]["n_points"] == res["zooms"][str(Z_MAX - 1)]["n_points"]
