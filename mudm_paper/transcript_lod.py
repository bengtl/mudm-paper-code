"""Level-of-detail fidelity evaluation for tiled 2D point (transcript) layers.

The muDM 2D tiler thins POINT features at coarser zooms deterministically by
feature id (`mudm-tools/rust/src/streaming2d.rs`):

    skip_factor = 1 << (max_zoom - z);  keep iff feature_id % skip_factor == 0

so zoom z_max-1 keeps about one half of the points, z_max-2 one quarter, and
so on. Points are never clipped across tiles, so the point that lives in the
finest tile (tx, ty) lives in tile (tx >> s, ty >> s) at zoom z_max - s.

This module asks what that scheme preserves for downstream analysis. For every
coarse zoom that contains points, all relative to the finest zoom, it reports

  1. retention: n_points, retained_fraction vs expected_fraction (2**-s);
  2. determinism: is the kept id set EXACTLY the modulo rule
     (id_rule_exact, n_missing, n_extra), and do kept points sit in the
     tile the shift predicts (tile_rule_exact, n_tile_mismatch);
  3. global per-gene composition: proportion errors for well-populated
     genes, Spearman over per-gene counts, genes lost;
  4. per-tile per-gene counts on the coarse tile grid (finest counts are
     re-binned onto that grid): Spearman over cells, relative error of the
     up-scaled coarse count, dropout of well-populated cells;
  5. bytes on disk and row totals per zoom (part files also hold polygons).

Reading is streamed one zoom at a time through pyarrow.dataset with only the
needed columns projected, a geom_type filter pushed down, and a vectorised
map_lookup for the tag values; per-row work is numpy on integer codes. Only
the finest-zoom arrays (feature_id, tile coords, gene codes) stay in memory.

CLI:  python -m mudm_paper.transcript_lod --parquet <dir> --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

READ_COLUMNS = ["feature_id", "tile_x", "tile_y", "geom_type", "tags"]
DEFAULT_BATCH_SIZE = 1 << 18

BYTES_NOTE = (
    "bytes_on_disk sums the sizes of all part files in the zoom directory; "
    "those files also hold the polygon (cell/nucleus) rows of that zoom, so "
    "this is the cost of the whole zoom, not of the point layer alone."
)

_ZOOM_DIR_RE = re.compile(r"^zoom=(\d+)$")


# ---------------------------------------------------------------------------
# Pyramid layout
# ---------------------------------------------------------------------------

def list_zoom_dirs(parquet_dir: str) -> dict[int, str]:
    """Map zoom -> `zoom=<Z>` directory for a Hive-partitioned pyramid."""
    out: dict[int, str] = {}
    for name in os.listdir(parquet_dir):
        m = _ZOOM_DIR_RE.match(name)
        path = os.path.join(parquet_dir, name)
        if m and os.path.isdir(path):
            out[int(m.group(1))] = path
    if not out:
        raise FileNotFoundError(f"no zoom=<Z> directories under {parquet_dir}")
    return out


def _part_files(zoom_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(zoom_dir, "*.parquet")))


# ---------------------------------------------------------------------------
# Gene vocabulary + streamed loading
# ---------------------------------------------------------------------------

class GeneVocab:
    """Incremental gene name -> integer code map; codes never change."""

    def __init__(self) -> None:
        self._codes: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._codes)

    @property
    def names(self) -> list[str]:
        return list(self._codes)  # insertion order == code order

    def code(self, name: str) -> int:
        c = self._codes.get(name)
        if c is None:
            c = len(self._codes)
            self._codes[name] = c
        return c

    def encode(self, values: pa.Array) -> np.ndarray:
        """Vectorised: dictionary-encode the batch, map its (small) dictionary
        to global codes once, gather per row. Nulls become -1."""
        d = pc.dictionary_encode(values)
        dictionary = d.dictionary.to_pylist()
        lut = np.fromiter((self.code(v) for v in dictionary), dtype=np.int32,
                          count=len(dictionary))
        idx = pc.fill_null(d.indices, -1).to_numpy(zero_copy_only=False)
        codes = np.full(len(idx), -1, dtype=np.int32)
        valid = idx >= 0
        if valid.any():
            codes[valid] = lut[idx[valid]]
        return codes


@dataclass
class PointArrays:
    fid: np.ndarray    # uint32
    tx: np.ndarray     # uint16
    ty: np.ndarray     # uint16
    gene: np.ndarray   # int32 code, -1 when the gene tag is missing

    @property
    def n(self) -> int:
        return int(self.fid.size)

    @property
    def n_without_gene(self) -> int:
        return int(np.count_nonzero(self.gene < 0))


def load_zoom_points(
    files: list[str],
    *,
    vocab: GeneVocab,
    point_geom_type: int = 1,
    layer_key: str | None = "layer_type",
    layer_value: str = "transcripts",
    gene_key: str = "gene_name",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PointArrays:
    """Stream the point population of one zoom into compact numpy arrays.

    Population = rows with geom_type == point_geom_type AND (when layer_key is
    truthy) tags[layer_key] == layer_value. The geom_type predicate is pushed
    into the pyarrow scanner; the tag predicates use pc.map_lookup.
    """
    fids, txs, tys, genes = [], [], [], []
    if not files:
        return _empty_points()
    dataset = ds.dataset(files, format="parquet")
    scanner = dataset.to_batches(
        columns=READ_COLUMNS,
        filter=ds.field("geom_type") == point_geom_type,
        batch_size=batch_size,
        batch_readahead=2,
        fragment_readahead=1,
    )
    for batch in scanner:
        if batch.num_rows == 0:
            continue
        tags = batch.column("tags")
        if layer_key:
            layer = pc.map_lookup(tags, query_key=layer_key, occurrence="first")
            mask = pc.fill_null(pc.equal(layer, layer_value), False)
            batch = batch.filter(mask)
            if batch.num_rows == 0:
                continue
            tags = batch.column("tags")
        gene = pc.map_lookup(tags, query_key=gene_key, occurrence="first")
        genes.append(vocab.encode(gene))
        fids.append(batch.column("feature_id").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False))
        txs.append(batch.column("tile_x").to_numpy(zero_copy_only=False).astype(np.uint16, copy=False))
        tys.append(batch.column("tile_y").to_numpy(zero_copy_only=False).astype(np.uint16, copy=False))
    if not fids:
        return _empty_points()
    return PointArrays(
        fid=np.concatenate(fids),
        tx=np.concatenate(txs),
        ty=np.concatenate(tys),
        gene=np.concatenate(genes),
    )


def _empty_points() -> PointArrays:
    return PointArrays(
        fid=np.zeros(0, np.uint32), tx=np.zeros(0, np.uint16),
        ty=np.zeros(0, np.uint16), gene=np.zeros(0, np.int32),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _finite_or_none(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rho; None when undefined (< 2 samples or a constant input)."""
    if a.size < 2 or b.size < 2:
        return None
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    rho = spearmanr(a, b)[0]
    return _finite_or_none(rho)


def _gene_counts(gene_codes: np.ndarray, n_vocab: int) -> np.ndarray:
    valid = gene_codes[gene_codes >= 0]
    return np.bincount(valid, minlength=n_vocab).astype(np.int64)


def _id_rule_stats(fid_sorted_max: np.ndarray, coarse: PointArrays, shift: int) -> dict:
    """Is the coarse id set exactly {fid at z_max : fid % 2**shift == 0}?"""
    mask = np.uint32((1 << shift) - 1)
    expected = np.unique(fid_sorted_max[(fid_sorted_max & mask) == 0])
    observed = np.unique(coarse.fid)
    n_common = np.intersect1d(expected, observed, assume_unique=True).size
    n_missing = int(expected.size - n_common)
    n_extra = int(observed.size - n_common)
    n_dup = int(coarse.fid.size - observed.size)
    return {
        "id_rule_exact": bool(n_missing == 0 and n_extra == 0 and n_dup == 0),
        "n_missing": n_missing,
        "n_extra": n_extra,
        "n_duplicate_ids": n_dup,
    }


def _tile_rule_stats(finest: PointArrays, order: np.ndarray, fid_sorted: np.ndarray,
                     coarse: PointArrays, shift: int) -> dict:
    """For coarse ids that exist at z_max: does (tx, ty) == finest >> shift?"""
    if coarse.n == 0 or fid_sorted.size == 0:
        return {"tile_rule_exact": True, "n_tile_mismatch": 0}
    pos = np.searchsorted(fid_sorted, coarse.fid)
    pos = np.minimum(pos, fid_sorted.size - 1)
    found = fid_sorted[pos] == coarse.fid
    src = order[pos[found]]
    mismatch = ((finest.tx[src] >> shift) != coarse.tx[found]) | \
               ((finest.ty[src] >> shift) != coarse.ty[found])
    n_mismatch = int(np.count_nonzero(mismatch))
    return {"tile_rule_exact": bool(n_mismatch == 0), "n_tile_mismatch": n_mismatch}


def _gene_stats(counts_max: np.ndarray, counts_z: np.ndarray, n_max: int, n_z: int,
                min_gene_count: int, names: list[str]) -> dict:
    present = counts_max > 0
    eval_mask = counts_max >= max(int(min_gene_count), 1)
    out = {
        "n_genes": int(np.count_nonzero(present)),
        "n_genes_eval": int(np.count_nonzero(eval_mask)),
        "n_genes_lost": int(np.count_nonzero(present & (counts_z == 0))),
        "n_genes_new": int(np.count_nonzero(~present & (counts_z > 0))),
        "median_rel_err": None,
        "max_rel_err": None,
        "max_rel_err_gene": None,
        "spearman_genes": _spearman(counts_max[present], counts_z[present]),
    }
    if eval_mask.any() and n_max > 0:
        p_max = counts_max[eval_mask] / float(n_max)
        p_z = counts_z[eval_mask] / float(n_z) if n_z > 0 else np.zeros_like(p_max)
        rel = np.abs(p_z - p_max) / p_max
        worst = int(np.argmax(rel))
        out["median_rel_err"] = float(np.median(rel))
        out["max_rel_err"] = float(rel[worst])
        out["max_rel_err_gene"] = names[int(np.flatnonzero(eval_mask)[worst])]
    return out


def _cell_keys(p: PointArrays, shift: int, n_vocab: int) -> np.ndarray:
    """Composite int64 key (coarse tile, gene) for rows that carry a gene."""
    has_gene = p.gene >= 0
    tx = p.tx[has_gene].astype(np.int64) >> shift
    ty = p.ty[has_gene].astype(np.int64) >> shift
    return ((tx << 16) | ty) * np.int64(max(n_vocab, 1)) + p.gene[has_gene].astype(np.int64)


def _cell_stats(finest: PointArrays, coarse: PointArrays, shift: int, n_vocab: int,
                min_cell_count: int, dropout_threshold: int) -> dict:
    keys_f, cnt_f = np.unique(_cell_keys(finest, shift, n_vocab), return_counts=True)
    keys_c, cnt_c = np.unique(_cell_keys(coarse, 0, n_vocab), return_counts=True)
    out = {
        "n_cells": int(keys_f.size),
        "n_cells_eval": 0,
        "n_cells_dropout_eval": 0,
        "n_cells_extra": 0,
        "spearman_cells": None,
        "median_abs_rel_err_cells": None,
        "dropout_fraction": None,
    }
    if keys_f.size == 0:
        out["n_cells_extra"] = int(keys_c.size)
        return out
    pos = np.minimum(np.searchsorted(keys_f, keys_c), keys_f.size - 1)
    hit = keys_f[pos] == keys_c
    coarse_on_fine = np.zeros(keys_f.size, dtype=np.int64)
    coarse_on_fine[pos[hit]] = cnt_c[hit]
    out["n_cells_extra"] = int(np.count_nonzero(~hit))
    out["spearman_cells"] = _spearman(cnt_f, coarse_on_fine)

    scaled = coarse_on_fine * np.int64(1 << shift)
    ev = cnt_f >= max(int(min_cell_count), 1)
    out["n_cells_eval"] = int(np.count_nonzero(ev))
    if ev.any():
        rel = np.abs(scaled[ev] - cnt_f[ev]) / cnt_f[ev]
        out["median_abs_rel_err_cells"] = float(np.median(rel))
    dv = cnt_f >= max(int(dropout_threshold), 1)
    out["n_cells_dropout_eval"] = int(np.count_nonzero(dv))
    if dv.any():
        out["dropout_fraction"] = float(np.mean(coarse_on_fine[dv] == 0))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _zoom_disk_stats(zoom_dir: str, files: list[str], point_geom_type: int) -> dict:
    dataset = ds.dataset(files, format="parquet") if files else None
    return {
        "n_files": len(files),
        "bytes_on_disk": int(sum(os.path.getsize(f) for f in files)),
        "n_rows_total": int(dataset.count_rows()) if dataset else 0,
        "n_point_geom_rows": int(dataset.count_rows(
            filter=ds.field("geom_type") == point_geom_type)) if dataset else 0,
    }


def evaluate_point_lod(
    parquet_dir: str,
    *,
    point_geom_type: int = 1,
    layer_key: str | None = "layer_type",
    layer_value: str = "transcripts",
    gene_key: str = "gene_name",
    min_gene_count: int = 1000,
    min_cell_count: int = 20,
    dropout_threshold: int = 8,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Evaluate the point LOD scheme of a 2D muDM pyramid; see module doc.

    Returns a JSON-serialisable dict: top-level metadata + `finest` (the
    reference zoom z_max) + `zooms` keyed by the coarse zoom as a string.
    """
    t0 = time.time()
    parquet_dir = os.path.abspath(parquet_dir)
    params = {
        "point_geom_type": int(point_geom_type),
        "layer_key": layer_key or None,
        "layer_value": layer_value,
        "gene_key": gene_key,
        "min_gene_count": int(min_gene_count),
        "min_cell_count": int(min_cell_count),
        "dropout_threshold": int(dropout_threshold),
        "batch_size": int(batch_size),
    }
    load_kw = dict(point_geom_type=point_geom_type, layer_key=layer_key,
                   layer_value=layer_value, gene_key=gene_key, batch_size=batch_size)

    zoom_dirs = list_zoom_dirs(parquet_dir)
    pyramid_zooms = sorted(zoom_dirs)
    disk = {}
    for z in pyramid_zooms:
        files = _part_files(zoom_dirs[z])
        disk[z] = _zoom_disk_stats(zoom_dirs[z], files, point_geom_type)
        disk[z]["files"] = files
        log.info("zoom %d: %d files, %d rows, %d point-geometry rows, %.1f MB",
                 z, disk[z]["n_files"], disk[z]["n_rows_total"],
                 disk[z]["n_point_geom_rows"], disk[z]["bytes_on_disk"] / 1e6)
    candidates = [z for z in reversed(pyramid_zooms) if disk[z]["n_point_geom_rows"] > 0]
    if not candidates:
        raise ValueError(f"no rows with geom_type == {point_geom_type} in {parquet_dir}")

    vocab = GeneVocab()

    # --- finest zoom: the reference, kept in memory --------------------------
    finest = None
    z_max = None
    for z in candidates:
        t = time.time()
        pts = load_zoom_points(disk[z]["files"], vocab=vocab, **load_kw)
        log.info("zoom %d: loaded %d points in %.1fs", z, pts.n, time.time() - t)
        if pts.n > 0:
            finest, z_max = pts, z
            break
    if finest is None:
        raise ValueError(
            f"no point rows with {layer_key}=={layer_value!r} in {parquet_dir}")
    n_max = finest.n
    order = np.argsort(finest.fid, kind="stable")
    fid_sorted = finest.fid[order]
    counts_max = _gene_counts(finest.gene, len(vocab))
    present_max = counts_max > 0

    finest_block = {
        "zoom": int(z_max),
        "n_points": n_max,
        "n_points_without_gene": finest.n_without_gene,
        "n_unique_ids": int(np.unique(fid_sorted).size),
        "n_genes": int(np.count_nonzero(present_max)),
        "n_genes_eval": int(np.count_nonzero(counts_max >= max(int(min_gene_count), 1))),
        **{k: disk[z_max][k] for k in ("n_files", "bytes_on_disk", "n_rows_total")},
    }

    # --- coarse zooms, one at a time -----------------------------------------
    zooms: dict[str, dict] = {}
    point_zooms = [z_max]
    for z in candidates:
        if z >= z_max:
            continue
        t = time.time()
        shift = z_max - z
        coarse = load_zoom_points(disk[z]["files"], vocab=vocab, **load_kw)
        if coarse.n == 0:
            log.info("zoom %d: no points after layer filter, skipped", z)
            continue
        point_zooms.append(z)
        n_vocab = len(vocab)
        if counts_max.size < n_vocab:  # vocab grew (unexpected genes at z)
            counts_max = _gene_counts(finest.gene, n_vocab)
        counts_z = _gene_counts(coarse.gene, n_vocab)

        entry = {
            "zoom": int(z),
            "shift": int(shift),
            "n_points": coarse.n,
            "n_points_without_gene": coarse.n_without_gene,
            "retained_fraction": coarse.n / n_max,
            "expected_fraction": 2.0 ** (-shift),
        }
        entry.update(_id_rule_stats(fid_sorted, coarse, shift))
        entry.update(_tile_rule_stats(finest, order, fid_sorted, coarse, shift))
        entry.update(_gene_stats(counts_max, counts_z, n_max, coarse.n,
                                 min_gene_count, vocab.names))
        entry.update(_cell_stats(finest, coarse, shift, n_vocab,
                                 min_cell_count, dropout_threshold))
        entry.update({k: disk[z][k] for k in ("n_files", "bytes_on_disk", "n_rows_total")})
        zooms[str(z)] = entry
        log.info("zoom %d: %d points (%.4f of finest, expected %.4f), id_rule_exact=%s, "
                 "genes lost=%d, cell spearman=%s, dropout=%s [%.1fs]",
                 z, coarse.n, entry["retained_fraction"], entry["expected_fraction"],
                 entry["id_rule_exact"], entry["n_genes_lost"], entry["spearman_cells"],
                 entry["dropout_fraction"], time.time() - t)
        del coarse

    return {
        "parquet_dir": parquet_dir,
        "z_max": int(z_max),
        "point_zooms": sorted(point_zooms),
        "pyramid_zooms": pyramid_zooms,
        "n_genes_total": len(vocab),
        "params": params,
        "bytes_note": BYTES_NOTE,
        "finest": finest_block,
        "zooms": zooms,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _summary_lines(res: dict) -> list[str]:
    lines = [f"finest zoom {res['z_max']}: {res['finest']['n_points']} points, "
             f"{res['finest']['n_genes']} genes, "
             f"{res['finest']['bytes_on_disk'] / 1e6:.1f} MB on disk"]
    hdr = (f"{'z':>3} {'n_points':>10} {'kept':>7} {'exp':>7} {'idrule':>6} "
           f"{'lost':>5} {'medRel':>7} {'maxRel':>7} {'rhoG':>6} {'rhoC':>6} "
           f"{'medCell':>8} {'dropout':>8} {'MB':>8}")
    lines.append(hdr)
    fmt = lambda v, f: ("   n/a" if v is None else format(v, f))  # noqa: E731
    for z in sorted(res["zooms"], key=int):
        r = res["zooms"][z]
        lines.append(
            f"{z:>3} {r['n_points']:>10d} {r['retained_fraction']:>7.4f} "
            f"{r['expected_fraction']:>7.4f} {str(r['id_rule_exact']):>6} "
            f"{r['n_genes_lost']:>5d} {fmt(r['median_rel_err'], '7.4f'):>7} "
            f"{fmt(r['max_rel_err'], '7.4f'):>7} {fmt(r['spearman_genes'], '6.3f'):>6} "
            f"{fmt(r['spearman_cells'], '6.3f'):>6} "
            f"{fmt(r['median_abs_rel_err_cells'], '8.4f'):>8} "
            f"{fmt(r['dropout_fraction'], '8.4f'):>8} {r['bytes_on_disk'] / 1e6:>8.1f}"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate what the 2D point LOD scheme (feature_id % 2**(z_max-z)) "
                    "preserves, per coarse zoom, relative to the finest zoom.")
    p.add_argument("--parquet", required=True, help="pyramid root holding zoom=<Z>/ dirs")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--point-geom-type", type=int, default=1)
    p.add_argument("--layer-key", default="layer_type",
                   help="tag key selecting the point layer; '' disables the layer filter")
    p.add_argument("--layer-value", default="transcripts")
    p.add_argument("--gene-key", default="gene_name")
    p.add_argument("--min-gene-count", type=int, default=1000)
    p.add_argument("--min-cell-count", type=int, default=20)
    p.add_argument("--dropout-threshold", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--quiet", action="store_true", help="suppress progress logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.quiet:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
    res = evaluate_point_lod(
        args.parquet,
        point_geom_type=args.point_geom_type,
        layer_key=args.layer_key or None,
        layer_value=args.layer_value,
        gene_key=args.gene_key,
        min_gene_count=args.min_gene_count,
        min_cell_count=args.min_cell_count,
        dropout_threshold=args.dropout_threshold,
        batch_size=args.batch_size,
    )
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, allow_nan=False)
    print("\n".join(_summary_lines(res)))
    print(f"wrote {args.out} ({res['elapsed_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
