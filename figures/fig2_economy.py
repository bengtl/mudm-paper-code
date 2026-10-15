#!/usr/bin/env python3
"""Main-text Fig. 2: the right-sizing economy.

Panel a: one representative MANC neuron (near-median size) at every pyramid
         level z0..z4, with its unique vertex / face counts.
Panel b: MANC balanced accuracy versus stored geometry per level (three tasks,
         mean +/- s.e.m. over 15 seeds) from ``results/manc_economy_*.json``.
Panel c: the same for the FlyWire subsample from ``results/flywire_economy_*.json``.

Inputs are deposited artifacts only (no GPU): the economy JSONs and
``data/manc_rep_neuron_lods.npz`` (feature 6546 / body 155112, cell type
IN13A065, extracted from the v0.6.4 MANC pyramid; border-duplicated faces removed).

Usage (from the repository root; needs the ``figures`` extra)::
    python figures/fig2_economy.py [--out DIR]    # default DIR: figures/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _palette import WONG_PALETTE, apply_rcparams  # noqa: E402

RESULTS = SCRIPT_DIR.parent / "results"
DATA = SCRIPT_DIR / "data"
TASKS = {
    "manc_economy": [("neuron_class", "neuron class (7)"), ("subclass", "subclass (28)"),
                     ("hemilineage", "hemilineage (39)")],
    "flywire_economy": [("super_class", "super class (10)"), ("cell_class", "cell class (28)"),
                        ("hemilineage", "hemilineage (70)")],
}
TITLES = {"manc_economy": "MANC, 23,546 neurons", "flywire_economy": "FlyWire, 16,379 neurons"}


def load_economy(prefix: str, results_dir: Path = RESULTS, file_prefix: str | None = None,
                 metric: str = "bal_acc"):
    """`prefix` selects the task list/title; `file_prefix` the artifact stem (default = prefix);
    `metric` = "bal_acc" (final epoch) or "bal_acc_at_best_val" (curve runs)."""
    storage = None
    series = []
    fp = file_prefix or prefix
    for key, label in TASKS[prefix]:
        d = json.load(open(results_dir / f"{fp}_{key}.json"))
        cells = d[key]["precomputed"]
        levels = sorted(cells, key=int)
        agg = [next(iter(cells[l].values()))[metric] for l in levels]
        series.append((label, np.array([a["mean"] for a in agg]), np.array([a["se"] for a in agg])))
        if storage is None:
            s = d["cost"]["storage_per_level"]
            storage = np.array([s[l]["uncompressed_bytes"] for l in sorted(s, key=int)]) / 1e9
    return storage, series


def project(pos: np.ndarray):
    """Rotate a vertex cloud into its principal frame; returns (xy, depth, R)."""
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    r = c @ vt.T
    return r[:, :2], r[:, 2], vt


def draw_mesh(ax, pos, idx, edges: bool, frame=None):
    """Draw a mesh in the principal frame `frame` (vt, centre) shared across levels."""
    vt, centre = frame if frame is not None else (project(pos)[2], pos.mean(0))
    p3 = (pos - centre) @ vt.T
    xy, depth = p3[:, :2], p3[:, 2]
    tri = idx
    n = np.cross(p3[tri[:, 1]] - p3[tri[:, 0]], p3[tri[:, 2]] - p3[tri[:, 0]])
    nz = np.abs(n[:, 2]) / (np.linalg.norm(n, axis=1) + 1e-12)
    order = np.argsort(depth[tri].mean(1))
    shade = 0.30 + 0.60 * nz[order]
    face = np.stack([shade, shade * 0.98, shade * 0.93, np.ones_like(shade)], 1)  # warm grey
    pc = PolyCollection(xy[tri[order]], facecolors=face,
                        edgecolors="#303030", linewidths=(0.45 if edges else 0.10), antialiased=True, rasterized=True)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return xy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR, help="output directory (default: figures/)")
    ap.add_argument("--results", type=Path, default=RESULTS, help="directory holding the economy JSONs")
    ap.add_argument("--manc-prefix", default="manc_economy", help="artifact stem for panel b")
    ap.add_argument("--flywire-prefix", default="flywire_economy", help="artifact stem for panel c")
    ap.add_argument("--metric", default="bal_acc", choices=["bal_acc", "bal_acc_at_best_val"])
    ap.add_argument("--name", default="fig2_economy", help="output file stem")
    args = ap.parse_args()
    out = args.out
    apply_rcparams()

    npz = np.load(DATA / "manc_rep_neuron_lods.npz")
    meta = json.loads(str(npz["meta"]))

    fig = plt.figure(figsize=(7.1, 5.4))
    gs = GridSpec(2, 10, figure=fig, height_ratios=[0.9, 1.35], hspace=0.28, wspace=0.55)
    gs_a = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], wspace=0.04)

    # --- a: one neuron across levels -------------------------------------
    xy4, _, vt4 = project(npz["pos4"])
    frame = (vt4, npz["pos4"].mean(0))
    lim = np.abs(xy4).max() * 1.05
    for z in range(5):
        ax = fig.add_subplot(gs_a[0, z])
        if z < 4:  # finest level ghosted underneath for context
            ghost = PolyCollection(xy4[npz["idx4"]], facecolors="#e4e2dc", edgecolors="none", rasterized=True)
            ax.add_collection(ghost)
        draw_mesh(ax, npz[f"pos{z}"], npz[f"idx{z}"], edges=(z <= 2), frame=frame)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        m = meta[f"z{z}"]
        ax.set_title(f"z{z}{' (coarsest)' if z == 0 else ' (finest)' if z == 4 else ''}\n"
                     f"{m['n_unique_verts']:,} v / {m['n_unique_tris']:,} f", fontsize=7, pad=2)
    fig.text(0.015, 0.95, "a", fontsize=12, fontweight="bold")

    # --- b, c: accuracy vs storage ---------------------------------------
    panels = [("manc_economy", "b", args.manc_prefix), ("flywire_economy", "c", args.flywire_prefix)]
    loaded = [load_economy(prefix, args.results, fp, args.metric) for prefix, _, fp in panels]
    lo = min((m - e).min() for _, series in loaded for _, m, e in series)
    hi = max((m + e).max() for _, series in loaded for _, m, e in series)
    pad = 0.04 * (hi - lo) + 0.02
    for col, ((prefix, letter, _), (storage, series)) in enumerate(zip(panels, loaded)):
        ax = fig.add_subplot(gs[1, 5 * col:5 * col + 5])
        for i, (label, mean, se) in enumerate(series):
            ax.errorbar(storage, mean, yerr=se, color=WONG_PALETTE[i], marker="o", ms=4,
                        lw=1.4, capsize=2, capthick=0.8, elinewidth=0.8, label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(storage)
        ax.set_xticklabels([f"z{i}\n{s:.2g}" for i, s in enumerate(storage)], fontsize=6.5)
        ax.minorticks_off()
        ax.set_xlabel("pyramid level / stored geometry (GB, uncompressed)")
        if col == 0:
            ax.set_ylabel("balanced accuracy")
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{TITLES[prefix]} ({storage[-1] / storage[0]:.0f}× storage span)")
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        ax.legend(loc="best", frameon=False, handlelength=1.6, borderaxespad=0.6)
        fig.text(0.015 if col == 0 else 0.495, 0.50, letter, fontsize=12, fontweight="bold")

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.name}.pdf", bbox_inches="tight", dpi=400)
    fig.savefig(out / f"{args.name}.png", dpi=200, bbox_inches="tight")
    print("wrote", out / f"{args.name}.pdf")


if __name__ == "__main__":
    main()
