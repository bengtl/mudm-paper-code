"""Extended Data Fig. 1: 2D polygon level-of-detail on PanNuke nuclei.

Real, data-driven figure. Reads two artifacts from data/:
  pannuke_2d_lod_fidelity.json  -- storage vs IoU across 38,078 nuclei
  pannuke_rep_nucleus.json      -- one representative nucleus boundary

Panel a: a representative nucleus at successive Douglas-Peucker tolerances
         (full outline in grey under the simplified outline), labelled with
         vertex count + IoU vs full detail.
Panel b: storage-vs-fidelity tradeoff (mean IoU vs retained-vertex storage %,
         log x) with a 5th-percentile worst-case band.

Simplification uses the same reduce-to-floor Douglas-Peucker as the tiling
pipeline (identical Rust and Python implementations; see
mudm_tools/mudm2vt/simplify.py and rust/src/simplify2d.rs), embedded here so
the figure is reproducible without the pipeline installed.

The data files are derived from PanNuke and carry its CC BY-NC-SA 4.0 license
(see LICENSE-DATA.md).

Usage (from the repository root; needs the ``figures`` extra)::
    python figures/fig_s_2d_lod.py [--out DIR]    # default DIR: figures/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _palette import WONG_PALETTE, apply_rcparams  # noqa: E402

DATA = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR  # figures/
BLUE = WONG_PALETTE[0]
GREY = "#9a9a9a"


# --- reduce-to-floor Douglas-Peucker (parity with the pipeline) -------------
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


def _dp_floor(pts, eps, mv):
    n = len(pts)
    fl = max(mv, 2)
    if n <= fl:
        return [[p[0], p[1]] for p in pts]
    imp = [0.0] * n
    imp[0] = math.inf
    imp[n - 1] = math.inf
    _dp_importance(pts, imp)
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
    return [[pts[i][0], pts[i][1]] for i in range(n) if keep[i]]


def _simp_ring(ring, eps, mv=4):
    fl = max(mv, 4)
    if len(ring) <= fl:
        return [[p[0], p[1]] for p in ring]
    return _dp_floor(ring, eps, fl)


def _iou(a, b):
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extended Data Fig. 1: 2D polygon LOD on PanNuke nuclei.")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory (default: figures/)")
    ap.add_argument("--data", type=Path, default=DATA, help="directory holding the pannuke_*.json inputs")
    args = ap.parse_args(argv)
    out_dir = args.out
    apply_rcparams()
    fid = json.loads((args.data / "pannuke_2d_lod_fidelity.json").read_text())
    rep = json.loads((args.data / "pannuke_rep_nucleus.json").read_text())
    ring = [(p[0], p[1]) for p in rep["ring"]]

    fig = plt.figure(figsize=(7.1, 4.6))
    gs = GridSpec(2, 5, height_ratios=[1.0, 1.5], hspace=0.45, wspace=0.25,
                  top=0.9, bottom=0.13, left=0.09, right=0.98)

    # Panel a: nucleus at LOD levels
    tol_thumbs = [0.0, 0.5, 1.0, 4.0, 8.0]
    cx = float(np.mean([p[0] for p in ring]))
    cy = float(np.mean([p[1] for p in ring]))
    fx = [p[0] for p in ring]
    fy = [p[1] for p in ring]
    for j, t in enumerate(tol_thumbs):
        ax = fig.add_subplot(gs[0, j])
        simp = ring if t == 0 else [tuple(p) for p in _simp_ring(ring, t, 4)]
        ax.plot(fx, fy, color=GREY, lw=0.6, zorder=1)
        sx = [p[0] for p in simp] + [simp[0][0]]
        sy = [p[1] for p in simp] + [simp[0][1]]
        ax.fill(sx, sy, color=BLUE, alpha=0.15, zorder=2)
        ax.plot(sx, sy, color=BLUE, lw=1.4, zorder=3)
        ax.plot([p[0] for p in simp], [p[1] for p in simp], "o", color=BLUE, ms=1.6, zorder=4)
        nv = len(simp)
        iu = 1.0 if t == 0 else _iou(ring, simp)
        lab = "full" if t == 0 else f"$\\tau$={t:g} px"
        ax.set_title(f"{lab}\n{nv} v · IoU {iu:.2f}", fontsize=7, pad=3)
        r = 1.15 * max(max(fx) - cx, cx - min(fx), max(fy) - cy, cy - min(fy))
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.015, 0.955, "a", fontsize=12, fontweight="bold")

    # Panel b: storage vs fidelity
    axb = fig.add_subplot(gs[1, :])
    L = [lv for lv in fid["levels"] if lv["eps_px"] > 0]
    stor = np.array([lv["storage_pct"] for lv in L])
    miou = np.array([lv["mean_iou"] for lv in L])
    p05 = np.array([lv.get("p05_iou", m) for lv, m in zip(L, miou)])
    tol = [lv["eps_px"] for lv in L]
    axb.fill_between(stor, p05, miou, color=BLUE, alpha=0.15, lw=0,
                     label="5th percentile band")
    axb.plot(stor, miou, "-o", color=BLUE, lw=1.6, ms=5, label="mean IoU")
    for s, m, t in zip(stor, miou, tol):
        axb.annotate(f"{t:g} px", (s, m), textcoords="offset points",
                     xytext=(4, 7), fontsize=7, color=BLUE)
    i1 = tol.index(1.0)
    axb.annotate("~9x fewer vertices\nat IoU $\\geq$ 0.95",
                 (stor[i1], miou[i1]), xytext=(stor[i1] * 2.2, 0.66),
                 fontsize=7.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))
    axb.set_xscale("log")
    axb.set_xlabel("stored geometry (% of full-detail vertices)")
    axb.set_ylabel("geometric fidelity (mean IoU)")
    axb.set_ylim(0.55, 1.02)
    axb.set_xticks([5, 10, 20, 40, 100])
    axb.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axb.grid(True, which="both", alpha=0.25, lw=0.5)
    axb.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.text(0.015, 0.60, "b", fontsize=12, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_s_2d_lod.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_s_2d_lod.png", dpi=200, bbox_inches="tight")
    print("wrote", out_dir / "fig_s_2d_lod.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
