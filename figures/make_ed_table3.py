#!/usr/bin/env python3
"""Regenerate Extended Data Table 3 (feature robustness across 2D polygon
levels of detail on PanNuke nuclei) from ``results/nyxus_stability_summary.json``.

The summary is written by ``scripts/stability_analysis.py``: for each
Douglas-Peucker tolerance tau (0.5, 1, 2, 4, 8 px) and each of the 144 Nyxus
features it stores the median absolute relative deviation from full detail
(tau = 0) and the Spearman correlation of per-nucleus values, both rounded to
four decimals.

Printed sections:

1. Extended Data Table 3: the curated nuclear morphometry and intensity
   features, grouped by regime, at tau = 1, 2 and 8 px (three decimals).
2. Extended Data Table 3 (continued): the eight features with the poorest
   rank stability at tau = 2 px (two decimals), plus the smallest rho among
   the remaining features with defined correlations.
3. Summary statistics quoted in the legend and text: the median deviation and
   median rho over all features at tau = 1 px, and the mean counterparts of
   the vertex-sensitive diameter statistics.

``--full`` additionally prints every feature at every level (the complete
feature-by-feature table the legend refers to); ``--csv PATH`` writes it as CSV.

Pure standard library. Usage (from the repository root)::

    python figures/make_ed_table3.py [--full] [--csv nyxus_stability_full.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = REPO_ROOT / "results" / "nyxus_stability_summary.json"

# Rows of the main panel, in manuscript order, grouped by observed regime.
GROUPS = [
    ("Size, ellipse shape, and extremal diameters (robust)",
     ["STAT_FERET_DIAM_MAX", "ASPECT_RATIO", "ECCENTRICITY", "ELONGATION",
      "ROUNDNESS", "COMPACTNESS"]),
    ("Intensity aggregates (robust; region membership shifts slightly)",
     ["MEAN", "MEDIAN", "STANDARD_DEVIATION", "ENTROPY"]),
    ("Area-coupled magnitudes (systematic shrinkage; rank-preserving)",
     ["AREA_PIXELS_COUNT", "INTEGRATED_INTENSITY"]),
    ("Boundary detail (degrades first)",
     ["PERIMETER", "CIRCULARITY", "EXTENT", "CONVEX_HULL_AREA", "SOLIDITY"]),
]
MAIN_TAUS = ["1", "2", "8"]
CONT_TAUS = ["1", "2"]
N_WORST = 8
# Vertex-sensitive diameter statistics and their mean counterparts.
MEAN_COUNTERPARTS = ["STAT_NASSENSTEIN_DIAM_MEAN", "STAT_MARTIN_DIAM_MEAN", "STAT_FERET_DIAM_MEAN"]


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def fmt(x, nd: int) -> str:
    return f"{x:.{nd}f}" if _finite(x) else "n/a"


def level(summary: dict, tau: str) -> dict:
    return summary["levels"][f"tau_{tau}"]


def worst_at(summary: dict, tau: str, n: int) -> tuple[list[str], float | None, str | None]:
    """The n features with the lowest defined rho at `tau`, and the smallest
    rho (with its feature) among the remaining defined features."""
    rho = level(summary, tau)["spearman"]
    defined = sorted((v, k) for k, v in rho.items() if _finite(v))
    worst = [k for _, k in defined[:n]]
    rest = defined[n:]
    return worst, (rest[0][0] if rest else None), (rest[0][1] if rest else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                    help="nyxus stability summary JSON (default: results/nyxus_stability_summary.json)")
    ap.add_argument("--full", action="store_true",
                    help="also print every feature at every level")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write the complete feature-by-feature table to this CSV")
    args = ap.parse_args(argv)
    s = json.loads(args.summary.read_text())
    taus = [k.removeprefix("tau_") for k in s["levels"]]
    n_matched = sorted({lv["n_matched"] for lv in s["levels"].values()})

    # --- 1. Main panel ------------------------------------------------------
    print(f"\nExtended Data Table 3: feature robustness across polygon levels of detail "
          f"(PanNuke; {s['n_base_instances']:,} nuclei at full detail, matched per level: "
          f"{', '.join(f'{n:,}' for n in n_matched)}; {s['n_features']} features)")
    print("dev. = median absolute relative deviation from tau = 0; rho = Spearman across nuclei\n")
    head = ["Feature"]
    for t in MAIN_TAUS:
        head += [f"τ = {t} px dev.", "ρ"]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for title, feats in GROUPS:
        print(f"| *{title}* |" + " |" * (len(head) - 1))
        for f in feats:
            cells = [f"`{f}`"]
            for t in MAIN_TAUS:
                lv = level(s, t)
                cells += [fmt(lv["median_abs_rel_dev"].get(f), 3), fmt(lv["spearman"].get(f), 3)]
            print("| " + " | ".join(cells) + " |")

    # --- 2. Continued panel -------------------------------------------------
    worst, next_rho, next_feat = worst_at(s, "2", N_WORST)
    print(f"\nExtended Data Table 3 (continued): the {N_WORST} features with the poorest "
          f"rank stability at τ = 2 px\n")
    head = ["Feature"]
    for t in CONT_TAUS:
        head += [f"τ = {t} px dev.", "ρ"]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for f in worst:
        cells = [f"`{f}`"]
        for t in CONT_TAUS:
            lv = level(s, t)
            cells += [fmt(lv["median_abs_rel_dev"].get(f), 2), fmt(lv["spearman"].get(f), 2)]
        print("| " + " | ".join(cells) + " |")
    if next_rho is not None:
        print(f"\nSmallest ρ at τ = 2 px among the remaining defined features: "
              f"{next_rho:.4f} ({next_feat})")
    undefined = [k for k, v in level(s, "2")["spearman"].items() if not _finite(v)]
    if undefined:
        print(f"Features without a defined ρ at τ = 2 px: {', '.join(sorted(undefined))}")

    # --- 3. Summary statistics ----------------------------------------------
    lv1 = level(s, "1")
    devs = [v for v in lv1["median_abs_rel_dev"].values() if _finite(v)]
    rhos = [v for v in lv1["spearman"].values() if _finite(v)]
    print(f"\nAt τ = 1 px over all features: median dev. = {statistics.median(devs):.4f} "
          f"({len(devs)} defined), median ρ = {statistics.median(rhos):.4f} ({len(rhos)} defined)")
    print("Mean counterparts of the vertex-sensitive diameters (ρ at τ = 1 / 2 px):")
    for f in MEAN_COUNTERPARTS:
        r1, r2 = lv1["spearman"].get(f), level(s, "2")["spearman"].get(f)
        print(f"  {f:28s} {fmt(r1, 4)} / {fmt(r2, 4)}")

    # --- Complete table -----------------------------------------------------
    feats = sorted(s["levels"][f"tau_{taus[0]}"]["spearman"])
    if args.full:
        print("\nComplete feature-by-feature table (dev. / ρ per tau)\n")
        head = ["Feature"] + [f"τ = {t} px dev. / ρ" for t in taus]
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for f in feats:
            cells = [f"`{f}`"] + [
                f"{fmt(level(s, t)['median_abs_rel_dev'].get(f), 4)} / "
                f"{fmt(level(s, t)['spearman'].get(f), 4)}" for t in taus]
            print("| " + " | ".join(cells) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["feature"] + [f"{m}_tau_{t}" for t in taus for m in ("dev", "rho")])
            for f in feats:
                vals = []
                for t in taus:
                    lv = level(s, t)
                    vals += [lv["median_abs_rel_dev"].get(f), lv["spearman"].get(f)]
                w.writerow([f] + [x if _finite(x) else "" for x in vals])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
