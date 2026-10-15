#!/usr/bin/env python3
"""Regenerate Extended Data Table 4 (multimodal fusion) and the fusion
statistics quoted in Results and Methods from the deposited artifacts.

Inputs, all in ``results/`` (one file per task)::

    manc_economy_<task>.json          geometry-only DGCNN (Table 1 arm)
    manc_fusion_<task>.json           DGCNN + four linked synapse-count features
    manc_countsonly_<task>.json       counts-only MLP
    manc_fusionperm_<task>.json       fusion with counts permuted among neurons
    manc_leakctrl_<task>.json         geometry-only, cell-type-held-out split
    manc_fusion_leakctrl_<task>.json  fusion, cell-type-held-out split

Printed sections:

1. Extended Data Table 4: per task and level, the geometry, counts-only and
   fusion seed means, Δ = fusion - geometry, the paired t-test p over the 15
   shared seeds, and the Holm-adjusted p across the five levels of each task.
2. Substitution test: fusion at the coarsest level versus geometry alone at
   the finest level (paired t over seeds).
3. Permutation control: permuted-counts fusion versus geometry alone per
   cell, with the maximum |Δ| and the number of Holm-significant cells.
4. Held-out control: fusion versus geometry under the cell-type-held-out
   split, with per-task Δ ranges, the largest p and the Holm-significant
   levels.

Every arm uses the same seeds (0..14) and splits, so per-seed values pair by
position. Statistics come from ``mudm_paper.stats`` (``paired_t``, ``holm``).

Usage (from the repository root)::

    python figures/make_ed_table4.py
    python figures/make_ed_table4.py --suffix 30 --metric bal_acc_at_best_val \\
        --results <dir_with_30_epoch_artifacts>   # 30-epoch recipe variant
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(REPO_ROOT))
from mudm_paper.stats import holm, paired_t  # noqa: E402

DEFAULT_TASKS = ["neuron_class", "subclass", "hemilineage"]
TASK_NAMES = {
    "neuron_class": "neuron class",
    "subclass": "subclass",
    "hemilineage": "hemilineage",
    "cell_type": "cell type",
}
ARMS = {
    "geometry": "economy",
    "fusion": "fusion",
    "counts": "countsonly",
    "perm": "fusionperm",
    "heldout_geometry": "leakctrl",
    "heldout_fusion": "fusion_leakctrl",
}
ALPHA = 0.05


def per_seed(cell: dict, metric: str) -> list[float]:
    """Per-seed values of `metric` for one (level, budget) cell, in seed order."""
    if metric == "bal_acc":
        return [float(x) for x in cell["raw"]["bal_acc"]]
    if metric == "bal_acc_at_best_val":
        return [float(c["test_bal_acc_at_best_val"]) for c in cell["raw"]["curves"]]
    raise ValueError(f"unknown metric {metric!r}")


def load_arm(results: Path, arm: str, task: str, suffix: str, metric: str) -> dict | None:
    """{level: [per-seed values]} for one arm and task, or None if absent."""
    path = results / f"manc_{arm}{suffix}_{task}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    budget = str(d["config"]["budgets"][0])
    cells = d[task]["precomputed"]
    out = {int(z): per_seed(cells[z][budget], metric) for z in cells}
    out["n_classes"] = d[task].get("meta", {}).get("n_classes")
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def fmt_p(p: float) -> str:
    """Manuscript style: two significant figures, scientific below 0.01."""
    if p != p:  # NaN
        return "nan"
    if p == 0:
        return "0"
    if p >= 0.01:
        return f"{p:.2g}"
    exp = math.floor(math.log10(p))
    mant = p / 10**exp
    if round(mant, 1) >= 10:
        mant, exp = mant / 10, exp + 1
    return f"{mant:.1f} × 10^{exp}"


def compare(a: dict, b: dict, levels_a: list[int], levels_b: list[int]) -> list[dict]:
    """Paired comparisons a[la] - b[lb] for matched level lists."""
    rows = []
    for la, lb in zip(levels_a, levels_b):
        r = paired_t(a[la], b[lb])
        rows.append({"la": la, "lb": lb, "a": mean(a[la]), "b": mean(b[lb]),
                     "diff": r["diff"], "p": r["p"]})
    return rows


def with_holm(rows: list[dict]) -> list[dict]:
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
    return rows


def sig_levels(rows: list[dict]) -> list[str]:
    return [f"z{r['la']}" for r in rows if r["p_holm"] < ALPHA]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results",
                    help="directory holding the manc_*_<task>.json artifacts (default: results/)")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated task list (default: %(default)s)")
    ap.add_argument("--suffix", default="",
                    help="artifact stem suffix, e.g. '30' for manc_fusion30_<task>.json")
    ap.add_argument("--metric", default="bal_acc", choices=["bal_acc", "bal_acc_at_best_val"],
                    help="bal_acc = final epoch (paper); bal_acc_at_best_val = curve runs")
    args = ap.parse_args(argv)
    tasks = [t for t in args.tasks.split(",") if t]

    data = {}
    for task in tasks:
        arms = {k: load_arm(args.results, stem, task, args.suffix, args.metric)
                for k, stem in ARMS.items()}
        missing = [f"manc_{ARMS[k]}{args.suffix}_{task}.json"
                   for k in ("geometry", "fusion", "counts") if arms[k] is None]
        if missing:
            raise SystemExit(f"missing in {args.results}: {', '.join(missing)}")
        data[task] = arms

    # --- 1. Extended Data Table 4 ------------------------------------------
    print(f"\nExtended Data Table 4: geometry fused with linked synapse counts "
          f"(metric={args.metric}; regenerated from {args.results})")
    print("Δ = fusion - geometry; p = paired t over shared seeds (uncorrected); "
          "p (Holm) = Holm-adjusted across the five levels of each task.\n")
    print("| Task (classes) | Level | Geometry | Counts | Fusion | Δ | p | p (Holm) |")
    print("|---|---|---|---|---|---|---|---|")
    main_rows = {}
    for task in tasks:
        arms = data[task]
        geo, fus, cnt = arms["geometry"], arms["fusion"], arms["counts"]
        levels = sorted(z for z in geo if isinstance(z, int))
        rows = with_holm(compare(fus, geo, levels, levels))
        main_rows[task] = rows
        label = f"{TASK_NAMES.get(task, task)} ({geo['n_classes']})"
        for i, r in enumerate(rows):
            z = r["la"]
            print(f"| {label if i == 0 else ''} | z{z} | {r['b']:.3f} | {mean(cnt[z]):.3f} "
                  f"| {r['a']:.3f} | {r['diff']:+.3f} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} |")

    print("\nPer-task summary (fusion - geometry):")
    for task in tasks:
        rows = main_rows[task]
        diffs = [r["diff"] for r in rows]
        cnt = data[task]["counts"]
        cvals = [mean(cnt[z]) for z in sorted(z for z in cnt if isinstance(z, int))]
        sig = sig_levels(rows)
        print(f"  {task:13s} Δ {min(diffs):+.3f} … {max(diffs):+.3f}; "
              f"Holm-significant at {len(sig)}/{len(rows)} levels ({', '.join(sig) or 'none'}); "
              f"counts-only {min(cvals):.3f}–{max(cvals):.3f}")
        nonsig = [r for r in rows if r["p_holm"] >= ALPHA]
        for r in nonsig:
            print(f"    z{r['la']}: raw p = {fmt_p(r['p'])}, Holm p = {r['p_holm']:.3f} (not significant)")

    # --- 2. Substitution test ----------------------------------------------
    print("\nSubstitution test: fusion at the coarsest level vs geometry at the finest level")
    for task in tasks:
        geo, fus = data[task]["geometry"], data[task]["fusion"]
        levels = sorted(z for z in geo if isinstance(z, int))
        r = compare(fus, geo, [levels[0]], [levels[-1]])[0]
        print(f"  {task:13s} fusion z{r['la']} {r['a']:.3f} vs geometry z{r['lb']} {r['b']:.3f}  "
              f"Δ {r['diff']:+.3f}  paired p = {fmt_p(r['p'])} ({r['p']:.3g})")

    # --- 3. Permutation control --------------------------------------------
    perm_tasks = [t for t in tasks if data[t]["perm"] is not None]
    if perm_tasks:
        print("\nPermutation control: counts permuted among neurons (one fixed permutation), "
              "fusion arm vs geometry alone")
        all_rows = []
        for task in perm_tasks:
            geo, perm = data[task]["geometry"], data[task]["perm"]
            levels = sorted(z for z in geo if isinstance(z, int))
            rows = with_holm(compare(perm, geo, levels, levels))
            all_rows += rows
            cells = ", ".join(f"z{r['la']} {r['diff']:+.3f} (p {fmt_p(r['p'])}, Holm {r['p_holm']:.2f})"
                              for r in rows)
            print(f"  {task:13s} {cells}")
        n_sig = sum(r["p_holm"] < ALPHA for r in all_rows)
        print(f"  max |Δ| over {len(all_rows)} task-level cells = "
              f"{max(abs(r['diff']) for r in all_rows):.4f}; Holm-significant cells: {n_sig}")
    else:
        print("\nPermutation control: no manc_fusionperm artifacts found")

    # --- 4. Held-out control -----------------------------------------------
    ho_tasks = [t for t in tasks
                if data[t]["heldout_fusion"] is not None and data[t]["heldout_geometry"] is not None]
    if ho_tasks:
        print("\nHeld-out control: fusion vs geometry under the cell-type-held-out split")
        for task in ho_tasks:
            geo, fus = data[task]["heldout_geometry"], data[task]["heldout_fusion"]
            levels = sorted(z for z in geo if isinstance(z, int))
            rows = with_holm(compare(fus, geo, levels, levels))
            diffs = [r["diff"] for r in rows]
            sig = sig_levels(rows)
            print(f"  {task:13s} Δ {min(diffs):+.3f} … {max(diffs):+.3f}; "
                  f"max raw p = {fmt_p(max(r['p'] for r in rows))}; "
                  f"Holm-significant at {len(sig)}/{len(rows)} ({', '.join(sig) or 'none'})")
            print("                " + ", ".join(
                f"z{r['la']} {r['diff']:+.3f} (p {fmt_p(r['p'])})" for r in rows))
    else:
        print("\nHeld-out control: manc_leakctrl / manc_fusion_leakctrl artifacts not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
