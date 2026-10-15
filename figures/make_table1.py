#!/usr/bin/env python3
"""Regenerate a storage-vs-balanced-accuracy table from deposited economy
artifacts in ``../results``.

By default it renders the main-text **Table 1** (MANC, ``manc_economy_*.json``)
for the three published tasks (neuron class, subclass, hemilineage). Pass
``--prefix`` to render a different artifact set the same way:

    python figures/make_table1.py                           # Table 1 (MANC)
    python figures/make_table1.py --prefix manc_leakctrl     # cell-type-held-out check
    python figures/make_table1.py --prefix manc_pointnet     # PointNet classifier
    python figures/make_table1.py --prefix flywire_economy   # FlyWire (Extended Data Table 2)
    python figures/make_table1.py --include-cell-type        # also the deposited cell_type run

The paper does not report the grouped ``cell_type`` task (its instance-grouped
split leaves most of the 117 classes without test neurons), so it is excluded
unless ``--include-cell-type`` or an explicit ``--tasks`` list asks for it.

Columns: "Storage (GB)" is the uncompressed geometry of each level
(``cost.storage_per_level[z].uncompressed_bytes / 1e9``) and "On disk (GB)"
the ZSTD-compressed Parquet size (``compressed_bytes / 1e9``). Accuracy cells
are seed means of balanced accuracy at the run's input budget (512 vertices in
the paper). The table is printed as Markdown, laid out like Table 1 in the
manuscript.

Pure standard library (``json`` only): no GPU, torch, or numpy. The sweep that
*produces* the artifacts is ``mudm_paper.run_manc``; this only reads them back.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

# top-level keys that are NOT the per-task result block
_NON_LABEL = {"benchmark_id", "timestamp", "hardware", "cost", "knee", "tost", "config"}

# Column order (coarsest -> finest class system), per known artifact prefix.
# Unknown prefixes fall back to alphabetical task order.
_MANC_ORDER = ["neuron_class", "subclass", "hemilineage", "cell_type"]
_TASK_ORDER = {
    "manc_economy": _MANC_ORDER,
    "manc_leakctrl": _MANC_ORDER,
    "manc_pointnet": _MANC_ORDER,
    "flywire_economy": ["super_class", "cell_class", "hemilineage", "cell_type"],
}
_DATASET_NAME = {
    "manc_economy": "MANC",
    "manc_leakctrl": "MANC (leakage-controlled, cell-type-held-out)",
    "manc_pointnet": "MANC (PointNet)",
    "flywire_economy": "FlyWire",
}
# Reported only on request (see module docstring).
_OPT_IN_TASKS = {"cell_type"}


def _label_key(d: dict) -> str:
    for k in d:
        if k not in _NON_LABEL:
            return k
    raise KeyError("no per-task label key found in result JSON")


def discover_tasks(results_dir: Path, prefix: str, include_cell_type: bool = False) -> list[str]:
    """Task names present as ``<prefix>_<task>.json`` in ``results_dir``, ordered
    coarsest->finest by the known ladder for this prefix (unknown tasks appended
    alphabetically). Stripping the exact ``<prefix>_`` keeps multi-word task
    names such as ``super_class`` / ``cell_type`` intact. ``cell_type`` is
    dropped unless ``include_cell_type``."""
    names = [
        Path(p).name[len(prefix) + 1 : -len(".json")]
        for p in glob.glob(str(results_dir / f"{prefix}_*.json"))
    ]
    if not include_cell_type:
        names = [n for n in names if n not in _OPT_IN_TASKS]
    order = _TASK_ORDER.get(prefix, [])
    known = [t for t in order if t in names]
    extra = sorted(t for t in names if t not in order)
    return known + extra


def load_task(results_dir: Path, prefix: str, task: str) -> dict | None:
    path = results_dir / f"{prefix}_{task}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    label = _label_key(d)
    budget = str(d["config"]["budgets"][0])  # 512 in the paper config
    arm = d[label]["precomputed"]
    bal = {int(z): arm[z][budget]["bal_acc"]["mean"] for z in arm}
    sem = {int(z): arm[z][budget]["bal_acc"]["se"] for z in arm}
    levels = d["cost"]["storage_per_level"]
    storage = {int(z): v["uncompressed_bytes"] / 1e9 for z, v in levels.items()}
    on_disk = {int(z): v["compressed_bytes"] / 1e9 for z, v in levels.items()}
    n_classes = d[label].get("meta", {}).get("n_classes")
    return {
        "bal": bal,
        "sem": sem,
        "storage": storage,
        "on_disk": on_disk,
        "n_classes": n_classes,
        "budget": budget,
        "seeds": d["config"].get("seeds"),
    }


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Pipe table with every column padded to its widest cell."""
    widths = [max(len(c) for c in col) for col in zip(header, *rows)]

    def fmt(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return [fmt(header), sep] + [fmt(r) for r in rows]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Regenerate an economy table from results/.")
    ap.add_argument(
        "--results",
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Directory holding <prefix>_<task>.json (default: ../results).",
    )
    ap.add_argument(
        "--prefix",
        default="manc_economy",
        help="Artifact prefix: manc_economy (Table 1, default), manc_leakctrl, "
        "manc_pointnet, or flywire_economy.",
    )
    ap.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task list, used as given "
        "(default: auto-discover from filenames, without cell_type).",
    )
    ap.add_argument(
        "--include-cell-type",
        action="store_true",
        help="Also report the deposited cell_type run (not reported in the paper).",
    )
    args = ap.parse_args(argv)
    results_dir = Path(args.results)
    prefix = args.prefix

    if args.tasks:
        tasks = [t for t in args.tasks.split(",") if t]
    else:
        tasks = discover_tasks(results_dir, prefix, args.include_cell_type)
    loaded = {t: load_task(results_dir, prefix, t) for t in tasks}
    present = [t for t in tasks if loaded[t] is not None]
    absent = [t for t in tasks if loaded[t] is None]
    if not present:
        raise SystemExit(
            f"no {prefix}_*.json found in {results_dir} "
            f"(run: python -m mudm_paper.run_manc ... --out-dir {results_dir} --prefix {prefix})"
        )

    first = loaded[present[0]]
    storage, on_disk = first["storage"], first["on_disk"]  # same pyramid for every task
    zooms = sorted(storage)

    name = _DATASET_NAME.get(prefix, prefix)
    title = "Table 1: " if prefix == "manc_economy" else ""
    print(f"\n{title}balanced accuracy versus storage across the {name} 3D-mesh pyramid")
    print(
        f"(prefix={prefix}; budget={first['budget']} vertices; "
        f"{first['seeds']} seeds; regenerated from {results_dir})\n"
    )
    header = ["Level", "Storage (GB)", "On disk (GB)"]
    header += [f"{t.replace('_', ' ')} ({loaded[t]['n_classes']})" for t in present]
    names = {zooms[0]: f"z{zooms[0]} (coarsest)", zooms[-1]: f"z{zooms[-1]} (finest)"}
    rows = []
    for z in zooms:
        row = [names.get(z, f"z{z}"), f"{storage[z]:.2f}", f"{on_disk[z]:.2f}"]
        row += [f"{loaded[t]['bal'][z]:.3f}" for t in present]
        rows.append(row)
    print("\n".join(markdown_table(header, rows)))
    if absent:
        print(f"\n(missing, skipped: {', '.join(f'{prefix}_{t}.json' for t in absent)})")

    zc, zf = zooms[0], zooms[-1]
    print("\nCoarsest-vs-finest retention:")
    for t in present:
        bal = loaded[t]["bal"]
        pct = 100 * bal[zc] / bal[zf] if bal[zf] else float("nan")
        print(f"  {t:13s} {pct:3.0f}%   (z{zc}={bal[zc]:.3f}, z{zf}={bal[zf]:.3f})")
    max_sem = max(max(loaded[t]["sem"].values()) for t in present)
    print(f"\nLargest s.e.m. over all cells: {max_sem:.3f}")
    print(
        f"Storage spans {storage[zf] / storage[zc]:.1f}x uncompressed "
        f"({storage[zc]:.2f} -> {storage[zf]:.2f} GB; coarsest = "
        f"{100 * storage[zc] / storage[zf]:.1f}% of finest) and "
        f"{on_disk[zf] / on_disk[zc]:.1f}x on disk "
        f"({on_disk[zc]:.2f} -> {on_disk[zf]:.2f} GB)."
    )


if __name__ == "__main__":
    main()
