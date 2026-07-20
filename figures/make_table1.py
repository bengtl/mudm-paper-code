#!/usr/bin/env python3
"""Regenerate a storage-vs-balanced-accuracy table from deposited economy
artifacts in ``../results``.

By default it renders the main-text **Table 1** (MANC, ``manc_economy_*.json``).
Pass ``--prefix`` to render a different artifact set the same way:

    python figures/make_table1.py                          # Table 1 (MANC)
    python figures/make_table1.py --prefix manc_leakctrl    # leakage-controlled check
    python figures/make_table1.py --prefix flywire_economy  # FlyWire replication (once run)

Pure standard library (``json`` only) -- no GPU, torch, or numpy. The sweep that
*produces* the artifacts is ``mudm_paper.run_manc``; this only reads them back.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

# top-level keys that are NOT the per-task result block
_NON_LABEL = {"benchmark_id", "timestamp", "hardware", "cost", "knee", "tost", "config"}

# Column order (coarsest -> finest), per known artifact prefix. Unknown prefixes
# fall back to alphabetical task order.
_TASK_ORDER = {
    "manc_economy": ["neuron_class", "subclass", "hemilineage", "cell_type"],
    "manc_leakctrl": ["neuron_class", "subclass", "hemilineage", "cell_type"],
    "flywire_economy": ["super_class", "cell_class", "hemilineage", "cell_type"],
}
_DATASET_NAME = {
    "manc_economy": "MANC",
    "manc_leakctrl": "MANC (leakage-controlled, cell-type-held-out)",
    "flywire_economy": "FlyWire",
}


def _label_key(d: dict) -> str:
    for k in d:
        if k not in _NON_LABEL:
            return k
    raise KeyError("no per-task label key found in result JSON")


def discover_tasks(results_dir: Path, prefix: str) -> list[str]:
    """Task names present as ``<prefix>_<task>.json`` in ``results_dir``, ordered
    coarsest->finest by the known ladder for this prefix (unknown tasks appended
    alphabetically). Stripping the exact ``<prefix>_`` keeps multi-word task
    names such as ``super_class`` / ``cell_type`` intact."""
    names = [
        Path(p).name[len(prefix) + 1 : -len(".json")]
        for p in glob.glob(str(results_dir / f"{prefix}_*.json"))
    ]
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
    # Storage = total uncompressed bytes per level (all columns); the deposited
    # cost block records it, and z0 ratio_to_finest gives the ~98x span.
    storage = {
        int(z): v["uncompressed_bytes"] / 1e9
        for z, v in d["cost"]["storage_per_level"].items()
    }
    n_classes = d[label].get("meta", {}).get("n_classes")
    return {"bal": bal, "storage": storage, "n_classes": n_classes}


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
        help="Artifact prefix: manc_economy (Table 1, default), manc_leakctrl, or flywire_economy.",
    )
    ap.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task order override (default: auto-discover from filenames).",
    )
    args = ap.parse_args(argv)
    results_dir = Path(args.results)
    prefix = args.prefix

    tasks = args.tasks.split(",") if args.tasks else discover_tasks(results_dir, prefix)
    loaded = {t: load_task(results_dir, prefix, t) for t in tasks}
    present = [t for t in tasks if loaded[t] is not None]
    if not present:
        raise SystemExit(
            f"no {prefix}_*.json found in {results_dir} "
            f"(run: mudm_paper.run_manc ... --out-dir {results_dir} --prefix {prefix})"
        )

    storage = loaded[present[0]]["storage"]  # same pyramid for every task
    zooms = sorted(storage)

    cols = [("Level", 14), ("Storage (GB)", 13)]
    cols += [(f"{t} ({loaded[t]['n_classes']})", 18) for t in present]

    def line(cells):
        print("  ".join(str(c).ljust(w) for c, (_, w) in zip(cells, cols)))

    name = _DATASET_NAME.get(prefix, prefix)
    title = "Table 1 — " if prefix == "manc_economy" else ""
    print(f"\n{title}storage vs balanced accuracy across the {name} 3D-mesh pyramid")
    print(f"(prefix={prefix}; regenerated from {results_dir})\n")
    line([h for h, _ in cols])
    line(["-" * w for _, w in cols])
    names = {zooms[0]: "z0 (coarsest)", zooms[-1]: f"z{zooms[-1]} (finest)"}
    for z in zooms:
        row = [names.get(z, f"z{z}"), f"{storage[z]:.2f}"]
        row += [f"{loaded[t]['bal'][z]:.3f}" for t in present]
        line(row)

    zc, zf = zooms[0], zooms[-1]
    print("\nCoarsest-vs-finest retention:")
    for t in present:
        bal = loaded[t]["bal"]
        pct = 100 * bal[zc] / bal[zf] if bal[zf] else float("nan")
        print(f"  {t:13s} {pct:3.0f}%   (z{zc}={bal[zc]:.3f}, z{zf}={bal[zf]:.3f})")
    print(
        f"\nStorage spans {storage[zf] / storage[zc]:.0f}x from coarsest to finest "
        f"({storage[zc]:.2f} -> {storage[zf]:.2f} GB)."
    )


if __name__ == "__main__":
    main()
