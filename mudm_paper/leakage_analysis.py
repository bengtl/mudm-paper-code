"""Serial/segmental-homolog leakage analysis for the MANC split (fix #4).

Reproduces the exposure figures cited in the Methods. Bilateral (_L/_R)
homologs are grouped (``grouped_stratified_split`` strips the side suffix from
``instance``), so no split group straddles train/test. This script quantifies
SERIAL-homolog exposure: test neurons whose ``cell_type`` (the finest homology
unit) also appears in TRAIN via a DIFFERENT instance group, i.e. a same-type
neuron the instance-grouping did not keep together.

It reads tags only (no geometry / GPU) and replays the exact split used for
Table 1, so a reader can regenerate the numbers behind the Methods sentence
(bilateral straddle 0.000; serial exposure 38-42% for the class/subclass/
lineage tasks, 100% for the cell-type task by necessity).

The complementary robustness check (rerunning the sweep under a cell-type-held-out
split) is produced by ``run_manc --group-field cell_type`` and deposited as
``results/manc_leakctrl_*.json``.

Usage:
    uv run python -m mudm_paper.leakage_analysis <parquet_dir> [seeds]
"""

import sys
from collections import defaultdict

import numpy as np

from mudm_paper.data import _SIDE_SUFFIX, grouped_stratified_split, load_labels

TASKS = ["neuron_class", "subclass", "hemilineage", "cell_type"]
MIN_CLASS = 20


def serial_homolog_exposure(
    body_ids: list,
    task_labels: dict,
    cell_type: dict,
    group_of: dict,
    seeds: int,
    min_class: int = MIN_CLASS,
) -> dict:
    """Replay the grouped split over ``seeds`` and measure homolog exposure.

    Returns a dict with mean fractions over seeds:
      - ``serial``: test neurons sharing a ``cell_type`` with a train neuron in
        a different split group (a serial homolog the grouping did not keep
        together);
      - ``straddle``: test neurons whose own group also appears in train (must
        be ~0 -- groups never cross the split by construction);
      - ``same_class``: test neurons whose task class is present in train
        (~1.0 for normal supervised classification).
    Plus ``n``, ``n_groups``, ``n_cell_types``, and ``frag`` (mean groups per
    cell type).
    """
    ct = {b: cell_type.get(b) for b in body_ids}
    ct_groups = defaultdict(set)
    for b in body_ids:
        if ct[b]:
            ct_groups[ct[b]].add(group_of[b])
    frag = float(np.mean([len(g) for g in ct_groups.values()])) if ct_groups else 0.0

    serial, straddle, same_class = [], [], []
    for seed in range(seeds):
        tr, _va, te = grouped_stratified_split(
            body_ids, task_labels, seed=seed, min_class=min_class, group_of=group_of
        )
        train_ct_groups = defaultdict(set)
        train_classes = set()
        for b in tr:
            if ct[b]:
                train_ct_groups[ct[b]].add(group_of[b])
            train_classes.add(task_labels[b])
        n_serial = n_straddle = n_same = 0
        for b in te:
            c = ct[b]
            if c and c in train_ct_groups:
                if train_ct_groups[c] - {group_of[b]}:
                    n_serial += 1
                if group_of[b] in train_ct_groups[c]:
                    n_straddle += 1
            if task_labels[b] in train_classes:
                n_same += 1
        nte = len(te) or 1
        serial.append(n_serial / nte)
        straddle.append(n_straddle / nte)
        same_class.append(n_same / nte)

    return {
        "n": len(body_ids),
        "n_groups": len({group_of[b] for b in body_ids}),
        "n_cell_types": len({ct[b] for b in body_ids if ct[b]}),
        "frag": frag,
        "serial": float(np.mean(serial)),
        "serial_sd": float(np.std(serial)),
        "straddle": float(np.mean(straddle)),
        "same_class": float(np.mean(same_class)),
    }


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    parquet = argv[0]
    seeds = int(argv[1]) if len(argv) > 1 else 15

    cell_type = load_labels(parquet, "cell_type")
    instance = load_labels(parquet, "instance")
    group_of = {b: _SIDE_SUFFIX.sub("", v) for b, v in instance.items()}

    print(f"seeds={seeds}  min_class={MIN_CLASS}")
    for task in TASKS:
        labels = load_labels(parquet, task)
        body_ids = sorted(set(labels) & set(group_of))  # mirror sweep eligibility
        r = serial_homolog_exposure(body_ids, labels, cell_type, group_of, seeds)
        print(
            f"\n{task}: n={r['n']} groups={r['n_groups']} "
            f"cell_types={r['n_cell_types']} frag(groups/type)={r['frag']:.2f}"
        )
        print(f"  serial-homolog-in-train (same cell_type, diff group): "
              f"{r['serial']:.3f} +/- {r['serial_sd']:.3f}")
        print(f"  group-straddle (bilateral leak; MUST be ~0): {r['straddle']:.4f}")
        print(f"  same-task-class-in-train (~1 = normal classification): "
              f"{r['same_class']:.3f}")


if __name__ == "__main__":
    main()
