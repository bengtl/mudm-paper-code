"""Leakage probe: how well do synapse counts ALONE predict MANC type labels?

Motivation (2026-08-11): before committing to a multimodal experiment
(geometry + synapse count), check whether synapse counts are quasi-label
information -- MANC systematic types are partly connectivity-defined, so a
high score here means adding counts risks circularity, not multimodal signal.

Protocol mirrors the paper harness: same tasks, MIN_CLASS=20, 15 seeds, and
the same grouped, leakage-safe split (side-stripped `instance` groups) via
mudm_paper.data.grouped_stratified_split. Features are log1p(pre),
log1p(post), log1p(pre+post), and pre/(pre+post). Two models bracket the
signal: multinomial logistic regression (linear) and gradient-boosted trees
(nonlinear ceiling). Metric: balanced accuracy on the held-out test split,
mean +/- sd over seeds -- directly comparable to the deposited DGCNN numbers
in results/manc_economy_<task>.json.

Run from the repository root (needs the probe extra):
    uv run python -m mudm_paper.synapse_probe [--out <file>]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from mudm_paper.data import _SIDE_SUFFIX, grouped_stratified_split

CACHE = Path(__file__).parents[1] / "results" / "manc_synapse_meta.json"
RESULTS_DIR = Path(__file__).parents[1] / "results"
TASKS = ["neuron_class", "subclass", "hemilineage", "cell_type"]
MIN_CLASS = 20
SEEDS = 15

_NEUPRINT_URL = "https://neuprint.janelia.org/api/custom/custom"
_DATASET = "manc:v1.2.1"
_CYPHER = """
MATCH (n :Neuron)
WHERE n.status = "Traced" AND n.type IS NOT NULL
RETURN n.bodyId AS bodyId, n.pre AS pre, n.post AS post,
       n.type AS cell_type, n.instance AS instance,
       n.class AS neuron_class, n.subclass AS subclass,
       n.hemilineage AS hemilineage
"""


def fetch() -> list[dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    token = Path("~/.neuprint_token").expanduser().read_text().strip()
    payload = json.dumps({"cypher": _CYPHER, "dataset": _DATASET}).encode()
    req = urllib.request.Request(
        _NEUPRINT_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = json.load(resp)
    cols = raw["columns"]
    rows = [dict(zip(cols, r)) for r in raw["data"]]
    CACHE.write_text(json.dumps(rows))
    return rows


def dgcnn_reference(task: str) -> dict:
    """Deposited DGCNN balanced accuracy (budget 512) per zoom, if present."""
    f = RESULTS_DIR / f"manc_economy_{task}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    pre = d.get(task, {}).get("precomputed", {})
    out = {}
    for zoom, budgets in pre.items():
        b = budgets.get("512") or next(iter(budgets.values()), None)
        if b and "bal_acc" in b:
            out[zoom] = b["bal_acc"]["mean"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RESULTS_DIR / "synapse_probe_results.json"),
                    help="output JSON (default: the deposited results/synapse_probe_results.json)")
    args = ap.parse_args()
    rows = fetch()
    rows = [r for r in rows if r.get("pre") is not None and r.get("post") is not None
            and r.get("instance")]
    print(f"neurons with pre/post + instance: {len(rows)}")

    body_ids_all = [str(r["bodyId"]) for r in rows]
    by_id = {str(r["bodyId"]): r for r in rows}
    group_of = {b: _SIDE_SUFFIX.sub("", by_id[b]["instance"]) for b in body_ids_all}

    def feats(b: str) -> list[float]:
        pre, post = float(by_id[b]["pre"]), float(by_id[b]["post"])
        tot = pre + post
        return [np.log1p(pre), np.log1p(post), np.log1p(tot),
                pre / tot if tot else 0.0]

    summary = {}
    for task in TASKS:
        labels = {b: by_id[b][task] for b in body_ids_all if by_id[b].get(task)}
        body_ids = sorted(labels)
        counts = {}
        for b in body_ids:
            counts[labels[b]] = counts.get(labels[b], 0) + 1
        kept = [b for b in body_ids if counts[labels[b]] >= MIN_CLASS]
        classes = sorted({labels[b] for b in kept})
        k = len(classes)
        cls_idx = {c: i for i, c in enumerate(classes)}

        lr_scores, gb_scores = [], []
        for seed in range(SEEDS):
            tr, _va, te = grouped_stratified_split(
                kept, labels, seed=seed, min_class=MIN_CLASS, group_of=group_of
            )
            Xtr = np.array([feats(b) for b in tr])
            ytr = np.array([cls_idx[labels[b]] for b in tr])
            Xte = np.array([feats(b) for b in te])
            yte = np.array([cls_idx[labels[b]] for b in te])

            lr = LogisticRegression(max_iter=2000)
            lr.fit(Xtr, ytr)
            lr_scores.append(balanced_accuracy_score(yte, lr.predict(Xte)))

            gb = HistGradientBoostingClassifier(random_state=seed)
            gb.fit(Xtr, ytr)
            gb_scores.append(balanced_accuracy_score(yte, gb.predict(Xte)))

        ref = dgcnn_reference(task)
        summary[task] = {
            "n": len(kept),
            "k_classes": k,
            "chance_bal_acc": 1.0 / k if k else None,
            "logreg_bal_acc": [float(np.mean(lr_scores)), float(np.std(lr_scores))],
            "gboost_bal_acc": [float(np.mean(gb_scores)), float(np.std(gb_scores))],
            "dgcnn_bal_acc_by_zoom_b512": ref,
        }
        print(f"\n{task}: n={len(kept)} K={k} chance={1.0 / k:.3f}" if k else task)
        print(f"  logreg  {np.mean(lr_scores):.3f} +/- {np.std(lr_scores):.3f}")
        print(f"  gboost  {np.mean(gb_scores):.3f} +/- {np.std(gb_scores):.3f}")
        if ref:
            zs = sorted(ref, key=lambda z: int(z))
            line = ", ".join(f"z{z}={ref[z]:.3f}" for z in zs)
            print(f"  DGCNN(b512) {line}")

    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
