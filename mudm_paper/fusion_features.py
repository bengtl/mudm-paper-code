"""Build the per-neuron scalar feature map consumed by the fusion arm.

Input: a JSON list of neuPrint neuron rows carrying synapse counts
(``[{"bodyId": ..., "pre": ..., "post": ...}, ...]`` -- e.g. the cached
fetch behind the counts-only leakage probe,
mudm_paper/synapse_probe.py). Output: ``{str(bodyId): [log1p(pre),
log1p(post), log1p(pre+post), pre/(pre+post)]}`` -- the probe's exact
feature definition, so the fusion arm (``dgcnn_fusion``), the counts-only
baseline (``counts_mlp``), and the probe all see identical inputs. Rows
missing either count are dropped (coverage filtering in run_sweep then
excludes those neurons from the cohort; no imputation).

Usage:
    python -m mudm_paper.fusion_features <neuprint_rows.json> <features.json>
    run_manc --extra-features features.json --model dgcnn_fusion ...
"""

from __future__ import annotations

import argparse
import json
import math


def features_from_rows(rows: list) -> dict:
    """See module docstring. Ratio is guarded to 0.0 on a zero total."""
    out = {}
    for r in rows:
        pre, post = r.get("pre"), r.get("post")
        if pre is None or post is None:
            continue
        pre, post = float(pre), float(post)
        total = pre + post
        out[str(r["bodyId"])] = [
            math.log1p(pre),
            math.log1p(post),
            math.log1p(total),
            (pre / total) if total else 0.0,
        ]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rows_json", help="neuPrint rows JSON (list of dicts with bodyId/pre/post)")
    ap.add_argument("out_json", help="output feature map JSON ({body_id: [4 floats]})")
    args = ap.parse_args(argv)
    with open(args.rows_json) as f:
        rows = json.load(f)
    features = features_from_rows(rows)
    with open(args.out_json, "w") as f:
        json.dump(features, f)
    print(f"wrote {len(features)} neurons (of {len(rows)} rows) -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
