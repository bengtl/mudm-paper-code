"""Per-feature robustness across polygon LOD levels (fixed stage3).

Joins Nyxus per-ROI features on (patch stem, ROI_label) between tau_0 (full
detail) and each simplified level, then reports per feature and level:
  - median absolute relative deviation (magnitude stability)
  - Spearman rank correlation across instances (ranking stability)
Writes features/stability_summary.json and prints a compact digest.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WORK = Path(sys.argv[1])
FEATS = WORK / "features"
LEVELS = ["tau_0.5", "tau_1", "tau_2", "tau_4", "tau_8"]
META = {"intensity_image", "mask_image", "ROI_label", "t_index", "stem"}


def load(level):
    df = pd.read_csv(FEATS / f"{level}.csv")
    df["stem"] = df["mask_image"].astype(str).str.replace(r"\.tif$", "", regex=True)
    return df.set_index(["stem", "ROI_label"]).sort_index()


base = load("tau_0")
feat_cols = [c for c in base.columns
             if c not in META and pd.api.types.is_numeric_dtype(base[c])]
print(f"tau_0 instances: {len(base)}  features: {len(feat_cols)}", flush=True)

summary = {"n_base_instances": len(base), "n_features": len(feat_cols), "levels": {}}
for level in LEVELS:
    cur = load(level)
    idx = base.index.intersection(cur.index)
    b = base.loc[idx, feat_cols].to_numpy(dtype=float)
    c = cur.loc[idx, feat_cols].to_numpy(dtype=float)
    denom = np.abs(b)
    denom[denom < 1e-12] = np.nan
    rel = np.abs(c - b) / denom                      # [n_inst, n_feat]
    med_rel = np.nanmedian(rel, axis=0)
    rho = np.full(len(feat_cols), np.nan)
    for j in range(len(feat_cols)):
        bj, cj = b[:, j], c[:, j]
        ok = np.isfinite(bj) & np.isfinite(cj)
        if ok.sum() > 10 and np.nanstd(bj[ok]) > 0 and np.nanstd(cj[ok]) > 0:
            rho[j] = spearmanr(bj[ok], cj[ok]).statistic
    summary["levels"][level] = {
        "n_matched": int(len(idx)),
        "median_abs_rel_dev": dict(zip(feat_cols, np.round(med_rel, 4).tolist())),
        "spearman": dict(zip(feat_cols, np.round(rho, 4).tolist())),
    }
    print(f"{level}: matched {len(idx)}  "
          f"median-of-medians rel dev {np.nanmedian(med_rel):.4f}  "
          f"median spearman {np.nanmedian(rho):.4f}", flush=True)

(FEATS / "stability_summary.json").write_text(json.dumps(summary, indent=1))

# Digest: most/least robust features at tau_2 (mid level) by rank stability
lv = summary["levels"]["tau_2"]
rho2 = {k: v for k, v in lv["spearman"].items() if v == v}  # drop NaN
ranked = sorted(rho2.items(), key=lambda kv: kv[1])
print("\nLeast rank-stable at tau_2:", flush=True)
for k, v in ranked[:8]:
    print(f"  {k:40s} rho={v:.3f} reldev={lv['median_abs_rel_dev'][k]:.3f}", flush=True)
print("Most rank-stable at tau_2:", flush=True)
for k, v in ranked[-8:]:
    print(f"  {k:40s} rho={v:.3f} reldev={lv['median_abs_rel_dev'][k]:.3f}", flush=True)
print("\nwrote", FEATS / "stability_summary.json", flush=True)
