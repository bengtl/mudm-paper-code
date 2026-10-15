"""Full PanNuke Nyxus robustness pipeline (shape features, all 7,556 patches).

Stages (idempotent; each skipped if its output exists):
  1. rasterize all patches at tau 0/0.5/1/2/4/8 (masks already pulled by caller)
  2. Nyxus *ALL_MORPHOLOGY* per level, intDir=segDir (shape features only,
     intensity-independent), CSV per level
  3. join on (file stem, label) and write a per-feature stability summary:
     median absolute relative deviation vs tau_0, per tau level

Run (needs the stability extra):  uv run python scripts/nyxus_pipeline.py <workdir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(sys.argv[1])
GEOJSON = WORK / "geojson"
MASKS = WORK / "masks"
FEATS = WORK / "features"
EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
LEVELS = ["tau_0", "tau_0.5", "tau_1", "tau_2", "tau_4", "tau_8"]


def stage1_rasterize():
    if (MASKS / "manifest.json").exists():
        print("stage1: masks exist, skipping", flush=True)
        return
    from mudm_paper.rasterize import main as ras_main
    rc = ras_main([str(GEOJSON), "--out", str(MASKS),
                   "--eps", *[str(e) for e in EPS], "--shape", "256", "256"])
    if rc != 0:
        raise SystemExit("rasterize failed")


def stage2_nyxus():
    # NB: intensity dir must DIFFER from the seg dir -- Nyxus silently switches
    # to whole-image (single-ROI) mode when they are the same path, which is
    # exactly the 7556-rows-of-ROI_label-1 failure the first run produced.
    import nyxus
    images = WORK / "images"
    n_img = len(list(images.glob("*.tif")))
    assert n_img > 0, f"no intensity images under {images} (run extract_pannuke_images.py)"
    FEATS.mkdir(exist_ok=True)
    for level in LEVELS:
        out_csv = FEATS / f"{level}.csv"
        if out_csv.exists():
            print(f"stage2: {level} exists, skipping", flush=True)
            continue
        nyx = nyxus.Nyxus(["*ALL_MORPHOLOGY*", "*ALL_INTENSITY*"])
        df = nyx.featurize_directory(str(images), str(MASKS / level))
        df.to_csv(out_csv, index=False)
        print(f"stage2: {level} -> {len(df)} rows", flush=True)


def stage3_summary():
    frames = {}
    for level in LEVELS:
        df = pd.read_csv(FEATS / f"{level}.csv")
        stem_col = next(c for c in df.columns if "mask" in c.lower() or "seg" in c.lower())
        df["stem"] = df[stem_col].astype(str).str.replace(r"\.tif$", "", regex=True)
        frames[level] = df.set_index(["stem", "label"])
    base = frames["tau_0"]
    feat_cols = [c for c in base.columns
                 if base[c].dtype.kind in "fi" and c not in ("stem", "label")]
    summary = {}
    for level in LEVELS[1:]:
        cur = frames[level]
        idx = base.index.intersection(cur.index)
        b, c = base.loc[idx, feat_cols], cur.loc[idx, feat_cols]
        denom = b.abs().replace(0, np.nan)
        rel = ((c - b).abs() / denom)
        summary[level] = {
            "n_instances": int(len(idx)),
            "median_abs_rel_dev": rel.median().round(4).to_dict(),
        }
    (FEATS / "stability_summary.json").write_text(json.dumps(summary, indent=1))
    print("stage3: wrote stability_summary.json", flush=True)


if __name__ == "__main__":
    stage1_rasterize()
    stage2_nyxus()
    try:
        stage3_summary()
    except Exception as e:  # feature CSVs are the primary artifact; summary is best-effort
        print(f"stage3 FAILED (non-fatal): {e}", flush=True)
    print("PIPELINE DONE", flush=True)
