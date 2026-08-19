"""Extract PanNuke intensity images matching the muDM geojson patch stems.

The raw folds are HuggingFace parquet (see gen_script.py): rows with an RGB
"image" and a list of per-nucleus binary "instances" masks. The muDM geojson
files are named fold{f}_{row:05d}.geojson, presumed to follow fold row order;
this script VERIFIES that by IoU between the union of a row's instance masks
and the rasterized tau_0 mask, on sampled patches, before extracting.

Writes images/<stem>.tif (8-bit grayscale) for every stem present in
geojson/, and prints alignment stats. Aborts (exit 1) if alignment fails.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

WORK = Path(sys.argv[1])
RAW, GEOJSON, MASKS = WORK / "raw", WORK / "geojson", WORK / "masks"
IMAGES = WORK / "images"

stems = {p.stem for p in GEOJSON.glob("*.geojson")}
print(f"geojson stems: {len(stems)}", flush=True)
IMAGES.mkdir(exist_ok=True)

ious, extracted = [], 0
for fold in (1, 2, 3):
    df = pd.read_parquet(next(RAW.glob(f"fold{fold}-*.parquet")))
    print(f"fold{fold}: {len(df)} rows", flush=True)
    check_rows = {0, len(df) // 2, len(df) - 1}
    for i, row in df.iterrows():
        stem = f"fold{fold}_{i:05d}"
        if stem not in stems:
            continue
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
        tifffile.imwrite(IMAGES / f"{stem}.tif", np.asarray(img, dtype=np.uint8))
        extracted += 1
        if i in check_rows:
            union = None
            for inst in row["instances"]:
                m = np.asarray(Image.open(io.BytesIO(inst["bytes"]))) > 0
                union = m if union is None else (union | m)
            if union is None:
                continue
            t0 = tifffile.imread(MASKS / "tau_0" / f"{stem}.tif") > 0
            inter = (union & t0).sum()
            uni = (union | t0).sum()
            ious.append(inter / uni if uni else 0.0)

print(f"extracted: {extracted} images", flush=True)
print(f"alignment IoUs (sampled): {[round(v, 3) for v in ious]}", flush=True)
if not ious or min(ious) < 0.8:
    print("ALIGNMENT FAILED — do not run Nyxus on these pairs", flush=True)
    sys.exit(1)
print("ALIGNMENT OK", flush=True)
