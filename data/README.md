# Data inputs

Nothing in `data/` is committed except this file (see `.gitignore`). The sweeps and
benchmarks read muDM Parquet pyramids, one directory per zoom level:

```
<parquet_dir>/zoom=<Z>/part_*.parquet      # Z = 0 (coarsest) .. 4 (finest) for the connectomes
```

Each row holds tile coordinates, a `feature_id`, packed float32 `positions`, packed
uint32 `indices` and a string `tags` map. The neuron-typing code needs the tags
`body_id`, `instance` and the task labels (`neuron_class`, `subclass`, `hemilineage`,
`cell_type` for MANC; `super_class`, `cell_class`, `hemilineage`, `cell_type` for FlyWire).
File names do not matter to the loader, so the catalog's `*.mu.parquet` parts work as they
are.

Commands below run from the repository root. `<mudm-tools>` is a git checkout of
<https://github.com/NovagenResearch/mudm-tools>: its `scripts/` folder is not part of the
PyPI wheel.

## MANC (Table 1, Fig. 2, fusion, benchmarks)

MANC v1.2.1, the *Drosophila* male adult nerve cord connectome (Janelia FlyEM; Takemura
et al. 2024, Marin et al. 2024), accessed through neuPrint, CC BY 4.0. The experiments use
its 23,546 traced, typed neurons.

### Option A: download the served pyramid

The catalog serves MANC as `manc-full`. Its tile descriptor,
<https://ngrf.mudm.io/tiles/manc-full/tilejson.json>, lists the Parquet parts under the
asset with `"role": "features"`. Each part is at

```
https://ngrf.mudm.io/tiles/manc-full/features.parquet/zoom=<Z>/part_<NNN>.mu.parquet
```

2,031 parts over zooms 0 to 4, 13.6 GB in total. (The dataset's Croissant manifest lists
only the 3D Tiles entry point, so use the tile descriptor.) The script below downloads the
parts a descriptor names and skips parts already present. It needs `requests` and `brotli` (the descriptor is served brotli-encoded): use the
`bench` extra, or prefix the command with `uv run --with requests --with brotli`. Writing it under
`data/` keeps it out of version control:

```bash
cat > data/fetch_pyramid.py <<'EOF'
"""Download the Parquet parts listed in a catalog tile descriptor.

usage: python data/fetch_pyramid.py <dataset_url/> <out_dir> <zooms, comma-separated>
"""
import pathlib
import sys

import requests  # from the bench extra; it also decodes the brotli-encoded descriptor

base, out, zooms = sys.argv[1], pathlib.Path(sys.argv[2]), set(sys.argv[3].split(","))
desc = requests.get(base + "tilejson.json", timeout=60).json()
features = next(a for a in desc["assets"] if a["role"] == "features")
for part in features["parts"]:
    if part.split("/")[0].removeprefix("zoom=") not in zooms:
        continue
    dst = out / part
    if dst.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(base + features["href"] + part, timeout=600)
    resp.raise_for_status()
    tmp = dst.with_name("." + dst.name + ".part")  # hidden, so readers skip it
    tmp.write_bytes(resp.content)
    tmp.rename(dst)
print("wrote", out)
EOF
uv run python data/fetch_pyramid.py https://ngrf.mudm.io/tiles/manc-full/ data/manc-full 0,1,2,3,4
```

Then pass `--parquet data/manc-full` to the commands in the README. The last argument
selects zoom levels, so `0` fetches only the coarsest level (41 parts, 0.16 GB).

The served pyramid is a tiling of the same LOD-2 neuPrint meshes made before the mudm-tools **v0.6.4** release
(catalog command:
`download_manc.py --download --tile --max-neurons 30000 --lod 2 --workers 48 --max-zoom 4`).
The deposited runs used a **v0.6.4** re-tile of those meshes (Option B). The served
pyramid carries the same labels: the loader finds 23,546 neurons with `neuron_class` and
21,952 with `instance`, as in the experiments. Table 1 should reproduce within seed-to-seed variation:
a full run of the Table 1 protocol on an earlier v0.6.3 tiling of these meshes differed
from the deposited values by at most 0.0052 in any cell. The served copy itself has not
been rerun.

### Option B: regenerate the v0.6.4 pyramid from source

Needs a neuPrint token (<https://neuprint.janelia.org>, Account page).

```bash
# 1. neuPrint metadata (manc:v1.2.1) and LOD-2 meshes of every traced, typed neuron.
#    Writes <manc_dir>/metadata.json and <manc_dir>/meshes/<body_id>.obj.
NEUPRINT_TOKEN=<token> uv run --with cloud-volume --with requests python \
    <mudm-tools>/scripts/download_manc.py --download --max-neurons 30000 --lod 2 \
    --workers 16 --data-dir <manc_dir>

# 2. neuPrint field names -> muDM tag names (empty fields dropped).
uv run python -m mudm_paper.manc_prep <manc_dir>/metadata.json <manc_dir>/meta.json

# 3. Tile into a five-level Parquet pyramid with the mudm-tools Rust tiler.
#    Point TMPDIR at a partition with room for intermediate shards.
TMPDIR=<large_scratch_dir> uv run python -m mudm_paper.tile_parquet \
    --obj-dir <manc_dir>/meshes --meta <manc_dir>/meta.json --out <manc_parquet> \
    --max-zoom 4 --base-cells 100
```

`--max-neurons 30000` exceeds the number of traced, typed MANC neurons, so the query
returns all of them. `<manc_dir>/meshes` is also the `<manc_meshes>` input of the format,
read-path and fragment benchmarks.

## FlyWire (Extended Data Table 2, Fig. 2c, full-brain tiling)

FlyWire adult brain connectome, release 783 (Dorkenwald et al. 2024, Schlegel et al. 2024),
CC BY-NC 4.0. Meshes download anonymously; no token is needed. Treat everything you
derive from them under CC BY-NC 4.0 (non-commercial, with attribution).

### The 16,379-neuron subsample

```bash
# 0. Pin the annotation table to the revision used for the runs (the downloader keeps an existing file).
mkdir -p <flywire_dir> && curl -L -o <flywire_dir>/Supplemental_file1_neuron_annotations.tsv \
    https://raw.githubusercontent.com/flyconnectome/flywire_annotations/17fc57722002e1a7d38cdd0c89ac382bf92718da/supplemental_files/Supplemental_file1_neuron_annotations.tsv

# 1. Stratified selection and LOD-1 meshes.
uv run --with cloud-volume --with requests python <mudm-tools>/scripts/download_flywire.py \
    --download --select per-superclass --max-neurons 30000 --lod 1 --data-dir <flywire_dir>
#    -> <flywire_dir>/meshes/<root_id>.obj
#    -> <flywire_dir>/Supplemental_file1_neuron_annotations.tsv

# 2. Annotation table -> muDM label map for the downloaded neurons (prints class survival at min_class 20).
uv run python -m mudm_paper.flywire_prep \
    --annotations <flywire_dir>/Supplemental_file1_neuron_annotations.tsv \
    --obj-dir <flywire_dir>/meshes --out <flywire_dir>/flywire_meta.json

# 3. Tile.
TMPDIR=<large_scratch_dir> uv run python -m mudm_paper.tile_parquet \
    --obj-dir <flywire_dir>/meshes --meta <flywire_dir>/flywire_meta.json \
    --out <flywire_parquet> --max-zoom 4
```

The sweep commands for this pyramid are in the README (R1, R3).
`scripts/run_flywire_economy.sh` wraps the same steps in two stages; set `DATA` (required)
and `MUDM_TOOLS` (default: the current directory); `TMPDIR` defaults to `$DATA/tmp`.

**How the subsample is chosen.** `--select per-superclass` groups the annotation table by
`super_class`, sorts each group by `root_id` (as text), and keeps the first
floor(30000 / number of super classes) neurons of each group. With FlyWire's class sizes
this gives 16,379 neurons. The 30,000 cap on the total never binds, so no cell type is
truncated. Step 0 pins the annotation table to revision 17fc577 (2026-05-04), the latest before the
runs; the downloader otherwise fetches the `main` branch, which has since changed. Both
revisions select the same 16,379 root ids. The root ids of the deposited sample are not yet in this repository.

### The complete brain (Parquet-only tiling record)

`results/flywire_full_tiling.json` records the tiling of all 139,244 proofread neurons:

```bash
uv run --with cloud-volume --with requests python <mudm-tools>/scripts/download_flywire.py \
    --download --select all --lod 1 --data-dir <flywire_full_dir>          # about 728 GB of OBJ
uv run python -m mudm_paper.flywire_prep \
    --annotations <flywire_full_dir>/Supplemental_file1_neuron_annotations.tsv \
    --obj-dir <flywire_full_dir>/meshes --out <flywire_full_dir>/flywire_meta.json
```

The tiling command is B4 in the README.

## Xenium transcript point level of detail (Methods)

Inputs to `mudm_paper.transcript_lod` are the catalog's served 2D pyramids of three 10x
Genomics Xenium sections (CC BY 4.0): `xenium-breast-full` (FFPE Human Breast Cancer Rep1;
Janesick et al. 2023), `xenium-lung-full` and `xenium-ovary-full`. Each lists its parts in
`https://ngrf.mudm.io/tiles2d/<dataset>/tilejson.json` (zooms 0 to 9; points at 6 to 9;
parts named `zoom=<Z>/<layer>_part_<NNN>.mu.parquet`), so the same script downloads them:

```bash
uv run python data/fetch_pyramid.py https://ngrf.mudm.io/tiles2d/xenium-breast-full/ \
    data/xenium-breast-full 0,1,2,3,4,5,6,7,8,9
uv run python -m mudm_paper.transcript_lod --parquet data/xenium-breast-full \
    --out results/transcript_lod_xenium-breast-full.json
```

Repeat with `xenium-lung-full` and `xenium-ovary-full` (README, B3). The served breast
pyramid is the input of the deposited run: its finest level has the same 240 parts and
614,058,004 bytes recorded in `results/transcript_lod_xenium-breast-full.json`. The
catalog built these pyramids from the 10x output bundles with
`python -m mudm_tools.converters.cli convert -f xenium -i <Xenium_outs_dir> -o <out_dir>`.

## PanNuke (Extended Data Table 3, Extended Data Fig. 1)

PanNuke (Gamper et al. 2019, 2020): 256 × 256 px H&E patches with nucleus instance masks
from 19 tissues, **CC BY-NC-SA 4.0**. Source: the three fold files of the Hugging Face
copy at <https://huggingface.co/datasets/RationAI/PanNuke> (`data/fold{1,2,3}-00000-of-00001.parquet`).

The feature-stability pipeline (README, P1) works in one directory:

```
<pannuke_work>/raw/fold{1,2,3}-*.parquet      the Hugging Face fold files
<pannuke_work>/geojson/fold<f>_<row>.geojson  one muDM FeatureCollection per patch (row zero-padded to 5 digits)
<pannuke_work>/masks/tau_<eps>/               written by mudm_paper.rasterize
<pannuke_work>/images/                        written by scripts/extract_pannuke_images.py
<pannuke_work>/features/                      written by scripts/nyxus_pipeline.py and scripts/stability_analysis.py
```

The GeoJSON patches are the muDM ingest of PanNuke: each nucleus mask traced as a
polygon in pixel coordinates without simplification, 7,556 patches and about 187,000
nuclei in total. The conversion script is not yet in this repository.
`scripts/extract_pannuke_images.py` checks that patch order matches the fold rows (by IoU
against the rasterized full-detail masks) before it writes the intensity images.

Extended Data Fig. 1 reads two small deposited files in `figures/data/`:
`pannuke_2d_lod_fidelity.json` (mean, median and 5th-percentile IoU and retained-vertex
storage per tolerance, over 38,078 nuclei in every fifth patch, 1,512 patches) and
`pannuke_rep_nucleus.json` (the 87-vertex representative nucleus). The script that
measured them is not yet in this repository. Both files are PanNuke-derived and
CC BY-NC-SA 4.0.

## Other inputs

- `figures/data/manc_rep_neuron_lods.npz` (Fig. 2a): one MANC neuron (feature 6546, body
  155112, cell type IN13A065, near the median size) at all five levels of the v0.6.4
  pyramid, with border-duplicated faces removed. CC BY 4.0, from MANC.
- The fusion runs use `results/manc_fusion_features.json`, built from the deposited
  neuPrint rows (README, F1).

## Citation

Cite the source datasets when you use these inputs: MANC (Takemura et al. 2024; Marin
et al. 2024) and neuPrint (Plaza et al. 2022); FlyWire (Dorkenwald et al. 2024; Schlegel
et al. 2024; Zheng et al. 2018); PanNuke (Gamper et al. 2019, 2020); Xenium (Janesick
et al. 2023 for the breast section; 10x Genomics for the lung and ovary sections). Full
references are in the paper. The catalog records provenance and licence per dataset at
<https://ngrf.mudm.io>.
