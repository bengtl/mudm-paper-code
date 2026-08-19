# Data — obtaining the MANC pyramid

The economy sweep reads a tiled muDM pyramid laid out as:

```
<parquet_dir>/zoom=<Z>/part_*.parquet     # Z = 0 (coarsest) .. 4 (finest)
```

Each row carries packed `float32` positions, `uint32` triangle indices, and a tag
map that must include `body_id`, `instance`, and the four classification targets
`neuron_class`, `subclass`, `hemilineage`, `cell_type`. Nothing in this directory is
committed (see `.gitignore`); obtain it one of two ways.

## Option A — fetch the served pyramid (fast)

The tiled MANC pyramid is published in the public catalog as **`manc-full`**:

- Dataset page: <https://ngrf.mudm.io> (browse to *MANC — Drosophila ventral nerve cord*)
- Machine-readable manifest (Croissant) enumerating the Parquet distributions:
  <https://ngrf.mudm.io/datasets/manc-full/croissant.json>

Download the `*.mu.parquet` tiles into `<parquet_dir>/zoom=<Z>/` preserving the
hive-partitioned `zoom=` layout above.

## Option B — regenerate from source (from scratch)

Requires the pinned `mudm-tools` (installed with this package) and neuPrint access.

```bash
# 1. Download MANC neuron meshes + annotations from neuPrint (mudm-tools script)
python -m mudm_tools.scripts.download_manc --download --out <objs_dir>
#    (equivalently: mudm-tools/scripts/download_manc.py --download)

# 2. Adapt neuPrint metadata into the muDM tag map
python -m mudm_paper.manc_prep <objs_dir>/metadata.json meta.json

# 3. Tile OBJ meshes into the multi-resolution Parquet pyramid
python -m mudm_paper.tile_parquet --obj-dir <objs_dir> --meta meta.json --out <parquet_dir>
```

Run `python -m mudm_paper.tile_parquet --help` / `python -m mudm_paper.manc_prep`
for exact arguments.

## Provenance / citation

MANC *Drosophila* ventral-nerve-cord connectome, obtained through neuPrint. Cite the
MANC connectome and neuPrint (references in the paper). Data are redistributed under
their upstream licence; the catalog records per-dataset provenance and licences.

## FlyWire second-connectome economy (subsample)

Reproduces the Table 1 trade-off on a second, independent connectome. FlyWire's
public release is **CC BY-NC 4.0**: the public catalog serves a small curated
FlyWire subsample as tiles under that licence with attribution, but the full
16,379-neuron experiment pyramid is not served — regenerate it locally as
follows, and treat the result as CC BY-NC 4.0 (non-commercial, attribute the
FlyWire papers). Step 1 needs
a `mudm-tools` git checkout (its `download_flywire.py` is not in the PyPI wheel) with
`cloud-volume` available; the remaining steps use this package.

```bash
# 1. Class-stratified subsample of FlyWire meshes + the annotation TSV
#    (anonymous cloud-volume, no CAVE token; run from the mudm-tools checkout):
python scripts/download_flywire.py --download \
    --select per-superclass --max-neurons 30000 --lod 1 --data-dir data/flywire
#    -> data/flywire/meshes/<root_id>.obj
#    -> data/flywire/Supplemental_file1_neuron_annotations.tsv

# 2. Annotations -> muDM label JSON (restricted to the downloaded subsample);
#    prints the per-target class-survival breakdown at min_class=20:
python -m mudm_paper.flywire_prep \
    --annotations data/flywire/Supplemental_file1_neuron_annotations.tsv \
    --obj-dir data/flywire/meshes --out data/flywire/flywire_meta.json

# 3. Tile to an f32 Parquet pyramid (QEM-decimated per level; set TMPDIR on the box):
TMPDIR=/data/ai/mudm-paper/tmp python -m mudm_paper.tile_parquet \
    --obj-dir data/flywire/meshes --meta data/flywire/flywire_meta.json \
    --out data/flywire/pyramid --max-zoom 4

# 4. Economy sweep, leakage-safe split grouped by cell_type:
python -m mudm_paper.run_manc --parquet data/flywire/pyramid --out-dir results/ \
    --labels super_class,cell_class,hemilineage --group-field cell_type \
    --prefix flywire_economy
```

`--max-neurons` caps by whole cell types (preserving bilateral symmetry). The
`cell_type` (finest) task is expected near-floor on FlyWire (thousands of types),
mirroring MANC's 117-class column; add it back with a coarser grouping if wanted.
The `super_class -> cell_class -> hemilineage` ladder is the FlyWire analog of MANC's
`neuron_class -> subclass -> hemilineage`.

