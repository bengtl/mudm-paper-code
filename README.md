# muDM paper: analysis code and result artifacts

Companion repository for *muDM: A Queryable Multi-Resolution Data Model for Vector
Geometry in Spatial Bioimaging* (Ljungquist, Hotaling and Schaub). It holds the
analysis code behind the paper's quantitative results, the result artifacts the
numbers are read from, and the scripts that render the tables and figures.

The muDM data model and the tiling pipeline are released separately and are
dependencies here:

- **`mudm`** (data model): <https://github.com/NovagenResearch/mudm>, PyPI `mudm==0.6.4`
- **`mudm-tools`** (pipeline, viewers, benchmark scripts): <https://github.com/NovagenResearch/mudm-tools>, PyPI `mudm-tools==0.6.4`
- **Public data catalog**: <https://ngrf.mudm.io> (dataset records at <https://ngrf.mudm.io/datasets.json>)
- **Format specification**: <https://novagenresearch.github.io/mudm/>

## Layout

```
mudm_paper/                analysis package; run modules as python -m mudm_paper.<module>
  run_manc.py              sweep entry point: Table 1 and every sweep variant (one JSON per task)
  sweep.py                 level x task x seed grid; per-epoch validation/test curves (--curve)
  data.py                  Parquet-pyramid loader and leakage-safe grouped train/val/test splits
  models.py                DGCNN (masked variable-size pooling, optional fused scalars), PointNet, counts-only MLP
  sampling.py              budget-limited point sampling (random, FPS, on-the-fly QEM)
  fps.py                   farthest-point sampling
  qem.py                   on-the-fly QEM decimation through the mudm-tools Rust simplifier
  cost.py                  per-level storage from Parquet metadata; FLOPs and peak-memory helpers
  stats.py                 mean and s.e.m., paired t-test, TOST, Holm correction
  knee.py                  legacy knee fields still written into each run JSON (not used by the paper)
  tile_parquet.py          OBJ meshes + label map -> Parquet mesh pyramid (mudm-tools Rust tiler)
  manc_prep.py             neuPrint MANC metadata -> muDM tag map
  flywire_prep.py          FlyWire annotation table -> muDM tag map
  fusion_features.py       neuPrint synapse rows -> per-neuron count features for the fusion runs
  permute_features.py      fixed permutation of the count features (fusion permutation control)
  synapse_probe.py         counts-only leakage probe (logistic regression, gradient-boosted trees)
  leakage_analysis.py      serial-homolog exposure of the instance-grouped split
  bench_formats.py         format spec-sheet, read-path and HTTP read benchmarks
  fragment_stats.py        per-level vertex/face counts, border duplicates, comparison with source OBJ
  transcript_lod.py        point level-of-detail evaluation for Xenium transcript layers
  rasterize.py             simplified polygon levels -> instance masks for Nyxus
  tests/                   pytest suite on synthetic pyramids (no GPU, no downloads)
figures/
  make_table1.py           Table 1 and the economy variants (retention, storage spans)
  make_ed_table3.py        Extended Data Table 3 (PanNuke feature stability); --full or --csv <file> for all 144 features
  make_ed_table4.py        Extended Data Table 4 (fusion), Holm, substitution test and fusion controls
  fig2_economy.py          Fig. 2
  fig_s_2d_lod.py          Extended Data Fig. 1
  _palette.py              shared colour palette and matplotlib settings
  data/                    figure inputs: manc_rep_neuron_lods.npz (Fig. 2a), pannuke_*.json (ED Fig. 1)
scripts/
  extract_pannuke_images.py, nyxus_pipeline.py, stability_analysis.py
                           PanNuke Nyxus pipeline behind Extended Data Table 3
  run_flywire_economy.sh   two-stage FlyWire download/tile/sweep driver used for Extended Data Table 2
results/                   deposited result JSONs, PROVENANCE.md (one entry per file), _schema_validator.py
data/README.md             how to obtain the inputs (served catalog or regeneration from source)
CITATION.cff  LICENSE  LICENSE-DATA.md  pyproject.toml  uv.lock
```

## Install

The project is managed with [uv](https://docs.astral.sh/uv/); `uv.lock` pins every
dependency.

```bash
uv sync --extra figures --extra dev      # tables, figures and the test suite
```

Add extras for specific tasks (`uv sync` installs exactly the extras you name, so list
all you need, or use `uv sync --all-extras`):

| Extra | Needed for |
|---|---|
| `figures` | matplotlib and shapely for `figures/*.py` |
| `dev` | pytest |
| `bench` | `bench_formats` (HTTP reads, Draco and Meshopt decoders), and the served-pyramid download in `data/README.md`. `meshoptimizer` is published only as a source release, so this extra needs a C++ compiler |
| `probe` | scikit-learn for `synapse_probe` |
| `stability` | Nyxus 0.11.0, tifffile, pandas, pillow for the PanNuke pipeline in `scripts/` |

The sweeps need a CUDA GPU in practice. `--device` defaults to `cuda:0` when CUDA is
available and to the CPU otherwise.

## Paper claim to artifact

Every row is a manuscript statement that cites the companion repository or a deposited
artifact. All files are in `results/` unless another folder is given; the last column
points to the command in [Reproduce](#reproduce). `results/PROVENANCE.md` describes each
file, how it was produced and with which software versions.

| Claim (where) | Artifact | Reproduce |
|---|---|---|
| Table 1: balanced accuracy per level, storage 0.29 to 28.55 GB (98-fold), on disk 0.16 to 10.14 GB (62-fold) (Results; Methods, economy readout) | `manc_economy_{neuron_class,subclass,hemilineage}.json` | R1, S1 |
| Fig. 2: (a) one MANC neuron at five levels, (b) MANC, (c) FlyWire | `figures/data/manc_rep_neuron_lods.npz`, `manc_economy_*.json`, `flywire_economy_*.json` | S1, R1 |
| FlyWire economy: 16,379 neurons, 0.6% of storage over a 155-fold span (119-fold on disk), 91 to 101% retention (Results; Extended Data Table 2) | `flywire_economy_{super_class,cell_class,hemilineage}.json` | R1, S2 |
| Read path: 0.91 ms per neuron, 32 to 197 times faster than 3D Tiles, Neuroglancer or OBJ; 3 × 3 window ninefold; 11,745 neurons with records in 2.1 s vs about 16 min (Results); 10, 83 and 330 ms per-object fetch (Methods); 28.8 and 70.0 ms per neuron (Extended Data, structural analysis) | `manc_access_benchmark.json` | B1 |
| Encoder peak memory 141 GB (3D Tiles) and 200 GB (Neuroglancer); per-format encode, decode, size and vertex fidelity (Methods, storage) | `manc_format_benchmark.json` | B1 |
| Finest level: unique vertices and faces identical to the source meshes for all 23,546 MANC neurons (Results; Methods, storage) | `obj_dedup_all.json` | producer pending, see PROVENANCE |
| Border duplicates: 3.3% of faces and 5.9% of vertices at the finest level (Methods, pyramid construction and neuron-typing protocol) | `manc_fragment_stats.json` | B2 |
| Coarse levels are triangle patches; median of about 100 components per MANC neuron (Methods; Fig. 2a legend) | `coarse_topology.json` | producer pending, see PROVENANCE |
| Point level of detail: three Xenium sections, 19 to 43 million transcripts, 541 codewords; median per-codeword deviation 1.0 to 1.5%, no codeword lost, 5 to 8% tile-gene dropout (Methods) | `transcript_lod_xenium-{breast,lung,ovary}-full.json` | B3 |
| Full FlyWire Parquet tiling: 139,244 neurons, 728 GB source, 5 h 4 min, 33 GB peak (Methods, storage; Discussion) | `flywire_full_tiling.json` (transcribed run record) | B4 |
| Feature stability across polygon levels; ρ ≥ 0.99 at the 11%-vertex level (Results; Extended Data Table 3 and continuation; Methods) | `nyxus_stability_summary.json`, `figures/data/pannuke_2d_lod_fidelity.json` | P1, S1 |
| Extended Data Fig. 1: polygon level of detail on 38,078 PanNuke nuclei | `figures/data/pannuke_2d_lod_fidelity.json`, `figures/data/pannuke_rep_nucleus.json` | S1 |
| Fusion with linked synapse counts: gains 0.07 to 0.12 and 0.04 to 0.09, counts alone 0.067, Holm p = 0.052, substitution 0.852 vs 0.800 (p = 4 × 10⁻³; subclass p = 0.08) (Results; Extended Data Table 4; Methods) | `manc_fusion_*.json`, `manc_countsonly_*.json`, `manc_economy_*.json` | F2, S1 |
| Fusion control 1, counts permuted among neurons: max abs change 0.013, no cell Holm-significant (Methods) | `manc_fusionperm_*.json` | F1, F2, S1 |
| Fusion control 2, cell types held out (Methods) | `manc_fusion_leakctrl_*.json`, `manc_leakctrl_*.json` | F2, S1 |
| Counts-only leakage probe run before the fusion experiment (Methods) | `synapse_probe_results.json` | F3 |
| Linked synapse counts from neuPrint `manc:v1.2.1`, retrieved 11 August 2026 (Methods; Data availability) | `manc_synapse_meta.json`, `manc_fusion_features.json` | F1 |
| Cell-type-held-out split gives the same trade-off; 38 to 42% serial-homolog exposure, no group crossing train/test (Methods; Supplementary Note 2) | `manc_leakctrl_*.json`; exposure printed by `leakage_analysis` | R2, S2 |
| PointNet retains 94, 93 and 99% (Methods; Supplementary Note 2) | `manc_pointnet_{neuron_class,subclass,hemilineage}.json` | R2, S2 |
| Input budgets 2,048 and 4,096: retention 93, 87, 99% at 512 and 98, 97, 99% at 4,096; absolute accuracy lower at larger budgets under either training length (Methods; Supplementary Note 2) | `manc_budget{2048,4096}_{neuron_class,hemilineage}.json`, `flywire_budget{2048,4096}_super_class.json`, `manc_curve{512,2048,4096}_neuron_class.json` | R3, S2 |
| 30-epoch recipe with validation-based epoch selection: neuron class 0.75 to 0.87 (coarsest) and 0.80 to 0.90 (finest), retention 91 to 96%, 93% at 2,048, hemilineage intermediate levels within 2.3% (Methods; Supplementary Note 2) | `manc_economy30_*.json`, `manc_curve2048_neuron_class.json` | R4, S3 |
| Cell type has no leakage-safe split; the per-neuron split gives 0.40 at the finest level with 76% retained (Methods; Supplementary Note 2) | `manc_ungrouped_cell_type.json`, `split_sizes.json` | R5, S2 (`split_sizes.json`: producer pending, see PROVENANCE) |
| Served connectomes: 23,546 MANC, 139,244 FlyWire, 176,443 male CNS, 134,934 BANC, 25,000 hemibrain (Results; Extended Data Table 5) | `deployed_counts_zoom0.json` | producer pending, see PROVENANCE |
| Xenium breast Rep1 ingest: 169,088 cells, 541 codewords (313 expression, 228 controls and blanks), 42.6 M transcripts (Methods; Extended Data Table 5) | `B2_xenium_ingest_validation.json` | see PROVENANCE |
| glTF traversal micro-probe on a separate tileset (Extended Data, structural analysis) | `B6_gltf_decoder_comparison.json` | B5 |
| OGC 3D Tiles validator report (Methods, mesh codecs; Supplementary Note 1) | `tile300_fixed_validation.json` | see PROVENANCE |
| Per-seed values for every reported mean (Methods) | the `raw` arrays inside each run JSON | |

The 30-epoch variants of the other sweeps (`manc_leakctrl30_*`, `flywire_economy30_*`,
`manc_fusion30_*`, `manc_countsonly30_*`, `manc_fusionperm30_*`,
`manc_fusion_leakctrl30_*`) and the `*_cell_type.json` runs are deposited for completeness;
the paper does not quote them.

**What this repository covers, and what lives elsewhere.** Every deposited file is
described in `results/PROVENANCE.md`, and the figure inputs and source datasets in
`data/README.md`. Most have a producing command in this repository. The exceptions are
listed in those two files: two producer scripts (the finest-level fidelity check and the
coarse-topology count) will be added; the split sizes, the served-catalog counts, the
figure inputs in `figures/data/` and the PanNuke polygon conversion were made with helpers
that are not yet included; two records were transcribed from run logs; and the root ids of
the 16,379-neuron FlyWire subsample are not deposited (they are regenerated
deterministically, see `data/README.md`).
Catalog contents and dataset scales (over 100 datasets, Extended Data Table 5 apart from
the rows above) are documented by the catalog itself at <https://ngrf.mudm.io>.
Properties of the pipeline (encoder byte identity, Neuroglancer determinism, the
lossless Draco path, shard hashing, clipping and simplification) are covered by the
`mudm-tools` test suite. The viewer checks in Supplementary Note 1 are not deposited here.

## Reproduce

Run everything from the repository root. Paths in angle brackets are placeholders.

### Render tables and figures from the deposited artifacts (no GPU, no downloads)

```bash
uv run python figures/make_table1.py                              # S1: Table 1
uv run python figures/make_ed_table3.py                           # S1: ED Table 3 (add --full or --csv <file> for all 144 features)
uv run python figures/make_ed_table4.py                           # S1: ED Table 4, Holm, substitution test, both fusion controls
uv run python figures/fig2_economy.py                             # S1: Fig. 2, writes figures/fig2_economy.pdf and .png
uv run python figures/fig_s_2d_lod.py                             # S1: writes figures/fig_s_2d_lod.pdf and .png
uv run python results/_schema_validator.py results/*.json         # header check of every result file
```

**S2, economy variants.** `make_table1.py` reads any five-level sweep prefix (for `manc_fusion`
add `--tasks neuron_class,subclass,hemilineage`) and prints coarsest-level
retention and the storage spans:

```bash
uv run python figures/make_table1.py --prefix flywire_economy     # ED Table 2
uv run python figures/make_table1.py --prefix manc_leakctrl       # cell types held out
uv run python figures/make_table1.py --prefix manc_pointnet       # PointNet
uv run python figures/make_table1.py --prefix manc_budget4096     # also manc_budget2048, flywire_budget2048, flywire_budget4096
uv run python figures/make_table1.py --prefix manc_ungrouped --tasks cell_type
```

**S3, 30-epoch recipe.** These runs also record `bal_acc_at_best_val`, the test score at
the epoch with the best validation score, which is what Supplementary Note 2 reports:

```bash
uv run python - <<'EOF'
import glob
import json

files = sorted(glob.glob("results/manc_economy30_*.json")) + ["results/manc_curve2048_neuron_class.json"]
for f in files:
    d = json.load(open(f))
    task, budget = d["config"]["labels"][0], str(d["config"]["budgets"][0])
    acc = {int(z): v[budget]["bal_acc_at_best_val"]["mean"] for z, v in d[task]["precomputed"].items()}
    levels = " ".join(f"z{z}={a:.3f}" for z, a in sorted(acc.items()))
    print(f"{f}: {levels}  retention {acc[min(acc)] / acc[max(acc)]:.0%}")
EOF
uv run python figures/make_ed_table4.py --suffix 30 --metric bal_acc_at_best_val
uv run python figures/fig2_economy.py --manc-prefix manc_economy30 --flywire-prefix flywire_economy30 \
    --metric bal_acc_at_best_val --name fig2_economy_30ep
```

### Obtain the pyramids

See [`data/README.md`](data/README.md): the served MANC pyramid from the catalog, the
v0.6.4 re-tile from source, the FlyWire subsample, PanNuke and the Xenium inputs. Below,
`<manc_parquet>` and `<flywire_parquet>` are pyramid roots holding `zoom=<Z>/` directories,
and `<manc_meshes>` is the directory of source `<body_id>.obj` meshes.

### Rerun the sweeps (GPU)

Each command writes `results/<prefix>_<task>.json`, one file per task. Without
`--labels`, the tasks are `neuron_class`, `subclass` and `hemilineage`; add
`--labels cell_type` to regenerate the deposited `*_cell_type.json` files.

**R1, Table 1 and the FlyWire economy** (DGCNN, levels z0 to z4, 512 vertices, 15 seeds,
8 epochs, batch 64, instance-grouped split):

```bash
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/
uv run python -m mudm_paper.run_manc --parquet <flywire_parquet> --out-dir results/ \
    --labels super_class,cell_class,hemilineage --group-field cell_type --prefix flywire_economy
```

**R2, cell types held out, PointNet and the exposure analysis:**

```bash
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --group-field cell_type --prefix manc_leakctrl
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --model pointnet --prefix manc_pointnet
uv run python -m mudm_paper.leakage_analysis <manc_parquet>       # optional second argument: number of seeds (default 15)
```

**R3, input budgets** (batch size halved per budget step):

```bash
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --labels neuron_class,hemilineage --budgets 2048 --batch 32 --prefix manc_budget2048
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --labels neuron_class,hemilineage --budgets 4096 --batch 16 --prefix manc_budget4096
uv run python -m mudm_paper.run_manc --parquet <flywire_parquet> --out-dir results/ \
    --labels super_class --group-field cell_type --budgets 2048 --batch 32 --prefix flywire_budget2048
uv run python -m mudm_paper.run_manc --parquet <flywire_parquet> --out-dir results/ \
    --labels super_class --group-field cell_type --budgets 4096 --batch 16 --prefix flywire_budget4096
```

**R4, 30-epoch recipe and learning curves.** `--curve` evaluates validation and test after
every epoch; the 30-epoch variants of the other sweeps take the same two flags and a
`30` suffix on the prefix (for example `--prefix manc_fusion30`):

```bash
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --epochs 30 --curve --prefix manc_economy30
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --labels neuron_class --levels 0,4 --seeds 3 --epochs 30 --curve --prefix manc_curve512
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --labels neuron_class --levels 0,4 --seeds 3 --epochs 30 --curve --budgets 2048 --batch 32 --prefix manc_curve2048
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --labels neuron_class --levels 0,4 --seeds 3 --epochs 30 --curve --budgets 4096 --batch 16 --prefix manc_curve4096
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --labels hemilineage --seeds 3 --epochs 30 --curve --prefix manc_curve512
```

**R5, cell type with a per-neuron split** (an empty grouping field disables grouping, so
bilateral twins can fall on both sides of the split):

```bash
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --labels cell_type --group-field "" --prefix manc_ungrouped
```

### Fusion inputs and runs

**F1, feature tables.** Both commands reproduce the deposited tables exactly:

```bash
uv run python -m mudm_paper.fusion_features results/manc_synapse_meta.json results/manc_fusion_features.json
uv run python -m mudm_paper.permute_features results/manc_fusion_features.json <permuted_features.json>   # seed 0
```

**F2, fusion arm, counts-only arm and both controls:**

```bash
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --model dgcnn_fusion --extra-features results/manc_fusion_features.json --prefix manc_fusion
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --model counts_mlp --extra-features results/manc_fusion_features.json --prefix manc_countsonly
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --model dgcnn_fusion --extra-features <permuted_features.json> --prefix manc_fusionperm
uv run python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ \
    --model dgcnn_fusion --extra-features results/manc_fusion_features.json --group-field cell_type --prefix manc_fusion_leakctrl
```

**F3, counts-only leakage probe** (needs the `probe` extra). It takes an optional `--out <file>`, reads
`results/manc_synapse_meta.json` (it queries neuPrint only when that file is missing, with
a token in `~/.neuprint_token`) and by default **overwrites** `results/synapse_probe_results.json` (use `--out <tmp.json>` for a non-destructive rerun):

```bash
uv run python -m mudm_paper.synapse_probe
```

### Benchmarks and pyramid statistics (CPU, large disk)

**B1, format spec-sheet and read path** (needs the `bench` extra). The spec-sheet run
encodes every format from one shared tile index into `<formats_dir>`; the access run then
reads that directory:

```bash
uv run python -m mudm_paper.bench_formats --mode spec-sheet --obj-dir <manc_meshes> \
    --out results/manc_format_benchmark.json --out-root <formats_dir> --n 200 --iters 10 --max-zoom 4 --no-sharded
uv run python -m mudm_paper.bench_formats --mode access --obj-dir <manc_meshes> --out-root <formats_dir> \
    --out results/manc_access_benchmark.json --n 50 --iters 5 --bulk-n 2000 \
    --records results/manc_fusion_features.json --join-n 200
```

**B2, border duplicates and per-level counts:**

```bash
uv run python -m mudm_paper.fragment_stats --parquet <manc_parquet> --zooms 0,1,2,3,4 \
    --obj-dir <manc_meshes> --workers 16 --out results/manc_fragment_stats.json
```

**B3, transcript point level of detail** (one run per Xenium pyramid; the defaults are
the deposited settings):

```bash
uv run python -m mudm_paper.transcript_lod --parquet <xenium_breast_parquet> --out results/transcript_lod_xenium-breast-full.json
```

**B4, full FlyWire Parquet tiling** (timed with `/usr/bin/time -v` on Linux; inputs in
`data/README.md`):

```bash
TMPDIR=<large_scratch_dir> /usr/bin/time -v uv run python -m mudm_paper.tile_parquet \
    --obj-dir <flywire_full_meshes> --meta <flywire_full_meta.json> --out <pyramid_dir> \
    --max-zoom 4 --base-cells 100 --io-threads 12
```

**B5, glTF traversal micro-probe.** The script lives in a `mudm-tools` checkout:

```bash
uv run python <mudm-tools>/scripts/benchmark_gltf_decoder.py --tiles-dir <glb_tiles_dir> \
    --sample 50 --iters 20 --output results/B6_gltf_decoder_comparison.json
```

### PanNuke feature stability (P1)

Needs the `stability` extra and a PanNuke work directory prepared as described in
`data/README.md`. The scripts take the work directory as their only argument:

```bash
uv run python -m mudm_paper.rasterize <pannuke_work>/geojson --out <pannuke_work>/masks --eps 0.5 1 2 4 8 --shape 256 256
uv run python scripts/extract_pannuke_images.py <pannuke_work>     # intensity images, with an alignment check
uv run python scripts/nyxus_pipeline.py <pannuke_work>             # Nyxus features per level
uv run python scripts/stability_analysis.py <pannuke_work>         # writes <pannuke_work>/features/stability_summary.json
cp <pannuke_work>/features/stability_summary.json results/nyxus_stability_summary.json
```

## Tests

```bash
uv run pytest -q
```

The suite runs on synthetic pyramids and needs no GPU or downloads. With only the
`figures` and `dev` extras, the HTTP benchmark tests (need `bench`) and one mask-writing
test of the rasterizer (needs `stability`) are skipped; everything else runs.

## Reproducibility notes

- The sweeps reproduce **within seed-to-seed variation, not bit for bit**: DGCNN's
  k-nearest-neighbour graph and cuDNN are not deterministic on GPU. Balanced accuracy is
  reported as mean and s.e.m. over seeds, and every JSON keeps the per-seed values.
- The deposited runs used an 80-core Linux host with two NVIDIA RTX A4500 GPUs, Python
  3.13.2, torch 2.12.0+cu130, pyarrow 23.0.1, numpy 2.4.4, scipy 1.17.1 and the released
  `mudm`/`mudm-tools` 0.6.4 wheels. `uv.lock` pins the same `mudm`/`mudm-tools` releases
  but later versions of torch (2.13.0), pyarrow (25.0.1), numpy (2.5.2) and scipy (1.18.0).
- Table 1 and every 8-epoch file report the final-epoch score (`bal_acc`). The 30-epoch
  files add `bal_acc_at_best_val` and per-epoch curves.
- The experiments read a v0.6.4 re-tile of the MANC meshes. The catalog serves a tiling
  of the same meshes made before the v0.6.4 release (see `data/README.md`).
- Paths and host names inside the JSON `config` blocks refer to the benchmark host.
- The `knee` block in each run JSON is legacy and not used by the paper.

## Citation

If you use this code or the deposited artifacts, please cite the paper (preprint DOI to
follow) and the `mudm` and `mudm-tools` releases. `CITATION.cff` carries the same
information.

> Ljungquist B., Hotaling N., Schaub N. muDM: A Queryable Multi-Resolution Data Model
> for Vector Geometry in Spatial Bioimaging. Preprint, 2026 (DOI to follow).

```bibtex
@misc{ljungquist2026mudm,
  title  = {muDM: A Queryable Multi-Resolution Data Model for Vector Geometry in Spatial Bioimaging},
  author = {Ljungquist, Bengt and Hotaling, Nathan and Schaub, Nick},
  year   = {2026},
  note   = {Preprint; DOI to follow. Code: https://github.com/bengtl/mudm-paper-code}
}
```

## Licence

Code is MIT-licensed (`LICENSE`), matching `mudm` and `mudm-tools`. Files derived from
third-party datasets keep the upstream licence: PanNuke-derived files are CC BY-NC-SA 4.0,
FlyWire-derived files CC BY-NC 4.0, and MANC- and Xenium-derived files CC BY 4.0.
`LICENSE-DATA.md` lists the files.
