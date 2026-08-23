# Benchmark Results

Result artifacts for the Nature Methods submission (right-sized level-of-detail
reframe). One JSON file per benchmark.

## Convention

Figures and tables in the paper are produced by reading these JSON files. Every
quantitative claim in the paper should trace back to one of these files (or to
the companion data catalogue, for dataset-scale numbers — see note below). If a
number in the paper cannot be traced, the claim must be removed or the benchmark
rerun.

## v0.6.4 re-run (2026-08-19/20)

Every ML result artifact below (economy, leakage controls, fusion + its two
controls, FlyWire economy, PointNet) was regenerated on 2026-08-19/20 under the
**released `mudm==0.6.4` / `mudm-tools==0.6.4` wheels**, on pyramids re-tiled
under v0.6.4, with the same training stack as the original runs (torch
2.12.0+cu130, Python 3.13.2, 80-core + 2x A4500 host). Values match the
superseded v0.6.3-era runs within seed noise (largest per-cell drift in Table 1:
0.005); the superseded artifacts are preserved in git history. The deployed
public catalog remains a v0.6.3 product and is documented as such in the paper.

## Files present

- `manc_economy_{neuron_class,subclass,hemilineage,cell_type}.json` — **source for
  Table 1** (storage versus balanced accuracy across the MANC 3D-mesh pyramid).
  DGCNN, 512-vertex budget, 15 seeds, uncapped, precomputed levels z0–z4. Each
  file carries per-seed balanced-accuracy `raw` arrays, per-level storage
  (`cost.storage_per_level`; z0–z4 uncompressed 0.29 / 0.54 / 1.39 / 3.89 / 28.55
  GB, a 98.2× range) and the run `config` (epochs 8, batch 64, seeds 15). Original
  E2-confirm run 2026-07-16; regenerated 2026-08-19/20 under v0.6.4 (see note above).
- `manc_leakctrl_{neuron_class,subclass,hemilineage}.json` — **leakage-controlled
  robustness check** for the serial-homolog concern (Methods). Same paper config,
  but the train/val/test split is grouped by `cell_type` (holding out entire cell
  types, so neither serial nor bilateral homologs appear in both train and test),
  15 seeds. Coarsest-level retention stays within a few points of the
  instance-grouped Table 1 (93/87/87% at neuron class / subclass / hemilineage,
  versus 93/89/89% here), so the storage-versus-accuracy trade-off is not an
  artifact of homolog leakage. Regenerate with
  `run_manc --out-dir <dir> --group-field cell_type --labels neuron_class,subclass,hemilineage --prefix manc_leakctrl`.
  The 38–42% serial-homolog exposure figure cited in Methods is reproduced by
  `python -m mudm_paper.leakage_analysis <parquet_dir>`.
- `hemibrain_5seed.json` — **RETIRED, do not cite.** Artifact of the disproven
  ``coarse-beats-fine'' LOD headline (Hemibrain, two-branch PointNet); the effect
  was traced to a vertex-cap truncation artifact and removed from the paper. No
  longer sources any figure or table; fails the current schema (no `timestamp`).
  Kept pending an owner decision to delete.
- `hemibrain_economy.json` — **RETIRED, do not cite.** FLOPs / memory / storage for
  the same disproven Hemibrain headline (former Fig. 2b), superseded by the MANC
  storage numbers in `manc_economy_*.json`; fails the current schema (no
  `timestamp`). Kept pending an owner decision to delete.
- `B2_xenium_ingest_validation.json` — Xenium FFPE Human Breast Cancer Rep1
  full-slide ingest validation at **mudm-tools v0.6.3** (the ingest that produced the
  companion-catalogue dataset `xenium-breast-full`): **169,088 cells**, 169,083 nuclei,
  42,638,083 transcripts, 541-codeword panel (313 expression [280 predesigned + 33 custom]
  + 228 controls/blanks), DAPI raster at 0.2125 µm/px. These are the numbers used in
  Methods and the Supplementary catalogue. Re-deposited 2026-06-04 from the v0.6.3
  convert log on oden, **superseding** the earlier v0.5.0 spike (which reported 167,780);
  the prior value is retained in the file's `supersedes` block.
- `B6_gltf_decoder_comparison.json` — Rust `gltf` decoder vs `pygltflib` per-tile
  decode timing. **Source for the Extended Data structural-analysis** glTF-traversal
  figure (≈195× gap; a benchmark probe isolating scene-graph traversal, not the
  Parquet ML read path).
- `hemibrain_fidelity.json` — single-seed (seed 42), single-branch PointNet
  round-trip check, **source for Extended Data Table 2** (lossless-geometry
  evidence). Parquet tiles vs raw OBJ on 4,074 hemibrain neurons spanning the 20
  most common cell types (Kenyon / lobula-columnar / ORN), 1,024 sampled points:
  accuracy 94.4 % vs 95.1 %, macro-F1 93.5 % vs 94.2 %, mean epoch 22.9 s vs
  24.6 s, peak 10.0 MB vs 9.7 MB (Parquet ≈ raw OBJ, lossless geometry). A
  separate dataset and model from the Table 1 economy experiment, by design.
  Regenerated on oden 2026-06-04 from a fresh top-20-type neuPrint download
  (`data/hemibrain_fidelity/`, 4,545 meshes → top-20 cohort). `benchmark_id` is
  set to the filename stem `hemibrain_fidelity`.
- `flywire_economy_{super_class,cell_class,hemilineage}.json` — **source for
  Extended Data Table 3** (FlyWire second-connectome economy). Class-stratified
  16,379-neuron subsample of the FlyWire adult-brain release (CC BY-NC 4.0,
  regenerated locally per `data/README.md`, not redistributed), identical DGCNN
  protocol (512-vertex budget, 15 seeds, levels z0–z4), split grouped by whole
  `cell_type`. Run on the same box, 2026-07-20. Regenerate with
  `run_manc --parquet <flywire_pyramid> --out-dir <dir> --labels super_class,cell_class,hemilineage --group-field cell_type --prefix flywire_economy`.
- `manc_fusion_{neuron_class,subclass,hemilineage,cell_type}.json` and
  `manc_countsonly_{...}.json` — **source for Extended Data Table 5** and the
  Results fusion subsection. Three pre-specified arms under identical splits and
  seeds: geometry-only (Table 1 files above), geometry+counts
  (`--model dgcnn_fusion --extra-features manc_fusion_features.json`; four
  log1p-scaled synapse-count scalars concatenated at the pooled layer), and the
  counts-only guardrail baseline (`--model counts_mlp`). Run 2026-08-11.
- `manc_fusionperm_{neuron_class,subclass,hemilineage}.json` — **permutation
  control** (Methods): count features shuffled across neurons; the fusion gain
  must vanish. Run 2026-08-12.
- `manc_fusion_leakctrl_{neuron_class,subclass,hemilineage}.json` —
  **cell-type-held-out fusion control** (Methods): the fusion arm rerun under
  the whole-cell-type-held-out split. Run 2026-08-12.
- `manc_fusion_features.json` — the per-neuron linked-feature table joined to
  each mesh by body id: `[log1p(pre), log1p(post), log1p(pre+post),
  pre/(pre+post)]`, derived from the neuPrint `manc:v1.2.1` annotations fetched
  2026-08-11 (raw rows in `manc_synapse_meta.json`). Regenerate with
  `python -m mudm_paper.fusion_features` (or from `manc_synapse_meta.json`).
- `synapse_probe_results.json` — the **counts-only leakage probe** run and
  deposited BEFORE the fusion experiment (Methods): counts-only classifiers
  score far below geometry on the learnable tasks, so linked counts are signal
  to fuse, not a label leak. Script: `mudm_paper/synapse_probe.py`. Run
  2026-08-11.
- `nyxus_stability_summary.json` — **source for Extended Data Table 4**
  (feature robustness across 2D polygon levels): 144 Nyxus morphology features
  over ~187k PanNuke nuclei rasterized at five simplification levels
  (τ = 0.5–8 px); per-level median absolute relative deviation and Spearman ρ
  vs the unsimplified masks. Pipeline: `scripts/extract_pannuke_images.py` →
  `mudm_paper.rasterize` → `scripts/nyxus_pipeline.py` →
  `scripts/stability_analysis.py`. Run 2026-08-11. **PanNuke-derived: this file
  is CC BY-NC-SA 4.0, not MIT — see `LICENSE-DATA.md`.** (Not part of the v0.6.4
  re-run: the rasterizer is harness-internal and parity-tested against the
  v0.6.4 simplification.)
- `manc_pointnet_{neuron_class,subclass,hemilineage,cell_type}.json` —
  **second-architecture robustness sweep** (PointNet instead of DGCNN, otherwise
  the exact Table 1 protocol: 512-vertex budget, 15 seeds, instance-grouped
  split, levels z0–z4). Coarsest-level retention 94/93/99/80% at neuron class /
  subclass / hemilineage / cell type, at or above DGCNN's 93/87/87/56%, so the
  storage-versus-accuracy economy is not specific to one point-cloud
  architecture. Run 2026-08-20 under v0.6.4. Regenerate with
  `run_manc --out-dir <dir> --model pointnet --prefix manc_pointnet`.

- `manc_budget{2048,4096}_{neuron_class,hemilineage}.json` and
  `flywire_budget{2048,4096}_super_class.json` — **input-budget sensitivity sweep**
  (run 2026-08-21/23 under v0.6.4): the exact economy protocol rerun at 2,048
  (batch 32) and 4,096 (batch 16) sampled vertices, against the deposited
  B=512 runs. Coarsest-level retention holds or rises with budget
  (neuron class 93/95/98%, hemilineage 87/89/97%, FlyWire super-class
  99/95/99% at 512/2048/4096), so the retention fractions are not an artifact
  of the input budget; the finest level gains no relative advantage at any
  budget. Absolute accuracies decline at larger budgets because the training
  recipe (8 epochs; batch size halved per budget step) is fixed rather than
  re-tuned, so comparisons are made within a budget. Regenerate with
  `run_manc --out-dir <dir> --labels <label> --budgets 2048 --batch 32
  --prefix manc_budget2048` (and analogously for 4096 / FlyWire with
  `--group-field cell_type`).
- `manc_access_benchmark.json` — **read-path / queryability benchmark** on the full
  23,546-neuron MANC store (v0.6.4 encodes, run 2026-08-21): bulk whole-level read
  (Parquet 0.91 ms/neuron vs 3D Tiles 28.8, Neuroglancer 70.0, OBJ 178.7),
  spatial-window predicate pushdown (9.0x vs whole-level scan), by-id fetch on the
  default and feature-sorted layouts vs Neuroglancer per-file (the honest contrast:
  NG wins single fetches at 10 ms), and the **feature-to-record JOIN arm** (the fusion
  experiment's access pattern): one pushdown query joins all 11,745 neurons in a
  3x3-tile window to their synapse-count records in 2.13 s, vs 954 s extrapolated
  for per-file fetch + join. bytes_read_* keys are /proc/self/io rchar deltas.
  Regenerate with `bench_formats --mode access --records manc_fusion_features.json`.
- `manc_format_benchmark.json` — **format spec-sheet** (v0.6.4, run 2026-08-21):
  encode time / stored bytes / peak RSS per output format from ONE shared tile index
  (Parquet 13.3 GB, 3D Tiles 9.0 GB, Neuroglancer 4.2 GB loose-unsharded), plus
  per-format decode timings, random access, and vertex fidelity on a 200-neuron
  sample. Supersedes the July v0.6.3 run for citation; `B6_gltf_decoder_comparison.json`
  remains as the separate scene-graph-traversal micro-probe.

## Reproducing Table 1

Table 1 (the `manc_economy_*.json` files above) is regenerated by
`mudm_paper.run_manc`. The committed non-smoke config in
`run_manc.build_config` is the exact E2-confirm configuration: DGCNN on
precomputed pyramid levels z0–z4, 512-vertex input budget, 15 seeds, 8 epochs,
batch 64, uncapped loader (`cap=600000`, above the largest MANC mesh),
instance-grouped leakage-safe splits, `min_class=20`. (The `knee`/`tost` fields
still written into each JSON are legacy and are not used by the paper, which
reports the storage-versus-accuracy trade-off directly.)

From a tiled MANC parquet pyramid:

```bash
uv run python -m mudm_paper.run_manc \
    --parquet /path/to/manc-pyramid \
    --out-dir paper/benchmark_results \
    --device cuda:1
```

This writes one validator-ready JSON per label
(`manc_economy_{neuron_class,subclass,hemilineage,cell_type}.json`).
Reproduction is up to GPU nondeterminism (DGCNN kNN / cuDNN), so numbers match
Table 1 within seed-to-seed variation, not bit-for-bit.

Building the pyramid from source (neuPrint MANC):

1. Download the traced, typed MANC neurons as `<body_id>.obj` meshes plus a
   `metadata.json` of neuPrint annotations (the `download_manc` script).
2. `uv run python -m mudm_paper.manc_prep metadata.json meta.json` — adapt
   neuPrint field names to muDM tag names (nulls dropped, so tags stay sparse,
   matching the deployed catalog).
3. `uv run python -m mudm_paper.tile_parquet --obj-dir <objs> --meta meta.json
   --out /path/to/manc-pyramid --max-zoom 4` — tile with the real muDM Rust
   tiler into `zoom={0..4}/part_*.parquet`. Set `TMPDIR` to a large partition.

Hardware for the deposited run: 80-core Linux host with 2× NVIDIA A4500 (each
file's `hardware` header records this).

## Companion data catalogue numbers

Dataset-scale figures in the Supplementary catalogue (per-tissue cell, nucleus,
transcript, and codeword counts) trace to the companion catalogue's authoritative
`site/datasets.json` in the `mudm-data` repository, not to a benchmark file here.

## Run logs

Benchmark runs write a tee'd stdout/stderr log to
`results/_logs/<benchmark_id>.log`. Logs are gitignored
(`results/_logs/*.log`); the directory tracks via `_logs/.gitkeep`.

## Python invocation

Run the schema validator and figure scripts from the paper repo:

```bash
uv run python results/_schema_validator.py results/*.json
uv run pytest results/_test_schema_validator.py -v
```

Two files are data tables, not benchmark results, and are exempt from the
benchmark header schema (they carry no `timestamp`/`benchmark_id`):
`manc_fusion_features.json` (the linked-feature table) and
`manc_synapse_meta.json` (raw neuPrint rows). The retired `hemibrain_5seed.json`
and `hemibrain_economy.json` fail the schema by design (see their entries).

The paper repo declares its own `pyproject.toml` with the dependencies needed for
figure scripts (matplotlib, numpy, pandas, pyarrow, brotli, scipy) and schema
validation (pytest).
