# Benchmark Results

Result artifacts for the Nature Methods submission (right-sized level-of-detail
reframe). One JSON file per benchmark.

## Convention

Figures and tables in the paper are produced by reading these JSON files. Every
quantitative claim in the paper should trace back to one of these files (or to
the companion data catalogue, for dataset-scale numbers — see note below). If a
number in the paper cannot be traced, the claim must be removed or the benchmark
rerun.

## Files present

- `manc_economy_{neuron_class,subclass,hemilineage,cell_type}.json` — **source for
  Table 1** (storage versus balanced accuracy across the MANC 3D-mesh pyramid).
  DGCNN, 512-vertex budget, 15 seeds, uncapped, precomputed levels z0–z4. Each
  file carries per-seed balanced-accuracy `raw` arrays, per-level storage
  (`cost.storage_per_level`; z0–z4 uncompressed 0.29 / 0.54 / 1.39 / 3.89 / 28.64
  GB, a 98.5× range) and the run `config` (epochs 8, batch 64, seeds 15). Run on
  the 80-core + 2× A4500 box, 2026-07-16 (E2-confirm).
- `manc_leakctrl_{neuron_class,subclass,hemilineage}.json` — **leakage-controlled
  robustness check** for the serial-homolog concern (Methods). Same paper config,
  but the train/val/test split is grouped by `cell_type` (holding out entire cell
  types, so neither serial nor bilateral homologs appear in both train and test),
  15 seeds. Coarsest-level retention stays within a few points of the
  instance-grouped Table 1 (94/86/88% at neuron class / subclass / hemilineage,
  versus 93/90/88% here), so the storage-versus-accuracy trade-off is not an
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
  is CC BY-NC-SA 4.0, not MIT — see `LICENSE-DATA.md`.**

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

The paper repo declares its own `pyproject.toml` with the dependencies needed for
figure scripts (matplotlib, numpy, pandas, pyarrow, brotli, scipy) and schema
validation (pytest).
