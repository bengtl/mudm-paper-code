# Provenance of the deposited results

This file has one entry per file in `results/`. Files that differ only by task share an
entry, and their names are spelled out with braces. Each entry gives what the file backs
in the paper, how it was produced, and when. The README maps manuscript claims to these
files.

## Conventions

- **Commands** are run from the repository root inside the project environment
  (`uv run python ...`; see the README for extras). Placeholders in angle brackets are
  explained in `data/README.md`: `<manc_parquet>` is the v0.6.4 MANC pyramid,
  `<flywire_parquet>` the FlyWire subsample pyramid, `<manc_meshes>` the 23,546 source
  OBJ meshes. `run_manc --out-dir results/` writes `<prefix>_<task>.json` for each task;
  without `--labels` the tasks are `neuron_class`, `subclass` and `hemilineage`, and
  `--labels cell_type` produces the `*_cell_type.json` files.
- **Software and hardware for the runs from 2026-08-19 onwards** (every `manc_*` and
  `flywire_*` record): released `mudm==0.6.4` and `mudm-tools==0.6.4` wheels, Python 3.13.2, torch
  2.12.0+cu130, pyarrow 23.0.1, numpy 2.4.4, scipy 1.17.1, on an 80-core Linux host with
  two NVIDIA RTX A4500 GPUs. Pyramids were tiled under v0.6.4 with
  `python -m mudm_paper.tile_parquet ... --max-zoom 4 --base-cells 100`. The `device`
  field in each config records which of the two GPUs ran the job.
- **Shared protocol** unless an entry says otherwise: DGCNN, levels z0 to z4, 512-vertex
  budget, 15 seeds, 8 epochs, batch 64, `cap=600000` (above the largest MANC mesh, so the
  finest level is never truncated), `min_class=20`, split grouped by the side-stripped
  `instance` annotation.
- **Dates** are the `timestamp` field (UTC) where a file has one, otherwise the run date from the run record.
- **Paths and host names** inside `config`, `params`, `parquet_dir`, `hardware` and `provenance`
  fields are those of the benchmark host. They are kept as recorded and do not need to
  exist on your machine.
- **Legacy fields.** The `knee` block in run files is not used by the paper, which reports
  the storage-versus-accuracy trade-off directly.
- **Readouts.** `bal_acc` is the final-epoch test balanced accuracy (Table 1 and all
  8-epoch files). Files run with `--curve` also hold `bal_acc_at_best_val`, the test score
  at the epoch with the best validation score, and the per-epoch curves under
  `raw.curves`.

## Schema check

```bash
uv run python results/_schema_validator.py results/*.json
```

Every run and benchmark record must carry `timestamp`, `hardware` and a `benchmark_id`
equal to the filename stem; run records also keep their `config` (`params` for the fragment statistics). Twelve files are data tables, summaries or
external tool reports; they are listed with a reason in the `EXEMPT` table of
`_schema_validator.py` and reported as SKIP. `validate_file()` itself stays strict.

## Neuron-typing economy, MANC, 8 epochs

**`manc_economy_{neuron_class,subclass,hemilineage}.json`.** Source of Table 1 and
Fig. 2b, the geometry-only arm of Extended Data Table 4, and the storage numbers in the
Methods economy readout: `cost.storage_per_level` holds per-level uncompressed bytes
(z0 to z4: 0.29, 0.54, 1.39, 3.89, 28.55 GB; 98-fold) and compressed bytes (0.16, 0.29,
0.72, 1.93, 10.14 GB; 62-fold), read from Parquet column metadata.
Produced by `python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/`.
Run 2026-08-19. A full run of the same protocol on an earlier v0.6.3 tiling of these
meshes differed by at most 0.0052 in any Table 1 cell.

**`manc_economy_cell_type.json`.** Same run, `--labels cell_type`. Not reported: under
instance grouping the 117-class task has test examples in only 15 classes (see
`split_sizes.json`), so no leakage-safe split exists for it.

**`manc_leakctrl_{neuron_class,subclass,hemilineage}.json`.** Cell types held out:
neither bilateral nor serial homologs appear in both train and test. Backs the Methods
sentence on repeating the sweeps with grouping by cell type, Supplementary Note 2
("Cell-type grouping"), and the geometry baseline of fusion control 2. Coarsest-level
retention 93, 89 and 89%. Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --group-field cell_type --prefix manc_leakctrl`.
Run 2026-08-20. The 38 to 42% serial-homolog exposure quoted in Methods is printed by
`python -m mudm_paper.leakage_analysis <manc_parquet>` (not stored as a file).

**`manc_pointnet_{neuron_class,subclass,hemilineage,cell_type}.json`.** PointNet under
the identical protocol; coarsest-level retention 94, 93 and 99% (Methods; Supplementary
Note 2, "Classifier"). The `cell_type` file is not reported. Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --model pointnet --prefix manc_pointnet`
(add `--labels cell_type` for the fourth file). Run 2026-08-20.

**`manc_ungrouped_cell_type.json`.** 117-class cell type with a per-neuron stratified
split, which cannot keep bilateral twins apart: 0.304 at z0 and 0.400 at z4, 76%
retained (Methods; Supplementary Note 2, "Cell type"). Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --labels cell_type --group-field "" --prefix manc_ungrouped`.
Run 2026-08-25.

## Input budget and training length

**`manc_budget{2048,4096}_{neuron_class,hemilineage}.json`,
`flywire_budget{2048,4096}_super_class.json`.** The economy protocol at 2,048 (batch 32)
and 4,096 (batch 16) sampled vertices, compared with the 512-vertex files above. Backs
the Methods budget sentence and Supplementary Note 2 ("Input budget"): coarsest-level
retention for MANC neuron class, MANC hemilineage and FlyWire super class is 93, 87 and
99% at 512 and 98, 97 and 99% at 4,096, and absolute accuracy is lower at the larger
budgets. Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --labels neuron_class,hemilineage --budgets 2048 --batch 32 --prefix manc_budget2048`,
the same with `--budgets 4096 --batch 16 --prefix manc_budget4096`, and for FlyWire
`python -m mudm_paper.run_manc --parquet <flywire_parquet> --out-dir results/ --labels super_class --group-field cell_type --budgets 2048 --batch 32 --prefix flywire_budget2048`
(and `--budgets 4096 --batch 16 --prefix flywire_budget4096`). Runs 2026-08-21 and
2026-08-22.

**`manc_curve{512,2048,4096}_neuron_class.json`.** Learning curves at 30 epochs, 3 seeds,
levels z0 and z4, budgets 512 (batch 64), 2,048 (batch 32) and 4,096 (batch 16). Backs
Supplementary Note 2: 93% retention at 2,048 under the 30-epoch recipe, and lower
absolute accuracy at larger budgets at 30 epochs. Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --labels neuron_class --levels 0,4 --seeds 3 --epochs 30 --curve --prefix manc_curve512`,
and the same with `--budgets 2048 --batch 32 --prefix manc_curve2048` or
`--budgets 4096 --batch 16 --prefix manc_curve4096`. Runs 2026-08-25.

**`manc_curve512_hemilineage.json`.** Hemilineage learning curve at all five levels,
30 epochs, 3 seeds. Supporting run for Supplementary Note 2; the hemilineage sentence
there quotes the 15-seed `manc_economy30_hemilineage.json`. Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --labels hemilineage --seeds 3 --epochs 30 --curve --prefix manc_curve512`.
Run 2026-08-25.

**`manc_economy30_{neuron_class,subclass,hemilineage}.json`.** The Table 1 protocol at
30 epochs with per-epoch evaluation; source of the 30-epoch recipe in Methods and
Supplementary Note 2 ("Training length"). With validation-based epoch selection
(`bal_acc_at_best_val`) neuron class reaches 0.867 at z0 and 0.901 at z4, retention is 96,
91 and 93%, and the hemilineage intermediate levels are within 2.3% of the finest.
Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --epochs 30 --curve --prefix manc_economy30`.
Run 2026-08-25.

**30-epoch variants of the other sweeps**, deposited for completeness and not quoted in
the paper. Each is the command of its 8-epoch counterpart plus `--epochs 30 --curve` and
the prefix shown:

- `manc_leakctrl30_{neuron_class,subclass,hemilineage}.json`: `--group-field cell_type --prefix manc_leakctrl30` (run 2026-08-26).
- `flywire_economy30_{super_class,cell_class,hemilineage}.json`: `--parquet <flywire_parquet> --labels super_class,cell_class,hemilineage --group-field cell_type --prefix flywire_economy30` (run 2026-08-27).
- `manc_fusion30_{neuron_class,subclass,hemilineage}.json`: `--model dgcnn_fusion --extra-features results/manc_fusion_features.json --prefix manc_fusion30` (run 2026-08-25).
- `manc_countsonly30_{neuron_class,subclass,hemilineage}.json`: `--model counts_mlp --extra-features results/manc_fusion_features.json --prefix manc_countsonly30` (run 2026-08-26).
- `manc_fusionperm30_{neuron_class,subclass,hemilineage}.json`: `--model dgcnn_fusion --extra-features <permuted_features.json> --prefix manc_fusionperm30` (run 2026-08-26).
- `manc_fusion_leakctrl30_{neuron_class,subclass,hemilineage}.json`: `--model dgcnn_fusion --extra-features results/manc_fusion_features.json --group-field cell_type --prefix manc_fusion_leakctrl30` (run 2026-08-27).

`figures/make_ed_table4.py --suffix 30 --metric bal_acc_at_best_val` renders the fusion
statistics for this set.

## FlyWire second connectome

**`flywire_economy_{super_class,cell_class,hemilineage}.json`.** Source of Extended Data
Table 2 and Fig. 2c: a class-stratified subsample of 16,379 FlyWire neurons (release 783),
tiled under v0.6.4, split grouped by whole cell type. Coarsest-level retention 99, 91 and
101% at 0.6% of the finest-level storage (155-fold uncompressed, 119-fold on disk).
Produced by
`python -m mudm_paper.run_manc --parquet <flywire_parquet> --out-dir results/ --labels super_class,cell_class,hemilineage --group-field cell_type --prefix flywire_economy`.
Run 2026-08-20. The subsample is regenerated from source (`data/README.md`); the list of
its root ids is not deposited.

**`flywire_full_tiling.json`.** Run record of the Parquet-only tiling of the complete
FlyWire brain: 139,244 neurons from 728 GB of LOD-1 OBJ meshes, 5 h 04 min wall clock,
33.2 GB peak resident memory, exit status 0, mudm-tools 0.6.4 (Methods, storage;
Discussion). Transcribed from the `/usr/bin/time -v` log of the run on the benchmark host;
the log and the pyramid are not deposited, and the file's `provenance` field says so.
Equivalent command:
`python -m mudm_paper.tile_parquet --obj-dir <flywire_full_meshes> --meta <flywire_full_meta.json> --out <pyramid_dir> --max-zoom 4 --base-cells 100 --io-threads 12`,
with inputs from `download_flywire.py --download --select all --lod 1` and
`mudm_paper.flywire_prep` (`data/README.md`). The per-zoom pyramid storage in the file comes from a storage scan of the finished pyramid;
the manuscript does not quote it. Run 2026-08-25.

## Fusion with linked synapse counts

**`manc_synapse_meta.json`.** Raw neuPrint rows for `manc:v1.2.1` (23,546 traced, typed
neurons: body id, presynaptic and postsynaptic counts, cell type, instance, class,
subclass, hemilineage), retrieved 11 August 2026 by the query in
`mudm_paper/synapse_probe.py`. Data table, exempt from the schema check.

**`manc_fusion_features.json`.** Per-neuron feature table keyed by body id:
`[log1p(pre), log1p(post), log1p(pre+post), pre/(pre+post)]`. Produced by
`python -m mudm_paper.fusion_features results/manc_synapse_meta.json results/manc_fusion_features.json`,
which reproduces the deposited file byte for byte. Data table, exempt from the schema
check.

**`synapse_probe_results.json`.** Counts-only leakage probe run before the fusion
experiment (Methods): logistic regression and gradient-boosted trees on the four count
features, same tasks, `min_class=20`, 15 seeds and instance-grouped split. Includes
`cell_type`. Produced by `python -m mudm_paper.synapse_probe` (needs the `probe` extra;
optional `--out <file>`; by default it reads `results/manc_synapse_meta.json` and overwrites this file). Run
2026-08-11. The DGCNN reference values embedded in the file were read from the economy
files of that date, an earlier v0.6.3-era run that the current `manc_economy_*.json`
files supersede; a rerun embeds the current values.

**`manc_fusion_{neuron_class,subclass,hemilineage,cell_type}.json`** and
**`manc_countsonly_{neuron_class,subclass,hemilineage,cell_type}.json`.** Source of
Extended Data Table 4 and the Results fusion paragraph: the fusion arm (DGCNN with the four
count features appended to the pooled representation) and the counts-only MLP, under the
splits and seeds of the geometry-only `manc_economy_*.json`. The `cell_type` files are not
reported. Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --model dgcnn_fusion --extra-features results/manc_fusion_features.json --prefix manc_fusion`
and the same with `--model counts_mlp --prefix manc_countsonly`. Runs 2026-08-19
(fusion) and 2026-08-20 (counts only). Render with `figures/make_ed_table4.py`.

**`manc_fusionperm_{neuron_class,subclass,hemilineage}.json`.** Fusion control 1: the
count features permuted among neurons with one fixed permutation (largest change 0.013,
no cell Holm-significant). The permuted table is embedded in each file's
`config.extra_features` and is regenerated by
`python -m mudm_paper.permute_features results/manc_fusion_features.json <permuted_features.json>`
(seed 0: keys sorted as strings, `numpy.random.default_rng(0).permutation`). Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --model dgcnn_fusion --extra-features <permuted_features.json> --prefix manc_fusionperm`.
Run 2026-08-20.

**`manc_fusion_leakctrl_{neuron_class,subclass,hemilineage}.json`.** Fusion control 2:
the fusion arm under the cell-type-held-out split, compared with `manc_leakctrl_*.json`.
Produced by
`python -m mudm_paper.run_manc --parquet <manc_parquet> --out-dir results/ --model dgcnn_fusion --extra-features results/manc_fusion_features.json --group-field cell_type --prefix manc_fusion_leakctrl`.
Run 2026-08-20.

## Formats, read path and pyramid structure (MANC)

**`manc_format_benchmark.json`.** Format spec-sheet from one shared tile index of the
23,546 MANC meshes under v0.6.4: encode time, stored bytes and peak resident memory per
output (Parquet 13.3 GB, 3D Tiles 9.0 GB, loose Neuroglancer 4.2 GB; peaks 141 GB for
3D Tiles and 200 GB for Neuroglancer, quoted in Methods), plus decode time, random access
and vertex fidelity on a 200-neuron sample. Produced by
`python -m mudm_paper.bench_formats --mode spec-sheet --obj-dir <manc_meshes> --out results/manc_format_benchmark.json --out-root <formats_dir> --n 200 --iters 10 --max-zoom 4 --no-sharded`
(needs the `bench` extra). Run 2026-08-21.

**`manc_access_benchmark.json`.** Read-path benchmark on the encodes in `<formats_dir>`
(Results; Methods, read-path benchmark; Supplementary Note 4): whole-level
read 0.91 ms per neuron for Parquet (feature-id and vertex-position columns) against
28.8 (3D Tiles), 70.0 (Neuroglancer) and 178.7 ms (OBJ) full-mesh decodes on a 2,000-neuron
sample. The 3D Tiles time includes a one-time neuron-to-tile index scan; counting that scan
once instead of scaling it (18.3 s, `manc_format_benchmark.json`
`random_access.tiles3d.index_build_s`) gives the 20.4 ms and the 22-fold lower bound the paper reports; 3 × 3-tile window pushdown 9.0 times faster than
a level scan; per-object fetch 330 ms (tile-sorted Parquet), 83 ms (feature-sorted) and
10 ms (Neuroglancer) over 50 identifiers; and the feature-to-record join, 11,745 window
neurons with their synapse-count records in 2.13 s against 954 s extrapolated from 200
per-file fetches. `bytes_read_*` values are deltas of the process read counter. Produced by
`python -m mudm_paper.bench_formats --mode access --obj-dir <manc_meshes> --out-root <formats_dir> --out results/manc_access_benchmark.json --n 50 --iters 5 --bulk-n 2000 --records results/manc_fusion_features.json --join-n 200`.
Run 2026-08-21.

**`manc_fragment_stats.json`.** Per-level vertex, row, duplicate-vertex and face counts
for the v0.6.4 MANC pyramid, and the comparison of the finest level with the source OBJ
meshes. Backs the border-duplicate figures in Methods: 5.9% duplicated vertices at the
finest level (`per_zoom.4.dup_fraction_weighted`) and 3.3% extra faces relative to the
source (`obj_comparison.totals.tris_excess_fraction`). Produced by
`python -m mudm_paper.fragment_stats --parquet <manc_parquet> --zooms 0,1,2,3,4 --obj-dir <manc_meshes> --workers 16 --out results/manc_fragment_stats.json`
(the deposited run also wrote a per-feature Parquet with `--per-feature-out`, not
deposited). Run 2026-08-25.

**`obj_dedup_all.json`.** Finest-level fidelity check behind the statement that the unique
vertices and faces of the Parquet store equal those of the source meshes (Results;
Methods, storage). For each of the 23,546 MANC neurons, the check hashed float32 vertex
coordinates and face vertex triples (sorted within each face), and compared the set of
unique vertices and the set of unique faces recovered from the finest level (z4) of the
v0.6.4 pyramid with the same sets from the source OBJ. Fields: `n_compared` 23,546;
`unique_verts_equal` and `unique_faces_equal` 23,546 each; `n_mismatch` 0; totals over
the source OBJ files of raw vertices (808,985,382), unique vertices (756,212,103), raw
faces (1,513,903,754) and unique faces (1,496,069,494). The difference between raw and
unique counts is duplication already present in the source meshes, which the tiler's
exact deduplication removes. Run 2026-08-25 on the benchmark host. **The producing
script is not yet in this repository and will be added.**

**`coarse_topology.json`.** Mesh structure per level for a 400-neuron MANC sample of the
v0.6.4 pyramid, behind the Methods statement that source meshes are chunked into a
median of about 100 components per neuron and that coarse levels are sets of small
triangle patches (also the Fig. 2a legend). For each level the file gives the mean vertex
count, mean unique-face count, median ratio of unique vertices to faces, median number of
connected components per neuron (96.5 at z0, 101.5 at z4), median faces per component
(1.6 at z0, 453 at z4), mean fraction of boundary edges (0.77 at z0, 0.004 at z4) and the
fraction of vertices referenced by a face. Run 2026-08-25 on the benchmark host. **The
producing script is not yet in this repository and will be added.**

**`split_sizes.json`.** Train, validation and test sizes (means over the 15 seeds), number
of classes, classes with and without test examples, smallest per-class test support, and
eligible neurons, for each task under three split schemes: instance-grouped (the paper's
split), ungrouped per-neuron, and cell types held out. Backs the Methods statement that
cell type has no leakage-safe split (15 of 117 classes have test examples under instance
grouping, test size about 137). Computed 2026-08-24 on the benchmark host with a helper
that is not yet in this repository. The eligible counts match the sweeps' eligibility
rules, for example 21,952 MANC neurons with an `instance` annotation.

**`tile300_fixed_validation.json`.** Report of the OGC `3d-tiles-validator` (version
0.6.1) on a 3D Tiles encode of 300 MANC neurons: 0 errors, 0 warnings, 1,217
informational messages (the validator does not check the `EXT_meshopt_compression`
extension). The encode used a correction to the Meshopt encoder made in `mudm-tools`
after the 0.6.4 release and not yet in a released version; the served catalog tiles
predate it. The tileset is not deposited. Run 2026-08-25. Methods, mesh codecs, and
Supplementary Note 1.

## Point level of detail (Xenium)

**`transcript_lod_xenium-{breast,lung,ovary}-full.json`.** What the deterministic point
thinning (a detection is kept at level z when its identifier is divisible by
2^(z_max - z)) preserves at each coarser point level, relative to the finest: retention,
per-gene proportions, per-tile counts and tile-gene dropout. Backs Methods (three
sections with 19 to 43 million transcripts and 541 codewords; median per-codeword deviation of
0.97, 1.28 and 1.54% at the coarsest point level, no codeword lost; 4.5, 6.1 and 8.4% of
tile-gene cells holding at least eight detections become empty). Inputs are the catalog's
served 2D pyramids `xenium-breast-full`, `xenium-lung-full` and `xenium-ovary-full`
(`data/README.md`); the served breast pyramid's finest level matches the input recorded
in the file (240 parts, 614,058,004 bytes). Produced by
`python -m mudm_paper.transcript_lod --parquet <xenium_parquet> --out results/transcript_lod_<dataset>.json`
with the default settings recorded under `params`. Runs 2026-08-24.

**`B2_xenium_ingest_validation.json`.** Counts from the full-slide conversion of the
10x Genomics Xenium FFPE Human Breast Cancer Rep1 bundle with mudm-tools 0.6.3 (mudm
0.6.1), the conversion that produced the catalog dataset `xenium-breast-full`: 169,088
cells, 169,083 nuclei, 42,638,083 transcripts, 541 codewords (280 predesigned and 33 custom
expression targets, 228 controls and blanks). Backs Methods (cross-format round trip) and
the breast row of Extended Data Table 5. The paper reports distinct cells: the served
pyramid holds 169,088 cell polygons for 167,780 distinct cell identifiers, because the converter splits 1,308
cell rings into two polygons (checked on the served pyramid on 2026-10-16). Transcribed from the conversion log of
`python -m mudm_tools.converters.cli convert -f xenium -i <Xenium_FFPE_Human_Breast_Cancer_Rep1_outs> -o <out_dir>`.
The file's `supersedes` block, which treats 169,088 as the cell count, predates that check. Dated 2026-06-02.

## Catalog and decoder probes

**`deployed_counts_zoom0.json`.** Number of zoom-0 parts, rows and distinct `feature_id`
values in the served Parquet pyramids of five connectomes, read over HTTP from
<https://ngrf.mudm.io>: MANC 23,546, FlyWire 139,244, male CNS 176,443, BANC 134,934,
hemibrain 25,000. Zoom 0 drops the smallest neurons of the male CNS and BANC; the finest
level and the served descriptors (`mudm:feature_count`) hold 176,570 and 134,942, the counts
the paper reports (Results; Extended Data Table 5, "Scale (served)"). Counted 2026-08-25
with a helper that is not yet in this repository.

**`B6_gltf_decoder_comparison.json`.** glTF traversal micro-probe: the compiled Rust
`gltf` reader in mudm-tools against `pygltflib` on 50 GLB tiles, 20 iterations each, on a
12-core host. The tiles come from a separate tileset that predates the v0.6.4 MANC
encodes. Not reported in the paper. Produced by the mudm-tools script
`scripts/benchmark_gltf_decoder.py --tiles-dir <glb_tiles_dir> --sample 50 --iters 20 --output results/B6_gltf_decoder_comparison.json`.
Run 2026-05-21.

## PanNuke feature stability

**`nyxus_stability_summary.json`.** Source of Extended Data Table 3 and its continuation,
and of the Results sentence on feature ranks at the 11%-vertex level: 144 Nyxus 0.11.0
morphology and intensity features for 187,249 PanNuke nuclei, matched at every level
between full detail and Douglas-Peucker tolerances of 0.5, 1, 2, 4 and 8 px; per feature
and level, the median absolute relative deviation and the Spearman correlation across
nuclei. Produced by the pipeline in the README (`mudm_paper.rasterize`,
`scripts/extract_pannuke_images.py`, `scripts/nyxus_pipeline.py`,
`scripts/stability_analysis.py`) on the 7,556 PanNuke patches converted to muDM polygons
(`data/README.md`). Run 2026-08-11. The rasterizer embeds the pipeline's reduce-to-floor
Douglas-Peucker simplification and is tested against it. Render with
`figures/make_ed_table3.py` (`--full` or `--csv` for every feature). PanNuke-derived:
CC BY-NC-SA 4.0 (see `LICENSE-DATA.md`). Summary table, exempt from the schema check.

The Extended Data Fig. 1 inputs live in `figures/data/` and are described in
`data/README.md`.

## Retired

**`hemibrain_fidelity.json`.** Retired; the current manuscript does not cite it. It
records an earlier hemibrain round-trip check (single-seed PointNet trained on Parquet
tiles and on raw OBJ for 4,074 neurons of the 20 most common cell types, 2026-06-04).
Finest-level fidelity is now reported for MANC by `obj_dedup_all.json`. Its producing
script is not in this repository.
