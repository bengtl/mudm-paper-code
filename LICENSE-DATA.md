# Data licensing

The **code** in this repository is MIT-licensed (see `LICENSE`), matching the `mudm` and
`mudm-tools` releases. So are the benchmark and run records that contain no third-party
data.

The files below are derived from third-party datasets and carry the upstream licence
instead of MIT. Raw meshes, images and transcript tables are not redistributed here;
`data/README.md` explains how to obtain them from source.

## PanNuke-derived files: CC BY-NC-SA 4.0

PanNuke (Gamper et al. 2019, 2020) is licensed CC BY-NC-SA 4.0, and its ShareAlike
condition applies to works derived from it. These files are distributed under
**CC BY-NC-SA 4.0** (<https://creativecommons.org/licenses/by-nc-sa/4.0/>):

- `results/nyxus_stability_summary.json`: per-level feature-stability statistics over
  PanNuke nuclei (Extended Data Table 3).
- `figures/data/pannuke_2d_lod_fidelity.json`: polygon level-of-detail fidelity over
  38,078 PanNuke nuclei (Extended Data Fig. 1b).
- `figures/data/pannuke_rep_nucleus.json`: one representative PanNuke nucleus boundary
  (Extended Data Fig. 1a).
- Tables and figures rendered from these files, such as the output of
  `figures/make_ed_table3.py` and `figures/fig_s_2d_lod.py`.

## FlyWire-derived files: CC BY-NC 4.0

The FlyWire public release is CC BY-NC 4.0 (non-commercial, with attribution to
Dorkenwald et al. 2024 and Schlegel et al. 2024). These files are distributed under
**CC BY-NC 4.0** (<https://creativecommons.org/licenses/by-nc/4.0/>):

- `results/flywire_economy_{super_class,cell_class,hemilineage}.json`
- `results/flywire_economy30_{super_class,cell_class,hemilineage}.json`
- `results/flywire_budget{2048,4096}_super_class.json`
- `results/flywire_full_tiling.json`

FlyWire meshes and pyramids you regenerate with `data/README.md` fall under the same
licence.

## MANC-derived files: CC BY 4.0

MANC v1.2.1 (Takemura et al. 2024; Marin et al. 2024), obtained through neuPrint (Plaza et
al. 2022), is licensed CC BY 4.0. These files are distributed under **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>) with attribution to those sources:

- `results/manc_*.json`: the neuPrint rows (`manc_synapse_meta.json`), the derived
  synapse-count feature table (`manc_fusion_features.json`), and every MANC run and
  benchmark record (economy, cell-type-held-out, PointNet, per-neuron cell type, budget,
  learning-curve, 30-epoch, fusion, counts-only and permutation runs; format, read-path
  and fragment statistics).
- `results/obj_dedup_all.json`, `results/coarse_topology.json`, `results/synapse_probe_results.json`,
  `results/split_sizes.json`, `results/tile300_fixed_validation.json`.
- `figures/data/manc_rep_neuron_lods.npz`: one MANC neuron mesh at five levels (Fig. 2a).

## Xenium-derived files: CC BY 4.0

The 10x Genomics Xenium datasets used here are licensed CC BY 4.0. These files are
distributed under **CC BY 4.0** with attribution to 10x Genomics (and to Janesick et al.
2023 for the breast section):

- `results/B2_xenium_ingest_validation.json`
- `results/transcript_lod_xenium-{breast,lung,ovary}-full.json`

## Hemibrain-derived file: CC BY 4.0

- `results/hemibrain_fidelity.json` (retired; see `results/PROVENANCE.md`) derives from
  the Janelia hemibrain v1.2.1 (Scheffer et al. 2020), CC BY 4.0.

## Catalog datasets

Per-dataset provenance and licences for everything in the public catalog are recorded at
<https://ngrf.mudm.io>. `results/deployed_counts_zoom0.json` holds only feature counts of
served pyramids.
