# Data licensing

The **code** in this repository is MIT-licensed (see `LICENSE`), matching the
`mudm` and `mudm-tools` releases.

The **data artifacts** below are derived from third-party datasets and carry the
upstream licence instead of MIT:

## PanNuke-derived artifacts — CC BY-NC-SA 4.0

PanNuke (Gamper et al. 2019/2020) is licensed CC BY-NC-SA 4.0; its ShareAlike
condition applies to works derived from it. The following files are therefore
distributed under **CC BY-NC-SA 4.0**
(<https://creativecommons.org/licenses/by-nc-sa/4.0/>), not MIT:

- `results/nyxus_stability_summary.json` — per-level feature-stability
  statistics computed over PanNuke nuclei (Extended Data Table 4).
- `figures/data/pannuke_2d_lod_fidelity.json` and
  `figures/data/pannuke_rep_nucleus.json` (when deposited) — polygon-LOD
  fidelity statistics and a representative nucleus boundary from the PanNuke
  ingest (Extended Data Fig. 1).
- Any other file whose name or provenance record marks it as PanNuke-derived.

## FlyWire — CC BY-NC 4.0 (not redistributed here)

FlyWire raw data and meshes are **not** included in this repository. The
FlyWire public release is CC BY-NC 4.0; regenerate the experiment subsample
locally per `data/README.md`, and treat locally regenerated FlyWire-derived
files under that licence (non-commercial, with attribution to Dorkenwald et
al. 2024 and Schlegel et al. 2024). The `results/flywire_economy_*.json`
files contain model accuracy statistics only.

## MANC / neuPrint

MANC data are obtained through neuPrint and redistributed under their upstream
terms; per-dataset provenance and licences are recorded in the public catalog
(<https://ngrf.mudm.io>).
