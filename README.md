# muDM paper — reproduction code

Companion code for the quantitative results of *muDM: A Queryable Multi-Resolution
Data Model for Vector Geometry in Spatial Bioimaging*. It reproduces the main-text
economy result (**Table 1**) and the 2D-polygon level-of-detail characterisation
(**Extended Data Fig. 1**), and ships the exact result artifacts the paper's numbers
are read from.

This repository is **analysis code only**. The muDM data model and the tiling
pipeline are released separately and are dependencies here:

- **`mudm`** (data model) — <https://github.com/NovagenResearch/mudm>, PyPI `mudm==0.6.4`
- **`mudm-tools`** (pipeline, viewers) — <https://github.com/NovagenResearch/mudm-tools>, PyPI `mudm-tools==0.6.4`
- **Public data catalog** — <https://ngrf.mudm.io> (the tiled MANC pyramid used below is served as `manc-full`)

## Layout

```
mudm_paper/        the economy/leakage harness (DGCNN sweep over pyramid levels)
  run_manc.py       entry point; `reproduce_table1` regenerates the results/ artifacts
  sweep.py          the level x task x seed grid
  data.py           Parquet-pyramid loader + bilateral-safe, leakage-controlled split
  models.py         DGCNN (masked variable-N pooling) and PointNet
  sampling.py cost.py stats.py knee.py qem.py fps.py   supporting modules
  tile_parquet.py manc_prep.py   build a pyramid from raw meshes (from-scratch path)
  leakage_analysis.py            serial-homolog exposure quantification
  tests/            unit suite (pytest)
results/           deposited result JSONs the paper reads from + PROVENANCE.md + schema validator
figures/           make_table1.py (regenerates Table 1 from results/); fig_s_2d_lod.py (ED Fig. 1)
data/              how to obtain the MANC pyramid (see data/README.md); inputs are not committed
```

## Install

```bash
uv venv && uv pip install -e ".[figures,dev]"     # or: pip install -e ".[figures,dev]"
```

This pulls the pinned `mudm` / `mudm-tools` wheels (native Rust, with a pure-Python
fallback) plus the analysis stack.

## Reproduce Table 1 (storage vs balanced accuracy, MANC 3D-mesh pyramid)

**1. Get the tiled MANC pyramid** (`<parquet_dir>/zoom=<0..4>/part_*.parquet`). Either
fetch the served pyramid from the catalog or regenerate it from source — see
[`data/README.md`](data/README.md).

**2. Run the sweep** (writes one artifact per task into `results/`):

```bash
python -m mudm_paper.run_manc --parquet <parquet_dir> --out-dir results/
```

This is the exact paper configuration (DGCNN; precomputed levels z0–z4; 512-vertex
budget; 15 seeds; 8 epochs; batch 64; instance-grouped, bilateral-safe split). It
regenerates `results/manc_economy_{neuron_class,subclass,hemilineage,cell_type}.json`.

**3. Render the table** from those artifacts (pure stdlib, no GPU):

```bash
python figures/make_table1.py
```

The same reader renders the other economy artifacts by prefix:
`python figures/make_table1.py --prefix manc_leakctrl` (leakage-controlled check)
and `--prefix flywire_economy` (the FlyWire second-connectome replication,
Extended Data Table 3; artifacts deposited in `results/`).

**Leakage-controlled robustness check** (holds out entire cell types, so neither serial
nor bilateral homologs appear in both train and test; regenerates `manc_leakctrl_*.json`):

```bash
python -m mudm_paper.run_manc --parquet <parquet_dir> --out-dir results/ \
    --group-field cell_type --labels neuron_class,subclass,hemilineage --prefix manc_leakctrl
```

Exposure numbers quoted in Methods: `python -m mudm_paper.leakage_analysis <parquet_dir>`.

## Extended Data Fig. 1 (2D polygon level-of-detail)

```bash
python figures/fig_s_2d_lod.py          # needs the .[figures] extra (matplotlib, shapely)
```

The reduce-to-floor Douglas–Peucker simplification itself lives in `mudm-tools`
(`rust/src/simplify2d.rs` and the mirrored Python `mudm2vt/simplify.py`); this script
characterises it (IoU vs stored geometry) on PanNuke nuclei. See the script header for
the expected input.

## Tests

```bash
pytest        # unit suite for the harness (loader/split, sampling, masking, stats, cost, knee)
```

## Reproducibility notes

- Results match Table 1 **within seed-to-seed variation, not bit-for-bit**: DGCNN's kNN
  and cuDNN introduce GPU nondeterminism. Balanced accuracy is reported as mean ± s.e.m.
  over 15 seeds.
- Paper runs used an 80-core Linux host with two NVIDIA A4500 GPUs.
- Every number in the paper traces to a file in `results/`; see `results/PROVENANCE.md`.

## Paper claim → artifact

| Claim | Artifact |
|---|---|
| Table 1 (storage vs balanced accuracy) | `results/manc_economy_*.json` → `figures/make_table1.py` |
| Leakage-controlled robustness (Methods) | `results/manc_leakctrl_*.json`; `mudm_paper/leakage_analysis.py` |
| Lossless-geometry fidelity (Extended Data Table 2) | `results/hemibrain_fidelity.json` |
| FlyWire second-connectome economy (Extended Data Table 3) | `results/flywire_economy_*.json` |
| Feature stability across polygon levels (Extended Data Table 4) | `results/nyxus_stability_summary.json`; `scripts/nyxus_pipeline.py`, `scripts/stability_analysis.py` |
| Multimodal fusion (Extended Data Table 5, Results) | `results/manc_fusion_*.json`, `results/manc_countsonly_*.json`; `mudm_paper/fusion_features.py` |
| Fusion controls: permutation + held-out types (Methods) | `results/manc_fusionperm_*.json`, `results/manc_fusion_leakctrl_*.json` |
| Counts-only leakage probe (Methods) | `mudm_paper/synapse_probe.py`; `results/synapse_probe_results.json` |
| Linked synapse-count feature table (Methods) | `results/manc_fusion_features.json` (provenance: `results/manc_synapse_meta.json`) |
| Read-path / queryability numbers (Results, Methods) | `results/manc_access_benchmark.json` |
| Format spec-sheet: encode/decode/fidelity per format (Extended Data) | `results/manc_format_benchmark.json` |
| Second-architecture robustness, PointNet (Methods) | `results/manc_pointnet_*.json` |
| Xenium ingest counts (Methods / catalog) | `results/B2_xenium_ingest_validation.json` |
| glTF traversal micro-probe (Extended Data) | `results/B6_gltf_decoder_comparison.json` |
| 2D polygon LOD (Extended Data Fig. 1) | `figures/fig_s_2d_lod.py` |

## License & citation

Code: MIT (matching `mudm` / `mudm-tools`). Data artifacts derived from
third-party datasets carry the upstream licence instead — in particular the
PanNuke-derived stability tables are **CC BY-NC-SA 4.0**; see
`LICENSE-DATA.md` for the file list. If you use this code, please cite the
muDM paper and the `mudm` / `mudm-tools` releases (Zenodo DOI on acceptance).

> Not included here: the retired Hemibrain "coarse-beats-fine" artifacts (a vertex-cap
> truncation artifact, removed from the paper) and the benchmark-result schema-emission
> helper; both remain in the paper working repository.
