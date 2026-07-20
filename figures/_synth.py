"""Synthetic placeholder data for figure scripts.

Each ``synth_<benchmark_id>()`` function returns a dict matching the shape
of the real ``paper/benchmark_results/<benchmark_id>.json`` it stands in for.
The intent: figure scripts can produce a complete PDF even before the real
benchmarks have run, so layout/styling iteration is not blocked.

Synthesised values are reasonable-looking but should never be quoted in
the paper -- ``_palette.watermark_preliminary()`` stamps figures that use them.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


def _header(benchmark_id: str) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": benchmark_id,
        "hardware": {"cpu_cores": 12, "cpu_model": "synthetic-placeholder"},
        "config": {"placeholder": True},
    }


def synth_B1_hemibrain_5seed() -> dict:
    """B1: 5-seed Hemibrain classification, Parquet vs Arrow IPC vs raw OBJ."""
    rng = np.random.default_rng(0)
    n_seeds = 5

    def arm(mean_acc, sd_acc, load_t_mean, load_t_sd):
        accs = rng.normal(mean_acc, sd_acc, size=n_seeds).tolist()
        f1s = rng.normal(mean_acc + 0.01, sd_acc, size=n_seeds).tolist()
        return {
            "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)),
            "load_time_mean_s": float(load_t_mean), "load_time_sd_s": float(load_t_sd),
            "epoch_time_mean_s": 40.0, "epoch_time_sd_s": 2.0,
            "peak_memory_mb_mean": 9.7, "peak_memory_mb_sd": 0.3,
            "per_seed_accuracy": accs, "per_seed_macro_f1": f1s,
        }

    return {
        **_header("B1_hemibrain_5seed"),
        "num_seeds": n_seeds,
        "parquet":   arm(0.955, 0.008, 88.0, 5.0),
        "arrow_ipc": arm(0.957, 0.007, 25.0, 1.5),
        "raw_obj":   arm(0.960, 0.009, 1894.0, 60.0),
        "paired_t_test_parquet_vs_raw": {
            "t_statistic": -1.18, "p_value": 0.30, "note": "synthetic; not statistically meaningful",
        },
    }


def synth_B3_xenium_al() -> dict:
    """B3-Xenium: per-seed AL trajectories, uncertainty vs random baseline."""
    rng = np.random.default_rng(1)
    n_seeds = 3
    n_rounds = 20
    budget = 50
    initial = 50

    def trajectory(asymptote, slope, jitter):
        rounds = []
        for r in range(n_rounds):
            x = initial + (r + 1) * budget
            acc = asymptote - (asymptote - 0.4) * np.exp(-slope * (r + 1)) + rng.normal(0, jitter)
            acc = float(np.clip(acc, 0.0, 1.0))
            f1 = float(np.clip(acc - rng.uniform(0.0, 0.02), 0.0, 1.0))
            rounds.append({
                "round": r, "queries_this_round": budget, "total_queries": x,
                "test_accuracy": acc, "test_macro_f1": f1,
                "round_wall_time_s": 4.5 + rng.normal(0, 0.3),
            })
        return {"total_queries": initial + n_rounds * budget, "rounds": rounds}

    return {
        **_header("B3_xenium_al"),
        "dataset": "Xenium_FFPE_Human_Breast_Cancer_Rep1 (synthetic)",
        "n_pool": 130000, "n_test": 33000,
        "config": {**_header("B3_xenium_al")["config"], "n_classes": 10,
                   "initial_labels": initial, "budget_per_round": budget, "n_rounds": n_rounds,
                   "seeds": list(range(n_seeds)), "uncertainty_fn": "entropy",
                   "mlp_hidden": 128, "mlp_epochs": 100},
        "per_seed": {
            str(s): {
                "uncertainty": trajectory(0.88, 0.35, 0.012),
                "random":      trajectory(0.78, 0.20, 0.018),
            }
            for s in range(n_seeds)
        },
    }


def synth_B3_latency() -> dict:
    """B3-latency: per-round latency samples for muDM vs re-extract."""
    rng = np.random.default_rng(2)
    n_seeds = 3
    n_rounds = 10

    def arm(median_ms, sd_ms, warmup_extra_ms=0.0):
        samples = []
        for r in range(n_rounds):
            base = rng.normal(median_ms, sd_ms)
            if r == 0:
                base += warmup_extra_ms  # round-0 PyTorch warmup
            samples.append(max(0.001, base) / 1000.0)
        return {
            "rounds": [
                {"round": i, "queries_this_round": 50, "total_queries": 50 + 50 * (i + 1),
                 "test_accuracy": 0.85 + 0.005 * i}
                for i in range(n_rounds)
            ],
            "latency_summary": {
                "n": n_rounds, "mean_s": float(np.mean(samples)),
                "stdev_s": float(np.std(samples)), "median_s": float(np.median(samples)),
            },
            "latency_samples_s": [float(s) for s in samples],
        }

    return {
        **_header("B3_latency"),
        "dataset": "Xenium_FFPE_Human_Breast_Cancer_Rep1 (synthetic)",
        "n_pool": 130000, "n_test": 33000,
        "per_seed": {
            str(s): {
                "mudm":      arm(median_ms=120.0, sd_ms=15.0, warmup_extra_ms=400.0),
                "reextract": arm(median_ms=4500.0, sd_ms=200.0, warmup_extra_ms=50.0),
            }
            for s in range(n_seeds)
        },
    }


def synth_B5_curriculum() -> dict:
    """B5: curriculum + spatial-batching ablation, accuracy curves over epochs."""
    rng = np.random.default_rng(3)
    n_epochs = 30
    n_seeds = 3

    def curve(asymptote, slope, jitter):
        per_seed = []
        for s in range(n_seeds):
            ys = []
            for e in range(n_epochs):
                y = asymptote - (asymptote - 0.3) * np.exp(-slope * (e + 1)) + rng.normal(0, jitter)
                ys.append(float(np.clip(y, 0.0, 1.0)))
            per_seed.append(ys)
        mean = np.mean(per_seed, axis=0).tolist()
        sd = np.std(per_seed, axis=0).tolist()
        return {"per_seed_accuracy": per_seed, "mean_accuracy": mean, "sd_accuracy": sd}

    return {
        **_header("B5_curriculum"),
        "epochs": list(range(1, n_epochs + 1)),
        "curriculum": {
            "uniform":     curve(0.85, 0.15, 0.012),
            "progressive": curve(0.90, 0.25, 0.012),
            "reverse":     curve(0.78, 0.10, 0.015),
        },
        "spatial_batching": {
            "random":  curve(0.85, 0.15, 0.012),
            "spatial": curve(0.89, 0.20, 0.010),
            "mixed":   curve(0.86, 0.17, 0.012),
        },
    }


def synth_B7_cnn_nmo() -> dict:
    """B7: NeuroMorpho CNN-NMO AL trajectory."""
    rng = np.random.default_rng(4)
    n_rounds = 15
    budget = 200

    rounds = []
    for r in range(n_rounds):
        acc = 0.80 - 0.40 * np.exp(-0.30 * (r + 1)) + rng.normal(0, 0.01)
        rounds.append({"round": r, "queries_this_round": budget,
                       "total_queries": (r + 1) * budget,
                       "test_accuracy": float(np.clip(acc, 0.0, 1.0))})
    return {
        **_header("B7_cnn_nmo"),
        "dataset": "NeuroMorpho.Org (synthetic)",
        "uncertainty": {"rounds": rounds, "total_queries": n_rounds * budget},
    }


def synth_B8_mouselight_refresh() -> dict:
    """B8: per-brain timings + output sizes across 38 MouseLight brains."""
    rng = np.random.default_rng(5)
    n_brains = 38
    per_brain = []
    for i in range(n_brains):
        parquet_t = float(rng.normal(45.0, 5.0))
        glb_t = float(rng.normal(80.0, 10.0))
        parquet_size = float(rng.normal(82_000_000, 4_000_000))
        glb_size = float(rng.normal(85_000_000, 5_000_000))
        brotli_size = glb_size * 0.67 + float(rng.normal(0, 1_000_000))
        per_brain.append({
            "brain": f"brain_{i+1:02d}",
            "parquet_wall_time_s": parquet_t,
            "glb_wall_time_s": glb_t,
            "parquet_size_bytes": int(parquet_size),
            "glb_size_bytes": int(glb_size),
            "glb_brotli_size_bytes": int(brotli_size),
        })
    total_wall = sum(b["parquet_wall_time_s"] + b["glb_wall_time_s"] for b in per_brain)
    return {
        **_header("B8_mouselight_refresh"),
        "config": {**_header("B8_mouselight_refresh")["config"],
                   "n_brains": n_brains, "compression": "meshopt"},
        "total_wall_time_s": total_wall,
        "per_brain": per_brain,
    }


def synth_encoding_progression() -> dict:
    """ED Fig 1 progression: stages = baseline glTF -> v1 -> v2 -> v3 -> final."""
    return {
        **_header("encoding_progression"),
        "stages": [
            {"name": "baseline glTF",          "size_mb": 232.0, "decode_ms": 1400.0},
            {"name": ".pbf3 v1 (varint)",      "size_mb":  98.0, "decode_ms":  220.0},
            {"name": "v2 (dictionary tags)",   "size_mb":  72.0, "decode_ms":   95.0},
            {"name": "v3 (vertex dedup)",      "size_mb":  47.0, "decode_ms":   25.0},
            {"name": "final (with gzip)",      "size_mb":  18.5, "decode_ms":   10.0},
        ],
    }


def synth_B3_consep_al() -> dict:
    """B3-CoNSeP AL trajectory (mirror of B3-Xenium for portability ED)."""
    rng = np.random.default_rng(6)
    n_seeds = 3; n_rounds = 20; budget = 30; initial = 40

    def trajectory(asymptote, slope, jitter):
        rounds = []
        for r in range(n_rounds):
            x = initial + (r + 1) * budget
            acc = asymptote - (asymptote - 0.35) * np.exp(-slope * (r + 1)) + rng.normal(0, jitter)
            acc = float(np.clip(acc, 0.0, 1.0))
            rounds.append({"round": r, "queries_this_round": budget, "total_queries": x,
                           "test_accuracy": acc,
                           "test_macro_f1": float(np.clip(acc - rng.uniform(0, 0.02), 0, 1)),
                           "round_wall_time_s": 3.0 + rng.normal(0, 0.2)})
        return {"total_queries": initial + n_rounds * budget, "rounds": rounds}

    return {
        **_header("B3_consep_al"),
        "dataset": "CoNSeP histopathology (synthetic)",
        "n_pool": 18000, "n_test": 4500,
        "per_seed": {
            str(s): {
                "uncertainty": trajectory(0.82, 0.30, 0.012),
                "random":      trajectory(0.72, 0.18, 0.018),
            }
            for s in range(n_seeds)
        },
    }


def synth_B11_hemibrain3d_al() -> dict:
    """B11 Hemibrain-3D AL trajectory (mirror of B3-Xenium for 3D portability)."""
    rng = np.random.default_rng(7)
    n_seeds = 3; n_rounds = 15; budget = 40; initial = 60

    def trajectory(asymptote, slope, jitter):
        rounds = []
        for r in range(n_rounds):
            x = initial + (r + 1) * budget
            acc = asymptote - (asymptote - 0.45) * np.exp(-slope * (r + 1)) + rng.normal(0, jitter)
            acc = float(np.clip(acc, 0.0, 1.0))
            rounds.append({"round": r, "queries_this_round": budget, "total_queries": x,
                           "test_accuracy": acc,
                           "test_macro_f1": float(np.clip(acc - rng.uniform(0, 0.02), 0, 1)),
                           "round_wall_time_s": 12.0 + rng.normal(0, 0.5)})
        return {"total_queries": initial + n_rounds * budget, "rounds": rounds}

    return {
        **_header("B11_hemibrain3d_al"),
        "dataset": "Janelia Hemibrain 3D neurons (synthetic)",
        "n_pool": 2500, "n_test": 600,
        "per_seed": {
            str(s): {
                "uncertainty": trajectory(0.92, 0.35, 0.010),
                "random":      trajectory(0.83, 0.22, 0.015),
            }
            for s in range(n_seeds)
        },
    }


def synth_B9_ontology_audit() -> dict:
    """B9 ontology audit: per-dataset stage retention.

    Returns a dict with one entry per dataset (Hemibrain, Xenium, CoNSeP, MouseLight,
    NeuroMorpho), each listing input/preserved counts at each stage."""
    return {
        **_header("B9_ontology_audit"),
        "datasets": {
            "Hemibrain":    {"input": 95,  "preserved_tiling": 95,  "preserved_encoding": 95,  "preserved_viewer": 95},
            "Xenium":       {"input": 313, "preserved_tiling": 313, "preserved_encoding": 313, "preserved_viewer": 313},
            "CoNSeP":       {"input": 7,   "preserved_tiling": 7,   "preserved_encoding": 7,   "preserved_viewer": 7},
            "MouseLight":   {"input": 730, "preserved_tiling": 729, "preserved_encoding": 729, "preserved_viewer": 729},
            "NeuroMorpho":  {"input": 47,  "preserved_tiling": 47,  "preserved_encoding": 47,  "preserved_viewer": 47},
        },
    }


SYNTH_BY_ID = {
    "B1_hemibrain_5seed":   synth_B1_hemibrain_5seed,
    "B3_xenium_al":         synth_B3_xenium_al,
    "B3_latency":           synth_B3_latency,
    "B5_curriculum":        synth_B5_curriculum,
    "B7_cnn_nmo":           synth_B7_cnn_nmo,
    "B8_mouselight_refresh": synth_B8_mouselight_refresh,
    "encoding_progression": synth_encoding_progression,
    "B3_consep_al":         synth_B3_consep_al,
    "B11_hemibrain3d_al":   synth_B11_hemibrain3d_al,
    "B9_ontology_audit":    synth_B9_ontology_audit,
}


def synthesize(benchmark_id: str) -> dict:
    """Return a synthetic placeholder for the given benchmark_id."""
    if benchmark_id not in SYNTH_BY_ID:
        raise KeyError(f"No synth available for benchmark_id={benchmark_id!r}; "
                       f"add one in paper/figures/scripts/_synth.py")
    return SYNTH_BY_ID[benchmark_id]()
