"""
Run the muDM economy sweep (MANC neuron classification).

This module assembles the experiment config and calls sweep.run_sweep.
`reproduce_table1` regenerates the four Table 1 artifacts deposited in
paper/benchmark_results/.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mudm_paper import sweep


def build_config(smoke: bool = False, device: str = "cuda:1") -> dict:
    """
    Build the experiment config for run_sweep.

    Args:
        smoke: If True, use reduced config for quick testing.
        device: CUDA device string (default "cuda:1").

    Returns:
        Configuration dict for sweep.run_sweep.

    The non-smoke config is the exact configuration behind Table 1 (the
    E2-confirm run): DGCNN on precomputed pyramid levels 0-4, a 512-vertex
    input budget, 15 seeds, 8 epochs, batch size 64, and an uncapped loader
    (cap 600000, above the largest MANC mesh) so the finest level is not
    spatially truncated. It sweeps `precomputed` only; the on-the-fly arms
    (`fine_random`, `fine_fps`, `fine_onthefly_qem`) and the multi-budget
    sweep ({256, 1024}) are supported by the harness but were not part of the
    paper run -- they would be re-enabled here for the decimation-baseline and
    budget-sensitivity ablations.
    """
    if smoke:
        return {
            "labels": ["neuron_class"],
            "budgets": [256],
            "levels": [0, 1, 2],
            "arms": ["precomputed", "fine_fps"],
            "seeds": 2,
            "epochs": 1,
            # Fix B: PointNet only for a quick smoke run (DGCNN as primary
            # arm is exercised via the full config below).
            "models": ["pointnet"],
            "device": device,
            "epsilon": 0.05,
            "min_class": 1,
            "group_field": None,
        }
    else:
        return {
            "labels": ["neuron_class", "subclass", "hemilineage", "cell_type"],
            "budgets": [512],
            "levels": [0, 1, 2, 3, 4],
            "arms": ["precomputed"],
            "seeds": 15,
            "epochs": 8,
            "batch_size": 64,
            # Uncapped loader: above the largest MANC mesh, so the finest
            # level is retained whole. A small cap silently truncates it to
            # the first N vertices and was the artifact behind the retired
            # coarse-beats-fine result.
            "cap": 600000,
            # DGCNN is the classifier reported in the paper (Table 1);
            # PointNet remains available via --model for comparison but is
            # not part of the paper run.
            "models": ["dgcnn"],
            "device": device,
            "epsilon": 0.02,
            "min_class": 20,
            "group_field": "instance",
        }


def _run_header(hardware: dict | None = None) -> tuple:
    """Return (timestamp, hardware) for a validator-ready result header.

    timestamp is the current UTC time in ISO-8601; hardware defaults to the
    host CPU count plus any visible CUDA device names.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if hardware is None:
        hardware = {"cpu_cores": os.cpu_count() or 0}
        try:
            import torch

            if torch.cuda.is_available():
                hardware["gpu"] = ", ".join(
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                )
        except Exception:
            pass
    return timestamp, hardware


def reproduce_table1(
    parquet_dir: str,
    out_dir: str,
    base_config: dict | None = None,
    device: str = "cuda:1",
    hardware: dict | None = None,
    prefix: str = "manc_economy",
) -> list:
    """Regenerate the four Table 1 artifacts (``manc_economy_<label>.json``).

    Runs the paper config once per label (single-label sweeps, DGCNN,
    precomputed levels 0-4) and writes one validator-ready JSON per label to
    ``out_dir`` as ``<prefix>_<label>.json``, matching the files deposited in
    paper/benchmark_results/. This is the committed equivalent of the ad-hoc
    oden runner that produced Table 1. Reproduction is up to GPU
    nondeterminism (DGCNN kNN / cuDNN), so numbers match Table 1 within
    seed-to-seed variation, not bit-for-bit.

    The same path drives the leakage-controlled robustness check: override
    ``base_config`` with ``group_field="cell_type"`` and the three coarse
    labels (and set ``prefix``) to hold out entire cell types.

    Returns the list of written file paths.
    """
    if base_config is None:
        base_config = build_config(device=device)
    model = base_config.get("model") or base_config.get("models", ["dgcnn"])[0]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp, hw = _run_header(hardware)

    written = []
    for label in base_config["labels"]:
        cfg = {**base_config, "labels": [label], "model": model}
        result = sweep.run_sweep(parquet_dir, cfg)
        stem = f"{prefix}_{label}"
        headed = {
            "benchmark_id": stem,
            "timestamp": timestamp,
            "hardware": hw,
            **result,
        }
        dest = out_path / f"{stem}.json"
        with open(dest, "w") as f:
            json.dump(
                headed,
                f,
                indent=2,
                default=lambda o: float(o) if hasattr(o, "item") else str(o),
            )
        written.append(str(dest))
        print(f"wrote {dest}")
    return written


def main(argv=None):
    """
    CLI entry point for the sweep runner.

    Args:
        argv: Command-line arguments (for testing).
    """
    parser = argparse.ArgumentParser(
        description="Run the muDM economy sweep on MANC neuron classification."
    )
    parser.add_argument(
        "--parquet", required=True, type=str, help="Path to parquet directory."
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="Output JSON file path (combined sweep, keyed by model).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=str,
        help=(
            "Directory to write the four Table 1 artifacts "
            "(manc_economy_<label>.json), reproducing Table 1 per label."
        ),
    )
    parser.add_argument(
        "--device", default="cuda:1", type=str, help="CUDA device (default: cuda:1)."
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help=(
            "Restrict the run to a single model (pointnet or dgcnn), "
            "overriding config['models']. Default: the paper config (dgcnn)."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use reduced config for quick testing."
    )
    parser.add_argument(
        "--seeds", type=int, default=None, help="Override number of seeds."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs."
    )
    parser.add_argument(
        "--group-field",
        default=None,
        type=str,
        help=(
            "Override the split grouping field (default 'instance'). Set to "
            "'cell_type' for the leakage-controlled robustness check."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        type=str,
        help="Comma-separated label subset override (default: all four).",
    )
    parser.add_argument(
        "--prefix",
        default="manc_economy",
        type=str,
        help="Output filename prefix for --out-dir mode (<prefix>_<label>.json).",
    )
    parser.add_argument(
        "--extra-features",
        default=None,
        type=str,
        help=(
            "JSON file mapping body_id -> [floats] of per-neuron scalar "
            "features (see fusion_features.py). Required for the fusion "
            "models (--model dgcnn_fusion or counts_mlp)."
        ),
    )

    args = parser.parse_args(argv)

    config = build_config(smoke=args.smoke, device=args.device)
    if args.model:
        config["models"] = [args.model]
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.group_field is not None:
        config["group_field"] = args.group_field
    if args.labels:
        config["labels"] = args.labels.split(",")
    if args.extra_features:
        with open(args.extra_features) as f:
            config["extra_features"] = json.load(f)

    # Table 1 reproduction (or the leakage-controlled variant via --group-field
    # / --labels / --prefix): one validator-ready JSON per label.
    if args.out_dir:
        reproduce_table1(
            args.parquet,
            args.out_dir,
            base_config=config,
            device=args.device,
            prefix=args.prefix,
        )
        print(f"Per-label artifacts written to {args.out_dir}")
        return

    if not args.out:
        parser.error(
            "one of --out (combined JSON) or --out-dir (per-label Table 1 "
            "artifacts) is required"
        )

    # General sweep: run once per model in config["models"], keyed by model
    # name. Each iteration gets its OWN config dict (not a shared, mutated one)
    # so each persisted "config" reflects its own model.
    result_by_model = {}
    for m in config["models"]:
        run_config = {**config, "model": m}
        result_by_model[m] = sweep.run_sweep(args.parquet, run_config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            result_by_model,
            f,
            indent=2,
            default=lambda o: float(o) if hasattr(o, "item") else str(o),
        )

    print(f"Sweep complete. Results written to {args.out}")


if __name__ == "__main__":
    main()
