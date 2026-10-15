"""
Run the muDM economy sweep (MANC neuron classification).

This module assembles the experiment config and calls sweep.run_sweep.
`reproduce_table1` regenerates the Table 1 artifacts deposited in results/
(one ``manc_economy_<label>.json`` per task).
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mudm_paper import sweep

# The three classification tasks reported in Table 1. `cell_type` remains
# available through --labels (its artifact is deposited as a reference).
TABLE1_LABELS = ["neuron_class", "subclass", "hemilineage"]


def default_device() -> str:
    """Return ``"cuda:0"`` when torch sees a CUDA device, else ``"cpu"``."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_config(smoke: bool = False, device: str | None = None) -> dict:
    """
    Build the experiment config for run_sweep.

    Args:
        smoke: If True, use reduced config for quick testing.
        device: Torch device string. Default: ``default_device()``, i.e.
            "cuda:0" when CUDA is available, else "cpu".

    Returns:
        Configuration dict for sweep.run_sweep.

    The non-smoke config is the exact configuration behind Table 1: DGCNN on
    precomputed pyramid levels 0-4, a 512-vertex input budget, 15 seeds,
    8 epochs, batch size 64, splits grouped by `instance`, the three tasks
    in `TABLE1_LABELS`, and an uncapped loader (cap 600000, above the largest
    MANC mesh) so the finest level is not spatially truncated. It sweeps
    `precomputed` only; the on-the-fly arms (`fine_random`, `fine_fps`,
    `fine_onthefly_qem`) are supported by the harness but were not part of
    the paper run. Other input budgets (the 2,048 and 4,096 sensitivity runs)
    are set with --budgets and --batch.
    """
    if device is None:
        device = default_device()
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
            "labels": list(TABLE1_LABELS),
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
    device: str | None = None,
    hardware: dict | None = None,
    prefix: str = "manc_economy",
) -> list:
    """Regenerate the Table 1 artifacts (``manc_economy_<label>.json``).

    Runs the paper config once per label (single-label sweeps, DGCNN,
    precomputed levels 0-4; by default the three tasks in `TABLE1_LABELS`)
    and writes one validator-ready JSON per label to ``out_dir`` as
    ``<prefix>_<label>.json``, matching the files deposited in results/.
    This is the committed equivalent of the runner that produced Table 1.
    Reproduction is up to GPU nondeterminism (DGCNN kNN / cuDNN), so numbers
    match Table 1 within seed-to-seed variation, not bit-for-bit.

    The same path drives the leakage-controlled robustness check: override
    ``base_config`` with ``group_field="cell_type"`` and the three coarse
    labels (and set ``prefix``) to hold out entire cell types. Likewise the
    learning-curve run: ``base_config["curve"] = True`` (CLI ``--curve``)
    persists per-epoch val/test balanced accuracy under each cell's
    ``raw["curves"]`` plus ``bal_acc_at_best_val`` / ``best_epoch``; every
    ``base_config`` key is forwarded to ``run_sweep`` unchanged.

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
            "Directory to write one artifact per label "
            "(<prefix>_<label>.json), reproducing Table 1 per label."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Torch device (default: cuda:0 when CUDA is available, else cpu).",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help=(
            "Restrict the run to a single model (dgcnn, pointnet, dgcnn_fusion or counts_mlp), "
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
        help=(
            "Comma-separated label override (default: "
            "neuron_class,subclass,hemilineage, the three Table 1 tasks). "
            "cell_type is available on request."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="manc_economy",
        type=str,
        help="Output filename prefix for --out-dir mode (<prefix>_<label>.json).",
    )
    parser.add_argument(
        "--budgets",
        default=None,
        type=str,
        help=(
            "Comma-separated input-budget override (default: the paper "
            "config, 512). Used for the budget-sensitivity sweep."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=(
            "Override batch size (default 64). Reduce for large budgets: "
            "DGCNN's pairwise-distance matrix scales with batch * budget^2."
        ),
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
    parser.add_argument(
        "--levels",
        default=None,
        type=str,
        help=(
            "Comma-separated pyramid-level override (default: the paper "
            "config, 0,1,2,3,4). E.g. '0,4' for a coarsest-vs-finest "
            "learning-curve run."
        ),
    )
    parser.add_argument(
        "--curve",
        action="store_true",
        help=(
            "Record a learning curve: evaluate validation and test balanced "
            "accuracy after every epoch and persist, per seed, the epoch of "
            "best validation balanced accuracy and the test balanced accuracy "
            "at that epoch (validation-selected early-stopping readout). Adds "
            "one val+test forward pass per epoch; final-epoch metrics and "
            "the default output structure are unchanged."
        ),
    )

    args = parser.parse_args(argv)

    device = args.device or default_device()
    config = build_config(smoke=args.smoke, device=device)
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
    if args.budgets:
        config["budgets"] = [int(b) for b in args.budgets.split(",")]
    if args.batch is not None:
        config["batch_size"] = args.batch
    if args.extra_features:
        with open(args.extra_features) as f:
            config["extra_features"] = json.load(f)
    if args.levels:
        config["levels"] = [int(z) for z in args.levels.split(",")]
    if args.curve:
        # Only set when requested: the persisted "config" (and the whole
        # result structure) stays byte-identical to Table 1 otherwise.
        config["curve"] = True

    # Table 1 reproduction (or the leakage-controlled variant via --group-field
    # / --labels / --prefix / --levels / --curve): one validator-ready JSON per
    # label. Every config key flows through reproduce_table1 unchanged.
    if args.out_dir:
        reproduce_table1(
            args.parquet,
            args.out_dir,
            base_config=config,
            device=device,
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
