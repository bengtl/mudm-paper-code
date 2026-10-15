"""Schema validator for the benchmark JSONs in results/.

Every benchmark JSON must include:
  - timestamp        : ISO-8601 UTC string
  - hardware         : dict with at least cpu_cores OR cpu_model
  - benchmark_id     : string matching the filename stem (e.g. "B2_xenium_ingest_validation")
  - config           : expected for run records (not checked; fragment statistics use "params")
  - (one of)
        status       : "passed" / "failed"          (validation-only benchmarks)
        results      : dict of benchmark-specific result keys
    or top-level domain keys (B1 / B3 / B6 use top-level summary keys directly)

Optional but expected:
  - git_sha_mudm_tools : str (commit SHA of mudm-tools at run time)

This validator is intentionally lenient on the result shape (each benchmark has
its own keys) but strict on the metadata header.

Files listed in EXEMPT are data tables, summaries or external tool reports, not
benchmark records. They carry no timestamp/benchmark_id header; the CLI reports
them as SKIP instead of FAIL. validate_file() itself stays strict.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REQUIRED_HEADER = ("timestamp", "hardware", "benchmark_id")
RESULT_INDICATORS = ("status", "results")  # one must be present OR domain-specific keys
ISO_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")

# Files in results/ that are not benchmark records (filename -> reason).
EXEMPT: dict[str, str] = {
    "manc_fusion_features.json": "data table: per-neuron linked features for the fusion runs",
    "manc_synapse_meta.json": "data table: raw neuPrint synapse rows",
    "nyxus_stability_summary.json": "summary table produced by scripts/stability_analysis.py",
    "synapse_probe_results.json": "summary of the counts-only synapse probe",
    "transcript_lod_xenium-breast-full.json": "point-LOD evaluation summary (transcript_lod)",
    "transcript_lod_xenium-lung-full.json": "point-LOD evaluation summary (transcript_lod)",
    "transcript_lod_xenium-ovary-full.json": "point-LOD evaluation summary (transcript_lod)",
    "obj_dedup_all.json": "finest-level fidelity summary: unique vertices/faces vs source OBJ",
    "coarse_topology.json": "per-level mesh topology summary (400-neuron sample)",
    "deployed_counts_zoom0.json": "feature counts of the served catalog pyramids over HTTP",
    "split_sizes.json": "train/val/test sizes per task and split scheme",
    "tile300_fixed_validation.json": "report of the OGC 3d-tiles-validator (external tool output)",
}


class SchemaError(Exception):
    pass


def validate_file(path: Path) -> dict:
    """Validate a single B*.json file. Returns the loaded dict on success;
    raises SchemaError on failure. The filename stem must match the
    benchmark_id field for self-consistency."""
    if not path.exists():
        raise SchemaError(f"file not found: {path}")
    data = json.loads(path.read_text())

    for key in REQUIRED_HEADER:
        if key not in data:
            raise SchemaError(f"{path.name}: missing required key '{key}'")

    if not ISO_TIMESTAMP_RE.match(str(data["timestamp"])):
        raise SchemaError(
            f"{path.name}: timestamp '{data['timestamp']}' is not ISO-8601 UTC"
        )

    if not isinstance(data["hardware"], dict):
        raise SchemaError(f"{path.name}: 'hardware' must be a dict")
    if not (data["hardware"].get("cpu_cores") or data["hardware"].get("cpu_model")):
        raise SchemaError(
            f"{path.name}: hardware dict must include at least cpu_cores or cpu_model"
        )

    expected_id = path.stem
    actual_id = data["benchmark_id"]
    if actual_id != expected_id:
        raise SchemaError(
            f"{path.name}: benchmark_id '{actual_id}' does not match filename stem '{expected_id}'"
        )

    return data


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: uv run python results/_schema_validator.py <file.json> [<file.json> ...]")
        return 2
    failures = 0
    for arg in argv[1:]:
        reason = EXEMPT.get(Path(arg).name)
        if reason is not None:
            print(f"SKIP {arg}: {reason}")
            continue
        try:
            validate_file(Path(arg))
            print(f"OK   {arg}")
        except SchemaError as exc:
            print(f"FAIL {arg}: {exc}")
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
