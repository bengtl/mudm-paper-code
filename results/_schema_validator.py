"""Schema validator for paper/benchmark_results/B*.json.

Per paper/benchmark_results/README.md, every benchmark JSON must include:
  - timestamp        : ISO-8601 UTC string
  - hardware         : dict with at least cpu_cores OR cpu_model
  - benchmark_id     : string matching the filename stem (e.g. "B1_hemibrain_5seed")
  - config           : dict containing the CLI args / config dict used
  - (one of)
        status       : "passed" / "failed"          (validation-only benchmarks)
        results      : dict of benchmark-specific result keys
    or top-level domain keys (B1 / B3 / B6 use top-level summary keys directly)

Optional but expected:
  - git_sha_mudm_tools : str (commit SHA of mudm-tools at run time)

This validator is intentionally lenient on the result shape (each benchmark has
its own keys) but strict on the metadata header.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REQUIRED_HEADER = ("timestamp", "hardware", "benchmark_id")
RESULT_INDICATORS = ("status", "results")  # one must be present OR domain-specific keys
ISO_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


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
        print("usage: uv run python paper/benchmark_results/_schema_validator.py <B*.json> [<B*.json> ...]")
        return 2
    failures = 0
    for arg in argv[1:]:
        try:
            validate_file(Path(arg))
            print(f"OK   {arg}")
        except SchemaError as exc:
            print(f"FAIL {arg}: {exc}")
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
