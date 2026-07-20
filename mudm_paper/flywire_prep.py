"""Adapt FlyWire (Schlegel et al. 2024) neuron annotations into the label map
`mudm_paper.tile_parquet` consumes, for the FlyWire second-connectome economy
(the MANC replication; see the scoping note in the paper repo).

FlyWire meshes + the annotation TSV are obtained with mudm-tools'
`scripts/download_flywire.py` (anonymous cloud-volume; no CAVE token). A
class-stratified subsample is selected AT DOWNLOAD time, e.g.:

    python scripts/download_flywire.py --download \
        --select per-superclass --max-neurons 30000 --lod 1 --data-dir data/flywire

which writes `<root_id>.obj` meshes and downloads
`Supplemental_file1_neuron_annotations.tsv`. This module maps that TSV into
`{root_id: {muDM_field: value}}` (empty values dropped), restricted to the
downloaded neurons, ready for:

    python -m mudm_paper.tile_parquet --obj-dir <meshes> --meta flywire_meta.json --out <pyramid>

The four classification targets mirror the MANC granularity ladder, coarsest to
finest: super_class -> cell_class -> hemilineage -> cell_type. FlyWire has no
`_L`/`_R` instance suffix, and bilateral homologs share a `cell_type`, so the
leakage-safe split groups by `cell_type` (run_manc --group-field cell_type),
which holds mirror-symmetric (and any serial) homologs in the same split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

# FlyWire annotation column -> muDM tag key. Mirrors mudm-tools
# download_flywire._TAG_MAP, restricted to the fields the economy sweep and its
# leakage-safe (cell_type-grouped) split actually use. Empty values are dropped,
# so the resulting tags are genuinely sparse -- exactly like the deployed catalog.
_FIELD_MAP = {
    "super_class": "super_class",          # target 1 (broadest)
    "cell_class": "cell_class",            # target 2
    "ito_lee_hemilineage": "hemilineage",  # target 3 (renamed)
    "cell_type": "cell_type",              # target 4 (finest) + leakage-split group key
    "cell_sub_class": "cell_sub_class",
    "hemibrain_type": "hemibrain_type",
    "side": "side",
    "flow": "flow",
    "top_nt": "predicted_nt",
}

# The four economy targets, coarsest -> finest, for the class-survival breakdown.
_TARGETS = ["super_class", "cell_class", "hemilineage", "cell_type"]

_ANNOT_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)


def _downloaded_ids(obj_dir: str | None) -> set[str] | None:
    """Root ids present as `<root_id>.obj` in `obj_dir` (the downloaded
    subsample), or None to keep every annotated neuron."""
    if obj_dir is None:
        return None
    return {p.stem for p in Path(obj_dir).glob("*.obj")}


def adapt(
    annotations_tsv: str,
    out_json: str,
    obj_dir: str | None = None,
    min_class: int = 20,
) -> int:
    """Read the FlyWire annotation TSV, write `{root_id: {muDM_field: value}}`
    (empty values dropped), restricted to the neurons downloaded into `obj_dir`
    when given. The TSV is fetched from the flywire_annotations release if absent.

    Prints, per economy target, how many neurons carry the field and how many
    classes clear `min_class` members -- the survival breakdown that decides
    whether each granularity is learnable (mirrors the sweep's min_class filter).
    Returns the number of neurons written.
    """
    tsv = Path(annotations_tsv)
    if not tsv.exists():
        tsv.parent.mkdir(parents=True, exist_ok=True)
        print(f"annotations not found; downloading {_ANNOT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(_ANNOT_URL, tsv)

    with open(tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    keep = _downloaded_ids(obj_dir)
    out: dict[str, dict] = {}
    for r in rows:
        rid = (r.get("root_id") or "").strip()
        if not rid.isdigit():
            continue  # skip rows without a numeric root_id (matches download_flywire)
        if keep is not None and rid not in keep:
            continue
        fields: dict[str, str] = {}
        for src, dst in _FIELD_MAP.items():
            v = r.get(src)
            v = v.strip() if isinstance(v, str) else v
            if v:
                fields[dst] = v
        out[rid] = fields

    with open(out_json, "w") as f:
        json.dump(out, f)

    print(f"wrote {len(out)} neurons -> {out_json}")
    for tgt in _TARGETS:
        counts = Counter(v[tgt] for v in out.values() if v.get(tgt))
        learnable = sum(1 for c in counts.values() if c >= min_class)
        print(
            f"  {tgt:12s}: {sum(counts.values()):6d} labelled, "
            f"{len(counts):4d} classes, {learnable:4d} with >={min_class} members"
        )
    return len(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Map FlyWire annotations -> muDM label JSON for the economy sweep."
    )
    p.add_argument(
        "--annotations",
        required=True,
        help="FlyWire Supplemental_file1_neuron_annotations.tsv (auto-downloaded if absent)",
    )
    p.add_argument(
        "--obj-dir",
        default=None,
        help="Restrict to <root_id>.obj meshes present here (the downloaded subsample)",
    )
    p.add_argument("--out", required=True, help="Output meta JSON: {root_id: {field: value}}")
    p.add_argument(
        "--min-class", type=int, default=20, help="Min members per class for the survival breakdown"
    )
    args = p.parse_args(argv)
    adapt(args.annotations, args.out, obj_dir=args.obj_dir, min_class=args.min_class)


if __name__ == "__main__":
    main()
