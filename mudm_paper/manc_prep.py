"""Adapt `download_manc.py`'s `metadata.json` into the label map `tile_parquet` wants.

`download_manc.py --download` writes `metadata.json` as
`{"neurons": [{"bodyId": ..., "neuronClass": ..., "somaSide": ..., "instance": ..., ...}]}`
using neuPrint field names. `mudm_paper.tile_parquet.tile_to_parquet` expects
`{body_id: {muDM_field: value}}` using muDM tag names (matching `download_manc._build_tags`),
which then ride in the parquet `tags` map and are read back by `mudm_paper.data`.

This renames neuPrint fields -> muDM tag names and **omits null/empty fields**, so the
resulting parquet tags are genuinely SPARSE — exactly like the deployed catalog — which is
what exercises the sweep's sparse-tag handling (`load_labels` guard) and the bilateral-group
eligibility filter on real manc data.
"""

from __future__ import annotations

import json
import sys

# neuPrint field name -> muDM tag name (mirrors download_manc._build_tags)
_FIELD_MAP = {
    "cellType": "cell_type",
    "instance": "instance",
    "systematicType": "systematic_type",
    "neuronClass": "neuron_class",
    "subclass": "subclass",
    "somaSide": "soma_side",
    "somaNeuromere": "soma_neuromere",
    "hemilineage": "hemilineage",
    "predictedNt": "predicted_nt",
    "entryNerve": "entry_nerve",
    "exitNerve": "exit_nerve",
    "status": "status",
    "statusLabel": "status_label",
}


def adapt(metadata_json: str, out_json: str) -> int:
    """Read download_manc metadata.json -> write {body_id: {muDM_field: value}} (nulls dropped)."""
    raw = json.loads(open(metadata_json).read())
    out: dict[str, dict] = {}
    for n in raw.get("neurons", []):
        bid = str(n["bodyId"])
        fields = {}
        for src, dst in _FIELD_MAP.items():
            v = n.get(src)
            if v is not None and v != "":
                fields[dst] = v
        out[bid] = fields
    with open(out_json, "w") as f:
        json.dump(out, f)
    return len(out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m mudm_paper.manc_prep <metadata.json> <out_meta.json>", file=sys.stderr)
        sys.exit(2)
    count = adapt(sys.argv[1], sys.argv[2])
    print(f"adapted {count} neurons -> {sys.argv[2]}")
