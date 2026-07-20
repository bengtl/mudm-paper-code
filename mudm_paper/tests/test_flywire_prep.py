import json

from mudm_paper.flywire_prep import adapt

_HEADER = ["root_id", "super_class", "cell_class", "ito_lee_hemilineage", "cell_type", "side"]
_ROWS = [
    ["1", "central", "CX", "CREa1", "EPG", "left"],
    ["2", "central", "CX", "CREa1", "EPG", "right"],
    ["3", "optic", "", "", "", "left"],          # sparse: only super_class + side
    ["abc", "central", "CX", "x", "y", "left"],  # non-numeric root_id -> skipped
]


def _write_tsv(path):
    with open(path, "w", newline="") as f:
        f.write("\t".join(_HEADER) + "\n")
        for r in _ROWS:
            f.write("\t".join(r) + "\n")


def test_maps_fields_renames_hemilineage_and_drops_nulls(tmp_path):
    tsv = tmp_path / "annot.tsv"
    _write_tsv(tsv)
    out = tmp_path / "meta.json"

    n = adapt(str(tsv), str(out))
    meta = json.loads(out.read_text())

    assert n == 3  # non-numeric root_id skipped
    assert "abc" not in meta
    assert meta["1"]["super_class"] == "central"
    assert meta["1"]["cell_class"] == "CX"
    assert meta["1"]["hemilineage"] == "CREa1"  # ito_lee_hemilineage -> hemilineage
    assert meta["1"]["cell_type"] == "EPG"
    # sparse neuron: empty columns dropped, present ones kept
    assert meta["3"] == {"super_class": "optic", "side": "left"}


def test_obj_dir_restricts_to_downloaded_subsample(tmp_path):
    tsv = tmp_path / "annot.tsv"
    _write_tsv(tsv)
    objs = tmp_path / "meshes"
    objs.mkdir()
    (objs / "1.obj").write_text("")
    (objs / "2.obj").write_text("")  # only root ids 1 and 2 were "downloaded"
    out = tmp_path / "meta.json"

    n = adapt(str(tsv), str(out), obj_dir=str(objs))
    meta = json.loads(out.read_text())

    assert n == 2
    assert set(meta) == {"1", "2"}
