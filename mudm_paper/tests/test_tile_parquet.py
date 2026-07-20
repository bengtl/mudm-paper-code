import os, json, tempfile
from mudm_paper.tests.fixtures import make_sphere_mesh, write_obj
from mudm_paper.tile_parquet import tile_to_parquet
from mudm_paper import data

def test_tile_two_meshes_to_parquet():
    md = tempfile.mkdtemp(); out = tempfile.mkdtemp()
    for bid in ["100", "200"]:
        p, idx = make_sphere_mesh(3)
        write_obj(os.path.join(md, f"{bid}.obj"), p, idx)
    meta = os.path.join(md, "meta.json")
    with open(meta, "w") as f:
        json.dump({"100": {"neuron_class": "t1"}, "200": {"neuron_class": "t2"}}, f)
    tile_to_parquet(md, meta, out, max_zoom=2, base_cells=10)
    lods = data.load_neuron_lods(out, zooms=(0, 1, 2))
    assert set(lods.keys()) == {"100", "200"}
    labs = data.load_labels(out, "neuron_class")
    assert set(labs.values()) == {"t1", "t2"}
