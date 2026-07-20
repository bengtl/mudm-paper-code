import numpy as np, os, tempfile
import pyarrow.dataset as ds
from mudm_paper.tests.fixtures import write_synthetic_pyramid
from mudm_paper import data

def test_loader_roundtrips_f32_and_labels():
    d = tempfile.mkdtemp(); write_synthetic_pyramid(d, n_neurons=6, zooms=(0,1,2))
    lods = data.load_neuron_lods(d, zooms=(0,1,2))
    assert len(lods) == 6
    bid = next(iter(lods))
    pos, idx = lods[bid][2]
    assert pos.dtype == np.float32 and pos.ndim == 2 and pos.shape[1] == 3
    assert idx.dtype == np.uint32 and idx.shape[1] == 3
    labs = data.load_labels(d, "neuron_class")
    assert len(labs) == 6 and set(labs.values()) <= {"t1","t2"}

def test_multitile_neuron_indices_valid():
    d = tempfile.mkdtemp(); write_synthetic_pyramid(d, n_neurons=6, zooms=(0,1,2))
    lods = data.load_neuron_lods(d, zooms=(0,1,2))
    pos, idx = lods["N0"][0]                       # N0 @ coarsest zoom was written as 2 tiles
    assert int(idx.max()) < len(pos)               # offsets applied correctly; no OOB index

    # The assertion above alone isn't discriminative: N0's two raw tiles at this
    # zoom are similarly sized, so an un-offset 2nd-tile index would still be
    # < len(pos) and the test would pass even with the offset removed. Independently
    # read the raw parquet rows for N0 at this zoom and confirm some concatenated
    # index actually lands at/after the 1st tile's vertex count -- only possible if
    # `load_neuron_lods` applied the running-vertex-count offset to the 2nd tile.
    raw = ds.dataset(os.path.join(d, "zoom=0")).to_table(columns=["positions", "tags"])
    n0_tile_positions = [
        np.frombuffer(pos_bytes, dtype="<f4").reshape(-1, 3)
        for pos_bytes, tags in zip(raw.column("positions").to_pylist(), raw.column("tags").to_pylist())
        if dict(tags)["body_id"] == "N0"
    ]
    assert len(n0_tile_positions) == 2              # sanity: fixture still writes N0 as 2 tiles
    v0 = len(n0_tile_positions[0])                  # 1st tile's vertex count
    assert int(idx.max()) >= v0                     # an index falls in the 2nd tile's offset region

def test_grouped_split_no_bilateral_leakage():
    bids, labs = [], {}
    for g in range(40):
        cls = "t1" if g % 2 == 0 else "t2"
        for side in ("_L","_R"):
            b = f"grp{g}{side}"; bids.append(b); labs[b] = cls
    tr, va, te = data.grouped_stratified_split(bids, labs, seed=0, min_class=1)
    grp = lambda b: b.rsplit("_",1)[0]
    tr_g, va_g, te_g = set(map(grp,tr)), set(map(grp,va)), set(map(grp,te))
    assert tr_g and va_g and te_g
    assert tr_g.isdisjoint(te_g) and tr_g.isdisjoint(va_g) and va_g.isdisjoint(te_g)
    for g in range(40):
        assert (f"grp{g}_R" in tr) == (f"grp{g}_L" in tr)   # homologs share a split

def test_grouped_split_drops_below_floor():
    labs = {f"a{i}_L":"big" for i in range(30)}; labs.update({f"a{i}_R":"big" for i in range(30)})
    labs.update({"rare_L":"rare","rare_R":"rare"})
    tr, va, te = data.grouped_stratified_split(list(labs), labs, seed=0, min_class=20)
    kept = set(tr)|set(va)|set(te)
    assert not any(b.startswith("rare") for b in kept)   # 2-member class dropped
    assert any(b.startswith("a") for b in kept)
