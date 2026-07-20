import tempfile
from mudm_paper.tests.fixtures import write_synthetic_pyramid
from mudm_paper.sweep import run_sweep

def test_sweep_runs_end_to_end_on_fixture():
    d = tempfile.mkdtemp(); write_synthetic_pyramid(d, n_neurons=40, zooms=(0,1,2))
    cfg = dict(labels=["neuron_class"], budgets=[64], levels=[0,1,2],
               arms=["precomputed","fine_random","fine_fps"], seeds=2, epochs=1,
               model="pointnet", device="cpu", epsilon=0.05, min_class=1)
    res = run_sweep(d, cfg)
    assert "neuron_class" in res
    assert set(res["neuron_class"]["precomputed"].keys()) == {0,1,2}     # precomputed at every level
    assert res["neuron_class"]["precomputed"][0][64]["acc"]["n"] == 2    # 2 seeds aggregated
    assert set(res["neuron_class"]["fine_fps"].keys()) == {2}            # fine_* only at finest
    assert "tost" in res and "neuron_class" in res["tost"]
    assert "knee" in res and "moves" in res["knee"]
    assert "cost" in res and "storage_per_level" in res["cost"]
