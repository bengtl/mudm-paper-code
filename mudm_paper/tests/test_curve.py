"""Learning-curve / validation-based early-stopping support for the economy
sweep (reviewer request).

`_train_eval(..., curve=True)` evaluates validation AND test balanced accuracy
after every epoch and reports the (1-based) epoch of best validation balanced
accuracy plus the test balanced accuracy AT that epoch. `run_sweep` with
`config["curve"] = True` threads the (previously unused) validation split
through and persists the per-seed curves next to the existing final-epoch
metrics. The `curve=False` path must stay byte-identical to the deposited
Table 1 results: no new keys, and an unchanged training trajectory.
"""

import tempfile

import pytest
import torch

from mudm_paper.models import PointNet
from mudm_paper.sweep import _train_eval, run_sweep
from mudm_paper.tests.fixtures import write_synthetic_pyramid


def _cfg(**over):
    base = dict(
        labels=["neuron_class"], budgets=[64], levels=[0, 1, 2], arms=["precomputed"],
        seeds=2, epochs=3, model="pointnet", device="cpu", epsilon=0.05, min_class=1,
    )
    base.update(over)
    return base


def _fixture_dir(n_neurons=40, zooms=(0, 1, 2)):
    d = tempfile.mkdtemp()
    write_synthetic_pyramid(d, n_neurons=n_neurons, zooms=zooms)
    return d


def _split_tensors(n, n_pts=32, n_classes=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n, 3, n_pts), generator=g)
    y = torch.arange(n) % n_classes
    mask = torch.ones((n, n_pts), dtype=torch.bool)
    return x, y, mask


# --- _train_eval ---------------------------------------------------------------

def test_train_eval_returns_four_tuple_with_none_when_curve_off():
    torch.manual_seed(0)
    x_tr, y_tr, m_tr = _split_tensors(8)
    x_te, y_te, m_te = _split_tensors(4, seed=1)
    out = _train_eval(
        PointNet(2), x_tr, y_tr, m_tr, x_te, y_te, m_te,
        epochs=1, lr=1e-3, batch_size=4, device="cpu", n_classes=2,
    )
    assert len(out) == 4
    acc, bal_acc, f1, curve_info = out
    assert curve_info is None
    for v in (acc, bal_acc, f1):
        assert 0.0 <= v <= 1.0


def test_train_eval_curve_requires_val_tensors():
    x_tr, y_tr, m_tr = _split_tensors(8)
    x_te, y_te, m_te = _split_tensors(4, seed=1)
    with pytest.raises(ValueError):
        _train_eval(
            PointNet(2), x_tr, y_tr, m_tr, x_te, y_te, m_te,
            epochs=1, lr=1e-3, batch_size=4, device="cpu", n_classes=2, curve=True,
        )


def test_train_eval_curve_tracks_every_epoch():
    torch.manual_seed(0)
    x_tr, y_tr, m_tr = _split_tensors(8)
    x_va, y_va, m_va = _split_tensors(4, seed=1)
    x_te, y_te, m_te = _split_tensors(4, seed=2)
    epochs = 4
    acc, bal_acc, f1, info = _train_eval(
        PointNet(2), x_tr, y_tr, m_tr, x_te, y_te, m_te,
        epochs=epochs, lr=1e-3, batch_size=4, device="cpu", n_classes=2,
        x_va=x_va, y_va=y_va, mask_va=m_va, curve=True,
    )
    assert set(info) == {"curve", "best_epoch", "test_bal_acc_at_best_val"}
    assert [e["epoch"] for e in info["curve"]] == list(range(1, epochs + 1))
    for e in info["curve"]:
        assert 0.0 <= e["val_bal_acc"] <= 1.0 and 0.0 <= e["test_bal_acc"] <= 1.0
    assert 1 <= info["best_epoch"] <= epochs
    vals = [e["val_bal_acc"] for e in info["curve"]]
    # best_epoch = FIRST epoch attaining the max validation balanced accuracy.
    assert info["best_epoch"] == vals.index(max(vals)) + 1
    assert info["test_bal_acc_at_best_val"] == info["curve"][info["best_epoch"] - 1]["test_bal_acc"]
    # Final-epoch semantics unchanged: the returned bal_acc IS the last curve point.
    assert bal_acc == pytest.approx(info["curve"][-1]["test_bal_acc"])


# --- run_sweep -----------------------------------------------------------------

def test_run_sweep_curve_true_persists_curves_and_aggregates():
    d = _fixture_dir()
    res = run_sweep(d, _cfg(curve=True))
    for level in (0, 1, 2):
        cell = res["neuron_class"]["precomputed"][level][64]
        curves = cell["raw"]["curves"]
        assert len(curves) == 2  # one curve_info per seed
        for info in curves:
            assert len(info["curve"]) == 3
            assert [e["epoch"] for e in info["curve"]] == [1, 2, 3]
            for e in info["curve"]:
                assert 0.0 <= e["val_bal_acc"] <= 1.0
                assert 0.0 <= e["test_bal_acc"] <= 1.0
            assert 1 <= info["best_epoch"] <= 3
            best = info["curve"][info["best_epoch"] - 1]
            assert best["val_bal_acc"] == max(e["val_bal_acc"] for e in info["curve"])
            assert info["test_bal_acc_at_best_val"] == best["test_bal_acc"]
        # Aggregates sit next to the final-epoch ones.
        agg = cell["bal_acc_at_best_val"]
        assert agg["n"] == 2
        assert agg["mean"] == pytest.approx(
            sum(c["test_bal_acc_at_best_val"] for c in curves) / 2
        )
        assert cell["best_epoch"]["raw"] == [c["best_epoch"] for c in curves]
        assert cell["best_epoch"]["mean"] == pytest.approx(
            sum(c["best_epoch"] for c in curves) / 2
        )
        # Final-epoch metrics are still there and equal the last curve point per seed.
        assert cell["bal_acc"]["n"] == 2
        for i, info in enumerate(curves):
            assert cell["raw"]["bal_acc"][i] == pytest.approx(info["curve"][-1]["test_bal_acc"])
    assert res["config"]["curve"] is True


def test_run_sweep_curve_false_has_no_new_keys():
    d = _fixture_dir()
    res = run_sweep(d, _cfg())  # no "curve" key at all -> default off
    cell = res["neuron_class"]["precomputed"][2][64]
    assert set(cell) == {"acc", "bal_acc", "f1", "eff_points", "raw"}
    assert set(cell["raw"]) == {"acc", "bal_acc", "f1"}
    assert "bal_acc_at_best_val" not in cell and "best_epoch" not in cell
    assert "curve" not in res["config"]


def test_curve_tracking_does_not_perturb_training_trajectory():
    """Per-epoch eval runs in eval mode under no_grad (no dropout RNG draw, no
    BatchNorm running-stat update), so switching curve on must leave the
    final-epoch metrics -- the deposited Table 1 path -- bit-identical."""
    d = _fixture_dir()
    off = run_sweep(d, _cfg())
    on = run_sweep(d, _cfg(curve=True))
    for level in (0, 1, 2):
        a = off["neuron_class"]["precomputed"][level][64]
        b = on["neuron_class"]["precomputed"][level][64]
        assert a["raw"]["acc"] == b["raw"]["acc"]
        assert a["raw"]["bal_acc"] == b["raw"]["bal_acc"]
        assert a["raw"]["f1"] == b["raw"]["f1"]
        # eff_points keeps its train+test meaning (val is not folded in).
        assert a["eff_points"] == b["eff_points"]


def test_run_sweep_curve_with_fusion_extra_val():
    """The fusion path must build extra features for the val split too."""
    d = _fixture_dir(zooms=(0, 1))
    extra = {f"N{i}": [float(i % 2), float(i), 0.5, 1.0] for i in range(40)}
    cfg = _cfg(
        levels=[0], model="dgcnn_fusion", epochs=2, seeds=1,
        extra_features=extra, curve=True,
    )
    res = run_sweep(d, cfg)
    curves = res["neuron_class"]["precomputed"][0][64]["raw"]["curves"]
    assert len(curves) == 1 and len(curves[0]["curve"]) == 2


# --- CLI -----------------------------------------------------------------------

def _fake_run_sweep(seen):
    def fake(parquet_dir, cfg):
        seen.append(cfg)
        label = cfg["labels"][0]
        return {label: {"precomputed": {}}, "cost": {}, "knee": {"knees": {}}, "config": {}}
    return fake


def test_cli_curve_and_levels_reach_config_out_dir(tmp_path, monkeypatch):
    """--curve and --levels thread into the config via reproduce_table1."""
    from mudm_paper import run_manc

    seen = []
    monkeypatch.setattr(run_manc.sweep, "run_sweep", _fake_run_sweep(seen))
    run_manc.main([
        "--parquet", "/fake/parquet",
        "--out-dir", str(tmp_path),
        "--labels", "neuron_class",
        "--levels", "0,4",
        "--budgets", "512",
        "--curve",
        "--prefix", "manc_curve512",
        "--device", "cpu",
    ])
    assert len(seen) == 1
    cfg = seen[0]
    assert cfg["curve"] is True
    assert cfg["levels"] == [0, 4]
    # Untouched paper settings survive the overrides.
    assert cfg["budgets"] == [512] and cfg["epochs"] == 8 and cfg["cap"] == 600000
    assert (tmp_path / "manc_curve512_neuron_class.json").exists()


def test_cli_curve_and_levels_reach_config_out(tmp_path, monkeypatch):
    """Same two flags through the combined --out path."""
    from mudm_paper import run_manc

    seen = []
    monkeypatch.setattr(run_manc.sweep, "run_sweep", _fake_run_sweep(seen))
    out = tmp_path / "combined.json"
    run_manc.main([
        "--parquet", "/fake/parquet",
        "--out", str(out),
        "--labels", "neuron_class",
        "--levels", "2,3",
        "--curve",
        "--device", "cpu",
    ])
    assert len(seen) == 1
    assert seen[0]["curve"] is True and seen[0]["levels"] == [2, 3]
    assert out.exists()


def test_cli_without_curve_leaves_config_untouched(tmp_path, monkeypatch):
    """No --curve -> no "curve" key, so the persisted config stays byte-identical."""
    from mudm_paper import run_manc

    seen = []
    monkeypatch.setattr(run_manc.sweep, "run_sweep", _fake_run_sweep(seen))
    run_manc.main([
        "--parquet", "/fake/parquet",
        "--out-dir", str(tmp_path),
        "--labels", "neuron_class",
        "--device", "cpu",
    ])
    assert "curve" not in seen[0]
    assert seen[0]["levels"] == [0, 1, 2, 3, 4]
