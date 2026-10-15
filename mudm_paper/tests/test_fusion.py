"""Tests for the multimodal fusion arm: geometry + per-neuron scalar features.

The fusion arm, gated by the synapse-count leakage probe
(mudm_paper/synapse_probe.py): a DGCNN that concatenates
per-neuron scalar features (log-synapse counts) to the pooled global feature
("dgcnn_fusion"), and a geometry-free MLP over the same scalars
("counts_mlp") as the counts-only baseline. Guardrail: fusion must beat BOTH
geometry-only (deposited Table 1) and counts-only under the identical
protocol, so all three arms run through the same sweep.
"""

import json
import tempfile
import warnings

import numpy as np
import pytest
import torch

from mudm_paper.tests.fixtures import write_synthetic_pyramid


def _cloud(b=4, n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((b, 3, n), generator=g), torch.ones((b, n), dtype=torch.bool)


# --- models ------------------------------------------------------------------

def test_dgcnn_fusion_shape_and_extra_sensitivity():
    from mudm_paper.models import DGCNN

    model = DGCNN(n_classes=3, n_extra=4).eval()
    x, mask = _cloud()
    with torch.no_grad():
        out1 = model(x, mask, torch.zeros((4, 4)))
        out2 = model(x, mask, torch.ones((4, 4)))
    assert out1.shape == (4, 3)
    assert not torch.allclose(out1, out2)  # extra features reach the head


def test_dgcnn_fusion_none_extra_means_zeros():
    # forward_flops (cost.py) probes with model(x) only; a fusion model must
    # accept extra=None and treat it as zeros rather than crash.
    from mudm_paper.models import DGCNN

    model = DGCNN(n_classes=3, n_extra=4).eval()
    x, mask = _cloud()
    with torch.no_grad():
        out_none = model(x, mask)
        out_zero = model(x, mask, torch.zeros((4, 4)))
    assert torch.allclose(out_none, out_zero)


def test_counts_mlp_ignores_geometry():
    from mudm_paper.models import CountsMLP

    model = CountsMLP(n_classes=3, n_extra=4).eval()
    xa, mask = _cloud(seed=1)
    xb, _ = _cloud(seed=2)
    e = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    with torch.no_grad():
        out_a = model(xa, mask, e)
        out_b = model(xb, mask, e)
    assert out_a.shape == (4, 3)
    assert torch.allclose(out_a, out_b)  # geometry has no path into the output


# --- feature assembly ---------------------------------------------------------

def test_build_extra_alignment_and_missing_key():
    from mudm_paper.sweep import _build_extra

    m = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    t = _build_extra(["b", "a"], m)
    assert t.dtype == torch.float32 and t.shape == (2, 2)
    assert float(t[0, 0]) == 3.0 and float(t[1, 0]) == 1.0
    assert _build_extra([], m).shape == (0, 2)
    with pytest.raises(KeyError):
        _build_extra(["a", "missing"], m)


def test_fusion_features_from_neuprint_rows():
    from mudm_paper.fusion_features import features_from_rows

    rows = [
        {"bodyId": 1, "pre": 10, "post": 30},
        {"bodyId": 2, "pre": 0, "post": 0},
        {"bodyId": 3, "pre": None, "post": 5},  # missing count: dropped
    ]
    m = features_from_rows(rows)
    assert set(m) == {"1", "2"}
    np.testing.assert_allclose(
        m["1"], [np.log1p(10), np.log1p(30), np.log1p(40), 0.25]
    )
    assert m["2"][3] == 0.0  # pre/(pre+post) guarded on zero total


def test_fusion_features_cli(tmp_path):
    from mudm_paper.fusion_features import main

    rows_file = tmp_path / "rows.json"
    out = tmp_path / "features.json"
    rows_file.write_text(json.dumps([{"bodyId": 7, "pre": 1, "post": 2}]))
    assert main([str(rows_file), str(out)]) == 0
    m = json.loads(out.read_text())
    assert "7" in m and len(m["7"]) == 4


# --- sweep integration --------------------------------------------------------

def _fusion_cfg(model, levels):
    return dict(labels=["neuron_class"], budgets=[64], levels=levels,
                arms=["precomputed"], seeds=2, epochs=1, model=model,
                device="cpu", epsilon=0.05, min_class=1)


def test_run_sweep_fusion_end_to_end():
    from mudm_paper.sweep import run_sweep

    d = tempfile.mkdtemp()
    write_synthetic_pyramid(d, n_neurons=40, zooms=(0, 1))
    extra = {f"N{i}": [float(i % 2), float(i), 0.5, 1.0] for i in range(40)}
    res = run_sweep(d, {**_fusion_cfg("dgcnn_fusion", [0, 1]), "extra_features": extra})
    assert "neuron_class" in res
    assert res["neuron_class"]["precomputed"][0][64]["acc"]["n"] == 2
    assert res["neuron_class"]["precomputed"][1][64]["acc"]["n"] == 2


def test_run_sweep_fusion_excludes_uncovered_neurons():
    # A neuron absent from extra_features must be dropped from eligibility
    # (mirroring the group_field coverage rule) -- never imputed.
    from mudm_paper.sweep import run_sweep

    d = tempfile.mkdtemp()
    write_synthetic_pyramid(d, n_neurons=40, zooms=(0, 1))
    extra = {f"N{i}": [1.0, 2.0] for i in range(40) if i != 0}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_sweep(d, {**_fusion_cfg("dgcnn_fusion", [0, 1]), "extra_features": extra})
    assert any("extra_features" in str(w.message) for w in caught)
    assert "neuron_class" in res


def test_run_sweep_fusion_requires_features():
    from mudm_paper.sweep import run_sweep

    d = tempfile.mkdtemp()
    write_synthetic_pyramid(d, n_neurons=8, zooms=(0,))
    with pytest.raises(ValueError, match="extra_features"):
        run_sweep(d, _fusion_cfg("dgcnn_fusion", [0]))


def test_run_sweep_counts_mlp_end_to_end():
    from mudm_paper.sweep import run_sweep

    d = tempfile.mkdtemp()
    write_synthetic_pyramid(d, n_neurons=40, zooms=(0,))
    extra = {f"N{i}": [float(i % 2)] for i in range(40)}
    res = run_sweep(d, {**_fusion_cfg("counts_mlp", [0]), "extra_features": extra})
    assert res["neuron_class"]["precomputed"][0][64]["acc"]["n"] == 2


# --- CLI wiring ----------------------------------------------------------------

def test_run_manc_extra_features_flag(tmp_path, monkeypatch):
    from mudm_paper import run_manc, sweep

    feat = tmp_path / "features.json"
    feat.write_text(json.dumps({"N1": [0.1, 0.2, 0.3, 0.4]}))
    captured = {}

    def fake_sweep(parquet_dir, cfg):
        captured.update(cfg)
        return {"config": cfg}

    monkeypatch.setattr(sweep, "run_sweep", fake_sweep)
    run_manc.main([
        "--parquet", "unused", "--out", str(tmp_path / "o.json"), "--smoke",
        "--model", "dgcnn_fusion", "--extra-features", str(feat),
    ])
    assert captured["model"] == "dgcnn_fusion"
    assert captured["extra_features"] == {"N1": [0.1, 0.2, 0.3, 0.4]}
