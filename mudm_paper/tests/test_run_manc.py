from mudm_paper.run_manc import build_config


def test_full_config_shape():
    c = build_config(smoke=False)
    # The three tasks reported in Table 1; cell_type only on request.
    assert c["labels"] == ["neuron_class", "subclass", "hemilineage"]
    assert c["levels"] == [0, 1, 2, 3, 4] and c["seeds"] == 15 and c["epochs"] == 8
    # Table 1 settings: single 512-vertex budget, batch 64, uncapped loader.
    assert c["budgets"] == [512] and c["batch_size"] == 64 and c["cap"] == 600000
    # The paper run swept the precomputed pyramid only (no on-the-fly arms).
    assert c["arms"] == ["precomputed"]
    assert c["group_field"] == "instance" and c["min_class"] == 20 and c["epsilon"] == 0.02
    # DGCNN is the classifier reported in Table 1.
    assert c["models"] == ["dgcnn"]


def test_smoke_config_is_reduced():
    c = build_config(smoke=True)
    assert c["labels"] == ["neuron_class"] and c["seeds"] == 2 and c["epochs"] == 1
    assert c["levels"] == [0, 1, 2] and c["min_class"] == 1
    assert "precomputed" in c["arms"] and "fine_fps" in c["arms"]
    assert c["models"] == ["pointnet"]


def test_device_passthrough():
    assert build_config(device="cuda:0")["device"] == "cuda:0"


def test_device_default_follows_cuda_availability(monkeypatch):
    """No --device: cuda:0 when torch sees a GPU, cpu otherwise."""
    import torch

    from mudm_paper import run_manc

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert run_manc.default_device() == "cuda:0"
    assert build_config()["device"] == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert run_manc.default_device() == "cpu"
    assert build_config()["device"] == "cpu"


def test_cli_default_device_and_explicit_cell_type(tmp_path, monkeypatch):
    """The CLI resolves the device default and still runs cell_type on request."""
    import torch

    from mudm_paper import run_manc

    seen = []

    def fake_run_sweep(parquet_dir, cfg):
        seen.append(cfg)
        label = cfg["labels"][0]
        return {label: {"precomputed": {}}, "cost": {}, "knee": {"knees": {}},
                "config": {}}

    monkeypatch.setattr(run_manc.sweep, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    run_manc.main([
        "--parquet", "/fake/parquet",
        "--out-dir", str(tmp_path),
        "--labels", "cell_type",
    ])

    assert [c["labels"] for c in seen] == [["cell_type"]]
    assert seen[0]["device"] == "cpu"
    assert (tmp_path / "manc_economy_cell_type.json").exists()


def test_cli_budgets_and_batch_override(tmp_path, monkeypatch):
    """--budgets and --batch thread into the config for the sensitivity sweep."""
    from mudm_paper import run_manc

    seen = []

    def fake_run_sweep(parquet_dir, cfg):
        seen.append(cfg)
        label = cfg["labels"][0]
        return {label: {"precomputed": {}}, "cost": {}, "knee": {"knees": {}},
                "config": {}}

    monkeypatch.setattr(run_manc.sweep, "run_sweep", fake_run_sweep)

    run_manc.main([
        "--parquet", "/fake/parquet",
        "--out-dir", str(tmp_path),
        "--labels", "neuron_class",
        "--budgets", "2048,4096",
        "--batch", "16",
        "--prefix", "manc_budget",
        "--device", "cpu",
    ])

    assert len(seen) == 1
    cfg = seen[0]
    assert cfg["budgets"] == [2048, 4096]
    assert cfg["batch_size"] == 16
    # Untouched paper settings survive the overrides.
    assert cfg["epochs"] == 8 and cfg["cap"] == 600000
    assert (tmp_path / "manc_budget_neuron_class.json").exists()


def test_reproduce_table1_writes_validator_ready_per_label(tmp_path, monkeypatch):
    """reproduce_table1 runs one single-label paper-config sweep per label and
    writes a validator-ready JSON (benchmark_id == filename stem) for each."""
    import json

    from mudm_paper import run_manc

    seen = []

    def fake_run_sweep(parquet_dir, cfg):
        seen.append(cfg)
        label = cfg["labels"][0]
        return {
            label: {"precomputed": {}},
            "cost": {},
            "knee": {"knees": {label: 4}},
            "config": {k: cfg[k] for k in ("labels", "arms", "budgets", "seeds",
                                           "epochs", "model")},
        }

    monkeypatch.setattr(run_manc.sweep, "run_sweep", fake_run_sweep)

    written = run_manc.reproduce_table1(
        "/fake/parquet", tmp_path, device="cpu", hardware={"cpu_cores": 4}
    )

    names = sorted(p.rsplit("/", 1)[-1] for p in written)
    assert names == [
        "manc_economy_hemilineage.json",
        "manc_economy_neuron_class.json",
        "manc_economy_subclass.json",
    ]
    # Each sweep is single-label, dgcnn, precomputed-only, at the paper settings.
    for cfg in seen:
        assert len(cfg["labels"]) == 1 and cfg["model"] == "dgcnn"
        assert cfg["arms"] == ["precomputed"] and cfg["budgets"] == [512]
        assert cfg["epochs"] == 8 and cfg["batch_size"] == 64 and cfg["cap"] == 600000
        assert cfg["levels"] == [0, 1, 2, 3, 4] and cfg["seeds"] == 15
        assert cfg["group_field"] == "instance"
    # Written files carry a validator-ready header (benchmark_id == stem).
    d = json.loads((tmp_path / "manc_economy_neuron_class.json").read_text())
    assert d["benchmark_id"] == "manc_economy_neuron_class"
    assert d["hardware"] == {"cpu_cores": 4}
    assert d["timestamp"].endswith("Z") and "T" in d["timestamp"]
