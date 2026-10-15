"""The table and figure scripts in figures/ reproduce the published values
from the deposited artifacts in results/ and figures/data/."""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "figures"
RESULTS = REPO / "results"

# Table 1 exactly as printed in the manuscript.
TABLE1 = """\
| Level         | Storage (GB) | On disk (GB) | neuron class (7) | subclass (28) | hemilineage (39) |
|---------------|--------------|--------------|------------------|---------------|------------------|
| z0 (coarsest) | 0.29         | 0.16         | 0.745            | 0.502         | 0.510            |
| z1            | 0.54         | 0.29         | 0.761            | 0.519         | 0.551            |
| z2            | 1.39         | 0.72         | 0.787            | 0.544         | 0.560            |
| z3            | 3.89         | 1.93         | 0.781            | 0.553         | 0.573            |
| z4 (finest)   | 28.55        | 10.14        | 0.800            | 0.576         | 0.585            |"""


def _load(name: str):
    path = FIGURES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_fig_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(FIGURES))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(FIGURES))
    return mod


def _run(mod, argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main(argv)
    return buf.getvalue()


def _needs(*names):
    missing = [n for n in names if not (RESULTS / n).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"missing artifacts: {missing}")


@_needs(*(f"manc_economy_{t}.json" for t in ("neuron_class", "subclass", "hemilineage")))
def test_table1_matches_manuscript():
    out = _run(_load("make_table1"), [])
    table = "\n".join(line for line in out.splitlines() if line.startswith("|"))
    assert table == TABLE1
    assert "cell type" not in out  # opt-in only
    assert "98.2x uncompressed" in out and "61.6x on disk" in out


@_needs("manc_economy_cell_type.json")
def test_table1_cell_type_is_opt_in():
    out = _run(_load("make_table1"), ["--include-cell-type"])
    assert "cell type (117)" in out


@_needs(
    *(f"manc_{arm}_{t}.json"
      for arm in ("economy", "fusion", "countsonly", "fusionperm", "leakctrl", "fusion_leakctrl")
      for t in ("neuron_class", "subclass", "hemilineage"))
)
def test_ed_table4_statistics():
    m = _load("make_ed_table4")
    levels = list(range(5))

    def arms(task, a, b):
        return (m.load_arm(RESULTS, a, task, "", "bal_acc"),
                m.load_arm(RESULTS, b, task, "", "bal_acc"))

    # Fusion vs geometry, Holm across five levels per task.
    sig = {}
    for task in ("neuron_class", "subclass", "hemilineage"):
        fus, geo = arms(task, "fusion", "economy")
        rows = m.with_holm(m.compare(fus, geo, levels, levels))
        assert all(r["diff"] > 0 for r in rows)
        sig[task] = m.sig_levels(rows)
        if task == "hemilineage":
            assert round(rows[0]["p_holm"], 3) == 0.052
    assert len(sig["neuron_class"]) == 5 and len(sig["subclass"]) == 5
    assert sig["hemilineage"] == ["z3"]

    # Substitution: fusion z0 vs geometry z4.
    fus, geo = arms("neuron_class", "fusion", "economy")
    r = m.compare(fus, geo, [0], [4])[0]
    assert (round(r["a"], 3), round(r["b"], 3)) == (0.852, 0.800)
    assert 3.5e-3 <= r["p"] < 4.5e-3  # "p = 4 × 10^-3"
    fus, geo = arms("subclass", "fusion", "economy")
    assert round(m.compare(fus, geo, [0], [4])[0]["p"], 2) == 0.08

    # Permutation control: max |Δ| 0.013, nothing survives Holm.
    perm_rows = []
    for task in ("neuron_class", "subclass", "hemilineage"):
        perm, geo = arms(task, "fusionperm", "economy")
        perm_rows += m.with_holm(m.compare(perm, geo, levels, levels))
    assert round(max(abs(r["diff"]) for r in perm_rows), 3) == 0.013
    assert not any(r["p_holm"] < m.ALPHA for r in perm_rows)

    # Held-out control: hemilineage Holm-significant at three levels.
    fus, geo = arms("hemilineage", "fusion_leakctrl", "leakctrl")
    assert len(m.sig_levels(m.with_holm(m.compare(fus, geo, levels, levels)))) == 3


def test_fmt_p_matches_manuscript_style():
    m = _load("make_ed_table4")
    assert m.fmt_p(6.567e-08) == "6.6 × 10^-8"
    assert m.fmt_p(0.002434) == "2.4 × 10^-3"
    assert m.fmt_p(0.01298) == "0.013"
    assert m.fmt_p(0.1926) == "0.19"
    assert m.fmt_p(9.96e-4) == "1.0 × 10^-3"


@_needs("nyxus_stability_summary.json")
def test_ed_table3_renders():
    out = _run(_load("make_ed_table3"), [])
    assert "| `STAT_FERET_DIAM_MAX` | 0.000 | 0.999 | 0.000 | 0.998 | 0.031 | 0.991 |" in out
    assert "| `STAT_MARTIN_DIAM_MIN` | 0.50 | 0.28 | 0.50 | 0.16 |" in out
    assert "0.6240 (MAXCHORDS_MAX_ANG)" in out


def test_fig_s_2d_lod_renders(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("shapely")
    if not (FIGURES / "data" / "pannuke_2d_lod_fidelity.json").exists():
        pytest.skip("PanNuke figure data not present")
    mod = _load("fig_s_2d_lod")
    assert _run(mod, ["--out", str(tmp_path)]).startswith("wrote")
    assert (tmp_path / "fig_s_2d_lod.pdf").stat().st_size > 0
