"""The permutation-control table regenerates byte for byte from the deposited inputs."""

import json
from pathlib import Path

import pytest

from mudm_paper.permute_features import main, permute_features

RESULTS = Path(__file__).resolve().parents[2] / "results"
FEATURES = RESULTS / "manc_fusion_features.json"
PERM_ARTIFACTS = [
    RESULTS / f"manc_fusionperm_{task}.json"
    for task in ("neuron_class", "subclass", "hemilineage")
]


def test_permute_is_a_bijection_on_values():
    feats = {str(i): [float(i), float(i) / 2] for i in range(50)}
    out = permute_features(feats, seed=0)
    assert list(out) == sorted(feats)  # string-sorted key order
    assert sorted(map(tuple, out.values())) == sorted(map(tuple, feats.values()))
    assert any(out[k] != feats[k] for k in feats)  # something actually moved
    assert permute_features(feats, seed=0) == out  # deterministic
    assert permute_features(feats, seed=1) != out  # seed matters


@pytest.mark.skipif(
    not FEATURES.exists() or not all(p.exists() for p in PERM_ARTIFACTS),
    reason="deposited fusion artifacts not present",
)
def test_regenerates_deposited_permutation_table(tmp_path):
    out_json = tmp_path / "manc_fusion_features_perm0.json"
    assert main([str(FEATURES), str(out_json)]) == 0
    regenerated_bytes = out_json.read_bytes()
    regenerated = json.loads(regenerated_bytes)

    for path in PERM_ARTIFACTS:
        raw = path.read_text()
        artifact = json.loads(raw)
        embedded = artifact["config"]["extra_features"]
        # Same keys in the same order with bit-identical float values.
        assert list(embedded) == list(regenerated)
        assert embedded == regenerated
        # The CLI output serializes exactly like the embedded table.
        assert json.dumps(embedded).encode() == regenerated_bytes
        # Splicing the regenerated table back reproduces the artifact file
        # byte for byte (run_manc writes json.dump(..., indent=2)).
        artifact["config"]["extra_features"] = regenerated
        assert json.dumps(artifact, indent=2) == raw, path.name
