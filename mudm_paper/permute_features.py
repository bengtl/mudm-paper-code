"""Build the fixed permuted feature table for the fusion permutation control.

The permutation control (Methods, "Multimodal fusion protocol") reruns the
fusion arm with the four per-neuron synapse-count features reassigned among
neurons by ONE fixed permutation, so that the model keeps the same extra input
capacity but the counts are no longer attached to the right neuron.

Recipe (reproduces the table embedded in the deposited
``results/manc_fusionperm_*.json`` configs exactly):

1. Sort the body-id keys of the feature map as strings (``sorted(keys)``).
2. Draw ``perm = numpy.random.default_rng(seed).permutation(n)`` with seed 0.
3. Key ``keys[i]`` receives the feature vector of ``keys[perm[i]]``.

The output keeps the string-sorted key order and is written with
``json.dump`` defaults, the same serialization as ``manc_fusion_features.json``.

Usage:
    python -m mudm_paper.permute_features results/manc_fusion_features.json <out.json>
    run_manc --extra-features <out.json> --model dgcnn_fusion --prefix manc_fusionperm ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np

DEFAULT_SEED = 0


def permute_features(features: dict, seed: int = DEFAULT_SEED) -> dict:
    """Reassign feature vectors among keys with one seeded permutation.

    See the module docstring for the exact recipe. The returned dict iterates
    in string-sorted key order; its values are the input vectors, unchanged,
    each attached to a different key.
    """
    keys = sorted(features)
    perm = np.random.default_rng(seed).permutation(len(keys))
    return {k: features[keys[int(j)]] for k, j in zip(keys, perm)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("features_json", help="feature map JSON ({body_id: [floats]})")
    ap.add_argument("out_json", help="output permuted feature map JSON")
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"numpy default_rng seed (default {DEFAULT_SEED}, the deposited control)",
    )
    args = ap.parse_args(argv)
    with open(args.features_json) as f:
        features = json.load(f)
    permuted = permute_features(features, args.seed)
    with open(args.out_json, "w") as f:
        json.dump(permuted, f)
    moved = sum(permuted[k] is not features[k] for k in permuted)
    print(
        f"wrote {len(permuted)} neurons ({moved} reassigned, seed={args.seed}) "
        f"-> {args.out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
