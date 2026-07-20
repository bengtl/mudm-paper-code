"""Parquet-pyramid loader + bilateral-safe train/val/test split.

Loads per-neuron, per-LOD (level-of-detail / zoom) mesh geometry and labels
from a muDM parquet pyramid laid out as `<dir>/zoom=<Z>/part_000.parquet`,
and splits neurons into train/val/test groups so that bilateral homologs
(ids differing only by a trailing `_L`/`_R`) never straddle a split
boundary. Leakage control here is correctness-critical: a homolog pair split
across train/test would let a classifier "cheat" on near-duplicate geometry,
invalidating any downstream result.
"""

import re

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


def _open_dataset(parquet_dir: str) -> ds.Dataset:
    return ds.dataset(
        parquet_dir,
        partitioning=ds.HivePartitioning(pa.schema([("zoom", pa.int32())])),
    )


def load_neuron_lods(
    parquet_dir: str, zooms, cap: int = 16384
) -> dict:
    """Load concatenated (positions, indices) mesh geometry per body_id per zoom.

    Returns ``{body_id: {zoom: (positions f32 [V,3], indices u32 [F,3])}}``.

    A neuron may be spread across several parquet rows (tiles) at the same
    zoom. Each row's `indices` are tile-local (index only into that row's
    own `positions`), so rows are concatenated by offsetting each row's
    indices by the running vertex count *before* concatenating, ensuring the
    returned indices index correctly into the concatenated `positions`.

    If a (body_id, zoom) mesh exceeds `cap` vertices, only the first `cap`
    positions are kept and any triangle referencing a dropped vertex (index
    >= cap) is dropped.
    """
    dataset = _open_dataset(parquet_dir)
    wanted_zooms = {int(z) for z in zooms}
    # `zoom` is the Hive partition column, so this filter is pushed down to the
    # partition scan: pyarrow prunes non-matching `zoom=*/` fragments before
    # reading them, rather than materializing the whole pyramid and filtering
    # in Python. Required for multi-hundred-GB manc/flywire pyramids, which
    # would not fit in memory if fully materialized.
    table = dataset.to_table(
        columns=["zoom", "positions", "indices", "tags"],
        filter=ds.field("zoom").isin(list(wanted_zooms)),
    )

    # accum[body_id][zoom] -> list of (positions, indices) chunks, one per row/tile
    accum: dict = {}
    for zoom, pos_bytes, idx_bytes, tags in zip(
        table.column("zoom").to_pylist(),
        table.column("positions").to_pylist(),
        table.column("indices").to_pylist(),
        table.column("tags").to_pylist(),
    ):
        body_id = dict(tags)["body_id"]
        pos = np.frombuffer(pos_bytes, dtype="<f4").reshape(-1, 3)
        idx = np.frombuffer(idx_bytes, dtype="<u4").reshape(-1, 3)
        accum.setdefault(body_id, {}).setdefault(zoom, []).append((pos, idx))

    result: dict = {}
    for body_id, per_zoom in accum.items():
        result[body_id] = {}
        for zoom, chunks in per_zoom.items():
            positions, indices, offset = [], [], 0
            for pos, idx in chunks:
                positions.append(pos)
                indices.append(idx.astype(np.int64) + offset)  # tile-local -> concatenated space
                offset += len(pos)
            all_pos = np.concatenate(positions, axis=0).astype(np.float32)
            all_idx = np.concatenate(indices, axis=0)

            if len(all_pos) > cap:
                all_pos = all_pos[:cap]
                keep = np.all(all_idx < cap, axis=1)  # drop faces touching a dropped vertex
                all_idx = all_idx[keep]

            result[body_id][zoom] = (all_pos, all_idx.astype(np.uint32))
    return result


_PLACEHOLDER_RUN_RE = re.compile(r"x{3,}", re.IGNORECASE)
_PLACEHOLDER_SENTINELS = {"", "unknown", "na"}


def _is_placeholder(cls: str) -> bool:
    """True if `cls` is a placeholder/unassigned class bucket rather than a
    real label (spec Sec 3.2) -- e.g. hemibrain's `"SNxxxx"` provisional-name
    marker for a subclass/cell_type that has not been assigned a real name
    yet.

    Deliberately conservative, not an exhaustive enumeration of every
    connectome atlas's placeholder convention: flags a literal run of 3+
    consecutive "x"/"X" characters anywhere in the string (so `"SNxxxx"`
    and `"xxx"` both match, but a real short class name like `"max"` or
    `"flex"` does not), plus the empty/whitespace-only/`"unknown"`/`"na"`
    sentinels (case-insensitive) some pipelines emit for "no assignment
    yet". A falsy value (`None`, `""`) counts as a placeholder too.
    """
    if not cls:
        return True
    stripped = cls.strip()
    lowered = stripped.lower()
    if lowered in _PLACEHOLDER_SENTINELS:
        return True
    if "xxxx" in lowered:
        return True
    return bool(_PLACEHOLDER_RUN_RE.search(stripped))


def load_labels(parquet_dir: str, field: str) -> dict:
    """Map body_id -> tags[field] (first row seen per neuron; assumed
    consistent across a neuron's rows and zooms).

    Real muDM tags are SPARSE: many neurons lack fields such as
    `hemilineage` or `subclass` entirely. A neuron missing `body_id` or
    the requested `field` is skipped rather than crashing the loader; it
    then falls out of `body_ids = set(lods) & set(labels_map)` in the
    sweep, which is the correct semantics for "no label available for
    this task". A neuron whose value for `field` is a placeholder bucket
    (`_is_placeholder`, e.g. `"SNxxxx"`) is skipped the same way --
    treating an unassigned/provisional label as equivalent to a missing
    one (spec Sec 3.2), so placeholder buckets never get counted as a
    real class or enter a train/val/test split.
    """
    dataset = _open_dataset(parquet_dir)
    table = dataset.to_table(columns=["tags"])

    labels: dict = {}
    for tags in table.column("tags").to_pylist():
        tag_map = dict(tags)
        body_id = tag_map.get("body_id")
        if body_id is None or field not in tag_map:
            continue
        value = tag_map[field]
        if _is_placeholder(value):
            continue
        labels.setdefault(body_id, value)
    return labels


_SIDE_SUFFIX = re.compile(r"_[LR]$")


def grouped_stratified_split(
    body_ids: list,
    labels: dict,
    seed: int,
    val: float = 0.15,
    test: float = 0.15,
    min_class: int = 20,
    group_of: dict | None = None,
) -> tuple:
    """Split body_ids into (train, val, test), grouped so homologs always
    land in the same split.

    By default the group key is the side-stripped id (bilateral homologs
    differing only by a trailing `_L`/`_R`). If `group_of` is given, the
    group key for a body_id is `group_of[body_id]` instead (falling back to
    the body_id itself if absent from the mapping) -- this lets callers
    group by an external homolog field (e.g. an `instance`/type tag) rather
    than the body_id's own side suffix.

    Classes with fewer than `min_class` body_ids (counting neurons, not
    groups) are dropped entirely before splitting. There is deliberately no
    fallback to an ungrouped split when a class is small -- that would
    silently reintroduce homolog leakage, defeating the purpose of this
    function. Deterministic given `seed`.
    """
    rng = np.random.RandomState(seed)

    class_counts: dict = {}
    for body_id in body_ids:
        cls = labels[body_id]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    kept_classes = {cls for cls, n in class_counts.items() if n >= min_class}

    # group_key (side-stripped id) -> {"class": cls, "members": [body_id, ...]}
    groups: dict = {}
    for body_id in body_ids:
        cls = labels[body_id]
        if cls not in kept_classes:
            continue
        if group_of is not None:
            group_key = group_of.get(body_id, body_id)
        else:
            group_key = _SIDE_SUFFIX.sub("", body_id)
        group = groups.setdefault(group_key, {"class": cls, "members": []})
        group["members"].append(body_id)

    class_to_groups: dict = {}
    for group_key, group in groups.items():
        class_to_groups.setdefault(group["class"], []).append(group_key)

    train, val_ids, test_ids = [], [], []
    for cls in sorted(class_to_groups):
        group_keys = sorted(class_to_groups[cls])
        order = rng.permutation(len(group_keys))
        group_keys = [group_keys[i] for i in order]

        n = len(group_keys)
        n_test = round(n * test)
        n_val = round(n * val)
        test_keys = group_keys[:n_test]
        val_keys = group_keys[n_test:n_test + n_val]
        train_keys = group_keys[n_test + n_val:]

        for group_key in train_keys:
            train.extend(groups[group_key]["members"])
        for group_key in val_keys:
            val_ids.extend(groups[group_key]["members"])
        for group_key in test_keys:
            test_ids.extend(groups[group_key]["members"])

    return train, val_ids, test_ids
