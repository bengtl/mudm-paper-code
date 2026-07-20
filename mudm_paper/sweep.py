"""The right-sizing grid experiment driver -- wires data/sampling/models/cost/
stats/knee into the paper's central economy result.

For each classification `label`, this trains real (small) PointNet/DGCNN
classifiers on point clouds sampled at matched `budget`s from muDM's
precomputed LOD pyramid (the `"precomputed"` arm, swept over every level)
and from on-the-fly baselines that decimate the finest mesh live (the
`"fine_*"` arms, recorded only at the finest level -- there is no pyramid to
sweep for a method that never touches anything but full-resolution
geometry). Repeating over `seeds` gives per-(label, arm, level, budget)
micro-accuracy/balanced-accuracy/macro-F1 with standard error
(`stats.aggregate`) plus the raw per-seed values, a TOST equivalence test of
the muDM read path against the fine-mesh FPS baseline at matched budget
(`stats.tost_equivalence`), and the moving-knee readout: the cheapest LOD
(by on-disk STORAGE, the muDM-attributable cost axis -- not FLOPs, which are
budget-driven workload context reported separately) within `epsilon` of
full-resolution accuracy, per label, and whether that knee shifts across
labels (`knee.moving_knee`). Balanced accuracy, not micro accuracy, drives
both the knee and the TOST: real muDM classes are imbalanced enough that a
majority class dominating micro accuracy would mask exactly the coarse/fine
LOD gap on rare classes this experiment is meant to detect.

Real muDM tags are also SPARSE (many neurons lack a given field) and
contain placeholder/provisional buckets (e.g. hemibrain's `"SNxxxx"`) --
`data.load_labels` already treats both as "no label" (see its docstring),
so a `label` here can legitimately collapse to fewer than 2 usable classes
or an empty train set purely from that filtering. `run_sweep` detects this
per label (via the split it computes for seed 0) and skips the label with
a `warnings.warn`, rather than crashing on `np.stack([])` downstream.

The same sparsity applies to `group_field` (e.g. an `instance` tag used to
group bilateral homologs for the split): `load_labels` drops any neuron
missing that tag from the resulting `group_of` map. Such a neuron must be
excluded from the experiment entirely (`_eligible_body_ids`) rather than
left in `body_ids` to reach `grouped_stratified_split`'s `.get(bid, bid)`
fallback, which would treat it as its own singleton group -- a group
distinct from its bilateral homolog's real group if that homolog DOES
carry an intact tag, letting the pair straddle train/test and leak
near-duplicate geometry.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from mudm_paper.cost import forward_flops, lod_storage
from mudm_paper.data import grouped_stratified_split, load_labels, load_neuron_lods
from mudm_paper.knee import moving_knee
from mudm_paper.models import DGCNN, PointNet
from mudm_paper.qem import onthefly_qem
from mudm_paper.sampling import sample_to_budget
from mudm_paper.stats import aggregate, tost_equivalence

_SIDE_SUFFIX = re.compile(r"_[LR]$")
_MODEL_FACTORY = {"pointnet": PointNet, "dgcnn": DGCNN}


def _arm_method_and_levels(arm: str, levels: list) -> tuple:
    """Map an experiment arm to its sampling method and the level(s) it is
    evaluated at.

    `"precomputed"` is the muDM read path: sample each neuron's mesh AT
    each precomputed LOD level via `method="precomputed"`, swept over every
    level in `levels`. A `"fine_*"` arm samples the FINEST level only
    (`max(levels)`), via the method named after its suffix (`"fine_fps"` ->
    `"fps"`, etc.) -- these are the no-pyramid / on-the-fly-decimation
    baselines, so there is nothing to sweep across levels.
    """
    if arm == "precomputed":
        return "precomputed", list(levels)
    if arm.startswith("fine_"):
        return arm[len("fine_"):], [max(levels)]
    raise ValueError(f"Unknown arm: {arm!r}")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over classes `0..n_classes-1`, implemented by hand
    (no sklearn dependency).

    Per class: precision = tp/(tp+fp), recall = tp/(tp+fn), f1 = 2PR/(P+R),
    each guarded to 0.0 on a zero denominator. This covers a class that is
    absent from both y_true and y_pred in this seed's test split (e.g. a
    class present in train but with no members drawn into test) without
    raising or propagating NaN.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Balanced accuracy (spec Sec 3.5): mean of per-class recall
    TP/(TP+FN) over classes `0..n_classes-1` that have >=1 true instance in
    this split. A class with zero true instances this seed (e.g. dropped
    by small-class val/test rounding) is excluded from the mean rather
    than contributing a spurious 0, guarded the same zero-division-safe
    way as `_macro_f1` above. This -- not micro accuracy -- is what should
    drive the knee/TOST: a majority class dominating micro accuracy would
    mask exactly the coarse/fine LOD gap on rare classes the experiment is
    meant to detect.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp + fn == 0:
            continue  # class has zero true instances this split; excluded from the mean
        recalls.append(tp / (tp + fn))
    return float(np.mean(recalls)) if recalls else 0.0


def _build_xy(
    body_ids: list, lods: dict, level: int, method: str, budget: int, seed: int,
    labels_map: dict, cls_idx: dict, qem_cache: dict | None = None,
) -> tuple:
    """Assemble one (X, y, mask, eff_points) batch for a set of neurons at a
    fixed (level, method, budget).

    Fix B / I2: `sample_to_budget` right-censors -- each neuron now
    contributes `n_i = min(budget, n_available)` REAL points, no duplicate
    padding. Since `n_i` varies per neuron (a coarse LOD's mesh can have
    fewer vertices than `budget`), this collates the batch itself: X is
    channel-first zero-padded to this batch's own max `N_max = max(n_i)`,
    shape [n_neurons, 3, N_max], plus a boolean `mask` [n_neurons, N_max]
    (True for each sample's first `n_i` columns, False for padding) that
    both PointNet and DGCNN honor via their `forward(x, mask)` contract.
    `eff_points` is the plain list of per-neuron `n_i` (same order as the
    batch), for the caller to report real-point-count statistics
    ("eff_points" in `run_sweep`'s result) rather than assuming every arm
    fed exactly `budget` points.

    `qem_cache`, if given, memoizes the (expensive) QEM bisection per
    `(body_id, budget)` -- it is seed- and label-independent, so re-running
    it every seed (as the pre-Fix-B code did) is wasted work. Only consulted
    for `method == "onthefly_qem"`; the post-QEM subsample-to-budget step
    (still seed-dependent when the decimated mesh has more vertices than
    budget) is delegated to `sample_to_budget(..., method="random", ...)`,
    which implements exactly the same "return all if <= budget, else
    subsample without replacement" contract on the cached decimated mesh.

    An empty `body_ids` returns empty tensors (`[0,3,budget]` / `[0]` /
    `[0,budget]` / `[]`) rather than crashing on `np.stack([])` -- can
    happen when a small class's val/test rounding sends zero members into a
    split even though the label as a whole has >=2 usable classes and a
    non-empty train set (the degenerate-whole-label case is handled
    separately, in `run_sweep`, before `_build_xy` is ever called for `tr`).
    """
    if not body_ids:
        return (
            torch.empty((0, 3, budget), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, budget), dtype=torch.bool),
            [],
        )
    if qem_cache is None:
        qem_cache = {}

    xs, ys, ns = [], [], []
    for body_id in body_ids:
        positions, indices = lods[body_id][level]
        if method == "onthefly_qem":
            key = (body_id, budget)
            if key not in qem_cache:
                qem_cache[key] = onthefly_qem(positions, indices, budget)
            pos_qem, idx_qem = qem_cache[key]
            points = sample_to_budget(pos_qem, idx_qem, budget, "random", seed)  # [n_i,3]
        else:
            points = sample_to_budget(positions, indices, budget, method, seed)  # [n_i,3]
        xs.append(points)
        ys.append(cls_idx[labels_map[body_id]])
        ns.append(points.shape[0])

    n_max = max(ns)
    b = len(xs)
    x = torch.zeros((b, 3, n_max), dtype=torch.float32)
    mask = torch.zeros((b, n_max), dtype=torch.bool)
    for i, points in enumerate(xs):
        n_i = points.shape[0]
        x[i, :, :n_i] = torch.from_numpy(points.T)  # [3,n_i]
        mask[i, :n_i] = True
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, mask, ns


def _batched_predict(
    model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor,
    batch_size: int, device: str,
) -> np.ndarray:
    """Argmax class predictions for every row of `x`, computed in mini-batches
    of `batch_size`. Returns a 1-D int64 numpy array of length `x.shape[0]`.

    Eval is mini-batched for the same reason training is: at manc-full scale a
    test split holds thousands of neurons, and running DGCNN over the whole
    split in one forward materializes an `[B, 2C, N, k]` edge-feature tensor
    that exhausts GPU memory (empirically ~8.5 GiB for B=3476 -> OOM on a 20 GB
    card). Batching bounds the peak forward allocation to `batch_size` samples.
    Each batch is moved to `device` on demand so the full (CPU-resident) test
    tensor is never copied to the GPU at once.

    Note DGCNN clamps its kNN `k` to the per-batch minimum valid-point count
    (see models.py), so batched eval is not bit-identical to a hypothetical
    single-forward eval -- but that batch-dependence is exactly the regime the
    model was trained under (training is already mini-batched at the same
    `batch_size`), and batching here is deterministic (fixed test order), so it
    is consistent across every arm/level/seed being compared. Assumes `model`
    is already in the desired mode and the caller owns any `torch.no_grad()`."""
    chunks = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].to(device)
        mb = mask[start:start + batch_size].to(device)
        chunks.append(model(xb, mb).argmax(dim=1).cpu())
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return torch.cat(chunks).numpy()


def _train_eval(
    model: torch.nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor, mask_tr: torch.Tensor,
    x_te: torch.Tensor, y_te: torch.Tensor, mask_te: torch.Tensor, epochs: int, lr: float,
    batch_size: int, device: str, n_classes: int,
) -> tuple:
    """Train `model` for `epochs` full passes over (x_tr, y_tr) with Adam +
    cross-entropy (shuffled mini-batches of `batch_size`), then evaluate on
    (x_te, y_te). Returns (accuracy, balanced_accuracy, macro_f1).

    Fix B / I2: `mask_tr`/`mask_te` (from `_build_xy`'s batch-max collate)
    are passed into every `model(x, mask)` call, train AND eval, so padding
    introduced by variable-N batching never contaminates a prediction --
    see `models.py`'s masked-invariance contract.

    Eval is mini-batched via `_batched_predict` (same `batch_size` as
    training): at manc-full scale a test split has thousands of neurons and a
    single whole-split DGCNN forward OOMs the GPU. Only the (small) train
    tensor is pinned to `device` up front; test batches are moved on demand.
    """
    model = model.to(device)
    x_tr, y_tr, mask_tr = x_tr.to(device), y_tr.to(device), mask_tr.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            if batch_idx.numel() < 2:
                # BatchNorm1d requires >=2 samples per channel in train mode;
                # a stray remainder batch of size 1 would otherwise crash.
                continue
            optimizer.zero_grad()
            logits = model(x_tr[batch_idx], mask_tr[batch_idx])
            loss = F.cross_entropy(logits, y_tr[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = _batched_predict(model, x_te, mask_te, batch_size, device)
    y_true = y_te.cpu().numpy()
    acc = float((preds == y_true).mean()) if len(y_true) else 0.0
    bal_acc = _balanced_accuracy(y_true, preds, n_classes)
    f1 = _macro_f1(y_true, preds, n_classes)
    return acc, bal_acc, f1


def _eligible_body_ids(
    lods: dict, labels_map: dict, group_of: dict | None, group_field: str | None
) -> list:
    """Body ids eligible to enter a label's split.

    Without `group_field`, this is just `set(lods) & set(labels_map)`:
    every body_id is its own group by construction
    (`grouped_stratified_split`'s default side-stripped-id key), so there
    is no group coverage to check.

    With `group_field`, a body_id must ALSO be present in `group_of`
    (built from `load_labels`, which already drops neurons missing or
    placeholder for that field -- see its docstring). An id absent from
    `group_of` must be EXCLUDED here rather than passed through to
    `grouped_stratified_split`: its `group_of.get(bid, bid)` fallback
    would treat a coverage-less id as its own singleton group, distinct
    from its bilateral homolog's real group if that homolog DOES carry an
    intact tag -- letting the pair land in different splits and leak
    near-duplicate geometry across train/test.
    """
    eligible = set(lods) & set(labels_map)
    if group_field:
        eligible &= set(group_of)
    return sorted(eligible)


def run_sweep(parquet_dir: str, config: dict) -> dict:
    """Run the right-sizing grid experiment end-to-end.

    See module docstring for the experiment design. `config` keys:
    labels, budgets, levels, arms, seeds, epochs, model, device, epsilon,
    min_class (default 20), group_field (default None), lr (default 1e-3),
    batch_size (default 32), cap (default 16384 -- max vertices kept per
    neuron per level by `load_neuron_lods`; the default TRUNCATES large
    finest-level meshes spatially, so raise it for a fair full-resolution
    baseline on connectomes with big neurons).

    Returns a dict keyed by label (each holding per-arm/level/budget
    `{"acc", "bal_acc", "f1"}` aggregates plus a `"raw"` sibling with the
    per-seed lists behind each, and a `"meta"` sibling with
    `{"n_classes", "classes", "support"}` from the LAST seed processed --
    class survival is driven by `min_class` filtering before any seed's
    split, so it is stable across seeds and the last seed is as
    representative as any other), plus top-level "tost", "knee", "cost",
    and "config" entries. A label that collapses to <2 usable classes or
    an empty train set (real tags are sparse/imbalanced) is skipped
    entirely -- absent from `result`, `result["tost"]`, and
    `result["knee"]` -- with a `warnings.warn`, rather than crashing.
    """
    labels = config["labels"]
    budgets = config["budgets"]
    levels = config["levels"]
    arms = config["arms"]
    seeds = config["seeds"]
    epochs = config["epochs"]
    model_name = config["model"]
    device = config["device"]
    epsilon = config["epsilon"]
    min_class = config.get("min_class", 20)
    group_field = config.get("group_field")
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 32)

    finest = max(levels)
    coarsest = min(levels)

    # I1: geometry is label- and seed-independent -- load once and reuse for
    # every label/seed rather than re-reading the whole pyramid per label.
    # `cap` bounds vertices kept per neuron per level; the loader default
    # (16384) silently truncates large finest-level meshes to the FIRST 16384
    # verts (spatial, not decimation), so callers benchmarking a fair
    # full-resolution level must raise it (see config docstring).
    cap = config.get("cap", 16384)
    lods = load_neuron_lods(parquet_dir, tuple(levels), cap=cap)

    # I1 leftover (Fix B): the QEM bisection is seed- and label-independent
    # given (body_id, budget) -- memoize it here (scoped to this run_sweep
    # call only, so it never leaks stale geometry across calls/tests) so the
    # fine_onthefly_qem arm doesn't redo the expensive bisection every seed.
    qem_cache: dict = {}

    # I6 preflight: fail loudly and BEFORE any training if the configured
    # `levels` aren't all present as tiled zooms in this pyramid, instead of
    # silently training on whatever subset of levels happened to load.
    available_zooms = {z for per_zoom in lods.values() for z in per_zoom}
    assert set(levels) <= available_zooms, (
        f"levels {levels} not all present in tiled zooms {sorted(available_zooms)}"
    )

    # group_field (e.g. an `instance` tag for bilateral grouping) is also
    # label- and seed-independent -- same reasoning as `lods` above.
    group_of = None
    if group_field:
        groups_map = load_labels(parquet_dir, group_field)
        group_of = {bid: _SIDE_SUFFIX.sub("", v) for bid, v in groups_map.items()}

    result: dict = {}
    processed_labels: list = []  # labels NOT skipped as degenerate; drives knee/TOST below
    last_n_classes = None

    for label in labels:
        labels_map = load_labels(parquet_dir, label)
        candidate_ids = set(lods) & set(labels_map)
        body_ids = _eligible_body_ids(lods, labels_map, group_of, group_field)
        if group_field:
            n_dropped = len(candidate_ids) - len(body_ids)
            if n_dropped:
                # Bilateral-leakage guard: see _eligible_body_ids and the
                # module docstring. Reported per label for auditability --
                # on real manc, `instance` is sparse enough (many sensory
                # neurons lack it) that this can drop a meaningful chunk of
                # otherwise-usable neurons.
                warnings.warn(
                    f"label {label!r}: dropping {n_dropped} neuron(s) lacking "
                    f"group_field {group_field!r} coverage (missing/placeholder "
                    f"tag) to avoid ungrouped-singleton bilateral leakage"
                )

        acc_seeds: dict = {}
        bal_acc_seeds: dict = {}
        f1_seeds: dict = {}
        eff_seeds: dict = {}
        for arm in arms:
            _, arm_levels = _arm_method_and_levels(arm, levels)
            acc_seeds[arm] = {lvl: {b: [] for b in budgets} for lvl in arm_levels}
            bal_acc_seeds[arm] = {lvl: {b: [] for b in budgets} for lvl in arm_levels}
            f1_seeds[arm] = {lvl: {b: [] for b in budgets} for lvl in arm_levels}
            eff_seeds[arm] = {lvl: {b: [] for b in budgets} for lvl in arm_levels}

        skip_label = False
        meta_classes = None
        meta_support = None

        for seed in range(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            tr, va, te = grouped_stratified_split(
                body_ids, labels_map, seed=seed, min_class=min_class, group_of=group_of
            )
            classes = sorted({labels_map[b] for b in tr})
            if len(classes) < 2 or len(tr) == 0:
                # C1: real muDM tags are sparse/imbalanced enough that a label
                # can legitimately have no usable classifier here (e.g. every
                # class but one fails `min_class`, or -- pathologically -- none
                # do). Skip the whole label rather than np.stack([])-crashing
                # downstream; class survival is set by min_class filtering
                # before any split, so this is stable across seeds and seed 0
                # is a sufficient witness.
                warnings.warn(
                    f"skipping label {label!r}: degenerate split at seed={seed} "
                    f"({len(classes)} usable class(es) in train, {len(tr)} train examples)"
                )
                skip_label = True
                break

            cls_idx = {c: i for i, c in enumerate(classes)}
            # Drop val/test neurons whose class didn't survive into train
            # (e.g. dropped by min_class, or never sampled into train by chance).
            va = [b for b in va if labels_map[b] in cls_idx]
            te = [b for b in te if labels_map[b] in cls_idx]
            last_n_classes = len(classes)

            # I4 meta: n_classes/classes/support from this (the LAST) seed.
            meta_classes = classes
            meta_support = {c: 0 for c in classes}
            for body_id in te:
                meta_support[labels_map[body_id]] += 1

            for arm in arms:
                method, arm_levels = _arm_method_and_levels(arm, levels)
                for level in arm_levels:
                    for budget in budgets:
                        x_tr, y_tr, mask_tr, ns_tr = _build_xy(
                            tr, lods, level, method, budget, seed, labels_map, cls_idx, qem_cache
                        )
                        x_te, y_te, mask_te, ns_te = _build_xy(
                            te, lods, level, method, budget, seed, labels_map, cls_idx, qem_cache
                        )
                        model = _MODEL_FACTORY[model_name](len(classes))
                        acc, bal_acc, f1 = _train_eval(
                            model, x_tr, y_tr, mask_tr, x_te, y_te, mask_te,
                            epochs, lr, batch_size, device, len(classes)
                        )
                        acc_seeds[arm][level][budget].append(acc)
                        bal_acc_seeds[arm][level][budget].append(bal_acc)
                        f1_seeds[arm][level][budget].append(f1)
                        # Fix B: effective (real, right-censored) point counts fed
                        # this cell, across both train and test neurons this seed.
                        eff_seeds[arm][level][budget].extend(ns_tr)
                        eff_seeds[arm][level][budget].extend(ns_te)

        if skip_label:
            continue

        result[label] = {
            arm: {
                level: {
                    budget: {
                        "acc": aggregate(acc_seeds[arm][level][budget]),
                        "bal_acc": aggregate(bal_acc_seeds[arm][level][budget]),
                        "f1": aggregate(f1_seeds[arm][level][budget]),
                        # Fix B / I2: real (right-censored) point counts actually
                        # fed to the model this cell -- lets a reader see coarse
                        # LODs feed fewer points rather than assuming `budget`.
                        "eff_points": {
                            "mean": float(np.mean(eff_seeds[arm][level][budget])),
                            "min": int(np.min(eff_seeds[arm][level][budget])),
                        },
                        "raw": {
                            "acc": acc_seeds[arm][level][budget],
                            "bal_acc": bal_acc_seeds[arm][level][budget],
                            "f1": f1_seeds[arm][level][budget],
                        },
                    }
                    for budget in budgets
                }
                for level in acc_seeds[arm]
            }
            for arm in arms
        }
        result[label]["meta"] = {
            "n_classes": len(meta_classes) if meta_classes is not None else 0,
            "classes": meta_classes if meta_classes is not None else [],
            "support": meta_support if meta_support is not None else {},
        }
        processed_labels.append(label)

    # --- cost axis: storage per LOD (muDM-attributable) + FLOPs (budget-driven context) ---
    storage = lod_storage(parquet_dir)
    cost_model = _MODEL_FACTORY[model_name](last_n_classes or 2)
    result["cost"] = {
        "storage_per_level": storage,
        "flops": {budget: forward_flops(cost_model, budget) for budget in budgets},
    }

    # --- moving knee: cheapest LOD (by STORAGE) within epsilon of full-res BALANCED
    # accuracy (I3), per label. Skipped (degenerate) labels are excluded via
    # `processed_labels` rather than the raw `labels` config list. ---
    if "precomputed" in arms:
        largest_budget = max(budgets)
        per_task = {
            label: [
                {
                    "level": z,
                    "acc_mean": result[label]["precomputed"][z][largest_budget]["bal_acc"]["mean"],
                    "cost": storage[z]["uncompressed_bytes"],
                }
                for z in levels
            ]
            for label in processed_labels
        }
        ref_by_task = {
            label: result[label]["precomputed"][finest][largest_budget]["bal_acc"]["mean"]
            for label in processed_labels
        }
        result["knee"] = moving_knee(per_task, ref_by_task, epsilon)
    else:
        result["knee"] = None

    # --- TOST: precomputed@coarsest vs fine_fps@finest, matched budget, on BALANCED
    # accuracy (I3). Sourced from the per-seed "raw" arrays persisted above (I4)
    # rather than a separate accumulator dict. ---
    if "precomputed" in arms and "fine_fps" in arms:
        result["tost"] = {
            label: {
                budget: tost_equivalence(
                    result[label]["precomputed"][coarsest][budget]["raw"]["bal_acc"],
                    result[label]["fine_fps"][finest][budget]["raw"]["bal_acc"],
                    epsilon,
                )
                for budget in budgets
            }
            for label in processed_labels
        }

    result["config"] = config
    return result
