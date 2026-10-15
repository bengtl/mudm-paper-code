"""Cost-axis instrumentation for the accuracy-vs-cost Pareto frontier.

Three measures, per spec Sec 3.6: forward FLOPs (compute) and peak GPU
memory are budget-driven workload numbers reported as context; on-disk
storage per LOD is the muDM-attributable economy metric.

  - forward_flops: model-agnostic forward-pass MAC count, from forward
    hooks on Conv1d/Conv2d/Linear. Not hardcoded to one architecture --
    unlike the retired two-branch analytic formula in
    mudm-tools/scripts/measure_hemibrain_economy.py
    (`analytical_flops_forward_per_sample`), this walks whatever layers the
    model actually contains, so it applies unchanged to PointNet, DGCNN, or
    any future model built from these layer types.
  - peak_memory: peak activation/param bytes for one forward+loss+backward
    step, via torch.cuda.max_memory_allocated. CUDA only; 0 on CPU (real
    measurement runs on the benchmark host's A4500 GPUs).
  - lod_storage: per-zoom on-disk storage read from parquet metadata only
    (no decompression/decoding), adapted from the reference script's
    `lod_storage_per_zoom` / `_positions_stats` metadata-only approach.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

_BYTES_PER_VERTEX = 12  # f32 xyz


def forward_flops(model: nn.Module, n_points: int) -> int:
    """Model-agnostic MAC count for one forward pass of one n_points sample.

    Registers a forward hook on every Conv1d/Conv2d/Linear submodule and
    accumulates ``output.numel() * per_output_MACs``, where per_output_MACs
    is ``in_channels_per_group * kernel_area`` for conv layers and
    ``in_features`` for Linear (bias adds, not MACs, are not counted). A
    single dummy forward pass with a `(1, 3, n_points)` input under
    `torch.no_grad()` (`model.eval()`, so BatchNorm doesn't choke on
    batch=1) drives every hook once; hooks are removed before returning so
    repeated calls don't accumulate.

    Returns a raw summed MAC count (not doubled to FLOPs=2*MACs); deterministic
    given model + n_points, and scales with n_points through the point-wise
    conv layers while the post-max-pool FC head contributes a constant term.
    """
    macs = 0

    def _count(module: nn.Module, _inputs, output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features
        else:  # nn.Conv1d / nn.Conv2d
            in_per_group = module.in_channels // module.groups
            kernel_area = 1
            for k in module.kernel_size:
                kernel_area *= k
            macs += output.numel() * in_per_group * kernel_area

    handles = [
        m.register_forward_hook(_count)
        for m in model.modules()
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))
    ]
    try:
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 3, n_points))
    finally:
        for h in handles:
            h.remove()

    return int(macs)


def peak_memory(model: nn.Module, n_points: int, device: str, bs: int = 64) -> int:
    """Peak activation/param bytes for one forward+loss+backward step at batch `bs`.

    Meaningful on CUDA only (`torch.cuda.max_memory_allocated` has no CPU
    equivalent in core torch); returns 0 on CPU by design rather than
    attempting a CPU memory measurement -- the real numbers behind the
    paper's cost axis are measured on the benchmark host's A4500 GPUs, so this path only
    needs to be safe to call from CPU-only test/CI environments.
    """
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return 0

    model = model.to(dev)
    model.train()  # backward() needs a training-mode graph through BatchNorm
    torch.cuda.reset_peak_memory_stats(dev)

    x = torch.randn(bs, 3, n_points, device=dev)
    out = model(x)
    target = torch.randint(0, out.shape[-1], (bs,), device=dev)
    loss = F.cross_entropy(out, target)
    loss.backward()

    return int(torch.cuda.max_memory_allocated(dev))


def lod_storage(parquet_dir: str) -> dict:
    """Per-zoom on-disk storage from parquet metadata only (no decompression).

    Walks ``<parquet_dir>/zoom=<Z>/part_*.parquet``. For each zoom, sums
    ``total_compressed_size`` / ``total_uncompressed_size`` across every
    column of every row group of every part file in that zoom's directory
    -- the whole tile's on-disk footprint (positions + indices + tags), not
    just vertex data. ``total_verts`` is derived separately, from only the
    `positions` column's uncompressed bytes / 12 (f32 xyz = 12 bytes/vertex).

    The `positions` column is located via `ColumnChunkMetaData.path_in_schema`:
    each column-chunk in row-group metadata already carries its column name
    as a plain string (no dotted/nested path, since `positions` is a
    top-level large_binary field in the muDM tiling schema -- see
    `mudm_paper/tests/fixtures.py::_ROW_SCHEMA`), so a direct string
    comparison identifies it without a separate schema/column-index lookup.

    ``ratio_to_finest`` = (finest zoom's uncompressed bytes) / (this zoom's
    uncompressed bytes), finest = max zoom present -- i.e. how many times
    MORE the finest LOD costs to store than this one (1.0 for the finest
    zoom itself); `None` if this zoom's uncompressed size is 0.
    """
    per_zoom: dict[int, dict] = {}

    for zoom_dir in sorted(Path(parquet_dir).glob("zoom=*")):
        try:
            zoom = int(zoom_dir.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue

        compressed = uncompressed = positions_uncompressed = 0
        for part in sorted(zoom_dir.glob("part_*.parquet")):
            meta = pq.ParquetFile(part).metadata
            for rg in range(meta.num_row_groups):
                row_group = meta.row_group(rg)
                for ci in range(row_group.num_columns):
                    col = row_group.column(ci)
                    compressed += col.total_compressed_size
                    uncompressed += col.total_uncompressed_size
                    if col.path_in_schema == "positions":
                        positions_uncompressed += col.total_uncompressed_size

        per_zoom[zoom] = {
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "total_verts": positions_uncompressed // _BYTES_PER_VERTEX,
        }

    if per_zoom:
        finest = max(per_zoom)
        finest_uncompressed = per_zoom[finest]["uncompressed_bytes"]
        for stats in per_zoom.values():
            stats["ratio_to_finest"] = (
                finest_uncompressed / stats["uncompressed_bytes"]
                if stats["uncompressed_bytes"]
                else None
            )

    return per_zoom
