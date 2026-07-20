"""Discriminative masked-invariance gate for Fix B (I2).

Coarse LODs feed fewer REAL points (no duplicate padding); batches pad to
the batch-max with a validity mask that both encoders must honor. The core
correctness property: in eval mode, appending masked GARBAGE points must
leave the model output UNCHANGED (atol 1e-4) -- if a pad point can move the
output at all, the masking leaks pad-point influence and the whole
right-censoring design (matched budget via fewer real points, not
duplicated ones) is unsound.
"""

import torch
from mudm_paper.models import PointNet, DGCNN


def _invariance(M):
    M.eval()
    x = torch.randn(2, 3, 40)
    full_mask = torch.ones(2, 40, dtype=torch.bool)
    out_ref = M(x)                                  # no mask
    out_ref_m = M(x, full_mask)                     # all-valid mask == no mask
    assert torch.allclose(out_ref, out_ref_m, atol=1e-5)
    # append 15 GARBAGE points, mark them invalid -> output must be UNCHANGED
    garbage = torch.randn(2, 3, 15) * 1e3
    x2 = torch.cat([x, garbage], dim=2)
    mask2 = torch.cat([full_mask, torch.zeros(2, 15, dtype=torch.bool)], dim=1)
    out_masked = M(x2, mask2)
    assert torch.allclose(out_ref, out_masked, atol=1e-4)   # pad points had ZERO effect


def test_pointnet_masked_invariance():
    _invariance(PointNet(7))


def test_dgcnn_masked_invariance():
    _invariance(DGCNN(7))


def test_dgcnn_small_valid_count():
    # only 5 valid points (< default k=20) among 30 padded -> must not crash, must give [B,n_cls]
    M = DGCNN(4).eval()
    x = torch.randn(2, 3, 30)
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[:, :5] = True
    assert M(x, mask).shape == (2, 4)
