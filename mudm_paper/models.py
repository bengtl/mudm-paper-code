"""Point-cloud classifiers sharing the contract forward([B,3,N]) -> [B,n_classes].

PointNet is the max-pool baseline; DGCNN is the sampling-robust EdgeConv arm.
"""

import torch
import torch.nn as nn


class PointNet(nn.Module):
    # adapted from mudm-tools/scripts/benchmark_hemibrain_ml.py (class PointNet)
    """Lightweight PointNet classifier.

    Architecture: shared MLP (3->64->128->256) with BatchNorm and ReLU,
    global max pooling, then FC 256->128->num_classes with dropout.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        # Shared MLP (applied per-point via Conv1d)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, N) point clouds (channel-first; no internal transpose
                needed since Conv1d already expects channels-first input --
                this differs from the source, which took (B, N, 3)).
            mask: (B, N) bool, True = valid point, False = padding (Fix B /
                I2's variable-N batching: samples are padded to the batch
                max with zeros and this mask marks which columns are real).
                `None` (default) preserves the original unmasked behavior --
                every point is treated as valid, matching every pre-Fix-B
                caller and test.

        Returns:
            (B, n_classes) logits. When `mask` is given, padding points have
            EXACTLY zero effect on the output (in eval mode): they are
            excluded from the global max-pool via a `-inf` fill before the
            max, so a padded column can never win the per-channel max.
            (In train mode, BatchNorm's batch statistics still see the pad
            points' pre-pool activations -- a minor, accepted contamination;
            the masked-invariance guarantee is exact in eval mode, where
            BatchNorm uses its fixed running stats instead.)
        """
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)                  # (B, 256, N)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), float("-inf"))
        # Global max pool: (B, 256, N) -> (B, 256)
        x = x.max(dim=2)[0]
        return self.fc(x)


def _knn_idx(x, k, mask=None):            # x: [B,C,N] -> [B,N,k]
    inner = -2 * torch.matmul(x.transpose(2, 1), x)          # [B,N,N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)              # [B,1,N]
    pair = -xx - inner - xx.transpose(2, 1)                  # neg sq dist [B,N,N]
    if mask is not None:
        # Never pick an invalid point as a NEIGHBOR (column), for any query
        # (row) -- including invalid queries, though their own output is
        # discarded downstream by the final pooling mask anyway.
        pair = pair.masked_fill(~mask.unsqueeze(1), float("-inf"))
    return pair.topk(k=k, dim=-1)[1]                         # [B,N,k]

def _edge_feature(x, k, mask=None):       # x: [B,C,N] -> [B,2C,N,k]
    B, C, N = x.size()
    idx = _knn_idx(x, k, mask)
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + base).view(-1)
    feat = x.transpose(2, 1).contiguous().view(B * N, C)[idx].view(B, N, k, C)
    center = x.transpose(2, 1).view(B, N, 1, C).repeat(1, 1, k, 1)
    return torch.cat((feat - center, center), dim=3).permute(0, 3, 1, 2).contiguous()

class DGCNN(nn.Module):
    """Compact EdgeConv DGCNN (sampling-robust arm)."""
    def __init__(self, n_classes: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv_out = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x, mask=None):      # x: [B,3,N] -> [B,n_classes]
        """See PointNet.forward for the `mask` contract (None preserves the
        original unmasked behavior). DGCNN additionally clamps its kNN `k`
        to the batch's minimum VALID point count (never dips into masked
        padding to fill out a neighbor list), and never lets a masked
        neighbor be selected in the first place (`_knn_idx`'s own mask
        handling) -- both needed for appending masked padding to leave the
        output exactly unchanged in eval mode."""
        if mask is not None:
            if mask.size(0) == 0:
                min_valid = 0                    # empty batch (e.g. a degenerate test split)
            else:
                min_valid = int(mask.sum(dim=1).min().item())
            k = max(1, min(self.k, min_valid))   # topk needs k <= every sample's valid count
        else:
            k = min(self.k, x.size(2))
        x1 = self.conv1(_edge_feature(x, k, mask)).max(dim=-1)[0]           # [B,64,N]
        x2 = self.conv2(_edge_feature(x1, k, mask)).max(dim=-1)[0]          # [B,64,N]
        x = self.conv_out(torch.cat((x1, x2), dim=1))                      # [B,256,N]
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), float("-inf"))
        x = x.max(dim=-1)[0]
        return self.head(x)
