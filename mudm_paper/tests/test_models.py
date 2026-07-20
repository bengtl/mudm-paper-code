import torch
from mudm_paper.models import PointNet, DGCNN

def test_forward_shapes():
    x = torch.randn(4, 3, 128)
    for M in (PointNet(7), DGCNN(7)):
        M.eval()
        y = M(x)
        assert y.shape == (4, 7)

def test_dgcnn_handles_small_pointset():
    x = torch.randn(4, 3, 8)          # N < default k=20
    y = DGCNN(5).eval()(x)
    assert y.shape == (4, 5)

def test_pointnet_handles_small_pointset():
    x = torch.randn(4, 3, 8)
    y = PointNet(5).eval()(x)
    assert y.shape == (4, 5)
