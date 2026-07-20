import tempfile
from mudm_paper.cost import forward_flops, peak_memory, lod_storage
from mudm_paper.models import PointNet
from mudm_paper.tests.fixtures import write_synthetic_pyramid

def test_flops_monotonic_in_points():
    m = PointNet(7)
    f_small = forward_flops(m, 256)
    f_big = forward_flops(m, 1024)
    assert f_big > f_small > 0

def test_peak_memory_nonneg_cpu():
    assert peak_memory(PointNet(5), 128, "cpu") >= 0     # 0 on CPU, must not crash

def test_lod_storage_reads_all_zooms():
    d = tempfile.mkdtemp(); write_synthetic_pyramid(d, n_neurons=4, zooms=(0,1,2))
    s = lod_storage(d)
    assert set(s.keys()) == {0, 1, 2}
    assert all(v["total_verts"] > 0 for v in s.values())
    assert all(v["uncompressed_bytes"] > 0 for v in s.values())
