"""`project()`'s per-pixel winner selection, pinned against the algorithm it replaced.

The contract is `range_image[v, u, 1:4] == points[inverse_index[v, u], :3]` with
the NEAREST point winning each pixel. The selection was rewritten from
`np.unique(key, return_index=True)` to a bounded-key scatter; these tests exist to
prove the same point wins every pixel, including where two points tie exactly.
"""
import numpy as np
import pytest
from vrgrid.perception.range_image import bin_widths, load_sensor_config, project


def _unique_reference(points, sensor_cfg=None):
    """The SHIPPED-BEFORE algorithm, transcribed verbatim, as the oracle."""
    cfg = sensor_cfg or load_sensor_config()
    h, w = cfg["num_rings"], cfg["num_azimuth"]
    phi_max = np.deg2rad(cfg["phi_max_deg"])
    d_theta, d_phi = bin_widths(cfg)

    pts = np.asarray(points)
    xyz = pts[:, :3]
    intensity = (pts[:, 3].astype(np.float32) if pts.shape[1] >= 4
                 else np.ones(pts.shape[0], dtype=np.float32))
    r = np.linalg.norm(xyz, axis=1)
    finite = r > 1e-6
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    z_over_r = np.divide(xyz[:, 2], r, out=np.zeros_like(r), where=finite)
    elevation = np.arcsin(np.clip(z_over_r, -1.0, 1.0))
    u = np.floor((azimuth + np.pi) / d_theta).astype(np.int64) % w
    v = np.clip(np.floor((phi_max - elevation) / d_phi).astype(np.int64), 0, h - 1)

    keep = finite
    u, v, r = u[keep], v[keep], r[keep]
    xyz, intensity = xyz[keep], intensity[keep]
    src_idx = np.flatnonzero(keep)

    range_image = np.full((h, w, 5), np.nan, dtype=np.float32)
    inverse_index = np.full((h, w), -1, dtype=np.int32)
    order = np.argsort(r, kind="stable")
    u, v, r = u[order], v[order], r[order]
    xyz, intensity, src_idx = xyz[order], intensity[order], src_idx[order]
    _, first = np.unique(v * w + u, return_index=True)
    u, v = u[first], v[first]
    range_image[v, u, 0] = r[first]
    range_image[v, u, 1:4] = xyz[first]
    range_image[v, u, 4] = intensity[first]
    inverse_index[v, u] = src_idx[first]
    return range_image, inverse_index


def _same(a, b):
    assert np.array_equal(a[0], b[0], equal_nan=True)
    assert np.array_equal(a[1], b[1])


def _cloud(seed, n=120_000, spread=60.0):
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 4), dtype=np.float64)
    pts[:, :3] = rng.uniform(-spread, spread, (n, 3))
    pts[:, 3] = rng.random(n)
    return pts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection_matches_the_unique_algorithm_on_random_clouds(seed):
    _same(project(_cloud(seed)), _unique_reference(_cloud(seed)))


def test_selection_matches_when_many_points_share_one_pixel():
    """A dense, narrow cloud puts hundreds of points in each pixel, which is where
    'first per pixel' actually has to choose."""
    pts = _cloud(7, n=60_000, spread=2.0)
    _same(project(pts), _unique_reference(pts))


def test_exact_range_ties_resolve_to_the_same_point():
    """[!] The tie-break is the part a rewrite is most likely to get wrong.

    Every point here is duplicated at an identical radius, so the winner is
    decided purely by the stable sort's original-order rule.
    """
    base = _cloud(3, n=20_000)
    pts = np.repeat(base, 2, axis=0)          # exact duplicate ranges, adjacent
    pts[1::2, 3] = 1.0 - pts[1::2, 3]         # differ only in intensity
    ours, ref = project(pts), _unique_reference(pts)
    _same(ours, ref)
    filled = ref[1] >= 0
    assert filled.sum() > 100
    assert np.array_equal(ours[1][filled], ref[1][filled])


def test_numpy_still_keeps_the_last_duplicate_write():
    """The rule the selection depends on, asserted directly.

    If numpy ever changed duplicate fancy-index assignment to keep the FIRST
    write, the projection would silently start choosing a different point per
    pixel. This test is the tripwire.
    """
    a = np.full(4, -1, dtype=np.int64)
    a[np.array([0, 0, 0, 2])] = np.array([10, 11, 12, 13])
    assert a[0] == 12, "numpy no longer keeps the last duplicate write"
    assert a[2] == 13 and a[1] == -1 and a[3] == -1


def test_inverse_index_contract_holds_on_a_real_scan():
    """The documented contract, on real data, end to end."""
    loader = pytest.importorskip("vrgrid.perception.loader")
    if not (loader.verify_sequence_exists("08") and loader._velodyne_path("08", 0).exists()):
        pytest.skip("KITTI seq 08 not present -- set VRGRID_DATA_ROOT")
    for frame in (0, 120, 400):
        pts = loader.load_velodyne_scan(loader._velodyne_path("08", frame))
        ri, inv = project(pts)
        _same((ri, inv), _unique_reference(pts))
        filled = inv >= 0
        assert np.array_equal(ri[filled][:, 1:4].astype(np.float64),
                              pts[inv[filled], :3].astype(np.float32).astype(np.float64))
