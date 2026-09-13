"""`normalise(..., with_incidence=False)`, pinned.

On the KITTI path (incidence_compensated=True) rho_hat is the intensity and never
uses cos_inc, yet the frame loop computed the full incidence normal field every
frame and threw it away. Skipping it is only acceptable if the bytes the pipeline
keeps are exactly the bytes it kept before.
"""
import numpy as np
import pytest
from vrgrid.perception.reflectivity import normalise, scatter_to_points


def _range_image(seed, h=64, w=512, fill=0.7):
    """A plausible (H, W, 5) range image with holes, on the real image shape."""
    rng = np.random.default_rng(seed)
    ri = np.full((h, w, 5), np.nan, dtype=np.float32)
    mask = rng.random((h, w)) < fill
    r = rng.uniform(2.0, 80.0, (h, w))
    az = np.linspace(-np.pi, np.pi, w, endpoint=False)[None, :].repeat(h, 0)
    el = np.linspace(0.05, -0.43, h)[:, None].repeat(w, 1)
    xyz = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az),
                    r * np.sin(el)], axis=2)
    ri[mask, 0] = r[mask]
    ri[mask, 1:4] = xyz[mask]
    ri[mask, 4] = rng.random(mask.sum())
    return ri


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_skipping_incidence_leaves_rho8_and_rho_hat_bit_identical(seed):
    ri = _range_image(seed)
    full = normalise(ri)
    lean = normalise(ri, with_incidence=False)
    assert np.array_equal(lean.rho8, full.rho8)
    assert np.array_equal(lean.rho_hat, full.rho_hat, equal_nan=True)
    assert lean.cos_inc is None and lean.flags is None


def test_default_still_computes_incidence():
    """The default must not change: tests/test_reflectivity.py reads flags and
    cos_inc from it, and so may any other caller."""
    res = normalise(_range_image(4))
    assert res.cos_inc is not None and res.flags is not None
    assert res.flags.shape == res.rho8.shape


def test_skipping_incidence_is_refused_on_the_raw_power_path():
    """With incidence_compensated=False rho_hat is divided by cos_inc, so the
    computation cannot be skipped -- refuse rather than return a wrong byte."""
    with pytest.raises(ValueError):
        normalise(_range_image(5), incidence_compensated=False, with_incidence=False)


def test_scatter_to_points_without_flags_keeps_rho8_identical():
    ri = _range_image(6)
    h, w = ri.shape[:2]
    n = 50_000
    inv = np.full((h, w), -1, dtype=np.int32)
    filled = np.isfinite(ri[:, :, 0])
    inv[filled] = np.random.default_rng(6).permutation(n)[:filled.sum()]

    rho8_full, flags_full = scatter_to_points(normalise(ri), inv)
    rho8_lean, flags_lean = scatter_to_points(normalise(ri, with_incidence=False), inv)
    assert np.array_equal(rho8_lean, rho8_full)
    assert flags_full is not None and flags_lean is None


def test_real_scan_rho8_is_identical_through_the_pipeline_functions():
    """On a real KITTI scan, the exact projection the pipeline uses."""
    loader = pytest.importorskip("vrgrid.perception.loader")
    if not (loader.verify_sequence_exists("00") and loader._velodyne_path("00", 0).exists()):
        pytest.skip("KITTI seq 00 not present -- set VRGRID_DATA_ROOT")
    from vrgrid.perception.range_image import project

    for frame in (0, 50, 100):
        pts = loader.load_velodyne_scan(loader._velodyne_path("00", frame))
        ri, inv = project(pts)
        a, _ = scatter_to_points(normalise(ri), inv)
        b, _ = scatter_to_points(normalise(ri, with_incidence=False), inv)
        assert np.array_equal(a, b), f"frame {frame}"
