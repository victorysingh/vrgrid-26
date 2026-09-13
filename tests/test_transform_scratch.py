"""The allocation-free `transform_points` path, pinned.

It exists to remove the largest per-stage p99 tail in the frame
(`reports/r-b-p99-tail-investigation.md` section 6), and it is only worth having
if it is bit-identical to the allocating path. These tests hold it to that, and
also pin the contract that makes it dangerous if misused: the result is a view
into the scratch and is overwritten by the next call.
"""
import tracemalloc

import numpy as np
import pytest
from vrgrid.perception.transforms import new_transform_scratch, transform_points


def _pose(rng):
    """A rigid transform with a non-trivial rotation and a large translation --
    large enough that float64 rounding is actually exercised, not a toy identity."""
    a, b, c = rng.uniform(-np.pi, np.pi, 3)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    T = np.eye(4)
    T[:3, :3] = rz @ ry @ rx
    T[:3, 3] = rng.uniform(-5000.0, 5000.0, 3)
    return T


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cols", [3, 4])
@pytest.mark.parametrize("n", [1, 997, 120_000])
def test_transform_points_scratch_is_bit_identical(dtype, cols, n):
    """Same bytes, not merely close. float32 input is in the parametrisation on
    purpose: the loader yields float32, and the float32 -> float64 widening is
    where a careless buffered path would diverge."""
    rng = np.random.default_rng(n * cols + (dtype is np.float32))
    pts = (rng.normal(size=(n, cols)) * 60.0).astype(dtype)
    T = _pose(rng)
    scratch = new_transform_scratch(150_000)

    ref = transform_points(pts, T)
    got = transform_points(pts, T, scratch=scratch)

    assert got.shape == ref.shape == (n, 3)
    assert got.dtype == ref.dtype == np.float64
    assert np.array_equal(got, ref)


def test_transform_points_scratch_matches_across_many_calls():
    """Reusing the scratch must not leak one call's data into the next -- in
    particular the ones column, which is written once and never again."""
    rng = np.random.default_rng(7)
    scratch = new_transform_scratch(4096)
    for _ in range(25):
        n = int(rng.integers(1, 4096))
        pts = rng.normal(size=(n, 4)).astype(np.float32) * 40.0
        T = _pose(rng)
        assert np.array_equal(transform_points(pts, T, scratch=scratch),
                              transform_points(pts, T))


def test_transform_points_scratch_falls_back_above_capacity():
    """A scan larger than the scratch is still transformed correctly -- it just
    takes the allocating path for that call instead of raising."""
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(300, 3))
    T = _pose(rng)
    assert np.array_equal(transform_points(pts, T, scratch=new_transform_scratch(100)),
                          transform_points(pts, T))


def test_transform_points_scratch_allocates_nothing():
    """The point of the change. The allocating path costs ~10.9 MB per call at
    ~123,000 points; the scratch path must not allocate the working arrays."""
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(120_000, 4)).astype(np.float32)
    T = _pose(rng)
    scratch = new_transform_scratch(150_000)
    transform_points(pts, T, scratch=scratch)            # warm

    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    transform_points(pts, T, scratch=scratch)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak < 64 * 1024, f"scratch path allocated {peak:,} B on 120,000 points"


def test_transform_points_scratch_result_is_overwritten_by_the_next_call():
    """[!] The contract, pinned so nobody enables buffer reuse where results are
    kept. A caller holding the first result sees it change after the second call.
    This is WHY iter_pipeline defaults reuse_buffers to False."""
    rng = np.random.default_rng(5)
    scratch = new_transform_scratch(1000)
    first = transform_points(rng.normal(size=(500, 3)), _pose(rng), scratch=scratch)
    kept = first.copy()
    transform_points(rng.normal(size=(500, 3)), _pose(rng), scratch=scratch)
    assert not np.array_equal(first, kept)


# ------------------------------------------------------------------ pipeline --


def _have(seq):
    loader = pytest.importorskip("vrgrid.perception.loader")
    return loader.verify_sequence_exists(seq) and loader._velodyne_path(seq, 0).exists()


def test_iter_pipeline_reuse_buffers_matches_the_default_path():
    """Frame by frame, consumed one at a time the way the frame loop consumes
    them, `points_world` from the reused-buffer path equals the default path
    exactly."""
    if not _have("00"):
        pytest.skip("KITTI seq 00 not present -- set VRGRID_DATA_ROOT")
    from vrgrid.run.__main__ import iter_pipeline

    plain = iter_pipeline("00", max_frames=4, use_patchworkpp=False)
    reused = iter_pipeline("00", max_frames=4, use_patchworkpp=False, reuse_buffers=True)
    n = 0
    for a, b in zip(plain, reused):
        assert np.array_equal(a.points_world, b.points_world)
        n += 1
    assert n == 4


def test_iter_pipeline_default_does_not_share_buffers_between_frames():
    """The default must stay safe for callers that keep frames -- which the test
    suite itself does with list(iter_pipeline(...))."""
    if not _have("00"):
        pytest.skip("KITTI seq 00 not present -- set VRGRID_DATA_ROOT")
    from vrgrid.run.__main__ import iter_pipeline

    frames = list(iter_pipeline("00", max_frames=3, use_patchworkpp=False))
    assert not np.shares_memory(frames[0].points_world, frames[1].points_world)
    assert not np.shares_memory(frames[1].points_world, frames[2].points_world)
