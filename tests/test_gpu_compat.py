"""The device seam's missing primitives, checked against numpy. [Shrestha]

`src/gpu/compat.py` exists because `array_module()` alone does not make the
kernels read the same on both backends -- four primitives are missing or differ
on cupy. These tests are the contract: **every one runs on both backends and
asserts the device answer is bit-identical to numpy's**, which is the property
the port depends on and the determinism gate would otherwise discover late.

GPU cases skip where cupy or a device is absent, so CI stays green on a runner
with no card. That is a real gap in coverage and not a claim that they passed --
`docs/gpu-lane/06-DAY3-CUPY-FINDINGS.md` records the machine they were run on.
"""
import numpy as np
import pytest
from vrgrid.gpu import compat

I16MAX = np.iinfo(np.int16).max


def _cupy_or_skip():
    """Skip unless cupy imports AND a device can actually run a kernel.

    Three states have to be handled separately, and only the first is what
    `pytest.importorskip` covers on its own:

      1. cupy absent            -- ModuleNotFoundError, the CI runner's case.
      2. cupy installed, broken -- ImportError from a CUDA library mismatch.
         Common, and it must skip rather than fail: a broken optional backend
         is not a failure of this project's code.
      3. cupy imports, no card  -- the first kernel launch raises instead.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.skip(f"cupy unavailable: {exc}")
    try:
        cp.zeros(1) + 1
    except Exception as exc:                                      # noqa: BLE001
        pytest.skip(f"cupy present but no usable device: {type(exc).__name__}")
    return cp


@pytest.fixture(params=["numpy", "cupy"])
def xp(request):
    if request.param == "numpy":
        return np
    return _cupy_or_skip()


def _host(a):
    return a.get() if hasattr(a, "get") else a


def test_take_clip_matches_numpy_for_in_range_indices(xp):
    table = xp.asarray(np.array([10, 20, 30, 40], np.int64))
    idx = xp.asarray(np.array([0, 3, 1, 2, 0], np.int64))
    out = xp.zeros(5, dtype=xp.int64)
    compat.take_clip(xp, table, idx, out=out)
    assert np.array_equal(_host(out), np.array([10, 40, 20, 30, 10]))


def test_take_clip_out_of_range_diverges_and_that_is_documented():
    """The trap, pinned so nobody 'fixes' the docstring away.

    numpy's clip clamps and cupy's default wraps. Callers guarantee the
    in-range invariant; this test exists so that if a future cupy gains `mode`
    and the divergence disappears, someone notices deliberately.
    """
    cp = _cupy_or_skip()
    host = np.array([10, 20, 30, 40], np.int64)
    dev = cp.asarray(host)
    for bad in ([9], [-1]):
        n = compat.take_clip(np, host, np.array(bad))
        c = _host(compat.take_clip(cp, dev, cp.asarray(np.array(bad))))
        assert not np.array_equal(n, c), (
            f"cupy now agrees with numpy's clip at {bad}; compat.take_clip's "
            "warning is stale and the callers' clamp requirement may have changed")


def test_sort_inplace_gives_a_total_order(xp):
    a = xp.asarray(np.array([9, 2, 7, 1, 5], np.int64))
    compat.sort_inplace(xp, a)
    assert np.array_equal(_host(a), np.array([1, 2, 5, 7, 9]))


def test_subtract_where_only_touches_masked_lanes(xp):
    a = xp.asarray(np.array([10, 20, 30, 40], np.int64))
    b = xp.asarray(np.array([1, 2, 3, 4], np.int64))
    mask = xp.asarray(np.array([True, False, True, False]))
    out = xp.asarray(np.array([10, 20, 30, 40], np.int64))
    scratch = xp.zeros(4, dtype=xp.int64)
    compat.subtract_where(xp, a, b, out=out, where=mask, scratch=scratch)
    assert np.array_equal(_host(out), np.array([9, 20, 27, 40]))


def test_segment_min_matches_minimum_reduceat(xp):
    vals16 = np.array([5, 3, 9, 1, 7, 2, 8], np.int16)
    starts = np.array([0, 2, 5], np.intp)
    expected = np.minimum.reduceat(vals16, starts)

    v = xp.asarray(vals16)
    s = xp.asarray(starts)
    out = xp.full(3, I16MAX, dtype=xp.int16)
    compat.segment_min(xp, v, s, out=out, n_segments=3)
    assert np.array_equal(_host(out), expected)


def test_segment_min_is_bit_identical_to_numpy_at_frame_scale(xp):
    """The property the port rests on, at a shape a frame actually produces."""
    rng = np.random.default_rng(11)
    n, k = 120_000, 8_000
    counts = rng.multinomial(n, np.full(k, 1.0 / k))
    counts = np.maximum(counts, 1)
    n = int(counts.sum())
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.intp)
    vals16 = rng.integers(-3000, 3000, n).astype(np.int16)
    expected = np.minimum.reduceat(vals16, starts)

    out = xp.full(k, I16MAX, dtype=xp.int16)
    compat.segment_min(xp, xp.asarray(vals16), xp.asarray(starts),
                       out=out, n_segments=k)
    assert np.array_equal(_host(out), expected)


def test_segment_min_is_order_independent_on_device():
    """Min is associative and commutative, so the atomics may land in any
    order. Asserted rather than assumed -- this is the same argument the
    integer height sum rests on, applied to a different operator.
    """
    cp = _cupy_or_skip()
    rng = np.random.default_rng(5)
    n, k = 200_000, 4_096
    counts = np.maximum(rng.multinomial(n, np.full(k, 1.0 / k)), 1)
    n = int(counts.sum())
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.intp)
    vals16 = rng.integers(-3000, 3000, n).astype(np.int16)

    v, s = cp.asarray(vals16), cp.asarray(starts)
    first = None
    for _ in range(20):
        out = cp.full(k, I16MAX, dtype=cp.int16)
        compat.segment_min(cp, v, s, out=out, n_segments=k)
        cp.cuda.Stream.null.synchronize()
        h = out.get().tobytes()
        if first is None:
            first = h
        assert h == first, "segment_min is not reproducible on device"
    assert first == np.minimum.reduceat(vals16, starts).tobytes()
