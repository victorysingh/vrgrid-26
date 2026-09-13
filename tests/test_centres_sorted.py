"""`MapEngine._centres(..., sorted_slots=True)`, pinned against the shipped path.

The cleanup's slot -> centre conversion cost ~11 ms and ~7.4 MB per frame, as much
as the visibility pass it feeds. The sorted fast path removes that, and is only
acceptable if every coordinate it produces is bit-identical to the shipped loop.
"""
import importlib.util
import pathlib
import tracemalloc

import numpy as np
import pytest
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.grid.fusion import occupancy_state

_spec = importlib.util.spec_from_file_location(
    "_engine_tests", pathlib.Path(__file__).with_name("test_engine.py"))
_engine_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine_tests)


def _engine():
    engine, _, _ = _engine_tests._run(ghost_removal=True)
    return engine


def _both(engine, slots, ego):
    n = len(slots)
    a = [np.full(max(n, 1), -7.0) for _ in range(3)]
    b = [np.full(max(n, 1), -7.0) for _ in range(3)]
    ra = engine._centres(slots, ego, *a)
    rb = engine._centres(slots, ego, *b, sorted_slots=True)
    return ra, rb


@pytest.mark.parametrize("ego", [np.array([12.5, -3.25, 1.73]), np.zeros(2)])
def test_sorted_centres_match_shipped_on_the_engines_occupied_set(ego):
    """The exact input the cleanup passes: the occupied set of a real engine run,
    for both ego forms (3-vector = vehicle frame, 2-vector = world z)."""
    engine = _engine()
    state = occupancy_state(engine.handle.grid, engine.thresholds)
    slots = np.flatnonzero(state == OCC_OCCUPIED)
    assert len(slots) > 100, "scene produced too few occupied cells to test"
    ra, rb = _both(engine, slots, ego)
    for x, y in zip(ra, rb):
        assert np.array_equal(x, y)


def test_sorted_centres_match_shipped_across_every_ring_and_shifted_windows():
    """Adversarial: sorted random slots spanning ALL rings, with every ring's window
    moved to large positive and negative origins, where the mod arithmetic in the
    shipped path is actually exercised."""
    engine = _engine()
    total = int(engine._ring_bounds[-1])
    rng = np.random.default_rng(42)
    saved = [(b.x0, b.y0) for b in engine.buffers]
    try:
        for trial in range(6):
            for b in engine.buffers:
                b.x0 = int(rng.integers(-40_000, 40_000))
                b.y0 = int(rng.integers(-40_000, 40_000))
            n = int(rng.integers(1, min(60_000, len(engine._cand_slots))))
            slots = np.sort(rng.choice(total, size=n, replace=False)).astype(np.int64)
            ego = rng.uniform(-3000, 3000, 3)
            ra, rb = _both(engine, slots, ego)
            for x, y in zip(ra, rb):
                assert np.array_equal(x, y), f"trial {trial}"
    finally:
        for b, (x0, y0) in zip(engine.buffers, saved):
            b.x0, b.y0 = x0, y0


def _peak(engine, n, sorted_slots):
    total = int(engine._ring_bounds[-1])
    slots = np.sort(np.random.default_rng(1).choice(total, size=n, replace=False)).astype(np.int64)
    out = [np.zeros(n) for _ in range(3)]
    engine._centres(slots, np.zeros(3), *out, sorted_slots=sorted_slots)   # warm
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    engine._centres(slots, np.zeros(3), *out, sorted_slots=sorted_slots)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return peak


def test_sorted_centres_allocation_does_not_grow_with_slot_count():
    """What is removed is the PER-SLOT allocation, not every byte.

    ⚑ The sorted path's peak is not zero: it saturates at ~68 KB, which is numpy's
      ufunc casting buffer (`np.getbufsize()` = 8192 elements x 8 B) used by the
      mixed int/float steps. Measured: 5,768 B at 500 slots, then 68,280 B flat
      at 20,000, 50,000 and 80,000, while the shipped path grows linearly to
      2,151,652 B. So the assertion is the property that matters -- a constant
      bound that does not scale with n -- not an arbitrary byte count.
    """
    engine = _engine()
    cap = len(engine._cand_slots)
    lo, hi = cap // 4, cap
    fast_lo, fast_hi = _peak(engine, lo, True), _peak(engine, hi, True)
    shipped_hi = _peak(engine, hi, False)
    assert fast_hi < 128 * 1024, f"sorted path allocated {fast_hi:,} B on {hi:,} slots"
    assert fast_hi - fast_lo < 8 * 1024, (
        f"sorted path grew {fast_lo:,} -> {fast_hi:,} B from {lo:,} to {hi:,} slots")
    assert shipped_hi > 10 * fast_hi, "control: the shipped path should allocate per slot"


def test_sorted_centres_fall_back_above_capacity():
    """More slots than the scratch holds: the shipped path runs instead."""
    engine = _engine()
    cap = len(engine._cand_slots)
    total = int(engine._ring_bounds[-1])
    n = min(cap + 1000, total)
    if n <= cap:
        pytest.skip("slot space smaller than the candidate cap")
    slots = np.arange(n, dtype=np.int64)
    ra, rb = _both(engine, slots, np.zeros(3))
    for x, y in zip(ra, rb):
        assert np.array_equal(x, y)


def test_sorted_slots_is_opt_in():
    """[!] Default stays the shipped path: map-view callers pass slots of unknown
    order, and the fast path silently assumes ascending slots."""
    import inspect
    sig = inspect.signature(type(_engine())._centres)
    assert sig.parameters["sorted_slots"].default is False
