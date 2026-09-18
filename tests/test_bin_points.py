"""Point -> slot binning, in one place at last. Math §2.1, §6.1. [Aakash]

Gate 3, item 2. The stage between perception and the grid had no owner: it was
composed by hand in `fusion.scatter`, `grid/transient.py`, `run/engine.py` and
`scripts/timing_table.py` -- four spellings of one step, in three directories.

These tests are almost entirely equivalence tests, and deliberately so. The
reference implementations (`ring_of`, `i_ring`, `RingBuffer.flat_slot`) are
what every other test in the suite is written against and what the partition
test proves things about; the frame-path twin is only allowed to exist if it
is bit-identical to them. Anything else is a second lattice, which is exactly
what `lattice.py`'s header exists to forbid.
"""

import numpy as np
import pytest
from vrgrid.gpu.shift import RingBuffer
from vrgrid.grid.lattice import (
    OUTSIDE,
    bin_points,
    d_aniso,
    d_aniso_into,
    i_ring,
    new_bin_scratch,
    ring_of,
    ring_of_into,
)
from vrgrid.grid.schedule import load

SCHEDULES = ["5/10/20/40", "5/10/50"]
SPEEDS = [0.0, 5.0, 15.0, 30.0]
# (vehicle world position, heading). Off-lattice positions, and headings that
# break the symmetries the block bound has: +x, the diagonal, backwards, and
# an angle with no special value.
HEADINGS = [((37.0, -11.0), 0.0), ((37.03, -11.12), 0.7853981633974483),
            ((-4.21, 250.4), 3.141592653589793), ((0.0, 0.0), -2.3)]


def _sweep(n=20_000, seed=0, reach=140.0):
    """A cloud that deliberately overruns the map: ring boundaries, the rear
    floor's 50 m edge, and points past the last ring all have to be covered,
    because those are the three places the two paths could diverge."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-reach, reach, n)
    y = rng.uniform(-reach, reach, n)
    return x, y


def _windows(handle, sched, wx=0.0, wy=0.0):
    """One toroidal window per ring, centred on the vehicle's WORLD position.

    This is what `recenter()` maintains and what `run/engine.py` builds. It
    matters for the fixture and not only for realism: a ring's window is only
    `side` cells across -- ring 0 is 400 cells of 5 cm, so 20 m -- and a window
    left at the lattice origin while the world coordinates sit 37 m away puts
    every ring-0 point out of view. Both paths return -1 for those and agree
    perfectly, which is a green equivalence test over an empty map.
    """
    bufs = []
    for r in handle.rings:
        k = round(r.cell_m / sched.base_cell_m)
        i = int(wx // sched.base_cell_m) // k
        j = int(wy // sched.base_cell_m) // k
        bufs.append(RingBuffer(side=r.side, offset=r.offset,
                               x0=i - r.side // 2, y0=j - r.side // 2))
    return bufs


def _ring_layouts(schedule):
    """`schedule.rings` carries cell_m and half_width_m but not the storage
    geometry, which lives in the allocation."""
    from vrgrid.gpu.allocators import allocate
    from vrgrid.grid.schedule import load_thresholds

    return allocate(schedule, load_thresholds(), commit_pages=False)


# --- the twins agree with the references ------------------------------------


@pytest.mark.parametrize("name", SCHEDULES)
@pytest.mark.parametrize("speed", SPEEDS)
def test_d_aniso_into_matches_d_aniso(name, speed):
    """Bit-identical, not close. The value is compared against a ring radius,
    so a point exactly on a boundary must land in the same ring on both paths
    or the partition test is measuring two different maps."""
    sched = load(name)
    x, y = _sweep()
    scratch = new_bin_scratch(len(x), sched)

    got = d_aniso_into(x, y, sched, speed,
                       scratch["f1"][:len(x)], scratch["f0"][:len(x)])
    assert np.array_equal(got, d_aniso(x, y, sched, speed))


@pytest.mark.parametrize("name", SCHEDULES)
@pytest.mark.parametrize("speed", SPEEDS)
def test_ring_of_into_matches_ring_of(name, speed):
    """Every rule the reference applies -- containment, the rear floor, the
    keep-what-fits clamp, OUTSIDE past the last ring -- reproduced exactly.

    ⚑ `5/10/50` is in the parametrisation on purpose: its ratios are 2 and 5,
      so it is the schedule that catches anything assuming a power of two, and
      `5/10/20/40` alone cannot.
    """
    sched = load(name)
    x, y = _sweep()
    scratch = new_bin_scratch(len(x), sched)

    got = ring_of_into(x, y, sched, speed, scratch["level"][:len(x)], scratch)
    want = ring_of(x, y, sched, speed)
    assert np.array_equal(got, want)

    # and away from the origin, at a heading, against shifted windows -- the
    # world coordinates the frame path takes versus the vehicle-relative ones
    # the reference takes, which is where a rounding difference would show
    handle = _ring_layouts(sched)
    for (vx, vy), yaw in HEADINGS:
        bufs = _windows(handle, sched, vx, vy)
        got = ring_of_into(x + vx, y + vy, sched, speed, scratch["level"][:len(x)],
                           scratch, bufs, (vx, vy), yaw)
        assert np.array_equal(got, ring_of(x, y, sched, speed, (vx, vy), yaw, bufs)), yaw
    got = ring_of_into(x, y, sched, speed, scratch["level"][:len(x)], scratch)

    assert np.array_equal(got, want)
    assert got.dtype == want.dtype
    assert (got == OUTSIDE).any(), "the sweep never left the map, so this is weak"
    assert (got >= 0).any(), "the sweep never entered the map either"


def test_ring_of_into_covers_every_ring_and_outside():
    """A guard on the fixture rather than the code: if a later edit to `_sweep`
    stopped producing points in ring 3, the equivalence tests above would still
    pass and would be testing less than they claim."""
    sched = load("5/10/20/40")
    x, y = _sweep()
    scratch = new_bin_scratch(len(x), sched)
    got = ring_of_into(x, y, sched, 0.0, scratch["level"][:len(x)], scratch)

    seen = set(np.unique(got).tolist())
    assert {OUTSIDE, 0, 1, 2, 3} <= seen, f"only reached {sorted(seen)}"


# --- the composition ---------------------------------------------------------


def _reference_bin(xv, yv, xw, yw, sched, buffers, speed=0.0, vehicle=(0.0, 0.0),
                   yaw=0.0):
    """The hand-rolled composition, written out one final time. This is the
    spelling `fusion.scatter` and `run/engine.py` used, and what `bin_points`
    has to reproduce before those call sites can be deleted."""
    level = ring_of(xv, yv, sched, speed, vehicle, yaw, buffers)
    idx = np.full(len(xv), -1, dtype=np.int64)
    for layout, buf in zip(sched.rings, buffers):
        sel = level == layout.ring
        if not sel.any():
            continue
        k = round(layout.cell_m / sched.base_cell_m)
        idx[sel] = buf.flat_slot(i_ring(xw[sel], sched.base_cell_m, k),
                                 i_ring(yw[sel], sched.base_cell_m, k))
    return idx


@pytest.mark.parametrize("name", SCHEDULES)
@pytest.mark.parametrize("speed", SPEEDS)
def test_bin_points_matches_the_reference_path(name, speed):
    """The whole stage, against the composition it replaces. Bit-identical,
    over both frozen schedules -- `5/10/50` has ratios 2 and 5, so it is what
    catches anything assuming a power of two."""
    sched = load(name)
    handle = _ring_layouts(sched)
    xv, yv = _sweep()
    scratch = new_bin_scratch(len(xv), sched)
    out = np.zeros(len(xv), np.int64)

    for (wx, wy), yaw in HEADINGS:        # world != vehicle, by odd offsets
        buffers = _windows(handle, sched, wx, wy)
        xw, yw = xv + wx, yv + wy
        got = bin_points(xw, yw, sched, buffers, out, scratch, speed, (wx, wy), yaw)
        want = _reference_bin(xv, yv, xw, yw, sched, buffers, speed, (wx, wy), yaw)

        assert np.array_equal(got, want), yaw
        binned = int((got >= 0).sum())
        assert binned > 0.3 * len(xv), (
            f"only {binned} of {len(xv)} points binned -- the fixture is testing "
            "an empty map, so the equivalence above proves little"
        )


@pytest.mark.parametrize("name", SCHEDULES)
def test_every_ring_is_actually_exercised(name):
    """A guard on the fixture, not the code. If a later edit stopped producing
    points in ring 3, or stopped centring the windows, the equivalence tests
    would still pass over a map nothing lands in."""
    sched = load(name)
    handle = _ring_layouts(sched)
    buffers = _windows(handle, sched, 37.0, -11.0)

    xv, yv = _sweep()
    xw, yw = xv + 37.0, yv - 11.0
    scratch = new_bin_scratch(len(xv), sched)
    out = np.zeros(len(xv), np.int64)
    got = bin_points(xw, yw, sched, buffers, out, scratch, 0.0, (37.0, -11.0))

    for layout in handle.rings:          # storage geometry, not the schedule
        lo = layout.offset
        hi = layout.offset + layout.side * layout.side
        assert ((got >= lo) & (got < hi)).any(), f"ring {layout.ring} got nothing"
    assert (got < 0).any(), "nothing fell outside, so the -1 path is untested"


def test_the_vehicle_position_is_not_optional():
    """⚑ The failure the old two-frame signature guarded against, in its new
    shape. `bin_points` takes world points and the vehicle position; leave the
    position at its default once the vehicle has driven 400 m and nothing
    raises and nothing is dropped -- the windows still hold every point -- but
    every block reads as 400 m away and lands in the coarsest ring. A map
    that is all 40 cm cells looks fine from a distance.
    """
    sched = load("5/10/20/40")
    handle = _ring_layouts(sched)
    buffers = _windows(handle, sched, 400.0, 0.0)

    xv, yv = _sweep(n=5000)
    xw, yw = xv + 400.0, yv               # the vehicle has driven 400 m
    scratch = new_bin_scratch(len(xv), sched)
    out = np.zeros(len(xv), np.int64)
    coarsest = handle.rings[-1].offset

    right = bin_points(xw, yw, sched, buffers, out, scratch, 0.0, (400.0, 0.0)).copy()
    wrong = bin_points(xw, yw, sched, buffers, out, scratch).copy()

    assert ((right >= 0) & (right < coarsest)).any(), "the correct call used no fine ring"
    assert np.array_equal(right >= 0, wrong >= 0), "the windows alone decide what is kept"
    assert not ((wrong >= 0) & (wrong < coarsest)).any(), (
        "without the vehicle position every block should read as far away -- "
        "if it does not, this test proves nothing"
    )


# --- the invariant this was built for ---------------------------------------


def test_bin_points_allocates_nothing_in_the_loop():
    """CLAUDE.md: no allocation inside the frame loop. `ring_of` allocated
    6.96 MB per 120,000-point sweep, which is the largest per-frame allocation
    in the system and more than twice what the scatter scratch cost before it
    was preallocated.

    Measured as peak transient bytes over one call, warm -- the first call
    faults in numpy's own machinery, which is startup, not the loop.
    """
    import tracemalloc

    sched = load("5/10/20/40")
    handle = _ring_layouts(sched)
    buffers = _windows(handle, sched, 37.0, -11.0)

    xv, yv = _sweep(n=120_000)
    xw, yw = xv + 37.0, yv - 11.0
    scratch = new_bin_scratch(len(xv), sched)
    out = np.zeros(len(xv), np.int64)

    for _ in range(3):                       # warm
        bin_points(xw, yw, sched, buffers, out, scratch, 0.0, (37.0, -11.0))

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    bin_points(xw, yw, sched, buffers, out, scratch, 0.0, (37.0, -11.0))
    peak = tracemalloc.get_traced_memory()[1] - before
    tracemalloc.stop()

    # The reference path allocates ~7 MB here. One full-length float64
    # temporary at this width is 0.96 MB, so the bound is set well below a
    # single array: anything that scales with the sweep cannot hide under it.
    assert peak < 16_000, f"{peak / 1e6:.3f} MB allocated inside bin_points"


def test_binning_does_not_allocate_more_as_the_sweep_grows():
    """The bound above is a constant, so it would also pass if allocation grew
    slowly. This is the shape test: quadrupling the points must not move it.

    That distinction is the whole point -- `ring_of`'s 6.96 MB was invisible
    precisely because nobody had asked how it scaled.
    """
    import tracemalloc

    sched = load("5/10/20/40")
    handle = _ring_layouts(sched)
    buffers = _windows(handle, sched, 37.0, -11.0)
    scratch = new_bin_scratch(120_000, sched)
    out = np.zeros(120_000, np.int64)

    peaks = []
    for n in (30_000, 120_000):
        xv, yv = _sweep(n=n)
        xw, yw = xv + 37.0, yv - 11.0
        for _ in range(3):
            bin_points(xw, yw, sched, buffers, out, scratch, 0.0, (37.0, -11.0))
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        bin_points(xw, yw, sched, buffers, out, scratch, 0.0, (37.0, -11.0))
        peaks.append(tracemalloc.get_traced_memory()[1] - before)
        tracemalloc.stop()

    assert peaks[1] <= peaks[0] + 8_000, (
        f"4x the points allocated {peaks[1] - peaks[0]} more bytes -- "
        "something on this path still scales with the sweep"
    )


def test_the_scratch_refuses_a_sweep_it_was_not_sized_for():
    """Growing the buffers on demand would allocate in the frame loop, which
    is the defect this whole function exists to remove. Refuse instead, and
    name the config key that sets the cap."""
    sched = load("5/10/20/40")
    handle = _ring_layouts(sched)
    buffers = _windows(handle, sched)

    scratch = new_bin_scratch(100, sched)
    xv, yv = _sweep(n=500)
    out = np.zeros(500, np.int64)

    with pytest.raises(ValueError, match="max_points_per_frame"):
        bin_points(xv, yv, sched, buffers, out, scratch)


def test_a_non_integer_ring_ratio_is_refused():
    """`_bin_geometry` bakes k per ring at startup. A non-integer ratio would
    silently round there, which is the float-lattice bug this module's header
    forbids -- so it raises instead. `schedule.validate()` should have caught
    it first; this is the second line of defence, not the first."""
    from types import SimpleNamespace

    from vrgrid.grid.lattice import _bin_geometry

    bad = SimpleNamespace(
        base_cell_m=0.05,
        rings=[SimpleNamespace(ring=0, cell_m=0.05, half_width_m=10.0),
               SimpleNamespace(ring=1, cell_m=0.075, half_width_m=25.0)],
        anisotropy=SimpleNamespace(rear_floor_cell_m=0.20),
    )
    with pytest.raises(ValueError, match="integer multiple"):
        _bin_geometry(bad)
