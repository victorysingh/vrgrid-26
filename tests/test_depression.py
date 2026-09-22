"""R10 — the wide shallow depression that the kerb-tuned stencil cannot see.

**Designed 2026-09-23. No prior specification existed**: the roadmap named
"R10 wide-depression gradient" and nothing else — no commit, no file, no
threshold, no predicate. These tests pin a fresh design, not a rediscovered one.

The case: a **2 m wide, 15 cm deep** depression. Bits 1 and 2 are differenced
over `baseline_m = 0.50`, tuned so a 12 cm kerb stays passable at every cell
size. Over that stencil the depression's walls read ~0.15 m/m against
`tan(20°) = 0.364`, and no 4-neighbour step reaches `s_max = 15 cm`. It is too
WIDE to see, not too shallow — which is the whole point of a second baseline.
"""
import numpy as np
import pytest

from vrgrid.grid import traversability as trav

SIDE = 64
CELL_M = 0.05          # ring 0 of 5/10/20/40
BASELINE_M = 4.0       # R10's baseline
DIP_MIN_CM = 15.0      # the depth the case names
RIM_CM = 0             # flat ground datum, in centimetres


def _flat():
    return np.full(SIDE * SIDE, RIM_CM, dtype=np.int32)


def _depression(width_m=2.0, depth_cm=15.0, centre=(32, 32)):
    """A smooth parabolic bowl, `width_m` across, `depth_cm` deep at the centre.

    ⚑ Shape matters, and a flat-bottomed box will NOT do. A box drops the full
      depth across one cell, which is a 0.30 m/m wall per axis and 0.42 on the
      diagonal — over `tan(20°) = 0.364`, so bit 1 catches it and the feature is
      a cliff rather than the case R10 exists for. The first version of this
      helper made exactly that mistake and
      `test_the_existing_kerb_baseline_does_NOT_catch_it` caught it.

      A radially symmetric bowl has gradient magnitude |dz/dr| in every
      direction, peaking at 2·depth/R = 0.30 m/m at the rim — under the limit,
      and smoothed further by the 0.5 m stencil. Genuinely invisible to bits 1
      and 2, visible to curvature.
    """
    r0, c0 = centre
    R = width_m / 2.0 / CELL_M                      # radius in cells
    rr, cc = np.ogrid[:SIDE, :SIDE]
    d = np.sqrt((rr - r0) ** 2 + (cc - c0) ** 2) / R
    bowl = np.where(d < 1.0, -depth_cm * (1.0 - d ** 2), 0.0)
    return np.rint(RIM_CM + bowl).astype(np.int32).reshape(-1)


def _seen():
    return np.ones(SIDE * SIDE, dtype=np.uint16)


def _mask(ground, ring_index=0):
    return trav.depression_mask(ground, _seen(), SIDE, CELL_M, ring_index,
                                BASELINE_M, DIP_MIN_CM)


# --- the case the spec named --------------------------------------------------

def test_the_2m_15cm_depression_is_detected():
    """The headline case. If this fails, R10 does nothing."""
    assert _mask(_depression()).any(), "2 m x 15 cm depression not detected"


def test_it_is_detected_at_the_bottom_not_the_rim():
    """The bit marks the cells that ARE the hazard, not the ground beside it."""
    hit = _mask(_depression()).reshape(SIDE, SIDE)
    assert hit[32, 32], "centre of the depression should be flagged"
    assert not hit[5, 5], "flat ground far from it must stay clear"


def test_flat_ground_is_never_flagged():
    assert not _mask(_flat()).any()


def test_a_shallower_depression_is_not_flagged():
    """5 cm is a puddle, not a hazard: the threshold has to mean something."""
    assert not _mask(_depression(depth_cm=5.0)).any()


def test_the_boundary_case_fails_safe():
    """A dip exactly at the limit counts, unlike bit 2's strict `>`.

    Deliberate: a dip equal to the step limit is as untraversable as a step of
    that size, and §7.1's convention is that the uncertain case fails safe.
    """
    exact = _mask(_depression(depth_cm=DIP_MIN_CM))
    assert exact.any()


# --- why the existing bits miss it, pinned so the premise cannot rot ----------

def test_the_existing_kerb_baseline_does_NOT_catch_it():
    """The justification for R10 existing at all.

    If a future threshold change makes bits 1/2 catch this on their own, this
    test fails and R10 should be re-examined rather than kept out of habit.
    """
    ground = _depression()
    th = trav.load_thresholds()["traversability"]
    dzdx, dzdy = trav.gradient(ground, SIDE, CELL_M, baseline_m=th["baseline_m"])
    slope = np.hypot(dzdx, dzdy)
    tan_max = np.tan(np.deg2rad(th["theta_max_deg"]))
    step = trav.max_step_cm(ground, SIDE, CELL_M, baseline_m=th["baseline_m"])

    assert slope.max() <= tan_max, "slope bit would already catch it"
    assert step.max() <= th["s_max_m"] * 100.0, "step bit would already catch it"


# --- the range limit ----------------------------------------------------------

@pytest.mark.parametrize("ring_index", [2, 3])
def test_rings_2_and_3_are_not_checked(ring_index):
    """Physics, not tuning: beyond ~15 m consecutive beams are >1 m apart, so a
    depression is not in the data to be found. See the note in traversability.py."""
    assert not _mask(_depression(), ring_index=ring_index).any()


@pytest.mark.parametrize("ring_index", [0, 1])
def test_rings_0_and_1_are_checked(ring_index):
    assert _mask(_depression(), ring_index=ring_index).any()


# --- the unobserved-neighbour rule -------------------------------------------

def test_unobserved_neighbours_are_not_differenced():
    """An unobserved cell holds ground_height 0, a default and not a measurement.
    Differencing against it fabricates a bowl -- the same trap bits 1 and 2 avoid."""
    ground = _depression()
    obs = _seen()
    obs.reshape(SIDE, SIDE)[30:35, 30:35] = 0          # blind the depression
    assert not trav.depression_mask(ground, obs, SIDE, CELL_M, 0,
                                    BASELINE_M, DIP_MIN_CM)[32 * SIDE + 32]


# --- the frozen surface is untouched -----------------------------------------

def test_no_trav_depression_bit_exists_yet():
    """R10 is deliberately UNWIRED until the frozen-file change is signed off.

    `TRAV_DEPRESSION = 1 << 6` must be declared in `include/vrgrid/cell.py`,
    which is a whole-team change regardless of who owns `src/grid/`. When that
    lands, this test is the reminder to wire the bit into `bitfield()`.
    """
    from vrgrid import cell
    assert not hasattr(cell, "TRAV_DEPRESSION"), (
        "TRAV_DEPRESSION now exists — wire depression_mask into bitfield() and "
        "update this test")
    assert cell.CELL_BYTES == 12, "R10 must not grow the frozen cell struct"
