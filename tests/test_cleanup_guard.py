"""The cleanup guard's lookup table, pinned against the np.isin it replaced.

`MapEngine._cleanup` used `np.copyto(guard, np.isin(occupied, touched))`, which
built an 8.18 MB sorted temporary every frame. It now uses a persistent boolean
table indexed by slot (`reports/r-b-p99-tail-investigation.md` section 10). The
whole-map hash was shown identical on 60 frames of real seq 08 by
`reports/harnesses/cleanup_lut_equivalence.py`; these tests keep the two
properties that equivalence rests on from regressing.
"""
import importlib.util
import pathlib

import numpy as np
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.grid.fusion import occupancy_state
from vrgrid.run.engine import MapEngine

# Reuse the synthetic scene the engine tests already trust, rather than a second
# hand-built one that could disagree with it about geometry.
_spec = importlib.util.spec_from_file_location(
    "_engine_tests", pathlib.Path(__file__).with_name("test_engine.py"))
_engine_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine_tests)


class _CheckedEngine(MapEngine):
    """Records, per frame, the guard np.isin WOULD have produced, and the guard
    the lookup table actually produced -- computed from the same inputs, in the
    same order, that _cleanup itself uses."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.checked = 0
        self.mismatches = 0
        self.lut_left_dirty = 0

    def _cleanup(self, frame, touched, ego, counters):
        state = occupancy_state(self.handle.grid, self.thresholds)
        occupied = np.flatnonzero(state == OCC_OCCUPIED)[:self.max_candidates]
        expected = np.isin(occupied, touched)

        super()._cleanup(frame, touched, ego, counters)

        m = len(occupied)
        if m:
            self.checked += 1
            if not np.array_equal(self._has_return[:m], expected):
                self.mismatches += 1
        if self._touched_lut.any():
            self.lut_left_dirty += 1


def _run_checked(present_for=3, total=12, seed=0):
    rng = np.random.default_rng(seed)
    engine = _CheckedEngine(_engine_tests.load("5/10/20/40"), max_points=40_000,
                            max_candidates=80_000, ghost_removal=True)
    for frame, _ in _engine_tests._sequence(rng, present_for, total):
        engine.step(frame)
    return engine


def test_cleanup_guard_matches_np_isin_on_every_frame():
    """The lookup table must give exactly the guard np.isin gave, on every frame
    that had candidates -- including the frames after the car leaves, which are
    the ones where the guard decides what gets cleared."""
    engine = _run_checked()
    assert engine.checked >= 8, "the scene exercised too few cleanup frames to mean anything"
    assert engine.mismatches == 0


def test_cleanup_guard_lut_is_all_false_between_frames():
    """[!] The table is set and cleared inside one frame. A slot left True would
    protect that cell from clearing on EVERY later frame -- a ghost that can never
    be removed, with no error anywhere. So it must be clean after each step."""
    engine = _run_checked()
    assert engine.lut_left_dirty == 0
    assert not engine._touched_lut.any()


def test_cleanup_guard_lut_is_sized_to_the_slot_space():
    """One entry per allocated slot, the same space `touched` and `occupied`
    index -- the precondition that lets lut[touched] never alias another cell."""
    engine = _run_checked(total=2)
    assert engine._touched_lut.size == engine.occ_state.size
    assert engine._touched_lut.dtype == np.bool_
