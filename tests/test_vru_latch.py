"""R5 — the sticky safety-critical (VRU) latch.

A VRU is routinely a MINORITY of the returns in a road-dominated cell, especially
past 25 m where the semantic label carries the detection burden because geometry
no longer resolves a 30 cm feature (§1.2). Boyer–Moore correctly reports "this
cell is mostly road" and the planner correctly never hears about the person.
`FLAG_VRU_SEEN` answers the other question — "has anything safety-critical been
seen here recently?" — as a latch, not a vote.
"""
import numpy as np
import pytest
from vrgrid.cell import (
    CELL_BYTES,
    CELL_DTYPE,
    FLAG_BLIND,
    FLAG_VRU_AGE_MASK,
    FLAG_VRU_AGE_SHIFT,
    FLAG_VRU_SEEN,
    alloc_soa,
)
from vrgrid.grid.fusion import (
    VRU_CLASS_IDS,
    age_vru_latch,
    boyer_moore_update,
    has_vru_latch,
    mark_vru_seen,
    unpack_class,
)
from vrgrid.grid.schedule import load_thresholds

ROAD = 9          # learning id for road
PERSON = 5
N_SLOTS = 16
CELL = 0


def _soa():
    soa = alloc_soa(N_SLOTS)
    soa["flags"][:] = 0
    soa["semantic_class"][:] = 0
    return soa


def _observe(soa, class_ids, slots=None):
    """One fusion step's worth of class evidence for `slots`."""
    slots = np.arange(len(class_ids)) if slots is None else np.asarray(slots)
    ids = np.asarray(class_ids)
    soa["semantic_class"][slots] = boyer_moore_update(soa["semantic_class"][slots], ids)
    mark_vru_seen(soa, slots, ids)


def _decay_n():
    return int(load_thresholds()["traversability"]["vru_decay_frames"])


# --- the failure case this exists for ----------------------------------------

def test_the_latch_survives_majority_voting_when_a_vru_is_a_small_minority():
    """THE case. One person observation against nine road observations.

    Build the real failure, not something easier: the vote must genuinely
    unseat the VRU, otherwise this proves nothing.
    """
    soa = _soa()
    _observe(soa, [PERSON], slots=[CELL])            # the transient VRU
    for _ in range(9):                                # a road-dominated cell
        _observe(soa, [ROAD], slots=[CELL])

    candidate, _counter = unpack_class(soa["semantic_class"][CELL])
    assert candidate == ROAD, "test is not reproducing the failure: VRU still winning the vote"
    assert has_vru_latch(soa, [CELL])[0], "latch lost exactly where it is needed"


@pytest.mark.parametrize("vru", VRU_CLASS_IDS)
def test_every_safety_critical_class_latches(vru):
    soa = _soa()
    _observe(soa, [vru], slots=[CELL])
    assert has_vru_latch(soa, [CELL])[0]


def test_non_safety_classes_never_latch():
    soa = _soa()
    for cid in (ROAD, 8, 10, 11, 15):
        _observe(soa, [cid], slots=[CELL])
    assert not has_vru_latch(soa, [CELL])[0]


# --- majority voting must be completely unaffected ---------------------------

def test_majority_voting_is_unchanged_for_non_safety_classes():
    """R5 is an override for specific classes, never a replacement for fusion."""
    a, b = _soa(), _soa()
    seq = [ROAD, 8, ROAD, 11, ROAD, ROAD, 15]
    for cid in seq:
        _observe(a, [cid], slots=[CELL])
        # b: the same votes, with the latch helper never called
        b["semantic_class"][[CELL]] = boyer_moore_update(b["semantic_class"][[CELL]],
                                                         np.array([cid]))
    assert soa_class(a) == soa_class(b), "the latch perturbed the vote"


def test_majority_voting_is_unchanged_even_when_a_vru_latches():
    a, b = _soa(), _soa()
    seq = [PERSON, ROAD, ROAD, ROAD]
    for cid in seq:
        _observe(a, [cid], slots=[CELL])
        b["semantic_class"][[CELL]] = boyer_moore_update(b["semantic_class"][[CELL]],
                                                         np.array([cid]))
    assert soa_class(a) == soa_class(b)
    assert has_vru_latch(a, [CELL])[0] and not has_vru_latch(b, [CELL])[0]


def soa_class(soa):
    return tuple(int(x) for x in unpack_class(soa["semantic_class"][CELL]))


# --- decay, and the frames_since_seen ambiguity ------------------------------

def test_the_latch_decays_once_the_cell_is_tested_enough_times():
    soa = _soa()
    _observe(soa, [PERSON], slots=[CELL])
    for _ in range(_decay_n() + 1):
        age_vru_latch(soa, [CELL])
    assert not has_vru_latch(soa, [CELL])[0]


def test_the_latch_does_not_decay_early():
    soa = _soa()
    _observe(soa, [PERSON], slots=[CELL])
    for _ in range(_decay_n()):
        age_vru_latch(soa, [CELL])
    assert has_vru_latch(soa, [CELL])[0], "cleared before N frames"


def test_a_continuously_observed_road_cell_still_decays():
    """⚑ THE AMBIGUITY THE DESIGN DOC FLAGS AS THE FAILURE MODE.

    `frames_since_seen` means "frames since this CELL was observed", so a cell
    the vehicle keeps looking at has it pinned at 0 forever. If the decay had
    been written against that field, one early transient VRU would latch this
    cell permanently — and a busy near-field road cell is exactly where a VRU is
    most likely to have been a transient minority.

    Here the cell is observed as road EVERY frame, so `frames_since_seen` never
    rises, and the latch must still clear.
    """
    soa = _soa()
    _observe(soa, [PERSON], slots=[CELL])            # one early transient VRU
    for _ in range(_decay_n() + 1):
        _observe(soa, [ROAD], slots=[CELL])          # continuously observed
        soa["frames_since_seen"][CELL] = 0           # ... so this stays pinned
        age_vru_latch(soa, [CELL])

    assert soa["frames_since_seen"][CELL] == 0, "precondition: cell is never unseen"
    assert not has_vru_latch(soa, [CELL])[0], (
        "latched forever on a continuously-observed cell — the decay is keyed to "
        "cell observation, not to VRU observation")


def test_a_new_sighting_resets_the_age():
    """Decay measures frames since a VRU was seen HERE, so a sighting restarts it."""
    soa = _soa()
    _observe(soa, [PERSON], slots=[CELL])
    for _ in range(_decay_n()):
        age_vru_latch(soa, [CELL])
    _observe(soa, [PERSON], slots=[CELL])            # seen again
    assert (soa["flags"][CELL] & FLAG_VRU_AGE_MASK) >> FLAG_VRU_AGE_SHIFT == 0
    for _ in range(_decay_n()):
        age_vru_latch(soa, [CELL])
    assert has_vru_latch(soa, [CELL])[0], "reset did not restart the countdown"


def test_cells_not_tested_this_frame_do_not_age():
    """Decay happens where the visibility pass TESTED, not everywhere."""
    soa = _soa()
    _observe(soa, [PERSON, PERSON], slots=[CELL, 1])
    for _ in range(_decay_n() + 1):
        age_vru_latch(soa, [CELL])                   # only cell 0 is tested
    assert not has_vru_latch(soa, [CELL])[0]
    assert has_vru_latch(soa, [1])[0], "an untested cell aged anyway"


# --- the frozen surface -------------------------------------------------------

def test_the_struct_did_not_grow():
    assert CELL_BYTES == 12
    assert CELL_DTYPE.itemsize == 12


def test_the_latch_does_not_disturb_other_flag_bits():
    soa = _soa()
    soa["flags"][CELL] = FLAG_BLIND
    _observe(soa, [PERSON], slots=[CELL])
    assert soa["flags"][CELL] & FLAG_BLIND, "FLAG_BLIND clobbered"
    assert soa["flags"][CELL] & FLAG_VRU_SEEN
