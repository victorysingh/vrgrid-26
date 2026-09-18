"""The fast gate path is the per-cell reference, decision for decision. [Shrestha]

`gate.apply` replaced a Python per-cell loop that cost ~50 ms a frame on real
data. Everything the gate decides -- which blocks are released, which requests
are acquired, evicted or refused, what each block is filled with, which cells
are flagged -- must be unchanged, because every refinement-pool number and the
§8 evaluation read it. So both implementations drive the same sequence on two
identical maps and are compared after EVERY frame, and once with a pool small
enough that eviction and refusal happen on almost every request.
"""

import copy

import numpy as np
import pytest
from vrgrid.eval import harness as H
from vrgrid.eval.synthetic import read_sequence, write_sequence
from vrgrid.grid import gate
from vrgrid.grid.lattice import migrate_ring, migrate_ring_many
from vrgrid.grid.schedule import load, load_thresholds


@pytest.fixture(scope="module")
def sequence(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate")
    write_sequence(root, "99", n_frames=14)
    return [(p, lab, np.ones(len(p), bool), T) for p, lab, T in read_sequence(root, "99")]


def _snapshot(gm):
    pool = gm.pool
    return (pool.owner_ring.copy(), pool.owner_slot.copy(), pool.levels.copy(),
            pool.score.copy(), {k: v.copy() for k, v in pool.cells.items()},
            gm.soa["flags"].copy())


def _run(sequence, apply_fn, thresholds):
    gm = H.build_gridmap(load("5/10/20/40"), thresholds=thresholds)
    log = []
    real = gate.apply

    def wrapped(g, slots, **kw):
        out = apply_fn(g, slots, **kw)
        log.append((out, _snapshot(g)))
        return out

    H.gate.apply = wrapped
    try:
        H.run_sequence(gm, sequence)
    finally:
        H.gate.apply = real
    return log


@pytest.mark.parametrize("blocks", [512, 8])
def test_apply_matches_the_reference(sequence, blocks):
    th = copy.deepcopy(load_thresholds())
    th.setdefault("refinement_pool", {})["blocks"] = blocks
    fast = _run(sequence, gate.apply, th)
    ref = _run(sequence, gate.apply_reference, th)
    assert len(fast) == len(ref) > 0
    for frame, ((c_f, s_f), (c_r, s_r)) in enumerate(zip(fast, ref)):
        assert c_f == c_r, f"frame {frame}: counts {c_f} vs {c_r}"
        for name, a, b in zip(("owner_ring", "owner_slot", "levels", "score"), s_f[:4], s_r[:4]):
            assert np.array_equal(a, b), f"frame {frame}: pool.{name} differs"
        for field, arr in s_f[4].items():
            assert np.array_equal(arr, s_r[4][field]), f"frame {frame}: pool cells.{field}"
        assert np.array_equal(s_f[5], s_r[5]), f"frame {frame}: map flags differ"
    totals = {k: sum(c[k] for c, _ in ref) for k in ("fired", "acquired", "refused", "released")}
    assert totals["fired"] > 0 and totals["acquired"] > 0
    if blocks == 8:
        assert totals["refused"] > 0, "the small pool must exercise refusal"


def test_migrate_ring_many_matches_the_scalar():
    s = load("5/10/20/40")
    rng = np.random.default_rng(11)
    x = rng.uniform(-110, 110, 3000)
    y = rng.uniform(-110, 110, 3000)
    cur = rng.integers(-1, 4, 3000)
    for speed, vehicle, yaw in ((0.0, (0.0, 0.0), 0.0), (12.0, (37.03, -11.12), 0.8)):
        many = migrate_ring_many(x, y, s, cur, speed, vehicle, yaw)
        one = [migrate_ring(float(a), float(b), s, int(c), speed, vehicle, yaw)
               for a, b, c in zip(x, y, cur)]
        assert np.array_equal(many, np.array(one))
