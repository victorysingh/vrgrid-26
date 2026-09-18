"""Stage attrition: every return ends in exactly one pipeline stage. [Shrestha]

Rule 3 of `vrgrid-recommended-changes.pdf` §6.6 -- report where returns leave
the pipeline, not only what the survivors scored. These tests build a frame
that exercises every terminal stage on purpose (a point cap, a return past the
map, ground far below the height band), check the counts partition the sweep
and match the per-point codes, and -- with a card -- that the CPU and CUDA
engines report the same thing return for return.
"""

import numpy as np
import pytest
import test_engine as te  # a failed import must fail, not skip
from vrgrid.gpu import attrition as A
from vrgrid.gpu import device as dev
from vrgrid.grid.schedule import load
from vrgrid.run.engine import MapEngine


def _frame(extra_far=40, extra_deep=30):
    rng = np.random.default_rng(3)
    frame, _ = next(iter(te._sequence(rng, 1, 1)))
    pts = frame.points_sensor[:, :3]
    far = np.column_stack([np.full(extra_far, 150.0), rng.uniform(-2, 2, extra_far),
                           np.full(extra_far, -1.7)])            # past the 100 m window
    deep = np.column_stack([rng.uniform(4, 6, extra_deep), rng.uniform(-1, 1, extra_deep),
                            np.full(extra_deep, -20.0)])         # ground 18 m below the band
    points = np.vstack([pts, far, deep])
    ground = np.concatenate([frame.ground, np.ones(extra_far, bool), np.ones(extra_deep, bool)])
    f = te._frame(0, points, ground)
    f.moving = np.zeros(len(points), bool)
    f.moving[:25] = True
    return f


def _engine(device, max_points):
    return MapEngine(load("5/10/20/40"), max_points=max_points, max_candidates=80_000,
                     device=device, attrition=True)


def test_every_return_ends_in_exactly_one_stage_and_every_stage_fires():
    f = _frame()
    n_all = len(f.points_sensor)
    eng = _engine("cpu", max_points=n_all - 10)          # the last 10 are capped
    c = eng.step(f).attrition
    assert sum(c[k] for k in A.NAMES) == c["points"] == n_all
    assert c["capped"] == 10
    assert c["outside_map"] >= 40                     # every far return is within the cap
    assert c["ground_out_of_band"] >= 1
    assert c["ground_fused"] > 0 and c["nonground"] > 0
    assert c["moving"] == 25
    assert 0 < c["projected"] <= n_all

    codes = eng.attrition_codes()
    assert codes.shape == (n_all,) and codes.dtype == np.uint8
    for code, name in enumerate(A.NAMES):
        assert int((codes == code).sum()) == c[name], name


def test_attrition_is_off_by_default():
    f = _frame()
    eng = MapEngine(load("5/10/20/40"), max_points=200_000, max_candidates=80_000)
    assert eng.step(f).attrition is None
    with pytest.raises(RuntimeError, match="attrition=True"):
        eng.attrition_codes()


@pytest.mark.skipif(not dev.cuda_available(), reason="no working CUDA device")
def test_attrition_is_identical_on_both_devices():
    f = _frame()
    n_all = len(f.points_sensor)
    cpu, gpu = _engine("cpu", n_all - 10), _engine("cuda", n_all - 10)
    assert cpu.step(f).attrition == gpu.step(f).attrition
    assert np.array_equal(cpu.attrition_codes(), gpu.attrition_codes())
