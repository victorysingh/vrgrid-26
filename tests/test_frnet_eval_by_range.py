"""`scripts/frnet_eval_by_range.py`: bands, groups and the IoU tally, pinned without a model.

The per-band numbers are only meaningful if the bands partition the points, the groups
partition the classes, and the tally computes the same IoU as `frnet_eval.py`'s loop.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from frnet_eval_by_range import (  # noqa: E402
    BAND_NAMES, CLASSES, CLASS_TO_GROUP, GROUPS, Tally, band_of, to_group)


def test_every_class_is_in_exactly_one_group():
    names = [c for g in GROUPS.values() for c in g]
    assert sorted(names) == sorted(CLASSES)
    assert len(names) == len(set(names))
    assert CLASS_TO_GROUP.shape == (19,)


def test_bands_follow_ring_edges_and_partition_points():
    xy = np.array([[0.0, 0.0], [9.99, 0], [10.0, 0], [0, 24.9], [25.0, 0],
                   [30, 40], [70.0, 70.0], [100.0, 0], [150.0, 0]])
    b = band_of(xy)
    # 0, 9.99 -> 0-10 ; 10 -> 10-25 ; 24.9 -> 10-25 ; 25 -> 25-50 ; 50 (3-4-5) -> 50-100 ;
    # 98.99 -> 50-100 ; 100 -> >100 ; 150 -> >100
    assert list(b) == [0, 0, 1, 1, 2, 3, 3, 4, 4]
    assert b.max() < len(BAND_NAMES)


def test_group_collapse_and_ignore():
    lab = np.array([CLASSES.index("car"), CLASSES.index("truck"), CLASSES.index("road"),
                    CLASSES.index("pole"), 19, -1])
    g = to_group(lab)
    movable, drivable, static = (list(GROUPS).index(k) for k in
                                 ("movable object", "drivable terrain", "static obstacle"))
    assert list(g) == [movable, movable, drivable, static, -1, -1]


def test_tally_matches_frnet_eval_loop_definition():
    """Same accumulation as frnet_eval.py: per-class (p & g), (p | g), support over gt >= 0."""
    rng = np.random.default_rng(0)
    gt = rng.integers(-1, 19, 50_000)
    pred = np.where(rng.random(50_000) < 0.8, gt, rng.integers(0, 20, 50_000))
    t = Tally(19)
    t.add(pred, gt)
    ok = gt >= 0
    for c in range(19):
        p, g = pred[ok] == c, gt[ok] == c
        assert t.inter[c] == int((p & g).sum())
        assert t.union[c] == int((p | g).sum())
        assert t.support[c] == int(g.sum())
    assert t.correct == int((pred[ok] == gt[ok]).sum()) and t.total == int(ok.sum())
