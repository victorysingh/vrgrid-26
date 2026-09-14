"""The label rule behind `scripts/plan_regret_frnet_delta.py`, pinned.

The delta that script reports is only a segmentation effect if the FRNet arm
changes CLASSES and nothing else. These tests pin that rule on its own, without
a model or data: which points keep their raw id, how FRNet's ignore slot maps,
and that an oracle prediction reproduces exactly what the harness gives the
ground-truth arm.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plan_regret_frnet_delta import compose_labels  # noqa: E402
from vrgrid.eval.harness import learning_ids  # noqa: E402
from vrgrid.grid.fusion import CLASS_UNLABELLED  # noqa: E402
from vrgrid.grid.transient import separate  # noqa: E402
from vrgrid.perception.semantics import semantic_labels  # noqa: E402

# Raw SemanticKITTI ids: 0 unlabeled, 10 car, 40 road, 70 vegetation,
# 252 moving-car, 254 moving-person; one with instance bits set.
RAW = np.array([0, 10, 40, 70, 252, 254, (7 << 16) | 10, 40], dtype=np.uint32)


def test_motion_gt_keeps_raw_moving_ids_and_relabels_only_static_points():
    pred = np.array([8, 0, 8, 14, 0, 5, 1, 9])
    out = compose_labels(RAW, pred, motion="gt")
    moving = separate(RAW)[1]
    assert moving.sum() == 2
    assert np.array_equal(out[moving], RAW[moving])
    assert np.array_equal(out[~moving], pred[~moving])
    assert np.array_equal(separate(out)[1], moving), "the same points must be routed out"


def test_motion_none_gives_every_point_the_prediction():
    pred = np.array([8, 0, 8, 14, 0, 5, 1, 9])
    out = compose_labels(RAW, pred, motion="none")
    assert np.array_equal(out, pred)
    assert not separate(out)[1].any(), "no motion information survives, by design"


def test_frnet_ignore_slot_and_out_of_range_map_to_unlabelled():
    pred = np.array([19, -1, 25, 18, 0, 0, 0, 0])
    out = compose_labels(RAW, pred, motion="none")
    assert list(out[:4]) == [CLASS_UNLABELLED, CLASS_UNLABELLED, CLASS_UNLABELLED, 18]


def test_oracle_prediction_reproduces_what_the_ground_truth_arm_scatters():
    """[!] The control, reduced to one scan: ground truth's own classes through
    compose_labels must give run_sequence exactly the static class ids it derives
    from the raw labels itself."""
    out = compose_labels(RAW, semantic_labels(RAW), motion="gt")
    static_gt, _ = separate(RAW)
    static_or, _ = separate(out)
    assert np.array_equal(static_gt, static_or)
    assert np.array_equal(learning_ids(out[static_or]), learning_ids(RAW[static_gt]))


def test_misaligned_predictions_are_refused():
    with pytest.raises(ValueError):
        compose_labels(RAW, np.zeros(RAW.size - 1, dtype=np.int64))
    with pytest.raises(ValueError):
        compose_labels(RAW, np.zeros(RAW.size, dtype=np.int64), motion="sometimes")
