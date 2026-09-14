"""The opt-in DL mode: `iter_pipeline(semantics_source="frnet", frnet=...)`, pinned.

SIH26053 asks for a deep-learning pipeline. JP's decision: FRNet labels as an OPT-IN mode,
ground truth stays the default, motion stays ground truth (disclosed). These tests hold
each part of that:
  - the default path is untouched and says so on every frame;
  - misuse fails loudly instead of silently falling back to ground truth;
  - in the DL mode the labels come from the predictor while motion does not;
  - the pipeline's FRNet labels equal `frnet_eval.py`'s on a real frame.
"""
from pathlib import Path

import numpy as np
import pytest
from vrgrid.perception import loader, semantics
from vrgrid.run.__main__ import build_parser, iter_pipeline

SEQ = "00"


def _need_seq(seq):
    if not (loader.verify_sequence_exists(seq) and loader._velodyne_path(seq, 0).exists()):
        pytest.skip(f"KITTI seq {seq} not present -- set VRGRID_DATA_ROOT")


class _Recorder:
    """A stand-in predictor: records what it is given, returns a known labelling.

    Not data -- it labels whatever real scan the pipeline hands it with the same
    deterministic rule, so the test can tell FRNet-mode labels from ground truth.
    """

    def __init__(self):
        self.calls = []

    def infer_points(self, points):
        self.calls.append(np.array(points, copy=True))
        n = len(points)
        return (np.arange(n) % 19).astype(np.int32)


def test_default_is_ground_truth_and_says_so():
    _need_seq(SEQ)
    frame = next(iter_pipeline(SEQ, 1, use_patchworkpp=False))
    raw = loader.load_labels(loader._velodyne_path(SEQ, 0).parents[1] / "labels" / "000000.label")
    assert frame.semantic_source == "gt"
    assert np.array_equal(frame.semantic, semantics.semantic_labels(raw))


def test_unknown_source_and_missing_predictor_fail_loudly():
    with pytest.raises(ValueError):
        next(iter_pipeline(SEQ, 1, semantics_source="pointnet"))
    with pytest.raises(ValueError):
        next(iter_pipeline(SEQ, 1, semantics_source="frnet"))


def test_frnet_mode_takes_labels_from_the_predictor_and_motion_from_ground_truth():
    _need_seq(SEQ)
    rec = _Recorder()
    gt_frame = next(iter_pipeline(SEQ, 1, use_patchworkpp=False))
    dl_frame = next(iter_pipeline(SEQ, 1, use_patchworkpp=False,
                                  semantics_source="frnet", frnet=rec))
    assert dl_frame.semantic_source == "frnet"
    assert len(rec.calls) == 1
    assert np.array_equal(rec.calls[0], dl_frame.points_sensor.astype(np.float32))
    assert np.array_equal(dl_frame.semantic, (np.arange(len(dl_frame.semantic)) % 19))
    assert not np.array_equal(dl_frame.semantic, gt_frame.semantic)
    # [!] motion is ground truth in both modes -- FRNet has no motion output
    assert np.array_equal(dl_frame.moving, gt_frame.moving)
    assert np.array_equal(dl_frame.points_world, gt_frame.points_world)


def test_cli_refuses_frnet_knobs_without_the_mode():
    from vrgrid.run.__main__ import main
    args = build_parser().parse_args(["--semantics", "frnet", "--threads", "1"])
    assert args.semantics == "frnet" and args.threads == 1
    assert build_parser().parse_args([]).semantics == "gt"
    with pytest.raises(SystemExit):
        main(["--seq", SEQ, "--frames", "1", "--no-map", "--fast-scatter"])


def test_pipeline_frnet_labels_equal_frnet_eval_on_a_real_frame():
    """[!] `FRNetInference` loads the checkpoint through its own key-mapping helper;
    `frnet_eval.py` uses load_state_dict(strict=False). The DL mode is only the model
    the project scores if the two produce the same labels. One real frame, one thread,
    loop path (reproducible per R-j)."""
    torch = pytest.importorskip("torch")
    ckpt = Path("checkpoints/frnet-semantickitti_seg.pth")
    if not ckpt.exists():
        pytest.skip("FRNet checkpoint not present")
    _need_seq("08")
    from vrgrid.perception.frnet import FRNet
    from vrgrid.run.__main__ import open_frnet

    saved = torch.get_num_threads()
    try:
        frnet = open_frnet(threads=1)
        frame = next(iter_pipeline("08", 1, semantics_source="frnet", frnet=frnet))

        model = FRNet(num_classes=20, ignore_index=19, output_shape=(64, 512),
                      fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG,
                      fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob.get("state_dict", blob), strict=False)
        model.eval()
        pts = loader.load_velodyne_scan(loader._velodyne_path("08", 0))
        with torch.no_grad():
            ref = model.predict([torch.from_numpy(pts).float()])[0].cpu().numpy()
        ref = np.where(ref >= 19, -1, ref).astype(np.int32)
    finally:
        torch.set_num_threads(saved)
    assert frame.semantic.shape == ref.shape
    assert np.array_equal(frame.semantic, ref), \
        f"{int((frame.semantic != ref).sum()):,} of {ref.size:,} labels differ from frnet_eval.py's model"
