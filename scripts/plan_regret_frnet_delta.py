#!/usr/bin/env python3
"""Plan regret with ground-truth labels vs FRNet-predicted labels, on real data.

    python scripts/plan_regret_frnet_delta.py --oracle --frames 40     # plumbing control
    python scripts/plan_regret_frnet_delta.py --fast-scatter           # the real run

[!] NAMING. This script is the "three-way plan-regret comparison" asked for when
  D11 was reopened on 2026-09-14. "R2" ELSEWHERE IN THE DOCS REFERS TO A
  DIFFERENT, PRIOR RESEARCH DOCUMENT (dynamics / segmentation), NOT THIS SCRIPT.
  Name the output by this file, never by "R2".

[!] OFFLINE EVALUATION ONLY. The mapping pipeline takes semantics from the
  SemanticKITTI `.label` files on purpose, and still does after this script
  exists. Nothing here changes `ground.py`, the run engine, `semantics.py`, or the
  production label source. FRNet predictions exist only inside this process and
  only to answer one question: how much worse do the map's DECISIONS get when
  the labels come from the model instead of the ground truth?

The comparison, all on the same frames of a real sequence:

  M*       the reference map, built from GROUND-TRUTH labels exactly as
           `eval_synthetic.py --seq` builds it (`build_from_scans(real_scans)`).
  M_gt     the shipped schedule's map, driven by `harness.real_scans` UNCHANGED.
  M_frnet  the same schedule, same scans, same poses, same ground masks, with
           each point's CLASS replaced by FRNet's prediction.

  R_gt, R_frnet  plan regret of each against M*, on their common support, via
                 `eval_synthetic.plan_regret_for`, imported rather than copied.
  delta          R_frnet - R_gt: the regret attributable to segmentation.

Three decisions, each stated because each changes what the delta means:

[!] MOTION STAYS GROUND TRUTH BY DEFAULT (`--motion gt`). `run_sequence` removes
  dynamic returns using RAW `moving-*` ids before it ever reads a class, and
  FRNet predicts 19 classes with no notion of motion. So in the FRNet arm a point
  whose ground truth is `moving-*` keeps its raw id and is still routed out of the
  persistent map, and only STATIC points take FRNet's class. That isolates
  segmentation quality, the same way the live pipeline already takes motion from
  the `.label` files. `--motion none` gives every point FRNet's class instead:
  moving objects are then welded into the map, and the delta mixes segmentation
  with the loss of motion information. Report which one was run.

[!] GROUND MASKS ARE IDENTICAL IN BOTH ARMS. Ground comes from Patchwork++ on the
  raw points. Labels only reach it through the fallback, which is used when
  Patchwork++ is absent. The script refuses to run on the fallback, where the two
  arms' ground masks would differ.

[!] THE PATCHWORK++ SINGLETON (D1) IS RESET BEFORE EVERY PASS. The estimator is a
  stateful module global, and a second pass through it does not reproduce the
  first. Each of the three passes (M*, M_gt, M_frnet) starts from a fresh
  estimator, so the arms differ only in labels. `ground.py` itself is not edited.

`--oracle` is the control and should be run first. It feeds ground truth's own
classes through the FRNet arm's code path. The two maps must then be
bit-identical, and the delta exactly 0.000. If they are not, the plumbing is
wrong, and no FRNet number from this script means anything.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_synthetic import PLAN_QUERY_FAMILIES, costmaps_for, plan_regret_for  # noqa: E402
from vrgrid.eval.harness import (  # noqa: E402
    build_gridmap, final_vehicle_xy, real_scans, run_sequence)
from vrgrid.eval.plan_regret import common_support  # noqa: E402
from vrgrid.eval.reference_map import build_from_scans  # noqa: E402
from vrgrid.grid.fusion import CLASS_UNLABELLED  # noqa: E402
from vrgrid.grid.schedule import load  # noqa: E402
from vrgrid.grid.transient import TrackList, is_moving  # noqa: E402

SCHEDULE = "5/10/20/40"
#: FRNet's 20th output is its ignore slot, not a class.
FRNET_IGNORE = 19
N_CLASSES = 19


def compose_labels(raw_labels, predicted, motion: str = "gt") -> np.ndarray:
    """The label array the FRNet arm hands to `run_sequence`.

    `raw_labels` are the RAW SemanticKITTI ids for one scan and `predicted` is one
    class per point, 0-19. Static points get the predicted class, with the ignore
    slot (and anything out of range) mapped to `CLASS_UNLABELLED` exactly as
    `harness.learning_ids` maps ground truth's unlabelled points.

    With `motion="gt"`, points whose RAW id is `moving-*` keep that raw id, so
    `transient.separate` routes them out of the map just as it does for ground
    truth. With `motion="none"`, every point carries the predicted class.

    Every value given to a static point is <= CLASS_MAX, so `learning_ids` passes
    the static subset through unchanged rather than re-mapping it as raw ids.
    """
    if motion not in ("gt", "none"):
        raise ValueError(f"motion must be 'gt' or 'none', not {motion!r}")
    raw = np.asarray(raw_labels)
    pred = np.asarray(predicted).astype(np.int64)
    if pred.shape != raw.shape:
        raise ValueError(f"{pred.shape[0] if pred.ndim else 0} predictions for "
                         f"{raw.shape[0] if raw.ndim else 0} points -- misaligned")
    cls = np.where((pred < 0) | (pred >= N_CLASSES), CLASS_UNLABELLED, pred)
    if motion == "none":
        return cls.astype(raw.dtype)
    return np.where(is_moving(raw), raw, cls.astype(raw.dtype))


def _fresh_ground():
    """D1: start every pass from a new Patchwork++ estimator. Not an edit to ground.py."""
    from vrgrid.perception import ground
    if not ground._HAVE_PATCHWORKPP:
        sys.exit("Patchwork++ is not installed: ground masks would come from the "
                 "labels and differ between the arms. Refusing to run.")
    ground._estimator = None


def relabelled_scans(sequence, frames, predict, motion):
    """`harness.real_scans`, with the label element replaced.

    Wraps the harness generator rather than re-implementing it, so poses, frames,
    vehicle-frame points and ground masks are the harness's own. The raw sensor
    points FRNet needs are re-read from the same loader, in the same order.
    """
    from vrgrid.perception import loader
    raw = loader.scans(sequence, max_frames=frames)
    for i, ((vehicle_pts, labels, gmask, T), (pts, labels2, _)) in enumerate(
            zip(real_scans(sequence, frames), raw)):
        if not np.array_equal(labels, labels2):
            raise AssertionError(f"frame {i}: harness and loader disagree on labels -- "
                                 "the two generators are not aligned")
        yield vehicle_pts, compose_labels(labels, predict(i, pts, labels), motion), gmask, T


def _digest(gm) -> str:
    soa = gm.soa
    h = hashlib.sha256()
    if isinstance(soa, np.ndarray):
        h.update(soa.tobytes())
    else:
        for k in sorted(soa):
            h.update(k.encode())
            h.update(np.ascontiguousarray(soa[k]).tobytes())
    return h.hexdigest()[:16]


def _build(sequence, frames, scans):
    _fresh_ground()
    gm = build_gridmap(load(SCHEDULE))
    tracks = TrackList(gm.allocation.max_tracks, arrays=gm.allocation.tracks)
    stats = run_sequence(gm, scans, tracks=tracks)
    return gm, stats


def _frnet_predictor(checkpoint: Path, fast_scatter: bool, score: dict):
    """Exactly `frnet_eval.py`'s construction and inference call."""
    import torch
    from vrgrid.perception import semantics
    from vrgrid.perception.frnet import FRNet

    if fast_scatter:
        from frnet_fast_scatter import enable
        enable(verify=True)
    if not checkpoint.exists():
        sys.exit(f"checkpoint not found: {checkpoint}")
    model = FRNet(num_classes=20, ignore_index=FRNET_IGNORE, output_shape=(64, 512),
                  fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG,
                  fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(blob.get("state_dict", blob), strict=False)
    model.eval()

    def predict(i, pts, labels):
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict([torch.from_numpy(pts).float()])[0].cpu().numpy()
        score["infer_s"] += time.perf_counter() - t0
        gt = semantics.semantic_labels(labels)
        ok = gt >= 0
        score["correct"] += int((pred[ok] == gt[ok]).sum())
        score["total"] += int(ok.sum())
        return pred
    return predict


def _oracle_predictor(score: dict):
    from vrgrid.perception import semantics

    def predict(i, pts, labels):
        gt = semantics.semantic_labels(labels)
        score["correct"] += int((gt >= 0).sum())
        score["total"] += int((gt >= 0).sum())
        return gt
    return predict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="08", help="a REAL SemanticKITTI sequence")
    ap.add_argument("--frames", type=int, default=200,
                    help="frames from 0; 200 is the slice frnet_eval.py scores")
    ap.add_argument("--checkpoint", default="checkpoints/frnet-semantickitti_seg.pth")
    ap.add_argument("--fast-scatter", action="store_true",
                    help="scripts/frnet_fast_scatter.py, verified before use")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch.set_num_threads(N) before the model is built. OPEN-ITEMS R-j: at "
                         "the default thread count FRNet predictions are NOT reproducible per point "
                         "on either path; with 1 thread --fast-scatter is bit-identical to the loops "
                         "and reproducible. Use 1 for any number that has to reproduce.")
    ap.add_argument("--motion", choices=["gt", "none"], default="gt",
                    help="gt: moving-* stays ground truth (isolates segmentation). "
                         "none: every point takes FRNet's class")
    ap.add_argument("--oracle", action="store_true",
                    help="CONTROL: feed ground truth through the FRNet arm; the maps "
                         "must hash identically and the delta must be 0.000")
    ap.add_argument("--json", default=None, help="also write the results here")
    args = ap.parse_args()

    if args.threads is not None and not args.oracle:
        import torch
        torch.set_num_threads(args.threads)
    score = {"correct": 0, "total": 0, "infer_s": 0.0}
    predict = (_oracle_predictor(score) if args.oracle
               else _frnet_predictor(Path(args.checkpoint), args.fast_scatter, score))
    arm = "oracle (ground truth through the FRNet path)" if args.oracle else "FRNet"
    print(f"sequence {args.seq}, frames 0-{args.frames - 1}, schedule {SCHEDULE}, "
          f"motion={args.motion}, second arm = {arm}, threads={args.threads or 'default'}")

    t0 = time.perf_counter()
    _fresh_ground()
    reference = build_from_scans(real_scans(args.seq, args.frames))
    vehicle_xy = final_vehicle_xy(args.seq, args.frames)
    print(f"reference map (ground truth): {reference}   [{time.perf_counter() - t0:.0f}s]")

    t0 = time.perf_counter()
    gm_gt, st_gt = _build(args.seq, args.frames, real_scans(args.seq, args.frames))
    print(f"M_gt built    [{time.perf_counter() - t0:.0f}s]  digest {_digest(gm_gt)}")
    t0 = time.perf_counter()
    gm_fr, st_fr = _build(args.seq, args.frames,
                          relabelled_scans(args.seq, args.frames, predict, args.motion))
    print(f"M_{'oracle' if args.oracle else 'frnet'} built  [{time.perf_counter() - t0:.0f}s]"
          f"  digest {_digest(gm_fr)}")
    identical = _digest(gm_gt) == _digest(gm_fr)
    if args.oracle:
        print(f"CONTROL maps bit-identical: {identical}")

    acc = score["correct"] / score["total"] if score["total"] else float("nan")
    print(f"per-point accuracy of the second arm's labels vs ground truth: {acc:.1%} "
          f"over {score['total']:,} labelled points")
    if not args.oracle:
        print(f"FRNet inference: {score['infer_s']:.0f}s total, "
              f"{score['infer_s'] / max(st_fr.frames, 1):.2f}s/frame (CPU)")

    mask = common_support(costmaps_for(gm_gt, reference, vehicle_xy)[1],
                          costmaps_for(gm_fr, reference, vehicle_xy)[1])
    print(f"common support: {mask.mean():.1%} of the planning window\n")

    out = {"sequence": args.seq, "frames": args.frames, "schedule": SCHEDULE,
           "motion": args.motion, "oracle": args.oracle, "fast_scatter": args.fast_scatter,
           "threads": args.threads,
           "checkpoint": None if args.oracle else args.checkpoint,
           "label_point_accuracy": acc, "maps_bit_identical": identical,
           "common_support": float(mask.mean()), "families": {}}
    print(f"  {'family':<13}{'R_gt':>9}{'R_second':>10}{'delta':>9}"
          f"{'found gt/2nd':>14}{'blocked gt/2nd':>16}")
    for family in PLAN_QUERY_FAMILIES:
        r_gt = plan_regret_for(gm_gt, reference, vehicle_xy, mask, family=family)
        r_fr = plan_regret_for(gm_fr, reference, vehicle_xy, mask, family=family)
        delta = r_fr.regret - r_gt.regret
        print(f"  {family:<13}{r_gt.regret:>9.3f}{r_fr.regret:>10.3f}{delta:>+9.3f}"
              f"{f'{r_gt.n_found}/{r_fr.n_found}':>14}"
              f"{f'{r_gt.n_blocked}/{r_fr.n_blocked}':>16}")
        out["families"][family] = {
            "R_gt": r_gt.regret, "R_second": r_fr.regret, "delta": delta,
            "sd_gt": r_gt.regret_sd, "sd_second": r_fr.regret_sd,
            "found": [r_gt.n_found, r_fr.n_found], "blocked": [r_gt.n_blocked, r_fr.n_blocked],
            "queries": r_gt.n_queries}
    print("\n  delta = R_second - R_gt, both scored on M* over the common support.")
    if args.oracle and not identical:
        print("\n[!] CONTROL FAILED: the oracle arm did not reproduce M_gt. Do not "
              "trust any FRNet run of this script until this is explained.")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0 if (identical or not args.oracle) else 1


if __name__ == "__main__":
    sys.exit(main())
