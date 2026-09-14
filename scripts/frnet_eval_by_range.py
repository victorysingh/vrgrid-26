#!/usr/bin/env python3
"""FRNet classification accuracy across distance bands. SIH26053's "Performance Metrics".

    python scripts/frnet_eval_by_range.py --fast-scatter --threads 1        # seq 08, 200 frames

The problem statement asks for "evidence of ... high accuracy in object classification
across varying distances". `frnet_eval.py` reports one pooled number per class, which
cannot show that. This script scores the same model, on the same frames, with the same
inference call, broken down by horizontal distance from the sensor.

[!] SAME MODEL, SAME CALL, SAME FRAMES AS `frnet_eval.py`. The FRNet construction,
  checkpoint load and `model.predict([torch.from_numpy(pts).float()])` are transcribed
  from it. Summed over every band, the point accuracy and the 15-class mIoU must
  reproduce `frnet_eval.py`'s figures on the same slice. The script prints that check,
  and a mismatch means the bands are wrong, not the model.

[!] REPRODUCIBILITY NEEDS `--threads 1` (OPEN-ITEMS R-j). At the default thread count,
  FRNet predictions change by roughly 0.06% of points between runs, on either reduction
  path. With one thread they are bit-identical, and `--fast-scatter` equals the port's
  loops exactly. Any number meant to reproduce must be run with `--threads 1`.

Distance is the HORIZONTAL range from the sensor, sqrt(x^2 + y^2) in the sensor frame.
The bands follow the map's ring edges (5/10/50 and 5/10/20/40 both start at +-10 m and
+-25 m) so accuracy can be read next to the resolution each band is mapped at. The rings
are square windows and these bands are radial, so a point at a ring corner can fall in
the next band out: the correspondence is approximate, and it is stated rather than implied.

Three groups follow the problem statement's own wording. Groups are scored by collapsing
both prediction and ground truth to the group, so a car predicted as truck is still a
correct "movable object":
  drivable terrain   road, parking, sidewalk, other-ground, terrain  (the map's §7.1 set)
  static obstacle    building, fence, vegetation, trunk, pole, traffic-sign
  movable object     car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist,
                     motorcyclist
"Movable" is a class property. FRNet has no notion of motion; this does NOT measure
moving-object detection.

[!] THIS IS NOT THE MAP'S SEMANTIC SOURCE. The mapping pipeline takes semantics from the
  `.label` files; this evaluates the model alongside it, as `frnet_eval.py` does.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
           "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
           "other-ground", "building", "fence", "vegetation", "trunk",
           "terrain", "pole", "traffic-sign"]
GROUPS = {
    "drivable terrain": ("road", "parking", "sidewalk", "other-ground", "terrain"),
    "static obstacle": ("building", "fence", "vegetation", "trunk", "pole", "traffic-sign"),
    "movable object": ("car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
                       "bicyclist", "motorcyclist"),
}
GROUP_NAMES = list(GROUPS)
#: class index -> group index. Every one of the 19 classes is in exactly one group.
CLASS_TO_GROUP = np.array([next(g for g, names in enumerate(GROUPS.values()) if c in names)
                           for c in CLASSES], dtype=np.int64)
#: Band edges in metres, matching the rings' +-10 / +-25 / +-50 / +-100 m extents.
EDGES_M = (0.0, 10.0, 25.0, 50.0, 100.0, float("inf"))
BAND_NAMES = ["0-10 m", "10-25 m", "25-50 m", "50-100 m", ">100 m"]


def band_of(xy: np.ndarray) -> np.ndarray:
    """(N, 2+) sensor-frame points -> (N,) band index by horizontal range."""
    r = np.hypot(xy[:, 0], xy[:, 1])
    return np.searchsorted(np.asarray(EDGES_M[1:-1]), r, side="right")


def to_group(labels: np.ndarray) -> np.ndarray:
    """19-class ids -> group ids. Anything outside 0-18 (ignore, unlabelled) -> -1."""
    lab = np.asarray(labels).astype(np.int64)
    out = np.full(lab.shape, -1, dtype=np.int64)
    ok = (lab >= 0) & (lab < len(CLASSES))
    out[ok] = CLASS_TO_GROUP[lab[ok]]
    return out


class Tally:
    """Dataset-level accumulation: intersections and unions summed, divided once."""

    def __init__(self, n_classes: int):
        self.n = n_classes
        self.inter = np.zeros(n_classes, np.int64)
        self.union = np.zeros(n_classes, np.int64)
        self.support = np.zeros(n_classes, np.int64)
        self.correct = 0
        self.total = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        ok = (gt >= 0) & (gt < self.n)
        p, g = pred[ok], gt[ok]
        self.correct += int((p == g).sum())
        self.total += int(ok.sum())
        # Confusion counts over (n + 1) prediction columns: column n holds every
        # prediction outside 0..n-1 (FRNet's ignore slot 19, or -1 after grouping), so
        # it lands in no class. The row stride must be n + 1 for that column to exist:
        # a stride of n makes column n spill into the NEXT class's row.
        pc = np.where((p >= 0) & (p < self.n), p, self.n)
        cm = np.bincount(g * (self.n + 1) + pc, minlength=self.n * (self.n + 1))
        cm = cm.reshape(self.n, self.n + 1)[:, :self.n]
        tp = np.diag(cm)
        gt_count = np.bincount(g, minlength=self.n)[:self.n]
        pred_count = np.bincount(p[(p >= 0) & (p < self.n)], minlength=self.n)[:self.n]
        self.inter += tp
        self.union += gt_count + pred_count - tp
        self.support += gt_count

    def summary(self) -> dict:
        seen = self.support > 0
        iou = np.zeros(self.n)
        iou[seen] = self.inter[seen] / np.maximum(self.union[seen], 1)
        return {"points": self.total,
                "accuracy": self.correct / self.total if self.total else float("nan"),
                "miou_present": float(iou[seen].mean()) if seen.any() else float("nan"),
                "present": int(seen.sum()), "iou": iou.tolist(), "support": self.support.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--checkpoint", default="checkpoints/frnet-semantickitti_seg.pth")
    ap.add_argument("--fast-scatter", action="store_true")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch.set_num_threads(N) first; use 1 for reproducible numbers (R-j)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import torch
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    from vrgrid.perception import loader, semantics
    from vrgrid.perception.frnet import FRNet

    if args.fast_scatter:
        sys.path.insert(0, str(Path(__file__).parent))
        from frnet_fast_scatter import enable
        enable(verify=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    model = FRNet(num_classes=20, ignore_index=19, output_shape=(64, 512),
                  fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG,
                  fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob.get("state_dict", blob), strict=False)
    model.to("cpu").eval()

    n_bands = len(BAND_NAMES)
    cls_all, grp_all = Tally(len(CLASSES)), Tally(len(GROUP_NAMES))
    cls_band = [Tally(len(CLASSES)) for _ in range(n_bands)]
    grp_band = [Tally(len(GROUP_NAMES)) for _ in range(n_bands)]

    root = Path(loader.DATA_ROOT) / "sequences" / args.seq
    frames = 0
    for i in range(args.frames):
        scan = root / "velodyne" / f"{i:06d}.bin"
        label = root / "labels" / f"{i:06d}.label"
        if not scan.exists() or not label.exists():
            break
        pts = loader.load_velodyne_scan(scan)
        gt = semantics.semantic_labels(loader.load_labels(label)).astype(np.int64)
        with torch.no_grad():
            pred = model.predict([torch.from_numpy(pts).float()])[0].cpu().numpy().astype(np.int64)
        band = band_of(pts)
        cls_all.add(pred, gt)
        grp_all.add(to_group(pred), to_group(gt))
        for b in range(n_bands):
            m = band == b
            if m.any():
                cls_band[b].add(pred[m], gt[m])
                grp_band[b].add(to_group(pred[m]), to_group(gt[m]))
        frames += 1
    if not cls_all.total:
        print("no labelled frames found", file=sys.stderr)
        return 1

    reductions = "torch.scatter_reduce (--fast-scatter)" if args.fast_scatter else "the port's own loops"
    print(f"\nsequence {args.seq}, {frames} frames, {cls_all.total:,} labelled points, "
          f"threads={torch.get_num_threads()}, reductions: {reductions}")
    s_all = cls_all.summary()
    print(f"  ALL BANDS  point accuracy {s_all['accuracy']:.4%}  mIoU over {s_all['present']} present "
          f"classes {s_all['miou_present']:.2%}   <- must equal frnet_eval.py on this slice")

    band_parts = sum(t.total for t in cls_band)
    print(f"  band points sum to {band_parts:,} (== all: {band_parts == cls_all.total})")

    print(f"\n  {'band':<10}{'points':>12}{'share':>8}{'accuracy':>10}{'mIoU':>8}{'classes':>9}"
          + "".join(f"{g[:14]:>16}" for g in GROUP_NAMES) + f"{'3-group acc':>13}")
    out_bands = []
    for b in range(n_bands):
        s, gs = cls_band[b].summary(), grp_band[b].summary()
        if not s["points"]:
            print(f"  {BAND_NAMES[b]:<10}{0:>12}")
            out_bands.append({"band": BAND_NAMES[b], "points": 0})
            continue
        gi = gs["iou"]
        print(f"  {BAND_NAMES[b]:<10}{s['points']:>12,}{s['points'] / cls_all.total:>8.1%}"
              f"{s['accuracy']:>10.2%}{s['miou_present']:>8.1%}{s['present']:>9}"
              + "".join(f"{('IoU ' + format(gi[k], '.1%')) if gs['support'][k] else '-':>16}"
                        for k in range(len(GROUP_NAMES)))
              + f"{gs['accuracy']:>13.2%}")
        out_bands.append({"band": BAND_NAMES[b], "class": s, "group": gs})
    g_all = grp_all.summary()
    print(f"  {'ALL':<10}{cls_all.total:>12,}{1:>8.0%}{s_all['accuracy']:>10.2%}{s_all['miou_present']:>8.1%}"
          f"{s_all['present']:>9}" + "".join(f"{'IoU ' + format(g_all['iou'][k], '.1%'):>16}"
                                            for k in range(len(GROUP_NAMES)))
          + f"{g_all['accuracy']:>13.2%}")

    print("\n  per-class IoU by band (classes with ground truth in that band; '-' = none there):")
    print(f"  {'class':<15}" + "".join(f"{n:>11}" for n in BAND_NAMES))
    for c in range(len(CLASSES)):
        if not cls_all.support[c]:
            continue
        cells = []
        for b in range(n_bands):
            t = cls_band[b]
            cells.append(f"{t.inter[c] / max(t.union[c], 1):>10.1%} " if t.support[c] else f"{'-':>11}")
        print(f"  {CLASSES[c]:<15}" + "".join(cells))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "sequence": args.seq, "frames": frames, "threads": torch.get_num_threads(),
            "fast_scatter": args.fast_scatter, "checkpoint": str(ckpt), "band_edges_m": list(EDGES_M[:-1]),
            "classes": CLASSES, "groups": GROUPS, "all": {"class": s_all, "group": g_all},
            "bands": out_bands}, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
