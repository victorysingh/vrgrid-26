# PROVENANCE -- written 2026-09-14 for reports/frnet-accuracy-by-distance.md.
#
# Produced: the "50 m label ceiling" table -- raw vs labelled points per distance band, the farthest
#           labelled and farthest raw point, and the raw semantic ids of points beyond 50 m.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset python reports/harnesses/semantickitti_label_range.py 08
#
# NOTE: Written because frnet_eval_by_range.py found exactly zero labelled points beyond 50 m over 200
#      frames. That could have been a band-assignment bug or a data property; this reads the scans and
#      labels directly, using the SAME band_of() the evaluation uses, to tell which.
#
# Measurement only. Changes nothing.
"""Where does SemanticKITTI ground truth stop? Raw vs labelled points by horizontal range."""
import sys
from pathlib import Path

import numpy as np
from vrgrid.perception import loader, semantics

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from frnet_eval_by_range import BAND_NAMES, band_of  # noqa: E402

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
FRAMES = list(range(0, 200, 20)) + [199]


def main():
    root = Path(loader.DATA_ROOT) / "sequences" / SEQ
    raw_n = np.zeros(len(BAND_NAMES), np.int64)
    lab_n = np.zeros(len(BAND_NAMES), np.int64)
    far_ids = {}
    max_lab_r = max_raw_r = 0.0
    for i in FRAMES:
        pts = loader.load_velodyne_scan(root / "velodyne" / f"{i:06d}.bin")
        raw = loader.load_labels(root / "labels" / f"{i:06d}.label")
        gt = semantics.semantic_labels(raw)
        r = np.hypot(pts[:, 0], pts[:, 1])
        b = band_of(pts)
        for k in range(len(BAND_NAMES)):
            raw_n[k] += int((b == k).sum())
            lab_n[k] += int(((b == k) & (gt >= 0)).sum())
        ids, cnt = np.unique(raw[r > 50] & 0xFFFF, return_counts=True)
        for a, c in zip(ids, cnt):
            far_ids[int(a)] = far_ids.get(int(a), 0) + int(c)
        if (gt >= 0).any():
            max_lab_r = max(max_lab_r, float(r[gt >= 0].max()))
        max_raw_r = max(max_raw_r, float(r.max()))

    print(f"sequence {SEQ}, frames sampled: {FRAMES}")
    print(f"{'band':<10}{'raw points':>12}{'labelled (gt>=0)':>18}{'labelled %':>12}")
    for k, name in enumerate(BAND_NAMES):
        pct = 100 * lab_n[k] / raw_n[k] if raw_n[k] else float("nan")
        print(f"{name:<10}{raw_n[k]:>12,}{lab_n[k]:>18,}{pct:>11.2f}%")
    print(f"max horizontal range, any point:      {max_raw_r:.1f} m")
    print(f"max horizontal range, labelled point: {max_lab_r:.1f} m")
    top = dict(sorted(far_ids.items(), key=lambda kv: -kv[1])[:10])
    print(f"raw semantic ids of points beyond 50 m (id: count): {top}")


if __name__ == "__main__":
    main()
