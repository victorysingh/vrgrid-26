# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/latency-gap-investigation.md
#           the isolated ground-stage cost: Patchwork++ 21.01 ms p50 vs fallback 0.33 ms, 63.6x, 96.5% agreement.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/ground_cost.py 08 60
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Isolate the cost of the ground stage alone.

Whole-frame timing on this machine is dominated by page cache and load, not by
the segmenter -- every stage moved 2-3x between two back-to-back runs. So:
load the scans ONCE into RAM, then time only ground.segment_ground vs
ground.ground_from_semantics, alternating, many repetitions. No disk in the
loop, no other stage, same points every time.

MEASUREMENT ONLY. Touches nothing.
"""
import sys
import time

import numpy as np
from vrgrid.perception import ground, loader, semantics

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

print(f"pypatchworkpp present: {ground._HAVE_PATCHWORKPP}")
print(f"preloading {N} scans of seq {SEQ} into RAM...")
scans = []
for i, (pts, raw, pose) in enumerate(loader.scans(SEQ, max_frames=N)):
    scans.append((pts, semantics.semantic_labels(raw)))
print(f"  {len(scans)} scans, {sum(len(s[0]) for s in scans):,} points total\n")

# warm both paths once (Patchwork++ builds its estimator lazily on first call)
ground.segment_ground(scans[0][0])
ground.ground_from_semantics(scans[0][1])

def bench(fn, arg_idx, label, reps=3):
    per_rep = []
    for _ in range(reps):
        ts = []
        for s in scans:
            t0 = time.perf_counter()
            fn(s[arg_idx])
            ts.append((time.perf_counter() - t0) * 1e3)
        a = np.array(ts)
        per_rep.append((np.median(a), np.percentile(a, 99), a.mean()))
    med = np.array([p[0] for p in per_rep])
    print(f"  {label:<26} p50 {med.mean():7.2f} ms  (reps: "
          f"{', '.join(f'{m:.2f}' for m in med)})  p99 "
          f"{np.mean([p[1] for p in per_rep]):7.2f}")
    return med.mean()

print("alternating, 3 reps each, same points in RAM:")
pw = bench(ground.segment_ground, 0, "Patchwork++ (geometric)")
fb = bench(ground.ground_from_semantics, 1, "semantic-class fallback")
print(f"\n  Patchwork++ costs {pw - fb:+.2f} ms/frame more than the fallback")
print(f"  ratio: {pw / fb:.1f}x")

# agreement, so we know they are answering the same question
agree = []
for pts, sem in scans[:20]:
    a = ground.segment_ground(pts)
    b = ground.ground_from_semantics(sem)
    valid = sem >= 0
    agree.append((a[valid] == b[valid]).mean())
print(f"  agreement on labelled points, 20 frames: {np.mean(agree)*100:.1f}%")
