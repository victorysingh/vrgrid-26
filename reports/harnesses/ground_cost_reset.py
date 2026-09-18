# PROVENANCE -- committed 2026-09-18 under OPEN-ITEMS.md item R-k.
#
# Produced: the TIMING half of R-k -- does R-a's published 21.01 ms p50 restate
#           once the Patchwork++ estimator is reset per repetition?
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/ground_cost_reset.py 08 60 {published|fresh}
#
# Measurement only. Reads the shipping code; changes nothing in src/, and does NOT
# modify reports/harnesses/ground_cost.py, whose output is published provenance.
#
# D8: run only on a state-trusted machine, checked BEFORE and AFTER every pass.
"""`ground_cost.py`'s Patchwork++ timing, with and without state carry-over.

Two configurations, one per process, chosen by argv[3]:

  published  exactly `ground_cost.py`: warm up once, then 3 reps over the same
             scans with NO reset, so rep 2 starts from what rep 1 left behind.
             This is the configuration that produced 21.01 ms p50.

  fresh      reset the estimator before each rep, then warm up on scans[0]
             UNTIMED, then time the rep. The untimed warm-up matters: without
             it, the lazy construction of the estimator would land inside the
             first frame's measurement and the comparison would be about
             construction cost rather than about adaptation state.

Only Patchwork++ is timed here; the fallback is not re-measured, because R-k is
about the estimator's state and the fallback has none.
"""
import sys
import time

import numpy as np
from vrgrid.perception import ground, loader, semantics

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MODE = sys.argv[3] if len(sys.argv) > 3 else "published"
assert MODE in ("published", "fresh"), MODE
REPS = 3

scans = [(pts, semantics.semantic_labels(raw))
         for pts, raw, _pose in loader.scans(SEQ, max_frames=N)]

if MODE == "published":
    ground.reset_estimator()                       # start from a known state
    ground.segment_ground(scans[0][0])             # the one warm-up, line 36

per_rep, pooled = [], []
for _ in range(REPS):
    if MODE == "fresh":
        ground.reset_estimator()
        ground.segment_ground(scans[0][0])         # UNTIMED: keeps construction out
    ts = []
    for pts, _sem in scans:
        t0 = time.perf_counter()
        ground.segment_ground(pts)
        ts.append((time.perf_counter() - t0) * 1e3)
    per_rep.append(float(np.median(ts)))
    pooled.extend(ts)

a = np.array(pooled)
print(f"mode={MODE} seq={SEQ} frames={N} reps={REPS}")
print(f"  rep medians   {', '.join(f'{m:.2f}' for m in per_rep)}")
print(f"  mean of those {np.mean(per_rep):.2f} ms      <- the statistic ground_cost.py prints")
print(f"  pooled p50    {np.percentile(a, 50):.2f} ms")
print(f"  pooled p99    {np.percentile(a, 99):.2f} ms")
print(f"  pooled n      {a.size}")
