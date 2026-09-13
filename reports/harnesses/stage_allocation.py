# PROVENANCE -- written 2026-09-13 under OPEN-ITEMS.md items R-b / R-h.
#
# Produced: reports/r-b-p99-tail-investigation.md, the per-stage allocation table
#           that tests "allocation causes the tail" across ALL stages rather than
#           one at a time.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/stage_allocation.py 08 30
#
# NOTE: transform_tail.py showed the tail IS allocation for one stage. If that
#      generalises, per-stage allocation should rank alongside per-stage p99
#      spread. If it does not, the remaining tail has a different cause and
#      chasing allocation further is wasted effort. Either answer is useful.
#
#      tracemalloc roughly doubles runtime, so the ABSOLUTE times here are not
#      comparable to p99_probe.py -- only the allocation column and the ranking
#      are. Frame count is small for the same reason.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Per-stage allocation, measured by wrapping the Timer's own stage() hook.

`Timer.stage` is the one place every stage boundary passes through, in both
halves of the pipeline, so wrapping it here gives per-stage allocation for the
whole frame without touching src/ and without a second list of stage names that
could drift from timing.STAGES.
"""
import sys
import tracemalloc
from contextlib import contextmanager

import numpy as np
from vrgrid.grid.schedule import load
from vrgrid.gpu.timing import STAGES, Timer
from vrgrid.run.__main__ import iter_pipeline
from vrgrid.run.engine import MapEngine

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30

alloc = {s: [] for s in STAGES}
_orig_stage = Timer.stage


@contextmanager
def _measuring_stage(self, name):
    before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    with _orig_stage(self, name):
        yield
    cur, peak = tracemalloc.get_traced_memory()
    alloc.setdefault(name, []).append((peak - before) / 1e6)


t = Timer(stages=STAGES, capacity=4096)
engine = MapEngine(load("5/10/20/40"), max_points=120_000, timer=t)

tracemalloc.start()
Timer.stage = _measuring_stage
try:
    # reuse_buffers=True: the same path scripts/timing_table.py times (12613df).
    # Without it the transform row measures the old, allocating path.
    frames = iter(iter_pipeline(SEQ, N + 1, timer=t, reuse_buffers=True))
    k = 0
    while True:
        f = next(frames, None)
        if f is None:
            break
        engine.step(f)
        k += 1
finally:
    Timer.stage = _orig_stage
    tracemalloc.stop()

print(f"  {k} frames, seq {SEQ}  (tracemalloc active: times inflated, "
      f"allocation exact)\n")
print(f"  {'stage':<14}{'peak alloc MB':>15}{'max MB':>9}{'time p50':>10}"
      f"{'time p99':>10}{'p99-p50':>9}")
print("  " + "-" * 68)
rows = []
for s in STAGES:
    if s == "total":
        continue
    v = np.asarray(alloc.get(s, []), dtype=np.float64)
    ts = np.asarray(t._samples(s), dtype=np.float64)
    if not v.size or not ts.size:
        continue
    v, ts = v[1:], ts[1:]                      # drop the startup frame
    if not v.size:
        continue
    rows.append((s, float(np.median(v)), float(v.max()),
                 float(np.median(ts)), float(np.percentile(ts, 99)),
                 float(np.percentile(ts, 99) - np.median(ts))))
for s, med, mx, p50, p99, spread in sorted(rows, key=lambda r: -r[1]):
    print(f"  {s:<14}{med:>15.2f}{mx:>9.2f}{p50:>10.2f}{p99:>10.2f}{spread:>9.2f}")

print(f"\n  total allocated per frame: "
      f"{sum(r[1] for r in rows):.2f} MB (median across stages, summed)")

print("\n  DOES ALLOCATION PREDICT THE TAIL?")
if len(rows) > 2:
    a = np.array([r[1] for r in rows])
    sp = np.array([r[5] for r in rows])
    print(f"    Spearman-ish rank corr(alloc, p99-p50) = "
          f"{np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(sp)))[0,1]:+.3f}")
    print(f"    Pearson  corr(alloc, p99-p50)          = "
          f"{np.corrcoef(a, sp)[0,1]:+.3f}")
    print("    A high rank correlation supports 'allocation causes the tail' as a")
    print("    general mechanism. A low one means transform was a special case and")
    print("    the rest of the tail is something else.")
