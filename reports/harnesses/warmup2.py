# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/latency-gap-investigation.md
#           the WHOLE-FRAME warm-up table (148.53 frame 0, 112.84 early, 100.59 best).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/warmup2.py 08 220
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""(b) Warm-up on the TRUE whole-frame metric -- perception + engine.step,
exactly as timing_table.run_real defines `total`. MEASUREMENT ONLY.
"""
import sys, time
import numpy as np
from vrgrid.grid.schedule import load
from vrgrid.run.__main__ import iter_pipeline
from vrgrid.run.engine import MapEngine

SEQ, N = sys.argv[1], int(sys.argv[2])
engine = MapEngine(load("5/10/20/40"), max_points=120_000)
frames = iter(iter_pipeline(SEQ, N + 1))
ts = []
while True:
    t0 = time.perf_counter()
    f = next(frames, None)
    if f is None: break
    engine.step(f)
    ts.append((time.perf_counter() - t0) * 1e3)
a = np.array(ts)
print(f"  {len(a)} whole frames (perception + engine.step)\n")
print(f"  {'window':<22}{'n':>5}{'p50':>9}{'p99':>9}")
for lo, hi in [(0,1),(1,21),(21,51),(51,101),(101,151),(151,201),(51,len(a)),(0,len(a))]:
    if hi > len(a) or lo >= hi: continue
    w = a[lo:hi]
    tag = f"[{lo}:{hi}]" + (" <- 50+ discarded" if lo==51 and hi==len(a) else
                            " <- frame 0 only" if hi-lo==1 else "")
    print(f"  {tag:<22}{len(w):>5}{np.median(w):>9.2f}{np.percentile(w,99):>9.2f}")
