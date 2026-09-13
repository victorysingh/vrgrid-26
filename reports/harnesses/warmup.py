# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/latency-gap-investigation.md
#           the front-half warm-up windows (62.4 ms flat from frame 1 onward).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/warmup.py 08 250
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""(b) Does a longer warm-up change the whole-frame number? MEASUREMENT ONLY.

Times each frame individually through the real pipeline, then reports p50/p99
over successive windows. If 'warm' means anything beyond frame ~1, a long tail
of frames should be measurably cheaper than an early window.
"""
import sys
import time
import numpy as np
from vrgrid.run.__main__ import iter_pipeline

N = int(sys.argv[2]) if len(sys.argv) > 2 else 250
frames = iter_pipeline(sys.argv[1], N)
ts = []
while True:
    t0 = time.perf_counter()
    f = next(frames, None)
    dt = (time.perf_counter() - t0) * 1e3
    if f is None:
        break
    ts.append(dt)
a = np.array(ts)
print(f"  {len(a)} frames timed end-to-end (includes load)\n")
print(f"  {'window':<18}{'n':>5}{'p50':>9}{'p99':>9}{'mean':>9}")
for lo, hi in [(0,1),(1,20),(20,50),(50,100),(100,150),(150,200),(200,len(a)),(50,len(a))]:
    if hi > len(a) or lo >= hi:
        continue
    w = a[lo:hi]
    tag = f"[{lo}:{hi}]" + (" <- 50+ warm" if lo == 50 and hi == len(a) else "")
    print(f"  {tag:<18}{len(w):>5}{np.median(w):>9.2f}{np.percentile(w,99):>9.2f}{w.mean():>9.2f}")
print(f"\n  ALL                {len(a):>5}{np.median(a):>9.2f}{np.percentile(a,99):>9.2f}{a.mean():>9.2f}")
