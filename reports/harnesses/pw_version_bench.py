# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/latency-gap-investigation.md
#           the version sweep: 1.4.1 18.58 ms vs 1.4.0 24.37, 1.3.1 23.88, 1.3.0 23.83, 1.2.0 25.05.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/pw_version_bench.py 40
#
# NOTE: Standalone: needs only numpy + pypatchworkpp, so it runs inside a throwaway
#      venv with a different pinned version. Passes 3 columns, not 4, so RNR
#      is inactive -- consistent across versions, which is what the sweep needs.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Standalone ground-stage cost. No vrgrid import, so it runs in a bare venv
with only numpy + pypatchworkpp. MEASUREMENT ONLY.

Reads velodyne .bin directly and applies the same sensor height ground.py uses,
so the number is comparable to the 20-21 ms figure in the latency report.
"""
import glob
import sys
import time
import numpy as np
import pypatchworkpp as pw

ROOT = "C:/KITTI/dataset/sequences/08/velodyne"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SENSOR_HEIGHT_M = 1.73

files = sorted(glob.glob(ROOT + "/*.bin"))[:N]
scans = [np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
         for f in files]

p = pw.Parameters()
p.sensor_height = SENSOR_HEIGHT_M
p.verbose = False
est = pw.patchworkpp(p)
for s in scans[:3]:
    est.estimateGround(s)
reps = []
for _ in range(3):
    ts = []
    for s in scans:
        t0 = time.perf_counter()
        est.estimateGround(s)
        ts.append((time.perf_counter() - t0) * 1e3)
    reps.append(float(np.median(ts)))
print(f"RESULT version={getattr(pw,'__version__','?')} "
      f"p50={np.mean(reps):.2f} reps={[round(r,2) for r in reps]} "
      f"scans={len(scans)}")
