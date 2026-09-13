# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/ring1-reproduction-investigation.md
#           the minimal statefulness reproduction: history does not matter, re-seeing a scan does.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/state_minimal.py
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Minimal reproduction: does a Patchwork++ estimator's history change its
verdict on a later scan? MEASUREMENT ONLY."""
import numpy as np
import pypatchworkpp as _pw
from vrgrid.perception import loader
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

scans = [np.asarray(p, dtype=np.float64)
         for p, _, _ in loader.scans("08", max_frames=6)]

def est():
    p = _pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    return _pw.patchworkpp(p)

def mask(e, s):
    e.estimateGround(s)
    m = np.zeros(len(s), bool)
    m[np.asarray(e.getGroundIndices(), dtype=np.int64)] = True
    return m

B = scans[5]
cold = mask(est(), B)                       # B on a virgin estimator
e = est()
for s in scans[:5]:
    mask(e, s)
warm = mask(e, B)                           # B after 5 prior frames
e2 = est()
for s in scans[:5]:
    mask(e2, s)
warm2 = mask(e2, B)                         # same history again

print(f"  scan B has {len(B):,} points")
print(f"  virgin estimator vs 5-frame history : {(cold != warm).sum():,} points differ "
      f"({(cold != warm).mean()*100:.3f}%)")
print(f"  same history twice (reproducible?)  : {(warm != warm2).sum():,} points differ")
print(f"  ground count  virgin {cold.sum():,}   warm {warm.sum():,}")
# and repeated calls on the SAME scan with the SAME estimator
e3 = est()
a1 = mask(e3, B)
a2 = mask(e3, B)
a3 = mask(e3, B)
print(f"  same scan, same estimator, 3x calls : "
      f"1v2 {(a1 != a2).sum():,} differ, 2v3 {(a2 != a3).sum():,} differ")
