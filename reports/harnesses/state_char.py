# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/ring1-reproduction-investigation.md
#           the characterisation: pass1 vs pass2 differs by 1,245 of 1,479,013 points; two separate estimators agree exactly.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/state_char.py
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Characterise it: is the 'second pass differs' effect general, and is it the
replay case specifically? MEASUREMENT ONLY."""
import numpy as np
import pypatchworkpp as _pw
from vrgrid.perception import loader
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

scans = [np.asarray(p, dtype=np.float64)
         for p, _, _ in loader.scans("08", max_frames=12)]

def est():
    p = _pw.Parameters(); p.sensor_height = SENSOR_HEIGHT_M; p.verbose = False
    return _pw.patchworkpp(p)

def mask(e, s):
    e.estimateGround(s)
    m = np.zeros(len(s), bool)
    m[np.asarray(e.getGroundIndices(), dtype=np.int64)] = True
    return m

print("A) PASS 1 vs PASS 2 over the same 12 frames, one estimator")
print("   (this is exactly what the determinism gate does: replay() == replay())")
e = est()
p1 = [mask(e, s) for s in scans]
p2 = [mask(e, s) for s in scans]
tot = sum(len(s) for s in scans)
diff = sum(int((a != b).sum()) for a, b in zip(p1, p2))
print(f"   {diff:,} of {tot:,} points differ ({diff/tot*100:.3f}%)  "
      f"frames differing: {sum(1 for a,b in zip(p1,p2) if (a!=b).any())}/{len(scans)}")

print("\nB) PASS 2 vs PASS 3 -- does it settle?")
p3 = [mask(e, s) for s in scans]
d23 = sum(int((a != b).sum()) for a, b in zip(p2, p3))
print(f"   {d23:,} points differ")

print("\nC) two SEPARATE estimators, one pass each -- the fix under test")
ea, eb = est(), est()
qa = [mask(ea, s) for s in scans]
qb = [mask(eb, s) for s in scans]
dab = sum(int((a != b).sum()) for a, b in zip(qa, qb))
print(f"   {dab:,} points differ  -> {'IDENTICAL' if dab == 0 else 'NOT identical'}")

print("\nD) per-frame breakdown of pass1 vs pass2 (first 12)")
for i, (a, b) in enumerate(zip(p1, p2)):
    n = int((a != b).sum())
    if n:
        print(f"   frame {i:>2}: {n:>6,} points differ  "
              f"({a.sum():,} -> {b.sum():,} ground)")
