# PROVENANCE -- written 2026-09-13 under OPEN-ITEMS.md item R-h (the `ground` half).
#
# Produced: reports/r-b-p99-tail-investigation.md, the attribution of the
#           `ground` stage's tail.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/ground_tail.py 08 200
#
# NOTE: `ground` has the LARGEST single per-stage p99-p50 spread (8.85 ms) on
#      modest allocation (4.75 MB), so the allocation story that explained
#      `transform` and `cleanup` does not apply. It is Patchwork++ C++, so the
#      allocation cannot be removed from here. What CAN be established is whether
#      the tail is driven by the input or is machine noise -- which decides
#      whether there is anything to fix at all.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Is `ground`'s tail input-driven or noise?

Three questions, all on preloaded scans so no disk is in the timed loop:

  1. Does per-frame cost track POINT COUNT? If the tail is just bigger sweeps,
     it is not a defect and the fix is nothing.
  2. Does it track the GROUND-POINT count (scene content -- open road vs
     clutter)? Patchwork++ fits planes per sector, so more ground could cost
     more.
  3. Are the SAME frames slow across two passes with FRESH estimators each time?
     Fresh, because the estimator is stateful (D1) and reusing one would confound
     input-dependence with history.
"""
import sys
import time

import numpy as np
import pypatchworkpp as pw
from vrgrid.perception import loader
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200

print(f"preloading {N} scans of seq {SEQ}...")
scans = [np.asarray(p, dtype=np.float64) for p, _, _ in loader.scans(SEQ, max_frames=N)]
npts = np.array([len(s) for s in scans])
print(f"  {len(scans)} scans, points {npts.min():,}-{npts.max():,} "
      f"(median {int(np.median(npts)):,}, spread {npts.max()/npts.min():.2f}x)\n")


def est():
    p = pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    return pw.patchworkpp(p)


def one_pass():
    """Fresh estimator, so history cannot confound input-dependence."""
    e = est()
    for s in scans[:3]:
        e.estimateGround(s)
    ts, ng = [], []
    for s in scans:
        t0 = time.perf_counter()
        e.estimateGround(s)
        ts.append((time.perf_counter() - t0) * 1e3)
        ng.append(len(np.asarray(e.getGroundIndices(), dtype=np.int64)))
    return np.array(ts), np.array(ng)


a, ga = one_pass()
b, gb = one_pass()
print(f"  pass 1: p50 {np.median(a):6.2f}  p99 {np.percentile(a,99):6.2f}  max {a.max():6.2f}")
print(f"  pass 2: p50 {np.median(b):6.2f}  p99 {np.percentile(b,99):6.2f}  max {b.max():6.2f}")
print(f"  ground points: {ga.min():,}-{ga.max():,} "
      f"({ga.max()/max(ga.min(),1):.2f}x spread)\n")

print("1. DOES COST TRACK POINT COUNT?")
print(f"   corr(time, n_points)       pass1 {np.corrcoef(a, npts)[0,1]:+.3f}   "
      f"pass2 {np.corrcoef(b, npts)[0,1]:+.3f}")
print("2. DOES COST TRACK GROUND-POINT COUNT?")
print(f"   corr(time, n_ground)       pass1 {np.corrcoef(a, ga)[0,1]:+.3f}   "
      f"pass2 {np.corrcoef(b, gb)[0,1]:+.3f}")
print("3. ARE THE SAME FRAMES SLOW IN BOTH PASSES?")
k = 15
wa = set(np.argsort(a)[::-1][:k].tolist())
wb = set(np.argsort(b)[::-1][:k].tolist())
inter = wa & wb
exp = k * k / len(a)
print(f"   worst-{k} overlap: {len(inter)} of {k}   (chance would be ~{exp:.1f})")
print(f"   corr(pass1, pass2) per frame: {np.corrcoef(a, b)[0,1]:+.3f}")
print("   verdict: "
      + ("DATA-DEPENDENT -- the same frames are genuinely expensive"
         if len(inter) > 3 * exp else
         "NOISE -- the slow frames are different each pass"))

print("\n4. IS THE TAIL A FEW OUTLIERS OR A WIDE DISTRIBUTION?")
for lbl, v in (("pass1", a), ("pass2", b)):
    over = v > 1.5 * np.median(v)
    print(f"   {lbl}: {over.sum():>3} of {v.size} frames above 1.5x median "
          f"({over.mean()*100:4.1f}%), worst {v.max()/np.median(v):.2f}x median")
