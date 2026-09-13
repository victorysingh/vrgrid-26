# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: pending-review/patchworkpp-num-iter-tradeoff.md
#           combined configs A (-5.42 ms, 0.72% flipped) and B (-9.30 ms, 1.72%).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/pw_combo.py 08 40
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Combined conservative Patchwork++ configs. MEASUREMENT ONLY, own estimators."""
import sys
import time
import numpy as np
import pypatchworkpp as pw
from vrgrid.perception import loader
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

scans = [p for p, _, _ in loader.scans(sys.argv[1], max_frames=int(sys.argv[2]))]

def run(label, mutate=None):
    p = pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    if mutate:
        mutate(p)
    est = pw.patchworkpp(p)
    for s in scans[:3]:
        est.estimateGround(np.asarray(s, dtype=np.float64))
    masks, ts = [], []
    for s in scans:
        a = np.asarray(s, dtype=np.float64)
        t0 = time.perf_counter()
        est.estimateGround(a)
        ts.append((time.perf_counter()-t0)*1e3)
        m = np.zeros(len(a), bool)
        m[np.asarray(est.getGroundIndices(), dtype=np.int64)] = True
        masks.append(m)
    return label, float(np.median(ts)), masks

def combo_a(p):
    p.num_iter = 2
    p.enable_RNR = False
    p.enable_RVPF = False
def combo_b(p):
    p.num_iter = 1
    p.enable_RNR = False
    p.enable_RVPF = False

rows = [run("SHIPPED"), run("A: num_iter2+RNRoff+RVPFoff", combo_a),
        run("B: num_iter1+RNRoff+RVPFoff", combo_b)]
base = rows[0][2]
b = rows[0][1]
for lab, ms, mk in rows:
    d = "" if lab == "SHIPPED" else f"{ms-b:+8.2f} ms   {np.mean([np.mean(x!=y) for x,y in zip(mk,base)])*100:5.2f}% flipped"
    print(f"  {lab:<30}{ms:7.2f} ms {d}")
