# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/latency-gap-investigation.md
#           the per-parameter speed/accuracy table (num_iter, RNR, TGR, RVPF, max_range, num_lpr).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/pw_params.py 08 40
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Patchwork++ parameter speed/accuracy tradeoff. MEASUREMENT ONLY.

Builds its OWN estimators. Does not import or modify src/perception/ground.py's
singleton, and applies nothing to the real pipeline.

Accuracy is measured as disagreement against the CURRENT shipped configuration
(what ground.py builds today), because there is no per-point ground-truth
ground mask -- SemanticKITTI gives classes, not a ground/non-ground label, and
`ground_from_semantics` is a class proxy, not truth. So "accuracy" here means
"how far does this drift from what we ship", which is the honest framing.
"""
import sys
import time

import numpy as np
import pypatchworkpp as pw
from vrgrid.perception import loader
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40

print(f"pypatchworkpp {getattr(pw, '__version__', '1.4.1 (per pip)')}")
print(f"preloading {N} scans of seq {SEQ}...")
scans = [pts for pts, _, _ in loader.scans(SEQ, max_frames=N)]
print(f"  {len(scans)} scans\n")


def base_params():
    p = pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    return p


def run(label, mutate=None):
    p = base_params()
    if mutate:
        mutate(p)
    est = pw.patchworkpp(p)
    # warm: the estimator is stateful (see the known determinism bug), so give
    # every configuration the same number of prior frames before timing.
    for s in scans[:3]:
        est.estimateGround(np.asarray(s, dtype=np.float64))
    masks, ts = [], []
    for s in scans:
        a = np.asarray(s, dtype=np.float64)
        t0 = time.perf_counter()
        est.estimateGround(a)
        ts.append((time.perf_counter() - t0) * 1e3)
        m = np.zeros(len(a), bool)
        m[np.asarray(est.getGroundIndices(), dtype=np.int64)] = True
        masks.append(m)
    return label, float(np.median(ts)), masks


rows = []
rows.append(run("SHIPPED (current default)"))
rows.append(run("num_iter 3 -> 2", lambda p: setattr(p, "num_iter", 2)))
rows.append(run("num_iter 3 -> 1", lambda p: setattr(p, "num_iter", 1)))
rows.append(run("enable_RNR off", lambda p: setattr(p, "enable_RNR", False)))
rows.append(run("enable_TGR off", lambda p: setattr(p, "enable_TGR", False)))
rows.append(run("enable_RVPF off", lambda p: setattr(p, "enable_RVPF", False)))
rows.append(run("max_range 80 -> 60", lambda p: setattr(p, "max_range", 60.0)))
rows.append(run("max_range 80 -> 50", lambda p: setattr(p, "max_range", 50.0)))
rows.append(run("num_lpr 20 -> 10", lambda p: setattr(p, "num_lpr", 10)))

base_masks = rows[0][2]
print(f"{'configuration':<28}{'p50 ms':>9}{'vs shipped':>12}{'pts changed':>13}")
print("-" * 62)
b = rows[0][1]
for label, ms, masks in rows:
    if label.startswith("SHIPPED"):
        print(f"{label:<28}{ms:>9.2f}{'--':>12}{'--':>13}")
        continue
    diff = np.mean([np.mean(m != bm) for m, bm in zip(masks, base_masks)])
    print(f"{label:<28}{ms:>9.2f}{ms - b:>+11.2f}{diff*100:>12.2f}%")
print("-" * 62)
print("'pts changed' = fraction of points whose ground/non-ground verdict flips\n"
      "relative to the shipped configuration. Not an accuracy number -- there is\n"
      "no per-point ground truth for 'is ground'.")
