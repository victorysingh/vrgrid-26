# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           the ring-0 gate and the +19-54% RMSE result.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/numiter_accuracy.py both 07 08 00
#
# NOTE: height_rmse_per_ring returns CENTIMETRES and a {ring: rmse} dict. An
#      earlier version multiplied by 100 and produced a table exactly 100x out
#      but otherwise plausible. The gate caught it; keep the gate.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Item 1: the ACCURACY half of the num_iter=2 tradeoff.

Runs the published accuracy chain twice -- once with the shipped Patchwork++
configuration, once with the proposed one -- and reports ring-0 RMSE for both.

MEASUREMENT ONLY. `src/perception/ground.py` is NOT modified. The proposed
configuration is installed by assigning `ground._estimator`, the module-level
singleton, before any call. A FRESH estimator is built per configuration
because that singleton is stateful (the known determinism bug), so a shared one
would let config A's history leak into config B's numbers.

Gate: the shipped run must reproduce the published ring-0 RMSE
(00 2.74 / 07 1.78 / 08 1.17 cm, 40 frames, schedule 5/10/20/40) before any
delta is believed. If the gate fails, nothing downstream means anything.

Usage:  numiter_accuracy.py gate            # baseline only
        numiter_accuracy.py both 07 08 00   # both configs, named sequences
"""
import sys

import numpy as np
import pypatchworkpp as _pw
from vrgrid.eval.harness import build_gridmap, real_scans, run_sequence
from vrgrid.eval.metrics import height_rmse_per_ring
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

FRAMES = 40
SCHEDULE = "5/10/20/40"
PUBLISHED = {"00": 2.74, "07": 1.78, "08": 1.17}   # ring-0 ALL RMSE, cm


def install(config: str):
    """Force `ground`'s singleton to a freshly built estimator."""
    p = _pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    if config == "proposed":
        p.num_iter = 2
        p.enable_RNR = False
        p.enable_RVPF = False
    elif config != "shipped":
        raise ValueError(config)
    ground._estimator = _pw.patchworkpp(p)


def ring0_rmse(seq: str, config: str) -> float:
    install(config)
    reference = build_from_scans(real_scans(seq, FRAMES))
    install(config)          # rebuild: the map pass must not inherit M*'s state
    gm = build_gridmap(load(SCHEDULE))
    run_sequence(gm, real_scans(seq, FRAMES))
    per_ring = height_rmse_per_ring(gm, reference)
    r0 = per_ring[0]
    return float(r0 if np.isscalar(r0) else r0[0])   # already cm


mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
seqs = sys.argv[2:] or ["07", "08", "00"]

print(f"{FRAMES} frames, schedule {SCHEDULE}, ring-0 ALL RMSE in cm\n")
print(f"  {'seq':<5}{'published':>11}{'shipped':>10}{'gate':>7}"
      f"{'proposed':>11}{'delta':>9}")
for seq in seqs:
    ship = ring0_rmse(seq, "shipped")
    pub = PUBLISHED[seq]
    ok = "PASS" if abs(ship - pub) <= 0.05 else "FAIL"
    if mode == "gate":
        print(f"  {seq:<5}{pub:>11.2f}{ship:>10.2f}{ok:>7}")
        continue
    prop = ring0_rmse(seq, "proposed")
    print(f"  {seq:<5}{pub:>11.2f}{ship:>10.2f}{ok:>7}"
          f"{prop:>11.2f}{prop - ship:>+9.3f}")
