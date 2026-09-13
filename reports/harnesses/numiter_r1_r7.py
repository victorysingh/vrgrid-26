# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           the R1 per-ring table under both configs.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/numiter_r1_r7.py 07 08 00
#
# NOTE: Its R7 half uses a WRONG window (centred on the vehicle) and does not
#      reproduce the published counts. Superseded by numiter_r7.py. Kept because
#      the R1 half is what the report's 3a table came from.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Item 1 continued: R1 (per-ring RMSE) and R7 (hazard misses) under the
proposed num_iter=2 configuration.

MEASUREMENT ONLY. ground.py untouched; the estimator singleton is swapped here.

Both halves are GATED on reproducing the published shipped numbers first. If a
gate fails the corresponding comparison is reported as unreproducible rather
than guessed at.

R1 gate  ring-0 ALL RMSE  07 1.78 / 08 1.17 / 00 2.74 cm
R7 gate  support / non-drivable / misses
         07 1,724 / 19 / 8    08 1,917 / 3 / 0    00 1,914 / 38 / 4
"""
import sys

import numpy as np
import pypatchworkpp as _pw
from vrgrid.eval.harness import (build_gridmap, final_vehicle_xy, real_scans,
                                 run_sequence)
from vrgrid.eval.metrics import height_rmse_per_ring
from vrgrid.eval.plan_regret import (common_support, costmap_from_gridmap,
                                     costmap_from_reference)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

FRAMES = 40
SCHEDULE = "5/10/20/40"
WIN_M, CELL_M = 11.0, 0.25
N = int(round(WIN_M / CELL_M))          # 44 -> 1,936 cells
PUB_R1 = {"00": 2.74, "07": 1.78, "08": 1.17}
PUB_R7 = {"07": (1724, 19, 8), "08": (1917, 3, 0), "00": (1914, 38, 4)}


def install(config):
    p = _pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    if config == "proposed":
        p.num_iter = 2
        p.enable_RNR = False
        p.enable_RVPF = False
    ground._estimator = _pw.patchworkpp(p)


def measure(seq, config):
    install(config)
    reference = build_from_scans(real_scans(seq, FRAMES))
    install(config)
    gm = build_gridmap(load(SCHEDULE))
    run_sequence(gm, real_scans(seq, FRAMES))

    d = height_rmse_per_ring(gm, reference)
    rmse = [float(d[k]) for k in sorted(d)]

    vx, vy = final_vehicle_xy(seq, FRAMES)
    x0, y0 = vx - WIN_M / 2, vy - WIN_M / 2
    cm_map = costmap_from_gridmap(gm, x0, y0, N, N, cell_m=CELL_M)
    cm_ref = costmap_from_reference(reference, x0, y0, N, N, cell_m=CELL_M)
    sup = common_support(cm_map, cm_ref)

    nd_ref = (cm_ref.trav != 0) & sup          # truly non-drivable
    dr_map = (cm_map.trav == 0) & sup          # map says drivable
    misses = int((nd_ref & dr_map).sum())
    false_alarms = int((~(cm_ref.trav != 0) & (cm_map.trav != 0) & sup).sum())
    return rmse, int(sup.sum()), int(nd_ref.sum()), misses, false_alarms


seqs = sys.argv[1:] or ["07", "08", "00"]
rows = {}
for seq in seqs:
    rows[seq] = {c: measure(seq, c) for c in ("shipped", "proposed")}

print("\n=== R1: per-ring RMSE (cm), 40 frames, 5/10/20/40 ===")
print(f"  {'seq':<5}{'config':<10}{'ring0':>8}{'ring1':>8}{'ring2':>8}{'ring3':>9}")
for seq in seqs:
    for c in ("shipped", "proposed"):
        r = rows[seq][c][0]
        cells = "".join(f"{v:>8.2f}" if np.isfinite(v) else f"{'--':>8}" for v in r[:3])
        r3 = f"{r[3]:>9.2f}" if len(r) > 3 and np.isfinite(r[3]) else f"{'--':>9}"
        print(f"  {seq:<5}{c:<10}{cells}{r3}")
    s, p = rows[seq]["shipped"][0][0], rows[seq]["proposed"][0][0]
    g = "PASS" if abs(s - PUB_R1[seq]) <= 0.05 else "FAIL"
    print(f"  {'':<5}{'gate ' + g:<10}ring-0 published {PUB_R1[seq]:.2f}  "
          f"shipped {s:.2f}  proposed {p:.2f}  delta {p - s:+.3f} "
          f"({(p / s - 1) * 100:+.0f}%)" if s else "")

print("\n=== R7: hazard misses on an 11 m planning window ===")
print(f"  {'seq':<5}{'config':<10}{'support':>9}{'non-driv':>10}"
      f"{'misses':>8}{'false alarms':>14}")
for seq in seqs:
    ps, pn, pm = PUB_R7[seq]
    for c in ("shipped", "proposed"):
        _, sup, nd, mi, fa = rows[seq][c]
        print(f"  {seq:<5}{c:<10}{sup:>9,}{nd:>10}{mi:>8}{fa:>14}")
    _, sup, nd, mi, _ = rows[seq]["shipped"]
    ok = (sup == ps and nd == pn and mi == pm)
    print(f"  {'':<5}{'gate ' + ('PASS' if ok else 'FAIL'):<10}"
          f"published {ps:,} / {pn} / {pm}")
