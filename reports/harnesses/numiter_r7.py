# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           R7 with the canonical window, which reproduced support 1,724/1,917/1,914.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/numiter_r7.py 07 08 00
#
# NOTE: The planning window is NOT centred on the vehicle: x0 = vx - 11.0,
#      y0 = vy - 5.5, 44x44, i.e. entirely behind it, over ground it has driven.
#      Centring it drops seq 07 support from 1,724 to 955.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""R7 hazard misses under the proposed config -- using the CANONICAL window.

The first attempt centred the window on the vehicle and omitted `vehicle_xy_m`.
`eval_synthetic.costmaps_for` does neither: the window sits ENTIRELY BEHIND the
vehicle (x0 = vx - 11.0, y0 = vy - 5.5, 44x44 at the default cell size) which is
ground the vehicle has actually driven over and therefore observed, and the
gridmap costmap is given `vehicle_xy_m`. Both differences shrink support, which
is why the reconstruction missed the published counts.

Several candidate definitions of "non-drivable" are printed side by side,
because R7's exact bit mask is not recorded and guessing one and reporting it as
R7 would be presenting a guess as a finding. Whichever row reproduces
07 1,724/19/8, 08 1,917/3/0, 00 1,914/38/4 for the SHIPPED config is the
definition R7 used; if none does, that is the result and it is reported as such.

MEASUREMENT ONLY. ground.py untouched.
"""
import sys

import pypatchworkpp as _pw
from vrgrid.eval.harness import (build_gridmap, final_vehicle_xy, real_scans,
                                 run_sequence)
from vrgrid.eval.plan_regret import (common_support, costmap_from_gridmap,
                                     costmap_from_reference)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

FRAMES, SCHEDULE = 40, "5/10/20/40"
PLAN_BEHIND_M, PLAN_Y0_M, PLAN_N = -11.0, -5.5, 44
PUB = {"07": (1724, 19, 8), "08": (1917, 3, 0), "00": (1914, 38, 4)}

HAZARD = {                      # candidate "non-drivable" masks
    "any bit":            0b111111,
    "no confidence":      0b011111,
    "no conf/clearance":  0b011110,
    "slope|step|rough":   0b001110,
}


def install(config):
    p = _pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    if config == "proposed":
        p.num_iter, p.enable_RNR, p.enable_RVPF = 2, False, False
    ground._estimator = _pw.patchworkpp(p)


def maps(seq, config):
    install(config)
    reference = build_from_scans(real_scans(seq, FRAMES))
    install(config)
    gm = build_gridmap(load(SCHEDULE))
    run_sequence(gm, real_scans(seq, FRAMES))
    vx, vy = final_vehicle_xy(seq, FRAMES)
    x0, y0 = vx + PLAN_BEHIND_M, vy + PLAN_Y0_M
    cm_ref = costmap_from_reference(reference, x0, y0, PLAN_N, PLAN_N)
    cm_map = costmap_from_gridmap(gm, x0, y0, PLAN_N, PLAN_N,
                                  vehicle_xy_m=(vx, vy))
    return cm_ref, cm_map


seqs = sys.argv[1:] or ["07", "08", "00"]
out = {}
for seq in seqs:
    out[seq] = {c: maps(seq, c) for c in ("shipped", "proposed")}

print("\n=== R7 reconstruction, canonical window (x0=vx-11, y0=vy-5.5, 44x44) ===")
for seq in seqs:
    ps, pn, pm = PUB[seq]
    print(f"\nseq {seq}   published: support {ps:,}  non-drivable {pn}  misses {pm}")
    print(f"  {'hazard mask':<20}{'config':<10}{'support':>9}{'non-driv':>10}"
          f"{'misses':>8}{'false al':>10}{'gate':>7}")
    for name, bits in HAZARD.items():
        for c in ("shipped", "proposed"):
            cm_ref, cm_map = out[seq][c]
            sup = common_support(cm_map, cm_ref)
            nd_ref = ((cm_ref.trav & bits) != 0) & sup
            nd_map = ((cm_map.trav & bits) != 0) & sup
            mi = int((nd_ref & ~nd_map).sum())
            fa = int((~nd_ref & nd_map).sum())
            g = ""
            if c == "shipped":
                g = "PASS" if (int(sup.sum()), int(nd_ref.sum()), mi) == (ps, pn, pm) else "fail"
            print(f"  {name:<20}{c:<10}{int(sup.sum()):>9,}{int(nd_ref.sum()):>10}"
                  f"{mi:>8}{fa:>10}{g:>7}")
