# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           the cached costmaps, so predicate variants can be tested without a 12-minute rebuild each time.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/dump_costmaps.py costmaps.npz
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Build the R7 costmaps once and dump them, so predicate variants can be
tested without a 12-minute rebuild each time. MEASUREMENT ONLY."""
import sys
import numpy as np
import pypatchworkpp as _pw
from vrgrid.eval.harness import (build_gridmap, final_vehicle_xy, real_scans,
                                 run_sequence)
from vrgrid.eval.plan_regret import costmap_from_gridmap, costmap_from_reference
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

FRAMES, SCHEDULE = 40, "5/10/20/40"
BEHIND, Y0, N = -11.0, -5.5, 44
OUT = sys.argv[1]

def install(c):
    p = _pw.Parameters(); p.sensor_height = SENSOR_HEIGHT_M; p.verbose = False
    if c == "proposed":
        p.num_iter, p.enable_RNR, p.enable_RVPF = 2, False, False
    ground._estimator = _pw.patchworkpp(p)

blobs = {}
for seq in ("07", "08", "00"):
    for c in ("shipped", "proposed"):
        install(c)
        ref = build_from_scans(real_scans(seq, FRAMES))
        install(c)
        gm = build_gridmap(load(SCHEDULE))
        run_sequence(gm, real_scans(seq, FRAMES))
        vx, vy = final_vehicle_xy(seq, FRAMES)
        x0, y0 = vx + BEHIND, vy + Y0
        cr = costmap_from_reference(ref, x0, y0, N, N)
        cmp_ = costmap_from_gridmap(gm, x0, y0, N, N, vehicle_xy_m=(vx, vy))
        for nm, o in (("ref", cr), ("map", cmp_)):
            blobs[f"{seq}_{c}_{nm}_trav"] = np.asarray(o.trav)
            blobs[f"{seq}_{c}_{nm}_cost"] = np.asarray(o.cost)
            blobs[f"{seq}_{c}_{nm}_unknown"] = np.asarray(o.unknown)
        print(f"  dumped {seq} {c}", flush=True)
np.savez_compressed(OUT, **blobs)
print("wrote", OUT)
