# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/ring1-reproduction-investigation.md
#           the scored-population check: seq 00 ring 1 differs by 61-89 cells while rings 0 and 2 match exactly.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/ring1_population.py 00,07
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Is seq 00's ring-1 residual a different POPULATION of scored cells, or the
same cells with different heights? Published n per ring is in the R1 report, so
comparing n discriminates directly. MEASUREMENT ONLY."""
import sys
import numpy as np
import pypatchworkpp as _pw
from vrgrid.eval.harness import build_gridmap, real_scans, run_sequence
from vrgrid.eval.metrics import _compared
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

FRAMES, SCHEDULE = 40, "5/10/20/40"
# published n and RMSE per ring, from reports/r1-accuracy-by-class-and-range-band.md
PUB = {"07": [(103182, 1.78), (50153, 3.60), (12703, 5.91)],
       "08": [(137034, 1.17), (141141, 2.31), (49073, 4.89)],
       "00": [(82868, 2.74), (41892, 6.77), (11275, 34.10)]}
_ORIG = ground._get_estimator

def build():
    p = _pw.Parameters(); p.sensor_height = SENSOR_HEIGHT_M; p.verbose = False
    return _pw.patchworkpp(p)

def run(seq, mode):
    ground._get_estimator = _ORIG
    ground._estimator = build()
    ref = build_from_scans(real_scans(seq, FRAMES))
    if mode == "fresh_phase":
        ground._estimator = build()
    gm = build_gridmap(load(SCHEDULE))
    run_sequence(gm, real_scans(seq, FRAMES))
    out = []
    for ring in range(3):
        _, _, ref_mean, _, mine = _compared(gm, ref, ring)
        rmse = float(np.sqrt(np.mean((mine - ref_mean) ** 2))) if mine.size else float("nan")
        out.append((int(mine.size), rmse))
    return out

for seq in sys.argv[1].split(","):
    print(f"\nseq {seq}")
    print(f"  {'mode':<14}" + "".join(f"{'ring'+str(r)+' n':>12}{'rmse':>8}" for r in range(3)))
    print(f"  {'PUBLISHED':<14}" + "".join(f"{n:>12,}{v:>8.2f}" for n, v in PUB[seq]))
    for mode in ("shared", "fresh_phase"):
        rows = run(seq, mode)
        line = "".join(f"{n:>12,}{v:>8.2f}" for n, v in rows)
        print(f"  {mode:<14}{line}", flush=True)
        for r in range(3):
            dn = rows[r][0] - PUB[seq][r][0]
            if dn:
                print(f"       ring{r}: n differs by {dn:+,} "
                      f"({dn/PUB[seq][r][0]*100:+.2f}%)")
