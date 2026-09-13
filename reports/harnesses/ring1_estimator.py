# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/ring1-reproduction-investigation.md
#           the shared vs fresh_phase comparison: seq 07 3.60 -> 3.04 cm.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/ring1_estimator.py 07,08,00 shared,fresh_phase
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Does the Patchwork++ singleton call pattern explain the ring-1 mismatch?

Published R1 (main @ 9b40ff2):   07 1.78 / 3.60 / 5.91   08 1.17 / 2.31 / 4.89
                                 00 2.74 / 6.77 / 34.10
My numiter harness reproduced:   07 1.77 / 3.04 / 5.91   08 1.17 / 2.31 / 4.89
                                 00 2.74 / 6.46 / 34.10

Rings 0 and 2 match exactly everywhere and seq 08 matches at every ring; ring 1
is off by -0.56 (07) and -0.31 (00). The eval path is byte-identical to
9b40ff2, so this is not code drift.

The one thing my harness did differently is the ESTIMATOR CALL PATTERN. It
installed a fresh Patchwork++ estimator before the M* build AND again before the
map build, to keep the stateful singleton from carrying history between them.
The normal path does not: `ground._get_estimator()` builds ONE estimator lazily
and reuses it for the whole process, so the map pass inherits whatever state the
M* pass left behind.

Three modes, same code otherwise:

  shared       one estimator for both phases  -- what the normal path does,
                                                 and what the published run did
  fresh_phase  a fresh estimator per phase    -- what my harness did
  fresh_frame  a fresh estimator per FRAME    -- maximum isolation; if state
                                                 matters at all, this is the
                                                 far end of the effect

If `shared` reproduces 3.60 / 6.77 and `fresh_phase` gives 3.04 / 6.46, the
mismatch IS the singleton's statefulness, and it is the same root cause as the
determinism gate failure -- found twice, independently.

MEASUREMENT ONLY. ground.py is not modified; the singleton is manipulated from
here.
"""
import sys

import pypatchworkpp as _pw
from vrgrid.eval.harness import build_gridmap, real_scans, run_sequence
from vrgrid.eval.metrics import height_rmse_per_ring
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

FRAMES, SCHEDULE = 40, "5/10/20/40"
PUBLISHED = {"07": (1.78, 3.60, 5.91), "08": (1.17, 2.31, 4.89),
             "00": (2.74, 6.77, 34.10)}
_ORIGINAL_GET = ground._get_estimator


def build_estimator():
    p = _pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    return _pw.patchworkpp(p)


def measure(seq, mode):
    # always start from a clean slate so no mode inherits another's state
    ground._get_estimator = _ORIGINAL_GET
    ground._estimator = None

    if mode == "fresh_frame":
        # every call to the segmenter gets its own estimator
        ground._get_estimator = build_estimator
    elif mode == "shared":
        ground._estimator = build_estimator()      # one, for both phases
    elif mode == "fresh_phase":
        ground._estimator = build_estimator()
    else:
        raise ValueError(mode)

    reference = build_from_scans(real_scans(seq, FRAMES))

    if mode == "fresh_phase":
        ground._estimator = build_estimator()      # the map pass starts clean

    gm = build_gridmap(load(SCHEDULE))
    run_sequence(gm, real_scans(seq, FRAMES))
    d = height_rmse_per_ring(gm, reference)
    return [float(d[k]) for k in sorted(d)]


seqs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["07", "08", "00"]
modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["shared", "fresh_phase"]

print(f"{FRAMES} frames, schedule {SCHEDULE}, RMSE in cm\n")
print(f"  {'seq':<5}{'mode':<14}{'ring0':>8}{'ring1':>8}{'ring2':>8}"
      f"{'ring1 vs published':>21}")
for seq in seqs:
    p0, p1, p2 = PUBLISHED[seq]
    print(f"  {seq:<5}{'PUBLISHED':<14}{p0:>8.2f}{p1:>8.2f}{p2:>8.2f}")
    for mode in modes:
        r = measure(seq, mode)
        d1 = r[1] - p1
        tag = "MATCH" if abs(d1) <= 0.02 else f"{d1:+.2f}"
        print(f"  {seq:<5}{mode:<14}{r[0]:>8.2f}{r[1]:>8.2f}{r[2]:>8.2f}{tag:>21}",
              flush=True)
