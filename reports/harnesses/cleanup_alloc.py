# PROVENANCE -- written 2026-09-13 under OPEN-ITEMS.md item R-h.
#
# Produced: reports/r-b-p99-tail-investigation.md, the attribution of the
#           `cleanup` stage's 9.61 MB/frame to candidate SELECTION rather than to
#           visibility_cleanup itself.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/cleanup_alloc.py 08 30
#
# NOTE: visibility_cleanup's own docstring promises "no allocation when handed a
#      scratch", and the engine does hand it one. So the stage's 9.61 MB cannot be
#      coming from the eq.(32) pass -- it is in the lines BEFORE the call. This
#      measures which.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Where does the `cleanup` stage's 9.61 MB/frame come from?

`MapEngine._cleanup` is written allocation-consciously -- `out=`, `scratch=`,
`np.copyto`, preallocated `_cand` / `_cand_slots` / `_has_return`. Three lines
still allocate, and this measures each against a realistic occupied-cell count
taken from a real populated map:

    state == OCC_OCCUPIED        a bool mask over every allocated slot
    np.flatnonzero(...)          int64 indices -- the suspected large one
    np.isin(occupied, touched)   the guard; np.isin sorts internally
"""
import sys
import time
import tracemalloc

import numpy as np
from vrgrid.eval.harness import build_gridmap, real_scans, run_sequence
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.grid.fusion import occupancy_state
from vrgrid.grid.schedule import load, load_thresholds

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30

print(f"building a real map: {N} frames of seq {SEQ}...")
gm = build_gridmap(load("5/10/20/40"))
run_sequence(gm, real_scans(SEQ, N))

grid = gm.soa
print(f"  allocated slots: {len(grid['log_odds']):,}")


def measure(label, fn, reps=30):
    fn()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    v = np.array(ts)
    print(f"  {label:<34}{(peak-base)/1e6:>9.2f} MB{np.median(v):>10.3f} ms"
          f"{np.percentile(v,99):>10.3f} ms")
    return (peak - base) / 1e6


# a real state array, via the shipping kernel
state = np.asarray(occupancy_state(grid, load_thresholds()))
occ_n = int((state == OCC_OCCUPIED).sum())
print(f"  occupied cells: {occ_n:,} of {state.size:,} "
      f"({occ_n/state.size*100:.1f}%)\n")

print(f"  {'operation':<34}{'peak alloc':>12}{'p50':>13}{'p99':>13}")
print("  " + "-" * 72)
total = 0.0
total += measure("state == OCC_OCCUPIED", lambda: state == OCC_OCCUPIED)
total += measure("np.flatnonzero(mask)", lambda: np.flatnonzero(state == OCC_OCCUPIED))

occupied = np.flatnonzero(state == OCC_OCCUPIED)
touched = occupied[::3].copy()          # a plausible 'touched this frame' subset
total += measure("np.isin(occupied, touched)", lambda: np.isin(occupied, touched))

print("  " + "-" * 72)
print(f"  {'sum of the three':<34}{total:>9.2f} MB")
print("\n  stage_allocation.py measured the whole `cleanup` stage at 9.61 MB.")

print("\n  ALLOCATION-FREE ALTERNATIVES (probes, not patches):")
mask_buf = np.empty(state.shape, dtype=bool)
idx_buf = np.empty(state.size, dtype=np.int64)


def prealloc_select():
    np.equal(state, OCC_OCCUPIED, out=mask_buf)
    n = np.count_nonzero(mask_buf)
    np.nonzero(mask_buf)[0]            # still allocates -- see the note printed below
    return n


measure("np.equal(..., out=) only", lambda: np.equal(state, OCC_OCCUPIED, out=mask_buf))
sorted_touched = np.sort(touched)
total_g = measure("np.searchsorted guard (vs isin)",
                  lambda: np.searchsorted(sorted_touched, occupied).clip(
                      0, len(sorted_touched) - 1))
print("\n  np.equal(out=) removes the mask allocation entirely.")
print("  np.nonzero has NO out= in numpy, so the index array is the hard part --")
print("  it needs either a persistent buffer plus a manual fill, or accepting it.")
