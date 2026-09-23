# PROVENANCE -- written 2026-09-23 for the RMSE-BASELINE investigation.
#
# Produced: reports/rmse-baseline-investigation-2026-09-23.md -- per-sequence and
#           per-ring ring-0 RMSE, with no gate, so the shift at df35fd5 can be
#           broken down instead of stopping at seq 07's failure.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/rmse_baseline_probe.py
#
# Measurement only. Reads the shipping code; changes nothing in src/. The
# comparison loop is transcribed from reports/harnesses/r1_accuracy_by_class.py
# (same builder, same frames, same schedule) with ONLY the gate removed, so the
# numbers are directly comparable to that harness's.
"""Ring-0 RMSE per sequence, without the gate that stops r1 at seq 07.

Runs against whichever tree it is imported from, so the same file can be pointed
at the current checkout and at a df35fd5~1 worktree to isolate that commit.
"""
import sys

import numpy as np
from vrgrid.eval import harness, metrics, reference_map

SEQS = ("07", "08", "00")
FRAMES = 40
SCHED = "5/10/20/40"
BASELINE_R0 = {"00": 2.73, "07": 1.76, "08": 1.16}   # r1's documented values


def per_sequence(seq):
    ref = reference_map.build_from_scans(harness.real_scans(seq, max_frames=FRAMES))
    gm = harness.build_gridmap(harness.load(SCHED))
    harness.run_sequence(gm, harness.real_scans(seq, max_frames=FRAMES))

    rmse = metrics.height_rmse_per_ring(gm, ref)
    # `_compared` is what the `serves` test lives in -- the thing df35fd5 changed.
    slots, n_ref, ref_mean, ref_var, mine = metrics._compared(gm, ref, 0)[:5]
    return rmse, int(np.size(slots)), int(np.sum(n_ref > 0))


def main():
    print(f"{'seq':<5}{'ring0':>9}{'doc':>8}{'delta%':>9}{'ring1':>9}{'ring2':>9}"
          f"{'ring3':>9}{'r0 cells':>10}{'r0 obs':>9}")
    print("-" * 77)
    for seq in SEQS:
        try:
            rmse, n_slots, n_obs = per_sequence(seq)
        except Exception as e:                      # a missing sequence is a fact, not a crash
            print(f"{seq:<5}  UNAVAILABLE: {type(e).__name__}: {str(e)[:48]}")
            continue
        r0 = rmse[0]
        want = BASELINE_R0[seq]
        d = (r0 - want) / want * 100.0
        cols = "".join(f"{rmse[L]:>9.4f}" if L < len(rmse) else f"{'-':>9}" for L in range(4))
        print(f"{seq:<5}{cols[:9]}{want:>8.2f}{d:>+9.1f}{cols[9:]}{n_slots:>10,}{n_obs:>9,}")
    print("\nring0 column is the gate quantity. `r0 cells` is how many ring-0 slots the")
    print("`serves` test routes to ring 0 -- df35fd5 changed that test, so a change in")
    print("this count is the direct mechanism rather than a downstream effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
