# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/r1-accuracy-by-class-and-range-band.md
#           the whole report: per-ring x per-class RMSE with n, and the ALL rows (07 1.78/3.60/5.91, 08 1.17/2.31/4.89, 00 2.74/6.77/34.10).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/r1_accuracy_by_class.py
#
# NOTE: [!] Runs all three sequences IN ONE PROCESS, so seq 00's Patchwork++
#      estimator carries ~160 frames of seq 07+08 history by the time it is
#      measured. That is the cause of the seq-00 ring-1 residual chased in
#      reports/ring1-reproduction-investigation.md -- see that report's
#      section on R-f. Do not 'fix' it here without restating the report.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""R1 -- per-range-band x per-class height accuracy against M*, with n.

Range band = ring (0-10 / 10-25 / 25-50 / 50-100 m), which is what the schedule
already partitions range into. Class = the semantic class the MAP stored for
that cell, unpacked from the 5-bit candidate field.

Uses metrics._compared, so the scored population is exactly the one
height_rmse_per_ring reports -- same exclusions, same M*, no second opinion.
Gated on ring-0 RMSE before anything is printed.
"""
import sys

import numpy as np
from vrgrid.eval import harness, metrics, reference_map
from vrgrid.grid.fusion import CLASS_UNLABELLED, unpack_class
from vrgrid.grid.traversability import class_ids

BASELINE_R0 = {"00": 2.73, "07": 1.76, "08": 1.16}
REACH = {0: "0-10 m", 1: "10-25 m", 2: "25-50 m", 3: "50-100 m"}


def name_table():
    ids = class_ids()
    out = {v: k for k, v in ids.items()}
    out[CLASS_UNLABELLED] = "unlabelled"
    return out


def run(seq, frames, sched_name):
    print(f"\n{'='*78}\nseq {seq}, {frames} frames, schedule {sched_name}\n{'='*78}")
    ref = reference_map.build_from_scans(harness.real_scans(seq, max_frames=frames))
    gm = harness.build_gridmap(harness.load(sched_name))
    harness.run_sequence(gm, harness.real_scans(seq, max_frames=frames))

    r0 = metrics.height_rmse_per_ring(gm, ref)[0]
    want = BASELINE_R0[seq]
    rel = abs(r0 - want) / want
    print(f"  [gate] ring-0 RMSE {r0:.4f} cm vs doc {want} ({rel*100:.1f}% apart)"
          f"  {'OK' if rel < 0.05 else '*** OUT OF TOLERANCE ***'}")
    if rel >= 0.05:
        sys.exit("gate failed")

    names = name_table()
    soa = gm.soa
    print(f"\n  {'band':<10}{'ring':<6}{'class':<16}{'n':>9}{'RMSE cm':>10}"
          f"{'mean bias':>11}{'share':>8}")
    print("  " + "-" * 68)
    rows = []
    for L in range(len(gm.schedule.rings)):
        slots, n_ref, ref_mean, ref_var, mine = metrics._compared(gm, ref, L)
        if not len(slots):
            print(f"  {REACH[L]:<10}{L:<6}{'(no scored cells)':<16}")
            continue
        cand, _ = unpack_class(soa["semantic_class"][slots])
        err = mine - ref_mean
        total = len(slots)
        # biggest classes first, so the table reads
        uniq, counts = np.unique(cand, return_counts=True)
        for cid in uniq[np.argsort(counts)[::-1]]:
            m = cand == cid
            n = int(m.sum())
            if n < 25:            # below this the RMSE is noise, not a number
                continue
            rmse = float(np.sqrt(np.mean(err[m] ** 2)))
            bias = float(np.mean(err[m]))
            nm = names.get(int(cid), f"id{int(cid)}")
            print(f"  {REACH[L]:<10}{L:<6}{nm:<16}{n:>9,}{rmse:>10.2f}"
                  f"{bias:>11.2f}{n/total:>7.0%}")
            rows.append((seq, sched_name, L, nm, n, rmse, bias))
        small = sum(int(c) for u, c in zip(uniq, counts) if c < 25)
        if small:
            print(f"  {'':<10}{'':<6}{'(classes n<25)':<16}{small:>9,}"
                  f"{'--':>10}{'--':>11}{small/total:>7.0%}")
        print(f"  {'':<10}{L:<6}{'ALL':<16}{total:>9,}"
              f"{float(np.sqrt(np.mean(err**2))):>10.2f}"
              f"{float(np.mean(err)):>11.2f}{1.0:>7.0%}")
        print("  " + "-" * 68)
    return rows


if __name__ == "__main__":
    allrows = []
    for seq in ("07", "08", "00"):
        allrows += run(seq, 40, "5/10/20/40")
    print(f"\n{'='*78}\nROLL-UP: same class across sequences, ring 0 and 1 only"
          f"\n{'='*78}")
    print(f"  {'ring':<6}{'class':<16}{'seq 07 n / RMSE':>22}"
          f"{'seq 08 n / RMSE':>22}{'seq 00 n / RMSE':>22}")
    idx = {(r[2], r[3], r[0]): r for r in allrows}
    classes = sorted({(r[2], r[3]) for r in allrows if r[2] <= 1})
    for L, nm in classes:
        cells = []
        for s in ("07", "08", "00"):
            r = idx.get((L, nm, s))
            cells.append(f"{r[4]:,} / {r[5]:.2f}" if r else "--")
        print(f"  {L:<6}{nm:<16}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")
