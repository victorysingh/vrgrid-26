# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/r7-hazard-miss-rate.md
#           the hazard-miss counts 8/19, 0/3, 4/38 and the false-alarm counts.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/r7_hazard_miss.py
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""R7 -- hazard miss rate with the correct denominator. MEASUREMENT ONLY.

    miss rate = |truly non-drivable AND map says drivable| / |truly non-drivable|

"Truly" = M*, via costmap_from_reference. "Map says" = costmap_from_gridmap.
Both on the same planning lattice, restricted to the common support, which is
exactly how eq. (23) pairs them -- so this is a like-for-like verdict
comparison, not a fill-rate measurement.

Gated on reproducing a known-good published number first: if the M* build is
wrong the miss rate is meaningless, and a wrong M* still produces a
confident-looking number.
"""
import sys

import numpy as np
from vrgrid.eval import harness, metrics, reference_map
from vrgrid.eval.plan_regret import (
    IMPASSABLE_BITS,
    common_support,
    costmap_from_gridmap,
    costmap_from_reference,
    restrict,
)

PLAN_BEHIND_M, PLAN_N, PLAN_Y0_M = -11.0, 44, -5.5
BASELINE_R0 = {"00": 2.73, "07": 1.76, "08": 1.16}   # known-limitations 2b


def build(seq, frames, sched_name):
    ref = reference_map.build_from_scans(harness.real_scans(seq, max_frames=frames))
    gm = harness.build_gridmap(harness.load(sched_name))
    harness.run_sequence(gm, harness.real_scans(seq, max_frames=frames))
    return ref, gm


def gate(seq, frames, ref, gm):
    """Refuse to report anything if M* does not reproduce the published RMSE."""
    r0 = metrics.height_rmse_per_ring(gm, ref)[0]
    want = BASELINE_R0.get(seq)
    # 5% tolerance, deliberately. PR #31 changed _ring_cells to filter on
    # ring_of(centre)==ring, so known-limitations 2b is legitimately stale --
    # eval_synthetic.py --seq 07 prints 1.78 today against the doc's 1.76
    # (1.1% apart). A ground-mask-less M*, the failure this gate exists to
    # catch, reads 22.10 cm on seq 07 -- an order of magnitude out. 5% passes
    # the first and catches the second.
    rel = abs(r0 - want) / want if want else float("nan")
    ok = want is not None and rel < 0.05
    print(f"  [gate] ring-0 RMSE {r0:.4f} cm vs doc {want if want else '--'} "
          f"({rel*100:.1f}% apart)  {'OK' if ok else '*** OUT OF TOLERANCE ***'}")
    if want is not None and not ok:
        sys.exit("gate failed -- M* is not trustworthy; nothing below would be either")
    return r0


def hazard_miss(seq, frames, sched_name):
    print(f"\n=== seq {seq}, {frames} frames, schedule {sched_name} ===")
    ref, gm = build(seq, frames, sched_name)
    gate(seq, frames, ref, gm)

    vx, vy = harness.final_vehicle_xy(seq, frames)
    x0, y0 = vx + PLAN_BEHIND_M, vy + PLAN_Y0_M
    star = costmap_from_reference(ref, x0, y0, PLAN_N, PLAN_N)
    mine = costmap_from_gridmap(gm, x0, y0, PLAN_N, PLAN_N, vehicle_xy_m=(vx, vy))

    mask = common_support(star, mine)

    # ⚑ NOT restrict(): it constructs CostMap without `trav`, which then
    #   defaults to None, so restrict(x).trav is None and the bitfield is
    #   gone. Mask the boolean verdicts directly instead -- same cells, and
    #   `trav` survives. (Latent defect in restrict(); no live caller hits it,
    #   because --confound reads low_confidence() off UNRESTRICTED maps.)
    truth_bad_full = (np.asarray(star.trav) & IMPASSABLE_BITS) != 0
    mine_bad_full = (np.asarray(mine.trav) & IMPASSABLE_BITS) != 0
    truth_bad = truth_bad_full[mask]
    mine_bad = mine_bad_full[mask]

    n_cells = int(truth_bad.size)
    n_truth_bad = int(truth_bad.sum())
    miss = truth_bad & ~mine_bad          # truly bad, we called it drivable
    false_alarm = ~truth_bad & mine_bad   # truly fine, we called it a hazard
    n_miss, n_fa = int(miss.sum()), int(false_alarm.sum())

    print(f"  cells in common support        {n_cells:,}")
    print(f"  truly NON-drivable (M*)        {n_truth_bad:,}"
          f"   ({n_truth_bad/max(n_cells,1)*100:.2f}% of support)")
    print(f"  map called drivable            {int((~mine_bad).sum()):,}")
    print(f"  MISSES (bad, called drivable)  {n_miss:,}")
    print(f"  false alarms (fine, called bad){n_fa:>8,}")
    if n_truth_bad:
        print(f"  >>> HAZARD MISS RATE           {n_miss/n_truth_bad*100:.2f}%"
              f"   ({n_miss}/{n_truth_bad})")
        print(f"      hazard recall              {(1-n_miss/n_truth_bad)*100:.2f}%")
    else:
        print("  >>> HAZARD MISS RATE           UNDEFINED -- M* has zero "
              "non-drivable cells in this window; the denominator is 0")
    return dict(seq=seq, frames=frames, sched=sched_name, cells=n_cells,
                truth_bad=n_truth_bad, miss=n_miss, fa=n_fa)


if __name__ == "__main__":
    rows = []
    for seq in ("07", "08", "00"):
        for sched in ("5/10/20/40", "5/10/50"):
            try:
                rows.append(hazard_miss(seq, 40, sched))
            except SystemExit:
                raise
            except Exception as e:
                print(f"  FAILED seq {seq} {sched}: {type(e).__name__}: {e}")
    print("\n=== SUMMARY ===")
    print(f"{'seq':>4} {'schedule':>11} {'support':>9} {'truly bad':>10} "
          f"{'miss':>6} {'miss rate':>10} {'false alarm':>12}")
    for r in rows:
        rate = f"{r['miss']/r['truth_bad']*100:.2f}%" if r["truth_bad"] else "UNDEF"
        print(f"{r['seq']:>4} {r['sched']:>11} {r['cells']:>9,} {r['truth_bad']:>10,} "
              f"{r['miss']:>6,} {rate:>10} {r['fa']:>12,}")
