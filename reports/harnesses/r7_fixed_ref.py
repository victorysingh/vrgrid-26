# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           R7 with M* HELD FIXED -- misses 8->10 (07), 4->5 (00).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/r7_fixed_ref.py costmaps.npz
#
# NOTE: The methodological point of the report: rebuilding M* with the config
#      under test makes a WORSE ground mask look like a SAFER map, because a
#      reference that cannot see a hazard cannot record a miss against it.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""R7 with the reference HELD FIXED at the shipped ground mask.

Rebuilding M* with the proposed mask changes what we measure against -- the
reference's own non-drivable count moves (19->16 on seq 07), so both sides of
the A/B shift and a degradation can partly hide. Holding M* at the shipped mask
asks the cleaner question: with the best available ground truth fixed, does the
proposed map miss more hazards?
"""
import sys
import numpy as np
from vrgrid.eval.plan_regret import common_support

Z = np.load(sys.argv[1])
BITS = 0b000110

class Fake:
    cell_m, x0_m, y0_m = 0.25, 0.0, 0.0
    def __init__(s, c, u, t): s.cost, s.unknown, s.trav = c, u, t
    def same_lattice(s, o): return True

def get(seq, cfg, w):
    return Fake(Z[f"{seq}_{cfg}_{w}_cost"], Z[f"{seq}_{cfg}_{w}_unknown"],
                Z[f"{seq}_{cfg}_{w}_trav"])
def nd(o, sup): return ((np.asarray(o.trav) & BITS) != 0) & sup

print("  reference held FIXED at the shipped ground mask")
print(f"  {'seq':<5}{'map from':<12}{'support':>9}{'non-driv':>10}{'misses':>8}"
      f"{'miss rate':>11}{'false al':>10}")
for seq in ("07", "08", "00"):
    ref = get(seq, "shipped", "ref")
    for cfg in ("shipped", "proposed"):
        mp = get(seq, cfg, "map")
        sup = common_support(mp, ref)
        r, m = nd(ref, sup), nd(mp, sup)
        mi, fa = int((r & ~m).sum()), int((~r & m).sum())
        print(f"  {seq:<5}{cfg:<12}{int(sup.sum()):>9,}{int(r.sum()):>10}{mi:>8}"
              f"{mi/max(int(r.sum()),1)*100:>10.2f}%{fa:>10}")
