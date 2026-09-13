# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           R7 shipped vs proposed with the confirmed predicate (gate PASS on all three).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/r7_final.py costmaps.npz
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""R7 shipped vs proposed, using the CONFIRMED predicate (TRAV_SLOPE|TRAV_STEP,
equivalently isinf(cost)) that reproduces the published counts exactly."""
import sys
import numpy as np
from vrgrid.eval.plan_regret import common_support

Z = np.load(sys.argv[1])
BITS = 0b000110          # TRAV_SLOPE | TRAV_STEP
PUB = {"07": (1724, 19, 8), "08": (1917, 3, 0), "00": (1914, 38, 4)}

class Fake:
    cell_m, x0_m, y0_m = 0.25, 0.0, 0.0
    def __init__(s, c, u, t): s.cost, s.unknown, s.trav = c, u, t
    def same_lattice(s, o): return True

def get(seq, cfg, w):
    return Fake(Z[f"{seq}_{cfg}_{w}_cost"], Z[f"{seq}_{cfg}_{w}_unknown"],
                Z[f"{seq}_{cfg}_{w}_trav"])

def nd(o, sup): return ((np.asarray(o.trav) & BITS) != 0) & sup

print(f"  {'seq':<5}{'config':<10}{'support':>9}{'non-driv':>10}{'misses':>8}"
      f"{'miss rate':>11}{'false al':>10}{'gate':>7}")
for seq in ("07", "08", "00"):
    ps, pn, pm = PUB[seq]
    for cfg in ("shipped", "proposed"):
        sup = common_support(get(seq, cfg, "map"), get(seq, cfg, "ref"))
        r = nd(get(seq, cfg, "ref"), sup)
        m = nd(get(seq, cfg, "map"), sup)
        mi, fa = int((r & ~m).sum()), int((~r & m).sum())
        rate = f"{mi/max(int(r.sum()),1)*100:.2f}%"
        g = ("PASS" if (int(sup.sum()), int(r.sum()), mi) == (ps, pn, pm)
             else "fail") if cfg == "shipped" else ""
        print(f"  {seq:<5}{cfg:<10}{int(sup.sum()):>9,}{int(r.sum()):>10}"
              f"{mi:>8}{rate:>11}{fa:>10}{g:>7}")
