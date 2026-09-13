# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           the recovery of R7's undocumented drivability predicate by enumerating all 63 bit masks: TRAV_SLOPE|TRAV_STEP, equivalently isinf(cost).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/find_r7_predicate.py costmaps.npz
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Which 'non-drivable' predicate reproduces R7's published counts?

Support already reproduces exactly (1,724 / 1,917 / 1,914) with the canonical
window, so the geometry is settled and only the predicate is unknown. This
enumerates every candidate against the published SHIPPED targets:

    seq 07  non-drivable 19, misses 8
    seq 08  non-drivable  3, misses 0
    seq 00  non-drivable 38, misses 4

If nothing matches all three, that is the answer and it gets reported as
"R7 could not be reproduced" rather than dressed up.
"""
import sys
import numpy as np
from vrgrid.eval.plan_regret import common_support

Z = np.load(sys.argv[1])
TARGET = {"07": (19, 8), "08": (3, 0), "00": (38, 4)}
SEQS = ("07", "08", "00")


class Fake:
    # All dumped maps share one lattice by construction (same window, same
    # call), so same_lattice is trivially true here.
    cell_m, x0_m, y0_m = 0.25, 0.0, 0.0

    def __init__(s, cost, unknown, trav):
        s.cost, s.unknown, s.trav = cost, unknown, trav

    def same_lattice(s, other):
        return True


def get(seq, cfg, which):
    return Fake(Z[f"{seq}_{cfg}_{which}_cost"],
                Z[f"{seq}_{cfg}_{which}_unknown"],
                Z[f"{seq}_{cfg}_{which}_trav"])


def support(seq, cfg):
    return common_support(get(seq, cfg, "map"), get(seq, cfg, "ref"))


preds = {}
for bits in range(1, 64):
    preds[f"trav & 0b{bits:06b}"] = lambda o, b=bits: (np.asarray(o.trav) & b) != 0
preds["isinf(cost)"] = lambda o: ~np.isfinite(np.asarray(o.cost))
preds["cost > 1"] = lambda o: np.asarray(o.cost) > 1.0
preds["cost > 10"] = lambda o: np.asarray(o.cost) > 10.0
preds["cost > 100"] = lambda o: np.asarray(o.cost) > 100.0

hits = []
for name, f in preds.items():
    row = {}
    for seq in SEQS:
        sup = support(seq, "shipped")
        nd = f(get(seq, "shipped", "ref")) & sup
        dm = ~f(get(seq, "shipped", "map")) & sup
        row[seq] = (int(nd.sum()), int((nd & dm).sum()))
    if all(row[s] == TARGET[s] for s in SEQS):
        hits.append((name, row))

print(f"loaded {len(Z.files)} arrays; support: "
      + ", ".join(f"{s}={int(support(s,'shipped').sum()):,}" for s in SEQS))
print("\ntargets: " + "  ".join(f"{s} {TARGET[s]}" for s in SEQS))

if hits:
    print(f"\nEXACT MATCHES ({len(hits)}):")
    for name, row in hits:
        print(f"  {name:<22}" + "  ".join(f"{s} {row[s]}" for s in SEQS))
else:
    print("\nNO predicate reproduces all three. Closest by total error:")
    scored = []
    for name, f in preds.items():
        row, err = {}, 0
        for seq in SEQS:
            sup = support(seq, "shipped")
            nd = f(get(seq, "shipped", "ref")) & sup
            mi = int((nd & (~f(get(seq, "shipped", "map")) & sup)).sum())
            row[seq] = (int(nd.sum()), mi)
            err += abs(row[seq][0] - TARGET[seq][0]) + abs(row[seq][1] - TARGET[seq][1])
        scored.append((err, name, row))
    for err, name, row in sorted(scored)[:8]:
        print(f"  err {err:>4}  {name:<22}" + "  ".join(f"{s} {row[s]}" for s in SEQS))
