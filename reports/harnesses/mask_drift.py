# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/ring1-reproduction-investigation.md
#           per-sequence drift and its radial distribution -- all of it inside 25 m, which is why ring 2 is untouched.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/mask_drift.py
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""How much do pass-1 and pass-2 ground masks differ, per sequence, over the
40 frames the R1 measurement uses? This is the exact inconsistency that sits
between M* and the map in `shared` mode. MEASUREMENT ONLY."""
import numpy as np
import pypatchworkpp as _pw
from vrgrid.perception import loader
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

def est():
    p = _pw.Parameters(); p.sensor_height = SENSOR_HEIGHT_M; p.verbose = False
    return _pw.patchworkpp(p)

def mask(e, s):
    e.estimateGround(s)
    m = np.zeros(len(s), bool)
    m[np.asarray(e.getGroundIndices(), dtype=np.int64)] = True
    return m

print(f"  {'seq':<5}{'points':>12}{'differ':>9}{'rate':>9}"
      f"{'frames hit':>12}{'range of hits':>16}")
for seq in ("07", "08", "00"):
    scans = [np.asarray(p, dtype=np.float64)
             for p, _, _ in loader.scans(seq, max_frames=40)]
    e = est()
    p1 = [mask(e, s) for s in scans]
    p2 = [mask(e, s) for s in scans]
    per = [int((a != b).sum()) for a, b in zip(p1, p2)]
    tot = sum(len(s) for s in scans); d = sum(per)
    hit = [i for i, n in enumerate(per) if n]
    rng = f"{min(hit)}-{max(hit)}" if hit else "none"
    print(f"  {seq:<5}{tot:>12,}{d:>9,}{d/tot*100:>8.3f}%"
          f"{len(hit):>9}/40{rng:>16}")
    # where, radially, are the differing points? ring 1 is 10-25 m
    if hit:
        r = np.concatenate([np.hypot(s[:, 0], s[:, 1])[a != b]
                            for s, a, b in zip(scans, p1, p2)])
        bands = [(0, 10), (10, 25), (25, 50), (50, 1e9)]
        share = "  ".join(
            f"{lo:.0f}-{hi:.0f}m {((r >= lo) & (r < hi)).mean()*100:.0f}%"
            if hi < 1e8 else f">{lo:.0f}m {(r >= lo).mean()*100:.0f}%"
            for lo, hi in bands)
        print(f"       radial distribution of differing points: {share}")
