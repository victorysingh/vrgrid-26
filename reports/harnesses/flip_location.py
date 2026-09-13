# PROVENANCE -- committed 2026-09-13 under OPEN-ITEMS.md item R-a.
#
# Produced: reports/numiter-tradeoff-accuracy-cost.md
#           where the flips land: slope 2.3-4.5x concentration, curbs clean, vegetation ~half of flips.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/flip_location.py 07 40
#
# NOTE: Passes all FOUR columns, as ground.segment_ground does, so enable_RNR is
#      live. Passing 3 silently disables RNR in both arms and changes the answer.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Item 1: WHERE do the num_iter=2 ground-verdict flips land?

Answers the question that decides the proposal: are the 0.72% of flipped points
spread evenly, or concentrated on curbs and slopes -- the two places where a
wrong ground verdict does real damage?

MEASUREMENT ONLY. `ground.py` untouched; both estimators are built here.

Four cuts, all per-point over the same frames:
  1. DIRECTION -- ground->non-ground loses surface samples; non-ground->ground
     contaminates the surface with facades and vegetation. Different damage.
  2. CLASS -- flip rate WITHIN each class, which is the cut that shows
     disproportion. A class that is 2% of points but 30% of flips matters.
  3. SLOPE -- local surface gradient from a 2 m mean-height grid.
  4. CURB PROXY -- SemanticKITTI has no `curb` class, so a curb is the
     road/sidewalk boundary: 0.5 m cells containing BOTH a road-ish and a
     sidewalk label.
"""
import sys

import numpy as np
import pypatchworkpp as _pw
from vrgrid.perception import loader, semantics
from vrgrid.perception.transforms import SENSOR_HEIGHT_M

SEQ = sys.argv[1] if len(sys.argv) > 1 else "07"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40

NAMES = {0: "car", 1: "bicycle", 2: "motorcycle", 3: "truck", 4: "other-vehicle",
         5: "person", 6: "bicyclist", 7: "motorcyclist", 8: "road",
         9: "parking", 10: "sidewalk", 11: "other-ground", 12: "building",
         13: "fence", 14: "vegetation", 15: "trunk", 16: "terrain",
         17: "pole", 18: "traffic-sign", 31: "unlabelled"}
ROADISH = {8, 9}        # road, parking
WALKISH = {10, 11}      # sidewalk, other-ground


def est(proposed: bool):
    p = _pw.Parameters()
    p.sensor_height = SENSOR_HEIGHT_M
    p.verbose = False
    if proposed:
        p.num_iter = 2
        p.enable_RNR = False
        p.enable_RVPF = False
    return _pw.patchworkpp(p)


def mask_of(e, pts64, n):
    e.estimateGround(pts64)
    m = np.zeros(n, bool)
    m[np.asarray(e.getGroundIndices(), dtype=np.int64)] = True
    return m


scans = []
for pts, raw, _ in loader.scans(SEQ, max_frames=N):
    scans.append((np.asarray(pts, dtype=np.float64),
                  semantics.semantic_labels(raw)))

a, b = est(False), est(True)
for p, _ in scans[:3]:
    a.estimateGround(p); b.estimateGround(p)

tot = fl = 0
g2n = n2g = 0
per_class = {}                      # cls -> [n_points, n_flips]
band_edges = [0, 10, 25, 50, 100, 1e9]
band = np.zeros((len(band_edges) - 1, 2))
slope_edges = [0, 0.02, 0.05, 0.10, 0.20, 1e9]
slope = np.zeros((len(slope_edges) - 1, 2))
curb = np.zeros((2, 2))             # [not-boundary, boundary] x [n, flips]

for pts, sem in scans:
    n = len(pts)
    ma, mb = mask_of(a, pts, n), mask_of(b, pts, n)
    f = ma != mb
    tot += n; fl += int(f.sum())
    g2n += int((ma & ~mb).sum()); n2g += int((~ma & mb).sum())

    for c in np.unique(sem):
        sel = sem == c
        e = per_class.setdefault(int(c), [0, 0])
        e[0] += int(sel.sum()); e[1] += int(f[sel].sum())

    r = np.hypot(pts[:, 0], pts[:, 1])
    bi = np.digitize(r, band_edges) - 1
    for i in range(len(band)):
        s = bi == i
        band[i, 0] += s.sum(); band[i, 1] += f[s].sum()

    # local slope from a 2 m mean-height grid over shipped-ground points only
    gx = np.floor(pts[:, 0] / 2.0).astype(np.int64)
    gy = np.floor(pts[:, 1] / 2.0).astype(np.int64)
    ox, oy = gx.min(), gy.min()
    W, H = int(gx.max() - ox + 3), int(gy.max() - oy + 3)
    zs = np.full((W, H), np.nan)
    ix, iy = gx - ox, gy - oy
    gp = ma
    np.add.at(zs, (ix[gp], iy[gp]), 0)          # touch cells so they exist
    acc = np.zeros((W, H)); cnt = np.zeros((W, H))
    np.add.at(acc, (ix[gp], iy[gp]), pts[gp, 2])
    np.add.at(cnt, (ix[gp], iy[gp]), 1)
    with np.errstate(invalid="ignore"):
        zs = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    dzx = np.full_like(zs, np.nan); dzy = np.full_like(zs, np.nan)
    dzx[1:-1, :] = (zs[2:, :] - zs[:-2, :]) / 4.0
    dzy[:, 1:-1] = (zs[:, 2:] - zs[:, :-2]) / 4.0
    grad = np.hypot(np.nan_to_num(dzx), np.nan_to_num(dzy))
    pslope = grad[ix, iy]
    si = np.digitize(pslope, slope_edges) - 1
    for i in range(len(slope)):
        s = si == i
        slope[i, 0] += s.sum(); slope[i, 1] += f[s].sum()

    # curb proxy: 0.5 m cells holding both a road-ish and a sidewalk-ish label
    cx = np.floor(pts[:, 0] / 0.5).astype(np.int64)
    cy = np.floor(pts[:, 1] / 0.5).astype(np.int64)
    key = (cx - cx.min()) * 100000 + (cy - cy.min())
    roadk = np.unique(key[np.isin(sem, list(ROADISH))])
    walkk = np.unique(key[np.isin(sem, list(WALKISH))])
    both = np.intersect1d(roadk, walkk, assume_unique=True)
    isb = np.isin(key, both)
    curb[1, 0] += isb.sum();  curb[1, 1] += f[isb].sum()
    curb[0, 0] += (~isb).sum(); curb[0, 1] += f[~isb].sum()

print(f"\nseq {SEQ}, {len(scans)} frames, {tot:,} points")
print(f"OVERALL flip rate: {fl/tot*100:.2f}%   "
      f"({fl:,} points: {g2n:,} ground->non-ground, {n2g:,} non-ground->ground)")

print("\n-- 2. per-class, sorted by flip rate within class --")
print(f"  {'class':<16}{'n':>11}{'flips':>9}{'rate':>8}{'share of flips':>16}")
for c, (npt, nf) in sorted(per_class.items(), key=lambda kv: -kv[1][1] / max(kv[1][0], 1)):
    if npt < 2000:
        continue
    print(f"  {NAMES.get(c, str(c)):<16}{npt:>11,}{nf:>9,}"
          f"{nf/npt*100:>7.2f}%{nf/max(fl,1)*100:>15.1f}%")

print("\n-- 3. by range band (ring boundaries) --")
for i, (lo, hi) in enumerate(zip(band_edges[:-1], band_edges[1:])):
    if band[i, 0] == 0:
        continue
    lbl = f"{lo:.0f}-{hi:.0f} m" if hi < 1e8 else f">{lo:.0f} m"
    print(f"  {lbl:<12}{int(band[i,0]):>11,}{int(band[i,1]):>9,}"
          f"{band[i,1]/band[i,0]*100:>7.2f}%")

print("\n-- 4. by local surface slope (2 m grid gradient) --")
for i, (lo, hi) in enumerate(zip(slope_edges[:-1], slope_edges[1:])):
    if slope[i, 0] == 0:
        continue
    lbl = f"{lo*100:.0f}-{hi*100:.0f}%" if hi < 1e8 else f">{lo*100:.0f}%"
    print(f"  slope {lbl:<10}{int(slope[i,0]):>11,}{int(slope[i,1]):>9,}"
          f"{slope[i,1]/slope[i,0]*100:>7.2f}%")

print("\n-- 5. curb proxy (0.5 m cell holding both road-ish and sidewalk-ish) --")
for i, lbl in enumerate(("away from boundary", "AT road/sidewalk boundary")):
    if curb[i, 0] == 0:
        continue
    print(f"  {lbl:<28}{int(curb[i,0]):>11,}{int(curb[i,1]):>9,}"
          f"{curb[i,1]/curb[i,0]*100:>7.2f}%")
