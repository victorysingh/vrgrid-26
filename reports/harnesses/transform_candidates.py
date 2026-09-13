# PROVENANCE -- written 2026-09-14 for D9 (transform_points allocation fix).
#
# Produced: the choice of allocation-free transform_points implementation, and
#           the proof that it is BIT-IDENTICAL to the shipped one.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/transform_candidates.py 08 200
#
# NOTE: the method is the one agreed for D10 -- keep the SAME operations in the
#      SAME order and change only where results are written. The shipped body is
#
#          pts   = np.asarray(points, dtype=np.float64)
#          xyz   = pts[:, :3]
#          pts_h = np.hstack([xyz, np.ones((n, 1))])
#          return (T @ pts_h.T).T[:, :3]
#
#      The matmul is a BLAS call. A different call SHAPE -- (N,4) @ (4,3) instead
#      of (4,4) @ (4,N) -- can sum in a different order and differ by ULPs, which
#      is exactly what cost scatter_mean 2 ULP. So every candidate is checked for
#      bit-identity against the shipped function on every frame, and anything that
#      is not exactly identical is rejected however fast it is.
#
# Measurement only. Changes nothing in src/.
"""Allocation-free transform_points candidates: bit-identity first, then cost."""
import sys
import time
import tracemalloc

import numpy as np
from vrgrid.perception import loader, transforms

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
CAP = 150_000                     # configs/thresholds.yaml scatter.max_points_per_frame

items = list(loader.scans(SEQ, max_frames=N))
scans = [p for p, _, _ in items]
Ts = [transforms.sensor_to_world(pose, sequence=SEQ) for _, _, pose in items]
print(f"{len(scans)} scans of seq {SEQ}, points "
      f"{min(len(s) for s in scans):,}-{max(len(s) for s in scans):,}\n")

shipped = transforms.transform_points

# ---- B: SAME call shape as shipped, written into reused buffers ---------------
# pts_h is (N,4) C-order with the ones column set ONCE; pts_h[:n].T is the same
# F-contiguous (4,N) view the shipped code hands to matmul. out4 is F-order so
# out4[:, :n] is contiguous, and the returned .T[:, :3] is the same layout the
# shipped function returns.
b_h = np.ones((CAP, 4), dtype=np.float64)
b_out = np.empty((4, CAP), dtype=np.float64, order="F")


def cand_B(points, T):
    n = len(points)
    b_h[:n, :3] = points[:, :3]
    np.matmul(T, b_h[:n].T, out=b_out[:, :n])
    return b_out[:, :n].T[:, :3]


# ---- C: the earlier probe -- DIFFERENT call shape (N,4) @ (4,3) ---------------
c_h = np.ones((CAP, 4), dtype=np.float64)
c_out = np.empty((CAP, 3), dtype=np.float64)


def cand_C(points, T):
    n = len(points)
    c_h[:n, :3] = points[:, :3]
    np.dot(c_h[:n], T[:3, :].T, out=c_out[:n])
    return c_out[:n]


# ---- D: explicit element-wise, no BLAS ----------------------------------------
d_out = np.empty((CAP, 3), dtype=np.float64)
d_tmp = np.empty(CAP, dtype=np.float64)
d_x = np.empty(CAP, dtype=np.float64)


def cand_D(points, T):
    n = len(points)
    x = d_x[:n]
    np.copyto(x, points[:, 0], casting="unsafe")
    y = np.asarray(points[:, 1], dtype=np.float64)
    z = np.asarray(points[:, 2], dtype=np.float64)
    for r in range(3):
        o = d_out[:n, r]
        np.multiply(T[r, 0], x, out=o)
        np.multiply(T[r, 1], y, out=d_tmp[:n])
        np.add(o, d_tmp[:n], out=o)
        np.multiply(T[r, 2], z, out=d_tmp[:n])
        np.add(o, d_tmp[:n], out=o)
        np.add(o, T[r, 3], out=o)
    return d_out[:n]


cands = {"B same-shape+buffers": cand_B, "C (N,4)@(4,3)": cand_C, "D elementwise": cand_D}

print("1. BIT-IDENTITY against shipped, every frame (rejection criterion)")
identical = {}
for name, fn in cands.items():
    worst_ulp, frames_diff = 0.0, 0
    for p, T in zip(scans, Ts):
        ref = shipped(p[:, :3], T)
        got = np.array(fn(p, T))
        if not np.array_equal(ref, got):
            frames_diff += 1
            d = np.abs(ref - got)
            with np.errstate(divide="ignore", invalid="ignore"):
                ulp = np.nanmax(d / np.spacing(np.abs(ref)))
            worst_ulp = max(worst_ulp, float(ulp))
    identical[name] = frames_diff == 0
    verdict = ("BIT-IDENTICAL on all frames" if frames_diff == 0 else
               f"DIFFERS on {frames_diff}/{len(scans)} frames, worst {worst_ulp:.1f} ULP -- REJECTED")
    print(f"   {name:<24} {verdict}")

print("\n2. ALLOCATION per call (tracemalloc peak; bytes, unaffected by machine state)")


def alloc_of(fn):
    fn(scans[0], Ts[0])
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    fn(scans[0], Ts[0])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return (peak - base) / 1e6


print(f"   {'shipped':<24} {alloc_of(lambda p, T: shipped(p[:, :3], T)):6.2f} MB")
for name, fn in cands.items():
    print(f"   {name:<24} {alloc_of(fn):6.2f} MB")

print("\n3. TIME (only meaningful on a calibrated machine; A/B back to back)")


def time_of(fn, reps=2):
    for p, T in zip(scans[:3], Ts[:3]):
        fn(p, T)
    ts = []
    for _ in range(reps):
        for p, T in zip(scans, Ts):
            t0 = time.perf_counter()
            fn(p, T)
            ts.append((time.perf_counter() - t0) * 1e3)
    v = np.array(ts)
    return np.median(v), np.percentile(v, 99), v.max(), int((v > 3 * np.median(v)).sum()), v.size


rows = [("shipped", lambda p, T: shipped(p[:, :3], T))] + list(cands.items())
for name, fn in rows:
    p50, p99, mx, spikes, n = time_of(fn)
    tag = "" if name == "shipped" or identical.get(name) else "  (rejected: not identical)"
    print(f"   {name:<24} p50 {p50:6.2f}  p99 {p99:6.2f}  max {mx:6.2f}  "
          f"spikes>3x {spikes}/{n}{tag}")
