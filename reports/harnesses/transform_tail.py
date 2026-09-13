# PROVENANCE -- written 2026-09-13 under OPEN-ITEMS.md item R-b.
#
# Produced: reports/r-b-p99-tail-investigation.md
#           the isolation of the p99 tail to transforms.transform_points, and
#           the allocation accounting behind it.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/transform_tail.py 08 220
#
# NOTE: the warm-cache p99 probe attributed 84% of the frame's p99 excess to the
#      `transform` stage -- a stage doing FIXED work on FIXED-size data at a p50
#      of 3.1 ms that occasionally takes 25-33 ms. This asks whether that tail
#      reproduces with the pipeline removed entirely: same scans, preloaded, only
#      transform_points in the loop.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Does the `transform` tail reproduce in isolation, and is it allocation?

Three arms on identical preloaded data:

  as shipped      transforms.transform_points, exactly as the pipeline calls it
  no-float64-cast the same, but fed float64 input, so np.asarray is a no-op --
                  isolates the float32 -> float64 conversion
  preallocated    an allocation-free equivalent writing into buffers created
                  once. NOT a proposed patch, a probe: if the tail vanishes here
                  and not above, the tail is allocation.

Also counts bytes allocated per call, so the claim is arithmetic rather than
assertion.
"""
import sys
import time
import tracemalloc

import numpy as np
from vrgrid.perception import loader, transforms

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 220

print(f"preloading {N} scans of seq {SEQ} (no disk in the timed loop)...")
scans = [np.asarray(p, dtype=np.float32) for p, _, _ in loader.scans(SEQ, max_frames=N)]
poses = [pose for _, _, pose in loader.scans(SEQ, max_frames=N)]
T = transforms.sensor_to_world(poses[0], sequence=SEQ)
npts = int(np.median([len(s) for s in scans]))
print(f"  {len(scans)} scans, median {npts:,} points\n")


def timed(label, fn, data):
    for d in data[:3]:
        fn(d)
    ts = []
    for d in data:
        t0 = time.perf_counter()
        fn(d)
        ts.append((time.perf_counter() - t0) * 1e3)
    v = np.array(ts)
    over = (v > 3 * np.median(v)).sum()
    print(f"  {label:<26}p50 {np.median(v):6.2f}  p99 {np.percentile(v,99):7.2f}  "
          f"max {v.max():7.2f}  spikes>3xp50 {over:>3}/{v.size}")
    return v


def shipped(p):
    return transforms.transform_points(p[:, :3], T)


f64 = [np.asarray(s[:, :3], dtype=np.float64) for s in scans]

# allocation-free equivalent: one buffer set, reused. Probe only.
cap = max(len(s) for s in scans)
buf_h = np.ones((cap, 4), dtype=np.float64)
buf_out = np.empty((cap, 3), dtype=np.float64)


def prealloc(p):
    n = len(p)
    buf_h[:n, :3] = p[:, :3]
    np.dot(buf_h[:n], T[:3, :].T, out=buf_out[:n])
    return buf_out[:n]


print("timing, identical data, 3 warm calls each:")
a = timed("as shipped (float32 in)", shipped, scans)
b = timed("float64 in (no cast)", shipped, f64)
c = timed("preallocated (probe)", prealloc, f64)

print("\nallocation per call, measured with tracemalloc:")
for label, fn, data in (("as shipped", shipped, scans),
                        ("float64 in", shipped, f64),
                        ("preallocated", prealloc, f64)):
    fn(data[0])
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    fn(data[0])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"  {label:<26}peak +{(peak-base)/1e6:6.2f} MB on a "
          f"{len(data[0]):,}-point scan")

print(f"\n  ratio of tails: shipped max / prealloc max = {a.max()/max(c.max(),1e-9):.1f}x")
print(f"  shipped spends {a.sum():.0f} ms over {a.size} frames; "
      f"prealloc {c.sum():.0f} ms ({a.sum()/max(c.sum(),1e-9):.1f}x)")
