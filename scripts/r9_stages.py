#!/usr/bin/env python3
"""R9's remaining rows: split/merge and the pyramid, on real data. [Shrestha]

    python scripts/r9_stages.py --seq 08 --frames 200

`timing_table.py --seq` times the frame loop `MapEngine` runs: load, transform,
range image, semantics, ground, reflectivity, bin, scatter, fuse, cleanup,
shift. R9 (`vrgrid-recommended-changes.pdf` §7.2; roadmap Day 1) asks for
split/merge and the conservative pyramid as well, and neither is in that loop:

  split_merge   the §5 refinement pool, driven per cell by `gate.apply` --
                release overtaken blocks, fire the gate, acquire and SPLIT
                parents into 16 children. It runs in the eval harness's map,
                which carries a pool; `MapEngine` does not allocate one.
  traversability  the §7.1 bitfield over every ring. Timed because the
                pyramid reduces its output and must run after it.
  pyramid       §7.2's max/min/AND/OR reduction over every ring window,
                allocated with `allocate(with_pyramid=True)`.

So this drives `harness.run_sequence` -- the real pool, the real §7.1 update,
the real `pyramid.build` -- on the same frames, and times exactly those calls
by wrapping them. Nothing is reimplemented. The first 10 frames are discarded
as warm-up. Pool activity per frame is reported next to the latency, because a
split/merge figure without how many cells it split is not interpretable.
"""

import argparse
import sys
import time
import warnings

import numpy as np

WARMUP = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--schedule", default="5/10/20/40")
    ap.add_argument("--traversability-device", default="cpu", choices=["cpu", "cuda"],
                    help="compute the 7.1 bitfield on the card (bit-identical)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    import vrgrid.eval.harness as H
    from vrgrid.gpu.allocators import allocate
    from vrgrid.gpu.pyramid import build, pyramid_bytes, scratch_bytes
    from vrgrid.gpu.timing import Timer
    from vrgrid.grid import gate
    from vrgrid.grid.schedule import load, load_thresholds
    from vrgrid.grid.transient import TrackList

    sched = load(args.schedule)
    gm = H.build_gridmap(sched)
    gm.traversability_device = args.traversability_device
    # The pyramid is sized from the same ring layouts as the map it reduces.
    pyr = allocate(sched, load_thresholds(), commit_pages=False, with_pyramid=True).pyramid
    rings = gm.allocation.rings
    timer = Timer(stages=("split_merge", "traversability", "pyramid"),
                  capacity=args.frames + WARMUP + 8)
    frame = {"i": 0, "pending": None}
    pool_log = []

    real_apply, real_trav = gate.apply, H._update_traversability

    # One frame is traversability + pyramid, then the gate. The harness also
    # updates traversability once more after its loop; that call has no gate
    # after it, so it is never committed and never counted as a frame.
    def timed_trav(g):
        t0 = time.perf_counter()
        real_trav(g)            # writes host bits; the download synchronises
        t1 = time.perf_counter()
        build(pyr, g.soa, rings)
        frame["pending"] = ((t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3)

    def timed_apply(*a, **kw):
        t0 = time.perf_counter()
        out = real_apply(*a, **kw)
        dt = (time.perf_counter() - t0) * 1e3
        if frame["i"] >= WARMUP:
            trav_ms, pyr_ms = frame["pending"]
            timer.record("traversability", trav_ms)
            timer.record("pyramid", pyr_ms)
            timer.record("split_merge", dt)
            pool_log.append(out)
        frame["i"] += 1
        return out

    H.gate.apply = timed_apply
    H._update_traversability = timed_trav

    scans = H.real_scans(args.seq, args.frames + WARMUP)
    tracks = TrackList(gm.allocation.max_tracks, arrays=gm.allocation.tracks)
    t0 = time.perf_counter()
    H.run_sequence(gm, scans, tracks=tracks)
    wall = time.perf_counter() - t0

    s = timer.summary()
    print(f"sequence {args.seq}, {args.frames} frames after {WARMUP} warm-up, schedule "
          f"{args.schedule}, eval-harness map with refinement pool, traversability on "
          f"{args.traversability_device} ({wall:.0f} s wall)")
    print(f"\n  {'stage':<15} {'p50 ms':>8} {'p99 ms':>8} {'max ms':>8} {'n':>5}")
    for name in ("split_merge", "traversability", "pyramid"):
        r = s[name]
        print(f"  {name:<15} {r['p50_ms']:>8.2f} {r['p99_ms']:>8.2f} {r['max_ms']:>8.2f} {r['n']:>5}")

    keys = ("fired", "acquired", "released", "refused")
    tot = {k: sum(int(p.get(k, 0)) for p in pool_log) for k in keys}
    per = {k: np.array([int(p.get(k, 0)) for p in pool_log]) for k in keys}
    print("\n  refinement pool per frame (p50 / max):  " + "  ".join(
        f"{k} {int(np.median(per[k]))}/{int(per[k].max())}" for k in keys))
    print(f"  totals over {len(pool_log)} frames: " + ", ".join(f"{k} {v:,}" for k, v in tot.items()))
    print(f"  pool occupancy at the end: {gm.pool.blocks - gm.pool.free_blocks}/{gm.pool.blocks} blocks")
    print(f"\n  pyramid memory: {pyramid_bytes(rings) / 1e6:.2f} MB nodes + "
          f"{scratch_bytes(rings) / 1e6:.2f} MB scratch")
    print("\n  Not in `MapEngine`'s frame loop: add to its FRAME row only if the pool "
          "and pyramid are enabled in the deployed pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
