# PROVENANCE -- written 2026-09-13 under OPEN-ITEMS.md item R-b.
#
# Produced: reports/r-b-p99-tail-investigation.md
#           the whole report: per-stage TAIL attribution, GC correlation,
#           frame-index reproducibility, and the drift/throttle check.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/p99_probe.py 08 220 2
#
# NOTE: p50 work to date has never explained the tail. This deliberately looks
#      only at the WORST frames, and at four candidate causes at once, because
#      the tail is where the 10 Hz claim actually fails.
#
# Measurement only. Reads the shipping code; changes nothing in src/.
"""Where does the p99 come from?

Four questions, one run each:

  1. WHICH STAGE owns the tail? Per-stage p99 and p99-p50 spread, plus a
     decomposition of the worst frames: on a slow frame, which stage was
     anomalous relative to its own median?
  2. Is it GARBAGE COLLECTION? gc callbacks record every collection and the
     frame it landed in, so slow frames can be tested against GC events
     directly rather than by assertion.
  3. Is it DATA-DEPENDENT or NOISE? The same run repeated: if the same frame
     indices are slowest each time, something about those frames is expensive.
     If the slow set is different each time, it is the machine.
  4. Is it THERMAL/DRIFT? Frame time against frame index, fitted, plus first-
     versus-last-decile comparison.
"""
import gc
import sys
import time

import numpy as np
from vrgrid.grid.schedule import load
from vrgrid.gpu.timing import STAGES, Timer
from vrgrid.run.__main__ import iter_pipeline
from vrgrid.run.engine import MapEngine

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 220
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
SCHEDULE = "5/10/20/40"


def one_run(tag):
    """Return (per-frame total ms, {stage: per-frame ms}, gc events, wall t0)."""
    gc_events = []          # (frame_index, generation, duration_ms)
    state = {"frame": -1, "t0": None}

    def cb(phase, info):
        if phase == "start":
            state["t0"] = time.perf_counter()
        elif state["t0"] is not None:
            gc_events.append((state["frame"], info.get("generation", -1),
                              (time.perf_counter() - state["t0"]) * 1e3))
            state["t0"] = None

    t = Timer(stages=STAGES, capacity=max(4096, N + 8))
    engine = MapEngine(load(SCHEDULE), max_points=120_000, timer=t)
    frames = iter(iter_pipeline(SEQ, N + 1, timer=t))

    gc.callbacks.append(cb)
    totals, wall = [], []
    try:
        k = 0
        while True:
            state["frame"] = k
            w0 = time.perf_counter()
            f = next(frames, None)
            if f is None:
                break
            engine.step(f)
            dt = (time.perf_counter() - w0) * 1e3
            totals.append(dt)
            wall.append(w0)
            k += 1
    finally:
        gc.callbacks.remove(cb)

    per_stage = {}
    for s in STAGES:
        if s == "total":
            continue
        v = np.asarray(t._samples(s), dtype=np.float64)
        if v.size:
            per_stage[s] = v
    print(f"  [{tag}] {len(totals)} frames timed, {len(gc_events)} gc collections")
    return np.array(totals), per_stage, gc_events, np.array(wall)


runs = [one_run(f"rep{i+1}") for i in range(REPS)]

# dump the per-frame arrays so the tail can be re-analysed without a re-run --
# the whole point of R-a. Saved beside the report.
_dump = {"total": runs[0][0]}
for _s, _v in runs[0][1].items():
    _dump[f"stage_{_s}"] = _v
for _i, (_a, _, _, _) in enumerate(runs):
    _dump[f"rep{_i+1}_total"] = _a
np.savez_compressed("p99_frames.npz", **_dump)
print(f"  wrote p99_frames.npz ({len(_dump)} arrays)")
a, per_stage, gc_events, wall = runs[0]

# frame 0 is startup and is excluded everywhere below, as run_real does
a0, sl = a[1:], slice(1, None)

print(f"\n{'='*74}\n1. WHICH STAGE OWNS THE TAIL\n{'='*74}")
print(f"  {'stage':<14}{'p50':>9}{'p99':>9}{'max':>9}{'p99-p50':>10}"
      f"{'share of p99 excess':>22}")
tot_p50, tot_p99 = np.median(a0), np.percentile(a0, 99)
excess = tot_p99 - tot_p50
rows = []
for s, v in per_stage.items():
    v = v[sl] if v.size > len(a0) else v
    if not v.size:
        continue
    p50, p99 = float(np.median(v)), float(np.percentile(v, 99))
    rows.append((s, p50, p99, float(v.max()), p99 - p50))
for s, p50, p99, mx, spread in sorted(rows, key=lambda r: -r[4]):
    print(f"  {s:<14}{p50:>9.2f}{p99:>9.2f}{mx:>9.2f}{spread:>10.2f}"
          f"{spread/excess*100:>21.0f}%")
print(f"  {'-'*70}")
print(f"  {'FRAME':<14}{tot_p50:>9.2f}{tot_p99:>9.2f}{a0.max():>9.2f}{excess:>10.2f}")
print("\n  'share of p99 excess' = this stage's own p99-p50 spread as a fraction of")
print("  the whole frame's. It sums to more than 100% when tails do not coincide.")

print(f"\n{'='*74}\n2. THE WORST FRAMES, DECOMPOSED\n{'='*74}")
worst = np.argsort(a0)[::-1][:8] + 1
print(f"  {'frame':>6}{'total':>9}{'vs p50':>9}   dominant stages (ms over their own median)")
for fi in worst:
    contrib = []
    for s, v in per_stage.items():
        if v.size <= fi:
            continue
        d = float(v[fi]) - float(np.median(v[sl]))
        contrib.append((d, s, float(v[fi])))
    contrib.sort(reverse=True)
    top = "  ".join(f"{s} +{d:.1f}" for d, s, _ in contrib[:4] if d > 0.5)
    ng = sum(1 for g in gc_events if g[0] == fi)
    print(f"  {fi:>6}{a[fi]:>9.2f}{a[fi]-tot_p50:>+9.2f}   {top}"
          + (f"   [gc x{ng}]" if ng else ""))

print(f"\n{'='*74}\n3. GARBAGE COLLECTION\n{'='*74}")
if not gc_events:
    print("  No collections ran during the timed loop. GC is NOT the cause.")
else:
    gms = np.array([g[2] for g in gc_events])
    byframe = {}
    for fi, gen, ms in gc_events:
        byframe.setdefault(fi, []).append(ms)
    gframes = np.array(sorted(k for k in byframe if 0 < k < len(a)))
    print(f"  collections: {len(gc_events)}  over {len(byframe)} distinct frames")
    print(f"  per-collection pause: p50 {np.median(gms):.3f} ms  "
          f"p99 {np.percentile(gms,99):.3f}  max {gms.max():.3f}")
    gen_counts = {}
    for _, gen, _ in gc_events:
        gen_counts[gen] = gen_counts.get(gen, 0) + 1
    print(f"  by generation: {dict(sorted(gen_counts.items()))}")
    if gframes.size:
        gc_tot = a[gframes]
        others = np.array([a[i] for i in range(1, len(a)) if i not in byframe])
        print(f"  frames WITH a collection:    n={gc_tot.size:>4}  "
              f"p50 {np.median(gc_tot):.2f}  max {gc_tot.max():.2f}")
        print(f"  frames WITHOUT a collection: n={others.size:>4}  "
              f"p50 {np.median(others):.2f}  max {others.max():.2f}")
        hit = len(set(worst.tolist()) & set(byframe))
        print(f"  of the 8 worst frames, {hit} had a collection")
        print(f"  total GC time over the run: {gms.sum():.1f} ms of "
              f"{a0.sum():.1f} ms ({gms.sum()/a0.sum()*100:.2f}%)")

print(f"\n{'='*74}\n4. DATA-DEPENDENT OR NOISE?\n{'='*74}")
if REPS < 2:
    print("  need REPS >= 2")
else:
    sets, tops = [], []
    for i, (ai, _, _, _) in enumerate(runs):
        w = set((np.argsort(ai[1:])[::-1][:15] + 1).tolist())
        sets.append(w)
        tops.append(sorted(w))
        print(f"  rep{i+1} worst 15 frame indices: {sorted(w)}")
    inter = set.intersection(*sets)
    print(f"\n  overlap across all {REPS} reps: {len(inter)} of 15  -> {sorted(inter)}")
    exp = 15 * 15 / max(len(runs[0][0]) - 1, 1)
    print(f"  expected overlap if purely random: ~{exp:.1f}")
    print("  >> strongly data-dependent" if len(inter) > 3 * exp
          else "  >> consistent with machine noise, not frame content")
    for i, (ai, _, _, _) in enumerate(runs):
        b = ai[1:]
        print(f"  rep{i+1}: p50 {np.median(b):.2f}  p99 {np.percentile(b,99):.2f}  "
              f"max {b.max():.2f}")

print(f"\n{'='*74}\n5. DRIFT / THERMAL\n{'='*74}")
idx = np.arange(a0.size)
slope, intercept = np.polyfit(idx, a0, 1)
dec = max(a0.size // 10, 1)
print(f"  linear fit: {slope:+.4f} ms per frame  ({slope*a0.size:+.1f} ms over the run)")
print(f"  first decile p50 {np.median(a0[:dec]):.2f}   last decile p50 {np.median(a0[-dec:]):.2f}")
print(f"  first decile p99 {np.percentile(a0[:dec],99):.2f}   "
      f"last decile p99 {np.percentile(a0[-dec:],99):.2f}")
q = a0.size // 4
for i in range(4):
    seg = a0[i*q:(i+1)*q]
    print(f"  quarter {i+1}: p50 {np.median(seg):7.2f}  p99 {np.percentile(seg,99):7.2f}  "
          f"max {seg.max():7.2f}")
print(f"  wall-clock span: {(wall[-1]-wall[0]):.1f} s")
