#!/usr/bin/env python3
"""The Day-6 per-stage latency table. [Shrestha]

Gate 6: the latency numbers on a slide come from here, not from memory.

    python scripts/timing_table.py [--schedule 5/10/20/40] [--frames 200]
    python scripts/timing_table.py --alloc      # + per-frame transient bytes

⚑ **This is a lower bound on frame latency, not the frame total.** It times
  the back half of the frame -- binning, the `src/gpu` stages, and the `fuse()`
  that consumes the scatter's aggregate. The perception front end (load,
  transform, range_image, semantics, motion) needs SemanticKITTI on disk. The
  unmeasured stages are printed as rows with their owner and why, rather than
  omitted: a table showing green rows and a healthy headroom, with the missing
  half silently dropped, is exactly the shape of a number that gets called on
  stage.

  **The whole frame HAS now been timed, elsewhere.** The two reasons this
  docstring used to give for there being no end-to-end loop are both obsolete:
  `src/run/engine.py:297-309` calls the real `scatter_sorted` and `fuse`, and
  the dataset is complete on disk. The end-to-end figure, 200 frames of real
  seq 08, is **p50 89.18 ms / p99 100.43 ms / max 109.28 ms** against a 100 ms
  budget -- see `docs/research-log.md` (the 2026-09-04 entry, twelve stages,
  flat and disjoint). Quote that for frame latency, and this table for the
  per-stage breakdown of the back half. Do not add the two together.

**Built against the real interfaces, not a private mock.** The points are a
synthetic HDL-64E sweep, but everything downstream of them is the shipping
code: `ring_of` and `i_ring` for lattice semantics, `RingBuffer.flat_slot` for
storage, `scatter_sorted` with the allocator's scratch, the real `fuse`, the
real `visibility_cleanup`, the real pyramid `build`. Swapping `make_sweep` for
`perception.loader` and sequence 07 is the only edit the real-data path needs,
which is the same seam `scripts/eval_synthetic.py` documents.

Three things that shape the table and are worth reading before quoting it:

**⚑ The back end alone spends 59 ms of the 100 ms budget.** 46 ms p50, 59 ms
p99, 1.7x headroom -- and that is before a single line of perception runs.
Whatever load, transform, range_image, semantics and motion cost has to fit
in the remaining ~41 ms, so the 10 Hz claim is not yet demonstrated; it is
bounded. Stable to within about 10% across three runs of 200 frames.

**✔ `bin` has an owner: `grid.lattice.bin_points`.** Gate 3, item 2. It was
composed by hand in four places -- here, `fusion.scatter`, `grid/transient.py`
and `run/engine.py` -- four spellings of one step across three directories,
agreeing by luck rather than by design. A binning bug does not crash; it
produces a plausible map.

The rewrite is one vectorised pass with no ring loop at all. The obvious shape
is a pass per ring over the points that fall in it, and that is what all four
copies did; it needs the selected world coordinates compacted into a buffer,
and numpy will not do that without allocating -- `np.compress(..., out=)`
still built 1.54 MB of internal index at 96,000 selected points. So the ring
became a per-POINT attribute: `k`, `side`, `x0`, `y0` and `offset` are
gathered by ring index, and the whole sweep is binned in one pass of
full-length ufuncs.

Measured here, 120,000 returns, same machine, before and after:

    hand-rolled     6.962 MB/frame    13.64 ms p50
    bin_points      0.002 MB/frame    12.35 ms p50

The allocation is gone -- that was the point, "no allocation in the frame
loop" is a hard invariant and a sentence in the report -- and it is 9% faster
as well, because one pass over the sweep beats four plus four compacting
copies.

⚑ The last two allocations to go are worth knowing about, because neither
  appears in a profile as a named allocation: `np.take(table, idx, out=)`
  builds a full-length bounds-check array in its default `mode="raise"`, and
  `int64 += bool` casts through numpy's fixed 64 kB internal buffer.

**Real geometry is not random indices.** `bench_scatter.py` draws `idx`
uniformly over the slots; a LiDAR sweep binned through the ring schedule is
spatially clustered and lands 47% of its returns in ring 1 and 6.6% in
ring 3. Against a cold grid the difference in scatter is small -- three
trials of 300 frames put real geometry about 7% above random on p50, 4.86
against 4.55 ms, with the p99 gap inside run-to-run noise. Against the warm
grid this script builds it is larger, because both scatter and fuse then do
real work on real prior state. Either way the scatter row here will not match
bench_scatter's exactly, and this is the reason.
"""

import argparse
import pathlib
import json
import platform
import subprocess
import time
import tracemalloc

import numpy as np
from vrgrid.gpu.allocators import allocate
from vrgrid.gpu.kernels import (
    Z_MAX_CM,
    Z_MIN_CM,
    measurement_variance_cm2,
    quantise_height,
    quantise_weight,
    scatter_sorted,
)
from vrgrid.gpu.pyramid import build
from vrgrid.gpu.shift import (
    RingBuffer,
    cells_per_shift,
    shift,
)
from vrgrid.gpu.timing import SENSOR_HZ, STAGES, Timer
from vrgrid.gpu.visibility import new_visibility_scratch, visibility_cleanup
from vrgrid.grid.fusion import fuse
from vrgrid.grid.lattice import bin_points, new_bin_scratch
from vrgrid.grid.schedule import load

# Range image geometry is locked to 64x512 sub-clouds (FLARES; research log
# 2026-09-01), so the gather in visibility_cleanup is sized off that and not
# off a full 64x2048 sweep.
IMAGE_SHAPE = (64, 512)

BLIND_CONE_M = 3.74     # master v4 §1.3; nothing returns inside it
MAX_RANGE_M = 99.0

# Who to ask when a row moves, and -- for the unmeasured rows -- what is in
# the way. Ownership is root CLAUDE.md's, not the execution plan's.
STAGE_OWNER = {
    "load": "JP", "transform": "JP", "range_image": "JP",
    "semantics": "JP", "motion": "JP",
    "bin": "Aakash", "scatter": "Shrestha", "fuse": "Aakash",
    "split_merge": "Aakash", "cleanup": "Shrestha", "pyramid": "Shrestha",
    "shift": "Shrestha",
}

BLOCKED = {
    "load": "needs SemanticKITTI on disk",
    "transform": "needs SemanticKITTI on disk",
    "range_image": "needs SemanticKITTI on disk",
    "semantics": "needs SemanticKITTI on disk",
    "motion": "needs SemanticKITTI on disk",
    "split_merge": "per-cell API; driven by the refinement pool, no frame batch",
}

MEASURED = ("bin", "scatter", "fuse", "cleanup", "pyramid", "shift")


def cpu_name() -> str:
    """The machine, named. A latency table without one is not reproducible."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_name() -> str:
    """Named if present. This is the CPU reference path either way -- every
    kernel here is numpy -- so an absent GPU is a fact to print, not to hide
    behind a blank cell."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5,
                             check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "none -- numpy CPU reference path"


def make_sweep(rng, n):
    """One HDL-64E sweep in the VEHICLE frame: (x, y, z, range).

    Returns thin out with range -- the gamma tail, not a uniform cube -- which
    is the effect the whole foveated design exists for, and the reason the
    ring split below is uneven. Same shape as `baseline_demo.synthetic_sweep`
    so the two scripts' scatter rows are comparable.

    Replace this with `perception.loader` + `transforms` for the real path;
    nothing downstream of it knows the difference.
    """
    r = np.clip(BLIND_CONE_M + rng.gamma(2.0, 12.0, n), BLIND_CONE_M, MAX_RANGE_M)
    az = rng.uniform(-np.pi, np.pi, n)
    el = rng.uniform(np.radians(-24.8), np.radians(2.0), n)
    ce = np.cos(el)
    return r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el) + 1.73, r


def ring_buffers(handle):
    """One toroidal window per ring, centred on the vehicle.

    `x0 = -side // 2` is the part that is easy to get wrong: `RingBuffer`
    defaults to a window at the lattice origin, but a sweep is centred on the
    sensor, so half of every ring would fall outside the window and bin to -1.
    """
    return [RingBuffer(side=r.side, offset=r.offset,
                       x0=-(r.side // 2), y0=-(r.side // 2))
            for r in handle.rings]


# `bin_points` used to be defined here, by hand, and the docstring said so:
# "the stage no module owns". It has an owner now -- `grid.lattice.bin_points`,
# one vectorised pass with every intermediate preallocated -- and this script
# times that rather than a fourth private copy of it. Gate 3, item 2.


def payload(z_m, range_m, rng, n):
    """The per-return columns scatter folds, from the sweep's own geometry.

    `class_id` is drawn over the full 0..19 learning set. It used to be drawn
    over 0..15, because `fusion.boyer_moore_update` rejected anything above 15
    under the old 4-bit candidate and this script could not otherwise run --
    which also meant the binning and fusion stages were being timed against a
    label distribution narrower than the real one. The byte was re-split 5 | 3
    on 1 Sep (§10.2); the draw is the real range now.
    """
    return {
        "z_cm": quantise_height(z_m),
        "w_q": quantise_weight(measurement_variance_cm2(range_m)),
        "refl": rng.integers(0, 256, n).astype(np.uint8),
        "class_id": rng.integers(0, 20, n).astype(np.uint8),
        "is_ground": z_m < 0.2,
    }


def fill_terrain(grid, rings, rng):
    """Terrain-shaped rather than uniform-random, matching bench_pyramid.py.

    The reductions are branch-free so the pyramid's timing does not depend on
    this, but running both scripts against different maps invites the question
    of why their numbers differ, and the answer should not be the fill.
    """
    for layout in rings:
        side, span = layout.side, slice(layout.offset, layout.offset + layout.slots)
        iy, ix = np.mgrid[0:side, 0:side]
        z = -173.0 + 2.0 * ix + 1.4 * iy + rng.integers(-2, 3, (side, side))
        hazard = rng.random((side, side)) < 0.02
        z[hazard] += rng.integers(20, 60, int(hazard.sum()))
        grid["ground_height"][span] = z.ravel()

        obs = rng.integers(3, 20, layout.slots).astype(np.uint8)
        thin = rng.random(layout.slots) < 0.05
        obs[thin] = rng.integers(0, 3, int(thin.sum()))
        grid["obs_count"][span] = obs
        grid["traversability"][span] = np.where(obs < 3, 1 << 5, 0)


def make_range_image(rng, shape):
    """A sweep with a horizon, some sky and some absorbed beams.

    The NO_RETURN pixels are the point: eq (32) must not clear through them,
    and a benchmark image without any would time a branch the real one takes.
    """
    v = np.arange(shape[0])[:, None]
    horizon = 8.0 + 70.0 * (v / shape[0])
    img = horizon + rng.normal(0.0, 3.0, shape)
    img[rng.random(shape) < 0.08] = np.inf          # sky and absorbed beams
    return np.maximum(img, 1.0)


def make_candidates(rng, n):
    """Occupied cell centres in the vehicle frame -- what cleanup is handed.

    A free cell has nothing to clear, so the candidate set is the occupied
    cells, not the whole map; `--cells` is the cap that ⚑ sizing the
    visibility scratch into `allocate()` would have to fix.
    """
    theta = rng.uniform(-np.pi, np.pi, n)
    r = np.sqrt(rng.uniform(1.0, MAX_RANGE_M ** 2, n))
    return r * np.cos(theta), r * np.sin(theta), rng.uniform(Z_MIN_CM / 100.0, Z_MAX_CM / 100.0, n)


class _Untimed:
    """A no-op stand-in for `Timer.stage`, so the warm-up and the allocation
    pass run the identical code path rather than copies of it that can drift."""

    def __init__(self, name):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Frame:
    """Everything the loop touches, built once. Nothing in `run` may allocate
    -- that is the property the table exists to measure."""

    def __init__(self, handle, sched, args, rng):
        self.handle, self.sched, self.args = handle, sched, args
        self.buffers = ring_buffers(handle)
        self.sweeps = []
        for _ in range(10):
            x, y, z, r = make_sweep(rng, args.points)
            self.sweeps.append((x, y, payload(z, r, rng, args.points)))

        cx, cy, cz = make_candidates(rng, args.cells)
        self.cand = (cx, cy, cz)
        self.image = make_range_image(rng, IMAGE_SHAPE)
        self.vis_scratch = new_visibility_scratch(args.cells, self.image.dtype)
        self.has_return = rng.random(args.cells) < 0.35

        # Ego-motion over one 10 Hz frame, in whole cells of each ring: 1.5 m
        # at 15 m/s is 30 cells of the 5 cm ring and 4 of the 40 cm ring,
        # which is why the clear is per ring and not one number for the map.
        self.per_frame_m = args.speed_mps / SENSOR_HZ
        self.dx = [max(1, round(self.per_frame_m / r.cell_m)) for r in handle.rings]

        # The sweep is centred on the SENSOR; the ring windows are anchored to
        # the LATTICE and march forward with the vehicle. Feeding the same
        # origin-centred sweep every frame while the windows advance walks the
        # map out from under the points: by frame 13 the 5 cm ring's window no
        # longer contains the origin, every point bins to -1, and scatter and
        # fuse start reporting sub-millisecond p50s for doing nothing at all.
        # Advancing the points with the vehicle is what the real loop does --
        # `transforms` hands the frame loop world-frame points -- and it is
        # JP's stage, so the add happens outside the timed region.
        self.ego = np.zeros(args.points, np.float64)
        self.bin_scratch = new_bin_scratch(args.points, self.sched)
        self.idx = np.zeros(args.points, np.int64)

    def cells_shifted(self) -> int:
        return sum(cells_per_shift(b, dx, 0) for b, dx in zip(self.buffers, self.dx))

    def step(self, i, ctx):
        """One frame. `ctx` is `Timer.stage` when timing and `_Untimed` when
        warming up or measuring allocation."""
        h = self.handle
        x, y, cols = self.sweeps[i % len(self.sweeps)]
        np.add(x, i * self.per_frame_m, out=self.ego)   # untimed: JP's transform
        with ctx("bin"):
            idx = bin_points(self.ego, y, self.sched, self.buffers,
                             self.idx, self.bin_scratch, 0.0,
                             (i * self.per_frame_m, 0.0))
        with ctx("scatter"):
            agg = scatter_sorted(idx, **cols, scratch=h.scratch)
        with ctx("fuse"):
            fuse(h.grid, agg)
        with ctx("cleanup"):
            visibility_cleanup(*self.cand, self.image,
                               has_return_now=self.has_return,
                               scratch=self.vis_scratch)
        if h.pyramid is not None:
            with ctx("pyramid"):
                build(h.pyramid, h.grid, h.rings)
        with ctx("shift"):
            for buf, dx in zip(self.buffers, self.dx):
                shift(buf, dx, 0, h.grid)


def run(frame, args):
    t = Timer(stages=MEASURED + ("measured",))
    frame.step(0, _Untimed)      # warm up: first-touch page faults are startup
    for i in range(args.frames):
        with t.stage("measured"):
            frame.step(i, t.stage)
    return t


def measure_alloc(frame, args):
    """Transient bytes per frame, per stage. A separate pass because
    tracemalloc costs several times the latency it would be reported beside --
    measuring the two at once would corrupt both."""
    peaks = {}
    for name in MEASURED:
        if name == "pyramid" and frame.handle.pyramid is None:
            continue

        def only(stage, _want=name):
            return _Untimed(stage) if stage != _want else _Peak(peaks, _want)

        tracemalloc.start()
        for i in range(min(args.frames, 20)):
            frame.step(i, only)
        tracemalloc.stop()
    return peaks


class _Peak:
    """Peak transient allocation inside one stage of one frame."""

    def __init__(self, into, name):
        self.into, self.name = into, name

    def __enter__(self):
        tracemalloc.reset_peak()
        self.before = tracemalloc.get_traced_memory()[0]
        return self

    def __exit__(self, *exc):
        peak = tracemalloc.get_traced_memory()[1] - self.before
        self.into[self.name] = max(self.into.get(self.name, 0), peak)
        return False


def print_real_table(t):
    """Every stage that produced a sample, in pipeline order, and a real total.

    No MEASURED-is-a-lower-bound caveat here: with `--seq` there are no
    unmeasured stages left except `split_merge`, which has no per-frame batch
    entry point at all (see the Gate 3 review).
    """
    summary = t.summary()
    budget = 1e3 / SENSOR_HZ
    head = f"{'stage':<13}{'p50 ms':>9}{'p99 ms':>9}{'max ms':>9}{'share':>8}{'x10Hz':>8}"
    print(head)
    print("-" * len(head))
    total = summary["total"]["p50_ms"]
    for name in STAGES:
        if name not in summary or name == "total":
            continue
        s = summary[name]
        print(f"{name:<13}{s['p50_ms']:>9.2f}{s['p99_ms']:>9.2f}{s['max_ms']:>9.2f}"
              f"{s['p50_ms'] / total:>7.0%}{budget / s['p99_ms']:>7.1f}x")
    print("-" * len(head))
    m, h = summary["total"], t.headroom("total")
    print(f"{'FRAME':<13}{m['p50_ms']:>9.2f}{m['p99_ms']:>9.2f}{m['max_ms']:>9.2f}"
          f"{1.0:>7.0%}{budget / m['p99_ms']:>7.1f}x")
    print(f"\n{h['fps_p50']:.1f} FPS p50 ({h['headroom_p50']:.1f}x), "
          f"{h['fps_p99']:.1f} FPS p99 ({h['headroom_p99']:.1f}x) against the "
          f"{budget:.0f} ms budget at {SENSOR_HZ:.0f} Hz")
    print("meets 10 Hz at p99" if h["meets_sensor_rate"] else "MISSES 10 Hz at p99")
    print("\nsplit_merge is absent because it has no per-frame batch entry point; "
          "it is\ndriven per cell by the refinement pool. Everything else in the "
          "frame is above.")
    print("Shares are p50/p50 and will not sum to exactly 100%: a percentile of "
          "sums is\nnot the sum of percentiles. They are for reading the shape, "
          "not for arithmetic.")


def print_table(t, alloc, frame):
    summary = t.summary()
    budget_ms = 1e3 / SENSOR_HZ
    wide = alloc is not None
    head = f"{'stage':<12}{'owner':<11}{'p50 ms':>9}{'p99 ms':>9}{'max ms':>9}{'x10Hz':>8}"
    if wide:
        head += f"{'MB/frame':>10}"
    print(head)
    print("-" * len(head))

    for name in MEASURED:
        if name not in summary:
            continue
        s = summary[name]
        row = (f"{name:<12}{STAGE_OWNER[name]:<11}{s['p50_ms']:>9.2f}"
               f"{s['p99_ms']:>9.2f}{s['max_ms']:>9.2f}"
               f"{budget_ms / s['p99_ms']:>7.1f}x")
        if wide:
            mb = alloc.get(name, 0) / 1e6
            row += f"{mb:>10.2f}" if mb >= 0.01 else f"{'~0':>10}"
        print(row)

    print("-" * len(head))
    m = summary["measured"]
    h = t.headroom("measured")
    row = (f"{'MEASURED':<12}{'':<11}{m['p50_ms']:>9.2f}{m['p99_ms']:>9.2f}"
           f"{m['max_ms']:>9.2f}{budget_ms / m['p99_ms']:>7.1f}x")
    if wide:
        row += f"{sum(alloc.values()) / 1e6:>10.2f}"
    print(row)
    print(f"\n{h['fps_p50']:.1f} FPS p50 ({h['headroom_p50']:.1f}x), "
          f"{h['fps_p99']:.1f} FPS p99 ({h['headroom_p99']:.1f}x) "
          f"-- against the {budget_ms:.0f} ms budget at {SENSOR_HZ:.0f} Hz")

    print("\nnot in the subtotal above:")
    for name, why in BLOCKED.items():
        print(f"  {name:<12}{STAGE_OWNER[name]:<11}{why}")

    print(f"\n[!] MEASURED is a LOWER BOUND on frame latency, not the frame total. "
          f"Six stages\n  above are unmeasured; the front end is the whole of "
          f"perception. The honest\n  sentence is \"the mapping back end costs "
          f"{m['p99_ms']:.1f} ms at p99\", never \"we run at {h['fps_p99']:.0f} FPS\".")
    if wide and alloc.get("bin", 0) > 1e6:
        print(f"\n[!] `bin` allocates {alloc['bin'] / 1e6:.2f} MB per frame. It is "
              f"`grid.lattice.bin_points`,\n  which is supposed to allocate "
              f"nothing -- so this is a regression, not a\n  known gap. Check "
              f"for a `np.take` without mode=\"clip\" or a mixed-dtype ufunc.")


def run_real(args, sched=None):
    """Time a real sequence: every stage, front end included.

    `sched` overrides `args.schedule`, so a caller sweeping schedules --
    `ablation_table.py --seq` -- can pass a constructed uniform baseline that
    has no config name to load.

    This is the table Day 6 asks for, and the one the synthetic path cannot
    produce -- `iter_pipeline` and `MapEngine` both take the same Timer, and
    the stage names are `timing.STAGES`, so the frame adds up instead of
    being a back-end subtotal with a caveat attached.
    """
    from vrgrid.run.__main__ import iter_pipeline
    from vrgrid.run.engine import MapEngine

    t = Timer(stages=STAGES)
    engine = MapEngine(load(args.schedule) if sched is None else sched,
                       max_points=args.points,
                       clip_class_ids=args.clip_class_ids, timer=t,
                       device=getattr(args, "device", "cpu"))

    # `total` has to span the WHOLE frame -- perception AND the map -- and the
    # perception half happens inside the generator, during `next()`. Wrapping
    # only `engine.step` gave a FRAME row smaller than several of its own
    # stages and shares that summed to 156%.
    # --semantics frnet: the opt-in DL mode. `getattr` because ablation_table.py calls
    # run_real with its own argument namespace, which has no such attributes.
    semantics_source = getattr(args, "semantics", "gt")
    frnet = None
    if semantics_source == "frnet":
        from vrgrid.run.__main__ import open_frnet
        frnet = open_frnet(fast_scatter=getattr(args, "fast_scatter", False),
                           threads=getattr(args, "threads", None))
    # reuse_buffers: this loop steps each frame before pulling the next, so the
    # allocation-free transform path is safe here -- see iter_pipeline.
    frames = iter(iter_pipeline(args.seq, args.frames + 1,
                                use_patchworkpp=not args.no_patchworkpp, timer=t,
                                reuse_buffers=True, semantics_source=semantics_source,
                                frnet=frnet, device=getattr(args, "device", "cpu")))
    n = 0
    while True:
        t0 = time.perf_counter()
        frame = next(frames, None)
        if frame is None:
            break
        engine.step(frame)
        t.record("total", (time.perf_counter() - t0) * 1e3)
        n += 1
        # The first frame faults every page in and warms every cache; that is
        # startup, not a frame, and it lands in the p99 the claim rests on.
        if n == 1:
            t.reset()
    if n < 2:
        raise SystemExit(f"sequence {args.seq} yielded {n} frames; need at least 2")
    return t, engine, n - 1


def run_free(args):
    """Whole-frame latency with NO stage timer, for the device path.

    The staged table synchronises the card at every stage boundary so each row
    is honest -- and that serialises the frame. Unsynchronised, the perception
    kernels run while Patchwork++ runs on the host, which is the configuration
    that actually ships. This pass times only the whole frame, same frames.
    """
    from vrgrid.run.__main__ import iter_pipeline
    from vrgrid.run.engine import MapEngine

    t = Timer(stages=("total",))
    engine = MapEngine(load(args.schedule), max_points=args.points,
                       clip_class_ids=args.clip_class_ids, device=args.device)
    frames = iter(iter_pipeline(args.seq, args.frames + 1,
                                use_patchworkpp=not args.no_patchworkpp,
                                device=args.device))
    n = 0
    while True:
        t0 = time.perf_counter()
        frame = next(frames, None)
        if frame is None:
            break
        engine.step(frame)
        t.record("total", (time.perf_counter() - t0) * 1e3)
        n += 1
        if n == 1:
            t.reset()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", default="5/10/20/40")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--points", type=int, default=120_000,
                    help="returns per sweep; a real HDL-64E sweep is ~120,000")
    ap.add_argument("--cells", type=int, default=200_000,
                    help="candidate (occupied) cells handed to visibility cleanup")
    ap.add_argument("--speed-mps", type=float, default=None,
                    help="ego speed for the shift row; default is the "
                         "schedule's anisotropy v_ref")
    ap.add_argument("--no-pyramid", action="store_true",
                    help="drop the stretch-item pyramid row (§7.2)")
    ap.add_argument("--seq", default=None,
                    help="time a REAL SemanticKITTI sequence end to end (needs "
                         "$VRGRID_DATA_ROOT). Without it, a synthetic sweep and "
                         "the back end only")
    ap.add_argument("--no-patchworkpp", action="store_true",
                    help="--seq only: use the semantic-class ground proxy")
    ap.add_argument("--clip-class-ids", action="store_true",
                    help="--seq only: clip semantic ids to 15 so fusion's 4-bit "
                         "candidate accepts them (math §10.2)")
    ap.add_argument("--semantics", default="gt", choices=["gt", "frnet"],
                    help="with --seq: gt (default) times the pipeline on .label classes; "
                         "frnet times the OPT-IN DL mode, FRNet inference inside the "
                         "semantics stage. For the real-time figure run frnet on the GPU "
                         "machine (AWS), not this CPU.")
    ap.add_argument("--fast-scatter", action="store_true",
                    help="with --semantics frnet: scripts/frnet_fast_scatter.py (verified)")
    ap.add_argument("--threads", type=int, default=None,
                    help="with --semantics frnet: torch.set_num_threads(N)")
    ap.add_argument("--frame-times", default=None, metavar="PATH",
                    help="with --seq: also write every frame's whole-frame time "
                         "(ms, startup frame excluded) as JSON, so p99 can be "
                         "pooled across runs instead of taken per run")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="--seq only: run scatter and cleanup on the card "
                         "(src/gpu/device.py). Publish it as a second column, "
                         "never in place of the CPU one")
    ap.add_argument("--alloc", action="store_true",
                    help="also report transient bytes per frame per stage, for the "
                         "MAPPING BACK END only (separate pass; tracemalloc "
                         "distorts latency). NOT supported with --seq: the "
                         "perception front end is not instrumented.")
    args = ap.parse_args()

    # --alloc is implemented only on the synthetic path: measure_alloc is called
    # from this function, never from run_real. So with --seq it used to be
    # SILENTLY IGNORED -- the table printed with no MB/frame column at all, and
    # the back-end-only figure got quoted as a whole-frame number. That is how
    # "8.15 -> 1.31 MB/frame" reached the README unqualified. Fail instead.
    if args.alloc and args.seq is not None:
        raise SystemExit(
            "--alloc is not supported with --seq; front-end allocation is not "
            "yet instrumented.\n"
            "  --alloc reports the MAPPING BACK END only (bin, scatter, fuse, "
            "cleanup, pyramid, shift).\n"
            "  The perception stages (load, transform, range_image, semantics, "
            "motion) are NOT measured,\n"
            "  and on real seq 08 they allocate ~39.5 MB/frame -- see "
            "reports/r-b-p99-tail-investigation.md.\n"
            "  Run --alloc WITHOUT --seq for the back-end figure, and do not "
            "quote it as a whole-frame number.")

    # Same failure class as --alloc above, the other way round: --frame-times is
    # written only on the --seq path, so without --seq it would be silently
    # ignored and a pooled p99 would be computed from a file that never existed.
    if args.semantics != "gt" and args.seq is None:
        raise SystemExit("--semantics frnet requires --seq; the synthetic path has no scans "
                         "for a network to label")
    if args.semantics == "gt" and (args.fast_scatter or args.threads is not None):
        raise SystemExit("--fast-scatter and --threads only apply to --semantics frnet")
    if args.frame_times and args.seq is None:
        raise SystemExit("--frame-times requires --seq; the synthetic path does not "
                         "record whole-frame samples")

    sched = load(args.schedule)
    if args.speed_mps is None:
        args.speed_mps = sched.anisotropy.v_ref_ms

    if args.seq is not None:
        t, engine, frames = run_real(args)
        print(f"CPU  {cpu_name()}")
        print(f"GPU  {gpu_name()}")
        print(f"numpy {np.__version__}, python {platform.python_version()}, "
              f"{platform.system()}\n")
        print(f"sequence {args.seq}, {frames} frames, schedule {args.schedule}, "
              f"{engine.handle.allocated_slots:,} slots, device {engine.device}\n")
        print_real_table(t)
        if args.frame_times:
            # The raw samples, not a percentile of them: a p99 gate judged over
            # several runs has to rank every frame together, and per-run p99s
            # cannot be pooled after the fact.
            pathlib.Path(args.frame_times).write_text(
                json.dumps([round(float(x), 4) for x in t._samples("total")]),
                encoding="utf-8")
        if args.device == "cuda":
            del engine
            free = run_free(args)
            m, h = free.summary()["total"], free.headroom("total")
            print(f"\nFREE-RUNNING (no per-stage synchronisation; the card and "
                  f"Patchwork++ overlap):\n  FRAME p50 {m['p50_ms']:.2f} ms  "
                  f"p99 {m['p99_ms']:.2f} ms  max {m['max_ms']:.2f} ms  -> "
                  f"{h['fps_p50']:.1f} FPS p50, {h['fps_p99']:.1f} FPS p99, "
                  + ("meets" if h["meets_sensor_rate"] else "MISSES")
                  + " 10 Hz at p99")
        return

    handle = allocate(sched, with_pyramid=not args.no_pyramid)
    rng = np.random.default_rng(0)
    fill_terrain(handle.grid, handle.rings, rng)
    frame = Frame(handle, sched, args, rng)

    print(f"CPU  {cpu_name()}")
    print(f"GPU  {gpu_name()}")
    print(f"numpy {np.__version__}, python {platform.python_version()}, "
          f"{platform.system()}\n")
    print(f"schedule {args.schedule}, {handle.allocated_slots:,} slots "
          f"({handle.logical_cells:,} logical), {args.frames} frames")
    print(f"{args.points:,} returns/sweep (synthetic HDL-64E, binned through the "
          f"real lattice)")
    print(f"{args.cells:,} candidate cells, range image "
          f"{IMAGE_SHAPE[0]}x{IMAGE_SHAPE[1]}")
    print(f"shift {args.speed_mps:.0f} m/s -> {frame.per_frame_m:.2f} m/frame, "
          f"{frame.cells_shifted():,} cells cleared across {len(handle.rings)} rings\n")

    t = run(frame, args)
    alloc = measure_alloc(frame, args) if args.alloc else None
    print_table(t, alloc, frame)


if __name__ == "__main__":
    main()
