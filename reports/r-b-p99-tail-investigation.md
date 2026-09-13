# R-b — where the p99 tail comes from

**Result, up front:**

- **It is not garbage collection.** Ruled out by direct measurement, not
  argument: 3 collections in a 220-frame run, **0.00% of runtime**, and **0 of
  the 8 worst frames had one.**
- **It is not the data.** The worst frames are *different every run* — the
  worst-15 overlap across reps is **1 of 15, exactly what chance predicts.**
- **It is not thermal.** Drift is +0.07 ms/frame and **plateaus after the first
  quarter**, over a 28-second run. That is warm-up, not heat soak.
- **Which stage owns the tail depends on page-cache state**, and that is itself a
  finding — cold: `load`, 60% of the excess. Warm: **`transform`, 84%.**
- **In the warm case the tail is ALLOCATION, localised to ~15 lines.**
  `transforms.transform_points` allocates **10.86 MB per call** and occasionally
  stalls 8–10×. An allocation-free equivalent has a **25× smaller tail and zero
  spikes**, on identical data.
- **Fixing it does not on its own reach 10 Hz.** Frame p99 goes from ~135 ms to
  roughly ~113 ms against a 100 ms budget. It is the largest single identified
  contributor and the first with a known mechanism, not a cure.

**Written:** 2026-09-13, against `main` @ `92fc7d0`.
**Harnesses:** `reports/harnesses/p99_probe.py`, `reports/harnesses/transform_tail.py`.
**Scope:** measurement only. Nothing in `src/` was modified.

---

## 1. Why this needed doing differently

Everything before this looked at p50. Every p50 lever was measured and none moved
the tail: not `num_iter`, not the optional Patchwork++ stages, not `max_range`,
not warm-up length, not frame count, not the library version. The tail was
treated as noise because nothing that moved the median touched it.

So this looks **only at the worst frames**, and at four candidate causes at once.

## 2. GC is not it

`gc.callbacks` recorded every collection and the frame it landed in, so this is
measured rather than reasoned about. Two independent 220-frame runs:

| | run 1 | run 2 |
|---|---|---|
| collections during the timed loop | 3 | 3 |
| generation | 0 only | 0 only |
| pause, p50 / max | 0.177 / 0.185 ms | 0.196 / 0.227 ms |
| **total GC time / total runtime** | **0.5 ms / 27,719 ms = 0.00%** | **0.6 ms / 24,074 ms = 0.00%** |
| **of the 8 worst frames, how many had a collection** | **0** | **0** |
| frames with a collection, p50 | 121.80 ms | 109.20 ms |
| frames without, p50 | 122.68 ms | 108.67 ms |

Frames *with* a collection are, if anything, marginally **faster**. Only
generation 0 ever runs, and its pauses are two orders of magnitude below the
30–70 ms excursions being explained. **Closed.**

## 3. The data is not it

If particular frames were expensive — denser scans, more objects — the same frame
indices would be slowest every run. They are not:

```
rep1 worst 15:  40  41  44  46  85  92 104 105 150 172 193 196 199 214 217
rep2 worst 15:   6  15  31  45  50  81  82  90  93  97 104 145 203 211 215

overlap: 1 of 15  (frame 104)
expected overlap if purely random: ~1.0
```

**The overlap is exactly chance.** The slow frames are a property of the machine
at that moment, not of the frames.

A second, blunter piece of the same evidence: the two runs disagree on the
*median*, 122.64 vs 104.95 ms — a **17% swing between identical invocations.**
Any single-run latency figure from this host carries that much uncertainty before
the tail is even discussed.

## 4. Thermal is not it either

| | |
|---|---|
| linear fit | **+0.0707 ms per frame** (+15.6 ms over 220) |
| first decile p50 → last decile p50 | 108.62 → 126.28 ms |
| quarter p50s | 115.63 → 125.26 → **122.43** → 125.06 |

There is real upward drift, but it **plateaus after the first quarter** and even
dips in the third — not the monotone slide thermal throttling produces. The whole
run is **27.8 seconds**, which is far too short to heat-soak a laptop CPU. This
looks like allocator and cache warm-up, which §6 then explains directly.

## 5. [!] Which stage owns the tail depends on the page cache

This is why the tail resisted attribution. Two runs of the identical command:

| stage | run 1 (cold) p50 → p99 | share of excess | run 2 (warm) p50 → p99 | share of excess |
|---|---|---|---|---|
| `load` | 17.73 → 52.27 | **60%** | 1.25 → 1.77 | 2% |
| `transform` | 3.09 → 26.98 | 41% | 3.13 → 25.28 | **84%** |
| `ground` | 21.17 → 24.71 | 6% | 21.13 → 32.26 | 42% |
| `cleanup` | 27.28 → 38.33 | 19% | 27.78 → 36.13 | 32% |
| `range_image` | 25.52 → 34.96 | 16% | 25.29 → 29.08 | 14% |
| **FRAME** | **122.64 → 180.33** | | **108.68 → 135.06** | |

**`load` swings 14×** between the two runs — p50 17.73 ms cold against 1.25 ms
warm. `r9-per-stage-latency-and-memory.md` already established this for the
median; it holds for the tail too, and more violently.

**The warm run is the one to reason from.** `load` is file I/O reading `.bin` and
`.label` off disk; a live sensor has no such stage, as `research-log.md` notes.
Attributing a deployment latency budget to it would be measuring the dataset, not
the pipeline.

*(Shares sum past 100% because stage tails do not coincide — which is itself the
point: no single stage is slow on every slow frame.)*

## 6. The warm-case tail is `transform`, and it is allocation

`transform` is the standout: **p50 3.13 ms, p99 25.28, max 33.54** — an 8–10×
excursion in a stage doing **fixed work on fixed-size data**. It is the dominant
contributor in **6 of the 8 worst frames** (+14 to +30 ms over its own median).

Fixed work with 8× variance is not algorithmic. `transform_points` is 6 lines:

```python
pts   = np.asarray(points, dtype=np.float64)          # float32 -> float64 copy
xyz   = pts[:, :3]                                    # view
pts_h = np.hstack([xyz, np.ones((n, 1))])             # np.ones + hstack
return (T @ pts_h.T).T[:, :3]                         # matmul into a fresh (4,N)
```

Four allocations per call, on ~123,000 points.

### Isolated, with the pipeline and the disk removed entirely

Same preloaded scans, three arms, `transform_points` the only thing in the loop:

| arm | p50 | p99 | **max** | spikes >3× p50 | **allocated per call** |
|---|---|---|---|---|---|
| **as shipped** (float32 in) | 3.19 | 20.46 | **34.61** | **4 / 220** | **10.86 MB** |
| float64 in (no cast) | 2.34 | 4.92 | 32.86 | 2 / 220 | 7.90 MB |
| **preallocated** (probe) | **1.06** | **1.29** | **1.37** | **0 / 220** | **0.00 MB** |

**The tail reproduces with nothing else running, and disappears when the
allocation does.** 34.61 ms → 1.37 ms max, a **25× smaller tail**, and not one
spike in 220 frames. p50 also improves 3× and total time 3.5×.

The middle arm localises it further: passing float64 in removes the 2.96 MB
conversion copy and drops p99 from 20.46 to 4.92 — but `max` stays at 32.86,
because the remaining `hstack`/matmul allocations can still stall. **Every
allocation has to go, not just the cast.**

### Why allocation produces a tail rather than a constant cost

10.86 MB per frame, freed each frame. Most of the time the allocator hands back
pages it already holds and the cost is invisible. Occasionally it must take fresh
pages from the OS, and the first touch of each faults — tens of milliseconds, on
an unpredictable subset of frames. That is exactly the observed signature:
**p50 unaffected, p99 8× worse, different frames each run, and warm-up drift that
plateaus.**

### [!] A gap in an invariant this project already enforces

`tests/test_engine.py::test_the_two_grid_allocations_stay_fixed` pins *"no
allocation inside the frame loop"* — and it passes. It covers `MapEngine.step`,
the grid half. **The perception half is not covered, and it allocates 10.86 MB
per frame in a single function.** The invariant is real and the test is correct;
its *scope* is narrower than the invariant's name suggests.

## 7. What this buys, honestly

`transform`'s p99−p50 spread is 22.14 ms of the frame's 26.38 ms excess. From the
dumped per-frame arrays, removing that stage's contribution entirely takes frame
p99 from **135.06 → 117.75 ms**; replacing it with the preallocated version
(p99 1.29 ms) lands in the same region, roughly **113 ms**.

**Against a 100 ms budget, that is still over.** So:

- It is the **largest single identified contributor to the tail**, and the only
  one so far with a mechanism rather than a correlation.
- It is **not sufficient**. The next candidates are `ground` (42% share in the
  warm run, and the worst single frame was `ground` +16.2 ms — inside the C++
  extension, so harder to attribute) and `cleanup` (32%).
- **p50 also improves**, by ~2 ms of frame time, which is a real if modest bonus.

## 8. Not attempted, deliberately

No fix is applied. `src/perception/transforms.py` is tracked and this is a
performance change to a hot path on the frame loop, not a docs correction — so it
is written up as a proposal in
**`pending-review/transform-points-allocation.md`** instead, with the preallocated
arm above as evidence rather than as a patch.

Two things that need deciding rather than guessing, both in that file: whether the
function keeps returning a fresh array (callers may retain it) and where a reused
buffer would live, given that `transform_points` is currently stateless and is
called from more than one place.

## 9. Reproduce

```sh
VRGRID_DATA_ROOT=C:/KITTI/dataset python reports/harnesses/p99_probe.py 08 220 2
VRGRID_DATA_ROOT=C:/KITTI/dataset python reports/harnesses/transform_tail.py 08 220
```

`p99_probe.py` writes `p99_frames.npz` with the per-frame per-stage arrays, so the
tail can be re-analysed without a 7-minute re-run. **Run it twice from a cold
cache and a warm one** — per §5, one run cannot tell you which stage owns the
tail.
