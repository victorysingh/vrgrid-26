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

## 9. Does the allocation mechanism generalise? Partly.

`transform` was found one stage at a time. The obvious next question is whether
allocation explains the *rest* of the tail, so: per-stage allocation for the whole
pipeline, measured by wrapping `Timer.stage` — the one hook every stage boundary
in both halves passes through, so no second list of stage names can drift from
`timing.STAGES`.

41 frames of seq 08:

| stage | peak alloc MB | time p50 | p99−p50 |
|---|---|---|---|
| `range_image` | **14.30** | 34.46 | 3.11 |
| `transform` | **10.87** | 3.85 | 0.81 |
| `cleanup` | **9.61** | 30.00 | 7.67 |
| `reflectivity` | 6.56 | 5.30 | 3.17 |
| `fuse` | 5.62 | 5.30 | 0.99 |
| `ground` | 4.75 | 21.31 | **8.85** |
| `scatter` | 4.56 | 8.32 | 0.94 |
| `semantics` | 2.17 | 0.86 | 0.18 |
| `motion` | 0.87 | 0.14 | 0.08 |
| `shift` | 0.07 | 0.87 | 0.70 |
| **`bin`** | **0.04** | 7.88 | 0.75 |
| **total per frame** | **59.40 MB** | | |

### Three things worth taking from this

**1. The pipeline allocates ~59 MB per frame.** At 10 Hz that is ~594 MB/s of
churn. `transform`'s 10.86 MB is under a fifth of it.

**2. `bin` at 0.04 MB is the control, and it is a good one.** `bin` is on the grid
path, which *is* covered by
`test_the_two_grid_allocations_stay_fixed` — and it allocates essentially nothing,
while every uncovered perception stage allocates 2–14 MB. **The invariant works
where it is enforced.** That is the clearest possible argument for R-g, extending
it to the perception half.

**3. Allocation is *a* mechanism, not *the* mechanism.** Rank correlation between
per-stage allocation and per-stage p99−p50 is **+0.64** (Pearson +0.37) —
supportive but not decisive, and two rows break the pattern outright:

- **`ground` has the largest spread (8.85 ms) on modest allocation (4.75 MB).**
  Its tail is most likely inside the Patchwork++ C++ extension, where none of this
  applies. It will not yield to an allocation fix.
- **`range_image` is the biggest allocator (14.30 MB) with a middling spread.**

### [!] The instrument perturbs what it measures

`tracemalloc` hooks every allocation, which both inflates absolute times and
appears to **smooth the very spikes under investigation** — `transform`'s spread
reads 0.81 ms here against **22.14 ms** in the uninstrumented probe. So the
correlation above is computed on a compressed tail and should be read as
*suggestive of ranking*, not as a measurement of effect size. The allocation
column is exact; the timing column, under this instrument, is not comparable to
§5's.

This is why `transform` was confirmed by an isolated A/B (§6) rather than by this
table. **Any further stage should be confirmed the same way** — allocation-free
arm against as-shipped arm, on identical preloaded data, without `tracemalloc`
running.

### So the next step for R-h

`cleanup` is the better target than `ground`: 9.61 MB allocated, 7.67 ms spread,
and it is **our own numpy** rather than a C++ extension, so the same
preallocation A/B that settled `transform` can be run against it directly.
`ground` should be treated as a separate problem with a different cause.

## 10. R-h: `cleanup`'s allocation is `np.isin`, in an argument

Following §9's recommendation to take `cleanup` next rather than `ground`, and
confirmed by the isolated A/B §9 demands rather than by the `tracemalloc` table.

**The stage is not allocating where it looks like it is.** `visibility_cleanup`
promises *"no allocation when handed a scratch"* and the engine hands it one, so
none of the 9.61 MB is in the eq. (32) pass. It is all in the three lines of
candidate selection before the call. On a real 30-frame seq-08 map (910,000
slots, 382,345 occupied):

| operation | peak alloc | p50 | p99 |
|---|---|---|---|
| `state == OCC_OCCUPIED` | 0.91 MB | 0.115 ms | 0.338 ms |
| `np.flatnonzero(mask)` | 3.97 MB | 2.333 ms | 2.632 ms |
| **`np.isin(occupied, touched)`** | **8.18 MB** | **5.677 ms** | **7.711 ms** |

`np.isin` dominates both columns and it sits inside an otherwise exemplary line:

```python
np.copyto(guard, np.isin(occupied, touched))
```

**The `np.copyto` into a preallocated buffer is right. The allocation is in the
argument.** `np.isin` builds an 8.18 MB temporary, sorting internally, before a
byte is copied — so the preallocation is defeated by the expression feeding it.
That is a different failure from `transform_points`, which allocates openly; this
one is invisible precisely *because* the surrounding line is careful.

A boolean lookup table — both arrays index the same slot space, so membership is a
lookup, not a search — gives **0.38 MB, 1.472 ms p50, 2.059 ms p99: 21× less
allocation, 3.9× faster, and `np.array_equal` identical output.**

Written up in `pending-review/cleanup-isin-guard.md`, not applied: `engine.py` is
Shrestha's and the LUT arguably belongs in `allocate()`.

**Two negative results worth keeping.** `np.take(out=)` is *slower* than plain
fancy indexing (3.633 vs 1.472 ms), so the obvious further optimisation backfires.
And a `searchsorted` probe read 51 ms — but it was **not a correct membership
test**, so that number measured the wrong thing and should not be cited as
evidence against `searchsorted`; the LUT is simply the right structure.

## 11. [!] `ground` — ATTEMPTED AND INCONCLUSIVE, and the reason matters more

> **Superseded by §12** — re-measured on a restarted, calibrated machine. The
> account below of *why* the first attempt failed stands; its conclusion does not.

I tried to settle `ground` the same way, and **I am not reporting a conclusion,
because the instrument failed.** The attempt is recorded because the failure is
the useful part.

### What went wrong

`reports/harnesses/ground_cost.py` measured Patchwork++ at **21.01 ms p50** earlier
in the session. Re-run late in the same session, unchanged, on the same data:
**93.44 ms p50** — a **4.4× slowdown of the same C++ call**, at 2% CPU load.

The machine, measured at that moment:

| | |
|---|---|
| CPU clock | **1520 MHz against a 2400 MHz base — 37% down**, at 2% load |
| commit charge | **23.61 GB against 15.73 GB physical** — ~8 GB over-committed |
| free physical | 2.2 GB |

So: the CPU had dropped into a low-power/thermal state after hours of sustained
work, **and** the machine was paging. 37% of clock does not explain 4.4×; the
paging does the rest.

**Every absolute timing taken in that window is invalid**, including the `ground`
numbers I measured there (p50 80–91 ms, where the pipeline reports 21 ms). The
`ground` correlations came out mutually contradictory across runs — `corr(time,
n_points)` read −0.592 then +0.532, worst-15 overlap 7/15 then 0/15 — which is
itself the signature of a measurement too noisy to conclude from. I am not
publishing a verdict off that.

### What this does NOT invalidate, and why

The `transform` and `cleanup` findings stand, because **both were A/B arms run
back to back in one process on identical data.** A machine-wide slowdown scales
both arms together, so the *ratios* — 25× smaller tail, 21× less allocation, 3.9×
faster — survive even though the absolutes would shift. That is the difference
between a controlled comparison and a headline number, and it is why §6 and §10
were built as A/Bs rather than as single measurements.

Allocation counts are unaffected entirely: `tracemalloc` counts bytes, not time.

### What a valid `ground` measurement needs

1. A machine **not** 8 GB over-committed, checked before and after rather than
   assumed.
2. CPU clock recorded alongside the timings — `Win32_Processor.CurrentClockSpeed`
   against `MaxClockSpeed`, because a 37% downclock is invisible in a timing table.
3. The A/B discipline of §6 even for a C++ stage: there is nothing to preallocate
   inside Patchwork++, but *fresh vs warm estimator* and *point count* can still be
   contrasted within one process.

### And it strengthens two existing items

- **D8 (one agreed reference host).** This host's absolute timings moved **4.4×
  within a single session**, unprompted. Any latency figure from it needs its
  machine state attached, not just its hostname.
- **R-b §3's warning** that two identical runs disagreed on the median by 17% was
  an understatement. The real spread over a session is far larger.

## 12. `ground`, measured properly: the tail is NOT in Patchwork++

Re-run after a restart, with the D8 discipline applied — machine state recorded
before and after every run, and a calibration gate first.

**Calibration.** `ground_cost.py` reproduced the known-good figure before anything
else was believed: reps **20.95 / 20.33 / 20.12 ms** against the early-session
21.29 / 20.82 / 20.90, flat across reps, fallback back to 0.32 ms, agreement
identical at 96.5%. Machine state held throughout: CPU 2400/2400 MHz, commit
12.85–12.89 GB against 15.73 GB physical.

**Result — `ground_tail.py`, seq 08, 60 preloaded scans, two runs × two passes,
fresh estimator per pass:**

| run / pass | p50 | p99 | max | frames > 1.5× median |
|---|---|---|---|---|
| 1 / 1 | 18.40 | **30.02** | 30.46 | 2 of 60 |
| 1 / 2 | 18.51 | 19.77 | 19.98 | 0 |
| 2 / 1 | 18.55 | 19.50 | 19.60 | 0 |
| 2 / 2 | 18.45 | 19.55 | 19.99 | 0 |

**In isolation Patchwork++ has almost no tail.** In three of four passes
p99 − p50 is **~1 ms** and the worst frame is **≤ 1.08× median**. The one wider
pass is the very first in a fresh process, and does not recur.

Compare the same stage *inside the pipeline*: p99 − p50 of **11.13 ms** in the warm
p99 probe (§5). **The segmenter on its own input does not produce that spread.**

The rest of the evidence agrees it is not the data:

- worst-15 frame overlap between passes: **2 of 15, then 4 of 15**, against ~3.8 by
  chance — noise.
- `corr(time, n_points)` swings between +0.04 and +0.78 across passes. With point
  counts spanning only 1.04× and times within ~1 ms, there is too little variance
  for a correlation to mean anything; it should not be read either way.

### What this establishes, and what it does not

**Established:** `ground`'s in-pipeline tail is **extrinsic** to Patchwork++. The C++
extension is not the thing to optimise, and `ground` comes off the list of stages
with an intrinsic tail.

**Not established — a hypothesis, stated as one:** the most plausible source is the
mechanism §6 proved for `transform`. The frame allocates ~59 MB of transient churn;
when the allocator faults fresh pages in, the stall lands in *whichever stage is
running*, and `ground` is one of the longest. That would make `ground`'s tail a
symptom of other stages' allocation rather than its own. **Testable, not yet
tested:** apply the `transform` and `cleanup` fixes (D9, `cleanup-isin-guard`) and
re-measure `ground`'s *in-pipeline* spread. If it collapses toward ~1 ms, the
hypothesis holds.

### Why this could be answered now and not last night

Same harness, same data. The difference is entirely the machine: the first attempt
ran at 1520/2400 MHz with ~8 GB over-committed and produced mutually contradictory
numbers. That is D8's upgraded scope demonstrated end to end — the question was
always answerable; the instrument was not, until its state was controlled.

## 9. Reproduce

```sh
VRGRID_DATA_ROOT=C:/KITTI/dataset python reports/harnesses/p99_probe.py 08 220 2
VRGRID_DATA_ROOT=C:/KITTI/dataset python reports/harnesses/transform_tail.py 08 220
```

`p99_probe.py` writes `p99_frames.npz` with the per-frame per-stage arrays, so the
tail can be re-analysed without a 7-minute re-run. **Run it twice from a cold
cache and a warm one** — per §5, one run cannot tell you which stage owns the
tail.
