# What really needs to be done — ranked, timeboxed

*Audit of `main` @ `8882ec3`, 2026-09-05. Every item below is a place where the
repo, the docs or the slides say something your own evidence does not support.
None of these are bugs in the code. They are all claims.*

**Do items 1–4 tonight. They are the ones a prepared judge can catch.**

| # | Item | Time | Owner | Severity |
|---|---|---|---|---|
| 1 | 69.8% mIoU is 65.2% | 15 min | Shrestha / JP | **Fatal if caught** |
| 2 | README claims plan regret is proven | 10 min | Aakash | **Fatal if caught** |
| 3 | Dataset-separation claim is not true | 20 min | Srinivas | **Fatal if caught** |
| 4 | 8.94 vs 29.06 MB needs one scripted sentence | 15 min | Shrestha | High |
| 5 | "GPU-accelerated" — there is no CUDA kernel | 20 min | Shrestha | High |
| 6 | Playbook Q8 contradicts known-limitations §5 | 5 min | Shrestha | High |
| 7 | Six docs still call FRNet non-functional | 15 min | JP | Medium |
| 8 | p99 is 100.43 ms against a 100 ms budget | 10 min | Shrestha | Medium |
| 9 | ρ quoted to two decimals it has not earned | 5 min | Aakash | Medium |
| 10 | Two different point-accuracy figures in circulation | 10 min | JP | Low |

---

## 1. 69.8% mIoU is wrong. It is 65.2%. — FATAL

**What happened.** The 15 per-class IoUs on 200 frames of seq 08 sum to 977.7.
Divided by 15 that is 65.18%. Divided by 14 it is 69.84%. The fifteenth class is
`other-ground`: 150 ground-truth points across 200 frames, IoU 0.0%, present in
the data and therefore counted. Someone dropped it. Point accuracy reproduces
exactly at 90.3%, so this is arithmetic, not a model or data difference.

The scripts were corrected. The documents were not.

**Still carrying 69.8%:**

```
docs/handover-2026-09-02.md:23     docs/handover-2026-09-02.md:169
docs/demo-runbook.md:229           docs/perception-dashboard-summary.md:150
src/perception/CLAUDE.md:9         src/perception/semantics.py:14
scripts/frnet_fast_scatter.py:33   scripts/frnet_fast_scatter.py:159
```

**And it is almost certainly on your slides.** Search the deck for "69.8" and
"69.8%" before you do anything else.

**Why it is fatal rather than embarrassing.** A judge who knows SemanticKITTI
knows the 19-class convention. If they ask how you got mIoU and you say "mean of
the per-class IoUs", the next question is "over how many classes", and the honest
answer exposes a number you presented as measured. Fix it and you have a
*better* story: "our first figure divided by the wrong class count; we caught it
in our own audit on 4 September and corrected it downward."

**Correct line:** *"90.3% point accuracy, 65.2% mIoU over 15 classes present, on
200 held-out frames of sequence 08, against the paper's 73.3%."*

---

## 2. The README claims plan regret is proven. It is not. — FATAL

**`README.md`, lines 5–10**, currently reads:

> *"It removes dynamic ghosts. And it proves the compression is free by showing
> it does not change the plan a robot would make. That last clause is the
> contribution."*

**Your own money plot, `known-limitations.md` §2 (seq 08, 20 frames, 64-query
mean, matched extent):**

| schedule | MB | R(S) |
|---|---|---|
| uniform 20 cm | 30.50 | **0.251** |
| **5/10/20/40 (ours)** | **29.06** | **0.488** |
| uniform 40 cm | 18.50 | **0.402** |

A uniform 20 cm map at essentially your memory plans **better**, and a uniform
40 cm map at 18.50 MB is cheaper *and* better. `regret_plot.py`'s own
monotonicity guard fires on that step, by design.

**This is not fatal to the project. It is fatal to that sentence.** The reason
the result does not appear is understood and documented: `PLAN_LANE_CELLS` runs a
single longitudinal lane down the middle of the window, unchanged since Day 0,
and a lane query rewards a map that is uniformly adequate along one line. It
structurally cannot reward a map that is sharp where the vehicle is looking. It
is proof the query cannot demonstrate the thesis, not proof the thesis is wrong.

**A patch is provided in `README.patch.md` in this directory.** Apply it or write
your own, but the sentence cannot ship as written.

**Say instead:** *"The two frozen schedules produce identical plans despite a
5.4 MB difference between them, and matched to the same ground we cost 29.06 MB
against uniform 10 cm's 78.50. What we have not yet shown is a regret knee —
our only planning query is a longitudinal lane that cannot discriminate between
resolutions, and designing one that can is the first thing on our Day 7 list."*

---

## 3. The dataset-separation claim is not true — FATAL

**`docs/defense-rehearsal-playbook.md` Q6** currently answers "did you tune on
your test set?" with a clean three-way split: 00 for scaffolding, 07 for tuning,
08 as "the completely unseen test sequence".

**`docs/known-limitations.md` §2b, first paragraph:**

> *"Everything in this project was measured on 07 and 08 until 2 Sep, and the
> honest reason for those two is that they downloaded first."*

These cannot both be true, and the second one is in the document that exists to
be honest. Worse, §2b goes on to flag that **07 and 08 sit at the good end of the
distribution** — their ring-1 ρ of 1.32 and 1.30 against a median of 1.45 across
eleven sequences.

**Why this is the most dangerous item on the list.** Items 1 and 2 are wrong
numbers. This one is a claimed methodology that did not happen, and a panel that
catches a fabricated experimental protocol stops believing everything else. It is
also completely unnecessary — you have a *better* answer available.

**Say instead:** *"Not a clean split, and we should say so precisely. 07 and 08
were the first two sequences we had on disk and most development happened against
them. What we did once the full 84.8 GB landed was re-run the accuracy table
across all eleven labelled sequences, and we report the distribution rather than
our best sequence: ρ median 1.45, range 1.26 to 1.59, n = 11. 07 and 08 are at
the good end of that range, which is exactly why we quote the range. The one
place a real hold-out exists is the segmentation model — the FRNet checkpoint was
trained on 00–10 excluding 08, and every FRNet number we quote is on 08."*

That answer is stronger than the fabricated one, because it is checkable.

**Edit Q6 in the playbook tonight.** Anyone rehearsing off the current version is
rehearsing a claim that will not survive contact.

---

## 4. 8.94 MB and 29.06 MB will meet on a slide

Both are correct and they measure different things:

- **8.94 MB** — 745,000 logical cells × 12 B. The **map**. All the compression
  ratios (21.5× vs uniform 2.5D's 192 MB, 286× vs dense 3D's 2.56 GB) are
  cell-count ratios computed on this.
- **29.06 MB** — what `allocate()` commits at startup: the map *plus* every
  frame-path working buffer. This is what the dashboard counter shows, and what
  `regret_plot.py` puts on the memory axis (against uniform 10 cm's 78.50 MB at
  matched extent, computed the same way).

`src/gpu/CLAUDE.md` already warns: *"Do not let the two meet on a slide."* They
will. Get in front of it.

**The scripted sentence, for whoever shows the memory slide:**

> *"Two numbers, and they answer different questions. The map is 8.94 megabytes —
> that is what the compression ratios are computed on, and they are pure
> cell-count ratios. The total we commit at startup is 29.06, because we
> preallocate every working buffer too rather than allocating in the frame loop.
> Against a uniform 10 cm grid covering the same ground, measured the same way,
> that is 29.06 against 78.50."*

Volunteering the less flattering number is the whole play. It costs one sentence.

---

## 5. "GPU" — be precise, because the code will not back the loose version

**What exists.** `src/gpu/` is a GPU-*shaped* design implemented in numpy on CPU.
`allocators.array_module()` returns `numpy` for `device="cpu"` and imports `cupy`
for anything else, but **cupy appears in no test and no script**, and every
latency figure you have was measured single-threaded on an Intel i7-14650HX.
`kernels.py`'s own docstring says its atomic path behaves "exactly as a CUDA
kernel would issue atomicAdd" — which is a statement about fidelity of the
reference, not about a kernel that runs.

The only thing in this project that actually executes on CUDA is FRNet inference,
through torch, and FRNet is not in the mapping pipeline.

**Why the loose version is dangerous.** "GPU-accelerated mapping" invites "show me
the kernel," and there isn't one. That is a bad thirty seconds.

**Why the precise version is still strong.** Every architectural decision in
`src/gpu/` is GPU-motivated and GPU-portable, and each one is measured:

| Decision | Motivation | Measured |
|---|---|---|
| Structure-of-arrays | coalesced access | layout, no runtime cost |
| int32 fixed-point, never float atomics | float atomics are non-associative → non-reproducible maps | determinism test is CI-blocking |
| `scatter_sorted` vs `scatter_atomic`, bit-identical | sort beats contention | p50 6.65 vs 20.56 ms @ 120k returns |
| Zero allocation in the **back end's** frame loop | allocation is the p99 | 8.15 → 1.31 MB/frame, p99 74.7 → 49.4 ms — **back end only** (synthetic path); the perception front end allocates ~39.5 MB/frame on real seq 08, and the CI test measures **retained growth, not churn** |
| Toroidal O(perimeter) shift | O(area) scroll is the naive version | 0.04 ms vs 15.2 ms |
| `np.take(mode="clip")`, intp indices | bounds-check copies | 3.2 MB → 1 KB/frame |
| Page-touch at startup | first-frame faults land in p99 | 29.06 claimed / 28.82 resident |

**Say:** *"The mapping pipeline is a CPU reference implementation in numpy, and
every number we quote is measured on it. It is written as GPU code would be —
structure-of-arrays, integer fixed-point accumulation so the result is bit-identical
run to run, zero allocation in the frame loop, and a device seam in the allocator
so the arrays can move to cupy without touching the kernels. We did not write CUDA
because determinism and the memory bound were the load-bearing claims and both are
harder to hold on a GPU, not easier. The one thing that does run on GPU is the
segmentation network."*

If asked "so where is the GPU scaling work?", point at the FRNet reduction swap:
seven `for`-loops replaced with `torch.scatter_reduce_`, **1408× on max and 541×
on mean at real shapes on CUDA**, which took a fine-tune from 3.3 hours to 2.2
minutes. That is real, measured, GPU work and it is yours.

---

## 6. Playbook Q8 contradicts known-limitations §5

**Playbook Q8** says: *"We did not simply fit a bigger number, because the peak
scales with sequence length — frames ×3.70 from 07 to 08, peak ×1.45."*

**`known-limitations.md` §5** says: *"The peak does **not** scale with sequence
length — 00 is the longest and the lowest, which refuted the first version of
this argument. It tracks scene density instead."*

Measured peaks: 314,442 (07, 1,101 frames), 455,714 (08, 4,071), **278,226 (00,
4,541)**. §5 is right; the playbook is carrying the refuted version.

**The corrected argument is better anyway,** because "unpredictable" is a stronger
case for a structural bound than "growing":

> *"We did not fit a bigger number, because we could not find a predictor to fit
> against. The peak does not track sequence length — sequence 00 is the longest
> and has the lowest peak. It tracks scene density, it varies 1.64× across the
> three sequences we measured, and nineteen sequences are unmeasured. That
> unpredictability is the case for making truncation impossible by construction
> rather than unlikely by measurement."*

---

## 7. Six documents still call FRNet non-functional

Flagged on 3 September, still unchanged. The port scores 90.3% point accuracy.

```
CLAUDE.md:66                    data/README.md:29
docs/team-assignments.md:110    docs/known-limitations.md:807
docs/execution-plan.md:37, :233 docs/master-v4.md:280
src/perception/semantics.py:295 ("Disabled -- the standalone port is non-functional")
```

**This matters on stage** because "we tried a deep learning model and it didn't
work" and "we ported a deep learning model to 90.3% accuracy and then chose not
to put it in the pipeline" are completely different stories, and the second one is
both true and much better. The choice not to use it is a *methodological* one:
using ground-truth labels isolates the mapping contribution from segmentation
error. Say that as a design decision, never as a fallback.

---

## 8. The p99 misses 10 Hz by 0.43 ms

End-to-end, 200 frames of seq 08, real data, both halves on one timer:
**p50 89.18 ms, p99 100.43 ms, max 109.28 ms**, against a 100 ms budget.

`src/gpu/timing.py`'s own docstring sets the standard this fails: *"A pipeline
that clears 10 Hz on the median and misses it one frame in a hundred has dropped
a frame of obstacles."*

Do not hide this and do not lead with it. If latency comes up:

> *"Median 89 ms against a 100 ms budget, so 10 Hz with about 11 ms to spare. The
> p99 is 100.4, so we miss one frame in a hundred by four tenths of a millisecond.
> That is a real miss and we are not going to round it away. Two things about it:
> the whole pipeline is single-threaded numpy on CPU with no kernel written yet,
> and the largest stage is visibility cleanup at 26 ms, which is the most
> parallel thing in the system."*

Also: `scripts/timing_table.py:13` still says there is no end-to-end loop to time.
That is now false — `src/run/engine.py:297–309` calls the real `scatter_sorted`
and `fuse`. Stale docstring, 2-minute fix, and it is the script a judge would
ask you to run.

---

## 9. ρ is quoted to two decimals it has not earned

`known-limitations.md` §7 flags it: the §9.2 band-filter fix changes the scored
population, and ρ moves by up to 0.06 per ring on the synthetic sequence. §2b's
eleven-sequence table was generated before that fix.

**On a slide, write ρ ≈ 1.45 (1.26–1.59, n = 11), not 1.45.** The finding survives
either way — a 0.06 shift does not move ρ out of its band — but the second decimal
is not currently backed. If asked, say the table wants regenerating with the band
filter and that is a Day 7 item.

Related, and worth knowing before it is asked: **ring 0 has no ρ on any sequence.**
This is arithmetic, not physics. `block_stats` counts observed *cells*, a ring-0
footprint is exactly one cell, so `n_ref` can never exceed 1 and the `n_ref > 1`
guard drops it every time. Closing it would score ring 0 at ρ 1.01–1.24, the best
of any ring. It is deferred because the same change moves ring 1 by −21.5% on 07
and invalidates every cached reference map, and every ρ moves in the direction
that flatters you — which is the worst direction in which to ship a headline
change the night before submission. Say it exactly that way.

---

## 10. Two point-accuracy figures are in circulation

**98.3%** is one frame (seq 00, frame 43), from the port-validation work.
**90.3%** is 200 frames of seq 08, which is the reported number.

Both appear in the repo without qualification (`src/perception/CLAUDE.md:9`,
`semantics.py:14`, `handover:169`). Only quote 90.3%, and if 98.3% comes up,
identify it as the single-frame port check.

---

## Suggested order tonight

1. Grep the deck and the script for `69.8`, `proves`, `unseen`, `GPU-accelerated`.
2. Apply the README patch (item 2).
3. Rewrite playbook Q6 and Q8 (items 3, 6).
4. Write the three scripted sentences (items 4, 5, 8) onto speaker notes.
5. Everything else is documentation hygiene and can slip to Day 7.
