# The ring-1 reproduction mismatch: the Patchwork++ singleton, confirmed on seq 07

**Result, up front:**

- **Seq 07 is solved, and it is the determinism bug.** The published ring-1
  figure of 3.60 cm is reproduced *exactly* when one Patchwork++ estimator is
  shared between the M\* build and the map build — the normal call pattern. Give
  each pass its own estimator and it becomes **3.04 cm**. The same root cause as
  the failing determinism gate, found independently from the other direction.
- **Seq 08 has nothing to explain** — it reproduces at every ring under either
  call pattern.
- **Seq 00 is now ALSO solved — same root cause, one step further.** It is the
  singleton again, but carrying state across *sequences*: the published harness
  measures 07, then 08, then 00 **in one process**, so by the time seq 00 runs the
  estimator has already processed ~160 frames of the other two. Every published R1
  figure reproduces **exactly** once that is replicated, seq 00's ring 1 included
  (41,892 cells @ 6.77 cm). **See §7.** This closed within minutes of the harness
  being committed under R-a, having resisted two investigations while it was
  missing.

**Written:** 2026-09-12, against `main` @ `23648e9`.
**Scope:** measurement only. `src/perception/ground.py` was **not** modified; the
singleton was manipulated from scratch scripts.

---

## 1. Ruled out first: code drift

R1 was measured on `main` @ `9b40ff2`. The eval path is **byte-identical**
between then and now:

```
git diff --stat 9b40ff2..HEAD -- src/eval src/grid include configs
    (empty)
```

`src/perception/` differs only in docstring prose (`f3a0337`, the mIoU wording)
and my `loader` error-message change (`fac61c2`) — no executable line in either.
So the mismatch is not a code change.

## 2. The mechanism, reproduced in six lines

The estimator accumulates state across calls, and **re-processing a scan it has
already seen gives a different verdict.** 12 frames of seq 08, one estimator:

| test | result |
|---|---|
| pass 1 vs **pass 2** over the same 12 frames | **1,245 of 1,479,013 points differ (0.084%)**, in 4 of 12 frames |
| pass 2 vs pass 3 | still 191 points differ — it does **not** settle |
| two separate estimators, one pass each | **0 points differ** |
| 5 frames of *different* history, then frame 5 | **0 points differ** vs a virgin estimator |

Two things follow, and the second is the surprise:

1. **A fresh estimator per pass is a complete fix for reproducibility** — 0
   points differ.
2. **It is not "history contaminates later frames".** Five frames of unrelated
   history change nothing. What changes the answer is seeing **the same scan a
   second time**. Ground count falls each time it re-sees a frame (83,598 ->
   83,208 on frame 0).

**This is exactly what `test_real_sequence_replay_is_identical` does** — it calls
`replay()` twice in one process, so the second replay is a second pass over the
same 50 scans on an estimator that has already seen them. The gate is not
flaky; it is correctly detecting this.

## 3. Why it lands on ring 1, and nowhere else

The drifting points are **all inside 25 m**, i.e. entirely within rings 0 and 1,
with the majority in ring 1. Pass-1 vs pass-2 masks over the 40 frames R1 uses:

| seq | points differing | rate | frames hit | 0-10 m | **10-25 m** | 25-50 m | >50 m |
|---|---|---|---|---|---|---|---|
| 07 | 10,424 | **0.213%** | 31/40 | 30% | **70%** | 0% | 0% |
| 08 | 1,437 | 0.029% | 5/40 | 0% | **100%** | 0% | 0% |
| 00 | 5,514 | 0.112% | 19/40 | 50% | **50%** | 0% | 0% |

This accounts for the whole pattern of the mismatch:

- **Ring 2 matches exactly on all three sequences** because **0% of the drift
  reaches beyond 25 m.** There is nothing there to perturb.
- **Ring 1 takes the largest share** of the drift on every sequence.
- **Ring 0 barely moves** (1.78 -> 1.77 on seq 07) despite receiving 30-50% of
  it: ring 0 holds ~2.8 M points inside 10 m, so a few thousand flipped verdicts
  are averaged away. Ring 1 has far fewer returns per cell, so the same absolute
  count shifts cell means.
- **Seq 08 is immune** because its drift is an order of magnitude smaller —
  1,437 points over 5 of 40 frames, against seq 07's 10,424 over 31 of 40.

## 4. The measurement

40 frames, schedule `5/10/20/40`, RMSE in cm. `shared` = one estimator for both
the M\* build and the map build (the normal path, and what the published run
did). `fresh_phase` = a fresh estimator for each.

| seq | mode | ring 0 | ring 1 | ring 2 | ring 1 vs published |
|---|---|---|---|---|---|
| 07 | *published* | *1.78* | *3.60* | *5.91* | |
| 07 | **shared** | **1.78** | **3.60** | **5.91** | **exact match, all 3 rings** |
| 07 | fresh_phase | 1.77 | **3.04** | 5.91 | −0.56 |
| 08 | *published* | *1.17* | *2.31* | *4.89* | |
| 08 | **shared** | **1.17** | **2.31** | **4.89** | **exact match** |
| 08 | fresh_phase | 1.17 | 2.31 | 4.89 | exact match |
| 00 | *published* | *2.74* | *6.77* | *34.10* | |
| 00 | shared | 2.74 | 6.45 | 34.10 | **−0.32** |
| 00 | fresh_phase | 2.74 | 6.46 | 34.10 | −0.31 |

**Seq 07 reproduces to the last digit in `shared` mode, on all three rings.**
That is the confirmation: the published harness used the shared singleton, and
my earlier harness did not.

### Seq 07, with cell counts — an exact match, and a third confirmation

| seq 07 | ring 0 n | rmse | ring 1 n | rmse | ring 2 n | rmse |
|---|---|---|---|---|---|---|
| *published* | *103,182* | *1.78* | *50,153* | *3.60* | *12,703* | *5.91* |
| **shared** | **103,182** | **1.78** | **50,153** | **3.60** | **12,703** | **5.91** |
| fresh_phase | 103,167 (−15) | 1.77 | 50,165 (+12) | **3.04** | **12,703** | 5.91 |

`shared` matches the published run on **every cell count and every value** —
there is no remaining ambiguity about what the published harness did.

And **ring 2's population is byte-identical in all three rows** (12,703). That is
§3's radial finding arriving independently: the drift does not reach past 25 m, so
ring 2 sees neither a different population nor a different value, under any call
pattern. The cells that move are all in rings 0 and 1, exactly where the drifting
points are.

## 5. What this means for the published number

**On seq 07, ~18% of the published ring-1 RMSE is an artifact of the bug.**

In `shared` mode, M\* is built on pass 1 and the map on pass 2 — so **the
reference and the map disagree about which points are ground.** Part of the
measured RMSE is then that disagreement, not the coarsening error the metric is
supposed to report. The direction confirms it: making the two consistent
*lowers* the error (3.60 -> 3.04), which is what removing a contaminant should
do.

So `3.04 cm` is the better estimate of seq 07's ring-1 coarsening error, and
`3.60` is inflated. **This is not a reproducibility annoyance — the bug is
inflating a published accuracy figure.**

`fresh_phase` is the right configuration on the merits, not just the
reproducible one: in production the map is built in a single pass, so each
consumer seeing each scan once is what the pipeline actually does. Two
estimators each making one pass agree exactly (§2), so both sides of the
comparison see the same ground mask.

## 6. Seq 00, as far as it could be taken WITHOUT the harness

Both call patterns give **6.45 / 6.46** against a published **6.77**. The
difference between the two modes is 0.01 cm — noise.

> **Superseded by §7.** I concluded here that the residual was therefore *not*
> the singleton. That was wrong, and wrong in an instructive way: both modes I
> tested started seq 00 from a *fresh* estimator, so the experiment held constant
> the very thing that turned out to matter. It is the singleton, carrying state
> across **sequences** rather than across passes.

### Narrowed: it is a POPULATION difference, not a height difference

Comparing the scored cell count `n` against the published table settles which
kind of difference it is:

| seq 00 | ring 0 n | rmse | ring 1 n | rmse | ring 2 n | rmse |
|---|---|---|---|---|---|---|
| *published* | *82,868* | *2.74* | *41,892* | *6.77* | *11,275* | *34.10* |
| shared | **82,868** | **2.74** | 41,953 **(+61)** | 6.45 | **11,275** | **34.10** |
| fresh_phase | **82,868** | **2.74** | 41,981 **(+89)** | 6.46 | **11,275** | **34.10** |

**Rings 0 and 2 match the published cell count exactly. Ring 1 does not** — I
score 61-89 more cells than the published run did, 0.15-0.21% more. So the
sequences are not disagreeing about heights; they are disagreeing about **which
cells to score**, and only at ring 1.

Note the direction: the published run scored **fewer** cells and got a **higher**
RMSE, so the cells it left out were low-error ones. 61 low-error cells removed
from ~42,000 is enough to move an RMSE by 0.3 cm when the distribution has a
heavy tail — and seq 00's ring 1 does (it contains the `building` at 18.45 cm
and `trunk` at 24.77 cm groups).

The seq-07 control confirms the method: in `shared` mode seq 07 matches the
published **n and RMSE exactly at all three rings** (103,182 / 50,153 / 12,703).
So the harness is right; seq 00 genuinely differs.

### Why it could not be closed at the time

**The published R1 harness was never committed.** The report says so plainly —
*"`scratchpad/r1_accuracy_by_class.py` (session scratch, not committed — a
measurement harness, not a deliverable)"* — and `git log --all --diff-filter=A`
confirms it has never existed in the repository.

That matters because the report also says its ALL row *"groups `_compared`'s
scored population by `unpack_class(...)`"*. If the published ALL row was computed
as a **sum over class groups** rather than over `_compared` directly, then any
cell whose class did not fall into a group would be silently absent from ALL —
which would produce exactly this signature: a slightly smaller n, at one ring,
with the same population everywhere the grouping is unambiguous.

**That is a hypothesis, not a finding, and it is not testable** — the code that
would confirm or refute it does not exist. Per instruction I am stopping here
rather than guessing further.

What *is* ruled out for seq 00:

- **Code drift** (§1) — byte-identical eval path.
- **The estimator call pattern** — measured both ways, 0.01 cm apart.
- **A height difference** — rings 0 and 2 agree exactly on both n and RMSE.
- **A different frame count.** Not formally swept, but ring 0 reproduces the
  published gate to four decimals (2.7377), as does seq 08's (1.1729); a
  different frame count would be unlikely to leave ring 0 matching to that
  precision. The R1 report states "40 frames" only in its seq-07 header.

**This is the second time in two nights that an uncommitted measurement harness
has made a published number unverifiable** — the other being the 80.78 ms
latency figure. The pattern is worth more attention than either individual
number: a harness that produces a figure a report will quote is a deliverable,
whatever its filename says.


---

## 7. Seq 00: CLOSED — the singleton again, one scope wider

**Every published R1 figure reproduces exactly** once the published harness is run
as written. Not approximately — exactly, on all four rings of all three sequences:

| seq | ring 0 | ring 1 | ring 2 | ring 3 |
|---|---|---|---|---|
| 07 | 103,182 @ **1.78** | 50,153 @ **3.60** | 12,703 @ **5.91** | 1,321 @ **16.93** |
| 08 | 137,034 @ **1.17** | 141,141 @ **2.31** | 49,073 @ **4.89** | 6,856 @ **54.86** |
| 00 | 82,868 @ **2.74** | **41,892 @ 6.77** | 11,275 @ **34.10** | 3,379 @ **9.11** |

Including the cell count that was the whole puzzle: **41,892**, not the
41,953/41,981 every separate run produced.

### The mechanism

`reports/harnesses/r1_accuracy_by_class.py` ends with:

```python
for seq in ("07", "08", "00"):
    allrows += run(seq, 40, "5/10/20/40")
```

**All three sequences, one process, one module-level estimator.** Each `run()`
builds M\* and then the map, so each sequence puts *two* passes of 40 frames
through `ground._estimator`. By the time seq 00 is measured, that estimator has
already processed **~160 frames of seq 07 and seq 08**.

That is a state no run of mine reproduced. Every experiment in §4 gave each
sequence a freshly built estimator — which is exactly why seq 00 refused to
match, and why I wrongly concluded the singleton was not responsible.

### Why this produces precisely the observed pattern

| seq | position in the loop | estimator state when measured | result |
|---|---|---|---|
| 07 | **first** | fresh | reproduced exactly in §4 — nothing to explain |
| 08 | second | ~80 frames of 07 | reproduced anyway: its drift is 0.029%, an order of magnitude too small to move a ring (§3) |
| 00 | **third** | ~160 frames of 07 + 08 | the only one that needed the accumulated state, and the only one that failed to match |

The controlled comparison, one variable — prior-sequence history:

| seq 00, shared estimator | ring 1 | n |
|---|---|---|
| fresh process, no prior sequences | 6.45 | 41,953 |
| **after 07 and 08 in the same process** | **6.77** | **41,892** |

Same code, same data, same call pattern. **Only the estimator's history differs,
and it moves ring-1 RMSE by 0.32 cm and the scored population by 61 cells.**

### What it means

The seq-00 published figure is **more** contaminated than seq 07's, not less. On
seq 07 the contamination is one extra pass over the same data; on seq 00 it is
160 frames of two *other* sequences. The consistent-mask estimate for seq 00
ring 1 is **~6.46 cm**, against a published 6.77.

So both open figures resolve the same way, and the D1 fix does more than turn a
CI gate green: it removes a contaminant from **two** published accuracy numbers.

### [!] And the ordering dependence is the sharper finding

A published accuracy figure depends on **which other sequences were measured
before it in the same process.** Nothing in the harness, the report or the metric
names that as an input. Reorder the tuple `("07", "08", "00")` and the numbers
change.

That is worse than non-reproducibility, because it is invisible: the harness is
deterministic, the data is fixed, the code is unchanged, and the number still
depends on evaluation order. Until D1 is fixed, **any harness measuring more than
one sequence in one process is unsafe**, and the honest mitigation is one process
per sequence.

### Credit where it is due: this is what R-a was for

This closed in minutes once `r1_accuracy_by_class.py` was committed, after
resisting two separate investigations while it was session scratch. The mechanism
was a three-line `for` loop at the bottom of a file nobody could read. Both of
this week's unverifiable figures had the same cause and the same cure.

## 8. Recommendation

**Fix the singleton, and prioritise it.** The argument is now stronger than
"a CI gate is red":

1. It **inflates a published accuracy number** by ~18% at ring 1 on seq 07.
2. It has now been found **twice, independently** — once as the determinism gate
   failure, once as this reproduction mismatch — which is what a root cause
   looks like.
3. The fix direction is **measured, not speculative**: two estimators making one
   pass each agree exactly (0 of 1.48 M points differ).
4. Every accuracy number built on the ground mask is currently measured against
   a reference that may disagree with the map about what ground is. That
   undermines the *method*, not just one figure.

**And until it is fixed, no accuracy comparison finer than ~0.5 cm at ring 1 can
be adjudicated.** That is a live constraint: the `num_iter` tradeoff I rejected
moved ring 1 by 1.71 cm on seq 07, comfortably above this floor, so that verdict
stands — but a smaller effect would have been unmeasurable.

I have deliberately **not** attempted the fix. Per standing instruction the
singleton is not to be patched without a real conversation, and the shape of the
fix is a genuine design decision — whether `segment_ground` should take an
estimator, whether the module should expose a reset, or whether the estimator
should be owned by the caller — not something to settle inside an investigation.

## 9. Reproduce

Session-scratch harnesses, not committed:

- `scratchpad/state_minimal.py`, `state_char.py` — the six-line mechanism (§2).
- `scratchpad/mask_drift.py` — per-sequence drift and its radial distribution (§3).
- `scratchpad/ring1_estimator.py` — the `shared` / `fresh_phase` / `fresh_frame`
  comparison (§4).
- `scratchpad/ring1_population.py` — scored-population check for §6.

Requires `VRGRID_DATA_ROOT=C:/KITTI/dataset`.
