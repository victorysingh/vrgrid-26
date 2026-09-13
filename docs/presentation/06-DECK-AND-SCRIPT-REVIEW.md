# Deck and script review — `SIH26053_vrgrid.pptx` + `VRgrid_Presentation_Script.docx`

*Audited 2026-09-05 against `main` @ `8882ec3`. The deck is well built and the
script is well written — the delivery notes are better than most SIH scripts I
would expect to see. The problems below are all **claims**, not craft.*

---

## 🔴 STOP — one claim in this deck is checkable and false

**Slide 3, section 5:** *"GPU ACCELERATION — CUDA kernels for project, fuse,
split, merge"*
**Slide 3, Stack:** *"Python 3.11 · C++17 · CUDA kernels"*
**Script, slide 3:** *"All of this runs as CUDA kernels — projection, fusion,
split, merge."*

**There is no CUDA in this repository. There is no C++ in this repository.**

```
$ find . -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.hpp"
(nothing)

$ ls include/vrgrid/
__init__.py   api.py   cell.py        ← the "C++17 frozen interfaces" are Python
```

`src/gpu/` is numpy on CPU. `allocators.array_module()` imports cupy for
`device != "cpu"`, but cupy appears in no test, no script, and no measurement.
Every latency number in the project was measured single-threaded on an Intel
i7-14650HX. The only thing that touches CUDA is FRNet inference through torch,
and FRNet is not in the mapping pipeline.

**Why this is the worst item in either file, by a distance.** Slide 6 gives the
judges `github.com/Stxtics03/vrgrid` and calls it public. A judge who clicks it
during or after your session and greps for `.cu` finds nothing. Every other
number in this deck is real and carefully derived, and this one claim would put
all of them in doubt. Wrong numbers are recoverable; a fabricated implementation
is not.

**The honest version is still a strong slide** — see §1 below for replacement
text. You have real, measured, systems engineering here. It just isn't CUDA.

---

## 1. Slide 3 — replacement text

**Replace section 5 entirely:**

> **5. DETERMINISTIC PARALLEL BACK END**
> Structure-of-arrays for coalesced access · int32 fixed-point accumulation, never
> float atomics · zero allocation in this back end's frame loop (8.15 → 1.31 MB/frame; front end not instrumented) ·
> toroidal O(perimeter) ego-shift, 0.04 ms against 15.2 ms for an O(area) scroll ·
> two scatter paths asserted bit-identical, sorted p50 6.65 ms vs atomic 20.56 ms

**Replace the Stack line:**

> Python 3.11 · NumPy SoA, int32 fixed-point atomics — integer addition is
> associative, so the map hash is bit-identical run to run · cupy device seam in
> the allocator · PyTorch/CUDA for the segmentation model

**Move `2.45 ms rebuild` out of the GPU section.** That figure is the
*conservative pyramid's* rebuild time, and the pyramid is a stretch item that is
**off by default** (enabling it moves the preallocated total 29.06 → 32.17 MB).
Presenting it as the headline throughput number under "GPU acceleration" is two
misattributions in one line, and it is on your script's "memorise cold" list.

**Add the number that is actually missing: end-to-end latency.**

> Frame latency, 200 frames of seq 08, both halves on one timer:
> **p50 89.18 ms, p99 100.43 ms** against a 100 ms budget

You currently have no latency claim anywhere in six slides, and "does it run in
real time?" is a certainty. Better to own the p99 than to be asked for it.

**Why the honest version reads better:** "we wrote CUDA" is a claim any team can
make. "We chose integer accumulation over float atomics because IEEE addition
isn't associative and a map that changes between runs can't be debugged" is a
claim that demonstrates you understand *why* parallel code is hard. That is the
more impressive sentence, and it is true.

---

## 2. Slide 4 — three fixes

**a) "Runs on a Jetson-class GPU. No discrete card needed."**
Never tested. No Jetson measurement exists anywhere in the repo, and there is no
GPU code to run on one. Delete or downgrade:

> Designed for Jetson-class deployment — 8.94 MB of map, no dynamic allocation
> in the frame loop, and no dependency on a discrete card. Not yet measured on
> target hardware.

The added sentence costs nothing and removes an unverifiable claim.

**b) "50 commits" — it is 206.** `git rev-list --count HEAD` = 206. Understating
your own work by 4×. Fix it; 206 commits with green CI is a much better line.

**c) "Nothing is trained"** — true for the shipped pipeline and worth keeping,
but it leaves you with no answer to *"where is the machine learning?"* under a
Smart Vehicles theme. You have a real answer and it is nowhere in the deck. See
§5.

**Correct and unchanged:** the risks table, the memory table, the `5/10/50`
ablation at 6.24 MB (520,000 × 12 B ✓), the sparse-3D disclosure at 130–240 MB,
the derived scope limits (3.74 m blind cone, 8.3 m pothole range). That memory
table with the unflattering sparse-3D row volunteered is the single best-judged
thing in the deck.

---

## 3. Slide 5 — the regret section is stale, and it contradicts its own notes

**The slide body says:** *"On the synthetic scene — R(S) = 0.000 for all six
schedules at 24 frames."*
**The slide's own speaker note says:** *"on the synthetic scene R(S) = 0.207."*

Both are superseded. The synthetic framing is no longer where this stands.

**What the slide says is the blocker:** *"the knee will be measured, not chosen"*
— implying a download is pending. **The download completed:** 43,552 scans,
84.8 GB, all 22 sequences. `scripts/data_status.py` exits 0.

**And the real-data result ran.** Sequence 08, 20 frames, 64-query mean, matched
extent:

| schedule | MB | R(S) |
|---|---|---|
| uniform 10 cm | 78.50 | 0.231 |
| uniform 20 cm | 30.50 | **0.251** |
| **5/10/20/40 (ours)** | **29.06** | **0.488** |
| uniform 40 cm | 18.50 | **0.402** |

**A uniform 20 cm map at essentially our memory plans better than we do.** If a
judge reads "the download is the blocker" and asks the obvious follow-up — "you
have the data now, what did it show?" — the current script has no answer and the
true answer is unfavourable. That is the worst possible way to encounter it.

**Replacement bullet for slide 5:**

> **Measured on real data, and the result is honest.** On seq 08 at matched
> extent, the two frozen schedules produce **identical plans** despite 5.4 MB
> between them, and the uniform series degrades monotonically with cell size.
> But a uniform 20 cm map at comparable memory currently scores better than
> ours — because our only planning query is a longitudinal lane, and a lane
> query cannot reward a map for being sharp where the vehicle is *looking*.
> **The query is the limit, not the metric.** Designing one that can
> discriminate resolution is the next item.

That is a genuinely good slide. It shows you built the evaluation, ran it,
got an inconvenient answer, and diagnosed why — which is a research result.

**Also on slide 5:**

- **"Coarsening ratio ρ = 1.18–1.84 on both sequences"** — supersede this. You
  now have **ρ median 1.45, range 1.26–1.59, n = 11 sequences** at ring 1. Eleven
  independent recordings is a far stronger claim than two, and 1.84 is 08's ring
  3, the weakest number in the set. Also: quote ρ as ≈1.45 with its range, not
  to two decimals — the §9.2 band-filter fix moves it by up to 0.06 and the
  eleven-sequence table predates it.
- **"Measured on SemanticKITTI sequences 07 and 08"** — change to eleven.
- **"(5.803 against 0.146; 1.793 after)"** — these predate the three metric
  fixes of 2 September. Drop them; they are unreproducible and invite a question
  you cannot answer.
- **"Ghost trails behind moving vehicles | 0 of 4,071 frames inert"** — the deck
  is *correct* but the row label invites exactly the misreading your script
  makes (see §4a). Reword the row to **"Ghost-removal coverage across seq 08"**
  and put the real removal figure beside it: **13.5% of trail removed, 4.96 M
  cells cleared, 429,012 cells spared by the current-return guard.**

---

## 4. Script — five corrections beyond the deck's

**a) Slide 5, and this one is a real misstatement.** The script says:

> *"Ghost trails behind moving vehicles: zero, across all 4,071 frames, once the
> elevation fix was toggled on."*

The measurement is **"0 of 4,071 frames inert"** — meaning the cleanup now fires
on every frame, where before the fix 57% of frames did nothing. It does **not**
mean zero ghosts remain. The measured removal is **13.5% of the trail**. Saying
"zero ghost trails" while your own figure script prints 13.5% is a contradiction
a judge can find in your repo.

Say instead: *"Before the fix, 2,304 of 4,071 frames on sequence 08 did nothing
at all — the cleanup silently stopped working above about 11 metres of
elevation. After it, zero inert frames, clearing 15 to 20 thousand cells on every
frame at every elevation across a 39-metre climb. Measured removal is 13.5% of
the trail, 4.96 million cells, with 429,000 cells protected by the guard that
stops it eating fences and poles."*

The before/after is a better story than the false absolute, and the "cells
protected" number is the one nobody volunteers.

**b) Slide 3, "the whole grid rebuilds in 2.45 milliseconds."** That is the
pyramid, not the grid, and the pyramid is off by default. Replace with the frame
latency (§1).

**c) Slide 4, "fifty commits."** 206.

**d) Slide 6, "with official KITTI ground-truth poses."** Contradicts slides 3
and 4, which correctly say SLAM poses on 00 and 08. Fix slide 6 to match — and
note slide 6 also says the dataset is "sequences 00 / 07 / 08" when your headline
accuracy result now spans eleven.

**e) "Numbers to have memorized cold."** Replace the list:

| Drop | Keep / add |
|---|---|
| ~~2.45 ms rebuild~~ | **89.18 / 100.43 ms** frame p50/p99 |
| ~~0 of 4,071 (as "zero ghosts")~~ | **13.5% trail removed, 429,012 spared** |
| | **ρ ≈ 1.45 (1.26–1.59, n = 11)** |
| | **8.94 MB map / 29.06 MB total preallocated** |
| Keep | 99.87%, 286× / 21.5×, 10.8 m at 50 m, 8.3 m pothole limit |

---

## 5. What is missing: your deep learning work

Six slides and no mention of FRNet. Under a **Smart Vehicles / Software** theme,
"where is the AI?" is likely, and right now the deck's only answer is *"nothing
is trained"* — which reads as an absence rather than a decision.

You have a strong, complete story sitting unused:

- Ported FRNet from ~15% to **90.3% point accuracy / 65.2% mIoU** on 200
  held-out frames of seq 08, against the paper's 73.3%, by finding three
  divergences: the wrong activation at 7 sites, a config overriding the trained
  projection FOV with the sensor's physical FOV, and a densification transform
  that had to be reproduced verbatim *including its off-by-one*.
- Found its forward pass was **10.5 s/frame with ~90% in a Python loop**, and
  replaced seven reduction loops with `torch.scatter_reduce_`: **1408× on max,
  541× on mean at real shapes on CUDA**, taking a fine-tune from 3.3 h to 2.2 min.
- **Fine-tuned it three ways and rejected all three on measurement**, because the
  checkpoint was already trained on this data and there was no domain gap.
- And then **deliberately kept it out of the mapping pipeline**, so segmentation
  error cannot contaminate the mapping result.

**That last point is the one that lands.** It reframes "nothing is trained" from
a limitation into experimental discipline.

**Suggested addition — one block on slide 3 or 4, or a backup slide:**

> **The model we built and chose not to use.** FRNet ported to 90.3% point
> accuracy / 65.2% mIoU on held-out seq 08 (paper: 73.3%), with a 1408× reduction
> speedup we contributed to its encoder. Fine-tuned three ways, rejected on
> measurement. **Kept out of the pipeline on purpose** — labels come from ground
> truth so the mapping contribution is evaluated independently of segmentation
> quality.

**This also does not exist in the deck and should:** ⚠️ **65.2%, never 69.8%.**
The 69.8 figure circulating in `handover-2026-09-02.md` and `demo-runbook.md` is
an arithmetic error — the 15 per-class IoUs sum to 977.7, and someone divided by
14, dropping `other-ground`. If 69.8 is anywhere in your notes, remove it.

---

## 6. Slide 2 — one small trap

*"Fixed budget — 745,000 cells × 12 B = 8.94 MB"* is correct and the derivation
on the slide is exactly right. But **29.06 MB** is the total you commit at
startup, it is what the dashboard counter shows, and it is what the regret plot
puts on its memory axis. If both surface, unexplained, it reads as
cherry-picking.

One clause fixes it: *"…8.94 MB of map, inside a 29.06 MB total preallocated
footprint — the ratios are on map memory and we'll show you both."*

---

## 7. What is genuinely good — do not change it

- **Slide 2's opening.** Leading with 10.8 m and 99.87% instead of "maps are
  big" reframes the whole project from thrift to physics. This is the strongest
  structural decision in the deck.
- **Slide 4's risk table**, and the delivery note telling you not to rush it.
  Right call.
- **Volunteering the sparse-3D baseline** at 130–240 MB next to your own 286×.
  Very few teams hand over the unflattering comparison. Keep it and say it out
  loud.
- **Slide 6's prior-art-first framing.** Naming Triebel, Droeschel and Losasso
  before anyone asks is exactly right.
- **The σ = 6.3 cm kerb example** on slide 2. Verified against `sih-math.md`
  §322 — correct, concrete, and it makes an abstract theorem tangible in one
  number.
- **The script's tone guidance** ("say the limitation lines calmly, not
  apologetically"). That instinct is what will carry this presentation.

---

## 8. Edit checklist

**Deck:**

- [ ] Slide 3 §5 — replace "CUDA kernels" block entirely (§1)
- [ ] Slide 3 Stack — remove "C++17 · CUDA kernels" (§1)
- [ ] Slide 3 — add frame latency p50 89.18 / p99 100.43 (§1)
- [ ] Slide 3 — move/delete "2.45 ms rebuild"
- [ ] Slide 4 — "50 commits" → 206
- [ ] Slide 4 — qualify the Jetson claim (§2a)
- [ ] Slide 5 — replace the synthetic R(S) = 0.000 bullet with the real-data result (§3)
- [ ] Slide 5 — ρ 1.18–1.84 on 2 seqs → ρ ≈ 1.45 (1.26–1.59, n = 11)
- [ ] Slide 5 — delete "(5.803 against 0.146; 1.793 after)"
- [ ] Slide 5 — reword the ghost row, add 13.5% / 429,012
- [ ] Slide 5 header — "sequences 07 and 08" → eleven labelled sequences
- [ ] Slide 6 — fix the pose contradiction, widen the dataset line
- [ ] Slide 2 — add the 29.06 clause
- [ ] Add the FRNet block (§5)
- [ ] Slide 1 — Team ID and demo link (your own note)

**Script:** use `07-CORRECTED-SCRIPT.md`, which has all of the above applied.
