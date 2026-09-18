# START HERE — the whole project, in the order you should learn it

> ⚠️ **Provisional as of 2026-09-05.** A stateful-singleton defect in
> `src/perception/ground.py` is under investigation and may affect published
> ring-1 accuracy figures. Any number in this directory derived from ρ should be
> treated as unconfirmed until that issue is closed. Nothing here has been
> updated in anticipation of the outcome — see the tracking issue.

*Written 2026-09-05 against `main` @ `8882ec3`. Nothing in `docs/presentation/`
is pushed. Read this file first; it is the map to the other five.*

| File | What it is | Who needs it |
|---|---|---|
| **`00-START-HERE.md`** | This. The mental model and the numbers you must know cold. | Everyone, first |
| **`01-CRITICAL-FIXES.md`** | Ten things that are wrong right now, ranked, with exact edits. | Do this tonight |
| **`02-WHAT-TO-PRESENT.md`** | Narrative arc, slide skeleton, demo choreography, speaker split. | Presenters |
| **`03-TECHNICAL-STUDY.md`** | The full technical document with references. | Report + deep questions |
| **`04-WHY-THIS-MATTERS.md`** | Impact and framing. Feeds your opening slide. | Whoever opens |
| **`05-PANEL-DEFENSE.md`** | Corrected Q&A. Supersedes parts of the old playbook. | Everyone, memorise |
| **`06-DECK-AND-SCRIPT-REVIEW.md`** | Audit of the .pptx and .docx, with exact slide edits. | Do this tonight |
| **`07-CORRECTED-SCRIPT.md`** | The script, rewritten and ready to read. | Presenters |
| **`08-JP-BRIEF.md`** | JP's scope, lines, questions and tonight's checklist. | JP |
| **`09-GPU-WHAT-TO-SAY.md`** | Every GPU claim: intended vs. what to say. Speakable. | Whoever presents slide 3 |
| **`10-DL-WHAT-TO-SAY.md`** | Same, for the segmentation model. Mostly material you aren't using. | JP |
| **`11-DLSS-DUALITY.md`** | The DLSS analogy, with the six claims its source PDF could not support removed. | Whoever pitches the framing |

---

## 1. The project in five sentences

You built a LiDAR map for a ground vehicle. Instead of storing the world at one
uniform resolution, cell size grows with distance from the vehicle: 5 cm nearby,
40 cm far away, in concentric square rings. The reason is physics rather than
thrift — at 50 m the sensor's beams land 10.8 m apart on the road, so a 5 cm cell
out there stores an interpolation, not a measurement. All the memory is allocated
before the first scan arrives and nothing in the frame loop grows, so the
footprint is a bound rather than an average. On top of that sit three things:
ghost removal for moving objects, curb/pothole detection, and an evaluation that
asks whether the compression changed the route a planner would pick.

**If you can only say one sentence:** *"We put the resolution where the sensor's
measurements actually are, under a memory bound fixed at startup, and we measured
what that coarsening cost in physical units rather than asserting it was free."*

## 2. The four ideas, in dependency order

Learn them in this order. Each one only makes sense after the one before it.

**① The rings, and why they are not arbitrary.**
Radial ground beam spacing grows as `s_rad(r) = r²·Δφ / h_s`. With the HDL-64E's
Δφ = 0.427° and sensor height 1.73 m, that is centimetres at 5 m and 10.8 m at
50 m. So a uniform 5 cm grid at 50 m is 99.87% empty in a single frame. The ring
schedule (5/10/20/40 cm) tracks that curve. This is the single most defensible
thing in the project, because it is a derivation and not a design taste.

**② Integer lattice, and why split/merge is hard.**
Cell indices are `i_L = i_fine // k_L` with `k_L` an integer, never
`floor(x/0.20)` — float lattices drift and open gaps at ring boundaries. When
four fine cells merge into one coarse cell, their variances combine by the **law
of total variance**, `σ²_p = Σwᵢσᵢ² + Σwᵢ(μᵢ−μ_p)²`. The second term is the one
everyone drops, and dropping it makes a merged cell *most confident exactly where
it straddles a kerb*. Splitting inflates variance and sets a `derived` bit, which
is what makes `merge(split(c)) == c` exact so a cell oscillating across a ring
boundary does not inflate its variance forever.

**③ The bound is structural, not measured.**
745,000 logical cells × 12 bytes = **8.94 MB** of map. Every array, the
refinement pool, the transient layer and the tracked-object list are preallocated.
The refinement pool is what lets semantics buy back resolution locally without
breaking the bound: a fixed 512 blocks, and under load the correct behaviour is
refusal and eviction, not growth.

**④ Plan regret, and the honest state of it.**
The idea: don't measure the map against ground truth in centimetres, measure it
against the *decision* — plan a route on the compressed map, score that route on
the 5 cm reference map, and see if it costs more than the optimal route. This is
the most original thing in the project **and it does not currently work in your
favour.** Read `01-CRITICAL-FIXES.md` §2 before you say anything about it.

## 3. Six numbers to know cold

| # | Number | What it is | Do not confuse it with |
|---|---|---|---|
| 1 | **8.94 MB** | the *map* — 745,000 cells × 12 B | the 29.06 MB total |
| 2 | **29.06 MB** | *total preallocated*, map + every frame-path buffer | the 8.94 MB map |
| 3 | **21.5× / 286×** | cell-count ratios vs uniform 5 cm 2.5D (192 MB) and dense 5 cm 3D (2.56 GB) | any ratio taken against 29.06 |
| 4 | **ρ ≈ 1.39**, range 1.16–1.53, n = 11, ring 1 (between-cell spread); **1.25** [1.11–1.33] with within-cell variance, which also gives ring 0 a ρ of **1.17** [1.13–1.29] — regenerated 2026-09-17, `known-limitations.md` §2b. *Was 1.45 [1.26–1.59].* | the coarsening-justification ratio: what coarsening cost, divided by what the terrain's own roughness cost | RMSE, which spans 6.0× because roads differ (5.3× before 17 Sep) |
| 5 | **p50 89.18 ms / p99 100.43 ms** | end-to-end frame, 200 frames of seq 08 | the 80.78/97.72 in the handover, which is the back half only |
| 6 | **90.3% point acc / 65.2% mIoU** | FRNet on 200 frames of seq 08 | **69.8%**, which is an arithmetic error still on your slides |

**Numbers 1 and 2 are the single biggest trap in this presentation.** Both are
true. 8.94 MB is the map; 29.06 MB is the map plus working memory. The compression
ratios are computed on 8.94. The dashboard counter shows the total. If a judge
sees both on different slides and you have not pre-empted it, it reads as
cherry-picking. See `01-CRITICAL-FIXES.md` §5 for the exact sentence to say.

## 4. Who owns what, for routing questions on stage

| Area | Owner | Redirect to them for |
|---|---|---|
| `src/grid/`, `src/eval/` | Aakash | lattice, split/merge, fusion, plan regret, metrics |
| `src/gpu/` | Shrestha | allocation, scatter, timing, visibility cleanup, memory bound |
| `src/perception/`, `dashboard/` | JP | loader, poses, range image, FRNet, ground segmentation, Rerun |
| prior art, novelty | Srinivas | Droeschel, clipmaps, OctoMap, Wavemap |
| dynamics, segmentation lit | Hriday | ghost removal baselines, MOS, FRNet lineage |
| traversability, eval theory | Pratyushi | plan regret formulation, ρ, Fréchet |

## 5. The three sentences that will lose you the room

Each of these is currently written somewhere in your own repo or slides, and
each is not supported by your own evidence. Full detail in `01-CRITICAL-FIXES.md`.

1. *"...and proves the compression is free by showing it does not change the plan
   a robot would make."* — `README.md` line 8. Your own money plot says a uniform
   20 cm map at the same memory plans **better**.
2. *"69.8% mIoU"* — an arithmetic error. It is 65.2%. Someone will divide 977.7
   by 15 in front of you.
3. *"Sequence 08 was the completely unseen test sequence."* — `defense-rehearsal-playbook.md`
   Q6. Your own `known-limitations.md` §2b says 07 and 08 were chosen because
   they downloaded first.

**And one worse than all three, found in the deck rather than the repo:**
slide 3 claims *"CUDA kernels for project, fuse, split, merge"* and a
*"Python 3.11 · C++17 · CUDA"* stack. **There is no CUDA and no C++ anywhere in
this repository**, and slide 6 hands the judges the public repo link. See
`06-DECK-AND-SCRIPT-REVIEW.md` — this is the first thing to fix.

The good news is that all three have honest replacements that are still strong,
and a panel that watches you volunteer a limitation trusts everything else you
say. That is the entire strategy of this presentation.

## 6. What to read next

- **Tonight, before anything else:** `01-CRITICAL-FIXES.md`. It is ranked and
  timeboxed, and items 1–3 are non-negotiable.
- **Then:** `02-WHAT-TO-PRESENT.md` and rehearse the demo twice.
- **On the train:** `05-PANEL-DEFENSE.md`.
- **For the written report:** `03-TECHNICAL-STUDY.md` is drafted to be lifted
  section by section.
