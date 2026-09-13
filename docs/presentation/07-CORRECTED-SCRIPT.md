# VRgrid — Presentation Script (corrected)

*SIH26053 — Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic
Environment Perception · Team: Chronicles.exe*

**Target: 6–7 minutes, 6 slides.**

*Your teammate's structure, pacing and tone guidance are kept — they were good.
Changes are marked ⚑ where a claim moved. Rationale for every change is in
`06-DECK-AND-SCRIPT-REVIEW.md`.*

*Practice with a timer. Trim slide 6 before you rush slide 4.*

---

# Slide 1 — Title *(10–15 s)*

"Good [morning/afternoon] judges. We're team Chronicles.exe, presenting our
solution to Problem Statement SIH26053: Adaptive Variable-Resolution 2.5D LiDAR
Mapping for Dynamic Environment Perception, under the Smart Vehicles theme."

*(Pause. Click.)*

---

# Slide 2 — Proposed Solution *(60–75 s)*

**Open with the pain point:**

"Here's the problem with how mapping is normally done. LiDAR doesn't sample the
world evenly — it samples it in rings that fan out from the sensor. At 50 metres,
two consecutive laser rings land almost 11 metres apart on the ground. So if you
build a uniform 5-centimetre grid, which is what most systems do, 99.87% of that
grid's cells out there cannot receive a single laser return in one frame. You're
allocating memory for detail the sensor physically cannot deliver."

**Second problem:**

"And moving vehicles leave stale elevation trails behind them in the accumulated
map — ghosts of where a car used to be, that a planner then routes around."

**The solution:**

⚑ "VRgrid fixes both. Instead of one uniform grid we use nested rings of
resolution — 5, 10, 20 and 40 centimetre cells going out to 10, 25, 50 and 100
metres, all sitting on one global 5-centimetre integer lattice. The map is a
fixed budget: 745,000 cells at 12 bytes each, 8.94 megabytes, allocated once at
startup. Our total preallocated footprint including every working buffer is
29.06 megabytes — we'll show you both numbers, and the compression ratios are
computed on the map."

> ⚑ *Added the 29.06 clause. It will appear on your dashboard and in the regret
> plot; volunteering it costs four seconds and removes a cherry-picking read.*

**Honest coarsening — say this confidently:**

"When we merge cells to save memory, we don't fake certainty. We use the law of
total variance, so merging and splitting are mathematically reversible — merge of
split equals the original — and a merged kerb cell correctly reports a sigma of
6.3 centimetres instead of claiming 1 centimetre of precision it doesn't have.
That matters because the published adaptive grid closest to ours uses
inverse-variance averaging, which makes a merged cell *most* confident exactly
where it straddles a kerb."

**Ghosts and determinism:**

"A visibility check against the range image clears the transient layer, so moving
vehicles don't leave stale trails. And every query at any resolution returns the
same struct, so it's a drop-in replacement for existing map consumers."

**Novelty:**

"What's novel isn't foveated grids by themselves — those exist and we cite them.
It's that our resolution schedule is driven by range and semantics together, that
we score it in planner regret rather than reconstruction error, and that it's
fully deterministic: the same input twice gives a bit-identical map hash."

*(Click.)*

---

# Slide 3 — Technical Approach *(70–90 s)*

"Let's walk the pipeline end to end."

**1. Data:**

⚑ "We're working with a real prototype on real data — SemanticKITTI, all 22
sequences on disk, 43,552 scans, 84.8 gigabytes. Label classes and motion flags
from the raw label files, KITTI ground-truth poses, with SLAM poses on sequences
00 and 08 for a reason I'll come to on the next slide."

**2. Preprocessing:**

"Each frame goes through a transform, a 64-by-512 range-image projection, deskew,
and Patchwork++ for ground segmentation."

**3. Resolution policy:**

"Resolution at each point comes from a single clamped function combining range,
object class and direction of travel — so every cell's resolution is computed the
same principled way, not hand-tuned per scene."

**4. Adaptive 2.5D grid:**

"Variance-driven split and merge, Kalman height fusion, and a Boyer–Moore-style
class byte — five bits of candidate, three of counter — to keep semantic tagging
to one byte per cell."

**5. ⚑ Deterministic parallel back end** *(this section is rewritten — see below)*

⚑ "The back end is written the way GPU code is written, and I want to be precise
about what that means. It's a numpy reference implementation on CPU, with a
device seam in the allocator so the arrays can move to cupy without touching the
kernels. Every number we quote is measured on that CPU path.

The design decisions are the parallel ones. Structure-of-arrays, so a kernel
reading one field touches contiguous memory. Integer fixed-point accumulation
rather than float atomics — because IEEE addition isn't associative and atomic
adds complete out of order, so a float map changes between runs and you can't
bisect a bug whose location moves. Zero allocation in this back end's frame loop,
which took us from 8.15 to 1.31 megabytes per frame — that is the mapping half;
the perception front end is not instrumented yet and does allocate. And a toroidal ego-motion shift that
moves the origin instead of the data: 0.04 milliseconds against 15.2 for the
obvious version.

End to end on 200 frames of sequence 08, the frame is 89 milliseconds at the
median against a 100 millisecond budget. The p99 is 100.4 — so one frame in a
hundred misses 10 Hz by four tenths of a millisecond. That's a real miss and
we're not rounding it away. The largest stage is visibility cleanup at 26 of
those 89 milliseconds, and it's the most parallel thing in the system, so we know
where the headroom is."

> ⚑ **This replaces "All of this runs as CUDA kernels."** There is no CUDA and no
> C++ in the repository, and slide 6 hands the judges the public repo link. The
> version above is true, and it demonstrates you understand *why* parallel
> correctness is hard — which is the more impressive thing to have said.
>
> ⚑ **The "2.45 ms rebuild" line is removed.** That was the conservative
> pyramid's rebuild time; the pyramid is a stretch item, off by default. The
> frame latency above is the number a judge actually wants.

**6. Evaluation:**

"We evaluate with planner regret and Fréchet distance, a coarsening ratio, per-ring
RMSE, and standard dynamic-removal metrics."

**Tie together:**

"The design point is that every stage is resolution-agnostic — a planner queries
by world coordinate, never by cell index, so it doesn't know or care that the
grid underneath is non-uniform."

**Stack — summarise, don't read:**

⚑ "Python 3.11, NumPy structure-of-arrays with int32 fixed-point atomics,
PyTorch and CUDA for the segmentation model, SemanticKITTI and Patchwork++ for
data, Rerun for visualisation, GitHub Actions and pytest enforcing CI gates that
block every merge. Our determinism and partition properties aren't just claimed —
they're theorem tests in CI, and they are proofs, not tuning targets."

*(Click.)*

---

# Slide 4 — Feasibility and Viability *(70–90 s — slow down here)*

"This slide is where we address feasibility head-on, because credibility matters
as much as the idea."

**Walk 2–3 risks, don't read all four:**

"First risk: odometry drift corrupting an accumulated map. We use ground-truth
poses everywhere except sequences 00 and 08, where the official ground truth puts
the same patch of road 16.6 centimetres apart from one frame to the next —
consistently, so a systematic offset rather than drift. There we fall back to
SemanticKITTI's SLAM poses, which give 1.04. We checked this across all eleven
labelled sequences and pinned the exception list in a test so it can't quietly
widen.

⚑ And the subtle part, because a judge may ask why only those two: per-frame
agreement turned out to be a *weak* predictor. Sequence 00 disagrees by only
2.27 centimetres per frame but accumulates almost 14 centimetres of bias by the
outer ring, while sequence 03 at a comparable per-frame number accumulates under
one. We overrode only the sequences with a measured win.

Second: far rings get very few beams, so a naive single-frame metric makes them
look worse than they are. We report far-ring metrics against
frames-since-first-observation, and the schedule comes from measured sampling
density rather than a guess.

Third: could a naive merge report lower variance than its children justify? We
prevent that by construction — law of total variance, with round-trip idempotence
as a blocking unit test.

And yes, a reviewer might say foveated grids already exist. They do, and we cite
them on our last slide. We claim the composition and the evaluation, nothing
more."

**Deployment facts:**

⚑ "This is software-only, built on a public dataset — no rig, no field trials, no
data collection. Nothing is trained for the shipped pipeline; classes and motion
flags come straight from the dataset's label files, and that's a deliberate
choice I'll come back to. The repository is public and green: **206 commits**, CI
passing, live Rerun demo. It's designed for Jetson-class deployment — 8.94
megabytes of map with no dynamic allocation in the frame loop and no need for a
discrete card — though we haven't yet measured on target hardware."

> ⚑ *"50 commits" was wrong by 4× — it's 206. And the Jetson claim was
> untested; the added half-sentence removes an unverifiable statement at
> almost no cost.*

**Memory table — land it:**

"Here's what that buys. A dense 5-centimetre 3D voxel grid at the same near-field
accuracy is about 2.56 gigabytes. A uniform 5-centimetre 2.5D grid is about 192
megabytes. VRgrid is 8.94 — a 286-times reduction against the dense baseline and
21.5 against the uniform one, at the same extent and vertical range.

And before you ask for it: a sparse or hashed 3D structure would be about 130 to
240 megabytes, which is 15 to 27 times rather than 286. That's the less
flattering comparison and it's on our slide because it's the one we'd want to
see."

**Viability:**

"Defence UGVs on unmapped ground, warehouse and yard AMRs, last-mile delivery,
ADAS perception stacks. No extra sensing hardware — it reuses the LiDAR that's
already there — and an entirely open-source stack on public datasets, so no
licensing cost to adopt."

*(Click.)*

---

# Slide 5 — Impact and Benefits *(60–75 s)*

⚑ "All of this was measured, not projected — and measured across **all eleven
labelled SemanticKITTI sequences**, not just the two we developed on."

**Results:**

⚑ "Map footprint fixed at 8.94 megabytes throughout. Our coarsening ratio — how
much the compression cost, divided by the terrain's own natural sub-cell spread —
comes out at a **median of about 1.45, range 1.26 to 1.59, across eleven
sequences** at ring 1. That's the number that matters, because raw RMSE across
those same eleven varies by a factor of 5.3 — it tracks how rough each road
happens to be. Rho divides that out and leaves what the *coarsening* cost, and it
varies by only 1.26. A rho near one means the coarsening cost about what the
terrain's own variability already did.

⚑ And we'll say this before you check: sequences 07 and 08, the two we developed
against, sit at the *good* end of that range. That's exactly why we quote the
distribution instead of our best sequence.

⚑ On ghosts: before our elevation fix, 2,304 of sequence 08's 4,071 frames did
nothing at all — the cleanup silently stopped working once the vehicle climbed
past about 11 metres. After the fix, zero inert frames, clearing 15 to 20
thousand cells on every frame across a 39-metre climb. Measured removal is 13.5%
of the trail, 4.96 million cells. And 429,000 cells were *protected* by the guard
that stops the cleanup eating fences, poles and sign posts — we quote that one
too, because it's the evidence the cleanup is conservative rather than
aggressive.

And two identical runs produce a bit-identical map hash."

> ⚑ **The original script said "ghost trails: zero, across all 4,071 frames."**
> The measurement is *"0 of 4,071 frames inert"* — the cleanup now fires every
> frame, where before it did nothing on 57% of them. It does not mean zero
> ghosts remain; measured removal is 13.5%. The before/after above is both true
> and a better story.

**Why plan regret:**

"Most adaptive-mapping papers report memory, latency and height RMSE — none of
them answer the question that actually matters: did compressing the map change
what the robot would decide to do? So we plan a path on the coarse map and on the
5-centimetre reference, then score *both* paths against the reference. The gap is
the regret. Scoring both on the reference is what makes it honest — a blurred
kerb the coarse map can't see produces infinite regret rather than false safety."

**⚑ Be honest about the result — this is the rewritten section:**

⚑ "And I want to be straight about what it says, because the answer isn't the one
we wanted.

We built the metric, and we found and fixed three real defects in it — a
confidence penalty that was growing with resolution, so we were penalising our
own fine rings; the two sides being evaluated on different lattices, which
invented 148 phantom walls; and a class penalty charged on one side only.

After all three, on sequence 08 at matched extent: our schedule reads 0.488, and
a uniform 20-centimetre map at essentially the same memory reads 0.251. It plans
better than we do.

Here's why, and it's diagnosed rather than hand-waved. We have exactly one
planning query, and it's a longitudinal lane down the middle of the window,
unchanged since day one. A lane query rewards a map that's uniformly adequate
along one line. It structurally cannot reward a map for being sharp where the
vehicle is *looking*, which is the entire thesis. So that isn't evidence the
thesis is wrong — it's evidence this query can't test it, and designing one that
can is the first item on our next list.

What does hold: our two schedules produce identical plans despite 5.4 megabytes
between them, and matched to the same ground we cost 29 megabytes against a
uniform 10-centimetre map's 78.5."

> ⚑ **The original said R(S) = 0.000 on a synthetic scene and "the download is
> the blocker."** The download completed — all 22 sequences, 84.8 GB — and the
> real-data run produced the table above. The slide also contradicted its own
> speaker note (0.000 in the body, 0.207 in the notes). If a judge reads
> "download is the blocker" and asks what the data showed, you need this answer
> ready.

**Impact:**

"Practically: 21.5 times less memory for the same near-field detail, so it runs
on lower-spec compute, on an open-source stack with no licensing or retraining
cost. Kerbs and ramps survive coarsening intact, so the paths are safer. And less
memory traffic means less energy per frame, using sensors already on the vehicle."

*(Click.)*

---

# Slide 6 — Research and References *(30–40 s)*

"Last slide, and we want to be upfront about where this sits in the literature,
because a strong submission should own that rather than hide it.

Foveated grids, elevation maps and dynamic-point removal are all published prior
art — Triebel, Pfaff and Burgard's multi-level surface maps; Droeschel, Stückler
and Behnke's nested ego-centric grids; Losasso and Hoppe's geometry clipmaps;
OctoMap and wavemap as volumetric baselines; Fankhauser's elevation mapping with
uncertainty; and Wodtko's adaptive patched grid mapping. Our repository documents
exactly what we build on.

What we contribute on top: one schedule driven by range and semantics together;
a hard memory envelope preallocated at startup with nothing allocated per frame;
mathematically correct coarsening through the law of total variance, so we never
report false confidence over a kerb; evaluation in units the planner actually
feels; and reproducibility by default — determinism and partition tests block
every merge."

**Close:**

⚑ "Everything — the maths, the theorems, the metric specs, and our known
limitations, root-caused and written down — is in the public repository at
github.com/Stxtics03/vrgrid. Thank you, and we're happy to take questions."

> ⚑ *Fix on the slide itself: slide 6 currently says "official KITTI GT poses,"
> contradicting slides 3 and 4. And it lists the dataset as sequences 00/07/08
> when your headline accuracy result now spans eleven.*

---

# ⚑ If asked: "Where is the machine learning?"

*Not in the deck, and it should be. Under a Smart Vehicles software theme this is
a likely question, and right now "nothing is trained" is your only answer.*

> "We built one and then chose not to put it in the pipeline, and the choice is
> the interesting part.
>
> We ported FRNet, a frustum-range segmentation network, from about 15 percent
> point accuracy to **90.3 percent, and 65.2 mIoU** on 200 held-out frames of
> sequence 08, against the paper's 73.3. Three causes: the wrong activation
> function at seven sites in the backbone, a config that was overriding the
> network's trained projection with the sensor's physical field of view, and a
> densification transform we had to reproduce verbatim — including an off-by-one
> we kept deliberately, because the published number was measured with it.
>
> We also found its forward pass was 10.5 seconds a frame with about 90 percent
> of that in a Python loop, and replaced seven reduction loops with
> `scatter_reduce`: 1408 times faster on max and 541 on mean at real shapes on
> CUDA, which took a fine-tune from three and a quarter hours to two minutes.
> Then we fine-tuned it three ways and rejected all three on measurement — the
> checkpoint was already trained on this data, so there was no domain gap to
> close.
>
> And we kept it out of the mapping pipeline on purpose. Labels come from ground
> truth, which means our mapping result is evaluated independently of
> segmentation quality — a bad coarsening ratio can't be blamed on a
> mis-segmented kerb, and a good one can't be credited to a strong segmenter."

⚠️ **65.2%, never 69.8%.** The 69.8 figure in older project docs is an arithmetic
error — the 15 per-class IoUs sum to 977.7 and someone divided by 14, dropping
`other-ground`.

---

# Delivery Notes

**Pacing.** Slides 3 and 4 carry the most content. Running long: trim slide 3's
stack list to one sentence and shorten slide 6 to the closing line. Do not trim
slide 5's honesty section — it is now the most valuable 40 seconds in the deck.

**Tone.** This deck is unusually rigorous for a hackathon pitch. Lean into it.
Deliver the limitation lines — slide 4's prior-art admission, slide 5's regret
result, slide 3's "there is no CUDA here" — **calmly and directly, not
apologetically.** Every one of them buys credibility for the numbers around it,
and you have enough real numbers to spend some.

⚑ **Whoever delivers slide 5's regret paragraph should be the most senior person
on the team.** Disclosure from a junior member reads as a slip; from the lead it
reads as rigour.

**⚑ Numbers to have memorised cold:**

| | |
|---|---|
| 99.87% | uniform-grid waste at 50 m |
| 10.8 m | beam spacing at 50 m |
| 8.94 MB / 29.06 MB | map / total preallocated |
| 286× / 21.5× | vs dense 3D / uniform 2.5D |
| ρ ≈ 1.45 (1.26–1.59, n = 11) | coarsening ratio, ring 1 |
| 89.18 / 100.43 ms | frame p50 / p99, 100 ms budget |
| 13.5% · 429,012 | trail removed · cells spared by the guard |
| 8.3 m | pothole detection limit, derived |
| 90.3% / 65.2% | FRNet point accuracy / mIoU |
| 206 | commits |

**Before presenting:** fill Team ID and the demo link on slide 1, and run
`./scripts/demo.sh check` and `bake` on the presenting machine.
