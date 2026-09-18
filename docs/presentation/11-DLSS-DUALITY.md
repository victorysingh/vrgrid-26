# The DLSS duality — the analogy, with the numbers it can actually carry

*Shrestha, 2026-09-12. A rewrite of `VRGRID_DLSS_DUALITY_AND_PPT_PITCH.pdf`.
The analogy in that document is the best framing device the project has. Six of
its claims are contradicted by our own measurements, and one of them is the
`01-CRITICAL-FIXES.md` item 2 overclaim arriving in new clothing. This file keeps
the framing and replaces the numbers with sourced ones.*

> ⚠️ Read `01-CRITICAL-FIXES.md` and `OPEN-ITEMS.md` before quoting anything here.
> Every figure below carries its source; if a figure has no source, it is not in
> this file on purpose.

---

## 1. Why the analogy is worth keeping

DLSS did not make rendering lossless. It made rendering **evaluated against the
consumer instead of the signal** — perceptual loss against the human visual
system, not per-pixel MSE against a reference frame. That reframing is what made
a 1080p internal render acceptable at 4K output, and it is the genuinely
transferable idea.

VRgrid's contribution is the same move in robotics: **stop scoring a map in
centimetres and start scoring it against the decision it produces.** Plan regret
is to a planner what perceptual loss is to an eye.

That is the whole analogy, and it survives contact with our evidence. What does
not survive is the claim that we have *won* on that metric.

## 2. The six pillars, corrected

| Pillar | NVIDIA DLSS | VRgrid — what is actually true |
|---|---|---|
| **Space domain** | 2D screen lattice, 1920×1080 → 3840×2160 | 2.5D world lattice, concentric rings 5/10/20/40 cm over 200 m × 200 m. **Correct as pitched.** |
| **Physics bottleneck** | Shading FLOPs cannot fill 8.3 M pixels at 60 FPS | Radial ground spacing `s_rad = r²Δφ/h_s`: **0.43 m at 10 m, 10.8 m at 50 m**, so a uniform 5 cm grid is **99.87% empty** in one frame. `sih-math.md` §3.2. **Correct as pitched.** |
| **Temporal history** | Sub-pixel jitter accumulated across frames | Ring-sweep fill: far rings populate over successive scans as the vehicle advances. **Correct as pitched.** |
| **Motion vectors** | 2D screen-space vectors + disocclusion mask | Ego-velocity anisotropic scaling `a_f(v)` + O(1) range-image visibility. **Correct as pitched.** The caveat that `a_f(v)` put the ring boundary at a non-lattice position (open item **D2 / R3**) is closed: ring membership is decided per lattice block, so no cell straddles a boundary (2026-09-17). |
| **Dynamic actors** | Colour clamping, history rejection | **Motion flags come from the SemanticKITTI `.label` files, not from FRNet.** See §3.1. |
| **Loss metric** | Perceptual loss (SSIM / LPIPS) vs the human eye | Plan regret R(S) vs the planner. **The metric is built. It does not currently favour us.** See §3.2. |
| **Numerics** | Tensor cores, FP16/INT8 | int32 fixed-point accumulation. Bit-exact **on the kernel path**; end-to-end replay currently fails on open item **D1**. See §3.4. |

## 3. The six claims that had to go

### 3.1 "FRNet semantic masking removes ghosts"

**FRNet is not in the mapping pipeline, by design.** Semantic classes and the
`moving-*` motion flags come straight from the SemanticKITTI `.label` files
(`src/perception/semantics.py`). The port exists and scores **90.3% point
accuracy / 65.2% mIoU** on 200 held-out frames of seq 08 against the paper's
73.3%, and it is reported *alongside* the map rather than wired into it, so the
mapping contribution is evaluated independently of segmentation error.

Say it as the methodological choice it is. "We ported a model to 90.3% and then
chose not to use it, so that our mapping result is not contaminated by our
segmentation result" is a stronger sentence than the pitch's version, and it is
true.

### 3.2 "Plan regret R(S) < 0.014 m", "zero decision loss", "mathematically free"

**This is the fatal one.** No 0.014 m figure exists anywhere in the repository.
The measured money plot — seq 08, 20 frames, 64-query mean, matched extent,
`known-limitations.md` §2 — is:

| schedule | MB | R(S) |
|---|---:|---:|
| uniform 20 cm | 30.50 | **0.251** |
| **5/10/20/40 (ours)** | **29.06** | **0.488** |
| uniform 40 cm | 18.50 | **0.402** |

**A uniform 20 cm map at comparable memory plans better than ours, and a uniform
40 cm map is cheaper and better.** Three real defects in the metric were found
and fixed on 2 September and the ordering survived all three.

The cause is understood and it is the query, not the thesis: `PLAN_LANE_CELLS` is
a single longitudinal lane down the centre of the window with no hazards on it,
so it structurally cannot reward a map for being sharp where the vehicle is
looking. A fine cell pools fewer returns, drops below `n_min`, is flagged
low-confidence, and the planner steers around a phantom made of our own sparsity.

**What to say instead:** *"We built the decision-theoretic metric, which is the
contribution. On the only query implemented so far it does not favour us, we know
exactly why, and designing a query that can discriminate is the first open item.
What we can show is that our two frozen schedules produce identical plans across
a 5.4 MB difference."*

A panel that hears that trusts the rest. A panel that hears "mathematically
guaranteed lossless" and then opens `known-limitations.md` does not.

### 3.3 "<15 ms total pipeline latency on Jetson / AWS G4dn"

**No such measurement exists, and neither platform has ever been run.** The
measured end-to-end frame, 200 frames of real seq 08, is **p50 89.18 ms / p99
100.43 ms** on one host and **108.65 / 127.23 ms** on another — open item **D8**,
which is unresolved precisely because two honest numbers disagree. The pitch
figure is off by roughly six to eight times against hardware nobody has touched.

The p99 misses the 100 ms budget by 0.43 ms on the better host. Volunteer that.

### 3.4 "CuPy/CUDA int32 modulo atomics" as a shipped invariant

The determinism *property* is real and now measured on device: int32
`scatter_add`, 2 M returns into 4,096 cells, **30/30 runs bit-identical**, against
a float32 control that differed on **29 of 29**
(`scripts/bench_cupy_seam.py --determinism`, `docs/gpu-lane/06-DAY3-CUPY-FINDINGS.md`).

But **nothing is ported yet.** The mapping pipeline is numpy on CPU; the cupy
seam was taken as a measurement on Day 3, not a merge. Presenting CuPy atomics as
a current hardware invariant is the same class of error as the deck's old "CUDA
kernels for project, fuse, split, merge", which we removed from the README this
week for being checkable and false.

Say: *"Integer accumulation makes the port deterministic for free, and we have
measured that on the device with a float control. The port itself is in
progress."*

### 3.5 "Our map never contains phantom trails"

Measured ghost removal is **13.5% of the trail removed, 4.96 M cells cleared, and
429,012 cells spared by the current-return guard**. The "0 / 4,071 frames" figure
in the README is about *inert* ghost trails after the elevation fix, which is a
narrower claim than "never contains phantom trails".

### 3.6 The azimuthal-spacing numbers are somebody else's numbers

The pitch's §6 states azimuthal spacing of **3.5 / 8.7 / 17.5 / 34.9 cm** at rings
0–3.

Those are not azimuthal spacings. **8.7 cm and 17.5 cm are the height standard
deviations σ_z at 50 m and 100 m**, from `sih-math.md:256` and `kernels.py:94`
— *"With σ_φ = 0.1°: σ_z = 8.7 cm at 50 m, 17.5 cm at 100 m."* The four values
are σ_z at 20, 50, 100 and 200 m, relabelled as a different quantity at different
ranges.

The correct physical argument is already in the deck and is stronger: radial
ground spacing `s_rad = r²Δφ/h_s` gives **0.43 m at 10 m and 10.8 m at 50 m**.
Use that. The "cell sizes match Nyquist within a factor of 1.15" claim needs
regenerating from `scripts/sampling_table.py` before anyone says it aloud.

## 4. The pitch, rewritten

### Mode A — the 30-second version

> "Everyone knows NVIDIA DLSS: it renders at 1080p and reconstructs 4K, because
> it stopped scoring frames per-pixel and started scoring them against the human
> eye. VRgrid is that idea for robot perception. LiDAR beams diverge
> quadratically — at 50 metres they land 10.8 metres apart, so a uniform 5 cm
> grid is 99.87% empty and stores interpolation, not measurement. We put
> resolution where the sensor actually samples, under a memory bound fixed before
> the first scan: 8.94 megabytes of map, 21.5× less than a uniform 5 cm 2.5D
> grid. And we score the result the way DLSS does — against the consumer. Ours is
> the planner, not the eye."

Note what it does not say. No "lossless", no "guaranteed", no regret figure.

### Mode B — the technical version

> "The mechanics map one-to-one. Motion compensation: DLSS uses screen-space
> motion vectors, we use ego-velocity anisotropic scaling and spherical
> range-image projection. Temporal accumulation: DLSS accumulates sub-pixel
> samples across frames, we ring-sweep fill far rings across successive scans.
> Numerics: DLSS leans on tensor cores, we lean on the fact that integer addition
> is associative — heights accumulate as int32 fixed-point in 1 cm units, so a
> GPU scatter is bit-identical regardless of how the scheduler interleaves
> blocks. We measured that: 30 of 30 runs identical on int32, 29 of 29 differing
> on a float32 control, same indices and same card. Most projects trade
> reproducibility for device speed; we do not, because of a decision made on
> Day 0. And the evaluation: DLSS discards MSE for perceptual loss, we discard
> elevation RMSE for plan regret. That metric is the contribution, and it is also
> where our result is weakest — I can take you through exactly why."

## 5. The judge questions, answered honestly

**"Is VRgrid hallucinating terrain with a neural net, like DLSS hallucinates
pixels?"** — No, and the distinction is real. Our elevation fusion is 100%
deterministic and geometric, governed by the law of total variance and
fixed-point arithmetic. No network touches the map at all: even the semantic
labels come from ground-truth files, not inference.

**"Why 8.94 MB — is that arbitrary?"** — It is 745,000 logical cells × 12 bytes,
preallocated at startup, and the ring boundaries come from beam geometry rather
than taste. The derivation is `sih-math.md` §3.2. Be careful to separate the
**8.94 MB map** from the **29.06 MB total committed** — both are true, the ratios
are computed on the map, and volunteering the larger one first is the whole play.

**"Why does fixed-point matter on a GPU with huge float throughput?"** — Because
IEEE-754 addition is not associative, so a float scatter gives a different map
every run and you cannot bisect a bug whose location moves. Integer `atomicAdd`
is exact and order-independent. We measured both on the same card.

**"Does your compression change the plan?"** — *That is the right question, and
it is the one we built the project to answer. Today the honest answer is that our
only planning query cannot tell, and on that query a uniform 20 cm map scores
better. Here is why the query is the limit, and here is what we would build to
fix it.*

---

## 6. Provenance

| Figure | Source |
|---|---|
| 8.94 MB, 745,000 cells, 12 B | `README.md`, `memory_table.py` |
| 29.06 MB committed, 10.3 / 32.17 | `allocate()`, `sih-math.md`, `gpu-lane/00` §3 |
| 21.5× / 286× | cell-count ratios on the map, `README.md` |
| 0.43 m at 10 m, 10.8 m at 50 m, 99.87% | `sih-math.md` §3.2 |
| σ_z 8.7 cm at 50 m / 17.5 cm at 100 m | `sih-math.md:256`, `kernels.py:94` |
| R(S) 0.251 / 0.488 / 0.402 | `known-limitations.md` §2 |
| 89.18 / 100.43 ms, 108.65 / 127.23 ms | research log; open item D8 |
| Ghosts 13.5%, 429,012 spared | `gpu-lane/00-WHERE-WE-ARE.md` §2 |
| FRNet 90.3% / 65.2% | `scripts/frnet_eval.py`, 200 frames seq 08 |
| int32 30/30, float32 29/29 | `scripts/bench_cupy_seam.py --determinism` |
| Replay differs by 1,245 / 1,479,013 | `reports/ring1-reproduction-investigation.md` |
