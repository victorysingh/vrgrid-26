# vrgrid — a technical study

**SIH26053 · Adaptive Variable-Resolution 2.5D LiDAR Mapping**
*Drafted 2026-09-05 against `main` @ `8882ec3`. Written to be lifted section by
section into the final report. Every number is sourced to the script that
produced it; where a number is disputed or superseded, that is stated inline
rather than resolved silently.*

---

## 1. The problem, stated physically

A ground vehicle carrying a Velodyne HDL-64E needs a local map it can plan on.
The obvious representation is a uniform occupancy grid. At 5 cm resolution over a
200 × 200 m footprint that is 16 million cells for 2.5D (192 MB at 12 B/cell) or
2.56 × 10⁹ voxels for dense 3D at 8 m of vertical extent (2.56 GB). Both are
unaffordable on an embedded platform, and the usual response is to coarsen
uniformly and accept the accuracy loss everywhere.

**The observation this project is built on is that the loss is not uniform,
because the sampling is not uniform.** For a sensor at height `h_s` with vertical
beam separation `Δφ`, the radial spacing between consecutive beams' ground
intersections grows quadratically with range:

```
s_rad(r) = r² · Δφ / h_s
```

For the HDL-64E (`Δφ` = 0.427°, `h_s` = 1.73 m) this is a few centimetres inside
10 m and **10.8 m at 50 m**. A uniform 5 cm grid at 50 m is therefore **99.87%
empty in a single frame**. Its cells do not hold measurements; they hold whatever
interpolation the fusion rule invents.

Two consequences follow, and they are the project's foundation:

1. **Coarsening the far field is not a compromise, it is a correction.** Storing
   the far field at 40 cm discards nothing the sensor measured.
2. **There is a hard limit on negative-obstacle detection.** A pothole of width
   `W` is physically invisible beyond `r_max = √(W·h_s / Δφ)`, which for a 30 cm
   pothole is **8.3 m** on a single scan. No grid resolution changes this. The
   correct behaviour beyond that range is to mark cells `unknown`, never `free`.

The second point is the difference between a mapping system and a safety system,
and it is worth making explicitly in the report: a uniform 5 cm map that reports
`free` at 50 m is not more accurate than a 40 cm map that reports `unknown`. It
is confidently wrong.

*Source: `docs/memo-r1-sensor-physics-and-ring-justification.md`; research log,
Srinivas, 2026-08-29.*

---

## 2. Data

### 2.1 The dataset

**SemanticKITTI** [1], built on the **KITTI odometry benchmark** [2].
Complete on disk: **43,552 scans, 84.8 GB, all 22 sequences**; point-wise labels
on sequences 00–10, poses on all. `python scripts/data_status.py` re-verifies
and exits 0 when whole. The dataset is not in the repository; the loader is
pointed at it by `VRGRID_DATA_ROOT`, and data-dependent tests **skip** rather
than fail when it is absent.

Layout consumed:

```
<root>/poses/<seq>.txt                    KITTI GT poses (Camera-0 → World_cam)
<root>/sequences/<seq>/velodyne/*.bin     sensor-frame points (N, 4)
<root>/sequences/<seq>/labels/*.label     raw SemanticKITTI ids
<root>/sequences/<seq>/calib.txt          Tr, sensor → camera
```

### 2.2 Labels

Both the 19-class semantic label and the motion flag come from the raw `.label`
files. Dynamic classes are the four `moving-*` raw ids — 252 (car), 253
(bicyclist), 254 (person), 255 (motorcyclist) — and these four are the sole
triggers for ghost removal.

**No network runs for labels.** This is deliberate and should be presented as a
methodological choice: it isolates the mapping contribution from segmentation
error, so a poor ρ cannot be blamed on a mis-segmented kerb and a good one cannot
be credited to a strong segmenter. Section 7 covers the network that exists
alongside this path.

### 2.3 Poses — the finding that cost the most time

KITTI ships **two** pose solutions and they are not interchangeable:

- **Official GT poses** — a GPS/IMU solution optimised for *trajectory*
  evaluation.
- **SemanticKITTI SLAM poses** — computed so that scans *register into a
  consistent map*.

Sequence 08's GT poses put the same patch of road **16.6 cm apart** between
consecutive frames, consistently (16.1–17.6 cm across every pair, so a systematic
offset rather than drift). Accumulated over N frames this gave the reference map
itself a **64.5 cm median standard deviation inside a 10 cm footprint** and put
08's per-ring RMSE at 162 cm. On SLAM poses the same measure is 1.04 cm.

Median absolute ground-height disagreement between consecutive frames, in 20 cm
cells both frames saw:

| seq | GT | SLAM | | seq | GT | SLAM |
|---|---|---|---|---|---|---|
| 00 | 2.27 | **1.05** | | 06 | 1.32 | 1.23 |
| 01 | 1.43 | 1.38 | | 07 | **0.47** | 0.64 |
| 02 | 1.20 | 1.20 | | 08 | **16.53** | **1.04** |
| 03 | 1.97 | 1.24 | | 09 | 1.21 | 1.26 |
| 04 | 1.25 | 1.11 | | 10 | 1.19 | 1.20 |
| 05 | 1.02 | 1.00 | | | | |

**The subtle part, and the one worth reporting.** Per-frame agreement turned out
to be a *weak predictor* of what actually matters, which is the bias that
**accumulates**. Sequence 00 disagrees by only 2.27 cm per frame yet accumulates
**−13.95 cm** of height bias by ring 3, while sequence 03 at a comparable
1.97 cm/frame accumulates −0.80 cm. Switching 00 to SLAM takes ring 2's bias from
−9.85 to −0.44 cm and ring 3's RMSE from 26.03 to 13.57.

Sequence 06, the next worst accumulator, was tested the same way and is a wash,
so it stays on GT. **Only sequences with a measured win are overridden.** The
final list is `{"00": "slam", "08": "slam"}`, pinned by
`test_only_08_needs_the_slam_poses` so it cannot quietly widen.

Ruled out along the way, each by measurement: the height datum, band saturation,
the ground mask, frame alignment (08 is 4,071/4,071/4,071), the calibration (07
and 08 have byte-identical `Tr`), and the transform composition itself.

*This is a good report section. It is a real dataset artifact, correctly
diagnosed, with the negative controls stated.*

### 2.4 A synthetic sequence, and why it exists

`src/eval/synthetic.py` writes a scene in the **real KITTI layout** — poses,
`velodyne/*.bin`, `labels/*.label`, `calib.txt` — sampled through a real beam
model. It is not a shortcut around the data; it is a scene whose surface is known
analytically, so heights can be *asserted* rather than eyeballed. It is what
exercises `reference_map.build()` end to end without the 84.8 GB download.

Getting it right required five conventions, three of which fail silently rather
than crash: a `.bin` holds **sensor**-frame points (vehicle origin is 1.73 m
below the laser); a `.label` holds **raw** ids, not 19-class learning ids (which
collide inside the valid range — 10 is `car`, 11 is `bicycle`); and a
`poses.txt` row is **Camera-0 → World_cam**, not vehicle → world.

**Caveat on synthetic results.** On sequence 08 the binding confidence channel is
*geometry* for rings 0–2, where the synthetic scene binds on *label* and
*evidence*. Real terrain sits near the slope and step thresholds and the analytic
scene does not. That is the clearest single argument in the project for not
reporting synthetic numbers as results.

---

## 3. Architecture

```
  SemanticKITTI                  src/perception/          [JP]
  .bin / .label / poses  ──►  loader → transforms → range_image
                                    → semantics → motion → ground (Patchwork++)
                                              │
                                              ▼
                              src/gpu/                    [Shrestha]
                        bin_points → scatter → shift → visibility
                                              │
                                              ▼
                              src/grid/                   [Aakash]
                   fusion → split/merge → gate/pool → traversability
                                    → features → confidence → query
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
                  src/eval/        [Aakash]              dashboard/       [JP]
          reference_map → metrics → plan_regret               Rerun app
```

`include/vrgrid/` holds frozen interfaces: the 12-byte cell struct and five
signatures. Changing it requires all three developers in one room, by design.

**The core has no ROS dependency.** The ROS adapter is an optional module under
`adapters/`, on the reasoning that if ROS breaks two days before submission the
framework must still run. It did not break; the discipline was still correct.

### 3.1 The cell

12 bytes, frozen on Day 0. Structure-of-arrays, never array-of-structs, so a
kernel reading only `ground_height` touches contiguous memory.

Heights are **int16 in 1 cm units**; ranges and gradients are float metres, and
variables carry `_cm` or `_m` suffixes so the two cannot mix silently. The class
byte is a **5-bit candidate and a 3-bit counter** — a Boyer–Moore majority
counter, re-split from 4|4 when it turned out 4 bits could not hold 20 learning
ids, which silently broke the semantic gate for `pole` (18) and `traffic-sign`
(19), the two classes semantic refinement exists for.

**An honest correction worth carrying into the report:** the majority counter
gives a *time constant, not a theorem*. Boyer–Moore's guarantee assumes an
unbounded counter; a saturating one discards the evidence the proof rests on. A
cap of 7 loses a 16-of-31 majority. The defensible sentence is that the cell is
exactly textbook Boyer–Moore on any sequence whose running excess stays within
C, and re-labels after more than C net contradicting observations.

### 3.2 The lattice

Cell indices are **integer**: `i_L = i_fine // k_L` where `i_fine = floor(x/0.05)`
and `k_L` is an integer ratio. Never `floor(x/0.20)` directly — float lattices
drift apart and produce gaps and double-counts at ring boundaries. `validate()`
rejects non-integer ratios between consecutive rings. Powers of two are a
convenience, not a requirement: 5/10/50 is legal because 10/5 = 2 and 50/10 = 5,
and it is the schedule that catches power-of-two assumptions in the code.

Rings are stored as **full squares, toroidally**: absolute lattice cell (ix, iy)
lives permanently at slot (iy mod W, ix mod W). Nothing is ever copied. Ego-motion
moves the *origin*, and the only work per shift is clearing the rows and columns
that just came into view — O(perimeter), about 1,000 cells for ring 3 against
250,000 for an O(area) scroll. **Measured: 0.04 ms against 15.2 ms.**

745,000 is the **logical** cell count and the only number the report's ratios
use. 910,000 is what is allocated (annuli stored as squares). These must never be
conflated on a slide.

---

## 4. The mathematics that is load-bearing

Three results carry weight. The rest of `docs/sih-math.md` is supporting.

### 4.1 Merge uses the law of total variance

Four children merging into a parent do **not** measure the same quantity — they
measure different *places*. So inverse-variance fusion is the wrong rule:

```
σ²_p = Σ wᵢσᵢ²  +  Σ wᵢ(μᵢ − μ_p)²
       └ within ┘    └── between ──┘
```

Dropping the second term is exactly what Adaptive Patched Grid Mapping [3] does,
and the consequence is that a merged cell becomes **most confident precisely
where it straddles a kerb** — the single place you would least want false
confidence. This is the clearest, most defensible technical criticism in the
project's prior-art analysis, and it should be in the report.

### 4.2 Split inflates variance and sets a `derived` bit

Children inherit the parent's mean and a strictly larger variance, and the
`derived` bit is what makes `merge(split(c)) == c` **exact**. Without it, a cell
oscillating across a ring boundary as the vehicle drives inflates its variance
every frame with no physical cause. Pinned by the §6.3 hysteresis test: a
boundary cell held for 1,000 frames of sinusoidal speed with the variance byte
unchanged.

### 4.3 Unknown ≠ free

Three occupancy states. `unknown` is decided by **observation count**, not by
log-odds sitting near zero. The 3.74 m blind cone under the sensor is `unknown`
forever, never `free`. `world/map/free` and `world/map/unknown` are separate
entities in the dashboard on purpose.

### 4.4 Visibility cleanup, and the guard that matters

For a map cell at `p`, project it into the current range image and compare:

```
(u, v) = proj(p),   r_expected = ‖p − p_sensor‖
see-through   iff   R_current(u, v) > r_expected + δ
```

with `δ = 3σ(r)` so the band widens with range, floored at the config's 0.30 m
for pose error. A flat 0.30 m would be 26.9σ at 5 m and 1.7σ at 100 m — barely
clearing in the near field where ghosts matter most, and clearing real structure
at range, which is what the guard exists to prevent.

**The guard: never clear a cell that has a return in the current scan.** Without
it the cleanup eats fences, poles and sign posts within a few frames. Measured:
**429,012 cells spared** by this guard in a 60-frame run. Quote that number
alongside the removal rate — it is the evidence the cleanup is conservative.

A cell's visibility height is **`ceiling_height`, not `ground_height`.** A cell
whose returns are all non-ground has `w_sum == 0` and reads `ground_height == 0`,
and 0 cm is the datum, not "no information". Projecting a parked car at 0 aims
the ray 1.73 m below the sensor and lands it on a different image row: measured,
all 379 car cells read ground 0 while the ceiling carried the real 34 cm, and not
one ghost cleared.

---

## 5. Perception

**Ground segmentation** is **Patchwork++** [4], wired in rather than
reimplemented. Without it the pipeline falls back to a semantic-class mask and
**says so, loudly** — the fallback includes `terrain` and admits embankments, so
the ground layer must not be demonstrated on it.

Installation is a known trap: `pip install pypatchworkpp` fails at every
published version (the sdist's CMake fetches `.../tags/v${CMAKE_PROJECT_VERSION}.tar.gz`,
the variable is empty under scikit-build-core, GitHub 404s). Building from a git
clone works because it takes the in-tree branch.

**Range image** is 64 × 512, following FLARES [5]: lower azimuth resolution with
sub-clouds improves both runtime and accuracy over full 64 × 2048 sweeps in
memory-constrained settings.

**Reflectivity.** KITTI intensity is already firmware range/incidence-compensated,
so the raw-power `·r²/cos` normalisation is deliberately **not** applied to it —
it saturated 62% of near-field road at the byte rail. The normalisation path is
retained for sensors that need it. This is a good "we read the sensor
documentation" detail.

**Height datum.** `quantise_height` clips to an 8 m band. Originally that band
was world-absolute at datum 0, which meant that on sequence 08's 45.7 m climb the
near-field heights saturated at the +6 m ceiling while the sensor sat tens of
metres higher, and **2,304 of 4,071 frames (57%) were fully inert** for ghost
removal. The band now slides in whole 1 m steps to follow the vehicle and
re-bases stored heights when it moves — the vertical counterpart of the toroidal
horizontal shift. Post-fix: **0 of 4,071 frames inert**, clearing flat at 15–20 k
cells/frame across every elevation band.

Note the fix keeps the band 8 m wide, which is why the dense-3D baseline counts
the same voxels and the 286× ratio is untouched. Widening the clamp would have
moved a headline number; moving the band did not.

---

## 6. Performance and scaling

### 6.1 What "GPU" means here, precisely

`src/gpu/` is a **GPU-shaped architecture implemented in numpy on CPU**.
`allocators.array_module()` returns `numpy` for `device="cpu"` and imports `cupy`
otherwise, but cupy appears in no test and no script. Every latency figure in
this project was measured single-threaded on an Intel i7-14650HX. **No CUDA
kernel exists in the mapping path.**

State it that way. The design decisions are all GPU-motivated and GPU-portable,
and each is measured; that is a real engineering contribution and it does not
need inflating.

### 6.2 Determinism as a design constraint

IEEE-754 addition is not associative, so GPU atomic float adds — which complete
in nondeterministic order — produce a different map every run, and **you cannot
bisect a bug whose location moves.** Heights are quantised to 1 cm anyway, so
everything accumulates as int32, which is exactly associative. `make
test-determinism` (same input twice → identical map hash) is CI-blocking, as is
the partition test (10⁶ random points, exactly one cell per ring).

Two scatter paths exist and are asserted **bit-identical**: `scatter_sorted`
(default, scratch sized by points) and `scatter_atomic` (dense accumulator, the
literal reading of the spec). If they diverge, the optimisation is the bug. At
120,000 returns into 745,000 cells: sorted **p50 6.65 / p99 9.81 ms**, atomic
**p50 20.56 / p99 30.51 ms**.

### 6.3 The optimisation record

> **Read the two rows marked "mapping back end" as exactly that.** Both read
> "Whole frame" until 2026-09-13 and neither was. They come from
> `timing_table.py --alloc`, which instruments the back end only and whose own
> output names `load`, `transform`, `range_image`, `semantics` and `motion` as
> *"not in the subtotal above"*. Two separate caveats follow:
> **(a) code** — the perception front end is not covered, and allocates
> ~39.5 MB/frame on real seq 08 (~59.4 MB whole-frame); **(b) metric** — the CI
> test measures retained growth across frames, not per-frame churn, so an
> allocate-and-free inside one frame does not trip it.

| Change | Before | After |
|---|---|---|
| Single-pass `bin_points` (no per-ring loop) | 6.962 MB/frame, 13.64 ms | **0.002 MB/frame, 12.35 ms** |
| `occupancy_state` with `out=`/`scratch=` | 8.19 MB/call | **0** |
| `np.take` with intp indices + `mode="clip"` | 3.2 MB/frame | **1 KB** |
| Mapping back end, allocation | 8.15 MB/frame | **1.31 MB/frame** |
| Mapping back end, p99 | 74.7 ms | **49.4 ms** |
| Toroidal shift vs annulus gather | 15.2 ms | **0.04 ms** |
| Refinement pool release (flaw E1) | 512/512 full, 15,791 refusals | **62/512, 0 refusals** |

Two allocations no profiler names, worth writing up because they are genuinely
non-obvious: `np.take(table, idx, out=)` builds a full-length bounds-check array
under its default `mode="raise"` (0.96 MB per call, six calls a frame), and
`int64 += bool` casts through numpy's fixed 64 kB internal buffer.

The `np.take` change alone took scatter's p99 from 8.9 to 6.1 ms and visibility's
from 17.5 to 11.4 ms.

### 6.4 The refinement pool

512 blocks × 16 cells × 12 B = 98 KB, preallocated. This is what lets semantics
buy back resolution locally without breaking the bound. Under load, the **correct
behaviour of a fixed pool is refusal and eviction, not growth** — a test
asserting "the crowd is fully mapped" would assert the opposite of the design.

Verified against a worst case designed to stress four different caps at once:
`synthetic.scan(crowd=N)` adds N pedestrians on the raw `moving-person` id, so
every return is dynamic (transient layer + track list), `person` is a refine
class (semantic gate + pool), they are small and close (many fine cells), and
they are separate objects a metre apart (the clustering worst case).

| crowd | returns | peak transient |
|---|---|---|
| 0 | 47,579 | 21.01 MB |
| 400 | 57,179 | 21.26 MB |

**20% more returns for 1.2% more peak.** The bound is flat in the scene, which is
the claim the report actually makes.

### 6.5 End-to-end latency — the uncomfortable number

200 frames of sequence 08, one `Timer` across both halves, stages flat and
disjoint:

| stage | p50 | p99 | | stage | p50 | p99 |
|---|---|---|---|---|---|---|
| cleanup | 26.22 | 33.43 | | reflectivity | 3.49 | 4.64 |
| range_image | 24.45 | 29.27 | | shift | 2.07 | 5.52 |
| ground | 12.41 | 14.50 | | transform | 1.53 | 2.56 |
| scatter | 7.06 | 8.20 | | load | 0.58 | 0.86 |
| bin | 6.84 | 8.33 | | semantics | 0.45 | 0.68 |
| fuse | 4.13 | 5.28 | | motion | 0.06 | 0.09 |
| | | | | **TOTAL** | **89.18** | **100.43** |

**The median meets 10 Hz with 10.8 ms to spare. The p99 is 0.43 ms over the
100 ms budget; the max is 109.28 ms.** Report it. Two caveats: `load` at 0.58 ms
is page-cache-warm and would not exist on a live sensor, and the 80.78/97.72
figure circulating in `handover-2026-09-02.md` is **not comparable** — it is the
back half only, on a synthetic sweep.

The largest stage is visibility cleanup at 26 ms, which is also the most
parallel thing in the system. That is where the headroom is.

---

## 7. The deep learning model

### 7.1 What it is

**FRNet** [6] — a frustum-range network for LiDAR semantic segmentation,
19-class SemanticKITTI, ~10 M parameters, paper mIoU 73.3%. Chosen for efficiency
(roughly 5× faster than voxel baselines) over accuracy.

### 7.2 The port, and the three divergences

Only a standalone port was available (no mmdet3d). It initially scored **~15%
point accuracy** — the network loaded and ran and produced nonsense. Three causes,
all found and fixed:

1. **Activation.** `nn.LeakyReLU` where the checkpoint trained with mmcv's
   HSwish, at 7 sites in the backbone. mmcv's HSwish is `x·relu6(x+3)/6`, which
   *is* `nn.Hardswish`.
2. **Field of view.** Not in the model — `semantics.py` was overriding the
   model's correct default (3.0° / −25.0°) with the HDL-64E's **physical** FOV
   (2.0° / −24.8°) out of the config. These are two different quantities: the
   checkpoint learned a *fixed* projection. Now pinned as constants a sensor
   config cannot reach.
3. **`range_interpolation`.** Described as "a missing densification", which reads
   as a resolution setting. It is a *test-pipeline transform to reproduce
   verbatim*, and it projects with the divergence-2 FOV — the two were coupled.
   Transcribed from upstream **including an off-by-one** (`proj_idx > 0`, so
   point index 0 reads invalid against a −1 sentinel) reproduced **deliberately**,
   because the paper's 73.3% was measured with it and "fixing" it makes the
   numbers incomparable.

The checkpoint loads 421 tensors, 0 missing, 8 unexpected (all `auxiliary_head.*`,
training-only, correctly unused at inference).

**Result: 90.3% point accuracy, 65.2% mIoU** over 200 frames of sequence 08,
against the paper's 73.3%. (A separate 98.3% figure in circulation is a
*single-frame* port check on seq 00 frame 43; do not quote it as the result.)

### 7.3 The mIoU correction

The reported 69.8% **does not reproduce and is arithmetic**. The 15 per-class
IoUs sum to 977.7: divided by 15 that is 65.18%, by 14 it is 69.84%. The missing
fifteenth class is `other-ground` — 150 ground-truth points over 200 frames, IoU
0.0%, present and therefore counted. Point accuracy reproduces exactly, and the
loop and fast-scatter paths agree at 90.3 / 65.2 / 61.1. **65.2% is the number.**

### 7.4 The forward pass was 10.5 s/frame, and ~90% was a Python loop

Measured on one seq 00 frame of 121,018 points, CUDA:

| stage | cost |
|---|---|
| disk load | 0.7 ms |
| `range_interpolation` | 60.5 ms |
| `frustum_region_group` | 0.4 ms |
| `voxel_encoder` | **4,678.5 ms** |
| full forward | **10,535.1 ms** |

`scatter_max` and `scatter_mean` were `for i in range(dim_size)` loops building a
full-length boolean mask per output slot. `dim_size` is the number of occupied
frustum pixels (~25,000), so each call is ~25,000 iterations over a 124,000-row
tensor — and there are **seven such calls per forward**.

`torch.scatter_reduce_` does the same reduction natively:

| | loop | `scatter_reduce_` | speedup |
|---|---|---|---|
| `scatter_max` (CUDA, real shapes) | — | — | **1408×** |
| `scatter_mean` (CUDA, real shapes) | — | — | **541×** |
| `scatter_max` (CPU) | 37,452 ms | 34.3 ms | 1093× |
| `scatter_mean` (CPU) | 34,452 ms | 10.7 ms | 3229× |

**Numerics, stated precisely.** `scatter_max` is bit-identical on both CPU and
CUDA (max is order-independent). `scatter_mean` is bit-identical on CPU but
differs by up to **2 float32 ulp (2.384e-07) on ~40% of slots on CUDA**, because
the native kernel sums a slot's rows in a different order and float addition is
not associative. The verifier gates max at exactly zero and mean at a stated ulp
bound rather than asserting a bit-identity that is not there. Backward: max is
exact, mean carries the same 2 ulp.

Consequence: a 600-step fine-tune went from **3.3 hours to 2.2 minutes**, and
`frnet_eval.py --frames 200` from ~35 minutes to about one.

The shim is applied at runtime behind `--fast-scatter` and does **not** edit
`src/perception/frnet/`, which is deliberately frozen as a reference port. One
trap worth recording: `frnet_backbone` does `from .frustum_encoder import
scatter_max` at import, which *binds the function object* — rebinding only the
encoder's copy leaves five of the seven calls on the slow path, and the run
merely looks disappointing rather than broken.

### 7.5 Fine-tuning: tried three ways, rejected on measurement

Held-out sequence 08, 200 frames:

| run | recipe | point acc | mIoU |
|---|---|---|---|
| — | pretrained checkpoint | 90.3% | **65.2%** |
| A | head only, 3× weights on terrain/vegetation, 600 steps | 89.8% | 64.6% |
| B1 | head, no class weights, lr 1e-4, 2,000 steps | 90.2% | 65.3% |
| B2 | head + backbone, lr 1e-4, 4,000 steps, batch 1 | 89.5% | 64.5% |

Per-class on run A: `terrain` moved +0.9 (the target), and everything else paid
for it — `trunk` −4.2, `building` −1.6, `bicycle` −1.6, `fence` −1.2. The gain on
the five classes the map actually consults was +0.3, inside noise.

**This is the expected answer, not a failure to tune.** The checkpoint was
already trained on 00–10 minus 08, so every recipe here retrains on its own
training set with no domain gap to close. The pretrained checkpoint stays the
reported model.

Two methodological points worth carrying into the report:

- **Training loss fell 0.167 → 0.1425 across run A.** That is exactly why
  training loss is not a result: a class-weighted loss falls when the head gets
  more confident on the weighted classes, whether or not it gets more correct.
  Sequence 08 was the only honest read.
- **Freezing weights is not freezing a module.** `model.train()` puts every
  BatchNorm into training mode, so a "frozen" backbone still drifts its running
  mean and variance on every forward pass — the weights hold still and the
  function the module computes does not. Frozen modules are held in `.eval()`.

A documented negative result that can be re-derived from a committed script is a
stronger position than never having tried. Present it as one slide.

---

## 8. Evaluation

Three independent claims, evaluated separately.

### 8.1 Memory — the strongest claim

Cell-count ratios, invariant to bytes per cell, over a 200 × 200 m footprint at
5 cm base resolution and 8 m vertical extent:

| | cells / voxels | bytes |
|---|---|---|
| dense 3D voxel | 2.56 × 10⁹ | 2.56 GB |
| uniform 2.5D | 16.0 × 10⁶ | 192 MB |
| **vrgrid, 4-ring** | **745,000** | **8.94 MB** |

**21.5× against uniform 2.5D, 286× against dense 3D.** Total preallocated
including working buffers is 29.06 MB; against uniform 10 cm at matched extent
(78.50 MB) that is 2.7×.

**A baseline that does not fault its pages in is not a baseline.** `np.zeros`
returns copy-on-write zero pages: `np.zeros(2_560_000_000, np.uint8)` moves
resident set size by **0.0 MB**. Built the obvious way, the dense-3D baseline
would show 0 MB beside our counter on stage, claiming 286× over something visibly
free. `baseline.commit()` touches one byte per page; 2.56 GB costs 0.23 s once.
Measured claimed/resident: ours 27.86/27.98, uniform 192.00/192.03, dense
2.56 GB/2.56 GB.

### 8.2 Geometric accuracy — the headline result

All eleven labelled sequences, 40 frames each, schedule 5/10/20/40, correct
per-sequence poses, Patchwork++:

```
ring 1:  ρ median 1.45  [1.26–1.59]    RMSE median  3.53 cm  [2.28–12.16]
ring 2:  ρ median 1.46  [1.18–2.32]    RMSE median  9.48 cm  [4.39–28.09]
```

**Lead with ρ, not RMSE.** The coarsening-justification ratio ρ = IL/spread
divides out how rough each road happens to be and leaves what the *coarsening*
cost. At ring 1, ρ spans 1.26× across the whole dataset while RMSE spans 5.3×.
"ρ ≈ 1.45, range 1.26–1.59, n = 11" is both more defensible and closer to the
actual claim than any single sequence's RMSE.

Three disclosures that belong next to it:

- **07 and 08 are at the good end, not typical.** Their ring-1 ρ of 1.32 and 1.30
  sit near the bottom of the range against a median of 1.45. Quote the
  distribution.
- **Ring 0 has no ρ on any sequence**, and this is *arithmetic, not physics*.
  `block_stats` counts observed **cells**; a ring-0 footprint is exactly one
  cell, so `n_ref` can never exceed 1 and the `n_ref > 1` guard drops it every
  time. The redundancy exists — the reference map holds >1 return in 92.8 / 97.0
  / 97.4% of scored ring-0 cells on 00 / 07 / 08 — but the sum of squares is
  computed at build time and discarded. Closing it would score ring 0 at
  **ρ 1.01–1.24, the best of any ring**, which is exactly what the foveation
  argument predicts. Deferred because the same change moves ring 1 by −21.5% on
  07 and invalidates every cached reference map.
- **Do not quote ρ to two decimals.** The §9.2 band-filter fix changes the scored
  population by up to 0.06 per ring, and the eleven-sequence table predates it.

### 8.3 Plan regret — the contribution, and the honest status

The idea, and it is a good one: do not measure the map against ground truth in
centimetres, measure it against the **decision**. Plan a path on the compressed
map `M_S`, plan the optimal path on the 5 cm reference `M*`, and score **both on
`M*`**:

```
R(S) = J_M*(π_S) − J_M*(π*)
```

Scoring both on `M*` is the key invariant: an unobserved obstacle or a blurred
kerb produces *infinite regret rather than false safety*.

**Three real defects were found and fixed in this metric**, and the debugging is
itself reportable:

1. **The fill-rate confound.** `costmap_from_gridmap` was OR-ing the confidence
   bit over the sub-cells of a planning cell, so **the handicap grew with
   resolution** — a penalty on every cell, for resolving finely. Inside the
   common support the frozen schedules went from **100.0% to 0.9%** of cells
   paying it. Worse, the diagnostic built to expose this read the wrong array and
   printed 0.0% where the real figure was 100.0%.
2. **The two lattices.** `M_S` set traversability bits at ring resolution while
   `M*` set them at 25 cm. A 12 cm kerb is a step at 5 cm and smooth at 25 cm, so
   the schedules invented **148 impassable cells** the reference did not have and
   missed 8 it did. Both sides are now evaluated at the planning cell from the
   same summed statistics: **0 invented, 0 missed.**
3. **Bit 4 on one side only.** The reference had no class opinion, so it charged
   0 class penalties against the schedules' 18 — a schedule paid pure regret for
   routing around ground it had *correctly* labelled non-drivable.

**And after all three, the result does not favour the schedules.** Sequence 08,
20 frames, 64-query mean, matched extent:

| schedule | MB | cells | R(S) | Fréchet |
|---|---|---|---|---|
| uniform 10 cm | 78.50 | 4,000,000 | 0.231 | 0.18 m |
| uniform 20 cm | 30.50 | 1,000,000 | **0.251** | 0.23 m |
| **5/10/20/40** | **29.06** | 745,000 | **0.488** | 0.33 m |
| 5/10/50 | 23.62 | 520,000 | 0.488 | 0.33 m |
| uniform 40 cm | 18.50 | 250,000 | **0.402** | 0.32 m |
| uniform 80 cm | 11.04 | 62,500 | 0.798 | 0.55 m |

Uniform 20 cm at essentially the same memory scores better; uniform 40 cm is
cheaper *and* better. `regret_plot.py`'s monotonicity guard fires on that step.

**What this is and is not evidence of.** One sequence, one window, one query
family — a single longitudinal lane down the middle of the window, unchanged
since Day 0 and never designed to discriminate between resolutions. **A lane
query rewards a map that is uniformly adequate along one line and cannot reward
one that is sharp where the vehicle is looking.** It is not proof the thesis is
wrong; it is proof this query cannot demonstrate it.

Two further constraints on any quoted figure: **R(S) is comparable across
schedules at a fixed window and not across windows** (5/10/20/40 reads 0.171 at
40 frames and 0.758 at 160), so every number must state its frame count. And the
memory axis must be at matched extent — the earlier figure compared 0.0400 km²
against 0.0023 km² and inverted the memory conclusion.

**What holds:** the two frozen schedules produce identical plans despite 5.4 MB
between them, and the uniform series rises with cell size at every window.

### 8.4 Curbs and potholes

Measured on all eleven sequences through the real pipeline. **Ring 0 returns
8.1–9.1 cm on every one of eleven sequences** — different recording dates,
different calibrations, a one-centimetre band. That consistency is the evidence
the detector measures a physical feature rather than an artifact, and it is a
stronger claim than any single number.

The rise with ring is systematic and physical (8–9 cm at 5 cm cells, 9–12 at
20 cm, 9–18 at 40 cm) because a coarser cell straddles the kerb face and averages
in sloped ground. Report per ring; ring 3's spread is where it stops being
reliable.

**The limitation: SemanticKITTI has no ground truth for curb or pothole
geometry.** There is no detection rate to quote, only counts and a plausibility
check on the height distribution. Potholes range 56–551 cells per sequence, a 10×
spread with no pattern — a demonstration, not a rate. Say this before you are
asked.

A methodological note worth reporting: a baseline sweep on three sequences
suggested moving `curb.baseline_m` from 0.20 to 0.30 m. Run across all eleven, it
raised the median 0.7 cm and **nearly tripled the cross-sequence spread** (0.36 →
0.89 sd). It was reverted. The tight band *is* the claim; trading it for 0.7 cm
of median trades the result for the number. Three sequences were not enough to
choose; eleven were.

### 8.5 Per-cell confidence

Four derived channels, combined by taking the weakest, nothing stored (the cell
stays 12 bytes). **It is not calibrated** — nothing has been fitted against
outcomes, so 0.6 is not a 60% chance of anything. Each channel is a margin with a
stated meaning. The `label` channel is additionally a *floor, not an estimate*:
the Boyer–Moore counter saturates at 7, so a cell observed 200 times unanimously
reports a *lower* share than one observed 8 times.

---

## 9. What was tried and rejected

A panel asks this. Having crisp answers is worth more than the successes.

| Tried | Outcome | Why it is worth reporting |
|---|---|---|
| **FRNet in the mapping pipeline** | Rejected by design | Isolates mapping from segmentation error |
| **Fine-tuning FRNet**, 3 recipes | Rejected on measurement | Checkpoint already converged on this data; documented negative result |
| **`curb.baseline_m` 0.20 → 0.30** | Reverted | Tripled cross-sequence spread for 0.7 cm of median |
| **Annulus storage** for rings | Rejected | Saves 1.98 MB, makes the shift a gather: 15.2 ms vs 0.04 ms |
| **Inverse-variance merge** | Rejected | Wrong: children measure different places, not the same quantity |
| **Flat 0.30 m clearing tolerance** | Replaced by 3σ(r) | 26.9σ at 5 m, 1.7σ at 100 m — wrong at both ends |
| **`visibility.max_candidate_cells = 150000`** | Replaced with the grid's slot count | Dropped 52.3% of 07's and 67.1% of 08's peak occupied set, *silently* |
| **Conservative pyramid** | Built, tested, **off by default** | Works (rebuild p50 2.45 ms, 32× headroom) but moves the total 29.06 → 32.17 MB, and a number on a slide does not move because a default did |
| **OptiX / RT-core visibility** | Not attempted | Unsupported on Jetson; cleanup is already O(1) per cell |
| **Raw-power reflectivity normalisation** | Not applied to KITTI | Intensity is already firmware-compensated; saturated 62% of near-field road |

---

## 10. What we would do next

1. **A planning query that can discriminate resolution.** The single item
   standing between this project and its headline claim. The current lane query
   is structurally incapable of testing the thesis.
2. **Ring-0 ρ**: store Σh² in the reference map. ~15 lines, two-day tail (every
   cached `M*` invalidates, the eleven-sequence table regenerates). Expected
   ρ 1.01–1.24.
3. **A real device path.** `array_module()` is the seam; move the arrays to cupy
   and re-measure. The determinism guarantee constrains what may move.
4. **Close the p99.** Visibility cleanup is 26 ms of 89 and is the most parallel
   stage.
5. **Range-stratified reference map** — (n, Σh, Σh²) per range band per 5 cm
   cell. Measured worth ≤0.05 cm on the synthetic scene, at 4× the memory; wants
   re-measuring on real data where the rear band has a kerb in it.
6. **Live semantics** via mmdet3d, reported *alongside* the GT-label result.
7. **BeautyMap-style static restoration** [7] if over-clearing of thin geometry
   appears on other sensors.

---

## 11. References

*Verified against peer-reviewed proceedings by the team's research modules; see
`docs/research-log.md` for the per-paper findings and `docs/prior-art-taxonomy-matrix.md`
for the comparison matrix.*

**Datasets and sensor**

[1] Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C.,
Gall, J. *SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR
Sequences.* ICCV 2019.

[2] Geiger, A., Lenz, P., Urtasun, R. *Are we ready for Autonomous Driving? The
KITTI Vision Benchmark Suite.* CVPR 2012.

**Representation and prior art**

Triebel, R., Pfaff, P., Burgard, W. *Multi-Level Surface Maps for Outdoor Terrain
Mapping and Loop Closing.* IROS 2006. — *the 2.5D lineage; cite next to the ring
diagram.*

Droeschel, D., Stückler, J., Behnke, S. *Local Multi-Resolution Representation
for 6D Motion Estimation and Mapping with a Continuously Rotating 3D Laser
Scanner.* ICRA 2014. — *the closest prior art; cite it first, unprompted.*

Losasso, F., Hoppe, H. *Geometry Clipmaps: Terrain Rendering Using Nested Regular
Grids.* ACM SIGGRAPH 2004. — *the toroidal scrolling lineage.*

Fankhauser, P., Bloesch, M., Gehring, C., Hutter, M., Siegwart, R.
*Robot-Centric Elevation Mapping with Uncertainty Estimates.* CLAWAR 2014. —
*confirms the range-dependent Kalman measurement-variance model σ²_m(r) = σ₀² + c·r².*

[3] Wodtko, T., Griebel, M., Buchholz, M. *Adaptive Patched Grid Mapping.*
arXiv:2308.03416, 2023. — *merges by inverse-variance averaging, dropping the
between-cell term. This is the criticism our §4.1 rests on.*

Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., Burgard, W. *OctoMap: An
Efficient Probabilistic 3D Mapping Framework Based on Octrees.* Autonomous
Robots, 2013.

Reijgwart, V., Cadena, C., Siegwart, R., Ott, L. *wavemap: Efficient Volumetric
Hierarchical Occupancy Mapping.* RSS 2023.

Yang, T., Cheng, K., Xue, J., Jiao, J., Liu, M. *Efficient Global Navigational
Planning in 3D Structures based on Point Cloud Tomography.* IEEE/ASME T-Mech,
2024. arXiv:2403.07631.

Patel, et al. *RoadRunner M&M: Learned Multi-Range Multi-Resolution Elevation
Mapping.* RA-L 2024.

Tevs, A., Ihrke, I., Seidel, H.-P. *Maximum Mipmaps for Fast, Accurate, and
Scalable Dynamic Height Field Rendering.* I3D 2008. — *the conservative pyramid's
graphics lineage.*

**Perception**

[4] Lee, S., Lim, H., Myung, H. *Patchwork++: Fast and Robust Ground
Segmentation Solving Partial Under-Segmentation Using 3D Point Cloud.* IROS 2022.

[5] *FLARES: sub-cloud range representations for LiDAR segmentation.*
arXiv:2502.09274, 2025. — *the 64 × 512 sub-cloud choice.*

[6] Xu, X., Kong, L., Shuai, H., Liu, Q. *FRNet: Frustum-Range Networks for
Scalable LiDAR Segmentation.* arXiv:2312.04484; IEEE TIP, 2025.

Vizzo, I., Guadagnino, T., Mersch, B., Wiesmann, L., Behley, J., Stachniss, C.
*KISS-ICP: In Defense of Point-to-Point ICP.* RA-L 2023.

**Dynamic removal**

Zhang, Q., et al. *A Dynamic Points Removal Benchmark in Point Cloud Maps.* ITSC
2023. — *defines Preservation Rate and Dynamic Removal; note their benchmark is
offline global map cleaning where ours is an online rolling local map.*

[7] *BeautyMap: Binary-Encoded Adaptable Ground Matrix for Dynamic Points Removal.*
RA-L 2024. — *range-visibility filtering over-clears thin geometry without static
restoration.*

Lim, H., Hwang, S., Myung, H. *ERASOR: Egocentric Ratio of Pseudo Occupancy-based
Dynamic Object Removal.* RA-L 2021.

Duberg, D., et al. *DUFOMap: Efficient Dynamic Awareness Mapping.* RA-L 2024.

Nuss, D., Reuter, S., Thom, M., Yuan, T., Krehl, G., Maile, M., Gern, A.,
Dietmayer, K. *A Random Finite Set Approach for Dynamic Occupancy Grid Maps with
Real-Time Application.* IJRR 2018. — *DOGMa; the particle-grid alternative.*

**Traversability and decision-sensitive evaluation**

Sivaprakasam, M., et al. *SALON: Self-supervised Adaptive Learning for Off-road
Navigation.* ICRA 2025.

Cai, X., et al. *EVORA: Deep Evidential Traversability Learning for Risk-Aware
Off-Road Autonomy.* IEEE T-RO.

Psomiadis, E., et al. *Communication-Aware Map Compression for Online
Path-Planning.* ICRA 2024. arXiv:2309.13451. — *closest work in decision-sensitive
compression; targets multi-robot bandwidth on generic 2D grids. Verdict: no
preemption.*

Larsson, D. T., et al. *Q-Tree Search: An Information-Theoretic Approach Toward
Hierarchical Abstractions for Agents with Computational Limitations.* RA-L 2021.

---

## Appendix — reproducing every number

Gate 6 rule: **every number on a slide comes from a script in `scripts/`.**

| Claim | Command |
|---|---|
| Memory table | `python scripts/memory_table.py` |
| Memory bound under load | `python scripts/memory_bound.py` |
| Per-stage latency | `python scripts/timing_table.py --seq 08 --frames 200` |
| Per-ring accuracy, ρ | `python scripts/eval_synthetic.py --seq <NN>` |
| Curbs and potholes | `python scripts/feature_report.py --seq <NN>` |
| Ghost removal figure | `python scripts/ghost_removal_figure.py --seq 08` |
| Plan regret / money plot | `python scripts/regret_plot.py --seq 08` |
| FRNet eval | `python scripts/frnet_eval.py --frames 200 --fast-scatter` |
| FRNet fine-tune | `python scripts/frnet_finetune.py --fast-scatter` |
| Reduction equivalence | `python scripts/frnet_fast_scatter.py` |
| Ring schedule derivation | `python scripts/sampling_table.py` |
| Reference map build | `python scripts/build_reference_map.py <NN>` |
| Dataset integrity | `python scripts/data_status.py` |

⚠️ `scripts/timing_table.py`'s docstring still claims there is no end-to-end loop
to time. That is stale — `src/run/engine.py:297–309` calls the real
`scatter_sorted` and `fuse`. Fix before anyone reads it.
