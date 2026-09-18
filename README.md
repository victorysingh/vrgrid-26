# VRgrid — Foveated 2.5D LiDAR Mapping

**Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception**

[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-orange)](https://www.sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/PS-SIH26053-blue)](https://www.sih.gov.in/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-CPU%20reference-013243.svg)](https://numpy.org/)

> **VRgrid is a deterministic-by-design, memory-bounded, foveated 2.5D LiDAR mapping system that allocates spatial resolution according to range, semantics and direction — while preserving uncertainty during coarsening and removing transient dynamic-object ghosts.**

**Smart India Hackathon 2026 · SIH26053 · DRDO · Smart Vehicles · Team Chronicles.exe**

---

## Overview

Conventional LiDAR mapping commonly uses a uniform spatial resolution across the entire sensing range.

That is wasteful.

At longer ranges, LiDAR beams become increasingly sparse. At 50 m, consecutive laser rings can land approximately **10.8 m apart on the ground**, meaning that **99.87% of the cells of a uniform 5 cm grid cannot receive a return in a single frame**.

At the same time, accumulated maps can retain stale elevation information from moving vehicles and other transient objects.

**VRgrid addresses both problems with one mapping representation:**

* Fine resolution where the sensor provides useful spatial information.
* Coarser resolution as range increases.
* Semantic information incorporated into the resolution policy.
* A fixed memory envelope allocated once at startup.
* Variance-preserving split/merge operations.
* Dynamic-object ghost removal using range-image visibility.
* Resolution-independent world-coordinate queries.
* Deterministic execution with bit-identical map hashes — **design target;
  not currently achieved**, see [Determinism](#determinism).

The result is a compact **2.5D elevation map** designed for downstream robotic planning and perception.

---

## Why VRgrid?

### The problem with a uniform grid

A uniform 5 cm grid assumes that the LiDAR sensor can populate that resolution everywhere.

It cannot.

As range increases, the physical spacing between consecutive LiDAR returns increases dramatically. Maintaining the same resolution therefore produces large regions of cells that cannot possibly receive observations during a frame.

VRgrid instead uses **nested resolution rings**:


<img width="446" height="378" alt="image" src="https://github.com/user-attachments/assets/13fc47fc-a7d6-4cd9-b69c-c25e53491713" />


### VRgrid resolution schedule

| Range    | Resolution |
| -------- | ---------: |
| 0–10 m   |       5 cm |
| 10–25 m  |      10 cm |
| 25–50 m  |      20 cm |
| 50–100 m |      40 cm |

All rings remain part of one global **5 cm lattice**, allowing the representation to change resolution without changing the interface presented to downstream systems.

---

## Key Results

| Metric                      |                     VRgrid |
| --------------------------- | -------------------------: |
| Fixed map footprint         |                **8.94 MB** |
| Cells                       |                **745,000** |
| Cell storage                |                   **12 B** |
| Memory vs uniform 5 cm 2.5D |            **21.5× lower** |
| Memory vs dense 5 cm 3D     |             **286× lower** |
| End-to-end frame latency    | **89.18 ms p50 / 100.43 p99** † |
| Ghost trails                |       **0 / 4,071 frames** |
| Determinism                 | **Bit-identical map hash** ‡ |

> ‡ **Determinism is currently qualified, and the qualification is open item D1.**
> The kernel path — scatter, fuse and the map hash — is bit-identical and
> CI-gated. **End-to-end replay is not.** `src/perception/ground.py` holds a
> stateful Patchwork++ singleton, so `test_real_sequence_replay_is_identical`
> fails: two replays of the same sequence in one process differ by **1,245 of
> 1,479,013 points**. Until D1 is closed, "bit-identical map hash" is true of the
> kernel path and false of a full replay. See `OPEN-ITEMS.md` §2.

> † **Quote this with its host attached** — open item D8. The tree currently
> holds two honest end-to-end figures for the same quantity on different
> machines: 89.18 / 100.43 ms here, and 108.65 / 127.23 ms in
> `docs/handover-2026-09-02.md`. One is 0.43 ms over the 100 ms budget and the
> other 27 ms over. Neither is "the" frame latency until D8 picks one reference
> host and one command.

The fixed footprint is allocated at startup:

```text
745,000 cells × 12 bytes = 8.94 MB
```

This avoids per-frame allocation and gives the mapping system a predictable memory envelope.

---

## The Core Idea

VRgrid combines **three signals** to determine spatial resolution:

```text
                 ┌─────────────┐
                 │    Range    │
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │ Resolution  │
                 │   Policy    │◄──── Semantics
                 └──────┬──────┘
                        │
                     Direction
                        │
                 ┌──────▼──────┐
                 │   2.5D      │
                 │    Grid     │
                 └─────────────┘
```

The policy is expressed as:

```text
s = clamp(
    s_min,
    s_max,
    α₁ g(range) + β h(class) + γ k(direction)
)
```

This is deliberately different from approaches that make resolution depend only on distance or only on semantic importance.

**VRgrid uses range AND semantics together.**

---

# Architecture

```text
                 SemanticKITTI
              ┌─────────────────┐
              │ LiDAR + Labels  │
              │ Motion Flags    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Frame Transform │
              │ Range Projection│
              │     Deskew      │
              │ Ground Segment. │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Resolution      │
              │ Policy          │
              │                 │
              │ Range           │
              │ + Semantics     │
              │ + Direction     │
              └────────┬────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │        Adaptive 2.5D Grid       │
        │                                 │
        │  Split / Merge                  │
        │  Variance Fusion                │
        │  Kalman Height Fusion           │
        │  Semantic Class Storage         │
        └───────────────┬─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Kernel Layer   │
              │  numpy · CPU    │
              │                 │
              │ Project         │
              │ Fuse            │
              │ Split           │
              │ Merge           │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Planner Query   │
              │ by World Coord. │
              └─────────────────┘
```

Every stage is designed to remain **resolution-agnostic**. A planner queries the map using world coordinates rather than depending on a particular cell index or resolution.

---

# Uncertainty-Honest Coarsening

Simply averaging child cells during a merge can create false confidence.

VRgrid instead uses the **law of total variance** so that spatial coarsening preserves the uncertainty represented by the original cells.

The important invariant is:

```text
merge(split(c)) = c
```

A coarse cell therefore does not pretend that its underlying terrain is more certain merely because several observations were combined.

For example, a merged kerb cell can report an uncertainty of approximately **σ = 6.3 cm**, rather than incorrectly collapsing that uncertainty toward something like 1 cm.

This makes the representation suitable for planners that need both elevation **and confidence**.

---

# Dynamic Object Handling

Accumulated elevation maps can retain stale observations from moving objects.

For example:

```text
Frame t:
             🚗
─────────────████──────────────

Frame t + N:
                  🚗
──────────────────████──────────

Naive accumulation:
─────────────████████████──────
              ↑ ghost trail
```

VRgrid separates transient observations and uses a **range-image visibility check** to remove stale dynamic-object elevation.

The prototype was evaluated on SemanticKITTI sequence 08 across **4,071 frames**, with **zero inert ghost trails after the elevation fix**.

---

# Determinism

Reproducibility is treated as a system requirement rather than an afterthought.

VRgrid uses:

* Integer/fixed-point accumulation where appropriate.
* Associative integer addition.
* Deterministic partitioning.
* Automated theorem/invariant tests.
* CI gates for every merge.

The same input should therefore produce the same map representation and the same
map hash, and a **bit-identical map hash** is the property the design targets.

The test suite includes a partition test over **10⁶ points**, ensuring that point partitioning does not alter the resulting map.

**Current status — the guarantee does not hold yet.** A known lifetime bug in the
ground-segmentation singleton breaks it on repeated passes over the same data:
`ground._estimator` is a module-level Patchwork++ estimator that accumulates
state, so re-processing a scan it has already seen returns a different
ground/non-ground verdict for that scan. Two replays of the same 50 frames in
one process therefore differ, and `test_real_sequence_replay_is_identical` —
which is CI-blocking, and correct to be — **currently fails**.

Measured: 1,245 of 1,479,013 points change verdict between a first and second
pass, in 4 of 12 frames. Two *separate* estimators making one pass each agree
exactly, 0 points differing — so the architecture is sound and the defect is
ownership of the estimator's lifetime, not the accumulation scheme.

The integer accumulation and associative addition above are real and do support
bit-identical determinism; **it is not currently guaranteed.** Tracked as
**Issue #1**; full diagnosis in `reports/ring1-reproduction-investigation.md`
and `OPEN-ITEMS.md` (D1). The fix is a design decision about who owns the
estimator and is deliberately not patched ad hoc.

---

# Performance

VRgrid is designed around a fixed computational and memory budget.

### Memory

| Representation             |      Memory | Relative |
| -------------------------- | ----------: | -------: |
| Dense 5 cm 3D voxel grid   |    ~2.56 GB |     286× |
| Uniform 5 cm 2.5D grid     |     ~192 MB |    21.5× |
| **VRgrid — 5/10/20/40 cm** | **8.94 MB** |   **1×** |

All measurements use the same spatial extent and vertical range.

### Compute

**The mapping pipeline is a CPU reference implementation in numpy. There is no
CUDA kernel in this repository, and every latency figure quoted here was measured
single-threaded on an Intel i7-14650HX.**

It is written the way GPU code is written, and each decision is measured:

* **Structure-of-arrays** throughout, for coalesced access.
* **int32 fixed-point accumulation, never float atomics** — float atomic adds are
  non-associative, so a float map differs run to run. Integer `atomicAdd` is exact
  and associative, which is why the determinism test is CI-blocking. (That test
  is currently **failing** for a reason unrelated to the accumulator — see
  [Determinism](#determinism).)
* **Zero allocation in the mapping back end's frame loop** — 8.15 → 1.31 MB/frame,
  p99 74.7 → 49.4 ms. †
* **A device seam in the allocator** (`allocators.array_module()`), so the arrays
  can move to cupy without touching the kernels.

> † **Two limits on that allocation figure, and they are different limits.**
>
> **(a) It covers the mapping back end only.** `bin`, `scatter`, `fuse`,
> `cleanup`, `shift` — not the perception front end. The figure comes from
> `timing_table.py --alloc`, whose own output lists `load`, `transform`,
> `range_image`, `semantics` and `motion` as *"not in the subtotal above"*.
> Measured on real seq 08, the **perception half allocates ~39.5 MB/frame** and
> the **whole frame ~59.4 MB/frame** of transient churn — `transform_points`
> alone is 10.86 MB. So the back end really is allocation-free; the pipeline is
> not. See `reports/r-b-p99-tail-investigation.md`.
>
> **(b) The CI test behind it measures RETAINED growth, not churn.**
> `test_no_allocation_inside_the_frame_loop` compares
> `tracemalloc.get_traced_memory()[0]` across frames, so it catches a buffer that
> grows with frame count — the failure that would make the compile-time bound
> false. A temporary allocated **and freed** inside one frame does not move it and
> will not fail the test. Its own docstring says so.
>
> Both gaps are real and neither implies the other. The back-end achievement is
> genuine and gated; the unqualified phrase *"in the frame loop"* was wider than
> either the code or the metric it rests on.

Porting to the device is the current cycle's work, tracked in
`docs/gpu-lane/03-CUDA-PORT-PLAN.md`. The determinism guarantee is expected to
survive it unchanged, because the accumulator is integer.

The one component that does run on CUDA today is FRNet inference, through torch —
and FRNet is deliberately not in the mapping pipeline (see below).

The intended deployment target is **Jetson-class hardware**.

---

# Evaluation

VRgrid does not evaluate mapping quality using RMSE alone.

The central question is:

> **Did compression change what the robot does?**

A map can have a small geometric error while still causing a planner to choose a different path.

Therefore VRgrid evaluates **planner regret**.

### Planner regret

Let:

* `S` = a resolution schedule
* `P(S)` = path planned using schedule `S`
* `C(P)` = cost of that path evaluated on the reference map

Then planner regret measures the cost difference between planning on the compressed map and planning on the high-resolution reference.

```text
R(S) = C(P(S)) - C(P(reference))
```

The evaluation first restricts both representations to common support so that finer resolution is not unfairly penalized simply for representing more space.

---

# Evaluation Metrics

VRgrid reports multiple complementary metrics:

| Metric                  | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| **Planner regret R(S)** | Measures whether compression changes decisions  |
| **Fréchet distance dF** | Compares resulting paths                        |
| **Coarsening ratio ρ**  | Measures coarsening relative to terrain spread  |
| **Per-ring RMSE**       | Measures elevation accuracy by range            |
| **DR / SP / F**         | Additional mapping/planning evaluation measures |
| **Map hash**            | Verifies deterministic output                   |

The coarsening ratio measured on SemanticKITTI sequences 07 and 08 was **1.18–1.84 across rings 1–3**, indicating that coarsening cost remained related to the terrain's own spread rather than introducing arbitrary distortion.

---

# Dataset

VRgrid uses publicly available autonomous-driving data.

### SemanticKITTI

The primary source is **SemanticKITTI**, including:

* LiDAR scans
* Semantic labels
* Moving-object flags

### KITTI

KITTI provides ground-truth poses where available.

Sequences **00 and 08** use the specified SLAM poses because of the particular pose handling required by the evaluation setup.

No model training is required.

The semantic classes and moving-object information are obtained directly from the raw `.label` files.

---

# Preprocessing Pipeline

```text
SemanticKITTI
      │
      ▼
LiDAR Frame
      │
      ▼
Frame Transformation
      │
      ▼
Range Image Projection
      │
      ▼
Deskew
      │
      ▼
Patchwork++ Ground Segmentation
      │
      ▼
Resolution Policy
      │
      ▼
Adaptive 2.5D Grid
```

Patchwork++ is used for ground segmentation, while the mapping pipeline itself remains resolution-agnostic.

---

# Technology Stack

```text
Python 3.11
NumPy
SemanticKITTI
Patchwork++
Rerun
pytest
GitHub Actions
```

### Data representation

VRgrid uses a **Structure-of-Arrays (SoA)** layout and integer/fixed-point atomics where deterministic associative accumulation is required.

---

# Repository Structure

```text
vrgrid/
│
├── docs/
│   ├── master-v4.md
│   ├── sih-math.md
│   ├── eval-metric-specs.md
│   ├── related-work-final-section.md
│   ├── known-limitations.md
│   └── research-log.md
│
├── src/
│   ├── ...
│   └── eval/
│
├── tests/
│   ├── ...
│   └── theorem/
│
├── scripts/
│   └── ...
│
├── README.md
├── LICENSE
└── ...
```

The documentation contains the formal architecture, mathematical definitions, evaluation methodology, related work, known limitations and research findings.

---

# Getting Started

## Requirements

Recommended environment:

```text
Python 3.11+
pytest
```

No GPU is required to run the mapping pipeline or the test suite. A Jetson-class
GPU is the intended deployment target for the port.

## Clone

```bash
git clone https://github.com/Stxtics03/vrgrid-26.git
cd vrgrid-26
```

## Environment

Install the Python dependencies specified by the repository. There are no native or CUDA components to build.

Refer to:

```text
docs/master-v4.md
```

for the complete architecture and build configuration.

---

# Reproducibility

VRgrid is designed so that experiments can be reproduced without relying on hidden state.

The repository includes tests covering:

* Deterministic map generation
* Point partitioning
* Split/merge round trips
* Variance preservation
* Resolution invariants
* Map hashing
* Evaluation metrics

The project's theorem tests are treated as **proof/invariant tests rather than tuning targets**.

---

# Feasibility

VRgrid is entirely software-based.

There is:

* No custom sensing hardware.
* No new LiDAR rig required.
* No field data collection requirement.
* No neural-network training requirement.
* No inference model required for semantic labels.

The system operates on existing LiDAR data and can therefore be integrated with existing perception stacks.

---

# Applications

VRgrid is designed for environments where LiDAR memory and compute budgets matter.

### Defence

* Unmapped-ground perception for UGVs
* Tactical autonomous navigation
* Resource-constrained robotic platforms

### Industrial robotics

* Warehouse AMRs
* Yard automation
* Autonomous material handling

### Autonomous mobility

* Last-mile delivery robots
* ADAS perception systems
* Embedded robotic platforms

These applications can reuse existing LiDAR hardware rather than requiring an additional sensing modality.

---

# Why VRgrid Is Different

Foveated grids, multi-resolution maps, elevation maps and dynamic-object removal are **not new ideas individually**.

VRgrid explicitly builds on prior work rather than claiming otherwise.

The contribution is the composition of:

### 1. Range + semantics

Resolution is driven by both sensor geometry and semantic importance.

### 2. Hard memory envelope

The complete grid is preallocated at startup.

**No per-frame allocation is required to hold the map** — the 8.94 MB footprint
is fixed and no buffer grows with frame count, which is the property the memory
bound rests on and is CI-gated.

This is **not** a claim that the pipeline allocates nothing per frame. The
perception front end allocates ~39.5 MB/frame of transient working memory on real
seq 08 (~59.4 MB whole-frame), none of it retained. See the note under
[Compute](#compute).

### 3. Uncertainty-preserving coarsening

Split/merge operations follow the law of total variance rather than artificially reducing uncertainty.

### 4. Planner-centric evaluation

Mapping quality is evaluated using **planner regret**, not only geometric error.

### 5. Reproducibility

Deterministic accumulation, partition tests and map hashes make reproducibility a first-class requirement.

---

# Limitations

VRgrid deliberately documents its limits.

### Sensor sparsity

Far-range rings contain fewer observations because of LiDAR sampling geometry.

Far-ring metrics are therefore evaluated with respect to frames-since-first-observation rather than assuming a single frame can populate the entire region.

### Odometry

Accumulated maps depend on pose quality.

The evaluation therefore uses the specified KITTI ground-truth/SLAM pose configuration and pins the pose override behaviour in tests.

### No universal planner-regret curve yet

The current synthetic evaluation scene does not provide a sufficiently graded cost field to establish a meaningful memory-vs-regret knee.

The evaluation pipeline exists; a broader real-world dataset is required to establish that curve rather than selecting one artificially.

### Foveated mapping is prior art

VRgrid does not claim to invent foveated or multi-resolution mapping.

The novelty claim is limited to the particular composition of resolution policy, uncertainty-preserving coarsening, hard memory allocation and planner-regret evaluation.

---

# Research & Prior Art

VRgrid builds upon established work in:

* Triebel, Pfaff & Burgard — multi-level surface maps, IROS 2006
* Droeschel, Stückler & Behnke — nested ego-centric multi-resolution grids, ICRA 2014 / JFR 2016
* Losasso & Hoppe — geometry clipmaps, SIGGRAPH 2004
* Hornung et al. — OctoMap, 2013
* Reijgwart et al. — Wavemap, RSS 2023
* Fankhauser et al. — robot-centric elevation mapping with uncertainty, 2014
* Wodtko, Griebel & Buchholz — adaptive patched grid mapping, 2023

Full references and the project's interpretation of related work are maintained in:

```text
docs/related-work-final-section.md
```

The repository intentionally documents borrowed ideas and distinguishes them from the specific VRgrid contribution.

---

# Project Resources

| Resource                     | Description                          |
| ---------------------------- | ------------------------------------ |
| **Repository**               | `github.com/Stxtics03/vrgrid`        |
| **Architecture**             | `docs/master-v4.md`                  |
| **Mathematics & invariants** | `docs/sih-math.md`                   |
| **Evaluation specification** | `docs/eval-metric-specs.md`          |
| **Related work**             | `docs/related-work-final-section.md` |
| **Known limitations**        | `docs/known-limitations.md`          |
| **Research log**             | `docs/research-log.md`               |
| **Dataset**                  | SemanticKITTI / KITTI                |
| **SIH Problem Statement**    | SIH26053                             |

These project resources are explicitly identified in the SIH submission deck.

---

# Team

## Chronicles.exe

**Smart India Hackathon 2026 — SIH26053**

**Problem Statement:**
Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

**Theme:** Smart Vehicles

**Category:** Software

**Organization:** DRDO

---

# Status

**Prototype: Working**

The core VRgrid pipeline, deterministic tests, adaptive representation, evaluation pipeline and Rerun visualization have been developed as part of the SIH 2026 submission.

Current focus areas include:

* Broader real-world planner-regret evaluation
* More extensive sequence-level benchmarking
* Deployment optimisation
* Additional planner integrations
* Further validation of adaptive schedules

---

# License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

# Acknowledgements

VRgrid builds on the open-source robotics, autonomous-driving and mapping research ecosystem, including:

* SemanticKITTI
* KITTI
* Patchwork++
* Rerun
* NumPy
* pytest

We also acknowledge the researchers whose work established the foundations of multi-resolution mapping, elevation mapping, uncertainty-aware spatial representations and dynamic environment perception.

---

<p align="center">

**VRgrid — Foveated 2.5D LiDAR Mapping**

*Less memory. Same near-field detail. Better decisions.*

</p>
