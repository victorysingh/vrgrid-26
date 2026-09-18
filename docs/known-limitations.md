# Known limitations

One issue was found in our own testing this week, root-caused, and fixed — it
is documented below (§1) as a resolved item because a reviewer may still see it
referenced in older notes. One genuinely open item remains (§2), and its status
is stated precisely. Everything else the team considers settled is listed at the
end.

---

## 1. Elevation / ghost-removal — FOUND, ROOT-CAUSED, FIXED

**Status: fixed on 2026-09-01** (`51bff0f`, "gpu/run: make the vertical band
follow the vehicle, not the world datum"; `337bb30` records its per-frame cost).
Independently verified three ways — see the bottom of this section.

### What the bug was

The visibility cleanup (math §10.4) — the stage that erases the trailing
"ghost" cells a moving object leaves behind — stopped clearing cells once the
vehicle's own elevation in the world frame rose past ~11.7 m. Heights entered
the map on a **world-absolute** vertical band of `[−2.0, +6.0] m`
(`quantise_height`, matching `vertical_extent_m` in the schedule config, which
scopes out overpasses and multi-storey structures by design), and the cleanup
was handed cell heights in that same world-absolute frame. On a climbing
sequence the near-field cell heights saturated at the +6 m ceiling while the
sensor sat tens of metres higher, so every cleanup candidate projected outside
the sensor's vertical FOV and was skipped.

*Pre-fix measurement, kept for context (SemanticKITTI seq 08, +45.7 m climb,
grid-wired soak, ghost removal ON):*

| vehicle world-z | cells cleared / frame (**pre-fix**) |
|---|---|
| −6.7 … +2.4 m | 15,580 |
| +2.4 … +11.6 m | 1,022 |
| +11.6 … +20.7 m | 35 |
| +20.7 … +39.0 m | 0 |

Pre-fix, **2,304 of seq 08's 4,071 frames (57 %) were fully inert**, and — a
finding from the re-soak below that the pre-fix notes missed — **seq 07 was
also degraded**: at its −4 to −6 m elevation it cleared only ~42–61 cells/frame
with ~70 % of candidates out of FOV, because −5.8 m saturates the *floor* of
the same world-absolute band.

### The fix

`quantise_height` now takes a `datum_m` the heights are measured from
(defaulting to 0.0 — bit-for-bit the old behaviour). `MapEngine._track_datum`
slides the 8 m band in whole 1 m steps to follow the vehicle's world-z, and
re-bases the stored ground/ceiling heights when it moves — the vertical
counterpart of the toroidal horizontal shift, and rare for the same reason.
`_centres` takes a 2- or 3-vector ego so the cleanup receives the vehicle-frame
z its contract asks for. **The band stays 8 m wide**, so the dense-3D baseline
in `dashboard/_config.py` counts the same voxels and the headline 286× memory
ratio is untouched — which is why moving the band was the right fix and
widening the clamp would have been the wrong one.

### Verification

1. **`tests/test_engine.py`** —
   `test_the_ghost_clears_at_any_vehicle_elevation`, parametrised at 0, −5.8,
   6.0, 12.0 and 39.0 m (seq 07's floor and seq 08's hill); and
   `test_the_band_follows_the_vehicle_rather_than_the_world_datum`.
2. **Aakash re-ran the Gate 3 scene through the real engine, kernels and
   projection** at −5.8, 0, 6, 12 and 39 m — the departed car's ghost now
   clears at every one; every one except 0 m failed on the pre-fix code.
3. **Full-sequence re-soak against the fixed engine** (`scratchpad/soak_elev_postfix_out.txt`):

| veh_z band (m) | frames | cleared/frame (**post-fix**) | protected/frame | prot_frac |
|---|---|---|---|---|
| −6.7 … 2.4 | ~1090 | 18,958 | 12,116 | 0.390 |
| 2.4 … 11.6 | ~848 | 20,524 | 11,499 | 0.359 |
| 11.6 … 20.7 | ~926 | 15,695 | 8,763 | 0.358 |
| 20.7 … 29.9 | ~760 | 17,775 | 10,614 | 0.374 |
| 29.9 … 39.0 | ~433 | 17,054 | 10,287 | 0.376 |

   Seq 08: **0 of 4,071 frames inert** (was 57 %), clearing flat at ~15–20 k
   cells/frame across every elevation band, `prot_frac` flat at ~0.36–0.39
   (was 0.37 → 0.22 → 0.13 → 0.00 → 0.00). Occupied-cell heights now track the
   vehicle — at frame 3000, veh_z 38.4 m, `occ_z` is `[36.0, 44.0] m` (was
   pinned at `[−2.0, 6.0]`). Seq 07 total cleared 373,846 → 15,731,026, all
   bands healthy. 0 NaN in the readout. Datum re-base fires 147 times in
   seq 08's 4,071 frames, median 3.17 ms each, 0.20 ms/frame amortised — the
   frames it fires on sit near 52 ms against the 100 ms budget.

**Consequence for the demo:** the ghost toggle is safe to demonstrate at any
vehicle elevation, including seq 08's full 39 m climb. `docs/demo-safe-ranges.md`
has been updated to drop the elevation-based frame restrictions.

---

## 2. Plan-regret evaluation — the two defects are closed; the scene is the limit

**The offline plan-regret pipeline is implemented and unit-tested** (`src/eval/`,
`tests/test_plan_regret.py`, `test_reference_map.py`, `test_metrics.py`) and has
been **run end-to-end on the synthetic sequence**. Two things stand between that
and a real §8.2 result, and the second is larger than a build step.

### The build path is fixed and works on real data

`reference_map.build()` — the only path from SemanticKITTI to the reference map
M\* — **had never been executed by anything** until this week: it raised
`ValueError: too many values to unpack` on its first line, behind two more
latent frame-convention bugs. Aakash fixed all three and added
`scripts/build_reference_map.py` (`74f555d`), with the whole real path now
exercised against the synthetic writer's KITTI layout so heights can be
asserted rather than eyeballed. `harness.FrameGuard` now checks frame 0 **and**
the first frame ≥ 10 m from the start (a `poses.txt` begins at the identity,
where the right and wrong compositions agree).

Verified on JP's machine: `python scripts/build_reference_map.py 00
--max-frames 5` builds a real M\* (226,485 observed cells, 621,510 returns,
median height −1.55 m), writes and reloads the cache. **Building M\* for 07/08
is one tested command; the ~40 GB SemanticKITTI download is the only thing left
on that path** — and it is not on the shared/CI infrastructure, only on JP's
dev box (`data/README.md` assigns it, execution-plan decision B).

### Both metric-semantics defects are fixed — 2 September

`docs/memo-shrestha-day5-plan-regret-query.md` (`bf03b8c`) named two defects
underneath the regret figure that no start/goal placement fixes. Both are now
closed, and a third was found while closing them.

1. **The fill-rate confound.** `common_support()` restricted on
   `CostMap.unknown` (never-observed, ~0.9% of the window) while `w_unknown`
   was charged on bit 5 (`n < n_min`, ~91.9%), and the diagnostic built to
   expose that read the array that hid it. Aakash root-caused it in `baa44b4`:
   `costmap_from_gridmap` was OR-ing the confidence bit over the sub-cells of a
   planning cell, so **the handicap grew with resolution**. Counts are now
   summed over the footprint's distinct cells and compared against `n_min`
   once. Inside the common support the frozen schedules went from **100.0% to
   0.9%**; uniform 20 cm from 4.2% to 0.0%.

2. **The two lattices.** M\_S set §7.1 bits at ring resolution while M\* set
   them at 25 cm. Both sides are now evaluated at `plan.cell_m` from the same
   summed statistics, and clearance is dropped from both — M\* is 2.5D ground
   and cannot set it. **0 invented walls, 0 missed**, both frozen schedules.
   Separately, §7.1 eq. (22a) now differences bits 1 and 2 over a fixed
   physical baseline rather than one cell, so a 12 cm kerb reads the same at
   5 cm and at 25 cm instead of being a wall on the fine rings only.

3. **Bit 4 was on one side only** — found while tracing the residue.
   `ReferenceMap` carries `class_id`, but `costmap_from_reference` built from
   `block_stats`, heights alone, so M\* charged 0 class penalties against the
   schedules' 18. Both paths are scored on M\*, so a schedule paid pure regret
   for routing around ground it had correctly labelled non-drivable. Symmetric
   now, via `ReferenceMap.block_class()`.

### What remains is the scene, not the metric

After all three, the frozen schedules report **R(S) = 0.207** and every uniform
baseline **0.000** on the synthetic sequence. That 0.207 is not a knee and must
not be drawn as one: it is

```
2 · (√2 − 1) · plan.cell_m  =  0.2071
```

— two diagonal steps, i.e. a path that jogs one 25 cm cell sideways and back.
**It is the smallest non-zero value the planning lattice can express.** Traced
to its cause it is a single cell: column 16 of the window holds one cell with
bit 5 set and column 17 holds none, so the fine schedules sidestep for the
length of the corridor. The uniform maps pool more observations per cell,
nothing falls under `n_min`, and they go straight.

So Shrestha's original reading survives the fixes: **the synthetic scene cannot
draw a knee.** M\* over the planning window has one passable cost value — 1,924
cells at 1.00× — and a graded curve needs a graded cost field. The real §8.2
plot needs sequence 08, which is now on disk. See
`docs/decisions-2026-09-02.md` for the query-design question this leaves open.

### The money plot on real data — a curve now, and what it does and does not show

`regret_plot.py --seq` exists (it had no `--seq`), R(S) is averaged over 64
seeded planning queries instead of one, and `common_support` equalises evidence
rather than only coverage. Sequence 08, four window lengths:

| schedule | MB | @20 | @40 | @80 | @160 |
|---|---|---|---|---|---|
| 5_10_20_40 | 29.06 | 0.488 | 0.171 | 0.714 | 0.758 |
| 5_10_50 | 23.62 | 0.488 | 0.171 | 0.714 | 0.758 |
| uniform 10 cm | 18.19 | 0.231 | 0.084 | 0.759 | 0.597 |
| uniform 20 cm | 10.71 | 0.251 | 0.104 | 0.827 | 0.791 |
| uniform 40 cm | 7.82 | 0.402 | 0.182 | 0.918 | 0.838 |
| uniform 80 cm | 7.09 | 0.798 | 0.130 | 1.165 | 0.819 |

**What holds.** The uniform series rises with cell size at every window —
strictly at 20 and 80 frames, and at 40 and 160 except for the 80 cm point
dipping below 40 cm. Fréchet distance tracks it. Before averaging, the same
runs gave multiples of the 0.207 lattice quantum and an ordering that inverted
with the frame count; that is gone.

**What does not.** The magnitude still moves with the window (5_10_20_40 reads
0.171 at 40 frames and 0.758 at 160), so **R(S) is comparable across schedules
at a fixed window and not across windows.** Any quoted number must state its
frame count.

**And the frozen schedules are not winning.** 5_10_20_40 scores worse than
uniform 10 cm at three of four windows. Before reading that as a result, see
the extent mismatch below — it is very likely an artifact of the x-axis.

### The money plot's memory axis — FIXED, extents now match

The uniform baselines were built at `half_width_m=24.0` against frozen
schedules reaching 100 m, so the figure's memory axis compared a map of
0.0400 km² with one of 0.0023 km² — a seventeenth of the ground — and drew them
as comparable points. `regret_plot.py` now matches the uniforms to the frozen
schedules' reach by default; `--uniform-half-width 24.0` reproduces the old
figure.

### What the money plot now says — seq 08, 20 frames, 64-query mean

| schedule | MB | cells | R(S) | Fréchet |
|---|---|---|---|---|
| uniform 10 cm | **78.50** | 4,000,000 | 0.231 | 0.18 m |
| uniform 20 cm | **30.50** | 1,000,000 | 0.251 | 0.23 m |
| **5/10/20/40** | **29.06** | 745,000 | 0.488 | 0.33 m |
| 5/10/50 | 23.62 | 520,000 | 0.488 | 0.33 m |
| uniform 40 cm | 18.50 | 250,000 | 0.402 | 0.32 m |
| uniform 80 cm | 11.04 | 62,500 | 0.798 | 0.55 m |

**The memory claim is now stated correctly and it is strong.** Matched to the
same ground, uniform 10 cm costs **78.50 MB against 29.06** — a 2.7× saving
that the old figure inverted into "we are more expensive than an 18.19 MB
baseline".

**The regret claim does not survive this query.** `uniform_20cm` at 30.50 MB —
essentially our memory — scores **0.251 against our 0.488**, and `uniform_40cm`
at 18.50 MB scores **0.402**: cheaper *and* better. The script's monotonicity
guard fires on that step. §8.2 claims foveation is free in decision terms; on
the only planning query that exists, it is not.

**What that is and is not evidence of.** One sequence, one window, one query
family — a single longitudinal lane down the middle of the window, unchanged
since Day 0 and never designed to discriminate between resolutions
(`docs/decisions-2026-09-02.md`, Decision 4). A lane query rewards a map that
is uniformly adequate along one line and cannot reward one that is sharp where
the vehicle is looking. It is not proof the thesis is wrong; it is proof this
query cannot demonstrate it.

### Scope

The memory reduction and the per-ring geometric accuracy against M\* are the
load-bearing quantitative claims and do not depend on the regret figure. The
regret result — "the compression does not change the plan a robot would make" —
is the project's strongest single claim, and its status is stated plainly here
so a reviewer knows exactly what has and has not run.

*(Current as of `origin/main` `38edfb5`, 2 Sep. No M\* / regret artifacts are
committed; `.gitignore` excludes them by design.)*

---

## 2b. Accuracy across ALL eleven labelled sequences — the headline result

### Regenerated 2026-09-17 — this table supersedes the one below it

`python scripts/eval_synthetic.py --seq <00..10> --frames 40`, schedule
5/10/20/40, per-sequence pose source, Patchwork++. Three changes since the
2 Sep table, all in the scored numbers and none in the map's memory:
§7's scoring fix (score a ring only where it still answers), §8's per-block
ring rule, and **`spread` now includes the within-cell variance**, which is
what gives ring 0 a ρ at all (the deferred fix described below, now done).

| seq | r0 RMSE | **r0 ρ** | r1 RMSE | r1 ρ | r1 ρ cons. | r2 RMSE | r2 ρ | r3 RMSE | r3 ρ |
|---|---|---|---|---|---|---|---|---|---|
| 00 | 2.70 | 1.24 | 6.42 | 1.26 | 1.39 | 33.53 | 2.20 | 9.44 | 1.26 |
| 01 | 1.59 | 1.23 | 2.09 | 1.30 | 1.53 | 3.71 | 1.32 | 8.80 | 1.28 |
| 02 | 0.84 | 1.13 | 2.29 | 1.22 | 1.36 | 13.26 | 1.28 | 14.80 | 1.43 |
| 03 | 5.26 | 1.29 | 12.60 | 1.30 | 1.41 | 15.92 | 1.38 | 6.88 | 1.15 |
| 04 | 0.91 | 1.14 | 3.95 | 1.20 | 1.25 | 12.21 | 1.25 | 18.24 | 1.24 |
| 05 | 1.26 | 1.17 | 3.39 | 1.33 | 1.48 | 9.62 | 1.36 | 12.22 | 1.21 |
| 06 | 1.25 | 1.18 | 3.07 | 1.25 | 1.39 | 8.82 | 1.29 | 12.52 | 1.14 |
| 07 | 1.87 | 1.27 | 2.82 | 1.15 | 1.24 | 5.88 | 1.14 | 13.57 | 1.27 |
| 08 | 1.17 | 1.15 | 2.31 | 1.16 | 1.22 | 3.79 | 1.08 | 3.88 | 1.06 |
| 09 | 1.83 | 1.16 | 3.27 | 1.25 | 1.40 | 4.75 | 1.25 | 10.38 | 1.42 |
| 10 | 1.60 | 1.16 | 2.71 | 1.11 | 1.16 | 4.22 | 1.17 | 2.77 | 1.15 |

```
         rho, incl. within-cell      rho, between-cell only (conservative)
ring 0:  1.17 [1.13-1.29]            -- (not computable)
ring 1:  1.25 [1.11-1.33]            1.39 [1.16-1.53]
ring 2:  1.28 [1.08-2.20]            1.30 [1.09-2.38]
ring 3:  1.24 [1.06-1.43]            1.25 [1.06-1.44]
RMSE medians (cm): r0 1.59 [0.84-5.26]  r1 3.07 [2.09-12.60]  r2 8.82 [3.71-33.53]  r3 10.38 [2.77-18.24]
```

**What to quote.** Ring 0 now has a ρ, and it is the best of any ring, which is
what §9.3 predicts for the finest cell. But the within-cell variance is sensor
noise and pose jitter as much as terrain, so it moves every ρ toward 1. **Quote
both**: "ρ = 1.39 at ring 1 (1.16–1.53, n = 11) on the between-cell spread —
the conservative definition the 2 Sep slides used — and 1.25 (1.11–1.33) with
the within-cell term, which also gives ring 0 a ρ of 1.17 (1.13–1.29)." Median
ring-1 RMSE is 3.07 cm either way. Both print from one run of
`eval_synthetic.py` (the `conservative` line under each table), and include
§8's ring rule, §11's band fix and §11's band rebalance.

Seq 00's ring-2 outlier (2.20) is examined in §9; ring 3 on 08, 09 and 10 was the band defect of §11.

### As published 2 Sep — superseded

Everything in this project was measured on 07 and 08 until 2 Sep, and the
honest reason for those two is that they downloaded first. All eleven labelled
sequences, 40 frames each, schedule 5/10/20/40, with the per-sequence pose
source and Patchwork++:

| seq | r0 RMSE | r1 RMSE | r1 ρ | r2 RMSE | r2 ρ |
|---|---|---|---|---|---|
| 00 | 2.73 | 6.54 | 1.52 | 28.09 | 2.32 |
| 01 | 1.59 | 2.28 | 1.59 | 4.39 | 1.46 |
| 02 | 1.34 | 7.15 | 1.45 | 15.53 | 1.36 |
| 03 | 5.25 | 12.16 | 1.48 | 24.26 | 1.76 |
| 04 | 0.91 | 3.65 | 1.26 | 11.50 | 1.31 |
| 05 | 1.26 | 3.53 | 1.55 | 11.17 | 1.50 |
| 06 | 4.18 | 3.02 | 1.43 | 9.48 | 1.52 |
| 07 | 1.76 | 3.48 | 1.32 | 6.13 | 1.18 |
| 08 | 1.16 | 2.55 | 1.30 | 6.46 | 1.22 |
| 09 | 1.85 | 3.67 | 1.50 | 6.37 | 1.43 |
| 10 | 1.61 | 3.51 | 1.29 | 8.57 | 1.56 |

```
ring 1:  rho median 1.45 [1.26-1.59]     RMSE median  3.53 cm [2.28-12.16]
ring 2:  rho median 1.46 [1.18-2.32]     RMSE median  9.48 cm [4.39-28.09]
```

**Lead with ρ, not RMSE.** At ring 1, ρ spans **1.26×** across the whole
dataset while RMSE spans **5.3×**. That is §9.3's decomposition doing exactly
what it is for: RMSE tracks how rough each road happens to be, ρ divides that
out and leaves what the coarsening cost. "ρ = 1.45, range 1.26–1.59, n = 11" is
both more defensible and closer to the actual claim than any single sequence's
RMSE.

⚑ **07 and 08 are at the good end, not typical.** Their ring-1 ρ of 1.32 and
  1.30 sit near the bottom of the range against a median of 1.45. Anyone who
  checks a third sequence gets a slightly worse number than the one we quoted
  first, so quote the distribution.

⚑ **Sequence 00 is the only ρ outlier** at 2.32 on ring 2, against 1.18–1.76
  everywhere else. Its systematic bias was a pose artifact and is fixed (see
  §6), but that only moved ρ from 2.40 to 2.32 — the rest is dispersion
  (spread 14.41 cm at ring 2) and is **unexplained**. 00 is a long urban loop
  and ring 2 spans 20–50 m where ground segmentation is hardest; that is a
  hypothesis, not a finding.

⚑ **Ring 0 has no ρ on any sequence.** The finest ring — the one the foveation
  argument is actually about — has RMSE (0.91–5.25 cm) and no ρ anywhere. It is
  excluded by the arithmetic of the metric, **not** by any shortage of data.
  Diagnosed 3 Sep; see immediately below.

### Ring 0 has no ρ — diagnosed, deferred, and not a sensor limitation

**This document previously said** that `coarsening_ratio_per_ring` "excludes
footprints holding a single reference return, and at 5 cm essentially every
footprint holds one." **That is wrong on both halves**, and it mattered because
it read as a permanent physical limit when it is a fixable one.

**What the guard actually tests.** `ReferenceMap._tables()` builds its
summed-area tables over `obs = self.observed`, a *boolean*, so `block_stats`
returns `n` = the number of **observed 5 cm cells** in a footprint, not the
number of returns. A ring-0 footprint is `k = 1` — exactly one cell — so
`n_ref` can be 0 or 1 and **nothing else**. `metrics.coarsening_ratio_per_ring`
then drops everything failing `n_ref > 1`. Ring 0 is therefore excluded **100%
of the time, on every sequence, by arithmetic** — not by evidence. Measured:
`max n_ref` is exactly 1 for ring 0 and exactly k² for rings 1–3, on 00, 07
and 08 alike.

**The redundant data exists and is being thrown away.** M\* is not a
one-point-per-cell map: `_Builder.finish` accumulates every static ground
return across every frame (`np.add.at(h_sum, …)`, `np.add.at(count, …)`) and
`count` is a true return count, persisted through `save`/`load`. What it
discards is the **sum of squares**, so the within-cell dispersion is computed
at build time and never stored. Of the ring-0 cells §9.2 actually scores:

| seq | ring-0 cells scored | M\* holds **>1 return** | median returns | `n_ref > 1` |
|---|---|---|---|---|
| 00 | 83,947 | **77,943 (92.8%)** | 6 | 0 |
| 07 | 105,366 | **102,207 (97.0%)** | 12 | 0 |
| 08 | 139,709 | **136,035 (97.4%)** | 10 | 0 |

**Nor is the sensor sparse at 5 cm.** In a *single* sweep, inside 10 m, the
HDL-64E puts more than one return into **56.8–65.5%** of the 5 cm columns it
touches (2.0–3.2 returns per column on average, up to 150). Multi-frame
accumulation is not needed to create redundancy — it is already there in one
frame, and M\* already sums it.

**What closing it would be worth.** Scoring ring 0 with the within-cell
variance as `spread²` — the law of total variance the project already mandates
for merge (§4.2, and a CLAUDE.md invariant) — gives:

| seq | usable cells | RMSE(bias) | spread | IL | **ρ ring 0** |
|---|---|---|---|---|---|
| 00 | 77,943 (92.8%) | 2.83 cm | 3.91 cm | 4.82 cm | **1.235** |
| 07 | 102,207 (97.0%) | 1.79 cm | 10.74 cm | 10.89 cm | **1.014** |
| 08 | 136,035 (97.4%) | 1.18 cm | 2.05 cm | 2.36 cm | **1.153** |

**ρ 1.01–1.24 would be the best of any ring on all three sequences** — the
finest ring paying least for its coarsening, which is precisely what §9.3
predicts and what the foveation argument needs.

**Why it is deferred rather than done.** `block_stats` is shared: adding the
within-cell term moves rings 1–3 as well, and moves them *downward*.

| seq | ring | ρ now | ρ after | move |
|---|---|---|---|---|
| 07 | 1 | 1.323 | 1.039 | **−21.5%** |
| 07 | 2 | 1.181 | 1.153 | −2.3% |
| 07 | 3 | 1.248 | 1.239 | −0.8% |
| 08 | 1 | 1.304 | 1.207 | **−7.5%** |
| 08 | 2 | 1.221 | 1.197 | −1.9% |
| 08 | 3 | 1.838 | 1.810 | −1.5% |

So it is roughly fifteen lines of code with a two-day tail: a new field on
`ReferenceMap` invalidates **every cached M\* `.npz`**, the §2b eleven-sequence
table and the headline "ρ median 1.45 [1.26–1.59], n = 11" both need
regenerating, and `block_stats` also feeds
`plan_regret.costmap_from_reference`, the §8.2 path stabilised only on 2 Sep.
Every ρ moves in the direction that flatters us, which is the worst direction
in which to ship a headline-metric change two days before submission.

**Status: DONE 2026-09-17.** `ReferenceMap` stores the within-cell variance,
`block_stats` adds it by the law of total variance, and the ρ guard counts
returns rather than cells. Ring 0 ρ across all eleven: **1.17 [1.13–1.29]** —
the prediction above held. Every cached M\* `.npz` is refused by `load()` until
rebuilt. The roughness bit in `plan_regret.costmap_from_reference` keeps the
between-cell variance only; including the within-cell term there took seq 08's
R(S) from 0.127 to 2.497 because the map side has no such term. The earlier
status follows for the record: **a diagnosed, deferred fix with a measured
cost and a known likely benefit — recommended as a Day-7 post-submission
item, not a permanent limitation.** Until it is done, the honest statement is that ring 0 is reported
on RMSE alone and carries no coarsening ratio, and that this is the metric's
construction rather than the sensor's reach.

*(A narrower variant — applying the within-cell term only at k = 1 — leaves
rings 1–3 untouched but makes ring 0's `spread` definitionally different from
theirs, which destroys the one thing a per-ring table exists to do. Worse, not
better.)*

## 3. Curb and pothole detection — real numbers, no ground truth to score against

`src/grid/features.py` answers the problem statement's own sentence about
curbs and potholes directly (§7.4). Measured on **all eleven labelled sequences**, 40 frames, schedule 5/10/20/40,
through the real loader → transforms → Patchwork++ → `run_sequence` →
`features.detect` path. Curb median by ring:

| seq | r0 | r1 | r2 | r3 | curb cells | pothole cells |
|---|---|---|---|---|---|---|
| 00 | 8.2 | 8.1 | 9.9 | 13.1 | 6,964 | 180 |
| 01 | 8.2 | 8.2 | 8.9 | 15.0 | 14,581 | 192 |
| 02 | 9.1 | 8.1 | 12.0 | 12.0 | 9,055 | 292 |
| 03 | 8.1 | 10.8 | 10.0 | 18.1 | 7,783 | 345 |
| 04 | 8.1 | 7.6 | 10.0 | 9.9 | 5,827 | 404 |
| 05 | 9.1 | 8.5 | 9.5 | 9.0 | 9,414 | 132 |
| 06 | 8.1 | 8.2 | 10.3 | 11.8 | 5,388 | 107 |
| 07 | 8.5 | 9.2 | 9.1 | — | 4,041 | 257 |
| 08 | 8.6 | 8.1 | 8.5 | 10.4 | 9,499 | 166 |
| 09 | 8.5 | 8.9 | 9.1 | 11.5 | 9,140 | 56 |
| 10 | 8.2 | 9.0 | 9.8 | 14.8 | 23,527 | 551 |

**Ring 0 returns 8.1–9.1 cm on every one of eleven sequences** — different
recording dates, different calibrations, a one-centimetre band. That
consistency is the evidence the detector measures a physical feature rather
than an artifact, and it is a stronger claim than any single number.

The rise with ring is systematic and physical — 8–9 cm at 5 cm cells, 9–12 at
20 cm, 9–18 at 40 cm — because a coarser cell straddles the kerb face and
averages in sloped ground. Report per ring; ring 3's spread (9.0–18.1) is where
it stops being reliable.

⚑ **"Reads 1–2 cm low" — WITHDRAWN, it was an assumption not a measurement.**
  I attributed the 8–9 cm reading to `curb.baseline_m` of 0.20 m sampling
  partway up the kerb face. Tested by sweeping the baseline, ring-0 median on
  three sequences:

  | baseline | 07 | 08 | 05 |
  |---|---|---|---|
  | 0.20 m | 8.5 | 8.6 | 9.1 |
  | 0.30 m | **9.1** | **9.8** | **9.5** |
  | 0.40 m | 9.1 | 9.0 | 10.3 |
  | 0.50 m | 9.1 | 7.6 | 11.0 |
  | 0.60 m | 8.5 | 7.1 | 11.0 |

  A longer baseline does **not** systematically recover height — past 0.30 m
  the sequences diverge, 05 rising and 08 falling, while cell counts grow
  throughout. So a longer baseline admits more and different features rather
  than measuring the same one better, and the kerbs in this data genuinely
  measure ~9 cm. Karlsruhe kerbs including dropped crossings at 9 cm is
  entirely ordinary.

  I then moved `curb.baseline_m` 0.20 → 0.30 on that three-sequence evidence
  and **reverted it** after running all eleven. It raises the ring-0 median
  0.7 cm and nearly triples the cross-sequence spread:

  | baseline | median | range | sd |
  |---|---|---|---|
  | **0.20 m** | 8.2 cm | **8.1–9.1** | **0.36** |
  | 0.30 m | 8.9 cm | 7.3–10.2 | 0.89 |

  The tight band *is* the claim — eleven independent recordings, different
  dates and calibrations, agreeing to 1.0 cm. Trading it for 0.7 cm of median
  trades the result for the number. Three sequences were not enough to choose
  this; eleven were.

  ⚑ It also only ever moved ring 0. `baseline_k` is an integer and
  `round(0.30/(2×0.10))` is `round(1.4999999999999998)` = 1, so at 10 cm cells
  and coarser the span stays `2×cell_m` whatever the config says. **Below
  `baseline_m`/2 the knob does nothing** — worth knowing before anyone tunes
  it.

⚑ **Potholes are a demonstration, not a claim.** 56 to 551 cells per sequence
  is a 10× spread with no pattern, and 00's ring 0 reports five cells at
  48.5 cm. The detector fires occasionally and correctly; there is no rate to
  quote.

**The limitation: SemanticKITTI has no ground truth for curb or pothole
geometry.** There is no detection rate to quote, only counts and a plausibility
check on the height distribution. The `road`/`sidewalk` label boundary is the
only cross-check available, and it locates curbs without measuring them. Quote
these as counts with that caveat attached, never as accuracy.

Two further honest notes. The median rises with cell size (9.1 → 11.3 →
14.2 cm) because a coarser cell averages across the kerb face and admits more
sloped ground — so it is reported per ring rather than pooled. And potholes are
bounded above at 50 cm (`pothole.max_depth_m`): ring 2 first reported 156
detections at a median 71.5 cm and a p90 of 200 cm, which are ditches, kerb-line
drop-offs and the space under parked cars. Those are real hazards and they need
a separate detector; this one does not claim them.

## 4. Per-cell confidence is a margin, not a probability

`src/grid/confidence.py` (§7.5) reports how far to trust each drivability
verdict, on four derived channels — nothing is stored, `CELL_BYTES` stays 12.

**It is not calibrated.** Nothing has been fitted against outcomes, so 0.6 is
not a 60% chance of anything; each channel is a margin with a stated meaning
and they are combined by taking the weakest. The `label` channel is
additionally a **floor, not an estimate**: the Boyer-Moore counter saturates at
7, so a cell observed 200 times unanimously reports a *lower* share than one
observed 8 times. `saturated()` flags that regime.

On sequence 08 the binding channel is **geometry** for rings 0–2, where the
synthetic scene binds on `label` and `evidence` — real terrain sits near the
slope and step thresholds and the analytic scene does not. That is the clearest
single argument in the project for not reporting synthetic numbers.

## 5. The visibility candidate cap moves a memory figure

`visibility.max_candidate_cells` is now `null`, meaning the grid's own slot
count — 910,000 at 5/10/20/40, **58.24 MB** of scratch, up from 9.60 MB at the
retired placeholder of 150,000. That placeholder dropped 52.3% of sequence 07's
peak occupied set and 67.1% of 08's, untested and in silence.

This is **working memory, not map memory**, so the report's cell-count ratios
are unaffected, and `with_visibility` is off by default so the 29.06 MB headline
does not move unless the cleanup's scratch is switched on. If a smaller declared
total is wanted, an explicit `600000` (38.40 MB) covers every measured sequence
at 1.32× the observed max.

Measured peaks: **314,442** (07, 1,101 frames), **455,714** (08, 4,071),
**278,226** (00, 4,541). The peak does **not** scale with sequence length — 00
is the longest and the lowest, which refuted the first version of this argument.
It tracks scene density instead, varies 1.64× across three sequences with no
available predictor, and 19 sequences remain unmeasured. That unpredictability,
not growth, is the case for the structural bound. See
`docs/decisions-2026-09-02.md`, Decision 2.

---

## 6. The eval harness had no height datum — FOUND, FIXED for 07, OPEN for 08

### What the bug was

`kernels.quantise_height` clips to an **8 m band, world-absolute at datum 0**:
[−2.00, +6.00] m. `MapEngine` tracks a moving datum and adds it back on
readout. **`harness.run_sequence` had none**, so every height it stored was
world-absolute.

Sequence 07's ground sits at world z ≈ −1.61 m. Anything more than 39 cm below
it — ditches, kerbside drops, the low side of the road camber — clipped
*upward* to −2.00 m. Measured: **69,470 of 2,158,949 ground returns, 3.2%**,
all clamped in the same direction.

That produced a positive height bias, concentrated in cells whose returns
spread lowest, which is disproportionately the far field. It looked exactly
like a range-dependent measurement error.

### The fix, and what it is worth

`run_sequence` now sets `gm.z_datum_m` from the first pose's elevation and
stores heights relative to it; `metrics._compared` adds it back before
comparing against the world-absolute M\*. **One** datum for the run, not a
moving one, deliberately: a constant offset cancels in every DIFFERENCE the map
computes — slope, step, curb height, pothole depth — so §7.1 and §7.4 are
untouched. A moving datum would not cancel and would put a spurious step
between any two cells last seen at different times.

Sequence 07, 40 frames, clipping falls from 3.2% to **1 return in 2.16 million**:

| ring | cell | cells | mean bias | sd | before |
|---|---|---|---|---|---|
| 0 | 5 cm | 102,988 | **−0.33 cm** | 2.17 | +1.24 |
| 1 | 10 cm | 54,320 | **−0.17 cm** | 3.23 | +3.98 |
| 2 | 20 cm | 11,231 | **−0.41 cm** | 5.67 | +23.46 |
| 3 | 40 cm | 47 | −18.52 cm | 94.08 | noise, 47 cells |

RMSE 22.02 → **3.23 cm** at ring 1; ρ 19.21 → **2.95**. Sub-centimetre
systematic bias at every ring that carries cells, with dispersion growing
sensibly with cell size. **This is the accuracy claim, and it is a good one.**

### ⚑ Corrections to what this document previously said

Two earlier conclusions here were wrong and are withdrawn.

- **"The bias is ring migration, structural, not a fusion bug."** It is not.
  That rested on a controlled experiment — one-ring uniform 20 cm reading
  +7.83 cm against the four-ring schedule's ring 2 at +23.46 cm — and that
  experiment was itself confounded by the clipping, which hit the two
  schedules differently. Re-run with the datum they agree: **−0.26 cm and
  −0.41 cm**. Migration costs nothing measurable.
- **"Grazing incidence and inverse-variance weighting are ruled out."** Those
  remain correctly ruled out, but for a better reason than given: neither could
  have mattered, because the clipping happens in `quantise_height` *before* any
  weight is applied.

### The moving datum, and where 08 actually fails

`gpu.shift.track_datum` now holds the one implementation and both callers use
it — `MapEngine._track_datum` delegates, and `run_sequence` calls it per frame.
The band slides in whole 1 m steps and **re-bases every stored height** as it
moves, so all cells stay relative to the same current datum and every
difference the map computes is unaffected.

Two metric defects surfaced while proving it, and both are fixed:

1. **`obs_count > 0` was not the right predicate.** It counts every return,
   and a cell whose returns were all NON-ground has no measured ground height
   — `fuse` leaves it at its initial 0, and 0 cm is not a neutral height, it is
   *the datum*. So the metric's answer moved with the datum: on seq 07, shifting
   it from −1.64 m to −2.00 m took ring 1's RMSE from 3.19 to 6.15 cm without
   changing a single measurement. `height_variance > 0` is the predicate — the
   codec maps code 0 to maximum variance exactly so "never fused" is
   distinguishable. The metric is now datum-independent, verified by A/B.
2. **Band-saturated cells were being scored.** A cell clamped at the band edge
   holds the edge, not a measurement. Excluded, and reported by
   `saturated_fraction_per_ring` so a ring that loses most of itself says so.

**Sequence 07 after all of it** — 40 frames, 5/10/20/40:

| ring | cell | cells | RMSE | mean bias | spread | rho |
|---|---|---|---|---|---|---|
| 0 | 5 cm | — | **1.16 cm** | — | — | — |
| 1 | 10 cm | 47,059 | **1.69 cm** | −0.08 | 1.07 | **1.95** |
| 2 | 20 cm | 8,323 | **1.02 cm** | +0.14 | 2.40 | **1.11** |

ρ near 1 is the thesis stated numerically: the coarsening cost only what the
terrain's own sub-cell variability costs.

### Sequence 08 — RESOLVED: it was the pose file

08's official KITTI ground-truth poses put the same patch of road **16.6 cm**
apart from one frame to the next, consistently (16.1–17.6 cm across every pair,
so a systematic offset rather than drift). A cell seen over N frames
accumulated about N × 16.6 cm, which made M\* itself carry a **64.5 cm median
standard deviation inside a 10 cm footprint** and put 08's per-ring RMSE at
162 cm.

Median absolute ground-height disagreement between consecutive frames, in
20 cm cells both frames saw:

| sequence | official GT poses | SemanticKITTI SLAM poses |
|---|---|---|
| 07 | **0.49 cm** | 0.66 cm |
| 08 | 16.63 cm | **1.04 cm** |

The two pose files are not interchangeable. KITTI's GT is a GPS/IMU solution
optimised for **trajectory** evaluation; SemanticKITTI computed its own SLAM
poses so that scans **register into a consistent map**. `README.md:21` chose GT
on Day 0 — right for most sequences, wrong for 08.

`loader.pose_source()` now decides per sequence: 08 reads SLAM, everything else
GT, and `VRGRID_POSE_SOURCE=gt|slam` forces one globally so the table above
stays reproducible.

**Checked across every labelled sequence**, same measure, 2 Sep:

| seq | GT | SLAM | | seq | GT | SLAM |
|---|---|---|---|---|---|---|
| 00 | 2.27 | 1.05 | | 06 | 1.32 | 1.23 |
| 01 | 1.43 | 1.38 | | 07 | **0.47** | 0.64 |
| 02 | 1.20 | 1.20 | | 08 | **16.53** | **1.04** |
| 03 | 1.97 | 1.24 | | 09 | 1.21 | 1.26 |
| 04 | 1.25 | 1.11 | | 10 | 1.19 | 1.20 |
| 05 | 1.02 | 1.00 | | | | |

**08 is the only pathological sequence on that measure.** But per-frame
agreement turned out to be a **weak predictor**, and choosing the override list
from it alone was wrong. What matters is the bias that ACCUMULATES, measured
per ring against M\*:

| seq | per-frame | mean_b r1 | mean_b r2 | mean_b r3 | |
|---|---|---|---|---|---|
| 00 | 2.27 cm | −2.86 | −9.85 | **−13.95** | → SLAM |
| 06 | 1.32 cm | −0.34 | −3.32 | −5.80 | wash, stays GT |
| 03 | 1.97 cm | −1.91 | +2.45 | −0.80 | fine |
| others | 1.0–1.4 cm | < \|0.8\| | < \|2.3\| | < \|2.9\| | fine |

Sequence 00 disagrees by only 2.27 cm per frame yet accumulates **−13.95 cm**
by ring 3, while 03 at a comparable 1.97 cm/frame accumulates −0.80. Switching
00 to SLAM takes ring 2's bias from −9.85 to **−0.44 cm**, ring 3's from −13.95
to −1.18, and ring 3 RMSE from 26.03 to **13.57**.

Sequence 06, the next worst accumulator, was tested the same way and is a
**wash** — GT better at ring 1, SLAM marginally better at rings 2–3 — so it
stays on GT. Only sequences with a measured win are overridden.

So the list is `{"00": "slam", "08": "slam"}`, and
`test_only_08_needs_the_slam_poses` pins it so it cannot quietly widen.

**Ruled out along the way, each by measurement:** the height datum (07 is clean
on the same code), band saturation, the ground mask, frame alignment (08 is
4,071/4,071/4,071), the calibration (07 and 08 have *byte-identical* `Tr`), and
`real_scans`'s own composition (bit-identical to `frames.md`'s textbook
`sensor_to_world` chain, 0.49 / 16.63 either way).

### Both sequences, 40 frames, 5/10/20/40, with Patchwork++ and the right poses

| | ring 0 (5 cm) | ring 1 (10 cm) | ring 2 (20 cm) | ring 3 (40 cm) |
|---|---|---|---|---|
| **07** RMSE | 1.76 cm | 3.48 cm | 6.13 cm | 16.34 cm |
| **07** rho | — | 1.32 | 1.18 | 1.25 |
| **08** RMSE | 1.16 cm | 2.55 cm | 6.46 cm | 52.12 cm |
| **08** rho | — | 1.30 | 1.22 | 1.84 |

**ρ between 1.18 and 1.84 on both sequences** is the thesis stated numerically:
the coarsening cost only what the terrain's own sub-cell variability costs.
Mean bias is under 2.4 mm everywhere except 08's ring 3.

⚑ 07's figures moved from an earlier 1.69 cm at ring 1 because **Patchwork++
  replaced the semantic-class ground fallback**, not because of the pose
  change — 07 still reads GT poses. The geometric segmenter admits genuine
  terrain the class mask missed, so `spread` rises (1.07 → 3.95 cm at ring 1)
  and more cells are scored (47,059 → 51,975; ring 3 goes from 0 cells to 949).
  RMSE rises and ρ *falls*, which is the more honest reading: the earlier
  number was over a narrower, flatter subset.

### Patchwork++ is now installed

`pip install pypatchworkpp` fails at every published version: the sdist's
`python/CMakeLists.txt` falls into an out-of-tree branch that fetches
`.../refs/tags/v${CMAKE_PROJECT_VERSION}.tar.gz`, the variable is empty under
scikit-build-core, and GitHub returns 404 for `tags/v.tar.gz`. Building from a
git clone takes the `if(EXISTS ../cpp/)` branch instead and works:

```
git clone --depth 1 https://github.com/url-kaist/patchwork-plusplus.git
.venv/bin/pip install ./patchwork-plusplus/python
```

`ground.segment_ground` is now the geometric segmenter rather than the
`ground_from_semantics` fallback, on this machine.

## 7. §9.2 scored each ring over its whole square window — FIXED, and it is not §6's bias

**What was wrong.** §9.2 eq. (26) does not say what `C_L` is, and
`metrics._ring_cells` read it as every cell in ring L's buffer. That buffer is
a square of half-width `R_L`, so it covers the region the finer rings serve,
while only its annulus `[R_{L-1}, R_L)` is ever written. No cell moves between
rings — what moves is the vehicle, and with it which ring answers for a place.
The interior is never cleared (a toroidal shift clears only the edge coming
into view, §2.4) and never read (`query()` routes those places to a finer
ring), so it holds values written when that ground was far away.

**Fixed.** `C_L` is now the cells ring L still *serves*, decided by `ring_of`
on the cell centre — the same function `query()` routes with, pinned against
`slot_of` rather than asserted. It lives in `_ring_cells`, so all §9.2/§9.3
metrics inherit it.

**⚑ This is NOT the range bias, and §6 is right to have withdrawn that.** §6's
controlled experiment — one-ring uniform 20 cm against the four-ring
schedule's ring 2, re-run with the datum fixed — gives **−0.26 cm and −0.41
cm**. A single-ring schedule structurally cannot carry this defect, so that
comparison is the direct test of it, and it says migration costs nothing
measurable in height bias. An earlier version of this section claimed the
confound distorted §8.2's money plot across schedules; **that claim is
withdrawn**. It rested on synthetic runs made before §6's datum fix, and §6's
real-data experiment is the better evidence.

**What it does change, and §6 does not cover.** The defect is in *which cells
are scored*, so it lands hardest on the population metrics rather than on
height. On the 12-frame synthetic sequence, ring 3's fill rate read **0.23**
against **0.83** over the cells it actually answers for — the §1.3 ring-sweep
claim understated by a factor of three and a half, because the unwritten
interior was counted as unfilled. IoU moves the same way. Those numbers are
synthetic and want re-running on real sequences.

**⚑ The sign question is answered — mechanistically, without real data.** Two
measurements of this fix on 07/08 disagreed, first "understated 3–12%", then
"no consistent bias, −40% to +21%". Neither was reproducible here, so the
mechanism was tested directly instead: three synthetic scenes, identical but
for where a roughness contrast sits relative to ring 2's inner boundary. Both
halves always rough, only the contrast moved. Ring 2, 60,000 returns/frame,
stable across 10/16/24/32 frames:

| scene | RMSE before | after | change |
|---|---|---|---|
| **rough-near** — stale interior is the rough half | 0.95 | 0.49 | **−48.2 %** |
| **rough-far** — live annulus is the rough half | 1.46 | 1.68 | **+15.1 %** |
| control — no contrast | 1.66 | 1.67 | +0.6 % |

**The correction spans ~55 percentage points on one codebase, one schedule and
one frame count, purely from where the roughness sits.** So the "no consistent
bias, −40% to +21%" report is not noise and not a defect — it is the predicted
consequence of terrain contrast varying by ring and sequence, and the earlier
"consistent 3–12%" was the reading that could not have been right.

**The consequence is that no correction factor exists.** A bias with a fixed
direction could have been divided out of the published numbers in prose; one
set by the terrain under each ring cannot be. Fixing the metric was the only
option, which is what this section records. Pinned in
`test_the_sign_of_the_band_filter_follows_the_terrain_not_the_code`.

⚑ **A second driver, smaller and disclosed:** the stale population was written
at longer range from fewer returns, so it is the worse estimate even on
identical ground, pushing the correction negative independently of roughness.
It shrinks as return density rises — at 25,000 returns/frame the control reads
≈ −10 %, at 60,000 it reads ≈ 0. **Prediction for the real-data A/B: each
sequence's sign should track its roughness contrast across each ring's inner
boundary, not a constant.** That is falsifiable and is the thing to check when
07/08 land.

**⚑ Resolved 2026-09-17 — §2b is regenerated.** *(Original note:)* ρ = 1.45 median (ring 1, n = 11) is
the claim we lead with. This fix changes the scored population, and on the
synthetic sequence ρ moves by up to **0.06 per ring**. **§2b's table should be
regenerated with this fix before ρ is quoted to two decimals.** The finding
survives in shape either way — a 0.06 shift does not move ρ out of its band —
but the second decimal is not currently earned.

**⚑ A separate caveat on ρ's denominator, found alongside.** `spread` is
estimated from the 5 cm reference cells of `F(c)` that M\* observed. Median
coverage on the 12-frame synthetic sequence is 1.00 / 0.25 / 0.06 / 0.02 for
rings 0–3 — ring 3's sub-cell variability comes from roughly one reference cell
in sixty-four. A spread estimated from two points is biased low and ρ divides
by it, so ρ on the coarse rings is biased **high**: the conservative direction
for a number we want near 1. `coarsening_ratio_per_ring` already drops
`n_ref ≤ 1`; at `k = 8` that guard admits a spread from two cells of
sixty-four. Disclosed rather than corrected, and coverage is now a `cov` column
printed next to ρ so the two cannot be read apart. This compounds §2b's "ring 0
has no ρ on any sequence" — between them, ρ is best evidenced at ring 1.

---

## 8. A coarse cell could share its footprint with a finer one — FIXED 2026-09-17 (open item D2 / R3)

*Shrestha, in Aakash's lane while he was away. `src/grid/lattice.py`,
`src/gpu/cuda_kernels.py`, and every caller of `bin_points` / `ring_of`.*

### What the defect was

Ring membership was decided per POINT: each return's own `d_aniso` against
`R_L`. The ring boundary is a real number and a cell is a block on the lattice,
so the boundary fell strictly inside blocks, and two returns in one 40 cm cell
could be filed into 5 cm and 40 cm. The 40 cm cell's footprint then contained
an occupied 5 cm cell — the §2.2 partition property that `block_stats`,
`_compared` and `query()` all assume, broken.

The 12 Sep design note framed this as an anisotropy problem (`a_f(v)` puts the
boundary off the lattice). **On real data it was worse than that, and live at
v = 0**, because the engine decided the ring from `points_sensor` — rotated
with the vehicle's heading — while the ring buffers are world-aligned squares.
Measured on seq 08, 30 frames, the engine's own binning:

| | before | after |
|---|---|---|
| frames with a coarse cell containing a finer occupied cell | 30 / 30 | 0 / 30 |
| coarse cells affected | 0.108% | 0 |
| returns dropped: ring chosen, then outside that ring's window | **0.224%** (~277 / frame) | 0 |

The dropped returns are the corners of the rotated square: a point at sensor
(9.9, 9.9) with the car facing 45° is ring 0 by Chebyshev distance and is at
world offset (0, 14) — outside ring 0's window, so `bin_points` returned -1.

### The fix

Per BLOCK, coarse to fine (`ring_of`'s docstring has the proof): start in the
coarsest ring if its window holds the point; a ring-L block splits into ring
L-1 only if every child lies in ring L-1's window **and** eq. (20) at the
block's nearest point is below R_{L-1} (or the rear floor forces it). Every
decision is a function of the block alone, so all points of a block stop at
the same level — no footprint can contain another and there is no gap.
Containment is an integer test against the real windows, so it cannot drop a
return a coarser ring has room for.

The design note's part B, snapping the boundary to the coarser lattice, is not
used: under a heading the boundary is not axis-aligned, and the per-block rule
makes the partition hold without it.

`bin_points` now takes world points plus the vehicle position and heading
(`MapEngine` reads the heading off the pose). `GridMap` gained
`vehicle_yaw_rad` (default 0.0, which is what the eval harness has always
assumed), and `query()`, `metrics._ring_cells` and `gate` pass the windows so
routing, scoring and binning use one rule.

### Proof, and cost

- `test_no_cell_footprint_contains_another_under_foveation` (CI-blocking):
  both schedules, every block within 0.6 m of an inner boundary, at speeds
  solved from the schedule so a stretched boundary lands ±1e-7 m either side
  of a lattice line, and at five vehicle positions / headings. **It fails on
  the old rule** at 20 of 28 speeds (5/10/20/40) and 10 of 20 (5/10/50), with
  the vehicle at the origin facing +x.
- `test_every_return_inside_the_map_is_binned` (CI-blocking).
- CPU and GPU paths: identical on 200/200 frames of seq 08. Final hash
  `4e180a12…`, was `4313df1a…`.
- `bin` p50: cpu 6.77 → 10.27 ms, cuda 0.18 → 0.35 ms. Whole frame on cuda
  unchanged at 22.26 / 26.74 ms. Bin scratch 50 → 84 B per point (12.6 MB at
  the 150,000-point cap) — declared at startup, outside the cell budget. **No
  memory figure on a slide moves**: rings are still preallocated at their
  fixed half-widths.

### What it does to the accuracy numbers — 40 frames, 5/10/20/40

`python scripts/eval_synthetic.py --seq 07 --frames 40` (and 08), `main` @
`6af6907` against the final state of 17 Sep (per-block rule, heading set in the
harness as in the engine). RMSE only — ρ's `spread` definition also changed
today (§2b), so ρ is not comparable across this table; RMSE is.

| | r0 RMSE | r1 RMSE | r2 RMSE | r3 RMSE | R(S), common support |
|---|---|---|---|---|---|
| 07 before | 1.77 | 3.04 | 5.91 | 16.93 | 1.420 |
| 07 after | 1.77 | **2.83** | 5.99 | **13.64** | **1.313** |
| 08 before | 1.17 | 2.31 | 4.89 | 54.86 | 0.171 |
| 08 after | 1.17 | 2.31 | **4.02** | 54.80 | **0.127** |

More cells are scored on every ring (07 ring 1: 42,227 → 46,084). 07 moves
more than 08 because its heading is further from an axis, which is exactly
where the rotated-frame rule and the world-aligned windows disagreed.

⚑ A diagnostic run at an intermediate state (heading not yet set in the
  harness) read 07's *unrestricted* regret 5.020 with the representative path
  30% unknown. That line measures fill rate, as the script says; at the final
  state it reads 1.960 / 7%, and the planning window's unknown share fell from
  7.6% to 5.8% with the fix. Recorded so the number is not rediscovered as a
  regression.



---

## 9. §9.2 against only what each ring received — DONE 2026-09-17

*Aakash's 2 Sep handover item: "each ring is scored against a reference
containing observations that ring never received … a metric comparing each
ring against a reference restricted to what that ring actually observed would
isolate coarsening properly." Done by Shrestha while Aakash was away.*

`reference_map.RingObservations` builds, per ring, a sparse M\* from exactly the
static ground returns that ring integrated — attributed during
`harness.run_sequence` with `lattice.ring_of_into`, the function `bin_points`
bins with, so no return can be credited to a ring that did not receive it
(`test_ring_observations_credit_each_return_to_the_ring_that_binned_it`). It
answers `block_stats` like M\*, so every §9 metric takes it unchanged.
`eval_synthetic.py` prints it under each per-ring table as `vs M*|ring`.

The synthetic one-off in `metrics.py`'s docstring put this at ≤ 0.05 cm and
said the direction on real data was **unknown**. On real data it is not small.
All eleven sequences, 40 frames, 5/10/20/40:

| seq | r0 ρ | r1 RMSE | r1 ρ | r2 RMSE | r2 ρ | r3 RMSE | r3 ρ |
|---|---|---|---|---|---|---|---|
| 00 | 1.06 | 3.38 | 1.10 | 1.54 | 1.02 | 1.59 | 1.01 |
| 01 | 1.05 | 0.78 | 1.08 | 0.87 | 1.03 | 1.32 | 1.01 |
| 02 | 1.06 | 1.02 | 1.05 | 3.12 | 1.02 | 1.66 | 1.01 |
| 03 | 1.06 | 5.51 | 1.10 | 3.52 | 1.03 | 1.50 | 1.01 |
| 04 | 1.05 | 1.69 | 1.05 | 3.55 | 1.03 | 4.95 | 1.03 |
| 05 | 1.10 | 0.91 | 1.06 | 2.40 | 1.04 | 2.85 | 1.02 |
| 06 | 1.06 | 1.42 | 1.07 | 3.15 | 1.05 | 2.18 | 1.01 |
| 07 | 1.08 | 1.91 | 1.08 | 2.83 | 1.04 | 4.20 | 1.04 |
| 08 | 1.12 | 1.28 | 1.06 | 2.46 | 1.04 | 2.44 | 1.02 |
| 09 | 1.07 | 1.77 | 1.10 | 1.08 | 1.02 | 0.84 | 1.01 |
| 10 | 1.05 | 1.79 | 1.05 | 1.86 | 1.03 | 1.08 | 1.03 |

```
M*|ring  ring 0  rho 1.06 [1.05-1.12]  RMSE 0.65 [0.45-0.98] cm  (vs M*: 1.59 [0.84-5.26])
M*|ring  ring 1  rho 1.07 [1.05-1.10]  RMSE 1.69 [0.78-5.51] cm  (vs M*: 3.07 [2.09-12.60])
M*|ring  ring 2  rho 1.03 [1.02-1.05]  RMSE 2.46 [0.87-3.55] cm  (vs M*: 8.82 [3.71-33.53])
M*|ring  ring 3  rho 1.01 [1.01-1.04]  RMSE 1.66 [0.84-4.95] cm  (vs M*: 10.38 [2.77-18.24])
```

### How to read it — and how not to

⚑ **ρ ≈ 1.03–1.08 is NOT a better headline, and must not be quoted as one.**
  When the cell and the reference average the same returns, the only thing
  left between them is the fusion itself — Kalman weighting against an
  unweighted mean, and 1 cm storage — so ρ near 1 is close to guaranteed. What
  this table measures is that **the map integrates what each ring receives
  faithfully**: 3–8% over the terrain's own spread. The §2b table remains the
  headline, because the map is used to answer for ground as the reference
  knows it.

**What the difference between the two tables is.** Everything §2b's ρ carries
above this table's is the ring's returns disagreeing with *other* returns of
the same ground — fired from another range, another frame, another ring. Ring
2's median RMSE is 8.82 cm against M\* and 2.46 cm against M\*|ring: about 70%
of ring 2's error on the median sequence is that disagreement, not the
coarsening of 20 cm cells. That is a statement about range-dependent
observation error (pose drift over the frames between the looks, beam
divergence, grazing incidence), and it is the error budget to attack next.

**Sequence 00's ring-2 outlier — narrowed, not closed.** §2b's ρ 2.20 (2.32 on
2 Sep) becomes **1.02** against M\*|ring, RMSE 33.54 → 1.54 cm. So the outlier
is not coarsening and not fusion: the 20–50 m returns ring 2 integrated
disagree by ~33 cm RMS with the returns of the same ground from other ranges.
00's systematic pose bias was already fixed (§6); what remains is dispersion
across looks. **Cause still open** — range-dependent registration on a long
urban loop is the leading hypothesis.

**Ring 3 on 08, 09 and 10 — found here, FIXED, see §11.** ρ 1.84 / 1.64 / 1.82
against M\*, and unchanged against M\*|ring, so the map disagreed with the very
returns it integrated. The cause was the 8 m height band. The table above is
from before that fix; §2b has the current numbers.

---

## 10. The money plot's non-monotone steps — both real ones were the metric, FIXED

*Aakash's 2 Sep open number: "the money plot's remaining non-monotone step,
which survives both the extent fix and 64-query averaging".*

`eval_synthetic.py` prints R(S) ± SE per schedule, and every adjacent step
**paired over the same 64 planning queries**. The schedules are planned on
identical start/goal pairs, so the per-query difference cancels how hard each
query is, which unpaired SEs do not.

### Seq 07's uniform 20 cm spike — found and fixed

Before, seq 07 read uniform 10 / 20 / 40 cm = 1.519 / **2.237** / 1.667. The
paired steps were +0.718 (8.5 SE) and −0.650 (7.9 SE), which is not query
noise. The spike was spread across queries: the top five carried only 24% of
it, so it was systematic.

**Cause: `costmap_from_gridmap` weighted each map cell under a 25 cm planning
cell by its observation COUNT.** A coarse cell clipping one corner of the
footprint got the weight of all its returns, so a neighbour's height leaked
into the planning cell. Where the map lattice beats against the 25 cm planning
lattice, every 1 m at 20 cm, the leak alternated into phantom steps. On M*'s
own optimal paths the 20 cm map set 23 slope and 27 step walls that M* did not
have, against 16 + 20 at 10 cm and 7 + 12 at 40 cm, and every detour around
them is scored as regret. Two other suspects were measured and ruled out.
Taking roughness and class off the OR of stored per-cell bits moved seq 07's
R(S) by less than 0.01. The class excess is the same with or without the OR,
and comes from content (fusion votes over all returns, M\* over ground only).

**Fix: weight by the share of the footprint each cell covers**, the number of
the 5 cm-spaced samples that land in it. That is the reference side's own
estimator: `block_stats` is a mean over the block's observed 5 cm cells, and for
a 5 cm map the two sides are now identical. Observation counts still decide
confidence (bit 5), where evidence is the question. Pinned by
`test_a_footprint_height_is_weighted_by_area_not_by_returns`, which fails under
count weighting (0.49 m against the reference's 0.10 m).

⚑ One test's premise changed with it, and is recorded rather than weakened.
  `test_regret_lattice.py` asserted that uniform 20 cm MISSES more of M*'s
  walls, the pothole rims, than 5/10/20/40. It did, through the same leak.
  With area weighting neither map misses one. The test now asserts that both
  find all 12, and that the coarse grid carries the larger depth error at them
  (it reads the 30 cm pothole at −26 cm), which is where the information loss
  physically is.

### Seq 09's 20 → 40 cm step: §7.1 bit 4 measured two different things — FIXED

After the area-weighting fix, seq 09 still read uniform 20 cm 0.101 against
40 cm 0.017 (−0.084, 2.5 SE). The weighting fix did not touch it, so the cause
was different. Six queries carried it, all starting at the same cells.

**Cause.** M\*'s optimal path ran along a road edge that M\* called drivable
and **every map marked non-drivable** (§7.1 bit 4), so each map detoured.
40 cm's detour happened to tie M\*'s cost and 20 cm's did not. The class bit
was not one measurement on the two sides. The map's class layer is fused from
**all** of a cell's returns, so a hedge or fence over the verge reads as
vegetation. M\* took its class from the **first ground return** in each cell,
so the same verge read as terrain. That asymmetry was everywhere, not only on
09. On seq 07, 5/10/20/40 set 173 class bits M\* did not have; on 09, 54.

**Fix.** `reference_map._Builder` now takes each 5 cm cell's class from the
**majority of every static return**, the same returns the map votes over.
Heights still come from ground returns only. On 09, 5/10/20/40's extra class
bits fell from 54 to 11, and what remains grows with cell size, which is what
real coarsening loss looks like. Pinned by
`test_class_is_the_majority_of_all_static_returns_height_is_ground_only`.

It moves regret on other sequences too, because class penalties were part of
every path. Seq 07's R(S) roughly halves (5/10/20/40 1.142 → 0.456); seq 00's
5/10/20/40 rises (0.075 → 0.205), within its SE of 0.094.

### All eleven sequences, after both fixes (40 frames)

| seq | 5/10/20/40 | 5/10/50 | uniform 10 cm | 20 cm | 40 cm | 80 cm |
|---|---|---|---|---|---|---|
| 00 | 0.205 ± 0.094 | 0.136 ± 0.067 | 0.028 ± 0.016 | 0.067 ± 0.031 | 0.028 ± 0.016 | 0.028 ± 0.016 |
| 01 | 0.074 ± 0.028 | 0.074 ± 0.028 | 0.089 ± 0.030 | 0.089 ± 0.030 | 0.049 ± 0.017 | 0.529 ± 0.041 |
| 02 | 0.061 ± 0.024 | 0.061 ± 0.024 | 0.077 ± 0.028 | 0.088 ± 0.029 | 0.110 ± 0.032 | 0.088 ± 0.029 |
| 03 | 1.278 ± 0.094 | 1.250 ± 0.107 | 1.215 ± 0.128 | 1.275 ± 0.129 | 1.281 ± 0.127 | 1.442 ± 0.124 |
| 04 | 0.012 ± 0.008 | 0.012 ± 0.008 | 0.008 ± 0.004 | 0.011 ± 0.007 | 0.026 ± 0.010 | 0.042 ± 0.015 |
| 05 | 0.011 ± 0.006 | 0.011 ± 0.006 | 0.023 ± 0.020 | 0.010 ± 0.006 | 0.009 ± 0.005 | 0.063 ± 0.021 |
| 06 | 0.464 ± 0.108 | 0.464 ± 0.108 | 0.382 ± 0.071 | 0.342 ± 0.071 | 0.669 ± 0.158 | 0.844 ± 0.172 |
| 07 | 0.456 ± 0.069 | 0.464 ± 0.069 | 0.377 ± 0.064 | 0.672 ± 0.109 | 0.611 ± 0.068 | 1.336 ± 0.106 |
| 08 | 0.077 ± 0.028 | 0.077 ± 0.028 | 0.123 ± 0.034 | 0.177 ± 0.045 | 0.121 ± 0.057 | 0.038 ± 0.015 |
| 09 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.038 ± 0.021 | 0.090 ± 0.031 | 0.121 ± 0.041 | 0.107 ± 0.026 |
| 10 | 1.490 ± 0.178 | 1.553 ± 0.187 | 1.150 ± 0.129 | 0.870 ± 0.157 | 0.939 ± nan | 1.268 ± 0.340 |

**What the paired steps say now (12 of 55 at |z| ≥ 2):**

- **No uniform curve has a backward step past 2 SE on any sequence.** Every
  significant step between neighbouring uniform sizes is positive: coarser
  costs regret or is indistinguishable.
- **Seq 07:** 10 → 20 cm +0.295 (2.4 SE), 20 → 40 cm −0.061 (0.7 SE, noise),
  40 → 80 cm +0.622 (7.7 SE).
- **Seq 09:** 10 → 20 cm +0.051 (2.1 SE), 20 → 40 cm +0.031 (2.6 SE).
- **Seq 08, the sequence the handover named:** noise throughout.
- **Coarsest-uniform cost is consistent:** 40 → 80 cm at ≥ 2 SE on 01, 03, 04, 05
  and 07 (01: +0.480, 11.9 SE).
- **The one significant negative step** is seq 10, 5/10/50 → uniform 10 cm
  (−0.460, 2.7 SE). The two schedules cover different extents, so this is not
  a monotonicity question. Seq 10's coarse uniform maps block most queries
  (n = 17 at 10 → 20 cm), so read them as unmeasured.

---

## 11. The 8 m band wrote clamped heights as measurements — FIXED 2026-09-17

*Found through §9: ring 3 on 08, 09 and 10 disagreed with its own returns.*

The map stores heights in an 8 m band (−2 m to +6 m at the time, **−3.5 to +4.5 m since the rebalance below**, about a datum that follows
the vehicle in 1 m steps). Two ways a height outside it became a confident,
wrong one:

1. **Fusion clamped, then averaged.** `quantise_height` clamps; `scatter` gave
   the clamped value full Kalman weight. On 08, ground returns 20–26 m below
   the road, 50–100 m out, with 6–11 m of spread inside one 40 cm cell, went in
   at the band floor. Nine cells carried 99.5% of ring 3's squared error while
   the median cell was off by 0.32 cm.
2. **Re-basing hid saturation.** When the datum stepped, a cell at the band
   edge was shifted and clamped again, landing 1 m inside the band where
   nothing could tell it was saturated. Seq 09 had 25 ring-3 cells at exactly
   +5.00 m against a reference of +5.7 to +8.0 m; seq 10 had 168 at exactly
   −1.00 m against −1.5 to −1.7 m.

**The fix, on both the CPU and CUDA paths:** a ground return outside the band
carries **no height weight** (`kernels.out_of_band`; it still counts as an
observation and still stays out of the ceiling). A stored height that leaves
the band on re-basing **loses its evidence** (variance code 0, the codec's
"never fused"), so the cell reads unknown until an in-band return arrives.
The scoring follows the same rule: M\* is built with `band=True`, which applies
the harness's own per-frame datum and counts what it leaves out (08: 28,679
ground returns over 40 frames; 07: 28). M\*|ring attributes only what `scatter`
fused. CPU and GPU remain identical on 200/200 frames of seq 08 (hash
`a9f979df…`); the device frame is unchanged at 22.3 / 25.9 ms.

| ring 3 | ρ before → after | RMSE before → after | vs own returns after |
|---|---|---|---|
| 08 | 1.84 → **1.03** | 54.80 → **2.97** cm | 2.11 cm |
| 09 | 1.64 → 1.48 | 25.55 → 14.81 cm | 0.97 cm |
| 10 | 1.82 → **1.13** | 10.58 → **2.57** cm | 1.10 cm |

09's remaining ring-3 error is cross-look disagreement (1.01 against its own
returns), not the map.

### What it cost, and the rebalance that recovered it

The fix alone made **seq 04's ring 3 worse** (20.61 → 24.47 cm). Traced cell by
cell: after a datum step the true ground sat 2 cm below the band floor, its
returns were (correctly) no longer fused, and a single ground-labelled return
3 m above the real surface, probably a Patchwork++ misclassification at 80 m,
defined the cell. The underlying problem was the band's placement. The datum
is `floor(road z)`, so the road is always 0–1 m above it, and the old
−2 / +6 m split kept only 2–3 m below the road and 5–6 m above.

**Rebalanced to −3.5 / +4.5 m, still 8 m, so no memory figure moves.** It was
chosen by survey, not by the eval. Ground returns beyond 10 m, every 10th
frame of all eleven labelled sequences, the share outside the band:

| split | lost, median sequence | lost, mean sequence |
|---|---|---|
| −2.0 / +6.0 (old) | 0.384% | 1.265% |
| −3.0 / +5.0 | 0.031% | 0.620% |
| **−3.5 / +4.5** | **0.009%** | **0.541%** |
| −4.0 / +4.0 | 0.052% | 0.604% |
| −4.5 / +3.5 | 0.110% | 0.828% |

Past −3.5 m, uphill ground starts leaving the top faster than downhill ground
returns to the bottom. The top stays clear of §7.1's clearance test: a road at
most 1 m above the datum, plus the 1.8 m vehicle height, leaves 1.7 m for
rising ground before a clamped ceiling could read as an obstruction. No 8 m
placement holds seq 01's embankments 11 m below the carriageway (3.3% of its
far ground).

**Result, eleven sequences, 40 frames:**

| ring 3 | before either fix | band fix only | **fix + rebalance** |
|---|---|---|---|
| median RMSE vs M\* | 13.64 cm | 12.96 cm | **10.38 cm** |
| worst RMSE vs M\* | 54.80 (08) | 24.47 (04) | **18.24 (04)** |
| ρ vs its own returns, range | 1.01–1.84 | 1.01–1.20 | **1.01–1.04** |
| RMSE vs its own returns, worst | 54.76 | 14.74 | **4.95** |

Seq 04 is back below its original 20.61 cm, and **every ring 3 on every
sequence now agrees with the returns it integrated to within 4%**. Ground
returns left outside the band on 08 fell from 28,679 to 5,833 over 40 frames.
Seq 10 went the other way (4,717 → 11,221: its uphill ground now meets the lower
top). Its ring 2 still improved (6.52 → 4.22 cm) and its ring 3 moved
2.57 → 2.77 cm.

**Still true:** a drop-off beyond the band is invisible. Such a cell reads
unknown, which is honest, but the map cannot say "cliff".

---

## What is not on this list

For the avoidance of doubt, the following are **settled**, not open questions:

- **FRNet is not used, on purpose.** Semantic and motion labels are ground truth
  from the SemanticKITTI `.label` files. This isolates the mapping contribution
  from segmentation error and is disclosed everywhere it matters. The one
  standalone FRNet port available does not reproduce the trained network; it is
  kept, flagged non-functional, for a possible future `mmdet3d` swap.
- **The KITTI reflectivity path is deliberate.** KITTI intensity is already
  firmware range/incidence-compensated, so the raw-power `·r²/cos` normalisation
  is not applied to it (it saturated 62 % of near-field road at the byte rail).
  The eq-(31) path is retained for sensors that need it.
- **Full-sequence robustness is verified.** Perception + mapping engine, every
  frame of seq 00 / 07 / 08, on both the pre- and post-elevation-fix engine:
  0 crashes, 0 NaN/Inf across ~3.2 × 10⁹ field values, no memory leak
  (`scratchpad/soak_grid_0708_out.txt`, `soak_elev_postfix_out.txt`).
