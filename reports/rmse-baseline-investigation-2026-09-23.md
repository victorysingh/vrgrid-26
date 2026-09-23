# Is 1.8660 cm the correct new ring-0 RMSE baseline? — investigation

*2026-09-23, `jp/p99-alloc-fixes` @ `1ec89dd`. Machine state OK for every run (clock 2400/2400 MHz,
commit 14.42 of 15.73 GB) — these are deterministic accuracy measurements, but logged per standard.*

---

## Executive summary — conclusion (a), high confidence

**1.8660 cm is very likely the correct new baseline, and the shift is an intended consequence of
`df35fd5` fixing a real defect.** The mechanism is specific and quantified, not inferred from
adjacency in time: `df35fd5` changed `ring_of` to decide ring membership against **the windows that
actually exist** rather than idealised ones, which recovered returns that were previously being
dropped at ring boundaries. On seq 07 that recovered **324 ring-0 cells (+0.31%)**, and those cells
are roughly **six times harder than the bulk** — implied RMS **10.46 cm** against a bulk of 1.77 —
which alone accounts for the entire rise from 1.7747 to 1.8660. The old figure was not more
accurate; it was **flattering, because it silently omitted the hardest cells**. Two independent
corroborations: the shift is **concentrated on seq 07 only** (08 moved −0.02%, 00 −0.14%), which
rules out a systematic corruption of the boundary logic; and `docs/known-limitations.md`, regenerated
by `df35fd5` itself, **already publishes `r0 RMSE 1.87` for seq 07** — the new value. What keeps this
at "very likely" rather than certain is stated in §6.

---

## 1. What `df35fd5` actually changed

### 1.1 The defect, mechanically

`ring_of` was rewritten from `ring_of(x, y, schedule, speed_ms)` to
`ring_of(x, y, schedule, speed_ms, vehicle_xy_m, yaw_rad, buffers)`. The commit's own words for the
reason:

> ring membership is decided against the windows that **EXIST**, not the ones that ought to.

Each ring's window is recentred on the vehicle in **discrete lattice steps**, so the window that
exists at any moment is offset from the ideal one by up to half a cell. The old code tested
membership against the ideal window. A point could therefore be assigned to a ring whose actual
window did not contain it — the reported symptoms being **0.108% of coarse cells with nested
footprints** and **0.224% of returns dropped**, both measured on seq 08 over 30 frames, both 0 after.

### 1.2 Does it touch ring 0 directly? **Yes — directly, not downstream**

Ring 0 has a window like every other ring, and it shifts in exactly the same discrete steps. The
relevant consumer for RMSE is `metrics._compared`, whose `serves` test decides which cells are scored
as ring L. `df35fd5` changed that call:

```
-    serves = ring_of(*_cell_centres_m(gm, ring, ix, iy), gm.schedule, gm.speed_ms) == ring
+    serves = ring_of(*_cell_centres_m(gm, ring, ix, iy), gm.schedule,
+                     gm.speed_ms, gm.vehicle_xy_m, gm.vehicle_yaw_rad, gm.buffers) == ring
```

So **which cells count as ring 0 changed**. That is the direct mechanism, and the shift was
foreseeable in principle.

### 1.3 A second change that does NOT apply here

`df35fd5` also introduced `reference_map.RingObservations` — a per-ring reference scoring each ring
against only the returns that ring received. That is opt-in, guarded by `hasattr(reference,
"for_ring")`. **R1's harness calls `build_from_scans`, which returns `ReferenceMap`, and
`ReferenceMap` has no `for_ring`** — verified directly. So the reference basis did **not** change for
R1/R7, and this larger change is not the cause. Ruling it out matters, because it is the change the
commit message emphasises.

### 1.4 Was accuracy impact mentioned? **Yes, explicitly**

`src/eval/metrics.py`, added in the same commit:

> Across all eleven sequences the effect is **NOT second order**: ring 2's median RMSE is 9.22 cm
> against M\* and 2.62 cm against its own returns.
>
> *(Regenerated 2026-09-17: ring 1 rho 1.39 [1.22–1.53] on the between-cell spread, 1.25 [1.14–1.37]
> with the within-cell term; §2b.)*

So the author knew the metric moved and **regenerated the figures he owned**. R1 and R7 are JP's, and
their baseline is hard-coded in `reports/harnesses/r1_accuracy_by_class.py` — a file Shrestha had no
reason to know existed.

## 2. Independent reproduction — exact, on a state-OK machine

The prior investigation ran while the machine was paging. Re-run fresh:

| point | ring-0 RMSE (seq 07) | gate vs doc 1.76 |
|---|---|---|
| current tree | **1.8660** | ✗ 6.0% |
| old band `[-2.0, 6.0]` restored | **1.8660** — identical | ✗ |
| at `df35fd5~1` (verified pre-fix: `ring_of` has no `vehicle` argument) | **1.7747** | ✓ 0.8% |

**Reproduces to four decimal places.** The height band is conclusively not involved.

## 3. Is the shift uniform or concentrated? — **concentrated on seq 07**

`reports/harnesses/rmse_baseline_probe.py` (new, committed) runs the same builder, frames and
schedule as R1 with the gate removed:

| seq | before (`df35fd5~1`) | after | Δ RMSE | ring-0 cells before → after | Δ cells | **implied RMS of added cells** |
|---|---|---|---|---|---|---|
| **07** | 1.7747 | **1.8660** | **+5.14%** | 103,167 → 103,491 | **+324 (+0.31%)** | **10.46 cm** |
| 08 | 1.1669 | 1.1667 | −0.02% | 136,956 → 138,008 | +1,052 (+0.77%) | 1.14 cm |
| 00 | 2.7377 | 2.7338 | −0.14% | 82,868 → 83,270 | +402 (+0.49%) | 1.75 cm |

Three things follow, and together they are the core of the conclusion:

1. **All three sequences gained ring-0 cells.** The fix recovers previously-dropped returns
   everywhere — consistent with the 0.224% figure — so the mechanism is general.
2. **Only 07's RMSE moved.** 08 gained the *most* cells (+1,052) and moved least (−0.02%). So this is
   not a systematic distortion of the boundary logic; if it were, all three would move together.
3. **The recovered cells in 07 are ~6× harder than the bulk** (10.46 cm implied RMS against 1.77).
   In 08 and 00 the recovered cells are ordinary (1.14 and 1.75 cm, at or below their bulk), which is
   exactly why those sequences did not move.

### 3.1 The error is a tiny tail, and its size matches

On the current tree, seq 07, ring 0:

```
cells scored 103,491    RMSE 1.8660 cm
|error| p50 0.46   p90 1.39   p99 3.60   p99.9 20.91 cm
cells with |error| > 10 cm:  281  (0.272%)
their share of total squared error:  74.5%
```

**281 cells carry three-quarters of ring-0's squared error**, and the fix added **324** cells whose
implied RMS is 10.46 cm. Those two populations are the same size to within 13%, and the implied RMS
sits inside the >10 cm tail. That is a closed, self-consistent account of the whole 5.14%.

**Stated precisely so it is not over-claimed:** I did not establish cell-by-cell identity between the
324 added cells and the 281 high-error cells — slot indices are not comparable across the two trees
without more work. What is established is that the counts and magnitudes align closely enough that no
other contributor is needed to explain the shift.

### 3.2 Per-class and spatial breakdown — not done

The existing harness computes per-class rows only *after* its gate, which fails. Rather than
re-plumb it, the tail analysis above answers the same question more directly. Spatial clustering
against ring boundaries was not tested — see §6.

## 4. Was the new number already validated elsewhere? — **yes, once**

`docs/known-limitations.md`, whose eleven-sequence table `df35fd5` regenerated (137 lines changed in
that file), carries:

```
| seq | r0 RMSE | r0 ρ | r1 RMSE | ...
| 07  |  1.87   | 1.27 |  2.82   | ...
| 08  |  1.17   | 1.15 |  2.31   | ...
```

**`1.87` for seq 07 is the new value**, matching this investigation's 1.8660; 08's 1.17 matches
1.1667. (00 reads 2.70 against 2.7338 — the table is over whole sequences rather than 40 frames, so
close agreement rather than identity is expected.)

So the post-fix ring-0 numbers were measured, and published, by the commit's own author. Nothing
else in `docs/gpu-lane/` consumes the ring-0 RMSE — searched, no hits — so this is the only
corroborating consumer, but it is a direct and authoritative one.

## 5. Conclusion — (a), with the reasoning that supports it

**(a) This is very likely an intended, correct consequence of `df35fd5`.** The supporting evidence,
in order of weight:

1. **A specific mechanism**, not adjacency in time: the `serves` test now routes cells using the
   windows that exist, and ring 0 is directly affected (§1.2).
2. **The magnitude is fully accounted for**: 324 recovered cells at ~10.46 cm implied RMS explain
   +5.14% with nothing left over, and a 281-cell tail carrying 74.5% of squared error corroborates
   the size (§3, §3.1).
3. **The pattern rules out systematic corruption**: concentrated on one sequence, absent on the one
   that gained the most cells (§3).
4. **The direction is the honest one**: recovering dropped boundary returns should *raise* RMSE,
   because boundary cells are the hardest. A fix that removed error would be the suspicious outcome.
5. **The author measured and published the new value** in `known-limitations.md` (§4).

**The old 1.76 was not a better measurement. It was a measurement taken with 324 of the hardest
cells silently excluded.**

## 6. What would change this answer, stated honestly

- **Cell-by-cell identity** between the added and high-error cells is inferred from matching counts
  and magnitudes, not proven (§3.1).
- **No spatial test**: I did not confirm the affected cells cluster at the ring-0/ring-1 boundary.
  That would be the final direct confirmation of the mechanism and it is not done.
- **`df35fd5`'s correctness is assumed, not re-derived.** Its own tests assert nested footprints and
  dropped returns are 0, and D2 was verified closed earlier; this investigation takes that as given
  rather than re-auditing the lattice change.
- **Shrestha's intent is still the most authoritative source.** This establishes that the shift is
  explainable, sized, localised and already published by him — which makes his confirmation a final
  check rather than the only evidence. It does not replace it.

If any of the above came back the other way — particularly if the high-error cells turned out to be
scattered rather than boundary-adjacent — the conclusion would move toward (c), indeterminate, not
straight to (b).
