# Handover, 17 September — for Aakash: what changed in your lane

*From Shrestha. You were away today and I was asked to close your open items
rather than leave them. Everything is on `main`. None of it was discussed with
you first, so here is all of it in one place — please review, and push back on
any call you would have made differently.*

---

## 1. D2 / R3 — the ring boundary. `src/grid/lattice.py`

**Ring membership is decided per world-lattice BLOCK, coarse to fine**, not per
point. A ring-L block splits into ring L-1 only if every child lies in ring
L-1's window and eq. (20) at the block's nearest point is below R_{L-1} (or the
rear floor forces it). `ring_of`'s docstring has the argument.

- **The defect was live at v = 0**, not just under anisotropy: the engine
  decided rings from `points_sensor` (rotated with heading) against
  world-aligned windows. Seq 08, 30 frames: nested footprints on every frame
  (0.108% of coarse cells) and **0.224% of returns dropped**. Both 0 now.
- **Part B of your R3 note (snapping) is not used.** Under a heading the
  boundary is not axis-aligned; the per-block rule makes the partition hold
  without it. Part A (nearest point) is kept — as a block lower bound that is
  exact at yaw 0.
- **Signatures changed.** `bin_points(xw, yw, schedule, buffers, out, scratch,
  speed_ms, vehicle_xy_m, yaw_rad)` — world points only. `ring_of(x, y,
  schedule, speed_ms, vehicle_xy_m, yaw_rad, buffers)`. `ring_of_into` takes
  WORLD coordinates. `migrate_ring` passes the same through. `GridMap` gained
  `vehicle_yaw_rad` (default 0.0). `MapEngine.bin(xw, yw)` defaults the vehicle
  to the last `step()`.
- `metrics._ring_cells`, `query.slot_of`, `gate.apply`, `fusion.scatter` and
  `transient.step` all pass the windows, so routing, scoring and binning use
  one rule. Your centre test in `_ring_cells` is now exact.
- **The harness now sets `gm.vehicle_yaw_rad` from the pose** in
  `run_sequence`, as the engine does.
- **Tests (both `partition`, CI-blocking):**
  `test_no_cell_footprint_contains_another_under_foveation` (fails on the old
  rule at 20/28 and 10/20 speeds) and
  `test_every_return_inside_the_map_is_binned`.
- **Cost:** bin scratch 50 → 84 B/point; CPU `bin` 6.77 → 10.27 ms p50; GPU
  frame unchanged. No memory figure moves.

Full write-up: `docs/known-limitations.md` §8; status block at the top of
`pending-review/r3-ring-boundary-under-anisotropy.md`.

## 2. Ring 0's ρ — the deferred fix, done. `src/eval/reference_map.py`, `metrics.py`

- `ReferenceMap` stores `within_var_cm2` per 5 cm cell; `block_stats` adds its
  mean to `var` (law of total variance, cells weighted equally).
  `block_returns` is new. The ρ guard counts **returns > 1**, not cells.
- `load()` **refuses** an `.npz` without the field — rebuild caches with
  `scripts/build_reference_map.py`.
- **`block_stats(..., within_cell=False)` in `plan_regret.costmap_from_reference`.**
  With the term in, seq 08's R(S) went 0.127 → 2.497: within-cell variance is
  sensor noise as much as terrain, and the map side of the costmap has none.
  Roughness keeps its old definition.
- Result (final, after the band fix): ring 0 ρ **1.17 [1.13–1.29]**, n = 11. Ring 1 reads 1.39 on the between-cell spread and 1.25 with the term —
  the flattering direction you predicted, so §2b publishes both.

## 3. §9.2 against only what each ring received. `reference_map.py`, `harness.py`

Your handover item. `RingObservations` (sparse, per ring) is filled by
`run_sequence(observed=...)` and read through the unchanged metrics.
`eval_synthetic.py` prints it as `vs M*|ring`. Result and how to read it:
`known-limitations.md` §9. Short version: ρ 1.03–1.08, which is *not* a
headline (same returns on both sides); ~70% of ring 2's median RMSE is
cross-look disagreement, not coarsening; **seq 00's ring-2 outlier is
cross-look disagreement** (ρ 2.20 → 1.02); **ring 3 on 08/09/10 was inside the
map** (unchanged under M\*|ring) — traced to the 8 m band and fixed, §3b.

⚑ One bug of mine on the way, found and fixed before any number was
  recorded: the packed cell key overflowed the sign bit. Pinned by
  `test_ring_observations_rebuild_m_star_exactly`.

## 3b. Ring 3 on 08/09/10 — the 8 m band. `fusion.py`, `harness.py`, `reference_map.py`

Out-of-band ground returns now carry no height weight in `scatter`, and M\* is
built with `band=True` so it leaves out exactly those. My half is
`gpu/kernels.py`, `gpu/shift.py` and the CUDA kernels. 08 ring 3: ρ 1.84 → 1.03.
The band is rebalanced to −3.5 / +4.5 m (still 8 m), chosen by a survey of all
eleven sequences; seq 04's ring 3, which the fix alone made worse, recovered to
18.2 cm. `known-limitations.md` §11.

## 4. §2b regenerated, all eleven sequences

`known-limitations.md` §2b — new table on top, the 2 Sep one kept below it
marked superseded. The deck docs quoting 1.45 are updated
(`00-START-HERE`, `02-WHAT-TO-PRESENT`, `05-PANEL-DEFENSE`,
`07-CORRECTED-SCRIPT`).

## 5. The money plot's non-monotone step

`eval_synthetic.py` now prints R(S) ± SE and each adjacent step **paired over
the same 64 queries**. See `known-limitations.md` §10. **Seq 08's step is noise**
(every paired step < 1.6 SE). Two others are real and new: **seq 07's uniform
20 cm** is worse than both neighbours by ~8 SE, and **seq 09's** 40 cm beats
20 cm at 2.6 SE. 40 → 80 cm costs regret at ≥ 2 SE on 7 of 11.

## Not done, and why

- ~~Seq 07's 20 cm regret spike~~ — fixed in `plan_regret.costmap_from_gridmap`
  (area-weighted footprint heights, §10). `test_regret_lattice.py`'s wall test
  premise changed with it; read §10 before reviewing that diff.
- ~~Seq 09's 20 → 40 cm step~~ — fixed in `reference_map._Builder`: M*'s class is
  now the majority of all static returns, as the map's is (heights unchanged,
  ground only). §10.
- **Seq 00's cross-look disagreement at ring 2** — narrowed (§9), cause open.
- **D1, the Patchwork++ singleton** (GitHub #1) — JP's file.
