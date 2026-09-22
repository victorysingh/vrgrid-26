# pending-review: `TRAV_DEPRESSION = 1 << 6` and its two thresholds — three-way sign-off

**Status: NOT applied. Two frozen/shared files need a change and neither has been touched.**
The grid-side logic is already implemented and tested in `src/grid/traversability.py` (in-lane since
the 2026-09-23 reassignment of `src/grid/` to JP); this note covers only the parts that are **not**
JP's to land alone.

**Needs:** @AakashH2006 @Stxtics03 @victorysingh — the same gate as R5.

---

## Why this is gated even though the lane moved

`src/grid/` was reassigned from Aakash to JP on 2026-09-23. **That changes nothing here.**
`CLAUDE.md` makes `include/vrgrid/` *"FROZEN interfaces — whole-team change only, never edit
unilaterally"*, and `.github/CODEOWNERS` keeps `/include/vrgrid/` and `/configs/` on all three
handles. Ownership of the *implementation* and ownership of the *interface* are separate, and only
the first one moved.

## Change 1 — `include/vrgrid/cell.py`

```diff
 TRAV_CLASS = 1 << 4       # class not in drivable_set
 TRAV_CONFIDENCE = 1 << 5  # n < n_min  (fail safe: unobserved is not traversable)
+TRAV_DEPRESSION = 1 << 6  # wide shallow depression: the cell sits >= dip_min below
+                          # the chord joining its +/- depression_baseline_m
+                          # neighbours. Rings 0-1 only -- beyond ~15 m consecutive
+                          # beams are >1 m apart and the feature is not in the data
+                          # (math §1.2). R10.
```

**No memory cost, and the 12-byte cell does not grow.** `traversability` is already a `uint8` with
bits 0–5 used; bits 6 and 7 are free. Verified: `CELL_BYTES == 12` before and after, and
`scripts/memory_table.py` still reports **8.94 MB** for the 4-ring schedule. A test pins both
(`tests/test_depression.py::test_no_trav_depression_bit_exists_yet`), and that test is written to
**fail once this lands**, as the reminder to wire the bit into `bitfield()`.

## Change 2 — `configs/thresholds.yaml`

```diff
 traversability:
   h_vehicle_m: 1.8         # bit 0  clearance
   theta_max_deg: 20.0      # bit 1  slope, compared as tan(theta_max)
   s_max_m: 0.15            # bit 2  step, max |z_c - z_nbr| over 4-neighbours
   baseline_m: 0.50         # bits 1 and 2 are differenced over THIS distance
+  depression_baseline_m: 4.0    # bit 6. SECOND baseline, alongside the 0.50 m one
+                                # above rather than replacing it. 0.50 m is tuned so
+                                # a 12 cm kerb stays passable at every cell size,
+                                # which is exactly why it cannot see a 2 m wide
+                                # feature: the stencil is shorter than the hazard.
+                                # 4.0 m straddles a 2 m depression with a full cell
+                                # of rim on each side.
+  depression_dip_min_m: 0.15    # bit 6 threshold, as a DEPTH below the chord. Set
+                                # equal to s_max_m deliberately: a 15 cm dip is as
+                                # untraversable as a 15 cm step, and tying them means
+                                # one number moves, not two.
```

**⚑ These two values are the part that most needs a second opinion.** `4.0` and `0.15` are defensible
(see below) but they were chosen tonight, not derived from a prior specification, and
`configs/thresholds.yaml` is frozen before schedules are compared — so picking them unilaterally is
exactly what that freeze exists to prevent.

## What was designed, and on what basis

**No prior spec existed.** The roadmap named *"R10 wide-depression gradient"* and nothing else — the
full-project audit found zero commits, zero files and zero mentions. Everything here is new.

- **Second difference, not a longer slope test.** A bowl is symmetric, so its centre has near-zero
  first derivative at any baseline. What distinguishes it is curvature: the cell sits below the chord
  joining its neighbours. Lengthening bit 1's baseline would not have worked.
- **Why the case is missed today.** A 2 m wide, 15 cm deep bowl has peak gradient `2·depth/R = 0.30`
  m/m against `tan(20°) = 0.364`, and no 4-neighbour step over the 0.5 m stencil reaches
  `s_max = 15 cm`. Pinned by `test_the_existing_kerb_baseline_does_NOT_catch_it`, which is written to
  fail if a future threshold change makes bits 1/2 catch it — at which point R10 should be
  re-examined rather than kept from habit.
- **Rings 0–1 only, and that is physics.** Radial spacing grows as `r²·dφ/h_s` (math §1.2): with
  `h_s = 1.73 m` and `dφ ≈ 0.4°`, beams land 1 m apart at ~15.7 m and 30 cm apart at ~8.6 m. Below two
  beams inside the feature there is nothing to detect, at any baseline. Honest caveat in the code:
  ring 1 runs to 25 m, past that limit, so the check degrades across ring 1 rather than stopping
  cleanly at its edge.
- **`>=` not `>`.** A dip equal to the limit fails safe, matching the convention bit 5 already sets.

## Verification already done

- `tests/test_depression.py` — **12 tests, all passing**: the 2 m × 15 cm case detected, detected at
  the bottom not the rim, flat ground clear, a 5 cm dip ignored, the boundary case failing safe,
  rings 2–3 skipped, rings 0–1 checked, unobserved neighbours not differenced, and the frozen struct
  unchanged.
- **Existing behaviour untouched:** 56 kerb/pothole/traversability tests pass; full suite **833
  passed, 28 skipped, 0 failed**.
- **Memory:** `scripts/memory_table.py` unchanged at 8.94 MB (4-ring) / 6.24 MB (3-ring).
- **Not done:** no real-sequence run. The logic is proven on synthetic ground only, so the false-alarm
  rate on real roads — cambered surfaces and drainage channels are shallow bowls too — is **unknown**
  and should be measured before the bit is trusted in the planner.
