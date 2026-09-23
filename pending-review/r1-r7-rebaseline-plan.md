# pending-review: re-baselining R1 and R7 to the post-`df35fd5` ring-0 values

**Status: PLAN ONLY. Nothing applied.** This changes *committed reports*, not a design doc, so it
waits for JP's explicit go-ahead — and JP has said he may wait for Shrestha's confirmation regardless
of what the investigation concluded. Basis:
`reports/rmse-baseline-investigation-2026-09-23.md`, conclusion **(a)**, high confidence.

**Do not execute any of this without that confirmation.**

---

## What changes, exactly

### 1. `reports/harnesses/r1_accuracy_by_class.py` — the gate constants

```python
BASELINE_R0 = {"00": 2.73, "07": 1.76, "08": 1.16}
```
becomes, from this investigation's measured values (40 frames, `5/10/20/40`):

```python
# Re-baselined 2026-09-__ after df35fd5 routed ring membership against the
# windows that exist. Seq 07 moved 1.7747 -> 1.8660 because the fix recovered
# 324 previously-dropped ring-0 cells whose implied RMS is ~10.46 cm; 08 and 00
# barely moved. See reports/rmse-baseline-investigation-2026-09-23.md.
BASELINE_R0 = {"00": 2.73, "07": 1.87, "08": 1.17}
```

**Keep the 5% tolerance unchanged.** Widening it would hide the next shift, which is the failure this
whole episode is about. `00` is left at 2.73 (measured 2.7338) and `08` moves 1.16 → 1.17 (measured
1.1667) so all three sit on their measured values rather than only the one that failed.

### 2. `reports/r1-accuracy-by-class-and-range-band.md` — regenerate

- Remove the `STALE 2026-09-23` banner and replace it with a dated note recording that the report was
  regenerated against the post-`df35fd5` map, with the old ring-0 values named so the change is
  traceable rather than silent.
- Re-run and paste the full table:
  `VRGRID_DATA_ROOT=... python reports/harnesses/r1_accuracy_by_class.py`
- The ALL rows in the harness's own provenance header (`07 1.78/3.60/5.91`, `08 1.17/2.31/4.89`,
  `00 2.74/6.77/34.10`) must be updated in the same commit, or the header lies about what it produces.

### 3. `reports/r7-hazard-miss-rate.md` — regenerate

Same shape. R7 shares the gate, so it has been unrunnable since 17 Sep for the same reason.
`VRGRID_DATA_ROOT=... python reports/harnesses/r7_hazard_miss.py`, then update the miss-rate table
and the harness header's `8/19, 0/3, 4/38` counts.

### 4. `OPEN-ITEMS.md` — close RMSE-BASELINE

Move it to §5 with the conclusion, the mechanism in one line, and a pointer to the investigation.
Remove the `provisionally invalidated` note from the BAND row's mention of these two reports, since
the band was ruled out — that row should end up saying the band affected **neither**.

## Order of operations

1. Confirm (JP, and ideally Shrestha).
2. Update `BASELINE_R0` **first**, in its own commit, so the re-baseline is reviewable separately
   from the regenerated numbers it unblocks.
3. Run both harnesses; check both gates now pass.
4. Regenerate both reports and their provenance headers in one commit.
5. Close the tracker row.

## What must NOT happen

- **Do not widen the 5% tolerance.** The gate caught a real change and has been correctly red for six
  days.
- **Do not regenerate without recording the old values.** Every dated correction on this project keeps
  the superseded number visible; a silent overwrite would make the 1.76 → 1.87 move undiscoverable.
- **Do not re-baseline `00` or `08` away from measurement** to make them look untouched — they moved
  by 0.1–0.6% and the new constants should reflect what was measured, not what is tidy.

## One open risk

The investigation could not prove cell-by-cell that the 324 recovered cells are the 281 high-error
cells; it established that the counts and magnitudes align with nothing left unexplained. If Shrestha
says the ring-0 effect was *not* expected, **stop and re-open** rather than proceeding — the numbers
would then need explaining before they are published, not after.
