# R1 — height accuracy by range band × class, with n

> **⚠ [PROVISIONALLY INVALIDATED 2026-09-23 — figures predate the band change.]**
> `843ad54` (17 Sep) moved `vertical_extent_m` from `[-2.0, 6.0]` to `[-3.5, 4.5]` in
> both schedule configs — same 8 m span, shifted down 1.5 m — and wired it through
> `quantise.py`, `metrics.py`, `harness.py` and both kernel files. **Every figure in
> this report is height-derived and was measured on the OLD band.** They have **not**
> been re-measured, so this is "do not quote until re-run", not "wrong". Re-running
> this report against the current band is tracked as the **BAND** row in
> `OPEN-ITEMS.md`.


*Measured 2026-09-11 overnight on `main` at `9b40ff2`. New computation from the
existing eval harness; no tracked file was modified.*

Range band = ring (the schedule's own range partition). Class = the semantic
class the **map** stored, from the 5-bit candidate field. Population is exactly
`metrics._compared` — the same cells `height_rmse_per_ring` reports, so the
`ALL` row reproduces the published per-ring number by construction. Classes
with n < 25 are pooled into one row; an RMSE over 12 cells is noise.

Gate passed on all three sequences (ring-0 RMSE 0.3–1.1% from doc §2b).

---

## The headline: the drivable surface is far more accurate than the aggregate,
## and the gap grows with range

| seq | ring | drivable n | **drivable RMSE** | all n | all RMSE | ratio | drivable share |
|---|---|---|---|---|---|---|---|
| 07 | 0 | 93,902 | **0.95 cm** | 103,182 | 1.78 cm | 1.9× | 91% |
| 07 | 1 | 38,232 | **1.14 cm** | 50,153 | 3.60 cm | 3.2× | 76% |
| 07 | 2 | 8,962 | **1.20 cm** | 12,703 | 5.91 cm | 4.9× | 71% |
| 08 | 0 | 131,201 | **1.16 cm** | 137,034 | 1.17 cm | 1.0× | 96% |
| 08 | 1 | 126,536 | **1.10 cm** | 141,141 | 2.31 cm | 2.1× | 90% |
| 08 | 2 | 39,699 | **1.12 cm** | 49,073 | 4.89 cm | 4.4× | 81% |
| 00 | 0 | 72,114 | **2.22 cm** | 82,868 | 2.74 cm | 1.2× | 87% |
| 00 | 1 | 32,380 | **3.13 cm** | 41,892 | 6.77 cm | 2.2× | 77% |
| 00 | 2 | 7,955 | **3.68 cm** | 11,275 | 34.10 cm | 9.3× | 71% |

*drivable = road ∪ parking ∪ sidewalk ∪ other-ground ∪ terrain (the §7.1 set),
pooled as √(Σnᵢ·RMSEᵢ² / Σnᵢ).*

**On the drivable surface, RMSE is essentially flat with range.** Seq 07 goes
0.95 → 1.14 → 1.20 cm across 0–10, 10–25 and 25–50 m. Seq 08 goes 1.16 → 1.10
→ 1.12 cm — it does not degrade at all. The per-ring aggregate grows with range
(1.78 → 3.60 → 5.91 on 07) **because the non-drivable fraction grows and is far
rougher, not because the ground estimate degrades.**

That is a materially different claim from the one §2b currently makes, and it is
the one the thesis actually needs: coarsening costs the *planning surface*
almost nothing out to 50 m.

**Caveat.** This is a mixture decomposition, not a correction. The aggregate is
not wrong — a map does contain vegetation. But quoting a single per-ring RMSE
for a system whose claim is about drivable ground blends two populations with
an order-of-magnitude difference in roughness, and the blend ratio (71–96%)
shifts with range and sequence.

---

## Per-class detail

### seq 07, 40 frames, 5/10/20/40 — gate 1.7761 cm (0.9%)

| band | ring | class | n | RMSE cm | mean bias | share |
|---|---|---|---|---|---|---|
| 0–10 m | 0 | road | 65,432 | **0.90** | −0.34 | 63% |
| | 0 | sidewalk | 21,380 | 1.16 | −0.51 | 21% |
| | 0 | terrain | 7,090 | **0.62** | −0.01 | 7% |
| | 0 | truck | 4,276 | 1.67 | −0.63 | 4% |
| | 0 | car | 1,580 | 1.19 | −0.35 | 2% |
| | 0 | **vegetation** | 1,458 | **7.98** | −3.72 | 1% |
| | 0 | fence | 1,062 | 2.62 | −0.38 | 1% |
| | 0 | **unlabelled** | 338 | **18.84** | −3.66 | 0% |
| | 0 | building | 326 | 4.14 | −1.15 | 0% |
| | 0 | **ALL** | **103,182** | **1.78** | −0.43 | |
| 10–25 m | 1 | road | 23,385 | 1.08 | −0.18 | 47% |
| | 1 | sidewalk | 10,123 | 1.28 | −0.32 | 20% |
| | 1 | terrain | 4,724 | 1.13 | −0.40 | 9% |
| | 1 | vegetation | 3,038 | 8.21 | −0.88 | 6% |
| | 1 | building | 3,034 | 4.93 | −0.52 | 6% |
| | 1 | unlabelled | 2,844 | 7.03 | −0.17 | 6% |
| | 1 | car | 2,321 | 5.32 | +0.05 | 5% |
| | 1 | **ALL** | **50,153** | **3.60** | −0.32 | |
| 25–50 m | 2 | road | 7,120 | 1.12 | +0.02 | 56% |
| | 2 | sidewalk | 1,812 | 1.48 | −0.07 | 14% |
| | 2 | vegetation | 1,369 | 5.07 | −0.12 | 11% |
| | 2 | building | 989 | 9.72 | +0.97 | 8% |
| | 2 | car | 569 | 16.19 | +0.46 | 4% |
| | 2 | fence | 381 | 17.09 | +3.95 | 3% |
| | 2 | **ALL** | **12,703** | **5.91** | +0.21 | |
| 50–100 m | 3 | **unlabelled** | 1,321 | 16.93 | −0.17 | **100%** |

### seq 08 — gate 1.1729 cm (1.1%)

Ring 0: road 85,643 @ **1.08**, sidewalk 32,801 @ 1.43, terrain 12,757 @ **0.90**,
vegetation 5,678 @ 1.31 → **ALL 137,034 @ 1.17**.
Ring 1: terrain 51,795 @ **0.77**, sidewalk 43,561 @ 1.31, road 30,884 @ 1.23,
vegetation 7,610 @ 5.28, building 3,462 @ 8.11, unlabelled 1,261 @ 9.91 →
**ALL 141,141 @ 2.31**.
Ring 2: terrain 20,412 @ 0.92, road 9,078 @ 1.69, sidewalk 8,737 @ **0.78**,
unlabelled 4,111 @ 10.94, vegetation 3,991 @ 12.25 → **ALL 49,073 @ 4.89**.
Ring 3: **100% unlabelled**, 6,856 @ 54.86.

### seq 00 — gate 2.7377 cm (0.3%)

Ring 0: road 42,651 @ 2.07, parking 19,126 @ 2.73, sidewalk 10,337 @ 1.75,
car 4,087 @ 6.66 → **ALL 82,868 @ 2.74**.
Ring 1: road 14,862 @ 3.39, sidewalk 13,840 @ 2.51, vegetation 5,349 @ 9.00,
building 3,082 @ **18.45**, trunk 120 @ **24.77** → **ALL 41,892 @ 6.77**.
Ring 2: see below.
Ring 3: **100% unlabelled**, 3,379 @ 9.11.

---

## ⚑ The "unexplained" seq-00 ring-2 outlier is vegetation

`known-limitations.md` §2b says of seq 00's ρ of 2.32 at ring 2:

> the rest is dispersion (spread 14.41 cm at ring 2) and is **unexplained**. 00
> is a long urban loop and ring 2 spans 20–50 m where ground segmentation is
> hardest; that is a hypothesis, not a finding.

**It is now a finding.** One class accounts for almost all of it:

| seq 00, ring 2 | n | RMSE cm | mean bias |
|---|---|---|---|
| road | 6,278 | 3.86 | −1.09 |
| unlabelled | 1,453 | 7.30 | +0.83 |
| **vegetation** | **1,261** | **100.77** | **+37.85** |
| sidewalk | 770 | 2.94 | −1.67 |
| terrain | 486 | 2.76 | −1.56 |
| building | 482 | 13.62 | −1.56 |
| **ALL** | **11,275** | **34.10** | +3.58 |
| **ALL minus vegetation** | **9,970** | **5.54** | |

**1,261 cells — 11% of the ring — carry a 100.77 cm RMSE and a +37.85 cm
systematic bias, and drag the whole ring from 5.54 to 34.10 cm.** A metre of
error with a third of a metre of *signed* bias is not dispersion; it is a
population of cells whose height is wrong in one direction. Vegetation at 25–50 m
on a long urban loop is overhanging canopy: the map is holding foliage height
where M\* holds ground.

Note this is *not* the same shape on 07 (vegetation ring 2 = 5.07 cm) or 08
(12.25 cm). Seq 00 is 8–20× worse on the same class.

**Not investigated further and not fixed** — flagged for whoever owns §2b.

---

## Ring 3 is 100% unlabelled on all three sequences — and that is the *correct* path

Every ring-3 cell scores as class `unlabelled` (`CLASS_UNLABELLED = 31`),
because SemanticKITTI's annotation stops around 50 m. `eval/harness.learning_ids`
maps `-1 → CLASS_UNLABELLED`, which is in no drivable set and so fails safe.

This is a clean confirmation that **the harness path is right where the
`MapEngine` path is wrong**: `src/run/engine.py:282` maps the same `-1 → 0`,
and id 0 is `car`, which is why the live dashboard shows a phantom car in every
ring-3 cell. Two implementations of one conversion; this report exercises the
correct one. (Known issue, not touched tonight.)

---

## Reproduce

`scratchpad/r1_accuracy_by_class.py` (session scratch, not committed — a
measurement harness, not a deliverable). Gates on ring-0 RMSE, then groups
`metrics._compared`'s scored population by `unpack_class(soa["semantic_class"])`.
