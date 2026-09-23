# R7 — hazard miss rate, with the correct denominator

> **✅ [REGENERATED 2026-09-23 against the post-`df35fd5` map.]**
> This report shared R1's ring-0 gate and was therefore unrunnable from 17 Sep to
> 23 Sep for the same reason: the gate was pinned to a baseline `df35fd5` had moved.
> See `reports/rmse-baseline-investigation-2026-09-23.md` for why 1.8660 is the
> correct ring-0 value and 1.76 was not.
>
> **Superseded counts, kept rather than overwritten:** hazard misses were
> **8/19, 0/3, 4/38** and are now **6/19, 1/3, 4/38** on `5/10/20/40`. Seq 07
> improved (8 → 6 misses), seq 08 worsened by one (0 → 1), seq 00 is unchanged.
>
> **⚑ Confirmed by evidence, not yet by the author.** If Shrestha says the ring-0
> effect was unintended, this becomes a live re-investigation and these counts
> revert.


*Measured 2026-09-11 overnight on `main` at `9b40ff2`. New computation from
existing data; no tracked file was modified.*

## Definition used

```
hazard miss rate = |truly non-drivable AND map calls it drivable|
                   ---------------------------------------------
                              |truly non-drivable|
```

- **"truly non-drivable"** — `costmap_from_reference(M*)`, a §7.1 bit in
  `IMPASSABLE_BITS = TRAV_CLEARANCE | TRAV_SLOPE | TRAV_STEP`.
- **"map calls it drivable"** — `costmap_from_gridmap`, same predicate, same
  planning lattice, none of those bits set.
- Restricted to `common_support(star, mine)`, so this is a **verdict**
  comparison and not a fill-rate one.
- Clearance is structurally absent from **both** sides (M\* is 2.5D ground and
  cannot set it), so in practice this is **slope ∪ step**.

This is the false-negative rate on hazards. Its complement is hazard recall.

## Gate — M\* proven before any number was taken off it

| seq | ring-0 RMSE measured | doc §2b | apart | |
|---|---|---|---|---|
| 07 | 1.7761 / 1.7032 | 1.76 | 1.0% / 3.2% | OK |
| 08 | 1.1729 / 1.1673 | 1.16 | 1.1% / 0.6% | OK |
| 00 | 2.7377 / 2.7323 | 2.73 | 0.3% / 0.1% | OK |

Tolerance is 5%, chosen deliberately: PR #31 changed `_ring_cells` to filter on
`ring_of(centre) == ring`, so §2b is *legitimately* stale — `eval_synthetic.py
--seq 07` prints **1.78** today against the doc's 1.76. The failure this gate
exists to catch is a ground-mask-less M\*, which reads **22.10 cm** on seq 07,
an order of magnitude out. 5% passes the first and catches the second.

---

## Result

| seq | schedule | support | truly non-drivable | misses | **miss rate** | 95% CI (Wilson) | false alarms |
|---|---|---|---|---|---|---|---|
| 07 | 5/10/20/40 | 1,724 | **19** | 8 | **42.11%** | [23.1%, 63.7%] | 71 |
| 07 | 5/10/50 | 1,723 | 19 | 8 | 42.11% | [23.1%, 63.7%] | 73 |
| 08 | 5/10/20/40 | 1,917 | **3** | 0 | **0.00%** | [0.0%, 56.2%] | 4 |
| 08 | 5/10/50 | 1,917 | 3 | 0 | 0.00% | [0.0%, 56.2%] | 4 |
| 00 | 5/10/20/40 | 1,914 | **38** | 4 | **10.53%** | [4.2%, 24.1%] | 59 |
| 00 | 5/10/50 | 1,914 | 38 | 4 | 10.53% | [4.2%, 24.1%] | 59 |

---

## ⚑ This is not a rate. It is a handful of cells.

**The denominators are 19, 3 and 38.** That is the whole finding, and it should
travel with the number everywhere it goes:

- seq 08's **"0.00%" is 0/3.** Its 95% interval runs to **56%**. It is not
  evidence of a safe map; it is evidence that this window had almost no
  hazards in it.
- seq 07's **42.11% is 8/19**, interval **40.6 points wide**.
- seq 00's 10.53% is the best-supported of the three and is still **±10 points**.

The three sequences disagree by 42 points and their intervals all overlap. **No
ordering between them is supported.** This is the same shape as the pothole
finding already in the handover — *"a demonstration, not a rate"* — and it
should be described the same way.

## ⚑ And it describes an 11 × 11 m patch, not a sequence

The planning window is `PLAN_N = 44` cells at `plan.cell_m = 0.25` m —
**11.0 m × 11.0 m**, placed at the vehicle's *final* pose (`final_vehicle_xy`).
40 frames of seq 08 is roughly 80 m of driving. So the measurement covers
**one square of ground at the end of the run**, not the route.

That is not a flaw in the harness — the planning window is what eq. (23) is
defined on, and this reuses it precisely so the verdicts are the same ones
plan-regret uses. But it does mean *"the hazard miss rate is X%"* is not a
sentence this number supports. *"On an 11 m window at the end of seq 07, 8 of
19 non-drivable cells were called drivable"* is.

## What would make it a rate

Sweep the window along the trajectory and pool, rather than sampling one
placement — the same fix `89d3551` applied to R(S) when it averaged 64 queries
instead of one, and for the same reason: *"one query was never an estimator."*
Pooling 40 placements over seq 08 would give a denominator in the hundreds.
**Not done tonight** — it needs a decision about placement spacing and overlap
handling, which is a design choice, not a measurement.

## Both schedules give the same answer

`5/10/20/40` and `5/10/50` are identical on 08 and 00 and differ by one cell of
support on 07. Expected: they share rings 0 and 1, and an 11 m window at the
vehicle sits entirely inside ring 0/ring 1 territory. **This measurement cannot
discriminate between schedules**, and should not be cited as if it could.

## False alarms, for context

The complementary error — truly fine, called a hazard — runs 4 to 73 cells.
On seq 07 there are **71 false alarms against 19 real hazards**, so the map is
considerably more likely to invent a hazard than to miss one. Conservative, and
the safe direction, but it is not free: every false alarm is ground a planner
will refuse to route through.

---

## Reproduce

Script: `scratchpad/r7_hazard_miss.py` (session scratch; not committed — it is
a measurement harness, not a deliverable). It gates on the ring-0 RMSE before
reporting, and masks the boolean verdicts directly rather than using
`restrict()`.

**Note for anyone re-implementing this:** `plan_regret.restrict()` builds
`CostMap(...)` **without `trav`**, which then defaults to `None` — so
`restrict(x).trav` is `None` and `restrict(x).low_confidence()` silently falls
back to `.unknown` alone, which is exactly the confound `baa44b4` fixed. No
live caller hits this (`--confound` reads `low_confidence()` off *unrestricted*
maps), so it is latent, not active. Logged, not fixed.
