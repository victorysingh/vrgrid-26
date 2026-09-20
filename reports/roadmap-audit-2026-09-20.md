# Roadmap audit — R1–R11 against what the repo actually contains

*2026-09-20. Read-only audit against `jp/p99-alloc-fixes` @ `6d16bb4` and `vrgrid26/main` @
`ae85979`. Every status below was checked against a commit, a file or a run — not against a status
line. Where a status line and the repo disagree, the repo wins and the disagreement is flagged.*

**The one-line finding: the roadmap's lane assignment no longer describes who does the work.** Of
the eleven R-items, JP authored the artifacts for seven and the design docs for two more; Shrestha
implemented one that was assigned to Aakash, and extended another. Two of the six named people have
made no commits at all this cycle.

---

## Part 1 — R1–R11 against the repo

### The correspondence table (old roadmap name → current tracker)

The project now runs two numbering systems, and they do not line up by name. `R3` the roadmap item
is `D2` the tracker item; `R2` the roadmap item is tracked as `D11 / R2`; and the tracker's own
`R-a … R-k` letters are a **third**, unrelated series created later for defects found during
reporting. Reading `R-b` as "roadmap item 2" is the mistake this table exists to prevent.

| roadmap | subject | current tracker id | artifact of record |
|---|---|---|---|
| **R1** | per-class / per-range accuracy table | *(closed row)* "FRNet accuracy by distance" | `reports/r1-accuracy-by-class-and-range-band.md` |
| **R2** | three-way plan-regret, model in the loop | **D11 / R2** | `scripts/plan_regret_frnet_delta.py` |
| **R3** | ring boundary under anisotropy | **D2** | `pending-review/r3-ring-boundary-under-anisotropy.md` → fixed in `df35fd5` |
| **R4** | CI test for the ring boundary | **D2** (same row) | `tests/test_lattice.py::test_no_cell_footprint_contains_another_under_foveation` |
| **R5** | sticky safety-critical (VRU) class bit | **D3** | `pending-review/r5-sticky-safety-critical-class-bit.md` |
| **R6** | ground-only variance | *(none — no trace)* | none found |
| **R7** | hazard miss rate with denominator | **D4** (wording only) | `reports/r7-hazard-miss-rate.md` |
| **R8** | README foveation claim | *(closed, pre-audit)* | `README.md:11`, `README.md:668` |
| **R9** | latency / VRAM table | **R-b**, **D8** descend from it | `reports/r9-per-stage-latency-and-memory.md` |
| **R10** | wide-depression gradient | *(none — no trace)* | none found |
| **R11** | limits page | *(closed row)* "R11 limits page" | `reports/r11-what-this-does-not-do.md` |

### Item by item

**R1 — per-class / per-range accuracy. DONE. Assigned JP+Hriday; done by JP alone.**
`7eb0d2d` (11 Sep, victorysingh) created the report. Later extended by JP as
`scripts/frnet_eval_by_range.py` + `a86b491`. ⚑ **Duplicated upstream:** Shrestha independently
shipped `26fe207` *"eval: classification accuracy against DISTANCE, which the statement asks for"*
(19 Sep) with its own `scripts/accuracy_by_range.py`. Two measurements of the same requirement, by
two people, neither aware of the other.

**R2 — three-way plan regret with a real model. DONE. Assigned JP+Hriday; done by JP alone.**
`a594255` (14 Sep) built `plan_regret_frnet_delta.py`; `4e81220` added paired per-query statistics
(+1.069, CI [+0.730, +1.436]; +0.567, CI [+0.301, +0.846]). Matches the plan.

**R3 + R4 — ring boundary and its CI test. DONE, but not by the assigned owner.**
JP wrote the design doc (`675d24b`, 12 Sep). The **implementation is Shrestha's** — `df35fd5`
(17 Sep), *"close Aakash's open items"*, whose own message says *"Done by Shrestha while Aakash was
away; every change to his lane is listed in `docs/handover-2026-09-17-aakash.md`"*. Verified by
reading the diff: ring membership now decided per world-lattice block, `bin_points` takes vehicle
position and heading, and R4's test landed as
`test_no_cell_footprint_contains_another_under_foveation` (parametrised by schedule) plus
`test_every_return_inside_the_map_is_binned`. Real closure, correctly handed over — but it is a
**lane breach that worked**, and it is why D2 is still sitting in §1 of this branch's OPEN-ITEMS
waiting on Aakash for something already done.

**R5 — sticky VRU bit. DESIGNED, NOT IMPLEMENTED.** `675d24b` (JP) wrote the design doc; nothing
implements it. Tracked as **D3**, still open, still a three-way decision. Assigned to no one in the
lane split that matches it — this is a grid-engine change, so JP's lane, but the class policy is a
room decision.

**R6 — ground-only variance. UNVERIFIABLE.** Zero mentions in any `.md` or `.py` in the tree. It was
reported closed before this audit's window and left no artifact behind. **Nothing here can confirm
or deny it.** If a number from R6 is still quoted anywhere, it has no traceable source.

**R7 — hazard miss rate. DONE (JP, `2811ebc`, 11 Sep)**, with the denominator stated, which was the
point of the item. What remains is **D4**: which of three drafted README wordings to use, and where.
JP said he would place it himself. Assigned to Aakash+Srinivas' "reporting" lane on paper; done by
JP.

**R8 — README foveation claim. DONE, pre-audit.** `README.md:11` carries the foveation claim and
`README.md:668` the *"does not claim to invent foveated or multi-resolution mapping"* disclaimer.
Intact.

**R9 — latency / VRAM table. DONE and since superseded twice.** `8239918` (11 Sep, JP) produced
`reports/r9-per-stage-latency-and-memory.md`. Shrestha extended it to the device in `adbe09d`
(*"R9 split/merge + pyramid rows, R4 cell cost, stage attrition"*). Its descendants are **R-b** (the
p99 tail) and **D8** (machine-state control), both still live. A self-contradiction inside R9 was
found and closed separately.

**R10 — wide-depression gradient. UNTOUCHED.** No commit, no file, no mention anywhere in the tree.
**This is the only roadmap item with no work of any kind**, and nothing in OPEN-ITEMS tracks it, so
it has been silently dropped rather than deferred.

**R11 — limits page. DONE (JP, `e4bd731`, 12 Sep).** `reports/r11-what-this-does-not-do.md`. Closed.

### Who actually did the R-work

| lane, as planned | planned owners | what actually happened |
|---|---|---|
| GPU / CUDA | Shrestha + Pratyushi | Shrestha alone. **Pratyushi (`Prathyushree`): 0 commits this cycle.** |
| DL + grid engine | JP + Hriday | JP alone. **Hriday (`zero-odds`): 0 commits this cycle.** |
| UI + everything else | Aakash + Srinivas | Aakash's last work was 14–15 Sep (dashboard). **His D2 was closed by Shrestha.** **Srinivas (`Shriniwas Gade`): 0 commits this cycle.** |

**Three of six named people have contributed nothing in this window.** All 29 upstream commits since
18 Sep are Shrestha's; all 83 unpushed commits on this branch are JP's. The project is currently two
people.

---

## Part 2 — D1–D11 and the R-letters, each verified

**Legend:** ✅ verified closed · ⚠️ marked closed, is not · 🔸 open · ❓ unverifiable

| id | status | verified how | waiting on |
|---|---|---|---|
| **D1** Patchwork++ lifetime | ✅ **DECIDED (A)**, 18 Sep | `tests/test_ground_reset_convention.py` 5/5; determinism 13/13. Upstream closed its §2 independently | nobody |
| **D2** ring boundary (R3/R4) | ✅ closed `df35fd5` | read the diff: lattice + 2 tests | nobody — **but still listed §1 on this branch** |
| **D3** sticky VRU bit (R5) | 🔸 open | design doc only, no implementation | the room |
| **D4** R7 README wording | 🔸 open | 3 drafts in `pending-review/` | **JP** |
| **D5** `alpha_m2` | 🔸 open, effectively answered | `alpha_m2: 0.0`; `test_alpha_would_break_the_flat_ground_remark` passes; **no reference map exists to calibrate against** | **JP** (nominally) |
| **D6** `kappa` 1/16 vs 1/12 | 🔸 open | both values pinned in `test_kappa_from_geometry_is_one_twelfth_at_every_ratio` | the room |
| **D7** CARLA scope | 🔸 open | zero code | **JP** |
| **D8** machine-state control | 🔸 open | harness exists; no reference machine agreed | **JP + Shrestha** |
| **D9** `transform_points` alloc | ✅ closed `12613df` | `reuse_buffers` + scratch present | nobody |
| **D10** ROS 2 adapter scope | 🔸 open | `pending-review/ros2-adapter-design.md` only | the room |
| **D11 / R2** FRNet + DL mode | ✅ **now complete** | real-time figure **exists**: `d48831a`, cuda+fp16 **p50 71.8 / p99 79.5 ms, 12.6 FPS, meets 10 Hz** on `5/10/50` | nobody |
| **R-a** commit the harnesses | ⚠️ **38 of 39** | `reports/harnesses/cleanup_tail_probe.py` has **no PROVENANCE header** | JP (one line) |
| **R-b** p99 tail | ⚠️ **two incomparable closures** | upstream closed at p99 28.40 ms (`--device cuda --schedule 5/10/50`); this branch measured 87.69 ms (CPU, `5/10/20/40`) | **JP** |
| **R-c** stdout UTF-8 | 🔸 staged, not applied | diff applies; 20/20 executed; **covers 20 of 27 entry points** now | **JP** |
| **R-d** ruff E741 | ⚠️ **false closure** | line 472 unchanged; ruff 0.12.0 → **10 errors** on upstream's tree; CI ruff **unpinned** | **JP / Shrestha** |
| **R-e** 99.87% vs 99.5% | ✅ closed | recomputation recorded | nobody |
| **R-f** seq 00 ring-1 residual | ✅ closed | singleton carry-over explanation | nobody |
| **R-g** perception invariant | ✅ closed | `test_no_allocation_inside_the_frame_loop` present | nobody |
| **R-h** cleanup/ground tail | 🔸 open | `d540618`, `697a2bd` unreviewed | **Shrestha** |
| **R-i** unify timing printers | 🔸 decision doc | snapshot net landed (`tests/test_timing_table_printers.py`, 8 tests) | **JP** |
| **R-j** `--fast-scatter` repro | 🔸 **CPU closed, GPU still open** | no upstream commit answers the CUDA half | nobody has claimed it |
| **R-k** `ground_cost.py` reset | ✅ closed, both halves | agreement 96.5% either way; timing +0.14 ms (0.7%) | nobody |
| **N-3** seq 00 ring 2 | 🔸 hypothesis refuted | `crosslook_probe.py` | Shrestha (cause open) |
| **N-4** stale GitHub #6 | ✅ closed after verifying | — | nobody |
| **S-1…S-5** | 🔸 new, 20 Sep | upstream §5c | see Part 4 |

### New false-closure instances found this pass

1. **R-a — "Done, each with a PROVENANCE header."** 38 of 39. `cleanup_tail_probe.py` lacks one. Small,
   but R-a exists *specifically* because untraceable numbers were the recurring defect, so an
   unprovenanced harness is the exact failure the item was created to prevent.
2. **D2 is stale in the other direction.** It is genuinely closed upstream, and this branch still
   lists it in §1 "waiting on a decision" with Aakash as owner. The same staleness upstream warned
   about, mirrored back at them.
3. **R6 and R10 are invisible.** R6 was closed with no surviving artifact; R10 was never started and
   nothing tracks it. Neither appears in OPEN-ITEMS, so neither can surface on its own.

(R-b and R-d were already found in the previous session and are re-confirmed above, not re-counted
as new.)

---

## Part 5 — Is the roadmap working?

### 1. Converging or diverging? **Diverging on coordination, converging on substance.**

The substance is genuinely good: D1 decided and enforced, D2 closed with tests, D9/R-e/R-f/R-g/R-k
closed and verified, T4 parity identical on 200 frames, the DL pipeline at 12.6 FPS. Item count is
falling.

But **the collision rate is rising, and it rose in the week the two active people were most
productive.** Every collision below happened between 14 and 20 September.

### 2. Duplicate-work incidents this cycle: **four, plus one near-miss**

| # | incident | evidence |
|---|---|---|
| 1 | **`--semantics frnet`** implemented twice | JP's branch and upstream `81843cc` — same flag name, same `choices`, same default |
| 2 | **Distance-accuracy measurement** done twice | JP `a86b491` + `frnet_eval_by_range.py`; Shrestha `26fe207` + `accuracy_by_range.py` |
| 3 | **AWS runbook** written twice | JP `aws-gpu-realtime-runbook.md`; Shrestha `docs/gpu-lane/02-AWS-RUNBOOK.md` (already demoted 18 Sep) |
| 4 | **The DL real-time figure** | JP scoped an AWS run and a measurement protocol for it; Shrestha produced the number on his own GPU (`d48831a`) before it ran |
| near-miss | parallel knobs on one flag | JP added `--fast-scatter`/`--threads`; Shrestha added `--semantics-precision`/`--semantics-every` |

**Common root cause — there are two, and the second is the one being under-weighted.**

- **Invisibility.** 83 commits on `jp/p99-alloc-fixes` have never been pushed. Work that cannot be
  seen gets rebuilt. This explains incidents 1, 2 and 4.
- **The shared status file is not trustworthy, and both sides now know it.** Upstream's own
  reconciliation commit says it plainly: *"This list said D1 blocked the project for over a week
  after the gate had started passing, and two people planned around that."* This audit found the
  same defect pointing the other way (D2 stale here) and three more (R-a, R6, R10). **OPEN-ITEMS.md
  is being used as a coordination mechanism while being wrong often enough that people route around
  it** — and routing around it is exactly what produces duplicate implementations.

Incident 3 is the tell: both parties wrote a runbook *because each believed the other had not*.

### 3. Against the 10-day plan: **ahead on depth, behind on breadth, and one item silently dropped**

- **Ahead:** R1, R2, R7, R9, R11 delivered, several with statistical rigour the plan did not ask for
  (paired CIs, bootstrap intervals, bit-identity proofs). The GPU lane is far ahead — device parity
  across two machines and a 10 Hz DL pipeline were not plan items at this depth.
- **On track:** R3+R4 (late, and by the wrong person, but done and tested), R8, D1.
- **Behind:** **R5/D3** (designed, never implemented), **D10 ROS 2** (scoping doc only — an entire
  assigned lane deliverable), **D3/D6** (room decisions nobody has convened).
- **Dropped:** **R10, wide-depression gradient. Zero work, zero tracking.** Nobody has raised it
  since the plan was written.
- **Unverifiable:** **R6**, closed with no artifact.

The pattern is that **every item assigned to a person who has not committed this cycle is the item
that slipped.** R5, R10 and D10 are not hard; they are unowned in practice.

### 4. What should change — plainly, yes, three things

1. **Push `jp/p99-alloc-fixes` today, even unfinished, even as a draft PR.** This is the single
   highest-value change and it costs nothing technical. Three of four duplicate incidents trace
   directly to those 83 commits being invisible. Every day it stays local is another day someone can
   rebuild work that already exists. It is JP's call because it is outward-facing — but the cost of
   *not* doing it is now measurable in duplicated days.
2. **Stop using OPEN-ITEMS.md as the coordination channel without a freshness rule.** Both sides have
   now planned against stale rows. Upstream already added the right rule to their copy — *"Re-run a
   gate before believing any status line here"* — and it should be adopted on both, with every
   status line carrying the date it was last verified and by what command. A status file that is
   wrong 5 times out of ~30 rows is worse than no status file, because it is trusted.
3. **Claim an item before building it, in one visible place.** The `--semantics frnet` collision was
   two people solving the same well-specified problem in the same week. Neither did anything wrong;
   there was simply nowhere to see "I am doing this now". A one-line claim in the tracker — name,
   date, item — would have prevented incidents 1, 2 and 4 outright.

A fourth, weaker suggestion: **pin ruff.** CI runs bare `ruff check .` with no version anywhere, so
"CI is green" is a statement about which ruff GitHub installed that morning. That is how R-d came to
be marked closed while ten errors were live.
