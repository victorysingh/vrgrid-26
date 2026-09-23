# Number-provenance audit — every load-bearing figure traced to source

*2026-09-20. Read-only tracing against `jp/p99-alloc-fixes` @ `1d24593` and `vrgrid26/main` @
`ae85979`. A claim is "verified" only if a committed command, script or artifact produces it and the
recipe is stated where the number appears. Status lines were not trusted; R-b and R-d had already
been caught once, and this pass applies the same standard to every row.*

**Headline: the project's two most quotable GPU figures — the one used to close R-b, and the 10 Hz
DL claim — have no committed measurement artifact and name no machine.** Both are almost certainly
real; neither can be re-run from what is in the repo. Separately, the one CUDA timing log that *is*
committed reaches the opposite verdict on a different schedule, and nothing in prose mentions it.

---

## A. Method note — a regex error in my own earlier pass

My first sweep searched for figures with patterns like `71.8`, where `.` matches **any** character,
so `479.51` matched `79.5` and the hit lists were inflated. Every figure below was re-checked with
escaped patterns (`71\.8`). Where an earlier session reported a number appearing "in 9 files", treat
that as wrong and this table as the correction.

## B. The audit table

**Key:** ✅ verified · ⚠️ needs correction · 🚩 provenance incomplete · ❓ cannot verify

| # | claim | where it appears | source traced to | recipe complete? | status |
|---|---|---|---|---|---|
| 1 | **p50 21.94 / p99 28.40 ms**, 45.6/35.2 FPS, "3.5× headroom" — used to **close R-b** | upstream `OPEN-ITEMS.md`, `docs/gpu-lane/13-PS-SCHEDULE.md`, `docs/research-log.md:561` | **prose only — no `.log` or `.json` in the repo contains either number** | recipe partly: seq 08, 200 frames, `5/10/50`, `--device cuda`. **Machine named only in `research-log.md`, not in `13-PS-SCHEDULE.md` or the OPEN-ITEMS row** | 🚩 **provenance incomplete** |
| 2 | **p50 71.8 / p99 79.5 ms**, 13.9/12.6 FPS — the **10 Hz DL claim** (`d48831a`) | upstream `OPEN-ITEMS.md`, `research-log.md:592`, commit body | **prose only — no artifact holds the pair** | seq/frames/schedule `5/10/50`, `--semantics frnet --semantics-precision fp16`, `--device cuda`. **No machine named anywhere, including the commit** | 🚩 **provenance incomplete** |
| 3 | **p50 67.42 / p99 100.18 ms — "MISSES 10 Hz at p99"** | `docs/gpu-lane/t4/timing_cuda.log` **only** | the committed T4 run: seq 08, 200 frames, `5/10/20/40`, device cuda | ✅ fully stated in the log header | ⚠️ **verified but unreported** — no prose document mentions this result anywhere |
| 4 | Hardware for the T4 column | `docs/gpu-lane/t4/host.log` | `nvidia-smi`: **2× Tesla T4, 15,360 MiB, driver 580.159.04, CUDA 13.0** | ✅ | ✅ verified — and consistent with Kaggle's free "T4 ×2" |
| 5 | "**AWS**" as the GPU environment | `reports/aws-dl-realtime-addendum.md`, `docs/gpu-lane/02-AWS-RUNBOOK.md`, `11-AWS-RESUME.md`, `scripts/aws/*`, my OPEN-ITEMS AWS row | `research-log.md:557`: *"the T4 column, on **Kaggle rather than AWS**… AWS was abandoned, not deferred… every GPU quota reads 0… **the whole T4 pass ran on Kaggle's free T4 (15.6 GB)**… Lifetime AWS spend **$0.00**"* | — | ⚠️ **needs correction** — see §C |
| 6 | Laptop GPU: RTX 5050 Laptop, 8,151 MiB, sm_120, driver 610.57.04 | `06-DAY3`, `07-LOCAL-BUILD`, `09-VRAM-CONTENTION`, `10-R9-R4`, `laptop/host.log` | `nvidia-smi` | ✅ | ✅ verified — the lane *does* name hosts elsewhere, which is why 1 and 2 stand out |
| 7 | **Pooled p99 89.58 ms**, p50 79.74, 3/1000 over 100 ms | my `OPEN-ITEMS.md` R-b row, `MORNING-SUMMARY-3/5` | `36a4dbe` | ✅ seq 08, 5 fresh processes × 200 frames, CPU, `5/10/20/40`, state-gated, per-run p99s listed | ✅ verified (scope since narrowed — see 8) |
| 8 | **Pooled p99 87.69 ms** warm / **101.70 ms** cold-start | my `OPEN-ITEMS.md`, `reports/r-b-post-merge-p99.md` | `6d16bb4` | ✅ 10 runs as 2×1,000 frames, CPU, `5/10/20/40`, state OK before *and* after each, both percentile methods shown | ✅ verified |
| 9 | **R-k**: agreement 96.5% vs 96.5%; masks differ 5,878/2,471,164 (0.2379%); timing 19.26 vs 19.40 ms | my `OPEN-ITEMS.md` | `1884358`, `10ba897`; `ground_estimator_carryover.py`, `ground_cost_reset.py` | ✅ seq 08, 60 scans, 20-frame window; 3 processes/config; state gated | ✅ verified |
| 10 | **R-a "Done, each with a PROVENANCE header"** | my `OPEN-ITEMS.md` | 39 harnesses | — | ✅ **now** accurate — was 38/39 until `1d24593` |
| 11 | ~~**R-d "`ruff check .` passes clean"**~~ **RETRACTED, see below** | upstream `OPEN-ITEMS.md` | `tests/test_metrics.py:472` unchanged; ruff 0.12.0 on `ae85979` → **10 errors** | CI runs bare `ruff check .`, **no version pinned** | ⚠️ **still a false closure** — re-confirmed today, not merely "previously caught" |
| 12 | **D2 / R3+R4 closed** | both copies | `df35fd5` diff read: lattice change + `test_no_cell_footprint_contains_another_under_foveation` + `test_every_return_inside_the_map_is_binned` | ✅ seq 08, 30 frames: 0.108% nested footprints, 0.224% returns dropped, both now 0 | ✅ verified |

| 13 | **D9** transform p99 22.07 → 1.45 ms | my `OPEN-ITEMS.md`, `pending-review/transform-points-allocation.md` | `12613df` | ✅ stated in the pending-review doc | ✅ verified |
| 14 | **D11/R2** plan regret +1.069 (CI [+0.730, +1.436]) and +0.567 (CI [+0.301, +0.846]) | my `OPEN-ITEMS.md` | `4e81220`, `scripts/plan_regret_frnet_delta.py` | ✅ real seq 08, paired per-query, seeded bootstrap | ✅ verified |
| 15 | **FRNet 90.3% point accuracy / 65.2% mIoU**, 200 frames seq 08 | `CLAUDE.md`, `checkpoints/…PROVENANCE.md`, gpu-lane docs | checkpoint provenance file + `scripts/frnet_eval.py` | ✅ frames and sequence stated; **CLAUDE.md explicitly warns 51.5% and 69.8% are arithmetic errors still in circulation, and 98.3% is a single-frame check** | ✅ verified, with a live ambiguity the doc itself manages |
| 16 | **FRNet determinism: "it is the model"**, CPU vs CPU 0.028–0.046% | upstream `research-log.md:594`, `d48831a` | measured over 3 frames | thread count **not stated** | ⚠️ **contradicts R-j** — see §D |
| 17 | Ground stage **21.01 ms p50 vs 0.33 ms fallback, 63.6×, 96.5% agreement** | my `OPEN-ITEMS.md` R-a/R-k rows, `reports/latency-gap-investigation.md` | `reports/harnesses/ground_cost.py 08 60` | ✅ now, incl. the carry-over caveat from R-k | ✅ verified; today's tree gives ~19.3 ms (logged as a machine/day difference, not a correction) |
| 18 | Memory **8.94 MB (5/10/20/40)** / **6.24 MB (5/10/50)**, 286× vs dense 3D | `README`, `MORNING-SUMMARY-5`, `13-PS-SCHEDULE` | `scripts/memory_table.py` | ✅ cell counts and 12 B/cell shown | ✅ verified |
| 19 | Test counts **821 passed** (mine) / **749 passed** (upstream) | this session's reports | both suites run; upstream run under an isolated import root | ✅ | ✅ verified |
| 20 | **N-3** seq 00 ring 2 ρ 2.20 → 1.02 | both copies | `scripts/crosslook_probe.py`; hypothesis **refuted** `8854b47` | ✅ | ✅ verified as *open with a refuted hypothesis* |


> **[CORRECTION 2026-09-24 — the R-d finding above is WRONG and is retracted.]**
> The claim that R-d was a false closure came from running **ruff 0.12.0**, an older
> version than CI installs. Re-tested with **ruff 0.16.8** — the exact version CI
> installs — in a clean venv, `--no-cache`, against `Stxtics03/vrgrid-26@8acbfe9`:
> **`All checks passed`**. Shrestha's closure (*"`ruff check .` passes clean on `main`;
> it is a CI gate and it is green"*) is **true and verified**.
>
> `pyproject.toml` sets no `[tool.ruff.lint] select`, so ruff uses its **default rule
> set**, and that set changed **in both directions** between versions: 0.16.8 drops the
> `E7xx` family from defaults (which is why `main`'s `E731`/`E741`/`E702` vanish) while
> enforcing `I001`, `RUF100`, `PLW1510` and `RUF059`. "Newer is stricter" was the wrong
> model.
>
> **What is true instead:** `jp/p99-alloc-fixes` carries **69 errors under CI's own
> ruff** — real lint debt in JP's own files (`plan_regret_frnet_delta.py` 10,
> `whole_frame_bench.py` 6, `test_plan_regret_frnet_delta.py` 5, `rmse_baseline_probe.py`
> 4, …), not inherited from `main` and not version drift. That is JP's to clean up.
> Pinning ruff in `ci.yml` remains worth doing so it cannot drift, but it explains
> nothing here.

### Items that cannot be verified

| claim | why |
|---|---|
| **R6 — ground-only variance** | Closed before this window with **no artifact**. Literal `R6` appears **0** times; `ground-only` appears twice, both unrelated (`research-log.md:541` regret step, `src/gpu/attrition.py:13` "height is ground-only"). ❓ **Cannot verify — but nothing cites it, so no untraceable number is in active use.** |
| **R10 — wide-depression gradient** | No commit, file or mention. Now tracked as unassigned (`1d24593`). ❓ n/a |

## C. Correction 1 — AWS never happened; the GPU work was Kaggle

**The repo already records this, in one place only.** `docs/research-log.md:557` (*"Module: D3 — the T4
column, on Kaggle rather than AWS"*) states that AWS was **abandoned, not deferred**: every GPU quota
on the account read 0, the increase request was refused because the free plan has no GPU tier, the
whole T4 pass ran on **Kaggle's free T4 (15.6 GB)** at zero cost, and **lifetime AWS spend was
$0.00** with the bucket and guardrails torn down.

That fact never propagated. Classification of every AWS reference:

| file | what it is | action |
|---|---|---|
| `docs/gpu-lane/11-AWS-RESUME.md` | **a plan, correctly labelled** — written while the account was still waiting to activate, and it documents the `OptInRequired`/`NotSignedUp` block | leave; **superseded note needed** (Shrestha's file — flagged, not edited) |
| `docs/gpu-lane/02-AWS-RUNBOOK.md`, `scripts/aws/*` | runbook + scripts, **never executed** | leave; superseded note needed (Shrestha's) |
| `reports/aws-dl-realtime-addendum.md` | **mine** — a measurement protocol for a run that never happened | **superseded note added** (Part 4) |
| my `OPEN-ITEMS.md` AWS row | says the lane is blocked on account activation | **corrected** (Part 4) |
| `dashboard/*.py`, `docs/data-access/…`, `sih-math.md` etc. | matched on the substring "aws" inside other words | no action |

**No figure was found that states it was measured "on AWS".** The provenance error is the opposite
one: the numbers say nothing about where they ran, and the *plans* say AWS while the *work* was
Kaggle. **Do not describe the T4 figures as AWS numbers, and do not describe Kaggle's card
generically** — it is specifically 2× Tesla T4 15,360 MiB, which `t4/host.log` records.

## D. The R-j contradiction (new this pass)

Upstream attributes FRNet's run-to-run label disagreement (0.028–0.046% of points) to **"the
model"** (`research-log.md:594`, `d48831a`), and does not state a thread count. **R-j on this branch
measured the opposite cause:** at PyTorch's default intra-op thread count neither `--fast-scatter`
nor the port's loops reproduce per point, and with **`torch.set_num_threads(1)` both are exact and
identical**. Both observations are consistent *only* if upstream's runs used the default thread
count — in which case the cause is the thread pool, not the model.

This is not a number error; it is an **attribution** error of the same class as the one withdrawn in
`1a47960` earlier in the project. It also strengthens Correction 2: the `--threads` knob is not a
convenience, it is the thing that makes the DL mode reproducible.

## E. What this audit changes about confidence

- Nothing on **this branch** failed verification. Every figure traced to a commit with its recipe.
- The two **upstream** GPU headlines are unbacked by artifacts. They should not be quoted in a
  submission until either the raw log is committed or the run is repeated with `--frame-times`.
- **The single most quotable sentence available today** — "the DL pipeline meets 10 Hz" — is exactly
  the one with the weakest provenance. That is worth fixing before it reaches a slide.
