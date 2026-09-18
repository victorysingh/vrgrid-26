# Open items — the master list

Single place for what is outstanding, so nothing falls between sessions.
Assembled 2026-09-12; **cleaned 2026-09-16** on `jp/p99-alloc-fixes`. The clean-up:
- **Stale and contradictory fragments removed.** Rows D11, R-j, R-c, R-i, R-h and R-b had accumulated
  notes appended over time that contradicted one another; each row now states its current status
  once.
- **Rows moved to where they belong.** Open items that sat in "Closed" (R-c, R-i, R-h) now sit under
  open work, and D9 (applied in `12613df`) moved to closed.
- **Rows added** for the DL mode, the distance-accuracy evaluation and the AWS runbook.

Nothing measured was dropped. The full history of every item is in `git log` and the reports each row
cites.

**How to read the status column:** *Decision* = waiting on a person, nothing to do until then.
*Staged* = built and verified, not applied. *Blocked* = waiting on access, a machine or time.

---

## 1. Waiting on a decision

| # | item | waiting on | why | file |
|---|---|---|---|---|
| **D1** | **The Patchwork++ singleton lifetime.** Does `segment_ground` take an estimator? Does the module expose a reset? Who owns the lifetime? | JP (design) | **[2026-09-18, VERIFIED after a fetch] The point-fix landed UPSTREAM, not on this branch.** It is **`3c26d47`** (14 Sep, **AakashH2006**), *"determinism: one Patchwork++ estimator was shared by every run in a process"* — **not** `03abd7b`, which is Shrestha's GPU frame-loop commit. Upstream calls `ground.reset_estimator()` inside `iter_pipeline` **before** the frame loop (`src/run/__main__.py:98`, loop at :100), i.e. **once per run, not per frame**, so it does not sit in the timed per-frame path; `harness.py:366`, `feature_report.py:84`, `gen_demo_rrds.py:82` and `gpu_parity.py:76` do the same, and `tests/test_ground.py` pins it. Our own determinism test still fails locally, because this branch does not contain that commit. **Nothing on `jp/p99-alloc-fixes` fixed D1** — every harness here resets the module global itself (`ground._estimator = None`) without editing `ground.py`. **The open question is unchanged and is NOT the point-fix:** who owns the estimator's lifetime, whether `segment_ground` should take one, and whether the API should expose an explicit reset. A reset at entry points is a call-site convention, not a lifetime design, and it leaves the same trap for any new caller that forgets it. See `reports/pre-merge-collision-check.md`. Three real options, none obviously right. **Highest-impact item on this list** — see §2. The determinism test `test_real_sequence_replay_is_identical` fails because of it. | `reports/ring1-reproduction-investigation.md` §7 |
| **D2** | **R3 — ring boundary under anisotropy.** Nearest-corner test + boundary snapping. | Aakash | **[2026-09-18] Reported CLOSED upstream** by `df35fd5` (17 Sep, Stxtics03): *"grid+eval: close Aakash's open items — D2 ring boundary, ring-0 rho, per-ring reference"*. Verified only as a commit on `vrgrid26/main`; its behaviour was not re-tested here. Nothing on this branch touches `lattice.py`. Touches `src/grid/lattice.py` (his) and has a cross-lane consequence in `metrics._ring_cells`. Relevant to SIH26053's "without alignment errors or data loss". | `pending-review/r3-ring-boundary-under-anisotropy.md` |
| **D3** | **R5 — sticky safety-critical class bit.** Which classes (5/6/7 only, or also 1/2)? Which of three decay designs? What is N (frozen `configs/thresholds.yaml`)? Does it affect traversability? | Aakash + three-way | Five sub-decisions; touches frozen `include/vrgrid/cell.py`, so three-way sign-off. | `pending-review/r5-sticky-safety-critical-class-bit.md` §6 |
| **D4** | **R7 README wording** — which of three drafted lengths, and where it goes. | JP | JP said he would place it himself. | `pending-review/r7-readme-counts-draft.md`, `r7-readme-wording.md` |
| **D5** | **`alpha_m2` — raise it or leave it at 0.** | JP | 0 is the only value consistent with sih-math §5.3 and the §5.4 unit test; raising it means restating both in the same commit. | `reports/r11-what-this-does-not-do.md` §4 |
| **D6** | **`kappa` 1/16 vs 1/12.** Knowingly 25% low; the stated geometry gives 1/12. | The room | Frozen config, explicitly "the room decides". Both values pinned in tests. | `configs/thresholds.yaml:133` |
| **D7** | **CARLA groundwork — scope.** | JP | Zero existing work; no scope supplied. Starting a simulator integration from a guess is the wrong move. | `MORNING-SUMMARY.md` §3 |
| **D8** | **Machine STATE control for latency, not just an agreed hostname.** | JP + Shrestha | Two honest end-to-end figures exist on two hosts: `README.md` 89.18 / 100.43 ms (Shrestha's i7-14650HX) and this laptop (i7-13620H). **One host also varied 4.4× in one session** (21.01 → 93.44 ms in `ground_cost.py`, with the CPU downclocked to 1520/2400 MHz and ~8 GB over-committed), so agreeing a hostname is not enough. It needs a checked machine state recorded with every figure: CPU clock vs max, commit vs physical RAM, and a warm-up/idle precondition. Controlled A/B ratios survive this; headline absolutes do not. The laptop harnesses now record state per run (`reports/harnesses/machine_state.py`). | `reports/r-b-p99-tail-investigation.md` §11; `docs/handover-2026-09-02.md` |
| **D10** | **ROS 2 adapter scope** — window extent, bitfield as one float layer or six boolean ones, where live labels come from, publish rate, and whether `export_gridmap()` needs arguments. | Three-way | The last changes a **frozen signature**. **The blocker is throughput:** `query()` is 54.7 µs, so a 20 m window at 5 cm is 8.75 s per frame (0.11 Hz), and a vectorised bulk reader pinned bit-identical to `query_region` is a prerequisite. **Method decided 2026-09-13:** transcribe `query()` rather than re-derive it. **Step 1 (transcription) done as prep.** Only `_refined` is new work, and `query()` contains no reduction (no ULP risk). Step 2 not started. Two traps are **named checks** (§6a): `test_bulk_query_keeps_the_two_outside_paths_independent` (the two OUTSIDE paths diverge on 361 of 361 swept points when a window is un-tracked; `OUTSIDE == -1`, so a conflated mask silently reads the last ring) and `test_bulk_query_masks_before_gathering`. | `pending-review/ros2-adapter-design.md` §4, §7; `pending-review/query-vectorisation-transcription.md` |
| **R-c** | **stdout UTF-8 at script entry points — apply or not.** | JP | **Staged, not applied** (`d0d84c3`). JP chose an in-code entry-point fix over `PYTHONUTF8=1`; `pending-review/r-c-stdout-utf8-entry-points.diff` covers 20 scripts. Verified: UTF-8 output byte-identical on 18/18 executed scripts, all 12 cp1252 `--help` crashes fixed, script tests 31 passed / 1 skipped on the patched tree. Not executed: `frnet_fast_scatter`, `gen_demo_rrds`. Does not cover `python -m vrgrid.run`, harnesses, or `open()` defaults. | `pending-review/r-c-stdout-utf8-entry-points.md` |
| **R-i** | **Unify the synthetic and real timing-table printers — or not.** | JP | **Decision doc, nothing implemented** (`d0d84c3`). The split caused both the 80.78 ms and 8.15 → 1.31 MB mislabellings. Recommendation: unify the printers, keep `run()` / `run_real()` separate, snapshot-test both current tables first, and update `whole_frame_bench.py`'s row regex in the same commit. | `pending-review/r-i-unify-timing-table-printers.md` |
| **PUSH** | **Push `jp/p99-alloc-fixes` and open a PR.** | JP | 38+ local commits, never pushed; no upstream. Also decides how code reaches the AWS instance (the runbook defaults to a git bundle). | — |

---

## 2. [!] The one that blocks other work

**D1, the Patchwork++ singleton.** Flagged separately because it is currently **distorting measurements
other decisions depend on**:

- `test_real_sequence_replay_is_identical` fails, correctly. Two replays in one process differ by
  1,245 of 1,479,013 points. Resetting the estimator between replays makes them identical (verified
  2026-09-14).
- **[2026-09-18]** It still fails **in this clone**, which does not contain the upstream point-fix (`3c26d47`, AakashH2006, verified present on `vrgrid26/main` after a fetch). That fix is a per-run `reset_estimator()` at the entry points — a call-site convention, not the lifetime design this row is about.
- **~18% of the published seq-07 ring-1 RMSE is artifact**, not coarsening error (3.60 → **3.04 cm**
  once the reference and the map use a consistent ground mask).
- **No ring-1 accuracy comparison finer than ~0.5 cm can be adjudicated until it is fixed.**

The fix direction is measured, not speculative: two estimators making one pass each agree exactly, with
0 of 1.48 M points differing. Every harness written this week resets the estimator before each pass
(`ground._estimator = None`) without editing `ground.py`.

---

## 3. In progress, staged, or blocked on access / people

| # | item | status | waiting on | evidence |
|---|---|---|---|---|
| **AWS** | **The DL pipeline's real-time figure, and GPU reproducibility** | **Blocked / paused by JP (2026-09-14).** Needs an AWS g4dn.xlarge; no local CUDA build by decision, and this laptop has no AWS access. **Runbook revised 2026-09-16 after a pre-launch audit:** git-bundle transfer (the branch is unpushed), frames 0–200 (494 MB), explicit CUDA torch + `.[perception]`, a Patchwork++ check, an on-instance 2-frame preflight, Linux-correct state rules, results fetched before termination, and a termination checklist. The laptop-testable parts were tested. | JP (launch the instance) | `reports/aws-gpu-realtime-runbook.md` |
| **D11** | **FRNet: checkpoint, verification, DL mode** | **Checkpoint acquired, verified, DL mode built; real-time GPU number still pending.** See the breakdown below. | the AWS run | `reports/d11-frnet-reverification-r2.md` |
| **R-j** | **`--fast-scatter` reproducibility** | **CPU: RESOLVED. GPU: OPEN, not assumed.** See the breakdown below. | the AWS run (steps e, f) | `reports/bench/rj_round2_pred_diff.json` |
| **R-h** | **`cleanup` / `ground` tail** | **`cleanup`:** the LUT guard (`d540618`) and the `_centres` fast path (`697a2bd`) are applied on the branch and **await Shrestha's review** (`engine.py` is his). Instrumenting every line showed `_centres` (7.09 MB, 10.78 ms p50) was the stage's largest allocator, with `np.isin` (6.40 MB) second; an earlier claim that `np.isin` set the peak was corrected in `dce017a`. After both fixes no single cleanup line spikes on slow frames, and `visibility_cleanup`'s ~11.5 ms is float compute, not allocation (not attempted: ULP risk). **`ground`:** in isolation Patchwork++ has p99−p50 ~1 ms, against 11.13 ms in-pipeline in an earlier measurement; in the later clean-machine benches its p99−p50 was ~1.4–2 ms at baseline and after. **Answered 2026-09-16 from existing trusted data, no new timing needed:** every whole-frame bench recorded a per-stage `ground` row with machine state checked. In-pipeline `ground` p99−p50 is a median **1.40 ms before any fix** (3 runs) and **2.46 ms after all fixes** (10 runs), max 7.67 ms. The 11.13 ms never reproduces on a trusted run, so the allocation hypothesis is **not supported** (the tail was already small before the fixes). The 11.13 ms came from a pre-gating probe with no state record: machine state is the likely, but unproven, cause. Rare slow frames (26–33 ms against p50 ~19.5) remain unattributed. `reports/r-b-p99-tail-investigation.md` §13, `reports/harnesses/ground_inpipeline_spread.py`. | Shrestha (review of the `cleanup` changes) | `pending-review/cleanup-isin-guard.md`; `reports/r-b-p99-tail-investigation.md` |
| **UPSTREAM** | **`vrgrid26/main` has moved a long way past this branch** | **Fetched 2026-09-18 (read-only, JP authorised): `7dee296`, 59 commits we do not have.** Shrestha ported the frame loop and then the whole pipeline to the card (`03abd7b`, `6af6907`; upstream reports 22 ms/frame, bit-identical to CPU), closed D2 (`df35fd5`), fixed two regret-costmap bugs, rebalanced the 8 m band, and scripted the AWS T4 lane (`scripts/aws/t4.sh`, `auto.sh`); Aakash rebuilt the dashboard and fixed the estimator sharing (`3c26d47`). **Dry-run merge (disposable worktree, nothing kept):** conflicts in `src/run/__main__.py` (6 hunks, in `iter_pipeline`/`perceive`/`main`), `scripts/timing_table.py` (3), `dashboard/__main__.py` (2), `OPEN-ITEMS.md` (2), `README.md` (1). **`src/run/engine.py` auto-merges** — upstream did not touch `_cleanup` or `_centres`, which are byte-identical to our base. A clean auto-merge is not proof of correctness; the merged file must be run. **Consequences:** our AWS runbook is superseded by theirs; R-b's pooled p99 89.58 ms describes the pre-port CPU path and needs restating, not retracting; the DL-mode wiring must be re-expressed against upstream's new `perceive()`. | JP (merge strategy) | `reports/pre-merge-collision-check.md` |
| **README** | **One README / report writing pass** | Waits for the AWS numbers, so every caveat lands together: motion from ground truth, the 50 m label ceiling, the real-time figure's hardware and GPU reproducibility, R-j's thread-count condition, and the opt-in DL mode. README still describes a ground-truth-label-only pipeline. | JP, after AWS | — |

**D11 — breakdown.** Offline evaluation plus an opt-in pipeline mode. The live default stays ground truth.
- **Checkpoint:** the 4 Sep file is gone (Shrestha doesn't have it; never tracked). The authors' public
  release was downloaded 2026-09-14T07:27:54Z, SHA-256 `09adea90…285e`, corroborated by Google's crc32c
  and by an independent browser download. It is **not** the 4 Sep file. `eff969a`, `8e20d57`.
- **Accuracy (Step 4):** 90.3% point accuracy / 65.2% mIoU (15 classes) / 61.1% drivable on seq 08,
  200 frames, matching the recorded figures. Fine-tuning was skipped: the figures are the pretrained
  checkpoint's. `4675e95`.
- **Plan-regret delta (Step 6):** reproducible at `--fast-scatter --threads 1`, two bit-identical runs.
  FRNet labels raise plan regret from 1.160 to 2.230 (**+1.069**) longitudinal and from 1.142 to 1.710
  (**+0.567**) lateral. The ground-truth arm is identical to the passed oracle control. This supersedes the
  10-thread +0.997 / +0.449. `5c19b96`. **Paired per query (2026-09-16):** longitudinal +1.069, 95% CI [+0.730, +1.436], 49/1/14 worse/equal/better, sign p=1.1e-05; lateral +0.567, 95% CI [+0.301, +0.846], 25/31/8, sign p=0.0046. Both CIs exclude zero. The 10-thread lateral value lies inside the CI, so the earlier sensitivity is bounded. The CI covers the 64 queries on one map and one slice, not other scenes.
- **Opt-in DL mode:** `--semantics frnet` in the pipeline, dashboard and timing table; motion stays
  ground truth. Its labels equal `frnet_eval.py`'s model on a real frame. `a26f7f0`.
- **Accuracy by distance:** `--threads 1`, reproducible. 93.2% (0–10 m) / 88.1% (10–25 m) / 85.4%
  (25–50 m). SemanticKITTI seq 08 has no labels beyond 50.0 m. `a86b491`.
- **Still pending:** the DL pipeline's real-time figure on GPU (AWS).

**R-j — breakdown.**
- **CPU, resolved (`b20e03e`):** the run-to-run variance is PyTorch's intra-op thread pool, on both
  paths. With `torch.set_num_threads(1)`, `--fast-scatter` equals the port's loops on every point and
  repeats exactly (0 of 2,471,164 points differ across four passes). At the default 10 threads
  neither path repeats: 1,895 points (fast) and 1,446 points (loop) differ between runs, with headline
  metrics unchanged. The reproducible like-for-like speedup is about 7.0× (1 thread, 20 frames:
  129 s vs 898–914 s). The earlier 5.82× (10 threads) is only a wall-time ratio.
- **GPU, open:** CUDA atomic-add ordering and cuDNN algorithm choice are a different mechanism, so the
  CPU answer is not carried over. Harnesses are ready (`988c3d8`): `frnet_pred_diff.py --device cuda`
  and `shim_determinism_probe.py --device cuda`; runbook steps (e) and (f). CPU reference hashes:
  scatter_max `51503245c7f03b0d`, scatter_mean `393d23dd065277e6`.
- **Rule until GPU is measured:** quote `--fast-scatter` as reproducible only at 1 thread on CPU.
  `a86b491`'s distance table was run that way (`threads: 1` in its JSON), so no R-j caveat applies to
  it.

---

## 4. Ready to pick up — scoped, no decision needed

| # | item | notes |
|---|---|---|
| **R-d** | **`ruff` E741 in `tests/test_metrics.py:472`** (ambiguous `l`). Pre-existing on `origin/main` from `014d388`, still red. | One-line fix; not assigned (tests/ is outside this lane). |

---

## 5. Closed — recorded so nobody re-raises them

| item | outcome | commit |
|---|---|---|
| **R-b** — the p99 tail | **Gate met on this laptop.** Pooled p99 **89.58 ms** over 1,000 frames (5 fresh-process runs, all state-trusted), p50 79.74 ms, against a baseline of 101.30 / 122.96. Five bit-identical fixes: transform scratch `12613df`, cleanup LUT `d540618`, reflectivity skips the unused incidence field `35b7b27`, `_centres` fast path `697a2bd`, `project()` winner selection `b9eef67`. **Limits:** 3 of 1,000 frames still exceeded 100 ms; one machine (D8); the `engine.py` fixes await review (R-h); D1 unchanged. | `36a4dbe` |
| **D9** — `transform_points` allocation | **Applied** with JP's approval: an opt-in `reuse_buffers` scratch, so callers that retain frames keep the allocating path. Bit-identical, and transform p99 fell from 22.07 to 1.45 ms. | `12613df` |
| **D11 checkpoint provenance** | Public release, SHA-256 `09adea90…285e`, recorded before first use and independently corroborated. | `eff969a`, `8e20d57` |
| **Opt-in DL mode** (SIH26053 "deep-learning pipeline") | `--semantics frnet` in `python -m vrgrid.run`, `python -m vrgrid.dash` and `timing_table.py --seq`; ground truth stays the default, motion ground truth (disclosed). Pipeline FRNet labels equal `frnet_eval.py`'s on a real frame. Suite 728 passed / 1 failed (D1) / 3 skipped. Smoke-tested again 2026-09-16 (2 frames, exit 0). | `a26f7f0` |
| **FRNet accuracy by distance** (SIH26053 "Performance Metrics") | 93.2 / 88.1 / 85.4% point accuracy at 0–10 / 10–25 / 25–50 m, with drivable / static / movable group IoUs; bands reproduce the whole-slice 90.3036% / 65.18%. **No ground truth beyond 50.0 m** in seq 08 (measured). A tally bug was caught by a unit test before the real run. | `a86b491` |
| **AWS runbook + GPU latency harness** | Written, then revised after the 2026-09-16 audit (see §3 AWS). | `5925f6b`, `988c3d8`, this clean-up's commit |
| `num_iter=2` latency tradeoff | **Rejected on measurement.** Buys 5.42 ms, costs 19–54% of ring-0 RMSE concentrated on slopes. | `c0ee403`, `860d58c` |
| The 80.78 / 97.72 ms latency claim | **Retired** across all seven files that carried it; reproduces to 1.1% only as a synthetic back-half benchmark. | `5cddaa7` |
| 69.8% mIoU | **Corrected to 65.2%** in the four files that carried it. | `c632027`, `f3a0337` |
| `pypatchworkpp` unpinned | **Pinned to ==1.4.1**, also the fastest published wheel. | `23648e9` |
| `r9` self-contradiction | Withdrawn in place. | `23648e9` |
| `VRGRID_DATA_ROOT` unset failing as "missing data" | **Fixed:** names the cause and the fix. | `fac61c2` |
| Ring-1 reproduction mismatch (seq 07) | **Root-caused:** the singleton (D1). | `870766c` |
| Build provenance (native build vs wheel) | **Closed as a documented unknown** on JP's call. | — |
| R11 limits page | **Done.** | `e4bd731` |
| **R-a** — commit the measurement harnesses | **Done**, each with a PROVENANCE header. | `2955b14`, `8318fed` |
| **R-f** — seq 00 ring-1 residual | **Closed:** the singleton carrying state across sequences. | `reports/ring1-reproduction-investigation.md` §7 |
| **R-e** — 99.87% vs 99.5% | **Resolved by recomputation:** 99.87% dead cells; README was right. | `3a43c19` |
| "Zero allocation in the frame loop" | **Scoped, not retracted:** back end only (the front end allocates ~39.5 MB/frame on seq 08), and the CI test measures retained growth, not churn. `--alloc` now fails loudly with `--seq`. | `235986d` |
| **R-g** — extend the invariant to perception | **Done:** `test_no_retained_growth_in_the_perception_frame_loop`, asserting retained growth only. | `8b40e44` |
| **DL** — `--fast-scatter` re-verification (11 Sep) | Done, exact in both directions on CPU. Superseded in scope by R-j. | `ff774fe` |
| ROS adapter scoping | **Done as design only.** | `0035850` |

---

## 6. The pattern worth acting on

Twice in two sessions a published figure turned out to be **unverifiable because its harness was
session scratch** (the p50 80.78 / p99 97.72 ms latency, and the seq 00 ring-1 RMSE 6.77 cm). In both
cases the number was probably fine; **the missing artifact was the defect.** Every measurement since has
committed its harness and its JSON beside the result (R-a).

---

## 7. Housekeeping state (2026-09-16)

- **Branch:** `jp/p99-alloc-fixes`, 38+ commits ahead of `main`, **never pushed**, no upstream. `main`
  still holds the local vrgrid-26 merge, also not pushed.
- **Untouched:** `src/perception/frnet/` (frozen), `src/perception/ground.py` (D1 paused),
  `src/grid/lattice.py`, and `include/vrgrid/`.
- **Modified on the branch, awaiting review:** `src/run/engine.py` (`d540618`, `697a2bd`), Shrestha's
  file.
- **Gates** (last full run, at `a26f7f0`): `pytest` 728 passed / 1 failed (the D1 determinism test) / 3
  skipped. `ruff` 1 error (R-d, pre-existing).
- **Checkpoint:** `checkpoints/frnet-semantickitti_seg.pth` present locally, **not tracked**, SHA-256
  re-verified 2026-09-16. `checkpoints/frnet-semantickitti_seg.pth.PROVENANCE.md` is tracked.
- **`pending-review/` inventory:**
  - **Applied:** `handover-latency-line-correction`, `r7b-mIoU-69.8-is-an-error`,
    `timing-table-unicode-crash`; `cleanup-isin-guard` and `transform-points-allocation` on this branch
    (their stale "nothing applied" headers were corrected in this clean-up).
  - **Withdrawn:** `patchworkpp-num-iter-tradeoff`.
  - **Superseded:** `stdout-utf8-at-entry` (by R-c), `front-end-allocation-instrumentation` (by R-i).
  - **Waiting:** D2 `r3-…`, D3 `r5-…`, D4 `r7-readme-*`, D10 `ros2-adapter-design` +
    `query-vectorisation-transcription`, R-c, R-i.
- **Summaries:** `MORNING-SUMMARY-3.md` (R-c / R-i pass), `-4.md` (D11 and R-j), `-5.md` (SIH26053
  fit).
