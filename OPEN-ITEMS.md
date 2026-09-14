# Open items — the master list

Single place for what is outstanding, so nothing falls between sessions.
Assembled 2026-09-12 by reading every file in `reports/` and `pending-review/`
plus `MORNING-SUMMARY.md`, against `main` @ `e4bd731`.

**How to read the status column:** *Decision* = waiting on JP, nothing to do
until then. *Ready* = scoped, could be picked up now. *Blocked* = waiting on
something external or someone else.

---

## 1. Waiting on your decision — nothing proceeds until these are called

| # | item | why it needs you | file |
|---|---|---|---|
| **D1** | **The Patchwork++ singleton lifetime.** Does `segment_ground` take an estimator? Does the module expose a reset? Who owns the lifetime? | Design decision with three real options, none obviously right. **Highest-impact item on this list** — see §2. | `reports/ring1-reproduction-investigation.md` §7 |
| **D2** | **R3 — ring boundary under anisotropy.** Nearest-corner test + boundary snapping. | Touches `src/grid/lattice.py` (Aakash's) and has a cross-lane consequence in `metrics._ring_cells`. Needs his call, not mine. | `pending-review/r3-ring-boundary-under-anisotropy.md` |
| **D3** | **R5 — sticky safety-critical class bit.** Which classes (5/6/7 only, or also 1/2)? Which of three decay designs? What is N, and it must live in frozen `configs/thresholds.yaml`? Does it affect traversability, and on which commit? | Five open sub-decisions, listed in the file. Also touches frozen `include/vrgrid/cell.py` — three-way sign-off. | `pending-review/r5-sticky-safety-critical-class-bit.md` §6 |
| **D4** | **R7 README wording** — which of three drafted lengths, and where it goes. | You said you would place it yourself. | `pending-review/r7-readme-counts-draft.md` |
| **D5** | **`alpha_m2` — raise it or leave it at 0.** | Not a calibration task: 0 is the only value consistent with sih-math §5.3 and the §5.4 unit test, so raising it means restating both in the same commit. | `reports/r11-what-this-does-not-do.md` §4 |
| **D6** | **`kappa` 1/16 vs 1/12.** Knowingly 25% low; the stated geometry gives 1/12. | Frozen config, explicitly "the room decides". Both values pinned in tests so the choice stays visible. | `configs/thresholds.yaml:133` |
| **D7** | **CARLA groundwork — scope.** | **Zero existing work** in this repo or any sibling folder, and no scope has been supplied across two sessions. Starting a simulator integration from a guess is the wrong move; this needs a paragraph from you on intended scope. | `MORNING-SUMMARY.md` §3 |
| **D8** | **[SCOPE UPGRADED 2026-09-14] Machine STATE control for latency, not just an agreed hostname.** | Every latency disagreement in the project reduces to this. Not a measurement problem. **Now concrete:** since the vrgrid-26 merge the tree states *two* honest end-to-end figures — `README.md` 89.18/100.43 (Shrestha's host) and `handover:20` 108.65/127.23 (this host). Same quantity, different machines; one is 0.43 ms over budget and the other 27 ms over. **The two hosts are identified:** this machine is an **i7-13620H**; the 89.18/100.43 figure was measured on an **i7-14650HX** per `README.md:288-289`, which is Shrestha's. *(Correction: an earlier version of this row suggested a possible third host. That was wrong — README:288-289 is the identification of the second host, not a third one. D8 is a two-host problem.)* Needs JP + Shrestha together. | `docs/handover-2026-09-02.md` latency note; `reports/latency-gap-investigation.md` | **[!] THE SCOPE CHANGED, and this is not a footnote.** D8 was framed as *two hosts disagree, pick one*. That framing is now insufficient: **one host varied 4.4x within a single session.** `ground_cost.py` read **21.01 ms** early and **93.44 ms** late — same script, same data, 2% CPU load — with the CPU **downclocked to 1520 MHz against a 2400 MHz base (37% off)** and the machine **~8 GB over-committed** (23.61 GB commit against 15.73 GB physical, 2.2 GB free). So nominating a reference *hostname* fixes nothing: the same box gives 21 ms or 93 ms depending on clock and memory state. **Resolving D8 now requires a defined and CHECKED machine state** — at minimum: CPU clock recorded alongside every figure (`CurrentClockSpeed` vs `MaxClockSpeed`, because a 37% downclock is invisible in a timing table), commit charge under physical RAM, and a stated warm-up/idle precondition. A latency number without its machine state attached is not checkable, whoever's host it came from. Corollary already proven: **controlled A/B ratios survive this** (both arms scale together) while headline absolutes do not — so prefer ratios in claims wherever the question allows it. See `reports/r-b-p99-tail-investigation.md` §11. |
| **D9** | **`transform_points` allocation fix** — return-a-fresh-array contract, and where a reused buffer lives. | It is the largest identified p99 contributor (84% of the warm-run excess) but the fix touches a stateless hot-path function called from three places. Returning a view into a reused buffer could silently mutate data a caller retained. | `pending-review/transform-points-allocation.md` §4 |
| **D10** | **ROS 2 adapter scope** — window extent, bitfield as one float layer or six boolean ones, where live labels come from, publish rate, and whether `export_gridmap()` needs arguments. | The last of those changes a **frozen signature**, so it needs the three-way *before* implementation rather than halfway through. **And the blocker is throughput, not the wire format:** `query()` is 54.7 µs, so a 20 m window at 5 cm is **8.75 s per frame (0.11 Hz)**. A vectorised bulk reader is a prerequisite and must be pinned bit-identical to `query_region`. **METHOD decided 2026-09-13:** vectorise the existing `query()` by transcribing its source, not by re-deriving from a spec. **STEP 1 (transcription) is DONE as prep** — `pending-review/query-vectorisation-transcription.md`. It found that `ring_of_into`, `flat_slot_into` (already pinned bit-identical), `occupancy_state(slots=)`, `i_fine`/`i_ring` and `unpack_class` are **already vectorised**, so only `_refined` is new work — and that `query()` contains **no reduction**, so there is no `scatter_mean`-style ULP risk. STEP 2 not started. **Two traps are now NAMED CHECKS** (§6a) rather than notes: `test_bulk_query_keeps_the_two_outside_paths_independent` — measured, the two OUTSIDE paths diverge on **361 of 361** swept points when a window is un-tracked and on **0 of 361** when tracked, and `OUTSIDE == -1` so a conflated mask silently reads the last ring — and `test_bulk_query_masks_before_gathering`. | `pending-review/ros2-adapter-design.md` §4, §7; `query-vectorisation-transcription.md` |
| **D11** | **The FRNet checkpoint.** **[IN PROGRESS 2026-09-14 — reopened NARROWLY for R2 only, by JP.]** Scope: retrain with `frnet_finetune.py`'s recipe, re-measure point accuracy / mIoU, confirm `--fast-scatter` end to end, run R2 (plan regret: GT labels vs FRNet-predicted labels, and the delta). Offline evaluation only: `ground.py`, the run engine and the production GT label source are not touched, and FRNet stays out of the mapping pipeline. **Blocked at a prerequisite before any run:** the recipe fine-tunes FROM `--init checkpoints/frnet-semantickitti_seg.pth`, which does not exist (no `checkpoints/` directory), and torch is CPU-only (`2.13.0+cpu`). Reported to JP. **[Progress 2026-09-14]** JP's decisions: checkpoint A (Shrestha) first, else B with URL + SHA-256 recorded; C off the table. CPU only; no local CUDA build; any fine-tune goes to AWS. Step 1: `--fast-scatter` was already wired into both scripts. Step 2: shim verify **passed** on a clean machine (max/mean both 0.000e+00 fwd+bwd on CPU; 156×/64× per call). Step 3: **skipped, not needed**, because 90.3%/65.2% are the PRETRAINED checkpoint's figures and every fine-tune was rejected on 4 Sep. Log: `reports/d11-frnet-reverification-r2.md`. Option B source identified (not downloaded): the official FRNet README links a Google Drive folder holding `frnet-semantickitti_seg.pth`, 38.5 MB, 7 Dec 2023; no checksum published. **[2026-09-14 second pass: STILL OPEN.]** The session brief's D11 line arrived with its placeholder unfilled (`[FILL THIS IN BEFORE SENDING …]`), so no location was given and DL items 1–3 were not run. Also needs JP to confirm FRNet is reopened for this: it was closed in late August (ship with ground-truth SemanticKITTI labels; do not revisit unless JP asks). `checkpoints/frnet-semantickitti_seg.pth` is absent; weights are gitignored by design and were never committed. | Blocks re-confirming 90.3% / 65.2% and blocks the three-way plan-regret comparison entirely. Fetching pretrained weights unprompted is not something I should do. | `reports/fast-scatter-reverification.md` |

---

## 2. [!] The one that blocks other work

**D1, the Patchwork++ singleton.** Flagging it separately because it is not just
another queue item — it is currently **distorting measurements other decisions
depend on**:

- `test_real_sequence_replay_is_identical` fails, correctly. Two replays in one
  process differ by 1,245 of 1,479,013 points.
- **~18% of the published seq-07 ring-1 RMSE is artifact**, not coarsening error
  (3.60 → **3.04 cm** once the reference and the map use a consistent ground
  mask).
- **No ring-1 accuracy comparison finer than ~0.5 cm can be adjudicated until it
  is fixed.** That is a live constraint on any future accuracy work, not a
  hypothetical.

Found twice independently — as the determinism gate failure, and as the ring-1
reproduction mismatch. The fix direction is measured, not speculative: two
estimators making one pass each agree exactly, 0 of 1.48 M points differing.

---

## 3. Ready to pick up — scoped, no decision needed

| # | item | notes |
|---|---|---|
| **R-d** | **`ruff` E741 in `tests/test_metrics.py:472`** (ambiguous `l`). Pre-existing on `origin/main` from `014d388`, unrelated to any of this week's work, still red. | One-line fix, not my lane (tests/). |

---

## 4. Closed this week — recorded so nobody re-raises them

| item | outcome | commit |
|---|---|---|
| `num_iter=2` latency tradeoff | **REJECTED on measurement.** Buys 5.42 ms, costs 19–54% of ring-0 RMSE concentrated on slopes. Curbs clean; slopes were the mechanism. | `c0ee403`, `860d58c` |
| The 80.78 / 97.72 ms latency claim | **RETIRED** across all seven files that carried it. Reproduces to 1.1% as a *synthetic* back-half benchmark; original run unrecoverable. | `5cddaa7` |
| 69.8% mIoU | **CORRECTED to 65.2%** in the four files that carried it, each now with its recipe. `research-log:402` determined **not** to be a defect. | `c632027`, `f3a0337` |
| `pypatchworkpp` unpinned | **PINNED to ==1.4.1** — also the fastest published wheel (18.58 ms vs 23.8–25.1 for every older release). | `23648e9` |
| `r9` self-contradiction | Withdrawn in place — the `--no-patchworkpp` inference was wrong. | `23648e9` |
| `VRGRID_DATA_ROOT` unset failing as "missing data" | **FIXED.** Names the cause and the fix. | `fac61c2` |
| Ring-1 reproduction mismatch (seq 07) | **ROOT-CAUSED** — the singleton. Seq 08 immune; seq 00 open (R-f). | `870766c` |
| Build provenance (native build vs wheel) | **Closed as a documented unknown** on your call — needs a multi-GB toolchain. | — |
| R11 limits page | **DONE.** | `e4bd731` |
| **R-a** — commit the measurement harnesses | **DONE.** 21 scripts in `reports/harnesses/`, each with a PROVENANCE header naming the report, which figures it produced, the invocation, and the traps. Lint-clean; every reformatting edit AST-verified. | `2955b14`, `8318fed` |
| **R-f** — seq 00 ring-1 residual | **CLOSED — and it was the singleton after all**, carrying state across *sequences* rather than only across passes. The published harness measures 07, 08, 00 in one process, so seq 00's estimator holds ~160 frames of the other two. Every published R1 figure then reproduces exactly, 41,892 @ 6.77 included. Closed within minutes of R-a committing the harness, after resisting two investigations. | `reports/ring1-reproduction-investigation.md` §7 |
| **R-e** — 99.87% vs 99.5% | **RESOLVED by recomputation, not by assumption.** 99.5% is the RADIAL term of `sih-math.md` eq. (4) alone; the full product is 0.001327, i.e. **99.87%** dead cells. README was right; `master-v4.md` and Hriday's R2 memo were corrected. | `3a43c19` |
| **R-c** — stdout UTF-8 | **[STAGED 2026-09-14 — `d0d84c3`, not applied.]** JP chose an in-code entry-point fix over `PYTHONUTF8=1`. `pending-review/r-c-stdout-utf8-entry-points.diff` (20 scripts) reconfigures stdout/stderr under each `__main__` guard. Verified: UTF-8 output byte-identical on 18/18 executed scripts; all 12 cp1252 `--help` crashes fixed with output equal to UTF-8; script tests 31 passed, 1 skipped on the patched tree. Not executed: `frnet_fast_scatter`, `gen_demo_rrds` (side effects). Does not cover `python -m vrgrid.run`, harnesses, or `open()` defaults. **MEASURED AND PROPOSED, not applied.** Scope is far larger than thought: **15 of 20** scripts hold a cp1252-unencodable char and **12 of 20 crash on `--help`**. Two candidates verified; `PYTHONUTF8=1` recommended (zero source edits). Needs a decision because it touches `demo.sh`, CI and a doc — and because the one *central* fix is blocked by the `include/vrgrid/` freeze. | `pending-review/stdout-utf8-at-entry.md` |
| **"Zero allocation in the frame loop"** | **SCOPED, not retracted** — same fix shape as `e3cda04`'s determinism split. 12 edits across 7 files — README x3, `research-log.md`, and **all 5 `docs/presentation/*` files carrying it (7 instances between them)**. `--alloc` now **fails loudly** with `--seq` instead of silently printing an incomplete table. Both scope gaps kept separate everywhere: **(a)** back end only, front end allocates ~39.5 MB/frame on real seq 08; **(b)** the CI test measures **retained growth, not churn**. | see below |
| **R-g** — extend the invariant to perception | **DONE.** `test_no_retained_growth_in_the_perception_frame_loop` added beside the grid one, so the contrast between them is visible. Asserts **retained growth only** (< 64 KB, the same bound as the grid test) on a **steady-state** window — the first window retains ~8.8 MB, which is one live `PerceptionFrame`, not a leak. Deliberately does **not** assert zero churn: the front end allocates ~39.5 MB/frame and would fail that today, which is D9. Suite 672 → **673 passed**. | `see below` |
| **R-i** — front-end allocation instrumentation | **[DECISION DOC 2026-09-14 — `d0d84c3`, JP to decide; nothing implemented.]** `pending-review/r-i-unify-timing-table-printers.md`: the case for and against unifying the synthetic and real table printers. Recommendation: unify the printers, keep `run()`/`run_real()` separate, write snapshot tests of both current tables first, and update `whole_frame_bench.py`'s row regex in the same commit. **PROPOSED, not applied.** The technique is already proven in `reports/harnesses/stage_allocation.py` (wrap `Timer.stage`, the one hook both halves pass through) — so this is "move ~10 lines into the shipping tool and decide how it prints", not "figure out how". The open decision is whether to **unify the two table printers**, since the synthetic/real split is what produced *both* mislabelling incidents. Not done tonight: it touches that script, and the machine degraded mid-session so no verification run would be trustworthy. | `pending-review/front-end-allocation-instrumentation.md` |
| **R-h** — `cleanup` / `ground` | **[GROUND RE-ATTEMPT DEFERRED 2026-09-14 — not attempted degraded.]** Gate check before any measurement: CPU clock 2400/2400 MHz (fine), but **commit 17.33 GB against 15.73 GB physical, 2.81 GB free — the machine is paging**. The trusted state used for the pooled p99 had lapsed: by process group, chrome (37 processes) 4.2 GB, Code 2.5 GB, msedgewebview2 1.1 GB, Spotify 0.7 GB. Per JP's rule the ground A/B was NOT run; it needs the machine back under physical RAM with state logged per run. **`cleanup` DONE and proposed; `ground` MEASURED — its tail is not intrinsic to Patchwork++.** **[!] CORRECTED 2026-09-14:** `np.isin` was the second-largest allocator in `_cleanup` (6.40 MB), not the largest -- `_centres` is (7.09 MB, 10.78 ms p50), found by instrumenting every line instead of three suspected ones. The LUT fix stands and is applied on `jp/p99-alloc-fixes`; `_centres` is the real target in this stage. **[TAIL PROBE 2026-09-14 — structure only, machine UNTRUSTED: commit 17.05–17.35 GB > 15.73 GB physical, user applications.]** After the `_centres` fix, per-frame timing of every `_cleanup` line over 180 warm seq-08 frames (transcription matches the shipped method's map hash): the slowest 5% of cleanups exceed the rest by 3.33 ms, spread across lines in proportion to their size — `visibility_cleanup` +1.91, `_centres` +0.92, every other line ≤0.15. **No single line spikes.** `visibility_cleanup` is already fully `out=`/scratch-based, so its ~11.5 ms is float compute (`arctan2`, `sqrt`, `arcsin` over ~320,000 cells), not allocation; cutting it means changing float arithmetic, which is a ULP/bit-identity risk and not attempted. Absolute times from this probe are not quoted. `cleanup`'s 9.61 MB is **not** in `visibility_cleanup` (scratch-based, allocation-free as documented) — it is `np.isin` inside an argument to an otherwise careful `np.copyto`. A boolean LUT is **21x less allocation, 3.9x faster, bit-identical**. `ground` **re-measured after a restart, calibration passed first** (20.95/20.33/20.12 ms vs known-good ~21). In isolation Patchwork++ has **p99−p50 ~1 ms, worst frame ≤1.08× median** in 3 of 4 passes — against **11.13 ms** in-pipeline. So `ground`'s tail is **extrinsic**, not the C++ extension. Hypothesis, untested: page-fault stalls from other stages' allocation landing in `ground`; test by applying D9 + the cleanup LUT and re-measuring in-pipeline. | `pending-review/cleanup-isin-guard.md` |
| **DL** — `--fast-scatter` re-verification | **DONE, still exact in both directions** through `frnet_eval.py` rather than the standalone check. The accuracy and plan-regret arms are blocked on D11. | `ff774fe` |
| **ROS adapter scoping** | **DONE as design only**, nothing under `adapters/` created. Needs no change to the frozen `api.py` as designed — but see D10. | `0035850` |
| **R-b** — the p99 tail | **[GATE MET 2026-09-14 — POOLED p99, the criterion JP chose; on THIS laptop only.]** `7e58ad7` (code identical to `b9eef67`), seq 08, 5 fresh-process runs × 200 frames = 1,000 frames, all 5 runs trusted (2400/2400 MHz, commit ~13.0/15.73 GB): **pooled p99 89.58 ms** (bootstrap 95% CI 88.35–95.55 over frames, 89.21–95.55 over runs), **p50 79.74 ms**. **Still open: 3 of 1,000 frames exceeded 100 ms** (102.2, 103.4, 111.8 — one each in runs 1, 2, 5), so this is a p99 pass, not a hard real-time bound. **Not yet a project-wide claim:** (a) D8 — measured on the i7-13620H, the reference machine is still not agreed with Shrestha; (b) the `engine.py` changes (d540618, 697a2bd) await Shrestha's review; (c) the determinism gate still fails on D1, unchanged by this work. `reports/bench/pooled_after_range_image.json`. **[UPDATE 2026-09-14 — p99 MEDIAN UNDER BUDGET, NOT EVERY RUN.]** On `jp/p99-alloc-fixes` (`b9eef67`), seq 08, 200 frames, 5 fresh-process reps, all trusted (2400/2400 MHz, commit ~12.5/15.73 GB): **FRAME p50 81.27 ms, p99 median 94.01 ms — but per-rep p99 was 114.38 / 92.80 / 90.64 / 94.01 / 102.93, so 2 of 5 runs still miss 100 ms** (rep 1 had a single 184 ms frame). Baseline on the same machine and harness was 101.30 / 122.96. Fixes, each bit-identical and tested: transform scratch (12613df), cleanup LUT (d540618), reflectivity skips the unused incidence field (35b7b27), `_centres` sorted fast path (697a2bd), `project()` winner selection without `np.unique` (b9eef67). Per-frame allocation ~59 → 36.80 MB before the last fix. **Whether 'median of reps' is the gate, or every run must pass, is not decided — not claimed as met.** Remaining tail now sits mostly in `cleanup` (p99 31.20 ms median, noisy) — `visibility_cleanup` is ~11 ms of it. Note: in these clean-machine runs `ground`'s p99−p50 is ~1.4–2 ms at baseline AND after, so the earlier 11.13 ms in-pipeline ground tail was not reproduced and the allocation fixes are NOT shown to be what cured it. **LOCALISED, not fixed.** GC ruled out (0.00% of runtime, 0 of the 8 worst frames). Data-dependence ruled out (worst-frame overlap = chance). Thermal ruled out (drift plateaus, 28 s run). Stage attribution depends on page cache: cold `load` 60%, warm **`transform` 84%** — and the warm tail is **allocation**, 10.86 MB per call, 25x smaller tail when preallocated. Fixing it gives ~113 ms p99, still over 100. Next: `ground` (42%), `cleanup` (32%). | `reports/r-b-p99-tail-investigation.md`; proposal `pending-review/transform-points-allocation.md` |

---

## 5. The pattern worth acting on

Twice in two sessions a published figure turned out to be **unverifiable because
its harness was session scratch**:

- **p50 80.78 / p99 97.72 ms** — no log, no artifact, no script, no method in the
  commit.
- **Seq 00 ring-1 RMSE 6.77 cm** — differs by a 61–89 cell population difference
  from a re-measurement; `scratchpad/r1_accuracy_by_class.py` has never existed
  in the repository.

In both cases the number itself is probably fine. **The missing artifact is the
defect**, and in both cases it cost more time to investigate than committing the
script would have cost. That is item **R-a**, and it is the cheapest thing on
this list.

---

## 6. Housekeeping state

- **Branch:** `main`. **vrgrid-26 has been merged in** (see below); nothing
  pushed anywhere, per standing instruction.
- **`vrgrid-26`:** remote `vrgrid26` added and fetched on JP's explicit
  instruction, superseding the earlier "do not check" rule. `vrgrid26/main`
  merged into local `main` as a true merge (**not** a rebase — both sides had
  independent commits worth keeping, and rebasing would have rewritten
  already-pushed history). Shared base `5e0ebf3`, their 5 commits against our
  17. Four conflicts, all resolved by reading both sides. **Not pushed.**
- **`src/perception/frnet/`:** frozen, untouched.
- **`src/perception/ground.py`:** untouched — D1 is paused pending your design
  call, not attempted.
- **Gates:** `pytest` 672 passed / 1 failed (the determinism gate, D1) / 3
  skipped. `ruff` 1 error (R-d, pre-existing and unrelated).
- **`pending-review/` inventory:** 4 awaiting decision (D2, D3, D4 ×2), 4 now
  marked applied (`handover-latency-line-correction`, `r7b-mIoU`,
  `timing-table-unicode-crash`, and `patchworkpp-num-iter-tradeoff` as
  withdrawn-on-evidence). Two of those four had **stale "Applied? NO" headers**
  despite having been applied in `f3a0337`; corrected while assembling this list.
