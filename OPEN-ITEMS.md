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
| **D11** | **The FRNet checkpoint.** `checkpoints/frnet-semantickitti_seg.pth` is absent; weights are gitignored by design and were never committed. | Blocks re-confirming 90.3% / 65.2% and blocks the three-way plan-regret comparison entirely. Fetching pretrained weights unprompted is not something I should do. | `reports/fast-scatter-reverification.md` |

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
| **R-c** — stdout UTF-8 | **MEASURED AND PROPOSED, not applied.** Scope is far larger than thought: **15 of 20** scripts hold a cp1252-unencodable char and **12 of 20 crash on `--help`**. Two candidates verified; `PYTHONUTF8=1` recommended (zero source edits). Needs a decision because it touches `demo.sh`, CI and a doc — and because the one *central* fix is blocked by the `include/vrgrid/` freeze. | `pending-review/stdout-utf8-at-entry.md` |
| **R-b** — the p99 tail | **LOCALISED to allocation in `transform_points`**, with GC, data-dependence and thermal all ruled out by measurement. See §3 R-h for what remains. | `0035850` |
| **"Zero allocation in the frame loop"** | **SCOPED, not retracted** — same fix shape as `e3cda04`'s determinism split. 12 edits across 7 files — README x3, `research-log.md`, and **all 5 `docs/presentation/*` files carrying it (7 instances between them)**. `--alloc` now **fails loudly** with `--seq` instead of silently printing an incomplete table. Both scope gaps kept separate everywhere: **(a)** back end only, front end allocates ~39.5 MB/frame on real seq 08; **(b)** the CI test measures **retained growth, not churn**. | see below |
| **R-g** — extend the invariant to perception | **DONE.** `test_no_retained_growth_in_the_perception_frame_loop` added beside the grid one, so the contrast between them is visible. Asserts **retained growth only** (< 64 KB, the same bound as the grid test) on a **steady-state** window — the first window retains ~8.8 MB, which is one live `PerceptionFrame`, not a leak. Deliberately does **not** assert zero churn: the front end allocates ~39.5 MB/frame and would fail that today, which is D9. Suite 672 → **673 passed**. | `see below` |
| **R-i** — front-end allocation instrumentation | **PROPOSED, not applied.** The technique is already proven in `reports/harnesses/stage_allocation.py` (wrap `Timer.stage`, the one hook both halves pass through) — so this is "move ~10 lines into the shipping tool and decide how it prints", not "figure out how". The open decision is whether to **unify the two table printers**, since the synthetic/real split is what produced *both* mislabelling incidents. Not done tonight: it touches that script, and the machine degraded mid-session so no verification run would be trustworthy. | `pending-review/front-end-allocation-instrumentation.md` |
| **R-h** — `cleanup` / `ground` | **`cleanup` DONE and proposed; `ground` deferred as a different mechanism.** `cleanup`'s 9.61 MB is **not** in `visibility_cleanup` (scratch-based, allocation-free as documented) — it is `np.isin` inside an argument to an otherwise careful `np.copyto`. A boolean LUT is **21x less allocation, 3.9x faster, bit-identical**. `ground` **attempted and INCONCLUSIVE** — not for lack of a hypothesis but because the instrument failed: the same harness that read 21.01 ms earlier read **93.44 ms** late in the session, with the **CPU downclocked to 1520/2400 MHz** and the machine **8 GB over-committed**. Every absolute timing from that window is void; the `transform`/`cleanup` A/Bs survive because both arms ran together. Needs a clean machine, with clock recorded. | `pending-review/cleanup-isin-guard.md` |
| **DL** — `--fast-scatter` re-verification | **DONE, still exact in both directions** through `frnet_eval.py` rather than the standalone check. The accuracy and plan-regret arms are blocked on D11. | `ff774fe` |
| **ROS adapter scoping** | **DONE as design only**, nothing under `adapters/` created. Needs no change to the frozen `api.py` as designed — but see D10. | `0035850` |
| **R-b** — the p99 tail | **LOCALISED, not fixed.** GC ruled out (0.00% of runtime, 0 of the 8 worst frames). Data-dependence ruled out (worst-frame overlap = chance). Thermal ruled out (drift plateaus, 28 s run). Stage attribution depends on page cache: cold `load` 60%, warm **`transform` 84%** — and the warm tail is **allocation**, 10.86 MB per call, 25x smaller tail when preallocated. Fixing it gives ~113 ms p99, still over 100. Next: `ground` (42%), `cleanup` (32%). | `reports/r-b-p99-tail-investigation.md`; proposal `pending-review/transform-points-allocation.md` |

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
