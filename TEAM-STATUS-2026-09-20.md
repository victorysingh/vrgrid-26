# VRgrid — team status and open asks

**As of 2026-09-20.** Compiled by JP from a full audit of `vrgrid26/main` @ `ae85979` and
`jp/p99-alloc-fixes` @ `1d24593` (now pushed to `victorysingh/vrgrid-26`). Every item below cites a commit, file or test so
it can be checked rather than taken on trust. Backing detail: `reports/roadmap-audit-2026-09-20.md`.

**Please read the first section even if you skip the rest** — it affects work already in flight.

---

## Why this page exists: we built the same thing twice, four times

Between 14 and 20 September, four pieces of work were done twice independently:

| what | one version | the other |
|---|---|---|
| the `--semantics frnet` flag | `jp/p99-alloc-fixes` (was unpushed) | `81843cc` |
| classification accuracy vs distance | `a86b491`, `scripts/frnet_eval_by_range.py` | `26fe207`, `scripts/accuracy_by_range.py` |
| the AWS/T4 runbook — **neither ran; the work went to Kaggle** | `reports/aws-dl-realtime-addendum.md` | `docs/gpu-lane/02-AWS-RUNBOOK.md` |
| the DL real-time figure | an AWS run was scoped and armed for it | `d48831a` measured it first, on Kaggle |

Nobody did anything wrong in any of these. Two causes, both fixable:

1. **84 commits on `jp/p99-alloc-fixes` had never been pushed.** Work nobody can see gets rebuilt.
   That was on JP. **It is fixed as of today — the branch is pushed and readable at
   `victorysingh/vrgrid-26`, branch `jp/p99-alloc-fixes`.** No PR yet; the `--semantics` question
   below should be settled first. Please look there before starting anything that touches
   perception, the DL mode or the timing harnesses.
2. **`OPEN-ITEMS.md` has been wrong often enough that we both route around it.** The upstream
   reconciliation on 20 Sep found D1 listed as a blocker for over a week after its gate started
   passing, and says outright that *two people planned around that*. This audit found the mirror
   image — D2 was closed upstream on 17 Sep and still sat in §1 of JP's copy waiting on Aakash —
   plus two more (R-a incomplete, R10 untracked). All three are now corrected and committed. R6 was
   also checked and came back clean.

**Two process asks come out of that, at the end of this page.** They cost nothing technical.

---

## What JP is doing (no action needed from anyone else unless named)

| # | item | what happens | blocked? |
|---|---|---|---|
| 1 | ~~Push `jp/p99-alloc-fixes`~~ | ✅ **DONE 2026-09-20** — 84 commits pushed to `victorysingh/vrgrid-26`, branch `jp/p99-alloc-fixes`. No PR until item 2 is settled | — |
| 2 | **`--semantics frnet` collision** | Resolve before merging. **The two versions are not the same size of thing** — see the ask to Shrestha. Joint call, not unilateral | needs 10 min with Shrestha |
| 3 | **Merge the 29 upstream commits** | Two conflicts: `src/run/__main__.py` (the flag) and `OPEN-ITEMS.md` | after #2 |
| 4 | **Reconcile R-b's two p99 numbers** | Upstream closed it at p99 **28.40 ms** (`--device cuda --schedule 5/10/50`); JP measured **87.69 ms** (CPU, `5/10/20/40`). Both are correct; they measure different things. JP will restate both with denominators attached | no |
| 5 | **Review the `range_image.py` float64 change** | `b5c9f95`/`67473fd` are a correct, well-evidenced portability fix. JP will fix one spliced comment that contradicts itself (says the transcendentals are rounded to float32; azimuth stays float64) | no |
| 6 | ~~R-a: one missing header~~ | ✅ **DONE** — `cleanup_tail_probe.py` now records what it produced | — |
| 7 | **Decisions on JP's desk** | D4 (README wording), D5 (`alpha_m2` — stays 0 until a reference map exists), D7 (CARLA scope), R-c (apply the UTF-8 diff), R-i (unify the timing printers) | no |
| 8 | **S-2 disclosure** | Motion is ground truth in *both* `--semantics gt` and `frnet`. JP owns saying this plainly wherever the DL mode is described | no |

**Fixed by this audit, already committed:** `R-a`'s one missing `PROVENANCE` header
(`cleanup_tail_probe.py`, 38 of 39 had one); **D2's status on JP's side** — it was closed upstream by
`df35fd5` on 17 Sep and still sat in "waiting on a decision" with Aakash's name on it; and **R10 is
back on the board** as explicitly unassigned (see below). **R6 was checked and is clean** — its
result is not cited in any current report, doc or README, so no untraceable number is in circulation.

**Closed by this audit, for the record:** D1 (decided, and now enforced by
`tests/test_ground_reset_convention.py`), R-k (`ground_cost.py`'s estimator carry-over — real but
immaterial: agreement 96.5% either way, timing +0.14 ms), and D11's real-time figure, which
`d48831a` answered before any rented instance was needed.

**AWS is closed out, for everyone's planning:** `docs/research-log.md` records it as *abandoned, not
deferred* — every GPU quota read 0, the increase was refused because the free plan has no GPU tier,
**lifetime spend $0.00**, bucket and guardrails torn down. The T4 pass ran on **Kaggle's free T4**
(`docs/gpu-lane/t4/host.log`: 2× Tesla T4, 15,360 MiB, driver 580.159.04). **Nobody needs to
provision or cancel anything.** JP's `reports/aws-dl-realtime-addendum.md` now says so at the top.

---

## Asks, by person

### Shrestha

1. **Review `d540618` and `697a2bd`** (R-h). Two `MapEngine` changes — the `_cleanup` LUT and the
   `_centres` sorted fast path. Both verified bit-identical and output-neutral on top of your port
   (host-path parity: identical counters and map hash on 20 frames vs `7dee296`). They have been
   waiting on your review since before the GPU port. **Note:** since the port these are host-path
   functions, so the "how much does this save" answer depends on which path is timed.
2. **The `--semantics frnet` flag — let's pick one implementation.** Both branches ship the same
   flag name, the same `choices`, the same default. **They are not the same size of thing, and that
   decides it more than preference does:**
   - **Yours (`81843cc`, `d48831a`) runs FRNet on the DEVICE path.** `perceive()` calls
     `frnet.infer_points()` *before* `perception.launch()` and passes `semantic_pred=pred` into the
     kernel, so the class is an input to the device semantics stage. That required widening
     `DevicePerception.launch()` — it is a cross-file feature, not a CLI flag. **It is also what
     produced the 10 Hz DL figure** (cuda + fp16, p50 71.8 / p99 79.5 ms).
   - **JP's is host-only by construction.** It raises `ValueError` on
     `--semantics frnet --device cuda`, because on the device path the model's labels would be
     silently ignored. What it adds is reproducibility: `--threads` (calling
     `torch.set_num_threads` *before* the model is built — at 1 thread FRNet's labels are
     reproducible run to run, and at the default they are not; OPEN-ITEMS R-j) and `--fast-scatter`.
   - **So the device integration cannot simply be "folded into" JP's version — it is the larger
     piece.** JP's proposal, for you to agree or reject: **take yours as the base** and port the two
     reproducibility knobs onto it, keeping `--semantics-precision` and `--semantics-every`. That
     keeps the 10 Hz result intact. Branch is pushed — `victorysingh/vrgrid-26`,
     `jp/p99-alloc-fixes`, see `src/run/__main__.py` and `open_frnet()`.
3. **R-d is marked closed but is not.** `OPEN-ITEMS` says *"`ruff check .` passes clean on `main`"*.
   `tests/test_metrics.py:472` is unchanged (`(p, l, T) for p, l, _, T in ...`) and ruff **0.12.0**
   reports **10 errors** on `ae85979` — that E741 plus `scripts/kaggle/elprobe.py` (E402, E741), two
   notebook E702/E701, and `scripts/mos_learned.py:233` (E702). **CI runs bare `ruff check .` with no
   version pinned anywhere**, so whether CI is green depends on which ruff GitHub installs that
   morning. Suggest pinning ruff in CI and reopening R-d.
4. **R-b's closure needs its denominators.** Closing it at "p50 21.94 / p99 28.40, 3.5× headroom" is
   fair for `--device cuda --schedule 5/10/50`, but the row does not say so, and the item was
   originally about the CPU path on `5/10/20/40`. Either denominator in the row, or two rows.
5. **Two GPU figures have no committed artifact — this is the one that matters most.** Traced with
   escaped patterns across every `.md`, `.log` and `.json` in the tree:
   - **p50 21.94 / p99 28.40 ms** (the R-b closure) appears in `OPEN-ITEMS.md`,
     `13-PS-SCHEDULE.md` and `research-log.md` — and **in no log or JSON**.
   - **p50 71.8 / p99 79.5 ms** (the 10 Hz DL claim) appears only in `research-log.md` and the
     `d48831a` message — **no artifact holds the pair, and no document names the machine.**
   - Meanwhile **`docs/gpu-lane/t4/timing_cuda.log` records p50 67.42 / p99 100.18 ms, "MISSES
     10 Hz at p99"** on `5/10/20/40` — and **no document mentions that result.**

   Both the favourable and unfavourable numbers can be true at once — different schedules, and
   `5/10/50` drops a ring. The problem is that the repo shows one without its raw log and leaves the
   other only in a log nothing cites. **"The DL pipeline meets 10 Hz" is currently our most quotable
   sentence and our least reproducible one.** A re-run with `--frame-times` committed, plus the host
   named the way `06`/`07`/`09`/`10` already name theirs, would close it.
6. **The AWS docs need a superseded note from you.** `docs/gpu-lane/02-AWS-RUNBOOK.md`,
   `11-AWS-RESUME.md` and `scripts/aws/` still read as live plans. They are your files, so JP has
   not touched them; `research-log.md` already has the real story.
7. **An attribution worth re-checking.** `d48831a` and `research-log.md` conclude that FRNet's
   0.028–0.046% run-to-run label disagreement is *"the model"*, without stating a thread count. R-j
   on JP's branch found the opposite cause: at PyTorch's default intra-op thread count the labels do
   **not** reproduce, and with **`torch.set_num_threads(1)` they are exact**. Consistent only if
   those runs used the default — in which case it is the thread pool, not the model. Worth one
   re-run at 1 thread to settle.
8. **N-3** — cause still open after your probe refuted the leading hypothesis (`8854b47`).
9. **S-3 (residual channels into FRNet)** reads as yours and as the largest accuracy win left.

*Also: thank you for `df35fd5` and the `docs/handover-2026-09-17-aakash.md` write-up — closing
someone else's lane item and documenting every change for their review is exactly right.*

### Aakash

1. **D2 is closed — please review the handover.** Shrestha implemented R3/R4 in `df35fd5` while you
   were away, with every change to your lane listed in `docs/handover-2026-09-17-aakash.md`. Ring
   membership is now decided per world-lattice block; R4's CI test landed as
   `test_no_cell_footprint_contains_another_under_foveation`. It needs your eyes, not your work.
2. **D6 — `kappa` 1/16 vs 1/12.** Your 28 Aug note says the stated geometry gives 1/12 and the config
   is knowingly 25% low. Both values are pinned in
   `test_kappa_from_geometry_is_one_twelfth_at_every_ratio`, so the choice is visible but unmade.
   This is a room decision and it has been open a long time.
3. **D3 — the sticky safety-critical (VRU) class bit.** Designed in
   `pending-review/r5-sticky-safety-critical-class-bit.md`, never implemented. Which classes are
   sticky is a room call.

### Pratyushi

Two things in the GPU lane are unclaimed, and both are real pieces of work rather than cleanup.
If either is already yours, say so and JP will mark it claimed:

- **R-j's GPU half.** On CPU it is resolved (with `torch.set_num_threads(1)`, `--fast-scatter`
  matches the loop exactly and repeats exactly). **Nobody has answered whether two CUDA runs of
  `--fast-scatter` agree point-for-point** — atomic-add ordering and cuDNN algorithm choice make it a
  genuinely open question, and it needs a card.
- **S-1**, the fourth gate reason in `grid/gate.py` — refine where perception is uncertain. The
  signal is already measured (at fixed range, the more uncertain half is 9–20 accuracy points worse).

### Srinivas

The "UI + everything else" lane includes ROS, docs and integration. One concrete gap there:

- **D10 — ROS 2 adapter scope.** `pending-review/ros2-adapter-design.md` exists; the scope questions
  (window extent, bitfield as one message or several) have never been answered, so nothing has been
  built. This is the one assigned lane deliverable with no implementation at all.

### Hriday

Two items sit unclaimed in the DL + grid-engine lane:

- **D3/R5 implementation** once the room decides which classes are sticky (see Aakash above).
- **R10 — the wide-depression gradient.** ⚑ It has **no commit, no file and no mention anywhere in
  the repo** — the only roadmap item with zero work of any kind. It was not deferred; nothing
  tracked it, so it could not surface. It is now listed in `OPEN-ITEMS.md` §1 as **unassigned**, and
  it needs either an owner or a deliberate decision to descope. Either is fine; silence is not.

### The whole room (not a lane decision)

- **S-4 — put the dynamic belief in the grid's own `log_odds`.** Worth +1.1 IoU as a prototype, but
  it touches the **frozen 12-byte cell struct**. Whole-team sign-off.
- **D3** (sticky classes), **D6** (`kappa`), **D10** (ROS scope) — three decisions that have each been
  open long enough to block work behind them.
- **R6 — ground-only variance** was closed before this window and left **no artifact in the tree**.
  If any current number descends from it, it has no traceable source. Does anyone remember where it
  went?

---

## Two process asks

**1. Claim an item before building it, in one visible line.** Name, date, item — in `OPEN-ITEMS.md`
or wherever we agree. Three of the four duplicate incidents above were two people solving the same
well-specified problem in the same week with nowhere to see "I'm on this."

**2. Date every status line, and re-verify before believing one.** Upstream already added the right
rule — *"Re-run a gate before believing any status line here"* — after D1 sat as a blocker for a week
past its fix. This audit found four more stale or wrong statuses (D2, R-a, R-d, R-b). A status file
that is wrong in five of ~30 rows is more dangerous than none, because it is trusted. Suggest: every
row carries the date it was last verified and the command that verified it.

**And on JP's side: the branch gets pushed.** That is the root cause of most of this and it is not
anyone else's to fix.
