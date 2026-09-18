# Pre-merge collision check — `jp/p99-alloc-fixes` against upstream

*Read-only investigation. First pass 2026-09-18 with stale refs; **re-run the same day after JP
authorised a fetch**, so the answers below come from upstream as it actually is. No rebase, no real
merge, nothing pushed, and no commit touching `src/run/engine.py` or `src/run/__main__.py`. The
dry-run merge ran in a disposable worktree that was deleted afterwards.*

**Upstream as fetched 2026-09-18:** `vrgrid26/main` @ **`7dee296`** (17 Sep) — **59 commits** we do not
have. `origin/main` (the old `Stxtics03/vrgrid`) is still `5e0ebf3` (5 Sep) and has nothing we lack;
the active repository is **vrgrid-26**.

---

## 1. `reset_estimator()` fires ONCE PER RUN, not per frame

**Verified in the upstream source**, `src/run/__main__.py`:

```
 61  def iter_pipeline(seq, max_frames, ...)
 ...
 97      # ... (see `ground.reset_estimator`). Runs here, at the first frame's pull.
 98      ground.reset_estimator()
 99      i = 0
100      while True:                      <-- the per-frame loop starts AFTER the reset
```

The call sits inside `iter_pipeline`, after the `loader.scans` generator is created and **before** the
frame loop. One estimator construction per generator, i.e. **per run/replay**, outside the timed
per-frame path. Other call sites follow the same shape: `harness.py:366`, `feature_report.py:84`,
`gen_demo_rrds.py:82`, `gpu_parity.py:76`, each once per sequence/run. `tests/test_ground.py` pins it
(`test_reset_estimator_drops_the_shared_patchworkpp_state`).

**So our per-frame timings are not invalidated by this change.** The 86.49 ms p50 result, the pooled
p99 89.58 ms, and the stage-by-stage breakdown all stand as measurements of the code they were run on.

**[!] Two corrections to the brief that requested this check:**

1. **`03abd7b` is not the `reset_estimator` commit.** It is Shrestha's *"gpu: the frame loop runs on
   the card, bit-identical to CPU on real seq 08"* (16 Sep).
2. **The point-fix is `3c26d47`** (14 Sep, **AakashH2006**), *"determinism: one Patchwork++ estimator
   was shared by every run in a process"*. `6af6907` touches the call again as part of the port.

**What does affect comparability, separately:** upstream's frame loop is now a different program.
`03abd7b` (+45/−14 on `engine.py`, +10/−2 on `__main__.py`) and `6af6907` (+96/−36 and +104/−47) add a
device seam — `DevicePerception`, `DeviceFrame`, `MapEngine(device="cuda")`, `src/gpu/cuda_kernels.py`
— and upstream reports **22 ms/frame on the card**. Our numbers describe the **CPU path before that
port**, on one laptop (D8). They are not wrong; they are no longer the headline configuration.

## 2. Do `d540618` (cleanup LUT) and `697a2bd` (`_centres` fast path) still apply? **Yes — exactly.**

Compared function body against function body, not by hunk headers
(`ast`-extracted from `main` and `vrgrid26/main`):

| function | our base vs current upstream | upstream still has |
|---|---|---|
| `MapEngine._cleanup` | **byte-identical** | `np.copyto(guard, np.isin(occupied, touched))` — the exact line the LUT replaces |
| `MapEngine._centres` | **byte-identical** | the per-ring `sel` mask loop; **no `_centres_sorted`** anywhere upstream |

So the GPU port did **not** rewrite either function. Both proposals apply to current upstream
unchanged, and neither is stale or redundant. The assumption behind `697a2bd` also still holds:
upstream's `_cleanup` still passes `np.flatnonzero(...)`, which is ascending, so the `sorted_slots`
fast path is still valid there.

**One caveat for the review conversation, not a blocker:** with the device path in place, these two
functions are on the **host** path. They still matter for CPU runs and for `MapEngine()` without
`device="cuda"`, but "how much does this save" now depends on which path is being timed. Worth saying
explicitly when the proposals go to Shrestha.

## 3. Dry-run merge against current upstream — the real conflict list

`git merge --no-commit --no-ff vrgrid26/main` at `8a546f8`, in a disposable worktree, nothing
resolved, worktree deleted.

| file | conflict hunks | where |
|---|---|---|
| **`src/run/__main__.py`** | **6** | `iter_pipeline` (×2), its inner `stage()`, `perceive()` (×2), `main()` |
| `scripts/timing_table.py` | 3 | — |
| `dashboard/__main__.py` | 2 | — |
| `OPEN-ITEMS.md` | 2 | — |
| `README.md` | 1 | — |
| **`src/run/engine.py`** | **0 — auto-merged** | both sides changed it; git reconciled them |

58 further files merged cleanly while being modified by upstream.

**Read that engine.py result carefully.** It auto-merged because upstream's `engine.py` changes are in
`__init__`, the device seam and the step order, while ours are inside `_cleanup` and `_centres`, which
upstream did not touch. **A clean auto-merge is not proof of semantic correctness** — the merged file
must still be run (parity script plus our tests) before anyone trusts it.

**The real work is `src/run/__main__.py`.** Upstream refactored `iter_pipeline` into a thin loop over a
new `perceive()` function and added the device branch; our side added the opt-in DL mode
(`semantics_source`, `frnet`, `open_frnet`) plus the `reuse_buffers` scratch wiring, in the same
places. That is a genuine rewrite-versus-rewrite overlap and needs hand-merging into upstream's
`perceive()` shape, not a mechanical resolution.

## 4. Upstream state worth knowing before deciding anything

- **D2/R3 closed upstream** by `df35fd5` (17 Sep), *"grid+eval: close Aakash's open items — D2 ring
  boundary, ring-0 rho, per-ring reference"*. Our OPEN-ITEMS listed it as waiting on Aakash.
- **Dashboard rebuilt** by AakashH2006 (14–15 Sep): the Demo/Details layout, KPI tiles and graphs.
- **GPU lane** now has `05-FLOAT-AUDIT` through `11-AWS-RESUME`, `src/gpu/cuda_kernels.py`,
  `scripts/gpu_parity.py`, and laptop logs under `docs/gpu-lane/laptop/`.
- **AWS is armed but blocked.** `docs/gpu-lane/11-AWS-RESUME.md` (17 Sep) records the account
  answering `OptInRequired` for EC2 and `NotSignedUp` for S3, with a 24 h activation window closing
  **2026-09-18T12:34Z (18:04 IST)**, and `scripts/aws/auto.sh` polling every 10 minutes to run
  preflight → dryrun → stage → launch → setup → run unattended. **No T4 results exist upstream yet**
  (the only JSON under `docs/gpu-lane/` is the VRAM-contention file).
- **`vrgrid26/staging`** is 0 ahead / 46 behind `main` — not where the work is.

### Consequences for our branch

1. **Our AWS runbook is superseded.** Upstream has `docs/gpu-lane/02-AWS-RUNBOOK.md` plus a scripted
   `scripts/aws/t4.sh` and `auto.sh`, with the dataset staging already solved and credit guardrails
   written. `reports/aws-gpu-realtime-runbook.md` should be withdrawn, keeping only the measurement
   steps it adds (the DL real-time figure, GPU reproducibility steps e/f, the 0–200 frame slice) as an
   addendum to theirs.
2. **R-b needs restating, not retracting.** "Gate met, pooled p99 89.58 ms" is true of the CPU path
   before the port. Upstream's 22 ms/frame is a different execution model on different hardware.
3. **The two engine.py proposals are still live** and should go to Shrestha as-is.
4. **The DL-mode work is the merge cost.** Our `__main__.py` changes have to be re-expressed against
   upstream's `perceive()`.
