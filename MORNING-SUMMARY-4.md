# Morning summary 4 — D11 reopened narrowly, and R-j (2026-09-14)

Covers everything from JP reopening D11 until the reproducible Step 6 result. Earlier passes
tonight are in `MORNING-SUMMARY-3.md`. All work is on the local branch `jp/p99-alloc-fixes`.
**Nothing was pushed and no PR was opened.**

---

## Still open, first

1. **Everything depends on pinning the thread count.** At the default 10 threads, FRNet predictions are
   **not reproducible per point, on either path** (with or without `--fast-scatter`). Any FRNet
   number that has to reproduce needs `torch.set_num_threads(1)` (`--threads 1` on the delta script).
2. **The lateral plan-regret delta is sensitive to small label changes.** About 0.06% of labels
   changing moved it by +0.118 (about 26%). It is clearly positive; its exact value is less firm than
   the longitudinal one.
3. **Scope limits on the deliverable:** one schedule (`5/10/20/40`), one 200-frame slice of seq 08,
   motion held at ground truth, and uncertainty estimated only roughly (no paired per-query analysis).
4. **The checkpoint is not the 4 Sep file.** That file is gone. Shrestha doesn't have it, and it was
   never tracked in either repo.

## The checkpoint (framing for every number below)

- **Source:** the authors' public release, a Google Drive folder linked from
  https://github.com/Xiangxu-0103/FRNet, file id `1Ez-fpwu2WFCBw8usjwxUz6XQruw-cGU4`.
- **Downloaded:** 2026-09-14T07:27:54Z.
- **SHA-256:** `09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`. This is the only
  provenance record; no checksum is published.
- **Corroborated:** the bytes matched Google's served `crc32c`, and JP's separate browser download
  hours later has the identical SHA-256.
- Record: `checkpoints/frnet-semantickitti_seg.pth.PROVENANCE.md`. The `.pth` itself is not tracked.

## D11, step by step

| step | outcome |
|---|---|
| 0 machine gate | enforced before every run; one launch aborted on a paging machine (see Corrections) |
| 1 `--fast-scatter` wiring | already wired in `frnet_eval.py` and `frnet_finetune.py`; nothing changed |
| 2 shim verify | passed on CPU: both reductions 0.000e+00 forward and backward |
| 3 fine-tune | **skipped**: 90.3% / 65.2% are the pretrained checkpoint's figures; every 4 Sep fine-tune was rejected |
| 4 evaluate | **90.3% point accuracy / 65.2% mIoU (15 classes) / 61.1% drivable**; matches the record at printed precision |
| 5 timing | 5.82× wall time at 10 threads, but the check caught per-class instability → R-j |
| 6 plan-regret delta | **reproducible: +1.069 longitudinal / +0.567 lateral** (below) |

### The deliverable: plan regret, ground-truth labels vs FRNet labels (real seq 08, 200 frames)

`scripts/plan_regret_frnet_delta.py --fast-scatter --threads 1`, run twice. **The two runs are
bit-identical**: FRNet map digest `41ad78eedfa97f06` both times.

| family | R, ground-truth labels | R, FRNet labels | **delta** |
|---|---|---|---|
| longitudinal | 1.160 | 2.230 | **+1.069** (+92%) |
| lateral | 1.142 | 1.710 | **+0.567** (+50%) |

- **The baseline is trusted.** The script's oracle control (ground truth fed through the FRNet code
  path) gave bit-identical maps and delta 0.000 at 40 and 200 frames. The ground-truth arm's digest
  (`2f0f5636f3033c9c`) is identical in every run.
- **The earlier 10-thread run** (+0.997 / +0.449) is superseded. It was one non-reproducible draw.
- FRNet labels: 90.3% point accuracy. All 64 queries found in both arms, none blocked, 100% common
  support.

## R-j: the `--fast-scatter` reproducibility finding (tracked separately from D11)

**Resolved.** A clean machine, gated before every pass, 20 frames, per-point predictions:

| comparison | points differing |
|---|---|
| fast @10 threads vs fast @10 threads | 1,895 |
| loop @10 threads vs loop @10 threads | 1,446 |
| fast @1 thread vs fast @1 thread | **0** |
| loop @1 thread vs loop @1 thread | **0** |
| **fast @1 thread vs loop @1 thread** | **0** |

- **The shim is exact** in the full model once threads are pinned. The run-to-run variance is PyTorch's
  intra-op thread pool, on **both** paths; the shim never introduced it.
- **The reproducible like-for-like speedup is about 7.0×** at 1 thread (129 s vs 898–914 s for 20
  frames, identical outputs). End to end on 200 frames, the FRNet arm took 6.1 s per frame at 1 thread
  with the shim, against 19.5 s per frame on the loop path at 10 threads.
- **The obvious first hypothesis** (the shim's `scatter_reduce` going multithreaded) was tested
  directly and refuted before the thread-pool mechanism was confirmed.

## Corrections made along the way (all on the record, none rewritten)

- **`8f74f30` blamed the shim** for the variance; `1a47960` withdrew that after the probe refuted it.
- **The first diagnostic launch ran on a paging machine** (commit 17.29 GB > 15.73 GB, Chrome had
  reopened), because only Option 2 was gated. It was stopped mid-pass with nothing used (`d4ef072`),
  and every later job gated before every pass.
- **Estimates that were wrong:** the single-thread loop pass took about 15 min, not about 8 (and not the
  40 first quoted). The 5.82× is a wall-time ratio between non-reproducible runs, not a like-for-like
  speedup.
- **Bash quoting failures** three times while staging text; each was confirmed to have run nothing, then
  redone with files.

## Commits in this pass (local only)

`aea5c68` D11 reopened · `8acd9f1` steps 1–3 · `91e354b` option B source · `a594255` delta script +
oracle control · `eff969a` checkpoint provenance · `4675e95` Step 4 · `8f74f30` Step 5 (attribution
later withdrawn) · `1a47960` correction + probe · `0de49f7` R-j opened · `d4ef072` aborted launch
recorded · `df6c040` R-j round 1 · `68076e3` Step 6 (10-thread) + D11 closed · `8e20d57` second
download corroborates · `b20e03e` R-j resolved + `--threads` · then the reproducible Step 6 and this
summary.

## Not touched

`ground.py`, the run engine, `semantics.py`, the production label source, `src/perception/frnet/`, and
`include/vrgrid/`. FRNet stays out of the mapping pipeline.
