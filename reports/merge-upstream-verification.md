# Verifying the upstream merge — what was and was not checked

*`jp/p99-alloc-fixes` ← `vrgrid26/main` @ `7dee296`, 2026-09-18. Rollback branch:
`jp/p99-pre-upstream-merge` @ `6882212`.*

**Read this limitation first: `scripts/gpu_parity.py` was NOT run.** It requires a CUDA
device; `torch.cuda.is_available()` is `False` on this laptop, and installing a CUDA build
of PyTorch locally is ruled out. **The GPU seam in the auto-merged `src/run/engine.py` is
therefore unverified by execution.** It must be run on the T4 before anyone relies on it.
What follows is the strongest check that *is* possible without a card.

## 1. Host-path parity — merged vs pure upstream: bit-identical

`engine.py` auto-merged (0 conflicts), which is not proof of correctness. The question that
matters for our side of it: do our two changes — the `_cleanup` LUT (`d540618`) and the
`_centres` sorted fast path (`697a2bd`) — alter upstream's output once layered on top?

Real seq 08, frames 0–19, GT labels, Patchwork++ on, schedule `5/10/20/40`. Each frame
folded into `MapEngine(ghost_removal=True)`; after every frame, all integer `StepCounters`
fields and `gpu.kernels.map_hash(engine.handle.grid)` (blake2b over the SoA arrays) recorded.

| comparison | result |
|---|---|
| **merged vs pure upstream `7dee296`** | **identical on all 20 frames** — every counter and every map hash |
| merged vs pre-merge `6882212` | differs on 20/20 frames — **expected, upstream's own grid changes** |

Final map hash, frame 19: merged `454885f091211284565145e6f6f8fd44` = upstream
`454885f091211284565145e6f6f8fd44`; pre-merge `cae34119aadb48996961c603f076e5e6`.

The merged-vs-pre-merge divergence begins at **frame 0** and is in `binned` (123,389 vs
123,254), `out_of_view` (1,430 vs 1,495), `protected`, `cells_touched`, `occupied`, `tested`
— i.e. which points reach the grid at all. That is the footprint of upstream's ring-boundary
and ring-0 rho work (`df35fd5`, D2), not of anything on this branch. **It does mean every
map-content number measured on the pre-merge branch describes a different grid.**

### A false pass, corrected

The first attempt at this comparison ran the pre-merge worktree with `PYTHONPATH` pointing at
it. **The editable install's meta-path finder takes precedence over `sys.path`, so both runs
imported the same merged file** (identical sha256) and "agreed" trivially. The real runs strip
that finder from `sys.meta_path` and reach each tree through a directory junction; the import
provenance is asserted before each run (pre-merge `4c938b5e…`, no device seam; upstream
`eb4f0ec6…`, no `_centres_sorted`; merged `021f473c…`, both).

*Machine was paging during these runs (commit 23.97 GB > 15.73 GB physical). That invalidates
timing, which is why none was taken here; hashes and counters are unaffected.*

## 2. Test suite on the merged tree

- **808 passed, 28 skipped, 0 failed** (full suite).
- `tests/test_determinism.py` **13 passed** — the D1 symptom is gone on this branch, because
  the merge brings in `3c26d47`'s per-run `ground.reset_estimator()`.
- Our six DL-mode/allocation test files: 46 passed.
- Ruff on the whole tree: 3 errors, none from the merge — `tests/test_metrics.py:472` E741
  (known, R-d, flag-only) and two upstream E731s in `scripts/engine_eval.py:81` and
  `scripts/eval_synthetic.py:287`.

## 3. The DL mode re-expressed against upstream's `perceive()`

`src/run/__main__.py` was the real work (6 conflict hunks). Upstream's file was taken whole and
our features re-applied into its new shape: `semantics_source` / `frnet` threaded through
`iter_pipeline` → `perceive`, `open_frnet()` lifted verbatim, the `reuse_buffers` scratch, and
the `--semantics` / `--fast-scatter` / `--threads` flags.

Smoke-tested on real frames, both modes: GT and `--semantics frnet` each complete and agree on
the map counters (`ghost removal: 7,524 cells cleared, 22,416 spared`, `ground: Patchwork++`).

**Guard ordering fixed during verification.** The `semantics_source` validation originally sat
after upstream's `resolve_device(device)`, so `--semantics frnet --device cuda` on a machine
without a card raised upstream's `RuntimeError` about a missing CUDA device instead of the real
reason. It now validates first; all three guards report their own cause:

```
device+frnet : semantics_source='frnet' is host-only; it cannot be combined with device='cuda'
missing model: semantics_source='frnet' needs frnet=open_frnet() (a semantics.FRNetInference)
unknown      : semantics_source must be 'gt' or 'frnet', not 'pointnet'
```

## 4. Still open after this merge

- **`gpu_parity.py` on a CUDA device** — the unverified part, above.
- **R-b restated, not retracted.** Pooled p99 89.58 ms describes the pre-port CPU path; and per
  §1 it describes a pre-`df35fd5` grid too. Upstream's 22 ms/frame is a different execution
  model on different hardware.
- ~~Our AWS runbook is superseded.~~ **DONE 2026-09-18:** `reports/aws-gpu-realtime-runbook.md`
  was demoted and renamed **`reports/aws-dl-realtime-addendum.md`**. Launch, connect and
  transfer now defer to `docs/gpu-lane/02-AWS-RUNBOOK.md` + `scripts/aws/t4.sh`; what stays is
  the DL-mode measurement protocol and four deltas (494 MB slice not 84.8 GB, 201 frames not
  200, git bundle for an unpushed branch, `tar --force-local` on Windows).
- `d540618` and `697a2bd` still awaiting Shrestha's review — now with the note that they are
  host-path functions since the port.
