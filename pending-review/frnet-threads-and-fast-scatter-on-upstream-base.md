# pending-review: `--threads` and `--fast-scatter` on Shrestha's device-path FRNet

**Status: written, tested, NOT applied and NOT merged anywhere.** This is a patch against
`vrgrid26/main` @ `ae85979`, produced in a throwaway worktree. Nothing on `jp/p99-alloc-fixes`
implements it, and nothing has been pushed to `vrgrid26`. **It needs Shrestha's agreement before it
goes anywhere** — the file it changes is the one his device-path work lives in.
**Patch:** `pending-review/frnet-threads-and-fast-scatter-on-upstream-base.diff`.

## What this is, and why this direction

Both branches shipped `--semantics {gt,frnet}` independently, same flag, same default. They are not
the same size of thing:

- **Shrestha's runs FRNet on the DEVICE path** — `perceive()` calls `frnet.infer_points()` *before*
  `perception.launch()` and passes `semantic_pred=pred` into the kernel, which required widening
  `DevicePerception.launch()`. **That is what produced the 10 Hz DL figure.**
- **JP's is host-only**, and adds the two things that make the mode reproducible: `--threads`
  (OPEN-ITEMS R-j) and `--fast-scatter`.

So the device integration is the larger piece and stays as the base. This patch moves only the two
reproducibility knobs onto it.

## The whole change is pre-construction

The knobs change **how the model is built, not how it is used**. `perceive()` only ever receives an
already-constructed `frnet` object, so **nothing in the device seam is touched**: not
`perception.launch()`, not `semantic_pred=`, not `semantics_from_prediction()`, not a kernel, not
`--semantics-every`'s label cache, not `--semantics-precision`.

| # | where | change |
|---|---|---|
| 1 | `build_parser()` | `--threads N`, `--fast-scatter` |
| 2 | `main()` | `SystemExit` if either is passed without `--semantics frnet` |
| 3 | `iter_pipeline()` signature | `semantic_threads: int \| None = None`, `semantic_fast_scatter: bool = False` |
| 4 | `iter_pipeline()` top | argument validation, before any work |
| 5 | `iter_pipeline()` FRNet block | `torch.set_num_threads()` and the shim, **before** `FRNetInference(...)` |
| 6 | `tests/` | `test_frnet_reproducibility_knobs.py`, 6 tests |

Call-site plumbing for `scripts/timing_table.py` and `dashboard/__main__.py` is **not** in this
patch — they do not currently expose `--semantics-precision` either, so that is one consistent
follow-up rather than two half-measures.

## `--fast-scatter` is host-only, deliberately

JP's branch verified the shim bit-identical to the port's frustum loops **on CPU at one thread**. On
the card it is **unverified** — that is R-j's open GPU half — and **the 10 Hz figure does not use
it.** So the patch refuses the combination rather than silently applying an unchecked shim:

```
ValueError: --fast-scatter is host-only: the shim is verified against the port's
loops on CPU at one thread, not on the device (OPEN-ITEMS R-j). Got device='cuda'.
```

## Two things found while writing it, both worth keeping

1. **Validation must come first in `iter_pipeline`.** Placed after `loader.scans(...)`, a rejected
   call still built a generator; placed after the device block, a machine with no card answers *"no
   CUDA device"* — true, and not the reason. It now validates before any work. The first placement
   was caught by a real failure: `test_build_allocates_nothing_per_frame` broke because an abandoned
   half-built generator perturbed its `tracemalloc` baseline. Validating first fixed it.
2. **`--threads` is not a convenience knob.** `d48831a` and `docs/research-log.md` attribute FRNet's
   0.028–0.046% run-to-run label disagreement to *"the model"*, without stating a thread count. R-j
   measured the cause as PyTorch's **intra-op thread pool**: at the default count the labels do not
   reproduce; at `torch.set_num_threads(1)` they are exact. If those runs used the default, this
   knob is the fix for that finding, not an optimisation.

## Verification

Against `ae85979` with the patch applied, under an isolated import root (the editable install's
meta-path finder otherwise resolves `vrgrid` to the wrong tree):

- **755 passed, 28 skipped, 0 failed** — upstream's 749 plus the 6 new ones.
- Upstream's own `-k "semantic or frnet"` selection: **23 passed**.
- `ruff check` clean on both changed files.
- Not run: anything on a CUDA device. The host-only guard is tested by the refusal path, not by a
  GPU run.
