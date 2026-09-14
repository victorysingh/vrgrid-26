# D11 reopened narrowly: FRNet re-verification and the plan-regret delta

**Scope, fixed by JP on 2026-09-14:** offline evaluation only. Re-measure FRNet's standalone
point accuracy and mIoU, confirm `--fast-scatter` end to end on a clean CPU machine, and run
the plan-regret delta between ground-truth labels and FRNet-predicted labels on real seq 08.
**Not in scope:** `ground.py`, the run engine, the production label source (ground truth), and
putting FRNet into the mapping pipeline. None of these is touched.
**Constraints:** CPU only; no CUDA build of PyTorch is installed locally. Any fine-tune goes to an
AWS g4dn.xlarge, not this machine. Stop at the first genuine blocker.

This is a running log, updated as each step closes.

---

## Status

| step | state |
|---|---|
| 0 machine gate | clean at every check so far (below) |
| 1 `--fast-scatter` wiring | **already wired in both scripts, no change** |
| 2 shim verify | **PASSED** |
| 3 fine-tune | **SKIPPED: not needed for the goal** (see below) |
| 4 eval pretrained checkpoint | **BLOCKED: checkpoint absent.** A (Shrestha) pending, then B |
| 5 end-to-end speedup | blocked on 4 |
| 6 plan-regret delta | script being built; FRNet arm blocked on 4 |

## Step 1: `--fast-scatter` wiring (read directly, not assumed)

- `scripts/frnet_eval.py`: `--fast-scatter` flag present. Calls
  `frnet_fast_scatter.enable(verify=True)` right after importing the port, **before** the
  checkpoint is loaded and before any `model.predict`.
- `scripts/frnet_finetune.py`: `--fast-scatter` flag present. Calls `enable(verify=True)` before
  building the model and before training. Saves `state_dict`, `optimizer`, `step` and the full
  `recipe` (including `fast_scatter`) beside the weights.

Neither script needed any integration.

## Step 2: the shim's own verify, fresh

Command `python scripts/frnet_fast_scatter.py`, `torch 2.13.0+cpu`, CUDA unavailable, 10 threads.

- State before: clock 2400/2400 MHz, commit 11.94/15.73 GB, free 7.17 GB. **OK.**
- State after: clock 2400/2400 MHz, commit 11.83/15.73 GB, free 7.51 GB. **OK.**

```
verifying on cpu: 20,000 rows x 64 ch into 4,000 slots (28 empty)
  scatter_max   forward  max abs diff 0.000e+00 (0.0 ulp), 0 of 254,208 values differ, empty-slot convention matches
  scatter_mean  forward  max abs diff 0.000e+00 (0.0 ulp), 0 of 256,000 values differ, empty-slot convention matches
  scatter_max   backward max abs diff 0.000e+00
  scatter_mean  backward max abs diff 0.000e+00
  verified: scatter_max exact in both directions, scatter_mean within float32 rounding

benchmark on cpu: 124,000 x 256 into 25,000 slots
  scatter_max   loop     3626.1 ms   shim   23.31 ms        156x
  scatter_mean  loop     3540.5 ms   shim   55.41 ms         64x
```

- **`scatter_max` is bit-identical** (gate: exactly 0).
- **`scatter_mean` is also bit-identical on CPU**, so it sits inside its 4-ulp bound with room to
  spare. The stated 2-ulp divergence is a CUDA effect and can't occur on this CPU-only machine.
- The per-call speedup (156× / 64×) agrees with the 11 Sep re-verification (144× / 64×). That is
  a per-call figure; the end-to-end figure is Step 5's job.

## Step 3: fine-tuning skipped, because the goal does not need it

The question is whether our port still reproduces **90.3% / 65.2%**. Those figures belong to the
**pretrained checkpoint** itself, not to any fine-tune. `docs/research-log.md`, 2026-09-04
(Shrestha), held-out seq 08, 200 frames:

| run | recipe | point acc | mIoU |
|---|---|---|---|
| — | **pretrained checkpoint** | **90.3%** | **65.2%** |
| A | head, 3× terrain/vegetation, 600 steps | 89.8% | 64.6% |
| B1 | head, no class weights, lr 1e-4, 2,000 steps | 90.2% | 65.3% |
| B2 | head+backbone, lr 1e-4, 4,000 steps | 89.5% | 64.5% |

Every fine-tune was tried and rejected on measurement, because the checkpoint was already trained
on 00–10 minus 08. So Step 4 (evaluating the pretrained checkpoint with `--fast-scatter`) answers
the question on its own. **Fine-tuning is an optional follow-up, and if wanted it runs on AWS, not
here.**

## Step 4 prerequisite: the checkpoint

`checkpoints/frnet-semantickitti_seg.pth` is absent, and there is no `checkpoints/` directory.
`*.pth` is gitignored. **No hash, size or download URL for the 4 Sep file is recorded anywhere** in
`docs/`, `configs/` or `reports/`. `configs/frnet.yaml` only names it and cites 73.3% mIoU.

- **A:** JP is asking Shrestha whether the 4 Sep file still exists. Preferred, because then the
  numbers are a direct re-check of the same file.
- **B, only if A fails:** fetch the authors' released SemanticKITTI checkpoint, recording the exact
  URL and SHA-256. [!] With no recorded hash for the 4 Sep file, B cannot prove it is *the same
  file*. Its numbers will be reported as their own result and compared with 90.3% / 65.2%, not
  claimed as "the same run".
- **C:** off the table.
