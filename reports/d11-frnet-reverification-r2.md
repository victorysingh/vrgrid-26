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
| 6 plan-regret delta | **script built; oracle control PASSED at 40 and 200 frames**; FRNet arm blocked on 4 |

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

### Option B's source, identified but NOT downloaded (2026-09-14)

Found by reading, not fetching. Nothing is downloaded while A is pending.

- **Official repository:** https://github.com/Xiangxu-0103/FRNet. Its README says: *"We provide
  the trained models for SemanticKITTI and nuScenes. The checkpoints can be downloaded from
  here"*, linking a Google Drive folder.
- **Folder:** https://drive.google.com/drive/folders/173ZIzO7HOSE2JQ7lz_Ikk4O85Mau68el
- **File:** `frnet-semantickitti_seg.pth`, **38.5 MB**, modified **7 Dec 2023**, as shown in the
  folder listing. It has the same name the scripts and `configs/frnet.yaml` expect. The folder
  also holds `frnet-nuscenes_seg.pth` (38.5 MB), which is **not** the one to fetch.
- **[!] No checksum is published,** by the README or the folder. If B is used, the SHA-256
  computed on download becomes the only provenance record, alongside the URL and the listed
  size and date. That still can't prove the file is identical to the 4 Sep copy, which has no
  recorded hash either.


## Step 6 prep: `scripts/plan_regret_frnet_delta.py`

Built while the checkpoint is pending. Real seq 08, not the synthetic sequence, whose own docs
say regret reads ~0.000 there. The docstring states that "R2" elsewhere in the docs is a
different, prior research document.

**What it compares,** on the same frames (0–199 by default, the slice `frnet_eval.py` scores):

- **M\*:** the reference map, built from ground-truth labels exactly as
  `eval_synthetic.py --seq` builds it.
- **M_gt:** the shipped schedule `5/10/20/40`, driven by `harness.real_scans` unchanged.
- **M_frnet:** the same schedule, scans, poses and ground masks, with each static point's class
  replaced by FRNet's prediction.
- **delta = R_frnet − R_gt**, via `eval_synthetic.plan_regret_for` (imported, not copied), on the
  two maps' common support. Both query families are reported; the script doesn't pick one.

**Design decisions, stated in the script:**

- **Motion stays ground truth by default (`--motion gt`).** `run_sequence` removes dynamic returns
  using raw `moving-*` ids before it reads a class, and FRNet has no motion output. Keeping those
  ids isolates segmentation quality, as the live pipeline already does. `--motion none` exists,
  and it mixes in the loss of motion information.
- **FRNet's ignore slot (19) maps to `CLASS_UNLABELLED`** (31), the same place ground truth's
  unlabelled points go.
- **Ground masks are identical in both arms.** They come from Patchwork++, and the script refuses
  to run on the label fallback.
- **The D1 singleton is reset before each of the three passes** (`ground._estimator = None`, no
  edit to `ground.py`).
- **Inference is `frnet_eval.py`'s exact construction and `model.predict` call.** The script also
  prints per-point accuracy over the same frames as a cross-check against Step 4.

**Tests:** `tests/test_plan_regret_frnet_delta.py`, 5 passed. They cover moving ids kept, static
points relabelled, the ignore slot mapped, oracle labels reproducing what the harness scatters,
and misaligned predictions refused.

### Oracle control: ground truth through the FRNet arm's code path, seq 08, frames 0–39

```
reference map (ground truth): ReferenceMap(3466x3174 @ 5 cm, 784,377 observed cells)   [2s]
M_gt built    [7s]  digest 930145f3707c0377
M_oracle built  [7s]  digest 930145f3707c0377
CONTROL maps bit-identical: True
per-point accuracy of the second arm's labels vs ground truth: 100.0% over 4,548,035 labelled points
common support: 99.0% of the planning window

  family            R_gt  R_second    delta  found gt/2nd  blocked gt/2nd
  longitudinal     0.171     0.171   +0.000         59/59             0/0
  lateral          0.104     0.104   +0.000         60/60             0/0
```

- **The plumbing is exact:** identical map digest, and a delta of exactly 0.000 in both families.
- **Real-data regret is non-zero** (0.171 / 0.104), so unlike the synthetic sequence this setup
  can register a difference.
- **It caught one defect:** an RMSE print read an attribute `Result` does not have and silently
  printed nothing. Removed, along with the unused `evaluate()` calls, then the control was re-run
  on the edited script (below).

### Oracle control at the full 200 frames, on the edited (committed) script

```
sequence 08, frames 0-199, schedule 5/10/20/40, motion=gt, second arm = oracle
reference map (ground truth): ReferenceMap(3620x6066 @ 5 cm, 2,657,064 observed cells)   [9s]
M_gt built    [35s]  digest 2f0f5636f3033c9c
M_oracle built  [35s]  digest 2f0f5636f3033c9c
CONTROL maps bit-identical: True
per-point accuracy of the second arm's labels vs ground truth: 100.0% over 22,741,893 labelled points
common support: 100.0% of the planning window

  family            R_gt  R_second    delta  found gt/2nd  blocked gt/2nd
  longitudinal     1.160     1.160   +0.000         64/64             0/0
  lateral          1.142     1.142   +0.000         64/64             0/0
```

- **Exact over the full slice the FRNet run will use:** identical digest, delta 0.000 in both
  families, all 64 queries found in both arms, none blocked.
- **R_gt at 200 frames is 1.160 / 1.142.** That is the ground-truth baseline the FRNet arm will be
  compared against on the same frames. It is not the same number as the 40-frame 0.171 / 0.104,
  because the planning window is placed at the vehicle's final pose and so moves with the frame
  count.
- **The map-building cost per arm is 35 s for 200 frames** (M\* 9 s). FRNet inference will be added
  to the second arm, and its CPU cost is measured in Steps 4–5, not estimated here.
