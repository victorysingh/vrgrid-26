# D11 reopened narrowly: FRNet re-verification and the plan-regret delta

**Scope, fixed by JP on 2026-09-14:** offline evaluation only. Re-measure FRNet's standalone
point accuracy and mIoU, confirm `--fast-scatter` end to end on a clean CPU machine, and run
the plan-regret delta between ground-truth labels and FRNet-predicted labels on real seq 08.
**Not in scope:** `ground.py`, the run engine, the production label source (ground truth), and
putting FRNet into the mapping pipeline. None of these is touched.
**Constraints:** CPU only; no CUDA build of PyTorch is installed locally. Any fine-tune goes to an
AWS g4dn.xlarge, not this machine. Stop at the first genuine blocker.

This is a running log, updated as each step closes.

> **[!] FRAMING, UP FRONT, FOR EVERY NUMBER BELOW.** Steps 4–6 use an **independently sourced
> checkpoint from the FRNet authors' public release**:
> - URL: https://drive.google.com/drive/folders/173ZIzO7HOSE2JQ7lz_Ikk4O85Mau68el (file id
>   `1Ez-fpwu2WFCBw8usjwxUz6XQruw-cGU4`), linked from https://github.com/Xiangxu-0103/FRNet
> - Downloaded: 2026-09-14T07:27:54Z
> - **SHA-256 `09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`**
>
> It is **NOT a reproduction of the original 4 Sep file** behind 90.3% / 65.2%, which no longer
> exists anywhere and never had a recorded hash. This SHA-256 is the only provenance record there
> will ever be. **If its numbers differ from 90.3% / 65.2%, that is expected and is reported as
> its own result, not chased as an error.** They may simply be two different, both legitimate
> checkpoints. Full record: `checkpoints/frnet-semantickitti_seg.pth.PROVENANCE.md`.

---

## Status

| step | state |
|---|---|
| 0 machine gate | clean at every check so far (below) |
| 1 `--fast-scatter` wiring | **already wired in both scripts, no change** |
| 2 shim verify | **PASSED** |
| 3 fine-tune | **SKIPPED: not needed for the goal** (see below) |
| 4 eval pretrained checkpoint | **DONE: 90.3% point accuracy / 65.2% mIoU / 61.1% drivable, on the independently sourced checkpoint** |
| 5 end-to-end speedup | **timed (5.82x wall, all runs trusted) but its correctness check FAILED: `--fast-scatter` runs disagree per class, run to run. STOPPED.** Cause not established |
| 6 plan-regret delta | **DONE, REPRODUCIBLE: +1.069 longitudinal / +0.567 lateral** (`--fast-scatter --threads 1`, two bit-identical runs); supersedes the 10-thread +0.997 / +0.449 |

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

## Step 4: evaluation of the downloaded checkpoint (`--fast-scatter`, CPU)

**Checkpoint:** the authors' public release, SHA-256
`09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`. **Independently sourced, NOT a
reproduction of the 4 Sep file.** Before use, the downloaded bytes matched Google's served
`crc32c=Qaiqow==`, so there was no transport corruption, and the SHA-256 was re-checked unchanged.

- State before: clock 2400/2400 MHz, commit 12.1/15.73 GB, free 6.89 GB. **OK.**
- State after: clock 2400/2400 MHz, commit 11.83/15.73 GB, free 7.65 GB. **OK.**
- `torch 2.13.0+cpu`, 10 threads. The shim self-verified at start-up: max and mean both 0.000e+00,
  forward and backward.
- Wall time **666 s** for 200 frames, end to end: start-up, verify, model load, data, inference,
  scoring. About 3.3 s per frame. This single run is not the Step 5 timing.

```
frnet-semantickitti_seg.pth on cpu: 0 missing, 8 unexpected tensors (auxiliary heads are training-only)
sequence 08, 200 frames, 22,741,893 labelled points
  reductions                torch.scatter_reduce (--fast-scatter)
  point accuracy             90.3%
  mIoU over 15 present classes   65.2%   (paper: 73.3% over all 4,071 frames)

  car 97.9  road 97.4  bicyclist 91.2  sidewalk 89.2  building 89.1  trunk 79.4  terrain 73.8
  vegetation 70.8  person 70.0  pole 55.1  bicycle 48.0  parking 45.2  traffic-sign 41.3
  fence 29.4  other-ground 0.0
  no ground truth in this slice, excluded: motorcycle, truck, other-vehicle, motorcyclist
  §7.1 drivable set only: mIoU 61.1% over 5 classes
```

### How this compares with the recorded 90.3% / 65.2%, stated plainly

- **The numbers match the recorded figures exactly at the printed precision:** point accuracy
  90.3%, mIoU 65.2% over the same 15 present classes, drivable-set 61.1%. The same 22,741,893
  labelled points were scored, and `other-ground` is again present at IoU 0.0% over 150 points.
- **The per-class IoUs agree too, at the level the record allows.** The 4 Sep entry records that
  the 15 per-class IoUs sum to 977.7. Today's printed, rounded values sum to 977.8, which is
  within rounding (15 values × ±0.05).
- **What this does and does not establish.** It shows that our port, with this public checkpoint,
  produces the reported figures on the reported slice. It does **not** prove that this file is
  byte-identical to the 4 Sep copy: that copy had no recorded hash and no longer exists. The most
  likely explanation is that the 4 Sep file *was* this same public release. It carries the same
  name, `configs/frnet.yaml` cites the same 73.3%-mIoU release, and the metrics match to every
  printed digit. That remains an inference, not a verification.

## Step 5: end-to-end timing, and the correctness check that failed

**Stopped here, and Step 6 was not run.** The timing itself is trustworthy. The check that is
supposed to make it a like-for-like speedup failed, and the cause is not established.

Harness: `reports/harnesses/frnet_eval_timing.py`. It runs `frnet_eval.py` end to end in fresh
processes, alternating the arms (fast, loop, loop, fast), records machine state before and after
every run, and re-checks the checkpoint SHA-256. Evidence: `reports/bench/frnet_eval_timing.json`.

| run | arm | wall | state before / after | acc | mIoU |
|---|---|---|---|---|---|
| 1 | fast | 670.2 s | OK / OK | 90.3 | 65.2 |
| 2 | loop | 3953.8 s | OK / OK | 90.3 | 65.2 |
| 3 | loop | 3855.6 s | OK / OK | 90.3 | 65.2 |
| 4 | fast | 671.3 s | OK / OK | 90.3 | 65.2 |

All four runs were trusted: 2400/2400 MHz throughout, commit 11.8–12.1 of 15.73 GB. **Median fast
670.7 s, median loop 3904.7 s, so the end-to-end wall-time ratio is 5.82×.** That ratio is measured,
not estimated. It is far below the 64–156× per-call figure, because the rest of the forward pass is a
large fixed cost on CPU.

### [!] The failed check: per-class IoU is not reproducible on the fast path

On CPU the shim was believed bit-identical, so every run in both arms had to print identical metrics.
They did not:

| class | Step 4 (fast) | run 1 fast | run 2 loop | run 3 loop | run 4 fast |
|---|---|---|---|---|---|
| bicycle | 48.0 | **48.2** | 48.0 | 48.0 | 48.0 |
| parking | 45.2 | **45.3** | 45.2 | 45.2 | **45.3** |
| person | 70.0 | **70.1** | 70.0 | 70.0 | 70.0 |
| traffic-sign | 41.3 | 41.3 | 41.3 | 41.3 | **41.4** |
| trunk | 79.4 | 79.4 | 79.4 | 79.4 | **79.3** |
| the other 10 classes | identical in every run | | | | |

- **The loop arm is reproducible:** runs 2 and 3 agree exactly.
- **The fast arm is not reproducible run to run.** Runs 1 and 4 differ from each other, and Step 4's
  fast run differs from both while matching the loop arm exactly.
- **The headline figures are unaffected at printed precision.** Point accuracy 90.3%, mIoU 65.2% and
  drivable 61.1% are the same in all five runs. The instability is 0.1–0.2 pp in five classes, mostly
  small ones (bicycle has 6,899 ground-truth points, person 33,383).

### The obvious cause was tested and is NOT it

Hypothesis: PyTorch's CPU `scatter_reduce` goes multithreaded at production shapes, so its summation
order, and with it the float rounding, varies between runs.
`reports/harnesses/shim_determinism_probe.py` calls the shim's reductions alone at the benchmark
shape (124,000 × 256 into 25,000 slots), repeated 6 times, at 10 threads and at 1 thread:

```
scatter_max   threads=10 repeats bit-identical: True  (0 differing)  == loop in 6/6
scatter_max   threads=1  repeats bit-identical: True  (0 differing)  == loop in 6/6
scatter_mean  threads=10 repeats bit-identical: True  (0 differing)  == loop in 6/6   max |fast-loop| 0.000e+00
scatter_mean  threads=1  repeats bit-identical: True  (0 differing)  == loop in 6/6   max |fast-loop| 0.000e+00
```

(For `scatter_max` the probe prints `max |fast-loop|` as `nan`, because empty slots are `-inf` in both
and `-inf − -inf` is `nan`. `torch.equal` is the real check, and it passed 6/6.)

**On random inputs at the real shape, the reductions are deterministic and exactly equal to the
loop.** So the hypothesis is refuted as stated.

- **Known:** in the full evaluation, three `--fast-scatter` runs gave three different per-class
  results, while two loop runs gave identical ones.
- **Not known:** where the divergence comes from. Untested candidates include real (not random) index
  and value patterns reaching a different kernel path, run-to-run variation elsewhere in the forward
  pass, and a chance agreement between the two loop runs. None is established, and none was chased
  further, as instructed.

**[!] Correction to commit `8f74f30`.** Its message says the shim's "bit-identical on CPU" does not hold
at production shapes. The probe above contradicts that: the reductions alone are exact at production
shapes on random inputs. The observed run-to-run disagreement stands, but its attribution to the shim
is withdrawn.

**What must not be said from this:** that `--fast-scatter` is a verified like-for-like 5.82× on CPU.
Only the wall-time ratio is established.

## R-j diagnostic, first launch: aborted on an untrusted machine (no result used)

The 20-frame per-point prediction diff (`reports/harnesses/frnet_pred_diff.py`: fast ×2, fast with `torch.set_num_threads(1)` ×2, loop ×1), chained to Option 2, was launched at 16:08:55. Its first state line read **commit 17.29 GB against 15.73 GB physical, free 3.17 GB: UNTRUSTED (paging)**. By process group, chrome had 48 processes and 4.9 GB. It was stopped during the first pass (commit had reached 19.51 GB, of which 2.3 GB was the pass itself). **Nothing from it is used.**

[!] **Process miss, corrected.** The launch gated only Option 2, not the diagnostic passes, against the standing rule that every run is gated. The relaunch gates before every pass and before Option 2, and stops the moment a gate fails. Option 2 had not started.

## R-j diagnostic (relaunched on a clean machine): the thread pool is the mechanism, and one question remains open

Relaunched at 16:15:42 after the apps were closed. **Every pass was gated first, and every gate
passed:** 2400/2400 MHz, commit 12.04–12.46 of 15.73 GB. The harness is
`reports/harnesses/frnet_pred_diff.py` (it transcribes `frnet_eval.py`'s inference and saves
per-point predictions). Frames 0–19 of seq 08, 2,471,164 points, checkpoint SHA-256 re-checked in
every pass. `--threads 1` calls `torch.set_num_threads(1)` as the first torch call in the process,
so it covers the whole evaluation, not just the shim. The inter-op pool stayed at its default of 10.

| pass | `--fast-scatter` | intra-op threads | wall | point accuracy (20 frames) |
|---|---|---|---|---|
| fastA | yes | 10 | 69 s | 92.6835% |
| fast1a | yes | **1** | 129 s | 92.6882% |
| fastB | yes | 10 | 69 s | 92.6886% |
| fast1b | yes | **1** | 129 s | 92.6882% |
| loop | no | 10 | 405 s | 92.6849% |

Pairwise differing points:

```
rj_fastA   vs rj_fastB    1,895 points differ, in 20/20 frames   (max 175 in one frame)
rj_fast1a  vs rj_fast1b       0 points differ, in  0/20 frames
rj_fastA   vs rj_loop     1,572 points differ, in 20/20 frames
rj_fastB   vs rj_loop     1,295 points differ, in 20/20 frames
rj_fast1a  vs rj_loop     3,088 points differ, in 20/20 frames   (fast1b identical to fast1a)
rj_fastA   vs rj_fast1a   3,172 points differ;  rj_fastB vs rj_fast1a 2,789
```

### What this establishes

- **Single-threading eliminates the run-to-run variance.** Two default-thread fast passes disagree on
  1,895 points, spread across every frame. Two single-thread fast passes agree on every point.
- **The mechanism is PyTorch's intra-op thread pool, inside the full graph.** The shim's reductions
  alone were already shown deterministic and loop-exact at 10 threads (Step 5 probe), so the variance
  comes from other multithreaded operations in the forward pass.
- **So `--fast-scatter`'s reproducibility has a precondition:** `torch.set_num_threads(1)`. Any
  speedup claim has to state the thread count. The 5.82× end-to-end ratio from Step 5 was measured at
  10 threads, where results are not reproducible run to run. At 20 frames the reproducible
  single-thread configuration costs about 1.9× the default-thread wall time (129 s vs 69 s).

### What this does NOT establish, and was not guessed at

- **Reproducible is not the same as equal to the loop.** Single-thread fast differs from the loop
  pass on 3,088 points. The loop pass ran at 10 threads, so this data cannot say whether the gap comes
  from the shim or from the thread count changing arithmetic elsewhere in the graph. Separating them
  needs a single-threaded loop pass (about 40 minutes for 20 frames). **Not run.**
- **The loop path's own per-point reproducibility at 10 threads is untested.** Step 5's two 200-frame
  loop runs agreed at the metric level, which is the only basis for calling that path reproducible.
  [!] Option 2 (Step 6, now running) uses that path at 10 threads, so its FRNet labels carry this
  caveat until it is tested.
- **Headline metrics stay insensitive:** 92.68% ± 0.005 pp across all five passes on this slice, and
  90.3% / 65.2% on 200 frames in every earlier run.

## Step 6: the plan-regret delta, ground-truth labels vs FRNet labels (the deliverable)

> **[!] Framing, restated:** the FRNet labels come from the **independently sourced public checkpoint**
> (https://drive.google.com/drive/folders/173ZIzO7HOSE2JQ7lz_Ikk4O85Mau68el, file id
> `1Ez-fpwu2WFCBw8usjwxUz6XQruw-cGU4`, downloaded 2026-09-14T07:27:54Z, SHA-256
> `09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`), **not** the 4 Sep file.
> This is offline evaluation: the mapping pipeline still takes ground-truth labels, and nothing in
> it was touched.

Run: `scripts/plan_regret_frnet_delta.py --seq 08 --frames 200 --motion gt`, **without
`--fast-scatter`**. That is Option 2, chosen because Step 5 found `--fast-scatter` not reproducible at
default threads (R-j). The gate before the run passed (2400/2400 MHz, commit 12.5/15.73 GB), state
after was OK (12.12/15.73 GB), and the checkpoint SHA-256 was re-verified first. Evidence:
`reports/bench/plan_regret_frnet_delta_loop.json`.

```
sequence 08, frames 0-199, schedule 5/10/20/40, motion=gt, second arm = FRNet
reference map (ground truth): ReferenceMap(3620x6066 @ 5 cm, 2,657,064 observed cells)   [7s]
M_gt built    [36s]  digest 2f0f5636f3033c9c
M_frnet built  [3942s]  digest 3b783d42de8d9f23
per-point accuracy of the second arm's labels vs ground truth: 90.3% over 22,741,893 labelled points
FRNet inference: 3903s total, 19.52s/frame (CPU)
common support: 100.0% of the planning window

  family            R_gt  R_second    delta  found gt/2nd  blocked gt/2nd
  longitudinal     1.160     2.157   +0.997         64/64             0/0
  lateral          1.142     1.591   +0.449         64/64             0/0
```

### The result

| family | R, ground-truth labels | R, FRNet labels | delta | relative |
|---|---|---|---|---|
| longitudinal | 1.160 (sd 0.94) | 2.157 (sd 1.28) | **+0.997** | +86% |
| lateral | 1.142 (sd 0.89) | 1.591 (sd 1.07) | **+0.449** | +39% |

**Swapping ground-truth labels for this FRNet checkpoint's predictions (90.3% point accuracy)
increases plan regret on real seq 08:** by about +0.997 on longitudinal queries and +0.449 on
lateral ones. Both query families found paths in all 64 queries in both arms, with none blocked, on
100% common support.

### Why the baseline is trusted

- **The ground-truth arm is bit-identical to the one validated by the oracle control:** digest
  `2f0f5636f3033c9c` in both. The oracle control put ground truth through the FRNet arm's own code
  path and got delta 0.000. So the delta comes from the labels and not from the plumbing, and the
  motion ids and ground masks are held equal by construction.
- **The FRNet labels score 90.3% point accuracy on the same 200 frames,** matching Step 4.

### How strong, and the limits

- **The uncertainty is rough.** The script records per-query standard deviations, not paired
  per-query regrets. An unpaired standard error of each mean (SD/sqrt(64)) gives delta/SE of about
  5.0 (longitudinal) and 2.6 (lateral). The queries share one map and are not fully
  independent, and a paired analysis would usually be tighter. So these are indicative, not a formal
  test.
- **One FRNet-arm run, one schedule (`5/10/20/40`), one 200-frame slice, motion held at ground truth.**
- **[!] R-j caveat:** this run used the loop path at the default 10 threads. That path's per-point
  reproducibility at 10 threads is untested; so far it rests only on metric-level agreement. How much
  the delta would move between runs is not measured.
- **Cost:** FRNet inference on CPU, loop path, 19.52 s per frame (3,903 s for 200 frames).

## R-j round 2: both open questions answered, R-j resolved

The same harness (`reports/harnesses/frnet_pred_diff.py`), frames 0–19 of seq 08, 2,471,164 points,
launched at 17:53:37. **Every pass was gated first, and every gate passed:** 2400/2400 MHz, commit
12.32–12.64 of 15.73 GB. Three new passes were added to the five from round 1:

| pass | `--fast-scatter` | intra-op threads | wall | accuracy (20 frames) |
|---|---|---|---|---|
| loop (round 1) | no | 10 | 405 s | 92.6849% |
| **loopB** | no | 10 | 395 s | 92.6862% |
| **loop1a** | no | **1** | 898 s | 92.6882% |
| **loop1b** | no | **1** | 914 s | 92.6882% |
| fast1a / fast1b (round 1) | yes | 1 | 129 s each | 92.6882% |
| fastA / fastB (round 1) | yes | 10 | 69 s each | 92.6835% / 92.6886% |

The pairs that answer the questions (full matrix in `reports/bench/rj_round2_pred_diff.json`):

```
rj_loop1a  vs rj_fast1a      0 points differ   (and loop1a=loop1b=fast1a=fast1b: all 0)
rj_loop    vs rj_loopB   1,446 points differ, in 20/20 frames
rj_fastA   vs rj_fastB   1,895 points differ, in 20/20 frames   (round 1)
```

### Answers

1. **Does single-thread `--fast-scatter` equal the single-thread loop? Yes, exactly: 0 differing
   points.** All four single-thread passes (two fast, two loop) are identical on every point. So the
   3,088-point gap in round 1 was the thread count, not the shim. **The shim is exact in the full
   model once the thread count is pinned.**
2. **Is the loop path reproducible per point at 10 threads? No: two passes differ in 1,446 points.**
   The loop path is as non-reproducible at default threads as `--fast-scatter` (1,895). **The run-to-run
   variance belongs to PyTorch's intra-op thread pool and affects both paths equally.** It was never
   introduced by the shim.

No reverse-engineering was needed: the diff separated the two effects cleanly. A module-level
divergence locator was prepared in case it did not, and it was deleted unused.

### What the standing `--fast-scatter` claim becomes

- **Correct:** with `torch.set_num_threads(1)`, `--fast-scatter` produces per-point predictions
  bit-identical to the port's own loops on real data, and both are reproducible run to run.
- **Correct:** at the default thread count, neither path is reproducible per point: roughly 0.05–0.08%
  of points change between runs. Headline metrics are unaffected at printed precision.
- **The reproducible, like-for-like speedup is about 7.0×:** at 1 thread, 20 frames, 129 s with
  `--fast-scatter` against 898–914 s without, with identical outputs. Step 5's 5.82× was measured at 10
  threads, so it compares two non-reproducible configurations; treat it as a wall-time ratio only.
- **Correction to an estimate made before this round:** the single-thread loop pass was estimated at
  about 8 minutes. It took about 15.

### Consequence for Step 6

Step 6's FRNet labels came from the loop path at 10 threads, so they carry this run-to-run noise, and
its effect on the regret delta was not measured. Step 6 is therefore re-run in the configuration now
proven exact and reproducible: `--fast-scatter` with 1 thread. It runs twice, so the deliverable's own
reproducibility is shown rather than assumed.

## Step 6, re-run reproducibly: `--fast-scatter --threads 1`, twice, and the runs are bit-identical

**Why:** R-j round 2 showed that at the default 10 threads neither path is reproducible per point, and
that with 1 thread `--fast-scatter` is bit-identical to the port's loops. The first Step 6 used the loop
path at 10 threads, so it was re-run twice in the proven configuration. Every run was gated and had its
checkpoint SHA-256 verified first; state after each run was OK (2400/2400 MHz, commit 12.35–12.62 of
15.73 GB). Evidence: `reports/bench/plan_regret_frnet_delta_t1_runA.json`, `..._runB.json`.

```
run A  M_gt digest 2f0f5636f3033c9c   M_frnet digest 41ad78eedfa97f06   acc 90.3% (0.9030358642528131)
       longitudinal  R_gt 1.160  R_frnet 2.230  delta +1.069191   64/64 found, 0 blocked
       lateral       R_gt 1.142  R_frnet 1.710  delta +0.567144   64/64 found, 0 blocked
run B  identical in every field: same M_frnet digest 41ad78eedfa97f06, same accuracy, same regrets
FRNet inference 6.15 / 6.09 s per frame (vs 19.52 s per frame on the 10-thread loop path)
```

### The reproducible result, which supersedes the 10-thread one

| family | R, ground-truth labels | R, FRNet labels | **delta (reproducible)** | 10-thread loop run | difference |
|---|---|---|---|---|---|
| longitudinal | 1.160 | 2.230 | **+1.069** | +0.997 | +0.073 |
| lateral | 1.142 | 1.710 | **+0.567** | +0.449 | +0.118 |

- **Two runs are bit-identical:** the FRNet map digest, label accuracy and every regret field match.
  The deliverable now reproduces exactly.
- **The ground-truth baseline is unchanged** (digest `2f0f5636f3033c9c`, the same as the oracle
  control).
- **The finding stands and is slightly larger:** FRNet labels raise plan regret by +1.069
  longitudinal (+92%) and +0.567 lateral
  (+50%).

### What the 10-thread vs 1-thread comparison teaches

The two configurations' labels differ in only about 0.06% of points: accuracy 0.9030359
against 0.9030913. **Yet the lateral delta moved by +0.118 (about
26%) and the longitudinal by +0.073
(about 7%).** Plan regret is sensitive to small label
changes. The lateral figure in particular is plainly positive, but its exact value should be read with that
sensitivity in mind. The longitudinal effect is robust across both configurations.

### Limits that still apply

One schedule (`5/10/20/40`), one 200-frame slice of seq 08, motion held at ground truth, and an
independently sourced checkpoint (not the 4 Sep file). Reproducibility now holds only for a pinned thread
count of 1.

## Step 6, paired per query: the uncertainty of the delta, measured properly (2026-09-16)

**Why.** The Step 6 delta was reported with a rough **unpaired** standard error (delta/SE about 5.0
longitudinal and 2.6 lateral), and the lateral value was flagged as sensitive. Both arms answer the
**same 64 planning queries**, so the correct uncertainty is over the per-query **differences**.

**How.** `scripts/plan_regret_frnet_delta.py --fast-scatter --threads 1 --per-query`, seq 08, 200
frames. `per_query_regrets()` is transcribed from `eval_synthetic.plan_regret_for` (same costmaps,
same common-support restriction, same `plan_queries`, same `regret()` calls in the same order). The
script **asserts** each arm's per-query mean equals `plan_regret_for`'s reported figure exactly, so
the transcription cannot drift silently. `paired_stats()` gives the mean, SD and SE of the
differences, a seeded 10,000-sample bootstrap 95% CI that resamples queries with each pair kept
together, worse/equal/better counts, and an exact two-sided sign test. It is pinned by
`tests/test_plan_regret_paired.py` (5 tests).

**Checks before trusting it:**
- **Oracle control, 40 frames:** the maps were bit-identical, the assertion held, and every paired
  statistic was **exactly zero** in both families (0 worse, 0 better, sign-test p = 1).
- **Real run:** FRNet map digest `41ad78eedfa97f06` and every aggregate field (R_gt, R_second, delta,
  SDs, found/blocked counts, label accuracy, common support) are **identical** to the earlier
  reproducible run `plan_regret_frnet_delta_t1_runA.json`.

Evidence: `reports/bench/plan_regret_frnet_delta_t1_paired.json` (includes all 64 per-query regrets
per arm).

| family | paired mean diff | SD | SE | t | 95% bootstrap CI | worse / equal / better | sign test |
|---|---|---|---|---|---|---|---|
| **longitudinal** | **+1.069** | 1.447 | 0.181 | 5.91 | **[+0.730, +1.436]** | 49 / 1 / 14 | p = 1.1e-05 |
| **lateral** | **+0.567** | 1.137 | 0.142 | 3.99 | **[+0.301, +0.846]** | 25 / 31 / 8 | p = 0.0046 |

### What this establishes

- **Both deltas are clearly above zero:** neither 95% CI comes close to it, and both sign tests are
  small. The paired SEs are tighter than the rough unpaired ones (0.199 → 0.181
  longitudinal, 0.173 → 0.142 lateral).
- **The longitudinal effect is broad:** FRNet labels make 49 of 64 queries worse and
  14 better.
- **The lateral effect is concentrated:** 31 of 64 lateral queries are unaffected; among the
  33 that change, 25 get worse and 8 better.
- **The earlier "lateral is sensitive" caveat is now bounded.** The non-reproducible 10-thread value
  (+0.449) lies inside the lateral CI, so run-to-run label noise at 10 threads moves the delta within
  its measured uncertainty, not outside it.

### What it does not establish

The CI is over **these 64 planning queries on one map**: one schedule (`5/10/20/40`), one 200-frame
slice of seq 08, and motion held at ground truth. The queries share that map, so this is uncertainty
from the choice of queries, not across scenes, sequences or schedules. A different slice could give a
different delta.
