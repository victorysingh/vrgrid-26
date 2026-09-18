# Morning summary 5 — fitting SIH26053: DL pipeline mode, real-time plan, accuracy by distance

Covers the pass that started from JP asking whether the project meets the SIH26053 problem statement
("Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception"). The earlier
D11 / R-j work is in `MORNING-SUMMARY-4.md`. Local branch `jp/p99-alloc-fixes`; **nothing pushed, no
PR.**

---

## Still open, first

1. **No real-time figure for the DL pipeline yet.** FRNet runs at 3–6 s per frame on the laptop CPU. JP
   decided the figure comes from an AWS g4dn.xlarge run, which **JP has to start**; the runbook and
   harness are ready (`reports/aws-dl-realtime-addendum.md`). Until then, the ~10 Hz figure is for the
   ground-truth-label pipeline only.
2. **Moving objects still come from ground-truth labels in the DL mode.** FRNet has no motion output.
   Decided as "keep and disclose"; it must be stated wherever the DL mode is reported.
3. **No classification accuracy can be measured beyond 50 m on SemanticKITTI:** seq 08 is not labelled
   past 50.0 m, while the grid extends to ±100 m.
4. **The project's written framing is not updated.** README and the report still describe the
   pipeline as ground-truth-label only, and don't mention the opt-in DL mode or the distance results.
   That is a docs pass for JP and the team, not done here.
5. **Unchanged from before:** ring-boundary alignment (D2 / r3, the statement's "without alignment
   errors or data loss"), Shrestha's review of the engine changes, the D8 reference machine, D1.

## Checked against the problem statement

| SIH26053 asks for | Status now |
|---|---|
| A deep-learning pipeline from raw lidar to the grid | **Opt-in DL mode added** (`--semantics frnet`); ground truth stays the default for benchmarks |
| Model segments terrain / static / moving | Terrain and static classes from FRNet; **motion still ground truth** (disclosed) |
| Variable-resolution grid, 5 cm → 50 cm up to 100 m | Already met: schedule `5/10/50` is exactly that |
| Memory reduction vs uniform high-res 3D | Already met: 8.94 MB, 286× less than a dense 5 cm 3D voxel grid |
| Real-time dashboard with colour coding | Already met; the dashboard now also accepts `--semantics frnet` |
| Low latency / high FPS | Ground-truth-label pipeline ~10 Hz (pooled p99 89.58 ms, laptop) — **[restated 2026-09-18] on the pre-port CPU path and the pre-`df35fd5` grid; see the R-b row in `OPEN-ITEMS.md`. Upstream's GPU pipeline reports 22 ms/frame, which is a different execution model on different hardware, not a replacement figure.** **DL pipeline pending the AWS run** |
| Accuracy in object classification across distances | **Now measured** to 50 m (below); beyond 50 m has no ground truth |

## What was done

### 1. Opt-in deep-learning mode (`a26f7f0`)

- `iter_pipeline(..., semantics_source="gt" | "frnet", frnet=None)`. The ground-truth default is the
  existing path, unchanged. `"frnet"` takes each frame's labels from `semantics.FRNetInference` on the
  raw scan.
- Every frame records `semantic_source`. `open_frnet(fast_scatter, threads)` is the one construction
  used by all three entry points.
- `--semantics / --fast-scatter / --threads` on `python -m vrgrid.run`, `python -m vrgrid.dash` and
  `scripts/timing_table.py --seq`, with loud refusals for misuse.
- **Verified:** the pipeline's FRNet labels are **identical** to `frnet_eval.py`'s model on a real frame
  (seq 08 frame 0, 1 thread). Full suite 728 passed, 3 skipped; the only failure is the known D1
  determinism test.
- **Not touched:** `semantics.py`, `engine.py`, `ground.py`, the frozen port.

### 2. AWS real-time runbook and GPU latency harness (`5925f6b`)

- `reports/harnesses/frnet_gpu_latency.py`: FRNet per-frame latency on CUDA (p50/p99/FPS), checkpoint
  SHA-256 enforced, `torch.cuda.synchronize`, warm-up excluded, and it refuses to run without CUDA.
  Smoke-tested on CPU (not a result).
- `reports/aws-dl-realtime-addendum.md`: instance setup, the data the loader really needs (checked), a
  Linux replacement for the Windows-only `machine_state.py`, network-only and **end-to-end DL
  pipeline** runs on the same instance, and a pooled-p99 report. It says GPU results are not
  bit-identical.

### 3. Classification accuracy by distance (this commit)

`scripts/frnet_eval_by_range.py --fast-scatter --threads 1`, seq 08, 200 frames, clean machine. All
bands together reproduce the known 90.3036% / 65.18%. Full report:
`reports/frnet-accuracy-by-distance.md`.

| band | point accuracy | drivable IoU | static obstacle IoU | movable object IoU |
|---|---|---|---|---|
| 0–10 m | **93.20%** | 94.1% | 67.2% | 98.9% |
| 10–25 m | **88.10%** | 86.1% | 82.3% | 92.9% |
| 25–50 m | **85.36%** | 78.1% | 88.9% | 82.0% |
| >50 m | no ground truth | — | — | — |

- **Found and fixed before the real run:** an IoU-tally indexing bug that would have corrupted per-band
  IoUs. The unit test caught it.
- **Found by checking, not assumed:** zero labelled points beyond 50 m is a property of the dataset.
  All 60,942 far points in 11 sampled frames are `unlabeled`, and the farthest labelled point is at
  50.0 m.

## Decisions JP made in this pass

- DL mode as **opt-in**, with ground truth as the default.
- Motion: **keep ground truth, disclose.**
- Real time: **AWS GPU run**, and no local CUDA install.

Recorded in memory so a later session doesn't widen or reverse them.
