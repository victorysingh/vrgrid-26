# Runbook: the DL pipeline's real-time figure on AWS (g4dn.xlarge, NVIDIA T4)

**Why this exists.** SIH26053 asks for "evidence of low latency (high FPS)" from a deep-learning
pipeline. On the local laptop, FRNet runs on CPU at 3–6 s per frame, so the DL mode
(`--semantics frnet`) cannot be real time there. JP decided (2026-09-14) that the real-time
figure comes from an AWS GPU run, and that **no CUDA build of PyTorch is installed on the laptop.**

**Two figures, both from the same instance:**

1. **The network alone:** FRNet inference latency per frame on the T4
   (`reports/harnesses/frnet_gpu_latency.py`).
2. **The whole DL pipeline end to end:** `scripts/timing_table.py --seq 08 --semantics frnet`,
   with FRNet inside the `semantics` stage and the map back end on the instance's CPU. This is the
   number that answers "real time". Quote it with the instance type, never next to the laptop's
   ground-truth-label figure as if the two were one machine.

---

## [!] Before you start: two things that do not carry over from the laptop

- **`reports/harnesses/machine_state.py` is Windows-only.** It reads counters through PowerShell and
  will not run on a Linux instance. Record the state by hand before and after **each** run (commands
  in step 4), the same discipline as every laptop number (D8).
- **Reproducibility on GPU is a different claim.** On CUDA, `--fast-scatter`'s `scatter_mean` differs
  from the loop by up to 2 float32 ulp, and cuDNN may pick non-deterministic kernels. The CPU result
  "bit-identical at 1 thread" (R-j) does **not** transfer. Report GPU accuracy next to GPU latency
  (the harness does), and do not claim bit-identity for GPU runs.

## 1. Instance and environment

- g4dn.xlarge (1× T4, 4 vCPU, 16 GB), an NVIDIA driver installed (e.g. a Deep Learning AMI).
- Clone the repo at the same commit as the laptop run: `git checkout <commit>`, then
  `git rev-parse --short HEAD`, and record it.
- A Python matching `pyproject.toml`, then `pip install -e .` plus a **CUDA** build of `torch` (on
  the instance only), plus the project's other requirements (e.g. `pypatchworkpp`, which ground
  segmentation needs).
- Check: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`,
  which must print `True` and `Tesla T4`.

## 2. Checkpoint (verify, don't trust)

- Copy `checkpoints/frnet-semantickitti_seg.pth` to the instance.
- `sha256sum checkpoints/frnet-semantickitti_seg.pth` must equal
  `09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`. The harness also refuses a
  mismatch.

## 3. Data (real only)

- Copy what the loader actually reads (`src/perception/loader.py`), keeping the directory layout:
  `poses/08.txt` (the **official** KITTI ground-truth poses, not `sequences/08/poses.txt`),
  `sequences/08/calib.txt` (the `Tr` extrinsic), and `sequences/08/velodyne/*.bin` plus
  `sequences/08/labels/*.label` for at least the frames you time (200 frames: roughly 400 MB).
  `times.txt` is not read by the pipeline; copying it is harmless but optional.
  Then `export VRGRID_DATA_ROOT=<dir holding poses/ and sequences/>`.
- **If a file is missing, stop and fetch it from the real source. Never substitute or synthesise
  anything.**

## 4. Record machine state around every run (Linux replacement for machine_state.py)

```sh
state() { echo "== $1 $(date -u +%FT%TZ)"; nproc; grep -E 'MemTotal|MemAvailable|Committed_AS' /proc/meminfo;
          nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks_throttle_reasons.active --format=csv,noheader;
          cat /proc/loadavg; }
```

Treat a run as untrusted if `clocks_throttle_reasons.active` is not `0x0000000000000000` (GPU
throttling), if `Committed_AS` exceeds `MemTotal`, or if anything else is using the GPU.

## 5. The runs

```sh
export PYTHONUTF8=1

# (a) network alone -- with the shim (this is the configuration that can be real time)
state before-a
python reports/harnesses/frnet_gpu_latency.py --fast-scatter --frames 200 --warmup 10 \
  --json reports/bench/frnet_gpu_latency_fast_t4.json
state after-a

# (b) network alone -- port loops, fewer frames (the loops are slow even on GPU)
state before-b
python reports/harnesses/frnet_gpu_latency.py --frames 40 --warmup 5 \
  --json reports/bench/frnet_gpu_latency_loop_t4.json
state after-b

# (c) THE END-TO-END DL PIPELINE on the same instance, 3 fresh processes
for r in 1 2 3; do
  state before-c$r
  python scripts/timing_table.py --seq 08 --frames 200 --semantics frnet --fast-scatter \
    --frame-times reports/bench/dl_pipeline_t4_rep$r.frames.json | tee reports/bench/dl_pipeline_t4_rep$r.txt
  state after-c$r
done

# (d) same instance, ground-truth labels, for a like-for-like comparison of the non-network cost
for r in 1 2 3; do
  state before-d$r
  python scripts/timing_table.py --seq 08 --frames 200 \
    --frame-times reports/bench/gt_pipeline_t4_rep$r.frames.json | tee reports/bench/gt_pipeline_t4_rep$r.txt
  state after-d$r
done
```

## 6. What to report

- **Pooled p99 over all frames of the three (c) runs.** That is the project's gate criterion; the
  `--frame-times` files make it computable. Report the pooled p50 and p99, the per-run p99s, and
  the count of frames over 100 ms.
- **The `semantics` stage row from (c)**, which is FRNet inside the pipeline, next to (a)'s
  network-only p50/p99. They should agree to within the host-to-GPU and GPU-to-host copy overhead.
- **(c) against (d), both on the instance:** the cost the network adds to the frame.
- **FPS** = 1000 / p50 ms and 1000 / p99 ms, stated with the instance type.
- **Point accuracy printed by (a)**, as a sanity check that the GPU path still labels correctly.
- **Motion stays ground truth** (`moving-*` labels) in the DL mode. Say so next to the figure.
- Commit the JSON and text outputs with the instance type, driver, CUDA and torch versions, and the
  recorded state blocks.
