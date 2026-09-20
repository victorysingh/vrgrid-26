# Addendum: measuring the DL pipeline's real-time figure on the T4

> **[SUPERSEDED 2026-09-20 — AWS was never used. The GPU work happened on Kaggle.]**
>
> This protocol was written for an AWS `g4dn.xlarge` that **never ran**. `docs/research-log.md`
> records that AWS was *abandoned, not deferred*: every GPU quota on the account read 0, the
> increase request was refused because the free plan has no GPU tier, and **lifetime AWS spend was
> $0.00**. The T4 pass ran instead on **Kaggle's free T4**, and `docs/gpu-lane/t4/host.log` records
> the actual hardware — **2× Tesla T4, 15,360 MiB, driver 580.159.04, CUDA 13.0**.
>
> Kept, not deleted, because the measurement protocol below (what to run, in what order, what to
> report) is independent of who provisions the card, and because deleting a plan hides that it was
> made. **Nothing here describes a measurement that was taken.** For anything that actually ran, see
> `docs/gpu-lane/` and `scripts/kaggle/`.

**[!] This is NOT a standalone runbook, and stopped being one on 2026-09-18.**
**The runbook is `docs/gpu-lane/02-AWS-RUNBOOK.md`** (Shrestha's, upstream), together with
`scripts/aws/t4.sh` and `scripts/aws/auto.sh`. It is more complete than this file was on every
shared topic and it is the one that is maintained: region and cost guardrails, AMI, storage,
security group, key pair, the launch checklist, first connect, S3 dataset staging, the environment,
`tmux` working discipline, failure modes, and what "done" looks like. **Follow that file for
everything up to and including a working instance with the repo and data on it.**

What survives here is only the part upstream's runbook does not cover: **the DL-mode measurement
protocol** — what to run, in what order, and what to report — because that answers a question
(SIH26053's "evidence of low latency (high FPS)" for a *deep-learning* pipeline) that the GPU lane
was not built to answer. Sections are numbered from 4 for that reason; 0–3 were launch and transfer
instructions that duplicated upstream's, and they are gone.

**Why the DL figure is needed at all.** On the laptop FRNet runs on CPU at 3–6 s per frame, so
`--semantics frnet` cannot be real time there. JP decided (2026-09-14) that the real-time figure
comes from an AWS GPU run, and that **no CUDA build of PyTorch is installed on the laptop.**

**What the run produces, all from the same instance:**

1. **GPU reproducibility:** do two CUDA runs of `--fast-scatter` agree point for point? Its own
   question, separate from R-j's CPU answer.
2. **The network alone:** FRNet inference latency per frame on the T4.
3. **The whole DL pipeline end to end:** `timing_table.py --seq 08 --semantics frnet`, with FRNet
   inside the `semantics` stage and the map back end on the instance's CPU. This is the number that
   answers "real time". Quote it with the instance type, never next to the laptop's
   ground-truth-label figure as if the two were one machine.

---

## Four deltas from upstream's runbook — read these before following it

1. **The dataset slice is ~494 MB, not 84.8 GB.** Upstream's §4 stages the full SemanticKITTI
   through S3 because the GPU lane maps whole sequences. This measurement needs **seq 08 frames
   0–200 only, plus `poses` and `calib`**. Note **201 frames, not 200**: `timing_table.py` reads
   `--frames + 1`, the first frame being start-up. At that size S3 staging is optional.
2. **The code may have to travel as a git bundle.** `jp/p99-alloc-fixes` has **never been pushed**,
   so a `git clone` gets neither the DL mode nor these harnesses. A bundle keeps it local and needs
   no authorisation; pushing the branch is outward-facing and needs JP's explicit go-ahead. Upstream
   assumes the repo is already reachable.
3. **On Windows, `tar` needs `--force-local`.** Without it GNU tar reads `C:/...` as a remote host
   and fails with "Cannot connect to C: resolve failed". Round-trip verified on the laptop.
4. **Upstream's "Bring nothing back" (their §4) is about the dataset, not the results.** Section 8
   below copies back a few small JSON files, which is the point of the run. Do that before
   terminating.

## [!] Read first

- **No durations are documented for any step.** Nothing has been run on a T4 yet. Every command
  below is prefixed with `time`, so this run produces the estimates for the next one. Keep the
  instance on only while a step is running or results are being copied.
- **Terminate, don't stop, when finished,** and only after the results are safely on the laptop
  (section 8). A stopped instance keeps billing for its disk.
- **GPU reproducibility is not assumed.** R-j showed that on **CPU**, with one thread,
  `--fast-scatter` matches the loop exactly and repeats exactly. On CUDA the mechanism differs
  (atomic-add ordering inside kernels, cuDNN algorithm choice), so it is measured here, first
  (step e/f).
- **Real data only.** If a file is missing on the instance, stop and copy it from the laptop. Never
  substitute or synthesise anything.

## 4. On the instance: code, environment, data, checkpoint

```sh
git clone -b jp/p99-alloc-fixes ~/vrgrid.bundle ~/vrgrid && cd ~/vrgrid
git rev-parse --short HEAD               # must equal the commit recorded on the laptop

python3 -m venv ~/venv && source ~/venv/bin/activate
python -m pip install --upgrade pip
# PyTorch with CUDA: choose the wheel index whose CUDA version is <= the "CUDA Version" nvidia-smi showed
# (see pytorch.org "Get Started"); e.g. cu121 or cu124
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[perception]"           # numpy, pyyaml, pypatchworkpp==1.4.1 (cupy is NOT needed)

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#   must print ... True Tesla T4
python -c "import pypatchworkpp; print('Patchwork++ OK')"
#   [!] if this fails, STOP. Without it the pipeline falls back to label-based ground, and the
#   timing is not comparable. Whether a pypatchworkpp 1.4.1 wheel exists for this Python on Linux
#   has NOT been verified on the laptop.

mkdir -p checkpoints data && mv ~/frnet-semantickitti_seg.pth checkpoints/
sha256sum checkpoints/frnet-semantickitti_seg.pth
#   must print 09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e -- else STOP
tar -xf ~/seq08_0-200.tar -C data
ls data/sequences/08/velodyne | wc -l    # must print 201
export VRGRID_DATA_ROOT=$PWD/data PYTHONUTF8=1
```

## 5. Preflight: the DL mode end to end on 2 real frames (same check as the laptop audit)

```sh
time python -m vrgrid.run --seq 08 --frames 2 --semantics frnet --fast-scatter
```

It must exit 0 and print **`[FRNet] Loaded 413 / 413 parameters`**, **`fast-scatter ENABLED`**, the DL-mode
motion disclosure, and **`ground: Patchwork++ (geometric segmenter)`**. If it prints
`SEMANTIC-CLASS FALLBACK`, stop and fix Patchwork++ first.

## 6. Record machine state around every run (Linux)

`reports/harnesses/machine_state.py` is Windows-only. Define this once per session:

```sh
state() {
  echo "== STATE $1 $(date -u +%FT%TZ)"
  grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
  cat /proc/loadavg
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  nvidia-smi -q -d PERFORMANCE | grep -iE 'Idle|Slowdown|Throttle|Power Cap|Thermal' | grep -iv 'not active'
}
```

**Treat a run as untrusted** if, before or after it:
- **Memory:** `MemAvailable` is under 2 GB, or swap is in use (`SwapFree` below `SwapTotal`).
- **GPU throttling:** any slowdown, throttle, power-cap or thermal reason is **Active**. `Idle: Active`
  alone is fine.
- **Sharing:** another process is using the GPU before the run starts.

(The first version's rule, `Committed_AS > MemTotal`, is wrong on Linux: overcommit there is normal and
does not mean paging.)

## 7. The runs, in this order

```sh
H=reports/harnesses/frnet_pred_diff.py

# (e) GPU reproducibility of the shim alone: twice, separate processes -- compare sha256[0] lines
state before-e1; time python reports/harnesses/shim_determinism_probe.py --device cuda | tee reports/bench/shim_probe_t4_run1.txt; state after-e1
state before-e2; time python reports/harnesses/shim_determinism_probe.py --device cuda | tee reports/bench/shim_probe_t4_run2.txt; state after-e2
time python scripts/frnet_fast_scatter.py | tee reports/bench/shim_verify_t4_run1.txt
time python scripts/frnet_fast_scatter.py | tee reports/bench/shim_verify_t4_run2.txt

# (f) GPU reproducibility of the whole model: --fast-scatter on CUDA twice, diff point for point;
#     then the cross-check against a CPU single-thread pass on the same 20 frames
state before-f1; time python $H run --device cuda --fast-scatter --frames 20 --out gpu_fastA.npz; state after-f1
state before-f2; time python $H run --device cuda --fast-scatter --frames 20 --out gpu_fastB.npz; state after-f2
python $H diff gpu_fastA.npz gpu_fastB.npz --json reports/bench/gpu_fast_scatter_pred_diff_t4.json
time python $H run --device cpu --fast-scatter --threads 1 --frames 20 --out cpu_fast1_t4host.npz
python $H diff gpu_fastA.npz cpu_fast1_t4host.npz --json reports/bench/gpu_vs_cpu1_pred_diff_t4.json

# (a) network alone, with the shim (the configuration that can be real time)
state before-a; time python reports/harnesses/frnet_gpu_latency.py --fast-scatter --frames 200 --warmup 10 --json reports/bench/frnet_gpu_latency_fast_t4.json; state after-a

# (b) network alone, port loops, fewer frames
state before-b; time python reports/harnesses/frnet_gpu_latency.py --frames 40 --warmup 5 --json reports/bench/frnet_gpu_latency_loop_t4.json; state after-b

# (c) THE END-TO-END DL PIPELINE, 3 fresh processes
for r in 1 2 3; do
  state before-c$r
  time python scripts/timing_table.py --seq 08 --frames 200 --semantics frnet --fast-scatter \
    --frame-times reports/bench/dl_pipeline_t4_rep$r.frames.json | tee reports/bench/dl_pipeline_t4_rep$r.txt
  state after-c$r
done

# (d) same instance, ground-truth labels: the like-for-like baseline for the non-network cost
for r in 1 2 3; do
  state before-d$r
  time python scripts/timing_table.py --seq 08 --frames 200 \
    --frame-times reports/bench/gt_pipeline_t4_rep$r.frames.json | tee reports/bench/gt_pipeline_t4_rep$r.txt
  state after-d$r
done
```

**Reading (e)/(f), and what not to do:**

- **`0 points differ` between `gpu_fastA` and `gpu_fastB`:** repeated GPU runs agree point for point at
  the settings recorded in the metadata (device, GPU, CUDA version, cuDNN flags). Say exactly that, for
  that configuration only.
- **Any points differ:** the GPU path is not reproducible at default settings. **Report the count as the
  finding** (points, frames affected, maximum in one frame), and state that (a), (c) and their accuracy
  inherit it. Do **not** quietly switch on `torch.use_deterministic_algorithms(True)` or cuDNN
  determinism and report only the clean run. A deterministic re-run, if wanted, is a separate, labelled
  pair, reported alongside.
- **GPU vs CPU single-thread** (`gpu_vs_cpu1_pred_diff_t4.json`) is expected to be non-zero: CUDA
  `scatter_mean` differs from the loop by up to 2 float32 ulp. It measures how far GPU labels sit from
  the CPU-reproducible ones.

## 8. Get the results back BEFORE terminating

```sh
# on the instance
cd ~/vrgrid && tar -czf ~/t4-results.tgz reports/bench/*t4* gpu_fastA.npz gpu_fastB.npz cpu_fast1_t4host.npz ~/aws-run.log
sha256sum ~/t4-results.tgz

# on the laptop (Git Bash, from vrgrid/)
scp -i <key.pem> ubuntu@<host>:~/t4-results.tgz "$HOME/"
sha256sum "$HOME/t4-results.tgz"        # must match the instance's value before terminating
tar -tzf "$HOME/t4-results.tgz"         # list it: every step's JSON/text and aws-run.log must be there
```

Only terminate once the laptop copy's hash matches and the listing is complete.

## 9. Terminate, and confirm nothing keeps billing

In the EC2 console (the laptop has no AWS CLI):

1. **Instances:** select it → *Instance state* → **Terminate instance** (not *Stop*). Wait for
   `terminated`.
2. **Volumes** (Elastic Block Store): confirm **no volume** from this instance shows state `available`.
   One would still bill. Delete any that do.
3. **Elastic IPs:** confirm **none** is allocated. If one is, release it.
4. **Snapshots / AMIs:** confirm you created none during the session. If you did, delete them.
5. **Optional:** delete the security group and key pair you made for this run. They cost nothing, but it
   keeps the account clean.

## 10. What to report

- **The GPU reproducibility result from (e)/(f)**, stated next to every GPU figure: whether two runs
  agreed point for point, under which settings, and the GPU-vs-CPU distance.
- **Pooled p99 over all frames of the three (c) runs** (the project's gate criterion): pooled p50 and
  p99, per-run p99s, and the count of frames over 100 ms.
- **The `semantics` row from (c)** (FRNet inside the pipeline) next to (a)'s network-only p50/p99. They
  should agree to within the host↔GPU copy overhead.
- **(c) against (d), both on the instance:** the cost the network adds to the frame.
- **FPS** = 1000 / p50 ms and 1000 / p99 ms, with the instance type.
- **Point accuracy printed by (a),** as a sanity check that the GPU path still labels correctly.
- **Motion stays ground truth** (`moving-*` labels) in the DL mode. Say so next to the figure.
- **Each step's `time` output,** so the next run has real durations.
- Commit the results on the laptop, with instance type, AMI name, driver, CUDA and torch versions, and
  the recorded state blocks.

## 11. What was verified on the laptop, and what was not

**Verified (2026-09-16, laptop, without AWS):**
- **Bundle route works.** `git bundle create` of `jp/p99-alloc-fixes`, then `git bundle verify`, then
  `git clone -b jp/p99-alloc-fixes <bundle>` gives exactly the laptop HEAD, with this runbook and
  `open_frnet` present (1.86 MB bundle).
- **The data list is complete.** All 404 files (201 velodyne + 201 labels + poses + calib) exist, and
  `seq -f "%06g"` works in Git Bash.
- **The tar command works.** `tar --force-local` archived a 2-frame subset and extracted it
  byte-identical. Without `--force-local` it fails ("Cannot connect to C: resolve failed").
- **Dependencies are what the install step says.** Third-party imports on the run path: `numpy`,
  `yaml`, `torch`, `pypatchworkpp`. `cupy` is imported only for `device != "cpu"` allocations, which the
  pipeline does not use.
- **The DL mode runs end to end on CPU.** The 2-frame smoke test: exit 0, 413/413 parameters, Patchwork++
  ground.
- **Both harnesses refuse cleanly without CUDA.** `--device cuda` on the laptop exits with a clear
  message; the CPU path is unchanged (0 of 2,471,164 points differ).

**Not verified (needs the instance):**
- `scp`.
- The AMI's driver and Python versions.
- A `pypatchworkpp==1.4.1` Linux wheel.
- The CUDA torch install.
- Anything running on CUDA.
- The `nvidia-smi -q` output wording on this driver.
- Every duration.
