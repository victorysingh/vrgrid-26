# Runbook: the DL pipeline's real-time figure on AWS (g4dn.xlarge, NVIDIA T4)

**Why this exists.** SIH26053 asks for "evidence of low latency (high FPS)" from a deep-learning
pipeline. On the laptop, FRNet runs on CPU at 3–6 s per frame, so the DL mode (`--semantics frnet`)
cannot be real time there. JP decided (2026-09-14) that the real-time figure comes from an AWS GPU
run, and that **no CUDA build of PyTorch is installed on the laptop.**

**What the run produces, all from the same instance:**

1. **GPU reproducibility:** do two CUDA runs of `--fast-scatter` agree point for point? This is its own
   question, separate from R-j's CPU answer.
2. **The network alone:** FRNet inference latency per frame on the T4.
3. **The whole DL pipeline end to end:** `timing_table.py --seq 08 --semantics frnet`, with FRNet inside
   the `semantics` stage and the map back end on the instance's CPU. This is the number that answers
   "real time". Quote it with the instance type, never next to the laptop's ground-truth-label figure
   as if the two were one machine.

**Revised 2026-09-16 after a pre-launch audit.** The first version could not have worked as written:
it told you to clone a branch that has never been pushed, listed one frame too few, left PyTorch and
Patchwork++ out of the install, used a Windows-only memory rule on Linux, and never said how to get
results back or terminate the instance.

---

## [!] Read first

- **No durations are documented for any step.** Nothing has been run on a T4 yet. Every command below
  is prefixed with `time`, so this run produces the estimates for the next one. Keep the instance on
  only while a step is running or results are being copied.
- **Terminate, don't stop, when finished,** and only after the results are safely on the laptop
  (section 8). A stopped instance keeps billing for its disk.
- **GPU reproducibility is not assumed.** R-j showed that on **CPU**, with one thread, `--fast-scatter`
  matches the loop exactly and repeats exactly. On CUDA the mechanism differs (atomic-add ordering
  inside kernels, cuDNN algorithm choice), so it is measured here, first (step e/f).
- **Real data only.** If a file is missing on the instance, stop and copy it from the laptop. Never
  substitute or synthesise anything.

## 0. Decide before launching: how the code gets there

`jp/p99-alloc-fixes` has **never been pushed**, so `git clone` from GitHub will **not** have the DL mode,
the harnesses or this runbook.

- **Default: a git bundle** (keeps everything local; tested on the laptop, see section 9). No
  authorisation needed.
- **Alternative: push the branch first.** That is an outward-facing action and needs JP's explicit
  go-ahead; it is not assumed here.

## 1. Launch the instance (EC2 console)

- **Instance type:** `g4dn.xlarge` (1× NVIDIA T4, 4 vCPU, 16 GB RAM).
- **AMI:** an **Ubuntu x86_64 AMI with the NVIDIA driver preinstalled**, e.g. AWS's *Deep Learning Base
  OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*. Exact names change between releases; the requirement is
  that `nvidia-smi` works and shows a `Tesla T4`. You install PyTorch yourself in a clean venv (step 4),
  so a PyTorch-specific AMI is not required. The default SSH user on Ubuntu AMIs is `ubuntu`.
- **Storage:** the default root volume is enough (the data is ~0.5 GB and PyTorch a few GB). On the
  Storage step, confirm **Delete on termination = Yes** for the root volume, and **add no extra volumes**.
- **Network:** a security group allowing SSH (port 22) **from your IP only**. **Do not** allocate an
  Elastic IP; the auto-assigned public IP is enough.
- **Key pair:** create or choose one and keep the `.pem` file. Below it is `<key.pem>`, and the instance's
  public DNS or IP is `<host>`.

## 2. Prepare on the laptop (Git Bash, from `vrgrid/`)

These commands were **tested on the laptop** (section 9), apart from the final `scp`, which needs the
instance.

```sh
# the code, as a bundle of the branch (about 1.9 MB)
git bundle create ~/vrgrid.bundle jp/p99-alloc-fixes
git rev-parse --short HEAD            # record it; the instance must show the same

# the checkpoint, and its hash to compare on arrival
sha256sum checkpoints/frnet-semantickitti_seg.pth
#   must print 09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e

# the data: frames 0-200 (201 frames), plus poses and calib, ~494 MB.
# 201, not 200: timing_table.py reads --frames + 1 (the first frame is start-up).
# --force-local is REQUIRED on Windows: without it GNU tar reads "C:/..." as a remote host.
cd /c/KITTI/dataset
tar --force-local -cf "$HOME/seq08_0-200.tar" poses/08.txt sequences/08/calib.txt \
  $(for i in $(seq -f "%06g" 0 200); do echo sequences/08/velodyne/$i.bin sequences/08/labels/$i.label; done)
cd -

# copy all three to the instance (NOT tested -- needs the instance)
scp -i <key.pem> "$HOME/vrgrid.bundle" "$HOME/seq08_0-200.tar" checkpoints/frnet-semantickitti_seg.pth ubuntu@<host>:~/
```

## 3. On the instance: record the session and check the GPU

```sh
ssh -i <key.pem> ubuntu@<host>
exec > >(tee -a ~/aws-run.log) 2>&1     # everything below is logged
nvidia-smi                               # must show Tesla T4; note the "CUDA Version" it reports
python3 --version                        # must be >= 3.10
```

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
