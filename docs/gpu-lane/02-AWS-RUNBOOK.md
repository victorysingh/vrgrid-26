# AWS runbook — the GPU instance

*Shrestha's Day 1. Everything here is checked against current AWS pricing
(Sep 2026), but **verify in the console before launching** — prices and AMI IDs
move.*

---

> **Scripted, 2026-09-17:** `scripts/aws/t4.sh` does every step below in order:
> `preflight` (spends nothing), `budget`, `stage`, `launch`, `setup`, `run`,
> `fetch`, then `stop`. It stages the FULL local dataset (84.8 GB, 250 GB disk), not only seq 08. It needs the AWS CLI (`~/.local/bin/aws`) configured with
> your own credentials. There is no terminate command, on purpose.

## 1. What we are provisioning, and why

**`g4dn.xlarge`** — <cite index="9-1">4 vCPUs, 16 GiB RAM, one NVIDIA T4 with 16 GiB of GPU memory</cite>. <cite index="5-1">On-demand pricing starts at $0.526/hr in us-east-1</cite>.

It is the right instance for this lane for three reasons:

- **The T4 is Turing (SM 7.5).** Same architecture family as most Jetson-class
  deployment targets, so what we learn about occupancy and memory behaviour
  transfers to the platform we claim in the deck.
- **16 GB VRAM is enough for the contention question.** Day 6 asks whether the
  grid engine and FRNet fit on one card together. Our arrays are ~29 MB; FRNet
  is ~10 M parameters at batch 1. If it does not fit in 16 GB, that is a real
  finding, not an artefact of a small card.
- **It is cheap enough to leave running through a work session** without anyone
  watching the bill.

**Do not use anything larger.** `g4dn.2xlarge` is $0.752/hr for twice the vCPU
and the *same single T4* — we are not CPU-bound. A bigger card (L4, A10G) would
make our numbers *less* comparable to a Jetson, which is the opposite of what the
deck needs.

### Region

**`ap-south-1` (Mumbai)** for interactive work — we are in Bengaluru and the RTT
matters when you are editing over SSH for eight days. Prices differ from
us-east-1; check the console.

Use `us-east-1` only if Mumbai has no `g4dn` capacity, which happens.

### Cost, honestly

| | |
|---|---|
| On-demand, ~6 h/day × 10 days = 60 h | **≈ $32** |
| Spot (variable, roughly a third to two-thirds of on-demand) | **≈ $10–20** |
| EBS, 150 GB gp3, 10 days | **≈ $4** |
| S3, 10 GB staged | **< $1** |
| **Realistic total** | **≈ $40** |

⚠️ **The only way this gets expensive is forgetting to stop the instance.**
Left running 24/7 for ten days: 240 h × $0.526 = **$126**, for maybe 60 h of
actual use. Stopping costs nothing but the EBS volume.

**Set a billing alarm at $50 before you launch anything.** Console → Billing →
Budgets. Two minutes, and it is the difference between a $40 cycle and a
conversation nobody wants to have.

**On spot:** cheaper, but instances can be reclaimed with two minutes' notice.
Fine for eval runs that checkpoint. Bad for a long fine-tune. Start on-demand;
move to spot once the workflow is scripted and restartable.

---

## 2. Launch

### AMI

Use the **AWS Deep Learning AMI (Ubuntu 22.04)**. It ships the NVIDIA driver,
CUDA toolkit and conda environments preinstalled. Installing the driver by hand
on a bare Ubuntu AMI is a reliable way to lose half of Day 1 to `nouveau`
blacklisting and kernel-header mismatches.

Search the AMI catalogue for *"Deep Learning Base OSS Nvidia Driver GPU AMI
(Ubuntu 22.04)"*. Pick the newest.

### Storage

**150 GB gp3.** The default 8 GB will not hold the AMI, let alone data.

Sizing: seq 08 velodyne (~7.8 GB) + labels (~2 GB) + FRNet checkpoint + conda
envs + room to stage a second sequence. 150 GB is comfortable and costs about
$0.40 for the ten days.

### Security group

- **Inbound: SSH (22) from your IP only.** Not `0.0.0.0/0`. A GPU instance open
  to the world gets found and mined within hours.
- Outbound: default (all).

If you want Jupyter or Rerun over the network, **tunnel it over SSH** rather than
opening a port:

```bash
ssh -i vrgrid-key.pem -L 8888:localhost:8888 ubuntu@<ip>
```

### Key pair

Create a new one, download the `.pem`, and:

```bash
chmod 400 vrgrid-key.pem     # ssh refuses a world-readable key
```

### Launch checklist

- [ ] Billing alarm at $50 set **first**
- [ ] Region: ap-south-1
- [ ] Instance: g4dn.xlarge
- [ ] AMI: Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04
- [ ] Storage: 150 GB gp3
- [ ] Security group: SSH from my IP only
- [ ] Key pair created, `.pem` chmod 400
- [ ] **Tag it** `Project=vrgrid`, `Owner=shrestha` so the bill is attributable

---

## 3. First connect, and verify

```bash
ssh -i vrgrid-key.pem ubuntu@<public-ip>

nvidia-smi                    # must show Tesla T4, 15360MiB
nvcc --version                # CUDA toolkit version — note it, it matters for cupy
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Record the driver and CUDA versions in the research log.** When a number
differs between your laptop and the instance, the first question is always which
CUDA it ran on.

---

## 4. ⚠️ Do not upload 84.8 GB

The full dataset is 43,552 scans across 22 sequences. **You do not need it.**

For everything in this lane's first week you need **sequence 08 only**:

| | size |
|---|---|
| `sequences/08/velodyne/` — 4,071 × ~1.9 MB | **~7.8 GB** |
| `sequences/08/labels/` — 4,071 × ~0.48 MB | **~2.0 GB** |
| `poses/08.txt`, `sequences/08/calib.txt` | trivial |
| FRNet checkpoint `.pth` | ~40 MB |
| **Total** | **~10 GB** |

That covers `frnet_eval.py --frames 200`, the fine-tune reproduction, the port
benchmarks, and `timing_table.py`. Add sequence 07 (~2.7 GB) later if you want
the traffic scene.

**10 GB instead of 84.8 GB is the difference between twenty minutes and three
hours of upload**, and inbound transfer to AWS is free either way.

### Route: S3 as a staging bucket

Direct `scp` of 10 GB over a domestic link will stall and you will restart it
twice. S3 handles resume properly.

```bash
# local — one-time
aws s3 mb s3://vrgrid-data-<something-unique> --region ap-south-1

aws s3 sync data/dataset/sequences/08 \
            s3://vrgrid-data-xxx/sequences/08 --region ap-south-1
aws s3 cp   data/dataset/poses/08.txt \
            s3://vrgrid-data-xxx/poses/08.txt --region ap-south-1

# on the instance — fast, S3 to EC2 in-region
aws s3 sync s3://vrgrid-data-xxx/ ~/vrgrid-data/
```

⚠️ **The layout trap.** The loader wants the directory holding *both* `poses/`
and `sequences/`. In our checkout that is `data/dataset`, not `data/`. On the
instance:

```bash
export VRGRID_DATA_ROOT=$HOME/vrgrid-data
python scripts/data_status.py     # verifies what you have
```

`data_status.py` will report the sequences it can see. It exits 0 only on a
complete dataset, so expect a partial report — that is correct here.

### Bring nothing back

Inbound is free; **outbound is charged.** Pull back only `.json`/`.md` results
and log files, never `.rrd` recordings or checkpoints. A 1.2 GB baked demo
dragged out of S3 is a pointless line on the bill.

---

## 5. Environment

```bash
git clone https://github.com/Stxtics03/vrgrid.git && cd vrgrid
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Patchwork++ will fail from PyPI.** Every published version's sdist has CMake
fetching `.../tags/v${CMAKE_PROJECT_VERSION}.tar.gz` with the variable empty
under scikit-build-core; GitHub 404s. Build from the clone:

```bash
git clone --depth 1 https://github.com/url-kaist/patchwork-plusplus.git
pip install ./patchwork-plusplus/python
```

**cupy — match the CUDA version `nvcc` reported:**

```bash
pip install cupy-cuda12x        # or cupy-cuda11x
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```

Then confirm the seam sees it:

```bash
python -c "from src.gpu.allocators import array_module; print(array_module('cuda'))"
```

**Verify before trusting anything:**

```bash
make test                                          # must be green
python scripts/frnet_fast_scatter.py               # reduction equivalence
python scripts/timing_table.py --seq 08 --frames 200
```

⚠️ `timing_table.py` on the instance will give **different numbers from your
laptop** — 4 vCPU on a T4 host is not an i7-14650HX. That is expected and it is
data, not an error. Record both. Our published p50/p99 stays the laptop number
until the port lands, and then we publish both columns.

---

## 6. Working discipline

**Stop, don't terminate.** Stopping preserves the EBS volume and everything on
it; you pay only ~$0.40/day for storage. Terminating destroys it.

```bash
aws ec2 stop-instances  --instance-ids i-xxxx
aws ec2 start-instances --instance-ids i-xxxx
```

⚠️ **The public IP changes on every stop/start** unless you attach an Elastic IP.
Attach one — it is free while associated with a running instance, and it saves
editing `~/.ssh/config` twice a day.

**Use `tmux` for anything over a minute.** SSH drops; a fine-tune that dies at
90% because the wifi blinked is a lost afternoon.

```bash
tmux new -s vrgrid
# ctrl-b d to detach, `tmux a -t vrgrid` to return
```

**End of each session:**
- [ ] Results copied off (`.md`, `.json`, logs only)
- [ ] Research-log entry written — Pratyushi, same day
- [ ] **Instance stopped**
- [ ] Billing console glanced at

---

## 7. Failure modes

| Symptom | Cause / fix |
|---|---|
| `InsufficientInstanceCapacity` on launch | No g4dn in that AZ. Try another AZ, then another region. |
| `nvidia-smi` not found | Wrong AMI — you launched plain Ubuntu. Relaunch on the Deep Learning AMI; do not install the driver by hand. |
| `cupy` imports but fails at first kernel | cupy build vs CUDA runtime mismatch. Reinstall matching `nvcc --version`. |
| `Permission denied (publickey)` | `.pem` not `chmod 400`, or wrong user — it is `ubuntu`, not `ec2-user`, on Ubuntu AMIs. |
| `FileNotFoundError: GT poses not found` | `VRGRID_DATA_ROOT` must point at the dir holding `poses/` **and** `sequences/`. |
| `[!] ground: SEMANTIC-CLASS FALLBACK` | Patchwork++ missing. Rebuild from the git clone. Numbers taken on the fallback are invalid — the proxy admits embankments. |
| SSH dies mid-run | Not using tmux. |
| Instance unreachable after restart | Public IP changed. Attach an Elastic IP. |
| Bill higher than expected | Instance left running. Check the console; stop it. |

---

## 8. What "done" looks like on Day 2

- [ ] Instance launched, tagged, billing alarm set
- [ ] `nvidia-smi` shows Tesla T4; driver and CUDA versions in the log
- [ ] ~10 GB of sequence 08 on the instance, `data_status.py` sees it
- [ ] `make test` green on the instance
- [ ] `frnet_fast_scatter.py` verify passes on **both** machines — max
      bit-identical, mean within the stated ULP bound
- [ ] `frnet_eval.py --frames 200 --fast-scatter` reproduces **90.3% / 65.2%**
- [ ] `timing_table.py` run on the instance, both columns logged
- [ ] Research-log entry (Pratyushi) recording all of it

**The gate that matters is the FRNet reproduction on a second machine with a
different GPU.** If 90.3 / 65.2 comes back identical there, that number is no
longer "what our laptop said" — it is a result. That is worth more at the finals
than anything else on Day 2's list.
