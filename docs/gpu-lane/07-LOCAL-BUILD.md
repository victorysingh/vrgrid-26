# The build — what "working" means, and how to get there

*Shrestha, 2026-09-13. The reference build, stood up before the AWS work so the
instance is reproducing something known-good instead of being guessed at.*

```bash
./scripts/setup_env.sh          # everything: package, Patchwork++, CuPy, Torch
./scripts/verify_env.py         # what is actually present, checked by using it
```

`verify_env.py` exits non-zero only when something the pipeline **requires** is
missing. GPU and dataset are reported and never fatal — CPU-only is a supported
configuration and CI is one. Run it on any machine before quoting a number from
that machine.

## The reference environment

| | |
|---|---|
| Python | **3.13.14** — see below, this is not a free choice |
| numpy | 2.5.3 |
| pypatchworkpp | **1.4.1**, prebuilt wheel (pinned; upstream `3e6903a`, tag v1.4.1) |
| cupy | 14.2.0 + `cupy-cuda12x[ctk]` |
| torch | cu128 build (sm_120 needs CUDA 12.8 or newer) |
| GPU | RTX 5050 Laptop, sm_120 Blackwell, 8151 MiB, driver 610.57.04 |
| OS | Kali rolling 2026.3, gcc 15.3.0, glibc 2.42 |

### Environment variables

```bash
source scripts/env.sh          # or: VRGRID_ASSETS=/mnt/data source scripts/env.sh
```

The three paths the pipeline needs that are not in git live outside every
checkout, in `~/Desktop/sih26/assets`:

| | | |
|---|---|---|
| `dataset/` | 90 GB | SemanticKITTI, sequences 00-21 plus poses |
| `checkpoints/` | 423 MB | FRNet `.pth`, including the fine-tuned ones |
| `figures/` | 50 MB | the PNG/SVG/CSV behind published figures |
| `rerun-recordings/` | 1.3 GB | **orphaned** — baked `.rrd` from the removed dashboard |

They are outside any checkout deliberately. During the vrgrid-26 migration two
checkouts existed at once and all of this lived inside the older one, so
deleting that checkout — which was the plan, once its remote is archived — would
have destroyed the dataset, the checkpoints and every figure behind a published
number. `VRGRID_ASSETS` overrides the location, which is how the AWS instance
will point at its own volume.

## Two things that cost time to discover

**Python 3.13, not 3.14.** `pypatchworkpp` publishes wheels for CPython 3.8–3.13
only. On 3.14 pip falls back to the sdist and builds from source, and that is
exactly how the binary behind our published numbers ended up unreproducible —
see `docs/patchwork-build-record.md`. On 3.13 the pinned 1.4.1 wheel installs
directly, and the pin matters: 1.4.1 is 26% faster in the ground stage than the
next published build, so a floating requirement would let two people compare
different segmenters.

**CuPy cannot reliably find its own CUDA headers.** It ships the toolkit as a pip
package and then does not always locate it, and **the failure is at first kernel
launch, not at import** — so it presents as a broken GPU rather than a missing
header. `setup_env.sh` installs a `.pth` that resolves `CUDA_PATH` relative to
the venv, so it is correct on any machine and nobody exports anything by hand.
Put this in the AWS runbook; it will bite there too.

## What the suite says on a complete environment

The first run with dataset, Patchwork++ and a GPU all present at once:

```
691 passed, 1 failed
```

**The one failure is `test_real_sequence_replay_is_identical` — open item D1, and
nothing else.** Every data-backed and GPU test now runs rather than skipping,
which is why the total jumps from the 622 CI sees.

This matters for reading CI. A green CI badge on this project currently means
"green on the subset a runner with no dataset, no Patchwork++ and no card can
execute". That is a real and useful signal, and it is a narrower one than it
looks.

### D1, reproduced independently with a controlled A/B

Same machine, same 50 frames of seq 08, same kernels; only the ground segmenter
changes:

| ground segmentation | `test_real_sequence_replay_is_identical` |
|---|---|
| Patchwork++ 1.4.1 installed | **fails** — two replays give different map hashes |
| absent, semantic fallback | **passes** |

Eleven other determinism tests pass either way. That isolates the defect to the
stateful Patchwork++ singleton and clears the mapping kernels, independently of
`reports/ring1-reproduction-investigation.md`, which reached the same conclusion
from the accuracy side on a different machine.

## What will differ on AWS

Named now so nobody treats a difference as a regression:

- **The card.** T4 is sm_75, this is sm_120. Both take the cu128 build, but the
  kernels JIT per architecture and the first launch pays for it.
- **Clocks.** T4s throttle; the laptop boosts to 3090 MHz and will not hold it.
  `03-CUDA-PORT-PLAN.md` §7 asks for `clocks.sm` logged beside every timing, and
  that is why.
- **The dataset has to get there.** 90 GB, and only sequence 08 is needed for
  most of it.
- **Two columns, never one.** A laptop number and an instance number are not the
  same experiment. Publish both.
