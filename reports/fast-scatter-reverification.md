# `--fast-scatter` re-verification

*Measured 2026-09-11 overnight, on `main` at `9b40ff2`. Pure measurement — the
shim and the frozen `src/perception/frnet/` were not modified.*

Command: `python scripts/frnet_fast_scatter.py` (the script's own self-verify +
benchmark mode), run **three times**.

Environment: `torch 2.13.0+cpu`, **`torch.cuda.is_available() == False`**,
10 torch threads of 16 logical CPUs.

---

## Verdict in one line

**The correctness claim reproduces exactly. The speed-up claim does not.**

---

## A. Correctness — reproduces exactly, both directions

```
verifying on cpu: 20,000 rows x 64 ch into 4,000 slots (28 empty)
  scatter_max   forward  max abs diff 0.000e+00 (0.0 ulp), 0 of 254,208 values differ, empty-slot convention matches
  scatter_mean  forward  max abs diff 0.000e+00 (0.0 ulp), 0 of 256,000 values differ, empty-slot convention matches
  scatter_max   backward max abs diff 0.000e+00
  scatter_mean  backward max abs diff 0.000e+00
  verified: scatter_max exact in both directions, scatter_mean within float32 rounding
```

Matches `docs/research-log.md` (3 Sep): **max abs diff 0.000e+00** for both
reductions, empty-slot conventions preserved in both directions. **No caveat.**

### ⚑ The CUDA caveat could not be tested and remains unverified here

The shim's own header records that bit-identity is a **CPU** result and that on
CUDA `scatter_mean` diverges by up to 2 float32 ulp on ~40% of slots. **This
machine has no CUDA-enabled torch** (`2.13.0+cpu`), so tonight's run exercises
only the CPU path. The CUDA claim is neither confirmed nor contradicted — it is
untested. Anyone quoting "bit-identical" should say *on CPU*.

---

## B. Speed-up — stable tonight, but ~7× lower than logged

Identical shapes both times: **124,000 × 256 into 25,000 slots, CPU.**

| run | `scatter_max` loop | shim | ratio | `scatter_mean` loop | shim | ratio |
|---|---|---|---|---|---|---|
| 1 | 4216.6 ms | 30.08 ms | 140× | 4133.7 ms | 60.16 ms | 69× |
| 2 | 3956.8 ms | 28.30 ms | 140× | 3953.7 ms | 63.45 ms | 62× |
| 3 | 4032.6 ms | 26.55 ms | 152× | 4031.5 ms | 67.30 ms | 60× |
| **tonight, mean** | **4068 ms** | **28.3 ms** | **~144×** | **4040 ms** | **63.6 ms** | **~64×** |
| *research-log, 3 Sep* | *37,452 ms* | *34.3 ms* | ***1093×*** | *34,452 ms* | *10.7 ms* | ***3229×*** |

Tonight's three runs agree to **±3%** — this is not measurement noise. The gap
to the logged figures is real and is in **both** terms:

- the **loop** is ~**9× faster** tonight (4.07 s vs 37.5 s);
- the **shim** is ~**6× slower** for `scatter_mean` (63.6 ms vs 10.7 ms) and
  about the same for `scatter_max` (28.3 ms vs 34.3 ms).

The loop term dominates. One incidental observation: the logged loop timings
differ between the two reductions (37,452 vs 34,452 ms) while tonight's are
near-identical (4032.6 vs 4031.5 ms).

### What this does and does not change

**Does not change:** the engineering conclusion. A ~64–144× speed-up on the
dominant cost is still decisive, the correctness gate still passes, and the
shim is still the right call.

**Does change:** any number derived from 1093×/3229×. `docs/research-log.md:411`
states *"A 600-step fine-tune ran in **2.2 min** against the estimated 3.3 h,
and `frnet_eval.py --frames 200` in about a minute against 35."* The **3.3 h**
and **35 min** baselines are extrapolations from the slow loop. If the loop is
really ~4 s per call on this machine rather than ~37 s, those baselines are
roughly 9× too large, and the *measured* 2.2 min / ~1 min figures would then
represent a much smaller saving than advertised.

I did **not** re-run the end-to-end fine-tune or `frnet_eval.py --frames 200` —
both need the checkpoint and a long run, and neither is on tonight's list.

### Causes not distinguished

I could not separate these, and did not guess: a different torch build (the
3 Sep entry does not record its version), a different thread count, machine load
on 3 Sep, or a different measurement harness. **The 3 Sep numbers are not
re-derivable from what the log records** — which is itself the finding worth
acting on.

**Not edited.** Logged per tonight's rules; the research-log entry is
Shrestha's and this contradicts it.

---

## Reproduce

```bash
python scripts/frnet_fast_scatter.py     # self-verify + benchmark, ~30 s
```

---

## Re-verified 2026-09-13 (OPEN-ITEMS item DL, sub-task 1)

Still exact, through `frnet_eval.py --fast-scatter` rather than the standalone
self-check, so the verification runs on the path the evaluation actually uses:

```
verifying on cpu: 20,000 rows x 64 ch into 4,000 slots (28 empty)
  scatter_max   forward  max abs diff 0.000e+00 (0.0 ulp), 0 of 254,208 values differ,
                         empty-slot convention matches
  scatter_mean  forward  max abs diff 0.000e+00 (0.0 ulp), 0 of 256,000 values differ,
                         empty-slot convention matches
  scatter_max   backward max abs diff 0.000e+00
  scatter_mean  backward max abs diff 0.000e+00
  verified: scatter_max exact in both directions, scatter_mean within float32 rounding

fast-scatter ENABLED in frustum_encoder, frnet_backbone (scatter_max)
             and frustum_encoder (scatter_mean)
```

**Both directions exact**, and the patch reaches all three binding sites — which
is the failure this shim was specifically written to avoid, since
`frnet_backbone` does `from .frustum_encoder import scatter_max` at import and so
binds the function object. Patching only `frustum_encoder.scatter_max` would
leave five of seven per-forward calls on the slow path and the run would merely
look disappointing rather than broken.

### [!] The accuracy half could NOT be re-confirmed — no checkpoint

`frnet_eval.py` stops after the verification above with:

```
checkpoint not found: checkpoints\frnet-semantickitti_seg.pth
```

`checkpoints/` does not exist on this machine, `.gitignore:18` excludes model
weights deliberately, and no `.pth` has ever been committed on any branch. So
**90.3% / 65.2% / 61.1% was not re-measured today** — the numbers stand on the
earlier run, not on this one. Recorded as a blocker rather than glossed:
`MORNING-SUMMARY-2.md`, "Needs your call".

The same gap blocks the three-way plan-regret comparison (GT labels vs
FRNet-predicted labels), because the predicted-label arm needs weights to infer
with. Nothing about that arm was attempted.
