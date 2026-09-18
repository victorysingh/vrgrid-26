# R-b re-measured on the merged tree — and the cold-start finding

*2026-09-18, `jp/p99-alloc-fixes` @ `e582857` (post-merge). Real seq 08, GT labels, CPU host path,
`scripts/timing_table.py --seq 08 --frames 200 --frame-times`. Machine state OK before **and**
after every one of the ten runs (clock 2400/2400 MHz, commit ~14.1 of 15.73 GB).*

**Read the limitation first: the gate is met only on a warm machine.** A pass whose first run
starts from a cold page cache **misses** it — pooled p99 **101.70 ms**. That is a real property of
this pipeline and it was not documented before, because the pre-merge measurement happened to run
warm. Both numbers below are true; they differ only in whether a cold first run is in the pool.

## Why this was needed

The figure of record, **89.58 ms** (`36a4dbe`), describes the **pre-port CPU path** *and* the
**pre-`df35fd5` grid**. The merge `ef4524e` changed which points reach the grid at all (host-parity
run: `binned` 123,254 → 123,389). So no current latency figure existed for the merged tree.

## Protocol, matched to the original

`36a4dbe` recorded "5 fresh-process runs × 200 frames … after three consecutive trusted readings
20 s apart", with per-run p99s of 89.28 / 98.40 / 89.21 / 91.61 / 91.82 — **no cold run among
them.** The pooled gate is JP's: *p99 over every frame of every run, ranked together.*

Ten runs were taken, as two independent five-run passes of 1,000 frames each. Nothing was
discarded; both passes are reported.

## Results

| pass | per-run p99 | pooled p50 | **pooled p99** | frames > 100 ms | gate |
|---|---|---|---|---|---|
| **2 — warm, settled** (matches the original's precondition) | 87.40 / 89.35 / 87.69 / 87.22 / 87.43 | 81.19 | **87.69 ms** | 1 of 1,000 | **MET** |
| **1 — first run cold** | 105.69 / 87.46 / 88.58 / 89.47 / 87.51 | 82.10 | **101.70 ms** | 19 of 1,000 | **MISSED** |
| *pre-merge `36a4dbe`, for reference* | 89.28 / 98.40 / 89.21 / 91.61 / 91.82 | 79.74 | *89.58 ms* | 3 of 1,000 | *MET* |

Warm pooled p99 **bootstrap 95% CI: 86.85 – 89.26 ms** (2,000 resamples over frames, seed
20260918).

**The merge did not regress latency.** Warm pooled p99 **87.69 ms** against the pre-merge
**89.58 ms**, with the pre-merge CI's lower bound at 88.35 — so the two are close, and if anything
the merged tree is slightly better at the tail. p50 moved the other way, 79.74 → 81.19 ms.

### The percentile method does not decide anything

The per-run `FRAME` row printed on stdout uses `method="higher"` over the timer's own samples,
which include the start-up frame (201); the `--frame-times` JSON holds 200. On a single run that
can differ visibly — run 7 printed p99 99.19 while its 200 frame times give 89.35, because the
tail there is a few isolated spikes. Pooled over 1,000 frames the two methods agree to ~0.5 ms:

| pass | p99 linear | p99 `method="higher"` |
|---|---|---|
| warm | 87.69 | 88.17 |
| cold-start | 101.70 | 102.41 |

Both verdicts hold under either method.

## What this figure is, precisely

**Pooled p99 87.69 ms — merged tree (`ef4524e`), post-`df35fd5` grid, CPU host path, ground-truth
labels, one laptop (i7-13620H, D8 unresolved), warm page cache, 1,000 frames over 5 fresh
processes.** Quote it with those qualifiers. It is *not* comparable to upstream's 22 ms/frame GPU
figure, which is a different execution model on different hardware.

## What is still not settled

- **Cold start misses the gate**, and nothing in the pipeline warms the cache deliberately. Whether
  the gate should be defined on a warm machine — or whether the cold path deserves its own fix — is
  a decision, not a measurement.
- **One machine.** D8 is still open; a reference machine has not been agreed.
- `d540618` and `697a2bd` still await Shrestha's review.
