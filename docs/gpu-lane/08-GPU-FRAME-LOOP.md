# The pipeline on the GPU

*Shrestha, 2026-09-17. Supersedes the 2026-09-16 version of this file, which
moved only scatter and the cleanup kernel and left the map on the host.*

```bash
python -m vrgrid.run --seq 08 --device cuda --viz                   # Rerun, on the card
python scripts/gpu_parity.py --seq 08 --frames 200                 # bit-identical, every frame
python scripts/timing_table.py --seq 08 --frames 200 --device cuda  # latency
```

**Headline: the whole frame runs at 22.3 ms p50 / 26.7 ms p99 on the card,
against 88.6 / 107.2 ms on the CPU on the same machine, same session. That is
45 FPS, and it meets 10 Hz at p99 with 3.7x headroom. The CPU pipeline misses
it. Range image, labels, reflectivity, the map grid and every map stage run on
the GPU, and the output is bit-identical to the CPU pipeline on every one of
200 real frames.**

## What runs where

| stage | where | how |
|---|---|---|
| load | host | disk |
| transform | host | one BLAS pose product, ~1.4 ms, uploaded |
| **range image** | **device** | CUDA kernel + 64-bit key sort (closest return wins) |
| **semantics, motion** | **device** | LUT built from JP's own functions over all 65,536 label words |
| ground | host | **Patchwork++**, a C++ CPU library, runs while the card works |
| **reflectivity** | **device** | per-pixel kernel, scattered to points |
| **shift, datum** | **device** | strip clear; re-base kernel |
| **bin** | **device** | ring + lattice + toroidal slot in one kernel |
| **scatter** | **device** | payload kernel + `scatter_sorted` on cupy |
| **fuse** | **device** | Kalman, ceiling, occupancy, Boyer-Moore, reflectivity, counts: one kernel |
| **occupancy, centres, guard, eq (32), misses** | **device** | kernels + cupy |

**The grid lives in device memory.** `MapEngine.handle.grid` becomes a
`MirroredGrid`: a read-only host copy that refreshes on the first read after a
frame. The dashboard, `map_hash`, the feature detectors and `occupied_cells()`
keep reading numpy and pay the 10.9 MB copy only when they look. A write to it
raises, because the next sync would silently overwrite it.

Frames from `iter_pipeline(device="cuda")` are `DeviceFrame`s. Their
perception outputs stay on the card for the engine. `semantic`,
`range_image` and the other perception fields download only if something reads
them (the dashboard does, a headless run does not), and they refuse to once
the next frame has reused the buffers.

What stays on the host: Patchwork++ (CLAUDE.md: wire it in, do not
reimplement it), the pose transform (its BLAS rounding cannot be promised on
device) and the disk read.

## Parity — seq 08, frames 0–199, Patchwork++, ghost removal on

For each scan `gpu_parity.py` runs perception on both paths and folds each
frame into its own engine. After every frame it compares exactly: the range
image (NaN-aware), inverse index, reflectivity bytes, semantic labels, motion
flags, every counter, and the full-grid hash. Patchwork++ runs once and both
paths get its mask; running it twice would compare D1, not the GPU.

- **identical on 200 / 200 frames**, final map hash `c680d22153c832658fedb256e41b40bf`
  (after the band rebalance to −3.5 / +4.5 m; `a9f979df7cb1910ac3358fe46191ea65` before it)
  (after the 8 m band fix, `known-limitations.md` §11: `payload` zeroes the weight
  of out-of-band ground and `rebase_heights` drops evidence that leaves the band;
  `4e180a121d7b5aaac95df6a97fe2374b` before it, same day).
  (Before 2026-09-17 it was `4313df1a58e68f0ed69a6e5417db4000`. The change is the
  per-block ring rule of open item D2 on BOTH paths -- 0.224% of returns that
  used to be dropped are now binned -- and the two paths still agree.)
- 7,609,197 cells cleared by §10.4 over the run, so the comparison exercises the cleanup
- also identical with `--max-points 100000`, where the engine truncates each scan

## On real data — every labelled sequence (2026-09-17)

Two questions, kept apart. **Does the GPU pipeline build the same map as the
CPU pipeline?** `gpu_parity.py` answers that, comparing perception outputs,
counters and the full-grid hash after **every** frame. **Is that map accurate?**
`scripts/engine_eval.py` answers that, scoring the engine's own map, not the
eval harness's CPU map, against the same M\* with the same §9 metrics, and
requiring the CPU and CUDA engines to agree on every number.

```bash
python scripts/gpu_parity.py --seq <00..10> --frames 200
python scripts/gpu_parity.py --seq 08 --frames 4071        # the whole drive, 45.7 m of climb
python scripts/engine_eval.py --seq <00..10> --frames 40    # accuracy of the GPU map
```

| seq | parity, frames identical | final hash | cells cleared §10.4 | GPU map ring 1 RMSE cm / ρ | ring 3 RMSE cm / ρ | CPU engine metrics identical |
|---|---|---|---|---|---|---|
| 00 | 200/200 | `375bc8d1…` | 2,326,258 | 6.48 / 1.26 | 10.94 / 1.34 | yes |
| 01 | 200/200 | `6a4058b7…` | 6,123,319 | 2.05 / 1.29 | 9.41 / 1.34 | yes |
| 02 | 200/200 | `cc794104…` | 6,658,543 | 2.30 / 1.22 | 15.03 / 1.44 | yes |
| 03 | 200/200 | `f2b86fb0…` | 4,729,703 | 12.39 / 1.29 | 6.84 / 1.15 | yes |
| 04 | 200/200 | `6f1bb46e…` | 2,605,289 | 4.02 / 1.20 | 18.23 / 1.25 | yes |
| 05 | 200/200 | `11f529c7…` | 3,253,047 | 3.37 / 1.33 | 12.16 / 1.21 | yes |
| 06 | 200/200 | `a210f555…` | 3,331,162 | 3.12 / 1.25 | 12.55 / 1.14 | yes |
| 07 | 200/200 | `d90f49f6…` | 2,393,890 | 2.57 / 1.13 | 13.89 / 1.28 | yes |
| 08 | 200/200 | `c680d221…` | 7,648,836 | 2.33 / 1.16 | 3.89 / 1.05 | yes |
| 09 | 200/200 | `47995658…` | 4,634,922 | 3.40 / 1.27 | 10.63 / 1.45 | yes |
| 10 | 200/200 | `b3e47136…` | 4,383,550 | 2.79 / 1.11 | 2.80 / 1.15 | yes |

Across all eleven labelled SemanticKITTI sequences, the GPU pipeline builds
**bit-identical maps to the CPU pipeline on every frame checked**, and those
maps score the same against M\*. The accuracy is in line with the eval
harness's table (`known-limitations.md` §2b). The two are not identical,
because the engine also runs §10.4 cleanup and slides its datum every frame.

**The whole of seq 08, all 4,071 frames** (the sequence with 45.7 m of climb,
so every datum step and band edge is exercised): **identical on every frame**,
final hash `33a54623145c7ae73b30b7410c0eae0b`, 94,903,565 cells cleared by
§10.4. cupy's pool held **145.08 MB used at the end of the drive, the same as
after frame 0**. The device memory bound holds over a full sequence, not only
a short one; reserved settled at 385.8 MB.

**What this does not cover.** Sequences 11–21 have no labels, and the pipeline
takes semantics from the `.label` files, so they cannot run. There is **no
simulator integration**: CARLA (open item D7) was never started. The synthetic
scenes the unit tests use run on the card in `tests/test_device.py`. They are
analytic, and do not substitute for a simulator.

## What it took to make "identical" true

Each of these was measured before it was relied on. Each would otherwise have
produced a map that looked right and hashed differently.

1. **NVRTC fuses `a*b + c` by default.** `x*0.1 + 0.7` over 1M doubles: 289,150
   mismatches with contraction, 0 with `--fmad=false`. Every kernel is compiled
   with the flag and written in the numpy reference's operation order.
2. **cupy's float `//` is not numpy's.** `1.0 // 0.1` is 9 in numpy and 10 in
   cupy, and 42,425 of 5M lattice-scale coordinates disagreed. `bin_points`
   floors every world coordinate onto the 5 cm lattice, so the kernel carries
   numpy's `npy_divmod` line for line.
3. **atan2 / asin.** The float32 device overloads put 116 of 24.5M points in a
   different range-image pixel over 200 frames. numpy's float32 results are
   correctly rounded on glibc 2.42, and double-precision device results rounded
   to float32 matched them on all 3.7M points tested. The kernel computes in
   double and rounds.
4. **The variance codec uses `log`,** which is not guaranteed to agree across
   libms. The kernel does not call it. `variance_code_thresholds()` bisects
   Aakash's `quantise_variance_cm2` over double bit patterns for the 255 code
   boundaries, checks monotonicity, and the kernel binary-searches the table.
   It matches the codec on 2M values including every boundary ±1 ulp.
5. **"Closest return wins"** is JP's stable argsort by range plus
   first-per-pixel. On device it is a sort of unique `pixel | range bits |
   index` keys, which selects the same winner including on exact range ties.

## Latency — seq 08, 200 frames, same session, back to back

| stage | cpu p50 | cpu p99 | **cuda p50** | **cuda p99** |
|---|---|---|---|---|
| load | 0.51 | 0.93 | 0.46 | 0.84 |
| transform | 1.45 | 2.66 | 1.43 | 2.16 |
| range_image | 22.96 | 29.05 | **1.13** | **2.85** |
| semantics + motion | 0.50 | 0.75 | **0.10** | **0.14** |
| ground (Patchwork++, host both) | 12.54 | 14.59 | 12.57 | 14.86 |
| reflectivity | 3.53 | 4.98 | **0.02** | **0.04** |
| bin | 6.77 | 9.82 | **0.18** | **0.23** |
| bin, after D2 (2026-09-17) | 10.27 | 12.92 | **0.35** | **0.42** |
| scatter | 7.08 | 10.86 | **1.47** | **2.02** |
| fuse | 4.11 | 5.70 | **0.05** | **0.07** |
| cleanup | 26.70 | 34.79 | **1.65** | **2.40** |
| shift | 2.11 | 5.81 | 2.94 | 4.64 |
| **FRAME** | **88.62** | **107.19** | **22.27** | **26.73** |
| 10 Hz at p99 | misses | | **meets, 3.7x** | |

Stage rows on cuda synchronise the device at each boundary, so each row is
real work and not a kernel launch. The free-running pass, with no per-stage
synchronisation, gives 22.23 / 27.89 ms: the staged table is not flattering
the device. On cuda, `shift` includes uploading the frame's ground mask.

The `bin` row after D2 is the per-block ring rule: three levels of a coarse-to-fine
descent per point instead of one comparison. Whole frame after it, same command:
cpu 91.73 / 102.77 ms, cuda **22.26 / 26.74 ms** -- the device frame does not
move, and the host frame still misses 10 Hz at p99 as it did before.

**Patchwork++ is now 56% of the frame.** It is the only large stage left and
it is a CPU library by project rule. Everything else in the frame totals under
10 ms.

## Device memory

134.96 MB preallocated on the card (grid 10.9 MB, scatter scratch, and the
eq (32) candidate buffers sized to the structural cap of 910,000 slots). The
pool reports ~145 MB used and up to ~416 MB reserved, because cupy caches the
per-frame temporaries of `flatnonzero`, `searchsorted` and the sort. The host
allocation and every memory figure on a slide are unchanged: `report()`
describes the CPU configuration, and the device bytes are declared separately
through `MapEngine.device_bytes()`.

## Tests

`tests/test_device.py`, 13 tests:
- each kernel against its reference on adversarial inputs: lattice-boundary
  coordinates, exact range ties, codec boundaries ±1 ulp
- the engine on both devices over the Gate 3 ghost scene, with ghost removal
  on and off, hashing the grid every frame
- stale-frame refusal, the read-only mirror, and host allocation per step

The codec table test needs no card and runs in CI; the rest skip without one.

## Hand-offs, not done here

- ~~`python -m vrgrid.dash` on CPU only~~ — `--device cuda`, done 2026-09-17.
- ~~`dashboard/gpu_stats.py` docstring~~ — corrected 2026-09-17.
- VRAM attribution and FRNet contention — done, `09-VRAM-CONTENTION.md`.
- **T4 column** on the AWS instance, per port plan §7 — still open.
