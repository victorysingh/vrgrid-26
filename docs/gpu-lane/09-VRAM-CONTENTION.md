# VRAM attribution and contention — the map and FRNet on one card

*Shrestha, 2026-09-17. Closes `03-CUDA-PORT-PLAN.md` §5 (R9b) and §6.*

```bash
python scripts/vram_contention.py --seq 08 --frames 200 --pair-seconds 40 \
    --out docs/gpu-lane/09-vram-contention.json
```

Measured on the **NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB)**, driver 610.57.04, CUDA runtime
12.9, cupy 14.2.0, torch 2.11.0+cu128. Seq 08, `main` after `6222625`. The card was idle at 15 MiB
before the run. Every configuration runs in its own process, so each gets a fresh CUDA context and
an empty allocator. Each process discards 10 warm-up frames, and the device is synchronised before
every clock stops. The raw record is `09-vram-contention.json`.

---

## 1. VRAM attribution (§5)

| Component | Claimed | Device-resident | Measured how |
|---|---|---|---|
| Map grid arrays | 8.94 MB (745,000 cells × 12 B) | **10.92 MB** | device grid `nbytes`; 910,000 slots, because a toroidal ring stores its full square |
| §10.4 cleanup buffers | — | **107.4 MB** | candidate and visibility scratch, sized to the 910,000-slot structural cap |
| Scatter / bin buffers | — | **16.6 MB** | point-sized, 150,000-return cap |
| Perception buffers | — | **10.1 MB** | `DevicePerception` arrays: points, range image, labels |
| **Our declared total** | — | **145.08 MB** | the four rows above |
| cupy pool, used / reserved | — | 145.1 / 382.1 MB | `device_bytes()` after 210 frames |
| CUDA context alone | — | 86 MiB (102 with torch) | `nvidia-smi`, context-only process |
| **Grid process on the card** | — | **594 MiB** | `nvidia-smi --query-compute-apps` |
| FRNet weights | — | 40.2 MB (10.02 M params) | torch allocated delta at `.to("cuda")` |
| FRNet peak, allocated / reserved | — | 1,599 / 4,129 MB | `torch.cuda.max_memory_*`, batch 1 |
| FRNet process on the card | — | 4,056 MiB | `nvidia-smi` |
| Both, one process | — | 4,564 MiB | `nvidia-smi` |
| Whole card, two processes, peak | — | 4,722 MiB of 8,151 | `nvidia-smi memory.used`, sampled every second |

**What it says.**

- **The declared figure is exact.** 145.08 MB declared, 145.1 MB in cupy's pool. Every byte our
  arrays hold is attributed, and `test_the_declared_device_memory_is_what_the_pool_holds` pins it.
- **The headline 8.94 MB is not what is on the card, and the difference is named.** The grid is
  10.92 MB because a toroidal ring stores its whole square (910,000 slots, not 745,000 cells), the
  same caveat `lattice.buffer_cells` carries on the host. **74% of our device memory is the §10.4
  cleanup**, sized to the structural cap on occupied cells.
- **The driver sees 594 MiB for a 145 MB claim.** 86 MiB is the context. 237 MB is cupy caching
  per-frame temporaries it has taken from the driver and not returned (pool 382 reserved − 145
  used). The rest is cupy's compiled kernels and library handles. A hard bound on *our arrays*
  holds; a hard bound on *the process* does not, and the difference is allocator caching we could
  cap with `set_limit`.
- **FRNet dominates.** 40 MB of weights, 1.6 GB of activations at batch 1, and 4.1 GB reserved by
  torch's caching allocator. **Both fit on this 8 GB card with 3.4 GB spare, and trivially on a
  16 GB T4.** Our map is ~13% of the combined process.

## 2. Contention (§6) — ms per frame, seq 08

| Configuration | p50 | p99 | max | frames |
|---|---|---|---|---|
| **Grid alone** | **21.97** | **27.80** | 198.02 | 200 |
| **FRNet alone** (batch 1, fast scatter) | **91.33** | **98.68** | 106.78 | 200 |
| One loop, whole frame | 122.81 | 130.30 | 284.47 | 200 |
|   … grid's share | 23.45 | 30.08 | 189.81 | 200 |
|   … FRNet's share | 99.47 | 105.21 | 123.20 | 200 |
| Two processes, grid | 52.78 | 69.10 | 382.90 | 749 |
| Two processes, FRNet | 104.21 | 132.08 | 143.07 | 378 |

The three questions §6 asked:

1. **Does it fit?** Yes: 4.7 GB peak across both, on an 8 GB card.
2. **What does contention cost?** It depends on how the two share the card.
   - **One loop:** the frame is **8.4% more than the sum of the parts** (122.8 vs 113.3 ms p50). The
     grid's share rises 7% (21.97 → 23.45) and FRNet's 9% (91.3 → 99.5).
   - **Two processes:** the card time-slices between two CUDA contexts. The grid's p50 rises
     **+140%** (22 → 53 ms) and its p99 **+149%**. FRNet's p50 rises +14% and its p99 +34%. The
     small, frequent grid kernels suffer most from time-slicing, as expected.
3. **Does the p99 survive?**
   - **The map's does, in both arrangements.** It is 30 ms in one loop and 69 ms as a separate
     process, both inside the 100 ms budget.
   - **The frame does not when FRNet is in the same loop:** 130 ms p99 misses 10 Hz. FRNet alone,
     at 91 / 99 ms, already sits on the budget.

**The decision this supports.** Keeping FRNet out of the map's loop is right on latency, not only on
the evaluation argument in `frnet_eval.py`. In the same loop it costs the 10 Hz frame. As its own
process it leaves the map at 10 Hz with 1.4× headroom at p99, but it costs the map its 3.6×
headroom. If segmentation ever goes in, it belongs in a separate process at its own rate, with the
map consuming its latest labels.

## 3. Caveats, stated

- **Power cap.** Every FRNet configuration ran with the **software power cap active** (throttle
  reason 0x4, 96 W peak); grid alone did not (0x0, 32 W). No thermal or hardware slowdown was
  flagged. The card peaked at 76 °C during the two-process run, which ran last. FRNet's figures
  therefore include laptop power limiting, and **a T4 (70 W) will limit differently**. The T4
  column is still open.
- **Max values.** Grid alone has a 198 ms max over 200 frames against a 27.8 ms p99. It is a single
  frame, and the max is reported, not hidden.
- **The two-process window** is 40 s. Grid cycles through the 200 frames several times, FRNet about
  twice. Both were released together after their own warm-up and stopped together, so every
  recorded frame was contended.
- **FRNet's labels do not enter the map** in the one-loop configuration. This measures cost, not
  what the labels would do (`CLAUDE.md`: semantics come from the `.label` files).
- **Fast scatter.** FRNet runs with `frnet_fast_scatter`, verified equivalent at start-up. The
  port's own loops are ~35× slower and could not be in any loop.

## 4. `03-CUDA-PORT-PLAN.md` §9, now

- [x] Float audit complete (`05-FLOAT-AUDIT.md`)
- [x] `bin_points`, `scatter`, `visibility_cleanup` on the device (and every other map stage, `08`)
- [x] Determinism on the device: bit-identical to CPU, 200/200 frames (`gpu_parity.py`)
- [x] int32 overflow bound asserted in a test, not only reasoned (`test_kernels.py::test_a_whole_frame_in_one_cell_does_not_overflow_int32`)
- [ ] Per-stage table, two columns: laptop ✓ / **T4 open** (no instance or AWS credentials on this machine)
- [x] VRAM attribution table, pool and card both shown — §1
- [x] Contention measured three ways (plus two-process) — §2
- [x] GPU, driver, CUDA and power state recorded with every number — this file and the JSON
- [x] Stages that got slower published — `08`, and the contention costs above
