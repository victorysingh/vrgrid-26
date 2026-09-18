# R9's missing rows, R4's cost, and stage attrition

*Shrestha, 2026-09-17. Roadmap Days 1, 4 and 7 in the GPU/CUDA column
(`VRgrid-10-Day-Roadmap.docx`), for the items that do not need AWS. All on the
RTX 5050 laptop, real SemanticKITTI seq 08.*

## 1. R9: split/merge and the pyramid (Day 1)

`timing_table.py --seq` covered ground, scatter and fuse, and printed
split/merge as "absent". R9 asks for split/merge and the pyramid too. Neither
runs in `MapEngine`'s frame loop. The refinement pool that splits cells lives in
the eval harness's map, and the pyramid reduces the §7.1 layer that only the
harness updates. So `scripts/r9_stages.py` drives `harness.run_sequence` and
times the real `gate.apply`, `traversability.update` and `pyramid.build` calls
on the same frames. It reimplements nothing.

```
python scripts/r9_stages.py --seq 08 --frames 200     # 10 warm-up frames discarded

  stage             p50 ms   p99 ms   max ms     n
  split_merge        49.59    63.50    67.79   200
  traversability     37.35    49.22    53.32   200
  pyramid             2.81     3.72     4.78   200

  refinement pool per frame (p50 / max): fired 4104/4923  acquired 141/522  released 5/115  refused 3492/4125
  pool occupancy at the end: 512/512 blocks
  pyramid memory: 2.73 MB nodes + 0.38 MB scratch
```

**What this said.** The pyramid was cheap (2.8 ms). The refinement pool and
the §7.1 pass were not: 87 ms p50 together, four times the 22 ms CUDA frame.
The pool fills after a few frames and then refuses ~3,500 gate requests a
frame, and each refusal paid for full owner-table scans in a Python loop.

### Sped up, same bits — 2026-09-17

```
python scripts/r9_stages.py --seq 08 --frames 200 --traversability-device cuda

  stage             p50 ms   p99 ms   max ms     n
  split_merge         8.50    18.17    22.22   200      was 49.59 / 63.50
  traversability      4.78     6.31     6.55   200      was 37.35 / 49.22  (30.07 / 37.80 on the host path)
  pyramid             2.80     3.74     3.78   200
```

**~87 ms → ~16 ms**, and **nothing the stages decide changed.** Over 60 frames
of seq 08, the fast paths and the originals produce the same map hash, the
same pool owner table, scores and cells, and the same gate counts on every
frame (`real_equiv` check, 0 frames differing; `819b5b5c…` both ways).

- **Split/merge (`gate.apply`).** Same sequential decisions, made cheaply.
  Rings, centres and priorities for all fired cells at once; blocks found
  through a dict mirroring the owner table; free blocks taken lowest index
  first; the eviction `argmin` recomputed only after a score changes (a
  refusal changes nothing). Release asks `lattice.migrate_ring_many` once for
  every held block. The per-cell original is kept as `gate.apply_reference`.
  `tests/test_gate_fast.py` compares them after every frame, with the normal
  pool and with an 8-block pool that evicts and refuses constantly. The test
  was mutation-checked: taking the highest free block, or evicting on a tied
  score, both fail it.
- **Traversability (`traversability.bitfield`).** On the host, per-ring
  stencils cached, the variance `exp` and the class `isin` replaced by
  256-entry lookup tables built from the same functions, and bits OR-ed in
  place: 37 → 30 ms. `traversability.update(device="cuda")` computes it on the
  card (`gpu/traversability_device.py`): 4.8 ms including upload and download.
  Every operation used is exact on the device except `hypot`, which differs
  from glibc in the last bit on ~30% of inputs. Cells whose device slope is
  within a relative 1e-12 of the threshold are therefore settled on the host
  with NumPy's `hypot`, so the bits are exact by construction. Opt-in (the
  harness reads `gm.traversability_device`); the default stays on the host.
  Pinned by `test_bitfield_matches_the_reference` and
  `test_device_bitfield_matches_host`.

⚑ **For Aakash, noticed while reading `gate.apply`, and deliberately left
  unchanged:** when a cell that already holds a block fires again, `acquire`
  returns the existing block and `_fill` re-splits the parent over it every
  frame. Whatever the refined children had accumulated is overwritten with
  the parent's split each time the gate fires. That may be intended (the
  pool has no separate child fusion yet) or it may not; changing it would
  change results, so it is his call.

## 2. R4: what the block-level ring rule costs in cells (Day 4)

The roadmap's gate is "0 overlapping footprints, cost ~0.1% cell increase,
8.94 MB unchanged". Overlaps are 0 by CI test (`known-limitations.md` §8). The
cost was never measured. Distinct cells written per frame, seq 08, 40 frames,
the old per-point rule (`6af6907`) against the per-block rule, same points,
same windows:

| ring | old | new | change |
|---|---|---|---|
| 0 | 29,105 | 29,005 | −0.34% |
| 1 | 23,148 | 23,144 | −0.02% |
| 2 | 9,514 | 9,489 | −0.26% |
| 3 | 2,176 | 2,198 | +1.01% |
| **all** | **63,944** | **63,837** | **−0.17%** |

The cost is effectively zero, slightly negative overall. The old figure includes
cells for returns the old rule then dropped at the window, which the new rule
keeps in a coarser ring instead. **8.94 MB cannot move:** every ring is
preallocated at its fixed half-width, whatever the ring rule decides.

## 3. Stage attrition (Day 7, Rule 3)

`MapEngine(attrition=True)` counts, per frame, the pipeline stage where each
return ends. The counts are identical on CPU and CUDA every frame, and
`gpu_parity.py` now compares them. `engine.attrition_codes()` returns the
per-point stage as `uint8`. That is the input for the "examined and rejected vs
never reached" map colouring in Aakash/Srinivas's Day 7 column (`gpu/attrition.py`).

| terminal stage | meaning |
|---|---|
| capped | beyond `max_points`, never binned |
| outside_map | past the coarsest ring's window |
| nonground | in the map; occupancy, class and ceiling only, no height |
| ground_out_of_band | ground, in the map, outside the 8 m band; height weight 0 |
| ground_fused | ground, in the map, height fused |

Seq 08, 200 frames (`gpu_parity.py --seq 08 --frames 200`):

```
capped 0.00%   outside_map 0.00%   nonground 36.65%   ground_out_of_band 0.18%   ground_fused 63.17%
beside the chain: moving 1.94%, projected 20.92%
```

**The number worth a slide is `projected`.** Only **20.9% of returns win a
64 × 512 range-image pixel.** The other 79% never feed §10.4's current-return
guard or reflectivity. That is exactly the "never reached the stage" share
Rule 3 says to publish next to any cleanup or reflectivity number. Nothing is
lost from the MAP: capping and out-of-map are both 0.00% on seq 08, and 99.8%
of in-map returns reach fusion. Tested in `tests/test_attrition.py`, including
CPU and CUDA agreement return by return.

## Still open in this column

- **AWS (Days 1, 2, 6, 9):** scripted in `scripts/aws/t4.sh`. Waiting for AWS
  credentials on this machine; the account is on the Free plan.
- **Day 5 kernel side of R5 (sticky VRU bit):** R5's design is JP and Hriday's,
  and not written yet.
- **Day 8 ROS-loop re-profile:** needs Aakash and Srinivas's ROS adapter.
