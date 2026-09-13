# pending-review: ROS 2 adapter — scoping for `export_gridmap()`

**Status:** design only. **Nothing implemented.** No file under `adapters/` was
created or modified.
**Code it would touch:** `adapters/` (new files only) — and **nothing else**.
`include/vrgrid/api.py` is **frozen** and this design requires no change to it.
**Written:** 2026-09-13, against `main` @ `92fc7d0`.

> **[!] The blocker is throughput, not message mapping.** `query()` costs 54.7 µs,
> so a 20 m × 20 m export at 5 cm is **8.75 s per published frame — a 0.11 Hz
> ceiling.** A vectorised bulk reader is a **prerequisite**, and it needs a
> decision about a second query path. **Do not start with the layer mapping** (§4).
>
> **[!] Untestable here.** `rclpy`, `grid_map_msgs`, `sensor_msgs`, `nav_msgs`,
> `std_msgs` and `tf2_ros` are all **absent** on this machine — verified, not
> assumed. Nothing below has been run. Message field names and layer semantics
> are from the published interface definitions and should be re-checked against
> the actual ROS 2 distribution before anyone writes code.

---

## 1. The constraint that shapes everything

`api.py` already states it, and it is worth restating because it determines the
whole design:

> `grid_map::GridMap` is uniform resolution and **cannot hold this map** —
> convenience only, never the primary output. Label it as lossy everywhere it
> appears.

So `export_gridmap()` is **not** "serialise the map". It is "project a foveated
map onto a uniform lattice, losing information on purpose, and say so". Every
decision below follows from choosing *which* information to lose.

## 2. What uniform resolution costs, quantified

A `GridMap` has one `resolution` for the whole grid. Three ways to pick it:

| choice | cells for a 200 m × 200 m map | what it loses |
|---|---|---|
| **5 cm** (ring 0) | 16,000,000 per layer | nothing in ring 0 — but **99.87% of the far-field cells can never hold a return in a single frame** (`sih-math.md` eq. 4, and R-e). 64 MB per float32 layer, overwhelmingly empty. |
| **40 cm** (ring 3) | 250,000 per layer | the entire near-field advantage — kerbs at 8–9 cm and 5 cm elevation accuracy both vanish. This throws away the project's headline result. |
| **5 cm over a bounded near-field window** | e.g. 20 m × 20 m = 160,000 | the far field entirely — but the far field is where uniform resolution was never real anyway. |

**Recommendation: the third.** Export a bounded near-field window at ring 0's
resolution. It is the only option where the exported grid is *honest*: every cell
in it is a cell the map actually resolves at that size.

It also matches what a consumer does with `grid_map` — local planning and
footprint checks in the near field. A consumer wanting the far field should use
`query()` / `query_conservative()`, which are resolution-agnostic by design and
are the interface the project actually stands behind.

The window extent and origin are a **decision for the owner** (§7).

## 3. Layer mapping — and where it gets ugly

`grid_map_msgs/GridMap` carries `layers` (string names) and `data`
(`std_msgs/Float32MultiArray` per layer). **Every layer is float32.** That is
fine for heights and awkward for everything else.

| layer | source (`CellQuery` field) | type reality |
|---|---|---|
| `elevation` | `ground_height` | float metres — **clean fit** |
| `ceiling` | `ceiling_height` | float metres — clean fit |
| `traversability` | `traversability` | **a 6-bit field crammed into a float.** See below |
| `confidence` | `confidence` | integer margin as float — acceptable, but it is *not a probability* (`known-limitations.md` §4) and the layer name will imply it is |
| `semantic_class` | `semantic_class` | integer class id as float — a consumer reading 8.0 must know to look up `road`, and float equality on class ids is a footgun |
| `occupancy` | `occupancy` | tri-state `OCC_UNKNOWN/FREE/OCCUPIED` as 0.0/1.0/2.0 |
| `dynamic` | `dynamic` | bool as 0.0/1.0 |

**`traversability` is the problem.** It is a bitfield —
`TRAV_CLEARANCE|SLOPE|STEP|ROUGHNESS|CLASS|CONFIDENCE`, bits 0–5 — and encoding
it as a float32 means a consumer must do integer bit tests on a value that
arrived as floating point. Two options, both worth the owner's attention:

- **One float layer holding the bitfield**, documented. Compact, and every
  consumer must get the cast right.
- **Six boolean layers**, one per hazard bit, plus a combined `traversable`.
  Larger on the wire, impossible to misread, and far more idiomatic for
  `grid_map` — consumers already expect one concern per layer.

I would take the **six-layer** form and additionally publish a single
`traversable` layer (0.0 = drivable) so the common case needs no bit logic at
all. But it is a wire-format decision and it is not mine.

**`NaN` is the right "no data" value.** `grid_map` treats NaN as empty, which is
exactly what an unobserved cell is. Do **not** publish 0.0 for unknown elevation:
0.0 is a perfectly plausible height and would read as flat ground. This is the
same failure mode as the blind-cone rule — *unknown, never free*.

## 4. [!] The throughput problem, and it is the real blocker

The adapter must read the map through the sanctioned door, not by touching the
SoA — that is what keeps `api.py` frozen and keeps one query semantics. The
vectorised door is `query_region(gm, xs, ys)`, and:

```python
# src/grid/query.py
return [query(gm, float(x), float(y)) for x, y in zip(xs, ys)]
```

**It is a Python loop over `query()`.** Its own docstring says *"a Python loop
over 10^5 points at 10 Hz is not a thing you can do"* — and then it is one,
deliberately, so that there is only one query implementation.

### Measured, and it is worse than "needs optimising"

`query()` costs **54.7 µs per call** on a warmed map (4,000 calls, real seq 08,
20 frames accumulated). So:

| window | cells | time per published frame | rate ceiling |
|---|---|---|---|
| **20 m × 20 m @ 5 cm** | 160,000 | **8.75 s** | **0.11 Hz** |
| 20 m × 20 m @ 10 cm | 40,000 | 2.19 s | 0.46 Hz |
| 10 m × 10 m @ 5 cm | 40,000 | 2.19 s | 0.46 Hz |

**[!] This corrects an earlier draft of this section**, which suggested publishing
below frame rate — *"2 Hz is usually plenty"* — as the honest first move. **It is
not available.** 2 Hz is unreachable by a factor of ~18 even on the smallest
window above. Throttling does not solve a problem of this size; it only changes
how often you pay 8.75 seconds.

So the three options are not peers, and the ordering in the earlier draft was
wrong:

1. **A vectorised bulk reader is a PREREQUISITE, not an optimisation.** Nothing
   ships without it, and it **must be pinned bit-identical to `query_region`**,
   exactly as `bin_points` is. That test is not optional.
   
   **[!] METHOD, decided 2026-09-13 — do not write it from a spec.** Vectorise the
   EXISTING `query()` by transcribing its actual source line by line and replacing
   each scalar operation with the array equivalent **in the same order** — not by
   re-deriving the behaviour from a description. Branches become masks that
   preserve the same per-cell decision, not shortcuts.
   
   **STEP 1 is already done:** `pending-review/query-vectorisation-transcription.md`
   holds the full transcription, every branch, and the hazards. Read it before
   writing anything. Its headline finding materially reduces this item:
   **`ring_of_into`, `flat_slot_into` (already pinned bit-identical),
   `occupancy_state(slots=...)`, `i_fine`/`i_ring` and `unpack_class` are ALREADY
   vectorised or batch-capable.** The only genuinely scalar step is `_refined`.
   So "a second implementation of query semantics" is the right caution for
   `_refined` and an overstatement for the rest, which is reuse.
   
   **And there is no reduction anywhere in `query()`'s chain**, so unlike
   `scatter_mean` there is no summation-order ULP risk — a correct vectorisation
   should be bit-identical by construction. The one real dtype hazard is
   float32/float64 promotion on the `int16 / 100.0` height conversions.
2. **Export off the frame loop** regardless. Necessary but nowhere near
   sufficient — a background thread that takes 8.75 s per export is still 8.75 s
   of CPU competing with a 10 Hz frame loop on the same cores. Note this also
   **interacts with D1**: while the ground segmenter is a stateful singleton, "a
   consistent snapshot" is less well-defined than it looks.
3. **Shrink the window and coarsen it** as a stopgap, understanding that at 10 cm
   the export no longer carries ring 0's resolution and the near-field advantage
   is what a `grid_map` consumer came for.

**The practical consequence for whoever picks this up: do not start with the
message mapping.** The layer and bitfield questions in §3 are real but they are
hours of work on a settled shape. The throughput problem is the one that decides
whether the adapter is possible at all, and it needs a design decision about a
second query path before any ROS code is written.

**Do not put export in the frame loop.** The frame budget is already contested —
whole-frame p99 is over 100 ms on every host measured (D8), and the 10 Hz claim
is recorded as *not proven*. Adding serialisation to that path would make a
contested claim indefensible.

## 5. Node structure

Two nodes, because the input and output halves have different failure modes and
different rates.

```
                 /points  (sensor_msgs/PointCloud2)
                 /tf      (vehicle -> world)
                     |
            [ vrgrid_mapping_node ]      <- owns the map, runs the frame loop
                     |
        +------------+-------------+
        |                          |
  /vrgrid/gridmap           /vrgrid/dynamic_objects
  (grid_map_msgs/GridMap)   (visualization_msgs/MarkerArray
   throttled, LOSSY          or a custom msg from dynamic_objects())
```

**Input side.** `PointCloud2` → `(N, 3)` plus labels. Two traps:

- **Labels.** The pipeline consumes SemanticKITTI `.label` ids and the map's
  semantics come from ground truth, deliberately — *"the model is reported
  alongside the map, never swapped into it"*. A live ROS topic has no ground
  truth, so an adapter either subscribes to a segmentation topic or runs with
  `semantic_class` unlabelled. **Which one is a scoping decision**, and it
  changes what the map means: the refine gate and `TRAV_CLASS` both consult
  class.
- **Frames.** ROS REP-103 is x-forward, y-left, z-up and `docs/frames.md` says
  the vehicle frame is *exactly* that — so no axis permutation is needed, which
  is a genuine piece of luck. But the pose handed to `scatter()` must be built by
  `transforms.vehicle_to_world`, never assembled from a TF lookup by hand:
  `reference_map.build`'s docstring records that passing a raw KITTI pose row
  through produced *"a complete map, in the wrong cells, and nothing downstream
  can tell"*. A TF-derived pose must go through the same composition, and the
  first frame must be checked with `harness.assert_world_is_z_up`.

**Output side.** `GridMap` header: `frame_id` = the world/odom frame,
`info.resolution` = 0.05, `info.length_x/length_y` = the window extent,
`info.pose` = the window centre. Publish on a timer, not per frame (§4).

**The core must not learn about ROS.** `adapters/__init__.py` already states it:
*"The core must never import from here."* So the adapter imports `vrgrid`, and
nothing under `src/` ever imports `rclpy`. Keeping `rclpy` out of
`pyproject.toml`'s hard dependencies is part of that — it belongs in an optional
extra, next to `dash` and `perception`.

## 6. Testing it without a robot

- **Round-trip on the synthetic scene.** `eval/synthetic.py` already writes a
  sequence; feed it through the adapter and assert the exported `elevation` layer
  matches `query()` cell-for-cell over the window. That is the only correctness
  property that matters and it needs no ROS running, only the message classes.
- **NaN discipline.** Assert every unobserved cell is NaN, never 0.0.
- **No allocation in the frame loop.** The existing
  `test_the_two_grid_allocations_stay_fixed` is the model; export must not break
  it, which is an argument for §4.2's background thread over an in-loop path.
- **Lossiness is asserted, not hoped for.** A test that the exported grid does
  **not** round-trip the far field, so nobody later mistakes the export for the
  map.

## 7. Decisions for the owner — I have not made these

1. **Window extent and origin.** 20 m × 20 m vehicle-centred is my suggestion,
   not a finding. It trades wire size against how much of ring 1 gets included.
2. **Bitfield as one float layer, or six boolean layers?** (§3) I lean six plus a
   combined `traversable`; it is a wire-format commitment.
3. **Where do labels come from in a live system?** (§5) This changes what the map
   means, not just how it is fed.
4. **Publish rate.** Tied to §4 and to whether export runs on the frame loop.
5. **Does `export_gridmap()`'s signature need arguments?** It currently takes
   none. A window extent and a layer selection are natural parameters — and
   **adding them changes a frozen signature**, so it needs the three-way. Worth
   settling *before* implementation rather than discovering it halfway.
