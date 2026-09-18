# R3 — Ring-boundary-under-anisotropy: nearest-corner test + boundary snapping

**Status:** **IMPLEMENTED 2026-09-17**, by Shrestha while Aakash was away, and
**not as designed below in one respect.** Read this block first; the design
underneath is kept as it was written.

- **Part A (nearest point, not centre) — kept**, but applied per BLOCK,
  coarse to fine: a ring-L block splits into ring L-1 only if every child fits
  ring L-1's window and eq. (20) at the block's nearest point is below
  R_{L-1} (or the rear floor forces it). Every point of a block therefore gets
  the same ring, which is what makes the partition hold.
- **Part B (snap the boundary to the coarser lattice) — not used.** In the
  engine the ring rule ran on the ROTATED sensor frame, so the boundary is not
  axis-aligned and no snap puts it on the lattice; the per-block decision makes
  the partition hold without it. §2's "per-axis" nearest-corner argument is
  also only exact at yaw 0; under a heading the bound is taken from the block
  centre ± (|cos|+|sin|)·c/2, a lower bound that is still a function of the
  block alone.
- **Found on the way, and larger than the anisotropy case:** the defect was
  live at v = 0 on real data. Seq 08, 30 frames: nesting on **every** frame
  (0.108% of coarse cells) and **0.224% of returns dropped** because the ring
  was chosen in the rotated frame and missed its world-aligned window. Both
  are 0 after.
- §6's open questions: (1) `metrics._ring_cells` moved with it — the centre
  test is now exact, not a convention; (2) the rear floor is a block predicate
  inside the descent; (3) OUTSIDE is now "past the coarsest window", decided
  in integers.
- Tests: `test_no_cell_footprint_contains_another_under_foveation` and
  `test_every_return_inside_the_map_is_binned` (both `partition`, CI-blocking).
  The first fails on the old rule at 20/28 speeds on 5/10/20/40 and 10/20 on
  5/10/50. Write-up: `docs/known-limitations.md` §8.
**Owner of the code it touches:** `src/grid/lattice.py` — Aakash.
**Written:** 2026-09-12, against `main` @ `f3a0337`.

---

## 1. The defect

`ring_of` decides a point's ring by comparing an **anisotropically scaled**
distance against **fixed** ring half-widths.

`lattice.py:34` — `stretch_factors`:

```python
a_f = min(max(1.0 + a.kappa_forward * t, 1.0), 2.0)   # t = speed_ms / v_ref_ms
a_s = 1.0 / (1.0 + a.kappa_side * t)
```

`lattice.py:49` — `d_aniso`:

```python
d = max( x⁺/a_f , x⁻/a_r , |y|/a_s )
```

`lattice.py:150` — the comparison:

```python
ring_aniso = np.searchsorted(radii, d, side="right")
```

`a_f` and `a_s` are **continuous functions of speed**. The effective ring
boundary in world coordinates is therefore `R_L · a_f(v)` forward and
`R_L · a_s(v)` laterally — **a real number that lands wherever it lands**, with
no relationship to the 5 cm base lattice or to any ring's cell size.

**Consequence.** A cell is a square block on the base lattice. Its ring is
decided by evaluating `ring_of` at **one point** — and because the boundary can
fall strictly inside a block, the block's *footprint* can span the boundary
while the tested point sits on one side of it. The block is then filed wholly
into one ring while part of the ground it covers belongs to the other.

Two cells at different levels can end up with **footprints that contain one
another** — which breaks the §2.2 partition property that every downstream
consumer assumes (`block_stats` treats a ring-L footprint as a clean `k × k`
slice of the base lattice; `_compared` assumes one cell answers for one patch).

**Why testing the centre is the wrong test.** Centre-testing gives the wrong
verdict on a straddling block whenever the boundary falls between the centre
and the nearer edge — i.e. on a meaningful fraction of exactly the blocks that
matter. It is wrong in the **unsafe** direction as often as the safe one: a
block whose near corner is inside ring L but whose centre is outside gets filed
coarse, so the finest ring silently declines ground it should have resolved.

---

## 2. The fix, in two independent parts

Both are needed. Part A alone still leaves a real-valued boundary; part B alone
still tests the wrong point.

### Part A — decide on the NEAREST CORNER, not the centre

For a candidate block at ring level `L` with cell size `c_L`, spanning
`[x₀, x₀+c_L) × [y₀, y₀+c_L)` in the vehicle frame, evaluate the ring predicate
at the corner **nearest the sensor** rather than at the centre.

Nearest corner, per axis, is the standard point-to-box clamp toward the origin:

```
x_near = 0 if x₀ <= 0 <= x₀+c_L else (x₀ if x₀ > 0 else x₀+c_L)
y_near = 0 if y₀ <= 0 <= y₀+c_L else (y₀ if y₀ > 0 else y₀+c_L)
```

i.e. `x_near = clamp(0, x₀, x₀+c_L)`. This is the point of the block with the
smallest `d_aniso`, because `d_aniso` is monotone in `|x|` and `|y|` on each
side of the axes — **and that monotonicity is what makes the corner the correct
test**: if the nearest corner is outside ring L, no part of the block is inside.

**Direction of the change.** Nearest-corner is *more inclusive* than centre —
a block is admitted to the finer ring if **any** of it belongs there. That is
the safe direction (resolve finely when in doubt) and is what produces the
small cell-count increase in §4.

⚑ `d_aniso` is **not** a Euclidean norm and the nearest-corner argument must be
made per-term, not by appealing to convexity of a ball. It holds because
`d = max(x⁺/a_f, x⁻/a_r, |y|/a_s)` is a max of three terms each monotone
non-decreasing in distance from the origin along its own axis. Minimising each
term independently over the block is exactly the per-axis clamp above, and the
max of the per-axis minima is attained at that corner. **Whoever implements
this should re-derive that rather than take it from here** — it is the one step
where a wrong assumption would be silent.

### Part B — snap the anisotropic boundary to the lattice before comparing

Replace the raw comparison against `R_L` with a comparison against a **snapped**
boundary:

```
R_L_eff(v) = floor( R_L · a(v) / c_snap ) · c_snap
c_snap     = max(c_L, c_{L+1})        # the COARSER of the two adjacent cells
```

**Why the coarser of the two.** The boundary separates ring `L` from ring `L+1`.
A block on the coarse side has size `c_{L+1}`; if the boundary is not a multiple
of `c_{L+1}` then a coarse block necessarily straddles it. Snapping to the finer
size would leave the coarse side still straddling. Snapping to the coarser makes
the boundary a lattice line for **both** sides, because `c_L` divides `c_{L+1}`
— guaranteed by `validate()`, which rejects non-integer ratios between
consecutive rings.

**`floor`, not `round`.** Rounding up would place the boundary beyond
`R_L · a(v)`, admitting ground into ring `L` that the anisotropy did not intend
and, worse, potentially past the ring's **fixed buffer half-width** — which is
the containment failure `ring_of`'s own docstring warns about at length ("the
index then wraps toroidally onto a cell on the far side of the map"). `floor` is
strictly conservative against that.

**Per-axis.** `a_f`, `a_s` and `a_r` differ, so the snap is per-axis: forward,
rear and lateral boundaries snap independently, each to the coarser adjacent
cell size.

---

## 3. Where to change it — and the trap

**Two functions implement this rule and both must change identically:**

| | file:line | role |
|---|---|---|
| `ring_of` | `src/grid/lattice.py:111` | reference, scalar + array |
| `ring_of_into` | `src/grid/lattice.py:346` | **allocation-free, on the frame path** |

`ring_of_into`'s own docstring states the contract: *"Every rule the reference
applies is applied here, in the same order and for the same reason."*

⚑ **This is the single highest-risk part of the change.** `ring_of_into` is
written against preallocated scratch lanes (`f0`, `f1`, `geom`, `mask`, `aux`)
under the hard "no allocation inside the frame loop" invariant. A nearest-corner
test needs the block's **extent**, not just a point — so it needs `c_L` per
candidate ring, which means either an extra scratch lane or restructuring the
`_count_at_or_below` comparison into a per-ring loop. **Do not add an allocation
here**: `tests/test_engine.py::test_the_two_grid_allocations_stay_fixed` and the
0.96 MB/frame budget in `reports/r9-per-stage-latency-and-memory.md` both pin it.

Also note the ordering constraints already encoded in `ring_of_into`, which the
fix must preserve: geometry first → `d_aniso` → containment (`max`) → rear floor
→ clamp → `OUTSIDE`. The snap belongs **inside the `d_aniso` comparison step**,
not after containment.

### Other call sites to check (not exhaustive — grep before starting)

- `bin_points` (`src/grid/lattice.py`) — consumes `ring_of_into`'s output; the
  research log records it as *"pinned bit-identical to `ring_of` + `i_ring` +
  `RingBuffer.flat_slot`"*, so its pinning test will fail loudly and correctly
  until the reference and the vectorised path agree again.
- `gate.ring_of_slot` (`src/grid/gate.py:134`) — inverse direction.
- `metrics._ring_cells` (`src/eval/metrics.py:161`) — since PR #31 this filters
  scored cells on `ring_of(centre) == ring`. **It tests the centre.** If the
  ring rule moves to nearest-corner, this must move with it or §9.2 will score
  a different population than the map serves. **This is a cross-lane
  consequence and it is easy to miss.**

---

## 4. Expected cost

- **Cell count: ~+0.1%.** Nearest-corner admits blocks that straddle inward;
  snapping moves each boundary outward by at most one coarse cell. Both are
  perimeter effects on a square buffer, so the increase scales with perimeter
  over area.
- **The memory bound does not move.** Rings are **preallocated at fixed
  half-widths** — `allocate()` sizes every ring from the schedule, not from
  occupancy. Nothing here changes `R_L`, `c_L` or the slot count, so
  **745,000 cells / 8.94 MB is untouched**, and no memory figure in the report
  needs recomputing. Say this explicitly when presenting the change; it looks
  like it should cost memory and it does not.
- **Latency:** the snap is `floor(R·a/c)·c` on ≤ 4 ring boundaries per frame,
  hoistable out of the per-point loop entirely — compute once per frame from
  `speed_ms` and reuse. The nearest-corner clamp is two extra `np.clip`-class
  operations per point. Against `bin` at 7.58 ms p50 this should be lost in the
  noise, **but measure it** rather than assuming.

---

## 5. The CI test

**Property:** for every pair of occupied cells at any two ring levels, neither
footprint contains the other.

```
for schedule in ("5/10/20/40", "5/10/50"):            # both frozen schedules
    for v in speeds:                                   # see sweep below
        occupy a scan-shaped spread of cells
        for (a, b) in pairs of occupied cells with ring(a) != ring(b):
            assert not footprint_contains(a, b)
            assert not footprint_contains(b, a)
```

`footprint(cell)` is `[i·k_L, (i+1)·k_L) × [j·k_L, (j+1)·k_L)` on the base
lattice — the same half-open interval `reference_map.block_stats` uses, so the
test and the metric agree about what a footprint is.

### The speed sweep is the point of the test

`a_f` **clamps at 2.0** (`lattice.py:44`), so it saturates at
`v = v_ref_ms / kappa_forward`. Sweeping past that adds nothing. The sweep must
cover, at minimum:

- `v = 0` — all three factors are exactly 1 and eq. (20) collapses to plain
  Chebyshev. This is the isotropic base case and it must still pass.
- the **clamp knee**, and either side of it.
- `v` where `a_f` saturates at 2.0.
- values chosen so `R_L · a_f(v)` lands **just above and just below** a
  multiple of `c_snap` — the adversarial cases. A uniform `linspace` will
  mostly miss these; **derive them from the schedule** by solving
  `R_L · a_f(v) = m · c_snap ± ε` for `v`.

⚑ A test that only samples round speeds (0, 5, 10, 15 m/s) can pass while the
bug is fully present. The failure is a measure-zero-ish set in `v`, and the
fraction of straddling blocks is what makes it *"a meaningful fraction"* rather
than rare — the boundary must be sampled deliberately, not stumbled onto.

### Also worth asserting

- **Coverage:** the union of occupied footprints has no gap at a ring boundary
  (the dual failure — snapping outward could in principle leave a strip
  unclaimed if the two sides snap differently). Partition means *neither*
  overlap *nor* gap.
- **Marker:** this is a partition property. `pytest.mark.partition` is already
  CI-blocking per CLAUDE.md, so it belongs there rather than in a plain test.

### It should fail before the fix

Worth confirming the test **reproduces the defect on current `main`** before the
fix lands. A property test that passes on the broken code is testing something
else.

---

## 6. Open questions for the implementer

1. **Does `metrics._ring_cells` move with it?** (§3). My reading is yes or §9.2
   silently scores a different population than the map serves. That is Aakash's
   call and it is cross-lane.
2. **Rear floor interaction.** `_rear_floor_ring` clamps the ring behind the
   vehicle within 50 m. Does the snapped boundary apply before or after that
   clamp? Current order is containment → floor → clamp. I believe the snap
   belongs with `d_aniso`, i.e. before all three, **but this is not obvious and
   I have not verified it.**
3. **`OUTSIDE` at the outermost boundary.** Snapping the last ring's boundary
   *inward* (floor) slightly shrinks the map. Points in the shaved strip become
   `OUTSIDE` rather than ring `N-1`. That is correct-by-construction but changes
   which returns are dropped — check it does not move the fill-rate numbers.
