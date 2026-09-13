# pending-review: D10 STEP 1 — `query()` transcribed, line by line

**Status:** **prep work only.** STEP 1 of the revised D10 methodology: read and
transcribe. **No vectorisation written, no code changed, nothing implemented.**
**Written:** 2026-09-13, against `main` @ `2cfd487`.

---

## 0. The headline for whoever picks up STEP 2

**Most of the chain is already vectorised, with bit-identical pinning tests
already in the repo as precedent.** This is a much smaller and much safer job
than "write a second query implementation" implied:

| step in `query()` | vectorised form | status |
|---|---|---|
| `ring_of` | **`ring_of_into`** (`lattice.py:346`) | **exists** |
| `i_fine`, `i_ring` | same function | **already array-capable by construction** |
| `flat_slot` | **`flat_slot_into`** (`shift.py:148`) | **exists, and PINNED bit-identical** by `test_flat_slot_into_matches_flat_slot` across side, offset, dx, dy — and 28% faster |
| `occupancy_state` | same function | **already takes a `slots` batch**, plus `out=`/`scratch=` for allocation-free use |
| `unpack_class` | same function | **already array-capable** (`np.asarray` inside) |
| `_transient` | flag test + int16→float | trivially element-wise |
| **`_refined`** | — | **the only genuinely scalar step. This is the work.** |
| `CellQuery` return | — | a *contract* difference, not a logic one (§4) |

**And there is no reduction anywhere in the chain** — see §3, which is the direct
answer to the ULP question.

## 1. The exact sequence, for one cell

Transcribed from `src/grid/query.py:162` and its callees, in order. Not
paraphrased — each line is the operation as written.

### 1a. `query(gm, x_m, y_m)`

```
 1.  ring, slot = slot_of(gm, x_m, y_m)
 2.  IF ring == OUTSIDE:  return OUT_OF_MAP                      [early exit]
 3.  soa, slot = _refined(gm, ring, slot, x_m, y_m)               [may REBIND both]
 4.  ground  = float(soa["ground_height"][slot])  / 100.0
 5.  ceiling = float(soa["ceiling_height"][slot]) / 100.0
 6.  occ     = int(occupancy_state(soa, gm.thresholds, [slot])[0])
 7.  trav    = int(soa["traversability"][slot])
 8.  cls     = int(unpack_class(soa["semantic_class"][slot])[0])
 9.  n       = int(soa["obs_count"][slot])
10.  dynamic = False
11.  IF gm.transient is not None:
12.      t_occ, t_ground = _transient(gm, ring, slot)
13.      IF t_occ == OCC_OCCUPIED:
14.          occ = OCC_OCCUPIED ; dynamic = True ; ground = t_ground
15.  return CellQuery(ground, ceiling, cls, trav, n, occ, dynamic)
```

Note step 3 rebinds **both** `soa` and `slot` — the later reads may come from the
refinement pool's array, not `gm.soa`. That is the single most important fact in
this document for a bulk implementation (§4).

### 1b. `slot_of(gm, x_m, y_m)`

```
 1.  ring = ring_of(x_m, y_m, gm.schedule, gm.speed_ms)     [VEHICLE frame]
 2.  IF ring == OUTSIDE: return (OUTSIDE, -1)
 3.  c0 = gm.schedule.base_cell_m
 4.  k  = gm.schedule.k(ring)
 5.  ix = i_ring(x_m + gm.vehicle_xy_m[0], c0, k)            [WORLD frame]
 6.  iy = i_ring(y_m + gm.vehicle_xy_m[1], c0, k)            [WORLD frame]
 7.  slot = int(gm.buffers[ring].flat_slot(ix, iy))
 8.  return (ring, slot) if slot >= 0 else (OUTSIDE, -1)     [second OUTSIDE path]
```

**The two-frame split is load-bearing and is the documented trap.** Ring is
decided in the **vehicle** frame (foveation follows the vehicle); cell index in
the **world** frame (cell identity is world-anchored). `slot_of`'s docstring
records what happens if you index in the vehicle frame: nothing raises, the slot
is simply *some other place's cell*, and *"a map holding 143,000 observed cells
answers 'never seen' five metres ahead"*. A bulk version must keep `+
vehicle_xy_m` applied to the lattice index and **not** to the ring decision.

Also note **two distinct OUTSIDE paths** — step 2 (ring) and step 8 (window). A
vectorised form needs both masks, not one.

### 1c. `i_ring(x, c0, k)` → `i_fine(x, c0) // int(k)`

```
i_fine:  q = x // base_cell_m ; return int(q) if scalar else q.astype(int64)
i_ring:  validate k is a positive int ; return i_fine(x, c0) // int(k)
```

`i_fine` is **already dual-path by design**: *"Scalar in -> int out; ndarray in ->
int64 array out. Both paths are the same floor-division operator, so the
vectorised path (scatter) and the scalar path (query) cannot answer
differently."* Nothing to write.

**Do not "simplify" `i_ring` to `floor(x / (k*c0))`.** Its docstring: the two
agree in exact arithmetic but *"in IEEE-754 they are two lattices that drift
apart, and near a boundary a point falls in both cells or in neither. Derive,
never recompute."*

### 1d. `RingBuffer.flat_slot(ix, iy)`

```
s = self.slot(ix, iy) ; return np.where(s < 0, -1, s + self.offset)
```

Already written with `np.where` — array-safe as it stands. And `flat_slot_into`
is the allocation-free twin, **already pinned bit-identical**.

### 1e. `_refined(gm, ring, slot, x_m, y_m)` — the only real work

```
 1.  IF gm.pool is None:            return (gm.soa, slot)          [exit A]
 2.  block = gm.pool.find(ring, slot)
 3.  IF block < 0:                  return (gm.soa, slot)          [exit B]
 4.  levels     = int(gm.pool.levels[block])
 5.  child_ring = ring - levels
 6.  c0 = gm.schedule.base_cell_m
 7.  k_child  = gm.schedule.k(child_ring)
 8.  k_parent = gm.schedule.k(ring)
 9.  m = k_parent // k_child
10.  wx = x_m + gm.vehicle_xy_m[0]
11.  wy = y_m + gm.vehicle_xy_m[1]
12.  ox = i_ring(wx, c0, k_child) - i_ring(wx, c0, k_parent) * m
13.  oy = i_ring(wy, c0, k_child) - i_ring(wy, c0, k_parent) * m
14.  inner = int(oy) * m + int(ox)
15.  return (gm.pool.cells, gm.pool.block_cells(block).start + inner)
```

Steps 4–9 are **per-cell scalars derived from `block`**, so `levels`,
`child_ring`, `k_child`, `m` all vary per cell. That is what makes this the hard
step: it is not one formula over an array, it is a gather over a per-cell
`block`.

### 1f. `_transient(gm, ring, slot)`

```
1. IF slot >= gm.transient["log_odds"].size:      return (OCC_UNKNOWN, 0.0)
2. flags = int(gm.transient["flags"][slot])
3. IF not flags & FLAG_DYNAMIC:                   return (OCC_UNKNOWN, 0.0)
4. return (OCC_OCCUPIED, float(gm.transient["ground_height"][slot]) / 100.0)
```

Note step 1: the transient layer may be **shorter** than the grid, so the bounds
test is part of the logic, not a safety net.

## 2. Every branch, and what it must become

| # | branch | vectorised form | hazard |
|---|---|---|---|
| 1 | `ring == OUTSIDE` (ring test) | mask | — |
| 2 | `slot < 0` (window test) | mask | **a second, independent OUTSIDE mask** |
| 3 | `gm.pool is None` | whole-array early exit | cheap, and worth keeping as a fast path |
| 4 | `block < 0` | mask | see §4 — the eager-evaluation trap |
| 5 | `gm.transient is not None` | whole-array early exit | — |
| 6 | `slot >= transient.size` | mask | must precede any transient gather |
| 7 | `flags & FLAG_DYNAMIC` | mask | — |
| 8 | `t_occ == OCC_OCCUPIED` | `np.where` on 3 outputs | `occ`, `dynamic` **and** `ground` all change together — one mask, three writes |

## 3. [!] The ULP question, answered: there is no reduction in this chain

The `scatter_mean` 2-ULP difference came from a **reduction** — summing many
values into one slot, where numpy's pairwise summation orders differently from a
scalar loop. **`query()` performs no reduction of any kind.** Every operation is
either:

- **exact integer arithmetic** — `//` floor division (`i_fine`, `i_ring`, `m`),
  `*`, `+`, `-` on int64; `>>` and `&` in `unpack_class`
- **element-wise float division by a constant** — `int16 / 100.0`, one operation
  per element, no accumulation
- **comparison / selection** — `occupancy_state`'s threshold tests, the masks above
- **a gather** — `array[slot]`

Divisions in `ring_of`'s `d_aniso = max(x⁺/a_f, x⁻/a_r, |y|/a_s)` are element-wise
with a `max` — a selection, not a sum.

**So a correct vectorisation should be bit-identical by construction, not by
luck.** That is a materially different risk position from the scatter work, and
it is worth stating to the room, because "second implementation" sounds like it
carries scatter's risk and it does not.

**The one caveat.** `float(int16)/100.0` in a loop versus `int16_array / 100.0`
vectorised: numpy may compute the array form in float32 if the input is int16 and
the scalar form promotes to Python float (float64). **Force float64 explicitly**
in the bulk path. This is a dtype-promotion difference, not an ordering one, and
it is the single most likely cause of a first-attempt mismatch.

## 4. [!] The trap: `np.where` is eager

`np.where(cond, a, b)` evaluates **both** `a` and `b` everywhere. The scalar code
relies on branches to *avoid* computing things:

- Step 1e.15 indexes `gm.pool.cells` and calls `block_cells(block)` **only when
  `block >= 0`**. Vectorised naively, `block_cells` is called for `block == -1`
  too, and `.start` for a non-existent block is either an exception or — worse —
  a plausible wrong index.
- Step 1f.2 reads `gm.transient["flags"][slot]` **only after** the bounds test.

**Mask before you gather, never gather then mask.** Every index used in a gather
must be clamped or filtered first, and the result overwritten under the mask
afterwards. The blind-cone rule applies to code as well as to cells: an invalid
index that produces a plausible number is the failure to design against.

## 5. The contract question, which is not a logic question

`query()` returns a `CellQuery` **dataclass** per cell. A bulk reader cannot
return 160,000 dataclasses — that is most of the 54.7 µs per call. It must return
**arrays**, one per field (a struct-of-arrays `CellQueryBatch`, which is also what
`grid_map` layers want).

So the pinning test compares **field arrays against a loop of `CellQuery`
objects**, not object against object. That is straightforward but it must be
written that way from the start, and it means the bulk function is *not* a drop-in
replacement for `query_region` — it is a sibling with a different return type.
**That is a signature decision and it belongs with D10's frozen-signature
question.**

## 6. What STEP 2 actually has to write

Given §0, the genuinely new code is small:

1. **A batched `_refined`** — the only real algorithm work. A gather over per-cell
   `block`, with `levels`/`k_child`/`m` varying per cell, and the eager-evaluation
   discipline of §4.
2. **Composition** of `ring_of_into`, `i_ring`, `flat_slot_into`,
   `occupancy_state(slots=...)`, `unpack_class` — all of which already exist and
   three of which already have pinning tests.
3. **A batched `_transient`** — masks and a gather.
4. **The SoA return type** (§5).

**Estimate shifts accordingly.** The earlier framing — "a second implementation of
query semantics, which is precisely the class of bug this project keeps designing
out" — is still the right caution for `_refined`, and is **overstated for the rest
of the chain**, which is reuse rather than reimplementation.

## 7. If the pinning test does not pass first time, what it probably means

Per STEP 3, a failure is information, not a debugging chore. Ranked by likelihood
from this read:

1. **float32/float64 promotion** on the `/100.0` height conversions (§3).
2. **The two OUTSIDE paths** conflated into one mask (§1b) — the ring test and the
   window test are independent and a point can pass the first and fail the second.
3. **`_refined` gathering before masking** (§4), giving a plausible wrong slot for
   `block < 0` cells rather than an error.
4. **The transient bounds test** dropped as if it were a safety net rather than
   logic (§1f.1).
5. **Something genuinely non-obvious in `pool.find`** — the one function in the
   chain I have read the *callers* of but not the internals. If a mismatch
   survives 1–4, look there first, and treat it as the finding rather than as an
   obstacle.
