# pending-review: `cleanup`'s allocation is `np.isin` in an argument — 21× and 3.9× available

**Status:** proposal, **nothing applied**. `src/run/engine.py` is Shrestha's
(`"""The map back end, as one frame loop. [Shrestha]"""`) and on the frame loop.
**Measurement:** `reports/r-b-p99-tail-investigation.md` §10, harness
`reports/harnesses/cleanup_alloc.py`.
**Written:** 2026-09-13, against `main` @ `8b40e44`.

---

## 1. What this does NOT fix — first, because it matters

**This does not get the frame to 10 Hz, and it is not the tail's main cause.**
`cleanup` is the *third* contributor (32% of the warm-run p99 excess) behind
`transform` (84%) and `ground` (42%). Fixing all three still leaves p99 over
budget on the measurements available. And `ground` — which has the **largest**
single spread — is inside the Patchwork++ C++ extension and none of this applies
to it.

What this is: a **~4.2 ms p50 and ~5.7 ms p99 saving in one line**, with
bit-identical output, on a stage that was written to avoid allocation and does
not.

## 2. The finding: careful code, defeated by its own argument

`MapEngine._cleanup` is written allocation-consciously — `out=`, `scratch=`,
`np.copyto`, preallocated `_cand`, `_cand_slots`, `_has_return`. And
`visibility_cleanup`'s docstring promises *"no allocation when handed a scratch"*,
which the engine does hand it. So the stage's 9.61 MB/frame is **not** in the
eq. (32) pass at all — it is in the three lines of candidate selection before the
call.

Measured on a real 30-frame seq-08 map (910,000 slots, 382,345 occupied, 42%):

| operation | peak alloc | p50 | p99 |
|---|---|---|---|
| `state == OCC_OCCUPIED` | 0.91 MB | 0.115 ms | 0.338 ms |
| `np.flatnonzero(mask)` | 3.97 MB | 2.333 ms | 2.632 ms |
| **`np.isin(occupied, touched)`** | **8.18 MB** | **5.677 ms** | **7.711 ms** |

*(Summed separately they exceed the 9.61 MB the stage measures as a peak, because
the three do not all peak simultaneously.)*

**`np.isin` dominates both columns**, and it sits here:

```python
guard = self._has_return[:m]
np.copyto(guard, np.isin(occupied, touched))
```

The `np.copyto` into a preallocated `guard` is exactly right. **The allocation is
in the argument being copied from** — `np.isin` builds an 8.18 MB temporary, and
sorts internally to do it, before a single byte is copied. The preallocation is
defeated by the expression feeding it.

## 3. The fix: a boolean lookup table

Both `occupied` and `touched` are index arrays into the same slot space, so
membership is a lookup, not a search:

```python
self._touched_lut[:] = False          # persistent, allocated once at startup
self._touched_lut[touched] = True
np.copyto(guard, self._touched_lut[occupied])
```

Measured, same data, 30 reps:

| variant | alloc | p50 | p99 | output identical to `np.isin`? |
|---|---|---|---|---|
| `np.isin` (shipped) | 8.18 MB | 5.677 ms | 7.711 ms | — |
| **boolean LUT** | **0.38 MB** | **1.472 ms** | **2.059 ms** | **yes, `np.array_equal` True** |
| LUT + `np.take(out=)` | 0.38 MB | 3.633 ms | 10.656 ms | yes |

**21× less allocation, 3.9× faster p50, 3.7× better p99, bit-identical output.**

The LUT costs one persistent `bool` array of `allocated_slots` — **910,000 bytes,
allocated once**, which is the same discipline as every other scratch in
`allocate()` and does not grow with frame count.

### [!] Two things that did NOT work, recorded so nobody retries them

- **`np.take(lut, occupied, out=guard_buf)` is *slower* than plain fancy
  indexing** — 3.633 ms against 1.472. The "obvious" further optimisation
  backfires, so the simple form is also the fast one.
- **A `searchsorted`-based membership test was a dead end.** My first probe ran
  51 ms, but it was also **not a correct membership test** — it did
  `searchsorted` + `clip` without the final equality check, so it was measuring
  the wrong thing *and* was slow. Stated because a half-correct probe that looks
  conclusive is worse than none; the LUT is the right structure here and
  `searchsorted` should not be revisited on the strength of that number.

## 4. The remaining 0.38 MB

That is `self._touched_lut[occupied]` — 382,345 booleans, the intermediate the
`np.copyto` reads. Removing it needs a genuine `out=` path for boolean fancy
indexing, and `np.take(out=)` is the obvious candidate and is slower (§3). **I
would leave it.** 0.38 MB against 8.18 is where the value is; chasing the last
sliver costs clarity for ~0.3 ms.

## 5. Decisions for the owner

1. **Where the LUT lives.** `allocate()` alongside the other scratches is the
   consistent answer, sized from `allocated_slots`. That touches the allocator,
   which is the file the memory bound is a claim about — so the 8.94 MB figure
   and the "two grid allocations" test both need checking, even though a
   910,000-byte bool array is not part of the 12-bytes-per-cell footprint. **How
   it is accounted for is the owner's call**, and it should be stated explicitly
   rather than absorbed silently.
2. **Whether `np.flatnonzero` is worth touching too** (3.97 MB, 2.33 ms). `np.equal(..., out=)`
   removes the mask allocation for free (0.91 → 0.00 MB, measured), but
   **`np.nonzero` has no `out=` in numpy**, so the index array itself needs a
   persistent buffer plus a manual fill. That is a real code change for ~2 ms and
   4 MB, and it is a separate decision from the guard.
3. **This is Shrestha's file.** The change is nine lines in `_cleanup` plus one
   buffer, but `engine.py` explicitly *"owns the ORDER of the map stages and
   nothing else"* — a LUT is arguably state that belongs in the allocator, not
   here, which is decision 1 restated from his side.
