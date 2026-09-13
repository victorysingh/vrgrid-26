# pending-review: `transform_points` allocates 10.86 MB per frame, and that is the p99 tail

**Status:** proposal, **nothing applied**. `src/perception/transforms.py` is
tracked, on the frame loop, and this is a performance change rather than a docs
correction — so it waits.
**Measurement:** `reports/r-b-p99-tail-investigation.md` §6.
**Written:** 2026-09-13, against `main` @ `92fc7d0`.

---

## 1. The finding

`transforms.transform_points` is the single largest identified contributor to the
whole-frame p99 tail — **84% of the frame's p99 excess** in a warm-cache run —
and the cause is allocation, not arithmetic.

Measured in isolation, pipeline and disk removed, same preloaded scans, 220
frames of ~123,000 points:

| arm | p50 | p99 | **max** | spikes >3× p50 | **allocated per call** |
|---|---|---|---|---|---|
| **as shipped** | 3.19 | 20.46 | **34.61** | **4 / 220** | **10.86 MB** |
| float64 input (no cast) | 2.34 | 4.92 | 32.86 | 2 / 220 | 7.90 MB |
| **preallocated probe** | **1.06** | **1.29** | **1.37** | **0 / 220** | **0.00 MB** |

**25× smaller tail, zero spikes, 3× better p50, 3.5× less total time.**

## 2. Where the 10.86 MB goes

```python
def transform_points(points, T):
    pts = np.asarray(points, dtype=np.float64)      # (1) float32 -> float64 copy
    xyz = pts[:, :3]                                #     view, free
    pts_h = np.hstack([xyz, np.ones((n, 1))])       # (2) np.ones  (3) hstack result
    return (T @ pts_h.T).T[:, :3]                   # (4) fresh (4, N) from the matmul
```

On 123,000 points, in float64: the input copy is ~3.9 MB, the `ones` column
~1.0 MB, the `hstack` result ~3.9 MB, the matmul output ~3.9 MB. Measured peak
with `tracemalloc`: **10.86 MB**, freed again every frame.

**Why that makes a tail and not a constant cost.** Most frames the allocator
returns pages it already holds and the cost is invisible. Occasionally it must
take fresh pages from the OS and first-touch faults them in — tens of
milliseconds, on an unpredictable subset of frames. That matches the observed
signature exactly: p50 unaffected, p99 8× worse, **different frames every run**,
and warm-up drift that plateaus.

The middle arm above is the useful control: feeding float64 removes only the
conversion copy, and p99 falls 20.46 → 4.92 — but `max` stays at 32.86, because
the remaining three allocations can still stall. **Removing one allocation is not
enough; the tail follows whichever one is left.**

## 3. [!] This is a gap in an invariant the project already enforces

`tests/test_engine.py::test_the_two_grid_allocations_stay_fixed` pins *"no
allocation inside the frame loop"*, and it passes. It covers `MapEngine.step` —
the grid half. **The perception half is not covered, and one function in it
allocates 10.86 MB per frame.**

The test is correct and the invariant is real; its **scope is narrower than its
name**. Whatever is decided about the code below, it is worth extending that test
to the perception path, because the same class of regression can reappear anywhere
in `iter_pipeline` and nothing currently notices.

## 4. The shape of a fix — and the two decisions it needs

The probe used one reused buffer set and `np.dot(..., out=)`:

```python
buf_h[:n, :3] = p[:, :3]                      # fill, no allocation
np.dot(buf_h[:n], T[:3, :].T, out=buf_out[:n])
return buf_out[:n]
```

That is a **probe, not a patch.** Two things must be decided first, and I am not
deciding them:

1. **Does the function still return a fresh array?** It currently returns a new
   object every call, and callers may reasonably retain it. Returning a view into
   a reused buffer means the *next* frame silently mutates data a caller kept —
   the worst kind of regression, because it is invisible until something
   downstream compares across frames. `iter_pipeline` passes `points_world`
   into `PerceptionFrame`, and whether anything holds that beyond the frame needs
   checking, not assuming.

2. **Where does the buffer live?** `transform_points` is stateless today and is
   called from more than one place (`iter_pipeline`, `eval/harness.real_scans`,
   `reference_map`). A module-level buffer makes it non-reentrant and adds exactly
   the kind of hidden shared state that D1 is currently teaching us about. The
   alternatives — an optional `out=` parameter, or a small caller-owned scratch
   object — are both more code but neither introduces a singleton.

**My recommendation** is an optional `out=` argument: callers on the frame loop
pass a buffer they own, everyone else keeps today's allocate-and-return
behaviour. It preserves the current contract by default, puts the buffer's
lifetime in the caller's hands, and avoids new module state. But it touches a
signature on a hot path and it is a real design choice.

## 5. What it buys, stated honestly

Frame p99 goes from **135.06 ms to roughly 113 ms** (from the dumped per-frame
arrays: removing the stage's contribution entirely gives 117.75, and the
preallocated arm lands in the same region).

**Against a 100 ms budget that is still over.** This is the largest single
identified contributor and the first with a mechanism rather than a correlation —
it is not a cure. The next candidates are `ground` (42% of the excess in the warm
run, and inside the C++ extension so harder to attribute) and `cleanup` (32%).

Do not let this be presented as "10 Hz fixed". It is one honest step.

## 6. Cheap partial, if the full change is not wanted now

Passing float64 in where the caller already has it removes the conversion copy
for free — `p99 20.46 -> 4.92`, no signature change, no buffer ownership
question. It leaves `max` at 32.86 so it does not fix the tail, but it is a
two-line change with no design consequences and it is strictly an improvement.

`iter_pipeline` already has `points` in hand and could pass
`points[:, :3].astype(np.float64)` once, reusing it for both the world transform
and anything else that wants float64 — though note that trades one allocation for
another unless it too is hoisted.
