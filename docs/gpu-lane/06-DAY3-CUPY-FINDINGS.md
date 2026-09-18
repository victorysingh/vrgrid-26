# Day 3 — the cupy seam, taken and measured

*Shrestha. Evidence for `03-CUDA-PORT-PLAN.md` §2 and §4. Everything here is
reproducible with `python scripts/bench_cupy_seam.py --all`.*

**Headline: the determinism claim is no longer an argument. It is measured, with
a control.** And the `bin_points` arithmetic runs ~60-70x faster on device,
bit-identical to the CPU result.

Two things stand between here and a merged port, both small, and one of them is
an ownership question rather than a technical one.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop, 8151 MiB, **sm_120** (Blackwell) |
| Driver | 610.57.04 |
| CUDA runtime | 12.9 (12090) |
| cupy | 14.2.0 (`cupy-cuda12x`) |
| CPU baseline | Intel i7-14650HX, numpy 2.5.3, Python 3.14.6 |
| SM clock | 180 MHz idle / 3090 MHz max |

⚠️ **This is a laptop, not the T4.** Per §7, a device number and a laptop number
are two columns, never one experiment. Everything below wants re-running on the
AWS instance before it goes near a slide. It is enough to make the Day-3
decision, which is what it is for.

⚠️ cupy 14.2.0 could not find the CUDA headers even with `cupy-cuda12x[ctk]`
installed; JIT fails with *"Failed to find CUDA headers"* until `CUDA_PATH`
points at `site-packages/nvidia/cuda_runtime`. Worth putting in the AWS runbook,
because the failure appears at first kernel launch, not at import.

## 1. Determinism on device — the claim, with a control

§2 calls this "the most valuable original systems claim our pair will have."
2,000,000 returns scattered into 4,096 cells — roughly 488 contending writes per
cell, far heavier than a real frame — repeated 30 times:

| Accumulator | Result |
|---|---|
| **int32** `scatter_add` | **30/30 runs bit-identical** |
| float32 `scatter_add` (control) | **29/29 runs differed from the first** |

The float32 row is what turns this from an assertion into a demonstration. Same
indices, same values, same card, same call — only the dtype changes, and the
reproducibility appears. That is the Day-0 fixed-point decision paying out
exactly as §2 predicts, and it is now a measurement anyone can re-run in about
four seconds.

**Say it with the control attached.** "Integer accumulation is deterministic" is
a claim about arithmetic. "We ran both and only one of them reproduced" is
evidence.

## 2. `bin_points` — 60-70x, bit-identical

The arithmetic chain from `grid/lattice.py:466-545`, allocation-free on both
sides as production is, 120,000 points, 200 iterations after 10 discarded:

| | p50 | p99 |
|---|---:|---:|
| CPU numpy | 7.2 - 8.4 ms | 8.6 - 11.8 ms |
| GPU cupy | **0.117 ms** | 0.157 - 0.315 ms |

**Speedup 61-73x on p50, and the outputs are bit-identical.**

The CPU side moves 7.2-8.4 ms between runs, so the ratio is a range and should
be quoted as one. The absolute device figure is stable at ~0.12 ms. For scale,
the port plan's stage table puts real `bin_points` at 6.84 ms p50, close enough
to the 7.2-8.4 ms measured here that the replica is representative.

§4 sets the bar: *"If cupy cannot win this one, stop and reconsider the whole
port."* It wins it comfortably. **Proceed.**

## 3. Two API gaps — the port is not a `np.` → `xp.` swap

`array_module()` is necessary and not sufficient. Probed on cupy 14.2.0:

| Call | cupy |
|---|---|
| `take(out=)`, `floor_divide(out=)`, `copyto(where=)`, `copyto(casting=)` | ok |
| `add.reduceat`, `minimum.at`, `argsort` | ok |
| **`take(..., mode="clip")`** | **rejected — `take()` has no `mode` argument at all** |
| **`subtract(..., out=, where=)`** | **rejected — ufuncs do not accept `where`** |

`bin_points` uses `mode="clip"` on all six of its takes and `where=` on two
subtractions, so both gaps are on the hot path. Neither is hard: `copyto(where=)`
works and covers the masked subtract, and the takes can drop `mode` **subject to
the trap below**.

## 4. The trap: cupy's `take` wraps where numpy's clip clamps

This is the finding that matters most, because it is silent.

| index into a length-4 table | numpy `mode="clip"` | cupy default |
|---|---|---|
| `9` (out of range) | `40` — clamped to the last element | **`20`** — wrapped |
| `-1` | `10` — clamped to the first | **`40`** — Python negative indexing |

`bin_points` clamps `lv` to `[0, n_rings)` immediately before every take, so no
index is ever out of range and the difference is unobservable **today**. Its
docstring already flags the fragility: *"if that clamp is ever removed, this
silently reads the wrong ring instead of raising."*

The port sharpens that warning. On CPU, removing the clamp gives a clamped read —
wrong, but bounded to a real ring. On device it gives a wrapped read, a
*different* wrong ring. So CPU and GPU would disagree only in the broken case,
which is the hardest kind of bug to find and precisely the failure shape §3 of
the float audit warned about for duplicate indices.

**The clamp must become an assertion, not a comment, before this ports.**

## 5. The blocker is ownership, not code

**`bin_points` lives in `src/grid/lattice.py`. That is Aakash's directory.**

This is collision ② from `01-THE-LANE.md` §2, arriving exactly where that
document predicted: the port plan assigns the work to us, and the code is not
ours. Per the same section's own recommendation — *"we should own R4 (the test)
outright and hand R3 (the fix) to the grid pair"* — I have not touched
`lattice.py`.

What is in this commit is entirely ours: the measurement, the benchmark script,
and this record. What needs a decision:

1. **Who applies the cupy changes to `bin_points`?** Either the grid pair takes
   the two rewrites, or we get explicit sign-off to touch `src/grid/`. Thirty
   minutes of conversation, and the alternative is discovering it on Day 5.
2. **Where does the clamp assertion go?** It belongs next to the takes, so
   whoever owns the file owns it.

## 6. What Day 3 hands to Day 4

- Determinism on device: **done, with a control.** Write it up for the report; it
  is the strongest systems claim in the project.
- `bin_points`: **measured and decided, not merged.** Blocked on §5 only.
- `scatter_sorted` is next by the §4 ordering, and its prerequisites are already
  green: `argsort`, `add.reduceat` and `minimum.at` all work, and the int32
  determinism result above is exactly the guarantee it needs.
- Then `visibility_cleanup` at 26.22 ms, which §4 calls the prize.
- Re-run all of this on the T4 before any of it is quoted.
