# The float audit — every reduction classified

*Shrestha. Closes item 1 of `03-CUDA-PORT-PLAN.md` §9, which §3 of that document
requires **before any port code is written**: "do it before Day 3, not after."*

**Verdict: no blocking issue. The port can proceed.** One site needs a test
rather than an assumption, and one code comment states a premise that is wrong
in a way that matters. Neither stops Day 3.

The rule being applied, from `03-CUDA-PORT-PLAN.md` §3: an *elementwise* float op
is deterministic on device, because each output depends on one input and no
ordering exists. A float *reduction over a variable number of contributors* is
not, because the GPU chooses the order. So the question at every site is: **does
this sum over a set whose order the device controls?**

## The table

| # | Site | File | Accumulator | Reduction? | Device-safe |
|---|---|---|---|---|---|
| 1 | height sum | `gpu/kernels.py:366` `add.reduceat` | **int64** | variable-contributor | ✅ integer add is associative |
| 2 | weight sum | `gpu/kernels.py:367` | int32/int64 | variable | ✅ |
| 3 | reflectivity sum | `gpu/kernels.py:368` | int32 | variable | ✅ |
| 4 | ceiling | `gpu/kernels.py:373` `minimum.reduceat` | int16 | variable | ✅ min is associative **and** commutative at any dtype |
| 5 | atomic height | `gpu/kernels.py:427` `add.at` | int64 | variable | ✅ |
| 6 | atomic weight / n / refl | `gpu/kernels.py:428-430` | int32 | variable | ✅ |
| 7 | atomic ceiling / class | `gpu/kernels.py:431-433` `minimum.at` | int | variable | ✅ as #4 |
| 8 | `mean_height_cm()` | `gpu/kernels.py:166` | integer division | elementwise | ✅ rounds half-away-from-zero **in integers**, so no float rounding mode enters |
| 9 | Kalman filter | `grid/fusion.py:193-231` | **float64** | **elementwise** on already-reduced inputs | ✅ see below |
| 10 | scatter-write of results | `grid/fusion.py:230-234` | — | duplicate-index hazard | ✅ verified unique, see below |
| 11 | law of total variance | `grid/splitmerge.py:280-282` `.sum(axis=-1)` | **float64** | **fixed-arity (4), last axis** | ⚠️ the one weak link |
| 12 | pyramid max/min | `gpu/pyramid.py` | int16 | associative | ✅ |
| 13 | `in_view.sum()`, `keep.sum()` | `gpu/visibility.py:303`, `gpu/baseline.py:141,172` | bool→int | count | ✅ integer, and diagnostic only |

## The three findings worth reading

### 1. The Kalman filter's float64 is genuinely safe, and the file already says why

`fusion.py:26` states it: *"What is non-associative is summation order, and every
sum has already happened in integers by the time a CellAggregate exists."*

That is correct and it is the crux of the whole port. Every float64 operation in
`fuse()` — `gain`, `post_mu`, `post_var` — is elementwise over a slot vector. The
reductions all happened upstream in `scatter_sorted`/`scatter_atomic`, in
integers. So identical inputs give bit-identical outputs on any device, and the
results land back in int16 cm and one uint8 code by fixed rounding rules.

**No action.** This is the largest float surface in the pipeline and it is clean.

### 2. The duplicate-index hazard does not apply — but only because of an invariant worth keeping

`fusion.py:230-234` writes results back with `soa["ground_height"][slots] = ...`.
On CPU, numpy resolves duplicate indices last-writer-wins in index order. **On
device, `cupy`'s `__setitem__` with duplicate indices has undefined order**, so if
`slots` could ever repeat a slot, this line would be nondeterministic on GPU and
perfectly deterministic on CPU — the worst possible failure shape.

It cannot, and both paths say so explicitly:

- `kernels.py:158` — `self.cells = cells  # int64, sorted, unique`
- `kernels.py:436` — `cells = np.flatnonzero(acc["n"])`, unique by construction

**Action: none now, but this invariant is load-bearing for the port in a way it
was not for the CPU code.** If anyone ever makes `cells` non-unique, CPU stays
green and device goes silently wrong. Worth a line in the port's test.

### 3. The law-of-total-variance merge is the only site that needs a test

```python
mu_p    = (w * mu).sum(axis=-1)
within  = (w * sigma2).sum(axis=-1)
between = (w * (mu - mu_p[..., None]) ** 2).sum(axis=-1)
```

These are float64 reductions. **But they are not the dangerous kind**: the axis is
the four children, fixed-arity and contiguous, not a variable set of contributors
funnelled through atomics. cupy reduces a size-4 last axis in a fixed order, so in
practice this is deterministic.

"In practice" is the problem. Every other site on this list is guaranteed by
associativity of integer addition — a property of the arithmetic. This one is
guaranteed by an implementation detail of cupy's reduction strategy for small
axes, which is a weaker promise and not one cupy documents.

**Action: pin it with a test rather than assume it.** Merge the same four children
a few hundred times on device and assert bit-identity. Cheap, and it converts an
assumption into a gate. Options (a) fixed-point and (c) sort-first from the port
plan are both available if it ever fails, but neither is worth doing pre-emptively.

## The overflow bound, measured rather than reasoned

`03-CUDA-PORT-PLAN.md` §9 asks for this "asserted in code, not just reasoned", and
§2 warns that int32 saturation is silent. The plan's own §2 arithmetic is built on
a premise that turns out to be wrong, so here is the measured version.

**The premise.** `kernels.py:503` justifies the accumulator widths with *"w_q up to
2^20"*. `WEIGHT_MAX` is indeed `1 << 20`, but that is a clip ceiling, not a
reachable value. Weight is `1024 / sigma_z^2`, and `measurement_variance_cm2()`
has a floor: the near-field term `(h_s/r)^2 sigma_r^2` blows up as r→0 and the
range term `r^2 sigma_phi^2` grows as r→∞, so variance bottoms out in between.

```
minimum variance   = 1.208 cm^2 at r = 4.45 m   (cos_incidence = 1.0, best case)
maximum weight     = 848
WEIGHT_MAX         = 1048576   -- 1236x above anything achievable
```

Reasoning from 2^20 gives "int32 `w_sum` overflows at 2048 returns per cell",
which looks alarming. Reasoning from 848 gives 2.53 million. The clip never binds.

**Measured, 30 frames of sequence 08 through the production `MapEngine` path:**

| Quantity | Peak observed | int32 headroom |
|---|---|---:|
| returns in one cell | 123 | — |
| `w_sum` | 16,739 | **128,292x** |
| \|`wz_sum`\| | 1,574,976 | 1,363x (stored int64) |

So there is no overflow risk at any site, by five orders of magnitude on `w_sum`
and three on `wz_sum`. `wz_sum` would in fact fit int32 at observed loads; int64
is the right call anyway, because the bound that matters is the adversarial one
(every return in a frame landing in one cell), not the observed one.

⚠️ Measured with the **semantic-class ground fallback**, not Patchwork++, which
was not installed in the measuring environment. The fallback marks *more* points
as ground, so it inflates `w_sum` — the direction that makes this bound
conservative. Re-run with Patchwork++ before quoting these as final.

**One inconsistency to note, not a defect.** `w_sum` is `int64` in the sorted path
(`kernels.py:479`) and `int32` in the atomic path (`kernels.py:507`), for the same
quantity. Both are safe by the numbers above. Worth unifying so the port does not
have to reason about two widths.

## What Day 3 should carry forward

1. Port `bin_points` first, as §4 says. Nothing in this audit touches it — it is
   pure elementwise integer division and clipping.
2. When `scatter_sorted` goes to device, assert `cells` is unique in the test, not
   just in a comment.
3. Add the merge bit-identity test before `splitmerge` goes anywhere near cupy.
4. Fix the `2^20` premise in `kernels.py:503` when someone is next in that file,
   and unify the two `w_sum` widths.
