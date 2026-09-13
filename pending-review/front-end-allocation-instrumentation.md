# pending-review: R-i — instrument front-end allocation

**Status:** proposal, **nothing applied**. Modifies `scripts/timing_table.py`.
**Why it exists:** `--alloc` covers the mapping back end only, which is how a
back-end figure became the README's whole-frame "zero allocation in the frame
loop" claim (`235986d`). The guard added there now *says* the front end is
uninstrumented; this is the work that would make that sentence unnecessary.
**Written:** 2026-09-13, against `main` @ `9b50111`.

---

## 1. The technique is already proven, not hypothetical

`reports/harnesses/stage_allocation.py` already measures per-stage allocation
across **both** halves of the pipeline on real data, and it does so without a
second list of stage names that could drift from `timing.STAGES`:

```python
_orig_stage = Timer.stage

@contextmanager
def _measuring_stage(self, name):
    before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    with _orig_stage(self, name):
        yield
    cur, peak = tracemalloc.get_traced_memory()
    alloc.setdefault(name, []).append((peak - before) / 1e6)
```

**`Timer.stage` is the single hook every stage boundary in both halves passes
through**, which is why this works where a per-function approach would not. It
produced the table in `reports/r-b-p99-tail-investigation.md` §9 — 59.40 MB/frame
whole-frame, with `bin` at 0.04 MB as the control showing the grid invariant
holding exactly where it is enforced.

So R-i is not "figure out how"; it is "move a proven ~10 lines into the shipping
tool, and decide how it prints".

## 2. What has to be decided, and it is not trivial

**The two table paths are the problem, and they are the same problem that
produced two mislabelling incidents.**

`timing_table.py` has two printers:

| path | function | columns | `--alloc` |
|---|---|---|---|
| synthetic | `main()` → `print_table(t, alloc, frame)` | `owner`, `MB/frame` | supported |
| real `--seq` | `run_real()` → `print_real_table(t)` | `share` | **not supported** |

The 80.78 ms latency figure and the 8.15 → 1.31 MB/frame allocation figure were
*both* back-end-only numbers from the synthetic path quoted as whole-frame. Adding
an `MB/frame` column to the real path is the fix — **and it is also the moment to
make the two tables impossible to confuse.**

Decisions for whoever takes it:

1. **Extend `print_real_table` with an `MB/frame` column**, or **unify the two
   printers**? Unifying is the better end state and the larger change; the
   `owner` column only makes sense on the synthetic path, and `share` only on the
   real one.
2. **Does `--alloc` on the real path print the `owner` column too?** If it does,
   the two tables converge and the confusion largely disappears.
3. **The header must say which is which.** The synthetic table already warns
   *"MEASURED is a LOWER BOUND on frame latency, not the frame total"*. The real
   table has no equivalent line and needs the converse: *this is the whole frame,
   both halves.*

## 3. [!] Two things the implementation must not get wrong

**`tracemalloc` roughly doubles runtime and smooths the latency tail.** Measured:
`transform`'s p99−p50 spread reads **0.81 ms** under `tracemalloc` against
**22.14 ms** without it. So the allocation pass must stay a **separate pass** from
the timing pass, exactly as `measure_alloc` already does on the synthetic path
(*"separate pass; tracemalloc distorts latency"*). **Never print allocation and
latency from the same run.**

**Restore `Timer.stage` in a `finally`.** The wrapper patches a class method; an
exception mid-run would leave every subsequent `Timer` in the process
instrumented, which would silently distort any later measurement in the same
session.

## 4. What it would have caught

Stated plainly, because it is the argument for doing it at all: with an
`MB/frame` column on the real path, *"zero allocation in the frame loop"* could
not have survived a single run of `timing_table.py --seq 08 --alloc`. The number
would have read ~59 MB/frame next to the claim.

That is the same class of defect as R-a's: **a figure nobody could check because
the tool that produces it does not measure the thing the claim is about.**

## 5. Not attempted tonight, and why

Two reasons, both worth stating rather than implying:

- **It touches the script at the centre of two mislabelling incidents**, and the
  right version of this change is partly a *table-unification* decision (§2),
  which is design rather than instrumentation.
- **The machine degraded during the session** — CPU downclocked to 1520/2400 MHz
  and ~8 GB over-committed (see `r-b-p99-tail-investigation.md` §11). Allocation
  counts would be unaffected, since `tracemalloc` counts bytes rather than time,
  but any verification run alongside them would not be trustworthy, and shipping
  a change to this particular script without a clean verification run is how the
  last two figures went wrong.
