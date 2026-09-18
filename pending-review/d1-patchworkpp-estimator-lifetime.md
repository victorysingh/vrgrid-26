# pending-review: D1 — who owns the Patchwork++ estimator's lifetime?

**Status:** decision doc. **Nothing implemented.** `src/perception/ground.py` is untouched, as
instructed. This is JP's call (OPEN-ITEMS lists D1 as *"JP (design)"* and as the highest-impact
item on the list).
**Written:** 2026-09-18, against `jp/p99-alloc-fixes` @ `95ce927` (post-merge).

---

## What is NOT the question

**The determinism symptom is fixed and this branch has the fix.** `3c26d47` (14 Sep, AakashH2006)
added `ground.reset_estimator()` at the entry points; the merge `ef4524e` brought it here, and
`tests/test_determinism.py` now passes 13/13 — including
`test_real_sequence_replay_is_identical`, which failed on this branch before the merge.

So nothing is broken today. **The question is whether a call-site convention is the design we want**,
because a convention holds only while every caller remembers it.

## The evidence that it is not holding

`ground.py` keeps one module-level `_estimator`, built lazily by `_get_estimator()` and cleared by
the public `reset_estimator()`. Counting how callers actually interact with that state, across the
whole tree:

| how the caller manages the estimator | count | who |
|---|---|---|
| public `reset_estimator()` | **5 calls, 5 files** | `src/run/__main__.py:117`, `src/eval/harness.py:366`, `scripts/feature_report.py:84`, `scripts/gen_demo_rrds.py:82`, `scripts/gpu_parity.py:76` |
| **assigns the private `ground._estimator` directly** | **13 assignments, 9 files** | 8 harnesses under `reports/harnesses/`, plus `scripts/plan_regret_frnet_delta.py:121` |

(Counts exclude `ground.py` itself and the tests.)

**More than twice as many assignments reach into the private global as there are calls to the public
API.** That is the finding. It is not carelessness: the public API cannot express what they need.

- `ground._estimator = None` (e.g. `cleanup_lut_equivalence.py:92`, `ring1_estimator.py:69`) is just
  `reset_estimator()` spelled privately — these could switch today.
- **`ground._estimator = _pw.patchworkpp(p)`** (`dump_costmaps.py:32`, `numiter_accuracy.py:58`,
  `numiter_r1_r7.py:56`, `numiter_r7.py:60`, `ring1_population.py:37`) installs a **differently
  configured** estimator — a different `num_iter`, for the R-7 / num-iter accuracy work. **There is
  no public way to do this at all.** `numiter_accuracy.py:19` documents the workaround in its own
  module docstring, which is how you can tell it is load-bearing rather than accidental.

So the real requirement is broader than "reset between runs": **some callers must supply their own
estimator.** Any design chosen here should be judged against that, not only against determinism.

## Three options

### A. Keep the module global; keep resetting at entry points (status quo)

Leave `ground.py` alone. Document the convention harder.

- **For:** zero code change and zero risk; it demonstrably works today (13/13); `segment_ground(points)`
  stays a one-argument function, which is why it reads well at every call site.
- **Against:** the trap is intact for the next caller who forgets — and the count above shows callers
  already route around the API rather than through it. The five public calls are load-bearing and
  nothing enforces them: **no test fails if a new entry point omits the reset**; the existing
  determinism test only covers the paths that already call it. Custom configuration stays a private-
  global poke forever.

### B. `segment_ground(points, estimator=None)` — caller may own it

Add an optional parameter. `None` keeps today's module-global behaviour; anything else is used as-is.

- **For:** smallest change that makes the *real* requirement expressible — the five num-iter harnesses
  stop poking a private name and pass their estimator explicitly. Fully backwards compatible: every
  current call site keeps working unchanged. Testable: a test can pass two independent estimators and
  assert they do not interfere.
- **Against:** it adds a parameter without removing the global, so both mechanisms exist at once and
  the trap is still reachable by anyone who passes nothing. Arguably the worst of both unless the
  global is later deprecated — which is a second decision, not this one.

### C. A `GroundSegmenter` object (or context manager) that owns the estimator

`seg = GroundSegmenter()` / `with ground.session() as seg:`; `seg.segment(points)`. The module-level
functions become thin wrappers over a default instance, or are removed.

- **For:** the lifetime becomes explicit and scoped — a run *is* an object, so "reset between runs"
  stops being something to remember and becomes something you cannot express wrongly. Custom
  configuration is just a constructor argument. This is the design the counts argue for.
- **Against:** the largest change, and it touches the seam the GPU port now runs through — upstream
  calls `reset_estimator()` inside `iter_pipeline`, and `gpu_parity.py` relies on ground running
  **once per scan with both paths receiving that one mask** (its docstring is explicit that running
  Patchwork++ twice would make the comparison about Patchwork++ rather than about the GPU). A
  lifetime redesign lands in Shrestha's active lane and would need coordinating, not just merging.
  It also breaks every existing call site at once.

## What I would recommend, and why it is still JP's call

**B, then C** — but only if the goal is to stop the private pokes; **A** is genuinely defensible if
the goal is to not disturb a working GPU lane mid-flight.

B is the change that matches the evidence (13 callers need injection, and B gives it to them) at a
cost of one optional parameter and no behavioural change. It can land without touching upstream's
seam. C is the better end state but should follow the GPU work, not race it.

**The reason this is not mine to decide:** all three are defensible, the trade-off is between API
honesty and disturbing an active lane, and that is a judgement about the team's next two weeks rather
than about the code.

## If anything is chosen, do this in the same commit

1. Add a test that **fails when an entry point forgets the reset** — today none does. Simplest form:
   assert that two `iter_pipeline` runs in one process hash identical maps, for each entry point, not
   only the one `test_real_sequence_replay_is_identical` covers.
2. Migrate the 13 private pokes. The 8 that are `= None` become the public call; the 5 that install a
   configured estimator are the ones that decide whether B or C is enough.
3. Leave `reset_estimator()` in place either way — it is upstream's convention now, in
   `iter_pipeline`, `harness.py`, `feature_report.py`, `gen_demo_rrds.py` and `gpu_parity.py`, and
   removing it is a separate, outward-facing change.
