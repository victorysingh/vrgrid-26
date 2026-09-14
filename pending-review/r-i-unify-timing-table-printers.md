# pending-review: R-i — should `timing_table.py` have one table printer or two?

**Status:** decision doc. **Nothing implemented** — this is JP's call.
**Written:** 2026-09-14, against `jp/p99-alloc-fixes` @ `dcbf0ac`.
**Supersedes the open question in:** `pending-review/front-end-allocation-instrumentation.md` §2.

---

## The question

Adding front-end allocation instrumentation is solved: `reports/harnesses/stage_allocation.py`
already does it by wrapping `Timer.stage`. The real question is **whether the synthetic and
real-data table printers should become one**, because that split produced both mislabelling
incidents.

Today `scripts/timing_table.py` has two separate paths from measurement to printed table:

| | synthetic (default) | real data (`--seq`) |
|---|---|---|
| runner | `run(frame, args)` | `run_real(args)` |
| printer | `print_table(t, alloc, frame)` | `print_real_table(t)` |
| stages timed | back end only: `MEASURED = (bin, scatter, fuse, cleanup, pyramid, shift)` | every stage in `timing.STAGES`, both halves |
| total row | `MEASURED` — a **lower bound**, not a frame | `FRAME` — the whole frame |
| columns | `owner`, `MB/frame` (with `--alloc`) | `share` |
| `--alloc` | supported | refused since `235986d` (it used to be silently ignored) |

## What the split has actually cost

1. **80.78 / 97.72 ms, "meets 10 Hz".** A synthetic back-end-only figure was quoted as a
   seq-08 whole-frame latency. It reproduced to 1.1% as a synthetic benchmark, was reconstructed
   in `295b7a6`, and was retired across every file that carried it in `5cddaa7`.
2. **"8.15 → 1.31 MB/frame, zero allocation in the frame loop."** `--alloc` was only
   implemented on the synthetic path and was *silently ignored* with `--seq`. So a back-end
   figure reached the README as a whole-frame claim. The loud-failure guard came in `235986d`.
3. **The same failure nearly happened a third time tonight.** `--frame-times` (`7e58ad7`)
   was written only on the `--seq` path, and on the synthetic path it was silently ignored
   until I added the same kind of guard in that commit. Nobody made a mistake that needed
   catching later; the structure invited it again within an hour.

The pattern is the argument: **every new feature lands on one path, and the other path
quietly does nothing.** The guards stop the silence, but they do it one flag at a time.

## The case FOR unifying the printers

- **It removes the structural cause instead of guarding its symptoms.** With one printer, a
  column like `MB/frame` or a flag like `--frame-times` is implemented once and appears for
  both scopes, or is visibly marked "not measured". No path silently lacks it.
- **Scope can be made impossible to miss.** A single printer can derive the label from *which
  stages produced samples*, not from which code path ran. If perception stages are missing,
  the total cannot be called `FRAME`. Both incidents were a back-end subtotal wearing a
  whole-frame label, and this makes that label unrepresentable.
- **One place to review.** Both incidents passed review because the synthetic table *looks*
  like the real one. One printer means one set of labels to check.
- **It is small.** The printers are about 80 lines together and called only from
  `timing_table.py`'s own `main()`. `ablation_table.py` uses `run_real`, not the printers, and
  no test calls either printer.

## The case AGAINST (what would be lost or complicated)

- **The synthetic table has columns that only make sense for it.** `owner` names the dev
  responsible for each back-end stage, and `share` only means something against a real frame
  total. A unified table either shows columns that are meaningless in one scope, or grows
  per-scope column logic that recreates the split inside one function.
- **Blank rows can look like zeros.** On the synthetic path the perception stages don't
  exist. If a unified table prints them, `range_image  —` sits one misread from `range_image 0`.
  "Not measured" has to be louder than a number, and getting that wrong recreates incident 1
  in a new form.
- **Scripts parse this output.** `reports/harnesses/whole_frame_bench.py` reads rows with
  `^(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s`. A unified layout that puts a text column (such
  as `owner`) between the stage name and the numbers breaks that regex. For `FRAME` it fails
  loudly ("printed no FRAME row"), but other stage rows would silently drop out of the JSON.
  Any change to the layout has to ship with the harness updated in the same commit.
- **No regression net exists.** There are no tests on either printer today. Unifying without
  first pinning the current output of both would be exactly the unverified change to this
  script that caused the last two incidents.
- **The docs quote the tables.** `docs/presentation/*` and `docs/research-log.md` quote
  figures, and in places layouts, from both tables. A changed layout means another pass over
  those files, and propagating corrections across them has been a multi-commit job every time
  tonight.

## Recommendation

**Unify the printers. Keep the two runners separate. Pin the current output with tests first.**

Concretely, in this order:

1. **Tests first, no behaviour change.** Snapshot tests for both current tables on a tiny
   deterministic run, plus a test that `whole_frame_bench.py`'s regex still extracts every row.
   This is the missing net, and it is worth having even if the answer below is "no".
2. **One printer, scope derived from the samples.** Print a mandatory first line,
   `SCOPE: whole frame (perception + map)` or
   `SCOPE: mapping back end only — NOT a frame latency`, decided by whether the perception
   stages have samples. Call the total row `FRAME` only in the first case; otherwise
   `BACK-END SUBTOTAL`. Print unmeasured stages as the word `not measured`, never as a blank
   or a zero.
3. **Keep `owner` on both, drop `share` from the synthetic scope.** Ownership is true in
   both scopes, and `share` is only computed where a real frame total exists.
4. **Do not merge `run()` and `run_real()`.** They measure genuinely different workloads (a
   synthetic sweep with `--points`/`--cells`/`--speed-mps` knobs versus a real sequence), and
   `ablation_table.py` depends on `run_real`'s return shape. The incidents came from the
   *printing*, not the running.
5. **Update `whole_frame_bench.py` in the same commit** as any layout change.

**If full unification is judged too much churn right now,** the smaller fallback is step 1
plus just the `SCOPE:` line and the `BACK-END SUBTOTAL` label on the existing synthetic
printer. That fixes the labelling (both incidents' failure mode), but not the structure (new
features still landing on one path). Given it nearly happened a third time tonight, I would
not stop there.

## What this doc does not do

It changes nothing. `scripts/timing_table.py`, the harness and the docs are untouched by it.
