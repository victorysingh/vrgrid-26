# Morning summary 3 — second pass, 2026-09-14

Covers only the brief that began *"Continuing tonight's session"*: D11/DL, `ground`, R-c,
R-i, OPEN-ITEMS. The p99 work before it is summarised in one line at the end.

All work is on local branch `jp/p99-alloc-fixes`. **Nothing pushed, no PR.**

---

## Needs your call

1. **D11 — the checkpoint location never arrived.** The brief's D11 line came with its
   placeholder unfilled (`[FILL THIS IN BEFORE SENDING …]`). I did not guess a path or fetch
   weights, so **none of the DL work ran**: no `--fast-scatter` verify, no `frnet_eval.py`
   90.3% / 65.2% check, no plan-regret comparison.
   **Also needs an explicit decision:** FRNet was closed in late August (ship with ground-truth
   SemanticKITTI labels, don't revisit unless you ask). If this brief is that ask, say so
   alongside the location.
2. **`ground` is deferred, not measured.** The gate check failed before any run. Commit charge
   was **17.33 GB against 15.73 GB physical** (2.81 GB free, clock fine at 2400/2400 MHz):
   the clean state from the pooled p99 run had lapsed. By process group: chrome (37 processes)
   4.2 GB, Code 2.5 GB, msedgewebview2 1.1 GB, Spotify 0.7 GB. Per your rule it was **not
   attempted degraded**. It needs the machine back under physical RAM, with state logged per
   run.
3. **R-c — apply or not.** The entry-point UTF-8 fix is staged and verified (below). It isn't
   applied because it touches all 20 scripts.
4. **R-i — unify the table printers or not.** Decision doc written, nothing implemented.

## What happened, by task

| Task | Outcome | Commit |
|---|---|---|
| D11 / DL | **Held.** Placeholder unfilled; FRNet-closed decision flagged. | `dcbf0ac` (OPEN-ITEMS D11) |
| 1 — `ground` | **Deferred.** Machine paging at the gate check; no measurement taken. | `dcbf0ac` (OPEN-ITEMS R-h) |
| 2 — R-c | **Staged, verified, not applied.** `pending-review/r-c-stdout-utf8-entry-points.{diff,md}` | `d0d84c3` |
| 3 — R-i | **Decision doc.** `pending-review/r-i-unify-timing-table-printers.md` | `d0d84c3` |
| 4 — OPEN-ITEMS | D11, R-h (ground), R-c and R-i rows updated with hashes. | `dcbf0ac`, `6e8d1c1` |

## R-c in detail

**Still not covered:** `frnet_fast_scatter.py` and `gen_demo_rrds.py` are patched but were
**not executed** (one runs a benchmark on start, the other writes `.rrd` files). Code outside
`scripts/` (`python -m vrgrid.run`, harnesses) doesn't get the fix, and `open()` still defaults
to cp1252.

**The change:** under each script's `if __name__ == "__main__":`, reconfigure stdout and stderr
to UTF-8, guarded by `hasattr` in case a stream is missing. It goes at the entry point, not at
import, because four test files import scripts under pytest's capture streams. There's no
helper module, so there's no `sys.path` dependency.

**Verified** (unpatched main tree against a patched git worktree, identical filenames):

- **Unchanged on a UTF-8 terminal:** 18 of 18 executed scripts byte-identical.
- **Crash fixed:** all 12 scripts that crashed on `--help` under cp1252 now exit 0, and their
  output equals the UTF-8 output.
- **Real default case:** with no forced encoding (locale cp1252, redirected stdout),
  `timing_table --help` goes from exit 1 / `UnicodeEncodeError` to exit 0.
- **Tests:** the five affected test files pass in the patched worktree (31 passed, 1 skipped).
  They locate scripts relative to their own file, so the patched copies were exercised.
  `ruff check scripts/` is clean.
- **Diff:** 20 files, 130 lines added, 0 removed. `git apply --check` is clean, including on a
  forced-CRLF copy, so a Windows checkout won't break it.
- `data_status` exits 1 in every arm, patched or not. That's its own data-presence result.

## R-i in brief

**Recommendation: unify the printers, keep `run()` and `run_real()` separate, and write
snapshot tests of both current tables first.**

The case for: the synthetic/real split is the structural cause of the 80.78 ms mislabelling
and the 8.15 → 1.31 MB mislabelling, and nearly caused a third (`--frame-times`, silently
ignored on one path until guarded in `7e58ad7`).

The case against, taken seriously in the doc:
- per-scope columns (`owner`, `share`)
- blank rows that could read as zeros
- `whole_frame_bench.py`'s row regex, which a layout change would break
- no existing tests on either printer
- the docs that quote the tables

A smaller fallback (label-only fix) is described, and I recommended against stopping there.

## One process note

One compound shell command failed on a quoting error while staging R-c. Bash rejected it before
running anything, so there was no partial state. I checked that (worktree intact, no stray
files or commits), then redid it with the note and commit messages written as files.

## Final git state

- Branch `jp/p99-alloc-fixes`, 17 commits since `main` including this summary.
- **No upstream; no remote-tracking branch contains any of these commits. Nothing pushed.**
- Working tree clean after this commit. No stashes; the temporary worktree was removed.
- Not touched: `ground.py`, `frnet/`, `lattice.py`, `include/vrgrid/`, `tests/test_metrics.py:472`,
  `engine.py`, D1/D2/D3/D6/D7/D10 implementation.

## Context: the p99 pass before this one

Pooled p99 **89.58 ms** over 1,000 frames on a trusted machine (`36a4dbe`). Three frames were
still over 100 ms, it's on one machine (D8 unagreed), and the two `engine.py` changes still
need Shrestha's review.
