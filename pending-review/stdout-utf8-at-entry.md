# pending-review: R-c — make stdout able to carry non-cp1252 text, once

**Status:** proposal, **nothing applied**. Modifies tracked files, so it waits.
**Recommendation:** **Candidate A — `PYTHONUTF8=1`.** Zero source edits, and it is
the direction CPython itself is going.
**Written:** 2026-09-13, against `main` @ `3a43c19`.

---

## 0. It is much bigger than "timing_table.py twice"

Measured, not assumed:

| | |
|---|---|
| scripts containing a character cp1252 cannot encode | **15 of 20** |
| scripts that **crash on `--help` right now** | **12 of 20** |

The crashing twelve: `ablation_table`, `build_reference_map`, `data_status`,
`eval_synthetic`, `feature_report`, `frnet_eval`, `frnet_finetune`,
`ghost_removal_figure`, `measure_visibility_cap`, `plan_query_survey`,
`regret_plot`, `timing_table`.

**`--help` is the worst possible place for this.** `argparse` prints the module
docstring as its description, so the failure hits the one command a new user runs
first, on more than half the tooling, with a traceback rather than a message. The
two `print()` sites fixed in `f3a0337` were a symptom, not the bug.

The offending characters are almost entirely **U+2691** (⚑, the house "watch out"
marker), plus **U+2714** (✔) in two files and **U+26A1** (⚡) in one.

> A note on how this was scoped: the script I wrote to *count* the affected files
> crashed on the bug it was counting, at `print()` of its own results. That is the
> most concise argument for fixing it centrally rather than per call site.

## 1. The obvious one-point fix is blocked

Every one of these scripts imports `vrgrid.*`, so the natural home for a single
fix is the package root — which is **`include/vrgrid/__init__.py`**, the
**frozen** interface directory. CLAUDE.md: *whole-team change only, never edit
unilaterally*, and CODEOWNERS routes it to all three devs.

So a two-line addition in the one correct place costs three signatures. That is
the constraint shaping both candidates below, and it is worth knowing that the
freeze is what makes this awkward rather than any property of the problem.

## 2. Candidate A — `PYTHONUTF8=1` (recommended)

Python's UTF-8 mode. **No source edits at all.**

```sh
PYTHONUTF8=1 python scripts/timing_table.py --help
```

**Verified:** fixes 4 of 4 sampled crashing scripts, including `timing_table.py`
and `ablation_table.py`.

Where it would be set — and these are the only files that change:

| file | change |
|---|---|
| `scripts/demo.sh` | `export PYTHONUTF8=1` beside the existing `VRGRID_DATA_ROOT` export |
| `.github/workflows/ci.yml` | `PYTHONUTF8: "1"` in the workflow `env:` block |
| `docs/demo-runbook.md` | one line in the setup section |

**Why this is the right answer, not just the cheap one:**

- **It is where CPython is going.** UTF-8 mode becomes the default in 3.15
  (PEP 686), so this is a temporary shim that expires on its own rather than code
  to carry forever.
- **It fixes every script at once**, including the five not yet affected and any
  written later. A per-file import fixes exactly the files someone remembers.
- **No `sys.path` assumption** — see Candidate B's footgun.
- It also fixes `open()` defaulting to cp1252, which is a *latent* version of the
  same bug anywhere the repo reads or writes text without an explicit `encoding=`.

**Honest downside:** it is environment, not code, so a developer invoking a script
directly without the variable set still crashes. Mitigation is the runbook line
plus the CI setting; the real removal of the footgun is 3.15.

## 3. Candidate B — a helper module imported per script

```python
# scripts/_console.py
"""Make stdout/stderr able to carry non-cp1252 text. Import before anything prints."""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
```

plus one line per script, after the shebang and docstring:

```python
import _console  # noqa: F401  -- stdout must accept non-cp1252 text
```

**Verified on a copy of `timing_table.py`:** `--help` goes from exit 1 with a
`UnicodeEncodeError` to exit 0, output intact.

**[!] Two real problems with it:**

1. **It relies on `scripts/` being `sys.path[0]`.** True for
   `python scripts/x.py`, which is how the docs invoke everything — and false for
   `python -m scripts.x`, for an editor's "run file" from a different cwd, and for
   anything importing a script as a module. The failure would be an
   `ImportError` on a helper whose entire purpose is to prevent a crash.
2. **20 insertion points, at two different offsets.** 18 scripts open with
   `#!/usr/bin/env python3` and 2 with a bare docstring, so the edit is not
   uniform. `noqa: F401` is needed on every one, and RUF100 on the newer CI ruff
   will flag it the moment the import is genuinely used for a side effect it can
   see.

## 4. Verification of the mechanism itself

The requirement was that it fix the crash and change nothing else. Tested with
**identical filenames** in both arms, because `argparse` prints the script name
into `usage:` and a renamed copy produces a spurious diff:

| stdout encoding | unpatched | patched | output |
|---|---|---|---|
| **UTF-8** (`PYTHONIOENCODING=utf-8`) | exit 0 | exit 0 | **byte-identical** |
| **cp1252** (this console's default) | **exit 1, `UnicodeEncodeError`** | **exit 0** | identical to the UTF-8 output |

Two things that matter in that table:

- **On a UTF-8-capable terminal the fix is a genuine no-op** — byte-identical
  output, not merely "looks the same".
- **On cp1252 the patched output equals the UTF-8 output**, so the real ⚑ is
  emitted rather than `errors="replace"` degrading it to `?`. The `replace`
  fallback only engages if reconfiguration itself fails.

## 5. What I recommend

1. **Take Candidate A.** Three one-line additions, no source churn, expires with
   Python 3.15.
2. **Do not chase the glyphs.** Replacing ⚑ with `[!]` across 15 files would also
   work and is tempting, but it is 15 files of churn to work around an encoding
   default, and it loses a marker the codebase uses deliberately and
   consistently.
3. **Worth raising at the next three-way:** whether
   `include/vrgrid/__init__.py` should carry the two-line reconfigure. That is the
   only genuinely central fix, it is two lines, and the freeze is the sole reason
   it is not the obvious answer. A frozen *interface* arguably should not be
   frozen against its own process setup.

## 6. What I did not do

Nothing is applied. No script, `demo.sh`, CI workflow or doc was modified — the
helper above exists only in this document and in a scratch copy used for the
verification in §4.
