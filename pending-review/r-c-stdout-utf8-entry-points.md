# pending-review: R-c — UTF-8 stdout at every script's entry point

**Status:** staged, **not applied**. The change is `r-c-stdout-utf8-entry-points.diff`
(20 scripts). Apply with `git apply pending-review/r-c-stdout-utf8-entry-points.diff`.
**Written:** 2026-09-14, against `jp/p99-alloc-fixes` @ `dcbf0ac`.
**Relation to `stdout-utf8-at-entry.md`:** that doc recommended `PYTHONUTF8=1` (environment,
no source edits). JP chose an in-code fix at each entry point. This is that fix, built so it
avoids both problems that doc found with its helper-module version (Candidate B).

---

## Still not covered

- ~~**`frnet_fast_scatter.py` and `gen_demo_rrds.py` are patched but were not executed.**~~
  **[2026-09-18] CLOSED — both executed, patched and unpatched, in a throwaway worktree.**
  `gen_demo_rrds.py` (with a filter matching no shot, so it exercises the print path without
  writing 4.7 GB): stdout **byte-identical**, exit 0 both ways. `frnet_fast_scatter.py`
  standalone (the equivalence check plus the benchmark): identical once the benchmark's **own**
  ms/speedup figures are normalised — those vary run to run (`scatter_max` 3655.2 vs 3928.5 ms)
  and are the only differing lines; the verification text matches exactly. **So the evidence is
  now 20 of 20 executed, not 18 of 18.** Neither crashes under `PYTHONIOENCODING=cp1252 --help`
  patched or unpatched, so neither was among the 12 crash fixes.
- **[2026-09-18] NEW GAP, opened by the upstream merge: the diff covers 20 of the tree's 27
  `scripts/` entry points.** The seven it does not touch arrived with `ef4524e` or were written
  after the diff was staged: `bench_cupy_seam.py`, `engine_eval.py`, `frnet_eval_by_range.py`,
  `gpu_parity.py`, `plan_regret_frnet_delta.py`, `r9_stages.py`, `vram_contention.py`. Tested
  under `PYTHONIOENCODING=cp1252 --help`: **`bench_cupy_seam.py` and `vram_contention.py` crash**
  (`UnicodeEncodeError` on `'⚑'`, the ⚑ flag character), the other five are clean. Both
  crashing scripts are GPU-lane files and belong to Shrestha's lane, so they are reported here
  rather than patched. Whatever is decided for R-c should cover them.
- **Code outside `scripts/` is not covered.** `python -m vrgrid.run` and the harnesses under
  `reports/harnesses/` do not get this. The only truly central fix is still
  `include/vrgrid/__init__.py`, which is frozen (D3).
- **`open()` still defaults to cp1252** without an explicit `encoding=`. That latent issue is
  something `PYTHONUTF8=1` would fix and this does not.

## The change

Inserted directly under each script's `if __name__ == "__main__":`, before `main()` runs:

```python
if __name__ == "__main__":
    # R-c: docstrings and tables here print non-cp1252 markers, which crash a
    # cp1252 console (--help included). Reconfigured at the entry point, not at
    # import, because tests import these modules under pytest's capture streams.
    import sys                                   # only in the 10 scripts lacking it
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
    main()
```

- **One pattern, applied to every script by a script.** It isn't a per-`print()` patch; the
  printed text is untouched.
- **At the entry point, not at import.** `tests/test_ablation_table.py`,
  `test_plan_query_survey.py`, `test_regret_plot.py` and `test_build_reference_map.py` import
  these modules, and pytest swaps `sys.stdout` for its own capture object. Reconfiguring at
  import would reach into that.
- **No helper module.** Candidate B's `import _console` needed `scripts/` on `sys.path` and
  broke under `python -m`. There is nothing to import here.
- **`hasattr` guard.** Under `pythonw` or an unusual host, `sys.stdout` can be `None` or lack
  `reconfigure`. The block then does nothing instead of raising.
- **stderr too.** Tracebacks and `SystemExit` messages go there.

## Verification

Unpatched main tree against a patched git worktree, with **identical script filenames** (argparse
prints the name into `usage:`), `PYTHONUTF8=0`:

| script | run | UTF-8: output byte-identical? | cp1252 before | cp1252 after | after == UTF-8 output? |
|---|---|---|---|---|---|
| ablation_table | `--help` | yes | **CRASH** | exit 0 | yes |
| baseline_demo | `--help` | yes | exit 0 | exit 0 | yes |
| bench_pyramid | `--help` | yes | exit 0 | exit 0 | yes |
| bench_scatter | `--help` | yes | exit 0 | exit 0 | yes |
| build_reference_map | `--help` | yes | **CRASH** | exit 0 | yes |
| eval_synthetic | `--help` | yes | **CRASH** | exit 0 | yes |
| feature_report | `--help` | yes | **CRASH** | exit 0 | yes |
| frnet_eval | `--help` | yes | **CRASH** | exit 0 | yes |
| frnet_finetune | `--help` | yes | **CRASH** | exit 0 | yes |
| ghost_removal_figure | `--help` | yes | **CRASH** | exit 0 | yes |
| measure_visibility_cap | `--help` | yes | **CRASH** | exit 0 | yes |
| memory_bound | `--help` | yes | exit 0 | exit 0 | yes |
| plan_query_survey | `--help` | yes | **CRASH** | exit 0 | yes |
| regret_plot | `--help` | yes | **CRASH** | exit 0 | yes |
| timing_table | `--help` | yes | **CRASH** | exit 0 | yes |
| memory_table | full run | yes | exit 0 | exit 0 | yes |
| sampling_table | full run | yes | exit 0 | exit 0 | yes |
| data_status | full run | yes | exit 1 | exit 1 | yes |

- **No change on a UTF-8 terminal:** **20 of 20** executed scripts are byte-identical (the last two closed 2026-09-18; see above).
- **Fixes the crash:** all 12 that crashed now exit 0, and their cp1252 output equals the
  UTF-8 output, so the real ⚑ is emitted rather than `?`. That is the same 12 the earlier doc
  counted.
- **`data_status` exits 1 in all four arms,** patched or not. That is its own "data not
  complete" result, and it is unchanged by this.
- **Also checked with no forced encoding at all** (the locale default for redirected output):
  see the `timing_table --help` check recorded in the commit that staged this file.
- **Nothing else regresses:** `tests/test_timing.py`, `test_ablation_table.py`,
  `test_build_reference_map.py`, `test_plan_query_survey.py` and `test_regret_plot.py` gave
  31 passed, 1 skipped in the patched worktree. The tests locate scripts relative to their own
  file (`Path(__file__).resolve().parents[1] / "scripts"`), so they imported the **patched**
  copies (checked). `ruff check scripts/` passes on the patched tree.
- **Applies cleanly** to the current tree (`git apply --check`).
