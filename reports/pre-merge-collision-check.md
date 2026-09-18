# Pre-merge collision check — `jp/p99-alloc-fixes` against upstream

*Run 2026-09-18 on JP's instruction, read-only. No rebase, no merge, no commit touching
`src/run/engine.py` or `src/run/__main__.py`. The dry-run merge happened in a disposable worktree
that was deleted afterwards.*

---

## [!] What could NOT be checked, and why

JP asked for the diffs of **`03abd7b`** and **`6af6907`** (Shrestha's GPU port) against `engine.py`
and `__main__.py`, and for a dry-run merge against **current `origin/main`**. Neither is possible
from this clone as it stands:

| asked | state here |
|---|---|
| read `03abd7b` | **commit not in this clone** (`git cat-file -t` fails) |
| read `6af6907` | **commit not in this clone** |
| `reset_estimator()` call frequency | **the symbol does not exist anywhere** in this clone: not in the working tree, not in any commit on any local branch (`git log --all -S`) |
| merge current `origin/main` | `origin/main` here is **`5e0ebf3`, fetched 2026-09-11**, and has **0 commits** we do not already have. A dry-run against it is a no-op and would falsely report "no conflicts" |

**No remote was contacted.** A fetch is still waiting on JP's explicit go-ahead.

Newest refs in this clone: `origin/main` `5e0ebf3` (11 Sep) · `vrgrid26/main` `0c769c9` (13 Sep) ·
local `main` `2c952dd` (14 Sep) · `HEAD` `4e81220` (16 Sep).

**Everything below about the 16–17 Sep GPU port, Aakash's `reset_estimator()` point-fix, D2/R3
closure and the AWS T4 work comes from JP's message of 2026-09-18, not from anything verified here.**

## 1. `reset_estimator()` — the question stands, unanswered

The question was: does it fire **once per `iter_pipeline()` call** (once per run) or **once per
frame**? It cannot be answered here, because neither the commit nor the symbol is present.

What can be said precisely, so the answer is immediately usable when the commit is readable:

- **If once per run:** our recorded timings stay valid *as descriptions of this branch's code*. A
  per-run reset costs one estimator construction per process, outside the per-frame loop, so p50 /
  p99 and the stage breakdown are unaffected.
- **If once per frame:** every per-frame number we hold would no longer be comparable to upstream,
  because upstream's `ground` stage would then carry an estimator construction **inside** the timed
  loop. Ours would need re-measuring against current `main`, not merely rebasing.
- **Either way**, our numbers were measured on **our** branch's code (`4e81220`), on one laptop
  (D8). They describe this branch, and they were never a claim about upstream `main`.

**How to settle it in one command once fetched:**
`git show 03abd7b -- src/run/__main__.py src/perception/ground.py`, then check whether the call sits
above the `while True:` loop in `iter_pipeline` (per run) or inside it (per frame).

**Independent of that:** our own local determinism test still fails today
(`tests/test_determinism.py::test_real_sequence_replay_is_identical`, re-run 2026-09-18), as expected
for a clone that does not contain the point-fix.

## 2. Do `d540618` and `697a2bd` still apply?

**Against the newest upstream visible here (`vrgrid26/main`, 13 Sep): yes, cleanly and exactly.**

- `git diff main vrgrid26/main -- src/run/engine.py` is **empty**: those 11 upstream commits do not
  touch `engine.py` at all. The same holds for `src/run/__main__.py` and `src/perception/ground.py`.
- So `_cleanup` and `_centres` on 13-Sep upstream are **byte-identical** to the versions our two
  proposals were written against, and both diffs apply unchanged.

**Against the 16–17 Sep GPU port: unknown, and genuinely at risk.** A port that moves split/merge and
traversability to the device is likely to touch the same functions:

- `d540618` replaces `np.isin(occupied, touched)` with a boolean LUT indexed by slot. If `_cleanup`'s
  candidate selection now runs on the device, the LUT would need to be a device array, and the fix as
  written is **CPU-shaped**.
- `697a2bd` adds `_centres(..., sorted_slots=True)`, a NumPy `out=`-based fast path over contiguous
  per-ring slices. If `_centres` is now a kernel, this is **redundant** rather than conflicting.

Neither can be judged without reading the port. **Recommendation: do not ask Shrestha to review
either proposal until the port is readable here** — a review request against a version he has already
rewritten wastes his time.

## 3. Dry-run merge — the real conflict surface available today

In a disposable worktree at `4e81220`, `git merge --no-commit --no-ff vrgrid26/main`, nothing
resolved, worktree deleted afterwards. **This is 13-Sep upstream, not "current main".**

```
merge exit 1
CONFLICT (content): Merge conflict in README.md
```

| result | detail |
|---|---|
| **conflicted files** | **`README.md` only — 1 conflict hunk** |
| merged cleanly, changed by their side | `docs/presentation/00-START-HERE.md`, `scripts/frnet_eval.py`, `scripts/frnet_finetune.py`, `src/gpu/kernels.py`, `tests/test_kernels.py` |
| upstream commits touching `src/run/engine.py` | **0** |
| upstream commits touching `src/run/__main__.py` | **0** |
| upstream commits touching `transforms.py`, `range_image.py` | **0** |

**So against 13-Sep upstream the collision surface is one README hunk, and nothing in the functions
our proposals touch.** The collision map's warning about `engine.py` and `__main__.py` is about the
*16–17 Sep* port, which this clone cannot see.

## 4. What this branch actually holds, for the merge conversation

43 commits, 59 files, +11,350 / −121 lines. The `src/` surface is small and concentrated:

| file | our change | owner |
|---|---|---|
| `src/run/engine.py` | +97 / −3 (cleanup LUT, `_centres` fast path) | Shrestha |
| `src/run/__main__.py` | +106 / −6 (opt-in DL mode, transform scratch wiring) | JP |
| `src/perception/transforms.py` | +49 / −1 | JP |
| `src/perception/range_image.py` | +31 / −10 | JP |
| `src/perception/reflectivity.py` | +22 / −6 | JP |

Everything else is additive: tests, harnesses, reports, scripts.

## 5. What would settle all three questions

One read-only fetch (`git fetch vrgrid26 && git fetch origin`), then:

1. `git show 03abd7b -- src/run/__main__.py src/perception/ground.py` — the reset's call frequency.
2. `git diff HEAD origin/main -- src/run/engine.py` — whether `_cleanup` / `_centres` survive in the
   form the two proposals assume.
3. The dry-run merge repeated against the fetched `origin/main`, for the real conflict list.
4. Whether `docs/gpu-lane/t4/` exists upstream — i.e. whether the FRNet fine-tune actually ran.

**Separately, already verified here:** `docs/gpu-lane/02-AWS-RUNBOOK.md` (Shrestha's, 297 lines, in
this clone since the 12 Sep migration) already covers instance choice, region, AMI, storage, security
group, launch checklist, S3 staging — explicitly *"do not upload 84.8 GB"* and *"bring nothing back"*
— environment, tmux discipline and failure modes. `reports/aws-gpu-realtime-runbook.md` (ours,
`9b784e2`) **duplicates that and contradicts its transfer route**, while adding measurement steps his
does not cover (the DL real-time figure, GPU reproducibility steps e/f, the 0–200 frame slice, result
retrieval). It should become an addendum to his runbook rather than a second runbook.
