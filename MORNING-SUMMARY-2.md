# Morning summary 2 — overnight run, 2026-09-13 / 14

> # [!] READ FIRST — this machine is not currently trustworthy for timing work
>
> **Restart it, or leave it idle for an extended period, before running anything
> timing-sensitive on it.**
>
> Measured near the end of the session, at **2% CPU load**:
>
> | | |
> |---|---|
> | CPU clock | **1520 MHz against a 2400 MHz base — 37% down** |
> | commit charge | **23.61 GB against 15.73 GB physical** — ~8 GB over-committed |
> | free physical | 2.2 GB |
>
> The effect is not subtle. `reports/harnesses/ground_cost.py` read **21.01 ms**
> early in the session and **93.44 ms** late — *same script, same data, same
> machine* — a **4.4× drift**.
>
> **What this means in practice:**
> - **Any absolute timing taken late in this session is void.** I did not publish
>   the `ground` numbers for exactly this reason (§4a).
> - **A/B ratios measured back-to-back in one process are still good** — a
>   machine-wide slowdown scales both arms. That is why `transform` (25×) and
>   `cleanup` (21× / 3.9×) stand.
> - **Allocation counts are unaffected entirely.** `tracemalloc` counts bytes, not
>   time, so every MB figure in tonight's work holds regardless.
> - **This upgraded D8's scope** — see §3.

**All eight list items addressed, plus the follow-on queue: R-g, R-h, R-i and
D10 STEP 1. 20 commits (`2955b14`..`HEAD`), all local, nothing pushed.** Blocked
items are in §3.

**The headline:** R-f closed, and it closed *because* R-a was done first — the
harness that resisted two investigations turned out to contain a three-line `for`
loop that explained everything. And R-b is no longer a mystery: the p99 tail is
**allocation**, localised to 15 lines, measured.

---

## 1. What got done, per task

| # | item | outcome | commits |
|---|---|---|---|
| 1 | **R-a** harnesses | **DONE** — 21 scripts in `reports/harnesses/` | `2955b14`, `8318fed` |
| 2 | **R-f** seq 00 ring-1 | **CLOSED** — exact reproduction | `7225f11` |
| 3 | **R-e** 99.87 vs 99.5 | **RESOLVED** by recomputation | `3a43c19` |
| 4 | **R-c** stdout UTF-8 | **proposed**, not applied | `92fc7d0` |
| 5 | **R-b** p99 cause | **localised**, not fixed | `0035850`, `ab5fe6c` |
| 6 | **DL** three-way regret | **1 of 3 sub-tasks**; rest blocked (§3) | `ff774fe` |
| 7 | **ROS** scoping | **DONE** as design only | `0035850` |
| 8 | **OPEN-ITEMS** current | **DONE** | `faebc41` |

### 1. R-a — the harnesses (`reports/harnesses/`, 21 scripts + README)

Each carries a `# PROVENANCE` header naming the report, **which figures in it**,
the exact invocation, and any trap that cost time. Chose `reports/harnesses/`
over `scripts/` deliberately: the point is verifiability of a claim, not a
reusable tool.

Committing them took repo-wide `ruff` from 1 error to 51, all stylistic — I fixed
them rather than excluding the directory, and **gated every reformatting edit on
`ast.dump(before) == ast.dump(after)`**. That guard earned its keep: the
mechanical pass tried to turn

```python
def combo_a(p): p.num_iter = 2; p.enable_RNR = False; p.enable_RVPF = False
```

into a one-statement function plus two module-level assignments — a real
behaviour change, silently plausible. Six files were refused and hand-split.

### 2. R-f — closed, and it is the singleton again

**Every published R1 figure reproduces exactly**, including the cell count that
was the whole puzzle: **seq 00 ring 1, 41,892 cells @ 6.77 cm.**

The cause is three lines at the bottom of the recovered harness:

```python
for seq in ("07", "08", "00"):
    allrows += run(seq, 40, "5/10/20/40")
```

All three sequences, **one process, one module-level estimator**, two passes each
— so by the time seq 00 is measured the estimator has processed ~160 frames of
the other two. Every experiment of mine gave each sequence a fresh estimator,
which is why seq 00 refused to match and why I previously concluded the singleton
was *not* responsible. Wrong, and wrong instructively: the experiment held
constant the one thing that mattered.

It explains the whole pattern, not just seq 00 — 07 was **first** (fresh
estimator, always matched), 08's drift is 0.029% (too small to move a ring), 00
was **third**.

Controlled, one variable:

| seq 00, shared estimator | ring 1 | n |
|---|---|---|
| fresh process | 6.45 | 41,953 |
| **after 07 and 08** | **6.77** | **41,892** |

**The sharper finding, and it is worse than non-reproducibility:** a published
accuracy figure depends on **which other sequences were measured before it in the
same process**. Nothing names evaluation order as an input. Reorder the tuple and
the numbers change — deterministic harness, fixed data, unchanged code. Until D1
is fixed, **any harness measuring more than one sequence per process is unsafe**;
the mitigation is one process per sequence.

**This strengthens D1 considerably.** It is now contaminating *two* published
figures — seq 07 ring 1 (3.60 vs a consistent-mask 3.04) and seq 00 ring 1 (6.77
vs ~6.46) — and seq 00 is the *more* contaminated.

### 3. R-e — 99.5% was the wrong one, and I checked rather than assumed

From `sih-math.md` eq. (4), `P_fill = min(1, c/s_rad) · min(1, c/s_az)` at
r = 50 m, c = 5 cm:

```
s_rad(50) = (2500/1.73) × 0.00745256 = 10.7696 m      (doc: 10.8)
s_az(50)  = 50 × 0.0035              = 17.50 cm       (doc: 17.5)

both terms   1 − 0.0013265 = 99.8674%     <- README, CORRECT
radial only  1 − 0.0046427 = 99.5357%     <- master-v4, INCOMPLETE
```

99.5% is the **radial term alone**. `master-v4.md`'s own sentence gives it away —
*"the ground returns are 10.8 m apart radially, so 99.5%…"*. A cell needs a ring
*through* it **and** an azimuthal sample *in* it.

Corrected in `master-v4.md` with the arithmetic inline. **README was already
right and is untouched.** Also corrected two inherited instances in Hriday's R2
memo — flagged because it is not my file, and worth knowing the fix **strengthens
his argument**: 754 dead cells per live one rather than 215, and he uses the
figure to motivate far-ring scene completion.

### 4. R-c — proposed, and the scope is much larger than we thought

**15 of 20 scripts** contain a cp1252-unencodable character; **12 of 20 crash on
`--help` right now**, because `argparse` prints the module docstring. The two
`print()` sites fixed earlier were a symptom.

> The script I wrote to *count* the affected files crashed on the bug it was
> counting, while printing its own results.

**The one central fix is blocked:** every affected script imports `vrgrid.*`, so
the correct single home is the package root — which is
`include/vrgrid/__init__.py`, **frozen**. A two-line addition in the one right
place costs three signatures. Raised in the proposal as worth a three-way: *a
frozen interface arguably should not be frozen against its own process setup.*

Recommended **`PYTHONUTF8=1`** (zero source edits, expires with Python 3.15 by
PEP 686) over a per-script helper (which needs `scripts/` on `sys.path` — false
under `python -m`, and the failure would be an `ImportError` on the helper meant
to prevent a crash).

Verified with **identical filenames in both arms**, since `argparse` prints the
script name into `usage:`: **byte-identical output on UTF-8 stdout**, exit 0
instead of exit 1 on cp1252.

### 5. R-b — the tail is allocation

Ruled out by measurement, not argument:

- **GC:** 3 collections per 220-frame run, generation 0 only, **0.00% of
  runtime**, and **0 of the 8 worst frames** had one. Frames *with* a collection
  are marginally faster.
- **Data:** worst-15 frame overlap across reps is **1 of 15 — exactly chance**.
- **Thermal:** +0.07 ms/frame that **plateaus after the first quarter**, over a
  27.8 s run.

**Why it resisted attribution: the answer depends on page-cache state.** Cold
run, `load` owns 60% of the excess; warm run, **`transform` owns 84%**. `load`
swings **14×** between identical invocations — and `load` is file I/O a live
sensor does not have.

**The warm tail is `transform`, and it is allocation.** Isolated, no pipeline, no
disk, identical preloaded scans:

| arm | p50 | p99 | **max** | spikes | **alloc/call** |
|---|---|---|---|---|---|
| as shipped | 3.19 | 20.46 | **34.61** | 4/220 | **10.86 MB** |
| float64 in | 2.34 | 4.92 | 32.86 | 2/220 | 7.90 MB |
| **preallocated** | 1.06 | **1.29** | **1.37** | **0/220** | **0.00 MB** |

**25× smaller tail, zero spikes.** The float64 arm is the control that matters:
removing only the cast drops p99 to 4.92 but leaves max at 32.86 — the tail
follows whichever allocation is left.

Then, across all stages: the pipeline allocates **~59 MB per frame**. The control
in that table is excellent — **`bin` allocates 0.04 MB**, and `bin` is on the grid
path, which *is* covered by `test_the_two_grid_allocations_stay_fixed`, while every
uncovered perception stage allocates 2–14 MB. **The invariant works exactly where
it is enforced.**

**Honest limits.** Fixing `transform` takes frame p99 from 135 to ~113 ms against
a 100 ms budget — **still over**. And allocation is *a* mechanism, not *the* one:
`ground` has the **largest** spread (8.85 ms) on modest allocation, so its tail is
likely inside the C++ extension and will not yield to this. Rank correlation is
+0.64, and `tracemalloc` **smooths the very spikes under study** (transform reads
0.81 ms instrumented vs 22.14 uninstrumented), so that number is suggestive of
ranking only.

### 6. DL — 1 of 3, and the reason is in §3

`--fast-scatter` **re-verified exact in both directions** through
`frnet_eval.py` rather than the standalone check, so it exercises the path the
evaluation uses: `scatter_max` 0.000e+00 forward *and* backward, `scatter_mean`
within float32 rounding, and the patch reaching **all three** binding sites —
which is the specific failure the shim exists to avoid.

The other two sub-tasks are blocked on a missing checkpoint. **90.3% / 65.2% /
61.1% was not re-measured tonight** and those figures stand on the earlier run.

### 7. ROS — design only

`pending-review/ros2-adapter-design.md`. Nothing under `adapters/` created. The
design needs **no change to the frozen `api.py`** as drafted.

The blocker I did not expect: **`query_region` is a Python loop over `query()`** —
its own docstring says a Python loop at 10 Hz *"is not a thing you can do"*, and
then it is one, deliberately, so there is only one query implementation. A
20 m × 20 m window at 5 cm is 160,000 `query()` calls per published frame. Three
ways out are laid out; publishing below frame rate is the honest first move.

Also verified: `rclpy`, `grid_map_msgs`, `sensor_msgs`, `nav_msgs`, `tf2_ros` are
**all absent** here, so nothing in that document has been run and I said so at the
top of it.

---

## 2. Everything in `pending-review/` and why each needs you

| file | needs a human because |
|---|---|
| **`transform-points-allocation.md`** (new) | **D9.** Returning a view into a reused buffer could silently mutate data a caller retained — invisible until something compares across frames. And the function is stateless today, called from three places, so a module-level buffer adds exactly the hidden shared state D1 is teaching us about. I recommend an optional `out=`; it touches a hot-path signature. |
| **`ros2-adapter-design.md`** (new) | **D10.** Five sub-decisions, and one of them — whether `export_gridmap()` takes arguments — **changes a frozen signature**, so it needs the three-way *before* implementation. |
| **`stdout-utf8-at-entry.md`** (new) | **R-c.** Touches `demo.sh`, CI and a doc; and the *central* fix is blocked by the `include/vrgrid/` freeze, which is itself worth raising. |
| `patchworkpp-num-iter-tradeoff.md` | Withdrawn on evidence. Kept as the record; no action. |
| `r3-ring-boundary-under-anisotropy.md` | **D2** — Aakash's `lattice.py`, cross-lane consequence in `metrics._ring_cells`. |
| `r5-sticky-safety-critical-class-bit.md` | **D3** — five sub-decisions, and frozen `include/vrgrid/cell.py`. |
| `r7-readme-counts-draft.md`, `r7-readme-wording.md` | **D4** — you said you would place the wording yourself. |
| `handover-latency-line-correction.md`, `r7b-mIoU-*.md`, `timing-table-unicode-crash.md` | Applied. Kept as rationale records. |

---

## 3. Needs your call

### D11 — the FRNet checkpoint is missing, and it blocks two sub-tasks

```
checkpoint not found: checkpoints/frnet-semantickitti_seg.pth
```

`checkpoints/` does not exist here, `.gitignore:18` excludes model weights
deliberately, and **no `.pth` has ever been committed on any branch**. So the
accuracy re-confirmation and the **entire three-way plan-regret comparison** (the
FRNet-predicted-label arm needs weights to infer with) could not run.

I did not go looking for weights to download. Fetching pretrained model files
unprompted is not something I should do on your machine, and the right source is
a decision — the original checkpoint, or a re-download from the FRNet authors.
**Tell me where it should come from and the DL item finishes in one run**
(`--fast-scatter` makes it ~1 minute rather than ~35).

### D8 — SCOPE UPGRADED, and it is no longer "pick a hostname"

D8 was *two hosts disagree, choose a reference*. That is now insufficient:
**one host varied 4.4× within a single session** (see the banner above). Naming a
reference machine fixes nothing when the same box gives 21 ms or 93 ms depending
on clock and memory state.

**Resolving D8 now requires a defined and checked machine state**, at minimum:

1. **CPU clock recorded alongside every figure** — `CurrentClockSpeed` against
   `MaxClockSpeed`. A 37% downclock is completely invisible in a timing table.
2. **Commit charge under physical RAM**, checked before *and* after the run.
3. **A stated warm-up / idle precondition.**

A latency number without its machine state attached is not checkable, whoever's
host it came from. The corollary is already proven tonight: **prefer controlled
ratios over headline absolutes** wherever the question allows it, because ratios
survive this and absolutes do not.

### D9 / D10 — summarised in §2, detailed in their files

Both are design decisions with real consequences, not preferences. D10 in
particular has a **frozen-signature** question inside it that is cheaper to settle
before anyone writes code.

### One thing I deliberately did not do

**CARLA (D7) remains untouched**, as instructed — zero scope has been supplied
across three sessions and starting from a guess is the wrong move. It is recorded,
not started.

---

## 4. Git state — nothing pushed, nothing touched

| | |
|---|---|
| branch | `main`, tree **clean** |
| commits tonight | **9**, all local |
| ahead of `origin/main` | **35** |
| **HEAD on any remote** | **0 remote branches contain it** |
| `origin` | 13 branches — **untouched** |
| `fork` | 23 branches — **untouched** |
| `vrgrid26` | 6 branches — **untouched**, `main` still Shrestha's `0c769c9` |
| `vrgrid26-fork` | 3 branches — **untouched** (PR #5's branch still at `aa74b3e`) |

**No push, no PR, no fetch of anything new.** PR #5 (the one-word headline fix) is
still open and unmerged, exactly as left.

### Gates

| | |
|---|---|
| `pytest` | **672 passed, 1 failed, 3 skipped** |
| the failure | `test_real_sequence_replay_is_identical` — **D1**, expected, and now understood better than ever (§1.2) |
| `ruff` | **1 error** — the pre-existing E741 at `tests/test_metrics.py:472` (**R-d**, not my lane, flagged only) |

Identical to where the night started. **Nothing I did tonight moved either
number.**

---

## 4a. Second half of the night — after your review

Nine further commits. Everything below is local.

| item | outcome | commit |
|---|---|---|
| **"Zero allocation in the frame loop"** | **SCOPED, not retracted** — 12 edits / 7 files, both scope gaps kept separate; `--alloc` now fails loudly with `--seq` | `235986d`, `b38b053` |
| **R-g** | **DONE** — retained-growth invariant extended to perception; suite 672 → **673 passed** | `8b40e44` |
| **R-h `cleanup`** | **DONE and proposed** — the 9.61 MB is `np.isin` *inside an argument*; a boolean LUT is 21× / 3.9× and bit-identical | `2cfd487` |
| **R-h `ground`** | **INCONCLUSIVE — instrument failed.** See below. | `9b50111` |
| **D10 method + STEP 1** | **DONE as prep** — transcription shrinks it to `_refined` alone; no reduction in the chain, so no ULP risk | `2f08ed0`, `6446648` |
| **R-i** | **PROPOSED** — technique already proven; the real decision is table unification | `e1b1056` |

### The finding I would read first: the machine degraded 4.4× mid-session

`ground_cost.py` read **21.01 ms** early and **93.44 ms** late — same script, same
data, 2% CPU load. Measured at that moment: **CPU downclocked to 1520 MHz against
a 2400 MHz base (37% off)** and **commit charge 23.61 GB against 15.73 GB
physical**, i.e. ~8 GB over-committed and paging.

**So every absolute timing from the late window is void**, including the `ground`
numbers, and I did not publish a `ground` verdict off them. Its correlations came
out mutually contradictory across runs (`corr(time, n_points)` −0.592 then +0.532),
which is the signature of noise rather than evidence either way.

**What survives, and why it survives:** `transform` and `cleanup` were **A/B arms
run back to back in one process on identical data**, so a machine-wide slowdown
scales both arms and the *ratios* hold — 25×, 21×, 3.9×. Allocation counts are
unaffected entirely; `tracemalloc` counts bytes, not time. That is the difference
between a controlled comparison and a headline number, and it is why those
sections were built as A/Bs.

**It also strengthens D8 considerably.** A host whose absolute timings move 4.4×
within one session cannot anchor a latency claim by hostname alone — the machine
state has to travel with the number.

### Note on the final gates

`ruff` is **1 error** (the pre-existing E741, R-d). The full suite was **not**
re-run at the end: it timed out past 10 minutes on the degraded machine, where it
had taken 2–6 earlier. It did not need re-running — `git diff 8b40e44..HEAD --
src/ tests/ include/ configs/ scripts/` is **empty**, so no code has changed since
the last full green run, and **673 passed / 1 failed (D1) / 3 skipped** stands.
Everything committed after that point is `reports/`, `pending-review/` and
`OPEN-ITEMS.md`.

### Two other things worth your eye

**The allocation claim was the determinism defect again**, third instance: a
back-end figure from `timing_table.py`'s synthetic path quoted as whole-frame,
exactly like 80.78 ms. Fixed the same way Shrestha fixed determinism — split, not
retract — because the back-end achievement is real (`bin` allocates 0.04 MB, and
that is the control proving the invariant works where enforced).

**The two D10 traps are now named checks with a measured fixture**, not notes.
`slot_of`'s two OUTSIDE paths diverge on **361 of 361** swept points when a window
is un-tracked and **0 of 361** when tracked — the worst possible shape, since one
mask passes every test written against a tracked map. And `OUTSIDE == -1`, so a
conflated mask feeds `-1` into a ring-indexed gather and silently reads the *last*
ring.

## 5. What I would pick up next

1. **D1.** It is now contaminating two published accuracy figures and it makes
   evaluation order an unnamed input. It is the highest-value item on the list by
   a clear margin.
2. **R-h — `cleanup`, not `ground`.** 9.61 MB allocated, 7.67 ms spread, and it is
   *our own numpy*, so the same isolated A/B that settled `transform` runs against
   it directly. `ground` is a separate problem with a different cause.
3. **R-g.** Extend the no-allocation invariant to the perception half. Independent
   of D9 and worth doing whichever way D9 goes — the `bin` = 0.04 MB control shows
   the invariant works wherever it is actually enforced.
