# Full project audit — every lane, every tracking system, every number

*2026-09-23. Read-only analysis against `jp/p99-alloc-fixes` @ `5a3ba9a` and `vrgrid26/main` @
`8acbfe9`. Ownership taken from `.github/CODEOWNERS` and `CLAUDE.md`, not assumed. No status line was
trusted: every "closed" was re-checked against a commit, a diff, an artifact or a run.*

---

## 0. Read this first — the finding that explains the rest

**The tracking system has outgrown the design it was built for. That is a structural fact, and it
explains nearly every individual defect in this document.**

> **~30 tracked items have no roadmap ancestor. 11 do.**

The 10-day roadmap defined R1–R11. What the team actually tracks today is `D1`–`D12`, `R-a`–`R-k`,
`N-1`–`N-6`, `S-1`–`S-6`, plus `GPU-LATENCY` and `BAND` — roughly three-quarters of it discovered
during the work rather than planned. A tracker designed for eleven known deliverables is being used
as the coordination channel for thirty-odd emergent ones, by two active people writing into two
copies of the same file from different branches.

The symptoms follow directly, and they are not independent accidents:

| symptom | count | detail |
|---|---|---|
| parallel identifier systems | **4** | roadmap R1–R11 · research modules R1–R3 · `D`-series · `R-`letters |
| live cross-system collisions | **2** | roadmap R1/R2/R3 vs research modules R1/R2/R3; `D9` across two copies |
| stale or false statuses found | **5** | R-b, R-d, S-6, D2, R-a — across three audit passes |
| duplicate-work incidents | **4** | the `--semantics` flag, classification-accuracy-by-range, the AWS runbook, the DL real-time figure |
| figures for one quantity | **6** | end-to-end latency (§3.4) |

**Fixing rows does not fix this.** Three passes have each found new stale statuses *after* the
previous pass "caught them all". The rate is not falling. Any plan that ends at "correct these five
rows" will be back here within a week.

What this document does *not* do is propose the redesign. That is a decision for the room, and it is
listed as such in §4.3.

---

## 1. Phase 1 — repo inventory, every area

Ownership has two authoritative sources that agree: `.github/CODEOWNERS` and `CLAUDE.md:43-47`.
Commit census since 2026-09-14.

| area | owner | upstream | mine | who actually committed | open items |
|---|---|---|---|---|---|
| `include/vrgrid/` | all three (frozen Day 0) | **0** | 0 | nobody | S-4 |
| `src/grid/` | @AakashH2006 | 5 | 0 | **Stxtics03 only** | D2 ✅, D3, D6, S-1 |
| `src/eval/` | @AakashH2006 | 7 | 0 | Stxtics03 + AakashH2006 | N-3, R-e ✅, R-f ✅ |
| `src/gpu/` | @Stxtics03 | 10 | 0 | Stxtics03 | R-j GPU half, GPU-LATENCY |
| `src/perception/` | **@victorysingh** | **8** | 4 | **Stxtics03 ×5, AakashH2006 ×3, JP ×0 upstream** | D1 ✅, R-k ✅, **D12**, D11 |
| `src/run/` | *nobody* — "belongs to nobody and everybody" | 11 | 6 | all three | R-c, R-i, `--semantics` collision |
| `dashboard/` | **@victorysingh** | **18** | 2 | **AakashH2006 ×17, Stxtics03 ×1, JP ×0 upstream** | N-4 ✅ |
| `configs/` | all three, **frozen** | **1** | 0 | **Stxtics03 alone** ⚑ | **BAND**, D5, D6 |
| `scripts/` | unassigned | 33 | 9 | all three | R-c, R-i |
| `tests/` | unassigned | 34 | 13 | all three | **R-d** |
| `docs/` | unassigned; `research-log.md` → @Parzival-7926 @zero-odds @Prathyushree | 35 | 1 | Stxtics03, AakashH2006 | R-b, provenance |
| `reports/`, `pending-review/` | JP by practice | 0 | 45 | JP only | the staged cross-lane work |

**Findings.** `configs/` was changed unilaterally (§3.1). Both of JP's lanes were worked on by other
people while JP has zero upstream commits in either — the structural cause of the collisions, since
CODEOWNERS only fires on PRs and there have been none. `docs/research-log.md` is owned by the three
research-module people; all ten of this cycle's commits to it are Shrestha's.
**`include/vrgrid/` has zero commits — the frozen shared surface has been respected by everyone all
cycle.**

---

## 2. Phase 2 — the tracking systems reconciled

### 2.1 Four systems, not three

| system | R1 | R2 | R3 |
|---|---|---|---|
| **10-day roadmap** | per-class/range accuracy | three-way plan regret | ring boundary under anisotropy |
| **research modules** (CODEOWNERS) | Representation & Prior Art — Shriniwas | Dynamics & Ghost Removal — Hriday | Traversability & Plan-Sensitivity — Prathyushree |

Thematically adjacent enough to be genuinely confusable: **research R3 is "plan-sensitivity";
roadmap R2 is "plan regret."** The `memo-r1-*.md` / `memo-r2-*.md` files belong to the *research*
system — anyone opening `memo-r2-future-scope-and-current-research.md` expecting roadmap R2's plan
regret gets the wrong document.

### 2.2 The D9 collision — resolved

Shrestha opened his `range_interpolation` finding as **D9** (`8acbfe9`); `D9` was already the closed
`transform_points` item here. **Resolution: JP's D9 keeps its ID** (older, already cited from
`pending-review/transform-points-allocation.md` and commit messages); **his becomes D12**, content
preserved and attributed. Staged for him: `pending-review/d9-collision-renumber-to-d12.md`.

### 2.3 Master correspondence table

| roadmap | subject | tracker | status | owner | evidence |
|---|---|---|---|---|---|
| R1 | **height** accuracy by band × class | closed row | ✅ verified-closed, **but band-invalidated** (§3.1) | JP | `7eb0d2d` |
| R2 | plan regret with a model | **D11/R2** | ✅ verified-closed | JP | `a594255`, `4e81220` |
| R3 | ring boundary | **D2** | ✅ verified-closed | Aakash → **done by Shrestha** | `df35fd5` diff |
| R4 | ring-boundary CI test | **D2** | ✅ verified-closed | ″ | `test_no_cell_footprint_contains_another_under_foveation` |
| R5 | sticky VRU bit | **D3** | 🔸 open, design only | the room | `pending-review/r5-…md` |
| R6 | ground-only variance | *none* | ❓ **unverifiable** | — | 0 mentions; **nothing cites it** |
| R7 | hazard miss rate | **D4** (wording) | ✅ closed, **band-invalidated** (§3.1) | JP | `2811ebc` |
| R8 | README foveation | *none* | ✅ verified-closed | — | `README.md:11`, `:668` |
| R9 | latency / VRAM | → R-b, D8 | ✅ closed; descendants live | JP/Shrestha | `8239918`, `adbe09d` |
| R10 | wide-depression gradient | §1 row | 🔸 **was silently dropped → now unassigned** | **nobody** | 0 commits/files/mentions |
| R11 | limits page | closed row | ✅ verified-closed | JP | `e4bd731` |

**No roadmap ancestor:** `R-a`–`R-k`, `D1`, `D5`–`D8`, `D10`, `D12`, `N-1`–`N-6`, `S-1`–`S-6`,
`GPU-LATENCY`, `BAND`.

### 2.4 Status verification — the fifth false status

⚑ **S-6 contradicts D12, and S-6 is the older, wrong version.** It still says the frnet
non-determinism *"is the model"* and that *"FRNet's `scatter_mean` is order-dependent on CUDA"*.
`8acbfe9` refutes both two days later: a CUDA-only mechanism cannot produce a CPU-vs-CPU result, and
the scatter ops measured bit-identical at 16 threads. It also states non-determinism as a property of
the mode when it **vanishes at one thread** (0.0000% on all three frames). The new row landed; S-6
was never updated. **His file, his row — flagged, not edited.**

**Verified genuine closures** (each checked against code or artifact, not label): D2, D9-mine, N-1
(`6222625`, +100 test lines), N-2 (`5e239f0`, +71), N-4, N-6 (`24c5dc2`), R-e, R-f, R-g, R-k, R1, R2,
R8, R11.

**Owned by people with zero commits this cycle:** D3, D6 (Aakash), D10 (unowned in practice), R10
(nobody), and the **entire research-module system** — `docs/research-modules.md` was written once by
Shrestha on 28 Aug and never touched by any of its three owners. Module R3 has no memo and zero
`[R3]`-tagged entries; its Day-3 verdict text does appear in four other docs, so it is **not** called
dropped, but its authorship is unclear.

---

## 3. Phase 3 — provenance sweep

### 3.1 The `configs/` change and what it invalidated

`843ad54` (17 Sep, Stxtics03) changed `vertical_extent_m` from **`[-2.0, 6.0]` to `[-3.5, 4.5]`** in
**both** schedule configs — same 8 m span, shifted down 1.5 m — wired through `quantise.py`,
`metrics.py`, `harness.py`, `kernels.py`, `cuda_kernels.py`.

**Process:** CODEOWNERS requires **@AakashH2006 @Stxtics03 @victorysingh** on `configs/`; `CLAUDE.md`
says configs are *"frozen before schedules are compared — changing one after that invalidates the
ablation."* The commit is linear, **no PR**, so no review fired.

**Technically it is well-evidenced**, and this should be said as clearly as the process point: the
split was chosen by surveying ground returns beyond 10 m on all eleven sequences, and CPU/GPU came out
bit-identical on 200 frames of all eleven plus all 4,071 frames of seq 08. **Shrestha also propagated
it** into `known-limitations.md` (137 lines) and four presentation docs.

**Three documents it did not reach — provisionally invalidated pending re-measure:**

| document | band-dependent figures | owner |
|---|---|---|
| `reports/r1-accuracy-by-class-and-range-band.md` | height RMSE per band × class (0.95 / 1.14 / 3.60 cm) | **JP** |
| `reports/r7-hazard-miss-rate.md` | drivability miss rates, from height gradients | **JP** |
| `docs/defense-rehearsal-playbook.md` | curb cells 4,041 / 9,499, ring medians, 52.3% occupied set | **Shrestha — panel-facing** |

**Verified band-INdependent, unaffected:** README memory ratios (21.5×, 286×), cell counts, FRNet
classification accuracy, latency.

### 3.2 N-5 — checked first; its closure **holds**

N-5's fix *is* `843ad54`, so the question was whether its closure inherits the invalidity. It does
not: N-5 rests on the ground-return survey, not on a schedule comparison. **N-5 = verified-closed on
its own evidence.** The process breach is real and stands separately. Two different failures; keep
them apart.

### 3.3 A correction to this audit's own earlier pass

**R1 is a HEIGHT metric, not classification.** The 20 Sep roadmap audit treated it as classification
accuracy and attached a duplicate-work flag to it. Shrestha's `26fe207` is *classification* accuracy;
they measure different things and **are not duplicates**. The genuine duplicate pair is JP's later
`a86b491` against `26fe207`. Incident count unchanged at four. Corrected in place in
`reports/roadmap-audit-2026-09-20.md`.

### 3.4 ⚑ Six figures for one quantity — the highest-priority finding

| figure | recipe | artifact |
|---|---|---|
| **89.18 / 100.43 ms** | README primary; **the deck quotes this one** | honest, footnoted |
| 108.65 / 127.23 ms | `docs/handover-2026-09-02.md` | honest, footnoted |
| 89.58 ms pooled p99 | JP, CPU, `5/10/20/40`, pre-merge | ✅ `36a4dbe` |
| 87.69 warm / 101.70 cold | JP, CPU, `5/10/20/40`, post-merge | ✅ `6d16bb4` |
| 21.94 / 28.40 ms | Kaggle T4, CUDA, `5/10/50` — **closed R-b** | ❌ **none** |
| 67.42 / 100.18 ms "MISSES 10 Hz" | Kaggle T4, CUDA, `5/10/20/40` | ✅ committed, **uncited** |

README's D8 footnote is the right instinct — *"Quote this with its host attached… Neither is 'the'
frame latency until D8 picks one reference host and one command"* — **but it names two of the six.
The other four appeared after it was written.** The deck quotes the README figure, so the stalest
summary is the one facing the panel.

### 3.5 Good news, verified

The presentation material does **not** carry the retired numbers as claims. `69.8`, `98.3` and
`8.15→1.31` all appear — as explicit corrections. `07-CORRECTED-SCRIPT.md`: *"65.2%, never 69.8%…
an arithmetic error."* `02-WHAT-TO-PRESENT.md` carries a say-this-not-that table. That work was done
properly.

---

## 4. Phase 4 — the consolidated list

### 4.1 Purely JP's to decide or do — no blocker

| # | item | what it needs |
|---|---|---|
| **1** | **⚑ The six latency figures (§3.4)** — highest priority in this audit, above the band-invalidated reports | A decision between two shapes: **(a)** pick one reference figure + host as primary and footnote the rest with full recipes, or **(b)** present all live figures together with recipes and rewrite the footnote to say *"six candidate figures currently exist, here they are"* rather than quietly citing two. **This audit deliberately does not recommend which.** The deck quotes the stalest summary, so this is the one with a live external consequence. |
| 2 | **`reports/r1-accuracy-by-class-and-range-band.md`** | Re-verify or re-measure height RMSE on the current band |
| 3 | **`reports/r7-hazard-miss-rate.md`** | Same — drivability derives from height gradients |
| 4 | The `--semantics frnet` merge | Patch staged; needs Shrestha's agreement then application |
| 5 | D4, D5, D7, R-c, R-i | Standing decisions, unchanged |
| 6 | Push the 4 local commits | Outward-facing |

### 4.2 Needs a specific named person

**Shrestha**
- **`docs/defense-rehearsal-playbook.md` — panel-facing, band-invalidated.** Flagged separately in
  `pending-review/playbook-band-invalidated.md` because of the stakes.
- **S-6** contradicts his own D12 (§2.4).
- **R-d**: pin ruff in CI, reopen the row — 10 errors under 0.12.0, CI version unpinned.

> **[CORRECTION 2026-09-24 — the R-d finding above is WRONG and is retracted.]**
> The claim that R-d was a false closure came from running **ruff 0.12.0**, an older
> version than CI installs. Re-tested with **ruff 0.16.8** — the exact version CI
> installs — in a clean venv, `--no-cache`, against `Stxtics03/vrgrid-26@8acbfe9`:
> **`All checks passed`**. Shrestha's closure (*"`ruff check .` passes clean on `main`;
> it is a CI gate and it is green"*) is **true and verified**.
>
> `pyproject.toml` sets no `[tool.ruff.lint] select`, so ruff uses its **default rule
> set**, and that set changed **in both directions** between versions: 0.16.8 drops the
> `E7xx` family from defaults (which is why `main`'s `E731`/`E741`/`E702` vanish) while
> enforcing `I001`, `RUF100`, `PLW1510` and `RUF059`. "Newer is stricter" was the wrong
> model.
>
> **What is true instead:** `jp/p99-alloc-fixes` carries **69 errors under CI's own
> ruff** — real lint debt in JP's own files (`plan_regret_frnet_delta.py` 10,
> `whole_frame_bench.py` 6, `test_plan_regret_frnet_delta.py` 5, `rmse_baseline_probe.py`
> 4, …), not inherited from `main` and not version drift. That is JP's to clean up.
> Pinning ruff in `ci.yml` remains worth doing so it cannot drift, but it explains
> nothing here.

- **R-b**: the 21.94/28.40 row says *"the laptop"*; `research-log.md:561` calls the same numbers *"the
  T4 column"*. One of the two is wrong.
- **`configs/` review** (§3.1) — the change is sound; the missing three-way review is the issue.
- **N-3** cause still open; **S-3** is his next piece of work.

**Aakash** — D2 handover review (`docs/handover-2026-09-17-aakash.md`); D6 `kappa` 1/16 vs 1/12;
`src/grid/` and `src/eval/` are being worked by Shrestha in his absence.

**Pratyushi** — R-j's GPU half (needs a card); S-1, the fourth gate reason.

**Srinivas** — D10, ROS 2 adapter scope: the one assigned lane deliverable with no implementation.

**Hriday** — D3/R5 implementation once classes are decided; **R10**, which needs an owner or a
deliberate descope.

### 4.3 Needs the room

- **The tracker redesign (§0)** — the structural finding. Four identifier systems, two live
  collisions, ~30 unplanned items against 11 planned, five stale statuses in three passes.
- **S-4** — dynamic belief in `log_odds`, touches the frozen 12-byte cell struct.
- **D3** (sticky classes), **D6** (`kappa`), **D10** (ROS scope).
- **D12** — pin the projection to one thread, or replace the duplicate write with a deterministic
  nearest-wins reduction. Both move published numbers.

### 4.4 Safe to complete independently — Phase 5

No cross-lane touch, no design decision, pure verification and documentation:

1. Stage the playbook flag for Shrestha (`pending-review/`).
2. Mark the two band-invalidated JP reports in place with dated notes, so neither is read as current.
3. Add R6's unverifiability and R10's status to the permanent record rather than leaving them in
   audit files only.
4. Re-verify the remaining unverified closed rows not yet checked in any pass.
