# What to present, and how

*Condenses `docs/demo-runbook.md`, `docs/demo-safe-ranges.md` and the corrected
playbook into one thing you can hold on stage. Read `01-CRITICAL-FIXES.md` first
— three sentences in the current deck should not be spoken.*

---

## 1. The strategy, in one paragraph

You have a genuinely good engineering result (a memory bound that is structural
rather than measured, an accuracy claim tested across eleven sequences, a ghost
removal that works at any vehicle elevation) and one ambitious idea whose result
has not landed (plan regret). The winning move is **to be the team that discloses
its own limitations before the panel finds them.** Your `known-limitations.md` is
better than most teams' entire submission. Use it. Every time you volunteer an
unflattering number, the panel's confidence in your other numbers goes up, and
you have enough good numbers to spend some credibility buying that.

The losing move is to present the README's original framing, get one probing
question, and spend the rest of the session defending.

---

## 2. The narrative arc

Six beats. Each answers the question the last one raises.

| # | Beat | The question it answers | Time |
|---|---|---|---|
| 1 | The problem is physics, not storage | Why not just buy more RAM? | 1 min |
| 2 | Rings, derived from beam geometry | Why *these* resolutions? | 1.5 min |
| 3 | The bound is structural | Is 8.94 MB a bound or an average? | 1 min |
| 4 | Ghosts — the live toggle | Does it actually work? | 2 min |
| 5 | What it cost — ρ across 11 sequences | What did you lose by coarsening? | 1.5 min |
| 6 | What we have not proven | Are you being straight with us? | 1 min |

**Beat 1 is the one most teams get wrong.** Do not open with "LiDAR maps are
big." Open with the sensor. At 50 m consecutive laser rings strike the road
**10.8 m apart**. A uniform 5 cm grid at that range is 99.87% empty in a single
frame — it is not storing measurements, it is storing interpolation. That
reframes the whole project from "we compressed a map" to "we stopped storing
data the sensor never collected," which is a much harder claim to attack.

The derivation, if asked: `s_rad(r) = r²·Δφ / h_s`, with Δφ = 0.427° and
h_s = 1.73 m for the HDL-64E.

---

## 3. Slide skeleton

I have not seen `SIH26053_vrgrid.pptx` — upload it and I will map this onto your
actual slides. This is the target structure to check yours against.

| # | Slide | Must contain | Must NOT contain |
|---|---|---|---|
| 1 | Title | SIH26053, team, one-line claim | — |
| 2 | **The sensor problem** | the 10.8 m figure, the beam diagram, 99.87% | "maps are big" |
| 3 | The idea | ring diagram, 5/10/20/40, blind cone in red | — |
| 4 | Architecture | perception → grid → eval, ownership | implementation detail |
| 5 | **Memory** | 8.94 map / 29.06 total, both, side by side | either number alone |
| 6 | The maths that is load-bearing | law of total variance, `merge(split(c))==c` | all of `sih-math.md` |
| 7 | **Demo** | live `check`, then baked scenes | a video |
| 8 | Ghost removal | 13.5% of trail, 4.96 M cells, **429,012 spared** | a rate without the guard number |
| 9 | **Accuracy** | ρ ≈ 1.39 (1.16–1.53, n=11) at ring 1; ring 0 ρ 1.17 (1.13–1.29) — regenerated 2026-09-17, `known-limitations.md` §2b; per-ring RMSE | one sequence's best number |
| 10 | Curbs & potholes | ring-0 8.1–9.1 cm on all 11 | any detection *rate* |
| 11 | Segmentation | 90.3% / **65.2%**, and why it is not in the pipeline | 69.8% |
| 12 | Latency | p50 89.18 / p99 100.43, budget 100 | p99 rounded down |
| 13 | **Known limitations** | regret, ring-0 ρ, potholes, p99 | anything hidden |
| 14 | Prior art | Droeschel 2014, Triebel 2006, Losasso 2004 | "we invented foveation" |
| 15 | What's next | the Day-7 list from §7 below | vague "future work" |

**Slide 13 is not a weakness slide, it is your strongest slide.** Put it before
the summary, not in an appendix. Title it "What we have not proven."

---

## 4. Demo choreography

Everything below was verified end-to-end on 2026-09-03. One launcher drives it.

### Before you leave

```bash
./scripts/demo.sh check      # must end "OK -- the pipeline runs on real data."
./scripts/demo.sh bake       # ~3 min, writes demo/*.rrd (~1.2 GB, gitignored)
ls -lh demo/                 # six recordings
make test                    # main must be green
```

Copy `demo/` plus the Rerun viewer to the presenting machine. Playback needs
nothing else — no venv, no dataset, no 84.8 GB.

⚠️ **The one thing that will break it.** The loader wants the directory holding
`poses/` and `sequences/`, which in this checkout is `data/dataset`, not `data/`.
`demo.sh` resolves it. If you type a raw command, `export
VRGRID_DATA_ROOT=$PWD/data/dataset` first.

### On stage — six minutes

**① Open live, 30 seconds.** This is the credibility purchase.

```bash
./scripts/demo.sh check
```

Runs the real pipeline on two real frames and prints live counters
(`3,099 cells cleared, 11,056 spared by the current-return guard`). Ten seconds.
Everything after it is credibly the same system, and it pre-empts "so it's a
video" before anyone thinks it.

**Then the numbers, in the terminal:**

```bash
./scripts/demo.sh numbers
```

> *"The map is 8.94 megabytes, fixed at startup. A uniform 5 cm 2.5D grid over
> the same footprint is 192 megabytes; a dense 5 cm 3D voxel grid is 2.56
> gigabytes. Our total preallocated footprint including working buffers is
> 29.06."*

Then volunteer the sparse-3D row (~130–240 MB, ~15–27×) yourself. Leading with
286× alone reads as cherry-picking; handing them the unflattering baseline reads
as good faith, and it is the number they were about to ask for anyway.

**② Foveation, 90 seconds.**

```bash
./scripts/demo.sh foveation
```

Ring circles and the red blind cone track the vehicle; cell size steps
5 → 10 → 20 → 40 cm outward. **Point at the red circle.** Say the words: the
3.74 m blind cone is `unknown`, never `free`. Three occupancy states, and
`world/map/free` and `world/map/unknown` are separate entities on purpose. A
panel that hears you distinguish those knows you have thought about what the
robot does not know, which is the difference between a mapping project and a
safety project.

**③ Ghosts, 2 minutes. This is the moment.**

```bash
./scripts/demo.sh ghosts-off    # trails stay in the map
./scripts/demo.sh ghosts-on     # same 60 frames, trails gone
```

Two windows side by side beats flipping. Same sequence, same frames, same
schedule; the only difference is whether §10.4 runs.

> *"A moving car writes occupancy into every cell it passes through. Visibility
> cleanup removes what the current scan can see through — but it never clears a
> cell that has a return in the current scan, and that guard is what stops it
> eating fences, poles and sign posts."*

Quote **both** numbers: 13.5% of the trail removed and 4.96 M cells cleared on
seq 08, and **429,012 cells spared by the guard**. The second number is the
evidence that the cleanup is conservative rather than aggressive, and nobody
volunteers it.

⚠️ **Keep `--show-ghosts` terminal output off screen.** It prints `0 occupied
cells, 0 cleared, 0 protected` because that counter is only computed inside the
ghost-removal branch. The map is fine, the picture is correct, and you do not
want to explain a counter artifact live.

**④ Against a uniform grid, 60 seconds.**

```bash
./scripts/demo.sh dense3d
```

1,544 variable-resolution boxes against 290,448 dense voxels — 188.1× — at a
20 m footprint. **Say the disclaimer before they read it:** this render uses a
reduced 20 m footprint because that is what a dev machine can allocate; the
ratio on screen is a local illustration and the 286× is the full-grid byte ratio
from `memory_table.py`. The script's own docstring says so. Being the one who
says it first costs nothing.

**⑤ Optional, if they are engaged.**

```bash
./scripts/demo.sh features       # curbs, potholes, per-cell confidence
./scripts/demo.sh traffic        # seq 07, dense moving traffic
./scripts/demo.sh reflectivity   # seq 00, lane paint
```

Two guards on `features`: confidence is **a margin, not a probability** (nothing
is calibrated against outcomes), and curb/pothole counts have **no ground truth
to score against**.

### Failure modes

| Symptom | Fix |
|---|---|
| `FileNotFoundError: GT poses not found: data/poses/00.txt` | `export VRGRID_DATA_ROOT=$PWD/data/dataset` |
| Viewer never appears | play the baked `.rrd`; `DISPLAY` is `:0.0` on this box |
| `PatchWorkpp` banner then long pause | normal, ~4 s init; baked scenes skip it |
| `[!] ground: SEMANTIC-CLASS FALLBACK` | `pypatchworkpp` missing. **Do not demo the ground layer on it** — the fallback admits embankments. Rebuild from the git clone, not PyPI |
| Ghost scene shows nothing moving | wrong frames; seq 00 needs ~0–60, best single frame is 10 |
| Scene slow live | 160 frames ≈ 35 s — play the baked file |

---

## 5. Speaker split

Six people, and a panel notices when one person answers everything.

| Beat | Speaker | Backup |
|---|---|---|
| Problem, sensor physics, rings | Srinivas | Hriday |
| Architecture, memory bound, demo driving | Shrestha | Aakash |
| Ghost removal, perception, segmentation | JP | Hriday |
| Accuracy, ρ, plan regret | Aakash | Pratyushi |
| Limitations slide | **whoever is most senior** | — |
| Prior art questions | Srinivas | — |

**Whoever delivers the limitations slide should be the person with the most
authority in the room.** Disclosure delivered by a junior member reads as a
slip; delivered by the lead it reads as rigour.

**One rule: if a question lands outside your directory, hand it over by name.**
"That's Aakash's — Aakash?" A team that routes cleanly looks like a team that
built something together.

---

## 6. Say this, not that

| Do not say | Say instead |
|---|---|
| "We proved the compression is free" | "We built the evaluation to test that. On our only query it does not show it, and we know why the query cannot." |
| "69.8% mIoU" | "65.2% mIoU over the 15 classes present" |
| "08 was our unseen test set" | "07 and 08 were what downloaded first. That's why we re-ran across all eleven and quote the range." |
| "GPU-accelerated" | "GPU-shaped, CPU reference implementation, every number measured on it" |
| "We detect potholes" | "We detect pothole-shaped features. There is no ground truth in SemanticKITTI to score a rate against." |
| "ρ = 1.39" | "ρ ≈ 1.39, range 1.16–1.53, n = 11" (was 1.45 / 1.26–1.59 before the 17 Sep regeneration) |
| "8.94 MB" (alone) | "8.94 MB of map, 29.06 total preallocated" |
| "Real-time at 10 Hz" | "Median 89 ms against a 100 ms budget. p99 is 100.4, so we miss one frame in a hundred by 0.4 ms." |
| "FRNet didn't work" | "FRNet works at 90.3%. We kept it out of the pipeline so segmentation error doesn't contaminate the mapping result." |

---

## 7. The Day-7 slide

Have a concrete next-steps list. Vague future work reads as no plan.

1. **A planning query that can discriminate resolution.** The current one is a
   longitudinal lane and structurally cannot. This is the item standing between
   the project and its headline claim.
2. **Ring-0 ρ.** Store the sum of squares in the reference map; roughly fifteen
   lines with a two-day tail because it invalidates every cached `M*`. Expected
   to score ρ 1.01–1.24, the best of any ring.
3. **A real device path.** `array_module()` is already the seam; move the arrays
   to cupy and re-measure.
4. **Live semantics** via an mmdet3d FRNet install, reported *alongside* the
   GT-label result rather than replacing it.
5. **The p99.** Visibility cleanup is 26 ms of the 89 and is the most parallel
   stage in the pipeline.
