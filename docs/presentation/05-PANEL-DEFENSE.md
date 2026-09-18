# Panel defense — corrected

*Supersedes `docs/defense-rehearsal-playbook.md` for **Q6** and **Q8**, which
carry claims contradicted by the project's own evidence. Q1–Q5 and Q7 there are
sound and are summarised here; read the original for their full wording.*

---

## The golden rule

**Acknowledge the literature immediately, state the distinction precisely, back
it with a number from a script.** Never claim you invented adaptive resolution,
foveation, or multi-level cells. Claim the composition, the variance-honest
split/merge guarantees, the preallocated bound, and the evaluation *methodology*.

**Second rule, and it is the one that wins rooms:** volunteer the unflattering
number before it is asked for. You have enough good results to spend some
credibility buying trust, and trust is what carries the results you cannot fully
prove yet.

---

## The four questions that can hurt you

These are not in the old playbook, and they are the likeliest hard questions
given what is actually in your repo.

### ⚠️ Q-A: "Your README says this proves the compression doesn't change the plan. Show me."

**This is the question to rehearse hardest.** Do not defend the sentence.

> *"That sentence is ahead of our evidence and we're changing it. Here is what we
> actually have. The metric is built and three real defects in it are fixed — a
> resolution-dependent handicap that was penalising our fine rings, two sides of
> the equation evaluated on different lattices, and a class penalty charged on
> one side only. After all three, on sequence 08 at a matched extent, our
> schedule reads 0.488 and a uniform 20 cm map at essentially the same memory
> reads 0.251. It plans better than we do.*
>
> *The reason is that we have exactly one planning query and it is a longitudinal
> lane down the middle of the window. A lane query rewards a map that is
> uniformly adequate along one line. It cannot reward a map for being sharp where
> the vehicle is looking, which is the entire thesis. So that is not evidence the
> thesis is wrong, it is evidence this query can't test it — and designing one
> that can is the first item on our next list.*
>
> *What does hold: our two schedules produce identical plans despite 5.4 MB
> between them, and matched to the same ground we cost 29.06 MB against uniform
> 10 cm's 78.50."*

**If they push: "so your headline result failed?"**

> *"Our headline result is the memory bound and the per-ring accuracy, and both
> are measured across eleven sequences. The regret evaluation is the ambitious
> part and it is honest work in progress. We'd rather show you the plot that
> doesn't cooperate than one we cherry-picked a window for — and we can, because
> the script refuses to draw a monotone story over those rows by design."*

### ⚠️ Q-B: "Where is the GPU code?"

> *"There is no CUDA kernel in the mapping path, and I want to be precise about
> that. The pipeline is a CPU reference implementation in numpy, and every
> latency number we quote is measured on it. It is written the way GPU code is
> written — structure-of-arrays for coalesced access, integer fixed-point
> accumulation instead of float atomics, zero allocation in the frame loop, and a
> device seam in the allocator so the arrays can move to cupy without touching
> the kernels.*
>
> *We didn't write CUDA because the two load-bearing claims are determinism and a
> hard memory bound, and both are harder to hold on a GPU rather than easier —
> float atomics complete out of order and IEEE addition isn't associative, so the
> map would change between runs. The place we did do GPU work is the segmentation
> network: we replaced seven Python reduction loops with `torch.scatter_reduce_`
> and measured 1408× on max and 541× on mean at real shapes on CUDA, which took a
> fine-tune from 3.3 hours to 2.2 minutes."*

### ⚠️ Q-C: "Did you tune on your test set?"

**The old playbook's answer claims a clean three-way split. Your own
`known-limitations.md` §2b contradicts it. Do not use it.**

> *"Not a clean split, and it's worth being exact. Sequences 07 and 08 were the
> first two that finished downloading and most development happened against them.
> Once the full 84.8 GB landed we re-ran the accuracy table across all eleven
> labelled sequences and we report the distribution rather than our best number:
> ρ median 1.39, range 1.16 to 1.53, n equals 11 — and the finest ring, ring 0,
> at 1.17 once spread counts the variance inside each 5 cm cell.*
>
> *And we flag in our own limitations document that 07 and 08 sit at the good end
> of that range — 1.24 and 1.22 against a median of 1.39 — which is exactly why
> we quote the range rather than a sequence.*
>
> ⚑ *Regenerated 2026-09-17 (`known-limitations.md` §2b); the 2 Sep figures were
> 1.45, 1.26–1.59, and 1.32 / 1.30. The between-cell figures are quoted here
> because they are the conservative ones.*
>
> *The one place a genuine hold-out exists is the segmentation model: the FRNet
> checkpoint was trained on 00 through 10 excluding 08, and every FRNet number we
> report is measured on 08."*

### ⚠️ Q-D: "You said 69.8% mIoU. I get a different number."

Only arises if item 1 in `01-CRITICAL-FIXES.md` was not done. If it was, and
someone asks about the discrepancy with an older document:

> *"65.2 percent, over the fifteen classes present in those 200 frames. An
> earlier draft said 69.8 and that was our arithmetic error — we divided by
> fourteen, dropping `other-ground`, which has 150 ground-truth points across the
> run and an IoU of zero. It's present in the data so it counts. We caught it in
> our own audit on 4 September and corrected downward. Point accuracy is
> unaffected at 90.3."*

---

## The six from the original playbook

Sound as written. Summarised; read `docs/defense-rehearsal-playbook.md` for the
full text.

**Q1 — "Isn't this just a clipmap, or Droeschel 2014?"** *(Srinivas)*
Yes on the lineage — cite Losasso & Hoppe 2004 and Droeschel 2014 yourself.
Droeschel targeted 3D MAV pose estimation and had no 2.5D elevation fusion, no
online semantic refinement, and no variance-conserving split/merge under a
compile-time bound. ⚠️ **Drop the phrase "and the Plan Regret proof" from the
golden rule as written there** — say "and the plan-sensitivity evaluation".

**Q2 — "Planners use uniform grids; don't you lose the savings on decompression?"**
Three reasons no: O(1) native multi-resolution `query(x, y)`; if a legacy planner
demands uniform, unroll only a 10 × 10 m local corridor at <1 MB; traversability
is pre-baked into a per-cell bitfield so there is no runtime costmap build.

**Q3 — "Why not OctoMap or Wavemap?"**
Ground vehicles navigate a 2D surface manifold; >98% of 3D voxel space outdoors
is empty air. Trees need dynamic heap allocation (~2.56 GB at 5 cm), which
produces the unpredictable p99 a safety case cannot absorb, and pointer chasing
gives warp divergence and uncoalesced reads. Flat 2.5D gives O(1) indexing and a
compile-time bound.

**Q4 — "Ring 0 is 5 cm, same as a uniform grid. Where's the gain?"**
The metric is accuracy per megabyte. Ring 0 delivers identical 5 cm fidelity in
the safety-critical near field; the saving is in not paying 192 MB to hold 5 cm
out to 100 m where the beams are 10.8 m apart.

**Q5 — "Can you detect a 30 cm pothole at 50 m?"**
No, and neither can a uniform 5 cm grid with a real sensor —
`r_max = √(W·h_s/Δφ) ≈ 8.3 m`. Beyond that, cells are marked `unknown`, never
`free`. ⚠️ **Do not claim a detection rate.** SemanticKITTI has no curb or pothole
ground truth. Quote counts and the ring-0 8.1–9.1 cm consistency across eleven
sequences, and say the caveat before you are asked.

**Q7 — "Your traversability predicate has a scale problem."**
It did, it was a real defect, found and fixed 2 September. Differenced over one
cell, a 12 cm kerb reads as gradient 1.200 at 5 cm and 0.240 at 25 cm against one
frozen threshold of 0.364 — a wall on fine rings and flat ground on coarse ones.
Eq. (22a) now differences over a **fixed 0.50 m physical baseline**, and that
baseline is bounded by the scene rather than chosen: above `0.12/tan 20° = 0.33 m`
so the kerb reads passable everywhere, below `0.40/tan 20° = 1.10 m` so a 40 cm
pothole rim still fails.

---

## Q8 — corrected

**"You claim a compile-time memory bound. Is it a bound, or a number you measured
once?"**

⚠️ The original answer argues *"we did not fit a bigger number, because the peak
scales with sequence length."* **`known-limitations.md` §5 refutes exactly that:
sequence 00 is the longest (4,541 frames) and has the lowest peak (278,226).**
Use this instead:

> *"For the map itself it's structural — every array is allocated at startup and
> nothing in the frame loop grows. For the ghost-removal scratch it was a
> measured guess until 2 September, and we changed it because the guess was wrong
> by a factor of three.*
>
> *The cap on candidate cells was a provisional 150,000. Measured on whole
> sequences that drops 52.3% of sequence 07's peak occupied set and 67.1% of 08's
> — and the failure is silent: dropped cells keep their occupancy, are never
> tested, and can't appear in the cleared count. So a truncating run prints a
> healthy ghost number while the map keeps its ghosts.*
>
> *We didn't just fit a bigger number, because we couldn't find a predictor to
> fit against. The peak does not track sequence length — sequence 00 is our
> longest and has the lowest peak. It tracks scene density, it varies 1.64×
> across the three sequences we measured, and nineteen are unmeasured. That
> unpredictability is the argument for making truncation impossible by
> construction rather than unlikely by measurement: the cap is now the grid's own
> slot count, because the occupied set cannot exceed the grid."*

**Expect the follow-up: "doesn't that cost you memory?"**

> *"Yes — 58.24 MB of scratch instead of 9.60. It's working memory, not map
> memory, so the cell-count ratios in the report are unaffected, and it's off by
> default. If a smaller declared total is wanted, an explicit 600,000 is 38.40 MB
> and covers every sequence we measured at 1.32× the observed peak."*

Say the number. Do not let them find it.

---

## Q9 — new: "Which memory number is real, 8.94 or 29.06?"

> *"Both, and they answer different questions. 8.94 megabytes is the map —
> 745,000 cells at 12 bytes — and that's what the compression ratios are computed
> on, because they're pure cell-count ratios against a uniform 5 cm 2.5D grid at
> 192 MB and a dense 5 cm 3D voxel grid at 2.56 GB. 29.06 is what we commit at
> startup in total, because we preallocate every working buffer rather than
> allocating in the frame loop. Against a uniform 10 cm grid covering the same
> ground, measured the same way, that's 29.06 against 78.50."*

---

## Q10 — new: "Do you hit real-time?"

> *"On the median, yes, with margin. End-to-end on 200 frames of sequence 08 the
> p50 is 89 milliseconds against a 100 millisecond budget. The p99 is 100.4, so
> we miss one frame in a hundred by four tenths of a millisecond, and our own
> timing module's documentation says a pipeline that misses one frame in a
> hundred has dropped a frame of obstacles. We're not going to round that away.*
>
> *Two things about it. The whole pipeline is single-threaded numpy on CPU with
> no kernel written. And the largest stage is visibility cleanup at 26 of the 89
> milliseconds, which is the most parallel thing in the system — so the headroom
> is in a known place."*

---

## Q11 — new: "Why don't you use your own segmentation network in the pipeline?"

> *"Deliberately. Semantic and motion labels come from the SemanticKITTI label
> files, which means the mapping contribution is evaluated independently of
> segmentation quality — a poor coarsening ratio can't be blamed on a
> mis-segmented kerb, and a good one can't be credited to a strong segmenter.*
>
> *The network exists and works: 90.3% point accuracy, 65.2% mIoU on 200
> held-out frames of sequence 08, against the paper's 73.3%. It's reported
> alongside the map, never swapped into it. We also fine-tuned it three ways and
> rejected all three on measurement — the checkpoint was already trained on this
> data, so there was no domain gap to close, and the gain on the five classes our
> map actually consults was +0.3, inside noise."*

---

## Rules for the room

1. **Route by name.** "That's Aakash's lane — Aakash?" A team that routes cleanly
   looks like a team that built something together.
2. **If you do not know, say so and say who would.** Never invent a number. Every
   number you have comes from a script and you can name the script.
3. **Volunteer the caveat before the question.** Sparse-3D baseline, the reduced
   20 m footprint on the dense3d scene, no pothole ground truth, the p99. Every
   one of these costs a sentence and buys the room.
4. **Never weaken a claim under pressure by inventing a hedge.** The honest
   version is already written down. Use the written one.
5. **If a demo fails, go to the numbers.** `./scripts/demo.sh numbers` needs no
   display and no dataset.
