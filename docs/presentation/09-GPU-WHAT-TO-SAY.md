# Every GPU claim — what you would have said, and what to say instead

*Two parts. **Part A** is the deck and the script: ten claims, each written out
as the full sentence you would actually have spoken, then the line to say
instead. **Part B** is the live demo: eight moments where GPU comes up while
you're standing at the screen, in the same format.*

**Why this matters more than any other correction:** slide 6 hands the judges
`github.com/Stxtics03/vrgrid` and calls it public. A judge who greps that repo
for `.cu` or `.cpp` finds nothing at all. Wrong numbers are survivable. A
fabricated implementation is not.

---

## The one-line summary

> There is no CUDA and no C++ anywhere in this project. `src/gpu/` is numpy on
> CPU. The only thing that touches a GPU is FRNet inference through PyTorch,
> and FRNet is not in the mapping pipeline.

```
$ find . -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.hpp"
(nothing)

$ ls include/vrgrid/          ← the "C++17 frozen interfaces"
__init__.py   api.py   cell.py
```

---

# PART A — the deck and the script

## ① The section heading

> 🗣️ **You would have said:**
> *"Fifth stage — GPU acceleration."*

**→ Say instead:**

> *"Fifth — the deterministic parallel back end."*

*Rename the slide section too. "GPU acceleration" sets up every question you
can't answer; "deterministic parallel back end" sets up every question you can.*

---

## ② The kernels

> 🗣️ **You would have said:**
> *"All of this runs as CUDA kernels — projection, fusion, split, merge — with a
> conservative max/min pyramid, and the whole grid rebuilds in 2.45
> milliseconds."*

**❌ Reality:** `src/gpu/kernels.py` is numpy. Its own docstring says the atomic
path behaves *"exactly as a CUDA kernel would issue atomicAdd"* — a statement
about the fidelity of a reference implementation, not about a kernel that runs.

**→ Say instead:**

> *"The back end is written the way GPU code is written, and I want to be precise
> about what that means. It's a numpy reference implementation on CPU, with a
> device seam in the allocator so the arrays can move to cupy without touching
> the kernels. Every number we quote is measured on that CPU path.*
>
> *The design decisions are the parallel ones. Structure-of-arrays, so a kernel
> reading one field touches contiguous memory. Zero allocation in this back end's
> frame loop — that took us from 8.15 megabytes per frame to 1.31, for the mapping
> half; the perception front end is not instrumented yet.* A toroidal ego-motion
> shift that moves the origin instead of the data: 0.04 milliseconds against 15.2
> for the obvious version. And two scatter paths, a sorted one and an atomic one,
> asserted bit-identical against each other — if they ever diverge, the
> optimisation is the bug."*

---

## ③ The determinism argument — **keep this, it's your best GPU line**

> 🗣️ **You would have said:**
> *"NumPy structure-of-arrays with fixed-point integer atomics — so integer
> addition stays associative and results are deterministic."*

**✅ True and load-bearing.** Do not cut it. But say the *reason*, not just the
property — the reason is what demonstrates you understand parallel correctness.

**→ Say instead:**

> *"Integer fixed-point accumulation rather than float atomics — because IEEE
> addition isn't associative and atomic adds complete out of order, so a float
> map changes between runs and you can't bisect a bug whose location moves.
> Heights are quantised to a centimetre anyway, so everything accumulates as
> int32, which is exactly associative. The determinism test blocks every merge to
> the codebase."*

*This is the single most impressive sentence available to you on this topic.
"We wrote CUDA" is a claim any team can make. "We chose integer accumulation
because a nondeterministic map can't be debugged" is a claim that shows you know
why parallel code is hard.*

⚠️ Minor: "atomics" is a design term here — `scatter_atomic` simulates
`atomicAdd` in numpy. If pressed on the word: *"the atomic path, which is our
reference for what a CUDA atomicAdd would do; the default path is a sort."*

---

## ④ The 2.45 ms

> 🗣️ **You would have said:**
> *"…with a conservative max/min pyramid, and the whole grid rebuilds in 2.45
> milliseconds."*
> — and it's on your memorised-numbers list as *"2.45 ms, rebuild time."*

**❌ Two misattributions in one line.** That figure is the *conservative
pyramid's* rebuild over 910,000 slots — not "the whole grid" — and **the pyramid
is off by default.** Enabling it moves the preallocated total from 29.06 to
32.17 MB. You'd be quoting a stretch feature's number as your throughput
headline.

**→ Delete it** from the deck, the script and the memorised list. Replace with ⑤.

If the pyramid comes up on its own:

> *"There's a conservative max/min pyramid for hierarchical queries — it rebuilds
> in 2.45 milliseconds over 910,000 slots, about 32× headroom against the frame
> budget. It's off by default, because turning it on moves our declared memory
> total, and a number on a slide shouldn't move because a default did."*

---

## ⑤ The number that's missing entirely

> 🗣️ **You would have said:**
> *…nothing.* Six slides and no frame-time claim anywhere. The only timing
> number you had was ④, which is a stretch feature's.

**"Does it run in real time?" is a certainty.**

**→ Add this, and own the miss:**

> *"End to end on 200 frames of sequence 08, the frame is 89 milliseconds at the
> median against a 100 millisecond budget — so 10 Hz with about 11 milliseconds
> to spare. The p99 is 100.4, so one frame in a hundred misses by four tenths of
> a millisecond. That's a real miss and we're not rounding it away. Two things
> about it: the whole pipeline is single-threaded numpy on CPU with no kernel
> written, and the largest stage is visibility cleanup at 26 of those 89
> milliseconds — which is the most parallel thing in the system. So the headroom
> is in a known place."*

*Your own `src/gpu/timing.py` says a pipeline that misses one frame in a hundred
has dropped a frame of obstacles. If a judge finds that line in your repo after
you claimed clean 10 Hz, it's much worse than saying it yourself.*

---

## ⑥ The stack line

> 🗣️ **You would have said:**
> *"Under the hood: Python 3.11 and C++17 with CUDA kernels, NumPy
> structure-of-arrays with fixed-point integer atomics, SemanticKITTI and
> Patchwork++ for data, Rerun for visualisation, and GitHub Actions with pytest
> enforcing CI gates that block every merge."*

**❌ Two of the first three are false.** `include/vrgrid/` — the "C++17 frozen
interfaces" — contains `__init__.py`, `api.py`, `cell.py`.

**→ Say instead:**

> *"Python 3.11, NumPy structure-of-arrays with int32 fixed-point accumulation,
> a cupy device seam in the allocator, and PyTorch with CUDA for the segmentation
> model. SemanticKITTI and Patchwork++ for data, Rerun for visualisation, GitHub
> Actions and pytest enforcing CI gates that block every merge."*

---

## ⑦ Jetson

> 🗣️ **You would have said:**
> *"And it runs on a Jetson-class GPU — no discrete card required."*

**❌ Never measured.** No Jetson appears anywhere in the repo, and there is no GPU
code to run on one.

**→ Say instead:**

> *"It's designed for Jetson-class deployment — 8.94 megabytes of map, no dynamic
> allocation in the frame loop, and no dependency on a discrete card — though we
> haven't yet measured on target hardware."*

*One added half-sentence. "Designed for" plus an honest caveat reads better than
an assertion you'd have to walk back.*

---

## ⑧ "Where is the GPU work, then?"

> 🗣️ **You would have said:**
> *…you had no answer prepared, because the deck asserted the work was the
> mapping kernels.*

**→ This is your answer, and it's genuinely strong:**

> *"The real GPU work is in the segmentation model. We found FRNet's forward pass
> was ten and a half seconds a frame, with about ninety percent of that inside a
> Python for-loop — its scatter_max and scatter_mean were looping once per output
> slot, twenty-five thousand iterations over a hundred-and-twenty-four-thousand-row
> tensor, seven times per forward.*
>
> *We replaced them with torch.scatter_reduce, which does the same reduction
> natively. 1408 times faster on max and 541 on mean, at real shapes on CUDA.
> That took a fine-tune from three and a quarter hours to two minutes, and an
> evaluation run from thirty-five minutes to about one.*
>
> *And we verified the numerics rather than assuming them: max is bit-identical
> on both CPU and CUDA, and mean differs by up to two float32 ulp on about forty
> percent of slots on CUDA, because the native kernel sums a slot's rows in a
> different order. Our verifier gates max at exactly zero and mean at a stated ulp
> bound, rather than claiming a bit-identity that isn't there."*

*The ulp detail is what makes a panel believe the rest.*

---

## ⑨ "Why didn't you write CUDA?"

> 🗣️ **You would have said:**
> *…nothing — under the old framing this question couldn't arise, so no answer
> existed.*

**→ Have this ready. It turns the absence into a decision:**

> *"Because the two load-bearing claims in this project are determinism and a
> hard memory bound, and a GPU makes both harder, not easier. Float atomics
> complete out of order and IEEE addition isn't associative, so the map would
> change between runs. And the memory bound is the thing we'd have to defend on a
> Jetson, where there's less headroom, not more.*
>
> *So we built the CPU reference first, made it bit-identical and bounded, and
> left a device seam in the allocator so the arrays can move without touching the
> kernels. Porting is the next step and it's constrained by the determinism
> guarantee — which is the right order to do it in."*

---

## ⑩ "Then why is the module called `src/gpu/`?"

> 🗣️ **You would have said:**
> *"Because that's where the CUDA kernels live."* — the answer that breaks, since
> they can open it.

**→ Say instead, short and unbothered:**

> *"Because that's where the device port lands. Everything in it is written to
> the constraints a kernel has — coalesced access, no allocation in the loop,
> deterministic reduction order. It's a reference implementation, and the seam is
> array_module() in the allocator."*

---

# PART B — during the live demo

*Eight moments, same format. The demo is where the GPU question most often
arrives unprompted, because the screen looks like GPU work even though it isn't.*

**The trap running through all of these:** the Rerun viewer renders on the GPU via
wgpu. Judges will be watching smooth, fast 3D. Nothing visible distinguishes
"the visualiser is on the GPU" from "the mapping is on the GPU" — so a CUDA claim
appears confirmed by the screen, right up until someone opens the repo. **Get in
front of it once, early, and it stops being a question for the rest of the
demo.**

---

## ⑪ Opening — `./scripts/demo.sh check`

> 🗣️ **You would have said:**
> *"This is the real pipeline running on two real frames — you can see the live
> counters."*

**→ Say the same thing, plus one clause:**

> *"This is the real pipeline on two real frames — 3,099 cells cleared, 11,056
> spared by the current-return guard. And that's running on CPU, single-threaded
> numpy. The visualiser you're about to see renders on the GPU, but the mapping
> itself doesn't — I'll come back to why."*

*Ten seconds, at the one moment when nobody is yet suspicious. It buys you the
rest of the demo, and makes everything after it credibly the same system.*

---

## ⑫ The foveation scene — it plays smoothly

> 🗣️ **You would have said:**
> *"Here you can see the ring circles and the blind cone tracking the vehicle."*

**❌ The risk isn't what you say, it's what they infer.** Smooth 3D at speed reads
as GPU compute. And it isn't — you're playing a **baked recording**.

**→ Add, once:**

> *"This is a baked recording, so playback speed here isn't our frame time. The
> measured frame is 89 milliseconds at the median, and I can show the full timing
> table if it's useful."*

*Skip this and a later "so what frame rate was that?" has you answering about a
rendering speed as though it were a compute number.*

---

## ⑬ The ghost scene — the 26 ms stage is on screen

> 🗣️ **You would have said:**
> *"Same sixty frames, same schedule — the only difference is whether visibility
> cleanup runs."*

**→ Good as is. This is where your one latency admission belongs, because the
thing causing it is literally on the screen:**

> *"Same sixty frames, same schedule — the only difference is whether visibility
> cleanup runs. 13.5% of the trail removed, 4.96 million cells cleared, and
> 429,000 cells spared by the guard that stops it eating fences and poles.*
>
> *And this stage is our single biggest cost — 26 milliseconds of an 89
> millisecond frame. It's also the most parallel thing in the system: every cell
> is an independent range-image lookup. So when we port to a device, this is the
> first thing that moves."*

*You've turned your worst latency number into a forward-looking one, standing in
front of a demo that shows exactly why it costs what it costs.*

---

## ⑭ The dense3d comparison — 290,448 voxels rendering at once

> 🗣️ **You would have said:**
> *"1,544 variable-resolution boxes against 290,448 dense voxels — 188 times."*

**❌ Two things to pre-empt**, and say both before they're asked:

**→ Say instead:**

> *"1,544 variable-resolution boxes against 290,448 dense voxels. Two things
> before you ask. This render uses a reduced 20-metre footprint, because that's
> what a dev machine can allocate — so the 188× on screen is a local
> illustration, and the 286× on our slide is the full-grid byte ratio from
> memory_table.py. And drawing a quarter of a million voxels is the viewer's GPU
> doing work, not ours; our side of this comparison is the 1,544."*

*The script's own docstring says the first part. Saying it before they read it
costs nothing.*

---

## ⑮ "Can we see GPU utilisation?" / someone opens `nvidia-smi`

> 🗣️ **You would have said:**
> *…this is the moment a CUDA claim dies in the room, in front of everyone.*

**→ If you've already said ⑪, it's a non-event:**

> *"You'll see the viewer using it and nothing else — the mapping is CPU. If you
> want to see us actually load the card, I can run the segmentation model; that's
> where our GPU work is."*

*Then run it if there's time. `scripts/frnet_eval.py --frames 200 --fast-scatter`
is about a minute and genuinely loads the GPU.*

---

## ⑯ `./scripts/demo.sh numbers` — the memory table

> 🗣️ **You would have said:**
> *"8.94 megabytes against 192 for uniform 2.5D and 2.56 gigabytes for dense
> 3D."*

**❌ The counter on screen shows 29.06, not 8.94.** Both surfacing unexplained
reads as cherry-picking.

**→ Say instead:**

> *"Two numbers, and they answer different questions. The map is 8.94 megabytes —
> that's what the ratios are computed on, and they're pure cell-count ratios
> against 192 for uniform 2.5D and 2.56 gigabytes for dense 3D. The total we
> commit at startup is 29.06, because we preallocate every working buffer rather
> than allocating in the frame loop. Against a uniform 10-centimetre grid covering
> the same ground, measured the same way, that's 29.06 against 78.50.*
>
> *And a baseline that doesn't fault in its pages isn't a baseline — np.zeros
> gives you copy-on-write zero pages, so a naive 2.56-gigabyte allocation shows
> zero resident. We touch a byte per page. Claimed against resident: ours 27.86
> to 27.98."*

*That last part is a genuinely sophisticated detail and it takes eight seconds.*

---

## ⑰ If the demo is slow, or a scene stutters

> 🗣️ **You would have said:**
> *…probably nothing, and let it look like a performance problem.*

**→ Say instead, without apologising:**

> *"That's the viewer streaming a recording, not our frame time. If it's useful I
> can show the timing table — 89 milliseconds median, 100.4 at p99, per stage."*

*A live scene is ~160 frames at roughly 35 seconds. Play the baked file.*

---

## ⑱ If a judge says "so this isn't really GPU-accelerated"

> 🗣️ **You would have said:**
> *…defended it, because the deck said it was.* That's the losing thirty seconds.

**→ Agree immediately, then redirect:**

> *"Correct — the mapping path isn't, and we should be precise about that. It's a
> CPU reference implementation and every number we quote is measured on it. It's
> built to port: structure-of-arrays, no allocation in the loop, deterministic
> reduction order, and a device seam in the allocator. The GPU work we've actually
> done is in the segmentation model, where we got 1408× on a reduction kernel.*
>
> *We took determinism and the memory bound as the load-bearing claims, and a GPU
> makes both harder rather than easier — so we built the reference first. Porting
> is next and it's constrained by the determinism guarantee."*

*Agreeing instantly is what defuses this. A team that concedes a precise point
and keeps its footing reads as confident; a team that argues reads as caught.*

---

# What you should actually say — the consolidated version

*If you take nothing else from this file, take this. It replaces slide 3
section 5 in full and runs about 45 seconds.*

> *"The back end is written the way GPU code is written, and I want to be precise
> about what that means. It's a numpy reference implementation on CPU, with a
> device seam in the allocator so the arrays can move to cupy without touching
> the kernels. Every number we quote is measured on that CPU path.*
>
> *The design decisions are all the parallel ones. Structure-of-arrays, so a
> kernel reading one field touches contiguous memory. Integer fixed-point
> accumulation rather than float atomics — because IEEE addition isn't
> associative and atomic adds complete out of order, so a float map changes
> between runs and you can't bisect a bug whose location moves. Zero allocation
> in this back end's frame loop, which took us from 8.15 megabytes per frame to
> 1.31 — the mapping half; the front end is not instrumented yet. And a
> toroidal ego-motion shift that moves the origin instead of the data: 0.04
> milliseconds against 15.2 for the obvious version.*
>
> *End to end on 200 frames of sequence 08, the frame is 89 milliseconds at the
> median against a 100 millisecond budget. The p99 is 100.4 — so one frame in a
> hundred misses 10 Hz by four tenths of a millisecond. That's a real miss and
> we're not rounding it away. The largest stage is visibility cleanup at 26 of
> those 89 milliseconds, and it's the most parallel thing in the system, so we
> know where the headroom is.*
>
> *The one place we do run on a GPU is the segmentation model, and there we
> replaced seven Python reduction loops with native scatter_reduce — 1408 times
> faster on max, 541 on mean, at real shapes on CUDA."*

---

## Numbers to swap in your memorised list

| ❌ Drop | ✅ Add |
|---|---|
| 2.45 ms rebuild | **89.18 / 100.43 ms** — frame p50 / p99, 100 ms budget |
| | **8.15 → 1.31 MB/frame** — allocation removed from the **back end's** loop (front end not instrumented; CI test covers retained growth, not churn) |
| | **0.04 vs 15.2 ms** — toroidal shift vs O(area) scroll |
| | **6.65 vs 20.56 ms** — sorted vs atomic scatter, p50 at 120k returns |
| | **1408× / 541×** — FRNet scatter_reduce on CUDA |
| | **26 of 89 ms** — visibility cleanup, where the headroom is |

---

## Checklist

**Deck:**
- [ ] Slide 3 — rename section 5 heading (①)
- [ ] Slide 3 — replace the "CUDA kernels" line (②)
- [ ] Slide 3 — keep the int32/associativity line, expand the reason (③)
- [ ] Slide 3 — delete "2.45 ms rebuild" (④)
- [ ] Slide 3 — add frame latency p50/p99 (⑤)
- [ ] Slide 3 — replace the stack line (⑥)
- [ ] Slide 4 — qualify Jetson (⑦)

**Script:** swap in the consolidated paragraph; swap the memorised numbers.

**Demo — rehearse these three, they're inferred rather than asked:**
- [ ] ⑪ the opening clause — say it once, early, and the rest gets easier
- [ ] ⑭ the dense3d double disclaimer
- [ ] ⑯ the 8.94 / 29.06 sentence
