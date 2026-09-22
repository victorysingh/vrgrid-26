# pending-review: your new D9 is renumbered D12 — and S-6 now contradicts it

**For Shrestha. Nothing edited in your files; this is a note so the two copies of `OPEN-ITEMS.md`
reconcile cleanly when they merge.** Raised 2026-09-23 during a full-project audit.

## 1. The `range_interpolation` work is excellent — separately from the numbering

`8acbfe9` is the strongest piece of debugging on this project so far, and it is worth saying so
plainly rather than burying it under the admin below:

- You took a flag that only said *"this attribution looks wrong"* and **re-measured it yourself**
  rather than accepting or dismissing it.
- You found that the follow-up guess — the scatter loops — was **also** wrong, and said so:
  *"at the real shapes and 16 threads `scatter_mean`, `scatter_max` and `torch.unique(dim=0)` are all
  bit-identical run to run."* Ruling out your own next hypothesis is the part most people skip.
- You localised it properly, by hooking every module across two forwards, to
  `voxel_encoder.pre_norm` at max |d| 9.5 — and read the **magnitude** as the tell that something
  discrete was changing, rather than a float-reordering ulp.
- The root cause is real and specific: `frnet.py:156`'s duplicate-index assignment relying on
  last-write-wins, which PyTorch documents as nondeterministic, with ~35% of a scan's points landing
  on an already-claimed pixel. `threads=16` → image identical **False**; `threads=1` → **True**.
- **And you disclosed the blast radius instead of minimising it:** *"Every frnet-path number — 86.5%
  agreement, 90.3% pooled, the by-range table, the fp16 comparison — was measured in that state."*
- You left the fix as a **decision** rather than taking it, because both candidate fixes move
  published numbers. That is the right call, and it is also correct on ownership: `frnet.py` is in
  `src/perception/`, which is JP's lane.

`scripts/frnet_determinism_probe.py` reproducing all four rows is what makes it checkable rather
than assertable.

## 2. The numbering collision

You opened it as **D9**. `D9` was already in use on `jp/p99-alloc-fixes` — the closed
`transform_points` allocation item (`12613df`, the opt-in `reuse_buffers` scratch). Two different
items now share one ID across two copies of the same file.

**JP's resolution, so both copies converge:**

- **JP's D9 (`transform_points`, closed) keeps its ID** — it is older and already referenced from
  `pending-review/transform-points-allocation.md` and from commit messages, so renumbering it would
  break existing citations.
- **Your `range_interpolation` item becomes D12** on JP's copy, with your content preserved and
  attributed to `8acbfe9`.
- `D10` (ROS 2 adapter scope) and `D11` (FRNet checkpoint / verification / DL mode) also exist only
  on JP's copy, which is why the next free number is 12 rather than 10.

**Nothing is being asked of you except to use D12 for it when the copies merge.**

## 3. ⚑ S-6 now contradicts D12, and it is the older, wrong version

`8acbfe9` added the D9/D12 row but **did not update S-6**, which still reads:

> **`--semantics frnet` is not bit-deterministic**, and it is the model: CPU vs CPU over two runs
> disagrees on 0.028–0.046% of points… **FRNet's `scatter_mean` is order-dependent on CUDA.**

Both halves of that are refuted by your own commit two days later: a CUDA-only mechanism cannot
produce a CPU-vs-CPU result, and you measured the scatter ops as bit-identical at 16 threads. It
also states the non-determinism as a property of the mode, when you showed it **disappears
completely at one thread** (0.0000% on all three frames).

Left for you to correct, since it is your file and your row. The project convention has been an
in-place dated note rather than an edit — the same shape as `1a47960`, and the same shape you used
when you left the 19 Sep entry standing with a pointer to the correction.

## 4. One consequence worth deciding together

If the projection is pinned to one thread, `--threads` stops being an opt-in reproducibility knob
and becomes a **correctness requirement** for the frnet path. That changes the framing of the patch
staged in `pending-review/frnet-threads-and-fast-scatter-on-upstream-base.diff`, which currently
presents `--threads` as optional. Worth settling in the same conversation.
