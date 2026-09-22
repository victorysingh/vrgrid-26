# One message, six items — from JP's full-project audit, 2026-09-23

*Draft for JP to send. Consolidates everything routed to Shrestha so it arrives as one list rather
than five pending-review files to discover separately.*

---

Shrestha — I ran a full audit across every lane, not just mine, and six things came out with your
name on them. Putting them in one message rather than scattering them.

**First, separately from all of it:** the `frnet.py:156` work in `8acbfe9` is the best debugging on
this project so far. I left that on the commit itself so it isn't buried in a list of asks.

---

**1. `docs/defense-rehearsal-playbook.md` — panel-facing, and it predates the band change.**
`843ad54` moved `vertical_extent_m` from `[-2.0, 6.0]` to `[-3.5, 4.5]`. You propagated that into
`known-limitations.md` (137 lines) and four presentation docs — the playbook wasn't in that set, and
its last commit is 2 Sep. The figures at risk are height-derived: the **4,041 curb cells on 07 /
9,499 on 08** with ring medians, and the **52.3% of sequence 07's peak occupied set**. I have not
re-measured them, so this is "don't quote until re-run", not "wrong" — the band is the same 8 m width
and only the floor and ceiling moved, so they may well survive. But a rehearsal doc is the wrong
place to find that out. Either re-run those two figures or put a dated banner on it; your call which.

For symmetry: the same sweep caught **two of my own reports** the same way
(`r1-accuracy-by-class-and-range-band.md`, `r7-hazard-miss-rate.md`). Both now carry banners. This
isn't a finding about your lane.

**2. `configs/` review gap — separate from the above, and the change itself is fine.**
`843ad54` edited both schedule configs. `.github/CODEOWNERS` requires all three of us on `configs/`,
and `CLAUDE.md` calls them frozen because changing one invalidates the ablation. The commit is linear
with no PR, so no review fired. **The technical work is sound** — the split came from surveying
ground returns beyond 10 m on all eleven sequences, and you verified CPU/GPU bit-identical on 200
frames of all eleven plus all 4,071 frames of seq 08. It's the review that was skipped, not the
judgement.

**3. S-6 now contradicts your own D9/D12.** S-6 still reads *"it is the model… FRNet's `scatter_mean`
is order-dependent on CUDA."* `8acbfe9` refutes both halves — a CUDA-only mechanism can't produce a
CPU-vs-CPU result, and you measured the scatter ops bit-identical at 16 threads. It also states the
non-determinism as a property of the mode when you showed it vanishes at one thread (0.0000% on all
three frames). The new row landed but S-6 wasn't updated. Same in-place-correction convention you
used for the 19 Sep entry.

**4. R-d is marked closed and isn't.** `tests/test_metrics.py:472` still reads
`(p, l, T) for p, l, _, T in ...`, the ruff config is unchanged with no E741 exemption, and ruff
0.12.0 reports **10 errors** on `ae85979` — that E741 plus `scripts/kaggle/elprobe.py` (E402, E741),
two notebook E702/E701, and `scripts/mos_learned.py:233`. The root cause is that CI runs bare
`ruff check .` with **no version pinned anywhere**, so "it is green" is a statement about whichever
ruff GitHub installed that morning. You hit this from the other side in `b977260`. Suggest pinning
ruff and reopening the row.

**5. R-b's row and your research log disagree about the machine.** The row says *"**The laptop** meets
10 Hz at p99: frame p50 21.94 / p99 28.40 ms on `5/10/50`"*. `docs/research-log.md:561` gives the
identical figures — 21.94 / 28.40, 45.6 / 35.2 FPS, 3.5× headroom — and calls them *"**The T4
column**, seq 08, 200 frames"*. The T4 column is Kaggle's 2× Tesla T4; the laptop is an RTX 5050.
They match to 0.01 ms, so it's one measurement with two machine labels, and
`13-PS-SCHEDULE.md` (which the row cites) names no machine at all. **Which one produced it?** That
answer decides whether the fix is a word in OPEN-ITEMS or a line in the research log.

Related and not a criticism: neither 21.94/28.40 nor the 71.8/79.5 DL figure appears in any committed
`.log` or `.json` — I checked with escaped patterns across the tree. Meanwhile
`docs/gpu-lane/t4/timing_cuda.log` **is** committed and records p50 67.42 / p99 100.18, *"MISSES 10 Hz
at p99"* on `5/10/20/40`, and nothing cites it. Both are true — different schedules, and `5/10/50`
drops a ring — but right now the favourable number has no log and the unfavourable one is invisible.
I've rewritten README's latency footnote to list **all six** candidate figures with their recipes
rather than quietly citing two, since D8 is still open and picking a winner would preempt a joint
call. A re-run with `--frame-times` committed would close it properly.

**6. Numbering: your new D9 is D12 on my copy.** `D9` was already taken here by the closed
`transform_points` item, which is cited from `pending-review/` and from commit messages, so
renumbering that one would break live references. Your `range_interpolation` item becomes **D12**,
content preserved and attributed to `8acbfe9`. `D10`/`D11` also exist only on my copy, which is why
the next free number is 12. Nothing needed from you except using D12 when the copies merge.

**One consequence worth deciding together:** if the projection gets pinned to one thread, `--threads`
stops being an opt-in reproducibility knob and becomes a **correctness requirement** for the frnet
path. That changes the framing of the patch I staged at
`pending-review/frnet-threads-and-fast-scatter-on-upstream-base.diff`, which currently presents it as
optional. That patch is on **your** device-path implementation as the base — yours produced the 10 Hz
result, so it stays — and it's 755 passed / 28 skipped with it applied. Yours to accept, change or
reject.

---

**Not asking for any of this today.** Items 3, 4 and 6 are small; 1 and 5 have deadline implications
because they touch panel-facing material; 2 is a process point for whenever we next talk about how
changes land.

— JP
