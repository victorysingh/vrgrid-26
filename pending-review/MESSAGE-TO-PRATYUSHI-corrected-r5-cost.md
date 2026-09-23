# Corrected R5 cost — for Pratyushi. Draft for JP to review and send.

*Same content as the note going to Shrestha. Both of them approved a version of R5 that does not
match what is actually shipping, so both approvals need re-confirming against the real cost.*

---

Pratyushi — I need to correct something I told you about R5 before you approved it. The approval you
gave was for a cheaper design than the one that actually works, and that's on me, not you.

**What you were told:** R5's sticky VRU bit, using decay option 2, costs **one bit** — `FLAG_VRU_SEEN`
at flags bit 4, leaving bits 5–7 free for future use. That was the basis on which option 2 looked
like the cheap choice next to option 1's per-cell countdown.

**What it actually costs: four bits.** Bit 4 is the latch, and **bits 5–7 are a 3-bit age counter**.
The flags byte is now fully consumed, 0 through 7.

**Why the counter turned out to be unavoidable.** The decay has to mean *"frames since a VRU was
observed in this cell"*. The struct already has `frames_since_seen`, but that means *"frames since
this **cell** was observed"* — a different question, and the exact failure the design doc warned
about: a cell the vehicle keeps looking at sits at 0 forever, so one transient person would latch a
busy road cell permanently. That is precisely the cell where a VRU is most likely to have been a
minority of the returns. No other field in the 12-byte struct was spare, so counting VRU age needs
its own bits.

**The consequence you should weigh, because it changes the comparison you approved:** four bits is
**the same total storage as option 1**, which we rejected *for that cost*. So option 2 is no longer
the cheap choice. What still argues for it is not storage but *where* the decay is keyed — in the
visibility pass, against cells that were actually tested, rather than a countdown ticking regardless.
That reason is unchanged and I still think it's right. But the storage argument is gone, and you
approved partly on the storage argument.

**Two further consequences:**

- `N`, the decay threshold, is **hard-capped at 7 frames** by 3 bits. It currently sits at 5 as a
  first guess, flagged as untuned in `configs/thresholds.yaml`.
- **The next flag bit anyone wants has no home** in this byte and will need a different one.

**Where things stand.** The code is written, wired and green — 850 tests passing, `CELL_BYTES` still
12, memory unchanged at 8.94 MB. **Nothing is committed.** The changes to `include/vrgrid/cell.py`
and `configs/thresholds.yaml` are sitting uncommitted until both you and Shrestha have replied to the
corrected cost, because a frozen-file change approved on wrong information isn't approved.

**What I'm asking:** does your approval still stand at four bits and a full flags byte? A plain yes
is enough. If it doesn't, say so and we redesign before anything touches the frozen files — that's a
better outcome than shipping on an approval that was given for something else.

For completeness, R10's `TRAV_DEPRESSION` is unaffected by any of this: it's one bit in the
*traversability* byte, bit 6, with bit 7 still free, and its cost is exactly what you were told.

— JP
