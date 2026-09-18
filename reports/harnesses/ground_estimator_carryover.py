# PROVENANCE -- committed 2026-09-18 under OPEN-ITEMS.md item R-k.
#
# Produced: the deterministic half of R-k -- does Patchwork++'s carried-over state
#           change WHICH points it calls ground, and does it move the 96.5%
#           agreement figure `ground_cost.py` published under R-a?
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/ground_estimator_carryover.py 08 60
#
# Measurement only. Reads the shipping code; changes nothing in src/, and does NOT
# modify reports/harnesses/ground_cost.py, whose output is published provenance.
"""Is `ground_cost.py`'s agreement figure affected by estimator carry-over?

`ground_cost.py` never resets the estimator (found by
`tests/test_ground_reset_convention.py`, 2026-09-18). Its agreement loop runs
*after* both `bench()` calls, so by then the one estimator has processed the
warm-up scan plus 3 reps x N scans. This reproduces that exact state sequence
and compares it against a fresh estimator, on the same scans in the same order.

**No timing is taken here** -- this laptop is paging (D8), which invalidates
latency but not masks. The timing half of R-k still needs a state-trusted
machine. What this settles is whether the carry-over is a correctness problem at all
or only a timing one.
"""
import hashlib
import sys

import numpy as np
from vrgrid.perception import ground, loader, semantics

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
WINDOW = 20                      # the agreement window ground_cost.py uses


def digest(mask):
    return hashlib.blake2b(np.ascontiguousarray(mask).tobytes(), digest_size=8).hexdigest()


def masks_and_agreement(scans):
    """The published loop, transcribed from ground_cost.py lines 62-68."""
    out, agree = [], []
    for pts, sem in scans[:WINDOW]:
        a = ground.segment_ground(pts)
        b = ground.ground_from_semantics(sem)
        valid = sem >= 0
        agree.append((a[valid] == b[valid]).mean())
        out.append(a)
    return out, float(np.mean(agree) * 100)


print(f"pypatchworkpp present: {ground._HAVE_PATCHWORKPP}")
print(f"preloading {N} scans of seq {SEQ} into RAM...")
scans = [(pts, semantics.semantic_labels(raw))
         for pts, raw, _pose in loader.scans(SEQ, max_frames=N)]
print(f"  {len(scans)} scans, {sum(len(s[0]) for s in scans):,} points total\n")

# --- A: fresh estimator, the state a per-run reset would give -------------------
ground.reset_estimator()
fresh, fresh_pct = masks_and_agreement(scans)

# --- B: ground_cost.py's state exactly -- warm-up, then 3 reps x N scans --------
ground.reset_estimator()
ground.segment_ground(scans[0][0])                    # the warm-up, line 36
ground.ground_from_semantics(scans[0][1])             # line 37
for _ in range(3):                                    # bench(), 3 reps, no timing
    for s in scans:
        ground.segment_ground(s[0])
for _ in range(3):
    for s in scans:
        ground.ground_from_semantics(s[1])
carried, carried_pct = masks_and_agreement(scans)

# --- compare --------------------------------------------------------------------
print(f"agreement on labelled points, {WINDOW} frames:")
print(f"  fresh estimator            {fresh_pct:.1f}%")
print(f"  carried-over (as published) {carried_pct:.1f}%")
print(f"  difference                 {carried_pct - fresh_pct:+.2f} points\n")

differing = [i for i, (a, b) in enumerate(zip(fresh, carried)) if not np.array_equal(a, b)]
print(f"per-frame ground masks: {len(differing)} of {WINDOW} differ")
if differing:
    tot = sum(int((a != b).sum()) for a, b in zip(fresh, carried))
    pts = sum(len(a) for a in fresh)
    print(f"  first differing frame: {differing[0]}")
    print(f"  points reclassified:   {tot:,} of {pts:,} ({tot / pts * 100:.4f}%)")
    for i in differing[:5]:
        d = int((fresh[i] != carried[i]).sum())
        print(f"    frame {i:>2}  {d:>6,} points  fresh {digest(fresh[i])}  carried {digest(carried[i])}")
else:
    print("  identical -- the carry-over is a TIMING question only, not a correctness one")
