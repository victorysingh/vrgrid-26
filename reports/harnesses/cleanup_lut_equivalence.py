# PROVENANCE -- written 2026-09-14 for the cleanup guard fix (R-h).
#
# Produced: the proof that replacing np.isin with a boolean lookup table in
#           MapEngine._cleanup changes NOTHING about the map, plus its cost.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/cleanup_lut_equivalence.py 08 60
#
# NOTE: the shipped line is
#
#          np.copyto(guard, np.isin(occupied, touched))
#
#      and the replacement keeps the SAME inputs, the SAME output buffer and the
#      SAME position in the stage -- only the membership test changes:
#
#          lut[touched] = True
#          np.copyto(guard, lut[occupied])
#          lut[touched] = False          # reset only what was set
#
#      That is only equivalent if `touched` never holds -1 or an out-of-range slot
#      (lut[-1] would silently mark the LAST slot, a failure np.isin never had).
#      scatter_sorted drops idx<0 before sorting and emits one cell per segment, so
#      it should hold -- but this checks it on every frame rather than trusting it.
#
#      Equivalence is tested at the strongest level available: the FULL map hash
#      after N frames, not just the guard array. Each run gets a FRESH Patchwork++
#      estimator, because the singleton is stateful (D1) and two runs sharing one
#      would differ for reasons that have nothing to do with this change.
#
# Measurement only. Patches the method on the class inside this process; changes
# nothing in src/.
"""Is the LUT cleanup guard bit-identical at the level of the whole map?"""
import sys
import time

import numpy as np
from vrgrid.gpu.kernels import map_hash
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.run import engine as engine_mod
from vrgrid.run.__main__ import iter_pipeline
from vrgrid.run.engine import MapEngine

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

_shipped_cleanup = MapEngine._cleanup
stats = {"frames": 0, "min": None, "max": None, "dups": 0, "neg": 0, "oob": 0}
guard_mismatch = {"frames": 0}


def _lut_cleanup(self, frame, touched, ego, counters):
    """The shipped _cleanup, with ONLY the membership line changed.

    Implemented by temporarily swapping np.isin inside the engine module for a
    LUT-backed equivalent, so every other line of the shipped method runs exactly
    as written -- nothing re-typed, nothing re-derived.
    """
    n_slots = self.occ_state.size
    if not hasattr(self, "_touched_lut"):
        self._touched_lut = np.zeros(n_slots, dtype=np.bool_)
    lut = self._touched_lut

    t = np.asarray(touched)
    stats["frames"] += 1
    if t.size:
        stats["min"] = int(t.min()) if stats["min"] is None else min(stats["min"], int(t.min()))
        stats["max"] = int(t.max()) if stats["max"] is None else max(stats["max"], int(t.max()))
        stats["neg"] += int((t < 0).sum())
        stats["oob"] += int((t >= n_slots).sum())
        stats["dups"] += int(t.size - np.unique(t).size)

    real_isin = engine_mod.np.isin

    def lut_isin(element, test_elements, *a, **k):
        # The one call site in _cleanup is np.isin(occupied, touched).
        lut[test_elements] = True
        out = lut[element]
        lut[test_elements] = False
        ref = real_isin(element, test_elements)
        if not np.array_equal(out, ref):
            guard_mismatch["frames"] += 1
        return out

    engine_mod.np.isin = lut_isin
    try:
        return _shipped_cleanup(self, frame, touched, ego, counters)
    finally:
        engine_mod.np.isin = real_isin


def run(label, patched):
    ground.reset_estimator()                     # fresh estimator: D1
    MapEngine._cleanup = _lut_cleanup if patched else _shipped_cleanup
    try:
        eng = MapEngine(load("5/10/20/40"), ghost_removal=True)
        t0 = time.perf_counter()
        for frame in iter_pipeline(SEQ, N):
            eng.step(frame)
        secs = time.perf_counter() - t0
        return map_hash(eng.handle.grid), secs
    finally:
        MapEngine._cleanup = _shipped_cleanup


h_ref, s_ref = run("shipped", patched=False)
h_lut, s_lut = run("lut", patched=True)
h_ref2, _ = run("shipped again", patched=False)

print(f"seq {SEQ}, {N} frames, full MapEngine with ghost removal\n")
print("1. INVARIANTS ON `touched` (the LUT's safety precondition), every frame")
print(f"   frames checked {stats['frames']}, slot range {stats['min']}..{stats['max']}, "
      f"negatives {stats['neg']}, out-of-range {stats['oob']}, duplicates {stats['dups']}")
safe = stats["neg"] == 0 and stats["oob"] == 0
print(f"   -> {'SAFE: lut[touched] cannot alias another slot' if safe else 'UNSAFE -- STOP'}")

print("\n2. GUARD ARRAY, per frame: LUT vs np.isin")
print(f"   frames where the guard differed: {guard_mismatch['frames']}")

print("\n3. WHOLE-MAP HASH after all frames")
print(f"   shipped        {h_ref}")
print(f"   LUT            {h_lut}")
print(f"   shipped again  {h_ref2}   (control: same code twice, fresh estimators)")
if h_ref != h_ref2:
    print("   -> CONTROL FAILED: the shipped path is not reproducible across runs here,")
    print("      so a hash comparison cannot adjudicate the LUT. Investigate before trusting.")
else:
    print(f"   -> {'BIT-IDENTICAL MAP' if h_lut == h_ref else 'MAPS DIFFER -- REJECT'}")
print(f"\n   (wall time, informational only: shipped {s_ref:.1f}s, LUT {s_lut:.1f}s)")
