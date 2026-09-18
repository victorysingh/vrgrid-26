"""Which line of MapEngine._cleanup makes its slow frames slow?

Per-frame timing of every line, transcribed from the shipped _cleanup (after
d540618 + 697a2bd), on a warm engine over real seq-08 frames through the same
iter_pipeline(reuse_buffers=True) path timing_table.py times. Read-only: the
method is replaced in-process only. The transcription's map hash is checked
against the shipped method on a separate identical run.
"""
import time

import numpy as np
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.gpu.kernels import map_hash
from vrgrid.gpu.visibility import apply_miss, visibility_cleanup
from vrgrid.grid.fusion import occupancy_state
from vrgrid.grid.schedule import load
from vrgrid.perception import ground
from vrgrid.run.__main__ import iter_pipeline
from vrgrid.run.engine import MapEngine

import sys
sys.path.insert(0, "reports/harnesses")
import machine_state  # noqa: E402

N = 200
LINES = ("occupancy_state", "flatnonzero", "centres", "range2d", "guard",
         "visibility", "apply_miss")
T = {k: [] for k in LINES}
TOT = []


def timed_cleanup(self, frame, touched, ego, counters):
    c0 = t = time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        T[name].append((now - t) * 1e3)
        t = now

    state = occupancy_state(self.handle.grid, self.thresholds,
                            out=self.occ_state, scratch=self.occ_scratch)
    lap("occupancy_state")
    occupied = np.flatnonzero(state == OCC_OCCUPIED)
    counters.occupied = len(occupied)
    lap("flatnonzero")
    if not len(occupied):
        return
    if len(occupied) > self.max_candidates:
        counters.truncated = len(occupied) - self.max_candidates
        occupied = occupied[:self.max_candidates]
    m = len(occupied)
    self._cand_slots[:m] = occupied
    cx, cy, cz = self._centres(occupied, ego, self._cand["x"], self._cand["y"],
                               self._cand["z"], sorted_slots=True)
    lap("centres")
    image = np.asarray(frame.range_image)
    if self.range2d.shape != image.shape[:2]:
        self.range2d = np.zeros(image.shape[:2], np.float32)
    np.copyto(self.range2d, image[:, :, 0])
    lap("range2d")
    guard = self._has_return[:m]
    lut = self._touched_lut
    lut[touched] = True
    try:
        np.copyto(guard, lut[occupied])
    finally:
        lut[touched] = False
    lap("guard")
    result = visibility_cleanup(
        cx, cy, cz, self.range2d, has_return_now=guard, sensor=self.sensor,
        floor_m=self.thresholds["visibility"]["range_tolerance_m"],
        protect_current_returns=True, scratch=self.vis_scratch)
    lap("visibility")
    occ = self.thresholds["occupancy"]
    apply_miss(self.handle.grid["log_odds"], self._cand_slots[:m],
               result.see_through, occ["log_odds_miss"], tuple(occ["log_odds_clamp"]))
    lap("apply_miss")
    counters.tested = result.tested
    counters.cleared = result.cleared
    counters.protected = result.protected
    counters.out_of_view = result.out_of_view
    TOT.append((time.perf_counter() - c0) * 1e3)


def run(cleanup=None, frames=N):
    ground.reset_estimator()                     # D1: fresh estimator per run
    orig = MapEngine._cleanup
    if cleanup is not None:
        MapEngine._cleanup = cleanup
    try:
        e = MapEngine(load("5/10/20/40"), max_points=120_000)
        for f in iter_pipeline("08", frames, reuse_buffers=True):
            e.step(f)
        return map_hash(e.handle.grid)
    finally:
        MapEngine._cleanup = orig


s0 = machine_state.snapshot()
print("  state before:", machine_state.line(s0))
h_ship = run(None, 40)
T.update({k: [] for k in LINES})
TOT.clear()
h_probe = run(timed_cleanup, 40)
print(f"  transcription == shipped _cleanup (map hash, 40 frames): {h_ship == h_probe}")

T.update({k: [] for k in LINES})
TOT.clear()
run(timed_cleanup, N)
s1 = machine_state.snapshot()
print("  state after: ", machine_state.line(s1))

W = 20                                            # warm-up frames dropped
tot = np.asarray(TOT[W:])
rows = {k: np.asarray(v[W:]) for k, v in T.items()}
p99 = np.percentile(tot, 99)
slow = tot >= np.percentile(tot, 95)               # the slowest 5% of cleanups
print(f"\n  _cleanup over {tot.size} warm frames: p50 {np.median(tot):.2f}  "
      f"p99 {p99:.2f}  max {tot.max():.2f} ms")
print(f"  {'line':<17}{'p50':>8}{'p99':>8}{'max':>8}{'p99-p50':>9}"
      f"{'slow-5% mean':>14}{'others mean':>13}{'excess':>8}")
excess_total = 0.0
for k, v in rows.items():
    ex = v[slow].mean() - v[~slow].mean()
    excess_total += ex
    print(f"  {k:<17}{np.median(v):>8.2f}{np.percentile(v, 99):>8.2f}{v.max():>8.2f}"
          f"{np.percentile(v, 99) - np.median(v):>9.2f}{v[slow].mean():>14.2f}"
          f"{v[~slow].mean():>13.2f}{ex:>8.2f}")
print(f"  slow-5% frames exceed the rest by {tot[slow].mean() - tot[~slow].mean():.2f} ms "
      f"in total; per-line excesses sum to {excess_total:.2f}")
worst = np.argsort(tot)[-5:][::-1]
print("\n  five slowest cleanups, per line (ms):")
for i in worst:
    parts = "  ".join(f"{k[:6]} {rows[k][i]:5.2f}" for k in LINES)
    print(f"    frame {i + W:>3}: total {tot[i]:6.2f} | {parts}")
