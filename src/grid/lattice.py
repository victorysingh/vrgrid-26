"""The integer lattice. Math §2. [Aakash — Day 0/1, first task]

Everything downstream indexes through this file, which is why it is the first
thing built and the partition test is CI-blocking.

The rule, and it has no exceptions: there is ONE lattice, at the base
resolution c0 = 5 cm. Coarser ring indices are derived from it by integer
division, never recomputed in floating point.

    i_fine(x) = floor(x / c0)
    i_L(x)    = floor(i_fine(x) / k_L),   k_L = c_L / c0 in Z+

Theorem (nested floor, math §2.2): floor(floor(x/c0)/k) == floor(x/(k*c0)).
So the ring-L lattice IS the direct lattice of size k*c0 -- the rings partition
the plane exactly. There is no tolerance to tune and no epsilon.

Computing floor(x/0.20) directly instead is the bug this file exists to
prevent: 0.2 is not representable in binary, the two lattices drift apart, and
near a boundary a point falls in both cells or neither.
"""

import math

import numpy as np
from vrgrid.cell import CELL_FIELDS, alloc_soa
from vrgrid.gpu.allocators import EMPTY_CELL

# A point beyond the last ring is not in the map. It is not ring 0 either.
OUTSIDE = -1

# The rear resolution floor applies within this range behind the vehicle.
# Math §6.2: "c_L <= 0.20 m whenever x < 0 and |x| < 50".
REAR_FLOOR_RANGE_M = 50.0


def stretch_factors(schedule, speed_ms: float):
    """(a_f, a_s, a_r) of math §6.2 eq. (20).

    a_f stretches the map forward with speed (clamped at 2x), a_s squeezes it
    laterally, a_r never stretches. At v = 0 all three are 1 and (20) collapses
    to the plain Chebyshev norm of (18) -- which is why there is no separate
    isotropic code path to keep in sync.
    """
    a = schedule.anisotropy
    t = speed_ms / a.v_ref_ms
    a_f = min(max(1.0 + a.kappa_forward * t, 1.0), 2.0)
    a_s = 1.0 / (1.0 + a.kappa_side * t)
    return a_f, a_s, a.rear_stretch


def d_aniso(x, y, schedule, speed_ms: float = 0.0):
    """Scaled L-infinity distance. Math §6.2 eq. (20).

        d = max( x+/a_f,  x-/a_r,  |y|/a_s )

    Note the direction of the lateral term: a_s < 1, so |y|/a_s > |y|. Points
    to the side are pushed OUT to a coarser ring -- the resolution is taken
    from the sides and spent forward, which is the whole idea.
    """
    a_f, a_s, a_r = stretch_factors(schedule, speed_ms)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    forward = np.maximum(x, 0.0) / a_f
    rear = np.maximum(-x, 0.0) / a_r
    side = np.abs(y) / a_s
    return np.maximum(np.maximum(forward, rear), side)


def _rear_floor_ring(schedule):
    """Coarsest ring whose cell still satisfies the rear floor, or None if the
    floor never binds for this schedule (the ablation's 50 cm ring is its own
    floor, so nothing is clamped)."""
    floor_m = schedule.anisotropy.rear_floor_cell_m
    allowed = [r.ring for r in schedule.rings if r.cell_m <= floor_m + 1e-9]
    if not allowed or len(allowed) == len(schedule.rings):
        return None
    return max(allowed)


def i_fine(x: float, base_cell_m: float) -> int:
    """Base-lattice index. Math §2.1 eq. (8).

    Floor, not truncation: `int(x / c0)` rounds toward zero, so -0.02 and
    +0.02 both land in cell 0 and the cell straddling the origin comes out
    twice the size of every other cell. Floor division is correct on both
    sides of zero.

    Scalar in -> int out; ndarray in -> int64 array out. Both paths are the
    same floor-division operator, so the vectorised path (scatter) and the
    scalar path (query) cannot answer differently.
    """
    q = x // base_cell_m
    return int(q) if np.ndim(q) == 0 else q.astype(np.int64)


def i_ring(x: float, base_cell_m: float, k: int) -> int:
    """Ring index, by integer division from the base lattice. Math §2.1 eq. (9).

    Deliberately NOT floor(x / (k * base_cell_m)). Theorem §2.2 proves the two
    agree in exact arithmetic; in IEEE-754 they are two lattices that drift
    apart, and near a boundary a point falls in both cells or in neither.
    Derive, never recompute.

    `k` must be a positive integer -- the same hard rule `schedule.validate()`
    enforces on the config. Powers of two are a convenience (the divide is a
    bit shift), not a requirement.
    """
    if k != int(k) or k < 1:
        raise ValueError(f"k must be a positive integer (math §2.1), got {k!r}")
    return i_fine(x, base_cell_m) // int(k)


def _ring_windows(schedule, vehicle_xy_m=(0.0, 0.0), buffers=None):
    """(k, side, x0, y0) per ring: the window each ring's cells live in.

    From `buffers` when the caller has them -- the engine, `GridMap`, anything
    whose windows have actually been shifted -- because ring membership is
    decided against the windows that EXIST, not the ones that ought to. With
    no buffers, the windows `recenter()` would build around `vehicle_xy_m`:
    each ring's own lattice index of the vehicle, minus half a side. That is
    the stationary map every unit test uses, and at the origin it reproduces
    eq. (18) exactly -- ring 0's window spans [-10, 10) m.
    """
    c0 = schedule.base_cell_m
    wins = []
    for L in range(len(schedule.rings)):
        k = schedule.k(L)
        if buffers is not None:
            b = buffers[L]
            wins.append((k, int(b.side), int(b.x0), int(b.y0)))
        else:
            W = ring_extent(schedule, L)
            wins.append((k, W,
                         i_ring(float(vehicle_xy_m[0]), c0, k) - W // 2,
                         i_ring(float(vehicle_xy_m[1]), c0, k) - W // 2))
    return wins


def _descent_constants(schedule, ks, speed_ms, vehicle_xy_m, yaw_rad):
    """Per-frame scalars for the block bound: the §6.2 stretch factors, the
    heading, and per ring (centre offset x, centre offset y, half-extent).

    The half-extent is how far a block of side c reaches along a unit
    direction at angle yaw: (|cos| + |sin|) c / 2. At yaw = 0 that is c / 2,
    and the bound below is the exact nearest point of the block.
    """
    a_f, a_s, a_r = stretch_factors(schedule, speed_ms)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    c0 = schedule.base_cell_m
    vx, vy = float(vehicle_xy_m[0]), float(vehicle_xy_m[1])
    per = []
    for k in ks:
        cell = k * c0
        per.append((0.5 * cell - vx, 0.5 * cell - vy,
                    0.5 * (abs(cy) + abs(sy)) * cell))
    return a_f, a_s, a_r, cy, sy, per


def ring_of(x, y, schedule, speed_ms: float = 0.0, vehicle_xy_m=(0.0, 0.0),
            yaw_rad: float = 0.0, buffers=None):
    """Which ring a place belongs to. Math §6.1 eq. (18), §6.2 eq. (20).

    `x`, `y` are metres from the vehicle along the WORLD axes -- the frame
    `query()` takes, and `x + vehicle_xy_m` is the world point. `yaw_rad` is
    the heading, and is used only to say which way is forward for §6.2.

    **The rule is decided per BLOCK, coarse to fine, never per point.**

        L = N-1, or OUTSIDE if the ring N-1 window does not hold the point
        while L > 0:
            B = the ring-L block containing the point
            split B into ring L-1 iff
                (i)  every ring-(L-1) child of B lies in ring L-1's window, and
                (ii) d_aniso at the NEAREST point of B is < R_{L-1}
                     (or §6.2's rear floor forces it)
            otherwise stop at L

    ⚑ Why per block, and it is the partition theorem, not taste (open item
      D2 / R3). The previous rule compared each POINT's own d_aniso against
      R_L. A ring boundary is a real number, and in the engine it was also
      rotated with the sensor, so it fell strictly inside blocks: two points
      in one 40 cm cell could be filed 5 cm and 40 cm, and the 40 cm cell's
      footprint then contained a 5 cm cell that was also occupied. Measured on
      real seq 08, 30 frames: every frame, 0.108% of coarse cells. Here every
      decision is a function of the block alone, and blocks nest (validate()
      guarantees integer ratios), so all points of one block stop at the same
      level -- no footprint can contain another, and every base cell has
      exactly one ring, so there is no gap either.

    ⚑ Why the WINDOW decides containment. Each ring is a fixed square buffer on
      the world lattice (§2.4). The old containment test was the point's
      Chebyshev distance in the sensor frame, which is a rotated square; its
      corners are outside the world-aligned window, and a point there was
      given ring L, missed ring L's window in `bin_points`, and was dropped --
      0.224% of all returns on seq 08. Testing the block against the integer
      window is exact, it cannot wrap toroidally, and it cannot drop a return
      a coarser ring has room for.

    ⚑ Nearest point, not centre (R3 part A). A block admitted to the finer
      ring if ANY of it belongs there is the safe direction: resolve finely
      when in doubt. d_aniso is a max of three terms, each monotone along its
      own heading-frame axis, and the heading-frame extent of a block of side
      c about its centre is +-(|cos|+|sin|) c/2, so max(term(centre -+ h), 0)
      bounds each term from below over the whole block. That is a lower bound
      on d, exact when yaw = 0, and -- the property the partition needs --
      a function of the block only.

    R3's part B, snapping the boundary to the coarser lattice, is not used:
    under a heading the boundary is not axis-aligned and no snap puts it on
    the lattice, while the per-block decision makes the partition hold without
    it.

    What survives unchanged from before: the lateral squeeze sends side blocks
    coarser sooner; the forward stretch cannot pull a block past the finer
    ring's buffer (condition i -- a memory decision, not a lattice one); the
    rear floor holds within 50 m behind wherever the block fits; the map keeps
    what the coarsest window holds whatever the speed; and a point past the
    last window is OUTSIDE, never ring 0.

    Scalar in -> int out; ndarray in -> int64 array out.
    """
    c0 = schedule.base_cell_m
    vx, vy = float(vehicle_xy_m[0]), float(vehicle_xy_m[1])
    xw = np.asarray(x, dtype=np.float64) + vx
    yw = np.asarray(y, dtype=np.float64) + vy
    fx = np.asarray(xw // c0).astype(np.int64)
    fy = np.asarray(yw // c0).astype(np.int64)

    wins = _ring_windows(schedule, (vx, vy), buffers)
    ks = [w[0] for w in wins]
    a_f, a_s, a_r, cy, sy, per = _descent_constants(schedule, ks, speed_ms,
                                                    (vx, vy), yaw_rad)
    radii = [r.half_width_m for r in schedule.rings]
    floor_ring = _rear_floor_ring(schedule)
    n = len(wins)

    k, W, x0, y0 = wins[-1]
    tx, ty = fx // k - x0, fy // k - y0
    level = np.where((tx >= 0) & (tx < W) & (ty >= 0) & (ty < W), n - 1, OUTSIDE)

    for M in range(n - 1, 0, -1):
        kM = ks[M]
        kP, WP, x0P, y0P = wins[M - 1]
        r = kM // kP
        bx, by = fx // kM, fy // kM

        # (i) every child in the finer window: [bx*r, bx*r + r) within [x0, x0+W)
        tx, ty = bx * r - x0P, by * r - y0P
        fits = (tx >= 0) & (tx <= WP - r) & (ty >= 0) & (ty <= WP - r)

        # (ii) eq. (20) at the block's nearest point, in the heading frame
        offx, offy, h = per[M]
        xc = np.asarray(bx * kM).astype(np.float64) * c0 + offx
        yc = np.asarray(by * kM).astype(np.float64) * c0 + offy
        u = cy * xc + sy * yc
        v = cy * yc - sy * xc
        fwd = np.maximum(u - h, 0.0) / a_f
        rear = np.maximum(-(u + h), 0.0) / a_r
        side = np.maximum(np.abs(v) - h, 0.0) / a_s
        admit = np.maximum(np.maximum(fwd, rear), side) < radii[M - 1]

        # §6.2's rear floor: never coarser than rear_floor_cell_m within 50 m
        # behind. Closing traffic is where a coarse cell hurts, so the stretch
        # is taken from the sides, never the back -- and only where the block
        # fits, which (i) already requires.
        if floor_ring is not None and M > floor_ring:
            admit = admit | ((u < 0.0) & (np.abs(u) < REAR_FLOOR_RANGE_M))

        level = np.where((level == M) & fits & admit, M - 1, level)

    return int(level) if np.ndim(level) == 0 else level.astype(np.int64)


def migrate_ring(x, y, schedule, current_ring, speed_ms: float = 0.0,
                 vehicle_xy_m=(0.0, 0.0), yaw_rad: float = 0.0, buffers=None):
    """Ring assignment WITH hysteresis, for per-frame migration. Math §6.3.

    A cell sitting exactly on a ring boundary while speed fluctuates would
    otherwise split and merge every frame: refinement-pool thrash, and by
    §5.4 unbounded variance inflation if the derived flag is ever cleared
    mid-cycle. So the thresholds are asymmetric (eq. 21):

        split (go finer)   when  d < R_L
        merge (go coarser) when  d > R_L (1 + eps)

    Between the two the cell stays where it is. `ring_of` is the eps = 0 case
    and is the right function for a fresh point with no history; this one is
    for a cell that already has a ring. `x`, `y`, the vehicle position,
    heading and windows mean what they mean in `ring_of`, and are passed
    straight through to it.
    """
    where = {"vehicle_xy_m": vehicle_xy_m, "yaw_rad": yaw_rad, "buffers": buffers}
    eps = schedule.hysteresis_eps
    d = d_aniso(x, y, schedule, speed_ms)
    radii = [r.half_width_m for r in schedule.rings]
    cur = int(current_ring)

    if cur == OUTSIDE:
        return ring_of(x, y, schedule, speed_ms, **where)

    # Coarser only once past the OUTER edge of the current ring, widened by
    # eps. Finer as soon as the inner boundary is genuinely crossed.
    if d > radii[cur] * (1.0 + eps):
        target = ring_of(x, y, schedule, speed_ms, **where)
        return target if target == OUTSIDE else max(target, cur + 1)
    if cur > 0 and d < radii[cur - 1]:
        return ring_of(x, y, schedule, speed_ms, **where)
    return cur



def migrate_ring_many(x, y, schedule, current_ring, speed_ms: float = 0.0,
                      vehicle_xy_m=(0.0, 0.0), yaw_rad: float = 0.0, buffers=None):
    """`migrate_ring` over arrays of cells, element for element identical.

    The refinement pool asks this for every block it holds, every frame -- up
    to 512 scalar calls, each rebuilding `ring_of`'s windows and constants,
    which was a measurable share of `gate.apply`. The branches of the scalar
    version become masks; `ring_of` is evaluated once over all cells, which
    gives the same per-cell answer because it is elementwise.
    `test_migrate_ring_many_matches_the_scalar` pins the equivalence.
    """
    where = {"vehicle_xy_m": vehicle_xy_m, "yaw_rad": yaw_rad, "buffers": buffers}
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cur = np.asarray(current_ring, dtype=np.int64)
    if x.size == 0:
        return np.zeros(0, dtype=np.int64)
    eps = schedule.hysteresis_eps
    d = np.atleast_1d(d_aniso(x, y, schedule, speed_ms))
    radii = np.array([r.half_width_m for r in schedule.rings], dtype=np.float64)
    target = np.atleast_1d(ring_of(x, y, schedule, speed_ms, **where)).astype(np.int64)

    outside = cur == OUTSIDE
    safe = np.where(outside, 0, cur)
    coarser = ~outside & (d > radii[safe] * (1.0 + eps))
    finer = ~outside & ~coarser & (safe > 0) & (d < radii[np.maximum(safe - 1, 0)])
    out = cur.copy()
    out[outside] = target[outside]
    out[finer] = target[finer]
    up = np.where(target == OUTSIDE, OUTSIDE, np.maximum(target, cur + 1))
    out[coarser] = up[coarser]
    return out


# --- the frame path: zero-allocation binning, math §2.1 + §6.1 --------------
#
# `ring_of`, `d_aniso` and `i_ring` above are the reference implementations:
# scalars or arrays in, allocate freely, and they are what every test compares
# against. Everything below is their frame-loop twin -- the same arithmetic in
# the same order, with every intermediate preallocated. Bit-identity is pinned
# by `test_ring_of_into_matches_ring_of` and
# `test_bin_points_matches_the_reference_path` over both frozen schedules,
# several speeds, and both ring parities.
#
# ⚑ Why the twin exists, and it is not tidiness. `ring_of` allocates roughly
#   seven full-length float64 temporaries per call -- 6.96 MB per
#   120,000-point sweep, measured by `scripts/timing_table.py --alloc` --
#   against CLAUDE.md's "no allocation inside the frame loop", which is a hard
#   invariant and a sentence in the report. Binning is also the largest single
#   stage in the frame, larger than `fuse` and larger than `cleanup`.


def _bin_geometry(schedule):
    """Per-schedule constants `bin_points` would otherwise rebuild every frame:
    ring radii as one array, the integer k per ring, and the rear-floor ring.

    Small allocations, but they are allocations in the frame loop, and
    `_rear_floor_ring` builds a list comprehension per call. Baked into the
    scratch at startup instead.
    """
    radii = np.array([r.half_width_m for r in schedule.rings], dtype=np.float64)
    ks = []
    for r in schedule.rings:
        k = round(r.cell_m / schedule.base_cell_m)
        if abs(k * schedule.base_cell_m - r.cell_m) > 1e-9 or k < 1:
            raise ValueError(
                f"ring {r.ring}: cell {r.cell_m} m is not a positive integer "
                f"multiple of the base {schedule.base_cell_m} m (math §2.1). "
                "schedule.validate() should have rejected this."
            )
        ks.append(int(k))
    return radii, ks, _rear_floor_ring(schedule)


def new_bin_scratch(max_points: int, schedule) -> dict:
    """Working set for `bin_points`, sized at startup like every other
    frame-path buffer.

    Six int64 lanes, four float64 and four bool: **84 B per point**, 12.6 MB at
    the 150,000-point cap in
    `configs/thresholds.yaml: scatter.max_points_per_frame`. It was 50 B
    before the per-block ring rule (open item D2): deciding a block needs its
    centre in the heading frame -- two more float lanes -- and the rear floor
    needs a third mask.

    That is the trade, stated plainly: declared startup footprint to remove
    per-frame churn, which `ring_of` would otherwise allocate at ~15 MB a
    sweep. Churn is invisible until someone profiles it; footprint is a number
    on a slide. It is not part of the map's cell budget.

    The lanes are reused aggressively and the comments in `ring_of_into` and
    `bin_points` say where, because fourteen buffers doing more jobs than that
    is only safe if the handover points are written down. The caller's `out`
    array is used as one more lane until the final write, for the same reason.
    """
    radii, ks, floor_ring = _bin_geometry(schedule)
    n_rings = len(schedule.rings)
    return {
        "f0": np.zeros(max_points, np.float64),   # base index, block centre x, fwd
        "f1": np.zeros(max_points, np.float64),   # block centre y, rear
        "f2": np.zeros(max_points, np.float64),   # u, the heading-frame forward
        "f3": np.zeros(max_points, np.float64),   # v, then side
        "level": np.zeros(max_points, np.int64),  # ring per point, whole pass
        "a": np.zeros(max_points, np.int64),      # block ix, then clipped ring
        "b": np.zeros(max_points, np.int64),      # block iy, then k, then side
        "c": np.zeros(max_points, np.int64),      # base ix, then col
        "d": np.zeros(max_points, np.int64),      # base iy, then row, then slot
        "e": np.zeros(max_points, np.int64),      # window offsets, block corner
        "live": np.zeros(max_points, np.bool_),
        "tmp": np.zeros(max_points, np.bool_),
        "aux": np.zeros(max_points, np.bool_),
        "aux2": np.zeros(max_points, np.bool_),
        # per-ring tables, gathered per point. x0/y0 move with the vehicle
        # every frame, so they are refilled per call -- into these arrays,
        # never rebuilt.
        "t_k": np.array(ks, dtype=np.int64),
        "t_side": np.zeros(n_rings, np.int64),
        "t_x0": np.zeros(n_rings, np.int64),
        "t_y0": np.zeros(n_rings, np.int64),
        "t_off": np.zeros(n_rings, np.int64),
        "t_dx": np.zeros(n_rings, np.int64),      # x0 mod side, the wrap offset
        "t_dy": np.zeros(n_rings, np.int64),
        # per-schedule constants, so the frame loop never rebuilds them
        "radii": radii,
        "ks": ks,
        "floor_ring": floor_ring,
        "max_points": int(max_points),
    }


def d_aniso_into(x, y, schedule, speed_ms, out, tmp):
    """`d_aniso` with both temporaries supplied. Math §6.2 eq. (20).

    The three terms are combined in the same order as the reference, so the
    float64 result is bit-identical rather than merely close. That matters:
    the value is compared against a ring radius, and a point exactly on a
    boundary has to land in the same ring on both paths or the partition test
    is measuring two different maps.
    """
    a_f, a_s, a_r = stretch_factors(schedule, speed_ms)
    np.maximum(x, 0.0, out=out)                  # forward
    np.divide(out, a_f, out=out)
    np.negative(x, out=tmp)                      # rear
    np.maximum(tmp, 0.0, out=tmp)
    np.divide(tmp, a_r, out=tmp)
    np.maximum(out, tmp, out=out)
    np.abs(y, out=tmp)                           # side
    np.divide(tmp, a_s, out=tmp)
    np.maximum(out, tmp, out=out)
    return out


def ring_of_into(xw, yw, schedule, speed_ms, out, scratch, buffers=None,
                 vehicle_xy_m=(0.0, 0.0), yaw_rad: float = 0.0):
    """`ring_of` over a whole sweep, allocation-free. Math §6.1, §6.2.

    ⚑ `xw`, `yw` are WORLD coordinates, unlike `ring_of`'s vehicle-relative
      ones: the frame path has world points in hand, and ring membership is
      now a property of the world-lattice block, so the vehicle position
      enters only through `vehicle_xy_m`. `ring_of(xw - vx, yw - vy, ...,
      vehicle_xy_m=(vx, vy))` is the same question, bit for bit -- pinned by
      `test_ring_of_into_matches_ring_of`.

    `out` is int64 and receives OUTSIDE (-1) past the coarsest window, exactly
    as the reference does. `buffers` are the ring windows; None means the
    windows centred on `vehicle_xy_m`, as in `ring_of`.

    Every rule the reference applies is applied here, in the same order and
    with the same float operations in the same order -- the block bound is
    compared against a radius, and a block exactly on a boundary must land in
    the same ring on both paths. `ring_of`'s docstring explains why each rule
    exists; this is the one that runs.

    Lane handover: on return `c` and `d` hold the base-lattice index of every
    point, which `bin_points` reuses rather than flooring the world twice.
    """
    c0 = schedule.base_cell_m
    n = len(xw)
    wins = _ring_windows(schedule, vehicle_xy_m, buffers)
    ks, radii, floor_ring = scratch["ks"], scratch["radii"], scratch["floor_ring"]
    a_f, a_s, a_r, cy, sy, per = _descent_constants(schedule, ks, speed_ms,
                                                    vehicle_xy_m, yaw_rad)
    f0, f1, f2, f3 = (scratch[k][:n] for k in ("f0", "f1", "f2", "f3"))
    a, b, c, d, e = (scratch[k][:n] for k in "abcde")
    act, adm, aux, aux2 = (scratch[k][:n] for k in ("live", "tmp", "aux", "aux2"))
    out = out[:n]

    # §2.1 eq. (8): the ONE base lattice, once per point.
    np.floor_divide(xw, c0, out=f0)
    np.copyto(c, f0, casting="unsafe")          # integer-valued float -> int64
    np.floor_divide(yw, c0, out=f0)
    np.copyto(d, f0, casting="unsafe")

    # Inside the coarsest window, or OUTSIDE.
    rings = len(wins)
    k, W, x0, y0 = wins[-1]
    np.floor_divide(c, k, out=a)
    np.subtract(a, x0, out=a)
    np.greater_equal(a, 0, out=act)
    np.less(a, W, out=aux)
    np.logical_and(act, aux, out=act)
    np.floor_divide(d, k, out=b)
    np.subtract(b, y0, out=b)
    np.greater_equal(b, 0, out=aux)
    np.logical_and(act, aux, out=act)
    np.less(b, W, out=aux)
    np.logical_and(act, aux, out=act)
    out[:] = rings - 1
    np.logical_not(act, out=aux)
    np.copyto(out, OUTSIDE, where=aux)

    for M in range(rings - 1, 0, -1):
        kM = ks[M]
        kP, WP, x0P, y0P = wins[M - 1]
        r = kM // kP
        offx, offy, h = per[M]

        np.equal(out, M, out=act)               # still descending
        np.floor_divide(c, kM, out=a)           # the ring-M block
        np.floor_divide(d, kM, out=b)

        # (i) every child in the finer window
        np.multiply(a, r, out=e)
        np.subtract(e, x0P, out=e)
        np.greater_equal(e, 0, out=aux)
        np.logical_and(act, aux, out=act)
        np.less_equal(e, WP - r, out=aux)
        np.logical_and(act, aux, out=act)
        np.multiply(b, r, out=e)
        np.subtract(e, y0P, out=e)
        np.greater_equal(e, 0, out=aux)
        np.logical_and(act, aux, out=act)
        np.less_equal(e, WP - r, out=aux)
        np.logical_and(act, aux, out=act)

        # (ii) the block centre, then eq. (20) at its nearest point
        np.multiply(a, kM, out=e)
        np.copyto(f0, e, casting="unsafe")
        np.multiply(f0, c0, out=f0)
        np.add(f0, offx, out=f0)                # xc
        np.multiply(b, kM, out=e)
        np.copyto(f1, e, casting="unsafe")
        np.multiply(f1, c0, out=f1)
        np.add(f1, offy, out=f1)                # yc
        np.multiply(f0, cy, out=f2)
        np.multiply(f1, sy, out=f3)
        np.add(f2, f3, out=f2)                  # u = cy*xc + sy*yc
        np.multiply(f1, cy, out=f3)
        np.multiply(f0, sy, out=f0)
        np.subtract(f3, f0, out=f3)             # v = cy*yc - sy*xc
        np.subtract(f2, h, out=f0)              # forward
        np.maximum(f0, 0.0, out=f0)
        np.divide(f0, a_f, out=f0)
        np.add(f2, h, out=f1)                   # rear
        np.negative(f1, out=f1)
        np.maximum(f1, 0.0, out=f1)
        np.divide(f1, a_r, out=f1)
        np.maximum(f0, f1, out=f0)
        np.abs(f3, out=f3)                      # side
        np.subtract(f3, h, out=f3)
        np.maximum(f3, 0.0, out=f3)
        np.divide(f3, a_s, out=f3)
        np.maximum(f0, f3, out=f0)
        np.less(f0, radii[M - 1], out=adm)

        if floor_ring is not None and M > floor_ring:
            np.less(f2, 0.0, out=aux)
            np.abs(f2, out=f3)
            np.less(f3, REAR_FLOOR_RANGE_M, out=aux2)
            np.logical_and(aux, aux2, out=aux)
            np.logical_or(adm, aux, out=adm)

        np.logical_and(act, adm, out=act)
        np.subtract(out, 1, out=out, where=act)
    return out


def _fill_ring_tables(buffers, scratch) -> None:
    """Per-ring constants as arrays indexed by ring, refreshed every call.

    `x0`/`y0` are the only state a toroidal shift changes (§2.4), so they move
    under us every frame and cannot be baked at startup like `k`. Written into
    preallocated arrays element by element -- a four-element Python loop, no
    array construction.
    """
    for L, buf in enumerate(buffers):
        W = buf.side
        scratch["t_side"][L] = W
        scratch["t_x0"][L] = buf.x0
        scratch["t_y0"][L] = buf.y0
        scratch["t_off"][L] = buf.offset
        scratch["t_dx"][L] = buf.x0 % W
        scratch["t_dy"][L] = buf.y0 % W


def bin_points(xw, yw, schedule, buffers, out, scratch, speed_ms: float = 0.0,
               vehicle_xy_m=(0.0, 0.0), yaw_rad: float = 0.0):
    """Points -> flat storage slots, in one place. Math §2.1, §2.4, §6.1.

    The stage between perception and the grid: ring membership, then the
    lattice index, then the slot in that ring's toroidal window. It runs every
    frame on every return, it is the largest single stage in the frame, and
    until now no module exported it -- it was composed by hand in
    `fusion.scatter`, `grid/transient.py`, `run/engine.py` and
    `scripts/timing_table.py`. Four spellings of one step in three
    directories, which will disagree eventually, and a binning bug does not
    crash: it produces a plausible map.

    ⚑ ONE frame now, and the vehicle as a parameter. Until open item D2 this
      took the points twice -- vehicle frame for the ring, world frame for the
      cell -- and the ring was decided per point on the rotated sensor frame.
      That filed points of one coarse cell into two rings (footprints that
      contain each other, every frame on seq 08) and dropped returns whose
      ring's world-aligned window did not hold them (0.224%). Ring membership
      is now decided per world-lattice block (`ring_of`), so it takes the
      world points plus where the vehicle is and which way it faces.

      `vehicle_xy_m` is not optional in practice. Omit it once the vehicle
      has driven away from the origin and every block reads as hundreds of
      metres away: nothing is dropped -- the windows still hold the points --
      but everything is filed in the coarsest ring. Pinned by
      `test_the_vehicle_position_is_not_optional`.

    **No ring loop for the slot.** The obvious shape is a pass per ring over the points that
    fall in it, and that is what all four hand-rolled copies did. It needs the
    selected world coordinates compacted into a buffer, and numpy will not do
    that without allocating: `np.compress(..., out=)` still built 1.54 MB of
    internal index at 96,000 selected points. So every per-ring constant --
    `k`, `side`, `x0`, `y0`, `offset` -- is instead gathered per POINT by ring
    index, and the sweep is binned in one pass of full-length ufuncs. That is
    allocation-free, and it is also less arithmetic than four iterations plus
    four compacting copies.

    Bit-identical to `ring_of` + `i_ring` + `RingBuffer.flat_slot`, pinned by
    `test_bin_points_matches_the_reference_path` over both frozen schedules,
    four speeds and several headings. It has to be: those are what the partition test proves things
    about, and a second lattice is what this module's header forbids.

    Returns a view of `out` of length len(xv): the flat slot per point, and -1
    for anything outside the map or outside its ring's window. Allocates
    nothing; `scratch` comes from `new_bin_scratch`.
    """
    n = len(xw)
    if n > scratch["max_points"]:
        raise ValueError(
            f"{n} points exceeds the {scratch['max_points']} this scratch was "
            "built for (configs/thresholds.yaml: scatter.max_points_per_frame). "
            "Sizing the frame path at startup is the point -- growing it here "
            "would allocate inside the frame loop."
        )
    _fill_ring_tables(buffers, scratch)

    level = scratch["level"][:n]
    ring_of_into(xw, yw, schedule, speed_ms, level, scratch, buffers,
                 vehicle_xy_m, yaw_rad)

    lv, kk = scratch["a"][:n], scratch["b"][:n]
    ix, iy = scratch["c"][:n], scratch["d"][:n]
    live, aux = scratch["live"][:n], scratch["tmp"][:n]
    gather = out[:n]                 # the sixth lane, free until the last write

    # OUTSIDE (-1) would index the tables from the back, so clamp for the
    # gather and mask those points out at the end instead.
    #
    # ⚑ Every `np.take` below passes `mode="clip"`, and not for the clipping.
    #   numpy's default `mode="raise"` performs its bounds check by building a
    #   full-length index array -- 0.96 MB per call, six calls a frame, which
    #   is most of what this rewrite exists to remove. `clip` skips that and is
    #   also 5x faster. It is safe only because `lv` is clamped to [0, n_rings)
    #   on the line above, so no index is ever out of range and clipping never
    #   actually clips; if that clamp is ever removed, this silently reads the
    #   wrong ring instead of raising.
    np.maximum(level, 0, out=lv)
    np.take(scratch["t_k"], lv, out=kk, mode="clip")

    # §2.1 eq. (9): the ONE base lattice, integer-divided by k. `ring_of_into`
    # left the base index in lanes c and d. Never floor(x / (k*c0)) -- see
    # this module's header.
    np.floor_divide(ix, kk, out=ix)
    np.floor_divide(iy, kk, out=iy)

    # k is spent; the lane becomes the ring side, which is needed to the end.
    W = kk
    np.take(scratch["t_side"], lv, out=W, mode="clip")

    # in_view: x0 <= ix < x0 + W, and the same for iy (§2.4). Computed as
    # col = ix - x0 in [0, W), which is also what the wrap below needs.
    np.take(scratch["t_x0"], lv, out=gather, mode="clip")
    np.subtract(ix, gather, out=ix)
    np.greater_equal(ix, 0, out=live)
    np.less(ix, W, out=aux)
    np.logical_and(live, aux, out=live)

    np.take(scratch["t_y0"], lv, out=gather, mode="clip")
    np.subtract(iy, gather, out=iy)
    np.greater_equal(iy, 0, out=aux)
    np.logical_and(live, aux, out=live)
    np.less(iy, W, out=aux)
    np.logical_and(live, aux, out=live)

    # The wrap, division-free, exactly as `flat_slot_into` argues it: in view,
    # (x0 + col) mod W == (x0 mod W) + col, minus W once if that overflowed,
    # since both terms are in [0, W) and their sum is in [0, 2W).
    np.take(scratch["t_dx"], lv, out=gather, mode="clip")
    np.add(ix, gather, out=ix)
    np.greater_equal(ix, W, out=aux)
    np.subtract(ix, W, out=ix, where=aux)

    np.take(scratch["t_dy"], lv, out=gather, mode="clip")
    np.add(iy, gather, out=iy)
    np.greater_equal(iy, W, out=aux)
    np.subtract(iy, W, out=iy, where=aux)

    np.multiply(iy, W, out=iy)
    np.add(iy, ix, out=iy)
    np.take(scratch["t_off"], lv, out=gather, mode="clip")
    np.add(iy, gather, out=iy)

    # A point past the last ring is not in ring 0, and must not take ring 0's
    # slot -- silently clamping it there is how a 100 m return ends up written
    # into a 5 cm cell at the origin.
    np.greater_equal(level, 0, out=aux)
    np.logical_and(live, aux, out=live)

    idx = out[:n]
    np.copyto(idx, iy, where=live)
    np.logical_not(live, out=aux)
    np.copyto(idx, -1, where=aux)
    return idx


# --- toroidal ring buffers, math §2.4 ---------------------------------------


def ring_extent(schedule, ring: int) -> int:
    """N_L: cells per side of ring L's square buffer.

    Careful with the notation: math §2.4 uses N_L for this LINEAR extent
    (Ring 3 -> 500), while §6.1 eq. (19) uses N_L for an annulus cell COUNT
    (Ring 3 -> 187,500). They are different numbers and the docs reuse the
    symbol. This function is the §2.4 one.
    """
    r = schedule.rings[ring]
    n = 2.0 * r.half_width_m / r.cell_m
    if abs(n - round(n)) > 1e-9:
        raise ValueError(
            f"ring {ring}: extent {2 * r.half_width_m} m / {r.cell_m} m = {n} "
            "is not a whole number of cells"
        )
    return round(n)


def ring_slice(schedule, ring: int) -> slice:
    """Where ring L's square buffer lives in the flat SoA arrays."""
    start = sum(ring_extent(schedule, level) ** 2 for level in range(ring))
    return slice(start, start + ring_extent(schedule, ring) ** 2)


def buffer_cells(schedule) -> int:
    """Total cells actually ALLOCATED: sum of N_L^2 over rings.

    This is NOT schedule.total_cells. That figure counts square annuli
    (eq. 19), which is the map's logical extent; a toroidal ring buffer has
    to store the full square per ring because the hole moves through the
    buffer as the vehicle drives. See the note in tests/test_lattice.py --
    the difference is a headline memory number and it is Shrestha's call.
    """
    return sum(ring_extent(schedule, level) ** 2 for level in range(len(schedule.rings)))


def alloc_ring_buffers(schedule) -> dict:
    """Preallocate every ring buffer plus its origin bookkeeping, once.

    A stand-in for gpu.allocators.allocate() so the grid is not blocked on
    it; it wraps the frozen alloc_soa() and adds nothing to the cell struct.
    `ring_origin[L]` is the world index of the first row/column of ring L's
    window -- the offsets o_L of eq. (10), kept as state instead of moving
    data.
    """
    soa = alloc_soa(buffer_cells(schedule))
    soa["ring_origin"] = np.zeros((len(schedule.rings), 2), dtype=np.int64)
    return soa


def toroidal_shift(soa, schedule, delta_cells) -> int:
    """Ego-motion shift of the ring buffers, O(perimeter) clear, in place.

    Math §2.4. Nothing is copied: the window slides by advancing the offset
    of eq. (10), and only the newly exposed strip is cleared. Ring 3 clears
    2N = 1,000 cells per unit step instead of N^2 = 250,000 -- the difference
    between a sub-millisecond shift and a 40 ms stall.

    `delta_cells` is (dx, dy) in COARSEST cells, which is the §2.4 constraint:
    the origin may only move in whole coarsest-cell steps (40 cm), or every
    ring boundary shifts by a fraction of a cell and you have to resample --
    precisely the "data loss during projection" the brief warns about. Finer
    rings therefore shift by a whole multiple, k_coarsest / k_L, which is an
    integer because the schedule validator demands integer ratios.

    Newly exposed cells get `allocators.EMPTY_CELL`, the same state
    `allocate()` starts the map in -- NOT raw zeros. Nine of the ten fields
    really are zero when empty (obs_count = 0, and §10.1 decides unknown by
    observation count, never by log-odds near zero); `ceiling_height` is the
    one whose empty value is a sentinel. Zeroing it says "something solid at
    the ground datum", so `ceiling - ground < h_vehicle` holds across the
    whole strip and §7.1 bit 0 marks it untraversable forever -- nothing ever
    raises a ceiling back up. `gpu.shift.shift()` has always filled this way;
    this function zeroed instead, so the two spellings of one operation
    disagreed about what an empty cell is.

    Returns the number of cells cleared, so the O(perimeter) claim is
    measurable rather than asserted.
    """
    dx, dy = (int(delta_cells[0]), int(delta_cells[1]))
    k_coarsest = schedule.k(len(schedule.rings) - 1)
    origins = soa["ring_origin"]
    cleared = 0

    for level in range(len(schedule.rings)):
        n = ring_extent(schedule, level)
        scale = k_coarsest // schedule.k(level)  # integer by validate()
        step = (dx * scale, dy * scale)

        # `axis` here is a WORLD axis: 0 is x, 1 is y, matching the (dx, dy) of
        # `delta_cells` and the (x0, y0) columns of `ring_origin`. It is not a
        # numpy axis -- see `_clear_strip`, which is where the two meet.
        for axis in (0, 1):
            s = step[axis]
            if s == 0:
                continue
            # Newly exposed slots are [origin + min(s,0), +|s|) mod N, for
            # either sign -- the window's leading edge in the direction of
            # travel maps onto the slots the trailing edge just vacated.
            start = int(origins[level, axis]) + min(s, 0)
            cleared += _clear_strip(soa, schedule, level, n, start, abs(s), axis)
            origins[level, axis] += s

    return cleared


def _clear_strip(soa, schedule, level, n, start, width, axis) -> int:
    """Empty `width` wrapped lattice lines of one ring, world `axis` 0 = x.

    ⚑ The world axis and the numpy axis are TRANSPOSED, and getting it wrong
      is silent. A slot is `iy * W + ix` (`bin_points`, `RingBuffer.slot`), so
      `reshape(n, n)[a, b]` is `[iy, ix]`: moving in **x** exposes a strip of
      constant `ix`, which is a numpy **column**, `view[:, idx]`. This
      function ran the two the other way round, so a pure +x shift cleared a
      row of constant y -- it kept stale cells in the strip the window had
      actually just uncovered and wiped live ones on an edge that had not
      moved. Every test shifted and then un-shifted by the same vector, which
      is symmetric in the bug, so nothing caught it.

    Filled with `EMPTY_CELL`, not zeros: see `toroidal_shift`.
    """
    if width <= 0:
        return 0
    sl = ring_slice(schedule, level)
    if width >= n:
        for name, _ in CELL_FIELDS:
            soa[name][sl] = EMPTY_CELL.get(name, 0)
        return n * n

    idx = np.arange(start, start + width) % n
    for name, _ in CELL_FIELDS:
        view = soa[name][sl].reshape(n, n)
        if axis == 0:
            view[:, idx] = EMPTY_CELL.get(name, 0)   # x moved -> a column strip
        else:
            view[idx, :] = EMPTY_CELL.get(name, 0)   # y moved -> a row strip
        soa[name][sl] = view.reshape(-1)
    return width * n
