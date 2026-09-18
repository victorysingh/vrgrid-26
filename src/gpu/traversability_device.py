"""The §7.1 bitfield on the card, bit-identical to the host. [Shrestha]

`traversability.bitfield` is NumPy over every cell of every ring window, 910,000
cells on 5/10/20/40, and stayed at ~30 ms a frame on real seq 08 after the host
rewrite. This is the same predicate in CuPy.

Every operation is exact on the device -- integer arithmetic, boolean masks,
`int -> float64 / 100`, division and comparisons were checked bit for bit
against NumPy on millions of values -- except one: **CUDA's `hypot` differs
from glibc's in the last bit on ~30% of inputs.** The slope bit compares
`hypot(dz/dx, dz/dy)` against `tan(theta_max)`, so a cell within an ulp of the
threshold could flip. It is handled rather than hoped away: the device computes
the slope, cells whose device slope lies within a relative 1e-12 of the
threshold -- a band ten thousand times wider than the device's error -- are
downloaded and decided with NumPy's `hypot`, and everything else is decided on
the card. The band is almost always empty, so the round trip costs nothing,
and the result is exact by construction. `test_device_bitfield_matches_host`
pins it, including with the band widened until most cells are settled on the host.
"""

import numpy as np
from vrgrid.cell import (
    TRAV_CLASS,
    TRAV_CLEARANCE,
    TRAV_CONFIDENCE,
    TRAV_ROUGHNESS,
    TRAV_SLOPE,
    TRAV_STEP,
)

BAND = 1e-12
_DEVICE_PLANS = {}


def _device_plan(cp, side, cell_m, th):
    from vrgrid.grid.traversability import _plan

    host = _plan(side, cell_m, th)
    key = id(host)
    plan = _DEVICE_PLANS.get(key)
    if plan is None:
        plan = {k: (cp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in host.items()}
        _DEVICE_PLANS[key] = plan
    return plan


def bitfield(cp, fields, side: int, cell_m: float, th):
    """`traversability.bitfield` for one ring, from device arrays. `fields` maps
    ground_height / ceiling_height / height_variance / obs_count /
    semantic_class to that ring's device arrays. Returns a device uint8 array."""
    P = _device_plan(cp, side, cell_m, th)
    ip, im = P["ip"], P["im"]

    ground = fields["ground_height"].astype(cp.int32)
    n = fields["obs_count"]
    out = cp.zeros(ground.size, dtype=cp.uint8)

    out |= cp.where((fields["ceiling_height"].astype(cp.int32) - ground) < P["h_cm"],
                    cp.uint8(TRAV_CLEARANCE), cp.uint8(0))

    seen = (n >= 1).reshape(side, side)
    geometric = (seen & seen[:, ip] & seen[:, im] & seen[ip, :] & seen[im, :]).reshape(-1)

    z = ground.astype(cp.float64).reshape(side, side) / 100.0
    dzdx = ((z[:, ip] - z[:, im]) / P["den_x"]).reshape(-1)
    dzdy = ((z[ip, :] - z[im, :]) / P["den_y"]).reshape(-1)
    slope = cp.hypot(dzdx, dzdy)
    tan = P["tan"]
    over = slope > tan
    near = geometric & (cp.abs(slope - tan) <= BAND * tan)
    if bool(near.any()):
        idx = cp.flatnonzero(near)
        host = np.hypot(dzdx[idx].get(), dzdy[idx].get()) > tan
        over[idx] = cp.asarray(host)
    out |= cp.where(geometric & over, cp.uint8(TRAV_SLOPE), cp.uint8(0))

    zi = ground.reshape(side, side)
    step = cp.abs(zi[:, ip] - zi)
    cp.maximum(step, cp.abs(zi[:, im] - zi), out=step)
    cp.maximum(step, cp.abs(zi[ip, :] - zi), out=step)
    cp.maximum(step, cp.abs(zi[im, :] - zi), out=step)
    out |= cp.where(geometric & (step.reshape(-1) > P["step_cm"]), cp.uint8(TRAV_STEP), cp.uint8(0))

    out |= cp.where(P["rough"][fields["height_variance"]], cp.uint8(TRAV_ROUGHNESS), cp.uint8(0))
    out |= cp.where(P["nondrivable"][fields["semantic_class"]], cp.uint8(TRAV_CLASS), cp.uint8(0))
    out |= cp.where((n < P["n_min"]) | P["border"], cp.uint8(TRAV_CONFIDENCE), cp.uint8(0))
    return out


FIELDS = ("ground_height", "ceiling_height", "height_variance", "obs_count", "semantic_class")


def update(soa, schedule, rings, th) -> None:
    """`traversability.update` with the bitfield computed on the card: uploads
    the five input fields once, computes every ring, downloads the result into
    `soa["traversability"]` in place."""
    import cupy as cp

    dev = {name: cp.asarray(soa[name]) for name in FIELDS}
    result = cp.empty(soa["traversability"].shape, dtype=cp.uint8)
    for level, (sl, side) in enumerate(rings):
        result[sl] = bitfield(cp, {k: v[sl] for k, v in dev.items()}, side,
                              schedule.rings[level].cell_m, th)
    result.get(out=soa["traversability"])
