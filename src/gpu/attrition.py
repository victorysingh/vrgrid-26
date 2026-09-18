"""Stage attrition: where each return of a sweep leaves the map pipeline. [Shrestha]

`vrgrid-recommended-changes.pdf` §6.6 Rule 3, roadmap Day 7: report STAGE
ATTRITION, not terminal accuracy. An accuracy number measured on what survives
the upstream stages says nothing about what never reached them, and the two
look identical on a map painted one grey.

Every return of a frame ends in exactly one terminal stage, in pipeline order:

    CAPPED              beyond `max_points`; never binned
    OUTSIDE_MAP         binned, but past the coarsest ring's window
    NONGROUND           in the map; updates occupancy, class and ceiling, but
                        carries no height (`fusion`: height is ground-only)
    GROUND_OUT_OF_BAND  ground, in the map, outside the 8 m height band; an
                        observation, but its height weight is zero
                        (`kernels.out_of_band`)
    GROUND_FUSED        ground, in the map, in the band: its height is fused

and two properties are counted beside the chain, because they are not stages a
return drops out at:

    moving              labelled `moving-*`; the map keeps it and §10.4's
                        cleanup clears the trail it leaves
    projected           won its range-image pixel; only these feed the
                        visibility guard and reflectivity

The same functions run on numpy and cupy (`xp`), over the engine's own frame
buffers -- the slots `bin_points` returned, the ground mask `scatter` read, the
height weights `scatter` used -- so the CPU and CUDA engines report identical
counts by construction. `test_attrition_is_identical_on_both_devices` pins it.
"""

import numpy as np

CAPPED, OUTSIDE_MAP, NONGROUND, GROUND_OUT_OF_BAND, GROUND_FUSED = range(5)
NAMES = ("capped", "outside_map", "nonground", "ground_out_of_band", "ground_fused")


def _xp(a):
    import sys
    cp = sys.modules.get("cupy")
    return cp if cp is not None and isinstance(a, cp.ndarray) else np


def counts(n_all: int, idx, ground, w_q, moving, inverse) -> dict:
    """Per-stage counts for one frame. `idx`, `ground` and `w_q` are the first
    `n` (capped) points; `moving` covers all `n_all`; `inverse` is the range
    image's pixel -> point index (-1 where empty)."""
    xp = _xp(idx)
    n = int(idx.shape[0])
    in_map = idx >= 0
    g = ground.astype(bool)
    ground_in = g & in_map
    oob = ground_in & (w_q == 0)
    out = {
        "points": int(n_all),
        "capped": int(n_all - n),
        "outside_map": int(n - int(xp.count_nonzero(in_map))),
        "nonground": int(xp.count_nonzero(in_map & ~g)),
        "ground_out_of_band": int(xp.count_nonzero(oob)),
        "ground_fused": int(xp.count_nonzero(ground_in & ~oob)),
        # These two may live on the host while the rest is on the card (a
        # PerceptionFrame fed to a CUDA engine), so each uses its own module.
        "moving": int(_xp(moving).count_nonzero(moving)),
        "projected": int(_xp(inverse).count_nonzero(inverse >= 0)),
    }
    assert sum(out[k] for k in NAMES) == n_all, "every return ends in exactly one stage"
    return out


def codes(n_all: int, idx, ground, w_q):
    """uint8 terminal stage per return, in point order, for a map colouring
    that splits "examined and rejected" from "never reached the pipeline"."""
    xp = _xp(idx)
    n = int(idx.shape[0])
    out = xp.full(n_all, CAPPED, dtype=xp.uint8)
    head = out[:n]
    head[:] = OUTSIDE_MAP
    in_map = idx >= 0
    g = ground.astype(bool)
    head[in_map & ~g] = NONGROUND
    head[in_map & g & (w_q == 0)] = GROUND_OUT_OF_BAND
    head[in_map & g & (w_q != 0)] = GROUND_FUSED
    return out
