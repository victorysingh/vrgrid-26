"""The operations `array_module()` does not cover. [Shrestha]

`allocators.array_module()` returns numpy or cupy and the intent was that the
kernels then read the same on both. Measured on cupy 14.2.0, they do not: four
of the primitives `scatter_sorted`, `bin_points` and `visibility_cleanup` are
built from either do not exist on device, or exist with different semantics.
See `docs/gpu-lane/06-DAY3-CUPY-FINDINGS.md` for the probe.

    | numpy spelling                | cupy 14.2.0                          |
    | ----------------------------- | ------------------------------------ |
    | `add.reduceat`                | works, matches, honours `out=`       |
    | `minimum.reduceat`            | NotImplementedError                  |
    | scatter-min on int16          | TypeError: int32/64, float32/64 only |
    | `arr.sort(kind="quicksort")`  | ValueError: only None or "stable"    |
    | `take(..., mode="clip")`      | TypeError: no `mode` argument        |
    | ufunc `..., where=mask`       | TypeError: unexpected keyword        |

⚑ THE CPU PATH MUST NOT MOVE. Every function here delegates to the exact numpy
  spelling the kernels already used when `xp is np`, so a CPU run is
  byte-identical to before this module existed. The determinism gate is
  CI-blocking and these are the reductions it gates; a "harmless" tidy-up of
  the numpy branch is not harmless.

⚑ The device branches are order-independent by construction -- integer add and
  min are associative and commutative -- so they are deterministic for the same
  reason the CPU ones are, not by accident of cupy's scheduling. That is the
  claim `docs/gpu-lane/03-CUDA-PORT-PLAN.md` §2 rests on, and
  `tests/test_gpu_compat.py` re-checks it against numpy on every backend.
"""
import numpy as np

__all__ = ["is_cupy", "segment_min", "sort_inplace", "subtract_where", "take_clip"]


def is_cupy(xp) -> bool:
    """True when `xp` is the cupy module. Cheap and import-free on CPU."""
    return xp.__name__ == "cupy"


def take_clip(xp, source, indices, out=None):
    """`np.take(source, indices, out=out, mode="clip")`, on either backend.

    ⚑ `mode="clip"` in the numpy path is NOT about clipping. numpy's default
      `mode="raise"` performs its bounds check by materialising a full-length
      index array -- 0.96 MB per call in `bin_points`, six calls a frame -- and
      `clip` skips it. Every caller already clamps its indices into range, so
      clipping never actually clips.

    ⚑ ON DEVICE THE OUT-OF-RANGE BEHAVIOUR DIFFERS AND IT IS SILENT. cupy has
      no `mode`, and its default WRAPS where numpy's clip CLAMPS: index 9 into
      a length-4 table gives 40 on numpy and 20 on cupy; -1 gives 10 against
      40. Callers are therefore responsible for the in-range invariant, and on
      device a broken caller reads a different wrong element than it would on
      CPU -- CPU and GPU would disagree only in the broken case. Assert the
      clamp at the call site; do not rely on this function to catch it.
    """
    if is_cupy(xp):
        return xp.take(source, indices, out=out)
    return np.take(source, indices, out=out, mode="clip")


def sort_inplace(xp, arr):
    """Sort `arr` in place, ascending.

    The kernels ask numpy for `kind="quicksort"`; cupy accepts only None or
    "stable". It does not matter here and the reason is worth stating rather
    than discovering: `scatter_sorted` packs a unique point id into every key,
    so no two keys compare equal and the order is total. Stability is a
    statement about ties, and there are none.
    """
    if is_cupy(xp):
        arr.sort()
    else:
        arr.sort(kind="quicksort")
    return arr


def subtract_where(xp, a, b, out, where, scratch=None):
    """`xp.subtract(a, b, out=out, where=where)`, on either backend.

    cupy's ufuncs reject `where=` outright. `copyto(where=)` does work, so the
    device path computes the difference and copies it back under the mask. That
    needs one full-width temporary; pass `scratch` to keep the frame path
    allocation-free, which is a hard invariant of this project.
    """
    if not is_cupy(xp):
        xp.subtract(a, b, out=out, where=where)
        return out
    tmp = xp.subtract(a, b) if scratch is None else xp.subtract(a, b, out=scratch)
    xp.copyto(out, tmp, where=where)
    return out


def segment_min(xp, values, seg_starts, out, n_segments=None, segment_id=None):
    """`np.minimum.reduceat(values, seg_starts, out=out)`, on either backend.

    This is the ceiling column: the lowest thing overhead in each cell, over a
    `values` array already sorted so that a cell's returns are contiguous.

    cupy implements no `minimum.reduceat` at all. The device path converts the
    segment STARTS into a segment ID per element with a cumsum, then reduces
    with an atomic scatter-min. Min is associative and commutative, so the
    result does not depend on the order the atomics land in -- this is exactly
    the argument that makes the integer height sum safe, applied to a different
    operator, and it is checked rather than assumed in the tests.

    ⚑ Widened to int32 on device and narrowed back. cupy's scatter-min accepts
      only int32/int64/float32/float64, and the ceiling column is int16 cm.
      Every int16 value is exactly representable in int32 and the reduction is
      a min, so nothing rounds and nothing saturates; the narrowing is exact.

    `segment_id` may be supplied to avoid rebuilding the cumsum each frame.
    """
    if not is_cupy(xp):
        np.minimum.reduceat(values, seg_starts, out=out)
        return out

    k = out.shape[0] if n_segments is None else n_segments
    if segment_id is None:
        flags = xp.zeros(values.shape[0], dtype=xp.int32)
        # A segment boundary at element 0 would set the id of the first element
        # to 1 and shift every id by one, so only the interior starts mark.
        if k > 1:
            flags[seg_starts[1:]] = 1
        segment_id = xp.cumsum(flags, dtype=xp.int64)

    wide = xp.full(k, np.iinfo(np.int16).max, dtype=xp.int32)
    xp.minimum.at(wide, segment_id, values.astype(xp.int32))
    xp.copyto(out, wide.astype(out.dtype))
    return out
