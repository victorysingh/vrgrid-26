#!/usr/bin/env python3
"""Day-3 evidence for the cupy port. [Shrestha]

Three things the port plan asks for, all measured rather than argued:

  1. `--determinism`  int32 scatter_add is bit-identical on device across
     repeated runs, and a float32 control is not. This is
     `docs/gpu-lane/03-CUDA-PORT-PLAN.md` §2, the cycle's headline claim.
  2. `--probe`        which of `bin_points`' numpy vocabulary cupy actually
     supports. Two gaps, both real; see §2 of 06-DAY3-CUPY-FINDINGS.md.
  3. `--bench`        the bin_points arithmetic chain, CPU vs device,
     allocation-free on both sides as production is.

    python scripts/bench_cupy_seam.py --all

⚑ cupy needs the CUDA headers to JIT. The pip toolkit ships them but cupy does
  not always find them, so export CUDA_PATH at the `nvidia/cuda_runtime`
  directory inside site-packages if you get "Failed to find CUDA headers".

Benchmark discipline is §7 of the port plan: 10 warmup iterations discarded,
an explicit stream synchronise before the timer stops, p50 and p99 never the
mean, and device and host reported as two columns rather than one ratio.
"""
import argparse
import time

import numpy as np

N_POINTS = 120_000
RINGS = 4


def _host_inputs(n=N_POINTS, seed=7):
    rng = np.random.default_rng(seed)
    return {
        "xw": rng.uniform(-100, 100, n), "yw": rng.uniform(-100, 100, n),
        "lv": rng.integers(0, RINGS, n).astype(np.int64),
        "t_k": np.array([1, 2, 4, 8], np.int64),
        "t_side": np.full(RINGS, 400, np.int64),
        "t_off": np.array([0, 160_000, 320_000, 480_000], np.int64),
    }


def _scratch(xp, n):
    return {
        "o": xp.zeros(n, xp.int64), "f": xp.zeros(n, xp.float64),
        "ix": xp.zeros(n, xp.int64), "iy": xp.zeros(n, xp.int64),
        "kk": xp.zeros(n, xp.int64), "tmp": xp.zeros(n, xp.int64),
        "live": xp.zeros(n, xp.bool_), "aux": xp.zeros(n, xp.bool_),
    }


def chain(xp, d, s, clip):
    """The `bin_points` arithmetic, allocation-free.

    `clip` selects the numpy spelling (`take(mode="clip")` and
    `subtract(where=)`); cupy supports neither, so the device path uses plain
    `take` and a `copyto(where=)`. Both produce the same answer **only while
    `lv` stays in range** -- see the wrap-vs-clamp note in the findings doc.
    """
    o, f, ix, iy, kk, tmp, live, aux = (
        s[k] for k in ("o", "f", "ix", "iy", "kk", "tmp", "live", "aux"))
    tk = {"mode": "clip"} if clip else {}

    xp.take(d["t_k"], d["lv"], out=kk, **tk)
    xp.floor_divide(d["xw"], 0.05, out=f)
    xp.copyto(ix, f, casting="unsafe")
    xp.floor_divide(ix, kk, out=ix)
    xp.floor_divide(d["yw"], 0.05, out=f)
    xp.copyto(iy, f, casting="unsafe")
    xp.floor_divide(iy, kk, out=iy)

    W = kk
    xp.take(d["t_side"], d["lv"], out=W, **tk)
    xp.greater_equal(ix, 0, out=live)
    xp.less(ix, W, out=aux)
    xp.logical_and(live, aux, out=live)
    xp.greater_equal(iy, 0, out=aux)
    xp.logical_and(live, aux, out=live)

    xp.greater_equal(ix, W, out=aux)
    if clip:
        xp.subtract(ix, W, out=ix, where=aux)
    else:
        xp.subtract(ix, W, out=tmp)
        xp.copyto(ix, tmp, where=aux)

    xp.multiply(iy, W, out=iy)
    xp.add(iy, ix, out=iy)
    xp.take(d["t_off"], d["lv"], out=o, **tk)
    xp.add(iy, o, out=iy)
    return iy


def _bench(fn, sync, iters=200, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


def run_determinism(cp, runs=30, n=2_000_000, cells=4096):
    import cupyx
    rng = np.random.default_rng(0)
    idx = cp.asarray(rng.integers(0, cells, n, dtype=np.int64))
    print(f"int32 vs float32 scatter_add, {n:,} returns into {cells:,} cells, "
          f"{runs} runs")
    for label, vals, dt in (
        ("int32  ", cp.asarray(rng.integers(-800, 800, n).astype(np.int32)), cp.int32),
        ("float32", cp.asarray(rng.random(n).astype(np.float32)), cp.float32),
    ):
        ref, differed = None, 0
        for _ in range(runs):
            t = cp.zeros(cells, dtype=dt)
            cupyx.scatter_add(t, idx, vals)
            cp.cuda.Stream.null.synchronize()
            h = t.get().tobytes()
            if ref is None:
                ref = h
            elif h != ref:
                differed += 1
        verdict = "BIT-IDENTICAL" if differed == 0 else f"{differed}/{runs - 1} differed"
        print(f"  {label}: {verdict}")


def run_probe(cp):
    print(f"bin_points vocabulary on cupy {cp.__version__}")
    n = 10
    a = cp.arange(n, dtype=cp.int64)
    b = cp.ones(n, dtype=cp.int64)
    o = cp.zeros(n, dtype=cp.int64)
    m = cp.ones(n, dtype=cp.bool_)
    lv = cp.zeros(n, dtype=cp.int64)
    tbl = cp.arange(RINGS, dtype=cp.int64)
    f = cp.zeros(n, dtype=cp.float64)
    cases = (
        ("take(out=, mode='clip')", lambda: cp.take(tbl, lv, out=o, mode="clip")),
        ("take(out=)", lambda: cp.take(tbl, lv, out=o)),
        ("floor_divide(out=)", lambda: cp.floor_divide(a, b, out=o)),
        ("subtract(out=, where=)", lambda: cp.subtract(a, b, out=o, where=m)),
        ("copyto(where=)", lambda: cp.copyto(o, a, where=m)),
        ("copyto(casting='unsafe')", lambda: cp.copyto(o, f, casting="unsafe")),
        ("add.reduceat", lambda: cp.add.reduceat(a, cp.array([0, 5]))),
        ("minimum.at", lambda: cp.minimum.at(o, cp.array([0, 1]), cp.array([1, 2]))),
        ("argsort", lambda: cp.argsort(a)),
    )
    for label, fn in cases:
        try:
            fn()
            print(f"  ok    {label}")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  FAILS {label:28s} {type(exc).__name__}")

    # The one that is a correctness trap rather than a missing keyword.
    host = np.array([10, 20, 30, 40], np.int64)
    dev = cp.asarray(host)
    for name, idx in (("out of range", [9]), ("negative", [-1])):
        h = np.take(host, np.array(idx), mode="clip")
        d = cp.take(dev, cp.array(idx)).get()
        note = "AGREE" if np.array_equal(h, d) else "DIVERGE"
        print(f"  {note:8s} {name}: numpy clip -> {h}, cupy -> {d}")


def run_bench(cp):
    d = _host_inputs()
    sh = _scratch(np, N_POINTS)
    p50c, p99c = _bench(lambda: chain(np, d, sh, True), lambda: None)

    dev = {k: cp.asarray(v) for k, v in d.items()}
    sd = _scratch(cp, N_POINTS)
    sync = cp.cuda.Stream.null.synchronize
    p50g, p99g = _bench(lambda: chain(cp, dev, sd, False), sync)

    same = np.array_equal(chain(np, d, sh, True), cp.asnumpy(chain(cp, dev, sd, False)))
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"bin_points arithmetic, allocation-free, N = {N_POINTS:,}")
    print(f"  CPU  numpy       : p50 {p50c:6.3f} ms   p99 {p99c:6.3f} ms")
    print(f"  GPU  {name[:14]:14s}: p50 {p50g:6.3f} ms   p99 {p99g:6.3f} ms")
    print(f"  p50 speedup      : {p50c / p50g:.1f}x")
    print(f"  bit-identical    : {'yes' if same else 'NO -- STOP'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not any((args.determinism, args.probe, args.bench, args.all)):
        args.all = True

    try:
        import cupy as cp
    except ImportError:
        raise SystemExit(
            "cupy is not installed. `pip install cupy-cuda12x[ctk]`, and see the "
            "CUDA_PATH note in this file's docstring.")

    for flag, fn in ((args.determinism, run_determinism),
                     (args.probe, run_probe), (args.bench, run_bench)):
        if flag or args.all:
            fn(cp)
            print()


if __name__ == "__main__":
    main()
