#!/usr/bin/env python3
"""The §9 accuracy table for the map the ENGINE builds -- on the card or not. [Shrestha]

    python scripts/engine_eval.py --seq 08 --frames 40 [--device both]

`eval_synthetic.py` scores the eval harness's map, which is built by
`harness.run_sequence` through `fusion.scatter` -- a CPU path of its own. The
pipeline that actually runs, on the GPU or off it, is `MapEngine`: its own
binning, datum, §10.4 cleanup and, with `--device cuda`, every map stage as a
CUDA kernel. Nothing measured that map against M* until this script. It
answers the question "does the GPU pipeline produce an accurate map on real
data", not only "does it produce the same map as the CPU engine"
(`gpu_parity.py`).

The reference is the same M* the harness uses, built from the same scans with
`band=True`, and the metrics are the same functions, read through a `GridMap`
view of the engine's grid, windows, datum, position and heading. `--device
both` runs the two engines and requires every metric to agree exactly.

The engine and the harness are expected to differ: the engine runs §10.4 ghost
cleanup and slides its datum every frame. The two tables are not meant to be
identical, and the difference is reported, not reconciled.
"""

import argparse
import sys
import warnings

import numpy as np


def run(seq, frames, device):
    from vrgrid.eval import metrics
    from vrgrid.grid.query import GridMap
    from vrgrid.grid.schedule import load
    from vrgrid.run.__main__ import iter_pipeline
    from vrgrid.run.engine import MapEngine

    sched = load("5/10/20/40")
    engine = MapEngine(sched, device=device)
    n = 0
    for frame in iter_pipeline(seq, frames, device=device):
        engine.step(frame)
        n += 1
    soa = {k: np.array(v) for k, v in engine.handle.grid.items()}
    gm = GridMap(soa=soa, schedule=sched, buffers=engine.buffers,
                 thresholds=engine.thresholds, z_datum_m=engine.z_datum,
                 vehicle_xy_m=engine._vehicle_xy, vehicle_yaw_rad=engine._yaw)
    return n, gm, metrics


def table(gm, metrics, reference):
    rmse = metrics.height_rmse_per_ring(gm, reference)
    rho = metrics.coarsening_ratio_per_ring(gm, reference)
    cons = metrics.coarsening_ratio_per_ring(gm, reference, within_cell=False)
    cells = {L: metrics._compared(gm, reference, L)[0].size for L in rmse}
    return {L: (cells[L], rmse[L], rho[L]["rho"], cons[L]["rho"]) for L in rmse}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--device", default="both", choices=["cpu", "cuda", "both"])
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    from vrgrid.eval.harness import real_scans
    from vrgrid.eval.reference_map import build_from_scans

    reference = build_from_scans(real_scans(args.seq, args.frames), band=True)
    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    results = {}
    for d in devices:
        n, gm, metrics = run(args.seq, args.frames, d)
        results[d] = table(gm, metrics, reference)
        print(f"\nsequence {args.seq}, {n} frames, MapEngine --device {d}, vs {reference}")
        print(f"  {'ring':>4} {'cells':>9} {'RMSE cm':>9} {'rho':>6} {'rho cons.':>10}")
        for L, (c, r, p, q) in results[d].items():
            f = (lambda v: f"{'--':>6}" if np.isnan(v) else f"{v:6.2f}")
            print(f"  {L:>4} {c:>9,} {r:>9.2f} {f(p)} {f(q):>10}")

    if len(devices) == 2:
        a, b = results["cpu"], results["cuda"]
        same = all(a[L][0] == b[L][0] and np.allclose(a[L][1:], b[L][1:], equal_nan=True, rtol=0, atol=0)
                   for L in a)
        print(f"\ncpu and cuda engines: {'IDENTICAL' if same else 'DIFFER'} on every ring and metric")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
