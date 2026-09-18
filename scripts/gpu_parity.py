#!/usr/bin/env python3
"""The GPU pipeline on real data: same outputs, bit for bit, and what it costs.
[Shrestha]

    python scripts/gpu_parity.py --seq 08 --frames 200

For every scan, perception runs twice -- the CPU stages (JP's functions) and
the device stages (`gpu.device.DevicePerception`) -- and each frame is folded
into its own engine: `MapEngine(device="cpu")` and `MapEngine(device="cuda")`,
whose grid lives on the card. After every frame this compares, exactly:

    range image (NaN-aware), inverse index, reflectivity bytes, semantic
    labels, motion flags, every StepCounters field, and the full-grid hash

and exits 1 naming the first frame and the first thing that differs.

Ground segmentation runs ONCE per scan and both paths receive that mask.
Patchwork++ is a stateful estimator and two replays of it disagree (open item
D1, `docs/gpu-lane/07-LOCAL-BUILD.md`), so running it twice would make the
comparison about Patchwork++ rather than about the GPU. It is also the one
stage that stays on the host in both configurations.

Timing is not taken here: two engines interleaved on one machine distort each
other. `scripts/timing_table.py --seq 08 --device cuda` is the latency table.
"""
import argparse
import sys
import warnings

import numpy as np


def _first_difference(host, dev):
    checks = {
        "range_image": lambda: np.array_equal(host.range_image, dev.range_image,
                                              equal_nan=True),
        "inverse_index": lambda: np.array_equal(host.inverse_index, dev.inverse_index),
        "reflectivity8": lambda: np.array_equal(host.reflectivity8, dev.reflectivity8),
        "semantic": lambda: np.array_equal(host.semantic, dev.semantic),
        "moving": lambda: np.array_equal(host.moving, dev.moving),
    }
    for name, ok in checks.items():
        if not ok():
            return name
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--schedule", default="5/10/20/40")
    ap.add_argument("--no-patchworkpp", action="store_true")
    ap.add_argument("--max-points", type=int, default=150_000,
                    help="engine point cap; below a scan's size it truncates")
    ap.add_argument("--show-ghosts", action="store_true",
                    help="run both engines with the §10.4 cleanup OFF")
    args = ap.parse_args(argv)

    from vrgrid.gpu.device import DevicePerception
    from vrgrid.gpu.kernels import map_hash
    from vrgrid.grid.schedule import load
    from vrgrid.perception import ground, loader, semantics
    from vrgrid.run.__main__ import perceive
    from vrgrid.run.engine import MapEngine

    sched = load(args.schedule)
    # attrition=True: the per-stage return counts are part of the counters
    # compared below, so stage attrition is checked on both devices every frame.
    engines = {d: MapEngine(sched, ghost_removal=not args.show_ghosts, device=d,
                            max_points=args.max_points, attrition=True)
               for d in ("cpu", "cuda")}
    perception = DevicePerception()
    ground.reset_estimator()
    use_pw = not args.no_patchworkpp

    n = cleared = 0
    stages = {}
    for i, (points, labels, pose) in enumerate(
            loader.scans(args.seq, max_frames=args.frames, start_frame=args.start_frame)):
        index = args.start_frame + i
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gres = ground.segment_ground_or_fallback(
                points, semantics.semantic_labels(labels), use_patchworkpp=use_pw)
        host = perceive(points, labels, pose, args.seq, index, ground_result=gres)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dev = perceive(points, labels, pose, args.seq, index,
                           perception=perception, ground_result=gres)

        what = _first_difference(host, dev)
        if what is not None:
            print(f"MISMATCH at frame {index}: perception output `{what}` differs")
            return 1

        c_cpu = engines["cpu"].step(host)
        c_gpu = engines["cuda"].step(dev)
        h_cpu = map_hash(engines["cpu"].handle.grid)
        h_gpu = map_hash(engines["cuda"].handle.grid)
        if c_cpu != c_gpu or h_cpu != h_gpu:
            print(f"MISMATCH at frame {index}: map\n  cpu  {h_cpu}  {c_cpu}\n"
                  f"  cuda {h_gpu}  {c_gpu}")
            return 1
        cleared += c_gpu.cleared
        for k, v in c_gpu.attrition.items():
            stages[k] = stages.get(k, 0) + v
        n += 1
        if n % 25 == 0:
            print(f"  frame {index}: identical -- perception and map; "
                  f"{c_gpu.occupied:,} occupied, {c_gpu.cleared:,} cleared, hash {h_gpu}")

    if n == 0:
        print("no frames read -- is VRGRID_DATA_ROOT set?")
        return 1
    b = engines["cuda"].device_bytes()
    print(f"\nsequence {args.seq}, frames {args.start_frame}..{args.start_frame + n - 1} "
          f"({n}), schedule {args.schedule}, ground {gres[1]}, ghost removal "
          f"{'OFF' if args.show_ghosts else 'ON'}")
    print(f"IDENTICAL on all {n} frames: range image, inverse index, reflectivity, "
          f"labels, motion, counters, map hash")
    print(f"final map hash {map_hash(engines['cuda'].handle.grid)}; "
          f"{cleared:,} cells cleared by §10.4")
    pts = stages["points"]
    print("stage attrition over the run (identical on both devices, every frame): "
          + ", ".join(f"{k} {stages[k] / pts:.2%}" for k in
                      ("capped", "outside_map", "nonground", "ground_out_of_band",
                       "ground_fused"))
          + f"; beside the chain: moving {stages['moving'] / pts:.2%}, "
            f"projected {stages['projected'] / pts:.2%}")
    print(f"device: {b['static'] / 1e6:.2f} MB map + buffers, pool used "
          f"{b['pool_used'] / 1e6:.2f} MB, reserved {b['pool_reserved'] / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
