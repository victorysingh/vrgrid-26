# PROVENANCE -- written 2026-09-14 for SIH26053's real-time requirement, to be RUN ON AWS
#            (g4dn.xlarge, NVIDIA T4) per JP's decision. Do NOT install a CUDA build of PyTorch
#            on the local laptop; this harness refuses to run without CUDA unless told otherwise.
#
# Produced: per-frame FRNet inference latency on a GPU -- p50 / p99 / max and FPS -- with and
#           without --fast-scatter, for the same checkpoint and the same predict() call the rest of
#           the project scores.
#
# Run (on the AWS instance, from the repo root, after the setup below):
#   python reports/harnesses/frnet_gpu_latency.py --fast-scatter --json reports/bench/frnet_gpu_latency_fast.json
#   python reports/harnesses/frnet_gpu_latency.py                --json reports/bench/frnet_gpu_latency_loop.json --frames 40
#
# Setup on the instance (real data only, nothing synthetic):
#   1. the repo at the same commit; `pip install -e .` plus a CUDA build of torch (on AWS only);
#   2. checkpoints/frnet-semantickitti_seg.pth copied over. The harness checks its SHA-256 against
#      09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e and refuses a mismatch;
#   3. SemanticKITTI sequences/08/velodyne and labels for at least the frames timed, under
#      $VRGRID_DATA_ROOT. If they are missing, the run stops: do not substitute anything.
#
# NOTE: What is timed, and why each choice.
#   - Scans are loaded into memory BEFORE timing. Disk I/O is the pipeline's `load` stage, measured
#     separately; including it here would time the instance's EBS volume, not the network.
#   - Each frame's timing INCLUDES the host->device copy of the points and the device->host copy of
#     the predictions. Both are unavoidable in a pipeline whose map runs on the CPU.
#   - torch.cuda.synchronize() brackets every measurement. CUDA calls return before the work is
#     done, so without it the numbers would be queue-submission times.
#   - The first --warmup frames are excluded (cuDNN autotuning, allocator growth, lazy init).
#   - [!] On CUDA, --fast-scatter's scatter_mean differs from the loop by up to 2 float32 ulp (see
#     scripts/frnet_fast_scatter.py). The harness reports point accuracy on the timed frames so a
#     GPU-side accuracy change would be visible next to the latency.
#   - [!] This is the NETWORK's latency on a T4, not the whole pipeline's. The end-to-end DL
#     pipeline figure must come from the full pipeline on the same instance; see the D11 report.
#
# Measurement only. Changes nothing in src/ or scripts/.
"""FRNet per-frame inference latency on CUDA (AWS), with and without --fast-scatter."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e"
ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--checkpoint", default="checkpoints/frnet-semantickitti_seg.pth")
    ap.add_argument("--fast-scatter", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="smoke-test the harness without CUDA; the numbers are then NOT a GPU result")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import torch
    from vrgrid.perception import loader, semantics
    from vrgrid.perception.frnet import FRNet

    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        sys.exit("CUDA is not available. This harness is for the AWS GPU run; pass --allow-cpu only "
                 "to smoke-test it, and never report that output as a GPU latency.")
    dev = torch.device("cuda" if cuda else "cpu")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(f"checkpoint SHA-256 {digest} != {EXPECTED_SHA256} -- refusing to time a different file")

    if args.fast_scatter:
        sys.path.insert(0, str(ROOT / "scripts"))
        from frnet_fast_scatter import enable
        enable(verify=True, device=str(dev))

    model = FRNet(num_classes=20, ignore_index=19, output_shape=(64, 512),
                  fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG, fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob.get("state_dict", blob), strict=False)
    model.to(dev).eval()

    root = Path(loader.DATA_ROOT) / "sequences" / args.seq
    scans, gts = [], []
    for i in range(args.frames):
        s, lab = root / "velodyne" / f"{i:06d}.bin", root / "labels" / f"{i:06d}.label"
        if not s.exists() or not lab.exists():
            sys.exit(f"frame {i} missing under {root} -- stopping; do not substitute data")
        scans.append(loader.load_velodyne_scan(s))
        gts.append(semantics.semantic_labels(loader.load_labels(lab)))
    if args.frames <= args.warmup:
        sys.exit(f"--frames ({args.frames}) must exceed --warmup ({args.warmup})")

    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if cuda else None,
           "cpu": platform.processor(), "python": platform.python_version(),
           "fast_scatter": args.fast_scatter, "checkpoint_sha256": digest}
    try:
        env["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        env["nvidia_smi"] = None
    print(json.dumps(env, indent=2))

    def sync():
        if cuda:
            torch.cuda.synchronize()

    times_ms, correct, total = [], 0, 0
    with torch.no_grad():
        for i, (pts, gt) in enumerate(zip(scans, gts)):
            sync()
            t0 = time.perf_counter()
            pred = model.predict([torch.from_numpy(pts).float().to(dev)])[0].cpu().numpy()
            sync()
            dt = (time.perf_counter() - t0) * 1e3
            if i >= args.warmup:
                times_ms.append(dt)
                ok = gt >= 0
                correct += int((pred[ok] == gt[ok]).sum())
                total += int(ok.sum())

    t = np.asarray(times_ms)
    res = {"frames_timed": int(t.size), "warmup": args.warmup,
           "p50_ms": float(np.median(t)), "p99_ms": float(np.percentile(t, 99)),
           "max_ms": float(t.max()), "mean_ms": float(t.mean()),
           "fps_p50": 1e3 / float(np.median(t)), "fps_p99": 1e3 / float(np.percentile(t, 99)),
           "point_accuracy_timed_frames": correct / total if total else None}
    print(f"\nFRNet inference on {env['gpu'] or 'CPU (smoke test, NOT a GPU result)'}"
          f"{' with --fast-scatter' if args.fast_scatter else ' (port loops)'}: "
          f"{res['frames_timed']} frames after {args.warmup} warm-up")
    print(f"  p50 {res['p50_ms']:.1f} ms  p99 {res['p99_ms']:.1f} ms  max {res['max_ms']:.1f} ms  "
          f"-> {res['fps_p50']:.1f} FPS p50, {res['fps_p99']:.1f} FPS p99")
    print(f"  point accuracy on the timed frames: {res['point_accuracy_timed_frames']:.2%}")
    print("  [!] network only -- the DL pipeline's end-to-end latency needs the full pipeline on this "
          "instance")
    if args.json:
        Path(args.json).write_text(json.dumps({"env": env, "result": res, "frame_ms": t.tolist()},
                                              indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
