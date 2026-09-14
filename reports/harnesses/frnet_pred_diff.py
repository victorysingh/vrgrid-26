# PROVENANCE -- written 2026-09-14 for the --fast-scatter reproducibility finding
#            (OPEN-ITEMS R-j), found during D11 Step 5.
#
# Produced: per-point FRNet predictions on the first N frames of a real sequence,
#           per pass, and a per-frame diff between passes -- to find out whether the
#           run-to-run per-class IoU variation seen on --fast-scatter comes from
#           PyTorch's intra-op thread pool.
# Run:      python reports/harnesses/frnet_pred_diff.py run  --fast-scatter --threads 1 --out a.npz
#           python reports/harnesses/frnet_pred_diff.py diff a.npz b.npz ...
#
# NOTE: `run` is frnet_eval.py's inference, transcribed: the same FRNet construction,
#      the same torch.load / load_state_dict(strict=False), the same
#      model.predict([torch.from_numpy(pts).float()]) call, the same frames in the same
#      order. It saves predictions instead of scoring them.
#
#      `--threads N` calls torch.set_num_threads(N) as the FIRST torch call in the
#      process -- before the shim is enabled, before the model is built -- so it holds
#      for the WHOLE evaluation, not just the shim. The inter-op pool is left at its
#      default and recorded, because the question asked was about set_num_threads.
#
# Measurement only. Changes nothing in src/ or scripts/.
"""Per-point FRNet predictions per pass, and the per-frame diff between passes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e"


def run(args):
    import torch
    if args.threads is not None:
        torch.set_num_threads(args.threads)        # first torch call: whole evaluation
    from vrgrid.perception import loader, semantics
    from vrgrid.perception.frnet import FRNet

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but CUDA is not available")

    ckpt = Path(args.checkpoint)
    h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if h != EXPECTED_SHA256:
        sys.exit(f"checkpoint SHA-256 {h} != expected {EXPECTED_SHA256}")

    if args.fast_scatter:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from frnet_fast_scatter import enable
        enable(verify=True, device=str(dev))

    model = FRNet(num_classes=20, ignore_index=19, output_shape=(64, 512),
                  fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG,
                  fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob.get("state_dict", blob), strict=False)
    model.to(dev).eval()

    meta = {"fast_scatter": args.fast_scatter, "threads_requested": args.threads,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "device": str(dev),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "cuda": torch.version.cuda,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "torch": torch.__version__, "seq": args.seq, "frames": args.frames}
    print("  " + json.dumps(meta))

    root = Path(loader.DATA_ROOT) / "sequences" / args.seq
    preds, gts = {}, {}
    for i in range(args.frames):
        pts = loader.load_velodyne_scan(root / "velodyne" / f"{i:06d}.bin")
        gts[f"{i:06d}"] = semantics.semantic_labels(
            loader.load_labels(root / "labels" / f"{i:06d}.label")).astype(np.int8)
        with torch.no_grad():
            p = model.predict([torch.from_numpy(pts).float().to(dev)])[0].cpu().numpy()
        preds[f"{i:06d}"] = p.astype(np.uint8)
    np.savez_compressed(args.out, meta=json.dumps(meta),
                        **{f"pred_{k}": v for k, v in preds.items()},
                        **{f"gt_{k}": v for k, v in gts.items()})
    ok = sum(int(((preds[k] == gts[k]) & (gts[k] >= 0)).sum()) for k in preds)
    tot = sum(int((gts[k] >= 0).sum()) for k in preds)
    print(f"  wrote {args.out}: {len(preds)} frames, point accuracy on this slice {ok / tot:.4%}")


def diff(args):
    passes = []
    for f in args.files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        frames = sorted(k[5:] for k in z.files if k.startswith("pred_"))
        passes.append((Path(f).stem, meta, {k: z[f"pred_{k}"] for k in frames},
                       {k: z[f"gt_{k}"] for k in frames}))
    frames = sorted(passes[0][2])
    for name, meta, p, g in passes:
        ok = sum(int(((p[k] == g[k]) & (g[k] >= 0)).sum()) for k in frames)
        tot = sum(int((g[k] >= 0).sum()) for k in frames)
        print(f"  {name:<14} device={meta.get('device', 'cpu'):<6} "
              f"fast={meta['fast_scatter']!s:<5} threads={meta['num_threads']:<2} "
              f"interop={meta['num_interop_threads']:<2} acc {ok / tot:.4%}")
    n_points = sum(len(passes[0][2][k]) for k in frames)
    print(f"\n  pairwise differing points over {len(frames)} frames ({n_points:,} points):")
    out = {}
    for a in range(len(passes)):
        for b in range(a + 1, len(passes)):
            na, _, pa, _ = passes[a]
            nb, _, pb, _ = passes[b]
            per = [int((pa[k] != pb[k]).sum()) for k in frames]
            nf = sum(1 for x in per if x)
            print(f"    {na:<14} vs {nb:<14} {sum(per):>7,} points differ, in {nf:>2}/{len(frames)} frames"
                  + (f"   (max {max(per):,} in one frame)" if nf else ""))
            out[f"{na}|{nb}"] = {"points": sum(per), "frames": nf, "per_frame": per}
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"passes": [{"name": n, **m} for n, m, _, _ in passes], "pairs": out}, indent=2),
            encoding="utf-8")
        print(f"\n  wrote {args.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seq", default="08")
    r.add_argument("--frames", type=int, default=20)
    r.add_argument("--checkpoint", default="checkpoints/frnet-semantickitti_seg.pth")
    r.add_argument("--fast-scatter", action="store_true")
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--device", default="cpu",
                   help="cpu (default, as R-j was measured) or cuda -- the GPU reproducibility test")
    r.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("files", nargs="+")
    d.add_argument("--json", default=None)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else diff(args)


if __name__ == "__main__":
    main()
