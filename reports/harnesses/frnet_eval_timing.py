# PROVENANCE -- written 2026-09-14 for D11 (reopened narrowly), Step 5.
#
# Produced: the end-to-end CPU timing of scripts/frnet_eval.py with and without
#           --fast-scatter, against checkpoints/frnet-semantickitti_seg.pth
#           (SHA-256 09adea90...285e, the authors' public release -- see
#           checkpoints/frnet-semantickitti_seg.pth.PROVENANCE.md).
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/frnet_eval_timing.py --reps 2 --frames 200
#
# NOTE: Same discipline as whole_frame_bench.py:
#   - every run is a FRESH PROCESS running the shipping script unchanged, so the
#     figure is the one a user of frnet_eval.py actually waits for (model load,
#     data read, inference, scoring -- the whole thing);
#   - machine state is recorded before AND after every run, and a run whose state
#     is out of bounds is kept in the JSON but excluded from the summary;
#   - arms ALTERNATE (fast, loop, fast, loop, ...) so slow drift in machine state
#     lands on both arms rather than on one;
#   - the checkpoint's SHA-256 is re-checked before the first run, so the timing
#     cannot silently be of a different file.
#
#   It ALSO checks correctness, because a speedup of a different answer is not a
#   speedup: on CPU the shim is bit-identical for both reductions (Step 2), so the
#   two arms must print IDENTICAL point accuracy, mIoU and per-class IoUs. Any
#   difference is reported as a failure, not averaged over.
#
# Measurement only. Changes nothing in src/ or scripts/.
"""End-to-end frnet_eval.py wall time, --fast-scatter vs the port's loops, CPU."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import machine_state  # noqa: E402

EXPECTED_SHA256 = "09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e"
ACC = re.compile(r"point accuracy\s+([\d.]+)%")
MIOU = re.compile(r"mIoU over\s+(\d+) present classes\s+([\d.]+)%")
CLS = re.compile(r"^\s{2}([a-z-]+)\s+([\d.]+)%\s+([\d,]+)\s*$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def one_run(arm, seq, frames, checkpoint):
    cmd = [sys.executable, "scripts/frnet_eval.py", "--seq", seq, "--frames", str(frames),
           "--checkpoint", checkpoint]
    if arm == "fast":
        cmd.append("--fast-scatter")
    env = dict(os.environ, PYTHONUTF8="1")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=6 * 3600)
    wall = time.perf_counter() - t0
    out = p.stdout
    acc = ACC.search(out)
    miou = MIOU.search(out)
    classes = {m.group(1): float(m.group(2)) for m in map(CLS.match, out.splitlines()) if m}
    if p.returncode != 0 or not acc or not miou:
        raise SystemExit(f"[{arm}] frnet_eval.py failed (exit {p.returncode}):\n"
                         + out[-2000:] + p.stderr[-2000:])
    return {"wall_s": round(wall, 2), "point_accuracy": float(acc.group(1)),
            "miou": float(miou.group(2)), "present_classes": int(miou.group(1)),
            "per_class_iou": classes, "stdout_tail": out[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2, help="runs PER ARM")
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--checkpoint", default="checkpoints/frnet-semantickitti_seg.pth")
    ap.add_argument("--arms", default="fast,loop",
                    help="order of the first pair; subsequent pairs alternate")
    ap.add_argument("--out", default="reports/bench/frnet_eval_timing.json")
    args = ap.parse_args()

    digest = sha256(args.checkpoint)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"checkpoint SHA-256 is {digest}, expected {EXPECTED_SHA256} -- "
                         "refusing to time a different file")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    first = args.arms.split(",")
    order = []
    for r in range(args.reps):
        order += first if r % 2 == 0 else first[::-1]
    print(f"frnet_eval timing @ {git}, seq {args.seq}, {args.frames} frames, "
          f"checkpoint sha256 {digest[:12]}..., order {order}")

    runs = []
    for i, arm in enumerate(order, 1):
        before = machine_state.snapshot()
        res = one_run(arm, args.seq, args.frames, args.checkpoint)
        after = machine_state.snapshot()
        trusted = bool(before.get("trusted")) and bool(after.get("trusted"))
        res.update({"arm": arm, "state_before": before, "state_after": after,
                    "trusted": trusted})
        runs.append(res)
        print(f"  run {i} [{arm:<4}] wall {res['wall_s']:>9.1f} s  acc {res['point_accuracy']}%  "
              f"mIoU {res['miou']}%  {'trusted' if trusted else 'UNTRUSTED'}")
        print(f"         before: {machine_state.line(before)}")
        print(f"         after:  {machine_state.line(after)}")
        pathlib.Path(args.out).write_text(json.dumps(
            {"git": git, "checkpoint_sha256": digest, "seq": args.seq,
             "frames": args.frames, "runs": runs}, indent=2), encoding="utf-8")

    good = [r for r in runs if r["trusted"]]
    summary = {}
    for arm in ("fast", "loop"):
        w = [r["wall_s"] for r in good if r["arm"] == arm]
        if w:
            summary[arm] = {"n": len(w), "median_s": round(statistics.median(w), 1),
                            "min_s": min(w), "max_s": max(w)}
    metrics = {(r["point_accuracy"], r["miou"], json.dumps(r["per_class_iou"], sort_keys=True))
               for r in runs}
    identical = len(metrics) == 1
    print(f"\n  trusted runs: {len(good)} of {len(runs)}")
    for arm, s in summary.items():
        print(f"  {arm:<4} median {s['median_s']} s over {s['n']} (range {s['min_s']}-{s['max_s']})")
    if "fast" in summary and "loop" in summary:
        summary["speedup_median"] = round(summary["loop"]["median_s"] / summary["fast"]["median_s"], 2)
        print(f"  END-TO-END SPEEDUP (median loop / median fast): {summary['speedup_median']}x")
    print(f"  accuracy/mIoU/per-class IoU identical across every run and both arms: {identical}")
    if not identical:
        print("  [!] THE ARMS DISAGREE. On CPU the shim is bit-identical, so this is a real "
              "problem -- do not quote the speedup.")
    pathlib.Path(args.out).write_text(json.dumps(
        {"git": git, "checkpoint_sha256": digest, "seq": args.seq, "frames": args.frames,
         "order": order, "runs": runs, "summary": summary,
         "metrics_identical_across_arms": identical}, indent=2), encoding="utf-8")
    print(f"  wrote {args.out}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
