# PROVENANCE -- written 2026-09-14 for the p99 fix programme (D8/D9/R-h).
#
# Produced: the before/after whole-frame latency figures for the transform and
#           cleanup allocation fixes.
# Run:      VRGRID_DATA_ROOT=C:/KITTI/dataset \
#             python reports/harnesses/whole_frame_bench.py --label baseline --reps 3
#
# NOTE: Each rep is a FRESH PROCESS running the shipping tool,
#      `scripts/timing_table.py --seq`, so the Patchwork++ singleton (D1) cannot
#      carry state from one rep into the next, and the number is the same one the
#      tool itself would print. Machine state is recorded before and after every
#      rep; a rep whose state is out of bounds is kept in the JSON but EXCLUDED
#      from the summary and named, rather than averaged in.
#
# Measurement only. Changes nothing in src/.
"""Whole-frame p50/p99 over several fresh-process reps, with machine state."""
import argparse
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
import numpy as np  # noqa: E402

ROW = re.compile(r"^(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s")


def one_rep(seq, frames, times_path):
    env = dict(os.environ, PYTHONUTF8="1")
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "scripts/timing_table.py", "--seq", seq, "--frames", str(frames),
         "--frame-times", str(times_path)],
        capture_output=True, text=True, env=env, timeout=1800)
    rows = {}
    for ln in out.stdout.splitlines():
        m = ROW.match(ln.strip())
        if m:
            rows[m.group(1)] = {"p50": float(m.group(2)), "p99": float(m.group(3)),
                                "max": float(m.group(4))}
    if "FRAME" not in rows:
        raise SystemExit("timing_table printed no FRAME row:\n" + out.stdout[-2000:]
                         + out.stderr[-2000:])
    frame_ms = json.loads(pathlib.Path(times_path).read_text(encoding="utf-8"))
    return rows, round(time.time() - t0, 1), frame_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="bench_results")
    args = ap.parse_args()

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                                capture_output=True, text=True).stdout.strip())
    print(f"[{args.label}] {branch}@{git}{' (DIRTY src/scripts)' if dirty else ''}, "
          f"seq {args.seq}, {args.frames} frames x {args.reps} fresh-process reps")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)
    reps = []
    for r in range(args.reps):
        before = machine_state.snapshot()
        rows, secs, frame_ms = one_rep(args.seq, args.frames,
                                       outdir / f".{args.label}.rep{r+1}.frames.json")
        after = machine_state.snapshot()
        trusted = bool(before.get("trusted")) and bool(after.get("trusted"))
        f = rows["FRAME"]
        print(f"  rep {r+1}: FRAME p50 {f['p50']:7.2f}  p99 {f['p99']:7.2f}  "
              f"max {f['max']:7.2f}  ({secs}s)  {'trusted' if trusted else 'UNTRUSTED'}")
        print(f"         before: {machine_state.line(before)}")
        print(f"         after:  {machine_state.line(after)}")
        reps.append({"rows": rows, "seconds": secs, "state_before": before,
                     "state_after": after, "trusted": trusted, "frame_ms": frame_ms})

    good = [x for x in reps if x["trusted"]]
    print(f"\n  trusted reps: {len(good)} of {len(reps)}")
    summary = {}
    if good:
        stages = sorted(set().union(*[x["rows"].keys() for x in good]))
        for st in stages:
            vals = [x["rows"][st] for x in good if st in x["rows"]]
            summary[st] = {k: round(statistics.median(v[k] for v in vals), 2)
                           for k in ("p50", "p99", "max")}
            summary[st]["p99_range"] = [min(v["p99"] for v in vals),
                                        max(v["p99"] for v in vals)]
        f = summary["FRAME"]
        print(f"  FRAME median of reps: p50 {f['p50']}  p99 {f['p99']}  "
              f"(p99 across reps {f['p99_range'][0]}-{f['p99_range'][1]})  -- per-run, NOT the gate")
        # THE GATE (JP, 2026-09-14): p99 over every frame of every trusted run,
        # ranked together. numpy's default linear percentile, as timing.py uses.
        pooled = np.asarray([x for rep in good for x in rep["frame_ms"]], dtype=np.float64)
        over = int((pooled > 100.0).sum())
        summary["FRAME_pooled"] = {
            "frames": int(pooled.size), "p50": round(float(np.median(pooled)), 2),
            "p99": round(float(np.percentile(pooled, 99)), 2),
            "max": round(float(pooled.max()), 2), "frames_over_100ms": over}
        g = summary["FRAME_pooled"]
        print(f"  FRAME POOLED over {g['frames']} frames: p50 {g['p50']}  p99 {g['p99']}  "
              f"max {g['max']}  ({over} frames > 100 ms)  "
              f"budget 100 ms -> p50 {'PASS' if g['p50'] < 100 else 'MISS'}, "
              f"p99 {'PASS' if g['p99'] < 100 else 'MISS'}")

    path = outdir / f"{args.label}.json"
    path.write_text(json.dumps({"label": args.label, "git": git, "branch": branch,
                                "dirty_src": dirty, "seq": args.seq,
                                "frames": args.frames, "reps": reps,
                                "summary_trusted_median": summary}, indent=2),
                    encoding="utf-8")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
