#!/usr/bin/env python3
"""VRAM attribution and GPU contention: the map pipeline and FRNet on one card.
[Shrestha]

    python scripts/vram_contention.py [--seq 08] [--frames 200] [--pair-seconds 40]
                                      [--out docs/gpu-lane/09-vram-contention.json]

Closes `docs/gpu-lane/03-CUDA-PORT-PLAN.md` §5 (VRAM attribution, R9b) and §6
(the contention question). Every configuration runs in its OWN process, so
each starts from a fresh CUDA context and an empty allocator -- a number taken
after another configuration has warmed the caches is not that configuration's
number.

Configurations, all on the same frames of the same sequence:

    context   CUDA context only (cupy, then torch): what any process pays
    grid      the map pipeline, `--device cuda`, alone
    frnet     FRNet batch 1 (fast scatter), alone
    both      both in ONE loop: each frame is perceived and mapped, and FRNet
              runs on the same scan between the two
    pair      grid and frnet as TWO concurrent processes, released together
              and stopped together, so every recorded frame was contended

Timing discipline (§7): the device is synchronised before every clock stops,
the first 10 frames of every process are discarded, p50 and p99 only, and
the GPU's temperature, power and throttle reasons are sampled once a second
throughout.

⚑ FRNet's labels are NOT fed to the map in `both`. The map takes semantics from
  the `.label` files on purpose (`frnet_eval.py`'s header), so that no mapping
  number depends on segmentation quality. `both` measures what running the
  network in the loop COSTS -- compute and memory on a shared card -- which is
  the question §6 asks, and nothing about what its labels would do.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

WARMUP = 10
SCHEDULE = "5/10/20/40"


# --- helpers shared by the workers ------------------------------------------------


def _process_vram_mib(pid=None):
    """This process's memory on the card, from the driver -- context, cupy's
    pool and torch's cache all included, which no allocator can see."""
    pid = os.getpid() if pid is None else pid
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        p, mem = (s.strip() for s in line.split(","))
        if int(p) == pid:
            return int(mem)
    return None


def _stats(ms):
    a = np.asarray(ms, np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()),
            "mean": float(a.mean())}


def _nbytes(arrays):
    return int(sum(getattr(a, "nbytes", 0) for a in arrays))


def _cupy_arrays(obj):
    import cupy
    found = []
    for v in vars(obj).values():
        if isinstance(v, cupy.ndarray):
            found.append(v)
        elif isinstance(v, dict):
            found += [x for x in v.values() if isinstance(x, cupy.ndarray)]
    return found


class _Grid:
    """The map pipeline on the card, pulled one frame at a time and cycling
    back to the start of the frame range when it runs out."""

    def __init__(self, seq, frames):
        import cupy
        from vrgrid.grid.schedule import load
        from vrgrid.run.__main__ import iter_pipeline
        from vrgrid.run.engine import MapEngine

        self.cp, self.seq, self.frames = cupy, seq, frames
        self._iter_pipeline = iter_pipeline
        self.engine = MapEngine(load(SCHEDULE), device="cuda")
        self._it = None

    def pull(self):
        """Perception for the next scan: load, device perception launch and
        Patchwork++ on the host. Returns the frame."""
        if self._it is None:
            self._it = self._iter_pipeline(self.seq, self.frames, device="cuda")
        frame = next(self._it, None)
        if frame is None:
            self._it = self._iter_pipeline(self.seq, self.frames, device="cuda")
            frame = next(self._it)
        return frame

    def step(self, frame):
        self.engine.step(frame)
        self.cp.cuda.Stream.null.synchronize()

    def memory(self):
        gpu = self.engine.gpu
        grid = _nbytes(gpu.grid.values())
        perception = self.engine_perception_bytes()
        db = gpu.device_bytes()
        cells = int(self.engine.sched.total_cells) if hasattr(self.engine.sched, "total_cells") else None
        return {"grid_bytes": grid, "grid_slots": int(gpu.n_slots),
                "schedule_cells": cells,
                "map_static_bytes": int(db["static"]),
                "perception_bytes": perception,
                "cupy_pool_used": int(db["pool_used"]),
                "cupy_pool_reserved": int(db["pool_reserved"])}

    def engine_perception_bytes(self):
        # The DevicePerception lives in the generator's frame; every frame
        # carries a reference to it.
        frame = getattr(self, "_last_frame", None)
        p = getattr(frame, "_p", None)
        return _nbytes(_cupy_arrays(p)) if p is not None else None


class _FRNet:
    def __init__(self, seq, frames):
        import torch

        sys.path.insert(0, str(Path(__file__).parent))
        from frnet_fast_scatter import enable
        from vrgrid.perception import loader, semantics
        from vrgrid.perception.frnet import FRNet

        enable(verify=True)
        self.torch, self.loader = torch, loader
        ckpt = Path(os.environ.get("VRGRID_FRNET_CHECKPOINT",
                                   "checkpoints/frnet-semantickitti_seg.pth"))
        model = FRNet(num_classes=20, ignore_index=19, output_shape=(64, 512),
                      fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG,
                      fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob.get("state_dict", blob), strict=False)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        self.model = model.to("cuda").eval()
        torch.cuda.synchronize()
        self.weights_bytes = int(torch.cuda.memory_allocated() - before)
        self.params = int(sum(p.numel() for p in self.model.parameters()))
        self.root = Path(loader.DATA_ROOT) / "sequences" / seq
        self.frames, self.i = frames, 0

    def load(self):
        scan = self.root / "velodyne" / f"{self.i % self.frames:06d}.bin"
        self.i += 1
        return self.loader.load_velodyne_scan(scan)

    def predict(self, pts):
        t = self.torch
        with t.no_grad():
            pred = self.model.predict([t.from_numpy(np.ascontiguousarray(pts)).float().to("cuda")])[0]
            out = pred.cpu().numpy()          # the map would need host labels
        t.cuda.synchronize()
        return out

    def memory(self):
        t = self.torch
        return {"frnet_params": self.params, "frnet_weights_bytes": self.weights_bytes,
                "torch_max_allocated": int(t.cuda.max_memory_allocated()),
                "torch_max_reserved": int(t.cuda.max_memory_reserved())}


# --- workers: each runs in its own process ----------------------------------------


def _wait_for(path, poll=0.02):
    while not Path(path).exists():
        time.sleep(poll)


def worker(args):
    res = {"config": args.worker, "pid": os.getpid()}
    ms, ms_a, ms_b = [], [], []

    if args.worker == "context":
        import cupy
        cupy.zeros(1)
        cupy.cuda.Stream.null.synchronize()
        res["vram_mib_cupy_context"] = _process_vram_mib()
        import torch
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        res["vram_mib_cupy_plus_torch_context"] = _process_vram_mib()

    elif args.worker in ("grid", "frnet", "both"):
        grid = _Grid(args.seq, args.frames) if args.worker in ("grid", "both") else None
        net = _FRNet(args.seq, args.frames) if args.worker in ("frnet", "both") else None
        if grid is not None:
            res["vram_mib_after_setup"] = _process_vram_mib()
        n_total = WARMUP + args.frames
        timed = args.release is None
        stop_at = None
        i = 0
        while True:
            if not timed and i == WARMUP:
                Path(args.ready).touch()
                _wait_for(args.release)
                stop_at = time.perf_counter() + args.pair_seconds
                timed = True
            if args.release is None and i >= n_total:
                break
            if stop_at is not None and time.perf_counter() >= stop_at:
                break
            t0 = time.perf_counter()
            if args.worker == "grid":
                frame = grid.pull()
                grid._last_frame = frame
                grid.step(frame)
            elif args.worker == "frnet":
                net.predict(net.load())
            else:
                frame = grid.pull()
                grid._last_frame = frame
                t1 = time.perf_counter()
                net.predict(np.asarray(frame.points_sensor))
                t2 = time.perf_counter()
                grid.step(frame)
                t3 = time.perf_counter()
                if i >= WARMUP:
                    ms_a.append((t1 - t0 + t3 - t2) * 1e3)   # the map pipeline's share
                    ms_b.append((t2 - t1) * 1e3)             # FRNet's share
            dt = (time.perf_counter() - t0) * 1e3
            if i >= WARMUP and (args.release is None or stop_at is not None):
                ms.append(dt)
            i += 1
        res["frame_ms"] = _stats(ms)
        if ms_a:
            res["grid_share_ms"] = _stats(ms_a)
            res["frnet_share_ms"] = _stats(ms_b)
        if grid is not None:
            res.update(grid.memory())
        if net is not None:
            res.update(net.memory())
        res["vram_mib_process"] = _process_vram_mib()
    Path(args.result).write_text(json.dumps(res, indent=1))


# --- the parent: runs the workers and prints the tables ---------------------------


class _Telemetry:
    FIELDS = "memory.used,utilization.gpu,temperature.gpu,power.draw,clocks_event_reasons.active"

    def __init__(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits",
             "-lms", "1000"], stdout=subprocess.PIPE, text=True)

    def stop(self):
        self.proc.terminate()
        rows = []
        for line in self.proc.communicate()[0].strip().splitlines():
            try:
                mem, util, temp, power, reasons = (s.strip() for s in line.split(","))
                rows.append((int(mem), int(util), int(temp), float(power), int(reasons, 16)))
            except ValueError:
                continue
        if not rows:
            return {}
        a = np.array([r[:4] for r in rows], np.float64)
        reasons = 0
        for r in rows:
            reasons |= r[4]
        # 0x4 is the SOFTWARE POWER CAP. It is reported, not counted as a
        # slowdown below, but it is not harmless: on the RTX 5050 laptop part
        # every FRNet configuration held it at ~96 W, so those timings include
        # power limiting, and a T4 (70 W cap) would limit differently. The
        # thermal and hardware slowdown bits are the ones flagged here.
        return {"card_mem_mib_max": int(a[:, 0].max()), "util_pct_mean": float(a[:, 1].mean()),
                "temp_c_max": int(a[:, 2].max()), "power_w_max": float(a[:, 3].max()),
                "throttle_reasons_or": hex(reasons),
                "thermal_or_hw_slowdown": bool(reasons & (0x20 | 0x40 | 0x8 | 0x80))}


def _run(cfg, args, tmp, extra=(), tag=""):
    result = Path(tmp) / f"{cfg}{tag}.json"
    cmd = [sys.executable, __file__, "--worker", cfg, "--seq", args.seq,
           "--frames", str(args.frames), "--result", str(result), *extra]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True), result


def _finish(proc, result, cfg):
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise SystemExit(f"{cfg} worker failed:\n{err[-3000:]}")
    return json.loads(Path(result).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--pair-seconds", type=float, default=40.0)
    ap.add_argument("--out", default=None, help="write every measurement as JSON")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--result", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ready", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--release", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         check=True).stdout.strip()
    import cupy
    import torch
    env = {"gpu": gpu, "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
           "cupy": cupy.__version__, "torch": torch.__version__, "seq": args.seq,
           "frames": args.frames, "warmup": WARMUP, "pair_seconds": args.pair_seconds}
    print(f"{gpu} | CUDA runtime {env['cuda_runtime']} | cupy {cupy.__version__} | torch {torch.__version__}")
    del cupy, torch

    out = {"env": env}
    with tempfile.TemporaryDirectory() as tmp:
        for cfg in ("context", "grid", "frnet", "both"):
            tel = _Telemetry()
            p, r = _run(cfg, args, tmp)
            out[cfg] = _finish(p, r, cfg)
            out[cfg]["telemetry"] = tel.stop()
            print(f"  {cfg:<8} done")

        ready = [Path(tmp) / "ready.grid", Path(tmp) / "ready.frnet"]
        release = Path(tmp) / "release"
        tel = _Telemetry()
        procs = []
        for cfg, rd in (("grid", ready[0]), ("frnet", ready[1])):
            p, r = _run(cfg, args, tmp, ("--ready", str(rd), "--release", str(release),
                                         "--pair-seconds", str(args.pair_seconds)), tag=".pair")
            procs.append((cfg, p, r))
        # Both warm up alone, then are released together: every recorded frame
        # of either process ran while the other was running.
        while not all(x.exists() for x in ready):
            if any(p.poll() is not None for _, p, _ in procs):
                break
            time.sleep(0.05)
        release.touch()
        pair = {}
        for cfg, p, r in procs:
            pair[cfg] = _finish(p, r, f"pair-{cfg}")
        out["pair"] = pair
        out["pair"]["telemetry"] = tel.stop()
        print("  pair     done")

    report(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return 0


def report(o):
    MB = 1e6
    g, f, b, c, pr = o["grid"], o["frnet"], o["both"], o["context"], o["pair"]

    print("\n=== VRAM attribution (03-CUDA-PORT-PLAN.md §5) ===")
    rows = [
        ("Map grid arrays", "8.94 MB (745,000 cells x 12 B)", f"{g['grid_bytes'] / MB:.2f} MB over {g['grid_slots']:,} slots", "sum of device grid nbytes"),
        ("Map frame-path buffers", "-", f"{(g['map_static_bytes'] - g['grid_bytes']) / MB:.2f} MB", "DeviceMap pool delta - grid"),
        ("Perception buffers", "-", f"{g['perception_bytes'] / MB:.2f} MB" if g.get("perception_bytes") else "n/a", "DevicePerception nbytes"),
        ("Our declared total", "-", f"{(g['map_static_bytes'] + (g.get('perception_bytes') or 0)) / MB:.2f} MB", "sum of the three"),
        ("cupy pool, used / reserved", "-", f"{g['cupy_pool_used'] / MB:.1f} / {g['cupy_pool_reserved'] / MB:.1f} MB", f"after {g['frame_ms']['n'] + WARMUP} frames"),
        ("CUDA context (cupy)", "-", f"{c['vram_mib_cupy_context']} MiB", "nvidia-smi, context-only process"),
        ("Grid process on card", "-", f"{g['vram_mib_process']} MiB", "nvidia-smi per process"),
        ("FRNet weights", "-", f"{f['frnet_weights_bytes'] / MB:.1f} MB ({f['frnet_params'] / 1e6:.2f} M params)", "torch allocated delta at .to(cuda)"),
        ("FRNet peak, allocated / reserved", "-", f"{f['torch_max_allocated'] / MB:.0f} / {f['torch_max_reserved'] / MB:.0f} MB", "torch.cuda.max_memory_*"),
        ("FRNet process on card", "-", f"{f['vram_mib_process']} MiB", "nvidia-smi per process"),
        ("Both, one process", "-", f"{b['vram_mib_process']} MiB", "nvidia-smi per process"),
        ("Whole card, pair run peak", "-", f"{pr['telemetry'].get('card_mem_mib_max')} MiB of {o['env']['gpu'].split(',')[2].strip()}", "nvidia-smi memory.used, 1 s"),
    ]
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    for r in rows:
        print("  " + "  ".join(r[i].ljust(w[i]) for i in range(4)))

    print("\n=== Contention (03-CUDA-PORT-PLAN.md §6), seq {seq}, ms per frame ===".format(**o["env"]))
    def line(name, s, extra=""):
        print(f"  {name:<34} p50 {s['p50']:7.2f}  p99 {s['p99']:7.2f}  max {s['max']:7.2f}  n {s['n']:>5}{extra}")
    line("grid alone", g["frame_ms"])
    line("FRNet alone", f["frame_ms"])
    line("both, one loop: whole frame", b["frame_ms"])
    line("both, one loop: grid share", b["grid_share_ms"])
    line("both, one loop: FRNet share", b["frnet_share_ms"])
    line("pair, two processes: grid", pr["grid"]["frame_ms"])
    line("pair, two processes: FRNet", pr["frnet"]["frame_ms"])
    s = g["frame_ms"]["p50"] + f["frame_ms"]["p50"]
    print(f"\n  sum of the parts, p50            {s:7.2f}   one loop measures {b['frame_ms']['p50']:.2f} "
          f"({(b['frame_ms']['p50'] / s - 1) * 100:+.1f}%)")
    for name, cfg in (("grid", g), ("frnet", f)):
        a, pp = cfg["frame_ms"], pr[name]["frame_ms"]
        print(f"  {name:<5} under contention (pair): p50 {(pp['p50'] / a['p50'] - 1) * 100:+.1f}%, "
              f"p99 {(pp['p99'] / a['p99'] - 1) * 100:+.1f}%")
    print(f"  10 Hz at p99: grid alone {'meets' if g['frame_ms']['p99'] < 100 else 'MISSES'}, "
          f"one loop {'meets' if b['frame_ms']['p99'] < 100 else 'MISSES'}, "
          f"pair grid {'meets' if pr['grid']['frame_ms']['p99'] < 100 else 'MISSES'}")
    print("\n  telemetry: " + "; ".join(
        f"{k} {o[k]['telemetry'].get('temp_c_max')} C max, {o[k]['telemetry'].get('power_w_max')} W max, "
        f"util {o[k]['telemetry'].get('util_pct_mean', 0):.0f}%, throttle {o[k]['telemetry'].get('throttle_reasons_or')}, "
        f"thermal/hw slowdown {o[k]['telemetry'].get('thermal_or_hw_slowdown')}"
        for k in ("grid", "frnet", "both", "pair")))


if __name__ == "__main__":
    sys.exit(main())
