"""Pipeline entry point -- `python -m vrgrid.run --seq 00 --frames 50`.

Thin wiring only: parse arguments, pull frames from the loader, run each
perception stage in order, hand the result to the dashboard. No algorithm
lives here -- it belongs to nobody and everybody. Keep it that way.

Stages (all JP's, `src/perception/`):

    loader.scans()          raw points + raw .label + GT pose, per frame
    transforms              sensor -> vehicle -> world  (docs/frames.md)
    range_image.project()   64x512 spherical image + inverse index (sensor frame)
    semantics               semantic_labels() 19-class + is_moving()  (GT .label)
    ground.segment_ground_or_fallback()  Patchwork++ mask, or the semantic-class
                            fallback (loudly) when pypatchworkpp is absent
    reflectivity.normalise() rho_hat -> one byte  (KITTI: rho_hat = I; the
                             eq-31 r^2/cos terms are firmware-redundant here)

then the map back end, in `engine.MapEngine` (see that file for the order):

    bin -> scatter -> fuse -> visibility cleanup -> shift

The dashboard (`--viz` / `--save`) renders the real per-frame output, replacing
the Day-0 synthetic plane/boxes/slope one layer at a time via `--color-by`.

âš‘ `--show-ghosts` is the Gate 3 toggle and it now drives BOTH halves: the
  viewer keeps the moving returns in the main cloud, AND the map stops running
  Â§10.4, so the ghost trails stay in the cells. Until the engine existed it
  drove only the first, which filters the input cloud on the ground-truth
  `moving-*` label and demonstrates nothing about the mapping engine.
"""

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from vrgrid.grid import schedule as schedule_mod
from vrgrid.run.engine import MapEngine


@dataclass
class PerceptionFrame:
    """One frame after every perception stage. Arrays are point-aligned to
    `points_sensor` unless noted."""

    index: int
    points_sensor: np.ndarray      # (N, 4) raw x,y,z,intensity
    points_world: np.ndarray       # (N, 3) after sensor->world
    pose: np.ndarray               # (3, 4) GT
    vehicle_xyz_world: np.ndarray  # (3,) vehicle origin in world
    semantic: np.ndarray           # (N,) 19-class, -1 ignore
    moving: np.ndarray             # (N,) bool
    ground: np.ndarray             # (N,) bool
    reflectivity8: np.ndarray      # (N,) uint8, 0 where not projected this frame
    range_image: np.ndarray        # (H, W, 5)
    inverse_index: np.ndarray      # (H, W) int32
    ground_method: str             # "patchworkpp" | "semantic_fallback" (ground.py)
    semantic_source: str = "gt"    # "gt" (.label files, default) | "frnet" (opt-in DL mode)


def iter_pipeline(seq: str, max_frames: int | None, use_patchworkpp: bool = True,
                  timer=None, start_frame: int = 0, device: str = "cpu",
                  reuse_buffers: bool = False, semantics_source: str = "gt", frnet=None):
    """Yield a PerceptionFrame per scan of `seq`.

    `start_frame` skips ahead before the first yield (default 0, so existing
    callers are unchanged); `max_frames` then counts from there, and
    `PerceptionFrame.index` carries the real sequence frame number so the
    dashboard timeline lines up with the sequence.

    `timer` is an optional `gpu.timing.Timer`. Passing one names each stage
    with the spelling in `timing.STAGES`, which is what lets
    `scripts/timing_table.py --seq` print a whole-frame latency table instead
    of the back end alone. The stage names were fixed in `timing.py` on Day 0
    precisely so the front end and the map would not invent two spellings of
    "range image"; this is the other half of that.

    `device="cuda"` runs the range image, reflectivity and label stages on the
    card (`gpu.device.DevicePerception`) and yields `DeviceFrame`s whose
    outputs stay there for a `MapEngine(device="cuda")` to consume without a
    copy. Patchwork++ still runs on the host, while the card works. The
    outputs are bit-identical to the CPU stages -- `scripts/gpu_parity.py`.

    `loader.scans` is a generator, so the `load` stage times the pull of one
    scan off it rather than the whole sequence -- which is the per-frame cost
    the 10 Hz budget is about.
    """
    from vrgrid.perception import ground, loader

    scans = loader.scans(seq, max_frames=max_frames, start_frame=start_frame)
    # `semantics_source="frnet"` is the OPT-IN deep-learning mode: each frame's
    # 19-class labels come from `semantics.FRNetInference` (build it with
    # `open_frnet()`) instead of the `.label` files. Ground truth stays the
    # default, and motion stays `is_moving(raw_labels)` in BOTH modes -- FRNet
    # has no motion output, which anything reporting the mode must say.
    if semantics_source not in ("gt", "frnet"):
        raise ValueError(f"semantics_source must be 'gt' or 'frnet', not {semantics_source!r}")
    if semantics_source == "frnet":
        if frnet is None:
            raise ValueError("semantics_source='frnet' needs frnet=open_frnet() "
                             "(a semantics.FRNetInference)")
        if device != "cpu":
            # The device path yields a DeviceFrame and never runs the host
            # semantics stage, so the model's labels would be silently ignored.
            raise ValueError("semantics_source='frnet' is host-only; it cannot be "
                             f"combined with device={device!r}")

    perception = None
    if device == "cuda":
        from vrgrid.gpu.device import DevicePerception, resolve_device
        resolve_device(device)
        perception = DevicePerception()
    # A fresh Patchwork++ estimator per run: it adapts from past scans, so a
    # shared one made a second run in the same process map differently (see
    # `ground.reset_estimator`). Runs here, at the first frame's pull.
    ground.reset_estimator()

    # `reuse_buffers` (default OFF): write `points_world` into one scratch reused
    # every frame instead of allocating ~10.9 MB per frame in `transform_points`.
    # [!] Each yielded frame's `points_world` is then OVERWRITTEN by the next
    # frame. Only a caller that finishes with a frame before pulling the next may
    # turn it on -- `main()` without a dashboard and `timing_table.py --seq` do.
    # Anything that keeps frames, `list(iter_pipeline(...))` included, must not;
    # the tests do exactly that, which is why the default is off.
    tscratch = None
    if reuse_buffers:
        from vrgrid.perception import transforms

        cap = int(schedule_mod.load_thresholds()["scatter"].get("max_points_per_frame",
                                                                150_000))
        tscratch = transforms.new_transform_scratch(cap)
    i = 0
    while True:
        # Timed by hand rather than with `stage("load")`, because the pull that
        # EXHAUSTS the generator must not be recorded: it is not a frame, and
        # counting it gave `load` one more sample than there were frames and
        # dragged its p99 down with a near-zero reading.
        t0 = time.perf_counter()
        item = next(scans, None)
        if item is None:
            break
        if timer is not None:
            timer.record("load", (time.perf_counter() - t0) * 1e3)
        points, raw_labels, pose = item
        yield perceive(points, raw_labels, pose, seq, start_frame + i,
                       use_patchworkpp=use_patchworkpp, timer=timer,
                       perception=perception, scratch=tscratch,
                       semantics_source=semantics_source, frnet=frnet)
        i += 1


def perceive(points, raw_labels, pose, seq: str, index: int, use_patchworkpp=True,
             timer=None, perception=None, ground_result=None, scratch=None,
             semantics_source="gt", frnet=None):
    """Every perception stage for one scan, on the host or on the card.

    `perception` is a `gpu.device.DevicePerception` for the device path, None
    for the CPU one. `ground_result` = `(mask, method)` skips ground
    segmentation and uses that result instead -- which is how the parity check
    feeds one Patchwork++ answer to both paths (the estimator is stateful, so
    running it twice would not be a controlled comparison).
    """
    from vrgrid.perception import ground, range_image, reflectivity, semantics, transforms

    if perception is not None:
        from vrgrid.gpu.device import synced_stage
        stage = synced_stage(timer)
    else:
        def stage(name):
            return timer.stage(name) if timer is not None else nullcontext()

    with stage("transform"):
        t_s_w = transforms.sensor_to_world(pose, sequence=seq)
        points_world = transforms.transform_points(points[:, :3], t_s_w, scratch=scratch)
        vehicle_xyz = transforms.vehicle_to_world(pose, sequence=seq)[:3, 3]

    if perception is not None:
        if semantics_source != "gt":
            raise ValueError("the device path yields a DeviceFrame and does not run the "
                             f"host semantics stage; semantics_source={semantics_source!r} "
                             "would be silently ignored")
        from vrgrid.gpu.device import DeviceFrame

        # Queued, not waited for: the card projects while the host segments.
        perception.launch(points, points_world, raw_labels,
                          stage=stage if timer is not None else None)
        with stage("ground"):
            if ground_result is None:
                # Patchwork++ needs no labels; the semantic fallback does, and
                # reading them back is its own cost only on that path.
                need_labels = not (use_patchworkpp and ground._HAVE_PATCHWORKPP)
                sem_host = (perception.sem[:len(points)].get() if need_labels
                            else None)
                gmask, ground_method = ground.segment_ground_or_fallback(
                    points, sem_host, use_patchworkpp=use_patchworkpp)
            else:
                gmask, ground_method = ground_result
        perception.finish()
        return DeviceFrame(perception, index=index, points_sensor=points,
                           points_world=points_world, pose=pose,
                           vehicle_xyz_world=vehicle_xyz, ground=gmask,
                           ground_method=ground_method)

    with stage("range_image"):
        ri, inv = range_image.project(points)
    with stage("semantics"):
        if semantics_source == "frnet":
            # The model's labels, on the raw sensor-frame scan -- the same input
            # scripts/frnet_eval.py scores. Same dtype and ignore convention (-1)
            # as semantic_labels(), so nothing downstream changes shape.
            semantic = frnet.infer_points(np.asarray(points, dtype=np.float32))
        else:
            semantic = semantics.semantic_labels(raw_labels)
    with stage("motion"):
        moving = semantics.is_moving(raw_labels)

    with stage("ground"):
        if ground_result is None:
            gmask, ground_method = ground.segment_ground_or_fallback(
                points, semantic, use_patchworkpp=use_patchworkpp)
        else:
            gmask, ground_method = ground_result

    with stage("reflectivity"):
        # with_incidence=False: cos_inc and flags are discarded just below and
        # PerceptionFrame stores neither, so skip computing them. rho8 is
        # unchanged by construction on the KITTI path (tests pin it).
        refl = reflectivity.normalise(ri, with_incidence=False)
        rho8, _ = reflectivity.scatter_to_points(refl, inv)
        if len(rho8) < len(points):  # pad points that never projected
            rho8 = np.concatenate([rho8, np.zeros(len(points) - len(rho8), np.uint8)])

    # The map back end (bin -> scatter -> fuse -> cleanup -> shift) runs in
    # `engine.MapEngine.step(frame)`, called by `main()` on each frame this
    # generator yields -- see the module docstring.

    return PerceptionFrame(
        index=index,
        points_sensor=points,
        points_world=points_world,
        pose=pose,
        vehicle_xyz_world=vehicle_xyz,
        semantic=semantic,
        moving=moving,
        ground=gmask,
        reflectivity8=rho8,
        range_image=ri,
        inverse_index=inv,
        ground_method=ground_method,
        semantic_source=semantics_source,
    )


def open_frnet(fast_scatter: bool = False, threads: int | None = None):
    """The opt-in DL mode's model: `semantics.FRNetInference`, set up reproducibly.

    `threads` calls `torch.set_num_threads` BEFORE the model is built. `threads=1` is
    what makes FRNet's labels reproducible run to run (OPEN-ITEMS R-j); the default
    thread count does not. `fast_scatter` applies `scripts/frnet_fast_scatter.py`
    after its own verify -- bit-identical to the port's loops at one thread, and the
    difference between seconds and a minute per frame on a CPU.

    The checkpoint is `configs/frnet.yaml`'s path (or $VRGRID_FRNET_CHECKPOINT),
    resolved from the working directory, so run from the repository root.
    """
    import torch

    if threads is not None:
        torch.set_num_threads(threads)
    if fast_scatter:
        import sys
        from pathlib import Path

        scripts = Path(__file__).resolve().parents[2] / "scripts"
        if not (scripts / "frnet_fast_scatter.py").exists():
            raise FileNotFoundError(f"--fast-scatter needs {scripts / 'frnet_fast_scatter.py'}")
        sys.path.insert(0, str(scripts))
        from frnet_fast_scatter import enable

        enable(verify=True)
    from vrgrid.perception.semantics import FRNetInference

    return FRNetInference()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vrgrid.run")
    p.add_argument("--seq", default="00", help="SemanticKITTI sequence")
    p.add_argument("--schedule", default="5/10/20/40", help="ring schedule name")
    p.add_argument("--thresholds", default="configs/thresholds.yaml")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--start-frame", type=int, default=0,
                   help="start from this frame index (default 0); --frames counts from here")
    p.add_argument("--viz", action="store_true", help="open the Rerun dashboard")
    p.add_argument("--save", default=None, help="write a Rerun .rrd recording here")
    p.add_argument(
        "--color-by",
        default="class",
        choices=["intensity", "class", "motion", "ground", "reflectivity"],
        help="how the dashboard colours the point cloud",
    )
    p.add_argument("--show-ghosts", action="store_true",
                   help="Gate 3 toggle OFF: keep moving points in the main cloud "
                        "and stop running the map's visibility cleanup, so ghost "
                        "trails stay in the cells (default: both on)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="cuda: perception (except Patchwork++) and the whole map run on "
                        "the card, grid in device memory. Bit-identical to cpu "
                        "(scripts/gpu_parity.py)")
    p.add_argument("--no-map", action="store_true",
                   help="perception only; skip the map back end entirely")
    p.add_argument("--clip-class-ids", action="store_true",
                   help="clip semantic ids to 15 so fusion's 4-bit candidate "
                        "accepts them (math Â§10.2). Corrupts the class layer; "
                        "the real fix is the 5/3 split, a room decision")
    p.add_argument("--palette", default="semantickitti", choices=["semantickitti", "groups"],
                   help="class colours: the 19-class standard, or 7 colourblind-safe groups")
    p.add_argument("--no-patchworkpp", action="store_true", help="use the semantic-class ground proxy")
    p.add_argument("--semantics", default="gt", choices=["gt", "frnet"],
                   help="gt (default): 19-class labels from the .label files, as every "
                        "benchmark uses. frnet: OPT-IN deep-learning mode, labels predicted "
                        "by FRNet on each raw scan (host only). Motion stays ground truth "
                        "in both.")
    p.add_argument("--fast-scatter", action="store_true",
                   help="with --semantics frnet: apply scripts/frnet_fast_scatter.py (verified)")
    p.add_argument("--threads", type=int, default=None,
                   help="with --semantics frnet: torch.set_num_threads(N); 1 is required for "
                        "reproducible labels (OPEN-ITEMS R-j)")
    p.add_argument("--features", action="store_true",
                   help="dashboard: draw the curb/pothole (math 7.4) and confidence "
                        "(7.5) layers. Recomputed every 20 frames, not every frame -- "
                        "the detector is a full-window pass and costs ~1.1 s")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sched = schedule_mod.load(args.schedule)
    print(
        f"schedule {sched.name}: {len(sched.rings)} rings, "
        f"{sched.total_cells:,} cells, {sched.total_cells * 12 / 1e6:.2f} MB"
    )

    engine = None
    if not args.no_map:
        engine = MapEngine(sched, ghost_removal=not args.show_ghosts,
                           clip_class_ids=args.clip_class_ids, device=args.device)
        print(f"map: {engine.handle.allocated_slots:,} slots preallocated, "
              f"ghost removal {'OFF' if args.show_ghosts else 'ON'}, "
              f"device {engine.device}")
        if engine.device_bytes() is not None:
            print(f"     {engine.device_bytes()['static'] / 1e6:.2f} MB on the card: "
                  "grid, scatter and cleanup buffers")

    if args.semantics != "frnet" and (args.fast_scatter or args.threads is not None):
        raise SystemExit("--fast-scatter and --threads only apply to --semantics frnet")
    frnet = None
    if args.semantics == "frnet":
        frnet = open_frnet(fast_scatter=args.fast_scatter, threads=args.threads)
        print("semantics: FRNet predictions (opt-in DL mode); motion: ground-truth "
              "moving-* labels -- FRNet has no motion output")

    view = None
    if args.viz or args.save:
        from vrgrid.dash.pipeline_view import PipelineView

        # `engine` is passed so the dashboard draws the map's occupied cells as
        # the real 2.5D surface, not just the point cloud -- this is what makes
        # `--show-ghosts` visibly change the screen (Gate 3).
        view = PipelineView(sched, spawn=args.viz, save_path=args.save,
                            color_by=args.color_by, ghost_removal=not args.show_ghosts,
                            palette=args.palette, engine=engine,
                            features=args.features)

    n, cleared, protected = 0, 0, 0
    truncated_frames, truncated_peak = 0, 0
    ground_method = None
    t_pull = time.perf_counter()
    # Buffers are reused only with no dashboard attached and on the host path: the
    # loop then finishes with each frame before pulling the next, and a DeviceFrame
    # retains points_world.
    for frame in iter_pipeline(args.seq, args.frames, use_patchworkpp=not args.no_patchworkpp,
                               start_frame=args.start_frame, device=args.device,
                               reuse_buffers=view is None and args.device == "cpu",
                               semantics_source=args.semantics, frnet=frnet):
        t_frame = time.perf_counter()          # the pull above was perception
        ground_method = frame.ground_method
        counters = engine.step(frame) if engine is not None else None
        t_step = time.perf_counter()
        if counters is not None:
            cleared += counters.cleared
            protected += counters.protected
            if counters.truncated:
                truncated_frames += 1
                truncated_peak = max(truncated_peak, counters.truncated)
        if view is not None:
            view.log_frame(frame, counters=counters,
                           timing_ms={"perception": (t_frame - t_pull) * 1e3,
                                      "engine": (t_step - t_frame) * 1e3})
        n += 1
        t_pull = time.perf_counter()           # the next pull starts now
        if n % 20 == 0:
            msg = f"  frame {frame.index}: {len(frame.points_sensor):,} pts"
            if counters is not None:
                msg += (f", {counters.occupied:,} occupied cells, "
                        f"{counters.cleared:,} cleared, {counters.protected:,} protected")
            print(msg)

    if view is not None:
        view.finish()   # final map + features state, whichever frame the run ended on
    print(f"done: {n} frames, sequence {args.seq}")
    if ground_method == "semantic_fallback":
        print("[!] ground: SEMANTIC-CLASS FALLBACK, not Patchwork++ -- every "
              "ground-derived number this run produced (heights, curbs, "
              "traversability) is on the fallback. See the RuntimeWarning above.")
    elif ground_method == "patchworkpp":
        print("ground: Patchwork++ (geometric segmenter)")
    if engine is not None:
        # The number the Gate 3 demo is actually about. With --show-ghosts it
        # is zero by construction, which is the point of printing it.
        print(f"ghost removal: {cleared:,} cells cleared, {protected:,} spared by "
              f"the current-return guard")
        # âš‘ Loud, and above any other summary, because it invalidates the line
        #   printed just before it. A truncated cell is never tested, keeps its
        #   occupancy, and cannot appear in `cleared` -- so a run that
        #   truncates reports a healthy ghost count while the map keeps its
        #   ghosts. Silence here used to be the only signal that the cap held.
        if truncated_frames:
            print(f"âš‘ visibility cap TRUNCATED on {truncated_frames} of {n} "
                  f"frames, up to {truncated_peak:,} occupied cells dropped "
                  f"and never tested.")
            print("  Raise visibility.max_candidate_cells; the ghost numbers "
                  "above are a floor, not a measurement.")
        elif cleared or protected:
            print("  visibility cap held on every frame: the whole occupied "
                  "set was tested.")
    if args.save:
        print(f"recording written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
