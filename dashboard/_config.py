"""Config- and schedule-derived values for the dashboard. [JP]

Imports no rerun. These read the frozen config files (`configs/thresholds.yaml`,
`configs/schedule_*.yaml`) via `vrgrid.grid.schedule`, and `tests/test_dashboard.py`
exercises them in CI where the optional `[dash]` extra (rerun-sdk) is not
installed. `pipeline_view.py` imports from here; `palettes.py` stays
colours-only, so the split is: colours in `palettes`, config/schedule wiring
here, rerun logging in `pipeline_view`.

Nothing in this project may hardcode a threshold or a ring size inline
(CLAUDE.md, "Don't"); this module is where the dashboard reads them instead.
"""

from vrgrid.cell import CELL_BYTES
from vrgrid.grid.schedule import CONFIG_DIR, load_thresholds
from vrgrid.grid.schedule import load as _load_schedule

# Bytes per voxel assumed for the dense-3D baseline. A dense voxel needs only an
# occupancy state, so 1 B is the charitable figure for the baseline -- the same
# convention `scripts/memory_table.py` uses for the report's 286x headline.
# (Our cell is CELL_BYTES = 12 because it also carries height, variance, class
# and flags; the ratio is deliberately measured against the baseline's best
# case, not ours.)
DENSE_VOXEL_BYTES = 1


def blind_cone_radius_m() -> float:
    """Blind-cone radius from `configs/thresholds.yaml` `sensor.blind_cone_m`.

    Math §1.4 eq (5): r_blind = h_s / tan|phi_min| = 1.73 / tan(24.8 deg) =
    3.74 m. The earlier plan assumed 1-2 m; master v4 corrected it. Read from
    the frozen config through the one cached reader (`load_thresholds`), never
    hardcoded -- a second literal is how the file and the running system drift.
    """
    return float(load_thresholds()["sensor"]["blind_cone_m"])


def playback_fps() -> float:
    """Timeline playback rate for recordings: the sensor's own frame rate,
    `1 / fusion.frame_dt_s` (10 Hz on KITTI), so a baked scene plays back in
    real time. Read from the frozen config, like every other rate here."""
    return 1.0 / float(load_thresholds()["fusion"]["frame_dt_s"])


def frame_budget_ms() -> float:
    """The per-frame budget: one sensor period, `1e3 * fusion.frame_dt_s`
    (100 ms at KITTI's 10 Hz). What the status feed colours each window against."""
    return 1e3 * float(load_thresholds()["fusion"]["frame_dt_s"])


def available_schedules() -> list[str]:
    """Schedule names discovered from `configs/schedule_*.yaml` -- no hardcoding.

    Returns the bare names (`5_10_20_40`, `5_10_50`) as `grid.schedule.load`
    accepts them.
    """
    return sorted(
        p.stem.removeprefix("schedule_") for p in CONFIG_DIR.glob("schedule_*.yaml")
    )


def schedule_legend_markdown(active: str) -> str:
    """Markdown table of every available schedule and its ring boundaries, read
    from the config files, with the active one marked.

    This is the (display-only) schedule selector: switching schedules mid-run is
    not wired -- `--schedule <name>` on `vrgrid.dash` / `vrgrid.run` picks one at
    startup. Ring half-widths and cell sizes come straight from the yaml via
    `grid.schedule.load`, so this cannot drift from what the engine uses.
    """
    lines = [
        "**Schedule selector** (display only -- `--schedule <name>` picks it at startup)",
        "",
        "| schedule | rings: half-width m / cell cm | cells | MB |",
        "|---|---|---|---|",
    ]
    for name in available_schedules():
        s = _load_schedule(name)
        rings = ", ".join(f"{r.half_width_m:g}/{r.cell_m * 100:g}" for r in s.rings)
        mark = " **(active)**" if name == active else ""
        lines.append(
            f"| `{name}`{mark} | {rings} | {s.total_cells:,} | "
            f"{s.total_cells * CELL_BYTES / 1e6:.2f} |"
        )
    return "\n".join(lines)


def dense_3d_baseline(schedule) -> dict:
    """The dense 5 cm 3-D voxel grid this map replaces, derived from the schedule.

    Not a magic number: the covered volume is the outer ring's square footprint
    (`2 * half_width` on a side) times the schedule's own vertical extent, at the
    schedule's finest resolution (`base_cell_m`) applied uniformly in all three
    axes -- the resolution the variable grid only spends in ring 0.

        footprint_m = 2 * schedule.rings[-1].half_width_m
        vertical_m  = hi - lo   from schedule.vertical_extent_m
        res_m       = schedule.base_cell_m
        voxels      = (footprint_m / res_m) ** 2 * (vertical_m / res_m)
        bytes       = voxels * DENSE_VOXEL_BYTES

    For `5_10_20_40`: (200 / 0.05)^2 * (8 / 0.05) = 2.56e9 voxels = 2.56 GB.
    """
    footprint_m = 2.0 * schedule.rings[-1].half_width_m
    lo, hi = schedule.vertical_extent_m
    vertical_m = hi - lo
    res_m = schedule.base_cell_m
    voxels = (footprint_m / res_m) ** 2 * (vertical_m / res_m)
    return {
        "footprint_m": footprint_m,
        "vertical_m": vertical_m,
        "res_m": res_m,
        "voxels": voxels,
        "bytes": voxels * DENSE_VOXEL_BYTES,
    }


def uniform_2_5d_baseline(schedule) -> dict:
    """A uniform grid at the schedule's finest cell over the outer ring's
    footprint, with the same `CELL_BYTES` cell -- the row `scripts/memory_table.py`
    prints, derived the same way. For `5_10_20_40`: (200 / 0.05)^2 * 12 B = 192 MB.
    It is a pure cell-count ratio against ours, so it does not depend on bytes
    per cell."""
    footprint_m = 2.0 * schedule.rings[-1].half_width_m
    cells = (footprint_m / schedule.base_cell_m) ** 2
    return {"footprint_m": footprint_m, "cells": cells, "bytes": cells * CELL_BYTES}


def grid_memory_stats(n_occupied: int, schedule) -> dict:
    """Live map memory vs the dense-3D baseline for `n_occupied` occupied cells.

    `live_bytes` is exactly `n_occupied * CELL_BYTES` -- the real storage the
    occupied cells take, not a fixed allocation figure. `ratio` is how many
    times bigger the dense-3D baseline for the same covered volume would be.
    """
    dense = dense_3d_baseline(schedule)
    live_bytes = int(n_occupied) * CELL_BYTES
    return {
        "n_occupied": int(n_occupied),
        "cell_bytes": CELL_BYTES,
        "live_bytes": live_bytes,
        "dense_voxels": dense["voxels"],
        "dense_bytes": dense["bytes"],
        "ratio": (dense["bytes"] / live_bytes) if live_bytes else float("inf"),
        "footprint_m": dense["footprint_m"],
        "vertical_m": dense["vertical_m"],
        "res_m": dense["res_m"],
    }


def _fmt_bytes(b: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= scale:
            return f"{b / scale:.2f} {unit}"
    return f"{b:.0f} B"


def memory_overlay_markdown(n_occupied: int, schedule) -> str:
    """The per-frame live memory overlay (a Rerun TextDocument).

    Shows the real occupied-cell storage now, the derived dense-3D baseline for
    the same covered volume, and the live ratio -- all recomputed each frame
    from `n_occupied`."""
    s = grid_memory_stats(n_occupied, schedule)
    return "\n".join([
        "**Live map memory**",
        "",
        "| | value |",
        "|---|---|",
        f"| occupied cells | {s['n_occupied']:,} |",
        f"| cell size | {s['cell_bytes']} B (`CELL_BYTES`) |",
        f"| **map in use now** | **{_fmt_bytes(s['live_bytes'])}** |",
        (
            f"| dense-3D baseline | {_fmt_bytes(s['dense_bytes'])} "
            f"({s['dense_voxels'] / 1e9:.2f} G voxels @ {DENSE_VOXEL_BYTES} B) |"
        ),
        f"| **live ratio** | **{s['ratio']:,.0f}x smaller** |",
        "",
        (
            f"dense volume: {s['footprint_m']:g} x {s['footprint_m']:g} x "
            f"{s['vertical_m']:g} m at {s['res_m'] * 100:g} cm uniform "
            f"= ({s['footprint_m']:g}/{s['res_m']:g})^2 x ({s['vertical_m']:g}/{s['res_m']:g}) voxels"
        ),
        "",
        (
            "_live ratio counts only occupied cells; the fixed preallocation is "
            f"{schedule.total_cells * CELL_BYTES / 1e6:.2f} MB "
            f"({s['dense_bytes'] / (schedule.total_cells * CELL_BYTES):,.0f}x -- the report figure)._"
        ),
    ])



def _ms(v) -> str:
    return "—" if v is None else f"{v:.0f} ms"


def _rate(ms) -> str:
    return "—" if not ms else f"{ms:.0f} ms · {1e3 / ms:.1f} fps"


# A frame whose time is above this fraction of the budget reads "near budget":
# a display threshold for the frame-time tile and the status feed, not a map one.
NEAR_BUDGET_FRACTION = 0.8

# Where the semantic labels come from: SemanticKITTI's ground-truth `.label`
# files (`perception/semantics.py`; CLAUDE.md "Don't"). Shown as a header badge
# so nobody reads the class colours as a segmentation result.
LABEL_SOURCE = "GT"


def header_markdown(frame_index: int, *, ghost_removal: bool) -> str:
    """The Demo tab's header: identity, frame, and the run's modes as badges.
    Inline code spans are the nearest thing to pills Rerun's Markdown draws."""
    return "\n".join([
        f"**VRgrid** · SIH26053 · Chronicles.exe · frame {frame_index:,}",
        "",
        f"`GHOST REMOVAL: {'ON' if ghost_removal else 'OFF'}` `LABELS: {LABEL_SOURCE}`",
    ])


def _bar(fraction: float, width: int = 20) -> str:
    """A text fill bar. A non-empty map never shows an empty bar."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    filled = round(fraction * width)
    if fraction > 0 and filled == 0:
        filled = 1
    return "█" * filled + "░" * (width - filled)


def kpi_memory_markdown(n_occupied: int, schedule, *, has_map: bool = True) -> str:
    """KPI tile: storage the occupied cells use, of the fixed allocation."""
    if not has_map:
        return "## —\n\nMap memory · back end off"
    alloc = schedule.total_cells * CELL_BYTES
    used = int(n_occupied) * CELL_BYTES
    # The big figure alone on its line: "1.67 / 8.94 MB" at heading size did not
    # fit a quarter-width tile, and a smaller font defeats the tile. The tile's
    # own title already says "Map memory", so no label line repeats it.
    # Bar and allocation share one line: on a third line the allocation fell
    # below the tile's height.
    return "\n".join([
        f"## {used / 1e6:.2f} MB",
        "",
        f"`{_bar(used / alloc, width=16)}` of {alloc / 1e6:.2f} MB fixed",
    ])


def kpi_frame_time_markdown(total_ms: float | None, budget_ms: float) -> str:
    """KPI tile: this frame's time and a verdict against the budget. Markdown
    cannot be coloured in Rerun, so the verdict is a symbol and words; the
    frame-time graph carries the green / red."""
    if total_ms is None:
        return "## —\n\nFrame time"
    if total_ms <= NEAR_BUDGET_FRACTION * budget_ms:
        verdict = "✓ under budget"
    elif total_ms <= budget_ms:
        verdict = "△ near budget"
    else:
        verdict = "✗ over budget"
    return "\n".join([f"## {total_ms:.0f} ms", "", f"**{verdict}** · {budget_ms:.0f} ms", "",
                      "Frame time"])


def kpi_moving_markdown(cleared_now: int | None, cleared_run: int | None, *,
                        ghost_removal: bool) -> str:
    """KPI tile: moving-object cells cleared this frame, and over the run."""
    if not ghost_removal:
        return "## off\n\nMoving-object cells cleared"
    now = "—" if cleared_now is None else f"{int(cleared_now):,}"
    run = "—" if cleared_run is None else f"{int(cleared_run):,}"
    return "\n".join([f"## {now}", "", f"this frame · {run} this run", "",
                      "Moving-object cells cleared"])


def kpi_deterministic_markdown() -> str:
    """KPI tile: determinism is a CI-blocking test, not something one run can
    show on its own -- the tile says where the check lives."""
    # Short lines: a quarter-width tile wrapped "identical map hash" onto a
    # third line and cut it.
    return "## ✓ Deterministic\n\nsame input → identical hash\n\nCI-blocking test"


# Results the SIH26053 deck quotes (slides 4-5), shown on the demo's side panel.
# Each is measured, and each entry names where: none is typed from memory.
DECK_MEASURED = (
    # docs/known-limitations.md §6 -- 40 frames, 5_10_20_40, Patchwork++, rings 1-3
    # rho = IL / spread (math §9.3 eq. 28): 1.0 means merging cost nothing
    # beyond the terrain's own roughness.
    ("Accuracy loss when merging, ρ (1.0 = none)", "1.18 – 1.84 (seq 07 / 08)"),
    # docs/known-limitations.md §1 -- post-fix re-soak of every seq 08 frame
    ("Cleanup failures", "0 of 4,071 frames (seq 08)"),
    # tests/test_determinism.py -- CI-blocking
    ("Deterministic", "two identical runs → bit-identical map hash"),
)
# scripts/memory_table.py -- the sparse / hashed 3D row is an estimate there too
SPARSE_3D_ROW = ("Sparse / hashed 3D (estimate)", "~130–240 MB", "~15–27×")
# scripts/sampling_table.py -- derived scope limits for the HDL-64E
POTHOLE_30CM_RANGE_M = 8.3
PEDESTRIAN_MOTION_RANGE_M = 25


def status_markdown(frame_index: int, n_occupied: int, schedule, *, ghost_removal: bool,
                    counters=None, run: dict | None = None, timing_ms: dict | None = None,
                    ground_method: str | None = None, has_map: bool = True,
                    gpu=None) -> str:
    """The "Run" tab's numbers: one heading, one line of context, one table.

    Only what changes while the demo plays -- this frame beside the whole run:
    frame time, the GPU drawing the dashboard (when `gpu`, a
    `gpu_stats.GpuReading`, is given), ghost cells removed / kept by the guard /
    skipped by the cap, and map memory against its fixed allocation. The
    memory comparison and the deck's measured results never change, so they
    live on the "Details" tab (`details_markdown`), logged once.

    `counters` is this frame's `StepCounters`; `timing_ms` this frame's
    `{"perception", "engine", "dashboard", "total"}`; `run` the view's running
    sums. Any of them may be None and shows as "—".
    """
    budget = frame_budget_ms()
    ground = ("" if not ground_method else
              " · ground Patchwork++" if ground_method == "patchworkpp" else
              " · ⚑ ground: semantic-class fallback")
    rows = [
        f"### Frame {frame_index:,}",
        "",
        f"ghost removal {'ON' if ghost_removal else 'OFF'}{ground} · budget {budget:.0f} ms",
        "",
        "| | this frame | whole run |",
        "|---|---|---|",
    ]
    t = timing_ms or {}
    n = (run or {}).get("n", 0)
    avg_total = run["total"] / n if run and n else None
    rows.append(f"| Frame time | {_rate(t.get('total'))} | {_rate(avg_total)} |")
    if gpu is not None:
        peak = (run or {}).get("gpu_peak_pct")
        rows.append(
            f"| GPU ({gpu.name.replace('NVIDIA GeForce ', '')}) | {gpu.util_pct:.0f}% · "
            f"{gpu.mem_used_mib / 1024:.1f} / {gpu.mem_total_mib / 1024:.1f} GB | "
            + ("—" if peak is None else f"peak {peak:.0f}%") + " |")

    if not has_map:
        rows.append("| Map | back end off (`--no-map`) | |")
        return "\n".join(rows)

    def now(key):
        return "—" if counters is None else f"{int(getattr(counters, key)):,}"

    def total(key):
        return "—" if run is None else f"{int(run[key]):,}"

    if ghost_removal:
        capped = run is not None and run["truncated"] > 0
        rows += [
            f"| Moving-object cells cleared | {now('cleared')} | {total('cleared')} |",
            # Not "static cells": the guard keeps ANY cell with a return in this
            # scan, including where a moving car is right now (math §10.4).
            f"| Cells kept (seen this scan) | {now('protected')} | {total('protected')} |",
            (f"| Skipped by candidate cap{' ⚑' if capped else ''} | {now('truncated')} | "
             f"{total('truncated')} |"),
        ]
    else:
        rows.append("| Moving-object cells cleared | off | off |")
    peak = max(int((run or {}).get("peak_occupied", 0)), int(n_occupied))
    # Two rows, not one "Map memory" figure: the storage the occupied cells
    # take is NOT the process footprint -- the whole grid is allocated once at
    # startup, and that fixed allocation is the headline claim.
    rows.append(f"| Map cells in use | {_fmt_bytes(int(n_occupied) * CELL_BYTES)} | "
                f"peak {_fmt_bytes(peak * CELL_BYTES)} |")
    rows.append(f"| Map allocation | {_fmt_bytes(schedule.total_cells * CELL_BYTES)}, "
                "fixed at startup | never grows |")
    return "\n".join(rows)


def details_markdown(schedule) -> str:
    """The "Details" tab: what does not change while the demo plays.

    1. Memory at the same extent -- the deck's slide-4 table, derived from the
       schedule exactly as `scripts/memory_table.py` derives it.
    2. Measured -- the deck's slide-4/5 results and scope limits (DECK_MEASURED).
    """
    alloc = schedule.total_cells * CELL_BYTES
    uniform = uniform_2_5d_baseline(schedule)["bytes"]
    dense = dense_3d_baseline(schedule)["bytes"]
    base = f"{schedule.base_cell_m * 100:g} cm"
    cells = "/".join(f"{r.cell_m * 100:g}" for r in schedule.rings)
    rows = [
        "**VRgrid · foveated 2.5D LiDAR map** · SIH26053 · Chronicles.exe",
        "",
        "### Memory at the same extent",
        "",
        "| map | size | vs vrgrid |",
        "|---|---|---|",
        f"| **vrgrid {cells} cm** | **{_fmt_bytes(alloc)}** | **1×** |",
        f"| Uniform {base} 2.5D | {_fmt_bytes(uniform)} | {uniform / alloc:.1f}× |",
        f"| {SPARSE_3D_ROW[0]} | {SPARSE_3D_ROW[1]} | {SPARSE_3D_ROW[2]} |",
        f"| Dense {base} 3D | {_fmt_bytes(dense)} | {dense / alloc:,.0f}× |",
        "",
        "### Measured",
        "",
        "From the evaluation runs named in each row, not from the recording playing now.",
        "",
        "| result | value |",
        "|---|---|",
    ]
    rows += [f"| {label} | {value} |" for label, value in DECK_MEASURED]
    rows += [
        f"| Blind spot radius | {blind_cone_radius_m():.2f} m — no ground seen closer |",
        f"| 30 cm pothole | detectable up to {POTHOLE_30CM_RANGE_M:g} m |",
        f"| Pedestrian motion | detectable up to {PEDESTRIAN_MOTION_RANGE_M:g} m |",
    ]
    return "\n".join(rows)


def map_legend_markdown(schedule, *, color_by: str, blind_cone_m: float,
                        palette_note: str | None = None, features: bool = False,
                        feature_interval: int = 20) -> str:
    """The long legend, on the "Details" tab. The Demo tab carries the short one
    -- a single row of real colour swatches (`pipeline_view.legend_items`).
    Ring sizes come from the schedule; nothing here is typed by hand.
    """
    points = (f"LiDAR points coloured by `{color_by}`"
              + (f" ({palette_note})" if palette_note else ""))
    lines = ["### Rings (squares around the car)", ""]
    lines += [f"- {r.cell_m * 100:g} cm cells, out to {r.half_width_m:g} m" for r in schedule.rings]
    lines += [
        "",
        "### Map",
        "",
        "- **occupied cells**: coloured by height, blue low → orange high",
        "- **free space**: seen and clear (translucent slate)",
        "- **unknown**: never assumed free (violet)",
        (f"- **blind spot {blind_cone_m:.2f} m** (red circle): the sensor cannot see the ground "
         "inside it right now (the map there is remembered from earlier frames)"),
        "- **moving objects** (pink dots) · **the car** (white arrow) · **path driven** (amber line)",
        f"- {points}, in the follow view only",
    ]
    if features:
        lines += [
            "",
            f"### Features (refreshed every {feature_interval} frames)",
            "",
            "- **orange boxes** curbs at their measured height",
            "- **vermillion boxes** potholes at their measured depth",
            "- **dark → light points** drivability confidence",
            ("- ⚑ Ring 3 confidence reads 0 on the live path: beyond ~50 m SemanticKITTI is "
             "unlabelled and `run/engine.py` stores unlabelled as class 0 (`car`), so dark there "
             "means unlabelled, not hazardous."),
        ]
    return "\n".join(lines)
