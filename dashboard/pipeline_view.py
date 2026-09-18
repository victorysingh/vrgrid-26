"""Rerun view of the real perception pipeline. [JP]

Replaces the Day-0 synthetic plane/boxes/slope (`demo_synthetic.py`) with the
actual per-frame output of `vrgrid.run`, one interpretation layer at a time via
`color_by`:

    intensity     raw Velodyne intensity, greyscale  (the "no semantics" baseline)
    class         semantic_labels() 19-class colours
    motion        is_moving(): static dim, moving bright red
    ground        segment_ground(): ground tan, non-ground steel blue
    reflectivity  reflectivity.normalise(): rho_hat byte, greyscale

Ghost toggle
------------
`world/points` holds the ghost-free cloud; the moving points are logged
separately to `world/ghosts`. Toggling `world/ghosts` visibility in the viewer
(the eye icon in the entity panel) is the ghost toggle -- ON shows the trails
behind moving objects, OFF removes them, static points untouched either way.

`get_display_points(frame, ghost_removal)` is the single swap point: it decides
which points land in `world/points`, currently by branching on the raw motion
mask `frame.moving`. When Aakash's `scatter()`/`fuse()` exist, only that
function's body changes -- it queries the grid's transient layer instead -- and
none of the logging wiring moves.

Ring boundaries and the blind cone are logged under the vehicle transform, so
they track the vehicle. Points are world-frame and accumulate on the timeline.
"""

import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from vrgrid.cell import CELL_BYTES, OCC_FREE, OCC_UNKNOWN
from vrgrid.grid.confidence import drivable_confidence
from vrgrid.grid.features import detect

from ._config import (
    NEAR_BUDGET_FRACTION,
    blind_cone_radius_m,
    details_markdown,
    frame_budget_ms,
    header_markdown,
    kpi_deterministic_markdown,
    kpi_frame_time_markdown,
    kpi_memory_markdown,
    kpi_moving_markdown,
    map_legend_markdown,
    playback_fps,
    status_markdown,
    uniform_2_5d_baseline,
)
from .gpu_stats import GpuSampler

# Every colour below is defined in `palettes.py`, which imports no rerun: the
# CVD audit (`cvd.py`) and tests/test_cvd.py check these numbers in CI, where
# rerun-sdk (the optional `[dash]` extra) is not installed. Imported rather than
# defined here so the dashboard and the audit cannot drift apart -- and so this
# module stays the import path the rest of the dashboard already uses.
from .palettes import (
    _CLASS_LUT,
    _GROUP_LUT,
    GHOST_RGB,
    GROUP_MEMBERS,
    GROUP_NAMES,
    GROUP_RGB,
)

PALETTES = ("semantickitti", "groups")

# Elevation ramp for the occupied-cell surface: blue -> green -> yellow ->
# vermillion over a FIXED [-3, 15] m band (z-up world), so a cell's colour does
# not change frame to frame as the visible height range moves. Okabe-Ito stops,
# an ordered light->dark progression that survives all three CVD types.
_HEIGHT_STOPS = np.array(
    [[0, 114, 178], [0, 158, 115], [240, 228, 66], [213, 94, 0]], dtype=np.float32
)


def _height_ramp_exact(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((np.asarray(z, np.float32) - lo) / (hi - lo), 0.0, 1.0) * 3.0
    i = np.clip(t.astype(np.int64), 0, 2)
    f = (t - i)[:, None]
    return (_HEIGHT_STOPS[i] * (1.0 - f) + _HEIGHT_STOPS[i + 1] * f).astype(np.uint8)


# The ramp, evaluated once at 1,024 heights over the fixed band: 1.8 cm a step,
# below what one uint8 colour channel can show. Interpolating ~180,000 cells per
# map redraw was 12 ms of the 34 ms `_log_occupied` took (profiled, seq 00);
# a table lookup is one index cast and one gather.
_HEIGHT_LUT_N = 1024
_HEIGHT_LO_M, _HEIGHT_HI_M = -3.0, 15.0
_HEIGHT_LUT = _height_ramp_exact(
    np.linspace(_HEIGHT_LO_M, _HEIGHT_HI_M, _HEIGHT_LUT_N), _HEIGHT_LO_M, _HEIGHT_HI_M)


def _height_ramp(z: np.ndarray, lo: float = _HEIGHT_LO_M, hi: float = _HEIGHT_HI_M) -> np.ndarray:
    """(N, 3) uint8 colour per cell from its world-z, clipped to [lo, hi]."""
    if (lo, hi) != (_HEIGHT_LO_M, _HEIGHT_HI_M):
        return _height_ramp_exact(z, lo, hi)
    idx = (np.asarray(z, np.float32) - lo) * ((_HEIGHT_LUT_N - 1) / (hi - lo)) + 0.5
    np.clip(idx, 0, _HEIGHT_LUT_N - 1, out=idx)
    return _HEIGHT_LUT[idx.astype(np.intp)]


# Occupancy layers are drawn as three visually distinct things -- "unknown is
# not free" is a hard invariant (CLAUDE.md, math §10.1) and the view has to keep
# them apart:
#   OCCUPIED  elevation-ramped solid boxes at the cell's height (`_log_occupied`)
#   FREE      flat translucent slate tiles at the ground datum (`_log_free`) --
#             "the sensor looked here and it is clear"
#   UNKNOWN   the blind cone, plus any cell the map still calls UNKNOWN despite
#             having been observed (`_log_unknown`); never-observed allocation
#             slots are left undrawn, they are not information
_FREE_RGBA = (110, 125, 140, 70)      # slate, ~27% opacity -- recedes behind occupied
_UNKNOWN_RGBA = (150, 90, 160, 90)    # muted violet, matches the blind-cone "unknown" hue family

# --- §7.4 features and §7.5 confidence -------------------------------------
#
# Three more read-only layers over fields `MapEngine.step` already fills. They
# are OFF by default and OFF the per-frame path, and that is a measurement
# rather than caution: `features.detect` is a neighbourhood pass over all
# 910,000 window slots and costs **1,137 ms p50** on this machine, eleven times
# the whole 100 ms frame budget (`drivable_confidence` over four rings is a
# further 89 ms). Logging them every frame would make the viewer unusable and
# would quietly triple the cost of every `--save` recording, so `--features`
# recomputes them every `FEATURE_INTERVAL` frames and once more at the end.
# What is drawn is therefore the map as of the last recompute, which is why
# the final call after the loop exists: the last frame is the one a still gets
# taken from.
FEATURE_INTERVAL = 20

# --- viewer frame rate -------------------------------------------------------
#
# The dense map layers (occupied / free / unknown / confidence) are drawn as
# `Points3D` sized to the cell, not `Boxes3D`, and redrawn every `MAP_INTERVAL`
# frames rather than every frame. Both are about the viewer, not the map:
#
#   * Rerun processes box instances one at a time on the CPU; points and
#     spheres take a GPU fast path the Rerun team puts at ~100x faster
#     (rerun-io/rerun#10276, which names voxel occupancy grids as the case).
#     Seq 00 has ~205,000 map cells a frame, which is where playback stalled.
#     A point of radius cell/2 still steps 5 -> 10 -> 20 -> 40 cm with range,
#     so the foveation reads exactly as it did.
#   * Resending every cell every frame is also most of the live-run overhead:
#     the pipeline is 81 ms p50, a live scene ~220 ms. The map changes slowly
#     and Rerun holds the last value between logs, so the point cloud, ghosts
#     and vehicle stay at the full frame rate while the map refreshes at 2 Hz.
#
# `finish()` redraws once after the loop so a run that stops between intervals
# still ends on the true final map -- the frame a still gets taken from.
MAP_INTERVAL = 5

_CURB_RGBA = (230, 159, 0, 235)       # Okabe-Ito orange -- a positive step, drawn standing up
_POTHOLE_RGBA = (213, 94, 0, 245)     # Okabe-Ito vermillion -- a negative one, drawn sunken

# Confidence is a scalar field, so it needs a ramp rather than a flat colour,
# and the ramp is ordered in LIGHTNESS (dark = no confidence -> light = full)
# so it survives all three CVD types without relying on hue. Deliberately not
# red-to-green, which is the one ramp a deuteranope cannot read at all.
_CONFIDENCE_STOPS = np.array(
    [[38, 54, 92], [0, 114, 178], [86, 180, 233], [240, 228, 66]], dtype=np.float32
)


def _confidence_ramp(c: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 from a drivable-confidence in [0, 1]. §7.5."""
    t = np.clip(np.asarray(c, np.float32), 0.0, 1.0) * 3.0
    i = np.clip(t.astype(np.int64), 0, 2)
    f = (t - i)[:, None]
    return (_CONFIDENCE_STOPS[i] * (1.0 - f) + _CONFIDENCE_STOPS[i + 1] * f).astype(np.uint8)


# --- the demo layout -----------------------------------------------------------
#
# One fixed layout, saved into every recording, so a baked scene opens the same
# way on any machine. Two top-level tabs:
#
#   Demo     the map on the left in two views -- the whole map from above, and a
#            follow camera close behind the car with the LiDAR sweep on it; on
#            the right a header with mode badges, four KPI tiles, a memory-over-
#            time graph and a frame-time graph; under both, one row of real
#            colour swatches as the legend.
#   Details  what a judge asks for: the full run table, the memory comparison,
#            measured results, the long legend, the moving-object and GPU
#            charts and the status feed.
#
# The map itself is drawn exactly as before the KPI redesign: points in the
# original colours. Flat tiles and three colour-mode layers were tried
# (0db3932) and taken back out -- they doubled the recording and the live demo
# stuttered.
#
# Rerun 0.37 cannot colour Markdown or fill the area under a line, so the KPI
# verdicts are symbols and words, and frame time is two lines -- green under
# budget, red over -- split with NaN gaps.
_BACKGROUND_RGB = (14, 17, 22)

_TRAIL_RGB = (240, 180, 60)
_BLIND_SPOT_RGB = (230, 60, 60)
_RING_LINE_RGB = (200, 205, 212)

# The vehicle marker: one flat white triangle pointing along +x (forward),
# 3.2 m long and 2 m wide, lifted 25 cm so it sits on top of the ground cells.
# A map-style "you are here" arrow -- position and heading, nothing else.
_MARKER_Z_M = 0.25
_MARKER_VERTS_M = [[2.0, 0.0, _MARKER_Z_M], [-1.2, 1.0, _MARKER_Z_M], [-1.2, -1.0, _MARKER_Z_M]]
_MARKER_RGB = (240, 240, 240)

_UNDER_RGB = (0, 210, 130)
_OVER_RGB = (255, 70, 70)
_BUDGET_RGB = (235, 235, 235)

# The status feed (Details tab): one coloured line per map-redraw window.
_FEED_OK_RGB = (0, 158, 115)
_FEED_WARN_RGB = (240, 228, 66)
_FEED_BAD_RGB = (213, 94, 0)
_FEED_NEAR_BUDGET = NEAR_BUDGET_FRACTION

_LEGEND_SPACING = 10.0


def legend_items(schedule):
    """The Demo tab's one-row legend, `[(label, rgb), ...]`: every mark on the
    map in the colour it is drawn with. Ring sizes come from the schedule."""
    sizes = "/".join(f"{r.cell_m * 100:g}" for r in schedule.rings)
    return [
        ("height: low", tuple(int(c) for c in _HEIGHT_STOPS[0])),
        ("mid", tuple(int(c) for c in _HEIGHT_STOPS[1])),
        ("high", tuple(int(c) for c in _HEIGHT_STOPS[3])),
        (f"rings {sizes} cm", _RING_LINE_RGB),
        ("moving", tuple(GHOST_RGB)),
        ("car", _MARKER_RGB),
        ("path", _TRAIL_RGB),
        ("blind spot", _BLIND_SPOT_RGB),
        ("free space", _FREE_RGBA[:3]),
        ("unknown", _UNKNOWN_RGBA[:3]),
    ]


def _series_styles(schedule):
    base = f"{schedule.base_cell_m * 100:g} cm"
    return {
        "stats/frame_ms/under": ("under budget", _UNDER_RGB),
        "stats/frame_ms/over": ("over budget", _OVER_RGB),
        "stats/frame_ms/budget": ("budget", _BUDGET_RGB),
        "stats/memory_mb/uniform": (f"uniform {base} grid", (230, 159, 0)),
        "stats/memory_mb/allocation": ("VRgrid allocation", (86, 180, 233)),
        "stats/memory_mb/in_use": ("VRgrid cells in use", (0, 158, 115)),
        "stats/moving/cleared": ("moving-object cells cleared", (0, 158, 115)),
        "stats/gpu_pct/usage": ("GPU usage", (0, 158, 115)),
        "stats/gpu_pct/memory": ("GPU memory", (204, 121, 167)),
    }


def _split_frame_time(total_ms: float, budget_ms: float):
    """`(under, over)` for the two frame-time lines: the value on one line and
    NaN -- a gap in Rerun -- on the other."""
    return (np.nan, total_ms) if total_ms > budget_ms else (total_ms, np.nan)


def _demo_blueprint(schedule, background=_BACKGROUND_RGB):
    # Both map views live in `/world/follow`, a frame that carries the vehicle's
    # POSITION only (see `log_frame`), and show everything under /world. So both
    # cameras go where the car goes. `tracking_entity` with a fixed eye did not
    # follow: by frame 1,000 of seq 00 the car was 370 m away and the view still
    # sat at the origin. Position but not heading, on purpose: a camera bolted
    # to the car's yaw swings on every small heading change between 10 Hz frames.
    def map_view(name, position, look_target, contents="/world/**"):
        return rrb.Spatial3DView(
            name=name, origin="/world/follow", contents=contents,
            background=rrb.Background(color=list(background)),
            line_grid=rrb.LineGrid3D(visible=False),
            eye_controls=rrb.EyeControls3D(position=position, look_target=look_target))

    maps = rrb.Vertical(
        # High and nearly straight down: the whole 200 m map, all four rings --
        # the MAP only. With the raw sweep drawn over it (class colours on top
        # of height colours) nobody could tell sensor data from map cells.
        map_view("Overview · the map, all four rings", [-20.0, 0.0, 150.0], [0.0, 0.0, 0.0],
                 contents=["+ /world/**", "- /world/points", "- /world/ghosts"]),
        # Low behind the car: the fine 5 cm ring up close, with the sweep on it.
        map_view("Around the car · map + LiDAR points", [-30.0, -18.0, 22.0], [15.0, 0.0, 0.0]),
        row_shares=[1, 1],
    )

    no_legend = rrb.PlotLegend(visible=False)
    budget = frame_budget_ms()
    uniform_mb = uniform_2_5d_baseline(schedule)["bytes"] / 1e6
    side = rrb.Vertical(
        rrb.TextDocumentView(name="VRgrid", origin="/panel/header"),
        rrb.Grid(
            rrb.TextDocumentView(name="Map memory", origin="/panel/kpi/memory"),
            rrb.TextDocumentView(name="Frame time", origin="/panel/kpi/frame_time"),
            rrb.TextDocumentView(name="Moving objects", origin="/panel/kpi/moving"),
            rrb.TextDocumentView(name="Determinism", origin="/panel/kpi/deterministic"),
            grid_columns=2,
        ),
        # The strongest picture: a flat VRgrid line against the uniform grid it replaces.
        rrb.TimeSeriesView(name="Memory MB · VRgrid vs uniform grid · click to jump",
                           origin="/stats/memory_mb",
                           axis_y=rrb.ScalarAxis(range=(0.0, 1.1 * uniform_mb))),
        rrb.TimeSeriesView(name="Frame time ms · green under budget · red over · click to jump",
                           origin="/stats/frame_ms", plot_legend=no_legend,
                           axis_y=rrb.ScalarAxis(range=(0.0, 2.5 * budget))),
        row_shares=[1.2, 3.0, 2.2, 2.2],       # 0.8 cut the header's badge line
    )
    n_items = len(legend_items(schedule))
    legend = rrb.Spatial2DView(
        name="Legend", origin="/panel/legend_swatches",
        visual_bounds=rrb.VisualBounds2D(
            x_range=[-0.6 * _LEGEND_SPACING, (n_items - 0.4) * _LEGEND_SPACING],
            y_range=[-2.2, 2.2]),
        background=rrb.Background(color=list(background)),
    )
    demo = rrb.Vertical(
        rrb.Horizontal(maps, side, column_shares=[7, 3]),
        legend,
        row_shares=[12, 1.3],
        name="Demo",
    )

    # Opened only when a judge asks.
    details = rrb.Vertical(
        rrb.Horizontal(
            rrb.TextDocumentView(name="Run details", origin="/panel/status"),
            rrb.TextDocumentView(name="Memory and results", origin="/panel/details"),
            rrb.TextDocumentView(name="Legend", origin="/panel/legend"),
            column_shares=[1, 1.2, 1],
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(name="Moving-object cells cleared per frame",
                               origin="/stats/moving", plot_legend=no_legend),
            rrb.TimeSeriesView(name="GPU rendering % · green usage · pink memory",
                               origin="/stats/gpu_pct", plot_legend=no_legend,
                               axis_y=rrb.ScalarAxis(range=(0.0, 100.0))),
            # Body only: each line already names its frames and its colour
            # already says green / yellow / red.
            rrb.TextLogView(
                name="Status feed", origin="/panel/feed",
                columns=rrb.archetypes.TextLogColumns(
                    timeline_columns=[rrb.components.TimelineColumn("frame", visible=False)],
                    text_log_columns=[
                        rrb.components.TextLogColumn("LogLevel", visible=False),
                        rrb.components.TextLogColumn("EntityPath", visible=False),
                        rrb.components.TextLogColumn("Body", visible=True),
                    ],
                ),
            ),
            column_shares=[1, 1, 1.3],
        ),
        row_shares=[3, 2],
        name="Details",
    )
    return rrb.Blueprint(
        rrb.Tabs(demo, details, active_tab=0),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        # Starts playing and loops, so a scene keeps running unattended for as
        # long as the demo lasts instead of stopping on its last frame.
        rrb.TimePanel(state="collapsed", timeline="frame", fps=playback_fps(),
                      play_state="playing", loop_mode="all"),
    )


def legend_markdown(palette: str) -> str:
    """Which raw classes fall into each colour, so nothing is lost in grouping."""
    if palette == "groups":
        lines = ["**Palette: groups** (colourblind-safe)", "", "| group | raw classes |", "|---|---|"]
        lines += [f"| {g} | {', '.join(GROUP_MEMBERS[g])} |" for g in GROUP_NAMES]
        return "\n".join(lines)
    return "**Palette: semantickitti** -- the standard 19-class map (not colourblind-safe)"


def _frame_colors(frame, color_by: str, palette: str = "semantickitti") -> np.ndarray:
    """(N, 3) uint8 colour per point of `frame`, for the chosen layer.

    `palette` only affects the `class` layer: "semantickitti" (default) or
    "groups" (19 classes -> 7 colourblind-safe super-groups).
    """
    if color_by == "intensity":
        g = np.clip(frame.points_sensor[:, 3] * 255.0 * 1.5, 0, 255).astype(np.uint8)
        return np.repeat(g[:, None], 3, axis=1)
    if color_by == "class":
        idx = np.clip(frame.semantic + 1, 0, 19)
        if palette == "groups":
            return GROUP_RGB[_GROUP_LUT[idx]]
        return _CLASS_LUT[idx]
    if color_by == "motion":
        c = np.full((len(frame.moving), 3), 90, dtype=np.uint8)  # static: neutral grey
        c[frame.moving] = GHOST_RGB
        return c
    if color_by == "ground":
        # tan vs steel-blue -- an orange/blue pair, the axis all three CVD types
        # preserve; min Delta-E 55 under every simulation (dashboard/cvd.py).
        c = np.empty((len(frame.ground), 3), dtype=np.uint8)
        c[frame.ground] = (170, 130, 90)
        c[~frame.ground] = (70, 130, 180)
        return c
    if color_by == "reflectivity":
        return np.repeat(frame.reflectivity8[:, None], 3, axis=1)
    raise ValueError(f"unknown color_by {color_by!r}")


COLOR_BY = ("intensity", "class", "motion", "ground", "reflectivity")


def get_display_points(frame, ghost_removal: bool, color_by: str = "class",
                       palette: str = "semantickitti"):
    """Points + colours for the `world/points` entity.

    Args:
        frame: a run.PerceptionFrame.
        ghost_removal: True drops the points flagged by `frame.moving`.
        color_by: which colour layer (see COLOR_BY).
        palette: "semantickitti" (default) or "groups" -- only affects `class`.

    Returns:
        (xyz (M, 3) float32, colors (M, 3) uint8).

    PLACEHOLDER: the mask is `frame.moving` (raw per-frame motion labels). When
    the grid's transient layer exists this becomes a grid query -- e.g.
    ``keep = ~grid.is_transient(frame.points_world)`` -- and callers do not change.
    """
    keep = ~frame.moving if ghost_removal else np.ones(len(frame.moving), dtype=bool)
    xyz = frame.points_world[keep].astype(np.float32)
    colors = _frame_colors(frame, color_by, palette)[keep]
    return xyz, colors


class PipelineView:
    def __init__(self, schedule, spawn: bool = False, save_path: str | None = None,
                 color_by: str = "class", ghost_removal: bool = True,
                 palette: str = "semantickitti", engine=None, features: bool = False,
                 map_interval: int = MAP_INTERVAL):
        self.color_by = color_by
        # Frames between map redraws -- see MAP_INTERVAL. 1 draws every frame.
        self.map_interval = max(1, int(map_interval))
        self.ghost_removal = ghost_removal
        self.palette = palette
        self.schedule = schedule
        # The map back end (`run.engine.MapEngine`), or None for perception-only
        # runs. When present, `log_frame` draws its occupied cells as the real
        # 2.5D surface -- see `_log_occupied`.
        self.engine = engine
        self._last_occupied_n = 0   # updated by _log_occupied, read by _log_memory
        # §7.4 curbs / potholes and §7.5 confidence. Off by default because
        # `features.detect` costs 1,137 ms -- see FEATURE_INTERVAL.
        self.features = bool(features) and engine is not None
        self._frames_logged = 0
        # Both a live viewer and a file: log to both at once. `rr.save` on its
        # own replaces the viewer connection, so a --viz --save run used to
        # record without rendering -- and the GPU chart then recorded an idle
        # GPU. Rendering while recording is what gives that chart its meaning.
        self.rendering_live = bool(spawn)
        if spawn and save_path:
            rr.init("vrgrid_pipeline", spawn=False)
            # 40%, not Rerun's 75% default: on the 15 GB demo laptop the default
            # outgrew free RAM and the OS watchdog killed the run. The viewer
            # drops its oldest frames at the cap; the file sink keeps them all.
            rr.spawn(connect=False, memory_limit="40%")
            rr.set_sinks(rr.GrpcSink(), rr.FileSink(save_path))
        else:
            rr.init("vrgrid_pipeline", spawn=spawn)
            if save_path:
                rr.save(save_path)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # The layout goes out with the first frame (`_send_blueprint`), once the
        # `frame` timeline exists. Sent here, before any data, the viewer logged
        # `Timeline "frame" not found` and the playback rate did not take.
        self._blueprint_sent = False
        self._budget_ms = frame_budget_ms()
        # Running sums behind the "whole run" column of Key numbers.
        self._run = {"n": 0, "perception": 0.0, "engine": 0.0, "dashboard": 0.0,
                     "total": 0.0, "cleared": 0, "protected": 0, "truncated": 0,
                     "peak_occupied": 0, "gpu_peak_pct": None}
        self._alloc_mb = schedule.total_cells * CELL_BYTES / 1e6
        # The status feed accumulates one map-redraw window of frames per line.
        self._feed_window = []                 # [(frame index, total ms), ...]
        self._feed_cleared = 0
        self._ground_method = None
        self._trail = []                       # vehicle positions, for the path driven
        palette_note = (("SemanticKITTI 19-class colours" if palette == "semantickitti"
                         else "7 colourblind-safe groups") if color_by == "class" else None)
        rr.log("panel/legend", rr.TextDocument(
            map_legend_markdown(
                schedule, color_by=color_by, blind_cone_m=blind_cone_radius_m(),
                palette_note=palette_note, features=self.features,
                feature_interval=FEATURE_INTERVAL),
            media_type=rr.MediaType.MARKDOWN), static=True)
        rr.log("panel/details", rr.TextDocument(details_markdown(schedule),
                                                media_type=rr.MediaType.MARKDOWN), static=True)
        rr.log("panel/kpi/deterministic", rr.TextDocument(
            kpi_deterministic_markdown(), media_type=rr.MediaType.MARKDOWN), static=True)
        items = legend_items(schedule)
        xs = np.arange(len(items), dtype=np.float32) * _LEGEND_SPACING
        rr.log("panel/legend_swatches",
               rr.Points2D(np.stack([xs, np.zeros_like(xs)], axis=1), radii=0.7,
                           colors=[rgb for _, rgb in items], labels=[t for t, _ in items],
                           show_labels=True),
               static=True)
        for path, (name, rgb) in _series_styles(schedule).items():
            rr.log(path, rr.SeriesLines(colors=[rgb], names=[name], widths=[2.5]), static=True)
        self._uniform_mb = uniform_2_5d_baseline(schedule)["bytes"] / 1e6
        self._log_rings(schedule)
        self._log_blind_cone(blind_cone_radius_m())
        self._log_marker()

        # GPU telemetry, polled off the frame path (gpu_stats.py). The first two
        # feed lines say what the map is given and what draws it.
        self._gpu = GpuSampler()
        # Static data on an entity replaces everything else logged there, so each
        # start-up line gets its own path and the per-frame lines another again.
        rr.log("panel/feed/startup/alloc", rr.TextLog(
            f"map allocated once at startup: {self._alloc_mb:.2f} MB · "
            f"{schedule.total_cells:,} cells · never grows",
            level=rr.TextLogLevel.INFO, color=_FEED_OK_RGB), static=True)
        gpu = self._gpu.latest()
        rr.log("panel/feed/startup/gpu", rr.TextLog(
            (f"GPU: {gpu.name} · {gpu.mem_total_mib / 1024:.1f} GB · "
             + ("usage recorded while rendering this run live" if self.rendering_live
                else "usage recorded while baking, no viewer rendering")
             if gpu is not None else "GPU telemetry unavailable (no NVIDIA GPU / nvidia-smi)"),
            level=rr.TextLogLevel.INFO if gpu is not None else rr.TextLogLevel.WARN,
            color=_FEED_OK_RGB if gpu is not None else _FEED_WARN_RGB), static=True)

    def _log_marker(self):
        """The vehicle as a small flat arrow under the vehicle transform, so it
        moves and turns with the car. Static: one triangle, logged once. The
        vertices wind counter-clockwise seen from above, so it faces up."""
        rr.log("world/vehicle/marker",
               rr.Mesh3D(vertex_positions=_MARKER_VERTS_M, triangle_indices=[[0, 1, 2]],
                         albedo_factor=_MARKER_RGB),
               static=True)

    def _log_rings(self, schedule):
        """Ring boundaries, straight from the passed `Schedule`, drawn as
        SQUARES. Ring membership is the L-infinity distance in the vehicle
        frame (`lattice.ring_of`, math §6.1 eq. 18), so ring 0 reaches 10 m
        along the axes and 14.1 m at its corners; the circles drawn here before
        showed the wrong boundary exactly where foveation is meant to be seen.
        Logged under the vehicle transform, so they turn with the heading, as
        membership does. Half-widths and cell sizes come from
        `configs/schedule_*.yaml` -- nothing here is hardcoded."""
        for ring in schedule.rings:
            hw = ring.half_width_m
            square = np.array([[hw, hw, 0], [-hw, hw, 0], [-hw, -hw, 0],
                               [hw, -hw, 0], [hw, hw, 0]], dtype=np.float32)
            rr.log(
                f"world/vehicle/rings/ring_{ring.ring}_{ring.cell_m * 100:g}cm",
                # No in-scene label: four labels plus the blind cone's piled up
                # on top of each other at the vehicle. The legend lists them.
                rr.LineStrips3D([square], colors=[_RING_LINE_RGB], radii=0.05),
                static=True,
            )

    def _log_blind_cone(self, radius_m: float):
        th = np.linspace(0, 2 * np.pi, 65)
        strip = np.stack([radius_m * np.cos(th), radius_m * np.sin(th), np.zeros_like(th)], axis=1)
        rr.log(
            "world/vehicle/blind_cone",
            rr.LineStrips3D([strip.astype(np.float32)], colors=[_BLIND_SPOT_RGB], radii=0.05),
            static=True,
        )

    def _cell_m_per_slot(self, slots: np.ndarray) -> np.ndarray:
        """Cell edge length (m) for each occupied slot, from the ring it lives
        in. This is what makes the foveation visible: a point drawn at a cell's
        own size grows from 5 cm near the vehicle to 40 cm at 100 m."""
        out = np.full(len(slots), self.engine.sched.base_cell_m, dtype=np.float32)
        for layout in self.engine.handle.rings:
            sel = (slots >= layout.offset) & (slots < layout.offset + layout.slots)
            out[sel] = layout.cell_m
        return out

    def _centres_world(self, slots: np.ndarray):
        """World-frame `(x, y, z)` for arbitrary slots, via the engine's own
        inverse of `flat_slot`. ego (0, 0) leaves the centres in the world
        frame, exactly as `MapEngine.occupied_cells` does it for the occupied
        set -- this just reuses it for the free / unknown sets."""
        n = len(slots)
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        if n:
            self.engine._centres(slots, np.zeros(2), x, y, z)
        return x, y, z

    def _log_occupied(self):
        """The real 2.5D occupied surface: every cell the map currently calls
        OCCUPIED, drawn as a point of radius cell/2 at its world xy, at its
        visibility height z
        (ceiling where one was seen, ground otherwise -- the same height §10.4
        tests), sized to its ring's cell. `--show-ghosts` keeps the moving
        car's cells here; the default clears them via §10.4, so this entity is
        where the toggle actually shows on screen.

        Calling `occupied_cells()` also refreshes `engine.occ_state`, which
        `_log_free` / `_log_unknown` then read -- so this runs first."""
        slots, x, y, z = self.engine.occupied_cells()
        self._last_occupied_n = len(slots)   # for the memory tile and graph, no second pass
        if len(slots) == 0:
            rr.log("world/map/occupied", rr.Clear(recursive=True))
            return
        cell_m = self._cell_m_per_slot(slots)
        centres = np.stack([x, y, z], axis=1).astype(np.float32)
        rr.log(
            "world/map/occupied",
            rr.Points3D(centres, radii=cell_m / 2.0, colors=_height_ramp(z)),
        )

    def _log_free(self):
        """FREE cells -- observed and clear -- as translucent slate points at
        the ground datum, sized to their ring's cell so the foveation still
        reads. Distinct from OCCUPIED (height-coloured) and from UNKNOWN
        (undrawn / blind cone): "looked and clear" is not "did not look"."""
        free = np.flatnonzero(self.engine.occ_state == OCC_FREE)
        if len(free) == 0:
            rr.log("world/map/free", rr.Clear(recursive=True))
            return
        x, y, z = self._centres_world(free)
        cell_m = self._cell_m_per_slot(free)
        centres = np.stack([x, y, z], axis=1).astype(np.float32)
        rr.log(
            "world/map/free",
            rr.Points3D(centres, radii=cell_m / 2.0, colors=[_FREE_RGBA]),
        )

    def _log_unknown(self):
        """UNKNOWN cells that were nonetheless OBSERVED at least once (blind-cone
        cells, cells whose evidence never cleared `n_min`) -- the planner-
        relevant unknown, drawn in the blind-cone hue. Never-observed allocation
        slots (the bulk of the grid) are deliberately not drawn: they carry no
        information, and 700k boxes would bury the ones that matter. The blind
        cone itself is always drawn as a circle under the vehicle transform."""
        obs = self.engine.handle.grid["obs_count"]
        seen_unknown = np.flatnonzero((self.engine.occ_state == OCC_UNKNOWN) & (obs > 0))
        if len(seen_unknown) == 0:
            rr.log("world/map/unknown", rr.Clear(recursive=True))
            return
        x, y, z = self._centres_world(seen_unknown)
        cell_m = self._cell_m_per_slot(seen_unknown)
        centres = np.stack([x, y, z], axis=1).astype(np.float32)
        rr.log(
            "world/map/unknown",
            rr.Points3D(centres, radii=cell_m / 2.0, colors=[_UNKNOWN_RGBA]),
        )

    def _ring_slices(self):
        """`[(slice, side), ...]` per ring -- the shape `features.detect` and
        `confidence.summarise` both take. Built from the engine's allocation so
        it cannot drift from the storage layout."""
        return [(slice(r.offset, r.offset + r.slots), r.side)
                for r in self.engine.handle.rings]

    def _flat(self, level: int, slot: np.ndarray) -> np.ndarray:
        """`Curbs.slot` / `Potholes.slot` are indices WITHIN a ring window --
        the dataclasses say so, and merging rings without this offset silently
        aliases ring 0's slot 5 onto ring 3's. Lift them to flat SoA slots."""
        return slot.astype(np.int64) + self.engine.handle.rings[level].offset

    def _log_curbs(self, curbs):
        """§7.4 curb edges, drawn STANDING UP at their measured rise.

        A curb is a step, so the box is drawn from the cell's surface up by
        `height_cm` rather than as a flat marker: the thing the problem
        statement says a 2D grid loses is exactly this height, and drawing it
        at its real magnitude is the difference between a detection and a
        measurement. Colour is flat -- height is already carried by the shape.
        """
        cent, half = [], []
        for level, c in enumerate(curbs):
            if not len(c):
                continue
            x, y, z = self._centres_world(self._flat(level, c.slot))
            h = np.maximum(c.height_cm.astype(np.float32), 1.0) / 100.0  # cm -> m
            cent.append(np.stack([x, y, z + h / 2.0], axis=1))
            half.append(np.stack([np.full_like(h, c.cell_m / 2.0),
                                  np.full_like(h, c.cell_m / 2.0), h / 2.0], axis=1))
        if not cent:
            rr.log("world/map/curbs", rr.Clear(recursive=True))
            return
        rr.log("world/map/curbs",
               rr.Boxes3D(centers=np.concatenate(cent).astype(np.float32),
                          half_sizes=np.concatenate(half).astype(np.float32),
                          colors=[_CURB_RGBA], fill_mode="solid"))

    def _log_potholes(self, holes):
        """§7.4 potholes, drawn SUNK to their measured depth below the rim.

        The mirror of `_log_curbs` and for the same reason: a negative obstacle
        is the other half of the sentence a 2D grid cannot answer, and a marker
        floating at the surface would show the detection while hiding the one
        number that says whether it matters.
        """
        cent, half = [], []
        for level, h in enumerate(holes):
            if not len(h):
                continue
            x, y, z = self._centres_world(self._flat(level, h.slot))
            d = np.maximum(h.depth_cm.astype(np.float32), 1.0) / 100.0   # cm -> m
            cent.append(np.stack([x, y, z - d / 2.0], axis=1))
            half.append(np.stack([np.full_like(d, h.cell_m / 2.0),
                                  np.full_like(d, h.cell_m / 2.0), d / 2.0], axis=1))
        if not cent:
            rr.log("world/map/potholes", rr.Clear(recursive=True))
            return
        rr.log("world/map/potholes",
               rr.Boxes3D(centers=np.concatenate(cent).astype(np.float32),
                          half_sizes=np.concatenate(half).astype(np.float32),
                          colors=[_POTHOLE_RGBA], fill_mode="solid"))

    def _log_confidence(self):
        """§7.5 per-cell confidence in the DRIVABILITY verdict, over observed
        cells, as flat tiles floating 15 cm above the surface.

        Floated deliberately: this is a scalar field over the same cells
        `_log_free` and `_log_occupied` already draw, and at the surface it
        would z-fight with both. Read it WITH the occupancy layers, never
        instead of them -- `drivable_confidence`'s own docstring makes the
        point that a cell can be traversable under §7.1 and still carry 0.1,
        which is the case the bitfield alone cannot express.

        ⚑ RING 3 CURRENTLY READS A FALSE 0.000, AND IT IS NOT THIS LAYER.
          Every ring-3 tile reports zero confidence with `binding` =
          "not-drivable", and the honest reason is a bug upstream, not a
          vegetation verge:

            * SemanticKITTI's annotation stops at roughly 50 m. Measured on
              seq 00 frames 0-59, **100.0%** of returns in the 50-100 m band
              (104,758 of 104,758) carry no label -- `semantic_labels` gives
              them -1. Ring 2 is 8.3% unlabelled, rings 0-1 under 1%.
            * `run/engine.py` maps `semantic < 0` to class **0**, and learning
              id 0 is `car`. So every ring-3 cell stores "car", `car` is not in
              `drivable_classes`, and the class gate zeroes the cell outright.
            * `eval/harness.py`'s `learning_ids()` does the same conversion
              CORRECTLY -- `-1 -> CLASS_UNLABELLED (31)`, which is in no
              drivable set and so fails safe as *unknown* rather than as a
              parked car. The two paths disagree about one conversion.

          Scope: this is the LIVE MapEngine path only (this dashboard,
          `vrgrid.run`, `timing_table`, `ghost_removal_figure`). The published
          §2b per-ring and rho tables go through `harness.run_sequence` and are
          NOT affected.

          Magnitude, so nobody over-reads the fix: bypassing the class gate,
          ring 3's worst-of-four margin is mean **0.008**, median 0.000, above
          zero on 3.6% of tiles -- `surface` is exactly zero on 91.8% of them
          and `geometry` on 60.5%. Corrected, ring 3 stays dark. What changes
          is that it would be dark for a true reason and `binding` would say
          `surface` / `geometry` instead of falsely saying "not-drivable" --
          low confidence honestly labelled unknown, rather than a phantom car.

          `run/engine.py` is not this lane, so this is documented here and
          flagged rather than fixed. A dark ring-3 tile in a recording made
          before that fix means "unlabelled beyond 50 m", not "hazard".
        """
        cent, cols, rad = [], [], []
        for level, (sl, side) in enumerate(self._ring_slices()):
            cell_m = self.engine.sched.rings[level].cell_m
            conf = drivable_confidence(self.engine.handle.grid, sl, side, cell_m,
                                       self.engine.thresholds)
            seen = np.flatnonzero(self.engine.handle.grid["obs_count"][sl] >= 1)
            if not seen.size:
                continue
            x, y, z = self._centres_world(seen + self.engine.handle.rings[level].offset)
            cent.append(np.stack([x, y, z + 0.15], axis=1))
            cols.append(_confidence_ramp(conf[seen]))
            rad.append(np.full(seen.size, cell_m / 2.0, np.float32))
        if not cent:
            rr.log("world/map/confidence", rr.Clear(recursive=True))
            return
        # Points, like the occupancy layers and for the same reason (see
        # MAP_INTERVAL): this layer covers every observed cell.
        rr.log("world/map/confidence",
               rr.Points3D(np.concatenate(cent).astype(np.float32),
                           radii=np.concatenate(rad), colors=np.concatenate(cols)))

    def log_features(self):
        """Recompute and draw the §7.4 / §7.5 layers. ~1.2 s -- see
        FEATURE_INTERVAL. Public so a caller can force one final pass after the
        loop, which is the state a still gets taken from."""
        if not self.features:
            return
        rings = self._ring_slices()
        curbs, holes = detect(self.engine.handle.grid, self.engine.sched, rings,
                              self.engine.thresholds, buffers=self.engine.buffers)
        self._log_curbs(curbs)
        self._log_potholes(holes)
        self._log_confidence()

    def _send_blueprint(self):
        """The demo layout (`_demo_blueprint`), sent once with the first frame."""
        self._blueprint_sent = True
        try:
            rr.send_blueprint(_demo_blueprint(self.schedule))
        except (AttributeError, TypeError):   # a viewer too old for the layout API
            pass

    def _log_trail(self):
        """The path driven so far, world frame. Redrawn with the map, not every
        frame: it is one line strip, but it grows with the run."""
        if len(self._trail) < 2:
            return
        rr.log("world/trajectory",
               rr.LineStrips3D([np.stack(self._trail)], colors=[_TRAIL_RGB], radii=0.12))

    def _log_stats(self, frame, counters, timing_ms, dashboard_ms):
        """Every frame: the Demo tab's header, KPI tiles and two graphs, and the
        Details tab's table, charts and feed. Nothing is ever blank between map
        redraws; the memory figures are the last redraw's."""
        timing = None
        if timing_ms and "perception" in timing_ms:
            timing = {"perception": timing_ms["perception"],
                      "engine": timing_ms.get("engine", 0.0), "dashboard": dashboard_ms}
            timing["total"] = sum(timing.values())
            self._run["n"] += 1
            for key, value in timing.items():
                self._run[key] += value
            under, over = _split_frame_time(timing["total"], self._budget_ms)
            rr.log("stats/frame_ms/under", rr.Scalars(under))
            rr.log("stats/frame_ms/over", rr.Scalars(over))
            rr.log("stats/frame_ms/budget", rr.Scalars(self._budget_ms))
        if counters is not None:
            for key in ("cleared", "protected", "truncated"):
                self._run[key] += int(getattr(counters, key))
            rr.log("stats/moving/cleared", rr.Scalars(counters.cleared))
        self._run["peak_occupied"] = max(self._run["peak_occupied"], self._last_occupied_n)
        if self.engine is not None:
            rr.log("stats/memory_mb/in_use", rr.Scalars(self._last_occupied_n * CELL_BYTES / 1e6))
            rr.log("stats/memory_mb/allocation", rr.Scalars(self._alloc_mb))
            rr.log("stats/memory_mb/uniform", rr.Scalars(self._uniform_mb))

        # GPU: the sampler's latest snapshot, never a blocking call on this path.
        gpu = self._gpu.latest()
        if gpu is not None:
            rr.log("stats/gpu_pct/usage", rr.Scalars(gpu.util_pct))
            rr.log("stats/gpu_pct/memory", rr.Scalars(gpu.mem_pct))
            peak = self._run["gpu_peak_pct"]
            self._run["gpu_peak_pct"] = gpu.util_pct if peak is None else max(peak, gpu.util_pct)

        self._log_feed(frame, counters, timing)
        tracked = self._run["n"] > 0 or counters is not None
        md = rr.MediaType.MARKDOWN
        rr.log("panel/status", rr.TextDocument(
            status_markdown(frame.index, self._last_occupied_n, self.schedule,
                            ghost_removal=self.ghost_removal, counters=counters,
                            run=self._run if tracked else None, timing_ms=timing,
                            ground_method=self._ground_method,
                            has_map=self.engine is not None, gpu=gpu),
            media_type=md))
        rr.log("panel/header", rr.TextDocument(
            header_markdown(frame.index, ghost_removal=self.ghost_removal), media_type=md))
        rr.log("panel/kpi/memory", rr.TextDocument(
            kpi_memory_markdown(self._last_occupied_n, self.schedule,
                                has_map=self.engine is not None), media_type=md))
        rr.log("panel/kpi/frame_time", rr.TextDocument(
            kpi_frame_time_markdown(timing["total"] if timing else None, self._budget_ms),
            media_type=md))
        rr.log("panel/kpi/moving", rr.TextDocument(
            kpi_moving_markdown(None if counters is None else int(counters.cleared),
                                self._run["cleared"] if tracked else None,
                                ghost_removal=self.ghost_removal), media_type=md))

    def _log_feed(self, frame, counters, timing):
        """One coloured status line per map-redraw window, plus an immediate red
        line whenever the candidate cap skips cells (a ghost could then persist).

        Green: every frame in the window within 80% of the budget. Yellow: the
        worst frame within the budget. Red: over it. Frames with no timing do
        not form windows, so a caller that passes none gets no feed lines."""
        if counters is not None and counters.truncated:
            rr.log("panel/feed/alerts", rr.TextLog(
                f"frame {frame.index:,}: candidate cap skipped {counters.truncated:,} cells",
                level=rr.TextLogLevel.ERROR, color=_FEED_BAD_RGB))
        if timing is None:
            return
        self._feed_window.append((frame.index, timing["total"]))
        if counters is not None:
            self._feed_cleared += int(counters.cleared)
        if len(self._feed_window) < self.map_interval:
            return
        totals = [ms for _, ms in self._feed_window]
        worst, avg = max(totals), sum(totals) / len(totals)
        if worst <= _FEED_NEAR_BUDGET * self._budget_ms:
            verdict, level, rgb = "within budget", rr.TextLogLevel.INFO, _FEED_OK_RGB
        elif worst <= self._budget_ms:
            verdict, level, rgb = "near budget", rr.TextLogLevel.WARN, _FEED_WARN_RGB
        else:
            verdict, level, rgb = "over budget", rr.TextLogLevel.ERROR, _FEED_BAD_RGB
        ghosts = (f" · {self._feed_cleared:,} moving-object cells cleared"
                  if self.ghost_removal and counters is not None else "")
        rr.log("panel/feed/frames", rr.TextLog(
            f"frames {self._feed_window[0][0]:,}–{self._feed_window[-1][0]:,} · "
            f"avg {avg:.0f} ms · worst {worst:.0f} ms · {verdict}{ghosts}",
            level=level, color=rgb))
        self._feed_window.clear()
        self._feed_cleared = 0

    def log_map(self):
        """Draw the occupancy layers at the current time. Called every
        `map_interval` frames by `log_frame`, and once more by `finish()`.
        No-op without an engine."""
        if self.engine is None:
            return
        self._log_occupied()   # also refreshes engine.occ_state
        self._log_free()
        self._log_unknown()

    def finish(self):
        """Call once after the loop: redraws the map and the §7.4 / §7.5
        layers so the recording ends on the true final state, whichever frame
        the run stopped on. Both would otherwise be up to an interval stale."""
        self.log_map()
        self.log_features()
        self._log_trail()
        self._gpu.stop()

    def log_frame(self, frame, counters=None, timing_ms=None):
        """Draw one frame.

        `counters` is what `MapEngine.step` returned for this frame, and
        `timing_ms` the caller's `{"perception": ms, "engine": ms}` for it.
        Both are optional: with them the side panels show live frame time and
        ghost-removal numbers; without them, what the view can know alone.
        """
        t_start = time.perf_counter()
        rr.set_time("frame", sequence=frame.index)
        if not self._blueprint_sent:
            self._send_blueprint()
        self._ground_method = getattr(frame, "ground_method", self._ground_method)

        xyz, colors = get_display_points(frame, self.ghost_removal, self.color_by, self.palette)
        rr.log("world/points", rr.Points3D(xyz, colors=colors, radii=0.03))

        if self.engine is not None:
            # Not every frame -- see MAP_INTERVAL. `finish()` draws the last.
            if self._frames_logged % self.map_interval == 0:
                self.log_map()
            # Not every frame: `features.detect` is 1,137 ms. The caller runs
            # one more pass after the loop so the final state is complete.
            if self.features and self._frames_logged % FEATURE_INTERVAL == 0:
                self.log_features()
        self._frames_logged += 1

        # The removed set, on its own entity -- this is what the demo toggles.
        ghosts = frame.points_world[frame.moving].astype(np.float32)
        rr.log("world/ghosts", rr.Points3D(ghosts, colors=list(GHOST_RGB), radii=0.09))

        # vehicle transform: origin + heading from the GT pose (world-frame yaw)
        fwd_world = frame.pose[:3, :3] @ np.array([0.0, 0.0, 1.0])  # camera z = forward
        yaw = np.arctan2(-fwd_world[0], fwd_world[2])  # into the z-up world convention
        rr.log("world/vehicle", rr.Transform3D(
            translation=frame.vehicle_xyz_world.astype(np.float32),
            rotation=rr.RotationAxisAngle(axis=[0, 0, 1], angle=float(yaw)),
        ))
        # The chase camera's frame: the vehicle's position, no rotation. The map
        # view's origin (`_demo_blueprint`), so the camera goes where the car goes.
        rr.log("world/follow", rr.Transform3D(
            translation=np.asarray(frame.vehicle_xyz_world, np.float32)))

        # The path driven. `_frames_logged` was already advanced above, so this
        # frame is a map-redraw frame when (count - 1) lands on the interval.
        self._trail.append(np.asarray(frame.vehicle_xyz_world, np.float32))
        if (self._frames_logged - 1) % self.map_interval == 0:
            self._log_trail()
        self._log_stats(frame, counters, timing_ms, (time.perf_counter() - t_start) * 1e3)
