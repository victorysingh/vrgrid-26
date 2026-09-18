"""The pipeline on the card: the map lives in device memory. [Shrestha]

`MapEngine(device="cuda")` and `iter_pipeline(device="cuda")` run every stage
that can run on a GPU there, against a grid that never leaves the card during
the frame loop:

    host    load (disk)  transform (pose)  Patchwork++ (C++ CPU library)
    device  range image  reflectivity  semantics/motion                (perception)
            shift  datum  bin  payload  scatter  fuse  occupancy
            slot->centre  guard  eq (32)  apply_miss                   (map)

What stays on the host, and why:
  * **Patchwork++** is a CPU library we wire in and do not reimplement
    (CLAUDE.md). It runs while the card is already working on the same frame:
    the perception kernels are queued before ground segmentation starts and
    nothing synchronises until it is done.
  * **The pose transform** is one BLAS product whose rounding the device cannot
    promise to reproduce, and it costs ~2 ms. Its output is uploaded.
  * **The disk read.**

**The contract is the CPU pipeline's output, bit for bit** -- range image,
inverse index, reflectivity, labels, and the full map hash after every frame.
`scripts/gpu_parity.py` checks all of it on real data; `gpu/cuda_kernels.py`
says what it took to make that true.

Readouts: `MapEngine.handle.grid` is a `MirroredGrid`. Reading any field copies
the device grid to host once and serves it until the next frame, so the
dashboard, the feature detectors and `map_hash` keep reading numpy exactly as
before and pay the copy only when they actually look. The host arrays are
read-only -- a write there would be silently overwritten by the next sync, so
it raises instead.
"""

import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
from vrgrid.cell import FLAG_BLIND
from vrgrid.gpu import cuda_kernels as K
from vrgrid.gpu.allocators import EMPTY_CELL
from vrgrid.gpu.kernels import (
    CEILING_NONE,
    SENSOR_HEIGHT_M,
    SIGMA_PHI_RAD,
    SIGMA_R_M,
    Z_MAX_CM,
    Z_MIN_CM,
    CellAggregate,
    new_sorted_scratch,
    scatter_sorted,
)
from vrgrid.gpu.shift import datum_step
from vrgrid.gpu.visibility import new_visibility_scratch, visibility_cleanup

DEVICES = ("cpu", "cuda")


def cuda_available() -> bool:
    """A usable card AND a cupy that can launch a kernel on it.

    Importing cupy is not enough: a missing CUDA header only fails at the first
    JIT, which is exactly the "broken GPU" `07-LOCAL-BUILD.md` warns about. So
    this launches one elementwise kernel and reads the answer back.
    """
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        return int((cupy.arange(4, dtype=cupy.int32) * 2).sum()) == 12
    except Exception:  # noqa: BLE001 -- any failure here means "not usable"
        return False


def resolve_device(device: str) -> str:
    """Validate a `--device` value. Fails loudly: a run asked for on the card
    that silently fell back to the CPU would publish a CPU number as a GPU one."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    if device == "cuda" and not cuda_available():
        raise RuntimeError(
            "--device cuda was requested but no working CUDA device was found "
            "(cupy import, device count, or first kernel launch failed). Run "
            "scripts/verify_env.py; the usual cause is cupy not finding its "
            "CUDA headers -- see docs/gpu-lane/07-LOCAL-BUILD.md.")
    return device


def synced_stage(timer):
    """`timer.stage`, synchronising the device before each stage's clock stops.

    CUDA calls return before the work is done, so an unsynchronised stage
    timer measures the launch. With no timer this is a no-op and nothing
    synchronises -- which is what lets the card and Patchwork++ overlap.
    """
    if timer is None:
        return lambda _name: nullcontext()
    import cupy

    @contextmanager
    def stage(name):
        with timer.stage(name):
            yield
            cupy.cuda.Stream.null.synchronize()
    return stage


# --- the host view of a device grid ---------------------------------------------

class MirroredGrid(dict):
    """`dict` of host arrays that refreshes itself from the device grid on read.

    Everything that reads `engine.handle.grid` -- the dashboard, `map_hash`,
    `occupancy_state`, the feature detectors -- gets ordinary numpy arrays,
    current as of the last completed frame. The copy (10.9 MB at 910,000
    slots) happens once per frame at most, and only if something reads.
    """

    def __init__(self, host: dict, device: dict):
        super().__init__(host)
        self._device = device
        self._dirty = False
        for arr in host.values():
            arr.flags.writeable = False

    def mark_dirty(self) -> None:
        self._dirty = True

    def sync(self) -> None:
        if not self._dirty:
            return
        for name, arr in dict.items(self):
            arr.flags.writeable = True
            try:
                self._device[name].get(out=arr)
            finally:
                arr.flags.writeable = False
        self._dirty = False

    def __getitem__(self, key):
        self.sync()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self.sync()
        return dict.get(self, key, default)

    def items(self):
        self.sync()
        return dict.items(self)

    def values(self):
        self.sync()
        return dict.values(self)


# --- perception -----------------------------------------------------------------

class DevicePerception:
    """Range image, reflectivity and labels on the card, for one run.

    Buffers are sized once. A `DeviceFrame` views them, so it is valid until
    the next `launch()` -- the same contract the scatter aggregate has.
    """

    def __init__(self, max_points: int = 150_000, sensor_cfg: dict | None = None):
        import cupy
        from vrgrid.perception import range_image, semantics

        if max_points > (1 << K.KEY_IDX_BITS):
            raise ValueError(f"max_points {max_points:,} exceeds the {1 << K.KEY_IDX_BITS:,} "
                             "the projection sort key can index")
        self.cp = cupy
        self.cfg = sensor_cfg if sensor_cfg is not None else range_image.load_sensor_config()
        self.h, self.w = int(self.cfg["num_rings"]), int(self.cfg["num_azimuth"])
        self.npix = self.h * self.w
        if self.npix > (1 << K.KEY_PIX_BITS):
            raise NotImplementedError(
                f"{self.h}x{self.w} = {self.npix:,} pixels does not fit the "
                f"{K.KEY_PIX_BITS}-bit pixel field of the projection sort key")
        self.d_theta, self.d_phi = range_image.bin_widths(self.cfg)
        self.phi_max = float(np.deg2rad(self.cfg["phi_max_deg"]))
        self.warn_frac = range_image.OUT_OF_FOV_WARN_FRAC

        # LUTs from JP's functions themselves, over every 16-bit label word --
        # which is all `semantic_labels` and `is_moving` ever look at.
        words = np.arange(1 << 16, dtype=np.uint32)
        lut = semantics.semantic_labels(words).astype(np.int32)
        self.max_class = int(lut.max())
        self.lut = cupy.asarray(lut)
        self.moving_lut = cupy.asarray(semantics.is_moving(words).astype(np.bool_))

        cap = max_points
        self.max_points = cap
        self.pts = cupy.zeros(cap * 4, np.float32)
        self.world = cupy.zeros(cap * 3, np.float64)
        self.labels = cupy.zeros(cap, np.uint32)
        self.key = cupy.zeros(cap, np.uint64)
        self.fov = cupy.zeros(cap, np.int8)
        self.planes = cupy.zeros(5 * self.npix, np.float32)
        self.inverse = cupy.zeros(self.npix, np.int32)
        self.rho = cupy.zeros(cap, np.uint8)
        self.sem = cupy.zeros(cap, np.int32)
        self.moving = cupy.zeros(cap, np.bool_)
        self.cls = cupy.zeros(cap, np.uint8)
        self.generation = 0
        self._n = 0

    def launch(self, points, points_world, raw_labels, stage=None):
        """Upload one scan and QUEUE its perception kernels. Returns without
        waiting for them -- ground segmentation runs on the host meanwhile."""
        stage = stage or (lambda _n: nullcontext())
        points = np.ascontiguousarray(points, dtype=np.float32)
        n, ncols = points.shape
        if ncols != 4:
            raise ValueError("device perception expects (N, 4) x, y, z, intensity")
        if n > self.max_points:
            raise ValueError(f"{n:,} points exceeds the device perception capacity "
                             f"of {self.max_points:,}")
        self.generation += 1
        self._n = n
        with stage("range_image"):
            self.pts[:4 * n].set(points.reshape(-1))
            self.world[:3 * n].set(np.ascontiguousarray(points_world, np.float64).reshape(-1))
            K.project_keys()(self.pts, 4, np.float32(np.pi), np.float32(self.d_theta),
                             self.phi_max, self.d_phi, self.h, self.w,
                             self.key[:n], self.fov[:n])
            self.key[:n].sort()
            self.planes.fill(np.nan)
            self.inverse.fill(-1)
            K.project_write()(self.key, self.pts, 4, self.npix, self.planes,
                              self.inverse, size=n)
        with stage("semantics"):
            self.labels[:n].set(np.ascontiguousarray(raw_labels, np.uint32))
            K.semantics()(self.labels, self.lut, self.moving_lut,
                          self.sem[:n], self.moving[:n], self.cls[:n])
        with stage("reflectivity"):
            self.rho[:n].fill(0)
            K.reflectivity_to_points()(self.planes, self.inverse, self.npix, self.rho,
                                       size=self.npix)

    def finish(self) -> None:
        """The one synchronising read: JP's out-of-FOV warning, same text."""
        n = self._n
        if not n:
            return
        clamped = int(self.cp.count_nonzero(self.fov[:n]))
        frac = clamped / n
        if frac > self.warn_frac:
            warnings.warn(
                f"range_image.project: {frac:.1%} of points fell "
                f"outside the vertical FOV [{self.cfg['phi_min_deg']}, "
                f"{self.cfg['phi_max_deg']}] deg and were clamped to an edge ring "
                f"-- check the sensor config and the point frame",
                stacklevel=3,
            )


class DeviceFrame:
    """A `PerceptionFrame` whose perception outputs are still on the card.

    Host fields are plain attributes. `semantic`, `moving`, `reflectivity8`,
    `range_image` and `inverse_index` download on first read -- the dashboard
    reads them, a headless run never does -- and refuse to once the next
    frame has overwritten the buffers they view.
    """

    def __init__(self, perception: DevicePerception, **host):
        self._p = perception
        self._generation = perception.generation
        self._n = perception._n
        self._cache = {}
        for k, v in host.items():
            setattr(self, k, v)

    def _current(self):
        if self._generation != self._p.generation:
            raise RuntimeError(
                f"frame {self.index}'s device buffers now hold a later frame; "
                "read its perception outputs before pulling the next one")

    def device_inputs(self):
        self._current()
        p, n = self._p, self._n
        return {"pts": p.pts, "ncols": 4, "dtype": np.float32, "world": p.world,
                "cls": p.cls[:n], "refl": p.rho[:n],
                "range": p.planes[:p.npix].reshape(p.h, p.w)}

    def _host(self, name, fn):
        if name not in self._cache:
            self._current()
            self._cache[name] = fn()
        return self._cache[name]

    @property
    def semantic(self):
        return self._host("semantic", lambda: self._p.sem[:self._n].get())

    @property
    def moving(self):
        return self._host("moving", lambda: self._p.moving[:self._n].get())

    @property
    def reflectivity8(self):
        return self._host("reflectivity8", lambda: self._p.rho[:self._n].get())

    @property
    def range_image(self):
        p = self._p
        return self._host("range_image", lambda: np.ascontiguousarray(
            p.planes.reshape(5, p.h, p.w).get().transpose(1, 2, 0)))

    @property
    def inverse_index(self):
        p = self._p
        return self._host("inverse_index", lambda: p.inverse.reshape(p.h, p.w).get())


# --- the map --------------------------------------------------------------------

class DeviceMap:
    """The grid and every per-frame map stage, on the card, for one engine.

    `MapEngine` still owns the ORDER of the stages and the ring windows; this
    owns where they run. Sizes come from the engine's host allocation so the two
    configurations refuse the same inputs.
    """

    def __init__(self, engine):
        import cupy
        from vrgrid.grid.lattice import REAR_FLOOR_RANGE_M

        self.cp = cupy
        self.pool = cupy.get_default_memory_pool()
        used_before = self.pool.used_bytes()
        handle = engine.handle
        th = engine.thresholds

        self.grid = {name: cupy.asarray(arr) for name, arr in handle.grid.items()}
        self.n_slots = handle.grid["log_odds"].size
        self.max_points = len(handle.scratch["key"])
        self.max_candidates = engine.max_candidates
        self.scatter_scratch = new_sorted_scratch(self.max_points, self.n_slots, xp=cupy)
        self.vis_scratch = new_visibility_scratch(self.max_candidates, np.float32, xp=cupy)

        # binning constants
        sched = engine.sched
        bs = engine.bin_scratch
        self.radii = cupy.asarray(bs["radii"])
        self.n_rings = len(sched.rings)
        self.sched = sched
        self.floor_ring = -1 if bs["floor_ring"] is None else int(bs["floor_ring"])
        self.rear_floor_m = REAR_FLOOR_RANGE_M
        self.base_cell_m = sched.base_cell_m
        self.tab_host = np.zeros(7 * self.n_rings, np.int64)
        self.tab_host[:self.n_rings] = bs["t_k"]
        self.tab = cupy.zeros(7 * self.n_rings, np.int64)
        self.per = cupy.zeros(3 * self.n_rings, np.float64)

        # payload constants, evaluated by the same Python expressions the host
        # functions evaluate, so the kernel multiplies by the same doubles
        self.sr2 = SIGMA_R_M ** 2
        self.sp2 = SIGMA_PHI_RAD ** 2
        self.cos2 = float(np.maximum(np.asarray(1.0, dtype=np.float64), 0.1) ** 2)

        # fuse / occupancy constants
        fus, occ = th.get("fusion", {}), th["occupancy"]
        q_cm2 = fus.get("process_noise_m2_per_s", 1e-4) * 1e4
        self.qdt = q_cm2 * fus.get("frame_dt_s", 0.1)
        self.hit = int(occ["log_odds_hit"])
        self.miss = int(occ["log_odds_miss"])
        self.clamp = tuple(int(c) for c in occ["log_odds_clamp"])
        self.l_occ = int(occ.get("log_odds_occupied", 0))
        self.unknown_below = int(occ["unknown_below_obs"])
        self.deq = cupy.asarray(K.dequantise_lut())
        self.thr = cupy.asarray(K.variance_code_thresholds())

        # frame buffers
        cap_p, cap_c = self.max_points, self.max_candidates
        self.idx = cupy.zeros(cap_p, np.int64)
        self.z_cm = cupy.zeros(cap_p, np.int16)
        self.w_q = cupy.zeros(cap_p, np.int32)
        self.ground = cupy.zeros(cap_p, np.bool_)
        self.occ_mask = cupy.zeros(self.n_slots, np.bool_)
        self.cand = {n: cupy.zeros(cap_c, np.float64) for n in "xyz"}
        self.guard = cupy.zeros(cap_c, np.bool_)
        self.work = cupy.zeros(cap_c, np.int64)
        self.row = cupy.zeros(cap_c, np.int64)
        self.ceil = cupy.zeros(cap_c, np.int16)
        self.gnd = cupy.zeros(cap_c, np.int16)
        self.pos = cupy.zeros(cap_c, np.int64)
        # host-frame uploads (tests, or a CPU perception pass feeding this map)
        self._up = {}
        self.static_bytes = self.pool.used_bytes() - used_before

    # -- inputs -------------------------------------------------------------

    def _upload_buffer(self, name, dtype, size):
        key = (name, np.dtype(dtype).name)
        buf = self._up.get(key)
        if buf is None or buf.size < size:
            buf = self.cp.zeros(size, dtype)
            self._up[key] = buf
        return buf

    def inputs(self, frame, n, cls_host=None):
        """Device views of one frame's map inputs. A `DeviceFrame` already has
        them; a host frame is uploaded into buffers kept for the next one."""
        if isinstance(frame, DeviceFrame):
            # Perception covers the whole scan; the map takes the first `n`,
            # exactly as the host path slices `[:n]` when max_points truncates.
            d = frame.device_inputs()
            d["cls"], d["refl"] = d["cls"][:n], d["refl"][:n]
        else:
            pts = np.ascontiguousarray(np.asarray(frame.points_sensor)[:n])
            ncols = pts.shape[1]
            dp = self._upload_buffer("pts", pts.dtype, max(n * ncols, 1))
            dp[:n * ncols].set(pts.reshape(-1))
            world = np.ascontiguousarray(np.asarray(frame.points_world, np.float64)[:n])
            dw = self._upload_buffer("world", np.float64, max(3 * n, 1))
            dw[:3 * n].set(world.reshape(-1))
            dc = self._upload_buffer("cls", np.uint8, max(n, 1))[:n]
            dc.set(np.ascontiguousarray(cls_host, np.uint8))
            dr = self._upload_buffer("refl", np.uint8, max(n, 1))[:n]
            dr.set(np.ascontiguousarray(np.asarray(frame.reflectivity8)[:n], np.uint8))
            plane = np.ascontiguousarray(np.asarray(frame.range_image)[:, :, 0], np.float32)
            dimg = self._upload_buffer("range", np.float32, plane.size)
            dimg[:plane.size].set(plane.reshape(-1))
            d = {"pts": dp, "ncols": ncols, "dtype": pts.dtype, "world": dw,
                 "cls": dc, "refl": dr,
                 "range": dimg[:plane.size].reshape(plane.shape)}
        self.ground[:n].set(np.ascontiguousarray(np.asarray(frame.ground)[:n], np.bool_))
        d["ground"] = self.ground[:n]
        return d

    # -- shift and datum ------------------------------------------------------

    def clear(self, flat_slots) -> None:
        """What `shift(..., soa)` writes into the newly visible strip."""
        if not len(flat_slots):
            return
        idx = self.cp.asarray(flat_slots)
        for name, arr in self.grid.items():
            arr[idx] = EMPTY_CELL.get(name, 0)

    def track_datum(self, datum_m, ego_z_m) -> float:
        """`shift.track_datum` against the device grid."""
        want, delta_cm = datum_step(datum_m, ego_z_m)
        if delta_cm is None:
            return want
        ground, ceiling = self.grid["ground_height"], self.grid["ceiling_height"]
        if abs(delta_cm) >= Z_MAX_CM - Z_MIN_CM:
            end = Z_MIN_CM if delta_cm > 0 else Z_MAX_CM
            seen = ceiling != CEILING_NONE
            ground.fill(end)
            self.grid["height_variance"].fill(0)
            self.cp.copyto(ceiling, np.int16(end), where=seen)
            return want
        K.rebase_heights()(delta_cm, Z_MIN_CM, Z_MAX_CM, CEILING_NONE, ground, ceiling,
                           self.grid["height_variance"])
        return want

    # -- bin, payload, scatter, fuse -------------------------------------------

    def bin(self, d, n, buffers, vehicle_xy_m=(0.0, 0.0), yaw_rad=0.0):
        """`lattice.bin_points` on the card. The per-frame constants come from
        the host function that builds them for `ring_of_into`, so both sides
        compare the same doubles."""
        from vrgrid.grid.lattice import _descent_constants

        t = self.tab_host
        R = self.n_rings
        for L, buf in enumerate(buffers):
            W = buf.side
            t[R + L], t[2 * R + L], t[3 * R + L] = W, buf.x0, buf.y0
            t[4 * R + L], t[5 * R + L], t[6 * R + L] = buf.offset, buf.x0 % W, buf.y0 % W
        self.tab.set(t)
        a_f, a_s, a_r, cy, sy, per = _descent_constants(
            self.sched, t[:R].tolist(), 0.0, vehicle_xy_m, yaw_rad)
        self.per.set(np.array(per, np.float64).reshape(-1))
        idx = self.idx[:n]
        K.bin_points()(d["world"], self.radii, R, a_f, a_s, a_r, self.floor_ring,
                       self.rear_floor_m, self.base_cell_m, cy, sy, self.per,
                       self.tab, idx)
        return idx

    def scatter(self, d, n, idx, z_datum) -> CellAggregate:
        z_cm, w_q = self.z_cm[:n], self.w_q[:n]
        K.payload(d["dtype"])(d["pts"], d["ncols"], d["world"], SENSOR_HEIGHT_M,
                              self.sr2, self.sp2, self.cos2, z_datum * 100.0,
                              bool(z_datum), z_cm, w_q)
        return scatter_sorted(idx, z_cm, w_q, d["refl"], d["cls"], d["ground"],
                              scratch=self.scatter_scratch)

    def fuse(self, agg: CellAggregate) -> None:
        k = len(agg.cells)
        if k == 0:
            return
        g = self.grid
        K.fuse()(agg.cells, agg.wz_sum, agg.w_sum, agg.n, agg.ceiling_cm, agg.refl_sum,
                 agg.class_id, g["ground_height"], g["ceiling_height"],
                 g["height_variance"], g["log_odds"], g["semantic_class"],
                 g["reflectivity"], g["obs_count"], g["frames_since_seen"],
                 self.deq, self.thr, self.qdt, self.hit, self.clamp[0], self.clamp[1],
                 size=k)

    # -- §10.4 --------------------------------------------------------------------

    def cleanup(self, touched, ego, z_datum, rings, buffers, image, sensor, floor_m,
                counters) -> None:
        """Occupied set -> centres -> guard -> eq (32) -> misses, all on the card.
        Mirrors `MapEngine._cleanup`; see it for the reasoning at each step."""
        cp, g = self.cp, self.grid
        K.occupied_mask()(g["log_odds"], g["obs_count"], g["flags"], self.l_occ,
                          self.unknown_below, np.uint8(FLAG_BLIND), self.occ_mask)
        occupied = cp.flatnonzero(self.occ_mask)
        total = int(occupied.size)
        counters.occupied = total
        if not total:
            return
        if total > self.max_candidates:
            counters.truncated = total - self.max_candidates
            occupied = occupied[:self.max_candidates]
        m = len(occupied)

        cp.take(g["ceiling_height"], occupied, out=self.ceil[:m])
        cp.take(g["ground_height"], occupied, out=self.gnd[:m])

        cx, cy, cz = self.cand["x"][:m], self.cand["y"][:m], self.cand["z"][:m]
        bounds = np.array([[r.offset, r.offset + r.slots] for r in rings], np.int64)
        edges = cp.searchsorted(occupied, cp.asarray(bounds.reshape(-1))).get()
        work, row = self.work, self.row
        for L, (layout, buf) in enumerate(zip(rings, buffers)):
            lo, hi = int(edges[2 * L]), int(edges[2 * L + 1])
            if lo == hi:
                continue
            W = buf.side
            local, r = work[lo:hi], row[lo:hi]
            # Same float64 expression, same order, as `MapEngine._centres`.
            cp.subtract(occupied[lo:hi], layout.offset, out=local)
            cp.floor_divide(local, W, out=r)
            cp.remainder(local, W, out=local)
            cp.subtract(local, buf.x0, out=local)
            cp.remainder(local, W, out=local)
            cp.add(local, buf.x0, out=local)
            cp.add(local, 0.5, out=cx[lo:hi])
            cp.multiply(cx[lo:hi], layout.cell_m, out=cx[lo:hi])
            cp.subtract(cx[lo:hi], float(ego[0]), out=cx[lo:hi])
            cp.subtract(r, buf.y0, out=r)
            cp.remainder(r, W, out=r)
            cp.add(r, buf.y0, out=r)
            cp.add(r, 0.5, out=cy[lo:hi])
            cp.multiply(cy[lo:hi], layout.cell_m, out=cy[lo:hi])
            cp.subtract(cy[lo:hi], float(ego[1]), out=cy[lo:hi])

        ceil, gnd = self.ceil[:m], self.gnd[:m]
        cp.copyto(gnd, ceil, where=ceil != CEILING_NONE)
        cp.divide(gnd, 100.0, out=cz)
        cp.add(cz, float(z_datum), out=cz)
        cp.subtract(cz, float(ego[2]), out=cz)

        k = len(touched)
        guard = self.guard[:m]
        if k:
            pos = self.pos[:m]
            pos[...] = cp.searchsorted(touched, occupied)
            cp.minimum(pos, k - 1, out=pos)
            cp.equal(touched[pos], occupied, out=guard)
        else:
            guard.fill(False)

        res = visibility_cleanup(cx, cy, cz, image, has_return_now=guard,
                                 sensor=sensor, floor_m=floor_m,
                                 protect_current_returns=True,
                                 scratch=self.vis_scratch)
        K.apply_miss()(occupied, res.see_through, g["log_odds"], self.miss,
                       self.clamp[0], self.clamp[1], size=m)
        counters.tested = res.tested
        counters.cleared = res.cleared
        counters.protected = res.protected
        counters.out_of_view = res.out_of_view

    # -- accounting ---------------------------------------------------------------

    def synchronize(self) -> None:
        self.cp.cuda.Stream.null.synchronize()

    def device_bytes(self) -> dict:
        """Card-side memory, pool used and reserved both -- cupy's pool caches,
        so `used` is what our arrays hold and `reserved` what cupy has taken
        from the driver (`03-CUDA-PORT-PLAN.md` §5)."""
        return {"static": self.static_bytes,
                "pool_used": self.pool.used_bytes(),
                "pool_reserved": self.pool.total_bytes()}
