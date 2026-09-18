"""The pipeline on the card: every device stage against its CPU reference. [Shrestha]

Each kernel in `gpu/cuda_kernels.py` is checked for EXACT equality with the
numpy function it replaces, on inputs chosen to hit the places where float
arithmetic disagrees: lattice-boundary coordinates for binning (cupy's own
float `//` gets 1.0 // 0.1 wrong), range boundaries for the projection, and
code boundaries for the variance codec. Then the engine as a whole:
`test_device_engine_is_bit_identical_to_cpu` drives the real `MapEngine` over
the Gate 3 ghost scene on both devices and compares the full grid hash after
every frame, and asserts the scene actually exercised clearing and the guard.

Device tests skip without a working card. The codec-table and
`resolve_device` tests need no card and run in CI.
"""

import warnings

import numpy as np
import pytest
from vrgrid.gpu import cuda_kernels as K
from vrgrid.gpu import device as dev
from vrgrid.gpu.kernels import map_hash
from vrgrid.grid.quantise import quantise_variance_cm2
from vrgrid.grid.schedule import load
from vrgrid.run.engine import MapEngine

needs_cuda = pytest.mark.skipif(not dev.cuda_available(),
                                reason="no working CUDA device")


def _frames(present_for=3, total=12, seed=0):
    import test_engine as te  # a failed import must fail, not skip
    rng = np.random.default_rng(seed)
    return [f for f, _ in te._sequence(rng, present_for, total)]


def _engine(device, **kw):
    return MapEngine(load("5/10/20/40"), max_points=40_000,
                     max_candidates=80_000, device=device, **kw)


# --- no card needed -------------------------------------------------------------

def test_resolve_device_rejects_unknown_names():
    assert dev.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        dev.resolve_device("gpu")


def test_cuda_without_a_card_fails_loudly(monkeypatch):
    monkeypatch.setattr(dev, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="no working CUDA device"):
        dev.resolve_device("cuda")


def test_cpu_engine_has_no_device_half():
    eng = _engine("cpu")
    assert eng.gpu is None and eng.device_bytes() is None


def test_variance_code_table_reproduces_the_codec_exactly():
    """The device codec is a binary search over these thresholds, so the table
    IS the codec on the card. Checked at every boundary and its neighbours,
    and across the whole dynamic range."""
    thr = K.variance_code_thresholds()

    def by_table(v):
        return np.array([np.searchsorted(-thr, -x, side="right") for x in v])

    rng = np.random.default_rng(0)
    v = np.exp(rng.uniform(np.log(1e-14), np.log(1e8), 20_000))
    bits = thr.view(np.uint64)
    edges = np.concatenate([thr, (bits + np.uint64(1)).view(np.float64),
                            (bits - np.uint64(1)).view(np.float64)])
    v = np.concatenate([v, edges])
    assert np.array_equal(by_table(v), quantise_variance_cm2(v).astype(np.int64))


# --- kernels against their references ---------------------------------------------

@needs_cuda
def test_floor_divide_is_numpys_not_cupys():
    import cupy as cp
    k = cp.ElementwiseKernel("float64 a, float64 b", "float64 q",
                             "q = np_floor_divide(a, b);", "t_floordiv",
                             preamble=K._PREAMBLE, options=K.OPTIONS)
    rng = np.random.default_rng(1)
    a = rng.uniform(-200, 200, 400_000)
    a[:100_000] = np.round(a[:100_000] / 0.05) * 0.05    # lattice boundaries
    a = np.concatenate([a, [1.0, 20.05, -0.05, 0.0, -0.0]])
    for b in (0.05, 0.1):
        assert np.array_equal(k(cp.asarray(a), b).get(), a // b)
    # and the reason the kernel exists at all
    assert (cp.asarray([1.0]) // 0.1).get()[0] != (np.array([1.0]) // 0.1)[0]


@needs_cuda
def test_device_bin_matches_bin_points():
    import cupy as cp
    rng = np.random.default_rng(2)
    eng = _engine("cuda")
    n = 40_000
    pts = np.column_stack([rng.uniform(-120, 120, n), rng.uniform(-120, 120, n),
                           rng.uniform(-3, 3, n), rng.uniform(0, 1, n)]).astype(np.float32)
    # world = vehicle + a small pose offset, so most points land in the windows
    world = np.column_stack([pts[:, 0].astype(np.float64) + 0.3175,
                             pts[:, 1].astype(np.float64) - 0.2125, pts[:, 2]])
    world[:5000, :2] = np.round(world[:5000, :2] / 0.05) * 0.05
    for buf in eng.buffers:           # a shifted window, not the origin one
        buf.x0 += 7
        buf.y0 -= 3
    d = {"pts": cp.asarray(pts.reshape(-1)), "ncols": 4, "dtype": np.float32,
         "world": cp.asarray(world.reshape(-1))}
    # headings chosen to break every symmetry the block bound has: facing
    # +x, the diagonal, backwards, and an angle with no special value
    for vehicle, yaw in [((0.3175, -0.2125), 0.0), ((0.35, 0.4), 0.7853981633974483),
                         ((-1.1, 2.05), 3.141592653589793), ((0.0, 0.0), -2.3)]:
        host = eng.bin(world[:, 0], world[:, 1], vehicle, yaw).copy()
        dev = eng.gpu.bin(d, n, eng.buffers, vehicle, yaw).get()
        assert np.array_equal(dev, host), f"yaw {yaw}"
        assert (host >= 0).sum() > n // 4 and (host < 0).sum() > 0


@needs_cuda
def test_device_projection_matches_range_image_project():
    ri = pytest.importorskip("vrgrid.perception.range_image")
    from vrgrid.perception import reflectivity
    rng = np.random.default_rng(3)
    n = 60_000
    az = rng.uniform(-np.pi, np.pi, n)
    el = np.radians(rng.uniform(-25, 3, n))
    r = rng.uniform(2, 80, n)
    pts = np.column_stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az),
                           r * np.sin(el), rng.uniform(0, 1, n)]).astype(np.float32)
    pts[:2000] = pts[2000:4000]                      # exact range ties
    labels = rng.integers(0, 260, n).astype(np.uint32)
    p = dev.DevicePerception(max_points=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img, inv = ri.project(pts)
        p.launch(pts, pts[:, :3].astype(np.float64), labels)
        p.finish()
    frame = dev.DeviceFrame(p, index=0)
    assert np.array_equal(frame.range_image, img, equal_nan=True)
    assert np.array_equal(frame.inverse_index, inv)
    rho, _ = reflectivity.scatter_to_points(reflectivity.normalise(img), inv)
    rho = np.concatenate([rho, np.zeros(n - len(rho), np.uint8)])
    assert np.array_equal(frame.reflectivity8, rho)


@needs_cuda
def test_a_stale_device_frame_refuses_to_read():
    rng = np.random.default_rng(4)
    pts = rng.uniform(-20, 20, (100, 4)).astype(np.float32)
    p = dev.DevicePerception(max_points=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.launch(pts, pts[:, :3].astype(np.float64), np.zeros(100, np.uint32))
        old = dev.DeviceFrame(p, index=0)
        p.launch(pts, pts[:, :3].astype(np.float64), np.zeros(100, np.uint32))
    with pytest.raises(RuntimeError, match="later frame"):
        old.semantic  # noqa: B018


# --- the engine -------------------------------------------------------------------

@needs_cuda
@pytest.mark.determinism
@pytest.mark.parametrize("ghost_removal", [True, False])
def test_device_engine_is_bit_identical_to_cpu(ghost_removal):
    cpu, gpu = _engine("cpu", ghost_removal=ghost_removal), \
        _engine("cuda", ghost_removal=ghost_removal)
    cleared = protected = 0
    for frame in _frames():
        c, g = cpu.step(frame), gpu.step(frame)
        assert c == g, f"counters diverge at frame {frame.index}"
        assert map_hash(cpu.handle.grid) == map_hash(gpu.handle.grid), \
            f"map diverges at frame {frame.index}"
        cleared += g.cleared
        protected += g.protected
    if ghost_removal:
        assert cleared > 0 and protected > 0, "scene did not exercise the cleanup"
    else:
        assert cleared == 0


@needs_cuda
def test_the_device_grid_is_the_map_and_the_host_copy_is_read_only():
    eng = _engine("cuda")
    frame = _frames(total=2)[0]
    eng.step(frame)
    assert int(eng.gpu.grid["obs_count"].sum()) == int(eng.handle.grid["obs_count"].sum()) > 0
    with pytest.raises(ValueError):
        eng.handle.grid["log_odds"][0] = 1


@needs_cuda
def test_device_step_allocates_almost_nothing_on_the_host():
    import tracemalloc

    frames = _frames(total=6)
    eng = _engine("cuda")
    for f in frames[:4]:
        eng.step(f)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        eng.step(frames[4])
        step = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert step < 2_000_000, f"device step allocates {step:,} B on the host"


@needs_cuda
def test_device_memory_is_declared():
    b = _engine("cuda").device_bytes()
    assert b["static"] > 0 and b["pool_reserved"] >= b["pool_used"] > 0


@needs_cuda
def test_device_rebase_drops_evidence_exactly_as_the_host_does():
    """`DeviceMap.track_datum` against `shift.track_datum`, heights spread
    across and past both band edges, for an in-range step and for a step
    wider than the band."""
    import cupy as cp
    from vrgrid.gpu.kernels import CEILING_NONE, Z_MAX_CM, Z_MIN_CM
    from vrgrid.gpu.shift import track_datum

    rng = np.random.default_rng(9)
    for ego in (-1.3, 3.7, -40.0):
        eng = _engine("cuda")
        n = eng.gpu.grid["ground_height"].size
        g = rng.integers(Z_MIN_CM, Z_MAX_CM + 1, n).astype(np.int16)
        c = np.where(rng.random(n) < 0.3, CEILING_NONE,
                     rng.integers(Z_MIN_CM, Z_MAX_CM + 1, n)).astype(np.int16)
        v = rng.integers(1, 256, n).astype(np.uint8)
        ref = {"ground_height": g.copy(), "ceiling_height": c.copy(), "height_variance": v.copy()}
        eng.gpu.grid["ground_height"].set(g)
        eng.gpu.grid["ceiling_height"].set(c)
        eng.gpu.grid["height_variance"].set(v)

        want_datum = track_datum(ref, 0.0, ego)
        got_datum = eng.gpu.track_datum(0.0, ego)
        assert got_datum == want_datum
        for k, want in ref.items():
            assert np.array_equal(cp.asnumpy(eng.gpu.grid[k]), want), (ego, k)
        assert (ref["height_variance"] == 0).any(), "the fixture must leave the band"


@needs_cuda
@pytest.mark.parametrize("band", [None, 1.0])
def test_device_bitfield_matches_host(band, monkeypatch):
    """The 7.1 bitfield on the card equals the host bitfield on every ring of
    both schedules. `band=1.0` widens the hypot guard band until most geometric
    cells are settled on the host, so the host-resolution path is exercised
    and must agree too."""
    import copy

    from vrgrid.gpu import traversability_device as TD
    from vrgrid.gpu.allocators import allocate
    from vrgrid.grid import traversability as T
    from vrgrid.grid.schedule import load_thresholds

    if band is not None:
        monkeypatch.setattr(TD, "BAND", band)
    th = copy.deepcopy(load_thresholds())
    rng = np.random.default_rng(4)
    for name in ("5/10/20/40", "5/10/50"):
        s = load(name)
        al = allocate(s, th, commit_pages=False)
        soa = {k: v.copy() for k, v in al.grid.items()}
        n = soa["ground_height"].size
        soa["ground_height"][:] = rng.integers(-350, 450, n)
        soa["ground_height"][: n // 2] = (np.arange(n // 2) % 37)      # smooth ramps too
        soa["ceiling_height"][:] = np.where(rng.random(n) < 0.3, 32767, rng.integers(-350, 450, n))
        soa["height_variance"][:] = rng.integers(0, 256, n)
        soa["obs_count"][:] = rng.integers(0, 6, n)
        soa["semantic_class"][:] = rng.integers(0, 256, n)
        rings = [(slice(r.offset, r.offset + r.side * r.side), r.side) for r in al.rings]
        host, dev_soa = soa, {k: v.copy() for k, v in soa.items()}
        T.update(host, s, rings, th)
        T.update(dev_soa, s, rings, th, device="cuda")
        assert np.array_equal(host["traversability"], dev_soa["traversability"]), name
        assert (host["traversability"] & 2).any(), "slope bit never set"
