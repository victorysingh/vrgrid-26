"""Reference map and the synthetic sequence. Math §9.1. [Aakash]

M* is the thing every other number is measured against, so an error here is
invisible everywhere and fatal everywhere. The two properties that matter most
-- that it strips moving points, and that it is schedule-independent -- are
asserted directly rather than inferred from a metric coming out plausible.
"""

import numpy as np
import pytest
from vrgrid.eval.reference_map import (
    ReferenceMap,
    build,
    build_from_scans,
    is_moving,
    load,
)
from vrgrid.eval.synthetic import (
    KERB_HEIGHT_M,
    KERB_Y_M,
    MOVING_CAR,
    POTHOLE_DEPTH_M,
    POTHOLE_XY_M,
    read_sequence,
    scan,
    terrain_height_m,
    write_sequence,
)


@pytest.fixture(scope="module")
def sequence(tmp_path_factory):
    root = tmp_path_factory.mktemp("seq")
    write_sequence(root, "99", n_frames=4)
    return root


@pytest.fixture(scope="module")
def reference(sequence):
    return build_from_scans(read_sequence(sequence, "99"))


# --- the synthetic surface ---------------------------------------------------


def test_the_surface_is_a_surface_not_a_sample():
    """The same place must give the same height on every frame, or the
    reference map is averaging a moving world and means nothing."""
    x = np.array([1.0, 12.0, 40.0])
    y = np.array([0.0, 3.5, -8.0])
    assert np.array_equal(terrain_height_m(x, y), terrain_height_m(x, y))


def test_the_surface_has_the_features_the_metrics_are_meant_to_catch():
    # The step is across the kerb, not from the crown of the road: the road is
    # cambered, so z(centre) - z(edge) is 6 cm of drainage camber on top of the
    # 12 cm kerb. Measuring from the centre would report half a kerb and look
    # like a fusion bug later.
    inside = terrain_height_m(5.0, KERB_Y_M - 0.01)
    outside = terrain_height_m(5.0, KERB_Y_M + 0.01)
    assert outside - inside == pytest.approx(KERB_HEIGHT_M, abs=0.01)
    assert terrain_height_m(5.0, 0.0) - inside == pytest.approx(0.06, abs=0.01)

    pot = terrain_height_m(*POTHOLE_XY_M)
    beside = terrain_height_m(POTHOLE_XY_M[0] + 2.0, POTHOLE_XY_M[1])
    assert beside - pot == pytest.approx(POTHOLE_DEPTH_M, abs=0.02)

    flat = terrain_height_m(20.0, 0.0)
    uphill = terrain_height_m(50.0, 0.0)
    assert uphill > flat + 1.0, "the ramp is not a slope worth testing bit 1 on"


def test_the_scan_has_the_sampling_geometry_the_rings_are_derived_from():
    """§1.2's quadratic radial spacing is the entire justification for the
    ring schedule. If the synthetic scan does not reproduce it, every fill
    rate measured on it is meaningless."""
    pts, _, _ = scan(moving_car=False)
    r = np.hypot(pts[:, 0], pts[:, 1])

    near = np.count_nonzero((r > 5) & (r < 10))
    far = np.count_nonzero((r > 50) & (r < 55))
    assert near > 5 * far, "returns are not thinning out quadratically with range"

    # and the blind cone: nothing lands inside 3.74 m (§1.4 eq. 5)
    assert np.count_nonzero(r < 3.0) == 0


# --- §9.1: what M* is and is not ---------------------------------------------


def test_moving_points_are_stripped():
    """§9.1. Ids 250-259 from the raw `.label` file, nothing retrained."""
    assert is_moving(MOVING_CAR)
    assert is_moving(np.array([250, 255, 259])).all()
    assert not is_moving(np.array([9, 11, 249, 260])).any()

    pts = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 5.0]])
    labels = np.array([9, MOVING_CAR])
    ref = build_from_scans([(pts, labels, np.eye(4))])
    assert ref.observed.sum() == 1
    assert ref.height_cm[ref.observed][0] == pytest.approx(0.0)


def test_reference_is_on_the_base_lattice_and_matches_the_surface(reference):
    """The reference must recover the analytic surface it was sampled from.

    Checked on cells that actually hold a return rather than on a chosen
    coordinate: at 6 m the beams land ~15 cm apart radially (§1.2), so most
    5 cm cells there are empty and picking one by hand tests the sampling
    pattern rather than the rasteriser.
    """
    obs = np.argwhere(reference.observed)
    rng = np.random.default_rng(1)
    for r, c in obs[rng.choice(len(obs), 300, replace=False)]:
        x = (r + reference.i0 + 0.5) * reference.cell_m
        y = (c + reference.j0 + 0.5) * reference.cell_m
        assert reference.height_cm[r, c] / 100.0 == pytest.approx(
            terrain_height_m(x, y), abs=0.06)


def test_a_uniform_5cm_grid_over_this_scene_is_almost_entirely_empty():
    """§1.3's headline claim, reproduced by the synthetic sampler: a uniform
    5 cm raster of what the sensor actually returned is >95% empty. That is
    the argument for the whole representation, and it is worth having the
    scaffold demonstrate it rather than assert it from the doc."""
    pts, _, _ = scan(moving_car=False)
    r = np.hypot(pts[:, 0], pts[:, 1])
    near = pts[r < 25.0]
    cells = {(int(np.floor(px / 0.05)), int(np.floor(py / 0.05)))
             for px, py in near[:, :2]}
    footprint = (2 * 25.0 / 0.05) ** 2
    assert len(cells) / footprint < 0.05


def test_reference_knows_nothing_about_schedules(reference):
    """Schedule-independence is what makes cross-schedule comparison valid. It
    is a structural property, so it is asserted structurally: there is no
    schedule anywhere in the object."""
    assert reference.cell_m == 0.05
    assert not any("ring" in s or "schedule" in s for s in ReferenceMap.__slots__)


def test_block_stats_matches_a_brute_force_reduction(reference):
    """The summed-area tables are an optimisation, and an optimisation of the
    thing every metric is built on. Checked against the obvious slow version
    on random blocks -- if these disagree, every number in the harness is
    wrong in a way no other test would catch."""
    assert reference.within_var_cm2.max() > 0, (
        "no cell of the fixture holds two different returns, so the "
        "within-cell term below is not being checked")
    rng = np.random.default_rng(5)
    H, W = reference.shape
    for k in (1, 2, 4, 8, 10):
        i_lo = rng.integers(reference.i0, reference.i0 + H - k, 40)
        j_lo = rng.integers(reference.j0, reference.j0 + W - k, 40)
        n, mean, var = reference.block_stats(i_lo, j_lo, k)

        for idx in range(len(i_lo)):
            r0 = i_lo[idx] - reference.i0
            c0 = j_lo[idx] - reference.j0
            block_obs = reference.observed[r0:r0 + k, c0:c0 + k]
            block_h = reference.height_cm[r0:r0 + k, c0:c0 + k][block_obs]
            block_w = reference.within_var_cm2[r0:r0 + k, c0:c0 + k][block_obs]

            assert n[idx] == block_obs.sum()
            if block_h.size:
                assert mean[idx] == pytest.approx(block_h.mean(), abs=1e-6)
                # law of total variance, cells weighted equally
                assert var[idx] == pytest.approx(block_h.var() + block_w.mean(), abs=1e-6)
        _, _, between = reference.block_stats(i_lo, j_lo, k, within_cell=False)
        for idx in range(len(i_lo)):
            r0, c0 = i_lo[idx] - reference.i0, j_lo[idx] - reference.j0
            block_obs = reference.observed[r0:r0 + k, c0:c0 + k]
            block_h = reference.height_cm[r0:r0 + k, c0:c0 + k][block_obs]
            if block_h.size:
                assert between[idx] == pytest.approx(block_h.var(), abs=1e-6)


def test_block_stats_clips_at_the_edge_rather_than_wrapping(reference):
    """A ring extends far past where a short sequence drove, so most blocks
    fall outside the reference entirely. They must come back with n = 0 so the
    caller drops them -- not wrap, and not score as agreement."""
    n, _, _ = reference.block_stats(np.array([reference.i0 - 10_000]),
                                    np.array([reference.j0 - 10_000]), 8)
    assert n[0] == 0


def test_save_and_load_round_trip(reference, tmp_path):
    path = tmp_path / "ref.npz"
    reference.save(path)
    back = load(path)

    assert back.cell_m == reference.cell_m
    assert (back.i0, back.j0) == (reference.i0, reference.j0)
    assert np.array_equal(back.count, reference.count)
    assert np.allclose(back.height_cm, reference.height_cm, atol=1e-2)

    # and the loaded one answers block queries identically
    i = np.array([reference.i0 + 10, reference.i0 + 50])
    j = np.array([reference.j0 + 10, reference.j0 + 50])
    for a, b in zip(reference.block_stats(i, j, 4), back.block_stats(i, j, 4)):
        assert np.allclose(a, b, atol=1e-2)


def test_an_empty_sequence_is_an_error_not_an_empty_map():
    """A reference map with no static points would silently score every
    schedule as perfect, because every comparison would be dropped."""
    pts = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no static points"):
        build_from_scans([(pts, np.array([MOVING_CAR]), np.eye(4))])


def test_poses_are_applied_so_the_map_is_world_anchored(sequence):
    """Frames matter more here than anywhere else: the whole sequence has to
    land in one world frame, or M* is a smear of every pose at once."""
    ref = build_from_scans(read_sequence(sequence, "99"))
    extent_m = ref.shape[0] * ref.cell_m
    assert extent_m > 100.0, "the map did not grow as the vehicle drove"

    # the surface at a place only the later frames could see
    x = 3 * 2.0 + 20.0                        # 4 frames, 2 m apart
    i = int(np.floor(x / 0.05)) - ref.i0
    j = int(np.floor(0.0 / 0.05)) - ref.j0
    if 0 <= i < ref.shape[0] and 0 <= j < ref.shape[1] and ref.count[i, j]:
        assert ref.height_cm[i, j] / 100.0 == pytest.approx(
            terrain_height_m(x, 0.0), abs=0.05)


def test_one_definition_of_moving():
    """⚑ `is_moving` was written out three times: `perception/semantics.py`,
    `grid/transient.py` and here. `perception/loader.py` states the id range a
    fourth time as a bare constant. JP found them while wiring the front end.

    They agree today, which is the only reason nothing has failed. They are
    also the predicate that decides which returns never enter the persistent
    map (§3.6) AND the predicate the reference map removes by (§9.1) -- so if
    two of them ever drift, the ghost-removal metric is scored against a
    reference that still contains the ghosts, and the number moves in the
    direction that looks like success.

    This file's copy now imports the transient layer's. The perception copies
    are a cross-directory change and stay pinned here rather than edited.
    """
    import numpy as np
    from vrgrid.eval import reference_map
    from vrgrid.grid import transient

    ids = np.arange(0, 400, dtype=np.uint32)
    core = transient.is_moving(ids)

    assert reference_map.is_moving is transient.is_moving, (
        "eval re-declared is_moving instead of importing it"
    )
    assert ids[core].tolist() == list(range(250, 260))

    semantics = pytest.importorskip("vrgrid.perception.semantics")
    assert np.array_equal(semantics.is_moving(ids), core), (
        "perception/semantics.py and grid/transient.py disagree about which "
        "raw label ids are moving -- consolidate before this reaches a metric"
    )

    loader = pytest.importorskip("vrgrid.perception.loader")
    assert list(loader.MOVING_LABEL_IDS) == list(range(250, 260)), (
        "perception/loader.py's MOVING_LABEL_IDS has drifted from the predicate"
    )


# --- the real-data path, before the real data --------------------------------

def _point_the_loader_at(tmp_path, monkeypatch, seq):
    """`loader.DATA_ROOT` is resolved from the environment at IMPORT time, so
    setting the variable inside a test does nothing. Patch the attributes."""
    from vrgrid.perception import loader, transforms

    monkeypatch.setattr(loader, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(loader, "GT_POSES_DIR", tmp_path / "poses")
    monkeypatch.setattr(loader, "VELODYNE_DIR", tmp_path / "sequences")
    monkeypatch.setattr(loader, "LABELS_DIR", tmp_path / "sequences")
    transforms._TR_CACHE.pop(seq, None)


def test_build_reads_a_sequence_through_the_real_loader(tmp_path, monkeypatch):
    """⚑ `reference_map.build()` had never been run, by anything.

    It is the first step of every real-data number in this project -- no M*,
    no metrics, no plan regret -- and it raised `ValueError: too many values
    to unpack` on its own first line. Nothing caught it because every test and
    every script called `build_from_scans` directly, and the one caller that
    would have exercised it needs 40 GB that has not landed.

    It does not need 40 GB. `eval/synthetic.write_sequence` writes the layout
    `perception.loader` reads, so the whole chain -- loader, calib, poses,
    sensor -> vehicle -> world -- runs here on a scene whose true surface is
    known analytically. That is a stronger check than the real data would give
    on its own: the heights can be asserted, not just eyeballed.
    """
    _point_the_loader_at(tmp_path, monkeypatch, "99")
    write_sequence(tmp_path, "99", n_frames=4)

    ref = build("99", cell_m=0.05)
    assert ref.count.sum() > 10_000

    # Every observed cell must sit on the analytic surface. This is what the
    # frame bugs would have broken silently: a whole map, in the wrong cells.
    obs = np.argwhere(ref.count > 0)
    i, j = obs[:, 0] + ref.i0, obs[:, 1] + ref.j0
    x = (i + 0.5) * ref.cell_m
    y = (j + 0.5) * ref.cell_m
    got = ref.height_cm[obs[:, 0], obs[:, 1]] / 100.0
    want = terrain_height_m(x, y, 0)
    # 5 cm cells over a crowned road, and the class-boundary cells straddle a
    # 12 cm kerb, so a handful are legitimately off. The median is the claim.
    assert np.median(np.abs(got - want)) < 0.02


def test_build_refuses_a_camera_convention_pose(tmp_path, monkeypatch):
    """The guard that makes the day-costing failure loud.

    `build()` used to hand `poses[i]` -- a Camera-0 -> World_cam row -- to a
    function expecting vehicle -> world. The result is not an error, it is a
    complete reference map rotated 90 degrees, and every metric measured
    against it is meaningless in a way no downstream check can see.

    ⚑ Eight frames, not two, and the number is the point. This test was
      written with two and did NOT raise -- which is how `FrameGuard` came to
      exist. A KITTI `poses.txt` begins at the identity, so on the first frame
      the wrong composition and the right one agree to well inside the guard's
      2 m tolerance; they only separate once the vehicle has driven. The guard
      now takes a second look after `GUARD_BASELINE_M` (10 m), and at this
      sequence's 2 m per frame that needs six. A guard that only ever saw
      frame 0 would have passed every real sequence too.
    """
    from vrgrid.eval.harness import FrameConventionError
    from vrgrid.perception import transforms

    _point_the_loader_at(tmp_path, monkeypatch, "99")
    write_sequence(tmp_path, "99", n_frames=8)

    # Skip the axis permutation: exactly the composition the old code did.
    monkeypatch.setattr(transforms, "vehicle_to_world",
                        lambda pose, sequence="00", tr=None:
                        np.vstack([np.asarray(pose).reshape(3, 4),
                                   [0.0, 0.0, 0.0, 1.0]]))
    with pytest.raises(FrameConventionError, match="not z-up"):
        build("99", cell_m=0.05)


def test_build_can_stop_early(tmp_path, monkeypatch):
    """`max_frames` is what makes a first pass over a real sequence a minute
    rather than an hour -- the difference between checking the frame
    convention before a full build and after one."""
    _point_the_loader_at(tmp_path, monkeypatch, "99")
    write_sequence(tmp_path, "99", n_frames=6)

    assert build("99", max_frames=2).count.sum() < build("99").count.sum()


# --- M* restricted to what each ring received (§9.2) -------------------------


def _sparse_from_dense(reference):
    from vrgrid.eval.reference_map import SparseReference

    obs = reference.observed
    r, c = np.nonzero(obs)
    return SparseReference(reference.cell_m, r + reference.i0, c + reference.j0,
                           reference.height_cm[obs], reference.within_var_cm2[obs],
                           reference.count[obs])


def test_sparse_block_stats_matches_the_dense_reference(reference):
    """`SparseReference` answers the question `ReferenceMap.block_stats`
    answers, over the same observed cells, including blocks that straddle the
    reference's edge and blocks wholly outside it. Every §9.2 metric goes
    through this contract, so the per-ring reference is only a fair comparison
    if the two agree on identical data."""
    sparse = _sparse_from_dense(reference)
    rng = np.random.default_rng(6)
    H, W = reference.shape
    for k in (1, 2, 4, 8, 10):
        i_lo = rng.integers(reference.i0 - 3 * k, reference.i0 + H + 3 * k, 400) // k * k
        j_lo = rng.integers(reference.j0 - 3 * k, reference.j0 + W + 3 * k, 400) // k * k
        n_d, m_d, v_d = reference.block_stats(i_lo, j_lo, k)
        n_s, m_s, v_s = sparse.block_stats(i_lo, j_lo, k)
        assert np.array_equal(n_s, n_d), k
        assert np.allclose(sparse.block_stats(i_lo, j_lo, k, within_cell=False)[2],
                           reference.block_stats(i_lo, j_lo, k, within_cell=False)[2],
                           atol=1e-4)
        assert np.array_equal(sparse.block_returns(i_lo, j_lo, k),
                              reference.block_returns(i_lo, j_lo, k)), k
        seen = n_d > 0
        assert seen.any() and (~seen).any(), "the fixture must hit both cases"
        assert np.allclose(m_s[seen], m_d[seen], atol=1e-6)
        assert np.allclose(v_s[seen], v_d[seen], atol=1e-4)


def test_ring_observations_rebuild_m_star_exactly(sequence):
    """Every static return credited to one ring must rebuild M* itself: same
    observed cells at the same lattice indices, same heights, same block
    statistics. This is the test that catches a broken key packing -- a check
    that compares two per-ring references against each other cannot, because
    both go through the same packing."""
    from vrgrid.eval.reference_map import RingObservations

    scans = list(read_sequence(sequence, "99"))
    m_star = build_from_scans(scans)
    obs = RingObservations(1)
    for pts, labels, pose in scans:
        T = np.asarray(pose, dtype=np.float64)
        keep = ~is_moving(labels)
        world = np.asarray(pts, dtype=np.float64)[keep] @ T[:3, :3].T + T[:3, 3]
        obs.add(np.zeros(len(world), np.int64), world)
    got = obs.for_ring(0)

    want = _sparse_from_dense(m_star)
    assert got.n_cells == want.n_cells
    order_g = np.lexsort((got.j, got.i))
    order_w = np.lexsort((want.j, want.i))
    assert np.array_equal(got.i[order_g], want.i[order_w])
    assert np.array_equal(got.j[order_g], want.j[order_w])
    assert np.allclose(got.height_cm[order_g], want.height_cm[order_w])
    assert np.allclose(got.within_var_cm2[order_g], want.within_var_cm2[order_w], atol=1e-6)
    assert np.array_equal(got.count[order_g], want.count[order_w])

    k = 8
    i_lo = (want.i // k * k)[::97]
    j_lo = (want.j // k * k)[::97]
    n_g, m_g, v_g = got.block_stats(i_lo, j_lo, k)
    n_d, m_d, v_d = m_star.block_stats(i_lo, j_lo, k)
    assert np.array_equal(n_g, n_d) and (n_d > 0).all()
    assert np.allclose(m_g, m_d, atol=1e-6) and np.allclose(v_g, v_d, atol=1e-4)


def test_the_packed_key_round_trips_negative_and_far_indices():
    from vrgrid.eval.reference_map import _pack, _unpack

    i = np.array([0, -1, 1, -(1 << 29), (1 << 29), 123456, -987654])
    j = np.array([0, 1, -1, (1 << 30), -(1 << 30), -5, 7])
    ui, uj = _unpack(_pack(i, j))
    assert np.array_equal(ui, i) and np.array_equal(uj, j)
    order = np.argsort(_pack(i, j))
    assert np.all(np.diff(i[order]) >= 0), "keys must sort by i first"


def test_an_empty_ring_reference_scores_nothing():
    from vrgrid.eval.reference_map import RingObservations

    obs = RingObservations(3)
    obs.add(np.array([0, 0]), np.array([[1.0, 2.0, 0.1], [1.01, 2.0, 0.3]]))
    n, _, _ = obs.for_ring(2).block_stats(np.array([0, 20]), np.array([40, 40]), 2)
    assert list(n) == [0, 0]
    n, mean, _ = obs.for_ring(0).block_stats(np.array([20]), np.array([40]), 1)
    assert n[0] == 1 and mean[0] == pytest.approx(20.0)   # both returns, one cell


def test_ring_observations_credit_each_return_to_the_ring_that_binned_it(sequence):
    """The per-ring reference is only meaningful if a return is credited to
    the ring whose cell integrated it. One frame through the harness, then the
    map's own `bin_points` on the same ground returns: the slot each lands in
    names its ring by buffer offset, and the per-ring references must hold
    exactly those returns -- same 5 cm cells, same mean heights."""
    from vrgrid.eval.harness import build_gridmap, run_sequence
    from vrgrid.eval.reference_map import RingObservations
    from vrgrid.grid.lattice import bin_points, new_bin_scratch
    from vrgrid.grid.schedule import load as load_schedule
    from vrgrid.grid.transient import separate

    frame = next(iter(read_sequence(sequence, "99")))
    pts, labels, pose = frame[0], frame[1], frame[-1]
    ground = np.ones(len(pts), bool)
    sched = load_schedule("5/10/20/40")
    gm = build_gridmap(sched)
    observed = RingObservations(len(sched.rings))
    run_sequence(gm, [(pts, labels, ground, pose)], observed=observed)

    T = np.asarray(pose, dtype=np.float64)
    world = np.asarray(pts, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
    static, _ = separate(labels)
    w = world[static]
    slots = bin_points(w[:, 0], w[:, 1], sched, gm.buffers,
                       np.zeros(len(w), np.int64), new_bin_scratch(len(w), sched),
                       gm.speed_ms, gm.vehicle_xy_m, gm.vehicle_yaw_rad)
    for L, b in enumerate(gm.buffers):
        mine = w[(slots >= b.offset) & (slots < b.offset + b.side * b.side)]
        want = RingObservations(1)
        want.add(np.zeros(len(mine), np.int64), mine)
        got, exp = observed.for_ring(L), want.for_ring(0)
        assert got.n_cells == exp.n_cells, L
        assert np.array_equal(got.i, exp.i) and np.array_equal(got.j, exp.j), L
        assert np.allclose(got.height_cm, exp.height_cm), L
    assert sum(observed.for_ring(L).n_cells for L in range(4)) > 1000


def test_ring_zero_has_a_rho(sequence):
    """Until the within-cell term, ring 0 -- the finest ring, the one the
    foveation argument is about -- had RMSE and no rho on any sequence,
    because its one-cell footprint could never pass a `cells > 1` guard. It is
    now scored wherever its cell holds more than one return. On this 4-frame
    scene that is about a third of the ring's scored cells; on real sequences
    at 40 frames it is 93-97% (`known-limitations.md` §2b)."""
    from vrgrid.eval import metrics
    from vrgrid.eval.harness import build_gridmap, run_sequence
    from vrgrid.grid.schedule import load as load_schedule

    scans = list(read_sequence(sequence, "99"))
    m_star = build_from_scans(scans)
    gm = build_gridmap(load_schedule("5/10/20/40"))
    run_sequence(gm, [(p, lab, np.ones(len(p), bool), T) for p, lab, T in scans])

    rho = metrics.coarsening_ratio_per_ring(gm, m_star)
    rmse_cells = metrics._compared(gm, m_star, 0)[0].size
    assert np.isfinite(rho[0]["rho"]) and rho[0]["n"] > 0
    assert rho[0]["n"] > 0.2 * rmse_cells, (rho[0]["n"], rmse_cells)
    assert rho[0]["spread_cm"] > 0


def test_a_banded_reference_leaves_out_exactly_what_the_map_cannot_hold(sequence):
    """`band=True` drops the ground returns outside the 8 m band around the
    vehicle -- the ones `scatter` gives no height weight -- counts them, and
    changes nothing else."""
    scans = list(read_sequence(sequence, "99"))
    plain = build_from_scans(scans)
    banded = build_from_scans(scans, band=True)
    assert banded.out_of_band_returns == 0          # the synthetic scene fits
    assert np.array_equal(banded.count, plain.count)

    pts, labels, pose = scans[0]
    deep = np.vstack([pts, [[30.0, 0.0, -25.0], [31.0, 0.0, -25.0]]])
    labels2 = np.concatenate([labels, np.array([40, 40], labels.dtype)])
    banded = build_from_scans([(deep, labels2, pose)] + scans[1:], band=True)
    assert banded.out_of_band_returns == 2
    assert banded.observed.sum() == plain.observed.sum()
    assert "outside the map's band" in repr(banded)


def test_class_is_the_majority_of_all_static_returns_height_is_ground_only():
    """One 5 cm cell: a road return on the ground, two vegetation returns
    above it, one moving car. M*'s height must come from the ground return
    alone, and its class from the majority of the STATIC returns --
    vegetation -- because that is what the map's class layer is fused from.
    Taking the class from the ground return made §7.1 bit 4 disagree at every
    road edge with something over it (seq 09's regret step)."""
    pts = np.array([[1.01, 2.01, -1.70], [1.02, 2.02, -0.40],
                    [1.03, 2.03, -0.10], [1.04, 2.04, -1.00]])
    labels = np.array([40, 70, 70, 252], dtype=np.uint32)     # road, veg, veg, moving car
    ground = np.array([True, False, False, False])
    ref = build_from_scans([(pts, labels, ground, np.eye(4))])
    i, j = int(np.floor(1.01 / ref.cell_m)), int(np.floor(2.01 / ref.cell_m))
    r, c = i - ref.i0, j - ref.j0
    assert ref.count[r, c] == 1
    assert ref.height_cm[r, c] == pytest.approx(-170.0)
    assert ref.class_id[r, c] == 70
