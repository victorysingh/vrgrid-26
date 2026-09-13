"""Coordinate transforms. [JP — Day 0, hour 0-2 with the whole team]

EVERY transform in this file is also written down in `docs/frames.md`, with
origin, axes, handedness and units. Frame confusion is the most common silent
bug in this project: the map looks entirely plausible and slowly rotates. It
costs three days if found on Day 4 and minutes if found now.

Three frames
------------
  Sensor  (Velodyne HDL-64E):   x forward, y left, z up.  Origin at the laser,
                                ~1.73 m above the ground.
  Vehicle:                      x forward, y left, z up.  Same axes as the
                                sensor; origin dropped to the ground plane
                                directly below the laser (z = 0 at the road).
  World:                        x forward, y left, z up.  Coincides with the
                                Vehicle frame of the FIRST frame of the
                                sequence and never moves after that.

KITTI odometry gives us two things and neither is Velodyne -> Vehicle directly:

  * `calib.txt` line `Tr:`  ->  Velodyne -> Camera-0    (4x4, measured extrinsic)
  * `poses.txt`  line i     ->  Camera-0 -> World_cam   (3x4, GT trajectory,
                                row-major [R | t], frame 0 is identity)

`World_cam` is in the camera convention (x right, y down, z forward). We rotate
it once, with a constant axis permutation, into the z-up World frame above so
that every downstream consumer (grid, reference map, dashboard) sees x-forward,
y-left, z-up.

The static-wall test (`tests/test_static_wall.py`, Gate Item 1) is the check
that this file is right: drive 100 frames past a building facade, transform
every scan into World, and confirm the wall's fitted plane does not rotate or
translate.
"""

import numpy as np

# --- Sensor -> Vehicle -------------------------------------------------------
#
# KITTI's odometry benchmark publishes no Velodyne -> Vehicle extrinsic. By the
# universal KITTI convention the Velodyne axes are already aligned with the
# vehicle axes (x forward, y left, z up), so the rotation is identity and the
# only difference is height: the laser sits 1.73 m above the road (KITTI spec
# sheet / HDL-64E mounting documentation, used throughout the literature).
#
# We put the Vehicle origin on the ground, so a point at the sensor origin is at
# vehicle-frame (0, 0, 1.73): the translation is +1.73 m in z.
SENSOR_HEIGHT_M = 1.73

T_S_V = np.eye(4, dtype=np.float64)
T_S_V[2, 3] = SENSOR_HEIGHT_M


def sensor_to_vehicle() -> np.ndarray:
    """Constant transform: Sensor (Velodyne) -> Vehicle frame.

    Identity rotation (KITTI convention: Velodyne axes align with the vehicle)
    and +1.73 m in z so the Vehicle origin sits on the road surface.

    Returns:
        (4, 4) float64 homogeneous matrix.
    """
    return T_S_V.copy()


# --- Camera-0 -> Vehicle-convention rotation --------------------------------
#
# Camera-0 frame:  x right, y down, z forward.
# Vehicle frame:   x forward, y left, z up.
#
#   vehicle_x =  camera_z
#   vehicle_y = -camera_x
#   vehicle_z = -camera_y
R_CAM0_TO_VEH = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# Vehicle -> Velodyne: inverse of sensor_to_vehicle (drop the ground origin
# back up to the laser). Used inside vehicle_to_world so the composed transform
# can be applied to points already lifted into the Vehicle frame.
_T_V_S = np.eye(4, dtype=np.float64)
_T_V_S[2, 3] = -SENSOR_HEIGHT_M


_TR_CACHE: dict[str, np.ndarray] = {}


def velo_to_cam0(sequence: str = "00") -> np.ndarray:
    """Velodyne -> Camera-0 (the `Tr:` line of `sequences/<seq>/calib.txt`).

    Cached per sequence. The matrix is near-identical across sequences 00/07/08
    (same rig), but we read the real one so a re-calibrated sequence is handled.

    Returns:
        (4, 4) float64 homogeneous matrix.
    """
    if sequence not in _TR_CACHE:
        from .loader import load_calib

        _TR_CACHE[sequence] = load_calib(sequence)["Tr_velo_to_cam0"]
    return _TR_CACHE[sequence].copy()


def _pose_to_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(3, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = pose
    return T


def vehicle_to_world(pose: np.ndarray, sequence: str = "00",
                     tr: np.ndarray | None = None) -> np.ndarray:
    """4x4 Vehicle -> World transform for one frame.

    Composition, applied right to left to a Vehicle-frame point:

        Vehicle -> Velodyne          (_T_V_S, undo the 1.73 m ground drop)
        Velodyne -> Camera-0         (Tr, from calib.txt)
        Camera-0 -> World_cam        (pose, from poses.txt)
        World_cam -> World (z-up)    (R_CAM0_TO_VEH, constant axis permutation)

    Args:
        pose: (3, 4) or (12,) row-major [R | t] from `poses.txt`, Camera-0 -> World_cam.
        sequence: which calib.txt to read `Tr` from. Default "00".
        tr: (4, 4) Velodyne -> Camera-0, supplied instead of reading it. For a
            sequence that does not live under the module-level DATA_ROOT --
            a test fixture, `eval/synthetic.py`. The composition below is the
            part that must not be duplicated; where `Tr` came from is not.

    Returns:
        (4, 4) float64 homogeneous matrix: point_Vehicle -> point_World.
    """
    T_pose = _pose_to_4x4(pose)
    T_tr = velo_to_cam0(sequence) if tr is None else np.asarray(tr, dtype=np.float64)

    R_flip = np.eye(4, dtype=np.float64)
    R_flip[:3, :3] = R_CAM0_TO_VEH

    return R_flip @ T_pose @ T_tr @ _T_V_S


def sensor_to_world(pose: np.ndarray, sequence: str = "00") -> np.ndarray:
    """4x4 Sensor -> World transform for one frame (convenience).

    Equivalent to ``vehicle_to_world(pose) @ sensor_to_vehicle()`` and, because
    the 1.73 m ground drop cancels, to the textbook KITTI chain
    ``R_flip @ pose @ Tr`` acting on raw Velodyne points.
    """
    return vehicle_to_world(pose, sequence) @ T_S_V


def new_transform_scratch(max_points: int) -> dict:
    """Buffers for an allocation-free `transform_points` on the frame loop.

    `h` is the homogeneous (N, 4) array with its ones column set ONCE, here;
    only the xyz columns are ever written. `out` is Fortran-ordered so that its
    leading `n` columns are contiguous and the matmul writes straight into them.
    Size it from `configs/thresholds.yaml: scatter.max_points_per_frame`, the
    same bound every other per-point scratch uses.
    """
    return {"h": np.ones((max_points, 4), dtype=np.float64),
            "out": np.empty((4, max_points), dtype=np.float64, order="F")}


def transform_points(points: np.ndarray, T: np.ndarray, scratch: dict | None = None
                     ) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to 3D points.

    Args:
        points: (N, 3) or (N, 4) array. A 4th column (intensity) is ignored and
            not returned.
        T: (4, 4) transform matrix.
        scratch: optional buffers from `new_transform_scratch`. Omit it and
            nothing changes -- a fresh array is returned, as always.

    Returns:
        (N, 3) float64 transformed points.

    [!] With `scratch`, the result is a VIEW INTO THE SCRATCH and is overwritten
      by the next call that uses the same scratch. Only pass one where the result
      is finished with before the next call -- the frame loop, not anything that
      keeps frames.

      Why it exists: without it this function allocates ~10.9 MB per call (the
      float64 copy, a ones column, the hstack and the matmul result) on ~123,000
      points, and those allocations produced the largest per-stage p99 tail in
      the whole frame -- p50 ~3 ms with 25-33 ms stalls when the allocator had to
      fault fresh pages in. `reports/r-b-p99-tail-investigation.md` section 6.

      It is BIT-IDENTICAL to the allocating path, and deliberately so: it makes
      the SAME (4, 4) @ (4, N) matmul call on the SAME F-contiguous view of an
      (N, 4) homogeneous array, and changes only where the result is written. A
      hand-written element-wise version was tried and changed the output on
      200 of 200 frames; different call shapes can sum in a different order.
      Pinned in `tests/test_transform_scratch.py`.

      A scan larger than the scratch falls back to the allocating path rather
      than raising: correct, merely not allocation-free for that frame.
    """
    if scratch is not None:
        pts_in = np.asarray(points)
        if pts_in.ndim != 2 or pts_in.shape[1] not in (3, 4):
            raise ValueError(f"points must be (N, 3) or (N, 4), got {pts_in.shape}")
        n = pts_in.shape[0]
        h, out = scratch["h"], scratch["out"]
        if n <= h.shape[0]:
            h[:n, :3] = pts_in[:, :3]
            np.matmul(T, h[:n].T, out=out[:, :n])
            return out[:, :n].T[:, :3]

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[1] not in (3, 4):
        raise ValueError(f"points must be (N, 3) or (N, 4), got {pts.shape}")

    xyz = pts[:, :3]
    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)])
    return (T @ pts_h.T).T[:, :3]
