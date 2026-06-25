"""Analytic grasp pose from object world position (shared teacher + geo-grasp inference)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# Fixed spawn / object height in Webots (meters).
DEFAULT_OBJECT_Y_M = 0.461

REACH_OFFSET = -0.095
SHIFT_OFFSET = 0.1
HEIGHT_OFFSET = 0.02
TEACHER_JITTER_M = 0.005

ROBOT_GEOMETRY: Dict[int, Dict[str, float]] = {
    1: {
        'robot_base_x': -0.685,
        'robot_base_z': 0.47235,
        'fallback_x': -0.685,
        'fallback_y': 0.44,
        'fallback_z': 0.55,
    },
    2: {
        'robot_base_x': -1.226,
        'robot_base_z': 0.47235,
        'fallback_x': -1.365,
        'fallback_y': 0.44,
        'fallback_z': 0.905,
    },
}


def fallback_grasp_pose(robot_id: int) -> List[float]:
    g = ROBOT_GEOMETRY[int(robot_id)]
    return [g['fallback_x'], g['fallback_y'], g['fallback_z'], 3.14, 0.0, 0.0]


def compute_grasp_pose_from_object_world(
    obj_x: float,
    obj_y: float,
    obj_z: float,
    robot_id: int,
    *,
    add_jitter: bool = False,
) -> List[float]:
    """
    Compute 6-DOF world grasp pose from object center (teacher + locator paths).

    Returns [x, y, z, rx, ry, rz] with rx=3.14, ry=0, rz=yaw toward object.
    """
    rid = int(robot_id)
    g = ROBOT_GEOMETRY[rid]
    base_x = g['robot_base_x']
    base_z = g['robot_base_z']

    dx = float(obj_x) - base_x
    dz = float(obj_z) - base_z
    dist_to_obj = math.sqrt(dx * dx + dz * dz)
    angle_to_obj = math.atan2(dz, dx)

    final_dist = dist_to_obj + REACH_OFFSET
    target_x = (
        base_x
        + (final_dist * math.cos(angle_to_obj))
        - (SHIFT_OFFSET * math.sin(angle_to_obj))
    )
    target_z = (
        base_z
        + (final_dist * math.sin(angle_to_obj))
        + (SHIFT_OFFSET * math.cos(angle_to_obj))
    )
    target_y = float(obj_y) + HEIGHT_OFFSET

    if add_jitter:
        target_x += np.random.uniform(-TEACHER_JITTER_M, TEACHER_JITTER_M)
        target_z += np.random.uniform(-TEACHER_JITTER_M, TEACHER_JITTER_M)

    yaw = angle_to_obj
    return [float(target_x), float(target_y), float(target_z), 3.14, 0.0, float(yaw)]


def local_xz_to_world(
    local_x: float,
    local_z: float,
    robot_id: int,
    *,
    obj_y: float = DEFAULT_OBJECT_Y_M,
) -> Tuple[float, float, float]:
    """Convert robot-local aux_position (x,z) to world object coordinates."""
    g = ROBOT_GEOMETRY[int(robot_id)]
    world_x = float(local_x) + g['robot_base_x']
    world_z = float(local_z) + g['robot_base_z']
    return world_x, float(obj_y), world_z
