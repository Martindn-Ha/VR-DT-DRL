"""Full-board fixed grid for Gomes-style DQN (world XZ meters, 112×112)."""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from spawn_geometry import load_fine_tune_config


# R1 platform (CurriculumManager)
_R1_CX = -0.646
_R1_CZ = 0.841
_R1_HALF_X = 0.143675
_R1_HALF_Z = 0.083675

# R2 platform bbox (CurriculumManagerRobot2)
_R2_X_MIN = -1.215 - 0.290
_R2_X_MAX = -1.215 - 0.010
_R2_Z_MIN = 0.755 + 0.010
_R2_Z_MAX = 0.755 + 0.180
_R2_ORIGIN_X = -1.215
_R2_ORIGIN_Z = 0.755
_R2_ARC_CX = 0.0241
_R2_ARC_CZ = -0.2965
_R2_ARC_R_INSET = 0.4771

# Spawn can sit on platform edges; yaml bounds may differ slightly from curriculum floats.
_BOUNDARY_EPS_M = 1e-6


def _clamp_board_xz(
    x: float, z: float, robot_id: int, cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    return (
        min(max(float(x), x_min), x_max),
        min(max(float(z), z_min), z_max),
    )


def default_board_dqn_config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "host_gpu_system" / "config" / "board_dqn_config.yaml"
    )


def load_board_dqn_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else default_board_dqn_config_path()
    return load_fine_tune_config(cfg_path)


def board_n(cfg: Optional[Dict[str, Any]] = None) -> int:
    return int((cfg or {}).get("grid", {}).get("n", 112))


def board_bounds(robot_id: int, cfg: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float, float]:
    """Return (x_min, x_max, z_min, z_max) in world frame."""
    plat = (cfg or {}).get("platform", {})
    key = str(int(robot_id))
    if key in plat:
        p = plat[key]
        return (
            float(p["x_min"]), float(p["x_max"]),
            float(p["z_min"]), float(p["z_max"]),
        )
    if robot_id == 2:
        return _R2_X_MIN, _R2_X_MAX, _R2_Z_MIN, _R2_Z_MAX
    return (
        _R1_CX - _R1_HALF_X, _R1_CX + _R1_HALF_X,
        _R1_CZ - _R1_HALF_Z, _R1_CZ + _R1_HALF_Z,
    )


def _r2_in_spawn_area(wx: float, wz: float) -> bool:
    lx = _R2_ORIGIN_X - wx
    lz = wz - _R2_ORIGIN_Z
    dist = math.hypot(lx - _R2_ARC_CX, lz - _R2_ARC_CZ)
    return dist <= _R2_ARC_R_INSET


def is_valid_board_world(x: float, z: float, robot_id: int,
                         cfg: Optional[Dict[str, Any]] = None) -> bool:
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    xf, zf = float(x), float(z)
    if xf < x_min - _BOUNDARY_EPS_M or xf > x_max + _BOUNDARY_EPS_M:
        return False
    if zf < z_min - _BOUNDARY_EPS_M or zf > z_max + _BOUNDARY_EPS_M:
        return False
    xc, zc = _clamp_board_xz(xf, zf, robot_id, cfg)
    if robot_id == 2:
        return _r2_in_spawn_area(xc, zc)
    return True


def board_cell_uv(cell: int, n: Optional[int] = None) -> Tuple[int, int]:
    n = n or 112
    row, col = divmod(int(cell), n)
    return row, col


def world_to_board_cell(
    x: float,
    z: float,
    robot_id: int,
    *,
    n: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Map world XZ to flat cell index; None if outside valid board."""
    n = n or board_n(cfg)
    if not is_valid_board_world(x, z, robot_id, cfg):
        return None
    xc, zc = _clamp_board_xz(x, z, robot_id, cfg)
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    cell_x = (xc - x_min) / max(x_max - x_min, 1e-9) * n
    cell_z = (z_max - zc) / max(z_max - z_min, 1e-9) * n
    col = int(cell_x)
    row = int(cell_z)
    col = min(max(col, 0), n - 1)
    row = min(max(row, 0), n - 1)
    return row * n + col


def board_cell_to_world(
    cell: int,
    robot_id: int,
    *,
    n: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    n = n or board_n(cfg)
    row, col = board_cell_uv(cell, n)
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    cell_w = (x_max - x_min) / float(n)
    cell_h = (z_max - z_min) / float(n)
    x = x_min + (col + 0.5) * cell_w
    z = z_max - (row + 0.5) * cell_h
    return float(x), float(z)


def board_cell_center_uv(cell: int, n: Optional[int] = None) -> Tuple[float, float]:
    """Cell center in grid index space (for projective warp)."""
    n = n or 112
    row, col = board_cell_uv(cell, n)
    return col + 0.5, row + 0.5


def calculate_board_reward(
    closest_dist_m: float,
    *,
    grasp_reward: float = 1.0,
    reward_dist_m: float = 0.02,
) -> float:
    """Gomes et al.: 1/(d+1) if d <= threshold else 0."""
    d = float(closest_dist_m)
    if d >= 9990.0 or d > float(reward_dist_m):
        return 0.0
    return float(grasp_reward) / (d + float(grasp_reward))


def board_shaping_q_targets(
    spawn_x: float,
    spawn_z: float,
    robot_id: int,
    n: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
):
    """Per-cell Q-shaping targets from spawn distance (Gomes session 2)."""
    import numpy as np
    n = n or board_n(cfg)
    reward_cfg = (cfg or {}).get("grid", {})
    grasp_reward = float(reward_cfg.get("grasp_reward", 1.0))
    reward_dist_m = float(reward_cfg.get("reward_dist_m", 0.02))
    targets = np.zeros(n * n, dtype=np.float32)
    sx, sz = float(spawn_x), float(spawn_z)
    for cell in range(n * n):
        wx, wz = board_cell_to_world(cell, robot_id, n=n, cfg=cfg)
        if not is_valid_board_world(wx, wz, robot_id, cfg):
            targets[cell] = 0.0
            continue
        d = math.hypot(wx - sx, wz - sz)
        targets[cell] = calculate_board_reward(
            d, grasp_reward=grasp_reward, reward_dist_m=reward_dist_m,
        )
    return targets


def epsilon_for_episode(episode: int, cfg: Dict[str, Any]) -> float:
    t = cfg.get("training", {})
    eps0 = float(t.get("epsilon_start", 0.9))
    eps1 = float(t.get("epsilon_end", 0.05))
    decay = max(1, int(t.get("epsilon_decay", 500)))
    frac = min(1.0, max(0.0, float(episode) / float(decay)))
    return eps0 + (eps1 - eps0) * frac


def invalid_cell_mask(robot_id: int, n: Optional[int] = None,
                      cfg: Optional[Dict[str, Any]] = None):
    """Boolean mask (n*n,) True where cell center is invalid (R2 arc)."""
    import numpy as np
    n = n or board_n(cfg)
    mask = np.zeros(n * n, dtype=bool)
    if robot_id != 2:
        return mask
    for cell in range(n * n):
        x, z = board_cell_to_world(cell, robot_id, n=n, cfg=cfg)
        if not is_valid_board_world(x, z, robot_id, cfg):
            mask[cell] = True
    return mask
