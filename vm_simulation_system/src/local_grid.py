"""Local table grid for hybrid locator + cell-picker grasp (world XZ meters)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from failure_taxonomy import FAR_MISS_DIST, NEAR_MISS_DIST, classify_outcome
from spawn_geometry import load_fine_tune_config


def default_grid_train_config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "host_gpu_system" / "config" / "grid_train_config.yaml"
    )


def load_grid_train_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else default_grid_train_config_path()
    return load_fine_tune_config(cfg_path)


def grid_params(cfg: Optional[Dict[str, Any]] = None) -> Tuple[int, float]:
    g = (cfg or {}).get("grid", {})
    return int(g.get("n", 5)), float(g.get("window_m", 0.04))


def teacher_grasp_xy(pose6: List[float]) -> Tuple[float, float]:
    return float(pose6[0]), float(pose6[2])


def world_to_cell(
    x: float,
    z: float,
    center_x: float,
    center_z: float,
    *,
    n: int = 5,
    window_m: float = 0.04,
) -> Optional[int]:
    half = window_m / 2.0
    cell_size = window_m / float(n)
    dx = float(x) - float(center_x)
    dz = float(z) - float(center_z)
    if abs(dx) > half or abs(dz) > half:
        return None
    col = int((dx + half) / cell_size)
    row = int((half - dz) / cell_size)
    col = min(max(col, 0), n - 1)
    row = min(max(row, 0), n - 1)
    return row * n + col


def cell_to_world(
    cell: int,
    center_x: float,
    center_z: float,
    *,
    n: int = 5,
    window_m: float = 0.04,
) -> Tuple[float, float]:
    half = window_m / 2.0
    cell_size = window_m / float(n)
    row, col = divmod(int(cell), n)
    x = float(center_x) - half + (col + 0.5) * cell_size
    z = float(center_z) + half - (row + 0.5) * cell_size
    return x, z


def calculate_local_bbox_center_reward(
    pick_x: float,
    pick_z: float,
    true_x: float,
    true_z: float,
    reward_cfg: Optional[Dict[str, Any]] = None,
    *,
    cell_action: Optional[int] = None,
    teacher_cell: Optional[int] = None,
) -> float:
    """Center-focused reward: pick world XZ vs sim GT block center (not lift)."""
    cfg = reward_cfg or {}
    dist = math.hypot(float(pick_x) - float(true_x), float(pick_z) - float(true_z))
    success_m = float(cfg.get("center_success_m", 0.002))
    fail_m = float(cfg.get("center_fail_m", 0.010))
    if fail_m <= success_m:
        fail_m = success_m + 1e-6
    if dist <= success_m:
        r = 1.0
    elif dist >= fail_m:
        r = 0.0
    else:
        r = 1.0 - (dist - success_m) / (fail_m - success_m)
    teacher_bonus = float(cfg.get("teacher_cell_bonus", 0.0))
    if (
        cell_action is not None
        and teacher_cell is not None
        and int(cell_action) == int(teacher_cell)
        and teacher_bonus > 0
    ):
        r = min(1.0, r + teacher_bonus)
    return float(r)


def calculate_grid_reward(
    success: bool,
    lifted_m: Optional[float],
    closest_dist_m: float,
    reward_cfg: Dict[str, Any],
    *,
    grasp_mode: str = "grid_explore",
    object_found: int = 1,
    closest_dist_xz_m: Optional[float] = None,
    cell_action: Optional[int] = None,
    teacher_cell: Optional[int] = None,
) -> float:
    """Pickup-oriented RL reward (XZ distance bands); logs still use failure_taxonomy."""
    lift_bonus = float(reward_cfg.get("lift_bonus", 1.0))
    miss_penalty = float(reward_cfg.get("miss_penalty", 0.0))
    dist_scale = float(reward_cfg.get("distance_scale", 0.5))
    mid_scale = float(reward_cfg.get("mid_miss_scale", 0.25))
    contact_scale = float(reward_cfg.get("contact_partial_scale", 0.3))
    teacher_bonus = float(reward_cfg.get("teacher_cell_bonus", 0.1))

    if success:
        return lift_bonus

    if int(object_found) == 0:
        return miss_penalty

    dist = float(closest_dist_m)
    if closest_dist_xz_m is not None and float(closest_dist_xz_m) < 9990.0:
        dist = float(closest_dist_xz_m)

    if dist >= 9990.0 or dist > FAR_MISS_DIST or dist_scale <= 0:
        r = miss_penalty
    elif dist <= NEAR_MISS_DIST:
        ramp = max(0.0, 1.0 - dist / NEAR_MISS_DIST)
        r = max(miss_penalty, dist_scale * ramp)
    elif FAR_MISS_DIST > NEAR_MISS_DIST:
        band = FAR_MISS_DIST - NEAR_MISS_DIST
        t = max(0.0, min(1.0, (FAR_MISS_DIST - dist) / band))
        r = max(miss_penalty, dist_scale * mid_scale * t)
    else:
        r = miss_penalty

    outcome = classify_outcome(
        success=False,
        grasp_mode=grasp_mode,
        object_found=object_found,
        closest_dist_m=closest_dist_m,
        lifted_m=lifted_m,
    )
    if outcome == "drop_or_push" and r > miss_penalty:
        r = miss_penalty + (r - miss_penalty) * contact_scale

    if (
        cell_action is not None
        and teacher_cell is not None
        and int(cell_action) == int(teacher_cell)
        and teacher_bonus > 0
    ):
        r += teacher_bonus

    return r


def epsilon_for_episode(episode: int, cfg: Dict[str, Any]) -> float:
    t = cfg.get("training", {})
    eps0 = float(t.get("epsilon_start", 0.3))
    eps1 = float(t.get("epsilon_end", 0.05))
    decay = max(1, int(t.get("epsilon_decay_episodes", 500)))
    frac = min(1.0, max(0.0, float(episode) / float(decay)))
    return eps0 + (eps1 - eps0) * frac
