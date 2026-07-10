#!/usr/bin/env python3
"""
Spawn geometry helpers for targeted BC fine-tuning and evaluation reports.

Quadrant 1-4 maps to Roman I-IV (0-90, 90-180, 180-270, 270-360 degrees robot frame),
matching analysis/spawn_spatial_report.py.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

QUADRANT_ANGLE_BINS = [(0, 90), (90, 180), (180, 270), (270, 360)]

# Inference-band radii from CurriculumManager.PHASE_CONFIG (phase index -> r_min, r_max)
PHASE_BAND_RADII = {
    0: (0.000, 0.000),
    1: (0.005, 0.015),
    2: (0.015, 0.035),
    3: (0.035, 0.070),
    4: (0.070, 0.115),
    5: (0.000, 0.000),
}


def angle_robot_deg(dx_m: Union[float, np.ndarray],
                    dz_m: Union[float, np.ndarray]) -> np.ndarray:
    dx = np.asarray(dx_m, dtype=float)
    dz = np.asarray(dz_m, dtype=float)
    return (np.degrees(np.arctan2(dz, -dx)) + 360) % 360


def quadrant_from_angle(angle_deg: float) -> int:
    """Return quadrant 1-4 for robot-frame angle in degrees."""
    a = float(angle_deg) % 360
    if a < 90:
        return 1
    if a < 180:
        return 2
    if a < 270:
        return 3
    return 4


def quadrant_from_spawn(spawn_x: float, spawn_z: float,
                        platform_center_x: float,
                        platform_center_z: float) -> int:
    dx = spawn_x - platform_center_x
    dz = spawn_z - platform_center_z
    return quadrant_from_angle(float(angle_robot_deg(dx, dz)))


def _quadrant_angle_range(quadrant: int) -> Tuple[float, float]:
    q = int(quadrant)
    if q < 1 or q > 4:
        raise ValueError(f"quadrant must be 1-4, got {quadrant}")
    lo, hi = QUADRANT_ANGLE_BINS[q - 1]
    return float(lo), float(hi)


def _offset_from_angle_radius(angle_deg: float, radius_m: float) -> Tuple[float, float]:
    """World-frame (dx, dz) from platform centre given robot-frame angle and radius."""
    rad = math.radians(angle_deg)
    dz = radius_m * math.sin(rad)
    dx = -radius_m * math.cos(rad)
    return dx, dz


def _get_phase_radii(phase: int) -> Tuple[float, float]:
    phase = max(0, min(int(phase), 5))
    return PHASE_BAND_RADII[phase]


def is_weak_cell(robot_id: int, band_phase: int, quadrant: int,
                 weak_regions: Dict[int, List[Dict[str, int]]]) -> bool:
    cells = weak_regions.get(int(robot_id), [])
    for cell in cells:
        if int(cell.get('phase', -1)) == int(band_phase) and int(cell.get('quadrant', -1)) == int(quadrant):
            return True
    return False


def classify_demo_bucket(robot_id: int,
                         spawn_phase: int,
                         spawn_x: float,
                         spawn_z: float,
                         platform_center_x: float,
                         platform_center_z: float,
                         weak_regions: Dict[int, List[Dict[str, int]]],
                         spawn_collection: Optional[str] = None) -> str:
    """
    Fallback bucket classifier when client omits demo_bucket.
    Prefer client-assigned demo_bucket at collection time.
    """
    if spawn_collection == 'full_grid':
        return 'normal'
    if spawn_collection == 'weak_cell':
        return 'weak'

    quadrant = quadrant_from_spawn(
        spawn_x, spawn_z, platform_center_x, platform_center_z
    )
    if is_weak_cell(robot_id, spawn_phase, quadrant, weak_regions):
        return 'weak'
    return 'normal'


def sample_spawn_in_phase_quadrant(
    band_phase: int,
    quadrant: int,
    platform_center_x: float,
    platform_center_z: float,
    half_size_x: float,
    half_size_z: float,
    in_spawn_area=None,
    max_attempts: int = 200,
) -> Tuple[float, float, float]:
    """
    Sample (spawn_x, spawn_z, radius_m) in an inference-style phase band and quadrant.

    band_phase: inference spawn band index (PHASE_CONFIG), not training curriculum phase.
    in_spawn_area: optional callable(wx, wz) -> bool for R2 curved platform.
    """
    r_min, r_max = _get_phase_radii(band_phase)
    angle_lo, angle_hi = _quadrant_angle_range(quadrant)
    cx, cz = platform_center_x, platform_center_z

    if r_max < 0.001:
        return cx, cz, 0.0

    for _ in range(max_attempts):
        angle_deg = np.random.uniform(angle_lo, angle_hi)
        radius = np.random.uniform(r_min, r_max)
        dx, dz = _offset_from_angle_radius(angle_deg, radius)
        sx = cx + dx
        sz = cz + dz

        sx = float(np.clip(sx, cx - half_size_x, cx + half_size_x))
        sz = float(np.clip(sz, cz - half_size_z, cz + half_size_z))

        if in_spawn_area is not None and not in_spawn_area(sx, sz):
            continue

        actual_r = math.hypot(sx - cx, sz - cz)
        if r_min <= actual_r <= r_max + 1e-6:
            q = quadrant_from_spawn(sx, sz, cx, cz)
            if q == int(quadrant):
                return sx, sz, actual_r

    # Fallback: centre of quadrant at mid-band radius
    mid_angle = (angle_lo + angle_hi) / 2
    mid_r = (r_min + r_max) / 2
    dx, dz = _offset_from_angle_radius(mid_angle, mid_r)
    sx = float(np.clip(cx + dx, cx - half_size_x, cx + half_size_x))
    sz = float(np.clip(sz, cz - half_size_z, cz + half_size_z))
    return sx, sz, math.hypot(sx - cx, sz - cz)


def default_fine_tune_config_path() -> Path:
    """Resolve fine_tune_config.yaml from repo root."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "host_gpu_system" / "config" / "fine_tune_config.yaml"


def default_locator_train_config_path() -> Path:
    """Resolve locator_train_config.yaml from repo root."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "host_gpu_system" / "config" / "locator_train_config.yaml"


def load_fine_tune_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else default_fine_tune_config_path()
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    weak_ratio = float(cfg.get('sampling', {}).get('weak_ratio', 0.7))
    normal_ratio = float(cfg.get('sampling', {}).get('normal_ratio', 0.3))
    if abs(weak_ratio + normal_ratio - 1.0) > 0.05:
        raise ValueError(
            f"weak_ratio + normal_ratio must sum to ~1.0, got {weak_ratio}+{normal_ratio}"
        )

    weak_regions_raw = cfg.get('weak_regions', {})
    weak_regions: Dict[int, List[Dict[str, int]]] = {}
    for key, cells in weak_regions_raw.items():
        weak_regions[int(key)] = [
            {'phase': int(c['phase']), 'quadrant': int(c['quadrant'])}
            for c in cells
        ]

    cfg['_weak_regions_parsed'] = weak_regions
    return cfg


def load_locator_train_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load locator_train_config.yaml (same schema as fine_tune_config)."""
    cfg_path = Path(path) if path else default_locator_train_config_path()
    return load_fine_tune_config(cfg_path)


def default_board_locator_train_config_path() -> Path:
    """Resolve board_locator_train_config.yaml from repo root."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "host_gpu_system" / "config" / "board_locator_train_config.yaml"


def load_board_locator_train_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load board_locator_train_config.yaml (same schema as fine_tune_config)."""
    cfg_path = Path(path) if path else default_board_locator_train_config_path()
    return load_fine_tune_config(cfg_path)


def pick_random_weak_cell(robot_id: int,
                          weak_regions: Dict[int, List[Dict[str, int]]]) -> Dict[str, int]:
    cells = weak_regions.get(int(robot_id), [])
    if not cells:
        raise ValueError(f"No weak regions configured for robot {robot_id}")
    idx = int(np.random.randint(0, len(cells)))
    return cells[idx]
