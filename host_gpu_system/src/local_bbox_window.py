"""Fixed-size world window + dense local grid around a detector center."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(_REPO_ROOT / "vm_simulation_system" / "src"))
from board_warp import warp_pixel_to_world_xz, world_xz_to_warp_pixel  # noqa: E402
from local_grid import cell_to_world, world_to_cell  # noqa: E402


def default_local_bbox_dqn_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "local_bbox_dqn_config.yaml"


def load_local_bbox_dqn_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else default_local_bbox_dqn_config_path()
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def local_bbox_grid_params(cfg: Optional[Dict[str, Any]] = None) -> Tuple[int, float]:
    g = (cfg or {}).get("grid", {})
    return int(g.get("n", 20)), float(g.get("window_m", 0.02))


def bbox_center_warp_px(bbox) -> Tuple[float, float]:
    return 0.5 * (float(bbox.x1) + float(bbox.x2)), 0.5 * (float(bbox.y1) + float(bbox.y2))


def bbox_center_world(
    bbox,
    robot_id: int,
    *,
    warp_size: int = 224,
    board_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    u, v = bbox_center_warp_px(bbox)
    return warp_pixel_to_world_xz(
        u, v, robot_id, out_size=warp_size, cfg=board_cfg, apply_post_flip=True,
    )


def window_xyxy_warp(
    center_x: float,
    center_z: float,
    robot_id: int,
    *,
    window_m: float = 0.02,
    warp_size: int = 224,
    board_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int, int]:
    """Axis-aligned warp-pixel rect covering the fixed world window."""
    half = float(window_m) / 2.0
    corners = [
        (center_x - half, center_z - half),
        (center_x + half, center_z - half),
        (center_x - half, center_z + half),
        (center_x + half, center_z + half),
    ]
    us, vs = [], []
    for wx, wz in corners:
        u, v = world_xz_to_warp_pixel(
            wx, wz, robot_id, out_size=warp_size, cfg=board_cfg, apply_post_flip=True,
        )
        us.append(u)
        vs.append(v)
    x1 = int(max(0, min(us)))
    x2 = int(min(warp_size - 1, max(us)))
    y1 = int(max(0, min(vs)))
    y2 = int(min(warp_size - 1, max(vs)))
    if x2 <= x1:
        x2 = min(warp_size - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(warp_size - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_fixed_window(
    rgb_w: np.ndarray,
    depth_w: Optional[np.ndarray],
    center_x: float,
    center_z: float,
    robot_id: int,
    *,
    window_m: float = 0.02,
    out_size: int = 224,
    board_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int]]:
    """Crop RGB (and optional depth) warp to the fixed window; resize to out_size."""
    h, w = rgb_w.shape[:2]
    x1, y1, x2, y2 = window_xyxy_warp(
        center_x, center_z, robot_id,
        window_m=window_m, warp_size=w, board_cfg=board_cfg,
    )
    rgb_c = rgb_w[y1:y2 + 1, x1:x2 + 1]
    if rgb_c.size == 0:
        rgb_c = rgb_w
        x1, y1, x2, y2 = 0, 0, w - 1, h - 1
    rgb_r = cv2.resize(rgb_c, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    depth_r = None
    if depth_w is not None:
        depth_c = depth_w[y1:y2 + 1, x1:x2 + 1]
        if depth_c.size == 0:
            depth_c = depth_w
        depth_r = cv2.resize(depth_c, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return rgb_r, depth_r, (x1, y1, x2, y2)


def local_cell_to_world(
    cell: int,
    center_x: float,
    center_z: float,
    *,
    n: int = 20,
    window_m: float = 0.02,
) -> Tuple[float, float]:
    return cell_to_world(cell, center_x, center_z, n=n, window_m=window_m)


def world_to_local_cell(
    x: float,
    z: float,
    center_x: float,
    center_z: float,
    *,
    n: int = 20,
    window_m: float = 0.02,
) -> Optional[int]:
    return world_to_cell(x, z, center_x, center_z, n=n, window_m=window_m)
