"""Projective board warp and grid↔world mapping (Gomes paper style)."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from board_grid import board_bounds, board_n, load_board_dqn_config


def _config_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "host_gpu_system" / "config"


def _default_world_corners(robot_id: int, cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    # Order matches board_grid row/col: top-left → bottom-right on warped output.
    return np.asarray([
        [x_min, z_max],
        [x_min, z_min],
        [x_max, z_max],
        [x_max, z_min],
    ], dtype=np.float64)


def _load_warp_yaml(robot_id: int) -> Dict[str, Any]:
    path = _config_root() / f"board_warp_r{robot_id}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_load_warp_yaml = lru_cache(maxsize=4)(_load_warp_yaml)


def image_corners_crop_local(robot_id: int, crop_w: int, crop_h: int) -> np.ndarray:
    """Default image corners: full cropped frame maps to platform."""
    return np.asarray([
        [0, 0],
        [0, crop_h],
        [crop_w, 0],
        [crop_w, crop_h],
    ], dtype=np.float64)


def get_warp_corners(
    robot_id: int,
    crop_w: int,
    crop_h: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = cfg or load_board_dqn_config()
    warp_cfg = (cfg.get("warp") or {}).get(str(int(robot_id)), {})
    yaml_extra = _load_warp_yaml(robot_id)

    world = warp_cfg.get("world_corners") or yaml_extra.get("world_corners")
    if world is not None:
        world_corners = np.asarray(world, dtype=np.float64)
    else:
        world_corners = _default_world_corners(robot_id, cfg)

    image = warp_cfg.get("image_corners") or yaml_extra.get("image_corners")
    if image is not None:
        image_corners = np.asarray(image, dtype=np.float64)
    else:
        image_corners = image_corners_crop_local(robot_id, crop_w, crop_h)

    return image_corners, world_corners


@lru_cache(maxsize=8)
def _grid_to_world_transform(robot_id: int, n: int) -> Any:
    from skimage.transform import ProjectiveTransform
    x_min, x_max, z_min, z_max = board_bounds(robot_id)
    src = np.asarray([[0, 0], [0, n], [n, 0], [n, n]], dtype=np.float64)
    dst = np.asarray([
        [x_min, z_max],
        [x_min, z_min],
        [x_max, z_max],
        [x_max, z_min],
    ], dtype=np.float64)
    t = ProjectiveTransform()
    if not t.estimate(src, dst):
        raise RuntimeError(f"board grid→world estimate failed (R{robot_id})")
    return t


def uv_to_world(u: float, v: float, robot_id: int, n: Optional[int] = None) -> Tuple[float, float]:
    n = n or board_n()
    t = _grid_to_world_transform(robot_id, n)
    xy = t((float(u), float(v))).squeeze()
    return float(xy[0]), float(xy[1])


def world_to_uv(x: float, z: float, robot_id: int, n: Optional[int] = None) -> Tuple[int, int]:
    n = n or board_n()
    t = _grid_to_world_transform(robot_id, n)
    inv = t.inverse
    uv = inv((float(x), float(z))).squeeze()
    u = int(np.clip(round(float(uv[0])), 0, n - 1))
    v = int(np.clip(round(float(uv[1])), 0, n - 1))
    return u, v


def world_to_board_cell_warp(
    x: float, z: float, robot_id: int, n: Optional[int] = None,
) -> Optional[int]:
    n = n or board_n()
    u, v = world_to_uv(x, z, robot_id, n)
    return v * n + u


def board_cell_to_world_warp(cell: int, robot_id: int, n: Optional[int] = None) -> Tuple[float, float]:
    n = n or board_n()
    row, col = divmod(int(cell), n)
    return uv_to_world(col + 0.5, row + 0.5, robot_id, n)


def _warp_dst_corners(out_size: int) -> np.ndarray:
    """Top-left, bottom-left, top-right, bottom-right on board grid (row 0 = z_max)."""
    s = float(out_size)
    return np.asarray([[0, 0], [0, s], [s, 0], [s, s]], dtype=np.float32)


def _legacy_image_corner_order(img_corners: np.ndarray) -> np.ndarray:
    """Old yaml used z_min as 'top-left' world — swap TL↔BL and TR↔BR for grid alignment."""
    out = img_corners.copy()
    out[0], out[1] = img_corners[1], img_corners[0]
    out[2], out[3] = img_corners[3], img_corners[2]
    return out


def _default_corner_permutation(robot_id: int) -> Optional[List[int]]:
    """Map yaml visual [TL, BL, TR, BR] → src order for _warp_dst_corners."""
    if int(robot_id) == 1:
        return [0, 2, 1, 3]
    if int(robot_id) == 2:
        return [3, 1, 2, 0]
    return None


def _warp_image_corners(
    robot_id: int, img_corners: np.ndarray, crop_w: int, crop_h: int,
) -> np.ndarray:
    yaml_extra = _load_warp_yaml(robot_id)
    order = str(yaml_extra.get('image_corner_order', 'visual')).lower()
    if order == 'legacy':
        img_corners = _legacy_image_corner_order(img_corners)
    elif order not in ('grid', 'visual'):
        raise ValueError(f"Unknown image_corner_order: {order}")

    perm = yaml_extra.get('image_corner_permutation')
    if perm is None:
        perm = _default_corner_permutation(robot_id)
    if perm is not None:
        idx = [int(i) for i in perm]
        if len(idx) != 4 or sorted(idx) != [0, 1, 2, 3]:
            raise ValueError(f"image_corner_permutation must be 4 unique indices 0-3, got {perm}")
        img_corners = img_corners[idx]
    return img_corners


def _post_flip_vertical(robot_id: int) -> bool:
    yaml_extra = _load_warp_yaml(robot_id)
    if 'post_flip_vertical' in yaml_extra:
        return bool(yaml_extra['post_flip_vertical'])
    return False


def _post_flip_horizontal(robot_id: int) -> bool:
    yaml_extra = _load_warp_yaml(robot_id)
    if 'post_flip_horizontal' in yaml_extra:
        return bool(yaml_extra['post_flip_horizontal'])
    return False


def _apply_warp(
    frame: np.ndarray,
    robot_id: int,
    out_size: int,
    cfg: Optional[Dict[str, Any]],
) -> np.ndarray:
    h, w = frame.shape[:2]
    img_corners, _ = get_warp_corners(robot_id, w, h, cfg)
    src = _warp_image_corners(robot_id, img_corners, w, h).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, _warp_dst_corners(out_size))
    out = cv2.warpPerspective(frame, matrix, (out_size, out_size))
    if _post_flip_vertical(robot_id):
        out = cv2.flip(out, 0)
    if _post_flip_horizontal(robot_id):
        out = cv2.flip(out, 1)
    return out


def get_warp_perspective_matrix(
    robot_id: int,
    crop_w: int,
    crop_h: int,
    out_size: int = 224,
    cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    img_corners, _ = get_warp_corners(robot_id, crop_w, crop_h, cfg)
    src = _warp_image_corners(robot_id, img_corners, crop_w, crop_h).astype(np.float32)
    return cv2.getPerspectiveTransform(src, _warp_dst_corners(out_size))


def world_xz_to_warp_pixel(
    x: float,
    z: float,
    robot_id: int,
    out_size: int = 224,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    apply_post_flip: bool = True,
) -> Tuple[int, int]:
    """Linear board-grid pixel on the warped CNN input (optionally after post-flip)."""
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    u = (float(x) - x_min) / max(x_max - x_min, 1e-9) * out_size
    v = (z_max - float(z)) / max(z_max - z_min, 1e-9) * out_size
    if apply_post_flip:
        if _post_flip_horizontal(robot_id):
            u = out_size - u
        if _post_flip_vertical(robot_id):
            v = out_size - v
    return int(round(u)), int(round(v))


def warp_pixel_to_world_xz(
    u: float,
    v: float,
    robot_id: int,
    out_size: int = 224,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    apply_post_flip: bool = True,
) -> Tuple[float, float]:
    """Inverse of world_xz_to_warp_pixel (undo post-flip first)."""
    uu, vv = float(u), float(v)
    if apply_post_flip:
        if _post_flip_vertical(robot_id):
            vv = out_size - vv
        if _post_flip_horizontal(robot_id):
            uu = out_size - uu
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    x = x_min + (uu / max(float(out_size), 1e-9)) * (x_max - x_min)
    z = z_max - (vv / max(float(out_size), 1e-9)) * (z_max - z_min)
    return float(x), float(z)


def warp_pixel_to_crop_pixel(
    u: float,
    v: float,
    robot_id: int,
    crop_w: int,
    crop_h: int,
    out_size: int = 224,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    apply_post_flip: bool = True,
) -> Tuple[float, float]:
    """Inverse-map a warp pixel to crop-local coords (undo post-flip first)."""
    if apply_post_flip:
        if _post_flip_vertical(robot_id):
            v = out_size - v
        if _post_flip_horizontal(robot_id):
            u = out_size - u
    m = get_warp_perspective_matrix(robot_id, crop_w, crop_h, out_size, cfg)
    pt = cv2.perspectiveTransform(
        np.array([[[float(u), float(v)]]], dtype=np.float32),
        np.linalg.inv(m),
    )[0, 0]
    return float(pt[0]), float(pt[1])


def warp_rgb_to_board(
    rgb: np.ndarray,
    robot_id: int,
    out_size: int = 224,
    cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    return _apply_warp(rgb, robot_id, out_size, cfg)


def warp_depth_to_board(
    depth: np.ndarray,
    robot_id: int,
    out_size: int = 224,
    cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    return _apply_warp(depth, robot_id, out_size, cfg)
