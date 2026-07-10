"""Synthetic block segmentation labels on the warped board (from sim oracle XZ)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_VM_SRC = Path(__file__).resolve().parent.parent.parent / "vm_simulation_system" / "src"
if str(_VM_SRC) not in sys.path:
    sys.path.insert(0, str(_VM_SRC))

from board_warp import world_xz_to_warp_pixel  # noqa: E402


def block_half_extents_m(locator_cfg: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
    """Return (half_width_x_m, half_depth_z_m) for the block footprint."""
    block = (locator_cfg or {}).get("block", {})
    w = float(block.get("width_m", 0.013))
    d = float(block.get("depth_m", 0.033))
    return w * 0.5, d * 0.5


def make_block_mask_warp(
    label_x: float,
    label_z: float,
    robot_id: int,
    *,
    out_size: int = 224,
    grid_n: int = 112,
    board_cfg: Optional[Dict[str, Any]] = None,
    locator_cfg: Optional[Dict[str, Any]] = None,
    half_width_m: Optional[float] = None,
    half_depth_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build teacher masks from world XZ (not from image pixels).

    Returns:
        mask_warp: uint8 (out_size, out_size), values 0 or 1
        mask_grid: float32 (grid_n, grid_n), values 0.0 or 1.0
    """
    if half_width_m is None or half_depth_m is None:
        hw, hd = block_half_extents_m(locator_cfg)
        half_width_m = hw if half_width_m is None else half_width_m
        half_depth_m = hd if half_depth_m is None else half_depth_m

    lx, lz = float(label_x), float(label_z)
    corners_xz = [
        (lx - half_width_m, lz - half_depth_m),
        (lx + half_width_m, lz - half_depth_m),
        (lx + half_width_m, lz + half_depth_m),
        (lx - half_width_m, lz + half_depth_m),
    ]
    pts = []
    for cx, cz in corners_xz:
        u, v = world_xz_to_warp_pixel(
            cx, cz, robot_id, out_size=out_size, cfg=board_cfg, apply_post_flip=True,
        )
        pts.append([u, v])

    mask_warp = np.zeros((out_size, out_size), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask_warp, [np.array(pts, dtype=np.int32)], 1)

    mask_grid = cv2.resize(
        mask_warp.astype(np.float32), (grid_n, grid_n), interpolation=cv2.INTER_AREA,
    )
    mask_grid = (mask_grid > 0.25).astype(np.float32)
    return mask_warp, mask_grid


def mask_grid_to_cell(
    mask_grid: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """Centroid cell of the mask blob; return (cell_index, confidence in [0,1])."""
    grid = np.asarray(mask_grid, dtype=np.float64)
    n = grid.shape[0]
    w = np.clip(grid, 0.0, None).reshape(-1)
    if invalid_mask is not None:
        inv = np.asarray(invalid_mask, dtype=bool).reshape(-1)
        w = w.copy()
        w[inv] = 0.0
    total = float(w.sum())
    if total <= 1e-9:
        return 0, 0.0
    rows = np.repeat(np.arange(n), n)
    cols = np.tile(np.arange(n), n)
    cr = int(round(float((w * rows).sum() / total)))
    cc = int(round(float((w * cols).sum() / total)))
    cr = min(max(cr, 0), n - 1)
    cc = min(max(cc, 0), n - 1)
    cell = cr * n + cc
    conf = float(total / w.size)
    return cell, conf


def blend_mask_overlay(
    rgb_board: np.ndarray,
    mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend a single-channel mask onto RGB warp image."""
    vis = rgb_board.astype(np.float32).copy()
    m = np.asarray(mask, dtype=np.float32)
    if m.shape[:2] != vis.shape[:2]:
        m = cv2.resize(m, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = np.clip(m, 0.0, 1.0)
    if m.ndim == 3:
        m = m[..., 0]
    color = np.array(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    vis = vis * (1.0 - alpha * m[..., None]) + color * (alpha * m[..., None])
    return np.clip(vis, 0, 255).astype(np.uint8)
