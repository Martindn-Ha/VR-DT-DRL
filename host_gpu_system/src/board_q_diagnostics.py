"""Board DQN Q-map diagnostics (sim-only measurement, no control changes)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_VM_SRC = Path(__file__).resolve().parent.parent.parent / "vm_simulation_system" / "src"
if str(_VM_SRC) not in sys.path:
    sys.path.insert(0, str(_VM_SRC))

from board_grid import board_cell_to_world, board_n, invalid_cell_mask


def _chebyshev_cell_dist(cell_a: int, cell_b: int, n: int) -> int:
    ra, ca = divmod(int(cell_a), n)
    rb, cb = divmod(int(cell_b), n)
    return max(abs(ra - rb), abs(ca - cb))


def _world_xz_dist_m(x0: float, z0: float, x1: float, z1: float) -> float:
    return math.hypot(float(x0) - float(x1), float(z0) - float(z1))


def compute_board_q_diagnostics(
    q: np.ndarray,
    teacher_cell: int,
    argmax_cell: int,
    robot_id: int,
    board_cfg: Optional[Dict[str, Any]] = None,
    *,
    top_k: int = 5,
    near_radius_cells: int = 2,
    label_x: Optional[float] = None,
    label_z: Optional[float] = None,
) -> Dict[str, float]:
    """Compute Q-map metrics comparing teacher vs argmax on the 112×112 grid."""
    n = board_n(board_cfg)
    q_flat = np.asarray(q, dtype=np.float64).reshape(-1)
    if q_flat.size != n * n:
        raise ValueError(f"q size {q_flat.size} != {n * n}")

    inv = invalid_cell_mask(robot_id, n=n, cfg=board_cfg)
    valid_idx = np.flatnonzero(~inv)
    if valid_idx.size == 0:
        raise ValueError("no valid board cells")

    teacher = int(teacher_cell)
    argmax = int(argmax_cell)
    q_valid = q_flat[valid_idx]
    q_teacher = float(q_flat[teacher])
    q_argmax = float(q_flat[argmax])

    order = np.argsort(-q_valid)
    ranks = np.empty(valid_idx.size, dtype=np.int64)
    ranks[order] = np.arange(1, valid_idx.size + 1)
    teacher_pos = int(np.searchsorted(valid_idx, teacher))
    if teacher_pos >= valid_idx.size or valid_idx[teacher_pos] != teacher:
        teacher_pos = int(np.where(valid_idx == teacher)[0][0])
    teacher_rank = int(ranks[teacher_pos])
    teacher_rank_pct = 100.0 * (teacher_rank - 1) / max(valid_idx.size - 1, 1)

    q_argmax_over_teacher = (
        float(q_argmax / q_teacher) if abs(q_teacher) > 1e-12 else float("inf")
    )

    q_shift = q_valid - np.max(q_valid)
    softmax = np.exp(q_shift)
    softmax /= softmax.sum()

    mass_near = 0.0
    for i, cell in enumerate(valid_idx):
        if _chebyshev_cell_dist(int(cell), teacher, n) <= near_radius_cells:
            mass_near += float(softmax[i])
    mass_near_teacher = mass_near

    k = min(int(top_k), valid_idx.size)
    top_local = order[:k]
    top_cells = valid_idx[top_local]
    top_scores = q_valid[top_local]
    w_shift = top_scores - np.max(top_scores)
    w = np.exp(w_shift)
    w /= w.sum()
    cx, cz = 0.0, 0.0
    for cell, wt in zip(top_cells, w):
        wx, wz = board_cell_to_world(int(cell), robot_id, n=n, cfg=board_cfg)
        cx += wt * wx
        cz += wt * wz
    topk_centroid_x_m = float(cx)
    topk_centroid_z_m = float(cz)

    if label_x is not None and label_z is not None:
        tx, tz = float(label_x), float(label_z)
    else:
        tx, tz = board_cell_to_world(teacher, robot_id, n=n, cfg=board_cfg)

    ax, az = board_cell_to_world(argmax, robot_id, n=n, cfg=board_cfg)
    topk_centroid_err_m = _world_xz_dist_m(tx, tz, cx, cz)
    argmax_err_m = _world_xz_dist_m(tx, tz, ax, az)

    return {
        "q_teacher": q_teacher,
        "q_argmax": q_argmax,
        "q_argmax_over_teacher": q_argmax_over_teacher,
        "teacher_rank": float(teacher_rank),
        "teacher_rank_pct": float(teacher_rank_pct),
        "mass_near_teacher": float(mass_near_teacher),
        "topk_centroid_x_m": topk_centroid_x_m,
        "topk_centroid_z_m": topk_centroid_z_m,
        "topk_centroid_err_m": float(topk_centroid_err_m),
        "argmax_err_m": float(argmax_err_m),
    }


def blend_q_heatmap_overlay(
    rgb_board: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Upsample 112×112 Q-map to board RGB size and blend JET heatmap."""
    base = np.asarray(rgb_board)
    h, w = base.shape[:2]
    q_norm = np.asarray(q, dtype=np.float32)
    if q_norm.ndim == 1:
        side = int(round(math.sqrt(q_norm.size)))
        q_norm = q_norm.reshape(side, side)
    q_up = cv2.resize(q_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    q_min, q_max = float(q_up.min()), float(q_up.max())
    if q_max > q_min:
        q_u8 = ((q_up - q_min) / (q_max - q_min) * 255.0).astype(np.uint8)
    else:
        q_u8 = np.zeros((h, w), dtype=np.uint8)
    heat_bgr = cv2.applyColorMap(q_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    blended = (
        (1.0 - alpha) * base.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
