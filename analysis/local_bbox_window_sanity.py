"""Sanity-check fixed-window crop + local grid mapping (no YOLO required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "host_gpu_system" / "src"))
sys.path.insert(0, str(_REPO / "vm_simulation_system" / "src"))

from local_bbox_window import (  # noqa: E402
    crop_fixed_window, local_cell_to_world, world_to_local_cell, window_xyxy_warp,
)
from board_grid import load_board_dqn_config  # noqa: E402


def main() -> None:
    cfg = load_board_dqn_config()
    robot_id = 1
    n, window_m = 20, 0.02
    # Approximate R1 platform center
    cx, cz = -0.646, 0.841
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[:] = (40, 40, 40)
    depth = np.full((224, 224), 0.7, dtype=np.float32)
    rgb_c, depth_c, xyxy = crop_fixed_window(
        rgb, depth, cx, cz, robot_id,
        window_m=window_m, out_size=224, board_cfg=cfg,
    )
    assert rgb_c.shape == (224, 224, 3)
    assert depth_c is not None and depth_c.shape == (224, 224)
    mid = world_to_local_cell(cx, cz, cx, cz, n=n, window_m=window_m)
    assert mid == (n * n) // 2 or mid is not None
    wx, wz = local_cell_to_world(mid, cx, cz, n=n, window_m=window_m)
    assert abs(wx - cx) < window_m and abs(wz - cz) < window_m
    win = window_xyxy_warp(cx, cz, robot_id, window_m=window_m, warp_size=224, board_cfg=cfg)
    print(f"OK window_xyxy={win} mid_cell={mid} world=({wx:.4f},{wz:.4f})")


if __name__ == "__main__":
    main()
