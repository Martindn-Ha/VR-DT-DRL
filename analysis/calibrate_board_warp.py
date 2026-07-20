#!/usr/bin/env python3
"""Offline helper: map platform world corners to crop-local pixel coords for board warp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "vm_simulation_system" / "src"))

from board_grid import board_bounds, load_board_dqn_config  # noqa: E402

R1_CROP = dict(crop_y0=0.22, crop_y1=0.7, crop_x0=0.37, crop_x1=0.60)
R2_CROP = dict(crop_y0=0.21, crop_y1=0.885, crop_x0=0.37, crop_x1=0.71)


def crop_size(full_h: int, full_w: int, crop: dict) -> tuple:
    y0 = int(crop["crop_y0"] * full_h)
    y1 = int(crop["crop_y1"] * full_h)
    x0 = int(crop["crop_x0"] * full_w)
    x1 = int(crop["crop_x1"] * full_w)
    return x1 - x0, y1 - y0


def default_corners(crop_w: int, crop_h: int) -> list:
    return [
        [0, 0],
        [0, crop_h],
        [crop_w, 0],
        [crop_w, crop_h],
    ]


def main():
    parser = argparse.ArgumentParser(description="Write board_warp_rN.yaml image_corners placeholder.")
    parser.add_argument("--robot-id", type=int, choices=[1, 2], required=True)
    parser.add_argument("--width", type=int, default=640, help="Full camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Full camera frame height")
    parser.add_argument("--config", type=str, default=None, help="board_dqn_config.yaml path")
    args = parser.parse_args()

    cfg = load_board_dqn_config(args.config)
    crop = R1_CROP if args.robot_id == 1 else R2_CROP
    crop_w, crop_h = crop_size(args.height, args.width, crop)
    x_min, x_max, z_min, z_max = board_bounds(args.robot_id, cfg)

    out = {
        "image_corners": default_corners(crop_w, crop_h),
        "image_corner_order": "visual",
        "post_flip_vertical": args.robot_id == 1,
        "world_corners": [
            [x_min, z_max],
            [x_min, z_min],
            [x_max, z_max],
            [x_max, z_min],
        ],
        "note": (
            "Click table corners in crop: visual TL, BL, TR, BR (SpawnArea overlay). "
            "Grid row 0 = z_max (back edge)."
        ),
    }
    dest = _REPO / "host_gpu_system" / "config" / f"board_warp_r{args.robot_id}.yaml"
    with open(dest, "w", encoding="utf-8") as f:
        yaml.dump(out, f, default_flow_style=False)
    print(f"Wrote {dest}")
    print(f"Crop size: {crop_w}x{crop_h}")


if __name__ == "__main__":
    main()
