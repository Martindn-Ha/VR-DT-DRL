#!/usr/bin/env python3
"""Preview synthetic board segmentation labels (spawn XZ → green mask on warp RGB)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "host_gpu_system" / "src"))
sys.path.insert(0, str(_REPO / "vm_simulation_system" / "src"))

from board_grid import load_board_dqn_config, world_to_board_cell  # noqa: E402
from board_seg_labels import make_block_mask_warp, blend_mask_overlay, mask_grid_to_cell  # noqa: E402
from spawn_geometry import load_board_locator_train_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview board seg teacher mask on warp RGB.")
    parser.add_argument("--warp-rgb", type=str, required=True,
                        help="Path to warped RGB image (e.g. debug/board_warp_r1_latest.jpg)")
    parser.add_argument("--spawn-x", type=float, required=True)
    parser.add_argument("--spawn-z", type=float, required=True)
    parser.add_argument("--robot-id", type=int, default=1, choices=[1, 2])
    parser.add_argument("--board-config", type=str, default=None)
    parser.add_argument("--locator-config", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    board_cfg = load_board_dqn_config(args.board_config)
    locator_cfg = load_board_locator_train_config(args.locator_config)
    n = int(board_cfg.get("grid", {}).get("n", 112))

    bgr = cv2.imread(args.warp_rgb)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.warp_rgb}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mask_warp, mask_grid = make_block_mask_warp(
        args.spawn_x, args.spawn_z, args.robot_id,
        grid_n=n, board_cfg=board_cfg, locator_cfg=locator_cfg,
    )
    cell, conf = mask_grid_to_cell(mask_grid)
    teacher_cell = world_to_board_cell(
        args.spawn_x, args.spawn_z, args.robot_id, n=n, cfg=board_cfg,
    )

    vis = blend_mask_overlay(rgb, mask_warp, color_rgb=(0, 255, 0), alpha=0.45)
    out_path = Path(args.out) if args.out else (
        _REPO / "host_gpu_system" / "debug" / f"board_seg_label_preview_r{args.robot_id}.jpg"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    meta = {
        "spawn_x": args.spawn_x,
        "spawn_z": args.spawn_z,
        "teacher_cell": teacher_cell,
        "mask_cell": cell,
        "mask_conf": conf,
        "warp_rgb": str(Path(args.warp_rgb).resolve()),
        "out": str(out_path.resolve()),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"teacher_cell={teacher_cell} mask_cell={cell} conf={conf:.3f}")


if __name__ == "__main__":
    main()
