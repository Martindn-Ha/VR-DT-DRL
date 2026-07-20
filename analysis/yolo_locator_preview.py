#!/usr/bin/env python3
"""Run Roboflow workflow on saved warp images; save bbox + random cell overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "host_gpu_system" / "src"))
sys.path.insert(0, str(_REPO / "vm_simulation_system" / "src"))

from board_grid import board_n, invalid_cell_mask, load_board_dqn_config  # noqa: E402
from yolo_locator import (  # noqa: E402
    LocalYoloLocator, load_yolo_locator_config, random_cell_in_bbox,
)


def _cell_pixel(cell: int, h: int, w: int, robot_id: int, n: int, board_cfg) -> tuple:
    from board_grid import board_cell_to_world
    from board_warp import world_xz_to_warp_pixel
    wx, wz = board_cell_to_world(cell, robot_id, n=n, cfg=board_cfg)
    return world_xz_to_warp_pixel(wx, wz, robot_id, out_size=w, cfg=board_cfg, apply_post_flip=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview local YOLO on warp PNGs.")
    parser.add_argument("--images-dir", type=str,
                        default=str(_REPO / "data" / "board_locator_dataset" / "r1" / "images"))
    parser.add_argument("--glob", type=str, default="*_warp.png")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--robot-id", type=int, default=1, choices=[1, 2])
    parser.add_argument("--yolo-config", type=str, default=None)
    parser.add_argument("--board-config", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    yolo_cfg = load_yolo_locator_config(args.yolo_config)
    board_cfg = load_board_dqn_config(args.board_config)
    n = board_n(board_cfg)
    client = LocalYoloLocator(yolo_cfg)

    img_dir = Path(args.images_dir)
    paths = sorted(img_dir.glob(args.glob))[: max(1, args.limit)]
    if not paths:
        raise SystemExit(f"No images matching {args.glob} in {img_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        _REPO / "host_gpu_system" / "debug" / "yolo_preview"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = invalid_cell_mask(args.robot_id, n, board_cfg)
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"skip unreadable: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bbox, raw = client.detect_bbox(rgb)
        vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        stem = path.stem
        if bbox is None:
            print(f"{stem}: no detection")
            cv2.putText(vis, "no detection", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in (bbox.x1, bbox.y1, bbox.x2, bbox.y2)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cell, count = random_cell_in_bbox(
                bbox, n, inv, h, w, args.robot_id, board_cfg=board_cfg,
            )
            label = f"conf={bbox.confidence:.2f} cells={count}"
            cv2.putText(vis, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if cell is not None:
                px, py = _cell_pixel(cell, h, w, args.robot_id, n, board_cfg)
                cv2.drawMarker(vis, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            print(f"{stem}: conf={bbox.confidence:.3f} cells_in_bbox={count} cell={cell}")
        out_path = out_dir / f"{stem}_yolo.jpg"
        cv2.imwrite(str(out_path), vis)
        with open(out_dir / f"{stem}_raw.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)


if __name__ == "__main__":
    main()
