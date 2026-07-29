#!/usr/bin/env python3
"""Offline helper: map platform world corners to crop-local pixel coords for board warp.

Modes:
  (default)  Write a placeholder board_warp_rN.yaml with full-crop corners.
  --click    Open the crop JPG, click visual TL/BL/TR/BR, write yaml.
"""

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

_LABELS = ("TL", "BL", "TR", "BR")


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


def _default_crop_candidates(robot_id: int) -> list[Path]:
    name = f"ai_vision_debug_rgb_sim_r{robot_id}.jpg"
    return [
        _REPO / "host_gpu_system" / "debug" / name,
        _REPO / "host_gpu_system" / name,
        _REPO / "host_gpu_system" / "src" / name,
        _REPO / name,
        _REPO / "host_gpu_system" / "debug" / f"board_crop_r{robot_id}_latest.jpg",
    ]


def _resolve_crop_image(robot_id: int, image: str | None) -> Path:
    if image:
        path = Path(image)
        if not path.is_file():
            raise SystemExit(f"Crop image not found: {path}")
        return path
    for cand in _default_crop_candidates(robot_id):
        if cand.is_file():
            return cand
    tried = "\n  ".join(str(p) for p in _default_crop_candidates(robot_id))
    raise SystemExit(
        f"No crop image for robot {robot_id}. Run one episode first, or pass --image.\nTried:\n  {tried}"
    )


def _world_corners(robot_id: int, cfg) -> list:
    x_min, x_max, z_min, z_max = board_bounds(robot_id, cfg)
    return [
        [x_min, z_max],
        [x_min, z_min],
        [x_max, z_max],
        [x_max, z_min],
    ]


def _write_warp_yaml(
    robot_id: int,
    image_corners: list,
    cfg,
    *,
    permutation: list | None = None,
    post_flip_vertical: bool = False,
    post_flip_horizontal: bool = False,
) -> Path:
    dest = _REPO / "host_gpu_system" / "config" / f"board_warp_r{robot_id}.yaml"
    lines = [
        f"# Click SpawnArea corners in ai_vision_debug_rgb_sim_r{robot_id}.jpg as VISUAL TL, BL, TR, BR.",
        "# permutation remaps clicks to grid order (TL, TR, BL, BR world corners).",
        "image_corner_order: visual",
    ]
    if permutation is not None:
        lines.append(f"image_corner_permutation: {permutation}")
    lines.append(f"post_flip_vertical: {'true' if post_flip_vertical else 'false'}")
    lines.append(f"post_flip_horizontal: {'true' if post_flip_horizontal else 'false'}")
    lines.append("image_corners:")
    for xy in image_corners:
        lines.append(f"- - {int(xy[0])}")
        lines.append(f"  - {int(xy[1])}")
    lines.append("world_corners:")
    for xz in _world_corners(robot_id, cfg):
        lines.append(f"- - {xz[0]}")
        lines.append(f"  - {xz[1]}")
    lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def click_corners(image_path: Path) -> list[list[int]]:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    vis = img.copy()
    pts: list[list[int]] = []
    win = f"click corners TL, BL, TR, BR — u=undo  Enter/q=save  Esc=abort"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal vis
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(pts) >= 4:
            return
        pts.append([int(x), int(y)])
        label = _LABELS[len(pts) - 1]
        cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(
            vis, f"{len(pts)-1}:{label}", (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )
        if len(pts) >= 2:
            cv2.line(vis, tuple(pts[-2]), tuple(pts[-1]), (0, 200, 255), 1)
        if len(pts) == 4:
            cv2.line(vis, tuple(pts[0]), tuple(pts[2]), (0, 200, 255), 1)
            cv2.line(vis, tuple(pts[1]), tuple(pts[3]), (0, 200, 255), 1)
        print(f"  {label}: ({x}, {y})")
        cv2.imshow(win, vis)

    print(f"Image: {image_path}")
    print("Click SpawnArea corners in order: TL, BL, TR, BR")
    print("Keys: u=undo last | Enter or q=save | Esc=abort")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, vis)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27,):  # Esc
            cv2.destroyAllWindows()
            raise SystemExit("Aborted (no yaml written).")
        if key in (ord("u"), ord("U")):
            if pts:
                pts.pop()
                vis = img.copy()
                for i, (x, y) in enumerate(pts):
                    cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
                    cv2.putText(
                        vis, f"{i}:{_LABELS[i]}", (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
                    )
                for i in range(1, len(pts)):
                    cv2.line(vis, tuple(pts[i - 1]), tuple(pts[i]), (0, 200, 255), 1)
                cv2.imshow(win, vis)
                print("  undo")
        if key in (13, ord("q"), ord("Q")):  # Enter / q
            if len(pts) != 4:
                print(f"Need 4 corners, have {len(pts)}")
                continue
            break

    cv2.destroyAllWindows()
    return pts


def main():
    parser = argparse.ArgumentParser(description="Calibrate board_warp_rN.yaml image_corners.")
    parser.add_argument("--robot-id", type=int, choices=[1, 2], required=True)
    parser.add_argument("--width", type=int, default=640, help="Full camera frame width (placeholder mode)")
    parser.add_argument("--height", type=int, default=480, help="Full camera frame height (placeholder mode)")
    parser.add_argument("--config", type=str, default=None, help="board_dqn_config.yaml path")
    parser.add_argument(
        "--click",
        action="store_true",
        help="Open crop image and click visual TL, BL, TR, BR to write yaml",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Crop JPG path (default: ai_vision_debug_rgb_sim_rN.jpg search)",
    )
    parser.add_argument(
        "--permutation",
        type=str,
        default=None,
        help="Optional image_corner_permutation as 4 ints, e.g. 0,2,1,3",
    )
    args = parser.parse_args()

    cfg = load_board_dqn_config(args.config)
    perm = None
    if args.permutation:
        perm = [int(x.strip()) for x in args.permutation.split(",")]
        if sorted(perm) != [0, 1, 2, 3]:
            raise SystemExit("--permutation must be 4 unique indices 0-3")

    if args.click:
        image_path = _resolve_crop_image(args.robot_id, args.image)
        pts = click_corners(image_path)
        # Keep existing flip flags if yaml already exists; else R1 historically used vertical flip off after perm.
        existing = _REPO / "host_gpu_system" / "config" / f"board_warp_r{args.robot_id}.yaml"
        fv, fh = False, False
        if existing.is_file():
            old = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
            fv = bool(old.get("post_flip_vertical", False))
            fh = bool(old.get("post_flip_horizontal", False))
            if perm is None and old.get("image_corner_permutation") is not None:
                perm = list(old["image_corner_permutation"])
        dest = _write_warp_yaml(
            args.robot_id, pts, cfg,
            permutation=perm,
            post_flip_vertical=fv,
            post_flip_horizontal=fh,
        )
        print(f"Wrote {dest}")
        for i, (x, y) in enumerate(pts):
            print(f"  {_LABELS[i]}: [{x}, {y}]")
        return

    crop = R1_CROP if args.robot_id == 1 else R2_CROP
    crop_w, crop_h = crop_size(args.height, args.width, crop)
    dest = _write_warp_yaml(
        args.robot_id,
        default_corners(crop_w, crop_h),
        cfg,
        permutation=perm,
        post_flip_vertical=(args.robot_id == 1),
        post_flip_horizontal=False,
    )
    print(f"Wrote placeholder {dest}")
    print(f"Crop size: {crop_w}x{crop_h}")


if __name__ == "__main__":
    main()
