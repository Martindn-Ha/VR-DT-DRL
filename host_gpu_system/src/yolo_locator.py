"""Local Ultralytics YOLO client for warp-space block localization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import cv2
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(_REPO_ROOT / "vm_simulation_system" / "src"))
from board_warp import _post_flip_horizontal, _post_flip_vertical  # noqa: E402


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = ""

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class BlockBBoxDetector(Protocol):
    """Detect a block bbox on a warped RGB board image."""

    def detect_bbox(self, rgb_w: np.ndarray) -> Tuple[Optional[BBox], Any]:
        ...


def default_yolo_locator_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "yolo_locator_config.yaml"


def load_yolo_locator_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else default_yolo_locator_config_path()
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_weights_path(cfg: Dict[str, Any]) -> Path:
    raw = str(cfg.get("weights", "host_gpu_system/models/block_yolo/weights/best.pt"))
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _filter_bbox(
    bbox: BBox,
    *,
    min_confidence: float,
    class_name: Optional[str],
) -> Optional[BBox]:
    if bbox.confidence < min_confidence:
        return None
    if class_name and bbox.class_name and bbox.class_name != class_name:
        return None
    return bbox


def cell_center_warp_px(
    cell: int,
    grid_n: int,
    warp_h: int,
    warp_w: int,
    robot_id: int,
) -> Tuple[float, float]:
    """Warp-space pixel center of a board cell (after post-flip)."""
    row, col = divmod(int(cell), grid_n)
    u = (col + 0.5) * warp_w / grid_n
    v = (row + 0.5) * warp_h / grid_n
    if _post_flip_horizontal(robot_id):
        u = warp_w - u
    if _post_flip_vertical(robot_id):
        v = warp_h - v
    return float(u), float(v)


def cells_in_bbox(
    bbox: BBox,
    grid_n: int,
    invalid_mask: Optional[np.ndarray],
    warp_h: int,
    warp_w: int,
    robot_id: int,
    board_cfg: Optional[Dict] = None,
) -> List[int]:
    """Return flat cell indices whose warp-space center lies inside bbox."""
    inv = None
    if invalid_mask is not None:
        inv = np.asarray(invalid_mask, dtype=bool).reshape(-1)

    flip_h = _post_flip_horizontal(robot_id)
    flip_v = _post_flip_vertical(robot_id)

    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    raw_x1, raw_x2 = (warp_w - x2, warp_w - x1) if flip_h else (x1, x2)
    raw_y1, raw_y2 = (warp_h - y2, warp_h - y1) if flip_v else (y1, y2)
    col_min = max(0, int(np.floor(raw_x1 * grid_n / warp_w - 0.5)))
    col_max = min(grid_n - 1, int(np.ceil(raw_x2 * grid_n / warp_w - 0.5)))
    row_min = max(0, int(np.floor(raw_y1 * grid_n / warp_h - 0.5)))
    row_max = min(grid_n - 1, int(np.ceil(raw_y2 * grid_n / warp_h - 0.5)))

    cells: List[int] = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            cell = row * grid_n + col
            if inv is not None and inv[cell]:
                continue
            px, py = cell_center_warp_px(cell, grid_n, warp_h, warp_w, robot_id)
            if bbox.contains(px, py):
                cells.append(cell)
    return cells


def random_cell_in_bbox(
    bbox: BBox,
    grid_n: int,
    invalid_mask: Optional[np.ndarray],
    warp_h: int,
    warp_w: int,
    robot_id: int,
    board_cfg: Optional[Dict] = None,
) -> Tuple[Optional[int], int]:
    """Pick a random valid cell inside bbox. Returns (cell, cells_in_bbox_count)."""
    cells = cells_in_bbox(
        bbox, grid_n, invalid_mask, warp_h, warp_w, robot_id, board_cfg=board_cfg,
    )
    if not cells:
        return None, 0
    return random.choice(cells), len(cells)


def best_q_cell_in_bbox(
    q_map: np.ndarray,
    bbox: BBox,
    grid_n: int,
    invalid_mask: Optional[np.ndarray],
    warp_h: int,
    warp_w: int,
    robot_id: int,
    board_cfg: Optional[Dict] = None,
) -> Tuple[Optional[int], int]:
    """Argmax Q among valid cells inside bbox. Returns (cell, cells_in_bbox_count)."""
    cells = cells_in_bbox(
        bbox, grid_n, invalid_mask, warp_h, warp_w, robot_id, board_cfg=board_cfg,
    )
    if not cells:
        return None, 0
    q_flat = np.asarray(q_map, dtype=np.float64).reshape(-1)
    best_cell = max(cells, key=lambda c: float(q_flat[c]))
    return int(best_cell), len(cells)


def dist_cell_to_bbox_px(
    cell: int,
    bbox: BBox,
    grid_n: int,
    warp_h: int,
    warp_w: int,
    robot_id: int,
) -> Tuple[bool, float]:
    """Return (outside_bbox, euclidean_px to nearest bbox edge; 0 if inside)."""
    px, py = cell_center_warp_px(cell, grid_n, warp_h, warp_w, robot_id)
    if bbox.contains(px, py):
        return False, 0.0
    cx = min(max(px, bbox.x1), bbox.x2)
    cy = min(max(py, bbox.y1), bbox.y2)
    return True, float(np.hypot(px - cx, py - cy))


def dist_cell_to_bbox_cells(
    cell: int,
    bbox: BBox,
    grid_n: int,
    invalid_mask: Optional[np.ndarray],
    warp_h: int,
    warp_w: int,
    robot_id: int,
    board_cfg: Optional[Dict] = None,
) -> Tuple[bool, Optional[int]]:
    """Return (outside_bbox, chebyshev dist to nearest in-bbox cell; 0 if inside)."""
    px, py = cell_center_warp_px(cell, grid_n, warp_h, warp_w, robot_id)
    if bbox.contains(px, py):
        return False, 0
    inside = cells_in_bbox(
        bbox, grid_n, invalid_mask, warp_h, warp_w, robot_id, board_cfg=board_cfg,
    )
    if not inside:
        return True, None
    r0, c0 = divmod(int(cell), grid_n)
    best: Optional[int] = None
    for c in inside:
        r, col = divmod(c, grid_n)
        d = max(abs(r0 - r), abs(c0 - col))
        if best is None or d < best:
            best = d
    return True, best


class LocalYoloLocator:
    """Run Ultralytics YOLO on warped RGB (H×W×3 uint8)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.warp_size = int(self.cfg.get("warp_size", 224))
        self.grid_n = int(self.cfg.get("grid_n", 112))
        self.min_confidence = float(self.cfg.get("min_confidence", 0.10))
        self.class_name = self.cfg.get("class_name")
        self.conf = float(self.cfg.get("conf", 0.25))
        self.imgsz = int(self.cfg.get("imgsz", self.warp_size))
        self.weights_path = _resolve_weights_path(self.cfg)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.weights_path}. "
                "Run analysis/train_block_yolo.py first."
            )
        from ultralytics import YOLO
        self._model = YOLO(str(self.weights_path))
        print(f"[YOLO] loaded local weights: {self.weights_path}", flush=True)

    def detect_bbox(self, rgb_w: np.ndarray) -> Tuple[Optional[BBox], Any]:
        """Returns (bbox, raw_result dict)."""
        if rgb_w.dtype != np.uint8:
            rgb_u8 = np.clip(rgb_w, 0, 255).astype(np.uint8)
        else:
            rgb_u8 = rgb_w
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            source=bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            max_det=1,
            verbose=False,
        )
        raw: Dict[str, Any] = {"predictions": [], "weights": str(self.weights_path)}
        if not results:
            return None, raw
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None, raw

        box = r0.boxes[0]
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
        names = getattr(r0, "names", {}) or {}
        cls_name = str(names.get(cls_id, "block"))

        x1, y1, x2, y2 = [float(v) for v in xyxy]
        s = float(self.warp_size)
        x1 = min(max(x1, 0.0), s - 1.0)
        y1 = min(max(y1, 0.0), s - 1.0)
        x2 = min(max(x2, 0.0), s - 1.0)
        y2 = min(max(y2, 0.0), s - 1.0)
        if x2 <= x1 or y2 <= y1:
            return None, raw

        bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_name=cls_name)
        raw["predictions"] = [{
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf, "class": cls_name, "class_id": cls_id,
        }]
        filtered = _filter_bbox(
            bbox,
            min_confidence=self.min_confidence,
            class_name=self.class_name,
        )
        return filtered, raw


YoloLocator = LocalYoloLocator
