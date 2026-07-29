"""Ollama VLM: point on board warp → nearest YOLO bbox (inference language select)."""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from yolo_locator import BBox

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_VLM_MODEL = "qwen3-vl:8b"
DEFAULT_TIMEOUT_S = 60.0

_SYSTEM_PROMPT = """You select a block on a top-down warped board image.
Left/right/top/bottom mean sides of THIS image.
Image quadrants (equal corners of the image):
  Q1 = top-left, Q2 = top-right, Q3 = bottom-left, Q4 = bottom-right.
The image is exactly {w}x{h} pixels.
Reply with ONLY one point on the referred block as: u,v
Use NORMALIZED coordinates only: both u and v must be floats in [0, 1].
Origin is top-left (u right, v down). Example: center is 0.5,0.5
Do NOT use pixel coordinates. Do NOT use values above 1.
If you cannot match the instruction, reply exactly: NONE
Do not explain. Do not use thinking. One line only."""


class VlmUnavailableError(RuntimeError):
    """Ollama down, timeout, or HTTP failure."""


class VlmSelectFailedError(RuntimeError):
    """Bad parse, NONE, out of image, or point too far from boxes."""


@dataclass(frozen=True)
class VlmSelectResult:
    bbox: BBox
    point_uv: Tuple[float, float]
    raw_text: str
    distance_px: float


def _bbox_center(b: BBox) -> Tuple[float, float]:
    return 0.5 * (b.x1 + b.x2), 0.5 * (b.y1 + b.y2)


def _bbox_half_diag(b: BBox) -> float:
    return 0.5 * math.hypot(b.x2 - b.x1, b.y2 - b.y1)


def nearest_bbox(
    point_uv: Tuple[float, float],
    boxes: Sequence[BBox],
    *,
    min_floor_px: float = 8.0,
) -> Tuple[BBox, float]:
    """Pick box with nearest center. Raises VlmSelectFailedError if too far."""
    if not boxes:
        raise VlmSelectFailedError("no YOLO boxes for nearest match")
    u, v = point_uv
    best: Optional[BBox] = None
    best_d = float("inf")
    for b in boxes:
        cx, cy = _bbox_center(b)
        d = math.hypot(u - cx, v - cy)
        if d < best_d:
            best_d = d
            best = b
    assert best is not None
    thresh = max(min_floor_px, _bbox_half_diag(best))
    if best_d > thresh:
        raise VlmSelectFailedError(
            f"point ({u:.1f},{v:.1f}) too far from nearest box "
            f"(dist={best_d:.1f}px > {thresh:.1f}px)"
        )
    return best, best_d


def parse_vlm_point(text: str, image_w: int, image_h: int) -> Tuple[float, float]:
    """Parse NONE or normalized u,v in [0,1]. Raises VlmSelectFailedError."""
    raw = (text or "").strip()
    if not raw:
        raise VlmSelectFailedError("empty VLM reply")
    # Drop common thinking / fence noise; keep last non-empty line preference.
    lines = [ln.strip() for ln in raw.replace("\r", "").split("\n") if ln.strip()]
    candidate = lines[-1] if lines else raw
    upper = candidate.upper()
    if upper == "NONE" or upper.startswith("NONE"):
        raise VlmSelectFailedError("VLM replied NONE")

    m = re.search(
        r"([+-]?\d*\.?\d+)\s*[,;\s]\s*([+-]?\d*\.?\d+)",
        candidate,
    )
    if not m:
        m = re.search(
            r"([+-]?\d*\.?\d+)\s*[,;\s]\s*([+-]?\d*\.?\d+)",
            raw,
        )
    if not m:
        raise VlmSelectFailedError(f"could not parse point from: {raw!r}")

    u = float(m.group(1))
    v = float(m.group(2))
    # Require normalized [0, 1] only (prompt asks for this).
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise VlmSelectFailedError(
            f"expected normalized u,v in [0,1], got ({u},{v})"
        )
    u = u * float(image_w - 1)
    v = v * float(image_h - 1)

    if not (0.0 <= u < float(image_w) and 0.0 <= v < float(image_h)):
        raise VlmSelectFailedError(
            f"point ({u:.1f},{v:.1f}) outside image {image_w}x{image_h}"
        )
    return u, v


def _rgb_to_jpeg_b64(rgb_w: np.ndarray, quality: int = 90) -> str:
    if rgb_w.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_w, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb_w
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise VlmUnavailableError("failed to encode warp JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ollama_chat(
    *,
    ollama_url: str,
    model: str,
    instruction: str,
    image_b64: str,
    image_w: int,
    image_h: int,
    timeout_s: float,
) -> str:
    url = ollama_url.rstrip("/") + "/api/chat"
    system = _SYSTEM_PROMPT.format(w=int(image_w), h=int(image_h))
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Image size: {int(image_w)}x{int(image_h)}. "
                    f"Instruction: {instruction}\n"
                    "Reply with only normalized u,v in [0,1] (example 0.5,0.5) or NONE."
                ),
                "images": [image_b64],
            },
        ],
        "options": {"temperature": 0.1},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise VlmUnavailableError(f"Ollama request failed: {exc}") from exc
    except TimeoutError as exc:
        raise VlmUnavailableError(f"Ollama timeout after {timeout_s}s") from exc

    msg = payload.get("message") or {}
    content = msg.get("content")
    if content is None:
        content = payload.get("response")
    if not isinstance(content, str):
        raise VlmUnavailableError(f"unexpected Ollama response: {payload!r}")
    return content


def select_bbox_by_vlm_point(
    rgb_w: np.ndarray,
    instruction: str,
    boxes: Sequence[BBox],
    *,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_VLM_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> VlmSelectResult:
    """Full warp + instruction → VLM point → nearest YOLO box."""
    instr = (instruction or "").strip()
    if not instr:
        raise VlmSelectFailedError("empty instruction")
    if not boxes:
        raise VlmSelectFailedError("no YOLO boxes")

    h, w = int(rgb_w.shape[0]), int(rgb_w.shape[1])
    image_b64 = _rgb_to_jpeg_b64(rgb_w)
    raw_text = _ollama_chat(
        ollama_url=ollama_url,
        model=model,
        instruction=instr,
        image_b64=image_b64,
        image_w=w,
        image_h=h,
        timeout_s=timeout_s,
    )
    point = parse_vlm_point(raw_text, w, h)
    bbox, dist = nearest_bbox(point, boxes)
    return VlmSelectResult(
        bbox=bbox, point_uv=point, raw_text=raw_text, distance_px=dist,
    )
