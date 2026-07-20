#!/usr/bin/env python3
"""Train local Ultralytics YOLO on Block Dataset.yolo26 warp export."""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO / "Block Dataset.yolo26" / "data.yaml"
_DEFAULT_PROJECT = _REPO / "host_gpu_system" / "models"
_DEFAULT_NAME = "block_yolo"


def _resolve_base_model(preferred: str) -> str:
    from ultralytics import YOLO
    for name in (preferred, "yolo26n.pt", "yolov8n.pt"):
        try:
            YOLO(name)
            return name
        except Exception:
            continue
    return "yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local block YOLO on warp dataset.")
    parser.add_argument("--data", type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--base-model", type=str, default="yolo26n.pt")
    parser.add_argument("--project", type=str, default=str(_DEFAULT_PROJECT))
    parser.add_argument("--name", type=str, default=_DEFAULT_NAME)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset yaml not found: {data_path}")

    from ultralytics import YOLO
    base = _resolve_base_model(args.base_model)
    print(f"Training from base model: {base}")
    print(f"Data: {data_path.resolve()}")

    model = YOLO(base)
    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Done. Best weights: {best.resolve()}")


if __name__ == "__main__":
    main()
