#!/usr/bin/env python3
"""Plain-English readout for board_warp_rN_latest.jpg sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain the latest board warp debug overlay (teacher vs CNN)."
    )
    parser.add_argument("--robot-id", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    debug_dir = _REPO / "host_gpu_system" / "debug"
    stem = f"board_warp_r{args.robot_id}_latest"
    jpg = debug_dir / f"{stem}.jpg"
    meta_path = debug_dir / f"{stem}.json"

    if not jpg.exists():
        print(f"No debug image yet: {jpg}")
        print("Run one board_dqn_train episode with gpu_server.py connected.")
        return 1

    print(f"Image: {jpg}")
    if not meta_path.exists():
        print("No sidecar JSON — restart gpu_server with latest code, run one episode.")
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    teacher = meta.get("teacher_cell")
    cnn = meta.get("cnn_cell")
    gap = meta.get("cell_index_gap")
    hit = meta.get("cnn_hit")

    print()
    print(f"  Object (sim): x={meta.get('label_x')}  z={meta.get('label_z')}")
    tw = meta.get("teacher_warp_px")
    if tw:
        print(f"  RED cross:    warp pixel ({tw[0]}, {tw[1]})  teacher cell {teacher}")
    else:
        print(f"  RED cross:    teacher cell {teacher}  (where sim XZ maps on grid)")
    print(f"  GREEN cross:  CNN cell {cnn}  (network pick)")
    if hit is not None:
        print(f"  Same cell?    {'YES (HIT)' if hit else 'NO (miss)'}  gap={gap} index units")
    print()

    q_keys = (
        "q_teacher", "q_argmax", "q_argmax_over_teacher", "teacher_rank",
        "teacher_rank_pct", "mass_near_teacher", "topk_centroid_err_m", "argmax_err_m",
    )
    if any(meta.get(k) is not None for k in q_keys):
        print("Q-map diagnostics (sim teacher known):")
        if meta.get("q_teacher") is not None:
            print(f"  Q at teacher cell:  {meta.get('q_teacher'):.4f}")
        if meta.get("q_argmax") is not None:
            print(f"  Q at argmax cell:   {meta.get('q_argmax'):.4f}")
        if meta.get("q_argmax_over_teacher") is not None:
            print(f"  Argmax/teacher Q:   {meta.get('q_argmax_over_teacher'):.3f}x")
        if meta.get("teacher_rank") is not None:
            pct = meta.get("teacher_rank_pct")
            pct_s = f"  ({pct:.2f}% percentile)" if pct is not None else ""
            print(f"  Teacher rank:       {int(meta['teacher_rank'])}{pct_s}")
        if meta.get("mass_near_teacher") is not None:
            print(f"  Mass near teacher:  {meta.get('mass_near_teacher'):.3f}  (softmax within 2 cells)")
        if meta.get("topk_centroid_err_m") is not None:
            print(f"  Top-5 centroid err: {meta.get('topk_centroid_err_m')*100:.1f} cm")
        if meta.get("argmax_err_m") is not None:
            print(f"  Argmax aim err:     {meta.get('argmax_err_m')*100:.1f} cm")
        print()
        mass = meta.get("mass_near_teacher")
        rank = meta.get("teacher_rank")
        rank_pct = meta.get("teacher_rank_pct")
        if mass is not None and rank is not None:
            if mass > 0.3 and (rank_pct is None or rank_pct <= 1.0):
                print("  Interpretation: heat clusters on/near block; top-K centroid may help vs argmax.")
            elif mass <= 0.3 or (rank_pct is not None and rank_pct > 10.0):
                print("  Interpretation: weak heat on block; argmax miss likely not fixed by top-K alone.")
            else:
                print("  Interpretation: mixed — check board_q_heatmap image for spatial pattern.")
        heat = debug_dir / f"board_q_heatmap_r{args.robot_id}_latest.jpg"
        if heat.exists():
            print(f"  Heatmap: {heat}")
        print()

    print("How to read the picture:")
    print("  1. On board_warp_rN_latest.jpg: click the block — compare to teacher_warp_px in JSON.")
    print("     Far apart -> fix board_warp_rN.yaml corners (see board_crop_rN_latest.jpg).")
    print("  2. On board_crop_rN_latest.jpg: RED must sit on the block inside the yellow quad.")
    print("     RED off block -> re-click the 4 SpawnArea corners in the crop image.")
    print("  3. GREEN on block too? -> CNN is aiming correctly this frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
