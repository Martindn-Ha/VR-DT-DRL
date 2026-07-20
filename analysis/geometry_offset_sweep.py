#!/usr/bin/env python3
"""
Grid-search REACH/SHIFT/HEIGHT offsets on oracle spawns (teacher object position).

Use after locator improves but geo grasps still miss — finds better constants per robot
without retraining CNN. Does not modify live code; prints best offsets to apply in
grasp_geometry.py.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SIM_SRC = _REPO / "vm_simulation_system" / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))

from grasp_geometry import (  # noqa: E402
    DEFAULT_OBJECT_Y_M,
    HEIGHT_OFFSET,
    REACH_OFFSET,
    SHIFT_OFFSET,
    compute_grasp_pose_from_object_world,
)
from spawn_geometry import quadrant_from_spawn  # noqa: E402

PLATFORM_CENTER = {1: (-0.646, 0.841), 2: (-1.365, 0.850)}


def pose_xy_error(pose: List[float], spawn_x: float, spawn_z: float) -> float:
  """Proxy: distance from grasp XY to spawn XY (lower = closer aim)."""
  return math.hypot(pose[0] - spawn_x, pose[2] - spawn_z)


def compute_with_offsets(
    obj_x: float, obj_z: float, robot_id: int,
    reach: float, shift: float, height: float,
) -> List[float]:
  import grasp_geometry as gg
  old_r, old_s, old_h = gg.REACH_OFFSET, gg.SHIFT_OFFSET, gg.HEIGHT_OFFSET
  try:
    gg.REACH_OFFSET = reach
    gg.SHIFT_OFFSET = shift
    gg.HEIGHT_OFFSET = height
    return compute_grasp_pose_from_object_world(
        obj_x, DEFAULT_OBJECT_Y_M, obj_z, robot_id, add_jitter=False,
    )
  finally:
    gg.REACH_OFFSET, gg.SHIFT_OFFSET, gg.HEIGHT_OFFSET = old_r, old_s, old_h


def sweep_quadrant(
    df: pd.DataFrame, robot_id: int, quadrant: int,
    reach_vals: List[float], shift_vals: List[float], height_vals: List[float],
) -> Tuple[float, float, float, float]:
  cx, cz = PLATFORM_CENTER[robot_id]
  sub = df[df.apply(
      lambda r: quadrant_from_spawn(r.spawn_x, r.spawn_z, cx, cz) == quadrant, axis=1
  )]
  if sub.empty:
    raise ValueError(f"No rows for Q{quadrant}")
  best = (1e9, REACH_OFFSET, SHIFT_OFFSET, HEIGHT_OFFSET)
  for reach, shift, height in itertools.product(reach_vals, shift_vals, height_vals):
    errs = []
    for r in sub.itertuples():
      pose = compute_with_offsets(r.spawn_x, r.spawn_z, robot_id, reach, shift, height)
      errs.append(pose_xy_error(pose, r.spawn_x, r.spawn_z))
    med = float(pd.Series(errs).median())
    if med < best[0]:
      best = (med, reach, shift, height)
  return best


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("xlsx", type=Path, help="Geo eval log with spawn_x/z")
  p.add_argument("--robot-id", type=int, required=True, choices=[1, 2])
  p.add_argument("--quadrant", type=int, default=2, help="Focus quadrant (default Q2)")
  p.add_argument("--reach-delta-mm", type=float, default=10.0)
  p.add_argument("--shift-delta-mm", type=float, default=10.0)
  p.add_argument("--height-delta-mm", type=float, default=5.0)
  p.add_argument("--step-mm", type=float, default=5.0)
  args = p.parse_args()

  df = pd.read_excel(args.xlsx.resolve())
  df["spawn_x"] = pd.to_numeric(df["spawn_x"], errors="coerce")
  df["spawn_z"] = pd.to_numeric(df["spawn_z"], errors="coerce")
  df = df.dropna(subset=["spawn_x", "spawn_z"])

  step = args.step_mm / 1000.0
  rd = args.reach_delta_mm / 1000.0
  sd = args.shift_delta_mm / 1000.0
  hd = args.height_delta_mm / 1000.0

  reach_vals = [REACH_OFFSET + i * step for i in range(int(-rd / step), int(rd / step) + 1)]
  shift_vals = [SHIFT_OFFSET + i * step for i in range(int(-sd / step), int(sd / step) + 1)]
  height_vals = [HEIGHT_OFFSET + i * step for i in range(int(-hd / step), int(hd / step) + 1)]

  med, reach, shift, height = sweep_quadrant(
      df, args.robot_id, args.quadrant, reach_vals, shift_vals, height_vals,
  )
  print(f"R{args.robot_id} Q{args.quadrant} oracle XY proxy (median err {med*1000:.1f}mm)")
  print(f"  REACH_OFFSET={reach:.4f}  SHIFT_OFFSET={shift:.4f}  HEIGHT_OFFSET={height:.4f}")
  print(f"  (defaults: {REACH_OFFSET}, {SHIFT_OFFSET}, {HEIGHT_OFFSET})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
