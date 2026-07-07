#!/usr/bin/env python3
"""
Verify grid cell labels: object center vs locator-centered grid.

Grasp aim XY is ~14 cm from object center (geometry offsets) — labels must use spawn/object XZ,
not teacher_grasp_xy, or every sample is falsely out_of_window.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SIM_SRC = Path(__file__).resolve().parent.parent / "vm_simulation_system" / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))

from local_grid import world_to_cell, cell_to_world, teacher_grasp_xy  # noqa: E402
from grasp_geometry import compute_grasp_pose_from_object_world  # noqa: E402


def main() -> int:
    n, window = 5, 0.04
    spawn_x, spawn_z = -0.70, 0.85
    pose = compute_grasp_pose_from_object_world(spawn_x, 0.461, spawn_z, 1, add_jitter=False)
    tx, tz = teacher_grasp_xy(pose)
    geo_off = math.hypot(tx - spawn_x, tz - spawn_z)

    # Correct: label from object center, grid centered on locator at spawn
    c0 = world_to_cell(spawn_x, spawn_z, spawn_x, spawn_z, n=n, window_m=window)
    assert c0 == 12, f"center expected 12, got {c0}"

    # Locator offset 1 cm — off-center cell, still in window
    loc_x, loc_z = spawn_x - 0.01, spawn_z
    c1 = world_to_cell(spawn_x, spawn_z, loc_x, loc_z, n=n, window_m=window)
    assert c1 is not None and c1 != 12, "offset locator should yield off-center cell"

    wx, wz = cell_to_world(c1, loc_x, loc_z, n=n, window_m=window)
    assert abs(wx - spawn_x) < 0.01 and abs(wz - spawn_z) < 0.01

    # Old bug: grasp aim vs locator at spawn → always out of window
    wrong = world_to_cell(tx, tz, spawn_x, spawn_z, n=n, window_m=window)
    assert wrong is None, "grasp XY must not be used for cell labels"

    print(f"spawn=({spawn_x},{spawn_z}) grasp_xy=({tx:.4f},{tz:.4f}) geo_offset={geo_off*1000:.1f}mm")
    print(f"label_cell={c1} (locator offset 1cm; center would be 12)")
    print("teacher_cell_label_verify: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
