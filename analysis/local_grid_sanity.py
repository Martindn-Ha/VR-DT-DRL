#!/usr/bin/env python3
"""Sanity checks for local_grid world↔cell mapping."""

from __future__ import annotations

import sys
from pathlib import Path

_SIM_SRC = Path(__file__).resolve().parent.parent / "vm_simulation_system" / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))

from local_grid import cell_to_world, teacher_grasp_xy, world_to_cell  # noqa: E402


def main() -> int:
    cx, cz = -0.70, 0.85
    n, window = 5, 0.04

    center_cell = world_to_cell(cx, cz, cx, cz, n=n, window_m=window)
    assert center_cell == 12, f"center expected 12 got {center_cell}"

    x, z = cell_to_world(12, cx, cz, n=n, window_m=window)
    assert abs(x - cx) < 1e-6 and abs(z - cz) < 1e-6

    off_cell = world_to_cell(cx + 0.01, cz - 0.01, cx, cz, n=n, window_m=window)
    assert off_cell is not None and off_cell != center_cell

    outside = world_to_cell(cx + 0.03, cz, cx, cz, n=n, window_m=window)
    assert outside is None, "expected out of window"

    tx, tz = teacher_grasp_xy([cx, 0.48, cz, 3.14, 0.0, 0.0])
    assert (tx, tz) == (cx, cz)

    print("local_grid sanity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
