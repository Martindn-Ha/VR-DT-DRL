#!/usr/bin/env python3
"""Compare local-grid (exploit_grid) phase-5 eval vs geo-grasp baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SIM_SRC = _REPO / "vm_simulation_system" / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))

from spawn_geometry import quadrant_from_spawn  # noqa: E402

PLATFORM_CENTER = {1: (-0.646, 0.841), 2: (-1.365, 0.850)}


def load_eval(path: Path, robot_id: int, grasp_mode: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "grasp_mode" in df.columns:
        df = df[df["grasp_mode"] == grasp_mode]
    cx, cz = PLATFORM_CENTER[robot_id]
    df = df.copy()
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    df["locator_err_m"] = pd.to_numeric(df.get("locator_err_m"), errors="coerce")
    df["spawn_x"] = pd.to_numeric(df["spawn_x"], errors="coerce")
    df["spawn_z"] = pd.to_numeric(df["spawn_z"], errors="coerce")
    df["q"] = [
        quadrant_from_spawn(x, z, cx, cz)
        for x, z in zip(df["spawn_x"], df["spawn_z"])
    ]
    return df


def summarize(df: pd.DataFrame, label: str) -> dict:
    succ = df["success"].mean() * 100 if len(df) else float("nan")
    fail = df[df["success"] == 0]
    fail_med = fail["locator_err_m"].median() * 1000 if len(fail) else float("nan")
    q2 = df[df["q"] == 2]["success"].mean() * 100 if (df["q"] == 2).any() else float("nan")
    return {
        "label": label,
        "n": len(df),
        "success_pct": succ,
        "q2_success_pct": q2,
        "fail_locator_med_mm": fail_med,
    }


def print_summary(stats: dict) -> None:
    print(
        f"  {stats['label']:12} n={stats['n']:4d}  "
        f"success={stats['success_pct']:5.1f}%  "
        f"Q2={stats['q2_success_pct']:5.1f}%  "
        f"fail_loc_med={stats['fail_locator_med_mm']:5.1f}mm"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path, help="Geo-grasp baseline xlsx (exploit_geo)")
    p.add_argument("candidate", type=Path, help="Local grid eval xlsx (exploit_grid)")
    p.add_argument("--robot-id", type=int, default=1, choices=[1, 2])
    args = p.parse_args()

    print(f"\n===== Robot {args.robot_id} (geo vs local grid) =====")
    d_geo = load_eval(args.baseline, args.robot_id, "exploit_geo")
    d_grid = load_eval(args.candidate, args.robot_id, "exploit_grid")
    s_geo = summarize(d_geo, "exploit_geo")
    s_grid = summarize(d_grid, "exploit_grid")
    print_summary(s_geo)
    print_summary(s_grid)
    print(f"  delta success: {s_grid['success_pct'] - s_geo['success_pct']:+.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
