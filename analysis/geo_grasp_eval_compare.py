#!/usr/bin/env python3
"""Compare geo-grasp phase-5 eval logs (pre vs post rebalance)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SIM_SRC = _REPO / "vm_simulation_system" / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))

from spawn_geometry import quadrant_from_spawn  # noqa: E402

PLATFORM_CENTER = {1: (-0.646, 0.841), 2: (-1.365, 0.850)}
ERR_BINS = [0, 0.005, 0.010, 0.015, 0.020, 0.030, 1.0]
ERR_LABELS = ["0-5mm", "5-10mm", "10-15mm", "15-20mm", "20-30mm", "30mm+"]


def load_eval(path: Path, robot_id: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "grasp_mode" in df.columns:
        df = df[df["grasp_mode"] == "exploit_geo"]
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


def compare_robot(pre: Path, post: Path, robot_id: int) -> None:
    print(f"\n===== Robot {robot_id} =====")
    d_pre = load_eval(pre, robot_id)
    d_post = load_eval(post, robot_id)
    s_pre = summarize(d_pre, "pre-bias")
    s_post = summarize(d_post, "post")
    print_summary(s_pre)
    print_summary(s_post)
    delta = s_post["success_pct"] - s_pre["success_pct"]
    print(f"  delta success: {delta:+.1f} pp")
    if robot_id == 2:
        d2 = s_post["q2_success_pct"] - s_pre["q2_success_pct"]
        print(f"  delta Q2:      {d2:+.1f} pp (target > 25.5% pre baseline)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pre-dir", type=Path, required=True, help="Pre-bias Eval folder")
    p.add_argument("--post-dir", type=Path, required=True, help="Post-rebalance Eval folder")
    p.add_argument("--robot-id", type=int, choices=[1, 2], default=None)
    args = p.parse_args()
    robots = [args.robot_id] if args.robot_id else [1, 2]
    for rid in robots:
        pre = args.pre_dir / f"episode_log_r{rid}_phase5.xlsx"
        post = args.post_dir / f"episode_log_r{rid}_phase5.xlsx"
        if not pre.exists():
            print(f"Missing {pre}")
            return 1
        if not post.exists():
            print(f"Missing {post} — run phase 5 geo eval first.")
            return 1
        compare_robot(pre, post, rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
