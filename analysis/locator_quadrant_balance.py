#!/usr/bin/env python3
"""Report quadrant balance in locator_train or geo-grasp episode logs."""

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
MIN_PER_QUADRANT = 200
MIN_SHARE_PCT = 15.0


def enrich(df: pd.DataFrame, robot_id: int) -> pd.DataFrame:
    cx, cz = PLATFORM_CENTER[int(robot_id)]
    out = df.copy()
    out["spawn_x"] = pd.to_numeric(out["spawn_x"], errors="coerce")
    out["spawn_z"] = pd.to_numeric(out["spawn_z"], errors="coerce")
    out["quadrant"] = [
        quadrant_from_spawn(x, z, cx, cz)
        for x, z in zip(out["spawn_x"], out["spawn_z"])
    ]
    return out


def report(path: Path, robot_id: int | None = None) -> int:
    df = pd.read_excel(path)
    rid = robot_id or int(df["robot_id"].iloc[0])
    df = enrich(df, rid)
    total = len(df)
    print(f"\n{path.name}  robot=R{rid}  rows={total}\n")
    print(f"{'Q':>3} {'count':>6} {'share':>7} {'min200':>8}")
    ok = True
    for q in sorted(df["quadrant"].dropna().unique()):
        n = int((df["quadrant"] == q).sum())
        share = 100.0 * n / total if total else 0.0
        flag = "OK" if n >= MIN_PER_QUADRANT and share >= MIN_SHARE_PCT else "LOW"
        if flag == "LOW":
            ok = False
        print(f"Q{int(q):>2} {n:>6} {share:>6.1f}% {flag:>8}")
    if ok:
        print("\nPASS: all quadrants >= 200 rows and >= 15% share.")
        return 0
    print(f"\nFAIL: target >= {MIN_PER_QUADRANT} per quadrant and >= {MIN_SHARE_PCT}% share.")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xlsx", type=Path, help="episode_log_r*_locator_train.xlsx or phase5 eval")
    p.add_argument("--robot-id", type=int, default=None)
    args = p.parse_args()
    return report(args.xlsx.resolve(), args.robot_id)


if __name__ == "__main__":
    raise SystemExit(main())
