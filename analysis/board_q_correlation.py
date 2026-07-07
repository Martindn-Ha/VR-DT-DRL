#!/usr/bin/env python3
"""Batch correlation: board Q diagnostics vs grasp success in episode logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent

Q_COLS = (
    "q_teacher",
    "q_argmax",
    "teacher_rank",
    "mass_near_teacher",
    "topk_centroid_err_m",
    "argmax_err_m",
)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pearson(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(x[mask].corr(y[mask]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate board Q-map diagnostics with grasp success."
    )
    parser.add_argument(
        "xlsx",
        nargs="?",
        default=str(_REPO / "data" / "episode_log_r1.xlsx"),
        help="Episode log xlsx path",
    )
    args = parser.parse_args()

    path = Path(args.xlsx)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    df = pd.read_excel(path, engine="openpyxl")
    if "grasp_mode" in df.columns:
        df = df[df["grasp_mode"].astype(str).str.lower() == "exploit_board"]
    for col in Q_COLS:
        if col in df.columns:
            df[col] = _to_num(df[col])
    if "success" in df.columns:
        df["success"] = _to_num(df["success"]).fillna(0).astype(int)
    else:
        print("No success column — cannot bin by outcome.")
        return 1

    has_q = df["mass_near_teacher"].notna() if "mass_near_teacher" in df.columns else pd.Series(False, index=df.index)
    sub = df[has_q].copy()
    if sub.empty:
        print("No rows with Q diagnostics (run inference with updated gpu_server + sim client).")
        return 1

    print(f"Rows: {len(sub)} exploit_board with Q fields (of {len(df)} filtered)")
    print()
    print("Summary (mean / median):")
    for col in Q_COLS:
        if col not in sub.columns:
            continue
        s = sub[col].dropna()
        if s.empty:
            continue
        print(f"  {col:22s}  mean={s.mean():.4f}  median={s.median():.4f}")

    if "mass_near_teacher" in sub.columns:
        tert = pd.qcut(sub["mass_near_teacher"], 3, duplicates="drop")
        print()
        print("Success rate by mass_near_teacher tertile:")
        for label, grp in sub.groupby(tert, observed=True):
            rate = grp["success"].mean()
            print(f"  {label}: n={len(grp)}  success={rate*100:.1f}%")

    if "topk_centroid_err_m" in sub.columns and "argmax_err_m" in sub.columns:
        r_topk = _pearson(sub["topk_centroid_err_m"], sub["success"])
        r_argmax = _pearson(sub["argmax_err_m"], sub["success"])
        print()
        print("Correlation with success (negative = lower error → more success):")
        print(f"  topk_centroid_err_m:  r={r_topk:.3f}")
        print(f"  argmax_err_m:         r={r_argmax:.3f}")
        if not np.isnan(r_topk) and not np.isnan(r_argmax):
            better = "top-K centroid" if abs(r_topk) > abs(r_argmax) else "argmax cell"
            print(f"  → {better} tracks success slightly better in this log.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
