#!/usr/bin/env python3
"""
Episode log report — spatial analysis + failure taxonomy (PDF).

Pages:
  1. Cover (source file and report datetime)
  2. Failure taxonomy color key (outcome_class definitions)
  3. Failure taxonomy counts for this log
  4. Spawn locations by success/failure (top-down, robot view)
  5. Spawn locations by outcome type (top-down, robot view)
  6. Distance from platform (near/mid/far) — success vs failure counts
  7. Distance — outcome_class breakdown
  8. Arc (8 directions, robot frame) — success vs failure counts
  9. Arc — outcome_class breakdown
  10. Quadrant (I–IV, 90° bins) — success vs failure counts
  11. Quadrant — outcome_class breakdown

Run:
    python spawn_spatial_report.py              # full PDF report, r1
    python spawn_spatial_report.py r2           # full PDF report, r2
    python spawn_spatial_report.py path.xlsx
    python spawn_spatial_report.py path.xlsx --taxonomy-only
    python spawn_spatial_report.py path.xlsx --output-csv out.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"

_SIM_SRC = REPO_ROOT / "vm_simulation_system" / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))

from failure_taxonomy import (  # noqa: E402
    CLAMP_EPS,
    DROP_LIFT,
    FAR_MISS_DIST,
    NAN_DIST_SENTINEL,
    NEAR_MISS_DIST,
    NEAR_MISS_LIFT,
    OUTCOME_CLASSES,
    REQUIRED_LIFT,
    classify_outcome_from_row,
    is_clamp_limited_from_row,
)

DATA_SEARCH_DIRS = (
    REPO_ROOT / "data" / "episode logs",
    REPO_ROOT / "data",
    REPO_ROOT / "vm_simulation_system" / "data",
    BASE_DIR,
)

# Loose spawn sanity bounds (world frame, meters) — drops misaligned CSV rows.
SPAWN_BOUNDS_BY_ROBOT = {
    1: {"spawn_x": (-0.95, -0.40), "spawn_z": (0.65, 1.10), "spawn_y": (0.40, 0.50)},
    2: {"spawn_x": (-1.55, -1.00), "spawn_z": (0.65, 1.10), "spawn_y": (0.40, 0.50)},
}

TARGET = "success"
REGION_LABELS = ["near", "mid", "far"]
PLATFORM_CENTER_BY_ROBOT = {1: (-0.646, 0.841), 2: (-1.365, 0.850)}

COLOR_SUCCESS = "#2ca02c"
COLOR_FAILURE = "#d62728"

OUTCOME_COLORS = {
    "success": COLOR_SUCCESS,
    "near_miss": "#ff7f0e",
    "mid_miss": "#17becf",
    "weak_lift": "#ffbb78",
    "drop_or_push": COLOR_FAILURE,
    "far_miss": "#9467bd",
    "object_not_found": "#8c564b",
    "sim_nan_abort": "#7f7f7f",
    "other_failure": "#bcbd22",
}

def _mm(meters: float) -> str:
    return f"{meters * 1000:.0f} mm"


# Matches classify_outcome() in vm_simulation_system/src/failure_taxonomy.py (first match wins).
OUTCOME_DEFINITIONS = {
    "success": f"success = 1 (lifted_m > {_mm(REQUIRED_LIFT)} block ΔY).",
    "sim_nan_abort": (
        f"grasp_mode = nan_abort, or closest_dist_m ≥ {NAN_DIST_SENTINEL:.0f}."
    ),
    "object_not_found": "object_found = 0.",
    "far_miss": f"Failed; closest_dist_m > {_mm(FAR_MISS_DIST)} (gripper–block).",
    "weak_lift": (
        f"Failed; 0 < lifted_m ≤ {_mm(REQUIRED_LIFT)} (lifted, below success bar)."
    ),
    "drop_or_push": "Failed; lifted_m < 0 mm (block ΔY downward).",
    "near_miss": (
        f"Failed; closest_dist_m ≤ {_mm(NEAR_MISS_DIST)} and "
        f"|lifted_m| < {_mm(NEAR_MISS_LIFT)}."
    ),
    "mid_miss": (
        f"Failed; {_mm(NEAR_MISS_DIST)} < closest_dist_m ≤ {_mm(FAR_MISS_DIST)}."
    ),
    "other_failure": (
        "Failed; close + |lifted_m| ≥ 20 mm (not weak/drop), or bad/missing fields."
    ),
}

CLAMP_DEFINITIONS = {
    "0": f"|ai_pose − clamp_pose| on X/Z ≤ {CLAMP_EPS:g} m (no clip).",
    "1": f"|ai_pose − clamp_pose| on X/Z > {CLAMP_EPS:g} m (X/Z clipped before move).",
}

ARC_BINS = [0, 45, 90, 135, 180, 225, 270, 315, 360]
ARC_LABELS = [
    "0-45", "45-90", "90-135", "135-180",
    "180-225", "225-270", "270-315", "315-360",
]
ARC_LABELS_ORDER = ARC_LABELS
QUADRANT_LABELS = ["I", "II", "III", "IV"]
QUADRANT_DISPLAY = [
    "I\n0–90°",
    "II\n90–180°",
    "III\n180–270°",
    "IV\n270–360°",
]
ROBOT_FACING_COLOR = "#222222"
ROBOT_ARC_XLABEL = "Arc (° clockwise from forward +spawn_z)"
ROBOT_QUADRANT_XLABEL = "Quadrant (θ, degrees)"
ROBOT_MAP_XLABEL = "Δspawn_x from platform (cm); axis mirrored for robot view"
ROBOT_MAP_YLABEL = "Δspawn_z from platform (cm)"


def angle_robot_deg(dx_m, dz_m) -> np.ndarray:
    dx = np.asarray(dx_m, dtype=float)
    dz = np.asarray(dz_m, dtype=float)
    return (np.degrees(np.arctan2(dz, -dx)) + 360) % 360


def add_robot_angle_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "angle_deg" not in out.columns:
        out["angle_deg"] = (
            np.degrees(np.arctan2(out["dz_platform_m"], out["dx_platform_m"])) + 360
        ) % 360
    out["angle_robot_deg"] = angle_robot_deg(out["dx_platform_m"], out["dz_platform_m"])
    return out


def assign_direction_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = add_robot_angle_columns(df)
    out["arc_sim"] = pd.cut(
        out["angle_deg"], bins=ARC_BINS, labels=ARC_LABELS,
        include_lowest=True, right=False,
    )
    out["arc"] = pd.cut(
        out["angle_robot_deg"], bins=ARC_BINS, labels=ARC_LABELS,
        include_lowest=True, right=False,
    )
    out["quadrant"] = pd.cut(
        out["angle_robot_deg"], bins=[0, 90, 180, 270, 360],
        labels=QUADRANT_LABELS, include_lowest=True, right=False,
    )
    out["quadrant_region"] = out["quadrant"]
    return out


def platform_to_robot_view_cm(dx_m, dz_m) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(dx_m, dtype=float) * 100
    z = np.asarray(dz_m, dtype=float) * 100
    return x, z


def configure_robot_view_axes(ax, x_cm: np.ndarray, z_cm: np.ndarray) -> None:
    m = float(np.nanmax(np.abs(np.r_[x_cm, z_cm])) or 1.0)
    m = max(m * 1.1, 5.0)
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_aspect("equal")
    ax.invert_xaxis()


def add_robot_facing_triangle(ax) -> None:
    pos = ax.get_position()
    pad = 0.09
    ax.set_position([pos.x0, pos.y0 + pad, pos.width, pos.height - pad])
    ax.scatter(
        0.5, -0.14, s=130, c=ROBOT_FACING_COLOR, marker="^",
        transform=ax.transAxes, clip_on=False, zorder=10,
    )


def safe_bool_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    vals = set(series.dropna().unique())
    bool_like = {True, False, 0, 1, "True", "False", "true", "false"}
    if vals.issubset(bool_like):
        return series.map({
            True: 1, False: 0, "True": 1, "False": 0,
            "true": 1, "false": 0, 1: 1, 0: 0,
        })
    return series


def find_default_log(robot_key: str) -> Path:
    """Prefer .xlsx under data/ (clean episode logs) over .csv exports."""
    stem = f"episode_log_{robot_key}"
    candidates: list[Path] = []
    for data_dir in DATA_SEARCH_DIRS:
        for ext in (".xlsx", ".csv"):
            path = data_dir / f"{stem}{ext}"
            if path.exists():
                candidates.append(path)
    if candidates:
        xlsx_files = [p for p in candidates if p.suffix.lower() == ".xlsx"]
        return xlsx_files[0] if xlsx_files else candidates[0]
    return DATA_SEARCH_DIRS[0] / f"{stem}.xlsx"


def robot_id_from_path(path: Path) -> int | None:
    name = path.name.lower()
    if "_r2" in name or name.startswith("episode_log_r2"):
        return 2
    if "_r1" in name or name.startswith("episode_log_r1"):
        return 1
    return None


def load_episode_log(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    if "timestamp_local" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"timestamp_local": "timestamp"})
    return df


def clean_episode_log(
    df: pd.DataFrame,
    *,
    expected_robot_id: int | None = None,
    inference_only: bool = True,
    exploit_only: bool = True,
    spawn_phase: str | int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop misaligned or out-of-scope rows; return cleaned frame and warning notes."""
    out = df.copy()
    notes: list[str] = []
    n0 = len(out)

    if expected_robot_id is not None and "robot_id" in out.columns:
        rid = pd.to_numeric(out["robot_id"], errors="coerce")
        out = out.loc[rid == expected_robot_id]
        if len(out) < n0:
            notes.append(f"dropped {n0 - len(out)} rows with robot_id != {expected_robot_id}")
            n0 = len(out)

    if inference_only and "run_mode" in out.columns:
        out = out.loc[out["run_mode"].astype(str) == "inference"]
        if len(out) < n0:
            notes.append(f"dropped {n0 - len(out)} non-inference rows")
            n0 = len(out)

    if exploit_only and "grasp_mode" in out.columns:
        modes = out["grasp_mode"].astype(str)
        out = out.loc[modes.eq("exploit") | modes.str.startswith("exploit_")]
        if len(out) < n0:
            notes.append(f"dropped {n0 - len(out)} non-exploit rows")
            n0 = len(out)

    if spawn_phase is not None and "spawn_phase" in out.columns:
        want = str(spawn_phase)
        out = out.loc[out["spawn_phase"].astype(str) == want]
        if len(out) < n0:
            notes.append(f"dropped {n0 - len(out)} rows with spawn_phase != {want}")
            n0 = len(out)

    robot_id = expected_robot_id
    if robot_id is None and "robot_id" in out.columns:
        mode = pd.to_numeric(out["robot_id"], errors="coerce").dropna()
        if not mode.empty:
            robot_id = int(mode.mode().iloc[0])

    bounds = SPAWN_BOUNDS_BY_ROBOT.get(robot_id or 1, SPAWN_BOUNDS_BY_ROBOT[1])
    for col, (lo, hi) in bounds.items():
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        valid = vals.between(lo, hi)
        dropped = (~valid).sum()
        if dropped:
            out = out.loc[valid]
            notes.append(f"dropped {dropped} rows with {col} outside [{lo}, {hi}]")

    return out.reset_index(drop=True), notes


def resolve_platform_center(data: pd.DataFrame) -> tuple[float, float, int]:
    if "robot_id" in data.columns:
        robot_id = int(pd.to_numeric(data["robot_id"], errors="coerce").dropna().mode().iloc[0])
    else:
        robot_id = 1
    cx, cz = PLATFORM_CENTER_BY_ROBOT.get(robot_id, PLATFORM_CENTER_BY_ROBOT[1])
    return cx, cz, robot_id


def add_sim_spatial_columns(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    cx, cz, robot_id = resolve_platform_center(out)
    out["platform_center_x_m"] = cx
    out["platform_center_z_m"] = cz
    out["robot_id_used"] = robot_id
    out["spawn_x_m"] = pd.to_numeric(out["spawn_x"], errors="coerce")
    out["spawn_z_m"] = pd.to_numeric(out["spawn_z"], errors="coerce")
    out["dx_platform_m"] = out["spawn_x_m"] - cx
    out["dz_platform_m"] = out["spawn_z_m"] - cz
    out["derived_radius_m"] = np.hypot(out["dx_platform_m"], out["dz_platform_m"])
    return assign_direction_bins(out)


def success_fail_summary(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    tmp = data.copy()
    tmp["_ok"] = pd.to_numeric(tmp[TARGET], errors="coerce")
    g = tmp.groupby(group_cols, observed=False)
    summary = g["_ok"].agg(episodes="count", successes="sum").reset_index()
    summary["successes"] = summary["successes"].fillna(0).astype(int)
    summary["failures"] = summary["episodes"] - summary["successes"]
    summary["success_rate"] = summary["successes"] / summary["episodes"].replace(0, np.nan)
    return summary


def build_region_bins(radius_m: pd.Series) -> pd.DataFrame:
    _, bins = pd.qcut(radius_m, q=len(REGION_LABELS), retbins=True, duplicates="drop")
    rows = []
    for i, lab in enumerate(REGION_LABELS[: len(bins) - 1]):
        lo, hi = float(bins[i]), float(bins[i + 1])
        rows.append({
            "region": lab,
            "radius_min_cm": round(lo * 100, 1),
            "radius_max_cm": round(hi * 100, 1),
            "radius_range_cm": f"{lo * 100:.1f}–{hi * 100:.1f}",
        })
    return pd.DataFrame(rows)


def ensure_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outcome_class" not in out.columns or out["outcome_class"].isna().all():
        out["outcome_class"] = out.apply(classify_outcome_from_row, axis=1)
    else:
        missing = out["outcome_class"].isna() | (
            out["outcome_class"].astype(str).str.strip() == ""
        )
        if missing.any():
            out.loc[missing, "outcome_class"] = out.loc[missing].apply(
                classify_outcome_from_row, axis=1,
            )
    if "clamp_limited" not in out.columns or out["clamp_limited"].isna().all():
        out["clamp_limited"] = out.apply(
            lambda row: int(is_clamp_limited_from_row(row)), axis=1,
        )
    return out


def active_outcome_order(data: pd.DataFrame) -> list[str]:
    present = set(data["outcome_class"].dropna().astype(str))
    order = [c for c in OUTCOME_CLASSES if c in present]
    for label in sorted(present):
        if label not in order:
            order.append(label)
    return order


def outcome_by_group_summary(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        data.groupby(group_cols + ["outcome_class"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def build_outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["outcome_class"].value_counts()
    n = len(df)
    rows = []
    seen = set()
    for label in OUTCOME_CLASSES:
        c = int(counts.get(label, 0))
        if c == 0:
            continue
        rows.append({"outcome_class": label, "count": c, "pct": 100.0 * c / n})
        seen.add(label)
    for label, c in counts.items():
        if label not in seen:
            rows.append({
                "outcome_class": label,
                "count": int(c),
                "pct": 100.0 * int(c) / n,
            })
    return pd.DataFrame(rows)


def print_failure_taxonomy_report(df: pd.DataFrame, title: str) -> None:
    n = len(df)
    print(title)
    print(f"Episodes: {n:,}")
    if n == 0:
        print("(no rows)")
        return

    summary = build_outcome_summary(df)
    print("")
    print("outcome_class          count    pct")
    print("-" * 36)
    for row in summary.itertuples():
        print(f"{row.outcome_class:22} {row.count:5,}  {row.pct:5.1f}%")

    clamp_n = int(pd.to_numeric(df["clamp_limited"], errors="coerce").fillna(0).sum())
    print("")
    print(f"clamp_limited=1: {clamp_n:,} ({100.0 * clamp_n / n:.1f}%)")

    if TARGET in df.columns:
        succ = int(pd.to_numeric(df[TARGET], errors="coerce").fillna(0).sum())
        print(f"success column:  {succ:,} ({100.0 * succ / n:.1f}%)")


def _style_table(table) -> None:
    """Tight rows: no vertical scale, minimal padding, uniform row height."""
    cells = table.get_celld()
    n_rows = max(r for r, _ in cells) + 1
    row_h = 1.0 / n_rows

    for (row, col), cell in cells.items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.8)
        cell.PAD = 0.02
        cell.set_height(row_h)

    for col in range(max(c for _, c in cells) + 1):
        header = table[(0, col)]
        header.set_facecolor("#e6e6e6")
        header.set_text_props(fontweight="bold", ha="center" if col == 0 else "left")


def plot_taxonomy_definition_page(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title(
        "Failure taxonomy — colors used in spawn maps and charts",
        fontsize=13, pad=16, y=0.98,
    )

    table_w = 0.78
    table_x0 = (1.0 - table_w) / 2

    n_out = len(OUTCOME_CLASSES) + 1
    outcome_h = 0.036 * n_out
    clamp_h = 0.10
    gap = 0.022
    subtitle_h = 0.042
    clamp_label_h = 0.032

    block_h = subtitle_h + gap + outcome_h + gap + clamp_label_h + gap + clamp_h
    block_bottom = 0.5 - block_h / 2
    block_top = block_bottom + block_h

    ax.text(
        0.5, block_top - subtitle_h / 2,
        "lifted_m = block ΔY after grasp (m). closest_dist_m = min gripper–block distance (m). "
        "First matching rule wins.",
        transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#555555",
    )

    outcome_bottom = block_bottom + clamp_h + gap + clamp_label_h + gap
    col_labels = ["Color", "outcome_class", "Definition"]
    rows = [
        ["", lab, OUTCOME_DEFINITIONS.get(lab, "")]
        for lab in OUTCOME_CLASSES
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colWidths=[0.10, 0.22, 0.68],
        bbox=[table_x0, outcome_bottom, table_w, outcome_h],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    _style_table(table)

    for row_idx, lab in enumerate(OUTCOME_CLASSES, start=1):
        table[(row_idx, 0)].set_facecolor(OUTCOME_COLORS.get(lab, "#888888"))
        table[(row_idx, 0)].get_text().set_text("")
        table[(row_idx, 1)].set_text_props(fontweight="bold")
        table[(row_idx, 1)].set_facecolor("#ffffff")
        table[(row_idx, 2)].set_facecolor("#ffffff")

    ax.text(
        0.5, block_bottom + clamp_h + gap + clamp_label_h / 2,
        "clamp_limited (not colored on maps)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=10, fontweight="bold",
    )

    clamp_table = ax.table(
        cellText=[
            ["0", CLAMP_DEFINITIONS["0"]],
            ["1", CLAMP_DEFINITIONS["1"]],
        ],
        colLabels=["Value", "Meaning"],
        loc="center",
        cellLoc="left",
        colWidths=[0.12, 0.88],
        bbox=[table_x0, block_bottom, table_w, clamp_h],
    )
    clamp_table.auto_set_font_size(False)
    clamp_table.set_fontsize(8.5)
    _style_table(clamp_table)


def plot_failure_taxonomy(ax, df: pd.DataFrame, robot_id: int) -> None:
    summary = build_outcome_summary(df)
    n = len(df)
    clamp_n = int(pd.to_numeric(df["clamp_limited"], errors="coerce").fillna(0).sum())

    labels = summary["outcome_class"].tolist()
    counts = summary["count"].to_numpy()
    colors = [OUTCOME_COLORS.get(lab, "#888888") for lab in labels]

    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    xmax = max(counts.max(), 1)
    for i, (c, p) in enumerate(zip(counts, summary["pct"])):
        ax.text(c + xmax * 0.02, i, f"{c:,} ({p:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Episode count")
    ax.set_title(
        f"Failure taxonomy — this log (robot {robot_id})\n"
        f"n = {n:,} episodes · clamp_limited = 1: {clamp_n:,} ({100.0 * clamp_n / n:.1f}%)",
    )


def plot_spawn_map(ax, spatial_df, robot_id):
    x, z = platform_to_robot_view_cm(
        spatial_df["dx_platform_m"], spatial_df["dz_platform_m"],
    )
    ok = pd.to_numeric(spatial_df[TARGET], errors="coerce").fillna(0).astype(int)
    n_total = len(spatial_df)
    colors = np.where(ok == 1, COLOR_SUCCESS, COLOR_FAILURE)
    ax.scatter(x, z, c=colors, s=16, alpha=0.55, edgecolors="none")
    ax.scatter([], [], c=COLOR_SUCCESS, s=40, label="Success")
    ax.scatter([], [], c=COLOR_FAILURE, s=40, label="Failure")
    ax.legend(loc="upper left", framealpha=0.9)

    ax.scatter(0, 0, s=160, c="black", marker="x", linewidths=2, zorder=5)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    configure_robot_view_axes(ax, x, z)
    ax.set_xlabel(ROBOT_MAP_XLABEL)
    ax.set_ylabel(ROBOT_MAP_YLABEL)
    ax.set_title(
        f"Spawn locations by success/failure (robot {robot_id}) — robot view\n"
        f"n = {n_total:,} spawns",
    )


def plot_spawn_map_by_taxonomy(
    ax, spatial_df, robot_id, outcome_order: list[str],
) -> None:
    x, z = platform_to_robot_view_cm(
        spatial_df["dx_platform_m"], spatial_df["dz_platform_m"],
    )
    n_total = len(spatial_df)
    outcomes = spatial_df["outcome_class"].astype(str)
    point_colors = [OUTCOME_COLORS.get(o, "#888888") for o in outcomes]

    ax.scatter(x, z, c=point_colors, s=16, alpha=0.55, edgecolors="none")
    for outcome in outcome_order:
        ax.scatter(
            [], [], c=OUTCOME_COLORS.get(outcome, "#888888"), s=40, label=outcome,
        )
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    ax.scatter(0, 0, s=160, c="black", marker="x", linewidths=2, zorder=5)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    configure_robot_view_axes(ax, x, z)
    ax.set_xlabel(ROBOT_MAP_XLABEL)
    ax.set_ylabel(ROBOT_MAP_YLABEL)
    ax.set_title(
        f"Spawn locations by outcome type (robot {robot_id}) — robot view\n"
        f"n = {n_total:,} spawns",
    )


def plot_stacked_success_failure(ax, summary, category_col, order, display_labels, title, xlabel):
    plot_df = summary.set_index(category_col).reindex(order).reset_index()
    plot_df = plot_df.fillna({"successes": 0, "failures": 0, "episodes": 0})
    x = np.arange(len(order))
    succ = plot_df["successes"].to_numpy()
    fail = plot_df["failures"].to_numpy()
    ax.bar(x, succ, color=COLOR_SUCCESS, label="Success")
    ax.bar(x, fail, bottom=succ, color=COLOR_FAILURE, label="Failure")
    ymax = max((succ + fail).max(), 1)
    for i, row in plot_df.iterrows():
        if row["episodes"] > 0:
            rate = row["successes"] / row["episodes"]
            ax.text(i, row["episodes"] + ymax * 0.02, f"{rate:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Episode count")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right")


def plot_stacked_outcomes(
    ax,
    summary: pd.DataFrame,
    category_col: str,
    order: list,
    display_labels: list,
    outcome_order: list[str],
    title: str,
    xlabel: str,
) -> None:
    plot_df = summary.set_index(category_col).reindex(order).fillna(0)
    x = np.arange(len(order))
    bottom = np.zeros(len(order))

    for outcome in outcome_order:
        if outcome not in plot_df.columns:
            continue
        counts = plot_df[outcome].to_numpy(dtype=float)
        ax.bar(
            x, counts, bottom=bottom,
            color=OUTCOME_COLORS.get(outcome, "#888888"),
            label=outcome,
        )
        bottom += counts

    ymax = max(float(bottom.max()), 1.0)
    if "success" in plot_df.columns:
        for i, total in enumerate(bottom):
            if total > 0:
                rate = plot_df.iloc[i]["success"] / total
                ax.text(i, total + ymax * 0.02, f"{rate:.0%}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Episode count")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def resolve_repo_path(path: Path, *, must_exist: bool = False) -> Path:
    """Resolve a user path from repo root or cwd (not analysis/)."""
    if path.is_absolute():
        resolved = path
    else:
        candidates = [Path.cwd() / path, REPO_ROOT / path]
        resolved = None
        for candidate in candidates:
            if candidate.exists() or not must_exist:
                resolved = candidate.resolve()
                break
        if resolved is None:
            resolved = (REPO_ROOT / path).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved


def resolve_input_path(arg: str) -> Path:
    inp = Path(arg)
    if inp.is_absolute():
        return inp

    candidates = [
        Path.cwd() / inp,
        REPO_ROOT / inp,
    ]
    for data_dir in DATA_SEARCH_DIRS:
        candidates.append(data_dir / inp.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (REPO_ROOT / inp).resolve()


def pdf_name_for_input(inp: Path) -> str:
    lower = inp.name.lower()
    if "_r1." in lower or lower.startswith("episode_log_r1."):
        return "spawn_spatial_report_r1.pdf"
    if "_r2." in lower or lower.startswith("episode_log_r2."):
        return "spawn_spatial_report_r2.pdf"
    return f"spawn_spatial_report_{inp.stem}.pdf"


def resolve_paths(arg: str | None, out_dir: Path | None = None) -> tuple[Path, Path]:
    dest = out_dir or OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)

    if arg is None or arg.lower() in ("r1", "1"):
        inp = find_default_log("r1")
        out = dest / "spawn_spatial_report_r1.pdf"
    elif arg.lower() in ("r2", "2"):
        inp = find_default_log("r2")
        out = dest / "spawn_spatial_report_r2.pdf"
    else:
        inp = resolve_input_path(arg)
        out = dest / pdf_name_for_input(inp)
    return inp, out


def write_pdf_report(
    *,
    input_file: Path,
    output_pdf: Path,
    spatial: pd.DataFrame,
    robot_id: int,
    region_summary: pd.DataFrame,
    arc_summary: pd.DataFrame,
    quad_summary: pd.DataFrame,
    region_cats: list[str],
    outcome_order: list[str],
    region_taxonomy: pd.DataFrame,
    arc_taxonomy: pd.DataFrame,
    quad_taxonomy: pd.DataFrame,
) -> None:
    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.55, input_file.name, ha="center", fontsize=16, fontweight="bold")
        ax.text(
            0.5, 0.48,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            ha="center", fontsize=12, color="#444444",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        plot_taxonomy_definition_page(ax)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_failure_taxonomy(ax, spatial, robot_id)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 9))
        plot_spawn_map(ax, spatial, robot_id)
        plt.tight_layout()
        add_robot_facing_triangle(ax)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 9))
        plot_spawn_map_by_taxonomy(ax, spatial, robot_id, outcome_order)
        plt.tight_layout()
        add_robot_facing_triangle(ax)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_stacked_success_failure(
            ax, region_summary, "region", REGION_LABELS, region_cats,
            f"Success vs failure by distance from platform (robot {robot_id})",
            "Distance band (cm from platform center)",
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_stacked_outcomes(
            ax, region_taxonomy, "region", REGION_LABELS, region_cats,
            outcome_order,
            f"Outcome types by distance from platform (robot {robot_id})",
            "Distance band (cm from platform center)",
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_stacked_success_failure(
            ax, arc_summary, "arc", ARC_LABELS_ORDER, ARC_LABELS_ORDER,
            f"Success vs failure by arc (robot {robot_id}) — robot view",
            ROBOT_ARC_XLABEL,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_stacked_outcomes(
            ax, arc_taxonomy, "arc", ARC_LABELS_ORDER, ARC_LABELS_ORDER,
            outcome_order,
            f"Outcome types by arc (robot {robot_id}) — robot view",
            ROBOT_ARC_XLABEL,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_stacked_success_failure(
            ax, quad_summary, "quadrant",
            QUADRANT_LABELS, QUADRANT_DISPLAY,
            f"Success vs failure by quadrant (robot {robot_id})",
            ROBOT_QUADRANT_XLABEL,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_stacked_outcomes(
            ax, quad_taxonomy, "quadrant",
            QUADRANT_LABELS, QUADRANT_DISPLAY,
            outcome_order,
            f"Outcome types by quadrant (robot {robot_id})",
            ROBOT_QUADRANT_XLABEL,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {output_pdf} ({output_pdf.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Episode log report: failure taxonomy + spatial analysis (PDF)",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="r1",
        help="r1, r2, or path to episode log .xlsx / .csv (default: r1)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory for PDF output (default: analysis/output/)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include training/explore rows (default: inference + exploit only)",
    )
    parser.add_argument(
        "--spawn-phase",
        default=None,
        help="Only episodes from this spawn phase (e.g. 4 for outer ring only)",
    )
    parser.add_argument(
        "--taxonomy-only",
        action="store_true",
        help="Print failure taxonomy to console only (no PDF)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write cleaned rows with outcome_class to CSV",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Only analyze the last N episodes (default: all)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = resolve_repo_path(out_dir)
    input_file, output_pdf = resolve_paths(args.input, out_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Input not found: {input_file}")

    df = load_episode_log(input_file)
    expected_robot = robot_id_from_path(input_file)
    if args.input and args.input.lower() in ("r1", "1"):
        expected_robot = 1
    elif args.input and args.input.lower() in ("r2", "2"):
        expected_robot = 2

    filter_modes = not args.include_all
    df, clean_notes = clean_episode_log(
        df,
        expected_robot_id=expected_robot,
        inference_only=filter_modes,
        exploit_only=filter_modes,
        spawn_phase=args.spawn_phase,
    )
    if clean_notes:
        print("Data cleanup:")
        for note in clean_notes:
            print(f"  - {note}")
    if df.empty:
        raise ValueError("No valid episodes left after cleanup — check input file.")

    if args.last > 0:
        df = df.tail(args.last).reset_index(drop=True)

    df[TARGET] = safe_bool_to_int(df[TARGET])
    df = ensure_outcome_columns(df)

    print_failure_taxonomy_report(df, f"=== {input_file.name} ===")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nWrote: {args.output_csv}")

    if args.taxonomy_only:
        return

    spatial = add_sim_spatial_columns(df)
    robot_id = int(spatial["robot_id_used"].iloc[0])

    region_bins = build_region_bins(spatial["derived_radius_m"])
    spatial["region"] = pd.qcut(
        spatial["derived_radius_m"],
        q=len(REGION_LABELS),
        labels=REGION_LABELS,
        duplicates="drop",
    )

    region_summary = success_fail_summary(spatial, ["region"]).merge(region_bins, on="region")
    arc_summary = success_fail_summary(spatial, ["arc"])
    quad_summary = success_fail_summary(spatial, ["quadrant"])

    outcome_order = active_outcome_order(spatial)
    region_taxonomy = outcome_by_group_summary(spatial, ["region"])
    arc_taxonomy = outcome_by_group_summary(spatial, ["arc"])
    quad_taxonomy = outcome_by_group_summary(spatial, ["quadrant"])

    region_cats = [
        f"{r.region}\n({r.radius_range_cm} cm)"
        for r in region_bins.itertuples()
    ]

    region_lines = "\n".join(
        f"  {r.region}: {r.radius_range_cm} cm" for r in region_bins.itertuples()
    )
    print(f"Region bands (data tertiles):\n{region_lines}")

    write_pdf_report(
        input_file=input_file,
        output_pdf=output_pdf,
        spatial=spatial,
        robot_id=robot_id,
        region_summary=region_summary,
        arc_summary=arc_summary,
        quad_summary=quad_summary,
        region_cats=region_cats,
        outcome_order=outcome_order,
        region_taxonomy=region_taxonomy,
        arc_taxonomy=arc_taxonomy,
        quad_taxonomy=quad_taxonomy,
    )


if __name__ == "__main__":
    main()
