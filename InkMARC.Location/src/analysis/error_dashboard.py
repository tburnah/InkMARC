# src/analysis/error_dashboard.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SampleRecord:
    file_id: str
    rel_path: str
    split: str

    participant_id: int
    task: str
    session_id: Optional[str]
    frame_idx: int

    # Labels
    touch_gt: Optional[int]
    p_gt: Optional[float]
    xy_mask: Optional[int]

    gt_x_px: float
    gt_y_px: float
    pred_x_px: float
    pred_y_px: float

    err_px: float
    err_x_px: float
    err_y_px: float

    # Regimes (optional, fill when you can)
    corner_quality: Optional[str] = None
    edge_proximity: Optional[str] = None  # e.g., "inboard", "near_edge", "touching_edge"
    warp_cond: Optional[float] = None     # condition number of homography (if available)

    # Motion (filled later)
    motion_gt_px: Optional[float] = None
    motion_pred_px: Optional[float] = None
    speed_bucket: Optional[str] = None


def _euclid(dx: float, dy: float) -> float:
    return float(np.sqrt(dx * dx + dy * dy))


def add_motion_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["file_id", "frame_idx"]).reset_index(drop=True)

    def per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["motion_gt_px"] = np.sqrt((g["gt_x_px"].diff() ** 2) + (g["gt_y_px"].diff() ** 2))
        g["motion_pred_px"] = np.sqrt((g["pred_x_px"].diff() ** 2) + (g["pred_y_px"].diff() ** 2))
        return g

    df = df.groupby("file_id", group_keys=False).apply(per_group)

    def bucket(v: Any) -> str:
        if pd.isna(v):
            return "unknown"
        v = float(v)
        if v < 1.0:
            return "static"
        if v < 5.0:
            return "slow"
        if v < 15.0:
            return "medium"
        return "fast"

    df["speed_bucket"] = df["motion_gt_px"].apply(bucket)
    return df


def summarise(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    def p(s: pd.Series, q: float) -> float:
        a = s.to_numpy(dtype=np.float64)
        a = a[~np.isnan(a)]
        return float(np.percentile(a, q)) if a.size else float("nan")

    g = df.groupby(group_cols, dropna=False)["err_px"]
    out = g.agg(
        n="count",
        mean="mean",
        median="median",
        p90=lambda s: p(s, 90),
        p95=lambda s: p(s, 95),
        p99=lambda s: p(s, 99),
    ).reset_index()
    return out.sort_values(group_cols)


def write_dashboard(records: List[SampleRecord], out_dir: str | Path, prefix: str = "val") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in records])
    df = add_motion_columns(df)

    raw_path = out_dir / f"{prefix}_per_frame.csv"
    df.to_csv(raw_path, index=False)

    slices: Dict[str, List[str]] = {
        "by_task": ["task"],
        "by_participant": ["participant_id"],
        "by_touch": ["touch_gt"],
        "by_xy_mask": ["xy_mask"],
        "by_speed": ["speed_bucket"],
        "by_task_touch": ["task", "touch_gt"],
        "by_task_speed": ["task", "speed_bucket"],
        "by_corner_quality": ["corner_quality"],
        "by_edge_proximity": ["edge_proximity"],
    }

    for name, cols in slices.items():
        summarise(df, cols).to_csv(out_dir / f"{prefix}_{name}.csv", index=False)

    print(f"Wrote dashboard:\n  {raw_path}\n  + {len(slices)} summary tables in {out_dir}")
