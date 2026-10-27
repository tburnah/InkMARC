# src/analysis/error_dashboard_phase2a.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Rec:
    # Identity
    file_id: str
    rel_path: str
    split: str
    participant_id: int
    task: str
    session_id: str
    frame_idx: int
    sample_id: str

    # Masks / regimes available now
    xy_mask: int
    heatmap_mask: int

    # GT / pred (normalised)
    gt_x: float
    gt_y: float
    pred_x: float
    pred_y: float

    # Error
    err_l1_norm: float
    err_eu_norm: float
    err_l1_px: float
    err_eu_px: float

    # Motion (computed later)
    motion_gt_px: Optional[float] = None
    motion_pred_px: Optional[float] = None
    speed_bucket: Optional[str] = None


def _meta_get(m: Dict[str, Any], key: str, default: Any) -> Any:
    v = m.get(key, default)
    # handle 0-d tensors if any sneak in
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def _add_motion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["file_id", "frame_idx"]).reset_index(drop=True)

    def per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["motion_gt_px"] = np.sqrt((g["gt_x_px"].diff() ** 2) + (g["gt_y_px"].diff() ** 2))
        g["motion_pred_px"] = np.sqrt((g["pred_x_px"].diff() ** 2) + (g["pred_y_px"].diff() ** 2))
        return g

    # Work in px for motion; add helper columns temporarily
    df["gt_x_px"] = df["gt_x"] * df["out_w"]
    df["gt_y_px"] = df["gt_y"] * df["out_h"]
    df["pred_x_px"] = df["pred_x"] * df["out_w"]
    df["pred_y_px"] = df["pred_y"] * df["out_h"]

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

    # Drop helpers
    df = df.drop(columns=["gt_x_px", "gt_y_px", "pred_x_px", "pred_y_px"])
    return df


def _p(series: pd.Series, q: float) -> float:
    a = series.to_numpy(dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def _summarise(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)[metric]
    out = g.agg(
        n="count",
        mean="mean",
        median="median",
        p90=lambda s: _p(s, 90),
        p95=lambda s: _p(s, 95),
        p99=lambda s: _p(s, 99),
    ).reset_index()
    return out.sort_values(group_cols)


def write_dashboard(
    recs: List[Rec],
    out_dir: str | Path,
    out_w: int,
    out_h: int,
    prefix: str = "val",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in recs])
    df["out_w"] = int(out_w)
    df["out_h"] = int(out_h)

    df = _add_motion(df)

    raw = out_dir / f"{prefix}_per_frame.csv"
    df.to_csv(raw, index=False)

    # Summaries: use euclidean px as primary
    metric = "err_eu_px"
    slices: Dict[str, List[str]] = {
        "by_task": ["task"],
        "by_participant": ["participant_id"],
        "by_session": ["session_id"],
        "by_speed": ["speed_bucket"],
        "by_xy_mask": ["xy_mask"],
        "by_task_speed": ["task", "speed_bucket"],
        "by_task_xy_mask": ["task", "xy_mask"],
    }

    for name, cols in slices.items():
        _summarise(df, cols, metric).to_csv(out_dir / f"{prefix}_{name}.csv", index=False)

    print(f"Wrote {raw} and {len(slices)} summary CSVs to {out_dir}")
