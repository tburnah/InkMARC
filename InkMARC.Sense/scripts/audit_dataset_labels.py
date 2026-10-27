#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_participant(stem: str) -> str:
    m = re.match(r"participant_([^_]+)_", stem)
    return m.group(1) if m else "UNKNOWN"


def parse_task(stem: str) -> str:
    parts = stem.split("_")
    return parts[2] if len(parts) >= 3 else "UNKNOWN"


def video_frame_count(path: Path) -> int | None:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None
    finally:
        cap.release()


def depth_frame_count(path: Path, key: str) -> int | None:
    if not path.exists():
        return None
    store = zarr.open(path, mode="r")
    arr = store if hasattr(store, "shape") else store[key]
    return int(arr.shape[0]) if len(arr.shape) >= 3 else 1


def infer_coordinate_type(x: np.ndarray, y: np.ndarray, image_w: float, image_h: float) -> str:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return "unknown_no_valid_xy"
    xv = x[valid]
    yv = y[valid]
    norm_frac = float(((0 <= xv) & (xv <= 1) & (0 <= yv) & (yv <= 1)).mean())
    image_frac = float(((0 <= xv) & (xv <= image_w) & (0 <= yv) & (yv <= image_h)).mean())
    if norm_frac >= 0.95:
        return "likely_normalized"
    if image_frac >= 0.95 and (np.nanmax(xv) > 1.5 or np.nanmax(yv) > 1.5):
        return "likely_image_pixels"
    if np.nanmax(xv) > image_w or np.nanmax(yv) > image_h:
        return "likely_tablet_or_surface_coordinates"
    return "ambiguous"


def summarize_values(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit resized InkMARC-Sense labels/depth/video consistency")
    parser.add_argument("--manifest", default="/data/baseAVIs_432x672/manifest_participant_splits.json")
    parser.add_argument("--output-dir", default="outputs/audits")
    parser.add_argument("--depth-key", default="depth_norm")
    parser.add_argument("--image-width", type=float, default=432)
    parser.add_argument("--image-height", type=float, default=672)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"manifest": str(manifest_path), "groups": {}}
    global_x: list[float] = []
    global_y: list[float] = []

    group_acc: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: defaultdict(int))

    for item in items:
        stem = str(item["stem"])
        split = str(item.get("split", "unknown"))
        participant_id = str(item.get("participant_id") or parse_participant(stem))
        task_id = str(item.get("task_id") or parse_task(stem))
        video_path = Path(item.get("output_video_path", ""))
        h5_path = Path(item.get("output_h5_path", ""))
        depth_path = Path(item.get("depth_path") or video_path.parent / "depth" / f"{stem}.zarr")
        manifest_frames = int(item.get("frame_count") or 0)
        video_frames = video_frame_count(video_path)
        depth_frames = depth_frame_count(depth_path, str(item.get("depth_key", args.depth_key))) if depth_path.exists() else None

        h5_exists = h5_path.exists()
        h5_frames = None
        valid_xy = pressure_pos = pressure_zero = pressure_neg = naninf = 0
        x_values: list[float] = []
        y_values: list[float] = []
        pressure_values: list[float] = []
        if h5_exists:
            try:
                with h5py.File(h5_path, "r") as h5:
                    labels = np.asarray(h5["labels"][:], dtype=np.float64)
                    masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
                    h5_frames = int(labels.shape[0])
                    x = labels[:, 0]
                    y = labels[:, 1]
                    p = labels[:, 2]
                    xy_mask = (masks[:, 0] != 0) & (masks[:, 1] != 0)
                    p_mask = masks[:, 2] != 0
                    finite_xy = np.isfinite(x) & np.isfinite(y)
                    finite_p = np.isfinite(p)
                    valid = xy_mask & finite_xy
                    valid_xy = int(valid.sum())
                    pressure_pos = int((p_mask & finite_p & (p > 0)).sum())
                    pressure_zero = int((p_mask & finite_p & (p == 0)).sum())
                    pressure_neg = int((p_mask & finite_p & (p < 0)).sum())
                    naninf = int((~finite_xy | ~finite_p).sum())
                    x_values = x[valid].tolist()
                    y_values = y[valid].tolist()
                    pressure_values = p[p_mask & finite_p].tolist()
                    global_x.extend(x_values)
                    global_y.extend(y_values)
            except OSError:
                h5_exists = False

        total_frames = manifest_frames or video_frames or depth_frames or h5_frames or 0
        row = {
            "split": split,
            "participant_id": participant_id,
            "task_id": task_id,
            "video_id": stem,
            "video_exists": video_path.exists(),
            "depth_exists": depth_path.exists(),
            "h5_exists": h5_exists,
            "manifest_frames": manifest_frames,
            "video_frames": video_frames,
            "depth_frames": depth_frames,
            "h5_frames": h5_frames,
            "count_mismatch": len({c for c in [manifest_frames, video_frames, depth_frames, h5_frames] if c is not None and c > 0}) > 1,
            "total_frames": total_frames,
            "frames_with_h5_labels": h5_frames or 0,
            "frames_without_h5_labels": 0 if h5_exists else total_frames,
            "valid_xy_true": valid_xy,
            "valid_xy_false": max(0, (h5_frames or 0) - valid_xy),
            "pressure_gt_0": pressure_pos,
            "pressure_eq_0": pressure_zero,
            "pressure_lt_0": pressure_neg,
            "naninf_xy_pressure": naninf,
            "x_inside_0_1": int(((0 <= np.asarray(x_values)) & (np.asarray(x_values) <= 1)).sum()) if x_values else 0,
            "y_inside_0_1": int(((0 <= np.asarray(y_values)) & (np.asarray(y_values) <= 1)).sum()) if y_values else 0,
            "xy_inside_image": int(
                (
                    (0 <= np.asarray(x_values))
                    & (np.asarray(x_values) <= args.image_width)
                    & (0 <= np.asarray(y_values))
                    & (np.asarray(y_values) <= args.image_height)
                ).sum()
            )
            if x_values
            else 0,
            "coordinate_inference": infer_coordinate_type(np.asarray(x_values), np.asarray(y_values), args.image_width, args.image_height),
            **{f"x_{k}": v for k, v in summarize_values(x_values).items()},
            **{f"y_{k}": v for k, v in summarize_values(y_values).items()},
            **{f"pressure_{k}": v for k, v in summarize_values(pressure_values).items()},
        }
        rows.append(row)
        for group_name, group_value in [
            ("split", split),
            ("participant_id", participant_id),
            ("task_id", task_id),
            ("video_id", stem),
        ]:
            acc = group_acc[(group_name, group_value)]
            for key in [
                "total_frames",
                "frames_with_h5_labels",
                "frames_without_h5_labels",
                "valid_xy_true",
                "valid_xy_false",
                "pressure_gt_0",
                "pressure_eq_0",
                "pressure_lt_0",
                "naninf_xy_pressure",
            ]:
                acc[key] += int(row[key] or 0)
            acc["videos"] += 1
            acc["missing_video"] += int(not row["video_exists"])
            acc["missing_depth"] += int(not row["depth_exists"])
            acc["missing_h5"] += int(not row["h5_exists"])
            acc["count_mismatch"] += int(row["count_mismatch"])

    for (group_name, group_value), values in group_acc.items():
        summary["groups"].setdefault(group_name, {})[group_value] = dict(values)
    summary["coordinate_inference_global"] = infer_coordinate_type(
        np.asarray(global_x), np.asarray(global_y), args.image_width, args.image_height
    )
    summary["x"] = summarize_values(global_x)
    summary["y"] = summarize_values(global_y)

    pd.DataFrame(rows).to_csv(output_dir / "dataset_label_audit.csv", index=False)
    (output_dir / "dataset_label_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"saved {output_dir / 'dataset_label_audit.json'}")
    print(f"saved {output_dir / 'dataset_label_audit.csv'}")
    print(f"global coordinate inference: {summary['coordinate_inference_global']}")
    mismatches = sum(1 for r in rows if r["count_mismatch"])
    missing_h5 = sum(1 for r in rows if not r["h5_exists"])
    print(f"videos={len(rows)} count_mismatches={mismatches} missing_h5={missing_h5}")


if __name__ == "__main__":
    main()
