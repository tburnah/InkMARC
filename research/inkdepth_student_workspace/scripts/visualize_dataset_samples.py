#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets.stylus_dataset import surface_to_image_xy


STATE_NAME = {0: "background", 1: "hover", 2: "touch"}
STATE_ID = {v: k for k, v in STATE_NAME.items()}


def parse_participant(stem: str) -> str:
    m = re.match(r"participant_([^_]+)_", stem)
    return m.group(1) if m else "UNKNOWN"


def parse_task(stem: str) -> str:
    parts = stem.split("_")
    return parts[2] if len(parts) >= 3 else "UNKNOWN"


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read {path} frame {frame_index}")
        return frame
    finally:
        cap.release()


def read_depth(path: Path, key: str, frame_index: int) -> np.ndarray:
    store = zarr.open(path, mode="r")
    arr = store if hasattr(store, "shape") else store[key]
    return np.asarray(arr[frame_index], dtype=np.float32)


def label_for_frame(path: Path, frame_index: int, threshold: float, surface_w: float, surface_h: float) -> dict[str, Any]:
    if not path.exists():
        return {"has_labels": False, "valid_xy": False, "state": None, "pressure": None, "xy_norm": None}
    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5["labels"][frame_index], dtype=np.float32)
        masks = np.asarray(h5["label_mask"][frame_index], dtype=np.uint8)
        corners = np.asarray(h5["corners"][frame_index], dtype=np.float32) if "corners" in h5 else None
        resize_attrs = dict(h5["preprocessing/resize_transform"].attrs) if "preprocessing/resize_transform" in h5 else {}
    valid_xy = bool(masks[0] and masks[1] and np.isfinite(labels[0]) and np.isfinite(labels[1]))
    has_pressure = bool(masks[2] and np.isfinite(labels[2]))
    pressure = float(labels[2]) if has_pressure else None
    touch = bool(has_pressure and pressure > threshold)
    xy_norm = None
    if valid_xy:
        xy_norm = surface_to_image_xy(
            float(labels[0]),
            float(labels[1]),
            corners,
            surface_w=surface_w,
            surface_h=surface_h,
            image_w=432.0,
            image_h=672.0,
            surface_model_w=480.0,
            surface_model_h=320.0,
            corners_coordinate_mode="resized",
            resize_attrs=resize_attrs,
        )
        valid_xy = xy_norm is not None
    state = 2 if valid_xy and touch else 1 if valid_xy else 0
    return {
        "has_labels": True,
        "valid_xy": valid_xy,
        "state": state,
        "pressure": pressure,
        "xy_norm": xy_norm if valid_xy else None,
        "raw_x": float(labels[0]) if np.isfinite(labels[0]) else None,
        "raw_y": float(labels[1]) if np.isfinite(labels[1]) else None,
    }


def colorize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    finite = np.isfinite(depth)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        norm = np.zeros_like(depth, dtype=np.float32)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    return cv2.resize(color, size, interpolation=cv2.INTER_LINEAR)


def put_lines(img: np.ndarray, lines: list[str], warning: bool = False) -> np.ndarray:
    out = img.copy()
    bg = (0, 0, 180) if warning else (0, 0, 0)
    y = 22
    for line in lines:
        cv2.rectangle(out, (4, y - 17), (min(out.shape[1] - 4, 10 + len(line) * 8), y + 5), bg, -1)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Save visual debug panels from dataset samples")
    parser.add_argument("--manifest", default="/data/baseAVIs_432x672/manifest_participant_splits.json")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--task", default=None)
    parser.add_argument("--participant-id", default=None)
    parser.add_argument("--state", choices=["background", "hover", "touch"], default=None)
    parser.add_argument("--touch-only", action="store_true")
    parser.add_argument("--valid-xy-only", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/debug_samples")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument("--surface-width", type=float, default=44800.0)
    parser.add_argument("--surface-height", type=float, default=29600.0)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    candidates: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    for item in manifest.get("items", []):
        stem = str(item["stem"])
        split = str(item.get("split", "unknown"))
        pid = str(item.get("participant_id") or parse_participant(stem))
        task = str(item.get("task_id") or parse_task(stem))
        if args.split != "all" and split != args.split:
            continue
        if args.task and task != args.task:
            continue
        if args.participant_id and pid != str(args.participant_id):
            continue
        h5_path = Path(item.get("output_h5_path", ""))
        frame_count = int(item.get("frame_count") or 0)
        stride = max(1, frame_count // 40) if frame_count else 1
        for frame_index in range(0, frame_count, stride):
            label = label_for_frame(h5_path, frame_index, args.pressure_touch_threshold, args.surface_width, args.surface_height)
            if args.state is not None and label.get("state") != STATE_ID[args.state]:
                continue
            if args.touch_only and not (label.get("pressure") is not None and label["pressure"] > args.pressure_touch_threshold):
                continue
            if args.valid_xy_only and not label.get("valid_xy"):
                continue
            candidates.append((item, frame_index, label))

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.num_samples]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (item, frame_index, label) in enumerate(selected):
        stem = str(item["stem"])
        split = str(item.get("split", "unknown"))
        pid = str(item.get("participant_id") or parse_participant(stem))
        task = str(item.get("task_id") or parse_task(stem))
        video_path = Path(item["output_video_path"])
        depth_path = Path(item.get("depth_path") or video_path.parent / "depth" / f"{stem}.zarr")
        frame = read_frame(video_path, frame_index)
        h, w = frame.shape[:2]
        depth_img = colorize_depth(read_depth(depth_path, str(item.get("depth_key", "depth_norm")), frame_index), (w, h))
        overlay = frame.copy()
        if label.get("valid_xy") and label.get("xy_norm") is not None:
            x = int(round(float(label["xy_norm"][0]) * (w - 1)))
            y = int(round(float(label["xy_norm"][1]) * (h - 1)))
            cv2.circle(overlay, (x, y), 9, (0, 255, 255), 2)
            cv2.drawMarker(overlay, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        state_name = STATE_NAME.get(label.get("state"), "unlabeled")
        warnings = []
        if label.get("state") == 0 and label.get("valid_xy"):
            warnings.append("WARNING background_with_valid_xy")
        if label.get("pressure") is not None and label["pressure"] > args.pressure_touch_threshold and label.get("state") != 2:
            warnings.append("WARNING pressure_gt_threshold_not_touch")
        if label.get("pressure") == 0 and label.get("valid_xy") and label.get("state") != 1:
            warnings.append("WARNING zero_pressure_valid_xy_not_hover")
        lines = [
            f"{split} pid={pid} task={task}",
            f"{stem} frame={frame_index}",
            f"state={state_name} valid_xy={label.get('valid_xy')} pressure={label.get('pressure')}",
            *warnings,
        ]
        overlay = put_lines(overlay, lines, warning=bool(warnings))
        panel = np.concatenate([frame, depth_img, overlay], axis=1)
        out = output_dir / f"{i:04d}_{split}_{pid}_{task}_{stem}_f{frame_index}.jpg"
        cv2.imwrite(str(out), panel)
    print(f"saved {len(selected)} panels to {output_dir}")
    if not selected:
        print("WARNING: no samples matched filters")


if __name__ == "__main__":
    main()
