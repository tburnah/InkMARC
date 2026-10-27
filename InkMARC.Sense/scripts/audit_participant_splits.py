#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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


def h5_state_counts(path: Path, pressure_threshold: float) -> dict[str, int]:
    if not path.exists():
        return {"background": 0, "hover": 0, "touch": 0, "valid_xy": 0, "touch_binary": 0}
    try:
        with h5py.File(path, "r") as h5:
            labels = np.asarray(h5["labels"][:], dtype=np.float32)
            masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
    except OSError:
        return {"background": 0, "hover": 0, "touch": 0, "valid_xy": 0, "touch_binary": 0}
    valid_xy = (masks[:, 0] != 0) & (masks[:, 1] != 0) & np.isfinite(labels[:, 0]) & np.isfinite(labels[:, 1])
    has_pressure = (masks[:, 2] != 0) & np.isfinite(labels[:, 2])
    touch = has_pressure & (labels[:, 2] > pressure_threshold)
    return {
        "background": int((~valid_xy).sum()),
        "hover": int((valid_xy & ~touch).sum()),
        "touch": int((valid_xy & touch).sum()),
        "valid_xy": int(valid_xy.sum()),
        "touch_binary": int(touch.sum()),
    }


def infer_frames(item: dict[str, Any], depth_path: Path) -> int:
    frames = int(item.get("frame_count") or 0)
    if frames > 0:
        return frames
    if depth_path.exists():
        try:
            store = zarr.open(depth_path, mode="r")
            arr = store if hasattr(store, "shape") else store[item.get("depth_key", "depth_norm")]
            if len(arr.shape) >= 3:
                return int(arr.shape[0])
        except Exception:
            return 0
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit participant split manifest")
    parser.add_argument("--manifest", default="/data/baseAVIs_432x672/manifest_participant_splits.json")
    parser.add_argument("--output-dir", default="outputs/audits")
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument("--min-val-test-touch", type=int, default=100)
    parser.add_argument("--min-val-test-valid-xy", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = manifest.get("items", [])

    split_participants: dict[str, set[str]] = defaultdict(set)
    split_tasks: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in items:
        stem = str(item["stem"])
        split = str(item.get("split", "unknown"))
        pid = str(item.get("participant_id") or parse_participant(stem))
        task = str(item.get("task_id") or parse_task(stem))
        video_path = Path(item.get("output_video_path", ""))
        depth_path = Path(item.get("depth_path") or video_path.parent / "depth" / f"{stem}.zarr")
        h5_path = Path(item.get("output_h5_path", ""))
        frames = infer_frames(item, depth_path)
        states = h5_state_counts(h5_path, args.pressure_touch_threshold)

        split_participants[split].add(pid)
        split_tasks[split][task] += 1
        c = split_counts[split]
        c["videos"] += 1
        c["frames"] += frames
        c["missing_video"] += int(not video_path.exists())
        c["missing_depth"] += int(not depth_path.exists())
        c["missing_h5"] += int(not h5_path.exists())
        for key, value in states.items():
            c[key] += int(value)

        task_rows.append({"split": split, "task_id": task, "videos": 1, "frames": frames, **states})
        participant_rows.append({"split": split, "participant_id": pid, "videos": 1, "frames": frames, **states})

    for a in ["train", "val", "test"]:
        for b in ["train", "val", "test"]:
            if a >= b:
                continue
            overlap = split_participants[a] & split_participants[b]
            if overlap:
                warnings.append(f"participant overlap {a}/{b}: {sorted(overlap)}")

    train_tasks = set(split_tasks["train"])
    for split in ["val", "test"]:
        missing_tasks = sorted(train_tasks - set(split_tasks[split]))
        if missing_tasks:
            warnings.append(f"tasks present in train but missing in {split}: {missing_tasks}")
        if split_counts[split]["touch"] < args.min_val_test_touch:
            warnings.append(f"{split} has few touch frames: {split_counts[split]['touch']}")
        if split_counts[split]["valid_xy"] < args.min_val_test_valid_xy:
            warnings.append(f"{split} has few valid_xy frames: {split_counts[split]['valid_xy']}")

    task_df = pd.DataFrame(task_rows).groupby(["split", "task_id"], as_index=False).sum(numeric_only=True)
    participant_df = pd.DataFrame(participant_rows).groupby(["split", "participant_id"], as_index=False).sum(numeric_only=True)
    task_df.to_csv(output_dir / "split_task_counts.csv", index=False)
    participant_df.to_csv(output_dir / "split_participant_counts.csv", index=False)

    audit = {
        "manifest": args.manifest,
        "exclusive_participants": not any("participant overlap" in w for w in warnings),
        "splits": {
            split: {
                **{k: int(v) for k, v in split_counts[split].items()},
                "participants": len(split_participants[split]),
                "participant_ids": sorted(split_participants[split]),
                "task_distribution": dict(split_tasks[split]),
            }
            for split in ["train", "val", "test"]
        },
        "warnings": warnings,
    }
    (output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit["splits"], indent=2))
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"saved {output_dir / 'split_audit.json'}")


if __name__ == "__main__":
    main()
