#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


STATE_NAMES = {0: "background", 1: "hover", 2: "touch", -1: "unknown"}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _read_manifest_records(manifest_path: Path, splits: set[str]) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest, list):
        rows = [dict(row) for row in manifest]
    elif isinstance(manifest, dict) and isinstance(manifest.get("frames"), list):
        rows = [dict(row) for row in manifest["frames"]]
    elif isinstance(manifest, dict) and isinstance(manifest.get("items"), list):
        rows = []
        for item in manifest["items"]:
            split = str(item.get("split", "unknown"))
            if splits and split not in splits:
                continue
            h5_path = item.get("output_h5_path") or item.get("label_h5_path")
            if not h5_path:
                continue
            frame_count = int(item.get("frame_count") or 0)
            if frame_count <= 0 and Path(h5_path).exists():
                with h5py.File(h5_path, "r") as h5:
                    frame_count = int(h5["labels"].shape[0])
            stride = max(1, int(item.get("frame_stride") or 1))
            stem = str(item.get("stem") or Path(str(h5_path)).stem.removesuffix("_corrected"))
            for frame_index in range(0, frame_count, stride):
                rows.append(
                    {
                        "split": split,
                        "video_id": stem,
                        "base_stem": stem,
                        "participant_id": item.get("participant_id"),
                        "task_id": item.get("task_id"),
                        "frame_index": frame_index,
                        "label_h5_path": h5_path,
                    }
                )
    else:
        raise ValueError(f"Unsupported manifest layout: {manifest_path}")
    if splits:
        rows = [row for row in rows if str(row.get("split", "unknown")) in splits]
    return rows


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _derive_state(labels: np.ndarray, mask: np.ndarray, threshold: float) -> tuple[int, bool, bool, float | None]:
    x_ok = bool(mask[0] != 0 and _finite(float(labels[0])))
    y_ok = bool(mask[1] != 0 and _finite(float(labels[1])))
    p_ok = bool(mask[2] != 0 and _finite(float(labels[2])))
    valid_xy = x_ok and y_ok
    pressure = float(labels[2]) if p_ok else None
    if valid_xy and p_ok and pressure is not None and pressure > threshold:
        return 2, valid_xy, p_ok, pressure
    if valid_xy:
        return 1, valid_xy, p_ok, pressure
    return 0, valid_xy, p_ok, pressure


def _nearest_transition_distance(states: np.ndarray, idx: int) -> int | None:
    if idx < 0 or idx >= len(states):
        return None
    diffs = np.flatnonzero(states[1:] != states[:-1]) + 1
    if diffs.size == 0:
        return None
    return int(np.min(np.abs(diffs - idx)))


def _summarize_values(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    qs = np.quantile(arr, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "p25": float(qs[3]),
        "median": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
        "p95": float(qs[7]),
        "p99": float(qs[8]),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit InkMARC-Sense H5 pressure labels around the hover/touch boundary."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/audits/pressure_boundary")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument("--near-pressure-thresholds", default="1,2,5,10,25,50,100,250,500,1000")
    parser.add_argument("--transition-windows", default="1,2,3,5,10")
    parser.add_argument(
        "--top-examples",
        type=int,
        default=200,
        help="Write this many lowest-positive and mismatch examples to CSV.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {str(split) for split in args.splits}
    near_thresholds = _parse_csv_floats(args.near_pressure_thresholds)
    transition_windows = _parse_csv_ints(args.transition_windows)

    records = _read_manifest_records(manifest_path, splits)
    occurrence_counts: Counter[tuple[str, int]] = Counter()
    representative: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in records:
        h5_path = str(rec.get("label_h5_path") or "")
        frame_index = int(rec.get("frame_index", rec.get("index", -1)))
        key = (h5_path, frame_index)
        occurrence_counts[key] += 1
        representative.setdefault(key, rec)

    keys_by_h5: dict[str, list[int]] = defaultdict(list)
    for h5_path, frame_index in occurrence_counts:
        keys_by_h5[h5_path].append(frame_index)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "splits": sorted(splits),
        "pressure_touch_threshold": args.pressure_touch_threshold,
        "manifest_rows": len(records),
        "unique_h5_frames": len(occurrence_counts),
        "unique_h5_files": len(keys_by_h5),
        "states": {},
        "pressure": {},
        "near_pressure": {},
        "transition": {},
        "manifest_hint_mismatches": {},
        "problems": {},
    }

    all_positive_pressures: list[float] = []
    touch_positive_pressures: list[float] = []
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    state_counts = Counter()
    weighted_state_counts = Counter()
    problem_counts = Counter()
    mismatch_counts = Counter()
    transition_counts = {window: Counter() for window in transition_windows}
    weighted_transition_counts = {window: Counter() for window in transition_windows}
    near_counts = {threshold: Counter() for threshold in near_thresholds}
    weighted_near_counts = {threshold: Counter() for threshold in near_thresholds}

    for h5_path, frame_indices in sorted(keys_by_h5.items()):
        path = Path(h5_path)
        if not h5_path or not path.exists():
            for frame_index in frame_indices:
                rec = representative[(h5_path, frame_index)]
                weight = occurrence_counts[(h5_path, frame_index)]
                row = {
                    "h5_path": h5_path,
                    "frame_index": frame_index,
                    "split": rec.get("split", "unknown"),
                    "video_id": rec.get("video_id", ""),
                    "base_stem": rec.get("base_stem", ""),
                    "manifest_occurrences": weight,
                    "problem": "missing_h5",
                    "derived_state": -1,
                    "pressure": math.nan,
                    "valid_xy": False,
                    "pressure_valid": False,
                    "transition_distance": math.nan,
                }
                rows.append(row)
                problem_counts["missing_h5"] += 1
            continue

        try:
            with h5py.File(path, "r") as h5:
                labels_all = np.asarray(h5["labels"][:], dtype=np.float32)
                masks_all = np.asarray(h5["label_mask"][:], dtype=np.uint8)
        except Exception:
            for frame_index in frame_indices:
                problem_counts["unreadable_h5"] += 1
                rec = representative[(h5_path, frame_index)]
                rows.append(
                    {
                        "h5_path": h5_path,
                        "frame_index": frame_index,
                        "split": rec.get("split", "unknown"),
                        "video_id": rec.get("video_id", ""),
                        "base_stem": rec.get("base_stem", ""),
                        "manifest_occurrences": occurrence_counts[(h5_path, frame_index)],
                        "problem": "unreadable_h5",
                        "derived_state": -1,
                        "pressure": math.nan,
                        "valid_xy": False,
                        "pressure_valid": False,
                        "transition_distance": math.nan,
                    }
                )
            continue

        full_states = np.empty(labels_all.shape[0], dtype=np.int16)
        for i in range(labels_all.shape[0]):
            full_states[i] = _derive_state(labels_all[i], masks_all[i], args.pressure_touch_threshold)[0]

        for frame_index in sorted(frame_indices):
            rec = representative[(h5_path, frame_index)]
            weight = occurrence_counts[(h5_path, frame_index)]
            split = str(rec.get("split", "unknown"))
            problem = ""
            if frame_index < 0 or frame_index >= labels_all.shape[0]:
                problem = "frame_out_of_range"
                derived_state, valid_xy, pressure_valid, pressure = -1, False, False, None
                distance = None
            else:
                derived_state, valid_xy, pressure_valid, pressure = _derive_state(
                    labels_all[frame_index], masks_all[frame_index], args.pressure_touch_threshold
                )
                distance = _nearest_transition_distance(full_states, frame_index)

            state_name = STATE_NAMES.get(derived_state, "unknown")
            state_counts[state_name] += 1
            weighted_state_counts[state_name] += weight
            split_counts[split][state_name] += 1
            split_counts[split][f"weighted_{state_name}"] += weight
            if problem:
                problem_counts[problem] += 1

            hint = rec.get("state_hint")
            hint_state = int(hint) if hint is not None and str(hint) != "" else None
            mismatch = hint_state is not None and derived_state >= 0 and hint_state != derived_state
            if mismatch:
                mismatch_counts[f"{STATE_NAMES.get(hint_state, hint_state)}_to_{state_name}"] += 1

            if pressure is not None and pressure > 0:
                all_positive_pressures.append(float(pressure))
                if derived_state == 2:
                    touch_positive_pressures.append(float(pressure))

            for threshold in near_thresholds:
                if pressure is not None and 0.0 < pressure <= threshold:
                    near_counts[threshold]["positive_pressure"] += 1
                    weighted_near_counts[threshold]["positive_pressure"] += weight
                    if derived_state == 2:
                        near_counts[threshold]["touch"] += 1
                        weighted_near_counts[threshold]["touch"] += weight

            for window in transition_windows:
                near_transition = distance is not None and distance <= window
                if near_transition:
                    transition_counts[window][state_name] += 1
                    weighted_transition_counts[window][state_name] += weight

            rows.append(
                {
                    "h5_path": h5_path,
                    "frame_index": frame_index,
                    "split": split,
                    "video_id": rec.get("video_id", ""),
                    "base_stem": rec.get("base_stem", ""),
                    "participant_id": rec.get("participant_id", ""),
                    "task_id": rec.get("task_id", ""),
                    "manifest_occurrences": weight,
                    "state_hint": hint_state,
                    "derived_state": derived_state,
                    "derived_state_name": state_name,
                    "manifest_hint_mismatch": mismatch,
                    "valid_xy": valid_xy,
                    "pressure_valid": pressure_valid,
                    "pressure": math.nan if pressure is None else pressure,
                    "transition_distance": math.nan if distance is None else distance,
                    "problem": problem,
                }
            )

    df = pd.DataFrame(rows)
    valid_df = df[df["derived_state"] >= 0].copy() if not df.empty else df
    touch_count = int((valid_df["derived_state"] == 2).sum()) if not valid_df.empty else 0
    weighted_touch_count = int(valid_df.loc[valid_df["derived_state"] == 2, "manifest_occurrences"].sum()) if not valid_df.empty else 0
    positive_count = int((valid_df["pressure"].fillna(0.0) > 0.0).sum()) if not valid_df.empty else 0
    weighted_positive_count = (
        int(valid_df.loc[valid_df["pressure"].fillna(0.0) > 0.0, "manifest_occurrences"].sum())
        if not valid_df.empty
        else 0
    )

    summary["states"] = {
        "unique": dict(state_counts),
        "manifest_weighted": dict(weighted_state_counts),
        "by_split_unique": {split: dict(counts) for split, counts in split_counts.items()},
    }
    summary["pressure"] = {
        "positive_all": _summarize_values(all_positive_pressures),
        "positive_touch": _summarize_values(touch_positive_pressures),
    }
    summary["near_pressure"] = {
        str(threshold): {
            "unique_positive_count": int(near_counts[threshold]["positive_pressure"]),
            "unique_touch_count": int(near_counts[threshold]["touch"]),
            "unique_touch_fraction": _safe_div(near_counts[threshold]["touch"], touch_count),
            "weighted_positive_count": int(weighted_near_counts[threshold]["positive_pressure"]),
            "weighted_touch_count": int(weighted_near_counts[threshold]["touch"]),
            "weighted_touch_fraction": _safe_div(weighted_near_counts[threshold]["touch"], weighted_touch_count),
            "unique_positive_fraction": _safe_div(near_counts[threshold]["positive_pressure"], positive_count),
            "weighted_positive_fraction": _safe_div(
                weighted_near_counts[threshold]["positive_pressure"], weighted_positive_count
            ),
        }
        for threshold in near_thresholds
    }
    summary["transition"] = {
        str(window): {
            "unique": dict(transition_counts[window]),
            "manifest_weighted": dict(weighted_transition_counts[window]),
            "unique_touch_fraction": _safe_div(transition_counts[window]["touch"], touch_count),
            "weighted_touch_fraction": _safe_div(weighted_transition_counts[window]["touch"], weighted_touch_count),
        }
        for window in transition_windows
    }
    summary["manifest_hint_mismatches"] = dict(mismatch_counts)
    summary["problems"] = dict(problem_counts)

    full_csv = output_dir / "pressure_boundary_frames.csv"
    summary_json = output_dir / "pressure_boundary_summary.json"
    df.to_csv(full_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    if args.top_examples > 0 and not valid_df.empty:
        positive_df = valid_df[valid_df["pressure"].fillna(0.0) > 0.0].sort_values("pressure")
        positive_df.head(args.top_examples).to_csv(output_dir / "lowest_positive_pressure_examples.csv", index=False)
        valid_df[valid_df["manifest_hint_mismatch"]].head(args.top_examples).to_csv(
            output_dir / "manifest_hint_mismatch_examples.csv", index=False
        )
        valid_df.sort_values(["transition_distance", "pressure"], na_position="last").head(args.top_examples).to_csv(
            output_dir / "nearest_transition_examples.csv", index=False
        )

    print(f"saved {summary_json}")
    print(f"saved {full_csv}")
    print(f"manifest rows: {len(records)}")
    print(f"unique h5/frame labels: {len(occurrence_counts)}")
    print(f"unique states: {dict(state_counts)}")
    print(f"manifest-weighted states: {dict(weighted_state_counts)}")
    print(f"positive pressure quantiles: {summary['pressure']['positive_all']}")
    if near_thresholds:
        first = near_thresholds[0]
        last = near_thresholds[-1]
        print(f"touch frames with 0<p<={first:g}: {summary['near_pressure'][str(first)]['unique_touch_fraction']:.4f}")
        print(f"touch frames with 0<p<={last:g}: {summary['near_pressure'][str(last)]['unique_touch_fraction']:.4f}")
    if transition_windows:
        window = transition_windows[min(2, len(transition_windows) - 1)]
        print(
            f"touch frames within {window} frames of a state transition: "
            f"{summary['transition'][str(window)]['unique_touch_fraction']:.4f}"
        )
    if mismatch_counts:
        print(f"manifest hint mismatches: {dict(mismatch_counts)}")
    if problem_counts:
        print(f"problems: {dict(problem_counts)}")


if __name__ == "__main__":
    main()
