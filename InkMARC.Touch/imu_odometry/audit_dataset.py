from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_SORT_COLUMN_CANDIDATES, resolve_feature_selection
from dataset import SensorWindowDataset, resolve_source_column
from utils import ensure_dir, save_history_csv, save_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit IMU odometry CSV sessions and target generation.")
    parser.add_argument("--csv-files", nargs="+", required=True)
    parser.add_argument("--feature-groups", nargs="*", default=["base", "gravity", "nav_frame"])
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument("--target-mode", type=str, default="delta_xy")
    parser.add_argument("--window-size", type=int, default=90)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--window-mode", type=str, default="rows", choices=["rows", "time"])
    parser.add_argument("--window-duration-s", type=float, default=1.0)
    parser.add_argument("--window-stride-s", type=float, default=0.10)
    parser.add_argument("--required-valid-flags", nargs="*", default=["tag_pose_valid"])
    parser.add_argument("--sort-columns", nargs="*", default=list(DEFAULT_SORT_COLUMN_CANDIDATES))
    parser.add_argument("--quaternion-frame-mode", type=str, default="body_to_nav", choices=["body_to_nav", "nav_to_body"])
    parser.add_argument("--min-valid-label-points", type=int, default=2)
    parser.add_argument("--min-target-magnitude-m", type=float, default=0.10)
    parser.add_argument("--loss-weight-mode", type=str, default="target_magnitude", choices=["none", "target_magnitude"])
    parser.add_argument("--loss-weight-reference-m", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="./runs/imu_odometry_audit")
    return parser


def _read_xyz(frame: pd.DataFrame, prefix: str) -> np.ndarray | None:
    cols = [
        resolve_source_column(f"{prefix}_x", frame.columns),
        resolve_source_column(f"{prefix}_y", frame.columns),
        resolve_source_column(f"{prefix}_z", frame.columns),
    ]
    if not all(cols):
        return None
    return np.stack([pd.to_numeric(frame[col], errors="coerce").to_numpy() for col in cols], axis=1).astype(float)


def _safe_stats(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    args = build_argparser().parse_args()
    feature_names = resolve_feature_selection(args.feature_groups, args.features)
    dataset = SensorWindowDataset(
        csv_files=args.csv_files,
        feature_names=feature_names,
        target_mode=args.target_mode,
        window_size=args.window_size,
        stride=args.stride,
        window_mode=args.window_mode,
        window_duration_s=args.window_duration_s,
        window_stride_s=args.window_stride_s,
        required_valid_flags=args.required_valid_flags,
        sort_column_candidates=args.sort_columns,
        quaternion_frame_mode=args.quaternion_frame_mode,
        loss_weight_mode=args.loss_weight_mode,
        loss_weight_reference_m=args.loss_weight_reference_m,
        min_valid_label_points=args.min_valid_label_points,
        min_target_magnitude_m=args.min_target_magnitude_m,
    )

    session_rows: list[dict[str, Any]] = []
    for session_idx, session in enumerate(dataset.sessions):
        frame = pd.read_csv(session.file_path, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        session_samples = [sample for sample in dataset.samples if sample.session_idx == session_idx]
        timestamps = session.timestamps_s.astype(np.float64)
        dt = np.diff(timestamps)
        magnitudes = np.asarray([sample.target_magnitude for sample in session_samples], dtype=np.float64)
        durations = np.asarray(
            [timestamps[sample.target_end] - timestamps[sample.target_start] for sample in session_samples],
            dtype=np.float64,
        )
        quat_norms = (
            np.linalg.norm(session.quaternion_xyzw.astype(np.float64), axis=1)
            if session.quaternion_xyzw is not None
            else np.asarray([], dtype=np.float64)
        )
        accel_xyz = _read_xyz(frame, "accel")
        gyro_xyz = _read_xyz(frame, "gyro")
        total_rows = int(session.features.shape[0])
        possible_windows = max(0, (total_rows - args.window_size) // args.stride + 1) if total_rows >= args.window_size else 0
        row = {
            "session_name": session.session_name,
            "file_path": session.file_path,
            "sort_column": session.sort_column,
            "rows": total_rows,
            "windows": len(session_samples),
            "possible_windows": possible_windows,
            "window_keep_fraction": float(len(session_samples) / possible_windows) if possible_windows > 0 else 0.0,
            "dt_mean_ms": _safe_stats(dt * 1000.0)["mean"],
            "dt_std_ms": _safe_stats(dt * 1000.0)["std"],
            "dt_min_ms": _safe_stats(dt * 1000.0)["min"],
            "dt_max_ms": _safe_stats(dt * 1000.0)["max"],
            "zero_dt_fraction": float(np.mean(dt == 0.0)) if dt.size else None,
            "positive_dt_fraction": float(np.mean(dt > 0.0)) if dt.size else None,
            "target_duration_mean_s": _safe_stats(durations)["mean"],
            "target_duration_std_s": _safe_stats(durations)["std"],
            "target_duration_min_s": _safe_stats(durations)["min"],
            "target_duration_max_s": _safe_stats(durations)["max"],
            "target_magnitude_mean": _safe_stats(magnitudes)["mean"],
            "target_magnitude_std": _safe_stats(magnitudes)["std"],
            "target_magnitude_min": _safe_stats(magnitudes)["min"],
            "target_magnitude_max": _safe_stats(magnitudes)["max"],
            "frac_target_magnitude_lt_0_05": float(np.mean(magnitudes < 0.05)) if magnitudes.size else None,
            "frac_target_magnitude_lt_0_10": float(np.mean(magnitudes < 0.10)) if magnitudes.size else None,
            "frac_target_magnitude_lt_0_15": float(np.mean(magnitudes < 0.15)) if magnitudes.size else None,
            "accel_mean_xyz": np.nanmean(accel_xyz, axis=0).round(6).tolist() if accel_xyz is not None else None,
            "accel_std_xyz": np.nanstd(accel_xyz, axis=0).round(6).tolist() if accel_xyz is not None else None,
            "gyro_mean_xyz": np.nanmean(gyro_xyz, axis=0).round(6).tolist() if gyro_xyz is not None else None,
            "gyro_std_xyz": np.nanstd(gyro_xyz, axis=0).round(6).tolist() if gyro_xyz is not None else None,
            "quaternion_norm_mean": _safe_stats(quat_norms)["mean"],
            "quaternion_norm_std": _safe_stats(quat_norms)["std"],
            "quaternion_norm_min": _safe_stats(quat_norms)["min"],
            "quaternion_norm_max": _safe_stats(quat_norms)["max"],
            "nan_rows_any_column": int(frame.isna().any(axis=1).sum()),
            "invalid_teacher_rows": int((~session.teacher_valid).sum()),
            "invalid_teacher_fraction": float(np.mean(~session.teacher_valid)),
        }
        session_rows.append(row)

    duration_means = np.asarray(
        [row["target_duration_mean_s"] for row in session_rows if row["target_duration_mean_s"] is not None],
        dtype=np.float64,
    )
    reference_duration = float(np.median(duration_means)) if duration_means.size else 0.0
    for row in session_rows:
        flags: list[str] = []
        target_duration_mean = row["target_duration_mean_s"]
        if target_duration_mean is not None and reference_duration > 0.0:
            ratio = target_duration_mean / reference_duration
            if ratio < 0.75 or ratio > 1.25:
                flags.append(f"window_duration_ratio={ratio:.2f}")
        zero_dt_fraction = row["zero_dt_fraction"]
        if zero_dt_fraction is not None and zero_dt_fraction > 0.85:
            flags.append("high_zero_dt_fraction")
        if row["invalid_teacher_fraction"] > 0.05:
            flags.append("high_invalid_teacher_fraction")
        if row["window_keep_fraction"] < 0.20:
            flags.append("low_window_keep_fraction")
        quat_norm_max = row["quaternion_norm_max"]
        quat_norm_min = row["quaternion_norm_min"]
        if quat_norm_max is not None and quat_norm_min is not None:
            if abs(quat_norm_max - 1.0) > 0.01 or abs(quat_norm_min - 1.0) > 0.01:
                flags.append("quaternion_norm_deviation")
        row["flags"] = flags

    output_dir = ensure_dir(Path(args.output_dir))
    payload = {
        "config": {
            "csv_files": [str(Path(path).resolve()) for path in args.csv_files],
            "feature_names": feature_names,
            "target_mode": args.target_mode,
            "window_size": args.window_size,
            "stride": args.stride,
            "window_mode": args.window_mode,
            "window_duration_s": args.window_duration_s,
            "window_stride_s": args.window_stride_s,
            "required_valid_flags": args.required_valid_flags,
            "sort_columns": args.sort_columns,
            "quaternion_frame_mode": args.quaternion_frame_mode,
            "min_valid_label_points": args.min_valid_label_points,
            "min_target_magnitude_m": args.min_target_magnitude_m,
            "loss_weight_mode": args.loss_weight_mode,
            "loss_weight_reference_m": args.loss_weight_reference_m,
        },
        "sessions": session_rows,
    }
    save_json(payload, output_dir / "session_audit.json")
    save_history_csv(session_rows, output_dir / "session_audit.csv")

    print(f"Audit output: {output_dir}")
    for row in session_rows:
        flags = ", ".join(row["flags"]) if row["flags"] else "none"
        dt_mean_ms = row["dt_mean_ms"]
        target_mag_mean = row["target_magnitude_mean"]
        duration_mean_s = row["target_duration_mean_s"]
        print(
            f"{Path(row['file_path']).name}: rows={row['rows']}, windows={row['windows']}, "
            f"dt_mean_ms={(dt_mean_ms if dt_mean_ms is not None else float('nan')):.3f}, "
            f"target_mag_mean={(target_mag_mean if target_mag_mean is not None else float('nan')):.4f}, "
            f"duration_mean_s={(duration_mean_s if duration_mean_s is not None else float('nan')):.3f}, "
            f"flags={flags}"
        )


if __name__ == "__main__":
    main()
