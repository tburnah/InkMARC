from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset

from config import COLUMN_ALIASES, NAV_FRAME_FEATURES, WINDOW_TIME_FEATURES
from utils import NormalizationStats


@dataclass
class SessionRecord:
    session_name: str
    file_path: str
    sort_column: str
    source_feature_columns: dict[str, str | None]
    features: np.ndarray
    timestamps_s: np.ndarray
    teacher_xyz: np.ndarray
    teacher_valid: np.ndarray
    accel_xyz: np.ndarray | None
    gyro_xyz: np.ndarray | None
    gravity_xyz: np.ndarray | None
    linear_accel_xyz: np.ndarray | None
    quaternion_xyzw: np.ndarray | None
    nav_accel_xyz: np.ndarray | None
    nav_gyro_xyz: np.ndarray | None


@dataclass
class WindowSample:
    session_idx: int
    window_start: int
    window_end: int
    target_start: int
    target_end: int
    target: np.ndarray
    true_target: np.ndarray
    classical_target: np.ndarray
    target_magnitude: float
    loss_weight: float
    actual_window_duration_s: float
    nonzero_dt_count: int
    zero_dt_fraction: float


def resolve_source_column(
    canonical_name: str,
    available_columns: Sequence[str],
) -> str | None:
    available_lookup = {name.lower(): name for name in available_columns}
    for alias in COLUMN_ALIASES.get(canonical_name, [canonical_name]):
        match = available_lookup.get(alias.lower())
        if match is not None:
            return match
    return None


def _coerce_numeric_series(frame: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name is None:
        return pd.Series(np.nan, index=frame.index, dtype=np.float32)
    series = pd.to_numeric(frame[column_name], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan)


def _coerce_boolean_series(frame: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name is None:
        return pd.Series(False, index=frame.index, dtype=bool)
    numeric = pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0)
    return numeric.astype(np.float32) > 0.0


def _infer_time_scale(column_name: str, values: np.ndarray) -> float:
    if column_name.endswith("_ns"):
        return 1e-9
    if column_name.endswith("_ms"):
        return 1e-3
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 1.0
    max_value = float(np.max(np.abs(finite_values)))
    if max_value > 1e15:
        return 1e-9
    if max_value > 1e11:
        return 1e-3
    return 1.0


def _fill_numeric(values: pd.Series) -> np.ndarray:
    filled = values.astype(np.float32).ffill().bfill().fillna(0.0)
    return filled.to_numpy(dtype=np.float32)


def _normalize_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normalized = quaternions_xyzw / norms
    invalid = ~np.isfinite(normalized).all(axis=1)
    if np.any(invalid):
        normalized[invalid] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return normalized.astype(np.float32)


def _rotate_body_to_nav(
    body_vectors: np.ndarray,
    quaternions_xyzw: np.ndarray,
    quaternion_frame_mode: str,
) -> np.ndarray:
    quaternions = _normalize_quaternions(quaternions_xyzw.astype(np.float32))
    if quaternion_frame_mode == "nav_to_body":
        quaternions = quaternions.copy()
        quaternions[:, :3] *= -1.0
    elif quaternion_frame_mode != "body_to_nav":
        raise ValueError(f"Unsupported quaternion frame mode: {quaternion_frame_mode}")

    q_xyz = quaternions[:, :3]
    q_w = quaternions[:, 3:4]
    cross_1 = np.cross(q_xyz, body_vectors)
    cross_2 = np.cross(q_xyz, cross_1)
    return (body_vectors + 2.0 * (q_w * cross_1 + cross_2)).astype(np.float32)


def _build_teacher_valid_mask(
    frame: pd.DataFrame,
    required_valid_flags: Sequence[str],
) -> np.ndarray:
    xyz_columns = [
        resolve_source_column("plane_x_m", frame.columns),
        resolve_source_column("plane_y_m", frame.columns),
        resolve_source_column("plane_z_m", frame.columns),
    ]
    if any(column is None for column in xyz_columns):
        raise ValueError(
            "Teacher label columns plane_x_m/plane_y_m/plane_z_m are required. "
            "Add aliases in config.py if your logs use different names."
        )
    xyz_values = np.stack(
        [_coerce_numeric_series(frame, column).to_numpy(dtype=np.float32) for column in xyz_columns],
        axis=1,
    )
    teacher_valid = np.isfinite(xyz_values).all(axis=1)
    for flag_name in required_valid_flags:
        teacher_valid &= _coerce_boolean_series(
            frame, resolve_source_column(flag_name, frame.columns)
        ).to_numpy(dtype=bool)
    return teacher_valid


def compute_classical_delta_target(
    session: SessionRecord,
    target_start: int,
    target_end: int,
    target_mode: str,
    quaternion_frame_mode: str,
) -> np.ndarray:
    if target_start >= target_end or session.timestamps_s.size <= target_end:
        target_dim = 2 if target_mode in {"delta_xy", "polar_xy"} else 3
        if target_mode in {"delta_z", "velocity_z"}:
            target_dim = 1
        return np.zeros((target_dim,), dtype=np.float32)

    quaternion_xyzw = session.quaternion_xyzw
    if quaternion_xyzw is None:
        target_dim = 2 if target_mode in {"delta_xy", "polar_xy"} else 3
        if target_mode in {"delta_z", "velocity_z"}:
            target_dim = 1
        return np.zeros((target_dim,), dtype=np.float32)

    window_quat = quaternion_xyzw[target_start : target_end + 1]
    times = session.timestamps_s[target_start : target_end + 1].astype(np.float64)
    if times.size < 2:
        target_dim = 2 if target_mode in {"delta_xy", "polar_xy"} else 3
        if target_mode in {"delta_z", "velocity_z"}:
            target_dim = 1
        return np.zeros((target_dim,), dtype=np.float32)

    if session.linear_accel_xyz is not None:
        body_linear_accel = session.linear_accel_xyz[target_start : target_end + 1].astype(np.float32)
        nav_linear_accel = _rotate_body_to_nav(body_linear_accel, window_quat, quaternion_frame_mode)
    elif session.accel_xyz is not None and session.gravity_xyz is not None:
        body_linear_accel = (
            session.accel_xyz[target_start : target_end + 1].astype(np.float32)
            - session.gravity_xyz[target_start : target_end + 1].astype(np.float32)
        )
        nav_linear_accel = _rotate_body_to_nav(body_linear_accel, window_quat, quaternion_frame_mode)
    elif session.accel_xyz is not None:
        nav_accel = _rotate_body_to_nav(
            session.accel_xyz[target_start : target_end + 1].astype(np.float32),
            window_quat,
            quaternion_frame_mode,
        )
        gravity_reference = np.median(nav_accel[: min(5, nav_accel.shape[0])], axis=0, keepdims=True)
        nav_linear_accel = nav_accel - gravity_reference
    else:
        target_dim = 2 if target_mode in {"delta_xy", "polar_xy"} else 3
        if target_mode in {"delta_z", "velocity_z"}:
            target_dim = 1
        return np.zeros((target_dim,), dtype=np.float32)

    nav_linear_accel = np.nan_to_num(nav_linear_accel.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    velocity = np.zeros(3, dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    for idx in range(times.size - 1):
        dt = float(times[idx + 1] - times[idx])
        if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            continue
        average_accel = 0.5 * (nav_linear_accel[idx] + nav_linear_accel[idx + 1])
        position += velocity * dt + 0.5 * average_accel * (dt * dt)
        velocity += average_accel * dt

    delta_xyz = position.astype(np.float32)
    if target_mode == "delta_xyz":
        return delta_xyz
    if target_mode == "delta_xy":
        return delta_xyz[:2]
    if target_mode == "delta_z":
        return np.asarray([delta_xyz[2]], dtype=np.float32)
    if target_mode == "polar_xy":
        planar = delta_xyz[:2]
        planar_distance = float(np.linalg.norm(planar))
        heading = float(np.arctan2(planar[1], planar[0])) if planar_distance > 1e-8 else 0.0
        return np.asarray([planar_distance, heading], dtype=np.float32)
    if target_mode == "velocity_xyz":
        return velocity.astype(np.float32)
    if target_mode == "velocity_z":
        return np.asarray([velocity[2]], dtype=np.float32)
    raise ValueError(f"Unsupported target mode for classical baseline: {target_mode}")


def load_session_csv(
    file_path: str,
    feature_names: Sequence[str],
    required_valid_flags: Sequence[str],
    sort_column_candidates: Sequence[str],
    quaternion_frame_mode: str,
) -> SessionRecord:
    frame = pd.read_csv(file_path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]

    sort_column = None
    for candidate in sort_column_candidates:
        resolved = resolve_source_column(candidate, frame.columns)
        if resolved is not None:
            sort_column = resolved
            break
    if sort_column is None:
        raise ValueError(
            f"Could not find a sort/timestamp column in {file_path}. "
            "Update --sort-columns or add aliases in config.py."
        )

    frame[sort_column] = pd.to_numeric(frame[sort_column], errors="coerce")
    frame = frame.sort_values(sort_column, kind="mergesort").reset_index(drop=True)

    source_feature_columns: dict[str, str | None] = {}
    feature_array_map: dict[str, np.ndarray] = {}
    if NAV_FRAME_FEATURES.intersection(feature_names):
        raw_dependency_names = [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "rot_w",
        ]
        raw_cache: dict[str, np.ndarray] = {}
        for dependency_name in raw_dependency_names:
            source_column = resolve_source_column(dependency_name, frame.columns)
            if source_column is None:
                raise ValueError(
                    f"Derived nav-frame feature requested, but '{dependency_name}' is missing in {file_path}."
                )
            source_feature_columns.setdefault(dependency_name, source_column)
            raw_cache[dependency_name] = _fill_numeric(_coerce_numeric_series(frame, source_column))

        raw_accel = np.stack(
            [raw_cache["accel_x"], raw_cache["accel_y"], raw_cache["accel_z"]],
            axis=1,
        ).astype(np.float32)
        raw_gyro = np.stack(
            [raw_cache["gyro_x"], raw_cache["gyro_y"], raw_cache["gyro_z"]],
            axis=1,
        ).astype(np.float32)
        raw_quat = np.stack(
            [raw_cache["rot_x"], raw_cache["rot_y"], raw_cache["rot_z"], raw_cache["rot_w"]],
            axis=1,
        ).astype(np.float32)
        nav_accel = _rotate_body_to_nav(raw_accel, raw_quat, quaternion_frame_mode)
        nav_gyro = _rotate_body_to_nav(raw_gyro, raw_quat, quaternion_frame_mode)
        feature_array_map.update(
            {
                "nav_accel_x": nav_accel[:, 0],
                "nav_accel_y": nav_accel[:, 1],
                "nav_accel_z": nav_accel[:, 2],
                "nav_gyro_x": nav_gyro[:, 0],
                "nav_gyro_y": nav_gyro[:, 1],
                "nav_gyro_z": nav_gyro[:, 2],
            }
        )

    for feature_name in feature_names:
        if feature_name in NAV_FRAME_FEATURES:
            source_feature_columns[feature_name] = "<derived>"
            continue
        if feature_name in WINDOW_TIME_FEATURES:
            source_feature_columns[feature_name] = "<dynamic>"
            continue
        source_column = resolve_source_column(feature_name, frame.columns)
        source_feature_columns[feature_name] = source_column
        feature_array_map[feature_name] = _fill_numeric(_coerce_numeric_series(frame, source_column))

    feature_arrays = [feature_array_map[feature_name] for feature_name in feature_names if feature_name not in WINDOW_TIME_FEATURES]
    num_rows = frame.shape[0]
    feature_matrix = (
        np.stack(feature_arrays, axis=1).astype(np.float32)
        if feature_arrays
        else np.zeros((num_rows, 0), dtype=np.float32)
    )

    raw_timestamps = _fill_numeric(_coerce_numeric_series(frame, sort_column)).astype(np.float64)
    timestamps_s = raw_timestamps * _infer_time_scale(sort_column, raw_timestamps)

    accel_columns = [
        resolve_source_column("accel_x", frame.columns),
        resolve_source_column("accel_y", frame.columns),
        resolve_source_column("accel_z", frame.columns),
    ]
    accel_xyz = (
        np.stack(
            [_fill_numeric(_coerce_numeric_series(frame, column)) for column in accel_columns],
            axis=1,
        ).astype(np.float32)
        if all(column is not None for column in accel_columns)
        else None
    )

    gyro_columns = [
        resolve_source_column("gyro_x", frame.columns),
        resolve_source_column("gyro_y", frame.columns),
        resolve_source_column("gyro_z", frame.columns),
    ]
    gyro_xyz = (
        np.stack(
            [_fill_numeric(_coerce_numeric_series(frame, column)) for column in gyro_columns],
            axis=1,
        ).astype(np.float32)
        if all(column is not None for column in gyro_columns)
        else None
    )

    gravity_columns = [
        resolve_source_column("gravity_x", frame.columns),
        resolve_source_column("gravity_y", frame.columns),
        resolve_source_column("gravity_z", frame.columns),
    ]
    gravity_xyz = (
        np.stack(
            [_fill_numeric(_coerce_numeric_series(frame, column)) for column in gravity_columns],
            axis=1,
        ).astype(np.float32)
        if all(column is not None for column in gravity_columns)
        else None
    )

    linear_accel_columns = [
        resolve_source_column("linear_accel_x", frame.columns),
        resolve_source_column("linear_accel_y", frame.columns),
        resolve_source_column("linear_accel_z", frame.columns),
    ]
    linear_accel_xyz = (
        np.stack(
            [_fill_numeric(_coerce_numeric_series(frame, column)) for column in linear_accel_columns],
            axis=1,
        ).astype(np.float32)
        if all(column is not None for column in linear_accel_columns)
        else None
    )

    quaternion_columns = [
        resolve_source_column("rot_x", frame.columns),
        resolve_source_column("rot_y", frame.columns),
        resolve_source_column("rot_z", frame.columns),
        resolve_source_column("rot_w", frame.columns),
    ]
    quaternion_xyzw = (
        np.stack(
            [_fill_numeric(_coerce_numeric_series(frame, column)) for column in quaternion_columns],
            axis=1,
        ).astype(np.float32)
        if all(column is not None for column in quaternion_columns)
        else None
    )

    nav_accel_xyz = None
    nav_gyro_xyz = None
    if accel_xyz is not None and gyro_xyz is not None and quaternion_xyzw is not None:
        nav_accel_xyz = _rotate_body_to_nav(accel_xyz, quaternion_xyzw, quaternion_frame_mode)
        nav_gyro_xyz = _rotate_body_to_nav(gyro_xyz, quaternion_xyzw, quaternion_frame_mode)

    teacher_xyz = np.stack(
        [
            _fill_numeric(_coerce_numeric_series(frame, resolve_source_column("plane_x_m", frame.columns))),
            _fill_numeric(_coerce_numeric_series(frame, resolve_source_column("plane_y_m", frame.columns))),
            _fill_numeric(_coerce_numeric_series(frame, resolve_source_column("plane_z_m", frame.columns))),
        ],
        axis=1,
    ).astype(np.float32)
    teacher_valid = _build_teacher_valid_mask(frame, required_valid_flags)

    session_id_column = resolve_source_column("session_id", frame.columns)
    if session_id_column is not None and frame[session_id_column].notna().any():
        session_name = str(frame[session_id_column].dropna().iloc[0])
    else:
        session_name = Path(file_path).stem

    return SessionRecord(
        session_name=session_name,
        file_path=str(Path(file_path).resolve()),
        sort_column=sort_column,
        source_feature_columns=source_feature_columns,
        features=feature_matrix,
        timestamps_s=timestamps_s.astype(np.float64),
        teacher_xyz=teacher_xyz,
        teacher_valid=teacher_valid.astype(bool),
        accel_xyz=accel_xyz,
        gyro_xyz=gyro_xyz,
        gravity_xyz=gravity_xyz,
        linear_accel_xyz=linear_accel_xyz,
        quaternion_xyzw=quaternion_xyzw,
        nav_accel_xyz=nav_accel_xyz,
        nav_gyro_xyz=nav_gyro_xyz,
    )


class SensorWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        csv_files: Sequence[str],
        feature_names: Sequence[str],
        target_mode: str,
        window_size: int,
        stride: int,
        required_valid_flags: Sequence[str],
        sort_column_candidates: Sequence[str],
        window_mode: str = "rows",
        window_duration_s: float = 1.0,
        window_stride_s: float = 0.10,
        quaternion_frame_mode: str = "body_to_nav",
        loss_weight_mode: str = "none",
        loss_weight_reference_m: float = 0.05,
        loss_weight_min: float = 1.0,
        loss_weight_max: float = 5.0,
        normalization: NormalizationStats | None = None,
        fit_normalization: bool = False,
        normalize_targets: bool = False,
        min_valid_label_points: int = 2,
        min_target_magnitude_m: float = 0.0,
        use_classical_residual_target: bool = False,
        classical_residual_mode: str = "additive",
    ) -> None:
        if not csv_files:
            raise ValueError("At least one CSV file is required.")
        self.csv_files = [str(Path(path).resolve()) for path in csv_files]
        self.feature_names = list(feature_names)
        self.static_feature_names = [name for name in self.feature_names if name not in WINDOW_TIME_FEATURES]
        self.dynamic_feature_names = [name for name in self.feature_names if name in WINDOW_TIME_FEATURES]
        self.target_mode = target_mode
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.window_mode = window_mode
        self.window_duration_s = float(window_duration_s)
        self.window_stride_s = float(window_stride_s)
        self.required_valid_flags = list(required_valid_flags)
        self.sort_column_candidates = list(sort_column_candidates)
        self.quaternion_frame_mode = quaternion_frame_mode
        self.loss_weight_mode = loss_weight_mode
        self.loss_weight_reference_m = float(loss_weight_reference_m)
        self.loss_weight_min = float(loss_weight_min)
        self.loss_weight_max = float(loss_weight_max)
        self.normalize_targets = normalize_targets
        self.min_valid_label_points = min_valid_label_points
        self.min_target_magnitude_m = float(min_target_magnitude_m)
        self.use_classical_residual_target = bool(use_classical_residual_target)
        self.classical_residual_mode = classical_residual_mode
        self.num_windows_filtered_by_target_magnitude = 0
        if self.window_mode not in {"rows", "time"}:
            raise ValueError(f"Unsupported window mode: {self.window_mode}")
        if self.window_mode == "time":
            if self.window_duration_s <= 0.0:
                raise ValueError("--window-duration-s must be > 0 when window-mode=time.")
            if self.window_stride_s <= 0.0:
                raise ValueError("--window-stride-s must be > 0 when window-mode=time.")
        if self.use_classical_residual_target:
            if self.target_mode != "delta_xy":
                raise ValueError("Classical residual target is currently supported only for target_mode=delta_xy.")
            if self.classical_residual_mode != "additive":
                raise ValueError(f"Unsupported classical residual mode: {self.classical_residual_mode}")

        self.sessions = [
            load_session_csv(
                file_path=path,
                feature_names=self.static_feature_names,
                required_valid_flags=self.required_valid_flags,
                sort_column_candidates=self.sort_column_candidates,
                quaternion_frame_mode=self.quaternion_frame_mode,
            )
            for path in self.csv_files
        ]
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(
                "No valid sliding windows were created. Reduce --window-size, lower "
                "--min-valid-label-points, or review your validity flags."
            )

        self.target_names = self._target_names()
        self.normalization = normalization
        if fit_normalization:
            self.normalization = self.compute_normalization_stats()
        if self.normalization is None:
            self.normalization = None

    def _target_names(self) -> list[str]:
        if self.target_mode == "delta_xyz":
            return ["delta_x", "delta_y", "delta_z"]
        if self.target_mode == "delta_xy":
            return ["delta_x", "delta_y"]
        if self.target_mode == "delta_z":
            return ["delta_z"]
        if self.target_mode == "polar_xy":
            return ["delta_distance_xy", "delta_heading"]
        if self.target_mode == "velocity_xyz":
            return ["vel_x", "vel_y", "vel_z"]
        if self.target_mode == "velocity_z":
            return ["vel_z"]
        raise ValueError(f"Unsupported target mode: {self.target_mode}")

    def _compute_target(
        self,
        session: SessionRecord,
        target_start: int,
        target_end: int,
    ) -> np.ndarray | None:
        delta_xyz = session.teacher_xyz[target_end] - session.teacher_xyz[target_start]
        if not np.all(np.isfinite(delta_xyz)):
            return None

        if self.target_mode == "delta_xyz":
            return delta_xyz.astype(np.float32)
        if self.target_mode == "delta_xy":
            return delta_xyz[:2].astype(np.float32)
        if self.target_mode == "delta_z":
            return np.asarray([delta_xyz[2]], dtype=np.float32)
        if self.target_mode == "polar_xy":
            planar_delta = delta_xyz[:2].astype(np.float32)
            planar_distance = float(np.linalg.norm(planar_delta))
            heading = float(np.arctan2(planar_delta[1], planar_delta[0])) if planar_distance > 1e-8 else 0.0
            return np.asarray([planar_distance, heading], dtype=np.float32)

        dt = float(session.timestamps_s[target_end] - session.timestamps_s[target_start])
        if dt <= 0.0:
            return None
        velocity_xyz = delta_xyz / dt
        if self.target_mode == "velocity_xyz":
            return velocity_xyz.astype(np.float32)
        if self.target_mode == "velocity_z":
            return np.asarray([velocity_xyz[2]], dtype=np.float32)
        raise ValueError(f"Unsupported target mode: {self.target_mode}")

    def _iter_row_windows(self, session: SessionRecord) -> list[tuple[int, int]]:
        total_rows = session.features.shape[0]
        if total_rows < self.window_size:
            return []
        return [
            (window_start, window_start + self.window_size)
            for window_start in range(0, total_rows - self.window_size + 1, self.stride)
        ]

    def _iter_time_windows(self, session: SessionRecord) -> list[tuple[int, int]]:
        times = session.timestamps_s.astype(np.float64)
        total_rows = times.size
        if total_rows < 2:
            return []
        windows: list[tuple[int, int]] = []
        start = 0
        while start < total_rows - 1:
            start_time = float(times[start])
            target_end_time = start_time + self.window_duration_s
            end_index = int(np.searchsorted(times, target_end_time, side="left"))
            if end_index >= total_rows:
                break
            window_end = end_index + 1
            if window_end - start < 2:
                start += 1
                continue
            actual_duration_s = float(times[window_end - 1] - times[start])
            if not np.isfinite(actual_duration_s) or actual_duration_s <= 0.0:
                start += 1
                continue
            windows.append((start, window_end))
            next_start_time = start_time + self.window_stride_s
            next_start = int(np.searchsorted(times, next_start_time, side="left"))
            if next_start <= start:
                next_start = start + 1
            start = next_start
        return windows

    def _window_ranges(self, session: SessionRecord) -> list[tuple[int, int]]:
        if self.window_mode == "rows":
            return self._iter_row_windows(session)
        return self._iter_time_windows(session)

    def _window_timing_stats(
        self,
        session: SessionRecord,
        window_start: int,
        window_end: int,
    ) -> tuple[float, int, float]:
        times = session.timestamps_s[window_start:window_end].astype(np.float64)
        if times.size < 2:
            return 0.0, 0, 1.0
        dt = np.diff(times)
        actual_window_duration_s = float(max(times[-1] - times[0], 0.0))
        nonzero_dt_count = int(np.sum(dt > 0.0))
        zero_dt_fraction = float(np.mean(dt <= 0.0)) if dt.size else 1.0
        return actual_window_duration_s, nonzero_dt_count, zero_dt_fraction

    def _time_feature_matrix(
        self,
        session: SessionRecord,
        window_start: int,
        window_end: int,
    ) -> np.ndarray:
        if not self.dynamic_feature_names:
            return np.zeros((window_end - window_start, 0), dtype=np.float32)
        times = session.timestamps_s[window_start:window_end].astype(np.float64)
        parts: dict[str, np.ndarray] = {}
        if "step_dt_s" in self.dynamic_feature_names:
            step_dt = np.diff(times, prepend=times[0]).astype(np.float32)
            step_dt[0] = 0.0
            parts["step_dt_s"] = step_dt
        if "elapsed_time_s" in self.dynamic_feature_names:
            elapsed = (times - times[0]).astype(np.float32)
            parts["elapsed_time_s"] = elapsed
        return np.stack([parts[name] for name in self.dynamic_feature_names], axis=1).astype(np.float32)

    def _build_samples(self) -> list[WindowSample]:
        samples: list[WindowSample] = []
        for session_idx, session in enumerate(self.sessions):
            for window_start, window_end in self._window_ranges(session):
                valid_relative = np.flatnonzero(session.teacher_valid[window_start:window_end])
                if valid_relative.size < self.min_valid_label_points:
                    continue
                target_start = window_start + int(valid_relative[0])
                target_end = window_start + int(valid_relative[-1])
                if target_end <= target_start:
                    continue
                target = self._compute_target(session, target_start, target_end)
                if target is None or not np.all(np.isfinite(target)):
                    continue
                classical_target = compute_classical_delta_target(
                    session=session,
                    target_start=target_start,
                    target_end=target_end,
                    target_mode=self.target_mode,
                    quaternion_frame_mode=self.quaternion_frame_mode,
                )
                if not np.all(np.isfinite(classical_target)):
                    classical_target = np.zeros_like(target)
                model_target = target
                if self.use_classical_residual_target:
                    model_target = (target - classical_target).astype(np.float32)
                target_magnitude = self._target_motion_magnitude(target)
                if target_magnitude < self.min_target_magnitude_m:
                    self.num_windows_filtered_by_target_magnitude += 1
                    continue
                loss_weight = self._compute_loss_weight(target_magnitude)
                actual_window_duration_s, nonzero_dt_count, zero_dt_fraction = self._window_timing_stats(
                    session=session,
                    window_start=window_start,
                    window_end=window_end,
                )
                samples.append(
                    WindowSample(
                        session_idx=session_idx,
                        window_start=window_start,
                        window_end=window_end,
                        target_start=target_start,
                        target_end=target_end,
                        target=model_target.astype(np.float32),
                        true_target=target.astype(np.float32),
                        classical_target=classical_target.astype(np.float32),
                        target_magnitude=target_magnitude,
                        loss_weight=loss_weight,
                        actual_window_duration_s=actual_window_duration_s,
                        nonzero_dt_count=nonzero_dt_count,
                        zero_dt_fraction=zero_dt_fraction,
                    )
                )
        return samples

    def _target_motion_magnitude(self, target: np.ndarray) -> float:
        if self.target_mode == "delta_xyz":
            return float(np.linalg.norm(target))
        if self.target_mode == "delta_xy":
            return float(np.linalg.norm(target))
        if self.target_mode == "delta_z":
            return float(abs(target[0]))
        if self.target_mode == "polar_xy":
            return float(abs(target[0]))
        if self.target_mode == "velocity_xyz":
            return float(np.linalg.norm(target))
        if self.target_mode == "velocity_z":
            return float(abs(target[0]))
        raise ValueError(f"Unsupported target mode: {self.target_mode}")

    def _compute_loss_weight(self, target_magnitude: float) -> float:
        if self.loss_weight_mode == "none":
            return 1.0
        if self.loss_weight_mode == "target_magnitude":
            reference = max(self.loss_weight_reference_m, 1e-6)
            weight = 1.0 + (target_magnitude / reference)
            return float(np.clip(weight, self.loss_weight_min, self.loss_weight_max))
        raise ValueError(f"Unsupported loss weight mode: {self.loss_weight_mode}")

    def compute_normalization_stats(self) -> NormalizationStats:
        if self.dynamic_feature_names:
            window_feature_rows = []
            for sample in self.samples:
                session = self.sessions[sample.session_idx]
                static_features = session.features[sample.window_start : sample.window_end].astype(np.float32)
                dynamic_features = self._time_feature_matrix(session, sample.window_start, sample.window_end)
                window_feature_rows.append(np.concatenate([static_features, dynamic_features], axis=1))
            feature_matrix = np.concatenate(window_feature_rows, axis=0)
        else:
            feature_matrix = np.concatenate([session.features for session in self.sessions], axis=0)
        feature_mean = feature_matrix.mean(axis=0)
        feature_std = feature_matrix.std(axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

        target_mean: np.ndarray | None = None
        target_std: np.ndarray | None = None
        if self.normalize_targets:
            target_matrix = np.stack([sample.target for sample in self.samples], axis=0)
            target_mean = target_matrix.mean(axis=0)
            target_std = target_matrix.std(axis=0)
            target_std = np.where(target_std < 1e-6, 1.0, target_std)

        return NormalizationStats(
            feature_names=list(self.feature_names),
            feature_mean=feature_mean.astype(np.float64).tolist(),
            feature_std=feature_std.astype(np.float64).tolist(),
            target_names=list(self.target_names),
            target_mean=None if target_mean is None else target_mean.astype(np.float64).tolist(),
            target_std=None if target_std is None else target_std.astype(np.float64).tolist(),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        session = self.sessions[sample.session_idx]

        static_features = session.features[sample.window_start : sample.window_end].copy()
        dynamic_features = self._time_feature_matrix(session, sample.window_start, sample.window_end)
        features = (
            np.concatenate([static_features, dynamic_features], axis=1)
            if dynamic_features.shape[1] > 0
            else static_features
        )
        if self.normalization is not None:
            features = (
                features - self.normalization.feature_mean_array
            ) / self.normalization.feature_std_array
        features = features.transpose(1, 0)

        target = sample.target.copy()
        if (
            self.normalize_targets
            and self.normalization is not None
            and self.normalization.target_mean is not None
            and self.normalization.target_std is not None
        ):
            target = (
                target - self.normalization.target_mean_array
            ) / self.normalization.target_std_array

        metadata = {
            "session_name": session.session_name,
            "session_idx": sample.session_idx,
            "file_path": session.file_path,
            "sequence_length": int(features.shape[1]),
            "window_start": sample.window_start,
            "window_end": sample.window_end,
            "target_start": sample.target_start,
            "target_end": sample.target_end,
            "start_time_s": float(session.timestamps_s[sample.target_start]),
            "end_time_s": float(session.timestamps_s[sample.target_end]),
            "target_duration_s": float(
                max(session.timestamps_s[sample.target_end] - session.timestamps_s[sample.target_start], 0.0)
            ),
            "target_magnitude": float(sample.target_magnitude),
            "actual_window_duration_s": float(sample.actual_window_duration_s),
            "nonzero_dt_count": int(sample.nonzero_dt_count),
            "zero_dt_fraction": float(sample.zero_dt_fraction),
            "teacher_start_xyz": session.teacher_xyz[sample.target_start].astype(np.float32),
            "teacher_end_xyz": session.teacher_xyz[sample.target_end].astype(np.float32),
            "true_target": sample.true_target.astype(np.float32),
            "classical_target": sample.classical_target.astype(np.float32),
            "loss_weight": float(sample.loss_weight),
        }
        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
            metadata,
        )

    def summary(self) -> dict[str, Any]:
        actual_window_durations = np.asarray([sample.actual_window_duration_s for sample in self.samples], dtype=np.float64)
        zero_dt_fractions = np.asarray([sample.zero_dt_fraction for sample in self.samples], dtype=np.float64)
        nonzero_dt_counts = np.asarray([sample.nonzero_dt_count for sample in self.samples], dtype=np.float64)
        return {
            "num_sessions": len(self.sessions),
            "num_samples": len(self.samples),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "window_size": self.window_size,
            "stride": self.stride,
            "window_mode": self.window_mode,
            "window_duration_s": self.window_duration_s,
            "window_stride_s": self.window_stride_s,
            "min_target_magnitude_m": self.min_target_magnitude_m,
            "num_windows_filtered_by_target_magnitude": self.num_windows_filtered_by_target_magnitude,
            "csv_files": self.csv_files,
            "quaternion_frame_mode": self.quaternion_frame_mode,
            "loss_weight_mode": self.loss_weight_mode,
            "use_classical_residual_target": self.use_classical_residual_target,
            "classical_residual_mode": self.classical_residual_mode,
            "timing_diagnostics": {
                "actual_window_duration_s_mean": float(np.mean(actual_window_durations)),
                "actual_window_duration_s_std": float(np.std(actual_window_durations)),
                "actual_window_duration_s_min": float(np.min(actual_window_durations)),
                "actual_window_duration_s_max": float(np.max(actual_window_durations)),
                "zero_dt_fraction_mean": float(np.mean(zero_dt_fractions)),
                "nonzero_dt_count_mean": float(np.mean(nonzero_dt_counts)),
            },
            "source_feature_columns": {
                session.session_name: session.source_feature_columns for session in self.sessions
            },
        }


def collate_sensor_windows(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    inputs, targets, metadata_list = zip(*batch)
    max_length = max(int(item.shape[1]) for item in inputs)
    padded_inputs: list[torch.Tensor] = []
    sequence_lengths: list[int] = []
    for item in inputs:
        current_length = int(item.shape[1])
        sequence_lengths.append(current_length)
        if current_length < max_length:
            pad = torch.zeros(
                (item.shape[0], max_length - current_length),
                dtype=item.dtype,
            )
            item = torch.cat([item, pad], dim=1)
        padded_inputs.append(item)
    collated_inputs = torch.stack(padded_inputs, dim=0)
    collated_targets = torch.stack(list(targets), dim=0)
    metadata = default_collate(list(metadata_list))
    metadata["sequence_length"] = torch.as_tensor(sequence_lengths, dtype=torch.int64)
    return collated_inputs, collated_targets, metadata


class SurfaceRolloutDataset(
    Dataset[tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]]
):
    def __init__(
        self,
        base_dataset: SensorWindowDataset,
        rollout_num_windows: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.rollout_num_windows = max(int(rollout_num_windows), 1)
        self.rollout_indices = self._build_rollout_indices()
        if not self.rollout_indices:
            raise ValueError(
                "No contiguous rollout samples were created. Reduce "
                "--surface-rollout-num-windows or review the window settings."
            )

    def _build_rollout_indices(self) -> list[list[int]]:
        if self.rollout_num_windows <= 1:
            return [[index] for index in range(len(self.base_dataset))]

        groups: list[list[int]] = []
        session_indices: dict[int, list[int]] = {}
        for index, sample in enumerate(self.base_dataset.samples):
            session_indices.setdefault(sample.session_idx, []).append(index)

        for indices in session_indices.values():
            for start in range(0, len(indices) - self.rollout_num_windows + 1):
                candidate = indices[start : start + self.rollout_num_windows]
                contiguous = True
                if self.base_dataset.window_mode == "rows":
                    for left, right in zip(candidate[:-1], candidate[1:]):
                        left_start = self.base_dataset.samples[left].window_start
                        right_start = self.base_dataset.samples[right].window_start
                        if right_start - left_start != self.base_dataset.stride:
                            contiguous = False
                            break
                if contiguous:
                    groups.append(candidate)
        return groups

    def __len__(self) -> int:
        return len(self.rollout_indices)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
        rollout_inputs: list[torch.Tensor] = []
        rollout_targets: list[torch.Tensor] = []
        rollout_metadata: list[dict[str, Any]] = []
        for window_index in self.rollout_indices[index]:
            inputs, targets, metadata = self.base_dataset[window_index]
            rollout_inputs.append(inputs)
            rollout_targets.append(targets)
            rollout_metadata.append(metadata)
        return rollout_inputs, rollout_targets, rollout_metadata

    def summary(self) -> dict[str, Any]:
        return {
            "num_rollout_samples": len(self.rollout_indices),
            "rollout_num_windows": self.rollout_num_windows,
            "base_num_windows": len(self.base_dataset),
            "window_mode": self.base_dataset.window_mode,
            "stride": self.base_dataset.stride,
        }


def collate_surface_rollouts(
    batch: list[tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not batch:
        raise ValueError("Surface rollout batch is empty.")

    rollout_num_windows = len(batch[0][0])
    if rollout_num_windows == 0:
        raise ValueError("Surface rollout batch contains an empty rollout sample.")

    flat_batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
    for rollout_inputs, rollout_targets, rollout_metadata in batch:
        if len(rollout_inputs) != rollout_num_windows:
            raise ValueError("Rollout batch contains inconsistent rollout lengths.")
        for inputs, targets, metadata in zip(rollout_inputs, rollout_targets, rollout_metadata):
            flat_batch.append((inputs, targets, metadata))

    flat_inputs, flat_targets, flat_metadata = collate_sensor_windows(flat_batch)
    batch_size = len(batch)
    num_channels = int(flat_inputs.shape[1])
    sequence_length = int(flat_inputs.shape[2])
    target_dim = int(flat_targets.shape[1])

    rollout_inputs = flat_inputs.reshape(batch_size, rollout_num_windows, num_channels, sequence_length)
    rollout_targets = flat_targets.reshape(batch_size, rollout_num_windows, target_dim)

    rollout_metadata: dict[str, Any] = {"flat_metadata": flat_metadata}
    for key, value in flat_metadata.items():
        if torch.is_tensor(value):
            rollout_metadata[key] = value.reshape(batch_size, rollout_num_windows, *value.shape[1:])
        else:
            flat_values = list(value)
            rollout_metadata[key] = [
                flat_values[start : start + rollout_num_windows]
                for start in range(0, len(flat_values), rollout_num_windows)
            ]
    rollout_metadata["rollout_num_windows"] = torch.full(
        (batch_size,),
        rollout_num_windows,
        dtype=torch.int64,
    )
    return rollout_inputs, rollout_targets, rollout_metadata
