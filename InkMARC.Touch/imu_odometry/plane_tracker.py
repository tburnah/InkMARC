from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from calibration import PlaneCalibration, build_plane_calibration


@dataclass
class PlaneRelativeTrackerConfig:
    position_blend: float = 0.70
    velocity_blend: float = 0.35
    uncertainty_scale: float = 0.10
    velocity_damping_per_s: float = 0.35
    height_constraint_gain: float = 8.0
    normal_accel_gain: float = 0.05
    max_height_offset_m: float = 0.05


class PlaneRelativeTracker:
    def __init__(
        self,
        calibration: PlaneCalibration,
        config: PlaneRelativeTrackerConfig,
    ) -> None:
        self.calibration = calibration
        self.config = config
        self.position_xy = np.zeros(2, dtype=np.float64)
        self.velocity_xy = np.zeros(2, dtype=np.float64)
        self.height_m = float(calibration.initial_height_m)

    def initialize(self, initial_velocity_xy: np.ndarray) -> None:
        self.position_xy = np.zeros(2, dtype=np.float64)
        self.velocity_xy = np.asarray(initial_velocity_xy, dtype=np.float64).reshape(2)
        self.height_m = float(self.calibration.initial_height_m)

    def propagate(
        self,
        segment_timestamps_s: np.ndarray,
        segment_linear_accel_nav: np.ndarray,
    ) -> None:
        if segment_timestamps_s.size < 2 or segment_linear_accel_nav.shape[0] < 2:
            return
        for idx in range(segment_timestamps_s.size - 1):
            dt = float(segment_timestamps_s[idx + 1] - segment_timestamps_s[idx])
            if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
                continue
            average_accel_nav = 0.5 * (
                segment_linear_accel_nav[idx].astype(np.float64)
                + segment_linear_accel_nav[idx + 1].astype(np.float64)
            )
            planar_accel_xy = average_accel_nav[:2]
            self.position_xy += self.velocity_xy * dt + 0.5 * planar_accel_xy * (dt * dt)
            self.velocity_xy += planar_accel_xy * dt
            damping = np.exp(-self.config.velocity_damping_per_s * dt)
            self.velocity_xy *= damping
            self.height_m += 0.5 * self.config.normal_accel_gain * float(average_accel_nav[2]) * (dt * dt)

        total_duration_s = float(segment_timestamps_s[-1] - segment_timestamps_s[0])
        if np.isfinite(total_duration_s) and total_duration_s > 0.0:
            pull = 1.0 - np.exp(-self.config.height_constraint_gain * total_duration_s)
            self.height_m += pull * (self.calibration.initial_height_m - self.height_m)
        min_height = self.calibration.initial_height_m - self.config.max_height_offset_m
        max_height = self.calibration.initial_height_m + self.config.max_height_offset_m
        self.height_m = float(np.clip(self.height_m, a_min=min_height, a_max=max_height))

    def update(
        self,
        measured_position_xy: np.ndarray,
        measured_velocity_xy: np.ndarray,
        measurement_sigma_xy: np.ndarray | None,
    ) -> tuple[float, float]:
        sigma_mean = (
            float(np.mean(np.clip(np.asarray(measurement_sigma_xy, dtype=np.float64), a_min=0.0, a_max=None)))
            if measurement_sigma_xy is not None
            else 0.0
        )
        attenuation = 1.0 / (1.0 + self.config.uncertainty_scale * sigma_mean)
        position_blend = float(np.clip(self.config.position_blend * attenuation, 0.0, 1.0))
        velocity_blend = float(np.clip(self.config.velocity_blend * attenuation, 0.0, 1.0))
        measured_position = np.asarray(measured_position_xy, dtype=np.float64).reshape(2)
        measured_velocity = np.asarray(measured_velocity_xy, dtype=np.float64).reshape(2)
        self.position_xy += position_blend * (measured_position - self.position_xy)
        self.velocity_xy += velocity_blend * (measured_velocity - self.velocity_xy)
        return position_blend, velocity_blend


def session_linear_accel_nav(session, quaternion_frame_mode: str) -> np.ndarray:
    if session.linear_accel_xyz is not None and session.quaternion_xyzw is not None:
        return _rotate_body_to_nav(
            np.asarray(session.linear_accel_xyz, dtype=np.float32),
            np.asarray(session.quaternion_xyzw, dtype=np.float32),
            quaternion_frame_mode=quaternion_frame_mode,
        )
    if (
        session.accel_xyz is not None
        and session.gravity_xyz is not None
        and session.quaternion_xyzw is not None
    ):
        body_linear_accel = (
            np.asarray(session.accel_xyz, dtype=np.float32)
            - np.asarray(session.gravity_xyz, dtype=np.float32)
        )
        return _rotate_body_to_nav(
            body_linear_accel,
            np.asarray(session.quaternion_xyzw, dtype=np.float32),
            quaternion_frame_mode=quaternion_frame_mode,
        )
    if session.nav_accel_xyz is not None:
        nav_accel = np.asarray(session.nav_accel_xyz, dtype=np.float32)
        gravity_reference = np.median(nav_accel[: min(nav_accel.shape[0], 5)], axis=0, keepdims=True)
        return (nav_accel - gravity_reference).astype(np.float32)
    return np.zeros_like(np.asarray(session.teacher_xyz, dtype=np.float32))


def rollout_plane_relative_tracker(
    session,
    session_sample_indices: list[int],
    learned_positions_xy: np.ndarray,
    learned_displacements_xy: np.ndarray,
    measurement_sigma_xy: np.ndarray | None,
    dataset,
    quaternion_frame_mode: str,
    tracker_config: PlaneRelativeTrackerConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if learned_positions_xy.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {
                "mean_position_blend": 0.0,
                "mean_velocity_blend": 0.0,
                "mean_abs_height_offset_m": 0.0,
                "max_abs_height_offset_m": 0.0,
            },
        )

    first_sample = dataset.samples[session_sample_indices[0]]
    calibration = build_plane_calibration(
        session,
        reference_index=first_sample.target_start,
        quaternion_frame_mode=quaternion_frame_mode,
    )
    linear_accel_nav = session_linear_accel_nav(session, quaternion_frame_mode=quaternion_frame_mode)
    tracker = PlaneRelativeTracker(calibration=calibration, config=tracker_config)

    first_duration = max(
        float(session.timestamps_s[first_sample.target_end] - session.timestamps_s[first_sample.target_start]),
        1e-6,
    )
    initial_velocity_xy = learned_displacements_xy[0].astype(np.float64) / first_duration
    tracker.initialize(initial_velocity_xy=initial_velocity_xy)

    positions_xy = np.zeros_like(learned_positions_xy, dtype=np.float32)
    heights_m = np.zeros((learned_positions_xy.shape[0],), dtype=np.float32)
    position_blends: list[float] = []
    velocity_blends: list[float] = []
    height_offsets: list[float] = []

    previous_end = int(first_sample.target_start)
    for local_idx, global_idx in enumerate(session_sample_indices):
        sample = dataset.samples[global_idx]
        target_end = int(sample.target_end)
        segment_start = max(previous_end, 0)
        segment_end = min(target_end, session.timestamps_s.shape[0] - 1)
        if segment_end > segment_start:
            tracker.propagate(
                segment_timestamps_s=np.asarray(
                    session.timestamps_s[segment_start : segment_end + 1],
                    dtype=np.float64,
                ),
                segment_linear_accel_nav=np.asarray(
                    linear_accel_nav[segment_start : segment_end + 1],
                    dtype=np.float32,
                ),
            )
        duration_s = max(
            float(session.timestamps_s[sample.target_end] - session.timestamps_s[sample.target_start]),
            1e-6,
        )
        measured_velocity_xy = learned_displacements_xy[local_idx].astype(np.float64) / duration_s
        sigma_xy = None if measurement_sigma_xy is None else measurement_sigma_xy[local_idx]
        position_blend, velocity_blend = tracker.update(
            measured_position_xy=learned_positions_xy[local_idx],
            measured_velocity_xy=measured_velocity_xy,
            measurement_sigma_xy=sigma_xy,
        )
        positions_xy[local_idx] = tracker.position_xy.astype(np.float32)
        heights_m[local_idx] = float(tracker.height_m)
        position_blends.append(position_blend)
        velocity_blends.append(velocity_blend)
        height_offsets.append(abs(float(tracker.height_m - calibration.initial_height_m)))
        previous_end = target_end

    diagnostics = {
        "mean_position_blend": float(np.mean(position_blends)) if position_blends else 0.0,
        "mean_velocity_blend": float(np.mean(velocity_blends)) if velocity_blends else 0.0,
        "mean_abs_height_offset_m": float(np.mean(height_offsets)) if height_offsets else 0.0,
        "max_abs_height_offset_m": float(np.max(height_offsets)) if height_offsets else 0.0,
        "calibration": calibration.to_dict(),
    }
    return positions_xy, heights_m, diagnostics


def aggregate_plane_tracker_diagnostics(session_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    if not session_diagnostics:
        return {"num_sessions": 0}
    return {
        "num_sessions": float(len(session_diagnostics)),
        "mean_position_blend": float(
            np.mean([item["mean_position_blend"] for item in session_diagnostics])
        ),
        "mean_velocity_blend": float(
            np.mean([item["mean_velocity_blend"] for item in session_diagnostics])
        ),
        "mean_abs_height_offset_m": float(
            np.mean([item["mean_abs_height_offset_m"] for item in session_diagnostics])
        ),
        "max_abs_height_offset_m": float(
            np.max([item["max_abs_height_offset_m"] for item in session_diagnostics])
        ),
        "first_session_calibration": session_diagnostics[0]["calibration"],
    }


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
