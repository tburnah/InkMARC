from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlaneCalibration:
    calibration_row_index: int
    plane_normal_nav: np.ndarray
    plane_axes_nav: np.ndarray
    planar_origin_xy_m: np.ndarray
    initial_height_m: float
    initial_quaternion_xyzw: np.ndarray | None
    initial_gravity_nav: np.ndarray
    frame_assumption: str

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_row_index": int(self.calibration_row_index),
            "plane_normal_nav": self.plane_normal_nav.astype(np.float64).tolist(),
            "plane_axes_nav": self.plane_axes_nav.astype(np.float64).tolist(),
            "planar_origin_xy_m": self.planar_origin_xy_m.astype(np.float64).tolist(),
            "initial_height_m": float(self.initial_height_m),
            "initial_quaternion_xyzw": (
                None
                if self.initial_quaternion_xyzw is None
                else self.initial_quaternion_xyzw.astype(np.float64).tolist()
            ),
            "initial_gravity_nav": self.initial_gravity_nav.astype(np.float64).tolist(),
            "frame_assumption": self.frame_assumption,
        }


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return fallback.astype(np.float64)
    return (vector / norm).astype(np.float64)


def build_plane_calibration(
    session,
    reference_index: int | None = None,
    quaternion_frame_mode: str = "body_to_nav",
) -> PlaneCalibration:
    teacher_valid = np.asarray(session.teacher_valid, dtype=bool)
    if reference_index is None:
        valid_indices = np.flatnonzero(teacher_valid)
        if valid_indices.size == 0:
            raise ValueError(f"Session '{session.session_name}' has no valid teacher rows for calibration.")
        calibration_row_index = int(valid_indices[0])
    else:
        calibration_row_index = int(reference_index)

    initial_teacher_xyz = np.asarray(session.teacher_xyz[calibration_row_index], dtype=np.float64)
    if not np.all(np.isfinite(initial_teacher_xyz)):
        raise ValueError(
            f"Session '{session.session_name}' calibration row {calibration_row_index} has non-finite teacher coordinates."
        )

    initial_quaternion_xyzw = None
    if session.quaternion_xyzw is not None:
        initial_quaternion_xyzw = np.asarray(
            session.quaternion_xyzw[calibration_row_index],
            dtype=np.float64,
        )

    if session.gravity_xyz is not None and session.quaternion_xyzw is not None:
        gravity_body = np.asarray(session.gravity_xyz[calibration_row_index], dtype=np.float64)
        gravity_nav = _rotate_body_to_nav(
            gravity_body.reshape(1, 3).astype(np.float32),
            np.asarray(session.quaternion_xyzw[calibration_row_index], dtype=np.float32).reshape(1, 4),
            quaternion_frame_mode=quaternion_frame_mode,
        )[0].astype(np.float64)
    else:
        gravity_nav = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    plane_normal_nav = _normalize(-gravity_nav, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    return PlaneCalibration(
        calibration_row_index=calibration_row_index,
        plane_normal_nav=plane_normal_nav.astype(np.float64),
        plane_axes_nav=np.eye(3, dtype=np.float64),
        planar_origin_xy_m=initial_teacher_xyz[:2].astype(np.float64),
        initial_height_m=float(initial_teacher_xyz[2]),
        initial_quaternion_xyzw=None if initial_quaternion_xyzw is None else initial_quaternion_xyzw.astype(np.float64),
        initial_gravity_nav=_normalize(gravity_nav, fallback=np.asarray([0.0, 0.0, -1.0], dtype=np.float64)),
        frame_assumption=(
            "Teacher plane coordinates are treated as the calibrated plane frame, "
            "and the calibration-time gravity-aligned nav frame is used as the plane tracker frame."
        ),
    )


def _rotate_body_to_nav(
    body_vectors: np.ndarray,
    quaternions_xyzw: np.ndarray,
    quaternion_frame_mode: str,
) -> np.ndarray:
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    quaternions = quaternions / norms
    if quaternion_frame_mode == "nav_to_body":
        quaternions = quaternions.copy()
        quaternions[:, :3] *= -1.0
    elif quaternion_frame_mode != "body_to_nav":
        raise ValueError(f"Unsupported quaternion frame mode: {quaternion_frame_mode}")
    q_xyz = quaternions[:, :3]
    q_w = quaternions[:, 3:4]
    cross_1 = np.cross(q_xyz, body_vectors.astype(np.float64))
    cross_2 = np.cross(q_xyz, cross_1)
    rotated = body_vectors.astype(np.float64) + 2.0 * (q_w * cross_1 + cross_2)
    return rotated.astype(np.float32)
