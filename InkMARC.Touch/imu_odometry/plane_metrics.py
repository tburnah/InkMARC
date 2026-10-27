from __future__ import annotations

from typing import Any

import numpy as np


def compute_plane_surface_metrics(
    predicted_positions_xy: np.ndarray,
    true_positions_xy: np.ndarray,
    predicted_height_m: np.ndarray | None = None,
    true_height_m: np.ndarray | None = None,
    short_horizon_steps: int = 3,
    surface_half_width_m: float = 0.15,
    surface_half_height_m: float = 0.10,
) -> dict[str, float]:
    predicted_positions = np.asarray(predicted_positions_xy, dtype=np.float64)
    true_positions = np.asarray(true_positions_xy, dtype=np.float64)
    translation_errors = np.linalg.norm(predicted_positions - true_positions, axis=1)

    surface_corners = np.asarray(
        [
            [-surface_half_width_m, -surface_half_height_m],
            [-surface_half_width_m, surface_half_height_m],
            [surface_half_width_m, -surface_half_height_m],
            [surface_half_width_m, surface_half_height_m],
        ],
        dtype=np.float64,
    )
    predicted_corners = predicted_positions[:, None, :] + surface_corners[None, :, :]
    true_corners = true_positions[:, None, :] + surface_corners[None, :, :]
    corner_errors = np.linalg.norm(predicted_corners - true_corners, axis=2)

    horizon = max(int(short_horizon_steps), 1)
    if predicted_positions.shape[0] > horizon:
        predicted_slip = predicted_positions[horizon:] - predicted_positions[:-horizon]
        true_slip = true_positions[horizon:] - true_positions[:-horizon]
        slip_errors = np.linalg.norm(predicted_slip - true_slip, axis=1)
    else:
        slip_errors = np.zeros(0, dtype=np.float64)

    metrics = {
        "plane_relative_position_mae_m": float(np.mean(translation_errors)) if translation_errors.size else 0.0,
        "plane_relative_position_rmse_m": (
            float(np.sqrt(np.mean(np.square(translation_errors))))
            if translation_errors.size
            else 0.0
        ),
        "surface_anchor_drift_mae_m": float(np.mean(translation_errors)) if translation_errors.size else 0.0,
        "surface_anchor_drift_final_m": float(translation_errors[-1]) if translation_errors.size else 0.0,
        "surface_anchor_drift_max_m": float(np.max(translation_errors)) if translation_errors.size else 0.0,
        "surface_slip_mae_m": float(np.mean(slip_errors)) if slip_errors.size else 0.0,
        "surface_slip_rmse_m": (
            float(np.sqrt(np.mean(np.square(slip_errors)))) if slip_errors.size else 0.0
        ),
        "surface_slip_max_m": float(np.max(slip_errors)) if slip_errors.size else 0.0,
        "surface_slip_horizon_steps": float(horizon),
        "corner_drift_mae_m": float(np.mean(corner_errors)) if corner_errors.size else 0.0,
        "corner_drift_final_m": (
            float(np.mean(corner_errors[-1])) if corner_errors.size else 0.0
        ),
        "corner_drift_max_m": float(np.max(corner_errors)) if corner_errors.size else 0.0,
    }

    if predicted_height_m is not None and true_height_m is not None:
        predicted_height = np.asarray(predicted_height_m, dtype=np.float64)
        true_height = np.asarray(true_height_m, dtype=np.float64)
        height_errors = np.abs(predicted_height - true_height)
        metrics.update(
            {
                "height_mae_m": float(np.mean(height_errors)) if height_errors.size else 0.0,
                "height_rmse_m": (
                    float(np.sqrt(np.mean(np.square(height_errors)))) if height_errors.size else 0.0
                ),
                "height_max_error_m": float(np.max(height_errors)) if height_errors.size else 0.0,
            }
        )
    return metrics


def aggregate_plane_surface_metrics(per_session_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not per_session_metrics:
        return {"num_sessions": 0}
    aggregated: dict[str, float] = {"num_sessions": float(len(per_session_metrics))}
    metric_names = sorted(per_session_metrics[0].keys())
    for metric_name in metric_names:
        values = [float(item[metric_name]) for item in per_session_metrics if metric_name in item]
        if not values:
            continue
        if metric_name.endswith("_max_m"):
            aggregated[metric_name] = float(np.max(values))
        else:
            aggregated[metric_name] = float(np.mean(values))
    return aggregated
