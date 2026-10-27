from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from config import (
    EvaluationConfig,
    TrainingConfig,
    build_eval_argparser,
    eval_config_from_args,
)
from dataset import SensorWindowDataset, collate_sensor_windows, compute_classical_delta_target
from ekf import PlanarEkf
from losses import (
    compute_direction_magnitude_metrics,
    compute_loss_values,
    compute_motion_loss_values,
    decode_predictions,
    summarize_loss,
)
from model import build_model
from plane_metrics import aggregate_plane_surface_metrics, compute_plane_surface_metrics
from plane_tracker import (
    PlaneRelativeTrackerConfig,
    aggregate_plane_tracker_diagnostics,
    rollout_plane_relative_tracker,
)
from utils import (
    denormalize_targets,
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_json,
    write_predictions_csv,
)


def expand_metadata_batch(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if not metadata:
        return []
    batch_size = len(metadata["session_name"])
    rows: list[dict[str, Any]] = []
    skip_keys = {"true_target", "classical_target"}
    for idx in range(batch_size):
        row: dict[str, Any] = {}
        for key, values in metadata.items():
            if key in skip_keys:
                continue
            if torch.is_tensor(values):
                value = values[idx]
                row[key] = value.item() if value.ndim == 0 else value.detach().cpu().tolist()
            elif isinstance(values, np.ndarray):
                row[key] = values[idx].item() if values.ndim == 1 else values[idx]
            else:
                row[key] = values[idx]
        rows.append(row)
    return rows


def unpack_model_outputs(
    model_outputs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(model_outputs, dict):
        return (
            model_outputs["regression"],
            model_outputs.get("motion_logits"),
            model_outputs.get("log_variance"),
            model_outputs.get("bias_correction"),
        )
    return model_outputs, None, None, None


def denormalize_delta_scale(
    values: np.ndarray,
    normalization,
) -> np.ndarray:
    if normalization is None or normalization.target_std is None:
        return values.astype(np.float32)
    return (values * normalization.target_std_array.reshape(1, -1)).astype(np.float32)


def sigma_from_log_variance(
    log_variance: np.ndarray | None,
    normalization,
    log_variance_min: float = -20.0,
    log_variance_max: float = 20.0,
) -> np.ndarray | None:
    if log_variance is None:
        return None
    bounded = np.clip(
        log_variance.astype(np.float64),
        a_min=log_variance_min,
        a_max=log_variance_max,
    )
    variance = np.exp(bounded)
    if normalization is not None and normalization.target_std is not None:
        target_scale = normalization.target_std_array.reshape(1, -1).astype(np.float64)
        variance = variance * np.square(target_scale)
    return np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calibration_bucket_table(
    predicted_std: np.ndarray,
    actual_abs_residual: np.ndarray,
    num_buckets: int,
) -> list[dict[str, float]]:
    if predicted_std.size == 0:
        return []
    bucket_count = max(int(num_buckets), 1)
    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
    bucket_edges = np.quantile(predicted_std.astype(np.float64), quantiles)
    rows: list[dict[str, float]] = []
    for bucket_index in range(bucket_count):
        lower = float(bucket_edges[bucket_index])
        upper = float(bucket_edges[bucket_index + 1])
        if bucket_index == bucket_count - 1:
            mask = (predicted_std >= lower) & (predicted_std <= upper)
        else:
            mask = (predicted_std >= lower) & (predicted_std < upper)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bucket_index": bucket_index,
                "predicted_std_lower_m": lower,
                "predicted_std_upper_m": upper,
                "num_samples": int(np.sum(mask)),
                "mean_predicted_std_m": float(np.mean(predicted_std[mask])),
                "mean_abs_residual_m": float(np.mean(actual_abs_residual[mask])),
                "median_abs_residual_m": float(np.median(actual_abs_residual[mask])),
                "max_abs_residual_m": float(np.max(actual_abs_residual[mask])),
            }
        )
    return rows


def magnitude_from_delta_xy(values_xy: np.ndarray) -> np.ndarray:
    if values_xy.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.linalg.norm(values_xy.astype(np.float64), axis=1).astype(np.float32)


def bucket_mask(
    magnitudes: np.ndarray,
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    mask = np.ones_like(magnitudes, dtype=bool)
    if lower is not None:
        mask &= magnitudes >= lower
    if upper is not None:
        mask &= magnitudes < upper
    return mask


def residual_covariance_diag(residual_xy: np.ndarray) -> np.ndarray:
    if residual_xy.size == 0:
        return np.full(2, 1e-6, dtype=np.float64)
    return np.mean(np.square(residual_xy.astype(np.float64)), axis=0)


def estimate_empirical_measurement_covariance(
    predictions_xy: np.ndarray,
    targets_xy: np.ndarray,
    bucket_edges: list[float],
    calibration_split_name: str,
) -> dict[str, Any]:
    residual_xy = predictions_xy.astype(np.float64) - targets_xy.astype(np.float64)
    predicted_magnitude = magnitude_from_delta_xy(predictions_xy).astype(np.float64)
    fixed_covariance_xy = residual_covariance_diag(residual_xy)
    bucket_rows: list[dict[str, Any]] = []
    for label, lower, upper in motion_bucket_labels(bucket_edges):
        mask = bucket_mask(predicted_magnitude, lower=lower, upper=upper)
        count = int(np.sum(mask))
        used_global_fallback = count == 0
        bucket_covariance_xy = (
            fixed_covariance_xy if used_global_fallback else residual_covariance_diag(residual_xy[mask])
        )
        bucket_rows.append(
            {
                "label": label,
                "lower_bound_m": None if lower is None else float(lower),
                "upper_bound_m": None if upper is None else float(upper),
                "num_samples": count,
                "used_global_fallback": used_global_fallback,
                "covariance_x_m2": float(bucket_covariance_xy[0]),
                "covariance_y_m2": float(bucket_covariance_xy[1]),
                "std_x_m": float(np.sqrt(max(bucket_covariance_xy[0], 1e-8))),
                "std_y_m": float(np.sqrt(max(bucket_covariance_xy[1], 1e-8))),
                "mean_predicted_displacement_m": (
                    float(np.mean(predicted_magnitude[mask])) if count > 0 else float("nan")
                ),
            }
        )
    return {
        "calibration_split": calibration_split_name,
        "bucket_key": "predicted_displacement_magnitude_m",
        "bucket_edges_m": [float(edge) for edge in bucket_edges],
        "num_samples": int(predictions_xy.shape[0]),
        "fixed_empirical": {
            "covariance_x_m2": float(fixed_covariance_xy[0]),
            "covariance_y_m2": float(fixed_covariance_xy[1]),
            "std_x_m": float(np.sqrt(max(fixed_covariance_xy[0], 1e-8))),
            "std_y_m": float(np.sqrt(max(fixed_covariance_xy[1], 1e-8))),
        },
        "bucketed_empirical": {
            "global_fallback_covariance_x_m2": float(fixed_covariance_xy[0]),
            "global_fallback_covariance_y_m2": float(fixed_covariance_xy[1]),
            "buckets": bucket_rows,
        },
    }


def empirical_sigma_from_covariance(
    predictions_xy: np.ndarray,
    empirical_measurement_covariance: dict[str, Any],
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    num_samples = int(predictions_xy.shape[0])
    if num_samples == 0:
        return np.zeros((0, 2), dtype=np.float32), {"mode": mode}
    if mode == "fixed_empirical":
        fixed = empirical_measurement_covariance["fixed_empirical"]
        covariance_xy = np.asarray(
            [fixed["covariance_x_m2"], fixed["covariance_y_m2"]],
            dtype=np.float64,
        )
        sigma_xy = np.sqrt(np.maximum(covariance_xy, 1e-8)).reshape(1, 2)
        return np.repeat(sigma_xy.astype(np.float32), repeats=num_samples, axis=0), {
            "mode": mode,
            "calibration_split": empirical_measurement_covariance["calibration_split"],
            "covariance_x_m2": float(covariance_xy[0]),
            "covariance_y_m2": float(covariance_xy[1]),
            "std_x_m": float(np.sqrt(max(covariance_xy[0], 1e-8))),
            "std_y_m": float(np.sqrt(max(covariance_xy[1], 1e-8))),
            "num_samples": num_samples,
        }
    if mode != "bucketed_empirical":
        raise ValueError(f"Unsupported empirical covariance mode: {mode}")
    predicted_magnitude = magnitude_from_delta_xy(predictions_xy).astype(np.float64)
    covariance_xy = np.zeros((num_samples, 2), dtype=np.float64)
    assignment_rows: list[dict[str, Any]] = []
    for bucket in empirical_measurement_covariance["bucketed_empirical"]["buckets"]:
        mask = bucket_mask(
            predicted_magnitude,
            lower=bucket["lower_bound_m"],
            upper=bucket["upper_bound_m"],
        )
        covariance_xy[mask, 0] = float(bucket["covariance_x_m2"])
        covariance_xy[mask, 1] = float(bucket["covariance_y_m2"])
        assignment_rows.append(
            {
                "label": bucket["label"],
                "lower_bound_m": bucket["lower_bound_m"],
                "upper_bound_m": bucket["upper_bound_m"],
                "calibration_num_samples": int(bucket["num_samples"]),
                "evaluation_num_samples": int(np.sum(mask)),
                "used_global_fallback": bool(bucket["used_global_fallback"]),
                "covariance_x_m2": float(bucket["covariance_x_m2"]),
                "covariance_y_m2": float(bucket["covariance_y_m2"]),
                "std_x_m": float(bucket["std_x_m"]),
                "std_y_m": float(bucket["std_y_m"]),
            }
        )
    sigma_xy = np.sqrt(np.maximum(covariance_xy, 1e-8)).astype(np.float32)
    return sigma_xy, {
        "mode": mode,
        "calibration_split": empirical_measurement_covariance["calibration_split"],
        "bucket_key": empirical_measurement_covariance["bucket_key"],
        "bucket_edges_m": empirical_measurement_covariance["bucket_edges_m"],
        "bucket_assignments": assignment_rows,
    }


def compute_uncertainty_diagnostics(
    predictions_xy: np.ndarray,
    targets_xy: np.ndarray,
    sigma_xy: np.ndarray | None,
    log_variance_xy: np.ndarray | None,
    log_variance_min: float,
    log_variance_max: float,
    num_buckets: int,
) -> dict[str, Any] | None:
    if sigma_xy is None:
        return None
    residual_xy = predictions_xy - targets_xy
    abs_residual_xy = np.abs(residual_xy)
    variance_xy = np.maximum(np.square(sigma_xy.astype(np.float64)), 1e-8)
    gaussian_nll_xy = 0.5 * (
        np.square(residual_xy.astype(np.float64)) / variance_xy
        + np.log(variance_xy)
        + np.log(2.0 * np.pi)
    )
    diagnostics: dict[str, Any] = {
        "mean_sigma_x_m": float(np.mean(sigma_xy[:, 0])),
        "mean_sigma_y_m": float(np.mean(sigma_xy[:, 1])),
        "mean_residual_x_m": float(np.mean(residual_xy[:, 0])),
        "mean_residual_y_m": float(np.mean(residual_xy[:, 1])),
        "mean_abs_residual_x_m": float(np.mean(abs_residual_xy[:, 0])),
        "mean_abs_residual_y_m": float(np.mean(abs_residual_xy[:, 1])),
        "std_abs_residual_correlation_x": safe_correlation(
            sigma_xy[:, 0].astype(np.float64),
            abs_residual_xy[:, 0].astype(np.float64),
        ),
        "std_abs_residual_correlation_y": safe_correlation(
            sigma_xy[:, 1].astype(np.float64),
            abs_residual_xy[:, 1].astype(np.float64),
        ),
        "nll_mean": float(np.mean(gaussian_nll_xy)),
        "nll_x_mean": float(np.mean(gaussian_nll_xy[:, 0])),
        "nll_y_mean": float(np.mean(gaussian_nll_xy[:, 1])),
        "calibration_by_bucket": {
            "x": calibration_bucket_table(
                sigma_xy[:, 0].astype(np.float64),
                abs_residual_xy[:, 0].astype(np.float64),
                num_buckets=num_buckets,
            ),
            "y": calibration_bucket_table(
                sigma_xy[:, 1].astype(np.float64),
                abs_residual_xy[:, 1].astype(np.float64),
                num_buckets=num_buckets,
            ),
        },
    }
    if log_variance_xy is not None:
        diagnostics["log_variance_clamp_fraction"] = {
            "below_min_x": float(np.mean(log_variance_xy[:, 0] <= log_variance_min)),
            "below_min_y": float(np.mean(log_variance_xy[:, 1] <= log_variance_min)),
            "above_max_x": float(np.mean(log_variance_xy[:, 0] >= log_variance_max)),
            "above_max_y": float(np.mean(log_variance_xy[:, 1] >= log_variance_max)),
        }
    return diagnostics


def aggregate_ekf_diagnostics(session_diagnostics: list[dict[str, float]]) -> dict[str, float]:
    if not session_diagnostics:
        return {}
    total_measurements = float(sum(item["num_measurements"] for item in session_diagnostics))
    total_updates = float(sum(item["num_measurement_updates"] for item in session_diagnostics))
    return {
        "num_sessions": float(len(session_diagnostics)),
        "num_measurements": total_measurements,
        "num_measurement_updates": total_updates,
        "measurement_update_fraction": (
            float(total_updates / total_measurements) if total_measurements > 0.0 else 0.0
        ),
        "mean_measurement_covariance_x": float(
            np.mean([item["mean_measurement_covariance_x"] for item in session_diagnostics])
        ),
        "mean_measurement_covariance_y": float(
            np.mean([item["mean_measurement_covariance_y"] for item in session_diagnostics])
        ),
        "covariance_floor_hit_fraction_x": float(
            np.mean([item["covariance_floor_hit_fraction_x"] for item in session_diagnostics])
        ),
        "covariance_floor_hit_fraction_y": float(
            np.mean([item["covariance_floor_hit_fraction_y"] for item in session_diagnostics])
        ),
        "covariance_ceiling_hit_fraction_x": float(
            np.mean([item["covariance_ceiling_hit_fraction_x"] for item in session_diagnostics])
        ),
        "covariance_ceiling_hit_fraction_y": float(
            np.mean([item["covariance_ceiling_hit_fraction_y"] for item in session_diagnostics])
        ),
    }


def summarize_delta_scale(
    values_xy: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    return {
        f"mean_abs_{prefix}_x_m": float(np.mean(np.abs(values_xy[:, 0]))),
        f"mean_abs_{prefix}_y_m": float(np.mean(np.abs(values_xy[:, 1]))),
        f"max_abs_{prefix}_x_m": float(np.max(np.abs(values_xy[:, 0]))),
        f"max_abs_{prefix}_y_m": float(np.max(np.abs(values_xy[:, 1]))),
    }


def compute_rollout_position_metrics(
    predicted_positions_xy: np.ndarray,
    true_positions_xy: np.ndarray,
) -> dict[str, float]:
    errors = np.linalg.norm(predicted_positions_xy - true_positions_xy, axis=1)
    final_error = float(errors[-1]) if errors.size else 0.0
    true_final_displacement = float(np.linalg.norm(true_positions_xy[-1])) if errors.size else 0.0
    return {
        "num_steps": int(errors.size),
        "position_mae_m": float(np.mean(errors)) if errors.size else 0.0,
        "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors.size else 0.0,
        "max_position_error_m": float(np.max(errors)) if errors.size else 0.0,
        "final_position_error_m": final_error,
        "true_final_displacement_m": true_final_displacement,
        "final_relative_error_percentage": (
            float(final_error / true_final_displacement * 100.0)
            if true_final_displacement > 1e-12
            else float("nan")
        ),
    }


def aggregate_rollout_metrics(per_session_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not per_session_metrics:
        return {"num_sessions": 0, "num_steps": 0}
    position_maes = [float(item["position_mae_m"]) for item in per_session_metrics]
    position_rmses = [float(item["position_rmse_m"]) for item in per_session_metrics]
    max_errors = [float(item["max_position_error_m"]) for item in per_session_metrics]
    final_errors = [float(item["final_position_error_m"]) for item in per_session_metrics]
    final_displacements = [float(item["true_final_displacement_m"]) for item in per_session_metrics]
    total_steps = sum(int(item["num_steps"]) for item in per_session_metrics)
    mean_final_error = float(np.mean(final_errors))
    mean_final_displacement = float(np.mean(final_displacements))
    return {
        "num_sessions": len(per_session_metrics),
        "num_steps": total_steps,
        "position_mae_m": float(np.mean(position_maes)),
        "position_rmse_m": float(np.mean(position_rmses)),
        "max_position_error_m": float(np.max(max_errors)),
        "final_position_error_m_mean": mean_final_error,
        "final_position_error_m_max": float(np.max(final_errors)),
        "mean_true_final_displacement_m": mean_final_displacement,
        "final_relative_error_percentage": (
            float(mean_final_error / mean_final_displacement * 100.0)
            if mean_final_displacement > 1e-12
            else float("nan")
        ),
    }


def build_plane_tracker_config(
    position_blend: float,
    velocity_blend: float,
    uncertainty_scale: float,
    velocity_damping_per_s: float,
    height_constraint_gain: float,
    normal_accel_gain: float,
    max_height_offset_m: float,
) -> PlaneRelativeTrackerConfig:
    return PlaneRelativeTrackerConfig(
        position_blend=float(position_blend),
        velocity_blend=float(velocity_blend),
        uncertainty_scale=float(uncertainty_scale),
        velocity_damping_per_s=float(velocity_damping_per_s),
        height_constraint_gain=float(height_constraint_gain),
        normal_accel_gain=float(normal_accel_gain),
        max_height_offset_m=float(max_height_offset_m),
    )


def rollout_velocity_integration(
    displacement_measurements_xy: np.ndarray,
    measurement_end_times_s: np.ndarray,
    measurement_durations_s: np.ndarray,
) -> np.ndarray:
    positions = np.zeros_like(displacement_measurements_xy, dtype=np.float32)
    if displacement_measurements_xy.shape[0] == 0:
        return positions
    safe_durations = np.clip(measurement_durations_s.astype(np.float64), a_min=1e-6, a_max=None)
    positions[0] = displacement_measurements_xy[0]
    for idx in range(1, displacement_measurements_xy.shape[0]):
        velocity_xy = displacement_measurements_xy[idx].astype(np.float64) / safe_durations[idx]
        dt_step = max(float(measurement_end_times_s[idx] - measurement_end_times_s[idx - 1]), 1e-6)
        positions[idx] = positions[idx - 1] + (velocity_xy * dt_step).astype(np.float32)
    return positions


def rollout_planar_ekf(
    displacement_measurements_xy: np.ndarray,
    measurement_end_times_s: np.ndarray,
    measurement_durations_s: np.ndarray,
    measurement_sigma_xy: np.ndarray | None,
    process_accel_std_mps2: float,
    process_noise_scale: float,
    measurement_covariance_scale: float,
    measurement_covariance_floor: float | None,
    measurement_covariance_ceiling: float | None,
    measurement_std_floor_mps: float,
    measurement_std_max_mps: float,
    initial_position_std_m: float,
    initial_velocity_std_mps: float,
    update_stride: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    positions = np.zeros_like(displacement_measurements_xy, dtype=np.float32)
    if displacement_measurements_xy.shape[0] == 0:
        return positions, {
            "num_measurements": 0.0,
            "num_measurement_updates": 0.0,
            "mean_measurement_covariance_x": 0.0,
            "mean_measurement_covariance_y": 0.0,
            "covariance_floor_hit_fraction_x": 0.0,
            "covariance_floor_hit_fraction_y": 0.0,
            "covariance_ceiling_hit_fraction_x": 0.0,
            "covariance_ceiling_hit_fraction_y": 0.0,
        }
    safe_durations = np.clip(measurement_durations_s.astype(np.float64), a_min=1e-6, a_max=None)
    effective_process_accel_std_mps2 = process_accel_std_mps2 * np.sqrt(max(process_noise_scale, 1e-8))
    ekf = PlanarEkf(process_accel_std_mps2=effective_process_accel_std_mps2)
    covariance_floor = (
        float(measurement_covariance_floor)
        if measurement_covariance_floor is not None
        else float(measurement_std_floor_mps * measurement_std_floor_mps)
    )
    covariance_ceiling = (
        float(measurement_covariance_ceiling)
        if measurement_covariance_ceiling is not None
        else float(measurement_std_max_mps * measurement_std_max_mps)
    )
    covariance_floor = max(covariance_floor, 1e-8)
    covariance_ceiling = max(covariance_ceiling, covariance_floor)
    effective_update_stride = max(int(update_stride), 1)
    initial_velocity_xy = displacement_measurements_xy[0].astype(np.float64) / safe_durations[0]
    initial_position_std = initial_position_std_m
    if measurement_sigma_xy is not None:
        initial_position_std = max(initial_position_std, float(np.mean(measurement_sigma_xy[0])))
    ekf.initialize(
        position_xy=displacement_measurements_xy[0].astype(np.float64),
        velocity_xy=initial_velocity_xy,
        position_std_m=initial_position_std,
        velocity_std_mps=initial_velocity_std_mps,
    )
    positions[0] = ekf.state[:2].astype(np.float32)
    raw_covariances_x: list[float] = []
    raw_covariances_y: list[float] = []
    floor_hits_x = 0
    floor_hits_y = 0
    ceiling_hits_x = 0
    ceiling_hits_y = 0
    update_count = 0
    for idx in range(1, displacement_measurements_xy.shape[0]):
        dt_step = max(float(measurement_end_times_s[idx] - measurement_end_times_s[idx - 1]), 1e-6)
        ekf.predict(dt_step)
        if idx % effective_update_stride != 0:
            positions[idx] = ekf.state[:2].astype(np.float32)
            continue
        measured_velocity_xy = displacement_measurements_xy[idx].astype(np.float64) / safe_durations[idx]
        if measurement_sigma_xy is None:
            raw_measurement_covariance_xy = np.full(
                2,
                measurement_std_floor_mps * measurement_std_floor_mps,
                dtype=np.float64,
            )
        else:
            raw_measurement_covariance_xy = np.square(
                measurement_sigma_xy[idx].astype(np.float64) / safe_durations[idx]
            )
        raw_measurement_covariance_xy = raw_measurement_covariance_xy * max(
            float(measurement_covariance_scale),
            1e-8,
        )
        raw_covariances_x.append(float(raw_measurement_covariance_xy[0]))
        raw_covariances_y.append(float(raw_measurement_covariance_xy[1]))
        floor_hits_x += int(raw_measurement_covariance_xy[0] < covariance_floor)
        floor_hits_y += int(raw_measurement_covariance_xy[1] < covariance_floor)
        ceiling_hits_x += int(raw_measurement_covariance_xy[0] > covariance_ceiling)
        ceiling_hits_y += int(raw_measurement_covariance_xy[1] > covariance_ceiling)
        measurement_covariance_xy = np.clip(
            raw_measurement_covariance_xy,
            a_min=covariance_floor,
            a_max=covariance_ceiling,
        )
        measurement_velocity_std_xy = np.sqrt(measurement_covariance_xy)
        ekf.update_velocity(measured_velocity_xy, measurement_velocity_std_xy)
        update_count += 1
        positions[idx] = ekf.state[:2].astype(np.float32)
    num_measurements = max(displacement_measurements_xy.shape[0] - 1, 0)
    return positions, {
        "num_measurements": float(num_measurements),
        "num_measurement_updates": float(update_count),
        "mean_measurement_covariance_x": float(np.mean(raw_covariances_x)) if raw_covariances_x else 0.0,
        "mean_measurement_covariance_y": float(np.mean(raw_covariances_y)) if raw_covariances_y else 0.0,
        "covariance_floor_hit_fraction_x": (
            float(floor_hits_x / max(len(raw_covariances_x), 1))
            if raw_covariances_x
            else 0.0
        ),
        "covariance_floor_hit_fraction_y": (
            float(floor_hits_y / max(len(raw_covariances_y), 1))
            if raw_covariances_y
            else 0.0
        ),
        "covariance_ceiling_hit_fraction_x": (
            float(ceiling_hits_x / max(len(raw_covariances_x), 1))
            if raw_covariances_x
            else 0.0
        ),
        "covariance_ceiling_hit_fraction_y": (
            float(ceiling_hits_y / max(len(raw_covariances_y), 1))
            if raw_covariances_y
            else 0.0
        ),
    }


def evaluate_sequential_rollouts(
    dataset: SensorWindowDataset,
    learned_displacements_xy: np.ndarray,
    classical_displacements_xy: np.ndarray,
    learned_sigma_xy: np.ndarray | None = None,
    ekf_measurement_covariance_mode: str = "learned_uncertainty",
    empirical_measurement_covariance: dict[str, Any] | None = None,
    planar_ekf_rollout_enabled: bool = False,
    planar_ekf_process_accel_std_mps2: float = 0.75,
    planar_ekf_process_noise_scale: float = 1.0,
    planar_ekf_measurement_covariance_scale: float = 1.0,
    planar_ekf_measurement_covariance_floor: float | None = None,
    planar_ekf_measurement_covariance_ceiling: float | None = None,
    planar_ekf_velocity_measurement_std_floor_mps: float = 0.05,
    planar_ekf_velocity_measurement_std_max_mps: float = 5.0,
    planar_ekf_initial_position_std_m: float = 0.10,
    planar_ekf_initial_velocity_std_mps: float = 0.25,
    planar_ekf_update_stride: int = 1,
    plane_relative_tracker_rollout_enabled: bool = False,
    plane_tracker_position_blend: float = 0.70,
    plane_tracker_velocity_blend: float = 0.35,
    plane_tracker_uncertainty_scale: float = 0.10,
    plane_tracker_velocity_damping_per_s: float = 0.35,
    plane_tracker_height_constraint_gain: float = 8.0,
    plane_tracker_normal_accel_gain: float = 0.05,
    plane_tracker_max_height_offset_m: float = 0.05,
    plane_tracker_short_horizon_steps: int = 3,
    plane_tracker_surface_half_width_m: float = 0.15,
    plane_tracker_surface_half_height_m: float = 0.10,
) -> dict[str, Any]:
    if dataset.target_mode != "delta_xy":
        return {}
    session_indices: dict[int, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        session_indices.setdefault(sample.session_idx, []).append(index)

    direct_metrics: list[dict[str, float]] = []
    direct_surface_metrics: list[dict[str, float]] = []
    classical_metrics: list[dict[str, float]] = []
    classical_surface_metrics: list[dict[str, float]] = []
    ekf_metrics: list[dict[str, float]] = []
    ekf_surface_metrics: list[dict[str, float]] = []
    ekf_diagnostics: list[dict[str, float]] = []
    plane_tracker_metrics: list[dict[str, float]] = []
    plane_tracker_surface_metrics: list[dict[str, float]] = []
    plane_tracker_diagnostics: list[dict[str, Any]] = []
    effective_sigma_all = learned_sigma_xy
    effective_measurement_covariance_floor = planar_ekf_measurement_covariance_floor
    empirical_mode_diagnostics: dict[str, Any] | None = None
    tracker_config = build_plane_tracker_config(
        position_blend=plane_tracker_position_blend,
        velocity_blend=plane_tracker_velocity_blend,
        uncertainty_scale=plane_tracker_uncertainty_scale,
        velocity_damping_per_s=plane_tracker_velocity_damping_per_s,
        height_constraint_gain=plane_tracker_height_constraint_gain,
        normal_accel_gain=plane_tracker_normal_accel_gain,
        max_height_offset_m=plane_tracker_max_height_offset_m,
    )
    if planar_ekf_rollout_enabled and ekf_measurement_covariance_mode != "learned_uncertainty":
        if empirical_measurement_covariance is None:
            raise ValueError("Empirical EKF covariance mode requires calibration statistics.")
        effective_sigma_all, empirical_mode_diagnostics = empirical_sigma_from_covariance(
            predictions_xy=learned_displacements_xy,
            empirical_measurement_covariance=empirical_measurement_covariance,
            mode=ekf_measurement_covariance_mode,
        )
        if effective_measurement_covariance_floor is None:
            effective_measurement_covariance_floor = 1e-8
    for session_idx, indices in session_indices.items():
        session = dataset.sessions[session_idx]
        anchor_xy = session.teacher_xyz[dataset.samples[indices[0]].target_start, :2].astype(np.float32)
        true_positions_xy = np.stack(
            [
                session.teacher_xyz[dataset.samples[index].target_end, :2].astype(np.float32) - anchor_xy
                for index in indices
            ],
            axis=0,
        ).astype(np.float32)
        true_heights_m = np.asarray(
            [float(session.teacher_xyz[dataset.samples[index].target_end, 2]) for index in indices],
            dtype=np.float32,
        )
        measurement_end_times_s = np.asarray(
            [float(session.timestamps_s[dataset.samples[index].target_end]) for index in indices],
            dtype=np.float64,
        )
        measurement_durations_s = np.asarray(
            [
                max(
                    float(
                        session.timestamps_s[dataset.samples[index].target_end]
                        - session.timestamps_s[dataset.samples[index].target_start]
                    ),
                    0.0,
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        session_learned_displacements_xy = learned_displacements_xy[indices].astype(np.float32)
        session_classical_displacements_xy = classical_displacements_xy[indices].astype(np.float32)
        session_sigma_xy = None if effective_sigma_all is None else effective_sigma_all[indices].astype(np.float32)

        direct_positions_xy = rollout_velocity_integration(
            session_learned_displacements_xy,
            measurement_end_times_s,
            measurement_durations_s,
        )
        classical_positions_xy = rollout_velocity_integration(
            session_classical_displacements_xy,
            measurement_end_times_s,
            measurement_durations_s,
        )
        direct_metrics.append(compute_rollout_position_metrics(direct_positions_xy, true_positions_xy))
        direct_surface_metrics.append(
            compute_plane_surface_metrics(
                predicted_positions_xy=direct_positions_xy,
                true_positions_xy=true_positions_xy,
                short_horizon_steps=plane_tracker_short_horizon_steps,
                surface_half_width_m=plane_tracker_surface_half_width_m,
                surface_half_height_m=plane_tracker_surface_half_height_m,
            )
        )
        classical_metrics.append(compute_rollout_position_metrics(classical_positions_xy, true_positions_xy))
        classical_surface_metrics.append(
            compute_plane_surface_metrics(
                predicted_positions_xy=classical_positions_xy,
                true_positions_xy=true_positions_xy,
                short_horizon_steps=plane_tracker_short_horizon_steps,
                surface_half_width_m=plane_tracker_surface_half_width_m,
                surface_half_height_m=plane_tracker_surface_half_height_m,
            )
        )

        if planar_ekf_rollout_enabled:
            ekf_positions_xy, ekf_session_diagnostics = rollout_planar_ekf(
                displacement_measurements_xy=session_learned_displacements_xy,
                measurement_end_times_s=measurement_end_times_s,
                measurement_durations_s=measurement_durations_s,
                measurement_sigma_xy=session_sigma_xy,
                process_accel_std_mps2=planar_ekf_process_accel_std_mps2,
                process_noise_scale=planar_ekf_process_noise_scale,
                measurement_covariance_scale=planar_ekf_measurement_covariance_scale,
                measurement_covariance_floor=effective_measurement_covariance_floor,
                measurement_covariance_ceiling=planar_ekf_measurement_covariance_ceiling,
                measurement_std_floor_mps=planar_ekf_velocity_measurement_std_floor_mps,
                measurement_std_max_mps=planar_ekf_velocity_measurement_std_max_mps,
                initial_position_std_m=planar_ekf_initial_position_std_m,
                initial_velocity_std_mps=planar_ekf_initial_velocity_std_mps,
                update_stride=planar_ekf_update_stride,
            )
            ekf_metrics.append(compute_rollout_position_metrics(ekf_positions_xy, true_positions_xy))
            ekf_surface_metrics.append(
                compute_plane_surface_metrics(
                    predicted_positions_xy=ekf_positions_xy,
                    true_positions_xy=true_positions_xy,
                    short_horizon_steps=plane_tracker_short_horizon_steps,
                    surface_half_width_m=plane_tracker_surface_half_width_m,
                    surface_half_height_m=plane_tracker_surface_half_height_m,
                )
            )
            ekf_diagnostics.append(ekf_session_diagnostics)
        if plane_relative_tracker_rollout_enabled:
            plane_tracker_positions_xy, plane_tracker_heights_m, plane_tracker_session_diagnostics = (
                rollout_plane_relative_tracker(
                    session=session,
                    session_sample_indices=indices,
                    learned_positions_xy=direct_positions_xy,
                    learned_displacements_xy=session_learned_displacements_xy,
                    measurement_sigma_xy=session_sigma_xy,
                    dataset=dataset,
                    quaternion_frame_mode=dataset.quaternion_frame_mode,
                    tracker_config=tracker_config,
                )
            )
            plane_tracker_metrics.append(
                compute_rollout_position_metrics(plane_tracker_positions_xy, true_positions_xy)
            )
            plane_tracker_surface_metrics.append(
                compute_plane_surface_metrics(
                    predicted_positions_xy=plane_tracker_positions_xy,
                    true_positions_xy=true_positions_xy,
                    predicted_height_m=plane_tracker_heights_m,
                    true_height_m=true_heights_m,
                    short_horizon_steps=plane_tracker_short_horizon_steps,
                    surface_half_width_m=plane_tracker_surface_half_width_m,
                    surface_half_height_m=plane_tracker_surface_half_height_m,
                )
            )
            plane_tracker_diagnostics.append(plane_tracker_session_diagnostics)

    rollout_metrics: dict[str, Any] = {
        "direct_model": aggregate_rollout_metrics(direct_metrics),
        "classical_baseline": aggregate_rollout_metrics(classical_metrics),
    }
    rollout_metrics["direct_model"]["surface_stability"] = aggregate_plane_surface_metrics(
        direct_surface_metrics
    )
    rollout_metrics["classical_baseline"]["surface_stability"] = aggregate_plane_surface_metrics(
        classical_surface_metrics
    )
    if planar_ekf_rollout_enabled:
        rollout_metrics["planar_ekf"] = aggregate_rollout_metrics(ekf_metrics)
        rollout_metrics["planar_ekf"]["surface_stability"] = aggregate_plane_surface_metrics(
            ekf_surface_metrics
        )
        rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"] = aggregate_ekf_diagnostics(
            ekf_diagnostics
        )
        rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"][
            "mode"
        ] = ekf_measurement_covariance_mode
        rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"]["process_noise_scale"] = float(
            planar_ekf_process_noise_scale
        )
        rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"][
            "measurement_covariance_scale"
        ] = float(planar_ekf_measurement_covariance_scale)
        rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"][
            "measurement_covariance_floor"
        ] = float(
            effective_measurement_covariance_floor
            if effective_measurement_covariance_floor is not None
            else planar_ekf_velocity_measurement_std_floor_mps
            * planar_ekf_velocity_measurement_std_floor_mps
        )
        rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"][
            "measurement_covariance_ceiling"
        ] = float(
            planar_ekf_measurement_covariance_ceiling
            if planar_ekf_measurement_covariance_ceiling is not None
            else planar_ekf_velocity_measurement_std_max_mps
            * planar_ekf_velocity_measurement_std_max_mps
        )
        rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"]["update_stride"] = float(
            planar_ekf_update_stride
        )
        if empirical_measurement_covariance is not None:
            rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"][
                "empirical_calibration"
            ] = empirical_measurement_covariance
        if empirical_mode_diagnostics is not None:
            rollout_metrics["planar_ekf"]["measurement_covariance_diagnostics"][
                "empirical_assignment"
            ] = empirical_mode_diagnostics
    if plane_relative_tracker_rollout_enabled:
        rollout_metrics["plane_relative_tracker"] = aggregate_rollout_metrics(plane_tracker_metrics)
        rollout_metrics["plane_relative_tracker"]["surface_stability"] = (
            aggregate_plane_surface_metrics(plane_tracker_surface_metrics)
        )
        rollout_metrics["plane_relative_tracker"]["tracker_diagnostics"] = (
            aggregate_plane_tracker_diagnostics(plane_tracker_diagnostics)
        )
    return rollout_metrics


def apply_motion_gate(
    predictions: np.ndarray,
    motion_probabilities: np.ndarray | None,
    target_names: list[str],
    motion_gate_inference: bool,
    probability_threshold: float = 0.5,
) -> np.ndarray:
    if motion_probabilities is None or not motion_gate_inference:
        return predictions
    gated = predictions.copy()
    static_mask = motion_probabilities < probability_threshold
    if not np.any(static_mask):
        return gated
    gated[static_mask, :] = 0.0
    if target_names == ["delta_distance_xy", "delta_heading"]:
        gated[static_mask, 1] = 0.0
    return gated


def apply_oracle_motion_gate(
    predictions: np.ndarray,
    target_magnitudes: np.ndarray,
    target_names: list[str],
    motion_threshold_m: float,
) -> np.ndarray:
    gated = predictions.copy()
    static_mask = target_magnitudes < motion_threshold_m
    if not np.any(static_mask):
        return gated
    gated[static_mask, :] = 0.0
    if target_names == ["delta_distance_xy", "delta_heading"]:
        gated[static_mask, 1] = 0.0
    return gated


def compute_target_motion_magnitudes(
    targets: np.ndarray,
    target_names: list[str],
) -> np.ndarray:
    is_polar_xy = target_names == ["delta_distance_xy", "delta_heading"]
    if is_polar_xy:
        return np.abs(targets[:, 0])
    return np.sqrt(np.sum(np.square(targets), axis=1))


def compute_motion_classification_metrics(
    motion_probabilities: np.ndarray,
    target_magnitudes: np.ndarray,
    motion_threshold_m: float,
    probability_threshold: float = 0.5,
) -> dict[str, float]:
    labels = target_magnitudes >= motion_threshold_m
    predictions = motion_probabilities >= probability_threshold
    tp = int(np.sum(predictions & labels))
    tn = int(np.sum((~predictions) & (~labels)))
    fp = int(np.sum(predictions & (~labels)))
    fn = int(np.sum((~predictions) & labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "num_positive": int(np.sum(labels)),
        "num_negative": int(np.sum(~labels)),
        "positive_rate": float(np.mean(labels)),
        "predicted_positive_rate": float(np.mean(predictions)),
    }


def magnitude_bucket_counts(
    magnitudes: np.ndarray,
    bucket_edges: list[float],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, lower, upper in motion_bucket_labels(bucket_edges):
        counts[label] = int(np.sum(bucket_mask(magnitudes, lower=lower, upper=upper)))
    return counts


def summarize_magnitude_subset(
    predicted_magnitude: np.ndarray,
    true_magnitude: np.ndarray,
    mask: np.ndarray,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    count = int(np.sum(mask))
    if count == 0:
        return {"num_samples": 0}
    pred_subset = predicted_magnitude[mask].astype(np.float64)
    true_subset = true_magnitude[mask].astype(np.float64)
    ratio_subset = pred_subset / np.maximum(true_subset, epsilon)
    return {
        "num_samples": count,
        "mean_true_magnitude_m": float(np.mean(true_subset)),
        "mean_predicted_magnitude_m": float(np.mean(pred_subset)),
        "median_true_magnitude_m": float(np.median(true_subset)),
        "median_predicted_magnitude_m": float(np.median(pred_subset)),
        "mean_predicted_to_true_ratio": float(np.mean(ratio_subset)),
        "median_predicted_to_true_ratio": float(np.median(ratio_subset)),
        "predicted_true_magnitude_correlation": safe_correlation(pred_subset, true_subset),
    }


def compute_magnitude_diagnostics(
    predicted_magnitude: np.ndarray,
    true_magnitude: np.ndarray,
    bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    predicted_magnitude = predicted_magnitude.astype(np.float64)
    true_magnitude = true_magnitude.astype(np.float64)
    ratio = predicted_magnitude / np.maximum(true_magnitude, epsilon)
    diagnostics: dict[str, Any] = {
        "mean_true_magnitude_m": float(np.mean(true_magnitude)),
        "mean_predicted_magnitude_m": float(np.mean(predicted_magnitude)),
        "median_true_magnitude_m": float(np.median(true_magnitude)),
        "median_predicted_magnitude_m": float(np.median(predicted_magnitude)),
        "mean_predicted_to_true_ratio": float(np.mean(ratio)),
        "median_predicted_to_true_ratio": float(np.median(ratio)),
        "predicted_true_magnitude_correlation": safe_correlation(predicted_magnitude, true_magnitude),
    }
    if bucket_edges:
        sorted_edges = sorted(bucket_edges)
        diagnostics["true_magnitude_bucket_counts"] = magnitude_bucket_counts(true_magnitude, sorted_edges)
        diagnostics["predicted_magnitude_bucket_counts"] = magnitude_bucket_counts(
            predicted_magnitude, sorted_edges
        )
    if moving_eval_thresholds:
        moving_summary: dict[str, Any] = {}
        for threshold_m in sorted(set(moving_eval_thresholds)):
            label = f"ge_{threshold_m:.2f}m"
            moving_summary[label] = summarize_magnitude_subset(
                predicted_magnitude,
                true_magnitude,
                mask=true_magnitude >= threshold_m,
                epsilon=epsilon,
            )
        diagnostics["moving_only_by_threshold"] = moving_summary
    return diagnostics


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    is_polar_xy = target_names == ["delta_distance_xy", "delta_heading"]
    if is_polar_xy:
        distance_errors = predictions[:, 0] - targets[:, 0]
        heading_errors = np.arctan2(
            np.sin(predictions[:, 1] - targets[:, 1]),
            np.cos(predictions[:, 1] - targets[:, 1]),
        )
        abs_errors = np.stack([np.abs(distance_errors), np.abs(heading_errors)], axis=1)
        squared_errors = np.stack([np.square(distance_errors), np.square(heading_errors)], axis=1)

        pred_xy = np.stack(
            [predictions[:, 0] * np.cos(predictions[:, 1]), predictions[:, 0] * np.sin(predictions[:, 1])],
            axis=1,
        )
        target_xy = np.stack(
            [targets[:, 0] * np.cos(targets[:, 1]), targets[:, 0] * np.sin(targets[:, 1])],
            axis=1,
        )
        vector_error = np.linalg.norm(pred_xy - target_xy, axis=1)
        predicted_vector_magnitude = np.abs(predictions[:, 0])
        true_vector_magnitude = np.abs(targets[:, 0])
    else:
        errors = predictions - targets
        abs_errors = np.abs(errors)
        squared_errors = np.square(errors)
        vector_error = np.sqrt(np.sum(squared_errors, axis=1))
        predicted_vector_magnitude = np.sqrt(np.sum(np.square(predictions), axis=1))
        true_vector_magnitude = compute_target_motion_magnitudes(targets, target_names)

    per_dim: dict[str, dict[str, float]] = {}
    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    for dim, name in enumerate(target_names):
        mae = float(np.mean(abs_errors[:, dim]))
        rmse = float(np.sqrt(np.mean(squared_errors[:, dim])))
        maes.append(mae)
        rmses.append(rmse)
        try:
            if is_polar_xy and name == "delta_heading":
                r2_value = float("nan")
            else:
                r2_value = float(r2_score(targets[:, dim], predictions[:, dim]))
        except ValueError:
            r2_value = float("nan")
        r2s.append(r2_value)
        per_dim[name] = {"mae": mae, "rmse": rmse, "r2": r2_value}
    zero_motion_vector_mae = float(np.mean(true_vector_magnitude))
    average_true_displacement_magnitude = zero_motion_vector_mae
    relative_error_percentage = (
        float(np.mean(vector_error) / zero_motion_vector_mae * 100.0)
        if zero_motion_vector_mae > 1e-12
        else float("nan")
    )
    metrics: dict[str, Any] = {
        "per_dim": per_dim,
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "vector_mae": float(np.mean(vector_error)),
        "vector_rmse": float(np.sqrt(np.mean(np.square(vector_error)))),
        "average_true_displacement_magnitude": average_true_displacement_magnitude,
        "zero_motion_baseline_vector_mae": zero_motion_vector_mae,
        "relative_error_percentage": relative_error_percentage,
        "r2_mean": (
            float(np.nanmean(r2s))
            if r2s and not np.all(np.isnan(np.asarray(r2s, dtype=np.float32)))
            else float("nan")
        ),
        "num_samples": int(predictions.shape[0]),
    }
    if motion_bucket_edges:
        metrics["motion_buckets"] = compute_motion_bucket_metrics(
            vector_error=vector_error,
            true_vector_magnitude=true_vector_magnitude,
            bucket_edges=motion_bucket_edges,
        )
    if target_names in (["delta_x", "delta_y"], ["delta_distance_xy", "delta_heading"]):
        metrics["magnitude_diagnostics"] = compute_magnitude_diagnostics(
            predicted_magnitude=predicted_vector_magnitude,
            true_magnitude=true_vector_magnitude,
            bucket_edges=motion_bucket_edges,
            moving_eval_thresholds=moving_eval_thresholds,
        )
    if moving_eval_thresholds:
        moving_metrics: dict[str, Any] = {}
        for threshold_m in sorted(set(moving_eval_thresholds)):
            moving_mask = true_vector_magnitude >= threshold_m
            label = f"ge_{threshold_m:.2f}m"
            if not np.any(moving_mask):
                moving_metrics[label] = {"num_samples": 0}
                continue
            moving_metrics[label] = compute_metrics(
                predictions[moving_mask],
                targets[moving_mask],
                target_names=target_names,
                motion_bucket_edges=motion_bucket_edges,
                moving_eval_thresholds=None,
            )
        metrics["moving_only_thresholds"] = moving_metrics
    if target_names == ["delta_z"] or target_names == ["vel_z"]:
        metrics["depth_mae"] = per_dim[target_names[0]]["mae"]
        metrics["depth_rmse"] = per_dim[target_names[0]]["rmse"]
    return metrics


def compute_masked_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
    mask: np.ndarray,
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {"num_samples": 0}
    return compute_metrics(
        predictions[mask],
        targets[mask],
        target_names=target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=moving_eval_thresholds,
    )


def motion_bucket_labels(bucket_edges: list[float]) -> list[tuple[str, float | None, float | None]]:
    edges = sorted(bucket_edges)
    labels: list[tuple[str, float | None, float | None]] = []
    lower = None
    for edge in edges:
        if lower is None:
            labels.append((f"lt_{edge:.2f}m", None, edge))
        else:
            labels.append((f"{lower:.2f}_to_{edge:.2f}m", lower, edge))
        lower = edge
    if lower is not None:
        labels.append((f"ge_{lower:.2f}m", lower, None))
    return labels


def compute_motion_bucket_metrics(
    vector_error: np.ndarray,
    true_vector_magnitude: np.ndarray,
    bucket_edges: list[float],
) -> dict[str, Any]:
    bucket_metrics: dict[str, Any] = {}
    for label, lower, upper in motion_bucket_labels(bucket_edges):
        mask = np.ones_like(true_vector_magnitude, dtype=bool)
        if lower is not None:
            mask &= true_vector_magnitude >= lower
        if upper is not None:
            mask &= true_vector_magnitude < upper
        count = int(np.sum(mask))
        if count == 0:
            bucket_metrics[label] = {"num_samples": 0}
            continue
        bucket_true = true_vector_magnitude[mask]
        bucket_error = vector_error[mask]
        zero_motion = float(np.mean(bucket_true))
        vector_mae = float(np.mean(bucket_error))
        bucket_metrics[label] = {
            "num_samples": count,
            "average_true_displacement_magnitude": zero_motion,
            "zero_motion_baseline_vector_mae": zero_motion,
            "vector_mae": vector_mae,
            "relative_error_percentage": (
                float(vector_mae / zero_motion * 100.0) if zero_motion > 1e-12 else float("nan")
            ),
        }
    return bucket_metrics


def metadata_rows_from_dataset(dataset: SensorWindowDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in dataset.samples:
        session = dataset.sessions[sample.session_idx]
        rows.append(
            {
                "session_name": session.session_name,
                "file_path": session.file_path,
                "window_start": sample.window_start,
                "window_end": sample.window_end,
                "target_start": sample.target_start,
                "target_end": sample.target_end,
                "start_time_s": float(session.timestamps_s[sample.target_start]),
                "end_time_s": float(session.timestamps_s[sample.target_end]),
                "target_magnitude": float(sample.target_magnitude),
                "loss_weight": float(sample.loss_weight),
            }
        )
    return rows


def dataset_true_targets(dataset: SensorWindowDataset) -> np.ndarray:
    return np.stack([sample.true_target for sample in dataset.samples], axis=0).astype(np.float32)


def dataset_classical_targets(dataset: SensorWindowDataset) -> np.ndarray:
    return np.stack([sample.classical_target for sample in dataset.samples], axis=0).astype(np.float32)


def extract_window_summary_features(dataset: SensorWindowDataset) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sample in dataset.samples:
        session = dataset.sessions[sample.session_idx]
        window_start = sample.window_start
        window_end = sample.window_end
        feature_parts: list[np.ndarray] = []
        for values in (
            session.accel_xyz,
            session.gyro_xyz,
            session.gravity_xyz,
            session.nav_accel_xyz,
            session.nav_gyro_xyz,
        ):
            if values is None:
                continue
            window_values = values[window_start:window_end].astype(np.float32)
            feature_parts.extend(
                [
                    np.mean(window_values, axis=0),
                    np.std(window_values, axis=0),
                    np.min(window_values, axis=0),
                    np.max(window_values, axis=0),
                    window_values[-1] - window_values[0],
                ]
            )
        rows.append(np.concatenate(feature_parts, axis=0).astype(np.float32))
    return np.stack(rows, axis=0).astype(np.float32)


def evaluate_prediction_baseline(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    return compute_metrics(
        predictions=predictions,
        targets=targets,
        target_names=target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=moving_eval_thresholds,
    )


def evaluate_zero_motion_baseline(
    dataset: SensorWindowDataset,
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    targets = dataset_true_targets(dataset)
    predictions = np.zeros_like(targets, dtype=np.float32)
    return evaluate_prediction_baseline(predictions, targets, dataset.target_names, motion_bucket_edges, moving_eval_thresholds)


def evaluate_train_mean_baseline(
    train_dataset: SensorWindowDataset,
    eval_dataset: SensorWindowDataset,
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    train_targets = dataset_true_targets(train_dataset)
    eval_targets = dataset_true_targets(eval_dataset)
    mean_target = np.mean(train_targets, axis=0, keepdims=True).astype(np.float32)
    predictions = np.repeat(mean_target, repeats=eval_targets.shape[0], axis=0)
    return evaluate_prediction_baseline(predictions, eval_targets, eval_dataset.target_names, motion_bucket_edges, moving_eval_thresholds)


def evaluate_previous_window_persistence_baseline(
    dataset: SensorWindowDataset,
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    targets = dataset_true_targets(dataset)
    predictions = np.zeros_like(targets, dtype=np.float32)
    previous_by_session: dict[int, np.ndarray] = {}
    for index, sample in enumerate(dataset.samples):
        if sample.session_idx in previous_by_session:
            predictions[index] = previous_by_session[sample.session_idx]
        previous_by_session[sample.session_idx] = sample.true_target
    return evaluate_prediction_baseline(predictions, targets, dataset.target_names, motion_bucket_edges, moving_eval_thresholds)


def evaluate_tabular_linear_regression_baseline(
    train_dataset: SensorWindowDataset,
    eval_dataset: SensorWindowDataset,
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    train_features = extract_window_summary_features(train_dataset)
    eval_features = extract_window_summary_features(eval_dataset)
    train_targets = dataset_true_targets(train_dataset)
    eval_targets = dataset_true_targets(eval_dataset)
    model = LinearRegression()
    model.fit(train_features, train_targets)
    predictions = model.predict(eval_features).astype(np.float32)
    metrics = evaluate_prediction_baseline(predictions, eval_targets, eval_dataset.target_names, motion_bucket_edges, moving_eval_thresholds)
    metrics["num_engineered_features"] = int(train_features.shape[1])
    return metrics


def classical_double_integration_prediction(
    dataset: SensorWindowDataset,
    sample_index: int,
) -> np.ndarray:
    sample = dataset.samples[sample_index]
    session = dataset.sessions[sample.session_idx]
    return compute_classical_delta_target(
        session=session,
        target_start=sample.target_start,
        target_end=sample.target_end,
        target_mode=dataset.target_mode,
        quaternion_frame_mode=dataset.quaternion_frame_mode,
    )


def evaluate_classical_double_integration(
    dataset: SensorWindowDataset,
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    predictions = np.stack(
        [classical_double_integration_prediction(dataset, sample_index) for sample_index in range(len(dataset))],
        axis=0,
    ).astype(np.float32)
    targets = dataset_true_targets(dataset)
    metrics = compute_metrics(
        predictions,
        targets,
        target_names=dataset.target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=moving_eval_thresholds,
    )
    return predictions, targets, metadata_rows_from_dataset(dataset), metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_mode: str,
    target_names: list[str],
    motion_bucket_edges: list[float] | None = None,
    moving_eval_thresholds: list[float] | None = None,
    loss_name: str | None = None,
    smooth_l1_beta: float = 1.0,
    motion_head_enabled: bool = False,
    use_motion_gate: bool = False,
    motion_gate_threshold_m: float = 0.10,
    motion_gate_prob_threshold: float = 0.5,
    motion_cls_loss_weight: float = 1.0,
    motion_reg_loss_weight: float = 1.0,
    motion_class_balance_mode: str = "none",
    delta_xy_parameterization: str = "cartesian",
    magnitude_loss_weight: float = 1.0,
    magnitude_loss_type: str = "l1",
    direction_loss_weight: float = 1.0,
    direction_loss_min_target_magnitude_m: float = 0.02,
    polar_distance_weight: float = 1.0,
    polar_heading_weight: float = 1.0,
    normalization=None,
    use_classical_residual_target: bool = False,
    dataset: SensorWindowDataset | None = None,
    hybrid_head_enabled: bool = False,
    hybrid_bias_correction_enabled: bool = False,
    hybrid_bias_regularization: float = 0.0,
    hybrid_log_variance_min: float = -6.0,
    hybrid_log_variance_max: float = 6.0,
    uncertainty_calibration_num_buckets: int = 5,
    planar_ekf_rollout_enabled: bool = False,
    ekf_measurement_covariance_mode: str = "learned_uncertainty",
    empirical_measurement_covariance: dict[str, Any] | None = None,
    planar_ekf_process_accel_std_mps2: float = 0.75,
    planar_ekf_process_noise_scale: float = 1.0,
    planar_ekf_measurement_covariance_scale: float = 1.0,
    planar_ekf_measurement_covariance_floor: float | None = None,
    planar_ekf_measurement_covariance_ceiling: float | None = None,
    planar_ekf_velocity_measurement_std_floor_mps: float = 0.05,
    planar_ekf_velocity_measurement_std_max_mps: float = 5.0,
    planar_ekf_initial_position_std_m: float = 0.10,
    planar_ekf_initial_velocity_std_mps: float = 0.25,
    planar_ekf_update_stride: int = 1,
    plane_relative_tracker_rollout_enabled: bool = False,
    plane_tracker_position_blend: float = 0.70,
    plane_tracker_velocity_blend: float = 0.35,
    plane_tracker_uncertainty_scale: float = 0.10,
    plane_tracker_velocity_damping_per_s: float = 0.35,
    plane_tracker_height_constraint_gain: float = 8.0,
    plane_tracker_normal_accel_gain: float = 0.05,
    plane_tracker_max_height_offset_m: float = 0.05,
    plane_tracker_short_horizon_steps: int = 3,
    plane_tracker_surface_half_width_m: float = 0.15,
    plane_tracker_surface_half_height_m: float = 0.10,
) -> tuple[
    float | None,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    np.ndarray | None,
]:
    model.eval()
    loss_numerator = 0.0
    loss_denominator = 0.0
    motion_loss_sum = 0.0
    motion_loss_count = 0.0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    motion_probabilities: list[np.ndarray] = []
    classical_targets: list[np.ndarray] = []
    true_targets: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    bias_corrections: list[np.ndarray] = []

    for batch_inputs, batch_targets, batch_metadata in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        sequence_lengths = batch_metadata.get("sequence_length")
        if torch.is_tensor(sequence_lengths):
            sequence_lengths = sequence_lengths.to(device)
        batch_outputs = model(batch_inputs, lengths=sequence_lengths)
        (
            batch_regression,
            batch_motion_logits,
            batch_log_variance,
            batch_bias_correction,
        ) = unpack_model_outputs(batch_outputs)
        if loss_name is not None:
            batch_weights = batch_metadata.get("loss_weight")
            if torch.is_tensor(batch_weights):
                batch_weights = batch_weights.to(device)
            batch_loss_values = compute_loss_values(
                batch_regression,
                batch_targets,
                loss_name=loss_name,
                smooth_l1_beta=smooth_l1_beta,
                target_mode=target_mode,
                delta_xy_parameterization=delta_xy_parameterization,
                magnitude_loss_weight=magnitude_loss_weight,
                magnitude_loss_type=magnitude_loss_type,
                direction_loss_weight=direction_loss_weight,
                direction_loss_min_target_magnitude_m=direction_loss_min_target_magnitude_m,
                polar_distance_weight=polar_distance_weight,
                polar_heading_weight=polar_heading_weight,
                predicted_log_variance=batch_log_variance if hybrid_head_enabled else None,
                predicted_bias_correction=batch_bias_correction if hybrid_head_enabled else None,
                hybrid_bias_regularization=hybrid_bias_regularization,
                hybrid_log_variance_min=hybrid_log_variance_min,
                hybrid_log_variance_max=hybrid_log_variance_max,
            )
            numerator, denominator = summarize_loss(batch_loss_values, batch_weights)
            loss_numerator += float(numerator.item())
            loss_denominator += float(denominator.item())
            if motion_head_enabled and batch_motion_logits is not None:
                target_magnitude = batch_metadata["target_magnitude"]
                if torch.is_tensor(target_magnitude):
                    motion_targets = (target_magnitude.to(device) >= motion_gate_threshold_m).float()
                else:
                    motion_targets = torch.as_tensor(
                        np.asarray(target_magnitude) >= motion_gate_threshold_m,
                        device=device,
                        dtype=torch.float32,
                    )
                motion_loss = compute_motion_loss_values(
                    batch_motion_logits.view(-1),
                    motion_targets,
                    balance_mode=motion_class_balance_mode,
                )
                motion_loss_sum += float((motion_loss * motion_cls_loss_weight).sum().item())
                motion_loss_count += float(motion_loss.numel())
        decoded_batch_regression = decode_predictions(
            batch_regression,
            target_mode=target_mode,
            delta_xy_parameterization=delta_xy_parameterization,
            predicted_bias_correction=batch_bias_correction if hybrid_bias_correction_enabled else None,
        )
        predictions.append(decoded_batch_regression.detach().cpu().numpy())
        targets.append(batch_targets.detach().cpu().numpy())
        if "classical_target" in batch_metadata:
            classical_value = batch_metadata["classical_target"]
            if torch.is_tensor(classical_value):
                classical_targets.append(classical_value.detach().cpu().numpy())
            else:
                classical_targets.append(np.asarray(classical_value, dtype=np.float32))
        if "true_target" in batch_metadata:
            true_value = batch_metadata["true_target"]
            if torch.is_tensor(true_value):
                true_targets.append(true_value.detach().cpu().numpy())
            else:
                true_targets.append(np.asarray(true_value, dtype=np.float32))
        metadata_rows.extend(expand_metadata_batch(batch_metadata))
        if batch_motion_logits is not None:
            motion_probabilities.append(torch.sigmoid(batch_motion_logits.view(-1)).detach().cpu().numpy())
        if batch_log_variance is not None:
            log_variances.append(batch_log_variance.detach().cpu().numpy())
        if batch_bias_correction is not None:
            bias_corrections.append(batch_bias_correction.detach().cpu().numpy())

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    log_variance_array = np.concatenate(log_variances, axis=0) if log_variances else None
    bias_correction_array = np.concatenate(bias_corrections, axis=0) if bias_corrections else None
    prediction_array = denormalize_targets(prediction_array, normalization)
    target_array = denormalize_targets(target_array, normalization)
    if use_classical_residual_target:
        if not classical_targets or not true_targets:
            raise ValueError("Residual evaluation requires classical_target and true_target metadata.")
        classical_array = np.concatenate(classical_targets, axis=0).astype(np.float32)
        true_target_array = np.concatenate(true_targets, axis=0).astype(np.float32)
        prediction_array = prediction_array + classical_array
        target_array = true_target_array
    target_magnitude_array = compute_target_motion_magnitudes(target_array, target_names)
    motion_probability_array = (
        np.concatenate(motion_probabilities, axis=0) if motion_probabilities else None
    )
    ungated_prediction_array = prediction_array.copy()
    gated_prediction_array = apply_motion_gate(
        prediction_array,
        motion_probability_array,
        target_names=target_names,
        motion_gate_inference=use_motion_gate,
        probability_threshold=motion_gate_prob_threshold,
    )
    oracle_prediction_array = apply_oracle_motion_gate(
        prediction_array,
        target_magnitude_array,
        target_names=target_names,
        motion_threshold_m=motion_gate_threshold_m,
    )

    mean_loss = None
    if loss_denominator > 0.0:
        mean_loss = float((loss_numerator * motion_reg_loss_weight) / loss_denominator)
        if motion_loss_count > 0.0:
            mean_loss += float(motion_loss_sum / motion_loss_count)
    active_prediction_array = gated_prediction_array if use_motion_gate else ungated_prediction_array
    metrics = compute_metrics(
        active_prediction_array,
        target_array,
        target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=moving_eval_thresholds,
    )
    ungated_metrics = compute_metrics(
        ungated_prediction_array,
        target_array,
        target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=moving_eval_thresholds,
    )
    gated_metrics = compute_metrics(
        gated_prediction_array,
        target_array,
        target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=moving_eval_thresholds,
    )
    oracle_metrics = compute_metrics(
        oracle_prediction_array,
        target_array,
        target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=moving_eval_thresholds,
    )
    motion_labels = target_magnitude_array >= motion_gate_threshold_m
    static_labels = ~motion_labels
    metrics["ungated_regression"] = ungated_metrics
    metrics["motion_gated"] = gated_metrics
    metrics["oracle_motion_gate"] = oracle_metrics
    metrics["gated_vector_mae"] = float(gated_metrics["vector_mae"])
    metrics["gated_relative_error_percentage"] = float(gated_metrics["relative_error_percentage"])
    metrics["ungated_vector_mae"] = float(ungated_metrics["vector_mae"])
    metrics["ungated_relative_error_percentage"] = float(ungated_metrics["relative_error_percentage"])
    metrics["regression_by_motion_label"] = {
        "moving": compute_masked_metrics(
            ungated_prediction_array,
            target_array,
            target_names,
            motion_labels,
            motion_bucket_edges=motion_bucket_edges,
            moving_eval_thresholds=None,
        ),
        "static": compute_masked_metrics(
            ungated_prediction_array,
            target_array,
            target_names,
            static_labels,
            motion_bucket_edges=motion_bucket_edges,
            moving_eval_thresholds=None,
        ),
    }
    metrics["motion_label_counts"] = {
        "moving": int(np.sum(motion_labels)),
        "static": int(np.sum(static_labels)),
    }
    if target_mode == "delta_xy" and delta_xy_parameterization == "direction_magnitude":
        direction_metrics = compute_direction_magnitude_metrics(
            torch.from_numpy(ungated_prediction_array),
            torch.from_numpy(target_array),
            min_target_magnitude_m=direction_loss_min_target_magnitude_m,
        )
        metrics["direction_magnitude"] = direction_metrics
    if motion_probability_array is not None:
        metrics["motion_classification"] = compute_motion_classification_metrics(
            motion_probability_array,
            target_magnitudes=target_magnitude_array,
            motion_threshold_m=motion_gate_threshold_m,
            probability_threshold=motion_gate_prob_threshold,
        )
    if log_variance_array is not None:
        sigma_array = sigma_from_log_variance(
            log_variance_array,
            normalization,
            log_variance_min=hybrid_log_variance_min,
            log_variance_max=hybrid_log_variance_max,
        )
        uncertainty_metrics = compute_uncertainty_diagnostics(
            predictions_xy=active_prediction_array,
            targets_xy=target_array,
            sigma_xy=sigma_array,
            log_variance_xy=log_variance_array,
            log_variance_min=hybrid_log_variance_min,
            log_variance_max=hybrid_log_variance_max,
            num_buckets=uncertainty_calibration_num_buckets,
        )
        if uncertainty_metrics is not None:
            metrics["hybrid_uncertainty"] = uncertainty_metrics
    else:
        sigma_array = None
    if bias_correction_array is not None:
        denormalized_bias_correction = denormalize_delta_scale(bias_correction_array, normalization)
        metrics["hybrid_bias_correction"] = summarize_delta_scale(
            denormalized_bias_correction,
            prefix="bias",
        )
    if dataset is not None and target_mode == "delta_xy":
        classical_array = dataset_classical_targets(dataset)
        metrics["sequential_rollout"] = evaluate_sequential_rollouts(
            dataset=dataset,
            learned_displacements_xy=active_prediction_array,
            classical_displacements_xy=classical_array,
            learned_sigma_xy=sigma_array,
            ekf_measurement_covariance_mode=ekf_measurement_covariance_mode,
            empirical_measurement_covariance=empirical_measurement_covariance,
            planar_ekf_rollout_enabled=planar_ekf_rollout_enabled,
            planar_ekf_process_accel_std_mps2=planar_ekf_process_accel_std_mps2,
            planar_ekf_process_noise_scale=planar_ekf_process_noise_scale,
            planar_ekf_measurement_covariance_scale=planar_ekf_measurement_covariance_scale,
            planar_ekf_measurement_covariance_floor=planar_ekf_measurement_covariance_floor,
            planar_ekf_measurement_covariance_ceiling=planar_ekf_measurement_covariance_ceiling,
            planar_ekf_velocity_measurement_std_floor_mps=(
                planar_ekf_velocity_measurement_std_floor_mps
            ),
            planar_ekf_velocity_measurement_std_max_mps=(
                planar_ekf_velocity_measurement_std_max_mps
            ),
            planar_ekf_initial_position_std_m=planar_ekf_initial_position_std_m,
            planar_ekf_initial_velocity_std_mps=planar_ekf_initial_velocity_std_mps,
            planar_ekf_update_stride=planar_ekf_update_stride,
            plane_relative_tracker_rollout_enabled=plane_relative_tracker_rollout_enabled,
            plane_tracker_position_blend=plane_tracker_position_blend,
            plane_tracker_velocity_blend=plane_tracker_velocity_blend,
            plane_tracker_uncertainty_scale=plane_tracker_uncertainty_scale,
            plane_tracker_velocity_damping_per_s=plane_tracker_velocity_damping_per_s,
            plane_tracker_height_constraint_gain=plane_tracker_height_constraint_gain,
            plane_tracker_normal_accel_gain=plane_tracker_normal_accel_gain,
            plane_tracker_max_height_offset_m=plane_tracker_max_height_offset_m,
            plane_tracker_short_horizon_steps=plane_tracker_short_horizon_steps,
            plane_tracker_surface_half_width_m=plane_tracker_surface_half_width_m,
            plane_tracker_surface_half_height_m=plane_tracker_surface_half_height_m,
        )
        sequential_rollout = metrics["sequential_rollout"]
        if planar_ekf_rollout_enabled and "planar_ekf" in sequential_rollout:
            metrics["hybrid_comparison"] = {
                "planar_ekf_vs_direct_position_mae_m_delta": (
                    sequential_rollout["planar_ekf"]["position_mae_m"]
                    - sequential_rollout["direct_model"]["position_mae_m"]
                ),
                "planar_ekf_vs_classical_position_mae_m_delta": (
                    sequential_rollout["planar_ekf"]["position_mae_m"]
                    - sequential_rollout["classical_baseline"]["position_mae_m"]
                ),
            }
        if plane_relative_tracker_rollout_enabled and "plane_relative_tracker" in sequential_rollout:
            plane_tracker_comparison = metrics.setdefault("hybrid_comparison", {})
            plane_tracker_comparison["plane_relative_tracker_vs_direct_position_mae_m_delta"] = (
                sequential_rollout["plane_relative_tracker"]["position_mae_m"]
                - sequential_rollout["direct_model"]["position_mae_m"]
            )
            plane_tracker_comparison["plane_relative_tracker_vs_classical_position_mae_m_delta"] = (
                sequential_rollout["plane_relative_tracker"]["position_mae_m"]
                - sequential_rollout["classical_baseline"]["position_mae_m"]
            )
            if "planar_ekf" in sequential_rollout:
                plane_tracker_comparison["plane_relative_tracker_vs_planar_ekf_position_mae_m_delta"] = (
                    sequential_rollout["plane_relative_tracker"]["position_mae_m"]
                    - sequential_rollout["planar_ekf"]["position_mae_m"]
                )
    return mean_loss, metrics, active_prediction_array, target_array, metadata_rows, motion_probability_array


def dataset_from_checkpoint_config(
    files: list[str],
    train_config: TrainingConfig,
    normalization,
    split_name: str,
) -> SensorWindowDataset:
    return SensorWindowDataset(
        csv_files=files,
        feature_names=train_config.resolved_features,
        target_mode=train_config.target_mode,
        window_size=train_config.window_size,
        stride=train_config.stride,
        window_mode=train_config.window_mode,
        window_duration_s=train_config.window_duration_s,
        window_stride_s=train_config.window_stride_s,
        required_valid_flags=train_config.required_valid_flags,
        sort_column_candidates=train_config.sort_columns,
        quaternion_frame_mode=train_config.quaternion_frame_mode,
        loss_weight_mode=train_config.loss_weight_mode,
        loss_weight_reference_m=train_config.loss_weight_reference_m,
        loss_weight_min=train_config.loss_weight_min,
        loss_weight_max=train_config.loss_weight_max,
        normalization=normalization,
        fit_normalization=False,
        normalize_targets=train_config.normalize_targets,
        min_valid_label_points=train_config.min_valid_label_points,
        min_target_magnitude_m=(
            train_config.min_target_magnitude_m if split_name == "train" else 0.0
        ),
        use_classical_residual_target=train_config.use_classical_residual_target,
        classical_residual_mode=train_config.classical_residual_mode,
    )


@torch.no_grad()
def calibrate_empirical_measurement_covariance(
    model: torch.nn.Module,
    calibration_dataset: SensorWindowDataset,
    calibration_dataloader: DataLoader,
    device: torch.device,
    train_config: TrainingConfig,
    target_names: list[str],
    normalization,
    calibration_split_name: str,
) -> dict[str, Any]:
    _, _, calibration_predictions, calibration_targets, _, _ = evaluate_model(
        model=model,
        dataloader=calibration_dataloader,
        device=device,
        target_mode=train_config.target_mode,
        target_names=target_names,
        motion_bucket_edges=train_config.motion_bucket_edges,
        moving_eval_thresholds=train_config.moving_eval_thresholds,
        loss_name=None,
        smooth_l1_beta=train_config.smooth_l1_beta,
        motion_head_enabled=train_config.motion_head_enabled,
        use_motion_gate=train_config.use_motion_gate,
        motion_gate_threshold_m=train_config.motion_gate_threshold_m,
        motion_gate_prob_threshold=train_config.motion_gate_prob_threshold,
        motion_cls_loss_weight=train_config.motion_cls_loss_weight,
        motion_reg_loss_weight=train_config.motion_reg_loss_weight,
        motion_class_balance_mode=train_config.motion_class_balance_mode,
        delta_xy_parameterization=train_config.delta_xy_parameterization,
        magnitude_loss_weight=train_config.magnitude_loss_weight,
        magnitude_loss_type=train_config.magnitude_loss_type,
        direction_loss_weight=train_config.direction_loss_weight,
        direction_loss_min_target_magnitude_m=train_config.direction_loss_min_target_magnitude_m,
        polar_distance_weight=train_config.polar_distance_weight,
        polar_heading_weight=train_config.polar_heading_weight,
        normalization=normalization,
        use_classical_residual_target=train_config.use_classical_residual_target,
        dataset=None,
        hybrid_head_enabled=train_config.hybrid_head_enabled,
        hybrid_bias_correction_enabled=train_config.hybrid_bias_correction_enabled,
        hybrid_bias_regularization=train_config.hybrid_bias_regularization,
        hybrid_log_variance_min=train_config.hybrid_log_variance_min,
        hybrid_log_variance_max=train_config.hybrid_log_variance_max,
        uncertainty_calibration_num_buckets=train_config.uncertainty_calibration_num_buckets,
        planar_ekf_rollout_enabled=False,
    )
    return estimate_empirical_measurement_covariance(
        predictions_xy=calibration_predictions,
        targets_xy=calibration_targets,
        bucket_edges=train_config.motion_bucket_edges,
        calibration_split_name=calibration_split_name,
    )


def main() -> None:
    args = build_eval_argparser().parse_args()
    eval_config = eval_config_from_args(args)

    checkpoint = load_checkpoint(eval_config.checkpoint, map_location="cpu")
    config_fields = set(TrainingConfig.__dataclass_fields__.keys())
    config_payload = {key: value for key, value in checkpoint["config"].items() if key in config_fields}
    train_config = TrainingConfig(**config_payload)
    if "use_motion_gate" not in config_payload:
        train_config.use_motion_gate = bool(
            train_config.motion_head_enabled and train_config.motion_gate_inference
        )
    if "motion_gate_threshold_m" not in config_payload:
        train_config.motion_gate_threshold_m = train_config.motion_threshold_m
    if "motion_cls_loss_weight" not in config_payload:
        train_config.motion_cls_loss_weight = train_config.motion_loss_weight
    if "motion_reg_loss_weight" not in config_payload:
        train_config.motion_reg_loss_weight = 1.0
    if eval_config.ekf_measurement_covariance_mode is not None:
        train_config.ekf_measurement_covariance_mode = eval_config.ekf_measurement_covariance_mode
    if eval_config.planar_ekf_process_noise_scale is not None:
        train_config.planar_ekf_process_noise_scale = eval_config.planar_ekf_process_noise_scale
    if eval_config.planar_ekf_measurement_covariance_scale is not None:
        train_config.planar_ekf_measurement_covariance_scale = (
            eval_config.planar_ekf_measurement_covariance_scale
        )
    if eval_config.planar_ekf_measurement_covariance_floor is not None:
        train_config.planar_ekf_measurement_covariance_floor = (
            eval_config.planar_ekf_measurement_covariance_floor
        )
    if eval_config.planar_ekf_measurement_covariance_ceiling is not None:
        train_config.planar_ekf_measurement_covariance_ceiling = (
            eval_config.planar_ekf_measurement_covariance_ceiling
        )
    if eval_config.planar_ekf_update_stride is not None:
        train_config.planar_ekf_update_stride = eval_config.planar_ekf_update_stride
    if eval_config.uncertainty_calibration_num_buckets is not None:
        train_config.uncertainty_calibration_num_buckets = (
            eval_config.uncertainty_calibration_num_buckets
        )
    if eval_config.plane_relative_tracker_rollout:
        train_config.plane_relative_tracker_rollout_enabled = True
    if eval_config.plane_tracker_position_blend is not None:
        train_config.plane_tracker_position_blend = eval_config.plane_tracker_position_blend
    if eval_config.plane_tracker_velocity_blend is not None:
        train_config.plane_tracker_velocity_blend = eval_config.plane_tracker_velocity_blend
    if eval_config.plane_tracker_uncertainty_scale is not None:
        train_config.plane_tracker_uncertainty_scale = eval_config.plane_tracker_uncertainty_scale
    if eval_config.plane_tracker_velocity_damping_per_s is not None:
        train_config.plane_tracker_velocity_damping_per_s = (
            eval_config.plane_tracker_velocity_damping_per_s
        )
    if eval_config.plane_tracker_height_constraint_gain is not None:
        train_config.plane_tracker_height_constraint_gain = (
            eval_config.plane_tracker_height_constraint_gain
        )
    if eval_config.plane_tracker_normal_accel_gain is not None:
        train_config.plane_tracker_normal_accel_gain = eval_config.plane_tracker_normal_accel_gain
    if eval_config.plane_tracker_max_height_offset_m is not None:
        train_config.plane_tracker_max_height_offset_m = eval_config.plane_tracker_max_height_offset_m
    if eval_config.plane_tracker_short_horizon_steps is not None:
        train_config.plane_tracker_short_horizon_steps = eval_config.plane_tracker_short_horizon_steps
    if eval_config.plane_tracker_surface_half_width_m is not None:
        train_config.plane_tracker_surface_half_width_m = eval_config.plane_tracker_surface_half_width_m
    if eval_config.plane_tracker_surface_half_height_m is not None:
        train_config.plane_tracker_surface_half_height_m = eval_config.plane_tracker_surface_half_height_m
    normalization = checkpoint["normalization"]
    target_names = checkpoint["target_names"]

    if eval_config.csv_files:
        files = eval_config.csv_files
        split_name = "custom"
    else:
        split_name = eval_config.split
        files = checkpoint["splits"].get(split_name, [])
    if not files:
        raise ValueError(f"No CSV files available for split '{split_name}'.")

    dataset = dataset_from_checkpoint_config(files, train_config, normalization, split_name=split_name)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_config.batch_size,
        shuffle=False,
        num_workers=eval_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sensor_windows,
    )

    device = resolve_device(eval_config.device)
    model = build_model(
        model_type=checkpoint["model_type"],
        input_channels=checkpoint["input_channels"],
        output_dim=checkpoint["output_dim"],
        conv_channels=checkpoint["conv_channels"],
        kernel_size=checkpoint["kernel_size"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
        activation=checkpoint["activation"],
        use_residual=checkpoint["use_residual"],
        motion_head_enabled=checkpoint.get("motion_head_enabled", train_config.motion_head_enabled),
        hybrid_head_enabled=checkpoint.get("hybrid_head_enabled", train_config.hybrid_head_enabled),
        hybrid_bias_correction_enabled=checkpoint.get(
            "hybrid_bias_correction_enabled",
            train_config.hybrid_bias_correction_enabled,
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    empirical_measurement_covariance = None
    if (
        train_config.planar_ekf_rollout_enabled
        and train_config.target_mode == "delta_xy"
        and train_config.ekf_measurement_covariance_mode != "learned_uncertainty"
    ):
        calibration_files = checkpoint["splits"].get("val", [])
        calibration_split_name = "val"
        if not calibration_files:
            calibration_files = files
            calibration_split_name = split_name
        calibration_dataset = dataset_from_checkpoint_config(
            calibration_files,
            train_config,
            normalization,
            split_name=calibration_split_name,
        )
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=eval_config.batch_size,
            shuffle=False,
            num_workers=eval_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_sensor_windows,
        )
        empirical_measurement_covariance = calibrate_empirical_measurement_covariance(
            model=model,
            calibration_dataset=calibration_dataset,
            calibration_dataloader=calibration_dataloader,
            device=device,
            train_config=train_config,
            target_names=target_names,
            normalization=normalization,
            calibration_split_name=calibration_split_name,
        )
    loss, metrics, predictions, targets, metadata_rows, _ = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        target_mode=train_config.target_mode,
        target_names=target_names,
        motion_bucket_edges=train_config.motion_bucket_edges,
        moving_eval_thresholds=train_config.moving_eval_thresholds,
        loss_name=checkpoint["loss_name"],
        smooth_l1_beta=checkpoint["smooth_l1_beta"],
        motion_head_enabled=train_config.motion_head_enabled,
        use_motion_gate=train_config.use_motion_gate,
        motion_gate_threshold_m=train_config.motion_gate_threshold_m,
        motion_gate_prob_threshold=train_config.motion_gate_prob_threshold,
        motion_cls_loss_weight=train_config.motion_cls_loss_weight,
        motion_reg_loss_weight=train_config.motion_reg_loss_weight,
        motion_class_balance_mode=train_config.motion_class_balance_mode,
        delta_xy_parameterization=train_config.delta_xy_parameterization,
        magnitude_loss_weight=train_config.magnitude_loss_weight,
        magnitude_loss_type=train_config.magnitude_loss_type,
        direction_loss_weight=train_config.direction_loss_weight,
        direction_loss_min_target_magnitude_m=train_config.direction_loss_min_target_magnitude_m,
        polar_distance_weight=train_config.polar_distance_weight,
        polar_heading_weight=train_config.polar_heading_weight,
        normalization=normalization,
        use_classical_residual_target=train_config.use_classical_residual_target,
        dataset=dataset,
        hybrid_head_enabled=train_config.hybrid_head_enabled,
        hybrid_bias_correction_enabled=train_config.hybrid_bias_correction_enabled,
        hybrid_bias_regularization=train_config.hybrid_bias_regularization,
        hybrid_log_variance_min=train_config.hybrid_log_variance_min,
        hybrid_log_variance_max=train_config.hybrid_log_variance_max,
        uncertainty_calibration_num_buckets=train_config.uncertainty_calibration_num_buckets,
        planar_ekf_rollout_enabled=train_config.planar_ekf_rollout_enabled,
        ekf_measurement_covariance_mode=train_config.ekf_measurement_covariance_mode,
        empirical_measurement_covariance=empirical_measurement_covariance,
        planar_ekf_process_accel_std_mps2=train_config.planar_ekf_process_accel_std_mps2,
        planar_ekf_process_noise_scale=train_config.planar_ekf_process_noise_scale,
        planar_ekf_measurement_covariance_scale=train_config.planar_ekf_measurement_covariance_scale,
        planar_ekf_measurement_covariance_floor=train_config.planar_ekf_measurement_covariance_floor,
        planar_ekf_measurement_covariance_ceiling=train_config.planar_ekf_measurement_covariance_ceiling,
        planar_ekf_velocity_measurement_std_floor_mps=(
            train_config.planar_ekf_velocity_measurement_std_floor_mps
        ),
        planar_ekf_velocity_measurement_std_max_mps=(
            train_config.planar_ekf_velocity_measurement_std_max_mps
        ),
        planar_ekf_initial_position_std_m=train_config.planar_ekf_initial_position_std_m,
        planar_ekf_initial_velocity_std_mps=train_config.planar_ekf_initial_velocity_std_mps,
        planar_ekf_update_stride=train_config.planar_ekf_update_stride,
        plane_relative_tracker_rollout_enabled=train_config.plane_relative_tracker_rollout_enabled,
        plane_tracker_position_blend=train_config.plane_tracker_position_blend,
        plane_tracker_velocity_blend=train_config.plane_tracker_velocity_blend,
        plane_tracker_uncertainty_scale=train_config.plane_tracker_uncertainty_scale,
        plane_tracker_velocity_damping_per_s=train_config.plane_tracker_velocity_damping_per_s,
        plane_tracker_height_constraint_gain=train_config.plane_tracker_height_constraint_gain,
        plane_tracker_normal_accel_gain=train_config.plane_tracker_normal_accel_gain,
        plane_tracker_max_height_offset_m=train_config.plane_tracker_max_height_offset_m,
        plane_tracker_short_horizon_steps=train_config.plane_tracker_short_horizon_steps,
        plane_tracker_surface_half_width_m=train_config.plane_tracker_surface_half_width_m,
        plane_tracker_surface_half_height_m=train_config.plane_tracker_surface_half_height_m,
    )
    if loss is not None:
        metrics["loss"] = loss
    if train_config.planar_ekf_rollout_enabled:
        metrics["ekf_measurement_covariance_mode"] = train_config.ekf_measurement_covariance_mode
    if train_config.plane_relative_tracker_rollout_enabled:
        metrics["plane_relative_tracker_rollout_enabled"] = True
    if empirical_measurement_covariance is not None:
        metrics["empirical_measurement_covariance"] = empirical_measurement_covariance
    baseline_predictions, baseline_targets, baseline_metadata_rows, baseline_metrics = (
        evaluate_classical_double_integration(
            dataset=dataset,
            motion_bucket_edges=train_config.motion_bucket_edges,
            moving_eval_thresholds=train_config.moving_eval_thresholds,
        )
    )
    metrics["classical_double_integration"] = baseline_metrics
    metrics["dataset_timing_diagnostics"] = dataset.summary().get("timing_diagnostics", {})
    train_files = checkpoint["splits"].get("train", [])
    train_dataset = (
        dataset_from_checkpoint_config(train_files, train_config, normalization, split_name="train")
        if train_files
        else None
    )
    metrics["zero_motion_baseline"] = evaluate_zero_motion_baseline(
        dataset=dataset,
        motion_bucket_edges=train_config.motion_bucket_edges,
        moving_eval_thresholds=train_config.moving_eval_thresholds,
    )
    metrics["previous_window_persistence_baseline"] = evaluate_previous_window_persistence_baseline(
        dataset=dataset,
        motion_bucket_edges=train_config.motion_bucket_edges,
        moving_eval_thresholds=train_config.moving_eval_thresholds,
    )
    if train_dataset is not None:
        metrics["train_mean_baseline"] = evaluate_train_mean_baseline(
            train_dataset=train_dataset,
            eval_dataset=dataset,
            motion_bucket_edges=train_config.motion_bucket_edges,
            moving_eval_thresholds=train_config.moving_eval_thresholds,
        )
        metrics["tabular_linear_regression_baseline"] = evaluate_tabular_linear_regression_baseline(
            train_dataset=train_dataset,
            eval_dataset=dataset,
            motion_bucket_edges=train_config.motion_bucket_edges,
            moving_eval_thresholds=train_config.moving_eval_thresholds,
        )

    output_dir = (
        ensure_dir(eval_config.output_dir)
        if eval_config.output_dir
        else ensure_dir(Path(eval_config.checkpoint).resolve().parent / f"eval_{split_name}")
    )
    save_json(metrics, output_dir / f"{split_name}_metrics.json")
    write_predictions_csv(
        output_dir / f"{split_name}_predictions.csv",
        metadata_rows=metadata_rows,
        predictions=predictions,
        targets=targets,
        target_names=target_names,
    )
    write_predictions_csv(
        output_dir / f"{split_name}_double_integration_predictions.csv",
        metadata_rows=baseline_metadata_rows,
        predictions=baseline_predictions,
        targets=baseline_targets,
        target_names=target_names,
    )

    print(f"Evaluated split: {split_name}")
    print(f"Samples: {metrics['num_samples']}")
    if "loss" in metrics:
        print(f"Loss: {metrics['loss']:.6f}")
    print(f"Mean MAE: {metrics['mae_mean']:.6f}")
    print(f"Mean RMSE: {metrics['rmse_mean']:.6f}")
    print(f"Vector MAE: {metrics['vector_mae']:.6f}")
    print(f"Vector RMSE: {metrics['vector_rmse']:.6f}")
    print(
        "Average true displacement magnitude: "
        f"{metrics['average_true_displacement_magnitude']:.6f}"
    )
    print(
        "Zero-motion baseline vector MAE: "
        f"{metrics['zero_motion_baseline_vector_mae']:.6f}"
    )
    print(f"Relative error percentage: {metrics['relative_error_percentage']:.2f}%")
    print(f"Gated vector MAE: {metrics['gated_vector_mae']:.6f}")
    print(f"Gated relative error percentage: {metrics['gated_relative_error_percentage']:.2f}%")
    print(f"Ungated vector MAE: {metrics['ungated_vector_mae']:.6f}")
    print(f"Ungated relative error percentage: {metrics['ungated_relative_error_percentage']:.2f}%")
    magnitude_diagnostics = metrics.get("magnitude_diagnostics")
    if magnitude_diagnostics is not None:
        print(
            "Magnitude diagnostics: "
            f"mean_true={magnitude_diagnostics['mean_true_magnitude_m']:.6f}, "
            f"mean_pred={magnitude_diagnostics['mean_predicted_magnitude_m']:.6f}, "
            f"median_true={magnitude_diagnostics['median_true_magnitude_m']:.6f}, "
            f"median_pred={magnitude_diagnostics['median_predicted_magnitude_m']:.6f}, "
            f"ratio={magnitude_diagnostics['mean_predicted_to_true_ratio']:.6f}, "
            f"corr={magnitude_diagnostics['predicted_true_magnitude_correlation']:.6f}"
        )
    motion_counts = metrics.get("motion_label_counts", {})
    if motion_counts:
        print(
            "Motion label counts: "
            f"moving={motion_counts['moving']}, static={motion_counts['static']}"
        )
    if "motion_classification" in metrics:
        motion = metrics["motion_classification"]
        print(
            "Motion classification: "
            f"acc={motion['accuracy']:.4f}, precision={motion['precision']:.4f}, "
            f"recall={motion['recall']:.4f}, f1={motion['f1']:.4f}, "
            f"tp={motion['tp']}, fp={motion['fp']}, tn={motion['tn']}, fn={motion['fn']}, "
            f"positive_rate={motion['positive_rate']:.4f}, "
            f"predicted_positive_rate={motion['predicted_positive_rate']:.4f}"
        )
    for label_name, split_metrics in metrics.get("regression_by_motion_label", {}).items():
        if split_metrics.get("num_samples", 0) == 0:
            print(f"Regression on {label_name}-only windows: samples=0")
            continue
        print(
            f"Regression on {label_name}-only windows: "
            f"vector_mae={split_metrics['vector_mae']:.6f}, "
            f"relative={split_metrics['relative_error_percentage']:.2f}% "
            f"over {split_metrics['num_samples']} windows"
        )
    if "direction_magnitude" in metrics:
        direction = metrics["direction_magnitude"]
        print(
            "Direction+magnitude: "
            f"magnitude_mae={direction['magnitude_mae']:.6f}, "
            f"mean_direction_cosine_similarity={direction['mean_direction_cosine_similarity']:.6f}, "
            f"eligible_samples={direction['num_direction_eligible_samples']}"
        )
    for label, moving in metrics.get("moving_only_thresholds", {}).items():
        if moving.get("num_samples", 0) == 0:
            print(f"Moving-only {label}: samples=0")
            continue
        print(
            f"Moving-only {label}: vector_mae={moving['vector_mae']:.6f}, "
            f"baseline={moving['zero_motion_baseline_vector_mae']:.6f}, "
            f"relative={moving['relative_error_percentage']:.2f}% "
            f"over {moving['num_samples']} windows"
        )
    oracle = metrics.get("oracle_motion_gate")
    if oracle is not None:
        print(
            "Oracle motion gate: "
            f"vector_mae={oracle['vector_mae']:.6f}, "
            f"relative={oracle['relative_error_percentage']:.2f}%"
        )
    if "classical_double_integration" in metrics:
        baseline = metrics["classical_double_integration"]
        print(
            "Classical double integration: "
            f"vector_mae={baseline['vector_mae']:.6f}, "
            f"baseline={baseline['zero_motion_baseline_vector_mae']:.6f}, "
            f"relative={baseline['relative_error_percentage']:.2f}%"
        )
        print(f"Learned vs classical delta: {metrics['vector_mae'] - baseline['vector_mae']:+.6f}")
        for label, moving in baseline.get("moving_only_thresholds", {}).items():
            if moving.get("num_samples", 0) == 0:
                print(f"Classical moving-only {label}: samples=0")
                continue
            learned = metrics.get("moving_only_thresholds", {}).get(label)
            delta_text = ""
            if learned is not None and learned.get("num_samples", 0) > 0:
                delta_text = f", learned_vs_classical={learned['vector_mae'] - moving['vector_mae']:+.6f}"
            print(
                f"Classical moving-only {label}: vector_mae={moving['vector_mae']:.6f}, "
                f"baseline={moving['zero_motion_baseline_vector_mae']:.6f}, "
                f"relative={moving['relative_error_percentage']:.2f}% "
                f"over {moving['num_samples']} windows{delta_text}"
            )
    for baseline_name in [
        "zero_motion_baseline",
        "train_mean_baseline",
        "previous_window_persistence_baseline",
        "tabular_linear_regression_baseline",
    ]:
        baseline = metrics.get(baseline_name)
        if baseline is None:
            continue
        print(
            f"{baseline_name}: vector_mae={baseline['vector_mae']:.6f}, "
            f"relative={baseline['relative_error_percentage']:.2f}%"
        )
    sequential_rollout = metrics.get("sequential_rollout", {})
    for rollout_name in ["direct_model", "classical_baseline", "planar_ekf", "plane_relative_tracker"]:
        rollout_metrics = sequential_rollout.get(rollout_name)
        if rollout_metrics is None:
            continue
        print(
            f"Sequential {rollout_name}: "
            f"position_mae={rollout_metrics['position_mae_m']:.6f}, "
            f"position_rmse={rollout_metrics['position_rmse_m']:.6f}, "
            f"final_position_error_mean={rollout_metrics['final_position_error_m_mean']:.6f}"
        )
        if rollout_name == "planar_ekf":
            covariance_diagnostics = rollout_metrics.get("measurement_covariance_diagnostics")
            if covariance_diagnostics is not None:
                print(
                    "Planar EKF covariance diagnostics: "
                    f"update_stride={int(covariance_diagnostics['update_stride'])}, "
                    f"update_fraction={covariance_diagnostics['measurement_update_fraction']:.4f}, "
                    f"cov_x_mean={covariance_diagnostics['mean_measurement_covariance_x']:.6f}, "
                    f"cov_y_mean={covariance_diagnostics['mean_measurement_covariance_y']:.6f}, "
                    f"floor_hit_x={covariance_diagnostics['covariance_floor_hit_fraction_x']:.4f}, "
                    f"floor_hit_y={covariance_diagnostics['covariance_floor_hit_fraction_y']:.4f}, "
                    f"ceiling_hit_x={covariance_diagnostics['covariance_ceiling_hit_fraction_x']:.4f}, "
                    f"ceiling_hit_y={covariance_diagnostics['covariance_ceiling_hit_fraction_y']:.4f}"
                )
        surface_stability = rollout_metrics.get("surface_stability")
        if surface_stability is not None:
            print(
                f"{rollout_name} surface stability: "
                f"anchor_drift_mae={surface_stability['surface_anchor_drift_mae_m']:.6f}, "
                f"anchor_drift_final={surface_stability['surface_anchor_drift_final_m']:.6f}, "
                f"slip_mae={surface_stability['surface_slip_mae_m']:.6f}, "
                f"corner_drift_mae={surface_stability['corner_drift_mae_m']:.6f}"
            )
        if rollout_name == "plane_relative_tracker":
            tracker_diagnostics = rollout_metrics.get("tracker_diagnostics")
            if tracker_diagnostics is not None:
                print(
                    "Plane-relative tracker diagnostics: "
                    f"position_blend={tracker_diagnostics['mean_position_blend']:.4f}, "
                    f"velocity_blend={tracker_diagnostics['mean_velocity_blend']:.4f}, "
                    f"mean_abs_height_offset={tracker_diagnostics['mean_abs_height_offset_m']:.6f}, "
                    f"max_abs_height_offset={tracker_diagnostics['max_abs_height_offset_m']:.6f}"
                )
    if "hybrid_comparison" in metrics:
        hybrid_comparison = metrics["hybrid_comparison"]
        if "planar_ekf_vs_direct_position_mae_m_delta" in hybrid_comparison:
            print(
                "Planar EKF vs direct rollout: "
                f"{hybrid_comparison['planar_ekf_vs_direct_position_mae_m_delta']:+.6f}"
            )
        if "planar_ekf_vs_classical_position_mae_m_delta" in hybrid_comparison:
            print(
                "Planar EKF vs classical rollout: "
                f"{hybrid_comparison['planar_ekf_vs_classical_position_mae_m_delta']:+.6f}"
            )
        if "plane_relative_tracker_vs_direct_position_mae_m_delta" in hybrid_comparison:
            print(
                "Plane-relative tracker vs direct rollout: "
                f"{hybrid_comparison['plane_relative_tracker_vs_direct_position_mae_m_delta']:+.6f}"
            )
        if "plane_relative_tracker_vs_classical_position_mae_m_delta" in hybrid_comparison:
            print(
                "Plane-relative tracker vs classical rollout: "
                f"{hybrid_comparison['plane_relative_tracker_vs_classical_position_mae_m_delta']:+.6f}"
            )
        if "plane_relative_tracker_vs_planar_ekf_position_mae_m_delta" in hybrid_comparison:
            print(
                "Plane-relative tracker vs planar EKF rollout: "
                f"{hybrid_comparison['plane_relative_tracker_vs_planar_ekf_position_mae_m_delta']:+.6f}"
            )
    if "hybrid_uncertainty" in metrics:
        uncertainty = metrics["hybrid_uncertainty"]
        print(
            "Hybrid uncertainty: "
            f"mean_sigma_x={uncertainty['mean_sigma_x_m']:.6f}, "
            f"mean_sigma_y={uncertainty['mean_sigma_y_m']:.6f}, "
            f"mean_abs_residual_x={uncertainty['mean_abs_residual_x_m']:.6f}, "
            f"mean_abs_residual_y={uncertainty['mean_abs_residual_y_m']:.6f}, "
            f"corr_x={uncertainty['std_abs_residual_correlation_x']:.6f}, "
            f"corr_y={uncertainty['std_abs_residual_correlation_y']:.6f}, "
            f"nll_mean={uncertainty['nll_mean']:.6f}"
        )
        clamp_fractions = uncertainty.get("log_variance_clamp_fraction")
        if clamp_fractions is not None:
            print(
                "Hybrid uncertainty clamp fractions: "
                f"below_min_x={clamp_fractions['below_min_x']:.4f}, "
                f"below_min_y={clamp_fractions['below_min_y']:.4f}, "
                f"above_max_x={clamp_fractions['above_max_x']:.4f}, "
                f"above_max_y={clamp_fractions['above_max_y']:.4f}"
            )
    if "hybrid_bias_correction" in metrics:
        bias_metrics = metrics["hybrid_bias_correction"]
        print(
            "Hybrid bias correction: "
            f"mean_abs_x={bias_metrics['mean_abs_bias_x_m']:.6f}, "
            f"mean_abs_y={bias_metrics['mean_abs_bias_y_m']:.6f}"
        )
    timing = metrics.get("dataset_timing_diagnostics")
    if timing is not None:
        print(
            "Timing: "
            f"duration_mean={timing['actual_window_duration_s_mean']:.3f}s, "
            f"duration_std={timing['actual_window_duration_s_std']:.3f}s, "
            f"duration_min={timing['actual_window_duration_s_min']:.3f}s, "
            f"duration_max={timing['actual_window_duration_s_max']:.3f}s, "
            f"zero_dt_fraction_mean={timing['zero_dt_fraction_mean']:.3f}, "
            f"nonzero_dt_count_mean={timing['nonzero_dt_count_mean']:.2f}"
        )
    for bucket_name, bucket_values in metrics.get("motion_buckets", {}).items():
        if bucket_values.get("num_samples", 0) == 0:
            print(f"{bucket_name}: samples=0")
            continue
        print(
            f"{bucket_name}: samples={bucket_values['num_samples']}, "
            f"vector_mae={bucket_values['vector_mae']:.6f}, "
            f"baseline={bucket_values['zero_motion_baseline_vector_mae']:.6f}, "
            f"relative={bucket_values['relative_error_percentage']:.2f}%"
        )
    for name in target_names:
        per_dim = metrics["per_dim"][name]
        print(
            f"{name}: MAE={per_dim['mae']:.6f}, RMSE={per_dim['rmse']:.6f}, R2={per_dim['r2']:.6f}"
        )


if __name__ == "__main__":
    main()
