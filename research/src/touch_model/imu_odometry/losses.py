from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_loss_fn(loss_name: str, smooth_l1_beta: float = 1.0, reduction: str = "mean") -> nn.Module:
    normalized = loss_name.lower()
    if normalized == "smooth_l1":
        return nn.SmoothL1Loss(beta=smooth_l1_beta, reduction=reduction)
    if normalized == "l1":
        return nn.L1Loss(reduction=reduction)
    if normalized == "mse":
        return nn.MSELoss(reduction=reduction)
    raise ValueError(f"Unsupported loss function: {loss_name}")


def summarize_loss(
    loss_values: Tensor,
    sample_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if loss_values.ndim == 0:
        scalar = loss_values.reshape(1)
        return scalar[0], torch.ones((), dtype=scalar.dtype, device=scalar.device)

    sample_loss = loss_values.reshape(loss_values.shape[0], -1).mean(dim=1)
    if sample_weights is None:
        return sample_loss.sum(), torch.tensor(
            float(sample_loss.numel()), dtype=sample_loss.dtype, device=sample_loss.device
        )

    weights = sample_weights.reshape(-1).to(sample_loss.device, dtype=sample_loss.dtype)
    return (sample_loss * weights).sum(), torch.clamp(weights.sum(), min=1e-8)


def reduce_loss(
    loss_values: Tensor,
    sample_weights: Tensor | None = None,
) -> Tensor:
    numerator, denominator = summarize_loss(loss_values, sample_weights)
    return numerator / denominator


def angular_difference(predictions: Tensor, targets: Tensor) -> Tensor:
    return torch.atan2(torch.sin(predictions - targets), torch.cos(predictions - targets))


def compute_motion_class_weights(
    labels: Tensor,
    balance_mode: str = "none",
) -> Tensor | None:
    normalized = balance_mode.lower()
    if normalized == "none":
        return None
    if normalized != "inverse_frequency":
        raise ValueError(f"Unsupported motion class balance mode: {balance_mode}")

    labels = labels.reshape(-1).float()
    if labels.numel() == 0:
        return None
    positive_fraction = labels.mean()
    negative_fraction = 1.0 - positive_fraction
    if positive_fraction <= 0.0 or negative_fraction <= 0.0:
        return torch.ones_like(labels)

    positive_weight = 0.5 / torch.clamp(positive_fraction, min=1e-8)
    negative_weight = 0.5 / torch.clamp(negative_fraction, min=1e-8)
    return torch.where(labels > 0.5, positive_weight, negative_weight)


def compute_motion_loss_values(
    logits: Tensor,
    labels: Tensor,
    balance_mode: str = "none",
) -> Tensor:
    labels = labels.reshape(-1).to(dtype=logits.dtype, device=logits.device)
    losses = F.binary_cross_entropy_with_logits(
        logits.reshape(-1),
        labels,
        reduction="none",
    )
    class_weights = compute_motion_class_weights(labels, balance_mode=balance_mode)
    if class_weights is not None:
        losses = losses * class_weights.to(device=losses.device, dtype=losses.dtype)
    return losses


def compute_loss_values(
    predictions: Tensor,
    targets: Tensor,
    loss_name: str,
    smooth_l1_beta: float,
    target_mode: str,
    delta_xy_parameterization: str = "cartesian",
    magnitude_loss_weight: float = 1.0,
    magnitude_loss_type: str = "l1",
    direction_loss_weight: float = 1.0,
    direction_loss_min_target_magnitude_m: float = 0.02,
    polar_distance_weight: float = 1.0,
    polar_heading_weight: float = 1.0,
    predicted_log_variance: Tensor | None = None,
    predicted_bias_correction: Tensor | None = None,
    hybrid_bias_regularization: float = 0.0,
    hybrid_log_variance_min: float = -6.0,
    hybrid_log_variance_max: float = 6.0,
) -> Tensor:
    normalized = loss_name.lower()

    def base_loss(pred: Tensor, tgt: Tensor) -> Tensor:
        if normalized == "smooth_l1":
            return F.smooth_l1_loss(pred, tgt, beta=smooth_l1_beta, reduction="none")
        if normalized == "l1":
            return F.l1_loss(pred, tgt, reduction="none")
        if normalized == "mse":
            return F.mse_loss(pred, tgt, reduction="none")
        raise ValueError(f"Unsupported loss function: {loss_name}")

    def magnitude_loss(pred_xy: Tensor, target_xy: Tensor) -> Tensor:
        pred_magnitude = torch.linalg.norm(pred_xy, dim=1)
        target_magnitude = torch.linalg.norm(target_xy, dim=1)
        if magnitude_loss_type == "l1":
            return F.l1_loss(pred_magnitude, target_magnitude, reduction="none")
        if magnitude_loss_type == "smooth_l1":
            return F.smooth_l1_loss(
                pred_magnitude,
                target_magnitude,
                beta=smooth_l1_beta,
                reduction="none",
            )
        raise ValueError(f"Unsupported magnitude loss type: {magnitude_loss_type}")

    if predicted_log_variance is not None:
        if target_mode != "delta_xy" or delta_xy_parameterization != "cartesian":
            raise ValueError(
                "Probabilistic hybrid loss currently supports only cartesian delta_xy targets."
            )
        corrected_predictions = predictions
        if predicted_bias_correction is not None:
            corrected_predictions = corrected_predictions + predicted_bias_correction
        bounded_log_variance = torch.clamp(
            predicted_log_variance,
            min=hybrid_log_variance_min,
            max=hybrid_log_variance_max,
        )
        squared_error = torch.square(corrected_predictions - targets)
        loss_values = 0.5 * (
            torch.exp(-bounded_log_variance) * squared_error
            + bounded_log_variance
            + math.log(2.0 * math.pi)
        )
        if magnitude_loss_weight > 0.0:
            auxiliary_magnitude_loss = (
                magnitude_loss(corrected_predictions, targets) * magnitude_loss_weight
            ).unsqueeze(1)
            loss_values = loss_values + auxiliary_magnitude_loss
        if predicted_bias_correction is not None and hybrid_bias_regularization > 0.0:
            loss_values = loss_values + hybrid_bias_regularization * torch.square(predicted_bias_correction)
        return loss_values

    if target_mode == "polar_xy":
        distance_loss = base_loss(predictions[:, 0], targets[:, 0]) * polar_distance_weight
        heading_delta = angular_difference(predictions[:, 1], targets[:, 1])
        heading_loss = base_loss(heading_delta, torch.zeros_like(heading_delta)) * polar_heading_weight
        return torch.stack([distance_loss, heading_loss], dim=1)

    if target_mode == "delta_xy" and delta_xy_parameterization == "direction_magnitude":
        predicted_xy, magnitude_pred, direction_pred = decode_delta_xy_direction_magnitude(
            predictions
        )
        del predicted_xy
        target_magnitude = torch.linalg.norm(targets, dim=1)
        safe_target_magnitude = torch.clamp(target_magnitude, min=1e-8)
        target_direction = targets / safe_target_magnitude.unsqueeze(1)

        magnitude_loss = base_loss(magnitude_pred, target_magnitude) * magnitude_loss_weight

        direction_mask = (target_magnitude >= direction_loss_min_target_magnitude_m).to(predictions.dtype)
        cosine_similarity = torch.sum(direction_pred * target_direction, dim=1).clamp(-1.0, 1.0)
        direction_loss = (1.0 - cosine_similarity) * direction_loss_weight
        direction_loss = direction_loss * direction_mask
        return torch.stack([magnitude_loss, direction_loss], dim=1)

    loss_values = base_loss(predictions, targets)
    if (
        target_mode == "delta_xy"
        and delta_xy_parameterization == "cartesian"
        and magnitude_loss_weight > 0.0
    ):
        auxiliary_magnitude_loss = (
            magnitude_loss(predictions, targets) * magnitude_loss_weight
        ).unsqueeze(1)
        loss_values = loss_values + auxiliary_magnitude_loss
    return loss_values


def decode_delta_xy_direction_magnitude(
    predictions: Tensor,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    magnitude = F.softplus(predictions[:, 0])
    direction_raw = predictions[:, 1:3]
    direction_norm = torch.linalg.norm(direction_raw, dim=1, keepdim=True)
    direction = direction_raw / torch.clamp(direction_norm, min=epsilon)
    decoded = magnitude.unsqueeze(1) * direction
    return decoded, magnitude, direction


def decode_predictions(
    predictions: Tensor,
    target_mode: str,
    delta_xy_parameterization: str = "cartesian",
    predicted_bias_correction: Tensor | None = None,
) -> Tensor:
    if (
        predicted_bias_correction is not None
        and target_mode == "delta_xy"
        and delta_xy_parameterization == "cartesian"
    ):
        predictions = predictions + predicted_bias_correction
    if target_mode == "delta_xy" and delta_xy_parameterization == "direction_magnitude":
        decoded, _, _ = decode_delta_xy_direction_magnitude(predictions)
        return decoded
    return predictions


def compute_direction_magnitude_metrics(
    predictions_xy: Tensor,
    targets_xy: Tensor,
    min_target_magnitude_m: float,
) -> dict[str, float]:
    pred_magnitude = torch.linalg.norm(predictions_xy, dim=1)
    target_magnitude = torch.linalg.norm(targets_xy, dim=1)
    magnitude_mae = torch.mean(torch.abs(pred_magnitude - target_magnitude))
    eligible_mask = target_magnitude >= min_target_magnitude_m
    if torch.any(eligible_mask):
        pred_dir = predictions_xy[eligible_mask] / torch.clamp(
            torch.linalg.norm(predictions_xy[eligible_mask], dim=1, keepdim=True),
            min=1e-8,
        )
        target_dir = targets_xy[eligible_mask] / torch.clamp(
            torch.linalg.norm(targets_xy[eligible_mask], dim=1, keepdim=True),
            min=1e-8,
        )
        cosine_similarity = torch.sum(pred_dir * target_dir, dim=1).mean()
        eligible_count = int(torch.sum(eligible_mask).item())
    else:
        cosine_similarity = torch.tensor(float("nan"), device=predictions_xy.device)
        eligible_count = 0
    return {
        "magnitude_mae": float(magnitude_mae.item()),
        "mean_direction_cosine_similarity": float(cosine_similarity.item()),
        "num_direction_eligible_samples": eligible_count,
    }


def compute_temporal_consistency_loss(
    decoded_predictions: Tensor,
    batch_metadata: dict[str, Any],
    target_mode: str,
    mode: str = "velocity_l1",
    expected_stride: int | None = None,
) -> Tensor:
    if mode != "velocity_l1":
        raise ValueError(f"Unsupported temporal consistency mode: {mode}")
    if target_mode != "delta_xy":
        return decoded_predictions.new_zeros(())

    session_names = batch_metadata.get("session_name")
    window_starts = batch_metadata.get("window_start")
    durations = batch_metadata.get("target_duration_s")
    if session_names is None or window_starts is None or durations is None:
        return decoded_predictions.new_zeros(())

    num_samples = decoded_predictions.shape[0]
    if num_samples < 2:
        return decoded_predictions.new_zeros(())

    if torch.is_tensor(window_starts):
        window_start_list = [int(value) for value in window_starts.detach().cpu().tolist()]
    else:
        window_start_list = [int(value) for value in window_starts]
    if torch.is_tensor(durations):
        duration_tensor = durations.to(device=decoded_predictions.device, dtype=decoded_predictions.dtype).reshape(-1)
    else:
        duration_tensor = torch.as_tensor(
            durations,
            device=decoded_predictions.device,
            dtype=decoded_predictions.dtype,
        ).reshape(-1)
    session_name_list = list(session_names)

    order = sorted(range(num_samples), key=lambda idx: (str(session_name_list[idx]), window_start_list[idx]))
    order_tensor = torch.as_tensor(order, device=decoded_predictions.device, dtype=torch.long)
    sorted_predictions = decoded_predictions.index_select(0, order_tensor)
    sorted_durations = duration_tensor.index_select(0, order_tensor)
    sorted_session_names = [str(session_name_list[idx]) for idx in order]
    sorted_window_starts = [window_start_list[idx] for idx in order]

    velocities = sorted_predictions / torch.clamp(sorted_durations.unsqueeze(1), min=1e-6)
    pair_losses: list[Tensor] = []
    for idx in range(num_samples - 1):
        if sorted_session_names[idx] != sorted_session_names[idx + 1]:
            continue
        if expected_stride is not None and sorted_window_starts[idx + 1] - sorted_window_starts[idx] != expected_stride:
            continue
        pair_losses.append(torch.mean(torch.abs(velocities[idx + 1] - velocities[idx])))

    if not pair_losses:
        return decoded_predictions.new_zeros(())
    return torch.stack(pair_losses).mean()


def _metadata_tensor(
    values: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(values, device=device, dtype=dtype)


def _surface_loss_values(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_type: str,
    smooth_l1_beta: float,
) -> Tensor:
    if loss_type == "l1":
        return torch.abs(prediction - target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, beta=smooth_l1_beta, reduction="none")
    raise ValueError(f"Unsupported surface anchor loss type: {loss_type}")


def _rollout_positions_from_windows(
    displacements_xy: Tensor,
    measurement_end_times_s: Tensor,
    measurement_durations_s: Tensor,
) -> Tensor:
    num_samples = int(displacements_xy.shape[0])
    positions = torch.zeros_like(displacements_xy)
    if num_samples == 0:
        return positions
    safe_durations = torch.clamp(measurement_durations_s.reshape(-1), min=1e-6)
    positions[0] = displacements_xy[0]
    for idx in range(1, num_samples):
        dt_step = torch.clamp(
            measurement_end_times_s[idx] - measurement_end_times_s[idx - 1],
            min=1e-6,
        )
        velocity_xy = displacements_xy[idx] / safe_durations[idx]
        positions[idx] = positions[idx - 1] + velocity_xy * dt_step
    return positions


def compute_surface_anchor_drift_loss(
    decoded_predictions: Tensor,
    batch_metadata: dict[str, Any],
    target_mode: str,
    *,
    loss_type: str = "l1",
    smooth_l1_beta: float = 1.0,
    expected_stride: int | None = None,
) -> Tensor:
    if target_mode != "delta_xy":
        return decoded_predictions.new_zeros(())

    session_names = batch_metadata.get("session_name")
    end_times = batch_metadata.get("end_time_s")
    durations = batch_metadata.get("target_duration_s")
    window_starts = batch_metadata.get("window_start")
    teacher_start_xyz = batch_metadata.get("teacher_start_xyz")
    teacher_end_xyz = batch_metadata.get("teacher_end_xyz")
    if (
        session_names is None
        or end_times is None
        or durations is None
        or window_starts is None
        or teacher_start_xyz is None
        or teacher_end_xyz is None
    ):
        return decoded_predictions.new_zeros(())

    num_samples = int(decoded_predictions.shape[0])
    if num_samples == 0:
        return decoded_predictions.new_zeros(())

    end_time_tensor = _metadata_tensor(
        end_times,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    duration_tensor = _metadata_tensor(
        durations,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    window_start_tensor = _metadata_tensor(
        window_starts,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_start_tensor = _metadata_tensor(
        teacher_start_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_end_tensor = _metadata_tensor(
        teacher_end_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    session_name_list = [str(item) for item in session_names]
    order = sorted(range(num_samples), key=lambda idx: (session_name_list[idx], float(end_time_tensor[idx].item())))
    order_tensor = torch.as_tensor(order, device=decoded_predictions.device, dtype=torch.long)
    sorted_predictions = decoded_predictions.index_select(0, order_tensor)
    sorted_end_times = end_time_tensor.index_select(0, order_tensor).reshape(-1)
    sorted_durations = duration_tensor.index_select(0, order_tensor).reshape(-1)
    sorted_window_starts = window_start_tensor.index_select(0, order_tensor).reshape(-1)
    sorted_teacher_start_xy = teacher_start_tensor.index_select(0, order_tensor)[:, :2]
    sorted_teacher_end_xy = teacher_end_tensor.index_select(0, order_tensor)[:, :2]
    sorted_session_names = [session_name_list[idx] for idx in order]

    group_losses: list[Tensor] = []
    group_start = 0
    while group_start < num_samples:
        group_end = group_start + 1
        while group_end < num_samples and sorted_session_names[group_end] == sorted_session_names[group_start]:
            group_end += 1
        segment_start = group_start
        while segment_start < group_end:
            segment_end = segment_start + 1
            while segment_end < group_end:
                if expected_stride is not None:
                    step = int(sorted_window_starts[segment_end].item() - sorted_window_starts[segment_end - 1].item())
                    if step != int(expected_stride):
                        break
                segment_end += 1
            predicted_positions_xy = _rollout_positions_from_windows(
                sorted_predictions[segment_start:segment_end],
                sorted_end_times[segment_start:segment_end],
                sorted_durations[segment_start:segment_end],
            )
            anchor_xy = sorted_teacher_start_xy[segment_start]
            true_positions_xy = sorted_teacher_end_xy[segment_start:segment_end] - anchor_xy.unsqueeze(0)
            group_losses.append(
                _surface_loss_values(
                    predicted_positions_xy,
                    true_positions_xy,
                    loss_type=loss_type,
                    smooth_l1_beta=smooth_l1_beta,
                ).mean()
            )
            segment_start = segment_end
        group_start = group_end

    if not group_losses:
        return decoded_predictions.new_zeros(())
    return torch.stack(group_losses).mean()


def compute_surface_slip_loss(
    decoded_predictions: Tensor,
    batch_metadata: dict[str, Any],
    target_mode: str,
    *,
    loss_type: str = "l1",
    smooth_l1_beta: float = 1.0,
    horizon_steps: int = 1,
    expected_stride: int | None = None,
) -> Tensor:
    if target_mode != "delta_xy":
        return decoded_predictions.new_zeros(())

    session_names = batch_metadata.get("session_name")
    end_times = batch_metadata.get("end_time_s")
    durations = batch_metadata.get("target_duration_s")
    window_starts = batch_metadata.get("window_start")
    teacher_start_xyz = batch_metadata.get("teacher_start_xyz")
    teacher_end_xyz = batch_metadata.get("teacher_end_xyz")
    if (
        session_names is None
        or end_times is None
        or durations is None
        or window_starts is None
        or teacher_start_xyz is None
        or teacher_end_xyz is None
    ):
        return decoded_predictions.new_zeros(())

    num_samples = int(decoded_predictions.shape[0])
    if num_samples < 2:
        return decoded_predictions.new_zeros(())

    end_time_tensor = _metadata_tensor(
        end_times,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    duration_tensor = _metadata_tensor(
        durations,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    window_start_tensor = _metadata_tensor(
        window_starts,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_start_tensor = _metadata_tensor(
        teacher_start_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_end_tensor = _metadata_tensor(
        teacher_end_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    session_name_list = [str(item) for item in session_names]
    order = sorted(range(num_samples), key=lambda idx: (session_name_list[idx], float(end_time_tensor[idx].item())))
    order_tensor = torch.as_tensor(order, device=decoded_predictions.device, dtype=torch.long)
    sorted_predictions = decoded_predictions.index_select(0, order_tensor)
    sorted_end_times = end_time_tensor.index_select(0, order_tensor).reshape(-1)
    sorted_durations = duration_tensor.index_select(0, order_tensor).reshape(-1)
    sorted_window_starts = window_start_tensor.index_select(0, order_tensor).reshape(-1)
    sorted_teacher_start_xy = teacher_start_tensor.index_select(0, order_tensor)[:, :2]
    sorted_teacher_end_xy = teacher_end_tensor.index_select(0, order_tensor)[:, :2]
    sorted_session_names = [session_name_list[idx] for idx in order]
    effective_horizon = max(int(horizon_steps), 1)

    group_losses: list[Tensor] = []
    group_start = 0
    while group_start < num_samples:
        group_end = group_start + 1
        while group_end < num_samples and sorted_session_names[group_end] == sorted_session_names[group_start]:
            group_end += 1
        segment_start = group_start
        while segment_start < group_end:
            segment_end = segment_start + 1
            while segment_end < group_end:
                if expected_stride is not None:
                    step = int(sorted_window_starts[segment_end].item() - sorted_window_starts[segment_end - 1].item())
                    if step != int(expected_stride):
                        break
                segment_end += 1
            segment_length = segment_end - segment_start
            if segment_length > effective_horizon:
                predicted_positions_xy = _rollout_positions_from_windows(
                    sorted_predictions[segment_start:segment_end],
                    sorted_end_times[segment_start:segment_end],
                    sorted_durations[segment_start:segment_end],
                )
                anchor_xy = sorted_teacher_start_xy[segment_start]
                true_positions_xy = sorted_teacher_end_xy[segment_start:segment_end] - anchor_xy.unsqueeze(0)
                predicted_slip_xy = predicted_positions_xy[effective_horizon:] - predicted_positions_xy[:-effective_horizon]
                true_slip_xy = true_positions_xy[effective_horizon:] - true_positions_xy[:-effective_horizon]
                group_losses.append(
                    _surface_loss_values(
                        predicted_slip_xy,
                        true_slip_xy,
                        loss_type=loss_type,
                        smooth_l1_beta=smooth_l1_beta,
                    ).mean()
                )
            segment_start = segment_end
        group_start = group_end

    if not group_losses:
        return decoded_predictions.new_zeros(())
    return torch.stack(group_losses).mean()


def compute_surface_anchor_drift_loss_rollout(
    decoded_predictions: Tensor,
    rollout_metadata: dict[str, Any],
    target_mode: str,
    *,
    loss_type: str = "l1",
    smooth_l1_beta: float = 1.0,
) -> Tensor:
    if target_mode != "delta_xy" or decoded_predictions.ndim != 3:
        return decoded_predictions.new_zeros(())

    end_times = rollout_metadata.get("end_time_s")
    durations = rollout_metadata.get("target_duration_s")
    teacher_start_xyz = rollout_metadata.get("teacher_start_xyz")
    teacher_end_xyz = rollout_metadata.get("teacher_end_xyz")
    if end_times is None or durations is None or teacher_start_xyz is None or teacher_end_xyz is None:
        return decoded_predictions.new_zeros(())

    end_time_tensor = _metadata_tensor(
        end_times,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    duration_tensor = _metadata_tensor(
        durations,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_start_tensor = _metadata_tensor(
        teacher_start_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_end_tensor = _metadata_tensor(
        teacher_end_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )

    predicted_positions = torch.stack(
        [
            _rollout_positions_from_windows(
                decoded_predictions[rollout_index],
                end_time_tensor[rollout_index].reshape(-1),
                duration_tensor[rollout_index].reshape(-1),
            )
            for rollout_index in range(decoded_predictions.shape[0])
        ],
        dim=0,
    )
    anchor_xy = teacher_start_tensor[:, 0, :2]
    true_positions = teacher_end_tensor[:, :, :2] - anchor_xy.unsqueeze(1)
    return _surface_loss_values(
        predicted_positions,
        true_positions,
        loss_type=loss_type,
        smooth_l1_beta=smooth_l1_beta,
    ).mean()


def compute_surface_slip_loss_rollout(
    decoded_predictions: Tensor,
    rollout_metadata: dict[str, Any],
    target_mode: str,
    *,
    loss_type: str = "l1",
    smooth_l1_beta: float = 1.0,
    horizon_steps: int = 1,
) -> Tensor:
    if target_mode != "delta_xy" or decoded_predictions.ndim != 3:
        return decoded_predictions.new_zeros(())

    end_times = rollout_metadata.get("end_time_s")
    durations = rollout_metadata.get("target_duration_s")
    teacher_start_xyz = rollout_metadata.get("teacher_start_xyz")
    teacher_end_xyz = rollout_metadata.get("teacher_end_xyz")
    if end_times is None or durations is None or teacher_start_xyz is None or teacher_end_xyz is None:
        return decoded_predictions.new_zeros(())

    effective_horizon = max(int(horizon_steps), 1)
    if decoded_predictions.shape[1] <= effective_horizon:
        return decoded_predictions.new_zeros(())

    end_time_tensor = _metadata_tensor(
        end_times,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    duration_tensor = _metadata_tensor(
        durations,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_start_tensor = _metadata_tensor(
        teacher_start_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )
    teacher_end_tensor = _metadata_tensor(
        teacher_end_xyz,
        device=decoded_predictions.device,
        dtype=decoded_predictions.dtype,
    )

    predicted_positions = torch.stack(
        [
            _rollout_positions_from_windows(
                decoded_predictions[rollout_index],
                end_time_tensor[rollout_index].reshape(-1),
                duration_tensor[rollout_index].reshape(-1),
            )
            for rollout_index in range(decoded_predictions.shape[0])
        ],
        dim=0,
    )
    anchor_xy = teacher_start_tensor[:, 0, :2]
    true_positions = teacher_end_tensor[:, :, :2] - anchor_xy.unsqueeze(1)
    predicted_slip = predicted_positions[:, effective_horizon:] - predicted_positions[:, :-effective_horizon]
    true_slip = true_positions[:, effective_horizon:] - true_positions[:, :-effective_horizon]
    return _surface_loss_values(
        predicted_slip,
        true_slip,
        loss_type=loss_type,
        smooth_l1_beta=smooth_l1_beta,
    ).mean()
