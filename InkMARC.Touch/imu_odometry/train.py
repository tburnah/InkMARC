"""
Train a pure sensor-input inertial odometry baseline from CSV sessions.

Expected CSV columns
- Required teacher columns: `plane_x_m`, `plane_y_m`, `plane_z_m`
- Default validity flag: `tag_pose_valid`
- Default base sensor inputs: `accel_x/y/z`, `gyro_x/y/z`, `rot_x/y/z/w`
- Optional sensor groups: gravity (`gravity_x/y/z` or `grav_x/y/z`),
  linear acceleration (`linear_accel_x/y/z` or `lin_x/y/z`), and
  magnetometer (`magnetometer_x/y/z` or `mag_x/y/z`)

How windows are formed
- Each CSV file is treated as one session.
- Rows are sorted by timestamp.
- Sliding windows of `window_size` rows are created with stride `stride`.
- The input is the full sensor sequence in that window.
- The target uses the first and last valid teacher rows inside the window.

Default target computation
- `delta_x = plane_x_m[last_valid] - plane_x_m[first_valid]`
- `delta_y = plane_y_m[last_valid] - plane_y_m[first_valid]`
- `delta_z = plane_z_m[last_valid] - plane_z_m[first_valid]`

Example run with your current logs
```
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python train.py \
  --train-files /home/t/tburnah/inkmarc/dtm_dtm_1774674722341.csv \
  --val-files /home/t/tburnah/inkmarc/dtm_dtm_1774675309496.csv \
  --window-size 60 \
  --stride 10 \
  --batch-size 128 \
  --epochs 30 \
  --output-dir /home/t/tburnah/TouchModel/runs/imu_odometry
```

If your logs use different column names, update the alias map in `config.py`.
"""

from __future__ import annotations

import glob
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from config import TrainingConfig, build_train_argparser, train_config_from_args
from dataset import (
    SensorWindowDataset,
    SurfaceRolloutDataset,
    collate_sensor_windows,
    collate_surface_rollouts,
)
from evaluate import (
    calibrate_empirical_measurement_covariance,
    evaluate_classical_double_integration,
    evaluate_model,
)
from evaluate import (
    evaluate_previous_window_persistence_baseline,
    evaluate_tabular_linear_regression_baseline,
    evaluate_train_mean_baseline,
    evaluate_zero_motion_baseline,
)
from export_onnx import export_checkpoint_to_onnx
from losses import (
    compute_loss_values,
    compute_motion_loss_values,
    compute_surface_anchor_drift_loss,
    compute_surface_anchor_drift_loss_rollout,
    compute_surface_slip_loss,
    compute_surface_slip_loss_rollout,
    compute_temporal_consistency_loss,
    decode_predictions,
    reduce_loss,
)
from model import build_model
from utils import (
    NormalizationStats,
    count_parameters,
    file_list_to_strings,
    flatten_metric_values,
    make_run_dir,
    plot_training_curves,
    resolve_device,
    save_checkpoint,
    save_history_csv,
    save_json,
    save_normalization,
    seed_everything,
    write_predictions_csv,
)


def count_motion_windows(dataset: SensorWindowDataset, threshold_m: float) -> dict[str, int]:
    moving = sum(1 for sample in dataset.samples if sample.target_magnitude >= threshold_m)
    return {"moving": moving, "static": len(dataset.samples) - moving}


class ContiguousRolloutBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SensorWindowDataset,
        batch_size: int,
        rollout_num_windows: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.rollout_num_windows = max(int(rollout_num_windows), 1)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._chunks = self._build_chunks()

    def _build_chunks(self) -> list[list[int]]:
        session_indices: dict[int, list[int]] = {}
        for index, sample in enumerate(self.dataset.samples):
            session_indices.setdefault(sample.session_idx, []).append(index)
        chunks: list[list[int]] = []
        for indices in session_indices.values():
            for start in range(0, len(indices), self.rollout_num_windows):
                chunk = indices[start : start + self.rollout_num_windows]
                if chunk:
                    chunks.append(chunk)
        return chunks

    def __iter__(self):
        chunks = [list(chunk) for chunk in self._chunks]
        if self.shuffle:
            random.shuffle(chunks)
        batch: list[int] = []
        for chunk in chunks:
            if batch and len(batch) + len(chunk) > self.batch_size:
                yield batch
                batch = []
            batch.extend(chunk)
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if not self._chunks:
            return 0
        total_indices = sum(len(chunk) for chunk in self._chunks)
        if self.drop_last:
            return total_indices // self.batch_size
        return math.ceil(total_indices / self.batch_size)


def _build_standard_loader(
    dataset: SensorWindowDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sensor_windows,
    )


def print_split_summary(split_name: str, metrics: dict[str, Any]) -> None:
    print(f"{split_name} vector MAE: {metrics['vector_mae']:.6f}")
    print(
        f"{split_name} average true displacement magnitude: "
        f"{metrics['average_true_displacement_magnitude']:.6f}"
    )
    print(
        f"{split_name} zero-motion baseline vector MAE: "
        f"{metrics['zero_motion_baseline_vector_mae']:.6f}"
    )
    print(
        f"{split_name} relative error percentage: "
        f"{metrics['relative_error_percentage']:.2f}%"
    )
    print(f"{split_name} gated vector MAE: {metrics['gated_vector_mae']:.6f}")
    print(
        f"{split_name} gated relative error percentage: "
        f"{metrics['gated_relative_error_percentage']:.2f}%"
    )
    print(f"{split_name} ungated vector MAE: {metrics['ungated_vector_mae']:.6f}")
    print(
        f"{split_name} ungated relative error percentage: "
        f"{metrics['ungated_relative_error_percentage']:.2f}%"
    )
    motion_counts = metrics.get("motion_label_counts", {})
    if motion_counts:
        print(
            f"{split_name} motion label counts: "
            f"moving={motion_counts['moving']}, static={motion_counts['static']}"
        )
    if "motion_classification" in metrics:
        motion = metrics["motion_classification"]
        print(
            f"{split_name} motion classification: "
            f"acc={motion['accuracy']:.4f}, precision={motion['precision']:.4f}, "
            f"recall={motion['recall']:.4f}, f1={motion['f1']:.4f}, "
            f"tp={motion['tp']}, fp={motion['fp']}, tn={motion['tn']}, fn={motion['fn']}, "
            f"positive_rate={motion['positive_rate']:.4f}, "
            f"predicted_positive_rate={motion['predicted_positive_rate']:.4f}"
        )
    for label_name, split_metrics in metrics.get("regression_by_motion_label", {}).items():
        if split_metrics.get("num_samples", 0) == 0:
            print(f"{split_name} regression on {label_name}-only windows: samples=0")
            continue
        print(
            f"{split_name} regression on {label_name}-only windows: "
            f"vector_mae={split_metrics['vector_mae']:.6f}, "
            f"relative={split_metrics['relative_error_percentage']:.2f}% "
            f"over {split_metrics['num_samples']} windows"
        )
    if "direction_magnitude" in metrics:
        direction = metrics["direction_magnitude"]
        print(
            f"{split_name} direction+magnitude: "
            f"magnitude_mae={direction['magnitude_mae']:.6f}, "
            f"mean_direction_cosine_similarity={direction['mean_direction_cosine_similarity']:.6f}, "
            f"eligible_samples={direction['num_direction_eligible_samples']}"
        )
    for label, moving in metrics.get("moving_only_thresholds", {}).items():
        if moving.get("num_samples", 0) == 0:
            print(f"{split_name} moving-only {label}: samples=0")
            continue
        print(
            f"{split_name} moving-only {label}: vector_mae={moving['vector_mae']:.6f}, "
            f"baseline={moving['zero_motion_baseline_vector_mae']:.6f}, "
            f"relative={moving['relative_error_percentage']:.2f}% "
            f"over {moving['num_samples']} windows"
        )
    for bucket_name, bucket_values in metrics.get("motion_buckets", {}).items():
        if bucket_values.get("num_samples", 0) == 0:
            print(f"{split_name} {bucket_name}: samples=0")
            continue
        print(
            f"{split_name} {bucket_name}: samples={bucket_values['num_samples']}, "
            f"vector_mae={bucket_values['vector_mae']:.6f}, "
            f"baseline={bucket_values['zero_motion_baseline_vector_mae']:.6f}, "
            f"relative={bucket_values['relative_error_percentage']:.2f}%"
        )
    oracle = metrics.get("oracle_motion_gate")
    if oracle is not None:
        print(
            f"{split_name} oracle motion gate: "
            f"vector_mae={oracle['vector_mae']:.6f}, "
            f"relative={oracle['relative_error_percentage']:.2f}%"
        )
    if "classical_double_integration" in metrics:
        baseline = metrics["classical_double_integration"]
        print(
            f"{split_name} classical double integration: "
            f"vector_mae={baseline['vector_mae']:.6f}, "
            f"baseline={baseline['zero_motion_baseline_vector_mae']:.6f}, "
            f"relative={baseline['relative_error_percentage']:.2f}%"
        )
        print(
            f"{split_name} learned vs classical delta: "
            f"{metrics['vector_mae'] - baseline['vector_mae']:+.6f}"
        )
        for label, moving in baseline.get("moving_only_thresholds", {}).items():
            if moving.get("num_samples", 0) == 0:
                print(f"{split_name} classical moving-only {label}: samples=0")
                continue
            learned = metrics.get("moving_only_thresholds", {}).get(label)
            delta_text = ""
            if learned is not None and learned.get("num_samples", 0) > 0:
                delta_text = f", learned_vs_classical={learned['vector_mae'] - moving['vector_mae']:+.6f}"
            print(
                f"{split_name} classical moving-only {label}: vector_mae={moving['vector_mae']:.6f}, "
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
            f"{split_name} {baseline_name}: "
            f"vector_mae={baseline['vector_mae']:.6f}, "
            f"relative={baseline['relative_error_percentage']:.2f}%"
        )
    sequential_rollout = metrics.get("sequential_rollout", {})
    for rollout_name in ["direct_model", "classical_baseline", "planar_ekf", "plane_relative_tracker"]:
        rollout_metrics = sequential_rollout.get(rollout_name)
        if rollout_metrics is None:
            continue
        print(
            f"{split_name} sequential {rollout_name}: "
            f"position_mae={rollout_metrics['position_mae_m']:.6f}, "
            f"position_rmse={rollout_metrics['position_rmse_m']:.6f}, "
            f"final_position_error_mean={rollout_metrics['final_position_error_m_mean']:.6f}"
        )
        if rollout_name == "planar_ekf":
            covariance_diagnostics = rollout_metrics.get("measurement_covariance_diagnostics")
            if covariance_diagnostics is not None:
                print(
                    f"{split_name} planar_ekf covariance: "
                    f"update_stride={int(covariance_diagnostics['update_stride'])}, "
                    f"update_fraction={covariance_diagnostics['measurement_update_fraction']:.4f}, "
                    f"floor_hit_x={covariance_diagnostics['covariance_floor_hit_fraction_x']:.4f}, "
                    f"floor_hit_y={covariance_diagnostics['covariance_floor_hit_fraction_y']:.4f}"
                )
        surface_stability = rollout_metrics.get("surface_stability")
        if surface_stability is not None:
            print(
                f"{split_name} {rollout_name} surface: "
                f"anchor_drift_mae={surface_stability['surface_anchor_drift_mae_m']:.6f}, "
                f"slip_mae={surface_stability['surface_slip_mae_m']:.6f}, "
                f"corner_drift_mae={surface_stability['corner_drift_mae_m']:.6f}"
            )
        if rollout_name == "plane_relative_tracker":
            tracker_diagnostics = rollout_metrics.get("tracker_diagnostics")
            if tracker_diagnostics is not None:
                print(
                    f"{split_name} plane_relative_tracker diagnostics: "
                    f"position_blend={tracker_diagnostics['mean_position_blend']:.4f}, "
                    f"velocity_blend={tracker_diagnostics['mean_velocity_blend']:.4f}, "
                    f"mean_abs_height_offset={tracker_diagnostics['mean_abs_height_offset_m']:.6f}"
                )
    if "hybrid_comparison" in metrics:
        hybrid_comparison = metrics["hybrid_comparison"]
        if "planar_ekf_vs_direct_position_mae_m_delta" in hybrid_comparison:
            print(
                f"{split_name} planar_ekf vs direct rollout: "
                f"{hybrid_comparison['planar_ekf_vs_direct_position_mae_m_delta']:+.6f}"
            )
        if "planar_ekf_vs_classical_position_mae_m_delta" in hybrid_comparison:
            print(
                f"{split_name} planar_ekf vs classical rollout: "
                f"{hybrid_comparison['planar_ekf_vs_classical_position_mae_m_delta']:+.6f}"
            )
        if "plane_relative_tracker_vs_direct_position_mae_m_delta" in hybrid_comparison:
            print(
                f"{split_name} plane_relative_tracker vs direct rollout: "
                f"{hybrid_comparison['plane_relative_tracker_vs_direct_position_mae_m_delta']:+.6f}"
            )
        if "plane_relative_tracker_vs_classical_position_mae_m_delta" in hybrid_comparison:
            print(
                f"{split_name} plane_relative_tracker vs classical rollout: "
                f"{hybrid_comparison['plane_relative_tracker_vs_classical_position_mae_m_delta']:+.6f}"
            )
    if "hybrid_uncertainty" in metrics:
        uncertainty = metrics["hybrid_uncertainty"]
        print(
            f"{split_name} hybrid uncertainty: "
            f"mean_sigma_x={uncertainty['mean_sigma_x_m']:.6f}, "
            f"mean_sigma_y={uncertainty['mean_sigma_y_m']:.6f}, "
            f"mean_abs_residual_x={uncertainty['mean_abs_residual_x_m']:.6f}, "
            f"mean_abs_residual_y={uncertainty['mean_abs_residual_y_m']:.6f}, "
            f"corr_x={uncertainty['std_abs_residual_correlation_x']:.6f}, "
            f"corr_y={uncertainty['std_abs_residual_correlation_y']:.6f}, "
            f"nll_mean={uncertainty['nll_mean']:.6f}"
        )
    magnitude_diagnostics = metrics.get("magnitude_diagnostics")
    if magnitude_diagnostics is not None:
        print(
            f"{split_name} magnitude: "
            f"mean_true={magnitude_diagnostics['mean_true_magnitude_m']:.6f}, "
            f"mean_pred={magnitude_diagnostics['mean_predicted_magnitude_m']:.6f}, "
            f"median_true={magnitude_diagnostics['median_true_magnitude_m']:.6f}, "
            f"median_pred={magnitude_diagnostics['median_predicted_magnitude_m']:.6f}, "
            f"ratio={magnitude_diagnostics['mean_predicted_to_true_ratio']:.6f}, "
            f"corr={magnitude_diagnostics['predicted_true_magnitude_correlation']:.6f}"
        )
    if "hybrid_bias_correction" in metrics:
        bias_metrics = metrics["hybrid_bias_correction"]
        print(
            f"{split_name} hybrid bias correction: "
            f"mean_abs_x={bias_metrics['mean_abs_bias_x_m']:.6f}, "
            f"mean_abs_y={bias_metrics['mean_abs_bias_y_m']:.6f}"
        )
    timing = metrics.get("dataset_timing_diagnostics")
    if timing is not None:
        print(
            f"{split_name} timing: duration_mean={timing['actual_window_duration_s_mean']:.3f}s, "
            f"duration_std={timing['actual_window_duration_s_std']:.3f}s, "
            f"duration_min={timing['actual_window_duration_s_min']:.3f}s, "
            f"duration_max={timing['actual_window_duration_s_max']:.3f}s, "
            f"zero_dt_fraction_mean={timing['zero_dt_fraction_mean']:.3f}, "
            f"nonzero_dt_count_mean={timing['nonzero_dt_count_mean']:.2f}"
        )


def resolve_file_splits(config: TrainingConfig) -> dict[str, list[str]]:
    if config.train_files or config.val_files or config.test_files:
        splits = {
            "train": file_list_to_strings(config.train_files),
            "val": file_list_to_strings(config.val_files),
            "test": file_list_to_strings(config.test_files),
        }
    else:
        if config.data_dir is None:
            raise ValueError("Provide --data-dir for automatic splitting, or explicit --train-files/--val-files.")
        pattern = str(Path(config.data_dir) / config.glob_pattern)
        files = sorted(glob.glob(pattern))
        if not files:
            raise ValueError(f"No CSV files matched '{pattern}'.")
        if len(files) < 2:
            raise ValueError("At least two sessions are required to produce train/val splits without leakage.")

        rng = random.Random(config.split_seed)
        rng.shuffle(files)

        train_count = max(1, int(round(len(files) * config.train_ratio)))
        val_count = max(1, int(round(len(files) * config.val_ratio)))
        if train_count + val_count > len(files):
            val_count = max(1, len(files) - train_count)
        test_count = max(0, len(files) - train_count - val_count)

        train_files = files[:train_count]
        val_files = files[train_count : train_count + val_count]
        test_files = files[train_count + val_count : train_count + val_count + test_count]
        if not val_files:
            raise ValueError("Automatic split produced an empty validation split.")
        splits = {"train": train_files, "val": val_files, "test": test_files}

    if not splits["train"]:
        raise ValueError("Training split is empty.")
    if not splits["val"]:
        raise ValueError("Validation split is empty. Provide --val-files or more sessions.")
    return {name: [str(Path(path).resolve()) for path in paths] for name, paths in splits.items()}


def create_dataloaders(
    config: TrainingConfig,
    splits: dict[str, list[str]],
) -> tuple[
    SensorWindowDataset,
    SensorWindowDataset,
    SensorWindowDataset | None,
    DataLoader,
    DataLoader,
    DataLoader | None,
]:
    train_dataset = SensorWindowDataset(
        csv_files=splits["train"],
        feature_names=config.resolved_features,
        target_mode=config.target_mode,
        window_size=config.window_size,
        stride=config.stride,
        window_mode=config.window_mode,
        window_duration_s=config.window_duration_s,
        window_stride_s=config.window_stride_s,
        required_valid_flags=config.required_valid_flags,
        sort_column_candidates=config.sort_columns,
        quaternion_frame_mode=config.quaternion_frame_mode,
        loss_weight_mode=config.loss_weight_mode,
        loss_weight_reference_m=config.loss_weight_reference_m,
        loss_weight_min=config.loss_weight_min,
        loss_weight_max=config.loss_weight_max,
        normalization=None,
        fit_normalization=True,
        normalize_targets=config.normalize_targets,
        min_valid_label_points=config.min_valid_label_points,
        min_target_magnitude_m=config.min_target_magnitude_m,
        use_classical_residual_target=config.use_classical_residual_target,
        classical_residual_mode=config.classical_residual_mode,
    )
    normalization = train_dataset.normalization
    val_dataset = SensorWindowDataset(
        csv_files=splits["val"],
        feature_names=config.resolved_features,
        target_mode=config.target_mode,
        window_size=config.window_size,
        stride=config.stride,
        window_mode=config.window_mode,
        window_duration_s=config.window_duration_s,
        window_stride_s=config.window_stride_s,
        required_valid_flags=config.required_valid_flags,
        sort_column_candidates=config.sort_columns,
        quaternion_frame_mode=config.quaternion_frame_mode,
        loss_weight_mode=config.loss_weight_mode,
        loss_weight_reference_m=config.loss_weight_reference_m,
        loss_weight_min=config.loss_weight_min,
        loss_weight_max=config.loss_weight_max,
        normalization=normalization,
        fit_normalization=False,
        normalize_targets=config.normalize_targets,
        min_valid_label_points=config.min_valid_label_points,
        min_target_magnitude_m=0.0,
        use_classical_residual_target=config.use_classical_residual_target,
        classical_residual_mode=config.classical_residual_mode,
    )
    test_dataset = None
    if splits["test"]:
        test_dataset = SensorWindowDataset(
            csv_files=splits["test"],
            feature_names=config.resolved_features,
            target_mode=config.target_mode,
            window_size=config.window_size,
            stride=config.stride,
            window_mode=config.window_mode,
            window_duration_s=config.window_duration_s,
            window_stride_s=config.window_stride_s,
            required_valid_flags=config.required_valid_flags,
            sort_column_candidates=config.sort_columns,
            quaternion_frame_mode=config.quaternion_frame_mode,
            loss_weight_mode=config.loss_weight_mode,
            loss_weight_reference_m=config.loss_weight_reference_m,
            loss_weight_min=config.loss_weight_min,
            loss_weight_max=config.loss_weight_max,
            normalization=normalization,
            fit_normalization=False,
            normalize_targets=config.normalize_targets,
            min_valid_label_points=config.min_valid_label_points,
            min_target_magnitude_m=0.0,
            use_classical_residual_target=config.use_classical_residual_target,
            classical_residual_mode=config.classical_residual_mode,
        )

    use_surface_rollout_batches = (
        (config.surface_anchor_loss_weight > 0.0 or config.surface_slip_loss_weight > 0.0)
        and config.surface_rollout_num_windows > 1
    )
    if use_surface_rollout_batches:
        train_rollout_dataset = SurfaceRolloutDataset(
            base_dataset=train_dataset,
            rollout_num_windows=config.surface_rollout_num_windows,
        )
        rollout_batch_size = max(config.batch_size // config.surface_rollout_num_windows, 1)
        train_loader = DataLoader(
            train_rollout_dataset,
            batch_size=rollout_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_surface_rollouts,
        )
    else:
        train_loader = _build_standard_loader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
        )
    val_loader = _build_standard_loader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = _build_standard_loader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def choose_monitor_value(metrics: dict[str, Any], monitor_metric: str) -> float:
    if monitor_metric in metrics:
        return float(metrics[monitor_metric])
    if "." in monitor_metric:
        current: Any = metrics
        for part in monitor_metric.split("."):
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            return float(current)
    raise KeyError(
        f"Metric '{monitor_metric}' not found. Available top-level keys: {sorted(metrics.keys())}"
    )


def monitor_requires_rollout_metrics(config: TrainingConfig) -> bool:
    if config.target_mode != "delta_xy":
        return False
    rollout_prefixes = (
        "sequential_rollout.",
        "hybrid_comparison.",
    )
    return config.monitor_metric.startswith(rollout_prefixes)


def build_optimizer(config: TrainingConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    total_loss_count = 0
    regression_loss_sum = 0.0
    regression_loss_count = 0
    motion_cls_loss_sum = 0.0
    motion_cls_loss_count = 0
    temporal_consistency_loss_sum = 0.0
    temporal_consistency_loss_count = 0
    surface_anchor_loss_sum = 0.0
    surface_anchor_loss_count = 0
    surface_slip_loss_sum = 0.0
    surface_slip_loss_count = 0
    for batch_inputs, batch_targets, batch_metadata in dataloader:
        rollout_batch_metadata = None
        flat_metadata = batch_metadata
        if batch_inputs.ndim == 4:
            rollout_batch_metadata = batch_metadata
            flat_metadata = batch_metadata["flat_metadata"]
            batch_size, rollout_num_windows, num_channels, max_sequence_length = batch_inputs.shape
            batch_inputs = batch_inputs.reshape(
                batch_size * rollout_num_windows,
                num_channels,
                max_sequence_length,
            )
            batch_targets = batch_targets.reshape(batch_size * rollout_num_windows, batch_targets.shape[-1])
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        sequence_lengths = flat_metadata.get("sequence_length")
        if torch.is_tensor(sequence_lengths):
            sequence_lengths = sequence_lengths.reshape(-1).to(device)
        batch_weights = flat_metadata.get("loss_weight")
        if torch.is_tensor(batch_weights):
            batch_weights = batch_weights.reshape(-1).to(device)

        optimizer.zero_grad(set_to_none=True)
        model_outputs = model(batch_inputs, lengths=sequence_lengths)
        if isinstance(model_outputs, dict):
            predictions = model_outputs["regression"]
            motion_logits = model_outputs.get("motion_logits")
            predicted_log_variance = model_outputs.get("log_variance")
            predicted_bias_correction = model_outputs.get("bias_correction")
        else:
            predictions = model_outputs
            motion_logits = None
            predicted_log_variance = None
            predicted_bias_correction = None
        loss_values = compute_loss_values(
            predictions,
            batch_targets,
            loss_name=config.loss_name,
            smooth_l1_beta=config.smooth_l1_beta,
            target_mode=config.target_mode,
            delta_xy_parameterization=config.delta_xy_parameterization,
            magnitude_loss_weight=config.magnitude_loss_weight,
            magnitude_loss_type=config.magnitude_loss_type,
            direction_loss_weight=config.direction_loss_weight,
            direction_loss_min_target_magnitude_m=config.direction_loss_min_target_magnitude_m,
            polar_distance_weight=config.polar_distance_weight,
            polar_heading_weight=config.polar_heading_weight,
            predicted_log_variance=predicted_log_variance,
            predicted_bias_correction=predicted_bias_correction,
            hybrid_bias_regularization=config.hybrid_bias_regularization,
            hybrid_log_variance_min=config.hybrid_log_variance_min,
            hybrid_log_variance_max=config.hybrid_log_variance_max,
        )
        regression_loss = reduce_loss(loss_values, batch_weights)
        loss = config.motion_reg_loss_weight * regression_loss
        if config.motion_head_enabled and motion_logits is not None:
            target_magnitudes = flat_metadata.get("target_magnitude")
            if target_magnitudes is None:
                raise ValueError("Motion-head training requires target_magnitude metadata from the dataset.")
            if not torch.is_tensor(target_magnitudes):
                target_magnitudes = torch.as_tensor(target_magnitudes, dtype=batch_targets.dtype)
            target_magnitudes = target_magnitudes.reshape(-1).to(device=device, dtype=batch_targets.dtype)
            motion_labels = (
                target_magnitudes >= config.motion_gate_threshold_m
            ).to(dtype=batch_targets.dtype)
            motion_loss_values = compute_motion_loss_values(
                motion_logits.view_as(motion_labels),
                motion_labels,
                balance_mode=config.motion_class_balance_mode,
            )
            motion_loss = reduce_loss(motion_loss_values, batch_weights)
            loss = loss + config.motion_cls_loss_weight * motion_loss
            motion_cls_loss_sum += float(motion_loss.item())
            motion_cls_loss_count += 1
        decoded_predictions = None
        decoded_predictions_rollout = None
        if (
            config.temporal_consistency_loss_weight > 0.0
            or config.surface_anchor_loss_weight > 0.0
            or config.surface_slip_loss_weight > 0.0
        ):
            decoded_predictions = decode_predictions(
                predictions,
                target_mode=config.target_mode,
                delta_xy_parameterization=config.delta_xy_parameterization,
                predicted_bias_correction=predicted_bias_correction,
            )
            if rollout_batch_metadata is not None:
                decoded_predictions_rollout = decoded_predictions.reshape(
                    rollout_batch_metadata["sequence_length"].shape[0],
                    rollout_batch_metadata["sequence_length"].shape[1],
                    decoded_predictions.shape[-1],
                )
        if config.temporal_consistency_loss_weight > 0.0:
            temporal_consistency_loss = compute_temporal_consistency_loss(
                decoded_predictions=decoded_predictions,
                batch_metadata=flat_metadata,
                target_mode=config.target_mode,
                mode=config.temporal_consistency_mode,
                expected_stride=config.stride,
            )
            loss = loss + config.temporal_consistency_loss_weight * temporal_consistency_loss
            temporal_consistency_loss_sum += float(temporal_consistency_loss.item())
            temporal_consistency_loss_count += 1
        if config.surface_anchor_loss_weight > 0.0:
            if rollout_batch_metadata is not None and decoded_predictions_rollout is not None:
                surface_anchor_loss = compute_surface_anchor_drift_loss_rollout(
                    decoded_predictions=decoded_predictions_rollout,
                    rollout_metadata=rollout_batch_metadata,
                    target_mode=config.target_mode,
                    loss_type=config.surface_anchor_loss_type,
                    smooth_l1_beta=config.smooth_l1_beta,
                )
            else:
                surface_anchor_loss = compute_surface_anchor_drift_loss(
                    decoded_predictions=decoded_predictions,
                    batch_metadata=flat_metadata,
                    target_mode=config.target_mode,
                    loss_type=config.surface_anchor_loss_type,
                    smooth_l1_beta=config.smooth_l1_beta,
                    expected_stride=config.stride,
                )
            loss = loss + config.surface_anchor_loss_weight * surface_anchor_loss
            surface_anchor_loss_sum += float(surface_anchor_loss.item())
            surface_anchor_loss_count += 1
        if config.surface_slip_loss_weight > 0.0:
            if rollout_batch_metadata is not None and decoded_predictions_rollout is not None:
                surface_slip_loss = compute_surface_slip_loss_rollout(
                    decoded_predictions=decoded_predictions_rollout,
                    rollout_metadata=rollout_batch_metadata,
                    target_mode=config.target_mode,
                    loss_type=config.surface_anchor_loss_type,
                    smooth_l1_beta=config.smooth_l1_beta,
                    horizon_steps=config.plane_tracker_short_horizon_steps,
                )
            else:
                surface_slip_loss = compute_surface_slip_loss(
                    decoded_predictions=decoded_predictions,
                    batch_metadata=flat_metadata,
                    target_mode=config.target_mode,
                    loss_type=config.surface_anchor_loss_type,
                    smooth_l1_beta=config.smooth_l1_beta,
                    horizon_steps=config.plane_tracker_short_horizon_steps,
                    expected_stride=config.stride,
                )
            loss = loss + config.surface_slip_loss_weight * surface_slip_loss
            surface_slip_loss_sum += float(surface_slip_loss.item())
            surface_slip_loss_count += 1
        loss.backward()
        optimizer.step()
        total_loss_sum += float(loss.item())
        total_loss_count += 1
        regression_loss_sum += float(regression_loss.item())
        regression_loss_count += 1

    return {
        "total_loss": total_loss_sum / max(total_loss_count, 1),
        "regression_loss": regression_loss_sum / max(regression_loss_count, 1),
        "motion_cls_loss": (
            motion_cls_loss_sum / motion_cls_loss_count if motion_cls_loss_count > 0 else 0.0
        ),
        "temporal_consistency_loss": (
            temporal_consistency_loss_sum / temporal_consistency_loss_count
            if temporal_consistency_loss_count > 0
            else 0.0
        ),
        "surface_anchor_loss": (
            surface_anchor_loss_sum / surface_anchor_loss_count
            if surface_anchor_loss_count > 0
            else 0.0
        ),
        "surface_slip_loss": (
            surface_slip_loss_sum / surface_slip_loss_count
            if surface_slip_loss_count > 0
            else 0.0
        ),
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    normalization: NormalizationStats | None,
    splits: dict[str, list[str]],
    epoch: int,
    best_metric: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(vars(config)),
        "normalization": normalization,
        "splits": splits,
        "target_names": config.target_names,
        "input_channels": len(config.resolved_features),
        "output_dim": config.model_output_dim,
        "model_type": config.model_type,
        "conv_channels": list(config.conv_channels),
        "kernel_size": config.kernel_size,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "activation": config.activation,
        "use_residual": config.use_residual,
        "delta_xy_parameterization": config.delta_xy_parameterization,
        "magnitude_loss_weight": config.magnitude_loss_weight,
        "magnitude_loss_type": config.magnitude_loss_type,
        "direction_loss_weight": config.direction_loss_weight,
        "direction_loss_min_target_magnitude_m": config.direction_loss_min_target_magnitude_m,
        "temporal_consistency_loss_weight": config.temporal_consistency_loss_weight,
        "temporal_consistency_mode": config.temporal_consistency_mode,
        "surface_anchor_loss_weight": config.surface_anchor_loss_weight,
        "surface_anchor_loss_type": config.surface_anchor_loss_type,
        "surface_slip_loss_weight": config.surface_slip_loss_weight,
        "surface_rollout_num_windows": config.surface_rollout_num_windows,
        "use_motion_gate": config.use_motion_gate,
        "use_classical_residual_target": config.use_classical_residual_target,
        "classical_residual_mode": config.classical_residual_mode,
        "motion_gate_threshold_m": config.motion_gate_threshold_m,
        "motion_gate_prob_threshold": config.motion_gate_prob_threshold,
        "motion_cls_loss_weight": config.motion_cls_loss_weight,
        "motion_reg_loss_weight": config.motion_reg_loss_weight,
        "motion_head_enabled": config.motion_head_enabled,
        "hybrid_head_enabled": config.hybrid_head_enabled,
        "hybrid_bias_correction_enabled": config.hybrid_bias_correction_enabled,
        "hybrid_bias_regularization": config.hybrid_bias_regularization,
        "hybrid_log_variance_min": config.hybrid_log_variance_min,
        "hybrid_log_variance_max": config.hybrid_log_variance_max,
        "planar_ekf_rollout_enabled": config.planar_ekf_rollout_enabled,
        "ekf_measurement_covariance_mode": config.ekf_measurement_covariance_mode,
        "planar_ekf_process_accel_std_mps2": config.planar_ekf_process_accel_std_mps2,
        "planar_ekf_process_noise_scale": config.planar_ekf_process_noise_scale,
        "planar_ekf_measurement_covariance_scale": config.planar_ekf_measurement_covariance_scale,
        "planar_ekf_measurement_covariance_floor": config.planar_ekf_measurement_covariance_floor,
        "planar_ekf_measurement_covariance_ceiling": config.planar_ekf_measurement_covariance_ceiling,
        "planar_ekf_update_stride": config.planar_ekf_update_stride,
        "planar_ekf_velocity_measurement_std_floor_mps": (
            config.planar_ekf_velocity_measurement_std_floor_mps
        ),
        "planar_ekf_velocity_measurement_std_max_mps": (
            config.planar_ekf_velocity_measurement_std_max_mps
        ),
        "planar_ekf_initial_position_std_m": config.planar_ekf_initial_position_std_m,
        "planar_ekf_initial_velocity_std_mps": config.planar_ekf_initial_velocity_std_mps,
        "uncertainty_calibration_num_buckets": config.uncertainty_calibration_num_buckets,
        "plane_relative_tracker_rollout_enabled": config.plane_relative_tracker_rollout_enabled,
        "plane_tracker_position_blend": config.plane_tracker_position_blend,
        "plane_tracker_velocity_blend": config.plane_tracker_velocity_blend,
        "plane_tracker_uncertainty_scale": config.plane_tracker_uncertainty_scale,
        "plane_tracker_velocity_damping_per_s": config.plane_tracker_velocity_damping_per_s,
        "plane_tracker_height_constraint_gain": config.plane_tracker_height_constraint_gain,
        "plane_tracker_normal_accel_gain": config.plane_tracker_normal_accel_gain,
        "plane_tracker_max_height_offset_m": config.plane_tracker_max_height_offset_m,
        "plane_tracker_short_horizon_steps": config.plane_tracker_short_horizon_steps,
        "plane_tracker_surface_half_width_m": config.plane_tracker_surface_half_width_m,
        "plane_tracker_surface_half_height_m": config.plane_tracker_surface_half_height_m,
        "loss_name": config.loss_name,
        "smooth_l1_beta": config.smooth_l1_beta,
    }


def save_split_predictions(
    split_name: str,
    run_dir: Path,
    train_dataset: SensorWindowDataset,
    dataset: SensorWindowDataset | None,
    dataloader: DataLoader | None,
    calibration_dataset: SensorWindowDataset | None,
    calibration_dataloader: DataLoader | None,
    model: torch.nn.Module,
    device: torch.device,
    save_predictions: bool,
    motion_bucket_edges: list[float],
    config: TrainingConfig,
) -> dict[str, Any] | None:
    if dataset is None or dataloader is None:
        return None
    empirical_measurement_covariance = None
    if (
        config.planar_ekf_rollout_enabled
        and config.target_mode == "delta_xy"
        and config.ekf_measurement_covariance_mode != "learned_uncertainty"
    ):
        effective_calibration_dataset = calibration_dataset if calibration_dataset is not None else dataset
        effective_calibration_dataloader = (
            calibration_dataloader if calibration_dataloader is not None else dataloader
        )
        calibration_split_name = "val" if calibration_dataset is not None else split_name
        empirical_measurement_covariance = calibrate_empirical_measurement_covariance(
            model=model,
            calibration_dataset=effective_calibration_dataset,
            calibration_dataloader=effective_calibration_dataloader,
            device=device,
            train_config=config,
            target_names=dataset.target_names,
            normalization=effective_calibration_dataset.normalization,
            calibration_split_name=calibration_split_name,
        )
    loss, metrics, predictions, targets, metadata_rows, _ = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        target_mode=config.target_mode,
        target_names=dataset.target_names,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=config.moving_eval_thresholds,
        loss_name=config.loss_name,
        smooth_l1_beta=config.smooth_l1_beta,
        motion_head_enabled=config.motion_head_enabled,
        use_motion_gate=config.use_motion_gate,
        motion_gate_threshold_m=config.motion_gate_threshold_m,
        motion_gate_prob_threshold=config.motion_gate_prob_threshold,
        motion_cls_loss_weight=config.motion_cls_loss_weight,
        motion_reg_loss_weight=config.motion_reg_loss_weight,
        motion_class_balance_mode=config.motion_class_balance_mode,
        delta_xy_parameterization=config.delta_xy_parameterization,
        magnitude_loss_weight=config.magnitude_loss_weight,
        magnitude_loss_type=config.magnitude_loss_type,
        direction_loss_weight=config.direction_loss_weight,
        direction_loss_min_target_magnitude_m=config.direction_loss_min_target_magnitude_m,
        polar_distance_weight=config.polar_distance_weight,
        polar_heading_weight=config.polar_heading_weight,
        normalization=dataset.normalization,
        use_classical_residual_target=config.use_classical_residual_target,
        dataset=dataset,
        hybrid_head_enabled=config.hybrid_head_enabled,
        hybrid_bias_correction_enabled=config.hybrid_bias_correction_enabled,
        hybrid_bias_regularization=config.hybrid_bias_regularization,
        hybrid_log_variance_min=config.hybrid_log_variance_min,
        hybrid_log_variance_max=config.hybrid_log_variance_max,
        uncertainty_calibration_num_buckets=config.uncertainty_calibration_num_buckets,
        planar_ekf_rollout_enabled=config.planar_ekf_rollout_enabled,
        ekf_measurement_covariance_mode=config.ekf_measurement_covariance_mode,
        empirical_measurement_covariance=empirical_measurement_covariance,
        planar_ekf_process_accel_std_mps2=config.planar_ekf_process_accel_std_mps2,
        planar_ekf_process_noise_scale=config.planar_ekf_process_noise_scale,
        planar_ekf_measurement_covariance_scale=config.planar_ekf_measurement_covariance_scale,
        planar_ekf_measurement_covariance_floor=config.planar_ekf_measurement_covariance_floor,
        planar_ekf_measurement_covariance_ceiling=config.planar_ekf_measurement_covariance_ceiling,
        planar_ekf_velocity_measurement_std_floor_mps=(
            config.planar_ekf_velocity_measurement_std_floor_mps
        ),
        planar_ekf_velocity_measurement_std_max_mps=(
            config.planar_ekf_velocity_measurement_std_max_mps
        ),
        planar_ekf_initial_position_std_m=config.planar_ekf_initial_position_std_m,
        planar_ekf_initial_velocity_std_mps=config.planar_ekf_initial_velocity_std_mps,
        planar_ekf_update_stride=config.planar_ekf_update_stride,
        plane_relative_tracker_rollout_enabled=config.plane_relative_tracker_rollout_enabled,
        plane_tracker_position_blend=config.plane_tracker_position_blend,
        plane_tracker_velocity_blend=config.plane_tracker_velocity_blend,
        plane_tracker_uncertainty_scale=config.plane_tracker_uncertainty_scale,
        plane_tracker_velocity_damping_per_s=config.plane_tracker_velocity_damping_per_s,
        plane_tracker_height_constraint_gain=config.plane_tracker_height_constraint_gain,
        plane_tracker_normal_accel_gain=config.plane_tracker_normal_accel_gain,
        plane_tracker_max_height_offset_m=config.plane_tracker_max_height_offset_m,
        plane_tracker_short_horizon_steps=config.plane_tracker_short_horizon_steps,
        plane_tracker_surface_half_width_m=config.plane_tracker_surface_half_width_m,
        plane_tracker_surface_half_height_m=config.plane_tracker_surface_half_height_m,
    )
    if loss is not None:
        metrics["loss"] = loss
    if config.planar_ekf_rollout_enabled:
        metrics["ekf_measurement_covariance_mode"] = config.ekf_measurement_covariance_mode
    if empirical_measurement_covariance is not None:
        metrics["empirical_measurement_covariance"] = empirical_measurement_covariance
    baseline_predictions, baseline_targets, baseline_metadata_rows, baseline_metrics = (
        evaluate_classical_double_integration(
            dataset=dataset,
            motion_bucket_edges=motion_bucket_edges,
            moving_eval_thresholds=config.moving_eval_thresholds,
        )
    )
    metrics["classical_double_integration"] = baseline_metrics
    metrics["dataset_timing_diagnostics"] = dataset.summary().get("timing_diagnostics", {})
    metrics["zero_motion_baseline"] = evaluate_zero_motion_baseline(
        dataset=dataset,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=config.moving_eval_thresholds,
    )
    metrics["train_mean_baseline"] = evaluate_train_mean_baseline(
        train_dataset=train_dataset,
        eval_dataset=dataset,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=config.moving_eval_thresholds,
    )
    metrics["previous_window_persistence_baseline"] = evaluate_previous_window_persistence_baseline(
        dataset=dataset,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=config.moving_eval_thresholds,
    )
    metrics["tabular_linear_regression_baseline"] = evaluate_tabular_linear_regression_baseline(
        train_dataset=train_dataset,
        eval_dataset=dataset,
        motion_bucket_edges=motion_bucket_edges,
        moving_eval_thresholds=config.moving_eval_thresholds,
    )
    save_json(metrics, run_dir / f"{split_name}_metrics.json")
    if save_predictions:
        write_predictions_csv(
            run_dir / f"{split_name}_predictions.csv",
            metadata_rows=metadata_rows,
            predictions=predictions,
            targets=targets,
            target_names=dataset.target_names,
        )
        write_predictions_csv(
            run_dir / f"{split_name}_double_integration_predictions.csv",
            metadata_rows=baseline_metadata_rows,
            predictions=baseline_predictions,
            targets=baseline_targets,
            target_names=dataset.target_names,
        )
    return metrics


def main() -> None:
    args = build_train_argparser().parse_args()
    config = train_config_from_args(args)
    seed_everything(config.seed)

    run_dir = make_run_dir(config.output_dir, config.run_name)
    device = resolve_device(config.device)
    splits = resolve_file_splits(config)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataloaders(
        config, splits
    )
    train_eval_loader = train_loader
    if not isinstance(train_loader.dataset, SensorWindowDataset):
        train_eval_loader = _build_standard_loader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )
    save_json({"splits": splits}, run_dir / "splits.json")
    save_json(config.to_dict(), run_dir / "config.json")
    save_json(train_dataset.summary(), run_dir / "train_dataset_summary.json")
    if hasattr(train_loader.dataset, "summary") and not isinstance(train_loader.dataset, SensorWindowDataset):
        save_json(train_loader.dataset.summary(), run_dir / "train_rollout_dataset_summary.json")
    save_json(val_dataset.summary(), run_dir / "val_dataset_summary.json")
    if test_dataset is not None:
        save_json(test_dataset.summary(), run_dir / "test_dataset_summary.json")
    if train_dataset.normalization is not None:
        save_normalization(train_dataset.normalization, run_dir / "normalization.json")

    model = build_model(
        model_type=config.model_type,
        input_channels=len(config.resolved_features),
        output_dim=config.model_output_dim,
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        activation=config.activation,
        use_residual=config.use_residual,
        motion_head_enabled=config.motion_head_enabled,
        hybrid_head_enabled=config.hybrid_head_enabled,
        hybrid_bias_correction_enabled=config.hybrid_bias_correction_enabled,
    ).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )
    best_metric = math.inf
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []
    use_rollout_monitor = monitor_requires_rollout_metrics(config)

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Train windows: {len(train_dataset)}")
    if hasattr(train_loader.dataset, "summary") and not isinstance(train_loader.dataset, SensorWindowDataset):
        print(f"Train rollout samples: {len(train_loader.dataset)}")
    print(f"Val windows: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"Test windows: {len(test_dataset)}")
    print(f"Resolved features ({len(config.resolved_features)}): {config.resolved_features}")
    print(f"Quaternion frame mode: {config.quaternion_frame_mode}")
    print(f"Delta-XY parameterization: {config.delta_xy_parameterization}")
    print(f"Magnitude loss weight: {config.magnitude_loss_weight}")
    print(f"Magnitude loss type: {config.magnitude_loss_type}")
    print(f"Loss weight mode: {config.loss_weight_mode}")
    print(f"Train min target magnitude (m): {config.min_target_magnitude_m}")
    print(f"Window mode: {config.window_mode}")
    if config.window_mode == "time":
        print(f"Window duration (s): {config.window_duration_s}")
        print(f"Window stride (s): {config.window_stride_s}")
    print(f"Classical residual target enabled: {config.use_classical_residual_target}")
    if config.use_classical_residual_target:
        print(f"Classical residual mode: {config.classical_residual_mode}")
    print(f"Temporal consistency loss weight: {config.temporal_consistency_loss_weight}")
    print(f"Surface anchor loss weight: {config.surface_anchor_loss_weight}")
    print(f"Surface anchor loss type: {config.surface_anchor_loss_type}")
    print(f"Surface slip loss weight: {config.surface_slip_loss_weight}")
    print(f"Surface rollout windows: {config.surface_rollout_num_windows}")
    print(f"Motion gate enabled: {config.use_motion_gate}")
    print(f"Motion head enabled: {config.motion_head_enabled}")
    print(f"Hybrid head enabled: {config.hybrid_head_enabled}")
    print(f"Monitor metric: {config.monitor_metric}")
    print(f"Monitor uses rollout metrics: {use_rollout_monitor}")
    if config.hybrid_head_enabled:
        print(f"Hybrid bias correction enabled: {config.hybrid_bias_correction_enabled}")
        print(f"Hybrid bias regularization: {config.hybrid_bias_regularization}")
        print(
            "Hybrid log variance clamp: "
            f"[{config.hybrid_log_variance_min:.2f}, {config.hybrid_log_variance_max:.2f}]"
        )
    print(f"Planar EKF rollout enabled: {config.planar_ekf_rollout_enabled}")
    if config.planar_ekf_rollout_enabled:
        print(
            f"Planar EKF measurement covariance mode: "
            f"{config.ekf_measurement_covariance_mode}"
        )
        print(f"Planar EKF process noise scale: {config.planar_ekf_process_noise_scale}")
        print(
            f"Planar EKF measurement covariance scale: "
            f"{config.planar_ekf_measurement_covariance_scale}"
        )
        print(f"Planar EKF update stride: {config.planar_ekf_update_stride}")
    print(f"Plane-relative tracker rollout enabled: {config.plane_relative_tracker_rollout_enabled}")
    if config.plane_relative_tracker_rollout_enabled:
        print(f"Plane tracker position blend: {config.plane_tracker_position_blend}")
        print(f"Plane tracker velocity blend: {config.plane_tracker_velocity_blend}")
        print(f"Plane tracker height constraint gain: {config.plane_tracker_height_constraint_gain}")
    if config.motion_head_enabled:
        print(f"Motion gate threshold (m): {config.motion_gate_threshold_m}")
        print(f"Motion gate probability threshold: {config.motion_gate_prob_threshold}")
        print(f"Motion cls loss weight: {config.motion_cls_loss_weight}")
        print(f"Motion reg loss weight: {config.motion_reg_loss_weight}")
        print(f"Motion class balance mode: {config.motion_class_balance_mode}")
    for split_label, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        if dataset is None:
            continue
        counts = count_motion_windows(dataset, config.motion_gate_threshold_m)
        print(
            f"{split_label} motion labels @ {config.motion_gate_threshold_m:.2f}m: "
            f"moving={counts['moving']}, static={counts['static']}"
        )
        timing = dataset.summary().get("timing_diagnostics", {})
        if timing:
            print(
                f"{split_label} timing: duration_mean={timing['actual_window_duration_s_mean']:.3f}s, "
                f"duration_std={timing['actual_window_duration_s_std']:.3f}s, "
                f"duration_min={timing['actual_window_duration_s_min']:.3f}s, "
                f"duration_max={timing['actual_window_duration_s_max']:.3f}s, "
                f"zero_dt_fraction_mean={timing['zero_dt_fraction_mean']:.3f}, "
                f"nonzero_dt_count_mean={timing['nonzero_dt_count_mean']:.2f}"
            )
    print(f"Export ONNX after training: {config.export_onnx}")
    if config.export_onnx:
        print(f"ONNX file name: {config.onnx_file_name}")
        print(f"ONNX opset version: {config.onnx_opset_version}")
        print(f"Verify ONNX export: {config.verify_onnx_export}")
    print(f"Model parameters: {count_parameters(model):,}")

    for epoch in range(1, config.epochs + 1):
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            config=config,
            device=device,
        )
        train_eval_loss, train_metrics, _, _, _, _ = evaluate_model(
            model=model,
            dataloader=train_eval_loader,
            device=device,
            target_mode=config.target_mode,
            target_names=train_dataset.target_names,
            motion_bucket_edges=config.motion_bucket_edges,
            moving_eval_thresholds=config.moving_eval_thresholds,
            loss_name=config.loss_name,
            smooth_l1_beta=config.smooth_l1_beta,
            motion_head_enabled=config.motion_head_enabled,
            use_motion_gate=config.use_motion_gate,
            motion_gate_threshold_m=config.motion_gate_threshold_m,
            motion_gate_prob_threshold=config.motion_gate_prob_threshold,
            motion_cls_loss_weight=config.motion_cls_loss_weight,
            motion_reg_loss_weight=config.motion_reg_loss_weight,
            motion_class_balance_mode=config.motion_class_balance_mode,
            delta_xy_parameterization=config.delta_xy_parameterization,
            magnitude_loss_weight=config.magnitude_loss_weight,
            magnitude_loss_type=config.magnitude_loss_type,
            direction_loss_weight=config.direction_loss_weight,
            direction_loss_min_target_magnitude_m=config.direction_loss_min_target_magnitude_m,
            polar_distance_weight=config.polar_distance_weight,
            polar_heading_weight=config.polar_heading_weight,
            normalization=train_dataset.normalization,
            use_classical_residual_target=config.use_classical_residual_target,
            dataset=train_dataset if use_rollout_monitor else None,
            hybrid_head_enabled=config.hybrid_head_enabled,
            hybrid_bias_correction_enabled=config.hybrid_bias_correction_enabled,
            hybrid_bias_regularization=config.hybrid_bias_regularization,
            hybrid_log_variance_min=config.hybrid_log_variance_min,
            hybrid_log_variance_max=config.hybrid_log_variance_max,
            uncertainty_calibration_num_buckets=config.uncertainty_calibration_num_buckets,
            planar_ekf_rollout_enabled=config.planar_ekf_rollout_enabled if use_rollout_monitor else False,
            ekf_measurement_covariance_mode=config.ekf_measurement_covariance_mode,
            planar_ekf_process_accel_std_mps2=config.planar_ekf_process_accel_std_mps2,
            planar_ekf_process_noise_scale=config.planar_ekf_process_noise_scale,
            planar_ekf_measurement_covariance_scale=config.planar_ekf_measurement_covariance_scale,
            planar_ekf_measurement_covariance_floor=config.planar_ekf_measurement_covariance_floor,
            planar_ekf_measurement_covariance_ceiling=config.planar_ekf_measurement_covariance_ceiling,
            planar_ekf_velocity_measurement_std_floor_mps=(
                config.planar_ekf_velocity_measurement_std_floor_mps
            ),
            planar_ekf_velocity_measurement_std_max_mps=(
                config.planar_ekf_velocity_measurement_std_max_mps
            ),
            planar_ekf_initial_position_std_m=config.planar_ekf_initial_position_std_m,
            planar_ekf_initial_velocity_std_mps=config.planar_ekf_initial_velocity_std_mps,
            planar_ekf_update_stride=config.planar_ekf_update_stride,
            plane_relative_tracker_rollout_enabled=(
                config.plane_relative_tracker_rollout_enabled if use_rollout_monitor else False
            ),
            plane_tracker_position_blend=config.plane_tracker_position_blend,
            plane_tracker_velocity_blend=config.plane_tracker_velocity_blend,
            plane_tracker_uncertainty_scale=config.plane_tracker_uncertainty_scale,
            plane_tracker_velocity_damping_per_s=config.plane_tracker_velocity_damping_per_s,
            plane_tracker_height_constraint_gain=config.plane_tracker_height_constraint_gain,
            plane_tracker_normal_accel_gain=config.plane_tracker_normal_accel_gain,
            plane_tracker_max_height_offset_m=config.plane_tracker_max_height_offset_m,
            plane_tracker_short_horizon_steps=config.plane_tracker_short_horizon_steps,
            plane_tracker_surface_half_width_m=config.plane_tracker_surface_half_width_m,
            plane_tracker_surface_half_height_m=config.plane_tracker_surface_half_height_m,
        )
        val_loss, val_metrics, _, _, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            target_mode=config.target_mode,
            target_names=val_dataset.target_names,
            motion_bucket_edges=config.motion_bucket_edges,
            moving_eval_thresholds=config.moving_eval_thresholds,
            loss_name=config.loss_name,
            smooth_l1_beta=config.smooth_l1_beta,
            motion_head_enabled=config.motion_head_enabled,
            use_motion_gate=config.use_motion_gate,
            motion_gate_threshold_m=config.motion_gate_threshold_m,
            motion_gate_prob_threshold=config.motion_gate_prob_threshold,
            motion_cls_loss_weight=config.motion_cls_loss_weight,
            motion_reg_loss_weight=config.motion_reg_loss_weight,
            motion_class_balance_mode=config.motion_class_balance_mode,
            delta_xy_parameterization=config.delta_xy_parameterization,
            magnitude_loss_weight=config.magnitude_loss_weight,
            magnitude_loss_type=config.magnitude_loss_type,
            direction_loss_weight=config.direction_loss_weight,
            direction_loss_min_target_magnitude_m=config.direction_loss_min_target_magnitude_m,
            polar_distance_weight=config.polar_distance_weight,
            polar_heading_weight=config.polar_heading_weight,
            normalization=val_dataset.normalization,
            use_classical_residual_target=config.use_classical_residual_target,
            dataset=val_dataset if use_rollout_monitor else None,
            hybrid_head_enabled=config.hybrid_head_enabled,
            hybrid_bias_correction_enabled=config.hybrid_bias_correction_enabled,
            hybrid_bias_regularization=config.hybrid_bias_regularization,
            hybrid_log_variance_min=config.hybrid_log_variance_min,
            hybrid_log_variance_max=config.hybrid_log_variance_max,
            uncertainty_calibration_num_buckets=config.uncertainty_calibration_num_buckets,
            planar_ekf_rollout_enabled=config.planar_ekf_rollout_enabled if use_rollout_monitor else False,
            ekf_measurement_covariance_mode=config.ekf_measurement_covariance_mode,
            planar_ekf_process_accel_std_mps2=config.planar_ekf_process_accel_std_mps2,
            planar_ekf_process_noise_scale=config.planar_ekf_process_noise_scale,
            planar_ekf_measurement_covariance_scale=config.planar_ekf_measurement_covariance_scale,
            planar_ekf_measurement_covariance_floor=config.planar_ekf_measurement_covariance_floor,
            planar_ekf_measurement_covariance_ceiling=config.planar_ekf_measurement_covariance_ceiling,
            planar_ekf_velocity_measurement_std_floor_mps=(
                config.planar_ekf_velocity_measurement_std_floor_mps
            ),
            planar_ekf_velocity_measurement_std_max_mps=(
                config.planar_ekf_velocity_measurement_std_max_mps
            ),
            planar_ekf_initial_position_std_m=config.planar_ekf_initial_position_std_m,
            planar_ekf_initial_velocity_std_mps=config.planar_ekf_initial_velocity_std_mps,
            planar_ekf_update_stride=config.planar_ekf_update_stride,
            plane_relative_tracker_rollout_enabled=(
                config.plane_relative_tracker_rollout_enabled if use_rollout_monitor else False
            ),
            plane_tracker_position_blend=config.plane_tracker_position_blend,
            plane_tracker_velocity_blend=config.plane_tracker_velocity_blend,
            plane_tracker_uncertainty_scale=config.plane_tracker_uncertainty_scale,
            plane_tracker_velocity_damping_per_s=config.plane_tracker_velocity_damping_per_s,
            plane_tracker_height_constraint_gain=config.plane_tracker_height_constraint_gain,
            plane_tracker_normal_accel_gain=config.plane_tracker_normal_accel_gain,
            plane_tracker_max_height_offset_m=config.plane_tracker_max_height_offset_m,
            plane_tracker_short_horizon_steps=config.plane_tracker_short_horizon_steps,
            plane_tracker_surface_half_width_m=config.plane_tracker_surface_half_width_m,
            plane_tracker_surface_half_height_m=config.plane_tracker_surface_half_height_m,
        )
        if train_eval_loss is not None:
            train_metrics["loss"] = train_eval_loss
        if val_loss is not None:
            val_metrics["loss"] = val_loss

        monitor_value = choose_monitor_value(val_metrics, config.monitor_metric)
        scheduler.step(monitor_value)

        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_losses["total_loss"],
            "train_regression_loss": train_losses["regression_loss"],
            "train_motion_cls_loss": train_losses["motion_cls_loss"],
            "train_temporal_consistency_loss": train_losses["temporal_consistency_loss"],
            "train_surface_anchor_loss": train_losses["surface_anchor_loss"],
            "train_surface_slip_loss": train_losses["surface_slip_loss"],
            "train_eval_loss": train_eval_loss if train_eval_loss is not None else float("nan"),
            "val_loss": val_loss if val_loss is not None else float("nan"),
        }
        epoch_row.update({f"train_{k}": v for k, v in flatten_metric_values(train_metrics).items()})
        epoch_row.update({f"val_{k}": v for k, v in flatten_metric_values(val_metrics).items()})
        history.append(epoch_row)

        print(
            f"Epoch {epoch:03d} | train_loss={train_losses['total_loss']:.6f} | "
            f"val_{config.monitor_metric}={monitor_value:.6f} | val_loss={val_loss:.6f}"
        )
        if config.temporal_consistency_loss_weight > 0.0:
            print(
                f"Epoch {epoch:03d} temporal_consistency_loss="
                f"{train_losses['temporal_consistency_loss']:.6f}"
            )
        if config.surface_anchor_loss_weight > 0.0 or config.surface_slip_loss_weight > 0.0:
            print(
                f"Epoch {epoch:03d} surface_losses="
                f"anchor:{train_losses['surface_anchor_loss']:.6f}, "
                f"slip:{train_losses['surface_slip_loss']:.6f}"
            )

        save_checkpoint(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                normalization=train_dataset.normalization,
                splits=splits,
                epoch=epoch,
                best_metric=best_metric,
            ),
            run_dir / "last_checkpoint.pt",
        )

        if monitor_value < best_metric:
            best_metric = monitor_value
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    normalization=train_dataset.normalization,
                    splits=splits,
                    epoch=epoch,
                    best_metric=best_metric,
                ),
                run_dir / "best_checkpoint.pt",
            )
            save_json(val_metrics, run_dir / "best_val_metrics.json")
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    save_history_csv(history, run_dir / "history.csv")
    if config.save_plots:
        plot_training_curves(history, run_dir / "training_curves.png")

    best_checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    final_summary = {
        "best_epoch": best_epoch,
        "best_monitor_metric": best_metric,
        "monitor_metric": config.monitor_metric,
    }
    save_json(final_summary, run_dir / "training_summary.json")

    val_metrics = save_split_predictions(
        split_name="val",
        run_dir=run_dir,
        train_dataset=train_dataset,
        dataset=val_dataset,
        dataloader=val_loader,
        calibration_dataset=val_dataset,
        calibration_dataloader=val_loader,
        model=model,
        device=device,
        save_predictions=config.save_predictions,
        motion_bucket_edges=config.motion_bucket_edges,
        config=config,
    )
    test_metrics = save_split_predictions(
        split_name="test",
        run_dir=run_dir,
        train_dataset=train_dataset,
        dataset=test_dataset,
        dataloader=test_loader,
        calibration_dataset=val_dataset,
        calibration_dataloader=val_loader,
        model=model,
        device=device,
        save_predictions=config.save_predictions,
        motion_bucket_edges=config.motion_bucket_edges,
        config=config,
    )

    onnx_export_result = None
    if config.export_onnx:
        onnx_export_result = export_checkpoint_to_onnx(
            checkpoint_path=run_dir / "best_checkpoint.pt",
            output_path=run_dir / config.onnx_file_name,
            opset_version=config.onnx_opset_version,
            verify=config.verify_onnx_export,
        )
        save_json(onnx_export_result, run_dir / "onnx_export_summary.json")

    print("Training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {config.monitor_metric}: {best_metric:.6f}")
    if onnx_export_result is not None:
        print(f"Exported ONNX: {onnx_export_result['onnx_path']}")
        if onnx_export_result["verification_path"] is not None:
            print(f"ONNX verification max abs diff: {onnx_export_result['max_abs_diff_by_output']}")
    if val_metrics is not None:
        print_split_summary("Validation", val_metrics)
    if test_metrics is not None:
        print_split_summary("Test", test_metrics)


if __name__ == "__main__":
    main()
