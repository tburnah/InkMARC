from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from typing import Any

FEATURE_GROUPS: dict[str, list[str]] = {
    "base": [
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
    ],
    "gravity": ["gravity_x", "gravity_y", "gravity_z"],
    "linear_accel": ["linear_accel_x", "linear_accel_y", "linear_accel_z"],
    "magnetometer": ["magnetometer_x", "magnetometer_y", "magnetometer_z"],
    "nav_frame": [
        "nav_accel_x",
        "nav_accel_y",
        "nav_accel_z",
        "nav_gyro_x",
        "nav_gyro_y",
        "nav_gyro_z",
    ],
    "time_features": [
        "step_dt_s",
        "elapsed_time_s",
    ],
}

NAV_FRAME_FEATURES: set[str] = {
    "nav_accel_x",
    "nav_accel_y",
    "nav_accel_z",
    "nav_gyro_x",
    "nav_gyro_y",
    "nav_gyro_z",
}

WINDOW_TIME_FEATURES: set[str] = {
    "step_dt_s",
    "elapsed_time_s",
}

DERIVED_FEATURES: set[str] = NAV_FRAME_FEATURES | WINDOW_TIME_FEATURES

COLUMN_ALIASES: dict[str, list[str]] = {
    "session_id": ["session_id"],
    "row_index": ["row_index"],
    "phone_wallclock_ms": ["phone_wallclock_ms"],
    "phone_elapsed_realtime_ns": ["phone_elapsed_realtime_ns"],
    "rs_timestamp_ns": ["rs_timestamp_ns"],
    "accel_x": ["accel_x"],
    "accel_y": ["accel_y"],
    "accel_z": ["accel_z"],
    "gyro_x": ["gyro_x"],
    "gyro_y": ["gyro_y"],
    "gyro_z": ["gyro_z"],
    "rot_x": ["rot_x", "quat_x"],
    "rot_y": ["rot_y", "quat_y"],
    "rot_z": ["rot_z", "quat_z"],
    "rot_w": ["rot_w", "quat_w"],
    "gravity_x": ["gravity_x", "grav_x"],
    "gravity_y": ["gravity_y", "grav_y"],
    "gravity_z": ["gravity_z", "grav_z"],
    "linear_accel_x": ["linear_accel_x", "lin_x"],
    "linear_accel_y": ["linear_accel_y", "lin_y"],
    "linear_accel_z": ["linear_accel_z", "lin_z"],
    "magnetometer_x": ["magnetometer_x", "mag_x"],
    "magnetometer_y": ["magnetometer_y", "mag_y"],
    "magnetometer_z": ["magnetometer_z", "mag_z"],
    "tag_pose_valid": ["tag_pose_valid"],
    "depth_valid": ["depth_valid"],
    "plane_x_m": ["plane_x_m"],
    "plane_y_m": ["plane_y_m"],
    "plane_z_m": ["plane_z_m"],
}

DEFAULT_SORT_COLUMN_CANDIDATES: list[str] = [
    "phone_elapsed_realtime_ns",
    "rs_timestamp_ns",
    "phone_wallclock_ms",
    "row_index",
]

TARGET_NAME_MAP: dict[str, list[str]] = {
    "delta_xyz": ["delta_x", "delta_y", "delta_z"],
    "delta_xy": ["delta_x", "delta_y"],
    "delta_z": ["delta_z"],
    "polar_xy": ["delta_distance_xy", "delta_heading"],
    "velocity_xyz": ["vel_x", "vel_y", "vel_z"],
    "velocity_z": ["vel_z"],
}


def alias_lookup() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            mapping[alias.lower()] = canonical_name
    return mapping


def canonicalize_name(name: str) -> str:
    lookup = alias_lookup()
    return lookup.get(name.lower(), name)


def resolve_feature_selection(
    feature_groups: list[str] | None,
    explicit_features: list[str] | None,
) -> list[str]:
    ordered: list[str] = []
    if explicit_features:
        candidates = explicit_features
    else:
        groups = feature_groups or ["base"]
        candidates = []
        for group in groups:
            if group not in FEATURE_GROUPS:
                raise ValueError(
                    f"Unknown feature group '{group}'. Valid groups: {sorted(FEATURE_GROUPS)}"
                )
            candidates.extend(FEATURE_GROUPS[group])

    for candidate in candidates:
        canonical = canonicalize_name(candidate)
        if canonical not in COLUMN_ALIASES and canonical not in DERIVED_FEATURES:
            raise ValueError(
                f"Unknown feature '{candidate}'. Add an alias in config.py if your logs use a different name."
            )
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


@dataclass
class TrainingConfig:
    data_dir: str | None = None
    glob_pattern: str = "*.csv"
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    feature_groups: list[str] = field(default_factory=lambda: ["base"])
    features: list[str] = field(default_factory=list)
    target_mode: str = "delta_xyz"
    required_valid_flags: list[str] = field(default_factory=lambda: ["tag_pose_valid"])
    sort_columns: list[str] = field(default_factory=lambda: list(DEFAULT_SORT_COLUMN_CANDIDATES))
    window_size: int = 60
    stride: int = 10
    window_mode: str = "rows"
    window_duration_s: float = 1.0
    window_stride_s: float = 0.10
    min_valid_label_points: int = 2
    min_target_magnitude_m: float = 0.0
    normalize_targets: bool = False
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler_patience: int = 4
    early_stopping_patience: int = 10
    min_lr: float = 1e-6
    loss_name: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    model_type: str = "temporal_cnn"
    conv_channels: list[int] = field(default_factory=lambda: [64, 128, 128])
    kernel_size: int = 5
    hidden_dim: int = 128
    dropout: float = 0.1
    activation: str = "gelu"
    use_residual: bool = True
    monitor_metric: str = "mae_mean"
    delta_xy_parameterization: str = "cartesian"
    magnitude_loss_weight: float = 1.0
    magnitude_loss_type: str = "l1"
    direction_loss_weight: float = 1.0
    direction_loss_min_target_magnitude_m: float = 0.02
    quaternion_frame_mode: str = "body_to_nav"
    motion_bucket_edges: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    moving_eval_thresholds: list[float] = field(default_factory=lambda: [0.10])
    loss_weight_mode: str = "none"
    loss_weight_reference_m: float = 0.05
    loss_weight_min: float = 1.0
    loss_weight_max: float = 5.0
    use_classical_residual_target: bool = False
    classical_residual_mode: str = "additive"
    temporal_consistency_loss_weight: float = 0.0
    temporal_consistency_mode: str = "velocity_l1"
    surface_anchor_loss_weight: float = 0.0
    surface_anchor_loss_type: str = "l1"
    surface_slip_loss_weight: float = 0.0
    surface_rollout_num_windows: int = 4
    use_motion_gate: bool = False
    motion_gate_threshold_m: float = 0.10
    motion_gate_prob_threshold: float = 0.5
    motion_cls_loss_weight: float = 1.0
    motion_reg_loss_weight: float = 1.0
    motion_head_enabled: bool = False
    motion_threshold_m: float = 0.10
    motion_loss_weight: float = 1.0
    motion_class_balance_mode: str = "inverse_frequency"
    motion_gate_inference: bool = True
    hybrid_head_enabled: bool = False
    hybrid_bias_correction_enabled: bool = False
    hybrid_bias_regularization: float = 1e-3
    hybrid_log_variance_min: float = -6.0
    hybrid_log_variance_max: float = 6.0
    planar_ekf_rollout_enabled: bool = False
    ekf_measurement_covariance_mode: str = "learned_uncertainty"
    planar_ekf_process_accel_std_mps2: float = 0.75
    planar_ekf_process_noise_scale: float = 1.0
    planar_ekf_measurement_covariance_scale: float = 1.0
    planar_ekf_measurement_covariance_floor: float | None = None
    planar_ekf_measurement_covariance_ceiling: float | None = None
    planar_ekf_update_stride: int = 1
    planar_ekf_velocity_measurement_std_floor_mps: float = 0.05
    planar_ekf_velocity_measurement_std_max_mps: float = 5.0
    planar_ekf_initial_position_std_m: float = 0.10
    planar_ekf_initial_velocity_std_mps: float = 0.25
    uncertainty_calibration_num_buckets: int = 5
    plane_relative_tracker_rollout_enabled: bool = False
    plane_tracker_position_blend: float = 0.70
    plane_tracker_velocity_blend: float = 0.35
    plane_tracker_uncertainty_scale: float = 0.10
    plane_tracker_velocity_damping_per_s: float = 0.35
    plane_tracker_height_constraint_gain: float = 8.0
    plane_tracker_normal_accel_gain: float = 0.05
    plane_tracker_max_height_offset_m: float = 0.05
    plane_tracker_short_horizon_steps: int = 3
    plane_tracker_surface_half_width_m: float = 0.15
    plane_tracker_surface_half_height_m: float = 0.10
    polar_distance_weight: float = 1.0
    polar_heading_weight: float = 1.0
    export_onnx: bool = False
    onnx_opset_version: int = 17
    onnx_file_name: str = "model.onnx"
    verify_onnx_export: bool = True
    output_dir: str = "./runs/imu_odometry"
    run_name: str = "temporal_cnn_baseline"
    seed: int = 42
    device: str = "auto"
    save_plots: bool = True
    save_predictions: bool = True

    @property
    def resolved_features(self) -> list[str]:
        return resolve_feature_selection(self.feature_groups, self.features)

    @property
    def target_names(self) -> list[str]:
        return TARGET_NAME_MAP[self.target_mode]

    @property
    def model_output_dim(self) -> int:
        if self.target_mode == "delta_xy" and self.delta_xy_parameterization == "direction_magnitude":
            return 3
        return len(self.target_names)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_features"] = self.resolved_features
        payload["target_names"] = self.target_names
        return payload


@dataclass
class EvaluationConfig:
    checkpoint: str
    split: str = "test"
    csv_files: list[str] = field(default_factory=list)
    batch_size: int = 256
    num_workers: int = 0
    device: str = "auto"
    output_dir: str | None = None
    save_plots: bool = True
    ekf_measurement_covariance_mode: str | None = None
    planar_ekf_process_noise_scale: float | None = None
    planar_ekf_measurement_covariance_scale: float | None = None
    planar_ekf_measurement_covariance_floor: float | None = None
    planar_ekf_measurement_covariance_ceiling: float | None = None
    planar_ekf_update_stride: int | None = None
    uncertainty_calibration_num_buckets: int | None = None
    plane_relative_tracker_rollout: bool = False
    plane_tracker_position_blend: float | None = None
    plane_tracker_velocity_blend: float | None = None
    plane_tracker_uncertainty_scale: float | None = None
    plane_tracker_velocity_damping_per_s: float | None = None
    plane_tracker_height_constraint_gain: float | None = None
    plane_tracker_normal_accel_gain: float | None = None
    plane_tracker_max_height_offset_m: float | None = None
    plane_tracker_short_horizon_steps: int | None = None
    plane_tracker_surface_half_width_m: float | None = None
    plane_tracker_surface_half_height_m: float | None = None


def build_train_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a temporal sensor-only inertial odometry baseline."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--glob-pattern", type=str, default="*.csv")
    parser.add_argument("--train-files", nargs="*", default=[])
    parser.add_argument("--val-files", nargs="*", default=[])
    parser.add_argument("--test-files", nargs="*", default=[])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--feature-groups",
        nargs="*",
        default=["base"],
        choices=sorted(FEATURE_GROUPS.keys()),
    )
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument(
        "--target-mode",
        type=str,
        default="delta_xyz",
        choices=sorted(TARGET_NAME_MAP.keys()),
    )
    parser.add_argument("--required-valid-flags", nargs="*", default=["tag_pose_valid"])
    parser.add_argument("--sort-columns", nargs="*", default=list(DEFAULT_SORT_COLUMN_CANDIDATES))
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--window-mode", type=str, default="rows", choices=["rows", "time"])
    parser.add_argument("--window-duration-s", type=float, default=1.0)
    parser.add_argument("--window-stride-s", type=float, default=0.10)
    parser.add_argument("--min-valid-label-points", type=int, default=2)
    parser.add_argument("--min-target-magnitude-m", type=float, default=0.0)
    parser.add_argument("--normalize-targets", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--loss-name", type=str, default="smooth_l1", choices=["smooth_l1", "l1", "mse"])
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument(
        "--model-type",
        type=str,
        default="temporal_cnn",
        choices=["temporal_cnn", "cnn_gru", "bilstm"],
    )
    parser.add_argument("--conv-channels", nargs="+", type=int, default=[64, 128, 128])
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"])
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--monitor-metric", type=str, default="mae_mean")
    parser.add_argument(
        "--delta-xy-parameterization",
        type=str,
        default="cartesian",
        choices=["cartesian", "direction_magnitude"],
    )
    parser.add_argument("--magnitude-loss-weight", type=float, default=None)
    parser.add_argument(
        "--magnitude-loss-type",
        type=str,
        default="l1",
        choices=["l1", "smooth_l1"],
    )
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-min-target-magnitude-m", type=float, default=0.02)
    parser.add_argument(
        "--quaternion-frame-mode",
        type=str,
        default="body_to_nav",
        choices=["body_to_nav", "nav_to_body"],
    )
    parser.add_argument("--motion-bucket-edges", nargs="*", type=float, default=[0.05, 0.10, 0.15])
    parser.add_argument("--moving-eval-thresholds", nargs="*", type=float, default=[0.10])
    parser.add_argument(
        "--loss-weight-mode",
        type=str,
        default="none",
        choices=["none", "target_magnitude"],
    )
    parser.add_argument("--loss-weight-reference-m", type=float, default=0.05)
    parser.add_argument("--loss-weight-min", type=float, default=1.0)
    parser.add_argument("--loss-weight-max", type=float, default=5.0)
    parser.add_argument("--use-classical-residual-target", action="store_true")
    parser.add_argument(
        "--classical-residual-mode",
        type=str,
        default="additive",
        choices=["additive"],
    )
    parser.add_argument("--temporal-consistency-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--temporal-consistency-mode",
        type=str,
        default="velocity_l1",
        choices=["velocity_l1"],
    )
    parser.add_argument("--surface-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--surface-anchor-loss-type",
        type=str,
        default="l1",
        choices=["l1", "smooth_l1"],
    )
    parser.add_argument("--surface-slip-loss-weight", type=float, default=0.0)
    parser.add_argument("--surface-rollout-num-windows", type=int, default=4)
    parser.add_argument("--use-motion-gate", action="store_true")
    parser.add_argument("--motion-gate-threshold-m", type=float, default=None)
    parser.add_argument("--motion-gate-prob-threshold", type=float, default=None)
    parser.add_argument("--motion-cls-loss-weight", type=float, default=None)
    parser.add_argument("--motion-reg-loss-weight", type=float, default=None)
    parser.add_argument("--motion-head", action="store_true")
    parser.add_argument("--motion-threshold-m", type=float, default=0.10)
    parser.add_argument("--motion-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--motion-class-balance-mode",
        type=str,
        default="inverse_frequency",
        choices=["none", "inverse_frequency"],
    )
    parser.add_argument("--no-motion-gating", action="store_true")
    parser.add_argument("--hybrid-head", action="store_true")
    parser.add_argument("--hybrid-bias-correction", action="store_true")
    parser.add_argument("--hybrid-bias-regularization", type=float, default=1e-3)
    parser.add_argument("--hybrid-log-variance-min", type=float, default=-6.0)
    parser.add_argument("--hybrid-log-variance-max", type=float, default=6.0)
    parser.add_argument("--planar-ekf-rollout", action="store_true")
    parser.add_argument(
        "--ekf-measurement-covariance-mode",
        type=str,
        default="learned_uncertainty",
        choices=["learned_uncertainty", "fixed_empirical", "bucketed_empirical"],
    )
    parser.add_argument("--planar-ekf-process-accel-std-mps2", type=float, default=0.75)
    parser.add_argument("--planar-ekf-process-noise-scale", type=float, default=1.0)
    parser.add_argument("--planar-ekf-measurement-covariance-scale", type=float, default=1.0)
    parser.add_argument("--planar-ekf-measurement-covariance-floor", type=float, default=None)
    parser.add_argument("--planar-ekf-measurement-covariance-ceiling", type=float, default=None)
    parser.add_argument("--planar-ekf-update-stride", type=int, default=1)
    parser.add_argument(
        "--planar-ekf-velocity-measurement-std-floor-mps",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--planar-ekf-velocity-measurement-std-max-mps",
        type=float,
        default=5.0,
    )
    parser.add_argument("--planar-ekf-initial-position-std-m", type=float, default=0.10)
    parser.add_argument("--planar-ekf-initial-velocity-std-mps", type=float, default=0.25)
    parser.add_argument("--uncertainty-calibration-buckets", type=int, default=5)
    parser.add_argument("--plane-relative-tracker-rollout", action="store_true")
    parser.add_argument("--plane-tracker-position-blend", type=float, default=0.70)
    parser.add_argument("--plane-tracker-velocity-blend", type=float, default=0.35)
    parser.add_argument("--plane-tracker-uncertainty-scale", type=float, default=0.10)
    parser.add_argument("--plane-tracker-velocity-damping-per-s", type=float, default=0.35)
    parser.add_argument("--plane-tracker-height-constraint-gain", type=float, default=8.0)
    parser.add_argument("--plane-tracker-normal-accel-gain", type=float, default=0.05)
    parser.add_argument("--plane-tracker-max-height-offset-m", type=float, default=0.05)
    parser.add_argument("--plane-tracker-short-horizon-steps", type=int, default=3)
    parser.add_argument("--plane-tracker-surface-half-width-m", type=float, default=0.15)
    parser.add_argument("--plane-tracker-surface-half-height-m", type=float, default=0.10)
    parser.add_argument("--polar-distance-weight", type=float, default=1.0)
    parser.add_argument("--polar-heading-weight", type=float, default=1.0)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-opset-version", type=int, default=17)
    parser.add_argument("--onnx-file-name", type=str, default="model.onnx")
    parser.add_argument("--no-verify-onnx-export", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./runs/imu_odometry")
    parser.add_argument("--run-name", type=str, default="temporal_cnn_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser


def build_eval_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained inertial odometry checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--csv-files", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--ekf-measurement-covariance-mode",
        type=str,
        default=None,
        choices=["learned_uncertainty", "fixed_empirical", "bucketed_empirical"],
    )
    parser.add_argument("--planar-ekf-process-noise-scale", type=float, default=None)
    parser.add_argument("--planar-ekf-measurement-covariance-scale", type=float, default=None)
    parser.add_argument("--planar-ekf-measurement-covariance-floor", type=float, default=None)
    parser.add_argument("--planar-ekf-measurement-covariance-ceiling", type=float, default=None)
    parser.add_argument("--planar-ekf-update-stride", type=int, default=None)
    parser.add_argument("--uncertainty-calibration-buckets", type=int, default=None)
    parser.add_argument("--plane-relative-tracker-rollout", action="store_true")
    parser.add_argument("--plane-tracker-position-blend", type=float, default=None)
    parser.add_argument("--plane-tracker-velocity-blend", type=float, default=None)
    parser.add_argument("--plane-tracker-uncertainty-scale", type=float, default=None)
    parser.add_argument("--plane-tracker-velocity-damping-per-s", type=float, default=None)
    parser.add_argument("--plane-tracker-height-constraint-gain", type=float, default=None)
    parser.add_argument("--plane-tracker-normal-accel-gain", type=float, default=None)
    parser.add_argument("--plane-tracker-max-height-offset-m", type=float, default=None)
    parser.add_argument("--plane-tracker-short-horizon-steps", type=int, default=None)
    parser.add_argument("--plane-tracker-surface-half-width-m", type=float, default=None)
    parser.add_argument("--plane-tracker-surface-half-height-m", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def train_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    motion_gate_enabled = bool(args.use_motion_gate or (args.motion_head and not args.no_motion_gating))
    motion_head_enabled = bool(args.motion_head or args.use_motion_gate)
    hybrid_head_enabled = bool(args.hybrid_head or args.hybrid_bias_correction)
    motion_gate_threshold_m = float(
        args.motion_gate_threshold_m if args.motion_gate_threshold_m is not None else args.motion_threshold_m
    )
    motion_gate_prob_threshold = float(
        args.motion_gate_prob_threshold if args.motion_gate_prob_threshold is not None else 0.5
    )
    motion_cls_loss_weight = float(
        args.motion_cls_loss_weight if args.motion_cls_loss_weight is not None else args.motion_loss_weight
    )
    motion_reg_loss_weight = float(
        args.motion_reg_loss_weight if args.motion_reg_loss_weight is not None else 1.0
    )
    if args.magnitude_loss_weight is None:
        magnitude_loss_weight = (
            1.0
            if args.target_mode == "delta_xy"
            and args.delta_xy_parameterization == "direction_magnitude"
            else 0.0
        )
    else:
        magnitude_loss_weight = float(args.magnitude_loss_weight)
    if hybrid_head_enabled:
        if args.target_mode != "delta_xy":
            raise ValueError("Hybrid head currently supports only --target-mode delta_xy.")
        if args.delta_xy_parameterization != "cartesian":
            raise ValueError(
                "Hybrid head currently supports only --delta-xy-parameterization cartesian."
            )
        if args.use_classical_residual_target:
            raise ValueError("Hybrid head is not currently supported with --use-classical-residual-target.")
    if args.plane_relative_tracker_rollout and args.target_mode != "delta_xy":
        raise ValueError("Plane-relative tracker rollout currently supports only --target-mode delta_xy.")
    return TrainingConfig(
        data_dir=args.data_dir,
        glob_pattern=args.glob_pattern,
        train_files=args.train_files,
        val_files=args.val_files,
        test_files=args.test_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        feature_groups=args.feature_groups,
        features=args.features,
        target_mode=args.target_mode,
        required_valid_flags=[canonicalize_name(flag) for flag in args.required_valid_flags],
        sort_columns=[canonicalize_name(name) for name in args.sort_columns],
        window_size=args.window_size,
        stride=args.stride,
        window_mode=args.window_mode,
        window_duration_s=args.window_duration_s,
        window_stride_s=args.window_stride_s,
        min_valid_label_points=args.min_valid_label_points,
        min_target_magnitude_m=args.min_target_magnitude_m,
        normalize_targets=args.normalize_targets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler_patience=args.scheduler_patience,
        early_stopping_patience=args.early_stopping_patience,
        min_lr=args.min_lr,
        loss_name=args.loss_name,
        smooth_l1_beta=args.smooth_l1_beta,
        model_type=args.model_type,
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
        use_residual=not args.no_residual,
        monitor_metric=args.monitor_metric,
        delta_xy_parameterization=args.delta_xy_parameterization,
        magnitude_loss_weight=magnitude_loss_weight,
        magnitude_loss_type=args.magnitude_loss_type,
        direction_loss_weight=args.direction_loss_weight,
        direction_loss_min_target_magnitude_m=args.direction_loss_min_target_magnitude_m,
        quaternion_frame_mode=args.quaternion_frame_mode,
        motion_bucket_edges=sorted(args.motion_bucket_edges),
        moving_eval_thresholds=sorted(set(args.moving_eval_thresholds)),
        loss_weight_mode=args.loss_weight_mode,
        loss_weight_reference_m=args.loss_weight_reference_m,
        loss_weight_min=args.loss_weight_min,
        loss_weight_max=args.loss_weight_max,
        use_classical_residual_target=args.use_classical_residual_target,
        classical_residual_mode=args.classical_residual_mode,
        temporal_consistency_loss_weight=args.temporal_consistency_loss_weight,
        temporal_consistency_mode=args.temporal_consistency_mode,
        surface_anchor_loss_weight=args.surface_anchor_loss_weight,
        surface_anchor_loss_type=args.surface_anchor_loss_type,
        surface_slip_loss_weight=args.surface_slip_loss_weight,
        surface_rollout_num_windows=args.surface_rollout_num_windows,
        use_motion_gate=motion_gate_enabled,
        motion_gate_threshold_m=motion_gate_threshold_m,
        motion_gate_prob_threshold=motion_gate_prob_threshold,
        motion_cls_loss_weight=motion_cls_loss_weight,
        motion_reg_loss_weight=motion_reg_loss_weight,
        motion_head_enabled=motion_head_enabled,
        motion_threshold_m=motion_gate_threshold_m,
        motion_loss_weight=motion_cls_loss_weight,
        motion_class_balance_mode=args.motion_class_balance_mode,
        motion_gate_inference=motion_gate_enabled,
        hybrid_head_enabled=hybrid_head_enabled,
        hybrid_bias_correction_enabled=bool(args.hybrid_bias_correction),
        hybrid_bias_regularization=args.hybrid_bias_regularization,
        hybrid_log_variance_min=args.hybrid_log_variance_min,
        hybrid_log_variance_max=args.hybrid_log_variance_max,
        planar_ekf_rollout_enabled=bool(args.planar_ekf_rollout),
        ekf_measurement_covariance_mode=args.ekf_measurement_covariance_mode,
        planar_ekf_process_accel_std_mps2=args.planar_ekf_process_accel_std_mps2,
        planar_ekf_process_noise_scale=args.planar_ekf_process_noise_scale,
        planar_ekf_measurement_covariance_scale=args.planar_ekf_measurement_covariance_scale,
        planar_ekf_measurement_covariance_floor=args.planar_ekf_measurement_covariance_floor,
        planar_ekf_measurement_covariance_ceiling=args.planar_ekf_measurement_covariance_ceiling,
        planar_ekf_update_stride=args.planar_ekf_update_stride,
        planar_ekf_velocity_measurement_std_floor_mps=(
            args.planar_ekf_velocity_measurement_std_floor_mps
        ),
        planar_ekf_velocity_measurement_std_max_mps=(
            args.planar_ekf_velocity_measurement_std_max_mps
        ),
        planar_ekf_initial_position_std_m=args.planar_ekf_initial_position_std_m,
        planar_ekf_initial_velocity_std_mps=args.planar_ekf_initial_velocity_std_mps,
        uncertainty_calibration_num_buckets=args.uncertainty_calibration_buckets,
        plane_relative_tracker_rollout_enabled=bool(args.plane_relative_tracker_rollout),
        plane_tracker_position_blend=args.plane_tracker_position_blend,
        plane_tracker_velocity_blend=args.plane_tracker_velocity_blend,
        plane_tracker_uncertainty_scale=args.plane_tracker_uncertainty_scale,
        plane_tracker_velocity_damping_per_s=args.plane_tracker_velocity_damping_per_s,
        plane_tracker_height_constraint_gain=args.plane_tracker_height_constraint_gain,
        plane_tracker_normal_accel_gain=args.plane_tracker_normal_accel_gain,
        plane_tracker_max_height_offset_m=args.plane_tracker_max_height_offset_m,
        plane_tracker_short_horizon_steps=args.plane_tracker_short_horizon_steps,
        plane_tracker_surface_half_width_m=args.plane_tracker_surface_half_width_m,
        plane_tracker_surface_half_height_m=args.plane_tracker_surface_half_height_m,
        polar_distance_weight=args.polar_distance_weight,
        polar_heading_weight=args.polar_heading_weight,
        export_onnx=bool(args.export_onnx),
        onnx_opset_version=args.onnx_opset_version,
        onnx_file_name=args.onnx_file_name,
        verify_onnx_export=not args.no_verify_onnx_export,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        save_plots=not args.no_plots,
        save_predictions=not args.no_save_predictions,
    )


def eval_config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint=args.checkpoint,
        split=args.split,
        csv_files=args.csv_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        ekf_measurement_covariance_mode=args.ekf_measurement_covariance_mode,
        planar_ekf_process_noise_scale=args.planar_ekf_process_noise_scale,
        planar_ekf_measurement_covariance_scale=args.planar_ekf_measurement_covariance_scale,
        planar_ekf_measurement_covariance_floor=args.planar_ekf_measurement_covariance_floor,
        planar_ekf_measurement_covariance_ceiling=args.planar_ekf_measurement_covariance_ceiling,
        planar_ekf_update_stride=args.planar_ekf_update_stride,
        uncertainty_calibration_num_buckets=args.uncertainty_calibration_buckets,
        plane_relative_tracker_rollout=bool(args.plane_relative_tracker_rollout),
        plane_tracker_position_blend=args.plane_tracker_position_blend,
        plane_tracker_velocity_blend=args.plane_tracker_velocity_blend,
        plane_tracker_uncertainty_scale=args.plane_tracker_uncertainty_scale,
        plane_tracker_velocity_damping_per_s=args.plane_tracker_velocity_damping_per_s,
        plane_tracker_height_constraint_gain=args.plane_tracker_height_constraint_gain,
        plane_tracker_normal_accel_gain=args.plane_tracker_normal_accel_gain,
        plane_tracker_max_height_offset_m=args.plane_tracker_max_height_offset_m,
        plane_tracker_short_horizon_steps=args.plane_tracker_short_horizon_steps,
        plane_tracker_surface_half_width_m=args.plane_tracker_surface_half_width_m,
        plane_tracker_surface_half_height_m=args.plane_tracker_surface_half_height_m,
    )
