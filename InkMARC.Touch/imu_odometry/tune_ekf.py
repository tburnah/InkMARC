from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from config import TrainingConfig
from dataset import collate_sensor_windows
from evaluate import dataset_from_checkpoint_config, evaluate_model
from model import build_model
from utils import ensure_dir, load_checkpoint, resolve_device, save_history_csv, save_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small EKF tuning sweep on a fixed hybrid checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--selection-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--report-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--process-noise-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument(
        "--measurement-covariance-scales",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--measurement-covariance-floors",
        nargs="+",
        type=float,
        default=[0.0004, 0.0025, 0.01],
    )
    parser.add_argument("--measurement-covariance-ceiling", type=float, default=None)
    parser.add_argument("--update-strides", nargs="+", type=int, default=[1, 2, 3])
    return parser


def load_train_config(checkpoint_payload: dict[str, Any]) -> TrainingConfig:
    config_fields = set(TrainingConfig.__dataclass_fields__.keys())
    config_payload = {
        key: value for key, value in checkpoint_payload["config"].items() if key in config_fields
    }
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
    return train_config


def build_dataset_and_loader(
    checkpoint_payload: dict[str, Any],
    train_config: TrainingConfig,
    normalization,
    split_name: str,
    batch_size: int,
    num_workers: int,
) -> tuple[Any, DataLoader]:
    files = checkpoint_payload["splits"].get(split_name, [])
    if not files:
        raise ValueError(f"No CSV files available for split '{split_name}'.")
    dataset = dataset_from_checkpoint_config(
        files=files,
        train_config=train_config,
        normalization=normalization,
        split_name=split_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sensor_windows,
    )
    return dataset, dataloader


def evaluate_with_ekf_overrides(
    model: torch.nn.Module,
    dataset,
    dataloader: DataLoader,
    device: torch.device,
    checkpoint_payload: dict[str, Any],
    train_config: TrainingConfig,
    normalization,
    process_noise_scale: float,
    measurement_covariance_scale: float,
    measurement_covariance_floor: float,
    measurement_covariance_ceiling: float | None,
    update_stride: int,
) -> dict[str, Any]:
    loss, metrics, _, _, _, _ = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        target_mode=train_config.target_mode,
        target_names=checkpoint_payload["target_names"],
        motion_bucket_edges=train_config.motion_bucket_edges,
        moving_eval_thresholds=train_config.moving_eval_thresholds,
        loss_name=checkpoint_payload["loss_name"],
        smooth_l1_beta=checkpoint_payload["smooth_l1_beta"],
        motion_head_enabled=train_config.motion_head_enabled,
        use_motion_gate=train_config.use_motion_gate,
        motion_gate_threshold_m=train_config.motion_gate_threshold_m,
        motion_gate_prob_threshold=train_config.motion_gate_prob_threshold,
        motion_cls_loss_weight=train_config.motion_cls_loss_weight,
        motion_reg_loss_weight=train_config.motion_reg_loss_weight,
        motion_class_balance_mode=train_config.motion_class_balance_mode,
        delta_xy_parameterization=train_config.delta_xy_parameterization,
        magnitude_loss_weight=train_config.magnitude_loss_weight,
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
        planar_ekf_rollout_enabled=True,
        planar_ekf_process_accel_std_mps2=train_config.planar_ekf_process_accel_std_mps2,
        planar_ekf_process_noise_scale=process_noise_scale,
        planar_ekf_measurement_covariance_scale=measurement_covariance_scale,
        planar_ekf_measurement_covariance_floor=measurement_covariance_floor,
        planar_ekf_measurement_covariance_ceiling=measurement_covariance_ceiling,
        planar_ekf_velocity_measurement_std_floor_mps=(
            train_config.planar_ekf_velocity_measurement_std_floor_mps
        ),
        planar_ekf_velocity_measurement_std_max_mps=(
            train_config.planar_ekf_velocity_measurement_std_max_mps
        ),
        planar_ekf_initial_position_std_m=train_config.planar_ekf_initial_position_std_m,
        planar_ekf_initial_velocity_std_mps=train_config.planar_ekf_initial_velocity_std_mps,
        planar_ekf_update_stride=update_stride,
    )
    if loss is not None:
        metrics["loss"] = loss
    return metrics


def main() -> None:
    args = build_argparser().parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    train_config = load_train_config(checkpoint)
    normalization = checkpoint["normalization"]
    output_dir = (
        ensure_dir(args.output_dir)
        if args.output_dir is not None
        else ensure_dir(Path(args.checkpoint).resolve().parent / "ekf_tuning")
    )
    save_json(vars(args), output_dir / "sweep_args.json")

    selection_dataset, selection_loader = build_dataset_and_loader(
        checkpoint_payload=checkpoint,
        train_config=train_config,
        normalization=normalization,
        split_name=args.selection_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    report_dataset, report_loader = build_dataset_and_loader(
        checkpoint_payload=checkpoint,
        train_config=train_config,
        normalization=normalization,
        split_name=args.report_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
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

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    for process_noise_scale, measurement_covariance_scale, measurement_covariance_floor, update_stride in itertools.product(
        args.process_noise_scales,
        args.measurement_covariance_scales,
        args.measurement_covariance_floors,
        args.update_strides,
    ):
        metrics = evaluate_with_ekf_overrides(
            model=model,
            dataset=selection_dataset,
            dataloader=selection_loader,
            device=device,
            checkpoint_payload=checkpoint,
            train_config=train_config,
            normalization=normalization,
            process_noise_scale=process_noise_scale,
            measurement_covariance_scale=measurement_covariance_scale,
            measurement_covariance_floor=measurement_covariance_floor,
            measurement_covariance_ceiling=args.measurement_covariance_ceiling,
            update_stride=update_stride,
        )
        direct_rollout = metrics["sequential_rollout"]["direct_model"]
        classical_rollout = metrics["sequential_rollout"]["classical_baseline"]
        ekf_rollout = metrics["sequential_rollout"]["planar_ekf"]
        row = {
            "process_noise_scale": process_noise_scale,
            "measurement_covariance_scale": measurement_covariance_scale,
            "measurement_covariance_floor": measurement_covariance_floor,
            "measurement_covariance_ceiling": (
                args.measurement_covariance_ceiling
                if args.measurement_covariance_ceiling is not None
                else train_config.planar_ekf_velocity_measurement_std_max_mps
                * train_config.planar_ekf_velocity_measurement_std_max_mps
            ),
            "update_stride": update_stride,
            "selection_vector_mae": float(metrics["vector_mae"]),
            "selection_direct_rollout_position_mae_m": float(direct_rollout["position_mae_m"]),
            "selection_classical_rollout_position_mae_m": float(classical_rollout["position_mae_m"]),
            "selection_planar_ekf_position_mae_m": float(ekf_rollout["position_mae_m"]),
            "selection_planar_ekf_vs_direct_delta_m": float(
                ekf_rollout["position_mae_m"] - direct_rollout["position_mae_m"]
            ),
            "selection_planar_ekf_vs_classical_delta_m": float(
                ekf_rollout["position_mae_m"] - classical_rollout["position_mae_m"]
            ),
            "selection_uncertainty_mean_sigma_x_m": float(
                metrics.get("hybrid_uncertainty", {}).get("mean_sigma_x_m", float("nan"))
            ),
            "selection_uncertainty_mean_sigma_y_m": float(
                metrics.get("hybrid_uncertainty", {}).get("mean_sigma_y_m", float("nan"))
            ),
            "selection_uncertainty_nll_mean": float(
                metrics.get("hybrid_uncertainty", {}).get("nll_mean", float("nan"))
            ),
        }
        rows.append(row)
        save_history_csv(rows, output_dir / "ekf_sweep.csv")
        if best_row is None or row["selection_planar_ekf_position_mae_m"] < best_row["selection_planar_ekf_position_mae_m"]:
            best_row = row
            best_metrics = metrics
            print(
                "New best selection EKF: "
                f"process_noise_scale={process_noise_scale}, "
                f"measurement_covariance_scale={measurement_covariance_scale}, "
                f"measurement_covariance_floor={measurement_covariance_floor}, "
                f"update_stride={update_stride}, "
                f"position_mae={row['selection_planar_ekf_position_mae_m']:.6f}"
            )

    if best_row is None or best_metrics is None:
        raise RuntimeError("EKF sweep produced no results.")

    report_metrics = evaluate_with_ekf_overrides(
        model=model,
        dataset=report_dataset,
        dataloader=report_loader,
        device=device,
        checkpoint_payload=checkpoint,
        train_config=train_config,
        normalization=normalization,
        process_noise_scale=best_row["process_noise_scale"],
        measurement_covariance_scale=best_row["measurement_covariance_scale"],
        measurement_covariance_floor=best_row["measurement_covariance_floor"],
        measurement_covariance_ceiling=best_row["measurement_covariance_ceiling"],
        update_stride=int(best_row["update_stride"]),
    )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selection_split": args.selection_split,
        "report_split": args.report_split,
        "num_settings": len(rows),
        "best_setting": best_row,
        "selection_metrics": best_metrics,
        "report_metrics": report_metrics,
    }
    save_json(summary, output_dir / "ekf_sweep_summary.json")

    direct_rollout = report_metrics["sequential_rollout"]["direct_model"]
    classical_rollout = report_metrics["sequential_rollout"]["classical_baseline"]
    ekf_rollout = report_metrics["sequential_rollout"]["planar_ekf"]
    print(f"Best setting saved to: {output_dir / 'ekf_sweep_summary.json'}")
    print(
        "Report split rollouts: "
        f"direct={direct_rollout['position_mae_m']:.6f}, "
        f"classical={classical_rollout['position_mae_m']:.6f}, "
        f"planar_ekf={ekf_rollout['position_mae_m']:.6f}"
    )
    print(
        "Planar EKF deltas: "
        f"vs_direct={ekf_rollout['position_mae_m'] - direct_rollout['position_mae_m']:+.6f}, "
        f"vs_classical={ekf_rollout['position_mae_m'] - classical_rollout['position_mae_m']:+.6f}"
    )


if __name__ == "__main__":
    main()
