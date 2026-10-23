"""
Run session-level cross-validation for the inertial odometry baseline.

This wrapper launches `train.py` repeatedly with file-level train/val/test splits.
It is intended for small numbers of CSV sessions where a single split is too noisy.

Typical usage:
```
/home/t/tburnah/miniconda3/envs/autokeras_env/bin/python cross_validate.py \
  --csv-files /path/to/session1.csv /path/to/session2.csv /path/to/session3.csv /path/to/session4.csv /path/to/session5.csv \
  --mode rotation \
  --output-dir /home/t/tburnah/TouchModel/runs/imu_odometry_cv \
  -- \
  --feature-groups base gravity nav_frame \
  --window-size 60 \
  --stride 10 \
  --loss-weight-mode target_magnitude \
  --loss-weight-reference-m 0.05 \
  --min-target-magnitude-m 0.10 \
  --monitor-metric vector_mae
```
"""

from __future__ import annotations

import argparse
import glob
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import ensure_dir, flatten_metric_values, load_json, make_run_dir, save_history_csv, save_json


@dataclass
class FoldSpec:
    fold_index: int
    train_files: list[str]
    val_file: str
    test_file: str

    @property
    def fold_name(self) -> str:
        val_name = Path(self.val_file).stem
        test_name = Path(self.test_file).stem
        return f"fold_{self.fold_index:02d}_val_{val_name}_test_{test_name}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run session-level cross-validation for the IMU odometry training pipeline."
    )
    parser.add_argument("--csv-files", nargs="*", default=[])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--glob-pattern", type=str, default="*.csv")
    parser.add_argument(
        "--mode",
        type=str,
        default="rotation",
        choices=["rotation", "all_pairs"],
        help="rotation: N folds with cyclic val/test holdout. all_pairs: every ordered val/test pair.",
    )
    parser.add_argument("--output-dir", type=str, default="./runs/imu_odometry_cv")
    parser.add_argument("--run-name", type=str, default="session_crossval")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=str, default=str(Path(__file__).with_name("train.py")))
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed through to train.py. Prefix them with --train-args -- or just --.",
    )
    return parser


def resolve_csv_files(csv_files: list[str], data_dir: str | None, glob_pattern: str) -> list[str]:
    if csv_files:
        files = [str(Path(path).resolve()) for path in csv_files]
    else:
        if data_dir is None:
            raise ValueError("Provide --csv-files or --data-dir.")
        files = sorted(glob.glob(str(Path(data_dir) / glob_pattern)))
        files = [str(Path(path).resolve()) for path in files]
    if len(files) < 3:
        raise ValueError("At least 3 CSV sessions are required for session-level cross-validation.")
    return files


def build_folds(files: list[str], mode: str) -> list[FoldSpec]:
    folds: list[FoldSpec] = []
    num_files = len(files)
    if mode == "rotation":
        for idx in range(num_files):
            test_file = files[idx]
            val_file = files[(idx + 1) % num_files]
            train_files = [path for path in files if path not in {test_file, val_file}]
            folds.append(
                FoldSpec(
                    fold_index=len(folds) + 1,
                    train_files=train_files,
                    val_file=val_file,
                    test_file=test_file,
                )
            )
        return folds

    if mode == "all_pairs":
        for test_idx, test_file in enumerate(files):
            for val_idx, val_file in enumerate(files):
                if test_idx == val_idx:
                    continue
                train_files = [path for idx, path in enumerate(files) if idx not in {test_idx, val_idx}]
                folds.append(
                    FoldSpec(
                        fold_index=len(folds) + 1,
                        train_files=train_files,
                        val_file=val_file,
                        test_file=test_file,
                    )
                )
        return folds

    raise ValueError(f"Unsupported fold mode: {mode}")


def sanitize_train_args(train_args: list[str]) -> list[str]:
    args = list(train_args)
    if args and args[0] == "--":
        args = args[1:]
    forbidden = {"--train-files", "--val-files", "--test-files", "--output-dir", "--run-name"}
    overlap = [arg for arg in args if arg in forbidden]
    if overlap:
        raise ValueError(
            f"Do not pass {sorted(set(overlap))} through cross_validate.py. "
            "Those are managed by the cross-validation runner."
        )
    return args


def discover_fold_run_dir(fold_output_root: Path) -> Path:
    candidates = [path for path in fold_output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No training run directory was created under {fold_output_root}.")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def run_fold(
    fold: FoldSpec,
    cv_run_dir: Path,
    python_bin: str,
    train_script: str,
    train_args: list[str],
) -> dict[str, Any]:
    fold_root = ensure_dir(cv_run_dir / "folds" / f"fold_{fold.fold_index:02d}")
    command = [
        python_bin,
        str(Path(train_script).resolve()),
        "--train-files",
        *fold.train_files,
        "--val-files",
        fold.val_file,
        "--test-files",
        fold.test_file,
        "--output-dir",
        str(fold_root),
        "--run-name",
        fold.fold_name,
        *train_args,
    ]

    print(
        f"[fold {fold.fold_index}] val={Path(fold.val_file).name} "
        f"test={Path(fold.test_file).name}"
    )
    print(f"[fold {fold.fold_index}] command: {' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, check=True, cwd=str(Path(train_script).resolve().parent))

    fold_run_dir = discover_fold_run_dir(fold_root)
    training_summary = load_json(fold_run_dir / "training_summary.json")
    val_metrics = load_json(fold_run_dir / "val_metrics.json")
    test_metrics = load_json(fold_run_dir / "test_metrics.json")

    row: dict[str, Any] = {
        "fold_index": fold.fold_index,
        "fold_name": fold.fold_name,
        "run_dir": str(fold_run_dir),
        "val_file": fold.val_file,
        "test_file": fold.test_file,
        "train_files": " | ".join(fold.train_files),
    }
    row.update({f"training_{key}": value for key, value in flatten_metric_values(training_summary).items()})
    row.update({f"val_{key}": value for key, value in flatten_metric_values(val_metrics).items()})
    row.update({f"test_{key}": value for key, value in flatten_metric_values(test_metrics).items()})
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    aggregates: dict[str, dict[str, float]] = {}
    for key in sorted(numeric_keys):
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        aggregates[key] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return {"num_folds": len(rows), "aggregates": aggregates}


def main() -> None:
    parser = build_argparser()
    args, unknown_args = parser.parse_known_args()
    train_args = sanitize_train_args(args.train_args + unknown_args)
    files = resolve_csv_files(args.csv_files, args.data_dir, args.glob_pattern)
    folds = build_folds(files, args.mode)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]

    cv_run_dir = make_run_dir(args.output_dir, args.run_name)
    save_json(
        {
            "mode": args.mode,
            "files": files,
            "python_bin": args.python_bin,
            "train_script": str(Path(args.train_script).resolve()),
            "train_args": train_args,
            "num_folds": len(folds),
        },
        cv_run_dir / "crossval_config.json",
    )

    rows: list[dict[str, Any]] = []
    for fold in folds:
        row = run_fold(
            fold=fold,
            cv_run_dir=cv_run_dir,
            python_bin=args.python_bin,
            train_script=args.train_script,
            train_args=train_args,
        )
        rows.append(row)
        save_history_csv(rows, cv_run_dir / "crossval_folds.csv")

    summary = summarize_rows(rows)
    summary["mode"] = args.mode
    summary["files"] = files
    save_json(summary, cv_run_dir / "crossval_summary.json")

    print(f"Cross-validation complete: {cv_run_dir}")
    for metric_name in [
        "val_vector_mae",
        "val_relative_error_percentage",
        "val_gated_vector_mae",
        "val_gated_relative_error_percentage",
        "val_motion_classification_f1",
        "val_regression_by_motion_label_moving_vector_mae",
        "val_sequential_rollout_direct_model_position_mae_m",
        "val_sequential_rollout_classical_baseline_position_mae_m",
        "val_sequential_rollout_planar_ekf_position_mae_m",
        "val_hybrid_comparison_planar_ekf_vs_direct_position_mae_m_delta",
        "val_hybrid_comparison_planar_ekf_vs_classical_position_mae_m_delta",
        "test_vector_mae",
        "test_relative_error_percentage",
        "test_gated_vector_mae",
        "test_gated_relative_error_percentage",
        "test_motion_classification_f1",
        "test_regression_by_motion_label_moving_vector_mae",
        "test_sequential_rollout_direct_model_position_mae_m",
        "test_sequential_rollout_classical_baseline_position_mae_m",
        "test_sequential_rollout_planar_ekf_position_mae_m",
        "test_hybrid_comparison_planar_ekf_vs_direct_position_mae_m_delta",
        "test_hybrid_comparison_planar_ekf_vs_classical_position_mae_m_delta",
        "val_moving_only_thresholds_ge_0.10m_vector_mae",
        "val_moving_only_thresholds_ge_0.15m_vector_mae",
        "test_moving_only_thresholds_ge_0.10m_vector_mae",
        "test_moving_only_thresholds_ge_0.15m_vector_mae",
    ]:
        aggregate = summary["aggregates"].get(metric_name)
        if aggregate is None:
            continue
        print(
            f"{metric_name}: mean={aggregate['mean']:.6f}, std={aggregate['std']:.6f}, "
            f"min={aggregate['min']:.6f}, max={aggregate['max']:.6f}"
        )


if __name__ == "__main__":
    main()
