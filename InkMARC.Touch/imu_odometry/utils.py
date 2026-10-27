from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass
class NormalizationStats:
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    target_names: list[str]
    target_mean: list[float] | None = None
    target_std: list[float] | None = None

    @property
    def feature_mean_array(self) -> np.ndarray:
        return np.asarray(self.feature_mean, dtype=np.float32)

    @property
    def feature_std_array(self) -> np.ndarray:
        return np.asarray(self.feature_std, dtype=np.float32)

    @property
    def target_mean_array(self) -> np.ndarray | None:
        if self.target_mean is None:
            return None
        return np.asarray(self.target_mean, dtype=np.float32)

    @property
    def target_std_array(self) -> np.ndarray | None:
        if self.target_std is None:
            return None
        return np.asarray(self.target_std, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationStats":
        return cls(**payload)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def make_run_dir(base_dir: str | Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = run_name.replace(" ", "_")
    return ensure_dir(Path(base_dir) / f"{timestamp}_{safe_name}")


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_normalization(stats: NormalizationStats, path: str | Path) -> None:
    save_json(stats.to_dict(), path)


def load_normalization(path: str | Path) -> NormalizationStats:
    return NormalizationStats.from_dict(load_json(path))


def save_history_csv(history: list[dict[str, Any]], path: str | Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, Any]], path: str | Path) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row.get("val_loss") for row in history]
    preferred_metrics = ["val_mae_mean", "val_vector_mae", "val_rmse_mean", "val_vector_rmse"]
    monitor_name = next((name for name in preferred_metrics if name in history[0]), None)
    if monitor_name is None:
        monitor_name = next(
            (key for key in history[0] if key.startswith("val_") and key not in {"val_loss"}),
            None,
        )

    fig, axes = plt.subplots(1, 2 if monitor_name else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    axes[0].plot(epochs, train_loss, label="train_loss")
    if any(v is not None for v in val_loss):
        axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if monitor_name:
        train_metric_name = monitor_name.replace("val_", "train_")
        if train_metric_name in history[0]:
            axes[1].plot(epochs, [row[train_metric_name] for row in history], label=train_metric_name)
        axes[1].plot(epochs, [row[monitor_name] for row in history], label=monitor_name)
        axes[1].set_title("Metric")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def denormalize_targets(
    values: np.ndarray,
    normalization: NormalizationStats | None,
) -> np.ndarray:
    if normalization is None or normalization.target_mean is None or normalization.target_std is None:
        return values
    return values * normalization.target_std_array + normalization.target_mean_array


def write_predictions_csv(
    path: str | Path,
    metadata_rows: list[dict[str, Any]],
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
) -> None:
    fieldnames = [
        "session_name",
        "file_path",
        "window_start",
        "window_end",
        "target_start",
        "target_end",
        "start_time_s",
        "end_time_s",
    ]
    for name in target_names:
        fieldnames.append(f"pred_{name}")
    for name in target_names:
        fieldnames.append(f"target_{name}")
    for name in target_names:
        fieldnames.append(f"error_{name}")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, meta in enumerate(metadata_rows):
            row = {
                "session_name": meta["session_name"],
                "file_path": meta["file_path"],
                "window_start": meta["window_start"],
                "window_end": meta["window_end"],
                "target_start": meta["target_start"],
                "target_end": meta["target_end"],
                "start_time_s": meta["start_time_s"],
                "end_time_s": meta["end_time_s"],
            }
            for dim, name in enumerate(target_names):
                row[f"pred_{name}"] = float(predictions[idx, dim])
                row[f"target_{name}"] = float(targets[idx, dim])
                row[f"error_{name}"] = float(predictions[idx, dim] - targets[idx, dim])
            writer.writerow(row)


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def vector_norm(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(values), axis=1))


def flatten_metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    stack: list[tuple[str, Any]] = list(metrics.items())
    while stack:
        key, value = stack.pop(0)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                nested_key = f"{key}_{sub_key}" if key else str(sub_key)
                stack.append((nested_key, sub_value))
        elif isinstance(value, (int, float, np.floating)) and not math.isnan(float(value)):
            flat[key] = float(value)
    return flat


def file_list_to_strings(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    return [str(path) for path in paths]
