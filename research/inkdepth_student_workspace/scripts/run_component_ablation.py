#!/usr/bin/env python
"""Run component ablations on the final RGBD+geometry student.

The trained network is held fixed. For each variant, excluded input paths are
disconnected by zeroing their first-layer weights. When depth is excluded, the
five heatmap-conditioned depth-statistic inputs to the task heads are also
disconnected. This avoids treating a raw zero tensor as a neutral input after
the model's per-channel normalization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn  # noqa: E402
from inkdepth.models import build_model_from_config  # noqa: E402
from inkdepth.training.validate import validate  # noqa: E402
from inkdepth.utils.checkpoints import load_checkpoint  # noqa: E402
from inkdepth.utils.config import load_config, parse_overrides  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/inkmarc_rgbd_geometry_presence_touch_location_export_best_state.yaml"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/inkmarc_rgbd_geometry_presence_touch_location_export_best_state"
    / "inkmarc_rgbd_geometry_presence_touch_location_best_state.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/component_ablation_best_final"


@dataclass(frozen=True)
class Variant:
    keep_rgb: bool
    keep_depth: bool
    keep_geometry: bool


VARIANTS = {
    "rgb": Variant(keep_rgb=True, keep_depth=False, keep_geometry=False),
    "rgb_geometry": Variant(keep_rgb=True, keep_depth=False, keep_geometry=True),
    "rgb_depth": Variant(keep_rgb=True, keep_depth=True, keep_geometry=False),
    "full": Variant(keep_rgb=True, keep_depth=True, keep_geometry=True),
    "geometry_only": Variant(keep_rgb=False, keep_depth=False, keep_geometry=True),
    "depth_only": Variant(keep_rgb=False, keep_depth=True, keep_geometry=False),
}
DEFAULT_VARIANTS = ["rgb", "rgb_geometry", "rgb_depth", "full"]

SUMMARY_METRICS = (
    "touch_accuracy",
    "touch_precision",
    "touch_recall",
    "touch_f1",
    "touch_best_f1",
    "state_accuracy",
    "state_f1_macro",
    "xy_mean_pix",
    "xy_median_pix",
)


class ZeroDepthBackbone(torch.nn.Module):
    """Skip live depth inference when all depth task paths are disconnected."""

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb.new_zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RGB, RGB+geometry, RGB+depth, and full inputs on the "
            "best final RGBD+geometry model."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=DEFAULT_VARIANTS,
        help="Variants to evaluate (default: rgb rgb_geometry rgb_depth full).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N split samples; intended for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, or 'auto' to select CUDA when available.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a variant's existing metrics_latest.json instead of evaluating it again.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _require_file(config_path, "config")
    _require_file(checkpoint_path, "checkpoint")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    overrides = parse_overrides(args.override)
    overrides["training.batch_size"] = args.batch_size
    overrides["training.num_workers"] = args.num_workers
    cfg = load_config(config_path, overrides)
    _validate_ablation_config(cfg)

    manifest_path = Path(str(cfg.data.manifest_path)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()
    _require_file(manifest_path, "dataset manifest")

    device = _resolve_device(args.device)
    _seed_everything(int(cfg.seed))
    dataset: torch.utils.data.Dataset = StylusDataset(cfg, split=args.split, training=False)
    full_split_size = len(dataset)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, full_split_size)))
    if len(dataset) == 0:
        raise RuntimeError(f"Manifest has no samples for split={args.split!r}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=stylus_collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for name in args.variants:
        print(f"\n=== {name}: {_retained_components(VARIANTS[name])} ===", flush=True)
        variant_dir = output_dir / name
        if args.reuse_existing and (variant_dir / "metrics_latest.json").is_file():
            results[name] = _load_existing_metrics(variant_dir / "metrics_latest.json")
            print(f"Reused {variant_dir / 'metrics_latest.json'}", flush=True)
            continue
        model = build_model_from_config(cfg).to(device).eval()
        load_checkpoint(checkpoint_path, model, map_location=device)
        disconnect_excluded_modalities(model, VARIANTS[name])
        metrics = validate(model, loader, cfg, device, epoch=0, output_dir=variant_dir)
        results[name] = {key: float(value) for key, value in metrics.items()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "study": "component_addition_to_rgb_baseline",
        "ablation_method": "disconnect_excluded_encoder_channels_and_depth_context_weights",
        "interpretation": (
            "Fixed-weight inference ablation comparing RGB, RGB+geometry, RGB+depth, "
            "and all inputs. This is a pathway-necessity test, not a comparison of "
            "separately retrained modality-specific models."
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "manifest": str(manifest_path),
        "split": args.split,
        "split_samples": full_split_size,
        "evaluated_samples": len(dataset),
        "device": str(device),
        "seed": int(cfg.seed),
        "reused_existing_metrics": bool(args.reuse_existing),
        "channel_layout": {"rgb": [0, 1, 2], "depth": [3], "geometry": [4, 5, 6]},
        "variants": {
            name: {
                "retained_components": _retained_component_list(VARIANTS[name]),
                "metrics": results[name],
                "change_vs_rgb": _metric_changes(results[name], results.get("rgb")),
                "change_vs_full": _metric_changes(results[name], results.get("full")),
            }
            for name in args.variants
        },
    }
    json_path = output_dir / "ablation_results.json"
    csv_path = output_dir / "ablation_results.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_summary_csv(csv_path, args.variants, results)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    _print_summary(args.variants, results)


def disconnect_excluded_modalities(model: torch.nn.Module, variant: Variant) -> None:
    """Disconnect excluded task inputs without changing kept input activations."""
    student = getattr(model, "student", model)
    backbone = getattr(student, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model has no student MobileNet backbone")
    first_conv = backbone[0][0]
    if not isinstance(first_conv, torch.nn.Conv2d) or first_conv.in_channels != 7:
        raise RuntimeError(
            "Component ablation requires the final seven-channel layout "
            "[RGB(3), depth(1), geometry(3)]; "
            f"found {getattr(first_conv, 'in_channels', None)} channels."
        )

    with torch.no_grad():
        if not variant.keep_rgb:
            first_conv.weight[:, 0:3].zero_()
        if not variant.keep_depth:
            first_conv.weight[:, 3:4].zero_()
            _disconnect_depth_context(student)
        if not variant.keep_geometry:
            first_conv.weight[:, 4:7].zero_()

    if not variant.keep_depth and hasattr(model, "da3_depth"):
        model.da3_depth = ZeroDepthBackbone().to(next(model.parameters()).device)


def _disconnect_depth_context(student: torch.nn.Module) -> None:
    if getattr(student, "state_context_mode", None) != "gap_xy_depth":
        return
    heads = [getattr(student, "state_head", None), getattr(student, "presence_head", None)]
    for head in heads:
        if head is None:
            continue
        layer = head.net[0]
        if not isinstance(layer, torch.nn.Linear) or layer.in_features < 5:
            raise RuntimeError("Could not identify the task head's five depth-statistic inputs")
        layer.weight[:, -5:].zero_()


def _validate_ablation_config(cfg: Any) -> None:
    model_name = str(getattr(cfg.model, "name", ""))
    if model_name not in {"InkDepthStudent", "InkDepthDA3Student"}:
        raise ValueError(f"Unsupported model for component ablation: {model_name}")
    geometry_cfg = getattr(cfg.data, "surface_geometry", None)
    if not bool(getattr(geometry_cfg, "enabled", False)):
        raise ValueError("data.surface_geometry.enabled must be true")
    if model_name == "InkDepthStudent":
        if str(getattr(cfg.data, "input_source", "")).lower() not in {"rgbd", "rgb_depth", "rgb+depth"}:
            raise ValueError("Cached student ablation requires data.input_source=rgbd")
        if int(getattr(cfg.model, "input_channels", 0)) != 7:
            raise ValueError("Cached student ablation requires model.input_channels=7")


def _metric_changes(metrics: dict[str, float], full: dict[str, float] | None) -> dict[str, float]:
    if full is None:
        return {}
    return {key: value - full[key] for key, value in metrics.items() if key in full}


def _load_existing_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("frame", payload)
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))
    }


def _write_summary_csv(path: Path, names: list[str], results: dict[str, dict[str, float]]) -> None:
    fields = ["variant", "retained_components"]
    for metric in SUMMARY_METRICS:
        fields.extend([metric, f"{metric}_change_vs_rgb", f"{metric}_change_vs_full"])
    rgb = results.get("rgb")
    full = results.get("full")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in names:
            row: dict[str, Any] = {
                "variant": name,
                "retained_components": _retained_components(VARIANTS[name]),
            }
            for metric in SUMMARY_METRICS:
                value = results[name].get(metric)
                row[metric] = "" if value is None else value
                row[f"{metric}_change_vs_rgb"] = (
                    "" if value is None or rgb is None or metric not in rgb else value - rgb[metric]
                )
                row[f"{metric}_change_vs_full"] = (
                    "" if value is None or full is None or metric not in full else value - full[metric]
                )
            writer.writerow(row)


def _print_summary(names: list[str], results: dict[str, dict[str, float]]) -> None:
    print("\nvariant       touch_f1  state_f1_macro  xy_mean_pix")
    for name in names:
        metrics = results[name]
        print(
            f"{name:<13} "
            f"{_format_metric(metrics.get('touch_f1')):>8}  "
            f"{_format_metric(metrics.get('state_f1_macro')):>14}  "
            f"{_format_metric(metrics.get('xy_mean_pix')):>11}"
        )


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _retained_component_list(variant: Variant) -> list[str]:
    return [
        name
        for name, keep in (
            ("rgb", variant.keep_rgb),
            ("depth", variant.keep_depth),
            ("geometry", variant.keep_geometry),
        )
        if keep
    ]


def _retained_components(variant: Variant) -> str:
    return "+".join(_retained_component_list(variant))


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label.capitalize()} not found: {path}")


if __name__ == "__main__":
    main()
