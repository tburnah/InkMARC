#!/usr/bin/env python
"""Train paired small location models with corrected vs uncorrected corners."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn  # noqa: E402
from inkdepth.models import build_model_from_config  # noqa: E402
from inkdepth.training.train import train  # noqa: E402
from inkdepth.training.validate import validate  # noqa: E402
from inkdepth.utils.checkpoints import load_checkpoint  # noqa: E402
from inkdepth.utils.config import load_config  # noqa: E402


DEFAULT_SOURCE = Path("/data/baseAVIs_432x672/manifest_original_no307_no321_frames_stride2_stable_w3.json")
DEFAULT_CONFIG = ROOT / "configs/final_rgb_geometry_location_base_plus_texture.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/corner_correction_ablation_small"
SRC_QUAD = np.float32([[0.0, 0.0], [479.0, 0.0], [479.0, 319.0], [0.0, 319.0]])
STATE_NAMES = {1: "hover", 2: "touch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train paired lightweight location models on identical frames, changing only "
            "corrected versus uncorrected training corners. Validation and test labels are corrected."
        )
    )
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--base-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-per-state", type=int, default=1500)
    parser.add_argument("--val-per-state", type=int, default=500)
    parser.add_argument("--test-per-state", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=8675309)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027, 4099])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", nargs=2, type=int, default=[336, 216], metavar=("H", "W"))
    parser.add_argument("--fpn-channels", type=int, default=48)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    for name in ("train_per_state", "val_per_state", "test_per_state", "epochs", "batch_size"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    source = Path(args.source_manifest).expanduser().resolve()
    base_config = Path(args.base_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _require_file(source, "source manifest")
    _require_file(base_config, "base config")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    manifests, subset_stats = build_paired_manifests(
        source,
        output_dir / "manifests",
        targets={
            "train": int(args.train_per_state),
            "val": int(args.val_per_state),
            "test": int(args.test_per_state),
        },
        seed=int(args.sample_seed),
    )

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for condition in ("corrected", "uncorrected_train"):
            run_dir = output_dir / f"seed_{seed}" / condition
            cfg = experiment_config(
                base_config,
                manifest=manifests[condition],
                output_dir=run_dir,
                seed=int(seed),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                input_size=tuple(int(x) for x in args.input_size),
                fpn_channels=int(args.fpn_channels),
            )
            checkpoint = run_dir / "best_xy.pt"
            if not (args.reuse_existing and checkpoint.is_file()):
                print(f"\n=== training seed={seed} condition={condition} ===", flush=True)
                train(cfg)
            else:
                print(f"\n=== reusing {checkpoint} ===", flush=True)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Training produced no best XY checkpoint: {checkpoint}")
            metrics = evaluate_test(cfg, checkpoint, run_dir / "test_corrected")
            scale_to_432x672 = 672.0 / float(args.input_size[0])
            row = {
                "seed": int(seed),
                "condition": condition,
                **metrics,
                "xy_mean_pix_432x672": metrics.get("xy_mean_pix", float("nan")) * scale_to_432x672,
                "xy_median_pix_432x672": metrics.get("xy_median_pix", float("nan")) * scale_to_432x672,
            }
            rows.append(row)

    report = build_report(args, source, base_config, manifests, subset_stats, rows)
    json_path = output_dir / "corner_correction_results.json"
    csv_path = output_dir / "corner_correction_results.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    print_summary(report)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def build_paired_manifests(
    source: Path,
    output_dir: Path,
    targets: dict[str, int],
    seed: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = [dict(row) for row in payload.get("frames", [])]
    if not frames:
        raise ValueError(f"Expected a non-empty top-level frames list: {source}")

    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    shifts: dict[tuple[str, int], tuple[float, float]] = {}
    by_h5: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        split = str(row.get("split", ""))
        state = int(row.get("state_hint", -1))
        if split in targets and state in STATE_NAMES:
            by_h5[str(row.get("label_h5_path", ""))].append(row)

    availability = defaultdict(int)
    for corrected_name, h5_rows in sorted(by_h5.items()):
        corrected_path = Path(corrected_name)
        uncorrected_path = uncorrected_sidecar(corrected_path)
        _require_file(corrected_path, "corrected sidecar")
        _require_file(uncorrected_path, "uncorrected sidecar")
        with h5py.File(corrected_path, "r") as corrected, h5py.File(uncorrected_path, "r") as uncorrected:
            if not (
                np.array_equal(corrected["labels"][:], uncorrected["labels"][:], equal_nan=True)
                and np.array_equal(corrected["label_mask"][:], uncorrected["label_mask"][:])
                and np.array_equal(corrected["timestamps_ns"][:], uncorrected["timestamps_ns"][:])
            ):
                raise RuntimeError(f"Sidecars differ in labels, masks, or timestamps: {corrected_path}")
            for row in h5_rows:
                frame_index = int(row["frame_index"])
                labels = np.asarray(corrected["labels"][frame_index], dtype=np.float32)
                mask = np.asarray(corrected["label_mask"][frame_index], dtype=np.uint8)
                if not (bool(mask[0]) and bool(mask[1]) and np.isfinite(labels[:2]).all()):
                    continue
                corrected_corners = np.asarray(corrected["corners"][frame_index], dtype=np.float32)
                uncorrected_corners = np.asarray(uncorrected["corners"][frame_index], dtype=np.float32)
                corrected_xy = project_label(labels, corrected_corners)
                uncorrected_xy = project_label(labels, uncorrected_corners)
                split = str(row["split"])
                state = int(row["state_hint"])
                if corrected_xy is None or uncorrected_xy is None:
                    availability[f"{split}/{STATE_NAMES[state]}/not_common_support"] += 1
                    continue
                out = dict(row)
                out["label_h5_path"] = str(corrected_path)
                candidates[(split, state)].append(out)
                key = (str(row["video_id"]), frame_index)
                label_shift = float(np.linalg.norm(corrected_xy - uncorrected_xy))
                corner_shift = float(np.mean(np.linalg.norm(corrected_corners - uncorrected_corners, axis=1)))
                shifts[key] = (label_shift, corner_shift)
                availability[f"{split}/{STATE_NAMES[state]}/common_support"] += 1

    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    sampled_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        for state in (1, 2):
            chosen = video_balanced_sample(candidates[(split, state)], targets[split], rng)
            if len(chosen) < targets[split]:
                raise RuntimeError(
                    f"Only {len(chosen)} common-support rows for {split}/{STATE_NAMES[state]}, "
                    f"requested {targets[split]}"
                )
            selected.extend(chosen)
            sampled_counts[f"{split}/{STATE_NAMES[state]}"] = len(chosen)
    rng.shuffle(selected)

    corrected_frames = [dict(row) for row in selected]
    uncorrected_train_frames = []
    for row in selected:
        out = dict(row)
        if str(out["split"]) == "train":
            out["label_h5_path"] = str(uncorrected_sidecar(Path(str(out["label_h5_path"]))))
        uncorrected_train_frames.append(out)

    selected_label_shifts = np.asarray(
        [shifts[(str(row["video_id"]), int(row["frame_index"]))][0] for row in selected], dtype=np.float64
    )
    selected_corner_shifts = np.asarray(
        [shifts[(str(row["video_id"]), int(row["frame_index"]))][1] for row in selected], dtype=np.float64
    )
    stats = {
        "source_manifest": str(source),
        "sample_seed": seed,
        "targets_per_split_state": targets,
        "sampled": sampled_counts,
        "availability": dict(sorted(availability.items())),
        "selected_label_shift_px": summarize_distribution(selected_label_shifts),
        "selected_mean_corner_shift_px": summarize_distribution(selected_corner_shifts),
        "protocol": (
            "Both training sets contain identical common-support frame keys. Only train-side label_h5_path "
            "changes. Validation and test use corrected sidecars in both conditions."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "corrected": output_dir / "corrected_all_splits.json",
        "uncorrected_train": output_dir / "uncorrected_train_corrected_eval.json",
    }
    for condition, condition_frames in (
        ("corrected", corrected_frames),
        ("uncorrected_train", uncorrected_train_frames),
    ):
        paths[condition].write_text(
            json.dumps({"frames": condition_frames, "corner_correction_ablation": stats}, indent=2),
            encoding="utf-8",
        )
    return paths, stats


def video_balanced_sample(rows: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_id"])].append(row)
    for video_rows in by_video.values():
        rng.shuffle(video_rows)
    videos = sorted(by_video)
    rng.shuffle(videos)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < target:
        added = False
        for video in videos:
            video_rows = by_video[video]
            if offset < len(video_rows):
                selected.append(video_rows[offset])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        offset += 1
    return selected


def project_label(labels: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    u = float(labels[0]) / 44800.0
    v = float(labels[1]) / 29600.0
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    transform = cv2.getPerspectiveTransform(SRC_QUAD, corners)
    projected = transform @ np.float64([u * 479.0, v * 319.0, 1.0])
    if abs(float(projected[2])) < 1e-8:
        return None
    xy = projected[:2] / projected[2]
    if not np.isfinite(xy).all() or xy[0] < 0 or xy[0] > 432 or xy[1] < 0 or xy[1] > 672:
        return None
    return xy.astype(np.float32)


def experiment_config(
    base_config: Path,
    manifest: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    input_size: tuple[int, int],
    fpn_channels: int,
) -> Any:
    cfg = deepcopy(load_config(base_config))
    cfg.seed = seed
    cfg.output_dir = str(output_dir)
    cfg.data.manifest_path = str(manifest)
    cfg.data.input_source = "rgb"
    cfg.data.input_size = list(input_size)
    cfg.data.variant_kind_filter = ["original"]
    cfg.data.surface_geometry.enabled = True
    cfg.augment.color_jitter = False
    cfg.augment.brightness = 0.0
    cfg.augment.contrast = 0.0
    cfg.augment.saturation = 0.0
    cfg.augment.hue = 0.0
    cfg.augment.blur_prob = 0.0
    cfg.augment.noise_std = 0.0

    cfg.model.name = "InkDepthStudent"
    cfg.model.input_channels = 6
    cfg.model.width_mult = 1.0
    cfg.model.pretrained = True
    cfg.model.fpn_channels = fpn_channels
    cfg.model.depth_out_size = [input_size[0] // 4, input_size[1] // 4]
    cfg.model.enable_depth_head = False
    cfg.model.use_depth_features_for_tasks = False
    cfg.model.state_context_mode = "gap"
    cfg.model.location_head_type = "heatmap"
    cfg.model.train_depth = False
    cfg.model.train_touch = False
    cfg.model.train_state = False
    cfg.model.train_location = True
    cfg.model.freeze_backbone = True
    cfg.model.freeze_neck = False
    cfg.model.freeze_depth_head = True
    cfg.model.freeze_state_head = True
    cfg.model.freeze_location_head = False
    cfg.model.freeze_presence_head = True

    cfg.loss.lambda_depth_ssi = 0.0
    cfg.loss.lambda_depth_l1 = 0.0
    cfg.loss.lambda_depth_roi_l1 = 0.0
    cfg.loss.lambda_depth_grad = 0.0
    cfg.loss.lambda_touch = 0.0
    cfg.loss.lambda_state = 0.0
    cfg.loss.lambda_xy = 1.0
    cfg.loss.lambda_heatmap = 0.5
    cfg.loss.xy_target = "all_valid_xy"

    cfg.training.epochs = epochs
    cfg.training.batch_size = batch_size
    cfg.training.num_workers = num_workers
    cfg.training.persistent_workers = num_workers > 0
    cfg.training.prefetch_factor = 2 if num_workers > 0 else None
    cfg.training.init_checkpoint = None
    cfg.training.resume = None
    cfg.training.sampler = "uniform"
    cfg.training.copy_manifest = False
    cfg.training.tensorboard = False
    cfg.training.save_visuals = False
    cfg.optimizer.backbone_lr = 0.0
    cfg.optimizer.head_lr = 0.0003
    cfg.scheduler.enabled = True
    cfg.scheduler.min_lr = 0.000001
    return cfg


def evaluate_test(cfg: Any, checkpoint: Path, output_dir: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)
    dataset = StylusDataset(cfg, split="test", training=False)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=stylus_collate_fn,
        persistent_workers=bool(cfg.training.num_workers),
    )
    return validate(model, loader, cfg, device, epoch=0, output_dir=output_dir)


def build_report(
    args: argparse.Namespace,
    source: Path,
    base_config: Path,
    manifests: dict[str, Path],
    subset_stats: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in ("corrected", "uncorrected_train")
    }
    aggregate = {}
    for condition, condition_rows in by_condition.items():
        aggregate[condition] = {}
        for metric in ("xy_mean_pix_432x672", "xy_median_pix_432x672", "xy_mean_norm", "xy_median_norm"):
            values = np.asarray([float(row[metric]) for row in condition_rows], dtype=np.float64)
            aggregate[condition][metric] = {
                "mean_across_seeds": float(values.mean()),
                "sample_std_across_seeds": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    paired = []
    for seed in args.seeds:
        corrected = next(row for row in rows if row["seed"] == seed and row["condition"] == "corrected")
        uncorrected = next(row for row in rows if row["seed"] == seed and row["condition"] == "uncorrected_train")
        paired.append(
            {
                "seed": int(seed),
                "mean_xy_improvement_px": float(
                    uncorrected["xy_mean_pix_432x672"] - corrected["xy_mean_pix_432x672"]
                ),
                "median_xy_improvement_px": float(
                    uncorrected["xy_median_pix_432x672"] - corrected["xy_median_pix_432x672"]
                ),
            }
        )
    mean_improvements = np.asarray([row["mean_xy_improvement_px"] for row in paired], dtype=np.float64)
    median_improvements = np.asarray([row["median_xy_improvement_px"] for row in paired], dtype=np.float64)
    return {
        "study": "corrected_vs_uncorrected_training_corners",
        "source_manifest": str(source),
        "base_config": str(base_config),
        "manifests": {key: str(value) for key, value in manifests.items()},
        "seeds": [int(seed) for seed in args.seeds],
        "epochs": int(args.epochs),
        "input_size": list(args.input_size),
        "model_protocol": "ImageNet MobileNetV3-Large backbone frozen; FPN and location head trained from identical seeds.",
        "evaluation_protocol": "Both conditions selected and evaluated on the identical corrected validation/test labels.",
        "subset": subset_stats,
        "runs": rows,
        "aggregate": aggregate,
        "paired_improvement_positive_means_corrected_is_better": {
            "by_seed": paired,
            "mean_xy_improvement_px_mean": float(mean_improvements.mean()),
            "mean_xy_improvement_px_sample_std": float(mean_improvements.std(ddof=1)) if len(paired) > 1 else 0.0,
            "median_xy_improvement_px_mean": float(median_improvements.mean()),
            "median_xy_improvement_px_sample_std": float(median_improvements.std(ddof=1)) if len(paired) > 1 else 0.0,
            "corrected_wins_mean_error_seeds": int((mean_improvements > 0).sum()),
            "corrected_wins_median_error_seeds": int((median_improvements > 0).sum()),
        },
    }


def summarize_distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(report: dict[str, Any]) -> None:
    print("\ncondition          mean_xy_px  median_xy_px")
    for condition in ("corrected", "uncorrected_train"):
        metrics = report["aggregate"][condition]
        print(
            f"{condition:<18} "
            f"{metrics['xy_mean_pix_432x672']['mean_across_seeds']:>10.4f}  "
            f"{metrics['xy_median_pix_432x672']['mean_across_seeds']:>12.4f}"
        )
    improvement = report["paired_improvement_positive_means_corrected_is_better"]
    print(
        "corrected improvement: "
        f"mean error {improvement['mean_xy_improvement_px_mean']:+.4f}px, "
        f"median error {improvement['median_xy_improvement_px_mean']:+.4f}px"
    )


def uncorrected_sidecar(corrected: Path) -> Path:
    suffix = "_corrected.h5"
    if not corrected.name.endswith(suffix):
        raise ValueError(f"Expected corrected H5 filename ending in {suffix}: {corrected}")
    return corrected.with_name(corrected.name[: -len(suffix)] + ".h5")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label.capitalize()} not found: {path}")


if __name__ == "__main__":
    main()
