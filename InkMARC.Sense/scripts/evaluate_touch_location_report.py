#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import build_model_from_config
from inkdepth.training.metrics import xy_metric_spec, xy_valid_mask
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate touch classification and XY error on a held-out split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--raw-threshold", type=float, default=0.30)
    parser.add_argument("--hyst-on", type=float, default=0.15)
    parser.add_argument("--hyst-off", type=float, default=0.05)
    parser.add_argument("--hyst-min-on", type=int, default=1)
    parser.add_argument("--hyst-min-off", type=int, default=1)
    parser.add_argument("--max-episode-gap", type=int, default=4)
    parser.add_argument(
        "--ablate-depth",
        action="store_true",
        help="Disconnect depth-only input paths from a copy of the loaded model.",
    )
    parser.add_argument(
        "--ablate-geometry",
        action="store_true",
        help="Disconnect surface-geometry input channels from a copy of the loaded model.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    if args.ablate_depth:
        ablate_depth_paths(model)
    if args.ablate_geometry:
        ablate_geometry_paths(model)

    ds = StylusDataset(cfg, split=args.split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stylus_collate_fn,
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", False)) and int(cfg.training.num_workers) > 0,
    )

    rows = collect_rows(model, loader, cfg, device)
    if not rows:
        raise RuntimeError("No rows were produced. Check the manifest, split, and config.")

    rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))
    write_csv(output_dir / "touch_location_frame_predictions.csv", rows)

    xy_by_touch = {
        "touch_true": xy_summary(rows, touch_true=1),
        "touch_false": xy_summary(rows, touch_true=0),
    }
    y = [int(row["touch_true"]) for row in rows if row["has_touch"]]
    raw_pred = [int(float(row["touch_prob"]) >= float(args.raw_threshold)) for row in rows if row["has_touch"]]
    raw_metrics = binary_metrics(y, raw_pred)

    hyst_pred_all = apply_hysteresis(
        rows,
        on_threshold=float(args.hyst_on),
        off_threshold=float(args.hyst_off),
        min_on_samples=int(args.hyst_min_on),
        min_off_samples=int(args.hyst_min_off),
    )
    hyst_pred = [pred for row, pred in zip(rows, hyst_pred_all) if row["has_touch"]]
    hyst_metrics = binary_metrics(y, hyst_pred)
    hyst_metrics.update(episode_metrics(rows, hyst_pred_all, max_gap=int(args.max_episode_gap)))

    report = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "depth_ablation": "disconnected_encoder_and_context_paths" if args.ablate_depth else "none",
        "geometry_ablation": "disconnected_encoder_channels" if args.ablate_geometry else "none",
        "split": args.split,
        "rows": len(rows),
        "valid_touch_rows": len(y),
        "xy_by_touch": xy_by_touch,
        "raw_threshold": {"threshold": float(args.raw_threshold), **raw_metrics},
        "hysteresis": {
            "on_threshold": float(args.hyst_on),
            "off_threshold": float(args.hyst_off),
            "min_on_samples": int(args.hyst_min_on),
            "min_off_samples": int(args.hyst_min_off),
            **hyst_metrics,
        },
    }
    (output_dir / "touch_location_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report_text = format_report(report)
    (output_dir / "touch_location_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)


class ZeroDepthBackbone(torch.nn.Module):
    """Skip depth inference after all depth-only task paths have been disconnected."""

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb.new_zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))


def ablate_depth_paths(model: torch.nn.Module) -> None:
    """Remove depth contributions while preserving RGB and surface-geometry paths."""
    if not hasattr(model, "da3_depth") or not hasattr(model, "student"):
        raise RuntimeError("--ablate-depth requires a DA3 student model.")
    student = model.student
    first_conv = student.backbone[0][0]
    if not isinstance(first_conv, torch.nn.Conv2d) or first_conv.in_channels < 4:
        raise RuntimeError("Could not identify the student's external depth input channel.")
    with torch.no_grad():
        first_conv.weight[:, 3].zero_()
        if student.state_context_mode == "gap_xy_depth":
            state_input = student.state_head.net[0]
            if not isinstance(state_input, torch.nn.Linear) or state_input.in_features < 5:
                raise RuntimeError("Could not identify the touch head's depth-statistic inputs.")
            state_input.weight[:, -5:].zero_()
            if student.presence_head is not None:
                presence_input = student.presence_head.net[0]
                presence_input.weight[:, -5:].zero_()
    model.da3_depth = ZeroDepthBackbone().to(next(model.parameters()).device)


def ablate_geometry_paths(model: torch.nn.Module) -> None:
    """Remove surface-mask and surface-coordinate contributions from the encoder."""
    if not hasattr(model, "student"):
        raise RuntimeError("--ablate-geometry requires a DA3 student model.")
    first_conv = model.student.backbone[0][0]
    if not isinstance(first_conv, torch.nn.Conv2d) or first_conv.in_channels <= 4:
        raise RuntimeError("Could not identify the student's surface-geometry input channels.")
    with torch.no_grad():
        first_conv.weight[:, 4:].zero_()


@torch.no_grad()
def collect_rows(model: torch.nn.Module, loader: DataLoader, cfg: Any, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xy_scale, _ = xy_metric_spec(cfg)
    scale = np.asarray(xy_scale, dtype=np.float32)
    for batch in tqdm(loader, desc="predict"):
        batch = {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}
        outputs = model(batch["image"])
        touch_logits = outputs.get("touch_logits")
        xy_pred = outputs.get("xy")
        if touch_logits is None:
            raise RuntimeError("Model produced no touch_logits.")
        if xy_pred is None:
            raise RuntimeError("Model produced no xy output.")
        probs = torch.sigmoid(touch_logits).detach().cpu().numpy().reshape(-1)
        xy_pred_np = xy_pred.detach().cpu().numpy()
        xy_true_np = batch["xy"].detach().cpu().numpy()
        xy_valid_np = xy_valid_mask(batch, cfg).detach().cpu().numpy().astype(bool)
        bsz = len(probs)
        for i in range(bsz):
            has_touch = bool(batch["has_touch"][i].detach().cpu())
            touch_true = int(float(batch["touch"][i].detach().cpu()) > 0.5) if has_touch else None
            valid_xy = bool(xy_valid_np[i])
            diff = xy_pred_np[i] - xy_true_np[i]
            row: dict[str, Any] = {
                "video_id": str(batch["video_id"][i]),
                "participant_id": str(batch["participant_id"][i]),
                "task_id": str(batch["task_id"][i]),
                "frame_index": int(batch["frame_index"][i].detach().cpu()),
                "has_touch": has_touch,
                "touch_true": touch_true,
                "touch_prob": float(probs[i]),
                "valid_xy": valid_xy,
                "xy_pred_x": "",
                "xy_pred_y": "",
                "xy_true_x": "",
                "xy_true_y": "",
                "xy_l1_norm": "",
                "xy_l2_norm": "",
                "xy_err_pix": "",
            }
            if valid_xy:
                row["xy_pred_x"] = float(xy_pred_np[i][0])
                row["xy_pred_y"] = float(xy_pred_np[i][1])
                row["xy_true_x"] = float(xy_true_np[i][0])
                row["xy_true_y"] = float(xy_true_np[i][1])
                row["xy_l1_norm"] = float(np.mean(np.abs(diff)))
                row["xy_l2_norm"] = float(np.linalg.norm(diff))
                row["xy_err_pix"] = float(np.linalg.norm(diff * scale))
            rows.append(row)
    return rows


def xy_summary(rows: list[dict[str, Any]], touch_true: int) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["has_touch"] and row["touch_true"] == touch_true and row["valid_xy"] and row["xy_l2_norm"] != ""
    ]
    if not selected:
        return {"count": 0}
    pix = np.asarray([float(row["xy_err_pix"]) for row in selected], dtype=np.float64)
    l1 = np.asarray([float(row["xy_l1_norm"]) for row in selected], dtype=np.float64)
    l2 = np.asarray([float(row["xy_l2_norm"]) for row in selected], dtype=np.float64)
    return {
        "count": int(len(selected)),
        "mean_euclidean_px": float(np.mean(pix)),
        "median_euclidean_px": float(np.median(pix)),
        "normalized_l1": float(np.mean(l1)),
        "normalized_l2": float(np.mean(l2)),
    }


def apply_hysteresis(
    rows: list[dict[str, Any]],
    on_threshold: float,
    off_threshold: float,
    min_on_samples: int,
    min_off_samples: int,
) -> list[int]:
    preds = [0] * len(rows)
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[str(row["video_id"])].append(i)
    for indices in grouped.values():
        active = False
        on_count = 0
        off_count = 0
        for index in indices:
            prob = float(rows[index]["touch_prob"])
            if active:
                if prob < off_threshold:
                    off_count += 1
                else:
                    off_count = 0
                if off_count >= min_off_samples:
                    active = False
                    off_count = 0
                on_count = 0
            else:
                if prob >= on_threshold:
                    on_count += 1
                else:
                    on_count = 0
                if on_count >= min_on_samples:
                    active = True
                    on_count = 0
                off_count = 0
            preds[index] = int(active)
    return preds


def binary_metrics(y: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 1)
    tn = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 0)
    fp = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 1)
    fn = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count": len(y),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def episode_metrics(rows: list[dict[str, Any]], pred: list[int], max_gap: int) -> dict[str, Any]:
    truth = [int(row["touch_true"]) if row["has_touch"] else 0 for row in rows]
    truth_episodes = collect_episodes(rows, truth, max_gap)
    pred_episodes = collect_episodes(rows, pred, max_gap)
    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    truth_by_video: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, episode in enumerate(truth_episodes):
        truth_by_video[episode["video_id"]].append((i, episode))
    for pred_i, pred_episode in enumerate(pred_episodes):
        for truth_i, truth_episode in truth_by_video.get(pred_episode["video_id"], []):
            if intervals_overlap(pred_episode, truth_episode):
                matched_pred.add(pred_i)
                matched_truth.add(truth_i)
                break
    precision = len(matched_pred) / max(1, len(pred_episodes))
    recall = len(matched_truth) / max(1, len(truth_episodes))
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "episode_precision": precision,
        "episode_recall": recall,
        "episode_f1": f1,
        "true_episodes": len(truth_episodes),
        "pred_episodes": len(pred_episodes),
    }


def collect_episodes(rows: list[dict[str, Any]], values: list[int], max_gap: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row, value in zip(rows, values):
        video_id = str(row["video_id"])
        frame_index = int(row["frame_index"])
        if value:
            if current is None or current["video_id"] != video_id or frame_index - int(current["end"]) > max_gap:
                if current is not None:
                    episodes.append(current)
                current = {"video_id": video_id, "start": frame_index, "end": frame_index}
            else:
                current["end"] = frame_index
        elif current is not None:
            episodes.append(current)
            current = None
    if current is not None:
        episodes.append(current)
    return episodes


def intervals_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return str(a["video_id"]) == str(b["video_id"]) and int(a["start"]) <= int(b["end"]) and int(b["start"]) <= int(a["end"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_report(report: dict[str, Any]) -> str:
    touch_true = report["xy_by_touch"]["touch_true"]
    touch_false = report["xy_by_touch"]["touch_false"]
    raw = report["raw_threshold"]
    hyst = report["hysteresis"]
    lines = [
        "Touch True",
        f"Mean Euclidean error:      {touch_true.get('mean_euclidean_px', 0.0):.2f} px",
        f"Median Euclidean error:    {touch_true.get('median_euclidean_px', 0.0):.2f} px",
        f"Normalised L1 error:       {touch_true.get('normalized_l1', 0.0):.6f}",
        f"Normalised L2 error:       {touch_true.get('normalized_l2', 0.0):.6f}",
        "Touch False",
        f"Mean Euclidean error:       {touch_false.get('mean_euclidean_px', 0.0):.2f} px",
        f"Median Euclidean error:     {touch_false.get('median_euclidean_px', 0.0):.2f} px",
        f"Normalised L1 error:        {touch_false.get('normalized_l1', 0.0):.6f}",
        f"Normalised L2 error:        {touch_false.get('normalized_l2', 0.0):.6f}",
        "",
        f"Touch-state performance was evaluated on the held-out {report['split']} set. This included {raw['count']:,} rows.",
        (
            f"A raw threshold of {raw['threshold']:.2f} produced an accuracy of {raw['accuracy']:.3f}, "
            f"precision of {raw['precision']:.3f}, recall of {raw['recall']:.3f}, and F1 score of {raw['f1']:.3f}."
        ),
        (
            "Applying hysteresis produced "
            f"accuracy={hyst['accuracy']:.3f}, precision={hyst['precision']:.3f}, "
            f"recall={hyst['recall']:.3f}, frame-level F1={hyst['f1']:.3f}, "
            f"and episode-level F1={hyst['episode_f1']:.3f}."
        ),
        "",
        "Method,Test rows,Accuracy,Precision,Recall,Frame F1,Episode F1,Parameters",
        (
            f"Raw threshold,{raw['count']},{raw['accuracy']:.3f},{raw['precision']:.3f},"
            f"{raw['recall']:.3f},{raw['f1']:.3f},,threshold = {raw['threshold']:.2f}"
        ),
        (
            f"Hysteresis,{hyst['count']},{hyst['accuracy']:.3f},{hyst['precision']:.3f},"
            f"{hyst['recall']:.3f},{hyst['f1']:.3f},{hyst['episode_f1']:.3f},"
            f"on = {hyst['on_threshold']:.2f}, off = {hyst['off_threshold']:.2f}, "
            f"min_on = {hyst['min_on_samples']}, min_off = {hyst['min_off_samples']}"
        ),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
