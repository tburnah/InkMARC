#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import build_model_from_config
from inkdepth.training.train import build_sampler
from inkdepth.training.metrics import xy_metric_spec
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


class ResidualRefiner(nn.Module):
    def __init__(self, in_channels: int, max_residual_px: float, scale_xy: tuple[float, float]) -> None:
        super().__init__()
        self.register_buffer(
            "max_residual_norm",
            torch.tensor(
                [float(max_residual_px) / float(scale_xy[0]), float(max_residual_px) / float(scale_xy[1])],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(64, 2),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(crop)) * self.max_residual_norm.to(dtype=crop.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen-base local residual XY refiner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--refiner-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--crop-span-px", type=float, default=96.0)
    parser.add_argument("--jitter-px", type=float, default=16.0)
    parser.add_argument("--max-residual-px", type=float, default=48.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--touch-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampler", default="touch_only")
    parser.add_argument("--raw-threshold", type=float, default=0.28)
    parser.add_argument("--hyst-on", type=float, default=0.60)
    parser.add_argument("--hyst-off", type=float, default=0.10)
    parser.add_argument("--hyst-min-on", type=int, default=1)
    parser.add_argument("--hyst-min-off", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(getattr(cfg, "seed", 1337)))

    if args.batch_size is not None:
        cfg.training.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.training.num_workers = int(args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(args.base_checkpoint, base, map_location=device)
    for param in base.parameters():
        param.requires_grad_(False)

    scale_xy, _ = xy_metric_spec(cfg)
    train_loader = make_loader(cfg, "train", training=True, shuffle=True, sampler_mode=str(args.sampler))
    val_loader = make_loader(cfg, "val", training=False, shuffle=False)

    sample = next(iter(train_loader))
    in_channels = int(sample["image"].shape[1]) + 1
    refiner = ResidualRefiner(in_channels, args.max_residual_px, scale_xy).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.refiner_checkpoint:
        load_refiner(Path(args.refiner_checkpoint), refiner, map_location=device)
    if args.eval_only:
        evaluate_test(base, refiner, cfg, args, device, scale_xy, out_dir)
        return

    best_median = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(base, refiner, train_loader, optimizer, scaler, cfg, args, device, scale_xy)
        val_metrics = evaluate_refiner(base, refiner, val_loader, cfg, args, device, scale_xy, split="val")
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        write_metrics_csv(out_dir / "metrics.csv", history)
        save_payload(out_dir / "latest_refiner.pt", refiner, optimizer, epoch, row, args, cfg)
        if val_metrics.get("refined_touch_true_median_px", float("inf")) < best_median:
            best_median = float(val_metrics["refined_touch_true_median_px"])
            save_payload(out_dir / "best_refiner.pt", refiner, optimizer, epoch, row, args, cfg)
        print(json.dumps(row, indent=2, allow_nan=False))

    best_path = out_dir / "best_refiner.pt"
    if best_path.exists():
        load_refiner(best_path, refiner, map_location=device)
    evaluate_test(base, refiner, cfg, args, device, scale_xy, out_dir)


def evaluate_test(
    base: nn.Module,
    refiner: nn.Module,
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
    scale_xy: tuple[float, float],
    out_dir: Path,
) -> None:
    test_loader = make_loader(cfg, "test", training=False, shuffle=False)
    test_rows, report = collect_report(
        base,
        refiner,
        test_loader,
        cfg,
        args,
        device,
        scale_xy,
        split="test",
        max_batches=int(args.max_test_batches),
    )
    write_rows(out_dir / "test_refined_predictions.csv", test_rows)
    (out_dir / "test_refined_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    text = format_report(report)
    (out_dir / "test_refined_report.txt").write_text(text, encoding="utf-8")
    print(text)


def make_loader(cfg: Any, split: str, training: bool, shuffle: bool, sampler_mode: str | None = None) -> DataLoader:
    ds = StylusDataset(cfg, split=split, training=training)
    sampler = None
    if training and sampler_mode and sampler_mode != "uniform":
        old_sampler = getattr(cfg.training, "sampler", "uniform")
        cfg.training.sampler = sampler_mode
        sampler = build_sampler(ds, cfg)
        cfg.training.sampler = old_sampler
    num_workers = int(cfg.training.num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.training.batch_size),
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": stylus_collate_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg.training, "persistent_workers", False))
    return DataLoader(ds, **kwargs)


def train_one_epoch(
    base: nn.Module,
    refiner: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
    scale_xy: tuple[float, float],
) -> dict[str, float]:
    refiner.train()
    losses: list[float] = []
    base_errors: list[float] = []
    refined_errors: list[float] = []
    used = 0
    scale = torch.tensor(scale_xy, device=device, dtype=torch.float32)
    for step, batch in enumerate(tqdm(loader, desc="train_refiner")):
        if args.max_train_steps and step >= int(args.max_train_steps):
            break
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = base(batch["image"])
            base_xy = outputs["xy"].detach()
            depth = outputs.get("depth")
            if depth is None:
                raise RuntimeError("Base model did not produce depth for refiner crops.")
        mask = valid_mask(batch, args)
        if not mask.any():
            continue
        center_xy = jitter_xy(base_xy, args.jitter_px, scale).clamp(0.0, 1.0)
        crop = crop_around_xy(torch.cat([batch["image"], depth.detach()], dim=1), center_xy, args.crop_size, args.crop_span_px)
        target_residual = batch["xy"] - center_xy
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            pred_residual = refiner(crop)
            pixel_err = (pred_residual[mask] - target_residual[mask]) * scale
            loss = F.smooth_l1_loss(pixel_err, torch.zeros_like(pixel_err), beta=3.0)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            refined_xy = (center_xy + pred_residual).clamp(0.0, 1.0)
            base_pix = torch.linalg.norm((base_xy[mask] - batch["xy"][mask]) * scale, dim=1)
            refined_pix = torch.linalg.norm((refined_xy[mask] - batch["xy"][mask]) * scale, dim=1)
        losses.append(float(loss.detach().cpu()))
        base_errors.extend(base_pix.detach().cpu().tolist())
        refined_errors.extend(refined_pix.detach().cpu().tolist())
        used += int(mask.sum().detach().cpu())
    return summarize_errors(losses, base_errors, refined_errors, used)


@torch.no_grad()
def evaluate_refiner(
    base: nn.Module,
    refiner: nn.Module,
    loader: DataLoader,
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
    scale_xy: tuple[float, float],
    split: str,
) -> dict[str, float]:
    _, report = collect_report(
        base,
        refiner,
        loader,
        cfg,
        args,
        device,
        scale_xy,
        split=split,
        max_batches=int(args.max_val_batches),
    )
    return {
        "base_touch_true_mean_px": report["base_xy_by_touch"]["touch_true"]["mean_euclidean_px"],
        "base_touch_true_median_px": report["base_xy_by_touch"]["touch_true"]["median_euclidean_px"],
        "refined_touch_true_mean_px": report["refined_xy_by_touch"]["touch_true"]["mean_euclidean_px"],
        "refined_touch_true_median_px": report["refined_xy_by_touch"]["touch_true"]["median_euclidean_px"],
    }


@torch.no_grad()
def collect_report(
    base: nn.Module,
    refiner: nn.Module,
    loader: DataLoader,
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
    scale_xy: tuple[float, float],
    split: str,
    max_batches: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base.eval()
    refiner.eval()
    scale_np = np.asarray(scale_xy, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval_{split}")):
        if max_batches and batch_idx >= int(max_batches):
            break
        batch = to_device(batch, device)
        outputs = base(batch["image"])
        base_xy = outputs["xy"].detach()
        depth = outputs.get("depth")
        if depth is None:
            raise RuntimeError("Base model did not produce depth for refiner crops.")
        crop = crop_around_xy(torch.cat([batch["image"], depth.detach()], dim=1), base_xy, args.crop_size, args.crop_span_px)
        residual = refiner(crop)
        refined_xy = (base_xy + residual).clamp(0.0, 1.0)
        probs = torch.sigmoid(outputs["touch_logits"]).detach().cpu().numpy().reshape(-1)
        base_np = base_xy.detach().cpu().numpy()
        refined_np = refined_xy.detach().cpu().numpy()
        true_np = batch["xy"].detach().cpu().numpy()
        valid_np = batch["valid_xy"].detach().cpu().numpy().astype(bool)
        for i in range(len(probs)):
            has_touch = bool(batch["has_touch"][i].detach().cpu())
            touch_true = int(float(batch["touch"][i].detach().cpu()) > 0.5) if has_touch else None
            valid_xy = bool(valid_np[i])
            row: dict[str, Any] = {
                "video_id": str(batch["video_id"][i]),
                "participant_id": str(batch["participant_id"][i]),
                "task_id": str(batch["task_id"][i]),
                "frame_index": int(batch["frame_index"][i].detach().cpu()),
                "has_touch": has_touch,
                "touch_true": touch_true,
                "touch_prob": float(probs[i]),
                "valid_xy": valid_xy,
                "base_x": "",
                "base_y": "",
                "refined_x": "",
                "refined_y": "",
                "true_x": "",
                "true_y": "",
                "base_err_px": "",
                "refined_err_px": "",
                "residual_px_x": float(residual[i, 0].detach().cpu()) * float(scale_np[0]),
                "residual_px_y": float(residual[i, 1].detach().cpu()) * float(scale_np[1]),
            }
            if valid_xy:
                base_diff = base_np[i] - true_np[i]
                refined_diff = refined_np[i] - true_np[i]
                row.update(
                    {
                        "base_x": float(base_np[i, 0]),
                        "base_y": float(base_np[i, 1]),
                        "refined_x": float(refined_np[i, 0]),
                        "refined_y": float(refined_np[i, 1]),
                        "true_x": float(true_np[i, 0]),
                        "true_y": float(true_np[i, 1]),
                        "base_err_px": float(np.linalg.norm(base_diff * scale_np)),
                        "refined_err_px": float(np.linalg.norm(refined_diff * scale_np)),
                    }
                )
            rows.append(row)
    rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))
    y = [int(row["touch_true"]) for row in rows if row["has_touch"]]
    raw_pred = [int(float(row["touch_prob"]) >= float(args.raw_threshold)) for row in rows if row["has_touch"]]
    hyst_all = apply_hysteresis(
        rows,
        on_threshold=float(args.hyst_on),
        off_threshold=float(args.hyst_off),
        min_on_samples=int(args.hyst_min_on),
        min_off_samples=int(args.hyst_min_off),
    )
    hyst_pred = [pred for row, pred in zip(rows, hyst_all) if row["has_touch"]]
    report = {
        "split": split,
        "rows": len(rows),
        "valid_touch_rows": len(y),
        "base_xy_by_touch": {
            "touch_true": xy_summary(rows, "base_err_px", touch_true=1),
            "touch_false": xy_summary(rows, "base_err_px", touch_true=0),
        },
        "refined_xy_by_touch": {
            "touch_true": xy_summary(rows, "refined_err_px", touch_true=1),
            "touch_false": xy_summary(rows, "refined_err_px", touch_true=0),
        },
        "raw_threshold": {"threshold": float(args.raw_threshold), **binary_metrics(y, raw_pred)},
        "hysteresis": {
            "on_threshold": float(args.hyst_on),
            "off_threshold": float(args.hyst_off),
            "min_on_samples": int(args.hyst_min_on),
            "min_off_samples": int(args.hyst_min_off),
            **binary_metrics(y, hyst_pred),
        },
    }
    return rows, report


def crop_around_xy(image: torch.Tensor, xy: torch.Tensor, crop_size: int, crop_span_px: float) -> torch.Tensor:
    b, _, h, w = image.shape
    device = image.device
    dtype = image.dtype
    half_x = (float(crop_span_px) * 0.5) / max(1.0, float(w - 1))
    half_y = (float(crop_span_px) * 0.5) / max(1.0, float(h - 1))
    lin_x = torch.linspace(-half_x, half_x, crop_size, device=device, dtype=dtype)
    lin_y = torch.linspace(-half_y, half_y, crop_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing="ij")
    gx = (xy[:, 0].to(dtype=dtype).view(b, 1, 1) + xx.view(1, crop_size, crop_size)) * 2.0 - 1.0
    gy = (xy[:, 1].to(dtype=dtype).view(b, 1, 1) + yy.view(1, crop_size, crop_size)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def jitter_xy(xy: torch.Tensor, jitter_px: float, scale: torch.Tensor) -> torch.Tensor:
    if jitter_px <= 0:
        return xy
    jitter = (torch.rand_like(xy) * 2.0 - 1.0) * float(jitter_px) / scale.view(1, 2)
    return xy + jitter


def valid_mask(batch: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    mask = batch["valid_xy"].bool()
    if bool(args.touch_only):
        mask = mask & batch["has_touch"].bool() & (batch["touch"].float() > 0.5)
    return mask


def xy_summary(rows: list[dict[str, Any]], key: str, touch_true: int) -> dict[str, Any]:
    vals = [
        float(row[key])
        for row in rows
        if row["has_touch"] and row["touch_true"] == touch_true and row["valid_xy"] and row[key] != ""
    ]
    if not vals:
        return {"count": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean_euclidean_px": float(np.mean(arr)),
        "median_euclidean_px": float(np.median(arr)),
    }


def binary_metrics(y: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 1)
    tn = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 0)
    fp = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 1)
    fn = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"accuracy": (tp + tn) / max(1, tp + tn + fp + fn), "precision": precision, "recall": recall, "f1": f1, "count": len(y)}


def apply_hysteresis(rows: list[dict[str, Any]], on_threshold: float, off_threshold: float, min_on_samples: int, min_off_samples: int) -> list[int]:
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
                off_count = off_count + 1 if prob < off_threshold else 0
                if off_count >= min_off_samples:
                    active = False
                    off_count = 0
                on_count = 0
            else:
                on_count = on_count + 1 if prob >= on_threshold else 0
                if on_count >= min_on_samples:
                    active = True
                    on_count = 0
                off_count = 0
            preds[index] = int(active)
    return preds


def summarize_errors(losses: list[float], base_errors: list[float], refined_errors: list[float], used: int) -> dict[str, float]:
    base = np.asarray(base_errors, dtype=np.float64)
    refined = np.asarray(refined_errors, dtype=np.float64)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "count": float(used),
        "base_mean_px": float(np.mean(base)) if len(base) else 0.0,
        "base_median_px": float(np.median(base)) if len(base) else 0.0,
        "refined_mean_px": float(np.mean(refined)) if len(refined) else 0.0,
        "refined_median_px": float(np.median(refined)) if len(refined) else 0.0,
    }


def save_payload(path: Path, refiner: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, Any], args: argparse.Namespace, cfg: Any) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "config": dict(cfg),
        },
        path,
    )


def load_refiner(path: Path, refiner: nn.Module, map_location: torch.device) -> None:
    payload = torch.load(path, map_location=map_location)
    refiner.load_state_dict(payload["model"])


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields = list(dict.fromkeys([*fields, *row.keys()]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_report(report: dict[str, Any]) -> str:
    b = report["base_xy_by_touch"]["touch_true"]
    r = report["refined_xy_by_touch"]["touch_true"]
    bf = report["base_xy_by_touch"]["touch_false"]
    rf = report["refined_xy_by_touch"]["touch_false"]
    raw = report["raw_threshold"]
    hyst = report["hysteresis"]
    return "\n".join(
        [
            "Touch True",
            f"Base mean/median:    {b.get('mean_euclidean_px', 0.0):.2f} / {b.get('median_euclidean_px', 0.0):.2f} px",
            f"Refined mean/median: {r.get('mean_euclidean_px', 0.0):.2f} / {r.get('median_euclidean_px', 0.0):.2f} px",
            "Touch False",
            f"Base mean/median:    {bf.get('mean_euclidean_px', 0.0):.2f} / {bf.get('median_euclidean_px', 0.0):.2f} px",
            f"Refined mean/median: {rf.get('mean_euclidean_px', 0.0):.2f} / {rf.get('median_euclidean_px', 0.0):.2f} px",
            "",
            f"Touch raw F1 remains {raw['f1']:.3f}; hysteresis F1 remains {hyst['f1']:.3f}.",
        ]
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
