#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets.stylus_dataset import surface_to_image_xy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal hover-vs-touch from local depth windows.")
    parser.add_argument(
        "--manifest",
        default="/data/baseAVIs_432x672/manifest_texture_replacement_v3_no_missing_h5_no_black_participant_splits.json",
    )
    parser.add_argument("--output-dir", default="outputs/temporal_touch_depth_original_w5")
    parser.add_argument("--variant-kind", default="original")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--offsets", default="-5,-3,-1,0,1,3,5")
    parser.add_argument("--center-stride", type=int, default=2)
    parser.add_argument("--exclude-transition-window", type=int, default=5)
    parser.add_argument("--min-touch-pressure", type=float, default=1000.0)
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--context-size", type=int, default=96)
    parser.add_argument(
        "--crop-normalize",
        choices=["none", "local_zscore", "center_median"],
        default="none",
        help="Normalize each local crop before stacking temporal channels.",
    )
    parser.add_argument(
        "--model-kind",
        choices=["local", "local_context"],
        default="local",
        help="local uses only crops; local_context also uses full-frame depth context and xy.",
    )
    parser.add_argument("--jitter-pixels", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--sampler",
        choices=["touch", "task_touch", "video_touch"],
        default="touch",
        help="touch balances hover/touch globally; task_touch and video_touch balance class counts within each group.",
    )
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = parse_int_list(args.offsets)
    cfg = vars(args).copy()
    cfg["offsets"] = offsets
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_ds = TemporalTouchDepthDataset(args, split="train", offsets=offsets, training=True)
    val_ds = TemporalTouchDepthDataset(args, split="val", offsets=offsets, training=False)
    if args.max_train_samples > 0:
        train_ds.samples = train_ds.samples[: args.max_train_samples]
        train_ds.recompute_counts()
    if args.max_val_samples > 0:
        val_ds.samples = val_ds.samples[: args.max_val_samples]
        val_ds.recompute_counts()

    train_sampler = build_sampler(train_ds, args.sampler)
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": temporal_collate,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=False, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_kind == "local_context":
        model = TemporalDepthTouchContextNet(in_channels=len(offsets)).to(device)
    else:
        model = TemporalDepthTouchNet(in_channels=len(offsets)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")

    print(f"train samples={len(train_ds)} counts={dict(train_ds.counts)}")
    print(f"val samples={len(val_ds)} counts={dict(val_ds.counts)}")
    print(f"offsets={offsets} crop_size={args.crop_size} context_size={args.context_size} model_kind={args.model_kind}")

    csv_path = output_dir / "metrics.csv"
    best_f1 = -1.0
    best_acc = -1.0
    best_swept_f1 = -1.0
    best_swept_acc = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        metrics = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        append_metrics(csv_path, metrics)
        save_payload(output_dir / "latest.pt", model, optimizer, epoch, metrics, cfg)
        if val_metrics.get("touch_f1", -1.0) > best_f1:
            best_f1 = val_metrics["touch_f1"]
            save_payload(output_dir / "best_touch.pt", model, optimizer, epoch, metrics, cfg)
        if val_metrics.get("touch_accuracy", -1.0) > best_acc:
            best_acc = val_metrics["touch_accuracy"]
            save_payload(output_dir / "best_accuracy.pt", model, optimizer, epoch, metrics, cfg)
        if val_metrics.get("touch_best_f1", -1.0) > best_swept_f1:
            best_swept_f1 = val_metrics["touch_best_f1"]
            save_payload(output_dir / "best_touch_swept.pt", model, optimizer, epoch, metrics, cfg)
        if val_metrics.get("touch_best_accuracy", -1.0) > best_swept_acc:
            best_swept_acc = val_metrics["touch_best_accuracy"]
            save_payload(output_dir / "best_accuracy_swept.pt", model, optimizer, epoch, metrics, cfg)
        print(format_metrics(metrics))

    train_ds.close()
    val_ds.close()


class TemporalTouchDepthDataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str, offsets: list[int], training: bool) -> None:
        self.args = args
        self.split = split
        self.offsets = offsets
        self.training = training
        self.crop_size = int(args.crop_size)
        self.context_size = int(args.context_size)
        self.jitter_pixels = int(args.jitter_pixels) if training else 0
        self.items = load_items(Path(args.manifest), split, args.variant_kind)
        self.samples: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self._depth_cache: OrderedDict[str, Any] = OrderedDict()
        self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        arr = self._open_depth(sample["depth_path"], sample["depth_key"])
        frames = []
        contexts = []
        for offset in self.offsets:
            frame_index = int(sample["frame_index"]) + int(offset)
            depth = np.asarray(arr[frame_index], dtype=np.float32).squeeze()
            crop = crop_around_xy(depth, sample["xy"], self.crop_size, self.jitter_pixels)
            crop = normalize_crop(crop, str(self.args.crop_normalize))
            frames.append(crop)
            if self.args.model_kind == "local_context":
                contexts.append(resize_depth(depth, self.context_size))
        x = torch.from_numpy(np.stack(frames, axis=0).astype(np.float32))
        row = {
            "x": x,
            "y": torch.tensor(float(sample["touch"]), dtype=torch.float32),
            "xy": torch.from_numpy(sample["xy"].astype(np.float32)),
            "video_id": sample["video_id"],
            "frame_index": int(sample["frame_index"]),
            "pressure": float(sample["pressure"]),
        }
        if self.args.model_kind == "local_context":
            row["context"] = torch.from_numpy(np.stack(contexts, axis=0).astype(np.float32))
        return row

    def _build_samples(self) -> None:
        min_offset = min(self.offsets)
        max_offset = max(self.offsets)
        for item_i, item in enumerate(self.items, start=1):
            h5_path = Path(str(item.get("output_h5_path") or item.get("label_h5_path") or ""))
            depth_path = Path(str(item.get("depth_path") or ""))
            if not h5_path.exists() or not depth_path.exists():
                continue
            try:
                with h5py.File(h5_path, "r") as h5:
                    labels = np.asarray(h5["labels"][:], dtype=np.float32)
                    masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
                    corners = np.asarray(h5["corners"][:], dtype=np.float32) if "corners" in h5 else None
                    resize_attrs = dict(h5.get("preprocessing/resize_transform", {}).attrs) if "preprocessing/resize_transform" in h5 else {}
            except Exception:
                continue
            if corners is None:
                continue
            n = min(int(item.get("frame_count") or len(labels)), len(labels))
            states = derive_states(labels[:n], masks[:n], float(self.args.pressure_touch_threshold))
            distances = transition_distances(states)
            for frame_index in range(max(0, -min_offset), min(n, n - max_offset), max(1, int(self.args.center_stride))):
                state = int(states[frame_index])
                if state not in {1, 2}:
                    continue
                if int(distances[frame_index]) <= int(self.args.exclude_transition_window):
                    continue
                pressure = float(labels[frame_index, 2])
                if state == 2 and pressure <= float(self.args.min_touch_pressure):
                    continue
                if not valid_xy_pressure(labels[frame_index], masks[frame_index]):
                    continue
                xy = convert_xy(labels[frame_index], corners[frame_index], resize_attrs)
                if xy is None:
                    continue
                touch = 1.0 if state == 2 else 0.0
                self.samples.append(
                    {
                        "video_id": str(item.get("stem") or item.get("base_stem") or h5_path.stem),
                        "frame_index": int(frame_index),
                        "depth_path": str(depth_path),
                        "depth_key": str(item.get("depth_key", "depth_norm")),
                        "xy": xy.astype(np.float32),
                        "pressure": pressure,
                        "touch": touch,
                    }
                )
                self.counts["touch" if touch > 0.5 else "hover"] += 1
            if item_i == 1 or item_i % 100 == 0:
                print(f"{self.split}: processed {item_i}/{len(self.items)} items samples={len(self.samples)}")

    def _open_depth(self, path: str, key: str) -> Any:
        store = self._depth_cache.get(path)
        if store is None:
            store = zarr.open(path, mode="r")
            self._depth_cache[path] = store
            while len(self._depth_cache) > 32:
                self._depth_cache.popitem(last=False)
        else:
            self._depth_cache.move_to_end(path)
        return store if hasattr(store, "shape") else store[key]

    def close(self) -> None:
        self._depth_cache.clear()

    def recompute_counts(self) -> None:
        self.counts = Counter("touch" if sample["touch"] > 0.5 else "hover" for sample in self.samples)


class TemporalDepthTouchNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class TemporalDepthTouchContextNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 96 + 2, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, xy: torch.Tensor | None = None) -> torch.Tensor:
        if context is None or xy is None:
            raise ValueError("TemporalDepthTouchContextNet requires context and xy tensors")
        feat = torch.cat([self.local(x), self.context(context), xy], dim=1)
        return self.head(feat).squeeze(1)


def load_items(manifest_path: Path, split: str, variant_kind: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [dict(item) for item in manifest.get("items", [])]
    if not items:
        raise ValueError(f"Expected manifest with items: {manifest_path}")
    out = []
    for item in items:
        if str(item.get("split", "unknown")) != split:
            continue
        if variant_kind and str(item.get("variant_kind", "original")) != variant_kind:
            continue
        out.append(item)
    return out


def derive_states(labels: np.ndarray, masks: np.ndarray, pressure_threshold: float) -> np.ndarray:
    valid_xy = (
        (masks[:, 0] != 0)
        & (masks[:, 1] != 0)
        & np.isfinite(labels[:, 0])
        & np.isfinite(labels[:, 1])
    )
    pressure_valid = (masks[:, 2] != 0) & np.isfinite(labels[:, 2])
    touch = valid_xy & pressure_valid & (labels[:, 2] > pressure_threshold)
    states = np.zeros(labels.shape[0], dtype=np.int8)
    states[valid_xy & ~touch] = 1
    states[touch] = 2
    return states


def transition_distances(states: np.ndarray) -> np.ndarray:
    distances = np.full(states.shape[0], fill_value=10**9, dtype=np.int32)
    transitions = np.flatnonzero(states[1:] != states[:-1]) + 1
    if transitions.size == 0:
        return distances
    indices = np.arange(states.shape[0], dtype=np.int32)
    insert_pos = np.searchsorted(transitions, indices)
    right = np.where(insert_pos < transitions.size, transitions[np.minimum(insert_pos, transitions.size - 1)], 10**9)
    left_pos = np.maximum(insert_pos - 1, 0)
    left = np.where(insert_pos > 0, transitions[left_pos], -10**9)
    return np.minimum(np.abs(indices - left), np.abs(indices - right)).astype(np.int32)


def valid_xy_pressure(labels: np.ndarray, masks: np.ndarray) -> bool:
    return bool(
        masks[0] != 0
        and masks[1] != 0
        and masks[2] != 0
        and np.isfinite(labels[0])
        and np.isfinite(labels[1])
        and np.isfinite(labels[2])
    )


def convert_xy(labels: np.ndarray, corners: np.ndarray, resize_attrs: dict[str, Any]) -> np.ndarray | None:
    return surface_to_image_xy(
        float(labels[0]),
        float(labels[1]),
        corners,
        surface_w=44800.0,
        surface_h=29600.0,
        image_w=432.0,
        image_h=672.0,
        surface_model_w=480.0,
        surface_model_h=320.0,
        corners_coordinate_mode="resized",
        resize_attrs=resize_attrs,
    )


def crop_around_xy(depth: np.ndarray, xy: np.ndarray, crop_size: int, jitter_pixels: int) -> np.ndarray:
    depth = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    depth = np.clip(depth, 0.0, 1.0)
    h, w = depth.shape[-2:]
    cx = int(round(float(xy[0]) * (w - 1)))
    cy = int(round(float(xy[1]) * (h - 1)))
    if jitter_pixels > 0:
        cx += random.randint(-jitter_pixels, jitter_pixels)
        cy += random.randint(-jitter_pixels, jitter_pixels)
    half = crop_size // 2
    x0, x1 = cx - half, cx - half + crop_size
    y0, y1 = cy - half, cy - half + crop_size
    crop = np.zeros((crop_size, crop_size), dtype=np.float32)
    sx0, sx1 = max(0, x0), min(w, x1)
    sy0, sy1 = max(0, y0), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    if sx1 > sx0 and sy1 > sy0:
        crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = depth[sy0:sy1, sx0:sx1]
    return crop


def normalize_crop(crop: np.ndarray, mode: str) -> np.ndarray:
    crop = np.asarray(crop, dtype=np.float32)
    if mode == "none":
        return crop
    valid = crop[np.isfinite(crop)]
    valid = valid[valid > 0.0]
    if valid.size < 16:
        return np.zeros_like(crop, dtype=np.float32)
    if mode == "local_zscore":
        center = float(np.median(valid))
        scale = float(np.percentile(valid, 90) - np.percentile(valid, 10))
        scale = max(scale, 1e-4)
        return np.clip((crop - center) / scale, -3.0, 3.0).astype(np.float32)
    if mode == "center_median":
        center = float(np.median(valid))
        return np.clip(crop - center, -0.5, 0.5).astype(np.float32)
    raise ValueError(f"Unsupported crop_normalize mode: {mode}")


def resize_depth(depth: np.ndarray, size: int) -> np.ndarray:
    depth = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    depth = np.clip(depth, 0.0, 1.0)
    t = torch.from_numpy(depth)[None, None]
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def build_touch_sampler(dataset: TemporalTouchDepthDataset) -> WeightedRandomSampler:
    counts = Counter(int(sample["touch"] > 0.5) for sample in dataset.samples)
    weights = [1.0 / max(1, counts[int(sample["touch"] > 0.5)]) for sample in dataset.samples]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def build_sampler(dataset: TemporalTouchDepthDataset, mode: str) -> WeightedRandomSampler:
    if mode == "touch":
        return build_touch_sampler(dataset)
    if mode == "task_touch":
        group_counts = Counter((task_from_video_id(str(sample["video_id"])), int(sample["touch"] > 0.5)) for sample in dataset.samples)
        weights = [
            1.0 / max(1, group_counts[(task_from_video_id(str(sample["video_id"])), int(sample["touch"] > 0.5))])
            for sample in dataset.samples
        ]
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
    if mode == "video_touch":
        group_counts = Counter((str(sample["video_id"]), int(sample["touch"] > 0.5)) for sample in dataset.samples)
        weights = [
            1.0 / max(1, group_counts[(str(sample["video_id"]), int(sample["touch"] > 0.5))])
            for sample in dataset.samples
        ]
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
    raise ValueError(f"Unsupported sampler mode: {mode}")


def temporal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "x": torch.stack([b["x"] for b in batch]).float(),
        "y": torch.stack([b["y"] for b in batch]).float(),
        "xy": torch.stack([b["xy"] for b in batch]).float(),
        "video_id": [b["video_id"] for b in batch],
        "frame_index": torch.tensor([b["frame_index"] for b in batch], dtype=torch.long),
        "pressure": torch.tensor([b["pressure"] for b in batch], dtype=torch.float32),
    }
    if "context" in batch[0]:
        out["context"] = torch.stack([b["context"] for b in batch]).float()
    return out


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        context = batch.get("context")
        if context is not None:
            context = context.to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(x, context, xy) if context is not None else model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        probs = torch.sigmoid(logits.detach()).float().cpu().numpy()
        ys.append(y.detach().float().cpu().numpy())
        ps.append(probs)
        losses.append(float(loss.detach().cpu()))
        if len(losses) % 20 == 0:
            pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})
    metrics = binary_metrics(np.concatenate(ys), np.concatenate(ps), threshold=0.5)
    metrics["loss"] = float(np.mean(losses))
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        context = batch.get("context")
        if context is not None:
            context = context.to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x, context, xy) if context is not None else model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        ys.append(y.float().cpu().numpy())
        ps.append(probs)
        losses.append(float(loss.cpu()))
    y = np.concatenate(ys)
    probs = np.concatenate(ps)
    metrics = binary_metrics(y, probs, threshold=0.5)
    metrics.update(best_threshold_metrics(y, probs))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def binary_metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probs >= threshold
    truth = y >= 0.5
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "touch_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "touch_precision": precision,
        "touch_recall": recall,
        "touch_f1": f1,
        "touch_count": float(len(y)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def best_threshold_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    best_f1: dict[str, float] | None = None
    best_acc: dict[str, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = binary_metrics(y, probs, float(threshold))
        metrics["threshold"] = float(threshold)
        if best_f1 is None or metrics["touch_f1"] > best_f1["touch_f1"]:
            best_f1 = metrics
        if best_acc is None or metrics["touch_accuracy"] > best_acc["touch_accuracy"]:
            best_acc = metrics
    assert best_f1 is not None and best_acc is not None
    return {
        "touch_best_f1": best_f1["touch_f1"],
        "touch_best_f1_threshold": best_f1["threshold"],
        "touch_best_f1_precision": best_f1["touch_precision"],
        "touch_best_f1_recall": best_f1["touch_recall"],
        "touch_best_accuracy": best_acc["touch_accuracy"],
        "touch_best_accuracy_threshold": best_acc["threshold"],
        "touch_best_accuracy_precision": best_acc["touch_precision"],
        "touch_best_accuracy_recall": best_acc["touch_recall"],
    }


def append_metrics(path: Path, row: dict[str, float]) -> None:
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            old = next(csv.reader(f), [])
        fieldnames = list(dict.fromkeys([*old, *fieldnames]))
    rows = []
    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append({key: row.get(key, "") for key in fieldnames})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_payload(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float], cfg: dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def format_metrics(metrics: dict[str, float]) -> str:
    keys = [
        "epoch",
        "train/loss",
        "train/touch_f1",
        "train/touch_accuracy",
        "val/loss",
        "val/touch_f1",
        "val/touch_accuracy",
        "val/touch_best_f1",
        "val/touch_best_accuracy",
        "val/touch_best_f1_threshold",
    ]
    parts = []
    for key in keys:
        if key in metrics:
            value = metrics[key]
            parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return " ".join(parts)


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def task_from_video_id(video_id: str) -> str:
    parts = video_id.split("_")
    return parts[2] if len(parts) >= 3 and parts[0] == "participant" else "UNKNOWN"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    main()
