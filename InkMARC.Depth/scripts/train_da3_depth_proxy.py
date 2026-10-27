#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SENSE_ROOT = ROOT.parent / "InkMARC.Sense"
# Shared dataset/configuration helpers; keep the Depth model independently packaged.
sys.path.insert(0, str(SENSE_ROOT))
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset
from inkmarc_depth import build_model_from_config
from inkdepth.utils.checkpoints import load_model_weights, save_checkpoint
from inkdepth.utils.config import ensure_dir, load_config, parse_overrides


class DA3ProxyCacheDataset(Dataset):
    def __init__(self, cfg: Any, cache_dirs: list[str | Path], split: str, input_size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.split = split
        self.input_size = tuple(input_size)
        self.base = StylusDataset(cfg, split=split, training=(split == "train"))
        self.records: list[dict[str, Any]] = []
        self._memmaps: dict[tuple[str, tuple[int, ...]], np.memmap] = {}
        for raw_cache_dir in cache_dirs:
            cache_dir = Path(raw_cache_dir).expanduser().resolve()
            index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
            entries = index.get("splits", {}).get(split, [])
            for entry in entries:
                source = str(entry.get("source", "stylus_dataset"))
                record = {"cache_dir": cache_dir, "entry": entry, "source": source}
                self.records.append(record)
        if not self.records:
            cache_list = ", ".join(str(Path(p).expanduser()) for p in cache_dirs)
            raise ValueError(f"No cache entries found for split={split!r} in: {cache_list}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        entry = record["entry"]
        if record["source"] == "npy_xz_raw":
            rgb = self._read_npy_xz_rgb(record["cache_dir"], entry)
        else:
            sample = self.base[int(entry["index"])]
            rgb = sample["image"][:3].float()
        if tuple(rgb.shape[-2:]) != self.input_size:
            rgb = F.interpolate(rgb.unsqueeze(0), size=self.input_size, mode="bilinear", align_corners=False).squeeze(0)
        target_path = record["cache_dir"] / entry["path"]
        with np.load(target_path) as npz:
            depth = np.asarray(npz["depth"], dtype=np.float32)
        depth_t = torch.from_numpy(depth)
        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0)
        return {"image": rgb.contiguous(), "depth": depth_t.contiguous()}

    def _read_npy_xz_rgb(self, cache_dir: Path, entry: dict[str, Any]) -> torch.Tensor:
        raw_path = cache_dir / entry["image_raw_path"]
        image_shape = tuple(int(x) for x in entry["image_raw_shape"])
        key = (str(raw_path), image_shape)
        mmap = self._memmaps.get(key)
        if mmap is None:
            mmap = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=image_shape)
            self._memmaps[key] = mmap
        image = np.asarray(mmap[int(entry["image_index"])])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB uint8 image, got shape={image.shape}")
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0


def normalize_per_frame_percentile(depth: torch.Tensor, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = depth.flatten(1)
    lo = torch.quantile(flat, low, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, high, dim=1).view(-1, 1, 1, 1)
    denom = (hi - lo).clamp_min(1.0e-8)
    return ((depth - lo) / denom).clamp(0.0, 1.0)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def depth_proxy_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Any) -> tuple[torch.Tensor, dict[str, float]]:
    raw_l1 = F.l1_loss(pred, target)
    grad = gradient_loss(pred, target)
    norm_l1 = F.l1_loss(normalize_per_frame_percentile(pred), normalize_per_frame_percentile(target))
    total = (
        float(getattr(cfg.loss, "lambda_raw_l1", 1.0)) * raw_l1
        + float(getattr(cfg.loss, "lambda_grad_l1", 0.2)) * grad
        + float(getattr(cfg.loss, "lambda_norm_l1", 1.0)) * norm_l1
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "raw_l1": float(raw_l1.detach().cpu()),
        "grad_l1": float(grad.detach().cpu()),
        "norm_l1": float(norm_l1.detach().cpu()),
    }
    return total, metrics


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["depth"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=bool(getattr(cfg.training, "amp", True)) and device.type == "cuda"):
                pred = model(image)
                loss, metrics = depth_proxy_loss(pred, target, cfg)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        batch_count = int(image.shape[0])
        count += batch_count
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_count
    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QNN-friendly depth proxy from cached DA3 teacher outputs.")
    parser.add_argument("--config", default=str(ROOT / "configs/da3_depth_proxy_504x322.yaml"))
    parser.add_argument(
        "--cache-dir",
        action="append",
        default=None,
        help="Depth proxy cache dir. Repeat to mix original and npy_xz caches.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to initialize model weights from. Optimizer/scheduler are reset.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(cfg.output_dir)
    input_size = tuple(int(x) for x in cfg.model.proxy_input_size)

    cache_dirs = args.cache_dir or [str(ROOT / "outputs/da3_proxy_depth_cache_504x322")]
    train_ds = DA3ProxyCacheDataset(cfg, cache_dirs, "train", input_size)
    val_ds = DA3ProxyCacheDataset(cfg, cache_dirs, "val", input_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model_from_config(cfg).to(device)
    if args.init_checkpoint:
        load_model_weights(args.init_checkpoint, model, map_location=device, strict=True)
        print(f"initialized model weights from {Path(args.init_checkpoint).expanduser().resolve()}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg.training.epochs)),
        eta_min=float(getattr(cfg.scheduler, "min_lr", 1.0e-6)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(cfg.training, "amp", True)) and device.type == "cuda")

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, cfg, device, optimizer=optimizer, scaler=scaler)
        val_metrics = run_epoch(model, val_loader, cfg, device)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        (Path(output_dir) / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(Path(output_dir) / "latest.pt", model, optimizer, epoch, val_metrics, scaler, scheduler, dict(cfg))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(Path(output_dir) / "best_val_loss.pt", model, optimizer, epoch, val_metrics, scaler, scheduler, dict(cfg))
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_norm_l1={val_metrics['norm_l1']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
