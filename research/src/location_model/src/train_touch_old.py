#!/usr/bin/env python3
"""
train_touch.py

Refactored touch classifier training that uses the new InkMARC data pipeline:

- dataset_core.InkMARCIndexedDataset for labels/masks/meta
- video_io.VideoFrameReader for fetching frames from AVI variants
- label_schema.extract_xy_touch for deriving (touch, touch_mask) from stored label vectors

Key training principles:
- Train on BOTH positives (touch) and negatives (hover), never "touch-only".
- Only compute loss on frames where touch supervision is valid (touch_mask == 1).
- Optionally correct class imbalance using pos_weight (recommended).

Example:
  PYTHONPATH=./src python ./src/train_touch.py \
    --dataset-root /data/LocationData \
    --version v1 \
    --epochs 10 --batch-size 32 --num-workers 12 \
    --img-size 448 \
    --amp \
    --save-dir runs/touch_resnet18_v1
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms.functional as TF

from data.dataset_core import InkMARCIndexedDataset
from data.label_schema import extract_xy_touch
from data.video_io import VideoFrameReader


# --------------------------
# Config
# --------------------------

@dataclass(frozen=True)
class Cfg:
    dataset_root: str
    version: str
    save_dir: str

    img_size: int = 448
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0

    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True

    amp: bool = False
    seed: int = 1234

    pressure_touch_threshold: float = 0.0
    require_xy_for_touch: bool = False

    # imbalance handling
    pos_weight: str = "auto"   # "auto" | "none"
    pos_weight_max_samples: int = 200_000

    # video reader
    video_cache_size: int = 4

    # export/checkpoint
    export_onnx: bool = True
    onnx_opset: int = 13


# --------------------------
# Transforms
# --------------------------

import torch.nn.functional as F
import torchvision.transforms.functional as TF

class TouchImageTransform:
    def __init__(self, size: int = 448) -> None:
        self.size = int(size)
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)

    def __call__(self, frame_bgr):
        # BGR uint8 -> RGB float tensor (3,H,W) in [0,1]
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

        c, h, w = t.shape

        # scale preserving aspect ratio
        scale = self.size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        t = TF.resize(t, [new_h, new_w], antialias=True)

        # pad to (size, size)
        pad_h = self.size - new_h
        pad_w = self.size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad expects (left, right, top, bottom)
        t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

        # normalize
        t = TF.normalize(t, mean=self.mean, std=self.std)
        return t

# --------------------------
# Dataset wrapper (canonical -> touch)
# --------------------------

class InkMARCTouchDataset(Dataset):
    """
    Wrap InkMARCIndexedDataset into a dict batch with:
      - image: (3, img_size, img_size) float32
      - touch: (1,) float32 in {0,1}
      - touch_mask: (1,) float32 in {0,1} (valid supervision)
      - meta: dict

    Note: we DO NOT filter to "touch-only". Negatives are essential.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        version: str,
        split: str,
        *,
        img_size: int,
        pressure_touch_threshold: float,
        require_xy_for_touch: bool,
        video_cache_size: int,
    ) -> None:
        self.split = str(split)

        self.core = InkMARCIndexedDataset(
            dataset_root=str(dataset_root),
            index_version=str(version),
            split=self.split,
            require_labels=True,
            transform=None,
        )

        self.vr = VideoFrameReader(
            dataset_root=str(dataset_root),
            manifest=self.core.manifest,
            cache_size=video_cache_size,
        )

        self.img_tf = TouchImageTransform(size=img_size)
        self.pressure_touch_threshold = float(pressure_touch_threshold)
        self.require_xy_for_touch = bool(require_xy_for_touch)

    def __len__(self) -> int:
        return len(self.core)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.core[idx]
        labels = s["labels"]
        label_mask = s["label_mask"]
        meta = s["meta"]

        touch, touch_mask, *_ = extract_xy_touch(
            labels,
            label_mask,
            pressure_touch_threshold=self.pressure_touch_threshold,
            require_xy_for_touch=self.require_xy_for_touch,
        )

        file_id = meta.get("file_id")
        frame_idx = int(meta["frame_idx"])

        frame_bgr = self.vr.get_frame(str(file_id), frame_idx, split=self.split)
        image = self.img_tf(frame_bgr)

        return {
            "image": image,
            "touch": torch.tensor([touch], dtype=torch.float32),
            "touch_mask": torch.tensor([touch_mask], dtype=torch.float32),
            "meta": meta,
        }

    def close(self) -> None:
        try:
            self.vr.release()
        except Exception:
            pass
        try:
            self.core.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def collate_touch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)          # (B,3,H,W)
    touch = torch.stack([b["touch"] for b in batch], dim=0)           # (B,1)
    touch_mask = torch.stack([b["touch_mask"] for b in batch], dim=0) # (B,1)
    meta = [b["meta"] for b in batch]
    return {"image": images, "touch": touch, "touch_mask": touch_mask, "meta": meta}


# --------------------------
# Model
# --------------------------

class ResNet18TouchLogits(nn.Module):
    """
    ResNet18 with a 1-logit head (use BCEWithLogitsLoss).
    """
    def __init__(self) -> None:
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        n = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(n, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)  # (B,1)


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_pos_weight(
    dataset_root: str,
    version: str,
    *,
    split: str,
    pressure_touch_threshold: float,
    require_xy_for_touch: bool,
    max_samples: int,
    seed: int,
) -> Optional[float]:
    """
    Estimate pos_weight = (#neg / #pos) over supervised frames only.

    Important: we do NOT decode video frames here; we only read labels/label_mask.
    """
    ds = InkMARCIndexedDataset(
        dataset_root=dataset_root,
        index_version=version,
        split=split,
        require_labels=True,
        transform=None,
    )

    n = len(ds)
    if n <= 0:
        return None

    rng = random.Random(seed)
    if max_samples > 0 and n > max_samples:
        idxs = rng.sample(range(n), k=max_samples)
    else:
        idxs = list(range(n))

    pos = 0
    neg = 0
    sup = 0

    for i in idxs:
        s = ds[i]
        touch, touch_mask, *_ = extract_xy_touch(
            s["labels"],
            s["label_mask"],
            pressure_touch_threshold=pressure_touch_threshold,
            require_xy_for_touch=require_xy_for_touch,
        )
        if touch_mask < 0.5:
            continue
        sup += 1
        if touch > 0.5:
            pos += 1
        else:
            neg += 1

    ds.close()

    if sup == 0 or pos == 0 or neg == 0:
        # If no positives/negatives under supervision, pos_weight isn't meaningful.
        return None

    return float(neg / pos)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_weight = 0.0

    correct = 0.0
    count = 0.0

    # optional: track precision/recall on supervised frames
    tp = fp = tn = fn = 0.0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["touch"].to(device, non_blocking=True)           # (B,1)
        m = batch["touch_mask"].to(device, non_blocking=True)      # (B,1)

        logits = model(x)
        per = loss_fn(logits, y)                                   # (B,1) because reduction="none"
        masked = per * m

        total_loss += float(masked.sum().item())
        total_weight += float(m.sum().item())

        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()

        # accuracy on supervised only
        correct += float(((pred == y).float() * m).sum().item())
        count += float(m.sum().item())

        # precision/recall confusion on supervised
        sup = (m > 0.5)
        if sup.any():
            yb = y[sup] > 0.5
            pb = pred[sup] > 0.5
            tp += float((pb & yb).sum().item())
            fp += float((pb & ~yb).sum().item())
            tn += float((~pb & ~yb).sum().item())
            fn += float((~pb & yb).sum().item())

    loss = total_loss / max(1.0, total_weight)
    acc = correct / max(1.0, count)

    precision = tp / max(1.0, (tp + fp))
    recall = tp / max(1.0, (tp + fn))
    f1 = 2 * precision * recall / max(1e-12, (precision + recall))

    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_supervised": total_weight,
    }


def save_checkpoint(save_dir: Path, name: str, model: nn.Module, opt: optim.Optimizer, epoch: int, extra: Dict[str, Any]) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / name
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "extra": extra,
    }
    torch.save(payload, path)
    return path


def export_onnx(save_dir: Path, model: nn.Module, img_size: int, device: torch.device, opset: int) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    out_path = save_dir / "touch_resnet18_logits.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(opset),
    )
    return out_path


# --------------------------
# Main train
# --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--save-dir", required=True)

    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-persistent-workers", action="store_true")
    ap.add_argument("--no-pin-memory", action="store_true")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    ap.add_argument("--require-xy-for-touch", action="store_true")

    ap.add_argument("--pos-weight", choices=["auto", "none"], default="auto")
    ap.add_argument("--pos-weight-max-samples", type=int, default=200_000)

    ap.add_argument("--video-cache-size", type=int, default=4)

    ap.add_argument("--no-export-onnx", action="store_true")
    ap.add_argument("--onnx-opset", type=int, default=13)

    args = ap.parse_args()

    cfg = Cfg(
        dataset_root=args.dataset_root,
        version=args.version,
        save_dir=args.save_dir,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        persistent_workers=(not args.no_persistent_workers),
        pin_memory=(not args.no_pin_memory),
        amp=bool(args.amp),
        seed=args.seed,
        pressure_touch_threshold=args.pressure_touch_threshold,
        require_xy_for_touch=bool(args.require_xy_for_touch),
        pos_weight=args.pos_weight,
        pos_weight_max_samples=args.pos_weight_max_samples,
        video_cache_size=args.video_cache_size,
        export_onnx=(not args.no_export_onnx),
        onnx_opset=args.onnx_opset,
    )

    set_seed(cfg.seed)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "cfg.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pos_weight
    pos_w_val: Optional[float] = None
    if cfg.pos_weight == "auto":
        pos_w_val = compute_pos_weight(
            cfg.dataset_root,
            cfg.version,
            split="train",
            pressure_touch_threshold=cfg.pressure_touch_threshold,
            require_xy_for_touch=cfg.require_xy_for_touch,
            max_samples=cfg.pos_weight_max_samples,
            seed=cfg.seed,
        )

    if pos_w_val is not None:
        pos_weight_t = torch.tensor([pos_w_val], device=device)
        print(f"[pos_weight] using auto pos_weight={pos_w_val:.4f} (neg/pos)")
    else:
        pos_weight_t = None
        if cfg.pos_weight == "auto":
            print("[pos_weight] auto requested but could not compute (no pos/neg supervised); using unweighted loss.")
        else:
            print("[pos_weight] disabled")

    # Datasets / loaders
    train_ds = InkMARCTouchDataset(
        cfg.dataset_root, cfg.version, "train",
        img_size=cfg.img_size,
        pressure_touch_threshold=cfg.pressure_touch_threshold,
        require_xy_for_touch=cfg.require_xy_for_touch,
        video_cache_size=cfg.video_cache_size,
    )
    val_ds = InkMARCTouchDataset(
        cfg.dataset_root, cfg.version, "val",
        img_size=cfg.img_size,
        pressure_touch_threshold=cfg.pressure_touch_threshold,
        require_xy_for_touch=cfg.require_xy_for_touch,
        video_cache_size=cfg.video_cache_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        collate_fn=collate_touch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        collate_fn=collate_touch,
    )

    model = ResNet18TouchLogits().to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Loss: reduction="none" so we can apply touch_mask
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_t)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0

        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["touch"].to(device, non_blocking=True)          # (B,1)
            m = batch["touch_mask"].to(device, non_blocking=True)     # (B,1)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.amp and device.type == "cuda")):
                logits = model(x)
                per = loss_fn(logits, y)         # (B,1)
                masked = per * m
                loss = masked.sum() / torch.clamp(m.sum(), min=1.0)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += float(masked.sum().item())
            total_weight += float(m.sum().item())

        train_loss = total_loss / max(1.0, total_weight)

        val_stats = evaluate(model, val_loader, device, loss_fn)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['loss']:.4f} "
            f"acc={val_stats['acc']:.4f} f1={val_stats['f1']:.4f} "
            f"(val_supervised={val_stats['n_supervised']:.0f})"
        )

        # Save last checkpoint each epoch
        save_checkpoint(
            save_dir,
            "ckpt_last.pt",
            model,
            opt,
            epoch,
            extra={"val": val_stats, "train_loss": train_loss, "pos_weight": pos_w_val},
        )

        # Save best by val loss
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                save_dir,
                "ckpt_best.pt",
                model,
                opt,
                epoch,
                extra={"val": val_stats, "train_loss": train_loss, "pos_weight": pos_w_val},
            )

    if cfg.export_onnx:
        out_path = export_onnx(save_dir, model, cfg.img_size, device, cfg.onnx_opset)
        print(f"ONNX exported: {out_path}")

    # Cleanup
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
