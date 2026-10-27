#!/usr/bin/env python3
import argparse
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor_image(x: np.ndarray) -> torch.Tensor:
    # x: H,W,3 uint8 -> float tensor [0,1], CHW
    t = torch.from_numpy(x)  # HWC, uint8
    t = t.permute(2, 0, 1).contiguous().float() / 255.0
    return t

# -----------------------------
# Dataset: (image, bounds)-> xy
# -----------------------------
class XYFromBoundsH5Dataset(Dataset):
    """
    Uses /images_reshaped as input, + /bounds (as 8-dim conditioning), target is /xy (2-dim).
    Accepts /bounds with shape (N, 8) or (N, 4, 2). Filters out rows with NaNs.
    Optionally keeps only frames where /touch == 1.
    """
    def __init__(self, h5_paths: List[str], only_touch: bool = False, color_only_augs: bool = True, units="pixels", cond_wh=False):
        self.files: List[h5py.File] = [h5py.File(p, "r") for p in h5_paths]
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, frame_idx)
        self.H_list, self.W_list = [], []
        self.bounds_format: List[str] = []  # "flat8" or "quad42" per file
        self.only_touch = only_touch
        self.units = units
        self.cond_wh = cond_wh

        # Preprocess / build index
        for fi, f in enumerate(self.files):
            imgs = f["/images_reshaped"]
            bnds = f["/bounds"]
            xy   = f["/xy"]
            assert imgs.ndim == 4 and imgs.shape[-1] == 3, "images_reshaped must be NHWC uint8"
            assert xy.shape[1] == 2, "/xy must be (N,2)"
            H, W = imgs.shape[1], imgs.shape[2]
            self.H_list.append(H)
            self.W_list.append(W)

            # detect bounds shape & NaN rows
            if bnds.ndim == 2 and bnds.shape[1] == 8:
                fmt = "flat8"
                b_all = np.array(bnds, dtype=np.float32)                # (N,8)
                b_nan = np.isnan(b_all).any(axis=1)                     # (N,)
            elif bnds.ndim == 3 and bnds.shape[1:] == (4, 2):
                fmt = "quad42"
                b_all = np.array(bnds, dtype=np.float32).reshape(bnds.shape[0], -1)  # (N,8)
                b_nan = np.isnan(b_all).any(axis=1)
            else:
                raise AssertionError(f"/bounds has unsupported shape {bnds.shape}; expected (N,8) or (N,4,2).")
            self.bounds_format.append(fmt)

            xy_all = np.array(xy, dtype=np.float32)                    # (N,2)
            xy_nan = np.isnan(xy_all).any(axis=1)

            if self.only_touch:
                if "/touch" not in f:
                    raise AssertionError("Requested --only-touch but /touch dataset is missing.")
                touch = np.array(f["/touch"]).reshape(-1)              # (N,)
                touch_mask = touch.astype(np.uint8) > 0
            else:
                touch_mask = np.ones(imgs.shape[0], dtype=bool)

            for i in range(imgs.shape[0]):
                # require: bounds valid & xy valid & (touch==1 if requested)
                if (not b_nan[i]) and (not xy_nan[i]) and touch_mask[i]:
                    self.index_map.append((fi, i))

        if len(self.index_map) == 0:
            raise RuntimeError("Dataset is empty after filtering (check only_touch and NaNs).")

        # Normalization (ImageNet) since we use a pretrained backbone
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Color-only augmentations to preserve geometry
        if color_only_augs:
            self.color_tf = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            ])
        else:
            self.color_tf = None

    def __len__(self):
        return len(self.index_map)

    def _load_bounds_as_quad42(self, f: h5py.File, fi: int, i: int) -> np.ndarray:
        """Return bounds as (4,2) float32."""
        fmt = self.bounds_format[fi]
        if fmt == "flat8":
            b = np.array(f["/bounds"][i], dtype=np.float32).reshape(4, 2)
        else:
            b = np.array(f["/bounds"][i], dtype=np.float32)  # (4,2)
        return b

    def __getitem__(self, idx):
        fi, i = self.index_map[idx]
        f = self.files[fi]
        img = f["/images_reshaped"][i]         # H,W,3 uint8
        b   = self._load_bounds_as_quad42(f, fi, i)  # (4,2) float32
        xy  = np.array(f["/xy"][i], dtype=np.float32)  # (2,)
        H, W = self.H_list[fi], self.W_list[fi]

        # Inputs
        img_t = to_tensor_image(img)  # [3,H,W], float32 in [0,1]
        if self.color_tf is not None:
            img_t = self.color_tf(img_t)
        img_t = self.norm(img_t)

        # Bounds conditioning
        if self.units == "norm":
            b_norm = b.copy()
            b_norm[:, 0] /= float(W)
            b_norm[:, 1] /= float(H)
            cond = b_norm.reshape(-1).astype(np.float32)  # (8,)
        else:
            cond = b.reshape(-1).astype(np.float32)       # (8,) in pixels
            if self.cond_wh:
                cond = np.concatenate([cond, np.array([float(W), float(H)], dtype=np.float32)])  # -> (10,)

        b_cond = torch.from_numpy(cond)

        # Target xy
        if self.units == "norm":
            target = np.array([xy[0] / float(W), xy[1] / float(H)], dtype=np.float32)
        else:
            target = xy.astype(np.float32)  # pixels

        target_t = torch.from_numpy(target)

        mask_t = torch.ones(2, dtype=torch.float32)
        meta = {"H": H, "W": W, "file_idx": fi, "frame_idx": i}
        return img_t, b_cond, target_t, mask_t, meta

    def close(self):
        for f in self.files:
            try:
                f.close()
            except Exception:
                pass

# -----------------------------
# Model: ResNet + bounds conditioning -> (x,y)
# -----------------------------
class ResNetXY(nn.Module):
    def __init__(self, pretrained: bool = True, cond_dim: int = 8):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features + cond_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2),  # (x,y) normalized to [0,1]
        )

    def forward(self, img: torch.Tensor, cond: torch.Tensor):
        feats = self.backbone(img)         # [B, F]
        x = torch.cat([feats, cond], dim=1)
        out = self.head(x)                 # [B,2]
        return out

# -----------------------------
# Loss & Metrics
# -----------------------------
def masked_smooth_l1_xy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """
    pred/target: [B, 2] in [0,1] (normalized)
    mask: [B, 2] (1 valid, 0 ignore)
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom

@torch.no_grad()
def pixel_mae_xy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 H: int, W: int, units: str = "pixels") -> float:
    if units == "norm":
        scale = torch.tensor([W, H], device=pred.device, dtype=pred.dtype)
        pred_px   = pred * scale
        target_px = target * scale
    else:
        pred_px, target_px = pred, target
    diff = torch.abs(pred_px - target_px) * mask
    denom = mask.sum().clamp(min=1.0)
    return (diff.sum() / denom).item()

# -----------------------------
# Training / Eval
# -----------------------------
def train_one_epoch(model, loader, opt, scaler, device, beta: float):
    model.train()
    total_loss = 0.0
    for imgs, conds, targets, masks, _ in loader:
        imgs    = imgs.to(device, non_blocking=True)
        conds   = conds.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks   = masks.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None and device.type=="cuda"):
            preds = model(imgs, conds)
            loss = masked_smooth_l1_xy(preds, targets, masks, beta=beta)
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, beta: float, units: str):
    model.eval()
    total_loss = 0.0
    total_mae_px = 0.0
    n_samples = 0
    for imgs, conds, targets, masks, metas in loader:
        imgs    = imgs.to(device, non_blocking=True)
        conds   = conds.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks   = masks.to(device, non_blocking=True)

        preds = model(imgs, conds)
        loss = masked_smooth_l1_xy(preds, targets, masks, beta=beta)
        total_loss += loss.item() * imgs.size(0)

        for j in range(imgs.size(0)):
            H = int(metas["H"][j]); W = int(metas["W"][j])
            mae_px = pixel_mae_xy(preds[j], targets[j], masks[j], H, W, units=units)
            total_mae_px += mae_px
        n_samples += imgs.size(0)

    return total_loss / len(loader.dataset), total_mae_px / max(1, n_samples)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train (x,y) regressor from image + bounds conditioning.")
    parser.add_argument("--h5", nargs="+", required=True, help="Paths to one or more .h5 files.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--only-touch", action="store_true", help="Use only frames with /touch==1.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoints_xy")
    parser.add_argument("--units", choices=["norm", "pixels"], default="pixels", help="Use normalized [0,1] targets/conds or raw pixels.")
    parser.add_argument("--huber-beta", type=float, default=None, help="Smooth-L1 delta. If omitted, auto-sets to 0.1 (norm) or 2.0 (pixels).")
    parser.add_argument("--cond-wh", action="store_true", help="Append image (W,H) to the bounds conditioning when using pixels.")
    args = parser.parse_args()

    if args.huber_beta is None:
        args.huber_beta = 0.1 if args.units == "norm" else 2.0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # Dataset
    dataset = XYFromBoundsH5Dataset(
        args.h5,
        only_touch=args.only_touch,
        color_only_augs=True,
        units=args.units,
        cond_wh=args.cond_wh,
)

    # Train/val split
    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(args.seed))

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=False, persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=False, persistent_workers=(args.num_workers > 0))

    # Model
    cond_dim = 8 + (2 if args.cond_wh and args.units == "pixels" else 0)
    model = ResNetXY(pretrained=not args.no_pretrained, cond_dim=cond_dim).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP scaler (new API)
    scaler = None
    if (not args.no_amp) and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, beta=args.huber_beta)
        val_loss, val_mae_px = evaluate(model, val_loader, device, beta=args.huber_beta, units=args.units)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val MAE(px) {val_mae_px:.2f}")

        # Save best by val loss
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = Path(args.out) / "best_model_xy.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    # Save final
    final_path = Path(args.out) / "last_model_xy.pt"
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, final_path)
    print(f"Saved final checkpoint to {final_path}")

    dataset.close()

if __name__ == "__main__":
    main()
