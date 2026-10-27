# train_bounds.py
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

def has_nan_row(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask [N] True if any value in row is NaN."""
    return np.isnan(arr).reshape(arr.shape[0], -1).any(axis=1)

def to_tensor_image(x: np.ndarray) -> torch.Tensor:
    # x: H,W,3 uint8 -> float tensor [0,1], CHW
    t = torch.from_numpy(x)  # HWC, uint8
    t = t.permute(2, 0, 1).contiguous().float() / 255.0
    return t

# -----------------------------
# Dataset
# -----------------------------
class BoundsH5Dataset(Dataset):
    """
    Accepts /bounds with shape (N, 8) or (N, 4, 2). NaNs allowed.
    """
    def __init__(self, h5_paths: List[str], keep_invalid: bool = False,
                 color_only_augs: bool = True, only_touch: bool = False):
        self.files: List[h5py.File] = [h5py.File(p, "r") for p in h5_paths]
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, frame_idx)
        self.H_list, self.W_list = [], []
        self.bounds_format: List[str] = []  # "flat8" or "quad42" per file
        self.only_touch = only_touch

        for fi, f in enumerate(self.files):
            imgs = f["/images_reshaped"]
            bnds = f["/bounds"]
            assert imgs.ndim == 4 and imgs.shape[-1] == 3, "images_reshaped must be NHWC uint8"
            H, W = imgs.shape[1], imgs.shape[2]
            self.H_list.append(H)
            self.W_list.append(W)

            # detect bounds shape & NaN rows
            if bnds.ndim == 2 and bnds.shape[1] == 8:
                fmt = "flat8"
                b_all = np.array(bnds, dtype=np.float32)                # (N,8)
                nan_mask = np.isnan(b_all).any(axis=1)                  # (N,)
            elif bnds.ndim == 3 and bnds.shape[1:] == (4, 2):
                fmt = "quad42"
                b_all = np.array(bnds, dtype=np.float32).reshape(bnds.shape[0], -1)  # (N,8)
                nan_mask = np.isnan(b_all).any(axis=1)                  # (N,)
            else:
                raise AssertionError(f"/bounds has unsupported shape {bnds.shape}; expected (N,8) or (N,4,2).")
            self.bounds_format.append(fmt)

            # optional /touch filter
            if self.only_touch:
                if "/touch" not in f:
                    raise AssertionError("Requested --only-touch but /touch dataset is missing.")
                touch = np.array(f["/touch"])  # shape (N,1) or (N,)
                touch = touch.reshape(touch.shape[0])                    # -> (N,)
                touch_mask = touch.astype(np.uint8) > 0                  # True where contact
            else:
                # include all by default
                touch_mask = np.ones(imgs.shape[0], dtype=bool)

            # build index map
            for i in range(imgs.shape[0]):
                # keep if touch==1 AND (valid bounds OR keep_invalid)
                if touch_mask[i] and (keep_invalid or (not nan_mask[i])):
                    self.index_map.append((fi, i))

        # Normalization (ImageNet) since we use a pretrained backbone
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Color-only augmentations to preserve geometry
        if color_only_augs:
            self.color_tf = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)
            ])
        else:
            self.color_tf = None

    def __len__(self):
        return len(self.index_map)

    def _load_bounds_as_quad42(self, f: h5py.File, fi: int, i: int) -> np.ndarray:
        """Return bounds as (4,2) float32 (with NaNs preserved)."""
        fmt = self.bounds_format[fi]
        if fmt == "flat8":
            b = np.array(f["/bounds"][i], dtype=np.float32)       # (8,)
            b = b.reshape(4, 2)
        else:  # "quad42"
            b = np.array(f["/bounds"][i], dtype=np.float32)       # (4,2)
        return b

    def __getitem__(self, idx):
        fi, i = self.index_map[idx]
        f = self.files[fi]
        img = f["/images_reshaped"][i]         # H,W,3 uint8
        b = self._load_bounds_as_quad42(f, fi, i)  # (4,2) float32 (may contain NaNs)
        H, W = self.H_list[fi], self.W_list[fi]

        # Inputs
        img_t = to_tensor_image(img)  # [3,H,W], float32 in [0,1]
        if self.color_tf is not None:
            img_t = self.color_tf(img_t)
        img_t = self.norm(img_t)

        # Targets: normalize to [0,1] by width/height
        target = np.copy(b)
        target[:, 0] = target[:, 0] / float(W)
        target[:, 1] = target[:, 1] / float(H)
        target = target.reshape(-1)  # (8,)
        target_t = torch.from_numpy(target)

        # Mask: 1 for valid coords, 0 for NaNs
        mask = ~np.isnan(target)
        mask_t = torch.from_numpy(mask.astype(np.float32))  # (8,)

        meta = {"H": H, "W": W, "file_idx": fi, "frame_idx": i}
        return img_t, target_t, mask_t, meta

    def close(self):
        for f in self.files:
            try:
                f.close()
            except Exception:
                pass

# -----------------------------
# Model
# -----------------------------
class ResNetRegressor(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 8):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = backbone.fc.in_features
        # Replace the classification head with a regression MLP
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)  # [B, in_features]
        out = self.head(feats)    # [B, 8] (x1,y1,x2,y2,x3,y3,x4,y4) TL,TR,BR,BL
        return out

# -----------------------------
# Loss & Metrics
# -----------------------------
def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """
    pred/target: [B, 8], in [0,1]
    mask: [B, 8] (1 valid, 0 ignore)
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    # Apply mask
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom

@torch.no_grad()
def pixel_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, H: int, W: int) -> float:
    """
    Mean absolute error in pixels, averaged across valid coordinates.
    pred/target in [0,1], mask in {0,1}
    """
    # denormalize to pixels
    scale = torch.tensor([W, H] * 4, device=pred.device, dtype=pred.dtype)  # [8]
    pred_px = pred * scale
    target_px = target * scale
    diff = torch.abs(pred_px - target_px)
    diff = diff * mask
    denom = mask.sum().clamp(min=1.0)
    return (diff.sum() / denom).item()

# -----------------------------
# Training / Eval
# -----------------------------
def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, targets, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            preds = model(imgs)
            loss = masked_smooth_l1(preds, targets, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae_px = 0.0
    n_samples = 0
    for imgs, targets, masks, metas in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        preds = model(imgs)
        loss = masked_smooth_l1(preds, targets, masks)
        total_loss += loss.item() * imgs.size(0)

        # Compute pixel MAE per sample using each sample's H,W
        for j in range(imgs.size(0)):
            H = metas["H"][j]
            W = metas["W"][j]
            mae_px = pixel_mae(preds[j], targets[j], masks[j], int(H), int(W))
            total_mae_px += mae_px
        n_samples += imgs.size(0)

    return total_loss / len(loader.dataset), total_mae_px / max(1, n_samples)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train quad-bounds regressor from HDF5 images.")
    parser.add_argument("--h5", nargs="+", required=True, help="Paths to one or more .h5 files.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # Dataset
    dataset = BoundsH5Dataset(args.h5, keep_invalid=False, color_only_augs=True, only_touch=True)

    # Train/val split
    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(args.seed))

    # Dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    # Model
    model = ResNetRegressor(pretrained=not args.no_pretrained, out_dim=8).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp) and device.type == "cuda")

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_mae_px = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val MAE(px) {val_mae_px:.2f}")

        # Save best by val loss
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = Path(args.out) / "best_model.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    # Save final
    final_path = Path(args.out) / "last_model.pt"
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, final_path)
    print(f"Saved final checkpoint to {final_path}")

    # Clean up
    dataset.close()

if __name__ == "__main__":
    main()
