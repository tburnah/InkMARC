import argparse
import math
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
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
# Dataset: (image, bounds) -> (x,y) target
# -----------------------------
class XYFromBoundsH5Dataset(Dataset):
    """
    Uses:
      /images_reshaped : (N, 448, 448, 3) uint8
      /bounds          : (N,8) or (N,4,2) float32 (no NaNs retained, filtered out)
      /xy              : (N,2) float32 (pixels)
      /touch           : (N,1) uint8 (optional)
    Returns img tensor [3,448,448], bounds conditioning (8 or 10), target (x,y) in pixels.
    """
    def __init__(
        self,
        h5_paths: List[str],
        only_touch: bool = False,
        color_only_augs: bool = True,
        cond_wh: bool = False,
    ):
        self.files: List[h5py.File] = [h5py.File(p, "r") for p in h5_paths]
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, frame_idx)
        self.H_list, self.W_list = [], []
        self.bounds_format: List[str] = []  # "flat8" or "quad42" per file
        self.only_touch = only_touch
        self.cond_wh = cond_wh

        # Build index
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
                b_nan = np.isnan(b_all).any(axis=1)
            elif bnds.ndim == 3 and bnds.shape[1:] == (4, 2):
                fmt = "quad42"
                b_all = np.array(bnds, dtype=np.float32).reshape(bnds.shape[0], -1)  # (N,8)
                b_nan = np.isnan(b_all).any(axis=1)
            else:
                raise AssertionError(f"/bounds has unsupported shape {bnds.shape}; expected (N,8) or (N,4,2).")
            self.bounds_format.append(fmt)

            # xy validity
            xy_all = np.array(xy, dtype=np.float32)                    # (N,2)
            xy_nan = np.isnan(xy_all).any(axis=1)

            # optional /touch filter
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

        # Confirm all images are same size (expected 448x448)
        if len(set(self.H_list)) > 1 or len(set(self.W_list)) > 1:
            raise AssertionError("All images must share the same size for batched heatmaps.")

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
        fmt = self.bounds_format[fi]
        if fmt == "flat8":
            b = np.array(f["/bounds"][i], dtype=np.float32).reshape(4, 2)
        else:
            b = np.array(f["/bounds"][i], dtype=np.float32)  # (4,2)
        return b

    def __getitem__(self, idx):
        fi, i = self.index_map[idx]
        f = self.files[fi]
        img = f["/images_reshaped"][i]                # H,W,3 uint8
        b   = self._load_bounds_as_quad42(f, fi, i)   # (4,2) float32
        xy  = np.array(f["/xy"][i], dtype=np.float32) # (2,)
        H, W = self.H_list[fi], self.W_list[fi]

        # Inputs
        img_t = to_tensor_image(img)  # [3,H,W]
        if self.color_tf is not None:
            img_t = self.color_tf(img_t)
        img_t = self.norm(img_t)

        # Bounds conditioning (pixels)
        cond = b.reshape(-1).astype(np.float32)       # (8,)
        if self.cond_wh:
            cond = np.concatenate([cond, np.array([float(W), float(H)], dtype=np.float32)])  # -> (10,)
        b_cond = torch.from_numpy(cond)               # (8,) or (10,)

        # Target xy in pixels
        target_t = torch.from_numpy(xy.astype(np.float32))  # (2,)

        meta = {"H": H, "W": W, "file_idx": fi, "frame_idx": i}
        return img_t, b_cond, target_t, meta

    def close(self):
        for f in self.files:
            try: f.close()
            except Exception: pass

# -----------------------------
# Model: ResNet18 + FiLM conditioning -> Heatmap
# -----------------------------
class HeatmapXY(nn.Module):
    """
    - Backbone: ResNet18 without avgpool/fc -> feature map [B, C=512, Hf, Wf]
    - FiLM: bounds cond -> gamma,beta in R^C, applied per-channel
    - Head: 1x1 conv -> logits [B,1,Hf,Wf]
    """
    def __init__(self, pretrained: bool = True, cond_dim: int = 8):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # chop off avgpool/fc
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = 512  # resnet18 layer4 out channels

        # FiLM: cond -> gamma,beta (each C)
        self.film = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.out_channels * 2)
        )

        # heatmap head
        self.head = nn.Conv2d(self.out_channels, 1, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B,512,Hf,Wf]
        return x

    def forward(self, img: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(img)  # [B,C,Hf,Wf]
        # FiLM
        gb = self.film(cond)            # [B, 2C]
        gamma, beta = torch.split(gb, self.out_channels, dim=1)  # [B,C], [B,C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)   # [B,C,1,1]
        f = (1.0 + gamma) * f + beta

        logits = self.head(f)           # [B,1,Hf,Wf]
        return logits

# -----------------------------
# Heatmap Utilities
# -----------------------------
def make_target_dist_xy(xy_px: torch.Tensor, Hf: int, Wf: int, stride: int, device: torch.device):
    """
    xy_px: [B,2] in pixels (x,y)
    Returns a normalized target distribution [B,1,Hf,Wf] that sums to 1.
    The Gaussian sigma is tied to stride (good default: sigma_cells = 1.5).
    """
    B = xy_px.shape[0]
    Hf = int(Hf); Wf = int(Wf); stride = int(stride)

    # grid cell centers in pixels
    xs_px = (torch.arange(Wf, device=device, dtype=torch.float32) + 0.5) * stride
    ys_px = (torch.arange(Hf, device=device, dtype=torch.float32) + 0.5) * stride
    Xs = xs_px.view(1, 1, 1, Wf)  # [1,1,1,Wf]
    Ys = ys_px.view(1, 1, Hf, 1)  # [1,1,Hf,1]

    x = xy_px[:, 0].view(B, 1, 1, 1)
    y = xy_px[:, 1].view(B, 1, 1, 1)

    # tie sigma to grid resolution
    sigma_cells = 1.5
    sigma_px = sigma_cells * stride
    denom = 2.0 * (sigma_px ** 2)

    dist2 = (Xs - x) ** 2 + (Ys - y) ** 2
    hm = torch.exp(-dist2 / denom)              # [B,1,Hf,Wf], unnormalized
    hm = hm / (hm.view(B, -1).sum(dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-12)
    return hm

def make_gaussian_heatmaps_xy(
    xy: torch.Tensor, Hf: int, Wf: int, stride: int, sigma_px: float, device: torch.device
) -> torch.Tensor:
    Hf = int(Hf); Wf = int(Wf)  # <— ensure Python ints
    B = xy.shape[0]

    xs_px = (torch.arange(Wf, dtype=torch.float32, device=device) + 0.5) * stride  # [Wf]
    ys_px = (torch.arange(Hf, dtype=torch.float32, device=device) + 0.5) * stride  # [Hf]
    Xs = xs_px.view(1, 1, 1, Wf)  # [1,1,1,Wf]
    Ys = ys_px.view(1, 1, Hf, 1)  # [1,1,Hf,1]

    x = xy[:, 0].view(B, 1, 1, 1)
    y = xy[:, 1].view(B, 1, 1, 1)

    dist2 = (Xs - x) ** 2 + (Ys - y) ** 2
    denom = 2.0 * (sigma_px ** 2)
    return torch.exp(-dist2 / denom)

def heatmap_soft_ce_loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,1,Hf,Wf] (unnormalized)
    target_dist: [B,1,Hf,Wf] with sum=1 over Hf*Wf
    Loss = - sum target * log_softmax(logits)
    """
    B, _, Hf, Wf = logits.shape
    logP = torch.log_softmax(logits.view(B, -1), dim=1)
    T = target_dist.view(B, -1).clamp_min(1e-12)
    return -(T * logP).sum(dim=1).mean()

@torch.no_grad()
def soft_argmax_xy(logits: torch.Tensor, stride: int, temp: float) -> torch.Tensor:
    """
    logits: [B,1,Hf,Wf]; returns [B,2] in pixels
    """
    B, _, Hf, Wf = logits.shape
    device = logits.device
    stride = int(stride)

    probs = torch.softmax(logits.view(B, -1) / max(1e-6, temp), dim=1).view(B, 1, Hf, Wf)

    xs_px = (torch.arange(Wf, device=device, dtype=torch.float32) + 0.5) * stride
    ys_px = (torch.arange(Hf, device=device, dtype=torch.float32) + 0.5) * stride
    Xs = xs_px.view(1, 1, 1, Wf)  # [1,1,1,Wf]
    Ys = ys_px.view(1, 1, Hf, 1)  # [1,1,Hf,1]

    x = (probs * Xs).sum(dim=(2, 3))
    y = (probs * Ys).sum(dim=(2, 3))
    return torch.cat([x, y], dim=1)  # [B,2]


# -----------------------------
# Loss & Metrics
# -----------------------------
def heatmap_loss_mse(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE between sigmoid(logits) and target heatmap [0,1].
    """
    pred = torch.sigmoid(logits)
    return torch.mean((pred - target) ** 2)

@torch.no_grad()
def pixel_mae_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> float:
    """
    pred_xy, target_xy: [B,2] in pixels
    """
    diff = torch.abs(pred_xy - target_xy)
    return diff.mean().item()

# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, opt, scaler, device, sigma_px: float, stride: int, temp: float):
    model.train()
    total_loss = 0.0
    for imgs, conds, targets_xy, metas in loader:
        imgs    = imgs.to(device, non_blocking=True)
        conds   = conds.to(device, non_blocking=True)
        targets_xy = targets_xy.to(device, non_blocking=True)

        # derive from batch tensor to avoid meta-collate issues
        H, W = imgs.shape[-2], imgs.shape[-1]
        Hf, Wf = H // stride, W // stride

        target_hm = make_gaussian_heatmaps_xy(targets_xy, Hf, Wf, stride, sigma_px, imgs.device)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            logits = model(imgs, conds)            # [B,1,Hf,Wf]
            Hf, Wf = logits.shape[-2], logits.shape[-1]
            stride = imgs.shape[-1] // Wf          # robust: infer stride
            target_dist = make_target_dist_xy(targets_xy, Hf, Wf, stride, imgs.device)
            loss = heatmap_soft_ce_loss(logits, target_dist)

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
def evaluate(model, loader, device, sigma_px: float, stride: int, temp: float):
    """
    Note: sigma_px and stride are unused here (kept for CLI/API compatibility).
    We infer the effective stride from the model output grid vs input size.
    """
    model.eval()
    total_loss = 0.0
    total_mae_px = 0.0
    n_samples = 0

    for imgs, conds, targets_xy, metas in loader:
        imgs = imgs.to(device, non_blocking=True)
        conds = conds.to(device, non_blocking=True)
        targets_xy = targets_xy.to(device, non_blocking=True)

        # Forward
        logits = model(imgs, conds)                     # [B,1,Hf,Wf]
        B = imgs.size(0)
        Hf, Wf = logits.shape[-2], logits.shape[-1]

        # Infer stride from input size vs output grid
        stride_h = imgs.shape[-2] // Hf
        stride_w = imgs.shape[-1] // Wf
        stride_model = int((stride_h + stride_w) // 2)

        # Build soft target distribution on the output grid
        target_dist = make_target_dist_xy(
            targets_xy, Hf=Hf, Wf=Wf, stride=stride_model, device=imgs.device
        )                                               # [B,1,Hf,Wf]

        # CE/KL-style loss with soft targets
        loss = heatmap_soft_ce_loss(logits, target_dist)
        total_loss += loss.item() * B

        # Decode with soft-argmax on the same grid
        pred_xy = soft_argmax_xy(logits, stride=stride_model, temp=temp)  # [B,2] in pixels

        # L1 MAE in pixels (|dx| + |dy|), averaged over batch
        batch_mae = (pred_xy - targets_xy).abs().sum(dim=1).mean().item()
        total_mae_px += batch_mae * B
        n_samples += B

    return total_loss / max(1, n_samples), total_mae_px / max(1, n_samples)

# -----------------------------
# Group (file-wise) split
# -----------------------------
def make_group_split(dataset: XYFromBoundsH5Dataset, val_frac: float, seed: int):
    rng = random.Random(seed)
    # collect indices by file id
    by_file: Dict[int, List[int]] = {}
    for k, (fi, _) in enumerate(dataset.index_map):
        by_file.setdefault(fi, []).append(k)
    files = list(by_file.keys())
    rng.shuffle(files)
    n_val_files = max(1, int(len(files) * val_frac))
    val_files = set(files[:n_val_files])
    val_idx  = [k for fi in files if fi in val_files for k in by_file[fi]]
    train_idx = [k for fi in files if fi not in val_files for k in by_file[fi]]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Heatmap head for (x,y) from image + bounds conditioning.")
    parser.add_argument("--h5", nargs="+", required=True, help="Paths to one or more .h5 files.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--only-touch", action="store_true", help="Use only frames with /touch==1.")
    parser.add_argument("--cond-wh", action="store_true", help="Append (W,H) to bounds conditioning.")
    parser.add_argument("--sigma-px", type=float, default=5.0, help="Gaussian sigma for target heatmap in pixels.")
    parser.add_argument("--softmax-temp", type=float, default=1.0, help="Softmax temperature for soft-argmax.")
    parser.add_argument("--split-by-file", action="store_true", help="Validate on held-out files/sessions.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoints_xy_hm")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # Dataset
    dataset = XYFromBoundsH5Dataset(
        args.h5, only_touch=args.only_touch, color_only_augs=True, cond_wh=args.cond_wh
    )

    # Constant sizes (expected 448x448)
    H = dataset.H_list[0]
    W = dataset.W_list[0]
    # ResNet18 output stride
    stride = 32
    assert H % stride == 0 and W % stride == 0, "Input size must be multiple of 32 for ResNet18 stride."
    Hf, Wf = H // stride, W // stride

    # Split
    if args.split_by_file:
        train_set, val_set = make_group_split(dataset, args.val_split, args.seed)
    else:
        # random frame-based split (less strict)
        n = len(dataset)
        val_len = max(1, int(n * args.val_split))
        idx = list(range(n))
        random.Random(args.seed).shuffle(idx)
        val_idx = idx[:val_len]
        train_idx = idx[val_len:]
        train_set, val_set = Subset(dataset, train_idx), Subset(dataset, val_idx)

    # Loaders
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0)
    )

    # Model
    cond_dim = 8 + (2 if args.cond_wh else 0)
    model = HeatmapXY(pretrained=not args.no_pretrained, cond_dim=cond_dim).to(device)

    # Optimizer & AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if ((not args.no_amp) and device.type == "cuda") else None

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            sigma_px=args.sigma_px, stride=stride, temp=args.softmax_temp
        )
        val_loss, val_mae_px = evaluate(
            model, val_loader, device,
            sigma_px=args.sigma_px, stride=stride, temp=args.softmax_temp
        )

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val MAE(px) {val_mae_px:.2f}")

        # Save best by val loss
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = Path(args.out) / "best_model_xy_heatmap.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "sigma_px": args.sigma_px,
                "softmax_temp": args.softmax_temp,
                "stride": stride,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    # Save final
    final_path = Path(args.out) / "last_model_xy_heatmap.pt"
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, final_path)
    print(f"Saved final checkpoint to {final_path}")

    dataset.close()

if __name__ == "__main__":
    main()
