# train_bounds_seq.py  (diagnostic build)
import argparse, math, os, random, sys, traceback
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_tensor_image(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x).permute(2, 0, 1).contiguous().float() / 255.0
    return t

def _meta_int(metas, key, j):
    v = metas[key]
    if isinstance(v, torch.Tensor):
        return int(v[j].item())
    if isinstance(v, (list, tuple, np.ndarray)):
        return int(v[j])
    return int(v)

# -----------------------------
# Datasets
# -----------------------------
class SequenceBoundsH5Dataset(Dataset):
    """
    Windowed sequence dataset with detailed diagnostics.
    """
    def __init__(self,
                 h5_paths: List[str],
                 seq_len: int = 5,
                 stride: int = 1,
                 keep_invalid: bool = False,
                 color_only_augs: bool = True,
                 only_touch: bool = False,
                 resize: int = None,
                 debug: bool = False):
        assert seq_len >= 1
        self.debug = debug
        # Expand and validate file list
        missing = [p for p in h5_paths if not os.path.exists(p)]
        if missing:
            print("[ERROR] The following --h5 paths do not exist:", flush=True)
            for m in missing: print("  -", m, flush=True)
            raise FileNotFoundError("Some --h5 files were not found.")

        self.files: List[h5py.File] = [h5py.File(p, "r") for p in h5_paths]
        self.seq_len, self.stride = seq_len, stride
        self.keep_invalid, self.only_touch = keep_invalid, only_touch
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, start_idx)
        self.bounds_format: List[str] = []
        self.H_list, self.W_list = [], []
        self.resize = resize

        # Print a banner
        print("=== Dataset Build ===", flush=True)
        print(f"Files: {len(self.files)}", flush=True)
        for i, f in enumerate(h5_paths):
            print(f"  [{i}] {f}", flush=True)

        total_candidates = 0
        total_kept = 0

        # Build window indices
        for fi, f in enumerate(self.files):
            if "/images_reshaped" not in f or "/bounds" not in f:
                raise KeyError(f"[ERROR] File {h5_paths[fi]} is missing required datasets "
                               f"('/images_reshaped' and/or '/bounds').")

            imgs = f["/images_reshaped"]  # [N,H,W,3] uint8
            N = imgs.shape[0]
            H, W = imgs.shape[1], imgs.shape[2]
            self.H_list.append(H); self.W_list.append(W)

            # bounds format & NaN mask
            bnds = f["/bounds"]
            if bnds.ndim == 2 and bnds.shape[1] == 8:
                fmt = "flat8"
                b_all = np.array(bnds, dtype=np.float32)
                nan_mask = np.isnan(b_all).any(axis=1)  # (N,)
            elif bnds.ndim == 3 and bnds.shape[1:] == (4, 2):
                fmt = "quad42"
                b_all = np.array(bnds, dtype=np.float32).reshape(N, -1)
                nan_mask = np.isnan(b_all).any(axis=1)
            else:
                raise AssertionError(f"[ERROR] /bounds has unsupported shape {bnds.shape}; "
                                     f"expected (N,8) or (N,4,2).")
            self.bounds_format.append(fmt)

            # /touch
            if self.only_touch:
                if "/touch" not in f:
                    raise AssertionError("[ERROR] Requested --only-touch but /touch dataset is missing.")
                touch = np.array(f["/touch"]).reshape(N)
                touch_mask = (touch.astype(np.uint8) > 0)
            else:
                touch_mask = np.ones(N, dtype=bool)

            # File-level stats
            num_touch = int(touch_mask.sum())
            num_nan = int(nan_mask.sum())
            print(f"File[{fi}] N={N} HxW={H}x{W} fmt={fmt} touch==1:{num_touch} NaN rows:{num_nan}", flush=True)

            # Build windows
            rejected_examples = 0
            for s in range(0, N - self.seq_len + 1, self.stride):
                e = s + self.seq_len
                total_candidates += 1
                win_touch_ok = touch_mask[s:e].all()
                win_nan_ok = not nan_mask[s:e].any() if not self.keep_invalid else True

                if win_touch_ok and win_nan_ok:
                    self.index_map.append((fi, s))
                    total_kept += 1
                elif self.debug and rejected_examples < 5:
                    reason = []
                    if not win_touch_ok: reason.append("touch==0 in window")
                    if not win_nan_ok:   reason.append("NaN bounds in window")
                    print(f"  reject window [{s}:{e}) -> {', '.join(reason)}", flush=True)
                    rejected_examples += 1

        # Normalization & color augs
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.color_tf = transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                               saturation=0.2, hue=0.02) if color_only_augs else None

        print(f"Total candidate windows: {total_candidates}", flush=True)
        print(f"Total kept windows:      {total_kept}", flush=True)
        print("======================", flush=True)
        if total_kept == 0:
            msg = (
                "[FATAL] Dataset has 0 windows.\n"
                "Possible fixes:\n"
                "  - Remove --only-touch if many frames are hover/no-touch.\n"
                "  - Add --keep-invalid to allow NaN bounds (they’ll be masked in loss).\n"
                "  - Reduce --seq-len so full-touch windows exist.\n"
                "  - Confirm /touch and /bounds are populated as expected."
            )
            print(msg, flush=True)
            # Close files before raising
            self.close()
            raise RuntimeError("Empty dataset after filtering.")

    def __len__(self): return len(self.index_map)

    def _load_bounds_one(self, f: h5py.File, fi: int, idx: int) -> np.ndarray:
        if self.bounds_format[fi] == "flat8":
            b = np.array(f["/bounds"][idx], dtype=np.float32).reshape(4, 2)
        else:
            b = np.array(f["/bounds"][idx], dtype=np.float32)
        return b

    def __getitem__(self, k):
        fi, s = self.index_map[k]
        f = self.files[fi]; T = self.seq_len; e = s + T

        imgs_np = f["/images_reshaped"][s:e]  # [T,H,W,3]
        frames = []
        for t in range(T):
            ft = to_tensor_image(imgs_np[t])
            if self.color_tf is not None: ft = self.color_tf(ft)
            frames.append(self.norm(ft))
        imgs = torch.stack(frames, dim=0)  # [T,3,H,W]
        if self.resize is not None:
            imgs = F.interpolate(imgs, size=(self.resize, self.resize),
                                 mode="bilinear", align_corners=False)

        H, W = self.H_list[fi], self.W_list[fi]
        targets = np.zeros((T, 8), dtype=np.float32)
        masks   = np.zeros((T, 8), dtype=np.float32)
        for t in range(T):
            b = self._load_bounds_one(f, fi, s + t)  # (4,2)
            flat = b.reshape(-1)
            m = ~np.isnan(flat)
            b_norm = np.copy(b)
            b_norm[:, 0] /= float(W); b_norm[:, 1] /= float(H)
            targets[t] = np.nan_to_num(b_norm.reshape(-1), nan=0.0)
            masks[t] = m.astype(np.float32)

        return imgs, torch.from_numpy(targets), torch.from_numpy(masks), {
            "H": H, "W": W, "file_idx": fi, "start_idx": s
        }

    def close(self):
        for f in self.files:
            try: f.close()
            except Exception: pass

# -----------------------------
# Model: CNN -> GRU
# -----------------------------
class TemporalRegressor(nn.Module):
    def __init__(self, out_dim=8, pretrained=True, hidden=512, num_layers=1, bidir=False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.rnn = nn.GRU(input_size=in_feats, hidden_size=hidden,
                          num_layers=num_layers, batch_first=True, bidirectional=bidir)
        head_in = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Linear(head_in, 512), nn.ReLU(True), nn.Dropout(0.2),
                                  nn.Linear(512, out_dim))

    def forward(self, x_seq: torch.Tensor):  # x_seq: [B,T,3,H,W]
        B,T,C,H,W = x_seq.shape
        x = x_seq.reshape(B*T, C, H, W)
        feats = self.backbone(x).view(B, T, -1)
        out, _ = self.rnn(feats)
        return self.head(out)  # [B,T,8]

# -----------------------------
# Loss & Metrics
# -----------------------------
def masked_smooth_l1(pred, target, mask, beta: float = 0.1):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom

def accel_penalty(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if preds.size(1) < 3: return torch.zeros((), device=preds.device)
    acc = preds[:, 2:] - 2 * preds[:, 1:-1] + preds[:, :-2]
    m = masks[:, 2:] * masks[:, 1:-1] * masks[:, :-2]
    denom = m.sum().clamp(min=1.0)
    return (acc.abs() * m).sum() / denom

@torch.no_grad()
def pixel_mae_seq(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, H: int, W: int) -> float:
    scale = torch.tensor([W, H] * 4, device=pred.device, dtype=pred.dtype)
    diff = torch.abs(pred * scale - target * scale) * mask
    denom = mask.sum().clamp(min=1.0)
    return (diff.sum() / denom).item()

# -----------------------------
# Training / Eval
# -----------------------------
from contextlib import nullcontext

def train_one_epoch(model, loader, opt, scaler, device, lambda_acc: float = 0.0,
                    autocast_cm=None, use_amp: bool = False):
    model.train()
    total_loss = 0.0
    for imgs, targets, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)       # [B,T,3,H,W]
        targets = targets.to(device, non_blocking=True) # [B,T,8]
        masks = masks.to(device, non_blocking=True)     # [B,T,8]

        opt.zero_grad(set_to_none=True)

        cm = autocast_cm(enabled=use_amp) if (autocast_cm is not None and use_amp) else nullcontext()
        with cm:
            preds = model(imgs)                         # [B,T,8]
            loss = masked_smooth_l1(preds, targets, masks)
            if lambda_acc > 0.0:
                loss = loss + lambda_acc * accel_penalty(preds, masks)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, scaler=None, autocast_cm=None, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_mae_px = 0.0
    n_samples = 0

    for imgs, targets, masks, metas in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        cm = autocast_cm(enabled=use_amp) if (autocast_cm is not None and use_amp) else nullcontext()
        with cm:
            preds = model(imgs)
            loss = masked_smooth_l1(preds, targets, masks)

        total_loss += loss.item() * imgs.size(0)

        # Per-sequence MAE averaged over frames
        B = imgs.size(0)
        for j in range(B):
            H = _meta_int(metas, "H", j)
            W = _meta_int(metas, "W", j)
            total_mae_px += pixel_mae_seq(preds[j], targets[j], masks[j], H, W)
        n_samples += B

    return total_loss / len(loader.dataset), total_mae_px / max(1, n_samples)

# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Train temporal quad-bounds regressor (CNN→GRU) from HDF5 images.")
    p.add_argument("--h5", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="checkpoints_seq")

    # Sequence & model
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--seq-stride", type=int, default=1)
    p.add_argument("--keep-invalid", action="store_true")
    p.add_argument("--only-touch", action="store_true")
    p.add_argument("--resize", type=int, default=None)
    p.add_argument("--rnn-hidden", type=int, default=512)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--bidir", action="store_true")
    p.add_argument("--lambda-acc", type=float, default=0.0)
    p.add_argument("--debug", action="store_true", help="Print reasons for rejected windows (first few).")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # Dataset
    dataset = SequenceBoundsH5Dataset(
        args.h5, seq_len=args.seq_len, stride=args.seq_stride,
        keep_invalid=args.keep_invalid, color_only_augs=True,
        only_touch=args.only_touch, resize=args.resize,
        debug=args.debug
    )
    print(f"[INFO] Dataset windows: {len(dataset)}", flush=True)

    # Train/val split
    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        print(f"[FATAL] Not enough windows for train/val split. "
              f"len={len(dataset)} val_len={val_len} train_len={train_len}", flush=True)
        print("Hint: reduce --val-split or relax filters (--keep-invalid / remove --only-touch / reduce --seq-len).", flush=True)
        dataset.close(); sys.exit(2)

    train_set, val_set = random_split(dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(args.seed))

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Model / Optim
    model = TemporalRegressor(out_dim=8, pretrained=not args.no_pretrained,
                              hidden=args.rnn_hidden, num_layers=args.rnn_layers,
                              bidir=args.bidir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP (new API with fallback)
    use_amp = (not args.no_amp) and (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        def autocast_cm(enabled):
            return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        def autocast_cm(enabled):
            return torch.cuda.amp.autocast(enabled=enabled)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                     lambda_acc=args.lambda_acc,
                                     autocast_cm=autocast_cm, use_amp=use_amp)
        val_loss, val_mae_px = evaluate(model, val_loader, device,
                                        scaler=scaler, autocast_cm=autocast_cm, use_amp=use_amp)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val MAE(px) {val_mae_px:.2f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = Path(args.out) / "best_model_seq.pt"
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}", flush=True)

    final_path = Path(args.out) / "last_model_seq.pt"
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs}, final_path)
    print(f"Saved final checkpoint to {final_path}", flush=True)
    dataset.close()

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"[EXIT] code={e.code}", flush=True)
        raise
    except Exception:
        print("[UNCAUGHT EXCEPTION]", flush=True)
        traceback.print_exc()
        sys.exit(1)
