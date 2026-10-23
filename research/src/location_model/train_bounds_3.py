# train_bounds_v2_2.py
import argparse, math, os, random, sys, traceback
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F

# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_tensor_image(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).permute(2,0,1).contiguous().float() / 255.0

def masked_smooth_l1(pred, target, mask, beta: float = 0.1):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5*(diff**2)/beta, diff - 0.5*beta)
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom

@torch.no_grad()
def pixel_mae_batch(pred, target, mask, H, W):
    scale = torch.stack([W, H], dim=1).repeat(1, 4)  # [B,8]
    diff = torch.abs(pred*scale - target*scale) * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (diff.sum(dim=1) / denom).mean()

def _linear_ramp_epoch(epoch, start_epoch, end_epoch, start, end):
    # epoch is 1-based
    if epoch <= start_epoch:
        return start
    if epoch >= end_epoch:
        return end
    t = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    return start + t * (end - start)

# =========================
# Homography helpers
# =========================
def dlt_homography_batch(src_xy: torch.Tensor, dst_xy: torch.Tensor) -> torch.Tensor:
    B = src_xy.shape[0]; device = src_xy.device; dtype = src_xy.dtype
    x,y = src_xy[...,0], src_xy[...,1]
    u,v = dst_xy[...,0], dst_xy[...,1]
    zeros = torch.zeros_like(x); ones = torch.ones_like(x)
    A1 = torch.stack([-x,-y,-ones, zeros,zeros,zeros, x*u,y*u,u], dim=2)
    A2 = torch.stack([ zeros,zeros,zeros, -x,-y,-ones, x*v,y*v,v], dim=2)
    A = torch.zeros((B,8,9), device=device, dtype=dtype)
    A[:,0::2,:] = A1; A[:,1::2,:] = A2
    _,_,Vh = torch.linalg.svd(A)
    h = Vh[:,-1,:]
    H = h.view(B,3,3)
    H = H / (H[:,2:3,2:3] + 1e-8)
    return H

def apply_H_to_points_norm(points_norm, H_mat, H_in, W_in, H_out, W_out):
    orig = points_norm.shape
    pts = points_norm.view(-1,4,2)
    B = pts.shape[0]
    x = pts[...,0] * W_in.view(B,1)
    y = pts[...,1] * H_in.view(B,1)
    p = torch.stack([x,y,torch.ones_like(x)], dim=-1)   # [B,4,3]
    p2 = torch.matmul(p, H_mat.transpose(1,2))
    x2 = p2[...,0] / p2[...,2].clamp_min(1e-8)
    y2 = p2[...,1] / p2[...,2].clamp_min(1e-8)
    x2n = x2 / W_out.view(B,1)
    y2n = y2 / H_out.view(B,1)
    return torch.stack([x2n,y2n], dim=-1).view(orig)

def random_homography_params(B, max_rot_deg=10.0, max_trans=0.05, max_scale=0.10, max_persp=0.001,
                             device="cpu", dtype=torch.float32):
    rot = (torch.rand(B, device=device, dtype=dtype)*2-1) * (max_rot_deg*math.pi/180)
    sx  = 1.0 + (torch.rand(B, device=device, dtype=dtype)*2-1) * max_scale
    sy  = 1.0 + (torch.rand(B, device=device, dtype=dtype)*2-1) * max_scale
    tx  = (torch.rand(B, device=device, dtype=dtype)*2-1) * max_trans
    ty  = (torch.rand(B, device=device, dtype=dtype)*2-1) * max_trans
    g   = (torch.rand(B, device=device, dtype=dtype)*2-1) * max_persp
    h   = (torch.rand(B, device=device, dtype=dtype)*2-1) * max_persp
    return rot,sx,sy,tx,ty,g,h

def params_to_H(B, W, H, rot,sx,sy,tx,ty,g,h):
    device = W.device; dtype = W.dtype
    cosr, sinr = torch.cos(rot), torch.sin(rot)
    RS = torch.zeros(B,3,3, device=device, dtype=dtype)
    RS[:,0,0] = cosr*sx; RS[:,0,1] = -sinr*sy
    RS[:,1,0] = sinr*sx; RS[:,1,1] =  cosr*sy
    RS[:,2,2] = 1.0
    T = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)
    T[:,0,2] = tx*W; T[:,1,2] = ty*H
    P = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)
    P[:,2,0] = g; P[:,2,1] = h
    Hmat = T @ RS @ P
    Hmat = Hmat / (Hmat[:,2:3,2:3] + 1e-8)
    return Hmat

def build_grid_from_H(H_inv: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    # Always compute the grid in float32 for stability
    B = H_inv.shape[0]
    device = H_inv.device
    ys, xs = torch.meshgrid(
        torch.arange(Ht, device=device, dtype=torch.float32),
        torch.arange(Wt, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    tgt = torch.stack([xs, ys, ones], dim=-1).view(1, Ht * Wt, 3).repeat(B, 1, 1)  # [B,N,3]
    src = torch.bmm(tgt, H_inv.transpose(1, 2))  # [B,N,3]
    x = src[:, :, 0] / src[:, :, 2].clamp_min(1e-8)
    y = src[:, :, 1] / src[:, :, 2].clamp_min(1e-8)
    x = (x / max(Wt - 1, 1)) * 2 - 1
    y = (y / max(Ht - 1, 1)) * 2 - 1
    return torch.stack([x, y], dim=-1).view(B, Ht, Wt, 2)  # float32


def warp_images_with_H(imgs: torch.Tensor, H_mat: torch.Tensor) -> torch.Tensor:
    """
    imgs: [B,3,H,W] (may be float16 under autocast)
    H_mat: [B,3,3]
    Returns warped imgs with the same dtype as imgs.
    """
    B, C, Ht, Wt = imgs.shape
    # Force float32 for the matrix inverse and grid
    H_inv32 = torch.linalg.inv(H_mat.to(torch.float32))
    grid32 = build_grid_from_H(H_inv32, Ht, Wt)  # float32

    # grid_sample requires input and grid to have the same dtype; run sampling in float32
    imgs32 = imgs.to(torch.float32)
    out32 = F.grid_sample(imgs32, grid32, mode="bilinear", padding_mode="border", align_corners=True)
    return out32.to(imgs.dtype)

def _linear_ramp(epoch, warmup, total, start, end):
    # epoch is 1-based in your loop; convert to 0-based for intuition
    e = max(0, epoch - 1)
    if e < warmup:
        return start
    span = max(1, total - warmup)
    t = min(1.0, (e - warmup) / span)
    return start + t * (end - start)

# =========================
# Dataset (pairs)
# =========================
class PairBoundsH5Dataset(Dataset):
    def __init__(self, h5_paths: List[str], only_touch=True, color_jitter=True):
        self.files = [h5py.File(p,"r") for p in h5_paths]
        self.index_map: List[Tuple[int,int]] = []
        self.H_list, self.W_list = [], []
        self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.color_tf = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02) if color_jitter else None
        self.only_touch = only_touch

        for fi,f in enumerate(self.files):
            imgs = f["/images_reshaped"]
            bnds = f["/bounds"]
            H,W = imgs.shape[1], imgs.shape[2]
            self.H_list.append(H); self.W_list.append(W)
            if bnds.ndim==2 and bnds.shape[1]==8:
                b_all = np.array(bnds, dtype=np.float32).reshape(-1,4,2)
            elif bnds.ndim==3 and bnds.shape[1:]==(4,2):
                b_all = np.array(bnds, dtype=np.float32)
            else:
                raise AssertionError(f"/bounds has unsupported shape {bnds.shape}")
            nan_mask = np.isnan(b_all).reshape(b_all.shape[0],-1).any(axis=1)
            if self.only_touch:
                if "/touch" not in f: raise AssertionError("Requested --only-touch but /touch missing.")
                touch = np.array(f["/touch"]).reshape(-1).astype(np.uint8) > 0
            else:
                touch = np.ones(b_all.shape[0], dtype=bool)
            valid = (~nan_mask) & touch
            for i in range(len(valid)-1):
                if valid[i] and valid[i+1]:
                    self.index_map.append((fi,i))

    def __len__(self): return len(self.index_map)

    def _load_bounds_quad(self, f, i):
        b = np.array(f["/bounds"][i], dtype=np.float32)
        if b.ndim==1: b = b.reshape(4,2)
        return b

    def __getitem__(self, idx):
        fi,i = self.index_map[idx]
        f = self.files[fi]; imgs = f["/images_reshaped"]
        b0 = self._load_bounds_quad(f,i); b1 = self._load_bounds_quad(f,i+1)
        H0,W0 = self.H_list[fi], self.W_list[fi]

        t0 = to_tensor_image(imgs[i]); t1 = to_tensor_image(imgs[i+1])
        if self.color_tf is not None: t0=self.color_tf(t0); t1=self.color_tf(t1)
        t0=self.norm(t0); t1=self.norm(t1)

        y0 = torch.from_numpy(np.stack([b0[:,0]/W0, b0[:,1]/H0],axis=1).reshape(-1))
        y1 = torch.from_numpy(np.stack([b1[:,0]/W0, b1[:,1]/H0],axis=1).reshape(-1))
        m0 = torch.from_numpy((~np.isnan(b0.reshape(-1))).astype(np.float32))
        m1 = torch.from_numpy((~np.isnan(b1.reshape(-1))).astype(np.float32))
        H_gt = dlt_homography_batch(torch.from_numpy(b0).float().unsqueeze(0),
                                    torch.from_numpy(b1).float().unsqueeze(0)).squeeze(0)
        meta = {"H0":H0,"W0":W0,"H1":H0,"W1":W0,"file_idx":fi,"i":i}
        return t0,t1,y0,y1,m0,m1,H_gt,meta

    def close(self):
        for f in self.files:
            try: f.close()
            except: pass

# =========================
# Model
# =========================
class ResNetRegressor(nn.Module):
    def __init__(self, pretrained=True, out_dim=8):
        super().__init__()
        weights = None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features,512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512,out_dim),
        )
    def forward(self,x):
        return self.head(self.backbone(x))

# =========================
# Losses
# =========================
def temporal_consistency_loss(p0,p1,m0,m1,Hgt,H0,W0,H1,W1):
    B = p0.shape[0]
    p0_to_1 = apply_H_to_points_norm(p0.view(B,4,2), Hgt, H0, W0, H1, W1).view(B,8)
    mask = (m0*m1)
    diff = torch.abs(p1 - p0_to_1) * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum()/denom

def equivariance_loss(model, imgs, preds, mask, H_img, W_img,
                      max_rot=10.0, max_trans=0.05, max_scale=0.10, max_persp=0.001):
    device = imgs.device
    B, _, Ht, Wt = imgs.shape

    # Build A in float32 regardless of autocast
    rot, sx, sy, tx, ty, g, h = random_homography_params(
        B, max_rot, max_trans, max_scale, max_persp, device=device, dtype=torch.float32
    )
    A = params_to_H(B,
                    W_img.to(device).to(torch.float32),
                    H_img.to(device).to(torch.float32),
                    rot, sx, sy, tx, ty, g, h)  # float32
    imgs_aug = warp_images_with_H(imgs, A)  # handles dtype conversions
    preds_aug = model(imgs_aug)
    preds_A = apply_H_to_points_norm(
        preds.view(B, 4, 2), A,
        H_img.to(device).to(torch.float32),
        W_img.to(device).to(torch.float32),
        H_img.to(device).to(torch.float32),
        W_img.to(device).to(torch.float32)
    ).view(B, 8)
    diff = torch.abs(preds_aug - preds_A) * mask.to(device)
    denom = mask.sum().clamp(min=1.0)
    return diff.sum()/denom

# =========================
# Train / Eval
# =========================
@torch.no_grad()
def pixel_and_norm_mae(pred, target, mask, H, W):
    # per-sample px MAE and normalized MAE (÷ max(W,H))
    scale = torch.stack([W, H], dim=1).repeat(1, 4)  # [B,8]
    diff = torch.abs(pred*scale - target*scale) * mask
    denom = mask.sum(dim=1).clamp(min=1.0)           # [B]
    per_px = diff.sum(dim=1) / denom                  # [B]
    per_norm = per_px / torch.maximum(W, H)
    return per_px.mean(), per_norm.mean()

@torch.no_grad()
def evaluate(model, loader, device, debug=False):
    model.eval()
    total_sup, total_mae, n = 0.0, 0.0, 0
    for batch in loader:
        t0,t1,y0,y1,m0,m1,_,meta = batch
        t0=t0.to(device); t1=t1.to(device)
        y0=y0.to(device); y1=y1.to(device)
        m0=m0.to(device); m1=m1.to(device)

        p0 = model(t0); p1 = model(t1)
        sup = 0.5*(masked_smooth_l1(p0,y0,m0)+masked_smooth_l1(p1,y1,m1))
        total_sup += sup.item()*t0.size(0)

        # meta is dict of lists
        H0 = torch.as_tensor(meta["H0"], device=device, dtype=p0.dtype)
        W0 = torch.as_tensor(meta["W0"], device=device, dtype=p0.dtype)
        H1 = torch.as_tensor(meta["H1"], device=device, dtype=p0.dtype)
        W1 = torch.as_tensor(meta["W1"], device=device, dtype=p0.dtype)

        mae0, nmae0 = pixel_and_norm_mae(p0,y0,m0,H0,W0)
        mae1, nmae1 = pixel_and_norm_mae(p1,y1,m1,H1,W1)

        total_mae += 0.5*(mae0+mae1).item()*t0.size(0)
        n += t0.size(0)
    return total_sup/max(1,n), total_mae/max(1,n)

def train_one_epoch(model, loader, opt, scaler, device,
                    lambda_temp, lambda_eq, eq_prob,
                    eq_rot, eq_trans, eq_scale, eq_persp,
                    clip_norm=1.0, debug=False):
    model.train()
    total = 0.0; count = 0
    for batch in loader:
        t0,t1,y0,y1,m0,m1,Hgt,meta = batch
        t0=t0.to(device); t1=t1.to(device)
        y0=y0.to(device); y1=y1.to(device)
        m0=m0.to(device); m1=m1.to(device)
        Hgt=Hgt.to(device)

        H0 = torch.as_tensor(meta["H0"], device=device, dtype=t0.dtype)
        W0 = torch.as_tensor(meta["W0"], device=device, dtype=t0.dtype)
        H1 = torch.as_tensor(meta["H1"], device=device, dtype=t0.dtype)
        W1 = torch.as_tensor(meta["W1"], device=device, dtype=t0.dtype)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=scaler is not None):
            p0 = model(t0); p1 = model(t1)
            L_sup  = 0.5*(masked_smooth_l1(p0,y0,m0) + masked_smooth_l1(p1,y1,m1))
            L_temp = temporal_consistency_loss(p0,p1,m0,m1,Hgt,H0,W0,H1,W1)
            if random.random() < eq_prob:
                L_eq = equivariance_loss(model, t0, p0.detach(), m0, H0, W0,
                                         max_rot=eq_rot, max_trans=eq_trans,
                                         max_scale=eq_scale, max_persp=eq_persp)
            else:
                L_eq = torch.tensor(0.0, device=device, dtype=t0.dtype)
            loss = L_sup + lambda_temp*L_temp + lambda_eq*L_eq

        # Numeric guard
        if not torch.isfinite(loss):
            print(f"!! Skipping batch: non-finite loss {loss.item()}")
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()
        total += loss.item()*t0.size(0); count += t0.size(0)
    return total/max(1,count)


# =========================
# Main
# =========================
def split_files_filewise(h5_paths: List[str], val_split: float, seed: int):
    rng = random.Random(seed); files = list(h5_paths); rng.shuffle(files)
    val_count = max(1, int(len(files)*val_split))
    val_files = files[:val_count]
    train_files = files[val_count:] if len(files) > val_count else files[:1]
    return train_files, val_files

def main():
    parser = argparse.ArgumentParser("Bounds v2.2 (temporal + homography equivariance)")
    parser.add_argument("--h5", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoints_v2_2")
    parser.add_argument("--no-color-jitter", action="store_true")
    parser.add_argument("--only-touch", action="store_true")
    parser.add_argument("--lambda-temp", type=float, default=0.5)
    parser.add_argument("--lambda-eq", type=float, default=0.5)
    parser.add_argument("--eq-prob", type=float, default=0.5)
    parser.add_argument("--eq-max-rot", type=float, default=10.0)
    parser.add_argument("--eq-max-trans", type=float, default=0.05)
    parser.add_argument("--eq-max-scale", type=float, default=0.10)
    parser.add_argument("--eq-max-persp", type=float, default=0.001)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lambda-temp-start", type=float, default=0.25)
    parser.add_argument("--lambda-temp-end", type=float, default=0.5)
    parser.add_argument("--eq-prob-start", type=float, default=0.0)
    parser.add_argument("--eq-prob-end", type=float, default=0.35)
    parser.add_argument("--ramp-end-epoch", type=int, default=8)

    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)  # early stopping patience (epochs)
    parser.add_argument("--cosine-lr", action="store_true")  # enable cosine annealing    
    args = parser.parse_args()

    try:
        print(">> Starting train_bounds_v2_2")
        print("argv:", " ".join(sys.argv))
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(args.out, exist_ok=True)

        train_files, val_files = split_files_filewise(args.h5, args.val_split, args.seed)
        print(f"Train files ({len(train_files)}): {train_files}")
        print(f" Val files ({len(val_files)}): {val_files}")

        train_set = PairBoundsH5Dataset(train_files, only_touch=args.only_touch, color_jitter=(not args.no_color_jitter))
        val_set   = PairBoundsH5Dataset(val_files,   only_touch=args.only_touch, color_jitter=False)
        print(f"Train pairs: {len(train_set)} | Val pairs: {len(val_set)}")
        if len(train_set)==0: print("!! No valid consecutive pairs in train_set (touch + non-NaN).")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)

        # Model + Optimizer + AMP
        model = ResNetRegressor(pretrained=not args.no_pretrained, out_dim=8).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.cosine_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr*0.1
            )
        scaler = None
        if (not args.no_amp) and device.type=="cuda":
            scaler = torch.amp.GradScaler('cuda')

        best_val = math.inf
        best_score = math.inf
        bad_epochs = 0
        min_epochs = 8

        for epoch in range(1, args.epochs+1):
            # Ramp values (warm-up then smooth to target)
            lam_temp_ep = _linear_ramp_epoch(epoch, args.warmup_epochs, args.epochs,
                                    args.lambda_temp_start, args.lambda_temp_end)
            eq_prob_ep  = _linear_ramp_epoch(epoch, args.warmup_epochs, args.epochs,
                                    args.eq_prob_start, args.eq_prob_end)

            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device,
                lambda_temp=lam_temp_ep, lambda_eq=args.lambda_eq,
                eq_prob=eq_prob_ep, eq_rot=args.eq_max_rot, eq_trans=args.eq_max_trans,
                eq_scale=args.eq_max_scale, eq_persp=args.eq_max_persp,
                clip_norm=args.clip_grad_norm
            )
            val_sup, val_mae = evaluate(model, val_loader, device)

            print(f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.4f} | "
                f"val sup loss {val_sup:.4f} | val MAE(px) {val_mae:.2f} | "
                f"lambda_temp {lam_temp_ep:.3f} | eq_prob {eq_prob_ep:.3f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}")

            # Use MAE as the score to minimize
            if val_mae < best_score - 1e-6:
                best_score = val_mae
                bad_epochs = 0
                ckpt = Path(args.out) / "best_by_mae.pt"
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "val_sup_loss": val_sup, "val_mae_px": val_mae,
                            "args": vars(args)}, ckpt)
                print(f"Saved best checkpoint to {ckpt}")
            else:
                bad_epochs += 1
                if epoch >= min_epochs and bad_epochs >= args.patience:
                    print(f"Early stopping after {epoch} (no val MAE improvement for {args.patience} epochs).")
                    break

            if scheduler is not None:
                scheduler.step()

        final_path = Path(args.out)/"last_model_v2_2.pt"
        torch.save({"model_state": model.state_dict(), "epoch": args.epochs, "args": vars(args)}, final_path)
        print(f"Saved final checkpoint to {final_path}")

        train_set.close(); val_set.close()
        print(">> Done")

    except Exception:
        print("!! Fatal error occurred:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
