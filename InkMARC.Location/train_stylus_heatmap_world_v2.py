# train_stylus_heatmap_world_v2.py
import argparse, os, sys, math, random, json, csv, traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import h5py, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models, transforms

# -----------------------------
# Utils
# -----------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_tensor_image(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).permute(2,0,1).contiguous().float() / 255.0  # [3,H,W]

# replace your softargmax2d with this safer version
def softargmax2d(hm: torch.Tensor, beta: float = 50.0):
    """hm: [B,1,Hm,Wm] logits -> x,y in heatmap coords (0..Wm-1 / 0..Hm-1) + normalized prob."""
    B, C, Hm, Wm = hm.shape; assert C == 1
    hm32 = hm.to(torch.float32)
    p32 = F.softmax(hm32.view(B,1,-1) * beta, dim=-1).view(B,1,Hm,Wm)
    xs = torch.linspace(0, Wm-1, Wm, device=hm.device, dtype=torch.float32).view(1,1,1,Wm)
    ys = torch.linspace(0, Hm-1, Hm, device=hm.device, dtype=torch.float32).view(1,1,Hm,1)
    x = (p32*xs).sum(dim=(2,3)).to(hm.dtype).view(B)
    y = (p32*ys).sum(dim=(2,3)).to(hm.dtype).view(B)
    return x, y, p32.to(hm.dtype)

def make_gaussian(Hm:int, Wm:int, cx:float, cy:float, sigma:float, device, dtype):
    ys = torch.arange(Hm, device=device, dtype=dtype).view(Hm,1)
    xs = torch.arange(Wm, device=device, dtype=dtype).view(1,Wm)
    g = torch.exp(-0.5 * ((xs-cx)**2 + (ys-cy)**2) / (sigma**2))
    return (g / g.max().clamp_min(1e-8)).unsqueeze(0)  # [1,Hm,Wm]

# ---- Homography helpers (float32 for stability)
def dlt_H(src_xy: torch.Tensor, dst_xy: torch.Tensor) -> torch.Tensor:
    """src_xy,dst_xy: [4,2] -> 3x3 H (src->dst)"""
    x,y = src_xy[:,0], src_xy[:,1]
    u,v = dst_xy[:,0], dst_xy[:,1]
    z,o = torch.zeros_like(x), torch.ones_like(x)
    A1 = torch.stack([-x,-y,-o,  z, z, z, x*u, y*u, u], dim=1)
    A2 = torch.stack([ z, z, z, -x,-y,-o, x*v, y*v, v], dim=1)
    A  = torch.stack([A1,A2], dim=1).reshape(-1,9)
    _,_,Vh = torch.linalg.svd(A)
    h = Vh[-1]; H = h.view(3,3)
    return H / (H[2,2].clamp_min(1e-8))

def apply_H(xy: torch.Tensor, H: torch.Tensor):
    """xy: [...,2]; H: [3,3] -> mapped [...,2]"""
    orig = xy.shape; pts = xy.view(-1,2)
    ones = torch.ones((pts.size(0),1), device=pts.device, dtype=pts.dtype)
    q = (H @ torch.cat([pts,ones], dim=1).T).T
    out = q[:,:2] / q[:,2:].clamp_min(1e-8)
    return out.view(orig)

def quad_area_xy(b: np.ndarray) -> float:
    # b: 4x2, any order; polygon area via shoelace
    x = b[:, 0]; y = b[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return 0.5 * abs(float(area2))

def valid_bounds(b: np.ndarray, min_area: float = 25.0) -> bool:
    if not np.isfinite(b).all(): return False
    if b.shape != (4,2): return False
    return quad_area_xy(b) >= min_area

def fix_bottom_order(b: np.ndarray) -> np.ndarray:
    """
    Ensure bounds are in TL, TR, BR, BL order.
    Assumes TL, TR are correct. If BL and BR are swapped, fix them.
    """
    b = np.asarray(b, dtype=np.float32).reshape(4,2)
    # TL, TR, BR, BL (expected order)
    TL, TR, C2, C3 = b

    # Decide which of C2, C3 is BR vs BL
    # Compare x-coordinates: BR.x > BL.x (since X increases rightward)
    if C2[0] < C3[0]:
        BL, BR = C2, C3
    else:
        BR, BL = C2, C3

    return np.array([TL, TR, BR, BL], dtype=np.float32)

def canonicalize_corners(b: np.ndarray) -> np.ndarray:
    """
    Robustly order 4 points to [TL, TR, BR, BL] in image pixel coords
    where y increases downward, x increases to the right.
    """
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    # sort by y (top row first)
    idx_by_y = np.argsort(b[:, 1])
    top2 = b[idx_by_y[:2]]
    bot2 = b[idx_by_y[2:]]

    # within each row, sort by x (left to right)
    TL, TR = top2[np.argsort(top2[:, 0])]
    BL, BR = bot2[np.argsort(bot2[:, 0])]

    return np.stack([TL, TR, BR, BL], axis=0).astype(np.float32)

# -----------------------------
# Session params (W,H,x0,y0, flips)
# -----------------------------
def load_session_map(path: Optional[str]) -> Dict[str, Dict]:
    """
    Accepts JSON or CSV mapping from file name (basename or full path) to:
      { "w": ..., "h": ..., "x0": 0.0, "y0": 0.0, "flip_u": 0/1, "flip_v": 0/1 }
    """
    if not path: return {}
    ext = Path(path).suffix.lower()
    mapping = {}
    if ext == ".json":
        with open(path,"r") as f:
            mapping = json.load(f)
    elif ext in (".csv",".tsv"):
        delim = "," if ext==".csv" else "\t"
        with open(path,"r", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                key = row.get("file") or row.get("name") or row.get("path")
                if not key: continue
                mapping[key] = {
                    "w": float(row.get("w", row.get("width", 1.0))),
                    "h": float(row.get("h", row.get("height",1.0))),
                    "x0": float(row.get("x0", 0.0)),
                    "y0": float(row.get("y0", 0.0)),
                    "flip_u": int(row.get("flip_u", 0)),
                    "flip_v": int(row.get("flip_v", 0)),
                }
    else:
        raise ValueError("world-sizes mapping must be .json, .csv, or .tsv")
    return mapping

def resolve_session_params(h5_path: str,
                           global_params: Dict,
                           mapping: Dict[str, Dict]) -> Dict:
    """
    Priority: explicit mapping by exact path -> by basename -> global defaults.
    """
    base = os.path.basename(h5_path)
    p = mapping.get(h5_path) or mapping.get(base) or {}
    out = {
        "w":   float(p.get("w",   global_params["w"])),
        "h":   float(p.get("h",   global_params["h"])),
        "x0":  float(p.get("x0",  global_params["x0"])),
        "y0":  float(p.get("y0",  global_params["y0"])),
        "flip_u": int(p.get("flip_u", global_params["flip_u"])),
        "flip_v": int(p.get("flip_v", global_params["flip_v"])),
    }
    return out

# -----------------------------
# Dataset
# -----------------------------
class StylusWorldDataset(Dataset):
    """
    Inputs per frame: image (/images_reshaped), corners in px (/bounds TL,TR,BR,BL), stylus world XY (/xy).
    Uses per-session world params (W,H,x0,y0, flips) to convert world <-> UV for supervision and metrics.
    """
    def __init__(self, h5_paths: List[str], world_key="/xy",
                 only_touch=False, heatmap_stride=4, sigma_hm=1.5,
                 global_params=None, mapping=None, color_jitter=True):
        self.files = [h5py.File(p,"r") for p in h5_paths]
        self.index: List[Tuple[int,int]] = []
        self.H_list, self.W_list = [], []
        self.world_key = world_key
        self.only_touch = only_touch
        self.sigma_hm = sigma_hm
        self.stride = heatmap_stride
        self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.cj = transforms.ColorJitter(0.25,0.25,0.2,0.02) if color_jitter else None

        # resolve per-file session params
        self.session_params = []
        for fi, h5p in enumerate(h5_paths):
            self.session_params.append(resolve_session_params(h5p, global_params, mapping or {}))

        for fi, f in enumerate(self.files):
            imgs = f["/images_reshaped"]
            H, W = imgs.shape[1], imgs.shape[2]
            self.H_list.append(H); self.W_list.append(W)
            if "/bounds" not in f: raise AssertionError("Missing /bounds")
            if self.world_key not in f: raise AssertionError(f"Missing {self.world_key}")

            # touch filter (optional)
            touch = np.ones(imgs.shape[0], dtype=bool)
            if self.only_touch:
                if "/touch" not in f: raise AssertionError("Requested --only-touch but /touch missing.")
                touch = np.array(f["/touch"]).reshape(-1).astype(np.uint8) > 0

            # adaptive area threshold: max(25 px², 0.0001 * image_area)
            min_area = max(25.0, 1e-4 * (H * W))

            # counters (optional, to see what's filtered)
            skipped_touch = skipped_bounds = skipped_xy = 0

            for i in range(imgs.shape[0]):
                if not touch[i]:
                    skipped_touch += 1
                    continue

                # bounds: handle (8,) or (4,2)                
                b = np.array(f["/bounds"][i], dtype=np.float32)
                b = b.reshape(4, 2) if b.ndim == 1 else b
                b = canonicalize_corners(b)
                if not valid_bounds(b, min_area):
                    skipped_bounds += 1
                    continue
                # world point (/xy)
                xy = np.array(f[self.world_key][i], dtype=np.float32).reshape(2)
                if not np.isfinite(xy).all():
                    skipped_xy += 1
                    continue

                # passed all checks → keep
                self.index.append((fi, i))

            # Optional: print per-file summary once
            kept = len([1 for j in range(imgs.shape[0]) if touch[j]]) - skipped_bounds - skipped_xy
            print(f"[{fi}] kept {kept} / {imgs.shape[0]} | skipped touch {skipped_touch}, "
                f"bounds {skipped_bounds}, xy {skipped_xy}")

    def __len__(self): return len(self.index)

    def _bounds(self, f, i):
        b = np.array(f["/bounds"][i], dtype=np.float32)
        b = b.reshape(4,2) if b.ndim==1 else b.astype(np.float32)
        return canonicalize_corners(b)

    def _world_xy(self, f, i):
        xy = np.array(f[self.world_key][i], dtype=np.float32).reshape(2)
        return xy

    def __getitem__(self, idx):
        fi,i = self.index[idx]; f = self.files[fi]
        img = f["/images_reshaped"][i]  # H,W,3
        H, W = self.H_list[fi], self.W_list[fi]
        bounds = self._bounds(f,i)      # (4,2) TL,TR,BR,BL in px
        xy_world = self._world_xy(f,i)  # (2,) in world units

        # session params
        sp = self.session_params[fi]
        w,h,x0,y0,fu,fv = sp["w"], sp["h"], sp["x0"], sp["y0"], sp["flip_u"], sp["flip_v"]

        # Image tensor
        img_t = to_tensor_image(img)
        if self.cj is not None: img_t = self.cj(img_t)
        img_t = self.norm(img_t)

        # World -> UV (0..1), apply flips
        u = (xy_world[0] - x0) / max(w, 1e-8)
        v = (xy_world[1] - y0) / max(h, 1e-8)
        if fu: u = 1.0 - u
        if fv: v = 1.0 - v
        uv = np.array([u, v], dtype=np.float32)

        # UV -> image px using per-frame homography
        uv_corners = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=torch.float32)
        H_uv2img = dlt_H(uv_corners, torch.from_numpy(bounds).to(torch.float32))
        xy_img = apply_H(torch.from_numpy(uv.reshape(1,2)), H_uv2img).view(2).numpy()

        # Heatmap target in heatmap coords
        Hm, Wm = H//self.stride, W//self.stride
        cx = float(np.clip(xy_img[0] / self.stride, 0, Wm-1))
        cy = float(np.clip(xy_img[1] / self.stride, 0, Hm-1))
        hm = make_gaussian(Hm, Wm, cx, cy, self.sigma_hm, img_t.device, img_t.dtype)  # [1,Hm,Wm]

        sample = {
            "image": img_t,
            "heatmap_target": hm,
            "bounds": torch.from_numpy(bounds),            # [4,2] px
            "xy_world": torch.from_numpy(xy_world),        # [2]
            "H_img": torch.tensor(H), "W_img": torch.tensor(W),
            "session": torch.tensor(fi),                   # file index
            "uv_gt": torch.tensor([u, v], dtype=torch.float32)
        }
        return sample

    def close(self):
        for f in self.files:
            try: f.close()
            except: pass

# -----------------------------
# Model: ResNet -> upsampled heatmap (1 channel)
# -----------------------------
class HeatmapHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        self.stem = nn.Sequential(*(list(net.children())[:-2]))  # [B,512,H/32,W/32]
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.ReLU(True),  # /16
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),  # /8
            nn.ConvTranspose2d(128, 64,4,2,1), nn.ReLU(True),  # /4
            nn.Conv2d(64,1,1)
        )
    def forward(self,x): return self.up(self.stem(x))  # [B,1,H/4,W/4]

# -----------------------------
# Losses / Metrics
# -----------------------------
def heatmap_mse(logits, target):
    return F.mse_loss(torch.sigmoid(logits), target)

def world_smooth_l1(pred_xy, tgt_xy):
    return F.smooth_l1_loss(pred_xy, tgt_xy, beta=0.02)

def world_smooth_l1_normalized(pred_xy, tgt_xy, ws, hs, beta=0.02):
    """
    Normalize world deltas by per-session width/height so this loss is ~O(1).
    pred_xy/tgt_xy: [B,2]; ws/hs: [B] tensors of world width/height.
    """
    diff = (pred_xy - tgt_xy) / torch.stack([ws, hs], dim=1).clamp_min(1e-8)
    # target is 0 after normalization
    return F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta)

def get_session_wh(batch, session_params, device):
    """Convenience: return tensors ws, hs (both [B]) from batch['session'] indices."""
    sess_idx = batch["session"].to(device)
    ws = torch.tensor([session_params[int(s.item())]["w"] for s in sess_idx],
                      device=device, dtype=torch.float32)
    hs = torch.tensor([session_params[int(s.item())]["h"] for s in sess_idx],
                      device=device, dtype=torch.float32)
    return ws, hs

# -----------------------------
# Train / Eval
# -----------------------------
def forward_world(model, batch, device, beta, stride_hw, session_params):
    img = batch["image"].to(device, non_blocking=True)
    bounds = batch["bounds"].to(device, non_blocking=True).to(torch.float32)

    hm_logits = model(img)  # [B,1,Hm,Wm]
    xh, yh, _ = softargmax2d(hm_logits, beta)

    # separate strides for width/height
    (stride_h, stride_w) = stride_hw
    xi, yi = xh*stride_w, yh*stride_h  # image pixels

    B = img.size(0)
    uv_corners = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],
                              device=img.device, dtype=torch.float32).unsqueeze(0).repeat(B,1,1)

    # Do homography in full float32 with autocast disabled
    with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=False):
        uv_pred = []
        for b in range(B):
            H_img2uv = dlt_H(bounds[b], uv_corners[b])               # float32
            uvb = apply_H(torch.stack([xi[b].to(torch.float32),
                                       yi[b].to(torch.float32)], dim=0).view(1,2),
                          H_img2uv).view(2)
            uv_pred.append(uvb)
        uv_pred = torch.stack(uv_pred, dim=0)                        # [B,2] float32

    # UV -> world using per-sample session params
    sess_idx = batch["session"].to(device)
    ws = torch.tensor([session_params[int(s.item())]["w"] for s in sess_idx],
                      device=device, dtype=torch.float32)
    hs = torch.tensor([session_params[int(s.item())]["h"] for s in sess_idx],
                      device=device, dtype=torch.float32)
    x0s= torch.tensor([session_params[int(s.item())]["x0"] for s in sess_idx],
                      device=device, dtype=torch.float32)
    y0s= torch.tensor([session_params[int(s.item())]["y0"] for s in sess_idx],
                      device=device, dtype=torch.float32)
    fus= torch.tensor([session_params[int(s.item())]["flip_u"] for s in sess_idx],
                      device=device, dtype=torch.bool)
    fvs= torch.tensor([session_params[int(s.item())]["flip_v"] for s in sess_idx],
                      device=device, dtype=torch.bool)

    u = uv_pred[:,0].clamp(0,1); v = uv_pred[:,1].clamp(0,1)
    u = torch.where(fus, 1.0 - u, u)
    v = torch.where(fvs, 1.0 - v, v)
    xy_world_pred = torch.stack([x0s + u*ws, y0s + v*hs], dim=1)     # [B,2] float32
    return hm_logits, xy_world_pred, uv_pred

@torch.no_grad()
def evaluate(model, loader, device, beta, session_params,
             hm_wt=1.0, uv_wt=1.0, w_wt=0.0, norm_world_loss=True):
    model.eval()
    total_loss, total_wmae, total_uvmae, n = 0.0, 0.0, 0.0, 0

    for batch in loader:
        img   = batch["image"].to(device, non_blocking=True)
        hm_t  = batch["heatmap_target"].to(device, non_blocking=True)
        xyw_t = batch["xy_world"].to(device, non_blocking=True).to(torch.float32)
        uv_t  = batch["uv_gt"].to(device).to(torch.float32)

        Hm, Wm = hm_t.shape[-2], hm_t.shape[-1]
        H,  W  = img.shape[-2], img.shape[-1]
        stride_hw = (H // Hm, W // Wm)

        hm_logits, xyw_pred, uv_pred = forward_world(
            model, batch, device, beta, stride_hw, session_params
        )

        if hm_logits.shape[-2:] != hm_t.shape[-2:]:
            hm_t = F.interpolate(hm_t, size=hm_logits.shape[-2:], mode="bilinear", align_corners=False)

        # numeric guard
        if (not torch.isfinite(hm_logits).all()) or \
           (not torch.isfinite(hm_t).all()) or \
           (not torch.isfinite(xyw_pred).all()) or \
           (not torch.isfinite(xyw_t).all()) or \
           (not torch.isfinite(uv_pred).all()) or \
           (not torch.isfinite(uv_t).all()):
            print("!! eval skip batch due to non-finite tensors")
            continue

        L_hm = heatmap_mse(hm_logits, hm_t)
        L_uv = F.smooth_l1_loss(uv_pred.clamp(0,1), uv_t, beta=0.01)

        if w_wt > 0.0:
            if norm_world_loss:
                ws, hs = get_session_wh(batch, session_params, device)
                L_w = world_smooth_l1_normalized(xyw_pred, xyw_t, ws, hs, beta=0.02)
            else:
                L_w = world_smooth_l1(xyw_pred, xyw_t)
        else:
            L_w = torch.tensor(0.0, device=device, dtype=img.dtype)

        # weighted loss (for reporting)
        loss = hm_wt * L_hm + uv_wt * L_uv + w_wt * L_w

        # metrics
        w_mae  = (xyw_pred - xyw_t).abs().mean(dim=0).sum() / 2.0
        uv_mae = (uv_pred.clamp(0,1) - uv_t).abs().mean()

        bs = img.size(0)
        total_loss   += loss.item() * bs
        total_wmae   += w_mae.item() * bs
        total_uvmae  += uv_mae.item() * bs
        n += bs

    return total_loss / max(1, n), total_wmae / max(1, n), total_uvmae / max(1, n)


def train_one_epoch(model, loader, opt, scaler, device, beta, session_params,
                    hm_wt=1.0, uv_wt=1.0, w_wt=0.0, norm_world_loss=True):
    model.train()
    total, n = 0.0, 0

    for batch in loader:
        img   = batch["image"].to(device, non_blocking=True)
        hm_t  = batch["heatmap_target"].to(device, non_blocking=True)
        xyw_t = batch["xy_world"].to(device, non_blocking=True).to(torch.float32)
        uv_t  = batch["uv_gt"].to(device).to(torch.float32)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu',
                                enabled=scaler is not None):
            # infer stride (supports variable input sizes)
            Hm, Wm = hm_t.shape[-2], hm_t.shape[-1]
            H,  W  = img.shape[-2], img.shape[-1]
            stride_hw = (H // Hm, W // Wm)

            hm_logits, xyw_pred, uv_pred = forward_world(
                model, batch, device, beta, stride_hw, session_params
            )

            # make sure targets match logits spatial size
            if hm_logits.shape[-2:] != hm_t.shape[-2:]:
                hm_t = F.interpolate(hm_t, size=hm_logits.shape[-2:], mode="bilinear", align_corners=False)

            # numeric guard (avoid NaNs propagating)
            if (not torch.isfinite(hm_logits).all()) or \
               (not torch.isfinite(hm_t).all()) or \
               (not torch.isfinite(xyw_pred).all()) or \
               (not torch.isfinite(xyw_t).all()) or \
               (not torch.isfinite(uv_pred).all()) or \
               (not torch.isfinite(uv_t).all()):
                print("!! skip batch due to non-finite tensors")
                continue

            L_hm = heatmap_mse(hm_logits, hm_t)
            L_uv = F.smooth_l1_loss(uv_pred.clamp(0,1), uv_t, beta=0.01)

            if w_wt > 0.0:
                if norm_world_loss:
                    ws, hs = get_session_wh(batch, session_params, device)
                    L_w = world_smooth_l1_normalized(xyw_pred, xyw_t, ws, hs, beta=0.02)
                else:
                    L_w = world_smooth_l1(xyw_pred, xyw_t)
            else:
                L_w = torch.tensor(0.0, device=device, dtype=img.dtype)

            loss = hm_wt * L_hm + uv_wt * L_uv + w_wt * L_w

        if not torch.isfinite(loss):
            print(f"!! skip batch non-finite loss {loss.item()}"); 
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        bs = img.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(1, n)

# -----------------------------
# Split (file-wise)
# -----------------------------
def split_files_filewise(h5_paths: List[str], val_split: float, seed: int):
    rng = random.Random(seed); files = list(h5_paths); rng.shuffle(files)
    val_count = max(1, int(len(files)*val_split))
    val_files = files[:val_count]
    train_files = files[val_count:] if len(files) > val_count else files[:1]
    return train_files, val_files

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Stylus heatmap: image + corners -> world XY (per-session W,H)")
    ap.add_argument("--h5", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="stylus_ckpts_v2")
    ap.add_argument("--only-touch", action="store_true")
    ap.add_argument("--world-key", type=str, default="/xy")
    ap.add_argument("--hm-wt", type=float, default=1.0)
    ap.add_argument("--uv-wt", type=float, default=1.0)
    ap.add_argument("--w-wt",  type=float, default=0.0)
    ap.add_argument("--norm-world-loss", action="store_true")

    # Global defaults (fallback if not provided per-session)
    ap.add_argument("--world-w", type=float, default=1.0)
    ap.add_argument("--world-h", type=float, default=1.0)
    ap.add_argument("--world-x0", type=float, default=0.0)
    ap.add_argument("--world-y0", type=float, default=0.0)
    ap.add_argument("--flip-u", type=int, default=0, help="1 to flip U axis")
    ap.add_argument("--flip-v", type=int, default=0, help="1 to flip V axis")

    # Optional mapping file (CSV/JSON) for per-session params
    ap.add_argument("--world-sizes-map", type=str, default=None,
                    help="CSV/JSON mapping: file -> w,h,x0,y0,flip_u,flip_v")

    # Heatmap settings
    ap.add_argument("--heatmap-stride", type=int, default=4)
    ap.add_argument("--sigma-hm", type=float, default=1.5)
    ap.add_argument("--heatmap-beta", type=float, default=50.0)
    args = ap.parse_args()

    try:
        print(">> Starting stylus heatmap (world per-session)")
        print("argv:", " ".join(sys.argv))
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(args.out, exist_ok=True)

        # Per-session mapping
        mapping = load_session_map(args.world_sizes_map)
        global_params = {
            "w": args.world_w, "h": args.world_h,
            "x0": args.world_x0, "y0": args.world_y0,
            "flip_u": int(args.flip_u), "flip_v": int(args.flip_v),
        }

        # File-wise split
        train_files, val_files = split_files_filewise(args.h5, args.val_split, args.seed)
        print(f"Train files ({len(train_files)}): {train_files}")
        print(f" Val files ({len(val_files)}): {val_files}")

        train_set = StylusWorldDataset(train_files, world_key=args.world_key,
                                       only_touch=args.only_touch,
                                       heatmap_stride=args.heatmap_stride,
                                       sigma_hm=args.sigma_hm,
                                       global_params=global_params,
                                       mapping=mapping,
                                       color_jitter=True)
        val_set   = StylusWorldDataset(val_files, world_key=args.world_key,
                                       only_touch=args.only_touch,
                                       heatmap_stride=args.heatmap_stride,
                                       sigma_hm=args.sigma_hm,
                                       global_params=global_params,
                                       mapping=mapping,
                                       color_jitter=False)
        print(f"Train frames: {len(train_set)} | Val frames: {len(val_set)}")
        if len(train_set)==0:
            print("!! No valid frames in train_set.")

        tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=False)
        va_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True, drop_last=False)

        # Model / Optim / AMP
        model = HeatmapHead(pretrained=not args.no_pretrained).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda') if (not args.no_amp and device.type=='cuda') else None

        best_wmae = float("inf")
        for epoch in range(1, args.epochs+1):
            tr_loss = train_one_epoch(
                model, tr_loader, opt, scaler, device,
                beta=args.heatmap_beta,
                session_params=train_set.session_params,
                hm_wt=args.hm_wt, uv_wt=args.uv_wt, w_wt=args.w_wt,
                norm_world_loss=args.norm_world_loss
            )

            va_loss, va_wmae, va_uv_mae = evaluate(
                model, va_loader, device,
                beta=args.heatmap_beta,
                session_params=val_set.session_params,
                hm_wt=args.hm_wt, uv_wt=args.uv_wt, w_wt=args.w_wt,
                norm_world_loss=args.norm_world_loss
            )
            print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
                  f"val World-MAE {va_wmae:.2f} | val UV-MAE {va_uv_mae:.4f}")

            if va_wmae < best_wmae - 1e-6:
                best_wmae = va_wmae
                ckpt = Path(args.out) / "best_world_hm.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_world_mae": va_wmae,
                    "args": vars(args)
                }, ckpt)
                print(f"Saved best to {ckpt}")

        last = Path(args.out) / "last_world_hm.pt"
        torch.save({"model_state": model.state_dict(), "epoch": args.epochs, "args": vars(args)}, last)
        print(f"Saved final to {last}")
        train_set.close(); val_set.close()
        print(">> Done")

    except Exception:
        print("!! Fatal error occurred:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
