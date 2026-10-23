from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def spatial_softmax(logits: torch.Tensor) -> torch.Tensor:
    b, c, h, w = logits.shape
    x = logits.view(b, c, h * w)
    x = F.softmax(x, dim=-1)
    return x.view(b, c, h, w)


def dsnt(prob: torch.Tensor) -> torch.Tensor:
    b, c, h, w = prob.shape
    device = prob.device
    dtype = prob.dtype

    grid_x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    grid_y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)

    prob_ = prob[:, 0, :, :]          # (B,H,W)
    px = prob_.sum(dim=1)             # (B,W)
    py = prob_.sum(dim=2)             # (B,H)

    x = (px * grid_x.unsqueeze(0)).sum(dim=1)
    y = (py * grid_y.unsqueeze(0)).sum(dim=1)
    return torch.stack([x, y], dim=1)


class CausalTSM(nn.Module):
    """
    Causal Temporal Shift:
      - shift a fraction of channels from t-1 to t (past -> present)
      - no future leakage
    Input:  x (B,T,C,H,W)
    Output: same shape
    """
    def __init__(self, fold_div: int = 8) -> None:
        super().__init__()
        self.fold_div = max(1, int(fold_div))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        fold = c // self.fold_div
        if fold == 0 or t == 1:
            return x

        out = x.clone()

        # shift "fold" channels from past: out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, 1:, :fold, :, :] = x[:, :-1, :fold, :, :]
        # keep t=0 unchanged for shifted channels (already clone)
        # remaining channels unchanged (causal)
        return out


class MobileNetV3_TSM_HeatmapDSNT(nn.Module):
    """
    Phase 4 Tier A:
      - MobileNetV3 features per frame
      - Causal TSM on feature tensor
      - heatmap head + DSNT, prediction for last frame only
    """
    def __init__(self, hm_h: int = 40, hm_w: int = 60, tsm_fold_div: int = 8) -> None:
        super().__init__()
        self.hm_h = hm_h
        self.hm_w = hm_w

        backbone = torchvision.models.mobilenet_v3_large(weights=None)
        self.features = backbone.features

        self.tsm = CausalTSM(fold_div=tsm_fold_div)

        last_ch = 960
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x_clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_clip: (B,T,C,H,W)
        """
        b, t, c, h, w = x_clip.shape
        x = x_clip.view(b * t, c, h, w)

        feats = self.features(x)  # (B*T, Cf, Hf, Wf)
        cf, hf, wf = feats.shape[1], feats.shape[2], feats.shape[3]
        feats = feats.view(b, t, cf, hf, wf)

        feats = self.tsm(feats)  # causal temporal mixing

        # Use last timestep only (causal)
        f_last = feats[:, -1, :, :, :]  # (B,Cf,Hf,Wf)

        logits = self.heatmap_head(f_last)
        logits = F.interpolate(logits, size=(self.hm_h, self.hm_w), mode="bilinear", align_corners=False)

        prob = spatial_softmax(logits)
        xy = dsnt(prob)

        return {
            "heatmap_logits": logits,
            "heatmap_prob": prob,
            "xy": xy,
        }
