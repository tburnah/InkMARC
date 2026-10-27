from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def spatial_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, 1, H, W)
    returns prob: (B, 1, H, W) with sum over H*W == 1
    """
    b, c, h, w = logits.shape
    x = logits.view(b, c, h * w)
    x = F.softmax(x, dim=-1)
    return x.view(b, c, h, w)


def dsnt(prob: torch.Tensor) -> torch.Tensor:
    """
    prob: (B, 1, H, W), sums to 1 over H*W
    returns xy in [0,1]: (B, 2)
    """
    b, c, h, w = prob.shape
    device = prob.device
    dtype = prob.dtype

    # grid_x: (W,), grid_y: (H,) in [0,1]
    grid_x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    grid_y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)

    # marginalize
    prob_ = prob[:, 0, :, :]                 # (B, H, W)
    px = prob_.sum(dim=1)                    # (B, W)
    py = prob_.sum(dim=2)                    # (B, H)

    x = (px * grid_x.unsqueeze(0)).sum(dim=1)  # (B,)
    y = (py * grid_y.unsqueeze(0)).sum(dim=1)  # (B,)

    return torch.stack([x, y], dim=1)        # (B, 2)


class MobileNetV3_HeatmapDSNT(nn.Module):
    def __init__(self, hm_h: int = 40, hm_w: int = 60, with_touch: bool = False) -> None:
        super().__init__()
        self.hm_h = hm_h
        self.hm_w = hm_w
        self.with_touch = with_touch

        backbone = torchvision.models.mobilenet_v3_large(weights=None)
        self.features = backbone.features  # keeps spatial maps

        # Light conv head -> 1-channel heatmap logits
        # We up/downsample to (hm_h, hm_w) explicitly for determinism.
        last_ch = 960  # mobilenet_v3_large final feature channels
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

        if self.with_touch:
            in_dim = backbone.classifier[0].in_features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.touch_head = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),  # logits
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.features(x)  # (B, C, Hf, Wf)
        logits = self.heatmap_head(feats)  # (B,1,Hf,Wf)

        logits = F.interpolate(logits, size=(self.hm_h, self.hm_w), mode="bilinear", align_corners=False)
        prob = spatial_softmax(logits)
        xy = dsnt(prob)  # (B,2) in [0,1]

        out: Dict[str, torch.Tensor] = {
            "heatmap_logits": logits,
            "heatmap_prob": prob,
            "xy": xy,
        }

        if self.with_touch:
            pooled = self.avgpool(feats)
            pooled = torch.flatten(pooled, 1)
            out["touch_logit"] = self.touch_head(pooled).squeeze(1)  # (B,)

        return out
