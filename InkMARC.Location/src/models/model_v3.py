# models/model_v3.py
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torchvision


class MobileNetV3_RawPlusCorners_XY(nn.Module):
    """
    Phase 1a (location-only):
      Inputs:
        - x_img: (B,3,H,W) raw RGB resized to out_w/out_h, normalized to [0,1]
        - x_corners: (B,8) corners (recommended: normalised, unclamped)
      Output:
        - xy: (B,2) in [0,1]
    """

    def __init__(self, corners_dim: int = 8, corner_embed: int = 64) -> None:
        super().__init__()

        backbone = torchvision.models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        img_feat_dim = backbone.classifier[0].in_features

        self.corner_mlp = nn.Sequential(
            nn.Linear(corners_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, corner_embed),
            nn.ReLU(inplace=True),
        )

        fused_dim = img_feat_dim + corner_embed

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.xy_head = nn.Sequential(
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, x_img: torch.Tensor, x_corners: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.features(x_img)
        feats = self.avgpool(feats)
        feats = torch.flatten(feats, 1)

        c = self.corner_mlp(x_corners)

        fused = torch.cat([feats, c], dim=1)
        fused = self.fuse(fused)

        xy = self.xy_head(fused)
        return {"xy": xy}
