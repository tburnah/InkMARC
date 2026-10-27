# Phase 1 - models/model_v2.py
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torchvision


class MobileNetV3LocationOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        in_dim = backbone.classifier[0].in_features

        self.xy_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Sigmoid(),  # keep only if labels are in [0,1]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.features(x)
        feats = self.avgpool(feats)
        feats = torch.flatten(feats, 1)
        xy = self.xy_head(feats)
        return {"xy": xy}


# Keep your alias pattern if other code expects "ResNet18..." naming.
ResNet18LocationOnly = MobileNetV3LocationOnly
