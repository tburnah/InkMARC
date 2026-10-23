from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


SUPPORTED_BACKBONES = ("mobilenet_v3_large", "mobilenet_v3_small")


def spatial_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert heatmap logits to a spatial probability distribution.

    Args:
        logits: Tensor shaped (B, 1, H, W).

    Returns:
        Tensor shaped (B, 1, H, W), where each sample sums to 1 over H*W.
    """
    b, c, h, w = logits.shape
    prob = F.softmax(logits.reshape(b, c, h * w), dim=-1)
    return prob.reshape(b, c, h, w)


def dsnt(prob: torch.Tensor) -> torch.Tensor:
    """
    Differentiable spatial-to-numerical transform.

    Args:
        prob: Tensor shaped (B, 1, H, W), normalized over H*W.

    Returns:
        Normalized xy coordinates shaped (B, 2), with x/y in [0, 1].
    """
    _, _, h, w = prob.shape
    device = prob.device
    dtype = prob.dtype

    grid_x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    grid_y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)

    prob_2d = prob[:, 0, :, :]
    px = prob_2d.sum(dim=1)
    py = prob_2d.sum(dim=2)

    x = (px * grid_x.unsqueeze(0)).sum(dim=1)
    y = (py * grid_y.unsqueeze(0)).sum(dim=1)
    return torch.stack([x, y], dim=1)


def build_mobilenet_features(backbone_name: str, pretrained: bool = False) -> Tuple[nn.Module, int]:
    """
    Build a MobileNetV3 feature extractor and return its final channel count.

    By default this matches the existing phase2 location model by using
    weights=None. Pretrained weights can be enabled for small-data experiments.
    """
    if backbone_name == "mobilenet_v3_large":
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.mobilenet_v3_large(weights=weights)
        return backbone.features, 960
    if backbone_name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        return backbone.features, 576
    raise ValueError(f"Unsupported backbone '{backbone_name}'. Choices: {SUPPORTED_BACKBONES}")


class CombinedTouchLocationModel(nn.Module):
    """
    Shared-backbone mobile model with a three-state classifier and heatmap/DSNT location head.
    """

    def __init__(
        self,
        *,
        backbone: str = "mobilenet_v3_large",
        hm_h: int = 40,
        hm_w: int = 60,
        state_hidden: int = 256,
        state_dropout: float = 0.1,
        pretrained_backbone: bool = False,
        imagenet_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.hm_h = int(hm_h)
        self.hm_w = int(hm_w)

        self.imagenet_normalize = bool(imagenet_normalize)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.features, backbone_channels = build_mobilenet_features(backbone, pretrained=bool(pretrained_backbone))
        self.backbone_channels = int(backbone_channels)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.backbone_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

        self.state_pool = nn.AdaptiveAvgPool2d(1)
        self.state_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.backbone_channels, int(state_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(state_dropout)),
            nn.Linear(int(state_hidden), 3),
        )
        self.touch_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.backbone_channels, int(state_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(state_dropout)),
            nn.Linear(int(state_hidden), 1),
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.imagenet_normalize:
            image = (image - self.imagenet_mean.to(dtype=image.dtype)) / self.imagenet_std.to(dtype=image.dtype)
        feats = self.features(image)

        heatmap_logits = self.heatmap_head(feats)
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=(self.hm_h, self.hm_w),
            mode="bilinear",
            align_corners=False,
        )
        heatmap_prob = spatial_softmax(heatmap_logits)
        xy = dsnt(heatmap_prob)

        pooled = self.state_pool(feats)
        state_logits = self.state_head(pooled)
        touch_logits = self.touch_head(pooled).squeeze(1)

        return {
            "state_logits": state_logits,
            "touch_logits": touch_logits,
            "xy": xy,
            "heatmap_logits": heatmap_logits,
            "heatmap_prob": heatmap_prob,
        }
