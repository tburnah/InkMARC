from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

from .heads import ConvBNAct


class Fixed504x322Decoder(nn.Module):
    def __init__(self, in_channels: dict[str, int], fpn_channels: int, decoder_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleDict(
            {name: nn.Conv2d(ch, fpn_channels, kernel_size=1) for name, ch in in_channels.items()}
        )
        self.up16_to8 = nn.ConvTranspose2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)
        self.up8_to4 = nn.ConvTranspose2d(
            fpn_channels,
            fpn_channels,
            kernel_size=(2, 3),
            stride=2,
            padding=(0, 1),
        )
        self.smooth8 = ConvBNAct(fpn_channels, fpn_channels)
        self.smooth4 = ConvBNAct(fpn_channels, fpn_channels)
        self.decode = nn.Sequential(
            ConvBNAct(fpn_channels, decoder_channels),
            ConvBNAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.up4_to2 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 3), stride=2, padding=(0, 1))
        self.up2_to1 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        p16 = self.lateral["s16"](feats["s16"])
        p8 = self.lateral["s8"](feats["s8"]) + self.up16_to8(p16)
        p8 = self.smooth8(p8)
        p4 = self.lateral["s4"](feats["s4"]) + self.up8_to4(p8)
        p4 = self.smooth4(p4)
        depth = self.decode(p4)
        depth = self.up4_to2(depth)
        return self.up2_to1(depth)


class DepthProxyMobileNet(nn.Module):
    """QNN-friendly CNN proxy trained to mimic DA3 raw depth."""

    def __init__(
        self,
        input_size: tuple[int, int] = (504, 322),
        backbone: str = "mobilenet_v2",
        width_mult: float = 1.0,
        pretrained: bool = True,
        fpn_channels: int = 64,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        if backbone == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v2(weights=weights, width_mult=width_mult).features
        elif backbone == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights, width_mult=width_mult).features
        elif backbone == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights, width_mult=width_mult).features
        else:
            raise ValueError("DepthProxyMobileNet supports mobilenet_v2, mobilenet_v3_small, or mobilenet_v3_large")
        self.input_size = tuple(input_size)
        self.feature_indices = self._infer_feature_indices(self.input_size)
        feat_channels = self._infer_feature_channels(self.input_size)
        if self.input_size != (504, 322):
            raise ValueError("The fixed QNN-friendly decoder currently supports proxy_input_size=(504, 322)")
        self.decoder = Fixed504x322Decoder(feat_channels, int(fpn_channels), int(decoder_channels))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    @classmethod
    def from_config(cls, cfg: Any) -> "DepthProxyMobileNet":
        m = cfg.model
        return cls(
            input_size=tuple(getattr(m, "proxy_input_size", cfg.data.input_size)),
            backbone=str(getattr(m, "backbone", "mobilenet_v2")),
            width_mult=float(getattr(m, "width_mult", 1.0)),
            pretrained=bool(getattr(m, "pretrained", True)),
            fpn_channels=int(getattr(m, "fpn_channels", 64)),
            decoder_channels=int(getattr(m, "decoder_channels", 64)),
        )

    def _infer_feature_indices(self, input_size: tuple[int, int]) -> dict[str, int]:
        candidates: dict[int, int] = {}
        with torch.no_grad():
            x = torch.zeros(1, 3, *input_size)
            h0, w0 = x.shape[-2:]
            for idx, block in enumerate(self.backbone):
                x = block(x)
                h, w = x.shape[-2:]
                stride = max(1, round(max(h0 / max(1, h), w0 / max(1, w))))
                candidates[stride] = idx

        def nearest(target: int) -> int:
            stride = min(candidates.keys(), key=lambda s: abs(s - target))
            return candidates[stride]

        return {"s4": nearest(4), "s8": nearest(8), "s16": nearest(16)}

    def _infer_feature_channels(self, input_size: tuple[int, int]) -> dict[str, int]:
        with torch.no_grad():
            feats = self._extract_features(torch.zeros(1, 3, *input_size), normalize=False)
        return {key: value.shape[1] for key, value in feats.items()}

    def _extract_features(self, x: torch.Tensor, normalize: bool = True) -> dict[str, torch.Tensor]:
        if normalize:
            x = (x - self.image_mean) / self.image_std
        selected: dict[str, torch.Tensor] = {}
        index_to_name = {idx: name for name, idx in self.feature_indices.items()}
        for idx, block in enumerate(self.backbone):
            x = block(x)
            name = index_to_name.get(idx)
            if name is not None:
                selected[name] = x
        return selected

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self._extract_features(image)
        return self.decoder(feats)
