from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
        )


class FPNNeck(nn.Module):
    def __init__(self, in_channels: dict[str, int], out_channels: int = 96) -> None:
        super().__init__()
        self.lateral = nn.ModuleDict(
            {name: nn.Conv2d(ch, out_channels, kernel_size=1) for name, ch in in_channels.items()}
        )
        self.smooth = nn.ModuleDict({name: ConvBNAct(out_channels, out_channels) for name in in_channels})

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p16 = self.lateral["s16"](feats["s16"])
        p8 = self.lateral["s8"](feats["s8"]) + F.interpolate(p16, size=feats["s8"].shape[-2:], mode="nearest")
        p4 = self.lateral["s4"](feats["s4"]) + F.interpolate(p8, size=feats["s4"].shape[-2:], mode="nearest")
        return {
            "p4": self.smooth["s4"](p4),
            "p8": self.smooth["s8"](p8),
            "p16": self.smooth["s16"](p16),
        }


class DepthHead(nn.Module):
    def __init__(self, channels: int, out_size: tuple[int, int]) -> None:
        super().__init__()
        self.out_size = out_size
        self.feature_channels = channels // 2
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = ConvBNAct(channels, self.feature_channels)
        self.pred = nn.Conv2d(self.feature_channels, 1, kernel_size=1)

    def forward(self, p4: torch.Tensor, input_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(p4)
        depth_features = self.conv2(x)
        depth_low = self.pred(depth_features)
        if depth_low.shape[-2:] != self.out_size:
            depth_low = F.interpolate(depth_low, size=self.out_size, mode="bilinear", align_corners=False)
            depth_features = F.interpolate(depth_features, size=self.out_size, mode="bilinear", align_corners=False)
        depth_full = F.interpolate(depth_low, size=input_size, mode="bilinear", align_corners=False)
        return depth_full, depth_low, depth_features


class StateHead(nn.Module):
    def __init__(self, in_channels: int, mode: str = "binary", num_states: int = 3) -> None:
        super().__init__()
        out_dim = 1 if mode == "binary" else num_states
        hidden = max(64, in_channels // 2)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, out_dim),
        )
        self.mode = mode

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        logits = self.net(features)
        if self.mode == "binary":
            return None, logits.squeeze(1)
        return logits, None


class LocationHead(nn.Module):
    def __init__(self, in_channels: int, heatmap_size: tuple[int, int]) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.net = nn.Sequential(
            ConvBNAct(in_channels, in_channels),
            ConvBNAct(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heatmap = self.net(features)
        if heatmap.shape[-2:] != self.heatmap_size:
            heatmap = F.interpolate(heatmap, size=self.heatmap_size, mode="bilinear", align_corners=False)
        xy = soft_argmax_2d(heatmap)
        return heatmap, xy


def soft_argmax_2d(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    b, _, h, w = logits.shape
    flat = (logits / max(temperature, 1e-6)).reshape(b, -1)
    prob = torch.softmax(flat, dim=1).reshape(b, h, w)
    xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    x = (prob * xs.view(1, 1, w)).sum(dim=(1, 2))
    y = (prob * ys.view(1, h, 1)).sum(dim=(1, 2))
    return torch.stack([x, y], dim=1)
