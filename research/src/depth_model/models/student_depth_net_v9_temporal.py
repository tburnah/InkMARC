import torch
import torch.nn as nn
import torch.nn.functional as F

from models.student_depth_net_v6 import DepthwiseSeparableConv
from models.student_depth_net_v8 import StudentDepthNetV8


class TemporalBottleneckFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels, channels),
        )
        self.state_refine = DepthwiseSeparableConv(channels, channels)

    def forward(self, current: torch.Tensor, prev_state: torch.Tensor):
        merged = torch.cat([current, prev_state], dim=1)
        gate = self.gate(merged)
        candidate = self.candidate(merged)
        fused = gate * current + (1.0 - gate) * candidate
        next_state = self.state_refine(fused)
        return fused, next_state


class StudentDepthNetV9TemporalLite(nn.Module):
    def __init__(self):
        super().__init__()

        base = StudentDepthNetV8()
        self.enc1 = base.enc1
        self.enc2 = base.enc2
        self.enc3 = base.enc3
        self.enc4 = base.enc4
        self.dec3 = base.dec3
        self.dec2 = base.dec2
        self.dec1 = base.dec1
        self.head = base.head

        self.state_channels = self._infer_state_channels()
        self.temporal_fusion = TemporalBottleneckFusion(self.state_channels)

    def _infer_state_channels(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
        return e4.shape[1]

    def init_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        device=None,
        dtype=None,
    ):
        with torch.no_grad():
            x = torch.zeros(
                batch_size,
                3,
                height,
                width,
                device=device,
                dtype=dtype or torch.float32,
            )
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
        return torch.zeros_like(e4)

    def _decode(self, e1, e2, e3, bottleneck, input_size):
        d3 = self.dec3(bottleneck, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        return self.head(out)

    def forward_step(self, x: torch.Tensor, prev_state: torch.Tensor | None = None):
        input_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        if prev_state is None:
            prev_state = torch.zeros_like(e4)

        fused, next_state = self.temporal_fusion(e4, prev_state)
        logits = self._decode(e1, e2, e3, fused, input_size)
        return logits, next_state

    def forward(
        self,
        x: torch.Tensor,
        prev_state: torch.Tensor | None = None,
        *,
        return_state: bool = False,
    ):
        logits, next_state = self.forward_step(x, prev_state)
        if return_state:
            return logits, next_state
        return logits
