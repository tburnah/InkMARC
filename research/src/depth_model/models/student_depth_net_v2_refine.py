import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = DepthwiseSeparableConv(in_ch + skip_ch, out_ch)
        self.refine = DepthwiseSeparableConv(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.refine(x)


class StudentDepthNetV2Refine(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = mobilenet_v3_large(weights=None)
        features = backbone.features

        # Keep the V2 encoder/decoder layout intact so existing checkpoints can
        # warm-start the shared backbone and coarse decoder weights.
        self.enc1 = nn.Sequential(*features[:2])
        self.enc2 = nn.Sequential(*features[2:4])
        self.enc3 = nn.Sequential(*features[4:9])
        self.enc4 = nn.Sequential(*features[9:12])

        ch1, ch2, ch3, ch4 = self._infer_channels()

        self.dec3 = DecoderBlock(ch4, ch3, 96)
        self.dec2 = DecoderBlock(96, ch2, 48)
        self.dec1 = DecoderBlock(48, ch1, 32)

        self.head = nn.Sequential(
            DepthwiseSeparableConv(32, 24),
            nn.Conv2d(24, 1, kernel_size=1),
        )

        # A lightweight high-resolution residual branch helps preserve thin
        # structures without changing inference outputs beyond one depth map.
        self.refine_fuse = DepthwiseSeparableConv(32 + ch1, 32)
        self.refine_head = nn.Sequential(
            DepthwiseSeparableConv(32, 24),
            nn.Conv2d(24, 1, kernel_size=1),
        )
        self._initialize_refine_branch()

    def _initialize_refine_branch(self):
        # Start from the original V2 prediction exactly; the residual branch
        # learns corrections during fine-tuning instead of perturbing warm-start
        # outputs on the first step.
        final_conv = self.refine_head[-1]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def _infer_channels(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
        return e1.shape[1], e2.shape[1], e3.shape[1], e4.shape[1]

    def forward(self, x):
        input_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        coarse_features = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        coarse_logits = self.head(coarse_features)

        high_res_skip = F.interpolate(e1, size=input_size, mode="bilinear", align_corners=False)
        refine_features = torch.cat([coarse_features, high_res_skip], dim=1)
        refine_features = self.refine_fuse(refine_features)
        refine_logits = self.refine_head(refine_features)

        return coarse_logits + refine_logits
