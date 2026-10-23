import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


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


class StudentDepthNetV5(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = mobilenet_v3_small(weights=None)
        features = backbone.features

        # Keep the same encoder/decoder layout as prior student variants so
        # V5 can warm-start from the original best checkpoint.
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

        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        return self.head(out)
