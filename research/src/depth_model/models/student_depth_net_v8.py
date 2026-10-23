import torch
import torch.nn as nn
import torch.nn.functional as F

from models.student_depth_net_v6 import (
    ConvBN,
    DecoderBlock,
    DepthwiseSeparableConv,
    UniversalInvertedBottleneck,
)


class MobileNetV4ConvMediumBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Block spec adapted from d-li14/mobilenetv4.pytorch MobileNetV4-Conv-Medium.
        block_specs = [
            ("conv_bn", 3, 2, 32),
            ("conv_bn", 3, 2, 128),
            ("conv_bn", 1, 1, 48),
            ("uib", 3, 5, 2, 80, 4.0),
            ("uib", 3, 3, 1, 80, 2.0),
            ("uib", 3, 5, 2, 160, 6.0),
            ("uib", 3, 3, 1, 160, 4.0),
            ("uib", 3, 3, 1, 160, 4.0),
            ("uib", 3, 5, 1, 160, 4.0),
            ("uib", 3, 3, 1, 160, 4.0),
            ("uib", 3, 0, 1, 160, 4.0),
            ("uib", 0, 0, 1, 160, 2.0),
            ("uib", 3, 0, 1, 160, 4.0),
            ("uib", 5, 5, 2, 256, 6.0),
            ("uib", 5, 5, 1, 256, 4.0),
            ("uib", 3, 5, 1, 256, 4.0),
            ("uib", 3, 5, 1, 256, 4.0),
            ("uib", 0, 0, 1, 256, 4.0),
            ("uib", 3, 0, 1, 256, 4.0),
            ("uib", 3, 5, 1, 256, 2.0),
            ("uib", 5, 5, 1, 256, 4.0),
            ("uib", 0, 0, 1, 256, 4.0),
            ("uib", 0, 0, 1, 256, 4.0),
            ("uib", 5, 0, 1, 256, 2.0),
            ("conv_bn", 1, 1, 960),
        ]

        layers = []
        channels = 3
        for block_type, *block_cfg in block_specs:
            if block_type == "conv_bn":
                kernel_size, stride, out_channels = block_cfg
                layer = ConvBN(channels, out_channels, kernel_size, stride)
            elif block_type == "uib":
                start_k, middle_k, stride, out_channels, expand_ratio = block_cfg
                layer = UniversalInvertedBottleneck(
                    channels,
                    out_channels,
                    expand_ratio,
                    start_k,
                    middle_k,
                    stride,
                )
            else:
                raise NotImplementedError(f"Unsupported block type: {block_type}")
            layers.append(layer)
            channels = out_channels

        self.features = nn.Sequential(*layers)


class StudentDepthNetV8(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = MobileNetV4ConvMediumBackbone()
        features = backbone.features

        # Match the V6/V7 decoder layout while swapping in the Medium encoder.
        # Tap /4, /8, /16, and /32 stages and skip the final 960-channel
        # classifier expansion to keep the head lightweight and export-friendly.
        self.enc1 = nn.Sequential(*features[:3])
        self.enc2 = nn.Sequential(*features[3:5])
        self.enc3 = nn.Sequential(*features[5:13])
        self.enc4 = nn.Sequential(*features[13:24])

        ch1, ch2, ch3, ch4 = self._infer_channels()

        self.dec3 = DecoderBlock(ch4, ch3, 80)
        self.dec2 = DecoderBlock(80, ch2, 48)
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
