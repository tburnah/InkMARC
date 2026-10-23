import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _fuse_sequential_conv_bn(seq: nn.Sequential, conv_idx: int, bn_idx: int) -> None:
    conv = seq[conv_idx]
    bn = seq[bn_idx]
    if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
        seq[conv_idx] = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        seq[bn_idx] = nn.Identity()


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


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                (kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio,
        start_dw_kernel_size,
        middle_dw_kernel_size,
        stride,
        middle_dw_downsample=True,
    ):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if start_dw_kernel_size:
            self.start_dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                start_dw_kernel_size,
                stride if not middle_dw_downsample else 1,
                (start_dw_kernel_size - 1) // 2,
                groups=in_channels,
                bias=False,
            )
            self.start_dw_norm = nn.BatchNorm2d(in_channels)

        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(in_channels, expand_channels, 1, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_channels)
        self.expand_act = nn.ReLU(inplace=True)

        if middle_dw_kernel_size:
            self.middle_dw_conv = nn.Conv2d(
                expand_channels,
                expand_channels,
                middle_dw_kernel_size,
                stride if middle_dw_downsample else 1,
                (middle_dw_kernel_size - 1) // 2,
                groups=expand_channels,
                bias=False,
            )
            self.middle_dw_norm = nn.BatchNorm2d(expand_channels)
            self.middle_dw_act = nn.ReLU(inplace=True)

        self.proj_conv = nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_channels)
        self.identity = stride == 1 and in_channels == out_channels

    def forward(self, x):
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)
        return x + shortcut if self.identity else x


def fuse_batch_norms_inplace(model: nn.Module) -> nn.Module:
    model.eval()

    for module in model.modules():
        if isinstance(module, DepthwiseSeparableConv):
            _fuse_sequential_conv_bn(module.block, 0, 1)
            _fuse_sequential_conv_bn(module.block, 3, 4)
        elif isinstance(module, ConvBN):
            _fuse_sequential_conv_bn(module.block, 0, 1)
        elif isinstance(module, UniversalInvertedBottleneck):
            if (
                hasattr(module, "start_dw_conv")
                and isinstance(module.start_dw_conv, nn.Conv2d)
                and isinstance(module.start_dw_norm, nn.BatchNorm2d)
            ):
                module.start_dw_conv = nn.utils.fusion.fuse_conv_bn_eval(
                    module.start_dw_conv,
                    module.start_dw_norm,
                )
                module.start_dw_norm = nn.Identity()

            if isinstance(module.expand_conv, nn.Conv2d) and isinstance(module.expand_norm, nn.BatchNorm2d):
                module.expand_conv = nn.utils.fusion.fuse_conv_bn_eval(
                    module.expand_conv,
                    module.expand_norm,
                )
                module.expand_norm = nn.Identity()

            if (
                hasattr(module, "middle_dw_conv")
                and isinstance(module.middle_dw_conv, nn.Conv2d)
                and isinstance(module.middle_dw_norm, nn.BatchNorm2d)
            ):
                module.middle_dw_conv = nn.utils.fusion.fuse_conv_bn_eval(
                    module.middle_dw_conv,
                    module.middle_dw_norm,
                )
                module.middle_dw_norm = nn.Identity()

            if isinstance(module.proj_conv, nn.Conv2d) and isinstance(module.proj_norm, nn.BatchNorm2d):
                module.proj_conv = nn.utils.fusion.fuse_conv_bn_eval(
                    module.proj_conv,
                    module.proj_norm,
                )
                module.proj_norm = nn.Identity()

    return model


def optimize_student_depth_net_v6_for_inference(
    model: nn.Module,
    *,
    use_channels_last: bool = False,
) -> nn.Module:
    fuse_batch_norms_inplace(model)
    if use_channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    return model


class MobileNetV4ConvLargeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Block spec adapted from d-li14/mobilenetv4.pytorch MobileNetV4-Conv-Large.
        block_specs = [
            ("conv_bn", 3, 2, 24),
            ("conv_bn", 3, 2, 96),
            ("conv_bn", 1, 1, 48),
            ("uib", 3, 5, 2, 96, 4.0),
            ("uib", 3, 3, 1, 96, 4.0),
            ("uib", 3, 5, 2, 192, 4.0),
            ("uib", 3, 3, 1, 192, 4.0),
            ("uib", 3, 3, 1, 192, 4.0),
            ("uib", 3, 3, 1, 192, 4.0),
            ("uib", 3, 5, 1, 192, 4.0),
            ("uib", 5, 3, 1, 192, 4.0),
            ("uib", 5, 3, 1, 192, 4.0),
            ("uib", 5, 3, 1, 192, 4.0),
            ("uib", 5, 3, 1, 192, 4.0),
            ("uib", 5, 3, 1, 192, 4.0),
            ("uib", 3, 0, 1, 192, 4.0),
            ("uib", 5, 5, 2, 512, 4.0),
            ("uib", 5, 5, 1, 512, 4.0),
            ("uib", 5, 5, 1, 512, 4.0),
            ("uib", 5, 5, 1, 512, 4.0),
            ("uib", 5, 0, 1, 512, 4.0),
            ("uib", 5, 3, 1, 512, 4.0),
            ("uib", 5, 0, 1, 512, 4.0),
            ("uib", 5, 0, 1, 512, 4.0),
            ("uib", 5, 3, 1, 512, 4.0),
            ("uib", 5, 5, 1, 512, 4.0),
            ("uib", 5, 0, 1, 512, 4.0),
            ("uib", 5, 0, 1, 512, 4.0),
            ("uib", 5, 0, 1, 512, 4.0),
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
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class StudentDepthNetV6(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = MobileNetV4ConvLargeBackbone()
        features = backbone.features

        # Keep the lightweight multi-scale decoder pattern from the best V2 line
        # while swapping in a stronger MobileNetV4-Large encoder. Tap the
        # natural /4, /8, /16, and /32 stages and skip the final 960-channel
        # classifier expansion to keep the depth head simpler and export-friendly.
        self.enc1 = nn.Sequential(*features[:3])
        self.enc2 = nn.Sequential(*features[3:5])
        self.enc3 = nn.Sequential(*features[5:16])
        self.enc4 = nn.Sequential(*features[16:29])

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
