from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .dsnt import dsnt_xy, heatmap_peak_confidence, spatial_softmax
from .normalization import ImageNetNormalize


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
        )


class InvertedResidualLite(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int) -> None:
        super().__init__()
        hidden = in_ch * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.Hardswish(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_residual:
            y = y + x
        return y


class TinyMobileNetV3Backbone(nn.Module):
    """MobileNetV3-style depthwise backbone sized for CPU smoke tests."""

    def __init__(self, width_mult: float = 0.5) -> None:
        super().__init__()

        def c(ch: int) -> int:
            return max(8, int(ch * width_mult + 7) // 8 * 8)

        channels = [c(v) for v in [16, 24, 40, 48, 96]]
        self.out_channels = channels[-1]
        self.features = nn.Sequential(
            ConvBNAct(3, channels[0], stride=2),
            InvertedResidualLite(channels[0], channels[1], stride=2, expand=2),
            InvertedResidualLite(channels[1], channels[1], stride=1, expand=2),
            InvertedResidualLite(channels[1], channels[2], stride=2, expand=3),
            InvertedResidualLite(channels[2], channels[2], stride=1, expand=3),
            InvertedResidualLite(channels[2], channels[3], stride=2, expand=3),
            InvertedResidualLite(channels[3], channels[4], stride=2, expand=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _try_torchvision_mobilenet(pretrained: bool, width_mult: float) -> Optional[nn.Module]:
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except Exception:
        return None
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights, width_mult=width_mult)
    features = model.features
    out_channels = None
    for module in reversed(features):
        if hasattr(module, "out_channels"):
            out_channels = int(module.out_channels)
            break
    if out_channels is None:
        out_channels = 576
    wrapper = nn.Sequential(features)
    wrapper.out_channels = out_channels  # type: ignore[attr-defined]
    return wrapper


class LocTouchNetLite(nn.Module):
    """Fast interaction-location and touch-state model for canonical RGB surfaces.

    ``touch_logit`` is the conditional touch-vs-hover logit given a valid
    interaction. The final unconditional touch probability is
    ``state_probs[:, 2]``.
    """

    def __init__(
        self,
        input_height: int = 320,
        input_width: int = 480,
        heatmap_height: int = 40,
        heatmap_width: int = 60,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = False,
        width_mult: float = 0.5,
        hidden_dim: int = 128,
        use_torchvision: bool = False,
        temporal_enabled: bool = False,
        num_motion_maps: int = 0,
        motion_hidden_dim: int = 0,
        temporal_input_mode: str = "motion_map",
        motion_type: str = "abs_gray_diff",
        motion_channels: int = 1,
    ) -> None:
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width
        self.temporal_enabled = temporal_enabled
        self.temporal_input_mode = temporal_input_mode
        self.motion_type = motion_type
        self.num_motion_maps = num_motion_maps
        self.normalize = ImageNetNormalize()

        tv_backbone = None
        if backbone == "mobilenet_v3_small" and use_torchvision:
            tv_backbone = _try_torchvision_mobilenet(pretrained=pretrained, width_mult=width_mult)
        if tv_backbone is not None:
            self.backbone = tv_backbone
            feat_ch = int(tv_backbone.out_channels)  # type: ignore[attr-defined]
        else:
            self.backbone = TinyMobileNetV3Backbone(width_mult=width_mult)
            feat_ch = self.backbone.out_channels

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, groups=feat_ch, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(feat_ch, 1, 1),
        )
        token_dim = feat_ch * 2 + 3
        self.motion_hidden_dim = max(0, int(motion_hidden_dim or feat_ch // 2))
        self.motion_in_channels = max(1, int(motion_channels or num_motion_maps or 1)) if temporal_enabled else 0
        if temporal_enabled:
            motion_dim = max(8, self.motion_hidden_dim)
            self.motion_encoder = nn.Sequential(
                ConvBNAct(self.motion_in_channels, motion_dim, stride=2),
                InvertedResidualLite(motion_dim, motion_dim, stride=2, expand=2),
                InvertedResidualLite(motion_dim, motion_dim, stride=2, expand=2),
            )
            self.motion_out_dim = motion_dim
        else:
            self.motion_encoder = None
            self.motion_out_dim = 0
        self.valid_head = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.Hardswish(), nn.Linear(hidden_dim, 1))
        self.touch_head = nn.Sequential(
            nn.Linear(token_dim + self.motion_out_dim, hidden_dim),
            nn.Hardswish(),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_config(cls, cfg: Dict) -> "LocTouchNetLite":
        model_cfg = cfg.get("model", cfg)
        temporal_cfg = cfg.get("temporal", {})
        return cls(
            input_height=int(model_cfg.get("input_height", cfg.get("input_height", 320))),
            input_width=int(model_cfg.get("input_width", cfg.get("input_width", 480))),
            heatmap_height=int(model_cfg.get("heatmap_height", cfg.get("heatmap_height", 40))),
            heatmap_width=int(model_cfg.get("heatmap_width", cfg.get("heatmap_width", 60))),
            backbone=str(model_cfg.get("backbone", cfg.get("backbone", "mobilenet_v3_small"))),
            pretrained=bool(model_cfg.get("pretrained", cfg.get("pretrained", False))),
            width_mult=float(model_cfg.get("width_mult", cfg.get("width_mult", 0.5))),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            use_torchvision=bool(model_cfg.get("use_torchvision", False)),
            temporal_enabled=bool(temporal_cfg.get("enabled", False)),
            num_motion_maps=int(temporal_cfg.get("num_motion_maps", temporal_cfg.get("motion_channels", 0))),
            motion_hidden_dim=int(temporal_cfg.get("motion_hidden_dim", 0)),
            temporal_input_mode=str(temporal_cfg.get("input_mode", "motion_map")),
            motion_type=str(temporal_cfg.get("motion_type", "abs_gray_diff")),
            motion_channels=int(temporal_cfg.get("motion_channels", temporal_cfg.get("num_motion_maps", 1))),
        )

    def _motion_from_prev(self, image: torch.Tensor, image_t_minus_1: torch.Tensor) -> torch.Tensor:
        if self.motion_type == "abs_gray_diff":
            return torch.mean(torch.abs(image - image_t_minus_1), dim=1, keepdim=True)
        if self.motion_type == "signed_gray_diff":
            return torch.mean(image - image_t_minus_1, dim=1, keepdim=True)
        if self.motion_type == "rgb_diff":
            return image - image_t_minus_1
        raise ValueError(f"Unknown motion_type: {self.motion_type}")

    def _motion_features(
        self,
        image: torch.Tensor,
        feat: torch.Tensor,
        image_t_minus_1: Optional[torch.Tensor],
        motion_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.temporal_enabled or self.motion_encoder is None:
            return feat.new_zeros((feat.shape[0], 0, feat.shape[2], feat.shape[3]))

        if self.temporal_input_mode == "motion_map" and motion_map is not None:
            motion_input = motion_map.to(dtype=image.dtype)
        elif image_t_minus_1 is not None:
            motion_input = self._motion_from_prev(image, image_t_minus_1)
        else:
            motion_input = image.new_zeros((image.shape[0], self.motion_in_channels, image.shape[2], image.shape[3]))

        motion_feat = self.motion_encoder(motion_input)
        return F.interpolate(motion_feat, size=feat.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        image: torch.Tensor,
        image_t_minus_1: Optional[torch.Tensor] = None,
        motion_map: Optional[torch.Tensor] = None,
        ablate_local_token: bool = False,
        ablate_global_token: bool = False,
        ablate_motion_token: bool = False,
        touch_ablation_mode: str = "rgb_motion",
    ) -> Dict[str, torch.Tensor]:
        x = self.normalize(image)
        feat = self.backbone(x)
        raw_heatmap_logits = self.heatmap_head(feat)
        heatmap_logits = F.interpolate(
            raw_heatmap_logits,
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=False,
        )
        heatmap_prob = spatial_softmax(heatmap_logits)
        xy = dsnt_xy(heatmap_prob).clamp(0.0, 1.0)
        loc_conf = heatmap_peak_confidence(heatmap_prob)

        global_token = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        local_prob = F.interpolate(heatmap_prob, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        local_prob = local_prob / local_prob.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        local_token = (feat * local_prob).sum(dim=(2, 3))
        motion_feat = self._motion_features(image, feat, image_t_minus_1, motion_map)
        motion_token = (motion_feat * local_prob).sum(dim=(2, 3))
        if ablate_global_token:
            global_token = torch.zeros_like(global_token)
        if ablate_local_token:
            local_token = torch.zeros_like(local_token)
        local_token_for_touch = local_token
        if ablate_motion_token and motion_token.numel() > 0:
            motion_token = torch.zeros_like(motion_token)
        if touch_ablation_mode == "no_motion" and motion_token.numel() > 0:
            motion_token = torch.zeros_like(motion_token)
        elif touch_ablation_mode == "motion_only":
            local_token_for_touch = torch.zeros_like(local_token)
        elif touch_ablation_mode == "rgb_motion":
            local_token_for_touch = local_token
        else:
            raise ValueError(f"Unknown touch_ablation_mode: {touch_ablation_mode}")

        token = torch.cat([global_token, local_token, xy, loc_conf.unsqueeze(1)], dim=1)
        touch_token = torch.cat([global_token, local_token_for_touch, xy, loc_conf.unsqueeze(1), motion_token], dim=1)

        valid_logit = self.valid_head(token).squeeze(1)
        touch_logit = self.touch_head(touch_token).squeeze(1)
        p_valid = torch.sigmoid(valid_logit)
        p_touch_cond = torch.sigmoid(touch_logit)
        p_background = 1.0 - p_valid
        p_hover = p_valid * (1.0 - p_touch_cond)
        p_touch = p_valid * p_touch_cond
        state_probs = torch.stack([p_background, p_hover, p_touch], dim=1).clamp_min(1e-6)
        state_probs = state_probs / state_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        state_logits = torch.log(state_probs.clamp_min(1e-6))

        return {
            "xy": xy,
            "heatmap_logits": heatmap_logits,
            "heatmap_prob": heatmap_prob,
            "loc_conf": loc_conf,
            "valid_logit": valid_logit,
            "touch_logit": touch_logit,
            "state_logits": state_logits,
            "state_probs": state_probs,
        }


class LocTouchNetLiteONNX(nn.Module):
    """ONNX wrapper returning a stable tuple of exported outputs."""

    def __init__(
        self,
        model: LocTouchNetLite,
        include_heatmap_output: bool = False,
        temporal_input_mode: str = "motion_map",
    ) -> None:
        super().__init__()
        self.model = model
        self.include_heatmap_output = include_heatmap_output
        self.temporal_input_mode = temporal_input_mode

    def forward(self, image: torch.Tensor, temporal_input: Optional[torch.Tensor] = None):
        kwargs = {}
        if self.model.temporal_enabled and temporal_input is not None:
            if self.temporal_input_mode == "image_t_minus_1":
                kwargs["image_t_minus_1"] = temporal_input
            elif self.temporal_input_mode == "motion_map":
                kwargs["motion_map"] = temporal_input
            else:
                raise ValueError(f"Unknown temporal_input_mode: {self.temporal_input_mode}")
        out = self.model(image, **kwargs)
        base = (
            out["xy"],
            out["loc_conf"],
            out["valid_logit"],
            out["touch_logit"],
            out["state_logits"],
        )
        if self.include_heatmap_output:
            return base + (out["heatmap_logits"],)
        return base
