from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .heads import DepthHead, DirectLocationHead, DirectPlusHeatmapLocationHead, FPNNeck, LocationHead, StateHead


class InkMARCSense(nn.Module):
    """Mobile RGB student with auxiliary depth, state, and location heads."""

    def __init__(
        self,
        input_size: tuple[int, int] = (672, 432),
        depth_out_size: tuple[int, int] = (168, 108),
        backbone: str = "mobilenet_v3_large",
        input_channels: int = 3,
        width_mult: float = 1.0,
        pretrained: bool = True,
        fpn_channels: int = 96,
        state_mode: str = "binary",
        num_states: int = 3,
        presence_num_states: int | None = None,
        use_depth_features_for_tasks: bool = True,
        depth_feature_source: str = "decoder",
        state_context_mode: str = "gap",
        location_head_type: str = "heatmap",
        train_depth: bool = True,
        train_touch: bool = True,
        train_state: bool = True,
        train_location: bool = True,
        enable_depth_head: bool = True,
        external_depth_channel: bool = False,
        freeze_backbone: bool = False,
        freeze_neck: bool = False,
        freeze_depth_head: bool = False,
        freeze_state_head: bool = False,
        freeze_location_head: bool = False,
        freeze_presence_head: bool = False,
    ) -> None:
        super().__init__()
        if backbone != "mobilenet_v3_large":
            raise ValueError("First implementation supports backbone='mobilenet_v3_large' only")
        if depth_feature_source not in {"decoder", "p4"}:
            raise ValueError("depth_feature_source must be 'decoder' or 'p4'")
        if state_context_mode not in {"gap", "gap_xy", "gap_xy_depth"}:
            raise ValueError("state_context_mode must be 'gap', 'gap_xy', or 'gap_xy_depth'")
        if location_head_type not in {"heatmap", "direct", "direct_plus_heatmap"}:
            raise ValueError("location_head_type must be 'heatmap', 'direct', or 'direct_plus_heatmap'")
        self.input_size = input_size
        self.input_channels = input_channels
        self.depth_out_size = depth_out_size
        self.state_mode = state_mode
        self.presence_num_states = int(presence_num_states or num_states)
        self.use_depth_features_for_tasks = use_depth_features_for_tasks
        self.depth_feature_source = depth_feature_source
        self.state_context_mode = state_context_mode
        self.location_head_type = location_head_type
        self.train_depth = train_depth
        self.train_touch = train_touch
        self.train_state = train_state
        self.train_location = train_location
        self.enable_depth_head = enable_depth_head
        self.external_depth_channel = external_depth_channel
        if not enable_depth_head and train_depth:
            raise ValueError("train_depth=True requires enable_depth_head=True")
        if not enable_depth_head and use_depth_features_for_tasks:
            raise ValueError("use_depth_features_for_tasks=True requires enable_depth_head=True")
        if not enable_depth_head and state_context_mode == "gap_xy_depth":
            raise ValueError("state_context_mode='gap_xy_depth' requires enable_depth_head=True")

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights, width_mult=width_mult).features
        self._adapt_first_conv(input_channels)
        self.feature_indices = self._infer_feature_indices(input_size)
        feat_channels = self._infer_feature_channels(input_size)
        self.neck = FPNNeck(feat_channels, out_channels=fpn_channels)
        self.depth_head = DepthHead(fpn_channels, out_size=depth_out_size) if enable_depth_head else None

        depth_task_channels = (
            self.depth_head.feature_channels
            if self.depth_head is not None and depth_feature_source == "decoder"
            else fpn_channels
        )
        pooled_channels = fpn_channels * 3
        task_channels = pooled_channels + (depth_task_channels if use_depth_features_for_tasks else 0)
        loc_channels = fpn_channels + (depth_task_channels if use_depth_features_for_tasks else 0)
        if state_context_mode in {"gap_xy", "gap_xy_depth"}:
            task_channels += loc_channels
        if state_context_mode == "gap_xy_depth":
            task_channels += 5
        self.state_head = StateHead(task_channels, mode=state_mode, num_states=num_states)
        self.presence_head = (
            StateHead(task_channels, mode="multiclass", num_states=self.presence_num_states)
            if state_mode == "binary" and train_state
            else None
        )
        if location_head_type == "direct":
            self.location_head = DirectLocationHead(loc_channels)
        elif location_head_type == "direct_plus_heatmap":
            self.location_head = DirectPlusHeatmapLocationHead(loc_channels, heatmap_size=depth_out_size)
        else:
            self.location_head = LocationHead(loc_channels, heatmap_size=depth_out_size)

        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        if input_channels > 3:
            mean_values.extend([0.5] * (input_channels - 3))
            std_values.extend([0.25] * (input_channels - 3))
        mean = torch.tensor(mean_values[:input_channels]).view(1, input_channels, 1, 1)
        std = torch.tensor(std_values[:input_channels]).view(1, input_channels, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        self._apply_freeze_config(
            freeze_backbone=freeze_backbone,
            freeze_neck=freeze_neck,
            freeze_depth_head=freeze_depth_head,
            freeze_state_head=freeze_state_head,
            freeze_location_head=freeze_location_head,
            freeze_presence_head=freeze_presence_head,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "InkMARCSense":
        m = cfg.model
        return cls(
            input_size=tuple(cfg.data.input_size),
            depth_out_size=tuple(m.depth_out_size),
            backbone=str(m.backbone),
            input_channels=int(getattr(m, "input_channels", 3)),
            width_mult=float(m.width_mult),
            pretrained=bool(m.pretrained),
            fpn_channels=int(m.fpn_channels),
            state_mode=str(m.state_mode),
            num_states=int(m.num_states),
            presence_num_states=int(getattr(m, "presence_num_states", m.num_states)),
            use_depth_features_for_tasks=bool(m.use_depth_features_for_tasks),
            depth_feature_source=str(getattr(m, "depth_feature_source", "decoder")),
            state_context_mode=str(getattr(m, "state_context_mode", "gap")),
            location_head_type=str(getattr(m, "location_head_type", "heatmap")),
            train_depth=bool(m.train_depth),
            train_touch=bool(m.train_touch),
            train_state=bool(m.train_state),
            train_location=bool(m.train_location),
            enable_depth_head=bool(getattr(m, "enable_depth_head", True)),
            external_depth_channel=bool(getattr(m, "external_depth_channel", False)),
            freeze_backbone=bool(getattr(m, "freeze_backbone", False)),
            freeze_neck=bool(getattr(m, "freeze_neck", False)),
            freeze_depth_head=bool(getattr(m, "freeze_depth_head", False)),
            freeze_state_head=bool(getattr(m, "freeze_state_head", False)),
            freeze_location_head=bool(getattr(m, "freeze_location_head", False)),
            freeze_presence_head=bool(getattr(m, "freeze_presence_head", False)),
        )

    def _apply_freeze_config(
        self,
        freeze_backbone: bool = False,
        freeze_neck: bool = False,
        freeze_depth_head: bool = False,
        freeze_state_head: bool = False,
        freeze_location_head: bool = False,
        freeze_presence_head: bool = False,
    ) -> None:
        self._frozen_modules: list[nn.Module] = []
        modules: list[tuple[nn.Module | None, bool]] = [
            (self.backbone, freeze_backbone),
            (self.neck, freeze_neck),
            (self.depth_head, freeze_depth_head),
            (self.state_head, freeze_state_head),
            (self.location_head, freeze_location_head),
        ]
        if self.presence_head is not None:
            modules.append((self.presence_head, freeze_presence_head))
        for module, freeze in modules:
            if module is None:
                continue
            if freeze:
                self._frozen_modules.append(module)
                for param in module.parameters():
                    param.requires_grad = False

    def set_frozen_modules_eval(self) -> None:
        for module in getattr(self, "_frozen_modules", []):
            module.eval()

    def _infer_feature_indices(self, input_size: tuple[int, int]) -> dict[str, int]:
        candidates: dict[int, int] = {}
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, *input_size)
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
            x = torch.zeros(1, self.input_channels, *input_size)
        feats = self._extract_features(x, normalize=False)
        return {key: value.shape[1] for key, value in feats.items()}

    def _adapt_first_conv(self, input_channels: int) -> None:
        if input_channels == 3:
            return
        first_conv = self.backbone[0][0]
        if not isinstance(first_conv, nn.Conv2d):
            raise TypeError("Expected MobileNetV3 first layer to be Conv2dNormActivation with Conv2d at [0][0]")
        new_conv = nn.Conv2d(
            input_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )
        with torch.no_grad():
            old_weight = first_conv.weight
            new_conv.weight[:, : min(3, input_channels)].copy_(old_weight[:, : min(3, input_channels)])
            if input_channels > 3:
                extra = old_weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:].copy_(extra.repeat(1, input_channels - 3, 1, 1))
            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)
        self.backbone[0][0] = new_conv

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

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | None]:
        input_size = tuple(image.shape[-2:])
        external_depth = image[:, 3:4] if self.external_depth_channel and image.shape[1] >= 4 else None
        feats = self._extract_features(image)
        pyr = self.neck(feats)
        needs_depth = self.train_depth or self.use_depth_features_for_tasks or (
            (self.train_state or self.train_touch) and self.state_context_mode == "gap_xy_depth"
        )
        if external_depth is not None and not self.train_depth and not self.use_depth_features_for_tasks:
            depth = external_depth
            depth_low = F.interpolate(external_depth, size=self.depth_out_size, mode="bilinear", align_corners=False)
            depth_decoder_features = None
        elif needs_depth:
            if self.depth_head is None:
                raise RuntimeError("This model needs depth features but enable_depth_head=False")
            depth, depth_low, depth_decoder_features = self.depth_head(pyr["p4"], input_size=input_size)
        else:
            depth = None
            depth_low = None
            depth_decoder_features = None
        depth_task_features = depth_decoder_features if self.depth_feature_source == "decoder" else pyr["p4"]
        if self.use_depth_features_for_tasks and depth_decoder_features is None:
            raise RuntimeError("use_depth_features_for_tasks requires the learned depth head")

        location_heatmap = None
        xy = None
        needs_location_context = self.train_location or (
            (self.train_state or self.train_touch) and self.state_context_mode in {"gap_xy", "gap_xy_depth"}
        )
        loc_feat = pyr["p4"]
        if self.use_depth_features_for_tasks:
            loc_feat = torch.cat([loc_feat, depth_task_features], dim=1)
        if needs_location_context:
            location_heatmap, xy = self.location_head(loc_feat)

        state_logits = None
        touch_logits = None
        if self.train_state or self.train_touch:
            pooled = [F.adaptive_avg_pool2d(pyr[key], 1).flatten(1) for key in ("p4", "p8", "p16")]
            if self.use_depth_features_for_tasks:
                pooled.append(F.adaptive_avg_pool2d(depth_task_features, 1).flatten(1))
            if self.state_context_mode in {"gap_xy", "gap_xy_depth"}:
                pooled.append(_heatmap_weighted_pool(loc_feat, location_heatmap))
            if self.state_context_mode == "gap_xy_depth":
                if depth_low is None:
                    raise RuntimeError("gap_xy_depth context requires a depth tensor")
                pooled.append(_heatmap_depth_stats(depth_low, location_heatmap))
            task_vec = torch.cat(pooled, dim=1)
            state_logits, touch_logits = self.state_head(task_vec)
            if self.train_state and state_logits is None:
                if self.presence_head is None:
                    raise RuntimeError("train_state=True with binary state_mode requires a presence_head")
                state_logits, _ = self.presence_head(task_vec)

        if not self.train_depth and depth is not None:
            depth = depth.detach()
        if not self.train_depth and depth_low is not None:
            depth_low = depth_low.detach()
        return {
            "depth": depth,
            "depth_lowres": depth_low,
            "state_logits": state_logits if self.train_state else None,
            "touch_logits": touch_logits if self.train_touch else None,
            "location_heatmap": location_heatmap if self.train_location else None,
            "xy": xy if self.train_location else None,
        }


def _heatmap_weighted_pool(features: torch.Tensor, heatmap: torch.Tensor | None) -> torch.Tensor:
    if heatmap is None:
        return F.adaptive_avg_pool2d(features, 1).flatten(1)
    if heatmap.shape[-2:] != features.shape[-2:]:
        heatmap = F.interpolate(heatmap, size=features.shape[-2:], mode="bilinear", align_corners=False)
    weights = torch.softmax(heatmap.flatten(2), dim=-1)
    pooled = (features.flatten(2) * weights).sum(dim=-1)
    return pooled


def _heatmap_depth_stats(depth: torch.Tensor, heatmap: torch.Tensor | None) -> torch.Tensor:
    if heatmap is not None and heatmap.shape[-2:] != depth.shape[-2:]:
        heatmap = F.interpolate(heatmap, size=depth.shape[-2:], mode="bilinear", align_corners=False)
    depth_flat = depth.flatten(2)
    global_mean = depth_flat.mean(dim=-1)
    global_var = (depth_flat - global_mean.unsqueeze(-1)).square().mean(dim=-1)
    global_std = torch.sqrt(global_var + 1e-6)

    if heatmap is None:
        local_mean = global_mean
        local_abs_contrast = torch.zeros_like(global_mean)
    else:
        weights = torch.softmax(heatmap.flatten(2), dim=-1)
        local_mean = (depth_flat * weights).sum(dim=-1)
        local_abs_contrast = ((depth_flat - local_mean.unsqueeze(-1)).abs() * weights).sum(dim=-1)

    return torch.cat(
        [local_mean, global_mean, local_mean - global_mean, global_std, local_abs_contrast],
        dim=1,
    )


# Compatibility with historical imports and serialized model objects.
InkDepthStudent = InkMARCSense
