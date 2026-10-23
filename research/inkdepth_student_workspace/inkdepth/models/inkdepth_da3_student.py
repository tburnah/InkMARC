from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .da3_backbone import DA3DepthBackbone
from .inkdepth_student import InkDepthStudent


def _surface_geometry_channels(cfg: Any) -> int:
    geometry_cfg = getattr(getattr(cfg, "data", None), "surface_geometry", None)
    return 3 if geometry_cfg is not None and bool(getattr(geometry_cfg, "enabled", False)) else 0


class InkDepthDA3Student(nn.Module):
    """RGB touch/location model with live DA3-small depth as a frozen backbone."""

    def __init__(
        self,
        student: InkDepthStudent,
        da3_depth: DA3DepthBackbone,
        student_input_mode: str = "rgbd",
    ) -> None:
        super().__init__()
        if student_input_mode not in {"rgbd", "depth_only"}:
            raise ValueError("student_input_mode must be 'rgbd' or 'depth_only'")
        self.student = student
        self.da3_depth = da3_depth
        self.student_input_mode = student_input_mode

    @classmethod
    def from_config(cls, cfg: Any) -> "InkDepthDA3Student":
        model_cfg = cfg.model
        geometry_channels = _surface_geometry_channels(cfg)
        student = InkDepthStudent(
            input_size=tuple(cfg.data.input_size),
            depth_out_size=tuple(model_cfg.depth_out_size),
            backbone=str(model_cfg.backbone),
            input_channels=4 + geometry_channels,
            width_mult=float(model_cfg.width_mult),
            pretrained=bool(model_cfg.pretrained),
            fpn_channels=int(model_cfg.fpn_channels),
            state_mode=str(model_cfg.state_mode),
            num_states=int(model_cfg.num_states),
            presence_num_states=int(getattr(model_cfg, "presence_num_states", model_cfg.num_states)),
            use_depth_features_for_tasks=bool(getattr(model_cfg, "use_depth_features_for_tasks", False)),
            depth_feature_source=str(getattr(model_cfg, "depth_feature_source", "decoder")),
            state_context_mode=str(getattr(model_cfg, "state_context_mode", "gap_xy_depth")),
            location_head_type=str(getattr(model_cfg, "location_head_type", "heatmap")),
            train_depth=False,
            train_touch=bool(model_cfg.train_touch),
            train_state=bool(model_cfg.train_state),
            train_location=bool(model_cfg.train_location),
            external_depth_channel=True,
            freeze_backbone=bool(getattr(model_cfg, "freeze_backbone", False)),
            freeze_neck=bool(getattr(model_cfg, "freeze_neck", False)),
            freeze_depth_head=True,
            freeze_state_head=bool(getattr(model_cfg, "freeze_state_head", False)),
            freeze_location_head=bool(getattr(model_cfg, "freeze_location_head", False)),
            freeze_presence_head=bool(getattr(model_cfg, "freeze_presence_head", False)),
        )
        da3_depth = DA3DepthBackbone(
            repo_root=str(getattr(model_cfg, "da3_repo_root", "/home/t/tburnah/src/depth_model/Depth-Anything-3")),
            model_id=str(getattr(model_cfg, "da3_model_id", "depth-anything/DA3-SMALL")),
            process_res=int(getattr(model_cfg, "da3_process_res", 322)),
            process_res_method=str(getattr(model_cfg, "da3_process_res_method", "lower_bound_resize")),
            num_workers=int(getattr(model_cfg, "da3_preprocess_workers", 1)),
            normalize=str(getattr(model_cfg, "da3_depth_normalize", "per_frame_percentile")),
        )
        return cls(
            student=student,
            da3_depth=da3_depth,
            student_input_mode=str(getattr(model_cfg, "da3_student_input_mode", "rgbd")),
        )

    @property
    def backbone(self) -> nn.Module:
        return self.student.backbone

    def set_frozen_modules_eval(self) -> None:
        self.student.set_frozen_modules_eval()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | None]:
        rgb = image[:, :3]
        geometry = image[:, 3:]
        depth = self.da3_depth(rgb)
        if self.student_input_mode == "depth_only":
            rgb = torch.zeros_like(rgb)
        rgbd = torch.cat([rgb, depth, geometry], dim=1)
        outputs = self.student(rgbd)
        outputs["depth"] = depth
        outputs["depth_lowres"] = F.interpolate(
            depth,
            size=tuple(self.student.depth_out_size),
            mode="bilinear",
            align_corners=False,
        )
        return outputs
