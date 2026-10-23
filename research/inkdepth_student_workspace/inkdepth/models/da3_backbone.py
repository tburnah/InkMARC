from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DA3DepthBackbone(nn.Module):
    """Frozen Depth Anything 3 depth provider for RGB tensors in [0, 1]."""

    def __init__(
        self,
        repo_root: str | Path,
        model_id: str = "depth-anything/DA3-SMALL",
        process_res: int = 322,
        process_res_method: str = "lower_bound_resize",
        num_workers: int = 1,
        normalize: str = "per_frame_percentile",
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.model_id = model_id
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.num_workers = int(num_workers)
        self.normalize = normalize

        src_path = str(self.repo_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from depth_anything_3.api import DepthAnything3  # noqa: WPS433

        model = DepthAnything3.from_pretrained(model_id).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        object.__setattr__(self, "model", model)

    @torch.inference_mode()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB tensor with shape [B,3,H,W], got {tuple(rgb.shape)}")
        device = rgb.device
        target_size = tuple(int(x) for x in rgb.shape[-2:])
        self._move_da3_to_device(device)
        frames = tensor_to_uint8_frames(rgb)
        imgs_cpu, _, _ = self.model.input_processor(
            frames,
            extrinsics=None,
            intrinsics=None,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
            num_workers=max(1, self.num_workers),
            sequential=self.num_workers <= 1,
            print_progress=False,
        )
        imgs = imgs_cpu[:, None].to(device, non_blocking=True).float()
        output = self.model.forward(
            imgs,
            None,
            None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        )
        depth = extract_da3_depth(output)
        depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=True)
        if self.normalize == "per_frame_percentile":
            depth = normalize_per_frame_percentile(depth)
        elif self.normalize == "none":
            depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Unsupported DA3 depth normalization: {self.normalize}")
        return depth

    def _move_da3_to_device(self, device: torch.device) -> None:
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = device
        if model_device != device:
            self.model.to(device)


def tensor_to_uint8_frames(rgb: torch.Tensor) -> list[np.ndarray]:
    arr = rgb.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(frame) for frame in arr]


def extract_da3_depth(output: Any) -> torch.Tensor:
    depth = output["depth"]
    if depth.ndim == 4:
        depth = depth.reshape(depth.shape[0], 1, depth.shape[-2], depth.shape[-1])
    elif depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.permute(0, 1, 4, 2, 3).reshape(depth.shape[0], 1, depth.shape[2], depth.shape[3])
    else:
        raise RuntimeError(f"Unexpected DA3 depth shape: {tuple(depth.shape)}")
    return depth.float()


def normalize_per_frame_percentile(depth: torch.Tensor, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = depth.flatten(1)
    lo = torch.quantile(flat, low, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, high, dim=1).view(-1, 1, 1, 1)
    denom = (hi - lo).clamp_min(1.0e-8)
    return ((depth - lo) / denom).clamp(0.0, 1.0)
