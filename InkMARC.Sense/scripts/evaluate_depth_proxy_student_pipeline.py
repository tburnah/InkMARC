#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.training.validate import validate
from inkdepth.utils.config import load_config, parse_overrides


class DepthProxyStudentPipeline(torch.nn.Module):
    def __init__(
        self,
        cfg: Any,
        proxy_onnx: str | Path,
        student_onnx: str | Path,
        proxy_provider: str = "CPUExecutionProvider",
        student_provider: str = "CPUExecutionProvider",
    ) -> None:
        super().__init__()
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.proxy_session = ort.InferenceSession(
            str(Path(proxy_onnx).expanduser().resolve()),
            sess_options=session_options,
            providers=[proxy_provider],
        )
        self.student_session = ort.InferenceSession(
            str(Path(student_onnx).expanduser().resolve()),
            sess_options=session_options,
            providers=[student_provider],
        )
        self.proxy_input_name = self.proxy_session.get_inputs()[0].name
        self.proxy_input_shape = [int(dim) for dim in self.proxy_session.get_inputs()[0].shape]
        self.proxy_output_name = self.proxy_session.get_outputs()[0].name
        self.student_input_name = self.student_session.get_inputs()[0].name
        self.student_input_shape = [int(dim) for dim in self.student_session.get_inputs()[0].shape]
        self.student_output_names = [out.name for out in self.student_session.get_outputs()]
        self.target_size = tuple(int(x) for x in cfg.data.input_size)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected fixed batch [1,C,H,W], got {tuple(image.shape)}")
        image_cpu = image.detach().float().cpu()
        rgb = image_cpu[:, :3]
        geometry = image_cpu[:, 3:]
        depth = self._infer_proxy_depth(rgb)
        student_input = torch.cat([rgb, depth, geometry], dim=1).numpy().astype(np.float32)
        if list(student_input.shape) != self.student_input_shape:
            raise ValueError(
                f"Student input shape mismatch: got {list(student_input.shape)}, expected {self.student_input_shape}"
            )
        student_outputs = self.student_session.run(self.student_output_names, {self.student_input_name: student_input})
        output_by_name = {
            name: torch.from_numpy(value.astype(np.float32))
            for name, value in zip(self.student_output_names, student_outputs)
        }
        return {
            "depth": depth,
            "depth_lowres": output_by_name.get("depth_lowres"),
            "state_logits": output_by_name.get("state_logits"),
            "touch_logits": output_by_name.get("touch_logits"),
            "location_heatmap": output_by_name.get("location_heatmap"),
            "xy": output_by_name.get("xy"),
        }

    def _infer_proxy_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        _, _, h, w = self.proxy_input_shape
        proxy_rgb = F.interpolate(rgb, size=(h, w), mode="bilinear", align_corners=False).numpy().astype(np.float32)
        raw_depth = self.proxy_session.run([self.proxy_output_name], {self.proxy_input_name: proxy_rgb})[0]
        depth = torch.from_numpy(raw_depth.astype(np.float32))
        if depth.ndim == 3:
            depth = depth[:, None]
        if tuple(depth.shape[-2:]) != self.target_size:
            depth = F.interpolate(depth, size=self.target_size, mode="bilinear", align_corners=True)
        return normalize_per_frame_percentile(depth)


def normalize_per_frame_percentile(depth: torch.Tensor, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = depth.flatten(1)
    lo = torch.quantile(flat, low, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, high, dim=1).view(-1, 1, 1, 1)
    denom = (hi - lo).clamp_min(1.0e-8)
    return ((depth - lo) / denom).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate depth proxy ONNX + deployable InkMARC-Sense student ONNX.")
    parser.add_argument("--config", default=str(ROOT / "configs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml"))
    parser.add_argument("--proxy-onnx", required=True)
    parser.add_argument(
        "--student-onnx",
        default=str(ROOT / "outputs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy/deployable_student_rgbd_geometry.onnx"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/depth_proxy_student_eval"))
    parser.add_argument("--proxy-provider", default="CPUExecutionProvider")
    parser.add_argument("--student-provider", default="CPUExecutionProvider")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    cfg.training.batch_size = 1
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.max_visuals = 0

    model = DepthProxyStudentPipeline(
        cfg=cfg,
        proxy_onnx=args.proxy_onnx,
        student_onnx=args.student_onnx,
        proxy_provider=args.proxy_provider,
        student_provider=args.student_provider,
    ).eval()
    ds = StylusDataset(cfg, split=args.split, training=False)
    if args.max_batches > 0:
        ds.records = ds.records[: int(args.max_batches)]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=stylus_collate_fn)
    output_dir = Path(args.output_dir)
    metrics = validate(model, loader, cfg, torch.device("cpu"), output_dir=output_dir)
    (output_dir / "depth_proxy_student_config.json").write_text(
        json.dumps(
            {
                "proxy_onnx": str(Path(args.proxy_onnx).expanduser().resolve()),
                "student_onnx": str(Path(args.student_onnx).expanduser().resolve()),
                "split": args.split,
                "records": len(ds),
                "proxy_provider": args.proxy_provider,
                "student_provider": args.student_provider,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics_depth_proxy_student.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
