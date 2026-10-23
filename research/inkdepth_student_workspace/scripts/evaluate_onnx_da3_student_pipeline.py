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


class ONNXDA3StudentPipeline(torch.nn.Module):
    def __init__(
        self,
        cfg: Any,
        da3_onnx: str | Path,
        student_onnx: str | Path,
        da3_repo_root: str | Path,
        da3_provider: str = "CPUExecutionProvider",
        student_provider: str = "CPUExecutionProvider",
        da3_preprocess: str = "da3",
        da3_square_size: int = 322,
    ) -> None:
        super().__init__()
        import onnxruntime as ort

        repo_src = str(Path(da3_repo_root).expanduser().resolve() / "src")
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: WPS433

        self.cfg = cfg
        self.input_processor = InputProcessor()
        self.process_res = int(getattr(cfg.model, "da3_process_res", 322))
        self.process_res_method = str(getattr(cfg.model, "da3_process_res_method", "lower_bound_resize"))
        self.target_size = tuple(int(x) for x in cfg.data.input_size)
        self.da3_preprocess = da3_preprocess
        self.da3_square_size = int(da3_square_size)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.da3_session = ort.InferenceSession(
            str(Path(da3_onnx).expanduser().resolve()),
            sess_options=sess_options,
            providers=[da3_provider],
        )
        self.student_session = ort.InferenceSession(
            str(Path(student_onnx).expanduser().resolve()),
            sess_options=sess_options,
            providers=[student_provider],
        )
        self.da3_input_name = self.da3_session.get_inputs()[0].name
        self.da3_input_shape = [int(dim) for dim in self.da3_session.get_inputs()[0].shape]
        self.da3_output_name = self.da3_session.get_outputs()[0].name
        self.student_input_name = self.student_session.get_inputs()[0].name
        self.student_input_shape = [int(dim) for dim in self.student_session.get_inputs()[0].shape]
        self.student_output_names = [out.name for out in self.student_session.get_outputs()]

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected fixed batch [1,C,H,W], got {tuple(image.shape)}")
        image_cpu = image.detach().float().cpu()
        rgb = image_cpu[:, :3]
        geometry = image_cpu[:, 3:]
        depth = self._infer_da3_depth(rgb)
        student_input = torch.cat([rgb, depth, geometry], dim=1).numpy().astype(np.float32)
        expected = self.student_input_shape
        if list(student_input.shape) != expected:
            raise ValueError(f"Student ONNX input shape mismatch: got {list(student_input.shape)}, expected {expected}")
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

    def _infer_da3_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        da3_input = self._preprocess_rgb(rgb)
        expected = self.da3_input_shape
        if list(da3_input.shape) != expected:
            raise ValueError(f"DA3 ONNX input shape mismatch: got {list(da3_input.shape)}, expected {expected}")
        raw_depth = self.da3_session.run([self.da3_output_name], {self.da3_input_name: da3_input})[0]
        depth = torch.from_numpy(raw_depth.astype(np.float32))
        if depth.ndim == 3:
            depth = depth[:, None]
        depth = F.interpolate(depth, size=self.target_size, mode="bilinear", align_corners=True)
        return normalize_per_frame_percentile(depth)

    def _preprocess_rgb(self, rgb: torch.Tensor) -> np.ndarray:
        if self.da3_preprocess == "da3":
            frames = tensor_to_uint8_frames(rgb)
            imgs_cpu, _, _ = self.input_processor(
                frames,
                extrinsics=None,
                intrinsics=None,
                process_res=self.process_res,
                process_res_method=self.process_res_method,
                num_workers=1,
                sequential=True,
                print_progress=False,
            )
            return imgs_cpu.unsqueeze(0).numpy().astype(np.float32)
        if self.da3_preprocess == "square":
            return square_imagenet_preprocess(rgb, self.da3_square_size)
        raise ValueError(f"Unsupported DA3 preprocess mode: {self.da3_preprocess}")


def tensor_to_uint8_frames(rgb: torch.Tensor) -> list[np.ndarray]:
    arr = rgb.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(frame) for frame in arr]


def square_imagenet_preprocess(rgb: torch.Tensor, size: int) -> np.ndarray:
    import cv2

    frame = tensor_to_uint8_frames(rgb)[0]
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, None]
    return np.ascontiguousarray(x.astype(np.float32))


def normalize_per_frame_percentile(depth: torch.Tensor, low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = depth.flatten(1)
    lo = torch.quantile(flat, low, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, high, dim=1).view(-1, 1, 1, 1)
    denom = (hi - lo).clamp_min(1.0e-8)
    return ((depth - lo) / denom).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DA3 ONNX + deployable InkDepth student ONNX.")
    parser.add_argument("--config", default=str(ROOT / "configs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml"))
    parser.add_argument("--da3-onnx", default=str(ROOT / "outputs/onnx/da3_small_depth_504x322.onnx"))
    parser.add_argument(
        "--student-onnx",
        default=str(ROOT / "outputs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy/deployable_student_rgbd_geometry.onnx"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/onnx_pipeline_eval"))
    parser.add_argument("--da3-provider", default="CPUExecutionProvider")
    parser.add_argument("--student-provider", default="CPUExecutionProvider")
    parser.add_argument("--da3-preprocess", choices=["da3", "square"], default="da3")
    parser.add_argument("--da3-square-size", type=int, default=322)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    cfg.training.batch_size = 1
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.max_visuals = 0
    if args.max_batches > 0:
        cfg.data.max_frames_per_video = int(args.max_batches)
        cfg.data.max_videos_per_split = 1

    model = ONNXDA3StudentPipeline(
        cfg=cfg,
        da3_onnx=args.da3_onnx,
        student_onnx=args.student_onnx,
        da3_repo_root=str(getattr(cfg.model, "da3_repo_root", "/home/t/tburnah/src/depth_model/Depth-Anything-3")),
        da3_provider=args.da3_provider,
        student_provider=args.student_provider,
        da3_preprocess=args.da3_preprocess,
        da3_square_size=args.da3_square_size,
    ).eval()

    ds = StylusDataset(cfg, split=args.split, training=False)
    if args.max_batches > 0:
        ds.records = ds.records[: int(args.max_batches)]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=stylus_collate_fn)
    output_dir = Path(args.output_dir)
    metrics = validate(model, loader, cfg, torch.device("cpu"), output_dir=output_dir)
    (output_dir / "onnx_pipeline_config.json").write_text(
        json.dumps(
            {
                "config": str(Path(args.config).resolve()),
                "da3_onnx": str(Path(args.da3_onnx).resolve()),
                "student_onnx": str(Path(args.student_onnx).resolve()),
                "split": args.split,
                "da3_provider": args.da3_provider,
                "student_provider": args.student_provider,
                "da3_preprocess": args.da3_preprocess,
                "records": len(ds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics_onnx_pipeline.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
