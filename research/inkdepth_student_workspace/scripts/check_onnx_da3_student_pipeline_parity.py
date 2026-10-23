#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import build_model_from_config
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides
from scripts.evaluate_onnx_da3_student_pipeline import ONNXDA3StudentPipeline


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare live PyTorch DA3 pipeline with DA3 ONNX + student ONNX.")
    parser.add_argument("--config", default=str(ROOT / "configs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy/latest.pt"))
    parser.add_argument("--da3-onnx", default=str(ROOT / "outputs/onnx/da3_small_depth_504x322.onnx"))
    parser.add_argument(
        "--student-onnx",
        default=str(ROOT / "outputs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy/deployable_student_rgbd_geometry.onnx"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--da3-provider", default="CUDAExecutionProvider")
    parser.add_argument("--student-provider", default="CUDAExecutionProvider")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    cfg.training.batch_size = 1
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, torch_model, map_location=device)
    onnx_model = ONNXDA3StudentPipeline(
        cfg=cfg,
        da3_onnx=args.da3_onnx,
        student_onnx=args.student_onnx,
        da3_repo_root=str(getattr(cfg.model, "da3_repo_root", "/home/t/tburnah/src/depth_model/Depth-Anything-3")),
        da3_provider=args.da3_provider,
        student_provider=args.student_provider,
    ).eval()

    ds = StylusDataset(cfg, split=args.split, training=False)
    ds.records = ds.records[: int(args.frames)]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=stylus_collate_fn)

    diffs: dict[str, list[float]] = {
        "depth": [],
        "depth_lowres": [],
        "state_logits": [],
        "touch_logits": [],
        "xy": [],
    }
    for batch in loader:
        image_cpu = batch["image"]
        torch_out = torch_model(image_cpu.to(device))
        onnx_out = onnx_model(image_cpu)
        for key in diffs:
            a = torch_out.get(key)
            b = onnx_out.get(key)
            if a is None or b is None:
                continue
            diffs[key].append(float((a.detach().cpu() - b.detach().cpu()).abs().max()))

    print(f"checked_frames={len(ds)}")
    for key, values in diffs.items():
        if not values:
            continue
        t = torch.tensor(values)
        print(f"{key}/max_abs_diff_max: {float(t.max()):.8f}")
        print(f"{key}/max_abs_diff_mean: {float(t.mean()):.8f}")


if __name__ == "__main__":
    main()
