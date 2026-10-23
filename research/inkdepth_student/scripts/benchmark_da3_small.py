#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Depth Anything 3 single-frame latency.")
    parser.add_argument("--repo-root", required=True, help="Depth-Anything-3 checkout")
    parser.add_argument("--model-id", default="depth-anything/DA3-SMALL")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--process-res", type=int, default=322)
    parser.add_argument("--process-res-method", default="lower_bound_resize")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--use-ray-pose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    sys.path.insert(0, str(repo_root / "src"))

    from depth_anything_3.api import DepthAnything3  # noqa: WPS433

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames = read_frames(Path(args.video), int(args.frames))
    if not frames:
        raise RuntimeError(f"No frames read from {args.video}")
    height, width = frames[0].shape[:2]

    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(args.model_id))
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start

    warmup_count = min(max(0, int(args.warmup_frames)), len(frames))
    for frame in frames[:warmup_count]:
        _ = infer_one(
            model,
            frame,
            process_res=int(args.process_res),
            process_res_method=str(args.process_res_method),
            target_size=(height, width),
            use_ray_pose=bool(args.use_ray_pose),
            timed=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    rows: list[dict[str, float]] = []
    start_total = time.perf_counter()
    for frame in frames:
        rows.append(
            infer_one(
                model,
                frame,
                process_res=int(args.process_res),
                process_res_method=str(args.process_res_method),
                target_size=(height, width),
                use_ray_pose=bool(args.use_ray_pose),
                timed=True,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_total = time.perf_counter() - start_total

    result = {
        "model_id": str(args.model_id),
        "repo_root": str(repo_root),
        "video": str(Path(args.video).resolve()),
        "frames": len(frames),
        "height": height,
        "width": width,
        "process_res": int(args.process_res),
        "process_res_method": str(args.process_res_method),
        "device": str(device),
        "use_ray_pose": bool(args.use_ray_pose),
        "warmup_frames": warmup_count,
        "load_seconds": load_seconds,
        "timed_seconds": elapsed_total,
        "timed_fps": len(frames) / elapsed_total if elapsed_total > 0 else 0.0,
        "mean_preprocess_ms": mean_ms(rows, "preprocess_seconds"),
        "mean_forward_ms": mean_ms(rows, "forward_seconds"),
        "mean_postprocess_ms": mean_ms(rows, "postprocess_seconds"),
        "mean_frame_ms_sum": mean_ms(rows, "total_seconds"),
        "processed_shape": rows[0].get("processed_shape", []) if rows else [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def read_frames(video: Path, limit: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < limit:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


@torch.inference_mode()
def infer_one(
    model: torch.nn.Module,
    frame_rgb: np.ndarray,
    process_res: int,
    process_res_method: str,
    target_size: tuple[int, int],
    use_ray_pose: bool,
    timed: bool,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    imgs_cpu, extrinsics, intrinsics = model.input_processor(
        [frame_rgb],
        extrinsics=None,
        intrinsics=None,
        process_res=process_res,
        process_res_method=process_res_method,
        num_workers=1,
        sequential=True,
    )
    t1 = time.perf_counter()
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
    output = model.forward(
        imgs,
        ex_t,
        in_t,
        export_feat_layers=[],
        infer_gs=False,
        use_ray_pose=use_ray_pose,
        ref_view_strategy="saddle_balanced",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    depth = output["depth"]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.permute(0, 1, 4, 2, 3).reshape(-1, 1, depth.shape[2], depth.shape[3])
    elif depth.ndim == 4:
        depth = depth.reshape(-1, 1, depth.shape[-2], depth.shape[-1])
    else:
        raise RuntimeError(f"Unexpected depth shape: {tuple(depth.shape)}")
    depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=True)
    _ = depth.float().cpu().numpy()
    t3 = time.perf_counter()
    if not timed:
        return {}
    return {
        "preprocess_seconds": t1 - t0,
        "forward_seconds": t2 - t1,
        "postprocess_seconds": t3 - t2,
        "total_seconds": t3 - t0,
        "processed_shape": list(imgs.shape),
    }


def mean_ms(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return 1000.0 * sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
