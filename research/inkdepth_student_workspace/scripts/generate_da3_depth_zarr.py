#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DA3 depth zarrs for manifest videos.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-id", default="depth-anything/DA3-SMALL")
    parser.add_argument("--process-res", type=int, default=322)
    parser.add_argument("--process-res-method", default="lower_bound_resize")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    sys.path.insert(0, str(repo_root / "src"))
    from depth_anything_3.api import DepthAnything3  # noqa: WPS433

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [dict(item) for item in manifest.get("items", [])]
    if int(args.limit_videos) > 0:
        items = items[: int(args.limit_videos)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(args.device) if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    load_start = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(args.model_id)).to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start

    records: list[dict[str, Any]] = []
    rewritten_items: list[dict[str, Any]] = []
    for item in tqdm(items, desc="videos"):
        video_path = Path(str(item.get("output_video_path") or item.get("video_path") or ""))
        if not video_path.exists():
            records.append({"video": str(video_path), "errors": ["video missing"]})
            continue
        stem = video_path.stem
        out_path = output_dir / f"{stem}.zarr"
        record = process_video(
            model=model,
            video_path=video_path,
            out_path=out_path,
            process_res=int(args.process_res),
            process_res_method=str(args.process_res_method),
            batch_size=int(args.batch_size),
            chunk_frames=int(args.chunk_frames),
            dtype=str(args.dtype),
            device=device,
            resume=bool(args.resume),
            warmup_frames=int(args.warmup_frames),
        )
        records.append(record)
        if not record.get("errors"):
            rewritten = dict(item)
            rewritten["depth_path"] = str(out_path)
            rewritten["depth_key"] = "depth_norm"
            rewritten_items.append(rewritten)

    output_manifest = {
        **{k: v for k, v in manifest.items() if k != "items"},
        "items": rewritten_items,
        "da3_depth_generation": {
            "source_manifest": str(manifest_path),
            "repo_root": str(repo_root),
            "model_id": str(args.model_id),
            "process_res": int(args.process_res),
            "process_res_method": str(args.process_res_method),
            "batch_size": int(args.batch_size),
            "dtype": str(args.dtype),
            "load_seconds": load_seconds,
            "records": records,
        },
    }
    out_manifest_path = Path(args.output_manifest)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_manifest_path}")


def process_video(
    model: torch.nn.Module,
    video_path: Path,
    out_path: Path,
    process_res: int,
    process_res_method: str,
    batch_size: int,
    chunk_frames: int,
    dtype: str,
    device: torch.device,
    resume: bool,
    warmup_frames: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "video": str(video_path),
        "output_depth": str(out_path),
        "errors": [],
    }
    if resume and out_path.exists():
        record["skipped"] = True
        return record

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        record["errors"].append("could not open video")
        return record
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        record["errors"].append(f"invalid video metadata frames={frame_count} size={width}x{height}")
        return record

    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(tmp_path), mode="w")
    group.attrs.update(
        {
            "source_video": str(video_path),
            "model": "depth-anything/DA3-SMALL",
            "process_res": process_res,
            "process_res_method": process_res_method,
            "shape": [frame_count, height, width],
            "normalization": "per-frame-percentile",
            "percentile_low": 2.0,
            "percentile_high": 98.0,
        }
    )
    out_dtype = np.float16 if dtype == "float16" else np.float32
    chunks = (min(max(1, chunk_frames), frame_count), height, width)
    raw_arr = create_array(group, "depth_raw", (frame_count, height, width), out_dtype, chunks)
    norm_arr = create_array(group, "depth_norm", (frame_count, height, width), out_dtype, chunks)

    try:
        if warmup_frames > 0:
            warmup = read_frame_batch(video_path, min(warmup_frames, frame_count), start_index=0)
            if warmup:
                _ = infer_batch(model, warmup, process_res, process_res_method, (height, width), device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        start_time = time.perf_counter()
        written = 0
        cap = cv2.VideoCapture(str(video_path))
        with tqdm(total=frame_count, desc=video_path.stem, leave=False, unit="frame") as progress:
            while written < frame_count:
                frames = []
                while len(frames) < batch_size and written + len(frames) < frame_count:
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        break
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if not frames:
                    break
                depth = infer_batch(model, frames, process_res, process_res_method, (height, width), device)
                norm = normalize_per_frame_percentile(depth)
                end = written + depth.shape[0]
                raw_arr[written:end] = depth.astype(out_dtype, copy=False)
                norm_arr[written:end] = norm.astype(out_dtype, copy=False)
                written = end
                progress.update(len(frames))
        cap.release()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        if written != frame_count:
            raise RuntimeError(f"wrote {written} frames, expected {frame_count}")
        record.update(
            {
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "inference_seconds": elapsed,
                "inference_fps": frame_count / elapsed if elapsed > 0 else 0.0,
            }
        )
    except Exception as exc:
        record["errors"].append(str(exc))
    finally:
        del group, raw_arr, norm_arr

    if record["errors"]:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        return record
    tmp_path.rename(out_path)
    return record


def create_array(group: Any, name: str, shape: tuple[int, int, int], dtype: Any, chunks: tuple[int, int, int]) -> Any:
    kwargs = {"shape": shape, "chunks": chunks, "dtype": dtype}
    try:
        import numcodecs

        kwargs["compressor"] = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    except Exception:
        pass
    try:
        return group.create_dataset(name, **kwargs)
    except Exception:
        kwargs.pop("compressor", None)
        return group.create_dataset(name, **kwargs)


def read_frame_batch(video_path: Path, count: int, start_index: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_index))
    frames = []
    try:
        while len(frames) < count:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


@torch.inference_mode()
def infer_batch(
    model: torch.nn.Module,
    frames_rgb: list[np.ndarray],
    process_res: int,
    process_res_method: str,
    target_size: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    imgs_cpu, _, _ = model.input_processor(
        frames_rgb,
        extrinsics=None,
        intrinsics=None,
        process_res=process_res,
        process_res_method=process_res_method,
        num_workers=1,
        sequential=True,
    )
    imgs = imgs_cpu[:, None].to(device, non_blocking=True).float()
    output = model.forward(
        imgs,
        None,
        None,
        export_feat_layers=[],
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    )
    depth = output["depth"]
    if depth.ndim == 4:
        depth = depth.reshape(depth.shape[0], 1, depth.shape[-2], depth.shape[-1])
    elif depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.permute(0, 1, 4, 2, 3).reshape(depth.shape[0], 1, depth.shape[2], depth.shape[3])
    else:
        raise RuntimeError(f"Unexpected depth shape: {tuple(depth.shape)}")
    depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=True)[:, 0]
    return depth.float().cpu().numpy().astype(np.float32, copy=False)


def normalize_per_frame_percentile(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32, copy=False)
    flat = depth.reshape(depth.shape[0], -1)
    lo = np.percentile(flat, 2.0, axis=1).astype(np.float32)
    hi = np.percentile(flat, 98.0, axis=1).astype(np.float32)
    denom = np.maximum(hi - lo, 1.0e-8)
    norm = (depth - lo[:, None, None]) / denom[:, None, None]
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


if __name__ == "__main__":
    main()
