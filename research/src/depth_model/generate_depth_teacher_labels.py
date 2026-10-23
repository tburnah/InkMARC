#!/usr/bin/env python3
"""
Generate teacher pseudo-depth labels for resized stylus-tracking videos.

This script uses a local Depth-Anything-V2 checkout as an offline teacher. The
saved arrays are pseudo-depth labels for student-model training, not metric
ground truth. RGB/depth PNGs and preview videos are inspection aids only.

Examples:
    python generate_depth_teacher_labels.py \
        --input-dir "D:/InkMARC/resized_432x672" \
        --output-dir "D:/InkMARC/depth_teacher_vitl_432x672" \
        --output-format zarr \
        --depth-anything-path "D:/models/Depth-Anything-V2" \
        --checkpoint "D:/models/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth" \
        --encoder vitl \
        --input-width 432 \
        --input-height 672 \
        --dtype float16 \
        --normalization per-frame-percentile \
        --profile-system \
        --auto-config \
        --preview

    # CPU-safe dry run:
    python generate_depth_teacher_labels.py \
        --input-dir ./videos --output-dir ./depth_labels \
        --depth-anything-path ./Depth-Anything-V2 \
        --checkpoint ./depth_anything_v2_vits.pth \
        --encoder vits --device cpu --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

NORMALIZATION_MODES = (
    "none",
    "per-frame-minmax",
    "per-video-minmax",
    "per-frame-percentile",
    "per-video-percentile",
)


@dataclass
class SelectedConfig:
    encoder: str
    device: str
    batch_size: int
    amp: bool
    output_dtype: str
    teacher_input_size: int
    max_vram_fraction: float
    cpu_only: bool
    auto_config: bool


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, set[str]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = detect_explicit_args(argv)

    parser = argparse.ArgumentParser(
        description="Generate Depth-Anything-V2 teacher pseudo-depth labels for videos."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-anything-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vitl")
    parser.add_argument("--input-width", type=int, default=432)
    parser.add_argument("--input-height", type=int, default=672)
    parser.add_argument("--video-extensions", type=str, default=".mp4,.avi,.mov,.mkv")
    parser.add_argument("--allow-different-size", action="store_true")

    parser.add_argument("--output-format", choices=("zarr", "h5", "npy"), default="zarr")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--stream-frames", type=int, default=256)
    parser.add_argument("--npy-layout", choices=("stacked", "per-frame"), default="stacked")

    parser.add_argument("--normalization", choices=NORMALIZATION_MODES, default="per-frame-percentile")
    parser.add_argument("--percentile-low", type=float, default=2.0)
    parser.add_argument("--percentile-high", type=float, default=98.0)
    parser.add_argument("--smoothing", choices=("none", "ema", "median"), default="none")
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--median-window", type=int, default=3)

    parser.add_argument("--profile-system", action="store_true")
    parser.add_argument("--auto-config", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-vram-fraction", type=float, default=0.85)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--teacher-input-size", type=int, default=518)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-video", action="store_true")
    parser.add_argument("--preview-max-frames", type=int, default=300)

    args = parser.parse_args(argv)
    if args.sample_every < 1:
        parser.error("--sample-every must be >= 1")
    if args.chunk_frames < 1:
        parser.error("--chunk-frames must be >= 1")
    if args.stream_frames < 1:
        parser.error("--stream-frames must be >= 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not 0.0 <= args.percentile_low < args.percentile_high <= 100.0:
        parser.error("--percentile-low must be < --percentile-high, both in [0, 100]")
    if not 0.0 < args.ema_alpha <= 1.0:
        parser.error("--ema-alpha must be in (0, 1]")
    if args.median_window < 1 or args.median_window % 2 == 0:
        parser.error("--median-window must be a positive odd integer")
    if not 0.0 < args.max_vram_fraction <= 1.0:
        parser.error("--max-vram-fraction must be in (0, 1]")
    return args, explicit


def detect_explicit_args(argv: list[str]) -> set[str]:
    names = {
        "encoder": "--encoder",
        "device": "--device",
        "cpu_only": "--cpu-only",
        "batch_size": "--batch-size",
        "dtype": "--dtype",
        "no_amp": "--no-amp",
        "teacher_input_size": "--teacher-input-size",
    }
    explicit = set()
    for name, flag in names.items():
        if any(arg == flag or arg.startswith(flag + "=") for arg in argv):
            explicit.add(name)
    return explicit


def find_videos(input_dir: Path, extensions: str) -> list[Path]:
    suffixes = {ext.strip().lower() for ext in extensions.split(",") if ext.strip()}
    suffixes = {ext if ext.startswith(".") else f".{ext}" for ext in suffixes}
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def profile_system_resources() -> dict[str, Any]:
    profile: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_cores": os.cpu_count(),
        "ram_total_gb": "unknown",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": False,
        "gpus": [],
        "opencv_version": getattr(cv2, "__version__", "unknown"),
    }

    try:
        import psutil

        mem = psutil.virtual_memory()
        profile["ram_total_gb"] = round(mem.total / (1024**3), 2)
        profile["ram_available_gb"] = round(mem.available / (1024**3), 2)
    except Exception as exc:
        profile["ram_error"] = str(exc)

    try:
        profile["cuda_available"] = bool(torch.cuda.is_available())
        if profile["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpu = {
                    "index": index,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / (1024**3), 2),
                    "free_vram_gb": "unknown",
                }
                profile["gpus"].append(gpu)
    except Exception as exc:
        profile["cuda_error"] = str(exc)

    add_nvidia_smi_vram(profile)
    return profile


def add_nvidia_smi_vram(profile: dict[str, Any]) -> None:
    if not shutil.which("nvidia-smi"):
        profile["nvidia_smi"] = "not found"
        return
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        rows = []
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) == 4:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "free_vram_gb": round(float(parts[1]) / 1024.0, 2),
                        "total_vram_gb": round(float(parts[2]) / 1024.0, 2),
                        "name": parts[3],
                    }
                )
        profile["nvidia_smi"] = rows
        for gpu in profile.get("gpus", []):
            for row in rows:
                if row["index"] == gpu["index"]:
                    gpu["free_vram_gb"] = row["free_vram_gb"]
    except Exception as exc:
        profile["nvidia_smi_error"] = str(exc)


def choose_device(args: argparse.Namespace) -> str:
    if args.cpu_only or args.device == "cpu":
        return "cpu"
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: --device cuda requested but CUDA is unavailable; using CPU.")
            return "cpu"
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_auto_config(
    args: argparse.Namespace, explicit: set[str], system_profile: dict[str, Any]
) -> SelectedConfig:
    device = choose_device(args)
    usable_vram_gb = 0.0
    if device == "cuda" and system_profile.get("gpus"):
        first_gpu = system_profile["gpus"][0]
        total = first_gpu.get("total_vram_gb")
        free = first_gpu.get("free_vram_gb")
        base = free if isinstance(free, (int, float)) else total
        usable_vram_gb = float(base or 0.0) * args.max_vram_fraction

    encoder = args.encoder
    batch_size = args.batch_size if args.batch_size is not None else 1

    if args.auto_config:
        if "encoder" not in explicit:
            if device == "cpu" or usable_vram_gb < 6.0:
                encoder = "vits"
            elif usable_vram_gb < 12.0:
                encoder = "vitb"
            else:
                encoder = "vitl"

        if "batch_size" not in explicit:
            if device == "cpu":
                batch_size = 1
            elif encoder == "vitl":
                batch_size = 1
            elif encoder == "vitb":
                batch_size = 2 if usable_vram_gb < 16.0 else 4
            else:
                batch_size = 4 if usable_vram_gb < 12.0 else 8

    amp = device == "cuda" and not args.no_amp
    return SelectedConfig(
        encoder=encoder,
        device=device,
        batch_size=batch_size,
        amp=amp,
        output_dtype=args.dtype,
        teacher_input_size=args.teacher_input_size,
        max_vram_fraction=args.max_vram_fraction,
        cpu_only=bool(args.cpu_only),
        auto_config=bool(args.auto_config),
    )


def load_depth_anything_v2(
    depth_anything_path: Path, checkpoint: Path, encoder: str, device: str
) -> torch.nn.Module:
    repo_path = depth_anything_path.resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Depth-Anything-V2 repo not found: {repo_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(repo_path))

    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(torch.device(device)).eval()
    return model


def build_teacher_transform(depth_anything_path: Path, input_size: int):
    sys.path.insert(0, str(depth_anything_path.resolve()))
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose

    return Compose(
        [
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )


def read_frames(
    video_path: Path,
    input_width: int,
    input_height: int,
    allow_different_size: bool,
    limit_frames: int | None,
    sample_every: int,
) -> tuple[list[np.ndarray], dict[str, Any], list[str]]:
    warnings = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    declared_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if (width, height) != (input_width, input_height):
        message = (
            f"Video size is {width}x{height}; expected {input_width}x{input_height}."
        )
        if allow_different_size:
            warnings.append(message)
        else:
            cap.release()
            raise ValueError(message + " Pass --allow-different-size to process anyway.")

    frames = []
    frame_indices = []
    source_index = 0
    decoded_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded_count += 1
        if source_index % sample_every == 0:
            frames.append(frame)
            frame_indices.append(source_index)
            if limit_frames is not None and len(frames) >= limit_frames:
                break
        source_index += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames selected from video: {video_path}")
    if sample_every != 1:
        warnings.append(f"Processed every {sample_every} frame(s); output contains sampled frames only.")
    if limit_frames is not None:
        warnings.append(f"Stopped after --limit-frames={limit_frames}.")

    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "source_frame_count_reported": declared_count,
        "decoded_frame_count": decoded_count,
        "selected_frame_count": len(frames),
        "frame_indices": frame_indices,
    }
    return frames, metadata, warnings


def inspect_video(
    video_path: Path,
    input_width: int,
    input_height: int,
    allow_different_size: bool,
    limit_frames: int | None,
    sample_every: int,
) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    declared_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if (width, height) != (input_width, input_height):
        message = f"Video size is {width}x{height}; expected {input_width}x{input_height}."
        if allow_different_size:
            warnings.append(message)
        else:
            raise ValueError(message + " Pass --allow-different-size to process anyway.")

    if sample_every != 1:
        warnings.append(f"Processed every {sample_every} frame(s); output contains sampled frames only.")
    if limit_frames is not None:
        warnings.append(f"Stopped after --limit-frames={limit_frames}.")

    if declared_count > 0:
        selected_count = (declared_count + sample_every - 1) // sample_every
        if limit_frames is not None:
            selected_count = min(selected_count, limit_frames)
    else:
        selected_count = 0

    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "source_frame_count_reported": declared_count,
        "decoded_frame_count": None,
        "selected_frame_count": selected_count,
        "frame_indices": [],
    }
    return metadata, warnings


def iter_frame_chunks(
    video_path: Path,
    sample_every: int,
    limit_frames: int | None,
    chunk_size: int,
) -> Iterable[tuple[list[np.ndarray], list[int], int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    source_index = 0
    decoded_count = 0
    selected_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded_count += 1
            if source_index % sample_every == 0:
                frames.append(frame)
                frame_indices.append(source_index)
                selected_count += 1
                if len(frames) >= chunk_size:
                    yield frames, frame_indices, decoded_count
                    frames = []
                    frame_indices = []
                if limit_frames is not None and selected_count >= limit_frames:
                    break
            source_index += 1

        if frames:
            yield frames, frame_indices, decoded_count
    finally:
        cap.release()


def count_selected_video_frames(video_path: Path, sample_every: int, limit_frames: int | None) -> int:
    count = 0
    source_index = 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            if source_index % sample_every == 0:
                count += 1
                if limit_frames is not None and count >= limit_frames:
                    break
            source_index += 1
    finally:
        cap.release()
    return count


def preprocess_batch(frames: list[np.ndarray], transform: Any) -> torch.Tensor:
    tensors = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        tensor = transform({"image": rgb})["image"]
        tensors.append(torch.from_numpy(tensor))
    return torch.stack(tensors, dim=0)


def autocast_context(device: str, enabled: bool):
    if device == "cuda" and enabled:
        try:
            return torch.autocast(device_type="cuda", enabled=True)
        except AttributeError:
            return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def run_inference(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    transform: Any,
    device: str,
    batch_size: int,
    amp: bool,
    target_height: int,
    target_width: int,
    warmup_frames: int,
) -> tuple[np.ndarray, int, dict[str, float]]:
    current_batch_size = max(1, batch_size)
    while current_batch_size >= 1:
        try:
            timings: dict[str, float] = {}
            if warmup_frames > 0:
                warmup = frames[: min(warmup_frames, len(frames))]
                _ = infer_batches(
                    model,
                    warmup,
                    transform,
                    device,
                    min(current_batch_size, len(warmup)),
                    amp,
                    target_height,
                    target_width,
                    show_progress=False,
                    description="warmup",
                )
                if device == "cuda":
                    torch.cuda.synchronize()

            start = time.perf_counter()
            depths = infer_batches(
                model,
                frames,
                transform,
                device,
                current_batch_size,
                amp,
                target_height,
                target_width,
                show_progress=True,
                description="inference",
            )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings["inference_seconds"] = elapsed
            timings["inference_fps"] = len(frames) / elapsed if elapsed > 0 else 0.0
            return depths, current_batch_size, timings
        except RuntimeError as exc:
            if device == "cuda" and is_cuda_oom(exc):
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise RuntimeError(
                        "CUDA out of memory at batch size 1. Try --encoder vitb/vits, "
                        "--device cpu, a smaller --teacher-input-size, or freeing GPU memory."
                    ) from exc
                current_batch_size = max(1, current_batch_size // 2)
                print(f"WARNING: CUDA OOM; retrying with --batch-size {current_batch_size}.")
                continue
            raise

    raise RuntimeError("Inference failed unexpectedly.")


def infer_batches(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    transform: Any,
    device: str,
    batch_size: int,
    amp: bool,
    target_height: int,
    target_width: int,
    show_progress: bool,
    description: str,
) -> np.ndarray:
    outputs = []
    iterator: Iterable[int] = range(0, len(frames), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=description, total=(len(frames) + batch_size - 1) // batch_size)

    with torch.inference_mode():
        for start in iterator:
            batch_frames = frames[start : start + batch_size]
            batch = preprocess_batch(batch_frames, transform).to(device, non_blocking=True)
            with autocast_context(device, amp):
                pred = model(batch)
                pred = F.interpolate(
                    pred[:, None],
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
            outputs.append(pred.float().cpu().numpy())
            del batch, pred
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def is_cuda_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def infer_chunk_with_retry(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    transform: Any,
    device: str,
    batch_size: int,
    amp: bool,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, int]:
    current_batch_size = max(1, batch_size)
    while current_batch_size >= 1:
        try:
            return (
                infer_batches(
                    model=model,
                    frames=frames,
                    transform=transform,
                    device=device,
                    batch_size=current_batch_size,
                    amp=amp,
                    target_height=target_height,
                    target_width=target_width,
                    show_progress=False,
                    description="inference",
                ),
                current_batch_size,
            )
        except RuntimeError as exc:
            if device == "cuda" and is_cuda_oom(exc):
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise RuntimeError(
                        "CUDA out of memory at batch size 1. Try --encoder vitb/vits, "
                        "--device cpu, a smaller --teacher-input-size, or freeing GPU memory."
                    ) from exc
                current_batch_size = max(1, current_batch_size // 2)
                print(f"WARNING: CUDA OOM; retrying with --batch-size {current_batch_size}.")
                continue
            raise
    raise RuntimeError("Inference failed unexpectedly.")


def resize_depth_to_frame(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.shape[-2:] == (height, width):
        return depth
    if depth.ndim == 2:
        return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth],
        axis=0,
    )


def normalize_depth(
    depth: np.ndarray,
    mode: str,
    percentile_low: float,
    percentile_high: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    stats: dict[str, Any] = {
        "mode": mode,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
    }
    if mode == "none":
        return None, stats

    depth_f32 = depth.astype(np.float32, copy=False)
    norm = np.empty_like(depth_f32, dtype=np.float32)

    if mode == "per-frame-minmax":
        mins = depth_f32.min(axis=(1, 2))
        maxs = depth_f32.max(axis=(1, 2))
        for i, (lo, hi) in enumerate(zip(mins, maxs)):
            norm[i] = normalize_with_bounds(depth_f32[i], float(lo), float(hi))
        stats["values"] = {"min": mins.tolist(), "max": maxs.tolist()}
    elif mode == "per-video-minmax":
        lo = float(depth_f32.min())
        hi = float(depth_f32.max())
        norm[:] = normalize_with_bounds(depth_f32, lo, hi)
        stats["values"] = {"min": lo, "max": hi}
    elif mode == "per-frame-percentile":
        lows = np.percentile(depth_f32, percentile_low, axis=(1, 2))
        highs = np.percentile(depth_f32, percentile_high, axis=(1, 2))
        for i, (lo, hi) in enumerate(zip(lows, highs)):
            norm[i] = normalize_with_bounds(depth_f32[i], float(lo), float(hi))
        stats["values"] = {"low": lows.tolist(), "high": highs.tolist()}
    elif mode == "per-video-percentile":
        lo = float(np.percentile(depth_f32, percentile_low))
        hi = float(np.percentile(depth_f32, percentile_high))
        norm[:] = normalize_with_bounds(depth_f32, lo, hi)
        stats["values"] = {"low": lo, "high": hi}
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return norm, stats


def normalize_depth_stream_chunk(
    depth: np.ndarray,
    mode: str,
    percentile_low: float,
    percentile_high: float,
) -> tuple[np.ndarray | None, dict[str, list[float]]]:
    if mode == "none":
        return None, {}

    depth_f32 = depth.astype(np.float32, copy=False)
    norm = np.empty_like(depth_f32, dtype=np.float32)

    if mode == "per-frame-minmax":
        mins = depth_f32.min(axis=(1, 2))
        maxs = depth_f32.max(axis=(1, 2))
        for i, (lo, hi) in enumerate(zip(mins, maxs)):
            norm[i] = normalize_with_bounds(depth_f32[i], float(lo), float(hi))
        return norm, {"min": mins.tolist(), "max": maxs.tolist()}

    if mode == "per-frame-percentile":
        lows = np.percentile(depth_f32, percentile_low, axis=(1, 2))
        highs = np.percentile(depth_f32, percentile_high, axis=(1, 2))
        for i, (lo, hi) in enumerate(zip(lows, highs)):
            norm[i] = normalize_with_bounds(depth_f32[i], float(lo), float(hi))
        return norm, {"low": lows.tolist(), "high": highs.tolist()}

    raise ValueError(f"Streaming does not support normalization mode: {mode}")


def normalize_with_bounds(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = hi - lo
    if denom <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / denom, 0.0, 1.0).astype(np.float32, copy=False)


def smooth_depth(depth: np.ndarray, mode: str, ema_alpha: float, median_window: int) -> np.ndarray | None:
    if mode == "none":
        return None
    depth_f32 = depth.astype(np.float32, copy=False)
    if mode == "ema":
        smoothed = np.empty_like(depth_f32, dtype=np.float32)
        smoothed[0] = depth_f32[0]
        for i in range(1, depth_f32.shape[0]):
            smoothed[i] = ema_alpha * depth_f32[i] + (1.0 - ema_alpha) * smoothed[i - 1]
        return smoothed
    if mode == "median":
        radius = median_window // 2
        padded = np.pad(depth_f32, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        smoothed = np.empty_like(depth_f32, dtype=np.float32)
        for i in range(depth_f32.shape[0]):
            smoothed[i] = np.median(padded[i : i + median_window], axis=0)
        return smoothed
    raise ValueError(f"Unknown smoothing mode: {mode}")


def cast_depth(depth: np.ndarray | None, dtype: str) -> np.ndarray | None:
    if depth is None:
        return None
    return depth.astype(np.float16 if dtype == "float16" else np.float32, copy=False)


def output_path_for_video(output_dir: Path, video_path: Path, output_format: str) -> Path:
    if output_format == "zarr":
        return output_dir / f"{video_path.stem}.zarr"
    if output_format == "h5":
        return output_dir / f"{video_path.stem}.h5"
    return output_dir / f"{video_path.stem}_npy"


def write_output(
    output_path: Path,
    output_format: str,
    raw: np.ndarray,
    norm: np.ndarray | None,
    smoothed: np.ndarray | None,
    metadata: dict[str, Any],
    chunk_frames: int,
    npy_layout: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "zarr":
        write_zarr(output_path, raw, norm, smoothed, metadata, chunk_frames)
    elif output_format == "h5":
        write_h5(output_path, raw, norm, smoothed, metadata, chunk_frames)
    elif output_format == "npy":
        write_npy(output_path, raw, norm, smoothed, metadata, npy_layout)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def write_zarr(
    output_path: Path,
    raw: np.ndarray,
    norm: np.ndarray | None,
    smoothed: np.ndarray | None,
    metadata: dict[str, Any],
    chunk_frames: int,
) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr output requested, but zarr is not installed. Install zarr or use --output-format h5.") from exc

    if output_path.exists():
        remove_existing_output(output_path)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs.update(json_safe(metadata))
    create_zarr_dataset(group, "depth_raw", raw, chunk_frames)
    if norm is not None:
        create_zarr_dataset(group, "depth_norm", norm, chunk_frames)
    if smoothed is not None:
        create_zarr_dataset(group, "depth_smoothed", smoothed, chunk_frames)


def open_zarr_for_stream(
    output_path: Path,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    metadata: dict[str, Any],
    chunk_frames: int,
    write_norm: bool,
) -> tuple[Any, Any, Any | None]:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr output requested, but zarr is not installed. Install zarr or use --output-format h5.") from exc

    if output_path.exists():
        remove_existing_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs.update(json_safe(metadata))
    raw_array = create_empty_zarr_dataset(group, "depth_raw", shape, dtype, chunk_frames)
    norm_array = None
    if write_norm:
        norm_array = create_empty_zarr_dataset(group, "depth_norm", shape, dtype, chunk_frames)
    return group, raw_array, norm_array


def create_zarr_dataset(group: Any, name: str, data: np.ndarray, chunk_frames: int) -> None:
    array = create_empty_zarr_dataset(group, name, data.shape, data.dtype, chunk_frames)
    array[:] = data


def create_empty_zarr_dataset(
    group: Any,
    name: str,
    shape: tuple[int, int, int] | list[int],
    dtype: np.dtype,
    chunk_frames: int,
) -> Any:
    chunks = (min(chunk_frames, int(shape[0])), int(shape[1]), int(shape[2]))
    compressor = None
    try:
        import numcodecs

        compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    except Exception:
        compressor = None

    kwargs = {"shape": tuple(shape), "chunks": chunks, "dtype": dtype}
    if compressor is not None:
        kwargs["compressor"] = compressor
    try:
        array = group.create_dataset(name, **kwargs)
    except Exception:
        kwargs.pop("compressor", None)
        array = group.create_dataset(name, **kwargs)
    return array


def write_h5(
    output_path: Path,
    raw: np.ndarray,
    norm: np.ndarray | None,
    smoothed: np.ndarray | None,
    metadata: dict[str, Any],
    chunk_frames: int,
) -> None:
    chunks = (min(chunk_frames, raw.shape[0]), raw.shape[1], raw.shape[2])
    with h5py.File(output_path, "w") as handle:
        handle.attrs["metadata_json"] = json.dumps(json_safe(metadata), indent=2)
        handle.attrs["training_warning"] = (
            "Open H5 files lazily inside each PyTorch DataLoader worker; do not share open handles across workers."
        )
        handle.create_dataset(
            "depth_raw",
            data=raw,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        if norm is not None:
            handle.create_dataset(
                "depth_norm",
                data=norm,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        if smoothed is not None:
            handle.create_dataset(
                "depth_smoothed",
                data=smoothed,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )


def write_npy(
    output_path: Path,
    raw: np.ndarray,
    norm: np.ndarray | None,
    smoothed: np.ndarray | None,
    metadata: dict[str, Any],
    npy_layout: str,
) -> None:
    if output_path.exists():
        remove_existing_output(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if npy_layout == "stacked":
        np.save(output_path / "depth_raw.npy", raw)
        if norm is not None:
            np.save(output_path / "depth_norm.npy", norm)
        if smoothed is not None:
            np.save(output_path / "depth_smoothed.npy", smoothed)
    else:
        write_npy_frames(output_path / "depth_raw", raw)
        if norm is not None:
            write_npy_frames(output_path / "depth_norm", norm)
        if smoothed is not None:
            write_npy_frames(output_path / "depth_smoothed", smoothed)
    (output_path / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")


def write_npy_frames(directory: Path, data: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(data):
        np.save(directory / f"{index:06d}.npy", frame)


def remove_existing_output(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def generate_previews(
    preview_dir: Path,
    video_path: Path,
    frames: list[np.ndarray],
    depth: np.ndarray,
    fps: float,
    write_video: bool,
    max_frames: int,
) -> dict[str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    first_rgb_path = preview_dir / f"{video_path.stem}_first_rgb.png"
    first_depth_path = preview_dir / f"{video_path.stem}_first_depth_color.png"
    first_side_path = preview_dir / f"{video_path.stem}_first_side_by_side.png"

    preview_frames = [resize_frame_for_preview(frame, depth.shape[2], depth.shape[1]) for frame in frames]
    first_color = colorize_depth(depth[0])
    cv2.imwrite(str(first_rgb_path), preview_frames[0])
    cv2.imwrite(str(first_depth_path), first_color)
    cv2.imwrite(str(first_side_path), np.concatenate([preview_frames[0], first_color], axis=1))

    result = {
        "first_rgb": str(first_rgb_path),
        "first_depth_color": str(first_depth_path),
        "first_side_by_side": str(first_side_path),
        "note": "Preview PNG/video files are for inspection only, not training labels.",
    }
    if write_video:
        preview_video_path = preview_dir / f"{video_path.stem}_preview.mp4"
        height, width = preview_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(preview_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width * 2, height),
        )
        for frame, depth_frame in zip(preview_frames[:max_frames], depth[:max_frames]):
            writer.write(np.concatenate([frame, colorize_depth(depth_frame)], axis=1))
        writer.release()
        result["preview_video"] = str(preview_video_path)
    return result


def resize_frame_for_preview(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth_f32 = depth.astype(np.float32, copy=False)
    lo = float(np.percentile(depth_f32, 2.0))
    hi = float(np.percentile(depth_f32, 98.0))
    normalized = normalize_with_bounds(depth_f32, lo, hi)
    u8 = (normalized * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def validate_saved_arrays(
    raw: np.ndarray,
    norm: np.ndarray | None,
    smoothed: np.ndarray | None,
    expected_frames: int,
    expected_height: int,
    expected_width: int,
) -> list[str]:
    warnings = []
    expected_shape = (expected_frames, expected_height, expected_width)
    for name, array in (("depth_raw", raw), ("depth_norm", norm), ("depth_smoothed", smoothed)):
        if array is None:
            continue
        if array.shape != expected_shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {expected_shape}.")
    if raw.shape[0] != expected_frames:
        warnings.append(f"Output frame count {raw.shape[0]} differs from selected frame count {expected_frames}.")
    return warnings


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def print_system_profile(profile: dict[str, Any]) -> None:
    print(json.dumps(json_safe(profile), indent=2))


def print_selected_config(config: SelectedConfig) -> None:
    print("Selected configuration:")
    print(json.dumps(asdict(config), indent=2))


def maybe_warn_missing_h5(video_path: Path) -> list[str]:
    candidates = [video_path.with_suffix(".h5"), video_path.with_suffix(".hdf5")]
    if not any(candidate.exists() for candidate in candidates):
        return [f"Matching H5 sidecar not found for {video_path.name}; continuing because it is not required."]
    return []


def can_stream_video(args: argparse.Namespace) -> bool:
    return (
        args.output_format == "zarr"
        and args.smoothing == "none"
        and args.normalization in {"none", "per-frame-minmax", "per-frame-percentile"}
    )


def process_video_streaming_zarr(
    video_path: Path,
    args: argparse.Namespace,
    config: SelectedConfig,
    model: torch.nn.Module,
    transform: Any,
    system_profile: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_video": str(video_path),
        "encoder": config.encoder,
        "checkpoint": str(args.checkpoint),
        "dtype": config.output_dtype,
        "normalization": args.normalization,
        "smoothing": args.smoothing,
        "warnings": [],
        "errors": [],
        "system_profile": system_profile,
        "streaming": True,
    }
    start = time.perf_counter()
    output_path = output_path_for_video(args.output_dir, video_path, args.output_format)
    temp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    record["output_depth"] = str(output_path)

    try:
        record["warnings"].extend(maybe_warn_missing_h5(video_path))
        if args.resume and output_path.exists():
            record["skipped"] = True
            record["warnings"].append("Skipped because output exists and --resume was supplied.")
            return record
        if temp_output_path.exists():
            remove_existing_output(temp_output_path)

        video_meta, read_warnings = inspect_video(
            video_path=video_path,
            input_width=args.input_width,
            input_height=args.input_height,
            allow_different_size=args.allow_different_size,
            limit_frames=args.limit_frames,
            sample_every=args.sample_every,
        )
        selected_count = int(video_meta["selected_frame_count"] or 0)
        if selected_count <= 0:
            selected_count = count_selected_video_frames(video_path, args.sample_every, args.limit_frames)
            video_meta["selected_frame_count"] = selected_count
        if selected_count <= 0:
            raise RuntimeError(f"No frames selected from video: {video_path}")

        record.update(
            {
                "frame_count": selected_count,
                "fps": video_meta["fps"],
                "width": video_meta["width"],
                "height": video_meta["height"],
                "video_metadata": video_meta,
            }
        )
        record["warnings"].extend(read_warnings)

        norm_stats: dict[str, Any] = {
            "mode": args.normalization,
            "percentile_low": args.percentile_low,
            "percentile_high": args.percentile_high,
        }
        if args.normalization == "per-frame-minmax":
            norm_stats["values"] = {"min": [], "max": []}
        elif args.normalization == "per-frame-percentile":
            norm_stats["values"] = {"low": [], "high": []}

        output_shape = (selected_count, args.input_height, args.input_width)
        output_metadata = {
            "source_video": str(video_path),
            "checkpoint": str(args.checkpoint),
            "encoder": config.encoder,
            "output_format": args.output_format,
            "dtype": config.output_dtype,
            "shape": list(output_shape),
            "input_width": args.input_width,
            "input_height": args.input_height,
            "teacher_input_size": config.teacher_input_size,
            "normalization": norm_stats,
            "smoothing": {
                "mode": args.smoothing,
                "ema_alpha": args.ema_alpha,
                "median_window": args.median_window,
            },
            "chunk_frames": args.chunk_frames,
            "stream_frames": args.stream_frames,
            "selected_config": asdict(config),
            "video_metadata": video_meta,
            "warnings": record["warnings"],
        }

        output_dtype = np.dtype(np.float16 if config.output_dtype == "float16" else np.float32)
        group, raw_array, norm_array = open_zarr_for_stream(
            output_path=temp_output_path,
            shape=output_shape,
            dtype=output_dtype,
            metadata=output_metadata,
            chunk_frames=args.chunk_frames,
            write_norm=args.normalization != "none",
        )

        preview_frames: list[np.ndarray] = []
        preview_depths: list[np.ndarray] = []
        frame_indices: list[int] = []
        decoded_count = 0
        written = 0
        used_batch_size = config.batch_size

        if args.warmup_frames > 0:
            warmup_frames = []
            for frames, _, decoded_count in iter_frame_chunks(
                video_path, args.sample_every, args.limit_frames, args.warmup_frames
            ):
                warmup_frames = frames
                break
            if warmup_frames:
                _ = infer_chunk_with_retry(
                    model=model,
                    frames=warmup_frames,
                    transform=transform,
                    device=config.device,
                    batch_size=config.batch_size,
                    amp=config.amp,
                    target_height=args.input_height,
                    target_width=args.input_width,
                )
                if config.device == "cuda":
                    torch.cuda.synchronize()

        inference_start = time.perf_counter()
        chunk_iter = iter_frame_chunks(
            video_path=video_path,
            sample_every=args.sample_every,
            limit_frames=args.limit_frames,
            chunk_size=args.stream_frames,
        )
        with tqdm(desc="inference", total=selected_count, unit="frame") as progress:
            for frames, chunk_indices, decoded_count in chunk_iter:
                raw_depth, used_batch_size = infer_chunk_with_retry(
                    model=model,
                    frames=frames,
                    transform=transform,
                    device=config.device,
                    batch_size=used_batch_size,
                    amp=config.amp,
                    target_height=args.input_height,
                    target_width=args.input_width,
                )
                raw_depth = resize_depth_to_frame(raw_depth, args.input_height, args.input_width)

                norm_depth, chunk_stats = normalize_depth_stream_chunk(
                    raw_depth,
                    mode=args.normalization,
                    percentile_low=args.percentile_low,
                    percentile_high=args.percentile_high,
                )
                if "values" in norm_stats:
                    for key, values in chunk_stats.items():
                        norm_stats["values"][key].extend(values)

                end = written + raw_depth.shape[0]
                if end > selected_count:
                    raise RuntimeError(
                        f"Decoded more selected frames than expected for {video_path}: {end} > {selected_count}."
                    )
                raw_array[written:end] = cast_depth(raw_depth, config.output_dtype)
                if norm_array is not None and norm_depth is not None:
                    norm_array[written:end] = cast_depth(norm_depth, config.output_dtype)

                if args.preview and len(preview_depths) < args.preview_max_frames:
                    remaining = args.preview_max_frames - len(preview_depths)
                    preview_frames.extend(frames[:remaining])
                    preview_depths.extend(raw_depth[:remaining])

                frame_indices.extend(chunk_indices)
                written = end
                progress.update(len(frames))
                del raw_depth, norm_depth, frames

        if config.device == "cuda":
            torch.cuda.synchronize()
        inference_elapsed = time.perf_counter() - inference_start
        if written != selected_count:
            raise RuntimeError(
                f"Decoded {written} selected frames from {video_path}; expected {selected_count}."
            )

        video_meta["decoded_frame_count"] = decoded_count
        video_meta["selected_frame_count"] = written
        video_meta["frame_indices"] = frame_indices
        output_metadata["shape"] = [written, args.input_height, args.input_width]
        output_metadata["normalization"] = norm_stats
        output_metadata["video_metadata"] = video_meta
        output_metadata["warnings"] = record["warnings"]
        group.attrs.update(json_safe(output_metadata))
        del group, raw_array, norm_array

        if output_path.exists():
            remove_existing_output(output_path)
        temp_output_path.rename(output_path)

        record["used_batch_size"] = used_batch_size
        record["inference_seconds"] = inference_elapsed
        record["inference_fps"] = written / inference_elapsed if inference_elapsed > 0 else 0.0

        if args.preview and preview_frames and preview_depths:
            preview_info = generate_previews(
                preview_dir=args.output_dir / "previews",
                video_path=video_path,
                frames=preview_frames,
                depth=np.stack(preview_depths, axis=0),
                fps=video_meta["fps"],
                write_video=args.preview_video,
                max_frames=args.preview_max_frames,
            )
            record["previews"] = preview_info

        elapsed = time.perf_counter() - start
        record["processing_time_seconds"] = elapsed
        record["overall_fps"] = written / elapsed if elapsed > 0 else 0.0
        record["video_metadata"] = video_meta
        record["frame_count"] = written
    except Exception as exc:
        if temp_output_path.exists():
            remove_existing_output(temp_output_path)
        record["errors"].append(str(exc))
        record["processing_time_seconds"] = time.perf_counter() - start
    return record


def process_video(
    video_path: Path,
    args: argparse.Namespace,
    config: SelectedConfig,
    model: torch.nn.Module,
    transform: Any,
    system_profile: dict[str, Any],
) -> dict[str, Any]:
    if can_stream_video(args):
        return process_video_streaming_zarr(
            video_path=video_path,
            args=args,
            config=config,
            model=model,
            transform=transform,
            system_profile=system_profile,
        )

    record: dict[str, Any] = {
        "source_video": str(video_path),
        "encoder": config.encoder,
        "checkpoint": str(args.checkpoint),
        "dtype": config.output_dtype,
        "normalization": args.normalization,
        "smoothing": args.smoothing,
        "warnings": [],
        "errors": [],
        "system_profile": system_profile,
    }
    start = time.perf_counter()
    output_path = output_path_for_video(args.output_dir, video_path, args.output_format)
    record["output_depth"] = str(output_path)

    try:
        record["warnings"].extend(maybe_warn_missing_h5(video_path))
        if args.resume and output_path.exists():
            record["skipped"] = True
            record["warnings"].append("Skipped because output exists and --resume was supplied.")
            return record

        frames, video_meta, read_warnings = read_frames(
            video_path=video_path,
            input_width=args.input_width,
            input_height=args.input_height,
            allow_different_size=args.allow_different_size,
            limit_frames=args.limit_frames,
            sample_every=args.sample_every,
        )
        record.update(
            {
                "frame_count": len(frames),
                "fps": video_meta["fps"],
                "width": video_meta["width"],
                "height": video_meta["height"],
                "video_metadata": video_meta,
            }
        )
        record["warnings"].extend(read_warnings)

        raw_depth, used_batch_size, timing = run_inference(
            model=model,
            frames=frames,
            transform=transform,
            device=config.device,
            batch_size=config.batch_size,
            amp=config.amp,
            target_height=args.input_height,
            target_width=args.input_width,
            warmup_frames=args.warmup_frames,
        )
        raw_depth = resize_depth_to_frame(raw_depth, args.input_height, args.input_width)
        record["used_batch_size"] = used_batch_size
        record.update(timing)

        norm_depth, norm_stats = normalize_depth(
            raw_depth,
            mode=args.normalization,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )
        smoothed_depth = smooth_depth(
            raw_depth,
            mode=args.smoothing,
            ema_alpha=args.ema_alpha,
            median_window=args.median_window,
        )

        raw_out = cast_depth(raw_depth, config.output_dtype)
        norm_out = cast_depth(norm_depth, config.output_dtype)
        smoothed_out = cast_depth(smoothed_depth, config.output_dtype)
        record["warnings"].extend(
            validate_saved_arrays(
                raw_out,
                norm_out,
                smoothed_out,
                expected_frames=len(frames),
                expected_height=args.input_height,
                expected_width=args.input_width,
            )
        )

        output_metadata = {
            "source_video": str(video_path),
            "checkpoint": str(args.checkpoint),
            "encoder": config.encoder,
            "output_format": args.output_format,
            "dtype": config.output_dtype,
            "shape": list(raw_out.shape),
            "input_width": args.input_width,
            "input_height": args.input_height,
            "teacher_input_size": config.teacher_input_size,
            "normalization": norm_stats,
            "smoothing": {
                "mode": args.smoothing,
                "ema_alpha": args.ema_alpha,
                "median_window": args.median_window,
            },
            "chunk_frames": args.chunk_frames,
            "selected_config": asdict(config),
            "video_metadata": video_meta,
            "warnings": record["warnings"],
        }
        write_output(
            output_path=output_path,
            output_format=args.output_format,
            raw=raw_out,
            norm=norm_out,
            smoothed=smoothed_out,
            metadata=output_metadata,
            chunk_frames=args.chunk_frames,
            npy_layout=args.npy_layout,
        )

        if args.preview:
            preview_info = generate_previews(
                preview_dir=args.output_dir / "previews",
                video_path=video_path,
                frames=frames,
                depth=raw_depth,
                fps=video_meta["fps"],
                write_video=args.preview_video,
                max_frames=args.preview_max_frames,
            )
            record["previews"] = preview_info

        elapsed = time.perf_counter() - start
        record["processing_time_seconds"] = elapsed
        record["overall_fps"] = len(frames) / elapsed if elapsed > 0 else 0.0
    except Exception as exc:
        record["errors"].append(str(exc))
        record["processing_time_seconds"] = time.perf_counter() - start
    return record


def dry_run(args: argparse.Namespace, videos: list[Path], config: SelectedConfig, profile: dict[str, Any]) -> None:
    print_selected_config(config)
    print(f"Found {len(videos)} video(s).")
    for video in videos:
        output_path = output_path_for_video(args.output_dir, video, args.output_format)
        try:
            cap = cv2.VideoCapture(str(video))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            status = "ok" if (width, height) == (args.input_width, args.input_height) else "size mismatch"
            print(f"{video} -> {output_path} [{width}x{height}, {frame_count} frames, {fps:.3f} fps, {status}]")
        except Exception as exc:
            print(f"{video} -> {output_path} [could not inspect: {exc}]")
    manifest = {
        "dry_run": True,
        "selected_config": asdict(config),
        "system_profile": profile,
        "video_count": len(videos),
        "videos": [str(video) for video in videos],
    }
    write_manifest(args.output_dir, manifest)


def main(argv: list[str] | None = None) -> int:
    args, explicit = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_profile = profile_system_resources()
    config = choose_auto_config(args, explicit, system_profile)
    system_profile["selected_device"] = config.device
    system_profile["selected_batch_size"] = config.batch_size
    system_profile["amp"] = config.amp

    if args.profile_system:
        print_system_profile(system_profile)
    print_selected_config(config)

    videos = find_videos(args.input_dir, args.video_extensions)
    if not videos:
        print(f"No videos found under {args.input_dir} with extensions {args.video_extensions}.")
        return 1

    if args.dry_run:
        dry_run(args, videos, config, system_profile)
        return 0

    model = load_depth_anything_v2(
        depth_anything_path=args.depth_anything_path,
        checkpoint=args.checkpoint,
        encoder=config.encoder,
        device=config.device,
    )
    transform = build_teacher_transform(args.depth_anything_path, config.teacher_input_size)

    records = []
    for video in tqdm(videos, desc="videos"):
        record = process_video(
            video_path=video,
            args=args,
            config=config,
            model=model,
            transform=transform,
            system_profile=system_profile,
        )
        records.append(record)
        if record.get("errors"):
            print(f"ERROR processing {video}: {record['errors'][-1]}")

    manifest = {
        "created_at_unix": time.time(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "output_format": args.output_format,
        "selected_config": asdict(config),
        "system_profile": system_profile,
        "normalization_modes": [args.normalization],
        "smoothing_mode": args.smoothing,
        "videos": records,
        "summary": {
            "total_videos": len(records),
            "succeeded": sum(1 for record in records if not record.get("errors") and not record.get("skipped")),
            "skipped": sum(1 for record in records if record.get("skipped")),
            "failed": sum(1 for record in records if record.get("errors")),
        },
    }
    manifest_path = write_manifest(args.output_dir, manifest)
    print(f"Wrote manifest: {manifest_path}")
    return 0 if manifest["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
