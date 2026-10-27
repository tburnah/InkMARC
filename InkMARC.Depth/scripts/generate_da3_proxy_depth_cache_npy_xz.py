#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SENSE_ROOT = ROOT.parent / "InkMARC.Sense"
# Shared dataset/configuration helpers; keep the Depth model independently packaged.
sys.path.insert(0, str(SENSE_ROOT))
sys.path.insert(0, str(ROOT))

from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DA3 teacher depth cache from raw XZ-compressed NPY image splits.")
    parser.add_argument("--config", default=str(ROOT / "configs/da3_depth_proxy_504x322.yaml"))
    parser.add_argument("--source-dir", default="/data/TouchData/npy_out/npy_merged")
    parser.add_argument("--da3-onnx", default=str(ROOT / "outputs/onnx/da3_small_depth_504x322.onnx"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/da3_proxy_depth_cache_504x322_npy_merged"))
    parser.add_argument("--source-split", action="append", default=None, help="Source split to use. Defaults to train/val/test.")
    parser.add_argument("--target-split", default="train", help="Cache split to place all source images into.")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    import onnxruntime as ort

    cfg = load_config(args.config, parse_overrides(args.override))
    repo_root = str(Path(getattr(cfg.model, "da3_repo_root", "/home/t/tburnah/src/depth_model/Depth-Anything-3")).expanduser().resolve() / "src")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sample_dir = output_dir / "samples"
    raw_dir = output_dir / "raw_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(Path(args.da3_onnx).expanduser().resolve()), sess_options=session_options, providers=[args.provider])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_shape = [int(dim) for dim in input_meta.shape]
    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    target_entries: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_split in args.source_split or ["train", "val", "test"]:
        labels = load_labels(source_dir / f"{source_split}_labels.npy.xz")
        source_count = int(labels.shape[0])
        total = source_count if args.max_samples_per_split <= 0 else min(source_count, int(args.max_samples_per_split))
        raw_count = total if args.max_samples_per_split > 0 else int(labels.shape[0])
        raw_path = ensure_raw_images(
            source_dir=source_dir,
            raw_dir=raw_dir,
            split=source_split,
            count=raw_count,
            source_count=source_count,
            image_size=int(args.image_size),
            channels=int(args.channels),
            resume=bool(args.resume),
        )
        image_shape = (raw_count, int(args.image_size), int(args.image_size), int(args.channels))
        images = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=image_shape)
        source_entries: list[dict[str, Any]] = []
        split_sample_dir = sample_dir / source_split
        split_sample_dir.mkdir(parents=True, exist_ok=True)
        for idx in tqdm(range(total), desc=f"cache npy {source_split}"):
            depth_path = split_sample_dir / f"{idx:08d}.npz"
            if not (args.resume and depth_path.exists()):
                frame = np.ascontiguousarray(images[idx])
                da3_input_np = fixed_da3_preprocess(frame, input_shape)
                if list(da3_input_np.shape) != input_shape:
                    raise ValueError(f"DA3 input shape mismatch: got {list(da3_input_np.shape)}, expected {input_shape}")
                depth = session.run([output_meta.name], {input_meta.name: da3_input_np})[0].astype(np.float32)
                if depth.ndim == 3:
                    depth = depth[:, None]
                np.savez_compressed(depth_path, depth=depth[0].astype(out_dtype, copy=False))
            entry = {
                "source": "npy_xz_raw",
                "source_split": source_split,
                "source_index": idx,
                "label": int(labels[idx]),
                "path": str(depth_path.relative_to(output_dir)),
                "image_raw_path": str(raw_path.relative_to(output_dir)),
                "image_raw_shape": list(image_shape),
                "image_index": idx,
            }
            source_entries.append(entry)
            target_entries.append(entry)
        source_summaries.append(
            {
                "source_split": source_split,
                "labels_path": str((source_dir / f"{source_split}_labels.npy.xz").resolve()),
                "images_path": str((source_dir / f"{source_split}_images.npy.xz").resolve()),
                "raw_images_path": str(raw_path),
                "source_count": int(labels.shape[0]),
                "cached_count": total,
            }
        )
        del images

    index = {
        "source": "npy_xz_raw",
        "source_dir": str(source_dir),
        "source_summaries": source_summaries,
        "da3_onnx": str(Path(args.da3_onnx).expanduser().resolve()),
        "provider": args.provider,
        "dtype": args.dtype,
        "input_name": input_meta.name,
        "input_shape": input_shape,
        "output_name": output_meta.name,
        "splits": {
            str(args.target_split): target_entries,
        },
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'index.json'}")


def load_labels(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as f:
        return np.load(f).astype(np.int32, copy=False)


def fixed_da3_preprocess(frame: np.ndarray, input_shape: list[int]) -> np.ndarray:
    import cv2

    if len(input_shape) != 5:
        raise ValueError(f"Expected DA3 input shape [1,1,3,H,W], got {input_shape}")
    _, _, channels, height, width = input_shape
    if channels != 3:
        raise ValueError(f"Expected 3-channel DA3 input, got {channels}")
    resized = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, None]
    return np.ascontiguousarray(x.astype(np.float32))


def ensure_raw_images(
    source_dir: Path,
    raw_dir: Path,
    split: str,
    count: int,
    source_count: int,
    image_size: int,
    channels: int,
    resume: bool,
) -> Path:
    name_count = str(count) if count == source_count else f"first{count}_of{source_count}"
    raw_path = raw_dir / f"{split}_images_{name_count}x{image_size}x{image_size}x{channels}.uint8.raw"
    expected_bytes = count * image_size * image_size * channels
    if resume and raw_path.exists() and raw_path.stat().st_size == expected_bytes:
        return raw_path
    source_path = source_dir / f"{split}_images.npy.xz"
    tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"decompressing {source_path} -> {raw_path}", flush=True)
    with lzma.open(source_path, "rb") as src, open(tmp_path, "wb") as dst:
        remaining = expected_bytes
        while remaining > 0:
            chunk = src.read(min(1024 * 1024 * 16, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    actual_bytes = tmp_path.stat().st_size
    if actual_bytes != expected_bytes:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Unexpected raw size for {source_path}: got {actual_bytes}, expected {expected_bytes}")
    tmp_path.rename(raw_path)
    return raw_path


if __name__ == "__main__":
    main()
