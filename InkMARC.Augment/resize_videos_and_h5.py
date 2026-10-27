#!/usr/bin/env python3
"""
Resize stylus-tracking videos and transform matching H5 surface corners.

Examples:
  python resize_videos_and_h5.py \
    --input-dir "D:/InkMARC/raw" \
    --output-dir "D:/InkMARC/resized_432x672" \
    --target-width 432 \
    --target-height 672 \
    --resize-mode direct \
    --corner-dataset "/corners" \
    --input-origin top-left \
    --debug-overlay

  python resize_videos_and_h5.py \
    --input-dir "D:/InkMARC/raw" \
    --output-dir "D:/InkMARC/resized_672x672_letterbox" \
    --target-width 672 \
    --target-height 672 \
    --resize-mode letterbox \
    --corner-dataset "/corners" \
    --input-origin top-left \
    --debug-overlay

  python resize_videos_and_h5.py \
    --input-dir "D:/InkMARC/raw" \
    --output-dir "D:/InkMARC/resized_432x672" \
    --list-h5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
EXPECTED_SOURCE_WIDTH = 720
EXPECTED_SOURCE_HEIGHT = 1120
DEFAULT_CORNER_DATASET = "/corners"


@dataclass
class VideoH5Pair:
    stem: str
    video_path: Path
    h5_path: Path


@dataclass
class VideoMetadata:
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass
class ResizeTransform:
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    resized_width: int
    resized_height: int
    scale_x: float
    scale_y: float
    pad_left: float
    pad_top: float
    crop_left: float
    crop_top: float
    input_origin: str
    output_origin: str
    resize_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize videos and transform matching H5 surface corner coordinates."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-width", type=int, default=432)
    parser.add_argument("--target-height", type=int, default=672)
    parser.add_argument("--resize-mode", choices=("direct", "letterbox"), default="direct")
    parser.add_argument("--corner-dataset", default=DEFAULT_CORNER_DATASET)
    parser.add_argument("--input-origin", choices=("top-left", "top-right"), default="top-left")
    parser.add_argument("--output-origin", choices=("top-left", "top-right"), default="top-left")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument(
        "--list-h5",
        action="store_true",
        help="Print structure of the first matching H5 file and exit.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest output path. Default: OUTPUT_DIR/manifest.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output video/H5 files.",
    )
    parser.add_argument(
        "--aspect-tolerance",
        type=float,
        default=1e-3,
        help="Aspect-ratio tolerance for direct mode warnings.",
    )
    return parser.parse_args()


def normalize_h5_path(path: str) -> str:
    return "/" + path.strip("/")


def find_pairs(input_dir: Path) -> tuple[list[VideoH5Pair], list[str]]:
    warnings: list[str] = []
    h5_by_stem = {path.stem: path for path in sorted(input_dir.glob("*.h5")) if path.is_file()}
    videos_by_stem: dict[str, list[Path]] = {}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos_by_stem.setdefault(path.stem, []).append(path)

    pairs: list[VideoH5Pair] = []
    for stem, videos in sorted(videos_by_stem.items()):
        if len(videos) > 1:
            warnings.append(
                f"Multiple videos for stem {stem}; using {videos[0].name} and ignoring "
                f"{', '.join(p.name for p in videos[1:])}"
            )
        h5_path = h5_by_stem.get(stem)
        if h5_path is None:
            warnings.append(f"Missing H5 for video: {videos[0]}")
            continue
        pairs.append(VideoH5Pair(stem=stem, video_path=videos[0], h5_path=h5_path))

    for stem, h5_path in sorted(h5_by_stem.items()):
        if stem not in videos_by_stem:
            warnings.append(f"Missing video for H5: {h5_path}")

    return pairs, warnings


def read_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        return VideoMetadata(width=width, height=height, frame_count=frame_count, fps=fps)
    finally:
        cap.release()


def compute_resize_transform(
    metadata: VideoMetadata,
    target_width: int,
    target_height: int,
    resize_mode: str,
    input_origin: str,
    output_origin: str,
    aspect_tolerance: float,
) -> tuple[ResizeTransform, list[str]]:
    warnings: list[str] = []
    if metadata.width != EXPECTED_SOURCE_WIDTH or metadata.height != EXPECTED_SOURCE_HEIGHT:
        warnings.append(
            f"Source video is {metadata.width}x{metadata.height}, expected "
            f"{EXPECTED_SOURCE_WIDTH}x{EXPECTED_SOURCE_HEIGHT}"
        )

    if resize_mode == "direct":
        resized_width = target_width
        resized_height = target_height
        scale_x = target_width / float(metadata.width)
        scale_y = target_height / float(metadata.height)
        pad_left = 0.0
        pad_top = 0.0
        source_aspect = metadata.width / float(metadata.height)
        target_aspect = target_width / float(target_height)
        if abs(source_aspect - target_aspect) > aspect_tolerance:
            warnings.append(
                "Direct resize changes aspect ratio: "
                f"source={source_aspect:.8f}, target={target_aspect:.8f}"
            )
        if abs(scale_x - scale_y) > aspect_tolerance:
            warnings.append(
                f"Direct resize uses different x/y scales: scale_x={scale_x:.8f}, "
                f"scale_y={scale_y:.8f}"
            )
    else:
        scale = min(target_width / float(metadata.width), target_height / float(metadata.height))
        resized_width = max(1, int(round(metadata.width * scale)))
        resized_height = max(1, int(round(metadata.height * scale)))
        scale_x = resized_width / float(metadata.width)
        scale_y = resized_height / float(metadata.height)
        pad_left = (target_width - resized_width) / 2.0
        pad_top = (target_height - resized_height) / 2.0

    transform = ResizeTransform(
        source_width=metadata.width,
        source_height=metadata.height,
        output_width=target_width,
        output_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        scale_x=scale_x,
        scale_y=scale_y,
        pad_left=pad_left,
        pad_top=pad_top,
        crop_left=0.0,
        crop_top=0.0,
        input_origin=input_origin,
        output_origin=output_origin,
        resize_mode=resize_mode,
    )
    return transform, warnings


def points_top_right_to_top_left(points: np.ndarray, width: float) -> np.ndarray:
    converted = points.astype(np.float32, copy=True)
    converted[..., 0] = width - converted[..., 0]
    return converted


def points_top_left_to_top_right(points: np.ndarray, width: float) -> np.ndarray:
    converted = points.astype(np.float32, copy=True)
    converted[..., 0] = width - converted[..., 0]
    return converted


def transform_points(points: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    pts = points.astype(np.float32, copy=True)
    if transform.input_origin == "top-right":
        pts = points_top_right_to_top_left(pts, transform.source_width)

    pts[..., 0] = (pts[..., 0] - transform.crop_left) * transform.scale_x + transform.pad_left
    pts[..., 1] = (pts[..., 1] - transform.crop_top) * transform.scale_y + transform.pad_top

    if transform.output_origin == "top-right":
        pts = points_top_left_to_top_right(pts, transform.output_width)
    return pts.astype(np.float32, copy=False)


def corner_shape_warning(data: np.ndarray, dataset_path: str) -> str | None:
    if data.ndim == 2 and data.shape == (4, 2):
        return None
    if data.ndim >= 3 and data.shape[-2:] == (4, 2):
        return None
    return (
        f"Corner dataset {dataset_path} has shape {data.shape}; expected (4, 2) "
        "or (..., 4, 2)"
    )


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def dataset_create_kwargs(src_dataset: h5py.Dataset) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if src_dataset.chunks is not None:
        kwargs["chunks"] = src_dataset.chunks
    if src_dataset.compression is not None:
        kwargs["compression"] = src_dataset.compression
        kwargs["compression_opts"] = src_dataset.compression_opts
    if src_dataset.shuffle:
        kwargs["shuffle"] = True
    if src_dataset.fletcher32:
        kwargs["fletcher32"] = True
    if src_dataset.maxshape is not None:
        kwargs["maxshape"] = src_dataset.maxshape
    return kwargs


def write_transform_metadata(h5_file: h5py.File, transform: ResizeTransform) -> None:
    group = h5_file.require_group("preprocessing")
    resize_group = group.require_group("resize_transform")
    for key, value in asdict(transform).items():
        resize_group.attrs[key] = value


def copy_h5_recursively(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    current_path: str,
    corner_dataset: str,
    transform: ResizeTransform,
    warnings: list[str],
    transformed_flag: dict[str, bool],
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)
    for name, item in src_group.items():
        item_path = f"{current_path.rstrip('/')}/{name}" if current_path != "/" else f"/{name}"
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(name)
            copy_h5_recursively(
                item,
                child,
                item_path,
                corner_dataset,
                transform,
                warnings,
                transformed_flag,
            )
        elif isinstance(item, h5py.Dataset):
            data = item[()]
            if item_path == corner_dataset:
                shape_warning = corner_shape_warning(np.asarray(data), corner_dataset)
                if shape_warning:
                    warnings.append(shape_warning)
                    output_data = data
                else:
                    output_data = transform_points(np.asarray(data), transform)
                    transformed_flag["value"] = True
            else:
                output_data = data
            try:
                created = dst_group.create_dataset(
                    name,
                    data=output_data,
                    dtype=output_data.dtype if hasattr(output_data, "dtype") else item.dtype,
                    **dataset_create_kwargs(item),
                )
            except TypeError:
                created = dst_group.create_dataset(name, data=output_data)
            copy_attrs(item.attrs, created.attrs)
        else:
            warnings.append(f"Skipping unsupported H5 object at {item_path}: {type(item).__name__}")


def transform_h5(
    source_h5: Path,
    output_h5: Path,
    corner_dataset: str,
    transform: ResizeTransform,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    transformed_flag = {"value": False}
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = output_h5.with_name(f"{output_h5.stem}.tmp{output_h5.suffix}")
    if tmp_h5.exists():
        tmp_h5.unlink()

    with h5py.File(source_h5, "r") as src, h5py.File(tmp_h5, "w") as dst:
        copy_h5_recursively(
            src,
            dst,
            "/",
            corner_dataset,
            transform,
            warnings,
            transformed_flag,
        )
        if not transformed_flag["value"]:
            warnings.append(f"Corner dataset missing or not transformed: {corner_dataset}")
        write_transform_metadata(dst, transform)

    tmp_h5.replace(output_h5)
    return transformed_flag["value"], warnings


def fourcc_for_output(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    if suffix in {".mp4", ".mov", ".m4v"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    if suffix == ".mkv":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def resize_frame(frame: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    resized = cv2.resize(
        frame,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_AREA,
    )
    if transform.resize_mode == "direct":
        return resized

    output = np.zeros((transform.output_height, transform.output_width, 3), dtype=frame.dtype)
    x0 = int(round(transform.pad_left))
    y0 = int(round(transform.pad_top))
    output[y0 : y0 + transform.resized_height, x0 : x0 + transform.resized_width] = resized
    return output


def resize_video(
    source_video: Path,
    output_video: Path,
    metadata: VideoMetadata,
    transform: ResizeTransform,
) -> tuple[int, np.ndarray | None]:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = output_video.with_name(f"{output_video.stem}.tmp{output_video.suffix}")
    if tmp_video.exists():
        tmp_video.unlink()

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video}")

    writer = cv2.VideoWriter(
        str(tmp_video),
        fourcc_for_output(output_video),
        metadata.fps,
        (transform.output_width, transform.output_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {tmp_video}")

    frame_count = 0
    first_output_frame: np.ndarray | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_frame = resize_frame(frame, transform)
            if first_output_frame is None:
                first_output_frame = out_frame.copy()
            writer.write(out_frame)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    tmp_video.replace(output_video)
    return frame_count, first_output_frame


def first_corner_quad(source_h5: Path, corner_dataset: str, transform: ResizeTransform) -> np.ndarray | None:
    with h5py.File(source_h5, "r") as h5:
        if corner_dataset not in h5:
            return None
        data = np.asarray(h5[corner_dataset][()])
    if corner_shape_warning(data, corner_dataset):
        return None
    if data.shape == (4, 2):
        quad = data
    else:
        quad = data.reshape((-1, 4, 2))[0]
    return transform_points(quad, transform)


def draw_debug_overlay(
    output_png: Path,
    frame: np.ndarray,
    quad: np.ndarray | None,
    label: str,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    overlay = frame.copy()
    if quad is not None:
        pts = np.rint(quad).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for idx, point in enumerate(pts.reshape((-1, 2))):
            x, y = int(point[0]), int(point[1])
            cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        overlay,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_png), overlay)


def list_h5_structure(h5_path: Path) -> None:
    print(f"H5 structure: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            path = "/" + name
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET {path} shape={obj.shape} dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"GROUP   {path}")

        h5.visititems(visitor)


def manifest_transform(transform: ResizeTransform) -> dict[str, Any]:
    return asdict(transform)


def process_one_pair(
    pair: VideoH5Pair,
    args: argparse.Namespace,
    corner_dataset: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    output_video = args.output_dir / pair.video_path.name
    output_h5 = args.output_dir / pair.h5_path.name
    debug_png = args.output_dir / "debug_overlays" / f"{pair.stem}.png"

    try:
        metadata = read_video_metadata(pair.video_path)
        transform, transform_warnings = compute_resize_transform(
            metadata,
            args.target_width,
            args.target_height,
            args.resize_mode,
            args.input_origin,
            args.output_origin,
            args.aspect_tolerance,
        )
        warnings.extend(transform_warnings)

        item: dict[str, Any] = {
            "stem": pair.stem,
            "source_video_path": str(pair.video_path),
            "source_h5_path": str(pair.h5_path),
            "output_video_path": str(output_video),
            "output_h5_path": str(output_h5),
            "debug_overlay_path": str(debug_png) if args.debug_overlay else None,
            "original_dimensions": {"width": metadata.width, "height": metadata.height},
            "output_dimensions": {"width": transform.output_width, "height": transform.output_height},
            "resized_dimensions": {"width": transform.resized_width, "height": transform.resized_height},
            "frame_count": metadata.frame_count,
            "fps": metadata.fps,
            "transform_parameters": manifest_transform(transform),
            "warnings": warnings,
            "errors": errors,
            "status": "planned" if args.dry_run else "processed",
        }

        if args.dry_run:
            return item

        if not args.force and output_video.exists() and output_h5.exists():
            item["status"] = "skipped_existing"
            return item

        if args.force or not output_video.exists():
            written_count, first_output_frame = resize_video(pair.video_path, output_video, metadata, transform)
            item["written_frame_count"] = written_count
            if metadata.frame_count > 0 and written_count != metadata.frame_count:
                warnings.append(
                    f"Output frame count differs from input: input={metadata.frame_count}, output={written_count}"
                )
        else:
            first_output_frame = None
            item["written_frame_count"] = None
            warnings.append(f"Output video already exists and was not rewritten: {output_video}")

        if args.force or not output_h5.exists():
            _, h5_warnings = transform_h5(pair.h5_path, output_h5, corner_dataset, transform)
            warnings.extend(h5_warnings)
        else:
            warnings.append(f"Output H5 already exists and was not rewritten: {output_h5}")

        if args.debug_overlay:
            if first_output_frame is None:
                cap = cv2.VideoCapture(str(output_video))
                ok, first_output_frame = cap.read()
                cap.release()
                if not ok:
                    first_output_frame = None
            if first_output_frame is not None:
                quad = first_corner_quad(pair.h5_path, corner_dataset, transform)
                draw_debug_overlay(debug_png, first_output_frame, quad, pair.stem)
            else:
                warnings.append("Could not read first output frame for debug overlay")

        item["warnings"] = warnings
        return item
    except Exception as exc:
        errors.append(str(exc))
        return {
            "stem": pair.stem,
            "source_video_path": str(pair.video_path),
            "source_h5_path": str(pair.h5_path),
            "output_video_path": str(output_video),
            "output_h5_path": str(output_h5),
            "warnings": warnings,
            "errors": errors,
            "status": "failed",
        }


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    corner_dataset = normalize_h5_path(args.corner_dataset)

    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2

    pairs, pair_warnings = find_pairs(args.input_dir)
    if not pairs:
        print(f"No video/H5 pairs found in {args.input_dir}", file=sys.stderr)
        for warning in pair_warnings:
            print(f"[WARN] {warning}", file=sys.stderr)
        return 1

    if args.list_h5:
        list_h5_structure(pairs[0].h5_path)
        return 0

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, Any]] = []
    for warning in pair_warnings:
        print(f"[WARN] {warning}", file=sys.stderr)

    for pair in pairs:
        print(f"[PROCESS] {pair.video_path.name}", flush=True)
        item = process_one_pair(pair, args, corner_dataset)
        manifest_items.append(item)
        for warning in item.get("warnings", []):
            print(f"[WARN] {pair.stem}: {warning}", file=sys.stderr)
        for error in item.get("errors", []):
            print(f"[ERROR] {pair.stem}: {error}", file=sys.stderr)

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "corner_dataset": corner_dataset,
        "dry_run": bool(args.dry_run),
        "items": manifest_items,
    }

    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        write_manifest(manifest_path, manifest)
        print(f"[DONE] Wrote manifest: {manifest_path}")

    failed = sum(1 for item in manifest_items if item.get("status") == "failed")
    if failed:
        print(f"[DONE] Completed with {failed} failed item(s).", file=sys.stderr)
        return 1
    print(f"[DONE] Processed {len(manifest_items)} item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
