#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2  # type: ignore
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.transforms.homography_warp import warp_to_canonical
from data.dataset_core import InkMARCIndexedDataset


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_homography(H: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    pt = np.array([[xy]], dtype=np.float64)  # (1,1,2)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def _fourcc_for_path(path: Path, codec: str | None) -> int:
    if codec:
        return cv2.VideoWriter_fourcc(*codec)
    if path.suffix.lower() == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter_fourcc(*"MJPG")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/LocationData")
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--index-sample",
        type=int,
        default=0,
        help="Index into the split (use to anchor file_id/frame_idx).",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="If set, use --file-id and --start instead of index.",
    )
    ap.add_argument("--file-id", default=None, help="Manifest file_id to visualize (no-index mode).")
    ap.add_argument("--start", type=int, default=0, help="Start frame (no-index mode).")
    ap.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Offset from the anchor frame when using index mode.",
    )
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--split", default="val", help="Split used to pick the anchor sample.")
    ap.add_argument(
        "--variant-index",
        type=int,
        default=0,
        help="Video variant index to use for all frames (keeps color consistent).",
    )
    ap.add_argument("--out", default="runs/phase1_sequence.mp4")
    ap.add_argument("--codec", default=None, help="FourCC codec (e.g., mp4v, MJPG, XVID)")
    ap.add_argument("--save-frames-dir", default=None, help="If set, save each frame as PNG in this dir.")
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--profile", action="store_true", help="Print timing breakdown.")
    seq_group = ap.add_mutually_exclusive_group()
    seq_group.add_argument(
        "--sequential-read",
        action="store_true",
        help="Avoid per-frame seek and read frames sequentially (default).",
    )
    seq_group.add_argument(
        "--no-sequential-read",
        action="store_true",
        help="Disable sequential read optimization (forces seek each frame).",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_path = dataset_root / "indices" / f"manifest_{args.version}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest['files'] missing/empty")

    if args.no_index:
        file_id = args.file_id or next(iter(files.keys()))
        entry = files.get(str(file_id))
        if not isinstance(entry, dict):
            raise KeyError(f"file_id '{file_id}' not found in manifest['files']")

        h5_rel = entry.get("h5_rel_path") or entry.get("rel_path")
        if not h5_rel:
            raise KeyError(f"file_id '{file_id}' missing h5_rel_path/rel_path in manifest")

        h5_path = dataset_root / str(h5_rel)
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 missing: {h5_path}")

        base_frame = int(args.start)
    else:
        ds = InkMARCIndexedDataset(
            dataset_root=dataset_root,
            index_version=args.version,
            split=args.split,
            require_labels=True,
            transform=None,
        )
        idx = max(0, min(int(args.index_sample), len(ds) - 1))
        sample = ds[idx]
        meta = sample.get("meta", {})
        file_id = meta.get("file_id")
        if file_id is None:
            raise KeyError("index sample missing meta.file_id")
        entry = files.get(str(file_id))
        if not isinstance(entry, dict):
            raise KeyError(f"file_id '{file_id}' not found in manifest['files']")

        h5_rel = meta.get("rel_path") or entry.get("h5_rel_path") or entry.get("rel_path")
        if not h5_rel:
            raise KeyError(f"file_id '{file_id}' missing h5_rel_path/rel_path in manifest")

        h5_path = dataset_root / str(h5_rel)
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 missing: {h5_path}")

        base_frame = int(meta.get("frame_idx", 0)) + int(args.start_offset)
        ds.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.save_frames_dir) if args.save_frames_dir else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        _fourcc_for_path(out_path, args.codec),
        float(args.fps),
        (args.out_w, args.out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    variants = entry.get("video_variants")
    if not isinstance(variants, list) or len(variants) == 0:
        raise KeyError(f"file_id '{file_id}' missing video_variants list in manifest")
    v_idx = int(args.variant_index)
    if v_idx < 0 or v_idx >= len(variants):
        raise ValueError(f"variant-index {v_idx} out of range (n={len(variants)})")
    variant_rel = str(variants[v_idx])
    variant_path = dataset_root / variant_rel
    if not variant_path.exists():
        raise FileNotFoundError(f"Video variant missing: {variant_path}")

    cap = cv2.VideoCapture(str(variant_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {variant_path}")

    try:
        with h5py.File(h5_path, "r") as h5:
            corners_ds = h5["corners"]
            labels_ds = h5["labels"]
            mask_ds = h5["label_mask"]

            n = int(corners_ds.shape[0])
            start = max(0, int(base_frame))
            end = min(n, start + int(args.count))
            if start >= n:
                raise ValueError(f"start {start} out of range (n={n})")

            t_video_read = 0.0
            t_warp = 0.0
            t_label = 0.0
            t_write = 0.0

            use_seq = not args.no_sequential_read
            if use_seq:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    raise RuntimeError(f"Failed to read frame {start} from {variant_rel}")
                prev_frame_idx = start

            for frame_idx in range(start, end):
                t0 = time.perf_counter()
                corners = np.asarray(corners_ds[frame_idx], dtype=np.float32)
                labels = np.asarray(labels_ds[frame_idx], dtype=np.float32)
                label_mask = np.asarray(mask_ds[frame_idx], dtype=np.uint8)

                if use_seq:
                    if frame_idx != prev_frame_idx:
                        ret, frame_bgr = cap.read()
                        if not ret or frame_bgr is None:
                            raise RuntimeError(f"Failed to read frame {frame_idx} from {variant_rel}")
                        prev_frame_idx += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        raise RuntimeError(f"Failed to read frame {frame_idx} from {variant_rel}")
                t1 = time.perf_counter()
                warp = warp_to_canonical(
                    frame_bgr,
                    corners,
                    out_w=args.out_w,
                    out_h=args.out_h,
                    margin_frac=args.margin_frac,
                )
                img = warp["img_surf_bgr"].copy()
                t2 = time.perf_counter()

                touch = float(np.clip(labels[0], 0.0, 1.0))
                xy_valid = (
                    touch >= 0.5
                    and float(label_mask[1]) > 0.0
                    and float(label_mask[2]) > 0.0
                )
                if xy_valid:
                    crop_origin = warp["crop_origin"]
                    H = warp["H_crop_to_surf"]
                    xy_full = (float(labels[1]), float(labels[2]))
                    xy_crop = (xy_full[0] - float(crop_origin[0]), xy_full[1] - float(crop_origin[1]))
                    xy_surf = _apply_homography(H, xy_crop)
                    x = int(round(xy_surf[0]))
                    y = int(round(xy_surf[1]))
                    x = max(0, min(x, args.out_w - 1))
                    y = max(0, min(y, args.out_h - 1))
                    cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                t3 = time.perf_counter()

                if frames_dir is not None:
                    frame_path = frames_dir / f"{file_id}_frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(frame_path), img)

                writer.write(img)
                t4 = time.perf_counter()

                t_video_read += (t1 - t0)
                t_warp += (t2 - t1)
                t_label += (t3 - t2)
                t_write += (t4 - t3)

    finally:
        writer.release()
        cap.release()

    if args.profile:
        frames = max(1, int(args.count))
        total = t_video_read + t_warp + t_label + t_write
        print("Timing breakdown (per frame):")
        print(f"  video_read: {t_video_read/frames*1000:.2f} ms")
        print(f"  warp:       {t_warp/frames*1000:.2f} ms")
        print(f"  label/draw: {t_label/frames*1000:.2f} ms")
        print(f"  write:      {t_write/frames*1000:.2f} ms")
        print(f"  total:      {total/frames*1000:.2f} ms")

    print(f"Wrote video: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
