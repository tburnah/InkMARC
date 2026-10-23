#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import cv2  # type: ignore
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.dataset_core import InkMARCIndexedDataset
from data.transforms.homography_warp import compute_crop_box


def _poly_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _apply_homography(H: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    pt = np.array([[xy]], dtype=np.float64)  # (1,1,2)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def _get_image_dims(h5_path: Path) -> Tuple[int | None, int | None]:
    try:
        with h5py.File(h5_path, "r") as h5:
            w = h5.attrs.get("width")
            h = h5.attrs.get("height")
            w = int(w) if w is not None else None
            h = int(h) if h is not None else None
            return w, h
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/LocationData")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--include-unlabeled", action="store_true")
    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--debug-count", type=int, default=5)
    args = ap.parse_args()

    ds = InkMARCIndexedDataset(
        dataset_root=args.dataset_root,
        index_version=args.version,
        split=args.split,
        require_labels=not args.include_unlabeled,
        transform=None,
    )

    dims_cache: Dict[str, Tuple[int | None, int | None]] = {}

    n = min(len(ds), int(args.max_samples))
    stats = defaultdict(int)
    xy_min = np.array([np.inf, np.inf], dtype=np.float64)
    xy_max = np.array([-np.inf, -np.inf], dtype=np.float64)

    debug_left = int(args.debug_count)
    for i in range(n):
        sample = ds[i]
        corners = np.asarray(sample["corners"], dtype=np.float64)
        labels = np.asarray(sample["labels"], dtype=np.float64)
        mask = np.asarray(sample["label_mask"], dtype=np.uint8)
        meta = sample.get("meta", {})

        if corners.shape != (4, 2):
            stats["bad_corners_shape"] += 1
            continue
        if not np.isfinite(corners).all():
            stats["corners_nonfinite"] += 1
            continue

        area = _poly_area(corners)
        if abs(area) < 1.0:
            stats["corners_degenerate"] += 1

        rel = meta.get("rel_path")
        if not rel:
            stats["missing_rel_path"] += 1
            continue
        h5_path = Path(args.dataset_root) / str(rel)
        key = str(h5_path)
        if key not in dims_cache:
            dims_cache[key] = _get_image_dims(h5_path)
        img_w, img_h = dims_cache[key]

        if img_w is not None and img_h is not None:
            tol = float(args.tolerance)
            if (corners[:, 0] < -tol).any() or (corners[:, 0] > img_w + tol).any():
                stats["corners_oob_x"] += 1
            if (corners[:, 1] < -tol).any() or (corners[:, 1] > img_h + tol).any():
                stats["corners_oob_y"] += 1
        else:
            stats["missing_dims"] += 1
        if debug_left > 0:
            print(
                f"sample {i} file={meta.get('file_id')} frame={meta.get('frame_idx')} "
                f"w={img_w} h={img_h} corners0={corners[0].tolist()} "
                f"xy=({labels[0]:.2f},{labels[1]:.2f}) p={labels[2]:.2f} tiltX={labels[3]:.2f}"
            )
            debug_left -= 1

        touch = float(np.clip(labels[2], 0.0, 1.0))
        xy_valid = (int(mask[0]) != 0 and int(mask[1]) != 0)
        if not xy_valid:
            stats["xy_skipped"] += 1
            continue

        xy_full = (float(labels[0]), float(labels[1]))
        if img_w is not None and img_h is not None:
            tol = float(args.tolerance)
            if xy_full[0] < -tol or xy_full[0] > img_w + tol:
                stats["xy_full_oob_x"] += 1
            if xy_full[1] < -tol or xy_full[1] > img_h + tol:
                stats["xy_full_oob_y"] += 1

        if img_w is None or img_h is None:
            stats["xy_missing_dims"] += 1
            continue

        try:
            x0, y0, x1, y1 = compute_crop_box(corners, img_w, img_h, margin_frac=args.margin_frac)
        except Exception:
            stats["crop_failed"] += 1
            continue

        corners_crop = corners - np.array([x0, y0], dtype=np.float64)
        src = corners_crop.astype(np.float32)
        dst = np.array(
            [
                [0.0, 0.0],
                [args.out_w - 1.0, 0.0],
                [args.out_w - 1.0, args.out_h - 1.0],
                [0.0, args.out_h - 1.0],
            ],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(src, dst)
        xy_crop = (xy_full[0] - float(x0), xy_full[1] - float(y0))
        xy_surf = _apply_homography(H, xy_crop)

        if not np.isfinite(xy_surf[0]) or not np.isfinite(xy_surf[1]):
            stats["xy_surf_nonfinite"] += 1
            continue

        xy_min = np.minimum(xy_min, np.array(xy_surf))
        xy_max = np.maximum(xy_max, np.array(xy_surf))

        if xy_surf[0] < -0.1 * args.out_w or xy_surf[0] > 1.1 * args.out_w:
            stats["xy_surf_oob_x"] += 1
        if xy_surf[1] < -0.1 * args.out_h or xy_surf[1] > 1.1 * args.out_h:
            stats["xy_surf_oob_y"] += 1

        stats["xy_valid"] += 1

    ds.close()

    print("Phase1 geometry validation")
    print(f"  samples_checked: {n}")
    print(f"  corners_degenerate: {stats['corners_degenerate']}")
    print(f"  corners_nonfinite: {stats['corners_nonfinite']}")
    print(f"  corners_oob_x: {stats['corners_oob_x']}")
    print(f"  corners_oob_y: {stats['corners_oob_y']}")
    print(f"  missing_dims: {stats['missing_dims']}")
    print(f"  xy_valid: {stats['xy_valid']}")
    print(f"  xy_skipped: {stats['xy_skipped']}")
    print(f"  xy_full_oob_x: {stats['xy_full_oob_x']}")
    print(f"  xy_full_oob_y: {stats['xy_full_oob_y']}")
    print(f"  xy_surf_oob_x: {stats['xy_surf_oob_x']}")
    print(f"  xy_surf_oob_y: {stats['xy_surf_oob_y']}")
    print(f"  xy_surf_nonfinite: {stats['xy_surf_nonfinite']}")
    if np.isfinite(xy_min).all() and np.isfinite(xy_max).all():
        print(f"  xy_surf_min: ({xy_min[0]:.2f}, {xy_min[1]:.2f})")
        print(f"  xy_surf_max: ({xy_max[0]:.2f}, {xy_max[1]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
