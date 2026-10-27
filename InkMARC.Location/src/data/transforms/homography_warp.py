from __future__ import annotations

from typing import Dict, Tuple

import cv2  # type: ignore
import numpy as np


def compute_crop_box(
    corners_xy: np.ndarray,
    img_w: int,
    img_h: int,
    margin_frac: float = 0.10,
) -> Tuple[int, int, int, int]:
    if corners_xy.shape != (4, 2):
        raise ValueError(f"corners_xy must be (4,2), got {corners_xy.shape}")

    min_xy = corners_xy.min(axis=0)
    max_xy = corners_xy.max(axis=0)
    w = max(float(max_xy[0] - min_xy[0]), 1.0)
    h = max(float(max_xy[1] - min_xy[1]), 1.0)

    margin_x = w * margin_frac
    margin_y = h * margin_frac

    x0 = int(np.floor(min_xy[0] - margin_x))
    y0 = int(np.floor(min_xy[1] - margin_y))
    x1 = int(np.ceil(max_xy[0] + margin_x))
    y1 = int(np.ceil(max_xy[1] + margin_y))

    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(1, min(x1, img_w))
    y1 = max(1, min(y1, img_h))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop box from corners: {(x0, y0, x1, y1)}")

    return x0, y0, x1, y1


def _order_corners(corners_xy: np.ndarray) -> np.ndarray:
    if corners_xy.shape != (4, 2):
        raise ValueError(f"corners_xy must be (4,2), got {corners_xy.shape}")
    # Order by sum and diff: TL, TR, BR, BL
    pts = corners_xy.astype(np.float32)
    s = pts.sum(axis=1)
    diff = (pts[:, 1] - pts[:, 0])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def warp_to_canonical(
    img_bgr: np.ndarray,
    corners_xy: np.ndarray,
    out_w: int = 480,
    out_h: int = 320,
    margin_frac: float = 0.10,
) -> Dict[str, np.ndarray | Tuple[int, int]]:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"img_bgr must be HxWx3, got {img_bgr.shape}")

    img_h, img_w = img_bgr.shape[:2]
    x0, y0, x1, y1 = compute_crop_box(corners_xy, img_w, img_h, margin_frac=margin_frac)
    crop = img_bgr[y0:y1, x0:x1]

    corners_crop = corners_xy.astype(np.float64) - np.array([x0, y0], dtype=np.float64)
    ordered_guess = corners_crop.astype(np.float32)
    if ordered_guess.shape != (4, 2):
        raise ValueError(f"corners_xy must be (4,2), got {corners_xy.shape}")

    ordered_fallback = _order_corners(corners_crop)
    max_delta = float(np.max(np.linalg.norm(ordered_guess - ordered_fallback, axis=1)))
    ordered = ordered_guess if max_delta < 5.0 else ordered_fallback

    dst = np.array(
        [
            [0.0, 0.0],
            [out_w - 1.0, 0.0],
            [out_w - 1.0, out_h - 1.0],
            [0.0, out_h - 1.0],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
    warped = cv2.warpPerspective(crop, H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    return {
        "img_surf_bgr": warped,
        "H_crop_to_surf": H,
        "crop_origin": (x0, y0),
    }
