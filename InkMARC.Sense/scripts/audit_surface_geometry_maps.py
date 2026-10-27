#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset
from inkdepth.datasets.stylus_dataset import _resize_points
from inkdepth.utils.config import load_config


def _label_corners_for_input(dataset: StylusDataset, label: dict[str, Any]) -> np.ndarray | None:
    corners = label.get("corners")
    if corners is None:
        return None
    c = np.asarray(corners, dtype=np.float32)
    if c.shape != (4, 2) or not np.isfinite(c).all():
        return None

    label_cfg = dataset.cfg.data.labels
    mode = str(getattr(label_cfg, "corners_coordinate_mode", "resized"))
    if mode == "original":
        c = _resize_points(c, label.get("resize_attrs", {}) or {})
    elif mode != "resized":
        raise ValueError(f"Unknown corners_coordinate_mode: {mode}")

    h, w = dataset.input_size
    image_w = float(getattr(label_cfg, "image_width", w))
    image_h = float(getattr(label_cfg, "image_height", h))
    if image_w <= 0 or image_h <= 0:
        return None
    c = c.copy()
    c[:, 0] *= float(w) / image_w
    c[:, 1] *= float(h) / image_h
    if abs(float(cv2.contourArea(c.astype(np.float32)))) < 1.0:
        return None
    return c


def _polygon_mask(shape: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    return mask.astype(bool)


def _mask_iou(mask: np.ndarray, corners: np.ndarray) -> float:
    expected = _polygon_mask(mask.shape, corners)
    actual = mask > 0.5
    union = np.logical_or(expected, actual).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(expected, actual).sum() / union)


def _sample_bilinear(arr: np.ndarray, x: float, y: float) -> float:
    h, w = arr.shape
    x = float(np.clip(x, 0.0, w - 1.0))
    y = float(np.clip(y, 0.0, h - 1.0))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    top = (1.0 - dx) * arr[y0, x0] + dx * arr[y0, x1]
    bottom = (1.0 - dx) * arr[y1, x0] + dx * arr[y1, x1]
    return float((1.0 - dy) * top + dy * bottom)


def _raw_uv(dataset: StylusDataset, rec: dict[str, Any]) -> tuple[float, float] | None:
    h5_path = rec.get("label_h5_path")
    if not h5_path:
        return None
    frame_index = int(rec.get("frame_index", rec.get("index", 0)))
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][frame_index], dtype=np.float32)
        masks = np.asarray(h5["label_mask"][frame_index], dtype=np.uint8)
    if not (bool(masks[0]) and bool(masks[1])):
        return None
    if not np.isfinite(labels[:2]).all():
        return None
    label_cfg = dataset.cfg.data.labels
    surface_w = float(getattr(label_cfg, "surface_width", getattr(label_cfg, "tablet_width", 44800.0)) or 0.0)
    surface_h = float(getattr(label_cfg, "surface_height", getattr(label_cfg, "tablet_height", 29600.0)) or 0.0)
    if surface_w <= 0 or surface_h <= 0:
        return None
    u = float(labels[0] / surface_w)
    v = float(labels[1] / surface_h)
    if not np.isfinite([u, v]).all() or u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return None
    return u, v


def _to_bgr(sample_image: np.ndarray) -> np.ndarray:
    rgb = np.clip(sample_image.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _color_map(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[mask <= 0.5] = 0
    return color


def _put_label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 22
    for line in lines:
        cv2.rectangle(out, (4, y - 17), (min(out.shape[1] - 4, 12 + len(line) * 8), y + 5), (0, 0, 0), -1)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def _write_panel(
    out_path: Path,
    sample: dict[str, Any],
    corners: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    image = sample["image"].detach().cpu().numpy()
    frame = _to_bgr(image[:3])
    mask, u_map, v_map = image[3], image[4], image[5]

    overlay = frame.copy()
    cv2.polylines(overlay, [np.round(corners).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    if sample.get("valid_xy") and sample.get("xy") is not None:
        h, w = mask.shape
        xy = sample["xy"].detach().cpu().numpy()
        x = int(np.clip(round(float(xy[0]) * w), 0, w - 1))
        y = int(np.clip(round(float(xy[1]) * h), 0, h - 1))
        cv2.circle(overlay, (x, y), 7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(overlay, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

    mask_bgr = cv2.cvtColor((mask * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    u_color = _color_map(u_map, mask)
    v_color = _color_map(v_map, mask)
    overlay = _put_label(
        overlay,
        [
            f"{sample['video_id']} f{int(sample['frame_index'])}",
            f"mask_iou={metrics.get('mask_iou', float('nan')):.5f}",
            f"uv_abs_err={metrics.get('uv_abs_err', float('nan')):.5f}",
        ],
    )
    panel = np.concatenate([frame, overlay, mask_bgr, u_color, v_color], axis=1)
    cv2.imwrite(str(out_path), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit surface geometry maps emitted by StylusDataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--output-dir", default="outputs/surface_geometry_audit")
    parser.add_argument("--stride", type=int, default=997)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = StylusDataset(cfg, split=args.split, training=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.split}_metrics.jsonl"

    rows: list[dict[str, Any]] = []
    step = max(1, int(args.stride))
    indices = list(range(0, len(dataset), step)) + list(range(min(step // 2, len(dataset)), len(dataset), step))
    seen: set[int] = set()
    selected = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_f:
        for idx in indices:
            if idx in seen or idx >= len(dataset):
                continue
            seen.add(idx)
            rec = dataset.records[idx]
            label = dataset._lookup_label(rec)
            corners = _label_corners_for_input(dataset, label)
            if corners is None:
                continue
            sample = dataset[idx]
            if sample["image"].shape[0] < 6:
                raise RuntimeError("Expected RGB + 3 surface geometry channels")
            geom = sample["image"][3:].detach().cpu().numpy()
            mask_iou = _mask_iou(geom[0], corners)
            row: dict[str, Any] = {
                "dataset_index": idx,
                "video_id": sample["video_id"],
                "frame_index": int(sample["frame_index"]),
                "valid_xy": bool(sample["valid_xy"]),
                "mask_iou": mask_iou,
            }
            uv = _raw_uv(dataset, rec)
            if uv is not None and sample["valid_xy"] and sample["xy"] is not None:
                h, w = geom[0].shape
                xy = sample["xy"].detach().cpu().numpy()
                px = float(np.clip(float(xy[0]) * w, 0.0, w - 1.0))
                py = float(np.clip(float(xy[1]) * h, 0.0, h - 1.0))
                u_hat = _sample_bilinear(geom[1], px, py)
                v_hat = _sample_bilinear(geom[2], px, py)
                row.update(
                    {
                        "raw_u": uv[0],
                        "raw_v": uv[1],
                        "map_u_at_label": u_hat,
                        "map_v_at_label": v_hat,
                        "uv_abs_err": 0.5 * (abs(u_hat - uv[0]) + abs(v_hat - uv[1])),
                    }
                )
            rows.append(row)
            metrics_f.write(json.dumps(row, sort_keys=True) + "\n")
            out_path = output_dir / f"{selected:03d}_{args.split}_{sample['video_id']}_f{int(sample['frame_index'])}.jpg"
            _write_panel(out_path, sample, corners, row)
            selected += 1
            if selected >= int(args.num_samples):
                break

    dataset.close()
    if not rows:
        print(f"No samples with valid corners found for split={args.split}")
        return
    mask_ious = np.array([r["mask_iou"] for r in rows if np.isfinite(r["mask_iou"])], dtype=np.float32)
    uv_errs = np.array([r["uv_abs_err"] for r in rows if "uv_abs_err" in r and np.isfinite(r["uv_abs_err"])], dtype=np.float32)
    print(f"saved {len(rows)} panels to {output_dir}")
    print(f"metrics: {metrics_path}")
    if mask_ious.size:
        print(f"mask_iou mean={mask_ious.mean():.5f} min={mask_ious.min():.5f}")
    if uv_errs.size:
        print(f"uv_abs_err mean={uv_errs.mean():.5f} max={uv_errs.max():.5f}")


if __name__ == "__main__":
    main()
