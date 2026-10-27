#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from .test_shadow_extraction import build_shadow_mask, make_overlay
except ImportError:
    from test_shadow_extraction import build_shadow_mask, make_overlay


WINDOW_NAME = "Shadow Threshold Tuner"


def load_image(path: Path, flags: int, label: str, required: bool = True) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {label}: {path}")
        return None
    image = cv2.imread(str(path), flags)
    if image is None and required:
        raise FileNotFoundError(f"Could not read {label}: {path}")
    return image


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (18, 18, 18), thickness=-1)
    cv2.putText(panel, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    return panel


def fit_for_screen(panel: np.ndarray, max_width: int = 1800, max_height: int = 1100) -> np.ndarray:
    h, w = panel.shape[:2]
    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale >= 0.999:
        return panel
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(panel, new_size, interpolation=cv2.INTER_AREA)


def build_preview(
    raw_frame: np.ndarray,
    shadow_field: np.ndarray,
    board_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    shadow_intensity: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shadow_mask = build_shadow_mask(shadow_field.astype(np.float32), board_mask, threshold)
    shadow_mask[occlusion_mask > 0] = 0

    overlay_intensity = shadow_intensity.copy()
    overlay_intensity[occlusion_mask > 0] = 0
    overlay = make_overlay(
        raw_frame,
        board_mask,
        shadow_mask,
        occlusion_mask,
        overlay_intensity,
    )

    heatmap = cv2.applyColorMap(shadow_field, cv2.COLORMAP_INFERNO)
    mask_bgr = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)

    board_pixels = max(int(np.count_nonzero(board_mask)), 1)
    shadow_pixels = int(np.count_nonzero(shadow_mask))
    coverage = 100.0 * shadow_pixels / float(board_pixels)

    top = np.hstack([
        label_panel(raw_frame, "Raw frame"),
        label_panel(heatmap, f"Shadow field threshold={threshold:.0f}"),
    ])
    bottom = np.hstack([
        label_panel(mask_bgr, f"Mask shadow_px={shadow_pixels} board_coverage={coverage:.2f}%"),
        label_panel(overlay, "Overlay"),
    ])
    return fit_for_screen(np.vstack([top, bottom])), shadow_mask, overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive slider for choosing a shadow-mask threshold.")
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent), help="Directory containing extractor outputs.")
    parser.add_argument("--prefix", required=True, help="Frame prefix, for example frame_000381.")
    parser.add_argument("--threshold", type=int, default=12, help="Initial threshold.")
    args = parser.parse_args()

    artifact_dir = Path(args.dir).resolve()
    prefix = args.prefix

    raw_frame = load_image(artifact_dir / f"{prefix}_raw_frame.jpg", cv2.IMREAD_COLOR, "raw frame")
    shadow_field = load_image(artifact_dir / f"{prefix}_shadow_field.png", cv2.IMREAD_GRAYSCALE, "shadow field", required=False)
    shadow_intensity = load_image(
        artifact_dir / f"{prefix}_shadow_intensity.png",
        cv2.IMREAD_GRAYSCALE,
        "shadow intensity",
        required=shadow_field is None,
    )
    board_mask = load_image(artifact_dir / f"{prefix}_board_mask.png", cv2.IMREAD_GRAYSCALE, "board mask", required=False)
    occlusion_mask = load_image(
        artifact_dir / f"{prefix}_occlusion_mask.png",
        cv2.IMREAD_GRAYSCALE,
        "occlusion mask",
        required=False,
    )

    if shadow_field is None:
        shadow_field = shadow_intensity.copy()
    if shadow_intensity is None:
        shadow_intensity = shadow_field.copy()
    if board_mask is None:
        board_mask = np.where(shadow_field > 0, 255, 0).astype(np.uint8)
    if occlusion_mask is None:
        occlusion_mask = np.zeros_like(board_mask, dtype=np.uint8)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("threshold", WINDOW_NAME, int(np.clip(args.threshold, 0, 255)), 255, lambda _value: None)

    last_threshold = int(np.clip(args.threshold, 0, 255))
    while True:
        threshold = cv2.getTrackbarPos("threshold", WINDOW_NAME)
        panel, shadow_mask, overlay = build_preview(
            raw_frame,
            shadow_field,
            board_mask,
            occlusion_mask,
            shadow_intensity,
            threshold,
        )
        last_threshold = threshold
        cv2.imshow(WINDOW_NAME, panel)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            mask_path = artifact_dir / f"{prefix}_shadow_mask_t{threshold:03d}.png"
            overlay_path = artifact_dir / f"{prefix}_overlay_t{threshold:03d}.png"
            cv2.imwrite(str(mask_path), shadow_mask)
            cv2.imwrite(str(overlay_path), overlay)
            print(f"Saved threshold {threshold} to {mask_path.name} and {overlay_path.name}")

    cv2.destroyAllWindows()
    print(f"Selected threshold: {last_threshold}")


if __name__ == "__main__":
    main()
