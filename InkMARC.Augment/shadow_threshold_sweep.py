#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from .shadow_threshold_ui import build_preview, load_image
except ImportError:
    from shadow_threshold_ui import build_preview, load_image


def assemble_contact_sheet(panels: list[np.ndarray], labels: list[str], columns: int = 3) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to assemble.")

    tile_h, tile_w = panels[0].shape[:2]
    rows = math.ceil(len(panels) / columns)
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)

    for idx, (panel, label) in enumerate(zip(panels, labels)):
        row = idx // columns
        col = idx % columns
        y0 = row * tile_h
        x0 = col * tile_w
        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = panel
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_w, y0 + 34), (18, 18, 18), thickness=-1)
        cv2.putText(
            sheet,
            label,
            (x0 + 12, y0 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

    return sheet


def load_artifacts(
    artifact_dir: Path,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_frame = load_image(artifact_dir / f"{prefix}_raw_frame.jpg", cv2.IMREAD_COLOR, "raw frame")
    shadow_field = load_image(artifact_dir / f"{prefix}_shadow_field.png", cv2.IMREAD_GRAYSCALE, "shadow field", required=False)
    shadow_intensity = load_image(
        artifact_dir / f"{prefix}_shadow_intensity.png",
        cv2.IMREAD_GRAYSCALE,
        "shadow intensity",
        required=shadow_field is None,
    )
    board_mask = load_image(artifact_dir / f"{prefix}_board_mask.png", cv2.IMREAD_GRAYSCALE, "board mask", required=False)
    occlusion_mask = load_image(artifact_dir / f"{prefix}_occlusion_mask.png", cv2.IMREAD_GRAYSCALE, "occlusion mask", required=False)

    if shadow_field is None:
        shadow_field = shadow_intensity.copy()
    if shadow_intensity is None:
        shadow_intensity = shadow_field.copy()
    if board_mask is None:
        board_mask = np.where(shadow_field > 0, 255, 0).astype(np.uint8)
    if occlusion_mask is None:
        occlusion_mask = np.zeros_like(board_mask, dtype=np.uint8)
    return raw_frame, shadow_field, board_mask, occlusion_mask, shadow_intensity


def threshold_values(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("--step must be positive.")
    if stop < start:
        raise ValueError("--stop must be >= --start.")
    return list(range(start, stop + 1, step))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate headless threshold sweeps for saved shadow artifacts.")
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent), help="Directory containing extractor outputs.")
    parser.add_argument("--prefixes", nargs="+", required=True, help="Frame prefixes, for example frame_000381 frame_000594.")
    parser.add_argument("--start", type=int, default=4, help="First threshold in the sweep.")
    parser.add_argument("--stop", type=int, default=24, help="Last threshold in the sweep.")
    parser.add_argument("--step", type=int, default=2, help="Threshold increment.")
    parser.add_argument("--columns", type=int, default=3, help="Columns in the contact sheet.")
    parser.add_argument("--outdir", default=None, help="Output directory for sweep images. Defaults to <dir>/threshold_sweeps.")
    args = parser.parse_args()

    artifact_dir = Path(args.dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else artifact_dir / "threshold_sweeps"
    outdir.mkdir(parents=True, exist_ok=True)

    thresholds = threshold_values(args.start, args.stop, args.step)

    for prefix in args.prefixes:
        prefix_dir = outdir / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)
        raw_frame, shadow_field, board_mask, occlusion_mask, shadow_intensity = load_artifacts(artifact_dir, prefix)

        panels: list[np.ndarray] = []
        labels: list[str] = []
        for threshold in thresholds:
            panel, shadow_mask, overlay = build_preview(
                raw_frame,
                shadow_field,
                board_mask,
                occlusion_mask,
                shadow_intensity,
                threshold,
            )
            panels.append(panel)
            labels.append(f"{prefix} threshold={threshold}")
            cv2.imwrite(str(prefix_dir / f"{prefix}_threshold_{threshold:03d}_panel.jpg"), panel)
            cv2.imwrite(str(prefix_dir / f"{prefix}_threshold_{threshold:03d}_mask.png"), shadow_mask)
            cv2.imwrite(str(prefix_dir / f"{prefix}_threshold_{threshold:03d}_overlay.png"), overlay)

        contact_sheet = assemble_contact_sheet(panels, labels, columns=max(1, args.columns))
        cv2.imwrite(str(prefix_dir / f"{prefix}_threshold_sweep_contact_sheet.jpg"), contact_sheet)
        print(f"Wrote sweep for {prefix} to {prefix_dir}")


if __name__ == "__main__":
    main()
