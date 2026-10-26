#!/usr/bin/env python3
"""Find visually usable MediaPipe hand-landmark frames in an InkMARC recording."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (17, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    return parser.parse_args()


def quality(points: np.ndarray) -> float:
    inside = np.logical_and(points >= 0.0, points <= 1.0).all(axis=1).mean()
    clipped = np.clip(points, 0.0, 1.0)
    width, height = clipped.ptp(axis=0)
    hull = cv2.convexHull((clipped * 1000).astype(np.float32))
    area = cv2.contourArea(hull) / 1_000_000.0
    palm_width = np.linalg.norm(clipped[5] - clipped[17])
    finger_spans = sum(np.linalg.norm(clipped[tip] - clipped[base]) for base, tip in [(5, 8), (9, 12), (13, 16), (17, 20)])
    return inside * (0.35 * width * height + 0.45 * area + 0.20 * min(1.0, palm_width + finger_spans / 2))


def overlay(frame: np.ndarray, points: np.ndarray, score: float, seconds: float) -> np.ndarray:
    image = frame.copy()
    pixels = np.clip(points, 0.0, 1.0) * np.array([frame.shape[1] - 1, frame.shape[0] - 1])
    for first, second in CONNECTIONS:
        a = tuple(np.rint(pixels[first]).astype(int))
        b = tuple(np.rint(pixels[second]).astype(int))
        cv2.line(image, a, b, (20, 20, 20), 8, cv2.LINE_AA)
        cv2.line(image, a, b, (0, 220, 255), 4, cv2.LINE_AA)
    for point in pixels:
        center = tuple(np.rint(point).astype(int))
        cv2.circle(image, center, 9, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(image, center, 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(image, (18, 18), (480, 88), (20, 20, 20), -1)
    cv2.putText(image, f"t={seconds:.3f}s  score={score:.3f}", (32, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = capture.get(cv2.CAP_PROP_FPS)
    start_index = max(0, round(args.start * fps))
    end_index = round(args.end * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    with vision.HandLandmarker.create_from_options(options) as detector:
        for index in range(start_index, end_index + 1):
            ok, raw = capture.read()
            if not ok:
                break
            if (index - start_index) % args.stride:
                continue
            frame = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
            if args.crop:
                x0, y0, x1, y1 = args.crop
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
                inference_frame = frame[y0:y1, x0:x1]
            else:
                inference_frame = frame
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)),
                round(index * 1000 / fps),
            )
            if not result.hand_landmarks:
                continue
            points = np.asarray([[point.x, point.y] for point in result.hand_landmarks[0]], np.float32)
            if args.crop:
                x0, y0, x1, y1 = args.crop
                points[:, 0] = (x0 + points[:, 0] * (x1 - x0)) / frame.shape[1]
                points[:, 1] = (y0 + points[:, 1] * (y1 - y0)) / frame.shape[0]
            candidates.append((quality(points), index / fps, frame, points))
    capture.release()
    candidates.sort(key=lambda item: item[0], reverse=True)
    for rank, (score, seconds, frame, points) in enumerate(candidates[: args.top], start=1):
        output = args.output_dir / f"{rank:02d}_t{seconds:07.3f}_score{score:.3f}.png"
        cv2.imwrite(str(output), overlay(frame, points, score, seconds))
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
