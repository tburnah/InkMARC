#!/usr/bin/env python3
"""Replay a session with MediaPipe hand and plane-residual shadow exclusion masks."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MEDIAPIPE = ROOT / ".analysis_tracking" / "mediapipe_deps"
if LOCAL_MEDIAPIPE.exists():
    sys.path.insert(0, str(LOCAL_MEDIAPIPE))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from annotate_session import Mp4Writer, nearest_marker, read_session_csv, view_point_to_frame
from experiment_video_tracker import (
    SparsePlanarTracker,
    TrackResult,
    canonical_patch,
    patch_similarity,
    quad_from_row,
    read_seed_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "app" / "src" / "main" / "assets" / "models" / "hand_landmarker.task",
    )
    return parser.parse_args()


def mask_ratio(mask: np.ndarray | None) -> float:
    return 0.0 if mask is None else float(np.count_nonzero(mask)) / mask.size


def combine_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    usable = [mask for mask in masks if mask is not None and np.any(mask)]
    if not usable:
        return None
    output = np.zeros(usable[0].shape, dtype=np.uint8)
    for mask in usable:
        cv2.bitwise_or(output, mask, output)
    return output


def ray_distance_to_boundary(
    origin: np.ndarray,
    direction: np.ndarray,
    width: int,
    height: int,
) -> float:
    distances: list[float] = []
    if direction[0] > 1e-6:
        distances.append(((width - 1.0) - origin[0]) / direction[0])
    elif direction[0] < -1e-6:
        distances.append((0.0 - origin[0]) / direction[0])
    if direction[1] > 1e-6:
        distances.append(((height - 1.0) - origin[1]) / direction[1])
    elif direction[1] < -1e-6:
        distances.append((0.0 - origin[1]) / direction[1])
    return min((value for value in distances if value >= 0.0), default=0.0)


class OfflineHandMasker:
    def __init__(self, model: Path, width: int, height: int) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.width = width
        self.height = height
        self.last_submission_ms = -1000
        self.history: list[tuple[int, np.ndarray]] = []
        self.detection_count = 0

    def update(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        if timestamp_ms - self.last_submission_ms >= 100:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect_for_video(image, timestamp_ms)
            self.last_submission_ms = timestamp_ms
            if result.hand_landmarks:
                self.detection_count += 1
                mask = self._build_mask(result.hand_landmarks)
                if mask is not None:
                    self.history.append((timestamp_ms, mask))
                    self.history = self.history[-2:]
        self.history = [item for item in self.history if timestamp_ms - item[0] <= 650]
        return combine_masks(*(mask for _, mask in self.history))

    def _build_mask(self, hands) -> np.ndarray | None:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        minimum = min(self.width, self.height)
        padding = int(np.clip(round(minimum * 0.075), 28, 140))
        for hand in hands:
            points = np.asarray(
                [
                    (
                        np.clip(landmark.x * (self.width - 1), 0, self.width - 1),
                        np.clip(landmark.y * (self.height - 1), 0, self.height - 1),
                    )
                    for landmark in hand
                ],
                dtype=np.float32,
            )
            if len(points) < 18:
                continue
            hull = cv2.convexHull(np.rint(points).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            wrist = points[0]
            palm_center = points[[5, 9, 13, 17]].mean(axis=0)
            direction = wrist - palm_center
            norm = float(np.linalg.norm(direction))
            if norm > 1.0:
                direction /= norm
                extension = min(
                    ray_distance_to_boundary(wrist, direction, self.width, self.height),
                    minimum * 0.45,
                )
                palm_width = float(np.linalg.norm(points[5] - points[17]))
                thickness = int(np.clip(max(56.0, palm_width * 0.65), 1, 280))
                endpoint = np.clip(
                    wrist + direction * extension,
                    (0.0, 0.0),
                    (self.width - 1.0, self.height - 1.0),
                )
                cv2.line(
                    mask,
                    tuple(np.rint(wrist).astype(int)),
                    tuple(np.rint(endpoint).astype(int)),
                    255,
                    thickness,
                    cv2.LINE_AA,
                )
        if not np.any(mask):
            return None
        kernel_size = padding * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel)

    def close(self) -> None:
        self.detector.close()


class OfflineDynamicMasker:
    """Python equivalent of DynamicOcclusionMasker for repeatable threshold tuning."""

    def __init__(self) -> None:
        self.scale = 0.5
        self.previous_gray: np.ndarray | None = None
        self.persistent: np.ndarray | None = None

    def reset(self, gray: np.ndarray | None = None) -> None:
        self.previous_gray = (
            None
            if gray is None
            else cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        )
        self.persistent = None

    def update(self, gray: np.ndarray, previous_to_current: np.ndarray | None) -> np.ndarray | None:
        scaled_gray = cv2.resize(
            gray,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_AREA,
        )
        previous = self.previous_gray
        self.previous_gray = scaled_gray.copy()
        if previous is None or previous_to_current is None or previous.shape != scaled_gray.shape:
            self.persistent = None
            return None

        height, width = scaled_gray.shape
        scaling = np.asarray(((self.scale, 0.0, 0.0), (0.0, self.scale, 0.0), (0.0, 0.0, 1.0)))
        inverse_scaling = np.asarray(
            ((1.0 / self.scale, 0.0, 0.0), (0.0, 1.0 / self.scale, 0.0), (0.0, 0.0, 1.0))
        )
        scaled_transform = scaling @ previous_to_current @ inverse_scaling
        warped = cv2.warpPerspective(previous, scaled_transform, (width, height))
        valid = cv2.warpPerspective(
            np.full_like(previous, 255),
            scaled_transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        valid = np.where(valid >= 250, 255, 0).astype(np.uint8)
        previous_blurred = cv2.GaussianBlur(warped, (7, 7), 0)
        current_blurred = cv2.GaussianBlur(scaled_gray, (7, 7), 0)
        valid_pixels = valid != 0
        if not np.any(valid_pixels):
            self.persistent = None
            return None
        brightness_delta = float(
            np.clip(
                current_blurred[valid_pixels].mean() - previous_blurred[valid_pixels].mean(),
                -32.0,
                32.0,
            )
        )
        adjusted = np.clip(previous_blurred.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)
        difference = cv2.absdiff(current_blurred, adjusted)
        raw = np.where((difference > 18) & valid_pixels, 255, 0).astype(np.uint8)
        raw = cv2.morphologyEx(
            raw,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        raw = cv2.morphologyEx(
            raw,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        filtered = np.zeros_like(raw)
        minimum_area = width * height * 0.0025
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
                filtered[labels == label] = 255
        filtered = cv2.dilate(
            filtered,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        if self.persistent is not None:
            prior = cv2.warpPerspective(
                self.persistent,
                scaled_transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
            )
            prior = cv2.erode(
                prior,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            filtered = cv2.bitwise_or(filtered, prior)
        filtered[~valid_pixels] = 0
        if not np.any(filtered) or mask_ratio(filtered) > 0.52:
            self.persistent = None
            return None
        self.persistent = filtered.copy()
        return cv2.resize(filtered, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)


def quad_transform(previous: np.ndarray | None, current: np.ndarray | None) -> np.ndarray | None:
    if previous is None or current is None:
        return None
    transform = cv2.getPerspectiveTransform(previous.astype(np.float32), current.astype(np.float32))
    return transform if np.isfinite(transform).all() else None


def quad_string(quad: np.ndarray | None) -> str:
    if quad is None:
        return ""
    return ";".join(f"{x:.3f}:{y:.3f}" for x, y in quad)


def corner_step(previous: np.ndarray | None, current: np.ndarray | None) -> float | None:
    if previous is None or current is None:
        return None
    return float(np.linalg.norm(current - previous, axis=1).max())


def tint(frame: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int], alpha: float) -> None:
    if mask is None or not np.any(mask):
        return
    overlay = np.empty_like(frame)
    overlay[:] = color
    pixels = mask != 0
    frame[pixels] = cv2.addWeighted(frame[pixels], 1.0 - alpha, overlay[pixels], alpha, 0.0)


def draw_quad(frame: np.ndarray, quad: np.ndarray | None, color: tuple[int, int, int]) -> None:
    if quad is not None:
        cv2.polylines(frame, [np.rint(quad).astype(np.int32)], True, color, 4, cv2.LINE_AA)


def summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def main() -> int:
    args = parse_args()
    output = args.output or args.video.with_name(f"{args.video.stem}_occlusion_experiment.mp4")
    result_csv = output.with_suffix(".csv")
    report_path = output.with_suffix(".json")
    metadata, seed_rows = read_seed_rows(args.csv)
    _, markers = read_session_csv(args.csv)
    marker_times = [marker.video_time_ns for marker in markers]

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {args.video}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"No frames found in {args.video}")
    height, width = frame.shape[:2]
    seeds = [
        (int(row["video_time_ns"]), quad)
        for row in seed_rows
        if row.get("event") == "camera_frame"
        and row.get("tracker_state") == "seeded"
        and (quad := quad_from_row(row, metadata, width, height)) is not None
    ]
    if not seeds:
        raise RuntimeError("No seeded tracker rows found")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(output, width, height, fps)
    hand_masker = OfflineHandMasker(args.model, width, height)
    dynamic_masker = OfflineDynamicMasker()
    baseline = SparsePlanarTracker()
    masked = SparsePlanarTracker()
    seed_index = 0
    active = False
    previous_phone_quad: np.ndarray | None = None
    previous_baseline_quad: np.ndarray | None = None
    previous_masked_quad: np.ndarray | None = None
    reference_patch: np.ndarray | None = None
    frame_index = 0
    metrics: dict[str, list[float]] = {
        "motion_mask_ratio": [],
        "combined_mask_ratio": [],
        "baseline_step_px": [],
        "masked_step_px": [],
        "baseline_similarity": [],
        "masked_similarity": [],
    }
    counts = {
        "active_frames": 0,
        "motion_mask_frames": 0,
        "hand_mask_frames": 0,
        "baseline_valid_frames": 0,
        "masked_valid_frames": 0,
        "baseline_catastrophic_steps": 0,
        "masked_catastrophic_steps": 0,
        "mask_rejected_tracks": 0,
    }

    with result_csv.open("w", newline="", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(
            [
                "frame",
                "time_s",
                "hand_mask_ratio",
                "motion_mask_ratio",
                "combined_mask_ratio",
                "baseline_model",
                "baseline_support",
                "baseline_step_px",
                "baseline_similarity",
                "baseline_quad",
                "masked_model",
                "masked_support",
                "masked_rejected_tracks",
                "masked_step_px",
                "masked_similarity",
                "masked_quad",
            ]
        )
        try:
            while ok:
                time_ns = round(frame_index * 1_000_000_000 / fps)
                timestamp_ms = round(time_ns / 1_000_000)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                marker = nearest_marker(markers, marker_times, time_ns)
                phone_quad = (
                    np.asarray(
                        [view_point_to_frame(x, y, metadata, width, height) for x, y in marker.quad],
                        dtype=np.float32,
                    )
                    if marker.quad is not None
                    else None
                )
                motion_transform = quad_transform(previous_phone_quad, phone_quad)
                hand_mask = hand_masker.update(frame, timestamp_ms)
                motion_mask = dynamic_masker.update(gray, motion_transform)
                combined = combine_masks(hand_mask, motion_mask)

                seeded_this_frame = False
                while seed_index < len(seeds) and time_ns >= seeds[seed_index][0]:
                    seed_quad = seeds[seed_index][1]
                    baseline_result = baseline.seed(gray, seed_quad)
                    masked_result = masked.seed(gray, seed_quad, combined)
                    reference_patch = canonical_patch(gray, seed_quad)
                    previous_baseline_quad = seed_quad.copy()
                    previous_masked_quad = seed_quad.copy()
                    dynamic_masker.reset(gray)
                    active = True
                    seeded_this_frame = True
                    seed_index += 1
                if active and not seeded_this_frame:
                    baseline_result = baseline.update(gray)
                    masked_result = masked.update(gray, combined)
                elif not active:
                    baseline_result = TrackResult(None, "waiting", 0, None)
                    masked_result = TrackResult(None, "waiting", 0, None)

                baseline_step = corner_step(previous_baseline_quad, baseline_result.quad)
                masked_step = corner_step(previous_masked_quad, masked_result.quad)
                baseline_similarity = patch_similarity(reference_patch, gray, baseline_result.quad)
                masked_similarity = patch_similarity(reference_patch, gray, masked_result.quad)
                if active:
                    counts["active_frames"] += 1
                    counts["motion_mask_frames"] += int(motion_mask is not None)
                    counts["hand_mask_frames"] += int(hand_mask is not None)
                    counts["baseline_valid_frames"] += int(baseline_result.quad is not None)
                    counts["masked_valid_frames"] += int(masked_result.quad is not None)
                    counts["mask_rejected_tracks"] += masked_result.mask_rejected
                    if baseline_step is not None:
                        metrics["baseline_step_px"].append(baseline_step)
                        counts["baseline_catastrophic_steps"] += int(baseline_step > 100.0)
                    if masked_step is not None:
                        metrics["masked_step_px"].append(masked_step)
                        counts["masked_catastrophic_steps"] += int(masked_step > 100.0)
                    if baseline_similarity is not None:
                        metrics["baseline_similarity"].append(baseline_similarity)
                    if masked_similarity is not None:
                        metrics["masked_similarity"].append(masked_similarity)
                    metrics["motion_mask_ratio"].append(mask_ratio(motion_mask))
                    metrics["combined_mask_ratio"].append(mask_ratio(combined))

                annotated = frame.copy()
                tint(annotated, motion_mask, (180, 0, 180), 0.28)
                tint(annotated, hand_mask, (0, 0, 255), 0.38)
                draw_quad(annotated, phone_quad, (0, 179, 255))
                draw_quad(annotated, baseline_result.quad, (255, 120, 0))
                draw_quad(annotated, masked_result.quad, (0, 255, 80))
                cv2.putText(
                    annotated,
                    f"t={time_ns / 1e9:.2f}s  motion={mask_ratio(motion_mask):.1%} "
                    f"hand={mask_ratio(hand_mask):.1%}",
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"baseline={baseline_result.support}  masked={masked_result.support} "
                    f"rejected={masked_result.mask_rejected}",
                    (18, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                result_writer.writerow(
                    [
                        frame_index,
                        f"{time_ns / 1e9:.6f}",
                        f"{mask_ratio(hand_mask):.8f}",
                        f"{mask_ratio(motion_mask):.8f}",
                        f"{mask_ratio(combined):.8f}",
                        baseline_result.model,
                        baseline_result.support,
                        "" if baseline_step is None else f"{baseline_step:.6f}",
                        "" if baseline_similarity is None else f"{baseline_similarity:.6f}",
                        quad_string(baseline_result.quad),
                        masked_result.model,
                        masked_result.support,
                        masked_result.mask_rejected,
                        "" if masked_step is None else f"{masked_step:.6f}",
                        "" if masked_similarity is None else f"{masked_similarity:.6f}",
                        quad_string(masked_result.quad),
                    ]
                )
                if baseline_result.quad is not None:
                    previous_baseline_quad = baseline_result.quad.copy()
                if masked_result.quad is not None:
                    previous_masked_quad = masked_result.quad.copy()
                previous_phone_quad = None if phone_quad is None else phone_quad.copy()
                frame_index += 1
                ok, frame = capture.read()
        finally:
            capture.release()
            writer.close()
            hand_masker.close()

    report = {
        "video": str(args.video),
        "frames": frame_index,
        "seeds": len(seeds),
        "mediapipe_hand_detections": hand_masker.detection_count,
        **counts,
        **{name: summary(values) for name, values in metrics.items()},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {result_csv}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
