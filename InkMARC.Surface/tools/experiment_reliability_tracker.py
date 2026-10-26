#!/usr/bin/env python3
"""Ablation replay for reliability-weighted, IMU-prior planar tracking.

The synchronized phone quad is used only as a replay-time frame-to-frame motion prior. It is a
proxy for the live IMU/plane prediction; output quality is still measured from image evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from annotate_session import Mp4Writer, nearest_marker, read_session_csv, view_point_to_frame
from experiment_occlusion_tracker import (
    OfflineDynamicMasker,
    OfflineHandMasker,
    combine_masks,
    corner_step,
    draw_quad,
    mask_ratio,
    quad_transform,
    summary,
    tint,
)
from experiment_video_tracker import (
    SparsePlanarTracker,
    TrackResult,
    canonical_patch,
    detect_grid_points,
    grid_coverage,
    patch_similarity,
    quad_from_row,
    read_seed_rows,
    validate_quad,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Replay only the baseline, guarded weighted tracker, and track-memory tracker.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "app" / "src" / "main" / "assets" / "models" / "hand_landmarker.task",
    )
    return parser.parse_args()


def perspective(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape((1, -1, 2)).astype(np.float32), transform)[0]


def cauchy(error: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(error / max(scale, 1e-6), -1e6, 1e6)
    return 1.0 / (1.0 + normalized * normalized)


def effective_count(weights: np.ndarray) -> float:
    total = float(weights.sum())
    squared = float(np.square(weights).sum())
    return 0.0 if squared <= 1e-9 else total * total / squared


def weighted_grid_coverage(points: np.ndarray, weights: np.ndarray, quad: np.ndarray) -> float:
    occupied: set[int] = set()
    minimum = quad.min(axis=0)
    maximum = quad.max(axis=0)
    size = maximum - minimum
    if (size <= 1e-6).any():
        return 0.0
    for point, weight in zip(points, weights):
        if weight < 0.25:
            continue
        normalized = (point - minimum) / size
        if (normalized < 0.0).any() or (normalized > 1.0).any():
            continue
        column = int(np.clip(int(normalized[0] * 4), 0, 3))
        row = int(np.clip(int(normalized[1] * 3), 0, 2))
        occupied.add(row * 4 + column)
    return len(occupied) / 12.0


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = float(np.linalg.norm(shifted, axis=1).mean())
    scale = math.sqrt(2.0) / max(mean_distance, 1e-6)
    transform = np.asarray(
        ((scale, 0.0, -scale * center[0]), (0.0, scale, -scale * center[1]), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def solve_weighted_homography(
    source: np.ndarray,
    destination: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray | None:
    if len(source) < 4 or float(weights.sum()) < 1.0:
        return None
    src, src_transform = normalize_points(source)
    dst, dst_transform = normalize_points(destination)
    rows: list[np.ndarray] = []
    for (x, y), (u, v), weight in zip(src, dst, weights):
        root = math.sqrt(max(float(weight), 1e-6))
        rows.append(root * np.asarray((-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u)))
        rows.append(root * np.asarray((0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v)))
    try:
        _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
        normalized_h = vh[-1].reshape((3, 3))
        homography = np.linalg.inv(dst_transform) @ normalized_h @ src_transform
    except np.linalg.LinAlgError:
        return None
    if abs(float(homography[2, 2])) > 1e-9:
        homography /= homography[2, 2]
    return homography if np.isfinite(homography).all() else None


def solve_weighted_affine(
    source: np.ndarray,
    destination: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray | None:
    if len(source) < 3 or float(weights.sum()) < 1.0:
        return None
    design = np.column_stack((source, np.ones(len(source))))
    root = np.sqrt(np.maximum(weights, 1e-6))[:, None]
    try:
        coefficients = np.linalg.lstsq(design * root, destination * root, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :] = coefficients.T
    return transform if np.isfinite(transform).all() else None


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[min(int(np.searchsorted(cumulative, cumulative[-1] * 0.5)), len(values) - 1)])


def solve_weighted_translation(
    source: np.ndarray,
    destination: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray | None:
    if len(source) < 2 or float(weights.sum()) < 1.0:
        return None
    delta = destination - source
    dx = weighted_median(delta[:, 0], weights)
    dy = weighted_median(delta[:, 1], weights)
    return np.asarray(((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0)))


def robust_weighted_model(
    source: np.ndarray,
    destination: np.ndarray,
    base_weights: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    solver = {
        "homography": solve_weighted_homography,
        "affine": solve_weighted_affine,
        "translation": solve_weighted_translation,
    }[model]
    minimum = {"homography": 4, "affine": 3, "translation": 2}[model]
    weights = base_weights.astype(np.float64).copy()
    transform: np.ndarray | None = None
    residual = np.full(len(source), np.inf)
    threshold = 6.0
    for _ in range(3):
        transform = solver(source, destination, weights)
        if transform is None:
            return None
        projected = perspective(source, transform)
        residual = np.linalg.norm(projected - destination, axis=1)
        finite = residual[np.isfinite(residual)]
        if len(finite) < minimum:
            return None
        median = float(np.median(finite))
        threshold = float(np.clip(1.0 + 2.5 * median, 2.5, 18.0))
        robust = cauchy(residual, threshold * 0.55)
        weights = base_weights * robust
    inliers = np.isfinite(residual) & (residual <= threshold) & (base_weights >= 0.02)
    if int(inliers.sum()) < minimum or float(base_weights[inliers].sum()) < minimum * 0.45:
        return None
    transform = solver(source[inliers], destination[inliers], weights[inliers])
    if transform is None:
        return None
    projected = perspective(source[inliers], transform)
    errors = np.linalg.norm(projected - destination[inliers], axis=1)
    score = float(np.average(errors, weights=np.maximum(base_weights[inliers], 1e-6)))
    return transform, inliers, score


def extract_patch(gray: np.ndarray, point: np.ndarray, radius: int = 7) -> np.ndarray | None:
    x, y = np.rint(point).astype(int)
    if x - radius < 0 or y - radius < 0 or x + radius >= gray.shape[1] or y + radius >= gray.shape[0]:
        return None
    return gray[y - radius : y + radius + 1, x - radius : x + radius + 1].copy()


def reacquire_patch(
    gray: np.ndarray,
    patch: np.ndarray | None,
    predicted: np.ndarray,
    radius: int = 30,
) -> tuple[np.ndarray, float] | None:
    if patch is None or patch.size == 0:
        return None
    half_x, half_y = patch.shape[1] // 2, patch.shape[0] // 2
    center_x, center_y = np.rint(predicted).astype(int)
    left = max(0, center_x - radius - half_x)
    top = max(0, center_y - radius - half_y)
    right = min(gray.shape[1], center_x + radius + half_x + 1)
    bottom = min(gray.shape[0], center_y + radius + half_y + 1)
    search = gray[top:bottom, left:right]
    if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
        return None
    response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    if not math.isfinite(score):
        return None
    point = np.asarray((left + location[0] + half_x, top + location[1] + half_y), dtype=np.float32)
    return point, float(score)


class ReliabilityTracker(SparsePlanarTracker):
    def __init__(self, reliability: bool, weighted_fit: bool, safe_hold: bool, memory: bool = False) -> None:
        super().__init__()
        self.use_reliability = reliability
        self.use_weighted_fit = weighted_fit
        self.use_safe_hold = safe_hold
        self.use_memory = memory
        self.reliability = np.empty(0, dtype=np.float64)
        self.templates: list[np.ndarray | None] = []
        self.dormant: list[dict[str, object]] = []
        self.hold_count = 0
        self.reacquired_total = 0

    def seed(
        self,
        gray: np.ndarray,
        quad: np.ndarray,
        exclusion_mask: np.ndarray | None = None,
    ) -> TrackResult:
        result = super().seed(gray, quad, exclusion_mask)
        self.reliability = np.full(len(self.previous_points), 0.65, dtype=np.float64)
        self.templates = [extract_patch(gray, point) for point in self.previous_points]
        self.dormant = []
        self.hold_count = 0
        self.reacquired_total = 0
        return result

    def update_with_prior(
        self,
        gray: np.ndarray,
        exclusion_mask: np.ndarray | None,
        frame_prior: np.ndarray | None,
    ) -> TrackResult:
        assert self.previous_gray is not None and self.reference_quad is not None
        if self.use_memory:
            retained_dormant: list[dict[str, object]] = []
            for item in self.dormant:
                item["age"] = int(item["age"]) + 1
                if frame_prior is not None:
                    item["point"] = perspective(
                        np.asarray(item["point"], dtype=np.float32).reshape((1, 2)), frame_prior
                    )[0]
                if int(item["age"]) <= 30:
                    retained_dormant.append(item)
            self.dormant = retained_dormant
        if len(self.previous_points) < 4:
            self.previous_gray = gray.copy()
            return self._hold(frame_prior, gray.shape, "too_few_points")
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.previous_points.reshape((-1, 1, 2)),
            None,
            winSize=(41, 41),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
            minEigThreshold=1e-4,
        )
        if next_points is None:
            self.previous_gray = gray.copy()
            return self._hold(frame_prior, gray.shape, "forward_flow")
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(41, 41),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
        )
        current = next_points.reshape((-1, 2))
        back = back_points.reshape((-1, 2)) if back_points is not None else np.zeros_like(current)
        fb_error = np.linalg.norm(back - self.previous_points, axis=1)
        valid = status.reshape(-1).astype(bool)
        if back_status is not None:
            valid &= back_status.reshape(-1).astype(bool)
        valid &= fb_error <= 3.0
        valid &= error.reshape(-1) <= 40.0
        valid &= (current[:, 0] >= 0) & (current[:, 0] < gray.shape[1])
        valid &= (current[:, 1] >= 0) & (current[:, 1] < gray.shape[0])

        masked = np.zeros(len(current), dtype=bool)
        if exclusion_mask is not None:
            rounded = np.rint(current).astype(np.int32)
            inside = (
                (rounded[:, 0] >= 0)
                & (rounded[:, 0] < gray.shape[1])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < gray.shape[0])
            )
            masked[inside] = exclusion_mask[rounded[inside, 1], rounded[inside, 0]] != 0
        mask_rejected = int(np.count_nonzero(valid & masked))

        if self.use_memory:
            for index in np.flatnonzero(valid & masked):
                self.dormant.append(
                    {
                        "reference": self.reference_points[index].copy(),
                        "point": current[index].copy(),
                        "reliability": float(min(self.reliability[index], 0.06)),
                        "patch": self.templates[index],
                        "age": 0,
                    }
                )
            self.dormant = self.dormant[-32:]

        if self.use_reliability:
            predicted = perspective(self.previous_points, frame_prior) if frame_prior is not None else self.previous_points
            prior_error = np.linalg.norm(current - predicted, axis=1)
            population = prior_error[valid & ~masked]
            prior_median = float(np.median(population)) if len(population) else 0.0
            prior_mad = float(np.median(np.abs(population - prior_median))) if len(population) else 0.0
            prior_scale = max(2.5, 1.0 + 2.5 * 1.4826 * prior_mad)
            excess = np.maximum(0.0, prior_error - prior_median - 1.0)
            evidence = np.sqrt(
                cauchy(error.reshape(-1), 18.0)
                * cauchy(fb_error, 2.5)
                * cauchy(excess, prior_scale)
            )
            evidence[masked] = 0.02
            alpha = np.where(evidence < self.reliability, 0.48, 0.14)
            self.reliability = np.clip(self.reliability + alpha * (evidence - self.reliability), 0.02, 1.0)
            if self.use_weighted_fit:
                valid &= self.reliability >= 0.02
                if self.use_memory:
                    valid &= ~masked
            else:
                valid &= ~masked
                valid &= self.reliability >= 0.12
        else:
            valid &= ~masked

        reference = self.reference_points[valid]
        current = current[valid]
        weights = self.reliability[valid] if self.use_reliability else np.ones(len(current))
        templates = [self.templates[index] for index in np.flatnonzero(valid)] if self.use_memory else []
        if self.use_memory and self.dormant:
            recovered_indices: list[int] = []
            for dormant_index, item in enumerate(self.dormant[:8]):
                predicted = np.asarray(item["point"], dtype=np.float32)
                rounded = np.rint(predicted).astype(int)
                if exclusion_mask is not None and (
                    0 <= rounded[0] < gray.shape[1]
                    and 0 <= rounded[1] < gray.shape[0]
                    and exclusion_mask[rounded[1], rounded[0]] != 0
                ):
                    continue
                match = reacquire_patch(gray, item["patch"], predicted)
                if match is None or match[1] < 0.74:
                    continue
                point = match[0]
                rounded = np.rint(point).astype(int)
                if exclusion_mask is not None and exclusion_mask[rounded[1], rounded[0]] != 0:
                    continue
                if len(current) and float(np.linalg.norm(current - point, axis=1).min()) < 5.0:
                    continue
                reference = np.vstack((reference, np.asarray(item["reference"], dtype=np.float32)))
                current = np.vstack((current, point))
                weights = np.concatenate((weights, np.asarray((max(0.45, float(item["reliability"])),))))
                templates.append(item["patch"])
                recovered_indices.append(dormant_index)
            if recovered_indices:
                recovered_set = set(recovered_indices)
                self.dormant = [item for index, item in enumerate(self.dormant) if index not in recovered_set]
                self.reacquired_total += len(recovered_indices)
        if self.use_weighted_fit:
            fit = self._weighted_fit(reference, current, weights, gray.shape[1], gray.shape[0])
            if fit is None:
                transform = None
                model = "held"
                inliers = None
                reprojection_error = None
                reason = "weighted_fit_failed"
            else:
                transform, model, inliers, reprojection_error = fit
                reason = ""
        else:
            transform, model, inliers, reprojection_error, reason = self._fit(
                reference, current, gray.shape[1], gray.shape[0]
            )
        if transform is None or inliers is None:
            self.reference_points = reference
            self.previous_points = current
            self.reliability = weights
            if self.use_memory:
                self.templates = templates
            self.previous_gray = gray.copy()
            held = self._hold(frame_prior, gray.shape, reason)
            held.mask_rejected = mask_rejected
            return held

        reference = reference[inliers]
        current = current[inliers]
        weights = weights[inliers]
        if self.use_memory:
            templates = [templates[index] for index in np.flatnonzero(inliers)]
        mapped_quad = perspective(self.reference_quad, transform)
        if self.use_safe_hold:
            support = effective_count(weights)
            coverage = weighted_grid_coverage(current, weights, mapped_quad)
            predicted_quad = perspective(self.quad, frame_prior) if frame_prior is not None else None
            innovation = (
                float(np.linalg.norm(mapped_quad - predicted_quad, axis=1).max())
                if predicted_quad is not None
                else 0.0
            )
            occlusion_ratio = mask_ratio(exclusion_mask)
            unsafe_innovation = occlusion_ratio >= 0.05 and innovation > 45.0
            if support < 5.0 or coverage < 0.20 or unsafe_innovation:
                self.reference_points = reference
                self.previous_points = current
                self.reliability = weights
                if self.use_memory:
                    self.templates = templates
                self.previous_gray = gray.copy()
                held = self._hold(
                    frame_prior,
                    gray.shape,
                    f"unsafe:{support:.2f}:{coverage:.2f}:{innovation:.1f}:{occlusion_ratio:.3f}",
                )
                held.mask_rejected = mask_rejected
                return held

        self.transform = transform
        self.quad = mapped_quad
        self.hold_count = 0
        needed = 72 - len(current)
        if needed > 0:
            extras = detect_grid_points(gray, self.quad, current, needed, exclusion_mask)
            if len(extras):
                try:
                    inverse = np.linalg.inv(transform)
                    extra_reference = perspective(extras, inverse)
                    reference = np.vstack((reference, extra_reference))
                    current = np.vstack((current, extras))
                    weights = np.concatenate((weights, np.full(len(extras), 0.65)))
                    if self.use_memory:
                        templates.extend(extract_patch(gray, point) for point in extras)
                except np.linalg.LinAlgError:
                    pass
        self.reference_points = reference.astype(np.float32)
        self.previous_points = current.astype(np.float32)
        self.reliability = weights
        if self.use_memory:
            self.templates = templates
        self.previous_gray = gray.copy()
        return TrackResult(self.quad.copy(), model, len(current), reprojection_error, mask_rejected=mask_rejected)

    def _hold(self, frame_prior: np.ndarray | None, shape: tuple[int, ...], reason: str) -> TrackResult:
        if not self.use_safe_hold or self.quad is None or frame_prior is None or self.hold_count >= 12:
            return TrackResult(None, "held", len(self.previous_points), None, reason)
        predicted_quad = perspective(self.quad, frame_prior)
        if validate_quad(predicted_quad, self.quad, shape[1], shape[0]) is not None:
            return TrackResult(None, "held", len(self.previous_points), None, reason)
        self.quad = predicted_quad
        self.hold_count += 1
        return TrackResult(self.quad.copy(), "prior_hold", len(self.previous_points), None, reason)

    def _weighted_fit(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        weights: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, str, np.ndarray, float] | None:
        candidates: list[tuple[np.ndarray, str, np.ndarray, float]] = []
        models = []
        if len(reference) >= 10 and grid_coverage(reference, self.reference_quad) >= 0.25:
            models.append(("homography", 0.35))
        if len(reference) >= 6:
            models.append(("affine", 0.20))
        if len(reference) >= 4:
            models.append(("translation", 0.0))
        for model, penalty in models:
            fit = robust_weighted_model(reference, current, weights, model)
            if fit is None:
                continue
            transform, inliers, error = fit
            quad = perspective(self.reference_quad, transform)
            if validate_quad(quad, self.quad, width, height) is not None:
                continue
            ratio = float(weights[inliers].sum()) / max(float(weights.sum()), 1e-6)
            candidates.append((transform, model, inliers, error + penalty + 3.0 * (1.0 - ratio)))
        if not candidates:
            return None
        transform, model, inliers, score = min(candidates, key=lambda item: item[3])
        return transform, model, inliers, score


def quad_string(quad: np.ndarray | None) -> str:
    return "" if quad is None else ";".join(f"{x:.3f}:{y:.3f}" for x, y in quad)


def main() -> int:
    args = parse_args()
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

    all_trackers: dict[str, ReliabilityTracker] = {
        "stage0_masked": ReliabilityTracker(False, False, False),
        "stage1_reliability": ReliabilityTracker(True, False, False),
        "stage2_weighted": ReliabilityTracker(True, True, False),
        "stage3_safe_hold": ReliabilityTracker(True, True, True),
        "stage4_track_memory": ReliabilityTracker(True, True, True, True),
    }
    trackers = (
        {name: all_trackers[name] for name in ("stage0_masked", "stage3_safe_hold", "stage4_track_memory")}
        if args.final_only
        else all_trackers
    )
    colors = {
        "stage0_masked": (255, 120, 0),
        "stage1_reliability": (0, 220, 255),
        "stage2_weighted": (220, 80, 220),
        "stage3_safe_hold": (0, 255, 80),
        "stage4_track_memory": (255, 220, 60),
    }
    metrics = {
        name: {"step": [], "similarity": [], "support": [], "valid": 0, "catastrophic": 0, "holds": 0}
        for name in trackers
    }
    previous_quads: dict[str, np.ndarray | None] = {name: None for name in trackers}
    reference_patch: np.ndarray | None = None
    previous_phone_quad: np.ndarray | None = None
    seed_index = 0
    active = False
    frame_index = 0
    dynamic_masker = OfflineDynamicMasker()
    hand_masker = OfflineHandMasker(args.model, width, height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(args.output, width, height, fps)
    result_path = args.output.with_suffix(".csv")
    with result_path.open("w", newline="", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(
            ["frame", "time_s", "mask_ratio"]
            + [f"{name}_{field}" for name in trackers for field in ("model", "support", "step", "similarity", "quad")]
        )
        try:
            while ok:
                time_ns = round(frame_index * 1_000_000_000 / fps)
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
                frame_prior = quad_transform(previous_phone_quad, phone_quad)
                hand_mask = hand_masker.update(frame, round(time_ns / 1_000_000))
                motion_mask = dynamic_masker.update(gray, frame_prior)
                exclusion = combine_masks(hand_mask, motion_mask)
                seeded = False
                while seed_index < len(seeds) and time_ns >= seeds[seed_index][0]:
                    seed_quad = seeds[seed_index][1]
                    for name, tracker in trackers.items():
                        tracker.seed(gray, seed_quad, exclusion)
                        previous_quads[name] = seed_quad.copy()
                    reference_patch = canonical_patch(gray, seed_quad)
                    dynamic_masker.reset(gray)
                    seed_index += 1
                    active = True
                    seeded = True
                results: dict[str, TrackResult] = {}
                for name, tracker in trackers.items():
                    if active and not seeded:
                        results[name] = tracker.update_with_prior(gray, exclusion, frame_prior)
                    elif active:
                        results[name] = TrackResult(tracker.quad.copy(), "seed", len(tracker.previous_points), 0.0)
                    else:
                        results[name] = TrackResult(None, "waiting", 0, None)

                row: list[object] = [frame_index, f"{time_ns / 1e9:.6f}", f"{mask_ratio(exclusion):.8f}"]
                annotated = frame.copy()
                tint(annotated, exclusion, (0, 0, 220), 0.22)
                for name, result in results.items():
                    step = corner_step(previous_quads[name], result.quad)
                    similarity = patch_similarity(reference_patch, gray, result.quad)
                    if active:
                        metrics[name]["valid"] += int(result.quad is not None)
                        metrics[name]["holds"] += int(result.model == "prior_hold")
                        metrics[name]["support"].append(float(result.support))
                        if step is not None:
                            metrics[name]["step"].append(step)
                            metrics[name]["catastrophic"] += int(step > 100.0)
                        if similarity is not None:
                            metrics[name]["similarity"].append(similarity)
                    draw_quad(annotated, result.quad, colors[name])
                    row.extend(
                        (
                            result.model,
                            result.support,
                            "" if step is None else f"{step:.6f}",
                            "" if similarity is None else f"{similarity:.6f}",
                            quad_string(result.quad),
                        )
                    )
                    if result.quad is not None:
                        previous_quads[name] = result.quad.copy()
                cv2.putText(
                    annotated,
                    f"t={time_ns / 1e9:.2f}s mask={mask_ratio(exclusion):.1%}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                for index, name in enumerate(trackers):
                    result = results[name]
                    cv2.putText(
                        annotated,
                        f"{name}: {result.model} n={result.support}",
                        (16, 58 + index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        colors[name],
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(annotated)
                result_writer.writerow(row)
                previous_phone_quad = None if phone_quad is None else phone_quad.copy()
                frame_index += 1
                ok, frame = capture.read()
        finally:
            capture.release()
            writer.close()
            hand_masker.close()

    report = {"video": str(args.video), "frames": frame_index, "seeds": len(seeds)}
    for name, values in metrics.items():
        report[name] = {
            "valid_frames": values["valid"],
            "catastrophic_steps": values["catastrophic"],
            "prior_hold_frames": values["holds"],
            "step_px": summary(values["step"]),
            "similarity": summary(values["similarity"]),
            "support": summary(values["support"]),
            "reacquired_tracks": trackers[name].reacquired_total,
        }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {result_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
