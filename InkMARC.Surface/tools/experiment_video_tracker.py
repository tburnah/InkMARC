#!/usr/bin/env python3
"""Compare image-only surface trackers on a recorded InkMARC session.

The synchronized CSV supplies repeatable seed frames and the phone track for comparison. The
experiment intentionally uses no IMU data. It writes an annotated MP4 plus a per-frame result CSV.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from annotate_session import Mp4Writer, read_session_csv, view_point_to_frame


@dataclass(frozen=True)
class SeedEvent:
    time_ns: int
    quad: np.ndarray


@dataclass
class TrackResult:
    quad: np.ndarray | None
    model: str
    support: int
    error: float | None
    reason: str = ""
    mask_rejected: int = 0


@dataclass
class SiftKeyframe:
    label: str
    frame_index: int
    quad: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray
    patch: np.ndarray


@dataclass
class GalleryMatch:
    reference: SiftKeyframe
    quad: np.ndarray
    support: int
    error: float
    similarity: float
    inlier_ratio: float
    quality: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--ecc", action="store_true", help="Also run the slow direct ECC tracker")
    return parser.parse_args()


def read_seed_rows(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for line in source:
            if line.startswith("#"):
                content = line[1:].strip()
                if "=" in content:
                    key, value = content.split("=", 1)
                    metadata[key.strip()] = value.strip()
            elif line.strip():
                data_lines.append(line)
    return metadata, list(csv.DictReader(data_lines))


def quad_from_row(row: dict[str, str], metadata: dict[str, str], width: int, height: int) -> np.ndarray | None:
    values = [row.get(f"quad_{axis}{index}", "") for index in range(4) for axis in ("x", "y")]
    if not all(value not in (None, "") for value in values):
        return None
    points = [
        view_point_to_frame(float(values[index]), float(values[index + 1]), metadata, width, height)
        for index in range(0, 8, 2)
    ]
    return np.asarray(points, dtype=np.float32)


def polygon_area(quad: np.ndarray) -> float:
    return abs(float(cv2.contourArea(quad.astype(np.float32))))


def quad_edges(quad: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)


def validate_quad(
    quad: np.ndarray | None,
    previous: np.ndarray | None,
    width: int,
    height: int,
) -> str | None:
    if quad is None or quad.shape != (4, 2) or not np.isfinite(quad).all():
        return "non_finite"
    contour = np.rint(quad).astype(np.int32).reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        return "non_convex"
    area = polygon_area(quad)
    frame_area = width * height
    if area < frame_area * 0.0015:
        return "too_small"
    if area > frame_area * 2.0:
        return "too_large"
    edges = quad_edges(quad)
    if float(edges.min()) < 8.0:
        return "short_edge"
    margin_x, margin_y = width * 0.35, height * 0.35
    if (
        (quad[:, 0] < -margin_x).any()
        or (quad[:, 0] > width + margin_x).any()
        or (quad[:, 1] < -margin_y).any()
        or (quad[:, 1] > height + margin_y).any()
    ):
        return "outside"
    if previous is not None:
        area_scale = area / max(1.0, polygon_area(previous))
        if not 0.40 <= area_scale <= 2.5:
            return "area_jump"
        motion = np.linalg.norm(quad - previous, axis=1).max()
        if motion > math.hypot(width, height) * 0.42:
            return "corner_jump"
    return None


def point_in_quad(point: np.ndarray, quad: np.ndarray) -> bool:
    return cv2.pointPolygonTest(quad.astype(np.float32), (float(point[0]), float(point[1])), False) >= 0


def cell_index(point: np.ndarray, quad: np.ndarray, columns: int = 4, rows: int = 3) -> int | None:
    if not point_in_quad(point, quad):
        return None
    minimum = quad.min(axis=0)
    maximum = quad.max(axis=0)
    size = maximum - minimum
    if (size <= 1e-6).any():
        return None
    normalized = (point - minimum) / size
    column = int(np.clip(int(normalized[0] * columns), 0, columns - 1))
    row = int(np.clip(int(normalized[1] * rows), 0, rows - 1))
    return row * columns + column


def grid_coverage(points: np.ndarray, quad: np.ndarray) -> float:
    occupied = {cell for point in points if (cell := cell_index(point, quad)) is not None}
    return len(occupied) / 12.0


def detect_grid_points(
    gray: np.ndarray,
    quad: np.ndarray,
    existing: np.ndarray | None = None,
    maximum: int = 72,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    if exclusion_mask is not None:
        mask[exclusion_mask != 0] = 0
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=360,
        qualityLevel=0.004,
        minDistance=6,
        mask=mask,
        blockSize=5,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    candidates = corners.reshape((-1, 2))
    buckets: list[list[np.ndarray]] = [[] for _ in range(12)]
    counts = [0] * 12
    if existing is not None:
        for point in existing:
            cell = cell_index(point, quad)
            if cell is not None:
                counts[cell] += 1
    for point in candidates:
        if existing is not None and len(existing) and np.min(np.linalg.norm(existing - point, axis=1)) < 5.0:
            continue
        cell = cell_index(point, quad)
        if cell is not None:
            buckets[cell].append(point)

    selected: list[np.ndarray] = []
    positions = [0] * 12
    per_cell = max(1, math.ceil(maximum / 12))
    made_progress = True
    while len(selected) < maximum and made_progress:
        made_progress = False
        for cell, bucket in enumerate(buckets):
            if len(selected) >= maximum or counts[cell] >= per_cell:
                continue
            while positions[cell] < len(bucket):
                point = bucket[positions[cell]]
                positions[cell] += 1
                if selected and np.min(np.linalg.norm(np.asarray(selected) - point, axis=1)) < 5.0:
                    continue
                selected.append(point)
                counts[cell] += 1
                made_progress = True
                break
    return np.asarray(selected, dtype=np.float32).reshape((-1, 2))


class SparsePlanarTracker:
    def __init__(self) -> None:
        self.reference_quad: np.ndarray | None = None
        self.reference_points = np.empty((0, 2), dtype=np.float32)
        self.previous_points = np.empty((0, 2), dtype=np.float32)
        self.previous_gray: np.ndarray | None = None
        self.transform = np.eye(3, dtype=np.float64)
        self.quad: np.ndarray | None = None

    def seed(
        self,
        gray: np.ndarray,
        quad: np.ndarray,
        exclusion_mask: np.ndarray | None = None,
    ) -> TrackResult:
        self.reference_quad = quad.copy()
        self.quad = quad.copy()
        self.transform = np.eye(3, dtype=np.float64)
        self.previous_gray = gray.copy()
        points = detect_grid_points(gray, quad, exclusion_mask=exclusion_mask)
        self.reference_points = points.copy()
        self.previous_points = points.copy()
        return TrackResult(quad.copy(), "seed", len(points), 0.0)

    def update(self, gray: np.ndarray, exclusion_mask: np.ndarray | None = None) -> TrackResult:
        assert self.previous_gray is not None and self.reference_quad is not None
        if len(self.previous_points) < 4:
            self.previous_gray = gray.copy()
            return TrackResult(None, "lost", len(self.previous_points), None, "too_few_points")
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.previous_points.reshape((-1, 1, 2)),
            None,
            winSize=(41, 41),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
            flags=0,
            minEigThreshold=1e-4,
        )
        if next_points is None:
            self.previous_gray = gray.copy()
            return TrackResult(None, "lost", 0, None, "forward_flow")
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
        valid = status.reshape(-1).astype(bool)
        if back_status is not None:
            valid &= back_status.reshape(-1).astype(bool)
        valid &= np.linalg.norm(back - self.previous_points, axis=1) <= 3.0
        valid &= error.reshape(-1) <= 40.0
        valid &= (current[:, 0] >= 0) & (current[:, 0] < gray.shape[1])
        valid &= (current[:, 1] >= 0) & (current[:, 1] < gray.shape[0])
        mask_rejected = 0
        if exclusion_mask is not None:
            rounded = np.rint(current).astype(np.int32)
            inside = (
                (rounded[:, 0] >= 0)
                & (rounded[:, 0] < gray.shape[1])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < gray.shape[0])
            )
            masked = np.zeros(len(current), dtype=bool)
            masked[inside] = exclusion_mask[rounded[inside, 1], rounded[inside, 0]] != 0
            mask_rejected = int(np.count_nonzero(valid & masked))
            valid &= ~masked
        reference = self.reference_points[valid]
        current = current[valid]

        transform, model, inliers, reprojection_error, reason = self._fit(reference, current, gray.shape[1], gray.shape[0])
        if transform is None or inliers is None:
            self.previous_points = current
            self.reference_points = reference
            self.previous_gray = gray.copy()
            return TrackResult(None, "held", len(current), None, reason, mask_rejected)

        reference = reference[inliers]
        current = current[inliers]
        self.transform = transform
        self.quad = cv2.perspectiveTransform(self.reference_quad.reshape((1, -1, 2)), transform)[0]
        needed = 72 - len(current)
        if needed > 0:
            extras = detect_grid_points(
                gray,
                self.quad,
                current,
                needed,
                exclusion_mask=exclusion_mask,
            )
            if len(extras):
                inverse = np.linalg.inv(transform)
                extra_reference = cv2.perspectiveTransform(extras.reshape((1, -1, 2)), inverse)[0]
                reference = np.vstack((reference, extra_reference))
                current = np.vstack((current, extras))
        self.reference_points = reference.astype(np.float32)
        self.previous_points = current.astype(np.float32)
        self.previous_gray = gray.copy()
        return TrackResult(
            self.quad.copy(),
            model,
            len(current),
            reprojection_error,
            mask_rejected=mask_rejected,
        )

    def _fit(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray | None, str, np.ndarray | None, float | None, str]:
        previous_quad = self.quad
        candidates: list[tuple[np.ndarray, str, np.ndarray, float]] = []
        reasons: list[str] = []
        if len(reference) >= 10 and grid_coverage(reference, self.reference_quad) >= 0.25:
            homography, mask = cv2.findHomography(reference, current, cv2.USAC_MAGSAC, 3.0)
            if homography is not None and mask is not None and int(mask.sum()) >= 8:
                inliers = mask.reshape(-1).astype(bool)
                quad = cv2.perspectiveTransform(self.reference_quad.reshape((1, -1, 2)), homography)[0]
                invalid = validate_quad(quad, previous_quad, width, height)
                if invalid is None:
                    projected = cv2.perspectiveTransform(reference[inliers].reshape((1, -1, 2)), homography)[0]
                    mean_error = float(np.linalg.norm(projected - current[inliers], axis=1).mean())
                    candidates.append((homography, "homography", inliers, mean_error))
                else:
                    reasons.append(f"homography:{invalid}")
        if len(reference) >= 6:
            affine, mask = cv2.estimateAffine2D(reference, current, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if affine is not None and mask is not None and int(mask.sum()) >= 6:
                transform = np.vstack((affine, (0.0, 0.0, 1.0)))
                inliers = mask.reshape(-1).astype(bool)
                quad = cv2.perspectiveTransform(self.reference_quad.reshape((1, -1, 2)), transform)[0]
                invalid = validate_quad(quad, previous_quad, width, height)
                if invalid is None:
                    projected = cv2.transform(reference[inliers].reshape((-1, 1, 2)), affine).reshape((-1, 2))
                    mean_error = float(np.linalg.norm(projected - current[inliers], axis=1).mean())
                    candidates.append((transform, "affine", inliers, mean_error + 0.15))
                else:
                    reasons.append(f"affine:{invalid}")
        if len(reference) >= 4:
            delta = current - reference
            translation = np.median(delta, axis=0)
            residual = np.linalg.norm(delta - translation, axis=1)
            inliers = residual <= 5.0
            if int(inliers.sum()) >= 4:
                transform = np.asarray(
                    ((1.0, 0.0, translation[0]), (0.0, 1.0, translation[1]), (0.0, 0.0, 1.0)),
                    dtype=np.float64,
                )
                quad = cv2.perspectiveTransform(self.reference_quad.reshape((1, -1, 2)), transform)[0]
                invalid = validate_quad(quad, previous_quad, width, height)
                if invalid is None:
                    candidates.append((transform, "translation", inliers, float(np.mean(residual[inliers])) + 0.35))
        if not candidates:
            return None, "held", None, None, "|".join(reasons) or "fit_failed"
        transform, model, inliers, score = min(candidates, key=lambda candidate: candidate[3])
        return transform, model, inliers, score, ""


class SiftReferenceTracker:
    def __init__(self) -> None:
        self.sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.008, edgeThreshold=14)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.reference_quad: np.ndarray | None = None
        self.reference_keypoints: list[cv2.KeyPoint] = []
        self.reference_descriptors: np.ndarray | None = None
        self.quad: np.ndarray | None = None

    def seed(self, gray: np.ndarray, quad: np.ndarray) -> TrackResult:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        self.reference_quad = quad.copy()
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors
        self.quad = quad.copy()
        return TrackResult(quad.copy(), "seed", len(keypoints), 0.0)

    def update(self, gray: np.ndarray) -> TrackResult:
        assert self.reference_quad is not None
        if self.reference_descriptors is None or len(self.reference_keypoints) < 6:
            return TrackResult(None, "lost", 0, None, "no_reference_descriptors")
        mask = np.zeros(gray.shape, dtype=np.uint8)
        if self.quad is not None:
            center = self.quad.mean(axis=0)
            expanded = center + (self.quad - center) * 1.6
            cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 255)
        else:
            mask.fill(255)
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < 6:
            return TrackResult(None, "held", 0, None, "no_current_descriptors")
        pairs = self.matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.78 * second.distance]
        if len(good) < 6:
            return TrackResult(None, "held", len(good), None, "too_few_matches")
        reference = np.asarray([self.reference_keypoints[match.queryIdx].pt for match in good], dtype=np.float32)
        current = np.asarray([keypoints[match.trainIdx].pt for match in good], dtype=np.float32)
        homography, inlier_mask = cv2.findHomography(reference, current, cv2.USAC_MAGSAC, 3.0)
        if homography is None or inlier_mask is None or int(inlier_mask.sum()) < 6:
            return TrackResult(None, "held", len(good), None, "homography_failed")
        quad = cv2.perspectiveTransform(self.reference_quad.reshape((1, -1, 2)), homography)[0]
        invalid = validate_quad(quad, self.quad, gray.shape[1], gray.shape[0])
        if invalid is not None:
            return TrackResult(None, "held", int(inlier_mask.sum()), None, invalid)
        inliers = inlier_mask.reshape(-1).astype(bool)
        projected = cv2.perspectiveTransform(reference[inliers].reshape((1, -1, 2)), homography)[0]
        error = float(np.linalg.norm(projected - current[inliers], axis=1).mean())
        self.quad = quad
        return TrackResult(quad.copy(), "sift_homography", int(inliers.sum()), error)


class KeyframeSiftTracker:
    """Frame-rate LK with periodic local SIFT correction and keyframe promotion."""

    def __init__(
        self,
        descriptor_interval: int = 5,
        promotion_interval: int = 15,
        origin_interval: int = 15,
    ) -> None:
        self.sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=0.008, edgeThreshold=14)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.lk = SparsePlanarTracker()
        self.descriptor_interval = descriptor_interval
        self.promotion_interval = promotion_interval
        self.origin_interval = origin_interval
        self.keyframe_quad: np.ndarray | None = None
        self.keyframe_keypoints: list[cv2.KeyPoint] = []
        self.keyframe_descriptors: np.ndarray | None = None
        self.keyframe_patch: np.ndarray | None = None
        self.origin_quad: np.ndarray | None = None
        self.origin_keypoints: list[cv2.KeyPoint] = []
        self.origin_descriptors: np.ndarray | None = None
        self.origin_patch: np.ndarray | None = None
        self.quad: np.ndarray | None = None
        self.frame_count = 0
        self.keyframe_age = 0

    def seed(self, gray: np.ndarray, quad: np.ndarray) -> TrackResult:
        result = self.lk.seed(gray, quad)
        self.quad = quad.copy()
        self.frame_count = 0
        self._promote(gray, quad)
        self.origin_quad = None if self.keyframe_quad is None else self.keyframe_quad.copy()
        self.origin_keypoints = list(self.keyframe_keypoints)
        self.origin_descriptors = (
            None if self.keyframe_descriptors is None else self.keyframe_descriptors.copy()
        )
        self.origin_patch = None if self.keyframe_patch is None else self.keyframe_patch.copy()
        return TrackResult(quad.copy(), "seed", result.support, 0.0)

    def update(self, gray: np.ndarray) -> TrackResult:
        self.frame_count += 1
        self.keyframe_age += 1
        lk_result = self.lk.update(gray)
        predicted_quad = lk_result.quad if lk_result.quad is not None else self.quad
        should_match = self.frame_count % self.descriptor_interval == 0 or lk_result.quad is None
        if should_match and predicted_quad is not None:
            local_correction = self._match_keyframe(gray, predicted_quad)
            should_close_origin = (
                self.frame_count % self.origin_interval == 0 or local_correction.quad is None
            )
            origin_correction = (
                self._match_origin(gray, predicted_quad)
                if should_close_origin
                else TrackResult(None, "held", 0, None, "origin_not_scheduled")
            )
            correction = origin_correction if origin_correction.quad is not None else local_correction
            if correction.quad is not None:
                self.quad = correction.quad.copy()
                reseed = self.lk.seed(gray, correction.quad)
                promote = (
                    self.keyframe_age >= self.promotion_interval
                    and correction.support >= 10
                    and correction.error is not None
                    and correction.error <= 1.8
                )
                if promote:
                    self._promote(gray, correction.quad)
                correction_source = correction.model.removesuffix("_correct")
                return TrackResult(
                    correction.quad.copy(),
                    f"{correction_source}_promote" if promote else correction.model,
                    max(correction.support, reseed.support),
                    correction.error,
                    correction.reason,
                )
        if lk_result.quad is not None:
            self.quad = lk_result.quad.copy()
            return TrackResult(
                lk_result.quad.copy(),
                f"keyframe_{lk_result.model}",
                lk_result.support,
                lk_result.error,
                lk_result.reason,
            )
        return TrackResult(None, "held", lk_result.support, lk_result.error, lk_result.reason)

    def _promote(self, gray: np.ndarray, quad: np.ndarray) -> None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < 6:
            return
        self.keyframe_quad = quad.copy()
        self.keyframe_keypoints = keypoints
        self.keyframe_descriptors = descriptors
        self.keyframe_patch = canonical_patch(gray, quad)
        self.keyframe_age = 0

    def _match_keyframe(self, gray: np.ndarray, predicted_quad: np.ndarray) -> TrackResult:
        if (
            self.keyframe_quad is None
            or self.keyframe_descriptors is None
            or self.keyframe_patch is None
            or len(self.keyframe_keypoints) < 6
        ):
            return TrackResult(None, "held", 0, None, "no_keyframe_descriptors")
        return self._match_reference(
            gray,
            predicted_quad,
            self.keyframe_quad,
            self.keyframe_keypoints,
            self.keyframe_descriptors,
            self.keyframe_patch,
            "keyframe",
            1.55,
        )

    def _match_origin(self, gray: np.ndarray, predicted_quad: np.ndarray) -> TrackResult:
        if (
            self.origin_quad is None
            or self.origin_descriptors is None
            or self.origin_patch is None
            or len(self.origin_keypoints) < 6
        ):
            return TrackResult(None, "held", 0, None, "no_origin_descriptors")
        return self._match_reference(
            gray,
            predicted_quad,
            self.origin_quad,
            self.origin_keypoints,
            self.origin_descriptors,
            self.origin_patch,
            "origin",
            2.20,
        )

    def _match_reference(
        self,
        gray: np.ndarray,
        predicted_quad: np.ndarray,
        reference_quad: np.ndarray,
        reference_keypoints: list[cv2.KeyPoint],
        reference_descriptors: np.ndarray,
        reference_patch: np.ndarray,
        source: str,
        search_scale: float,
    ) -> TrackResult:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        center = predicted_quad.mean(axis=0)
        expanded = center + (predicted_quad - center) * search_scale
        cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 255)
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < 6:
            return TrackResult(None, "held", 0, None, f"{source}_no_current_descriptors")
        pairs = self.matcher.knnMatch(reference_descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
        if len(good) < 6:
            return TrackResult(None, "held", len(good), None, f"too_few_{source}_matches")
        reference_points = np.asarray(
            [reference_keypoints[match.queryIdx].pt for match in good],
            dtype=np.float32,
        )
        current_points = np.asarray([keypoints[match.trainIdx].pt for match in good], dtype=np.float32)
        homography, inlier_mask = cv2.findHomography(
            reference_points,
            current_points,
            cv2.USAC_MAGSAC,
            3.0,
        )
        if homography is None or inlier_mask is None:
            return TrackResult(None, "held", len(good), None, f"{source}_homography_failed")
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < 6:
            return TrackResult(None, "held", inlier_count, None, f"too_few_{source}_inliers")
        if grid_coverage(reference_points[inliers], reference_quad) < 0.16:
            return TrackResult(None, "held", inlier_count, None, f"{source}_coverage")
        quad = cv2.perspectiveTransform(reference_quad.reshape((1, -1, 2)), homography)[0]
        invalid = validate_quad(quad, self.quad, gray.shape[1], gray.shape[0])
        if invalid is not None:
            return TrackResult(None, "held", inlier_count, None, f"{source}_{invalid}")
        similarity = patch_similarity(reference_patch, gray, quad)
        if similarity is None or similarity < 0.10:
            return TrackResult(None, "held", inlier_count, similarity, f"{source}_patch_mismatch")
        projected = cv2.perspectiveTransform(reference_points[inliers].reshape((1, -1, 2)), homography)[0]
        error = float(np.linalg.norm(projected - current_points[inliers], axis=1).mean())
        return TrackResult(quad.copy(), f"{source}_correct", inlier_count, error)


class GallerySiftTracker:
    """Frame-rate LK corrected by a bounded gallery of verified SIFT viewpoints."""

    def __init__(
        self,
        descriptor_interval: int = 5,
        promotion_interval: int = 8,
        max_keyframes: int = 5,
        origin_interval: int = 15,
    ) -> None:
        self.sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.008, edgeThreshold=14)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.lk = SparsePlanarTracker()
        self.descriptor_interval = max(1, descriptor_interval)
        self.promotion_interval = max(1, promotion_interval)
        self.max_keyframes = max(2, max_keyframes)
        self.origin_interval = max(1, origin_interval)
        self.references: list[SiftKeyframe] = []
        self.quad: np.ndarray | None = None
        self.frame_count = 0
        self.last_promotion_frame = 0
        self.next_keyframe_index = 1

    @property
    def gallery_size(self) -> int:
        return len(self.references)

    def seed(self, gray: np.ndarray, quad: np.ndarray) -> TrackResult:
        result = self.lk.seed(gray, quad)
        self.references.clear()
        self.quad = quad.copy()
        self.frame_count = 0
        self.last_promotion_frame = 0
        self.next_keyframe_index = 1
        origin = self._capture_reference(gray, quad, "origin")
        if origin is not None:
            self.references.append(origin)
        return TrackResult(quad.copy(), "seed", result.support, 0.0, f"gallery={self.gallery_size}")

    def update(self, gray: np.ndarray) -> TrackResult:
        self.frame_count += 1
        lk_result = self.lk.update(gray)
        predicted_quad = lk_result.quad if lk_result.quad is not None else self.quad
        should_match = self.frame_count % self.descriptor_interval == 0 or lk_result.quad is None
        failure_reason = "gallery_not_scheduled"
        if should_match and predicted_quad is not None and self.references:
            matches: list[GalleryMatch] = []
            reasons: list[str] = []
            keypoints, descriptors = self._detect_current(gray, predicted_quad, 1.65)
            if descriptors is None or len(keypoints) < 6:
                reasons.append("local:no_current_descriptors")
            else:
                for reference in self.references:
                    match, reason = self._match_reference(
                        gray,
                        predicted_quad,
                        keypoints,
                        descriptors,
                        reference,
                    )
                    if match is not None:
                        matches.append(match)
                    elif reason:
                        reasons.append(f"{reference.label}:{reason}")
            should_close_origin = self.frame_count % self.origin_interval == 0 or not matches
            if should_close_origin and self.references[0].label == "origin":
                wide_keypoints, wide_descriptors = self._detect_current(gray, predicted_quad, 2.20)
                if wide_descriptors is None or len(wide_keypoints) < 6:
                    reasons.append("origin_wide:no_current_descriptors")
                else:
                    wide_match, wide_reason = self._match_reference(
                        gray,
                        predicted_quad,
                        wide_keypoints,
                        wide_descriptors,
                        self.references[0],
                    )
                    if wide_match is not None:
                        matches.append(wide_match)
                    elif wide_reason:
                        reasons.append(f"origin_wide:{wide_reason}")
            if matches:
                correction = max(matches, key=lambda match: match.quality)
                self.quad = correction.quad.copy()
                reseed = self.lk.seed(gray, correction.quad)
                promoted = self._should_promote(correction)
                if promoted:
                    label = f"k{self.next_keyframe_index}"
                    new_reference = self._capture_reference(gray, correction.quad, label)
                    if new_reference is not None:
                        self.next_keyframe_index += 1
                        self.references.append(new_reference)
                        self.last_promotion_frame = self.frame_count
                        while len(self.references) > self.max_keyframes:
                            self.references.pop(1)
                    else:
                        promoted = False
                action = "promote" if promoted else "correct"
                diagnostic = (
                    f"source={correction.reference.label};similarity={correction.similarity:.3f};"
                    f"ratio={correction.inlier_ratio:.3f};quality={correction.quality:.3f};"
                    f"gallery={self.gallery_size}"
                )
                return TrackResult(
                    correction.quad.copy(),
                    f"gallery_{action}_{correction.reference.label}",
                    max(correction.support, reseed.support),
                    correction.error,
                    diagnostic,
                )
            failure_reason = "|".join(reasons) or "gallery_no_match"
        if lk_result.quad is not None:
            self.quad = lk_result.quad.copy()
            return TrackResult(
                lk_result.quad.copy(),
                f"gallery_{lk_result.model}",
                lk_result.support,
                lk_result.error,
                failure_reason if should_match else lk_result.reason,
            )
        return TrackResult(None, "held", lk_result.support, lk_result.error, failure_reason or lk_result.reason)

    def _detect_current(
        self,
        gray: np.ndarray,
        predicted_quad: np.ndarray,
        search_scale: float,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        center = predicted_quad.mean(axis=0)
        expanded = center + (predicted_quad - center) * search_scale
        cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 255)
        return self.sift.detectAndCompute(gray, mask)

    def _capture_reference(
        self,
        gray: np.ndarray,
        quad: np.ndarray,
        label: str,
    ) -> SiftKeyframe | None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) < 6:
            return None
        return SiftKeyframe(
            label,
            self.frame_count,
            quad.copy(),
            keypoints,
            descriptors,
            canonical_patch(gray, quad),
        )

    def _match_reference(
        self,
        gray: np.ndarray,
        predicted_quad: np.ndarray,
        current_keypoints: list[cv2.KeyPoint],
        current_descriptors: np.ndarray,
        reference: SiftKeyframe,
    ) -> tuple[GalleryMatch | None, str]:
        pairs = self.matcher.knnMatch(reference.descriptors, current_descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
        if len(good) < 6:
            return None, f"matches={len(good)}"
        reference_points = np.asarray(
            [reference.keypoints[match.queryIdx].pt for match in good],
            dtype=np.float32,
        )
        current_points = np.asarray(
            [current_keypoints[match.trainIdx].pt for match in good],
            dtype=np.float32,
        )
        homography, inlier_mask = cv2.findHomography(
            reference_points,
            current_points,
            cv2.USAC_MAGSAC,
            3.0,
        )
        if homography is None or inlier_mask is None:
            return None, "homography"
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < 6:
            return None, f"inliers={inlier_count}"
        coverage = grid_coverage(reference_points[inliers], reference.quad)
        if coverage < 0.16:
            return None, f"coverage={coverage:.3f}"
        quad = cv2.perspectiveTransform(reference.quad.reshape((1, -1, 2)), homography)[0]
        invalid = validate_quad(quad, self.quad, gray.shape[1], gray.shape[0])
        if invalid is not None:
            return None, invalid
        similarity = patch_similarity(reference.patch, gray, quad)
        if similarity is None or similarity < 0.10:
            return None, "patch_mismatch" if similarity is None else f"similarity={similarity:.3f}"
        projected = cv2.perspectiveTransform(reference_points[inliers].reshape((1, -1, 2)), homography)[0]
        error = float(np.linalg.norm(projected - current_points[inliers], axis=1).mean())
        if error > 2.5:
            return None, f"error={error:.3f}"
        inlier_ratio = inlier_count / len(good)
        prediction_error = float(np.linalg.norm(quad - predicted_quad, axis=1).mean())
        frame_diagonal = math.hypot(gray.shape[1], gray.shape[0])
        quality = (
            similarity
            + 0.25 * min(1.0, inlier_count / 20.0)
            + 0.15 * inlier_ratio
            - 0.05 * min(error, 3.0)
            - 0.10 * min(1.0, prediction_error / (0.10 * frame_diagonal))
        )
        return (
            GalleryMatch(
                reference,
                quad.copy(),
                inlier_count,
                error,
                similarity,
                inlier_ratio,
                quality,
            ),
            "",
        )

    def _should_promote(self, correction: GalleryMatch) -> bool:
        return (
            self.frame_count - self.last_promotion_frame >= self.promotion_interval
            and correction.support >= 8
            and correction.error <= 1.8
            and correction.similarity >= 0.12
            and correction.inlier_ratio >= 0.45
        )


class EccReferenceTracker:
    def __init__(self, scale: float = 0.5) -> None:
        self.scale = scale
        self.reference_small: np.ndarray | None = None
        self.mask_small: np.ndarray | None = None
        self.reference_quad: np.ndarray | None = None
        self.quad: np.ndarray | None = None
        self.transform = np.eye(3, dtype=np.float64)

    def seed(self, gray: np.ndarray, quad: np.ndarray) -> TrackResult:
        self.reference_small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.mask_small = np.zeros(self.reference_small.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            self.mask_small,
            np.rint(quad * self.scale).astype(np.int32),
            255,
        )
        self.reference_quad = quad.copy()
        self.quad = quad.copy()
        self.transform = np.eye(3, dtype=np.float64)
        return TrackResult(quad.copy(), "seed", 0, 1.0)

    def update(self, gray: np.ndarray) -> TrackResult:
        assert self.reference_small is not None and self.mask_small is not None and self.reference_quad is not None
        current_small = cv2.resize(gray, self.reference_small.shape[::-1], interpolation=cv2.INTER_AREA)
        scaling = np.asarray(((self.scale, 0.0, 0.0), (0.0, self.scale, 0.0), (0.0, 0.0, 1.0)))
        inverse_scaling = np.asarray(((1.0 / self.scale, 0.0, 0.0), (0.0, 1.0 / self.scale, 0.0), (0.0, 0.0, 1.0)))
        warp = (scaling @ self.transform @ inverse_scaling).astype(np.float32)
        try:
            correlation, warp = cv2.findTransformECC(
                self.reference_small,
                current_small,
                warp,
                cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 35, 1e-4),
                self.mask_small,
                5,
            )
        except cv2.error:
            return TrackResult(None, "held", 0, None, "ecc_failed")
        transform = inverse_scaling @ warp.astype(np.float64) @ scaling
        quad = cv2.perspectiveTransform(self.reference_quad.reshape((1, -1, 2)), transform)[0]
        invalid = validate_quad(quad, self.quad, gray.shape[1], gray.shape[0])
        if invalid is not None or not math.isfinite(correlation) or correlation < 0.55:
            return TrackResult(None, "held", 0, float(correlation), invalid or "low_correlation")
        self.transform = transform
        self.quad = quad
        return TrackResult(quad.copy(), "ecc_homography", 0, float(correlation))


def canonical_patch(gray: np.ndarray, quad: np.ndarray, width: int = 180, height: int = 360) -> np.ndarray:
    target = np.asarray(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)), dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return cv2.warpPerspective(gray, transform, (width, height), flags=cv2.INTER_LINEAR)


def patch_similarity(reference_patch: np.ndarray | None, gray: np.ndarray, quad: np.ndarray | None) -> float | None:
    if reference_patch is None or quad is None:
        return None
    patch = canonical_patch(gray, quad, reference_patch.shape[1], reference_patch.shape[0])
    reference_gradient = cv2.Laplacian(reference_patch, cv2.CV_32F)
    current_gradient = cv2.Laplacian(patch, cv2.CV_32F)
    margin_y = max(2, reference_patch.shape[0] // 20)
    margin_x = max(2, reference_patch.shape[1] // 20)
    reference_gradient = reference_gradient[margin_y:-margin_y, margin_x:-margin_x]
    current_gradient = current_gradient[margin_y:-margin_y, margin_x:-margin_x]
    score = cv2.matchTemplate(reference_gradient, current_gradient, cv2.TM_CCOEFF_NORMED)[0, 0]
    return float(score) if math.isfinite(float(score)) else None


def draw_quad(frame: np.ndarray, quad: np.ndarray | None, color: tuple[int, int, int], thickness: int) -> None:
    if quad is None:
        return
    points = np.rint(quad).astype(np.int32)
    cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    output = args.output or args.video.with_name(f"{args.video.stem}_tracking_experiment.mp4")
    result_csv = output.with_suffix(".csv")
    metadata, rows = read_seed_rows(args.csv)
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
        SeedEvent(int(row["video_time_ns"]), quad)
        for row in rows
        if row.get("event") == "camera_frame"
        and row.get("tracker_state") == "seeded"
        and (quad := quad_from_row(row, metadata, width, height)) is not None
    ]
    if not seeds:
        raise RuntimeError("No tracker_state=seeded camera rows found")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(output, width, height, fps)
    sparse = SparsePlanarTracker()
    hybrid = SparsePlanarTracker()
    sift = SiftReferenceTracker()
    ecc = EccReferenceTracker() if args.ecc else None
    seed_index = 0
    active_seed = -1
    frame_index = 0
    reference_patch: np.ndarray | None = None
    summaries = {
        "phone": [0 for _ in seeds],
        "sparse": [0 for _ in seeds],
        "hybrid": [0 for _ in seeds],
        "sift": [0 for _ in seeds],
        "ecc": [0 for _ in seeds],
        "frames": [0 for _ in seeds],
    }
    similarity_sums = {
        name: [0.0 for _ in seeds] for name in ("phone", "sparse", "hybrid", "sift", "ecc")
    }
    similarity_counts = {
        name: [0 for _ in seeds] for name in ("phone", "sparse", "hybrid", "sift", "ecc")
    }
    with result_csv.open("w", newline="", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(
            [
                "frame",
                "time_s",
                "seed_index",
                "sparse_model",
                "sparse_support",
                "sparse_error",
                "sparse_reason",
                "sparse_similarity",
                "sparse_quad",
                "hybrid_model",
                "hybrid_support",
                "hybrid_error",
                "hybrid_reason",
                "hybrid_similarity",
                "hybrid_quad",
                "sift_model",
                "sift_support",
                "sift_error",
                "sift_reason",
                "sift_similarity",
                "sift_quad",
                "ecc_model",
                "ecc_correlation",
                "ecc_reason",
                "ecc_similarity",
                "ecc_quad",
                "phone_similarity",
            ]
        )
        try:
            while ok:
                time_ns = round(frame_index * 1_000_000_000 / fps)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                while seed_index < len(seeds) and time_ns >= seeds[seed_index].time_ns:
                    active_seed = seed_index
                    sparse_result = sparse.seed(gray, seeds[seed_index].quad)
                    hybrid_result = hybrid.seed(gray, seeds[seed_index].quad)
                    sift_result = sift.seed(gray, seeds[seed_index].quad)
                    ecc_result = (
                        ecc.seed(gray, seeds[seed_index].quad)
                        if ecc is not None
                        else TrackResult(None, "disabled", 0, None)
                    )
                    reference_patch = canonical_patch(gray, seeds[seed_index].quad)
                    seed_index += 1
                if active_seed >= 0 and not (
                    abs(time_ns - seeds[active_seed].time_ns) <= round(0.5 * 1_000_000_000 / fps)
                ):
                    sparse_result = sparse.update(gray)
                    hybrid_result = hybrid.update(gray)
                    sift_result = sift.update(gray)
                    ecc_result = ecc.update(gray) if ecc is not None else TrackResult(None, "disabled", 0, None)
                elif active_seed < 0:
                    sparse_result = TrackResult(None, "waiting", 0, None)
                    hybrid_result = TrackResult(None, "waiting", 0, None)
                    sift_result = TrackResult(None, "waiting", 0, None)
                    ecc_result = TrackResult(None, "waiting", 0, None)

                marker_index = bisect.bisect_left(marker_times, time_ns)
                marker_index = min(max(marker_index, 0), len(markers) - 1)
                if marker_index > 0 and time_ns - marker_times[marker_index - 1] <= marker_times[marker_index] - time_ns:
                    marker_index -= 1
                phone_quad = markers[marker_index].quad
                phone_frame_quad = (
                    np.asarray(
                        [view_point_to_frame(x, y, metadata, width, height) for x, y in phone_quad],
                        dtype=np.float32,
                    )
                    if phone_quad is not None
                    else None
                )

                sift_similarity = patch_similarity(reference_patch, gray, sift_result.quad)
                if (
                    sift_result.quad is not None
                    and sift_result.support >= 8
                    and sift_similarity is not None
                    and sift_similarity >= 0.12
                ):
                    reset_result = hybrid.seed(gray, sift_result.quad)
                    hybrid_result = TrackResult(
                        reset_result.quad,
                        "sift_reset",
                        reset_result.support,
                        sift_result.error,
                    )

                if active_seed >= 0:
                    summaries["frames"][active_seed] += 1
                    summaries["phone"][active_seed] += int(phone_frame_quad is not None)
                    summaries["sparse"][active_seed] += int(sparse_result.quad is not None)
                    summaries["hybrid"][active_seed] += int(hybrid_result.quad is not None)
                    summaries["sift"][active_seed] += int(sift_result.quad is not None)
                    summaries["ecc"][active_seed] += int(ecc_result.quad is not None)

                similarities = {
                    "phone": patch_similarity(reference_patch, gray, phone_frame_quad),
                    "sparse": patch_similarity(reference_patch, gray, sparse_result.quad),
                    "hybrid": patch_similarity(reference_patch, gray, hybrid_result.quad),
                    "sift": sift_similarity,
                    "ecc": patch_similarity(reference_patch, gray, ecc_result.quad),
                }
                if active_seed >= 0:
                    for name, score in similarities.items():
                        if score is not None:
                            similarity_sums[name][active_seed] += score
                            similarity_counts[name][active_seed] += 1

                draw_quad(frame, phone_frame_quad, (0, 179, 255), 3)
                draw_quad(frame, sparse_result.quad, (80, 220, 80), 3)
                draw_quad(frame, hybrid_result.quad, (255, 120, 40), 3)
                draw_quad(frame, sift_result.quad, (220, 80, 220), 3)
                draw_quad(frame, ecc_result.quad, (255, 220, 60), 3)
                cv2.putText(frame, "phone", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 179, 255), 2)
                cv2.putText(
                    frame,
                    f"LK {sparse_result.model} n={sparse_result.support}",
                    (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 220, 80),
                    2,
                )
                cv2.putText(
                    frame,
                    f"HYB {hybrid_result.model} n={hybrid_result.support}",
                    (16, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 120, 40),
                    2,
                )
                cv2.putText(
                    frame,
                    f"SIFT {sift_result.model} n={sift_result.support}",
                    (16, 114),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (220, 80, 220),
                    2,
                )
                cv2.putText(
                    frame,
                    f"ECC {ecc_result.model} c={ecc_result.error or 0.0:.2f}",
                    (16, 142),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 220, 60),
                    2,
                )
                writer.write(frame)
                quad_text = lambda quad: "" if quad is None else ";".join(f"{value:.3f}" for value in quad.reshape(-1))
                score_text = lambda score: "" if score is None else f"{score:.6f}"
                result_writer.writerow(
                    [
                        frame_index,
                        f"{time_ns / 1e9:.6f}",
                        active_seed,
                        sparse_result.model,
                        sparse_result.support,
                        "" if sparse_result.error is None else f"{sparse_result.error:.6f}",
                        sparse_result.reason,
                        score_text(similarities["sparse"]),
                        quad_text(sparse_result.quad),
                        hybrid_result.model,
                        hybrid_result.support,
                        "" if hybrid_result.error is None else f"{hybrid_result.error:.6f}",
                        hybrid_result.reason,
                        score_text(similarities["hybrid"]),
                        quad_text(hybrid_result.quad),
                        sift_result.model,
                        sift_result.support,
                        "" if sift_result.error is None else f"{sift_result.error:.6f}",
                        sift_result.reason,
                        score_text(similarities["sift"]),
                        quad_text(sift_result.quad),
                        ecc_result.model,
                        "" if ecc_result.error is None else f"{ecc_result.error:.6f}",
                        ecc_result.reason,
                        score_text(similarities["ecc"]),
                        quad_text(ecc_result.quad),
                        score_text(similarities["phone"]),
                    ]
                )
                frame_index += 1
                ok, frame = capture.read()
        finally:
            capture.release()
            writer.close()

    print(
        "attempt,frames,phone_valid,lk_valid,hybrid_valid,sift_valid,ecc_valid,"
        "phone_similarity,lk_similarity,hybrid_similarity,sift_similarity,ecc_similarity"
    )
    for index in range(len(seeds)):
        average = lambda name: (
            similarity_sums[name][index] / similarity_counts[name][index]
            if similarity_counts[name][index]
            else float("nan")
        )
        print(
            f"{index + 1},{summaries['frames'][index]},{summaries['phone'][index]},"
            f"{summaries['sparse'][index]},{summaries['hybrid'][index]},"
            f"{summaries['sift'][index]},{summaries['ecc'][index]},"
            f"{average('phone'):.3f},{average('sparse'):.3f},{average('hybrid'):.3f},"
            f"{average('sift'):.3f},{average('ecc'):.3f}"
        )
    print(f"Wrote {output}")
    print(f"Wrote {result_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
