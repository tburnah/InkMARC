#!/usr/bin/env python3
"""Evaluate XFeat as a throttled reference-frame recovery matcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from annotate_session import Mp4Writer
from experiment_video_tracker import (
    canonical_patch,
    grid_coverage,
    patch_similarity,
    quad_from_row,
    read_seed_rows,
    validate_quad,
)


ROOT = Path(__file__).resolve().parents[1]
XFEAT_ROOT = ROOT / ".analysis_tracking" / "accelerated_features"
sys.path.insert(0, str(XFEAT_ROOT))

from modules.xfeat import XFeat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval-frames", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=1024)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def draw_quad(frame: np.ndarray, quad: np.ndarray | None, color: tuple[int, int, int]) -> None:
    if quad is not None:
        cv2.polylines(frame, [np.rint(quad).astype(np.int32)], True, color, 4, cv2.LINE_AA)


def quad_step(previous: np.ndarray | None, current: np.ndarray | None) -> float | None:
    if previous is None or current is None:
        return None
    return float(np.linalg.norm(current - previous, axis=1).max())


def main() -> int:
    args = parse_args()
    if not XFEAT_ROOT.exists():
        raise RuntimeError(f"Clone the official XFeat repository to {XFEAT_ROOT}")
    metadata, rows = read_seed_rows(args.csv)
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
        raise RuntimeError("No video frames")
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, 640.0 / source_width)
    width = max(1, round(source_width * scale))
    height = max(1, round(source_height * scale))
    seeds = [
        (int(row["video_time_ns"]), quad)
        for row in rows
        if row.get("event") == "camera_frame"
        and row.get("tracker_state") == "seeded"
        and (quad := quad_from_row(row, metadata, width, height)) is not None
    ]
    if not seeds:
        raise RuntimeError("No seeded rows")

    matcher = XFeat(top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(args.output, width, height, fps)
    result_path = args.output.with_suffix(".csv")
    seed_index = 0
    active_seed_index = -1
    frame_index = 0
    reference_frame: np.ndarray | None = None
    reference_quad: np.ndarray | None = None
    reference_patch: np.ndarray | None = None
    accepted_quad: np.ndarray | None = None
    previous_correction_quad: np.ndarray | None = None
    attempted = accepted = catastrophic = 0
    latencies_ms: list[float] = []
    supports: list[float] = []
    inlier_ratios: list[float] = []
    errors: list[float] = []
    steps: list[float] = []
    similarities: list[float] = []
    with result_path.open("w", newline="", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(
            (
                "frame",
                "time_s",
                "seed_index",
                "attempted",
                "accepted",
                "matches",
                "inliers",
                "inlier_ratio",
                "error_px",
                "latency_ms",
                "step_px",
                "similarity",
                "quad",
            )
        )
        try:
            while ok:
                if scale < 1.0:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                time_ns = round(frame_index * 1_000_000_000 / fps)
                seeded = False
                while seed_index < len(seeds) and time_ns >= seeds[seed_index][0]:
                    active_seed_index = seed_index
                    reference_frame = frame.copy()
                    reference_quad = seeds[seed_index][1].copy()
                    accepted_quad = reference_quad.copy()
                    previous_correction_quad = reference_quad.copy()
                    reference_patch = canonical_patch(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), reference_quad)
                    seed_index += 1
                    seeded = True

                did_attempt = False
                did_accept = False
                match_count = inlier_count = 0
                ratio = error = latency_ms = step_px = similarity = None
                if (
                    active_seed_index >= 0
                    and not seeded
                    and frame_index % max(1, args.interval_frames) == 0
                    and reference_frame is not None
                    and reference_quad is not None
                ):
                    did_attempt = True
                    attempted += 1
                    started = time.perf_counter()
                    source_points, current_points = matcher.match_xfeat(
                        reference_frame,
                        frame,
                        top_k=args.top_k,
                        min_cossim=0.76,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    latencies_ms.append(latency_ms)
                    match_count = len(source_points)
                    if match_count >= 12:
                        transform, inlier_mask = cv2.findHomography(
                            source_points,
                            current_points,
                            cv2.USAC_MAGSAC,
                            3.0,
                        )
                        if transform is not None and inlier_mask is not None:
                            inliers = inlier_mask.reshape(-1).astype(bool)
                            inlier_count = int(inliers.sum())
                            ratio = inlier_count / match_count
                            if inlier_count >= 10:
                                projected = cv2.perspectiveTransform(
                                    source_points[inliers].reshape((1, -1, 2)), transform
                                )[0]
                                error = float(np.linalg.norm(projected - current_points[inliers], axis=1).mean())
                                candidate = cv2.perspectiveTransform(
                                    reference_quad.reshape((1, -1, 2)), transform
                                )[0]
                                coverage = grid_coverage(source_points[inliers], reference_quad)
                                invalid = validate_quad(candidate, previous_correction_quad, width, height)
                                if ratio >= 0.25 and error <= 4.0 and coverage >= 0.25 and invalid is None:
                                    step_px = quad_step(previous_correction_quad, candidate)
                                    similarity = patch_similarity(
                                        reference_patch,
                                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        candidate,
                                    )
                                    accepted_quad = candidate
                                    previous_correction_quad = candidate.copy()
                                    did_accept = True
                                    accepted += 1
                                    supports.append(float(inlier_count))
                                    inlier_ratios.append(float(ratio))
                                    errors.append(error)
                                    if step_px is not None:
                                        steps.append(step_px)
                                        catastrophic += int(step_px > 100.0)
                                    if similarity is not None:
                                        similarities.append(similarity)

                annotated = frame.copy()
                draw_quad(annotated, accepted_quad, (0, 255, 80))
                status = "accepted" if did_accept else "rejected" if did_attempt else "held"
                cv2.putText(
                    annotated,
                    f"XFeat recovery {status} matches={match_count} inliers={inlier_count}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                result_writer.writerow(
                    (
                        frame_index,
                        f"{time_ns / 1e9:.6f}",
                        active_seed_index,
                        did_attempt,
                        did_accept,
                        match_count,
                        inlier_count,
                        "" if ratio is None else f"{ratio:.6f}",
                        "" if error is None else f"{error:.6f}",
                        "" if latency_ms is None else f"{latency_ms:.3f}",
                        "" if step_px is None else f"{step_px:.6f}",
                        "" if similarity is None else f"{similarity:.6f}",
                        "" if accepted_quad is None else ";".join(f"{x:.3f}:{y:.3f}" for x, y in accepted_quad),
                    )
                )
                frame_index += 1
                ok, frame = capture.read()
        finally:
            capture.release()
            writer.close()

    report = {
        "video": str(args.video),
        "frames": frame_index,
        "seeds": len(seeds),
        "interval_frames": args.interval_frames,
        "attempted_recoveries": attempted,
        "accepted_recoveries": accepted,
        "acceptance_ratio": accepted / max(1, attempted),
        "catastrophic_corrections": catastrophic,
        "latency_ms": summarize(latencies_ms),
        "inlier_support": summarize(supports),
        "inlier_ratio": summarize(inlier_ratios),
        "reprojection_error_px": summarize(errors),
        "correction_step_px": summarize(steps),
        "similarity": summarize(similarities),
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
