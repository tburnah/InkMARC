"""Summarise a synchronized ARCore-versus-fallback capture.

Usage:
  python tools/analyze_arcore_comparison.py <capture>/planes.ndjson

Writes comparison_frames.csv and comparison_summary.json next to the input log unless an output
directory is supplied. The frame log is deliberately plain CSV so it can be opened in Excel.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from math import hypot
from pathlib import Path


def finite_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="planes.ndjson emitted by TrackingDatasetRecorder")
    parser.add_argument("--output-dir", type=Path, help="defaults to the capture directory")
    args = parser.parse_args()

    output_dir = args.output_dir or args.frames.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    outcomes: Counter[str] = Counter()
    point_counts: list[float] = []
    fallback_confidences: list[float] = []
    no_plane_frames = 0
    fallback_without_plane_frames = 0
    depth_placement_frames = 0
    corner_errors: list[float] = []
    quad_ious: list[float] = []
    plane_corner_errors: list[float] = []
    plane_quad_ious: list[float] = []

    with args.frames.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at line {line_number}: {error}") from error

            fallback = frame.get("fallback") or {}
            comparison = frame.get("comparison") or {}
            placement = frame.get("placement") or {}
            outcome = str(comparison.get("outcome", "unknown"))
            outcomes[outcome] += 1
            point_count = finite_number(frame.get("pointCloudPointCount"))
            confidence = finite_number(fallback.get("confidence"))
            if point_count is not None:
                point_counts.append(point_count)
            if confidence is not None:
                fallback_confidences.append(confidence)
            has_horizontal_plane = (frame.get("trackedHorizontalPlaneCount") or 0) > 0
            if not has_horizontal_plane:
                no_plane_frames += 1
                if comparison.get("fallbackUsable"):
                    fallback_without_plane_frames += 1
            if placement.get("source") == "depth":
                depth_placement_frames += 1
            agreement = quad_agreement(fallback.get("quadView"), placement.get("cornersView"))
            if outcome == "both_tracking" and agreement is not None:
                corner_errors.append(agreement["mean_corner_error_px"])
                quad_ious.append(agreement["quad_iou"])
                if placement.get("source") == "plane":
                    plane_corner_errors.append(agreement["mean_corner_error_px"])
                    plane_quad_ious.append(agreement["quad_iou"])
            rows.append(
                {
                    "frame_timestamp_ns": frame.get("frameTimestampNs"),
                    "arcore_camera_state": frame.get("cameraTrackingState"),
                    "arcore_tracking_failure_reason": frame.get("cameraTrackingFailureReason"),
                    "arcore_horizontal_planes": frame.get("trackedHorizontalPlaneCount"),
                    "arcore_point_cloud_points": point_count,
                    "fallback_state": fallback.get("state"),
                    "fallback_has_quad": fallback.get("hasQuad"),
                    "fallback_active_tracks": fallback.get("activeTrackCount"),
                    "fallback_inliers": fallback.get("inlierCount"),
                    "fallback_confidence": confidence,
                    "fallback_rejection_reason": fallback.get("rejectionReason"),
                    "arcore_placement_source": placement.get("source"),
                    "arcore_placement_tracking_state": placement.get("trackingState"),
                    "corner_mean_error_px": agreement["mean_corner_error_px"] if agreement else None,
                    "quad_iou": agreement["quad_iou"] if agreement else None,
                    "comparison_outcome": outcome,
                }
            )

    fields = list(rows[0]) if rows else ["comparison_outcome"]
    with (output_dir / "comparison_frames.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    summary = {
        "frame_count": total,
        "outcomes": dict(outcomes),
        "outcome_ratios": {key: value / total for key, value in outcomes.items()} if total else {},
        "point_cloud_points": describe(point_counts),
        "fallback_confidence": describe(fallback_confidences),
        "pre_plane_evidence": {
            "frames_without_tracked_horizontal_plane": no_plane_frames,
            "fallback_usable_without_plane_frames": fallback_without_plane_frames,
            "depth_backed_placement_frames": depth_placement_frames,
        },
        "corner_agreement_when_both_usable": {
            "all_placements": {
                "mean_corner_error_px": describe(corner_errors),
                "quad_iou": describe(quad_ious),
            },
            "plane_backed_placements_only": {
                "mean_corner_error_px": describe(plane_corner_errors),
                "quad_iou": describe(plane_quad_ious),
            },
        },
        "notes": {
            "arcore_usable": "camera TRACKING with at least one non-subsumed horizontal-upward plane",
            "fallback_usable": "fallback emitted a quadrilateral in seeded or tracking state",
            "comparison": "Both trackers share the user placement, but comparison mode does not let the fallback seed from ARCore planes or pause with ARCore tracking.",
        },
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def quad_agreement(fallback: object, arcore: object) -> dict[str, float] | None:
    fallback_quad = parse_quad(fallback)
    arcore_quad = parse_quad(arcore)
    if fallback_quad is None or arcore_quad is None:
        return None
    errors = [hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(fallback_quad, arcore_quad)]
    intersection = polygon_area(clip_polygon(fallback_quad, arcore_quad))
    union = polygon_area(fallback_quad) + polygon_area(arcore_quad) - intersection
    return {
        "mean_corner_error_px": sum(errors) / len(errors),
        "quad_iou": intersection / union if union > 1e-6 else 0.0,
    }


def parse_quad(value: object) -> list[tuple[float, float]] | None:
    if not isinstance(value, list) or len(value) != 8:
        return None
    try:
        return [(float(value[index]), float(value[index + 1])) for index in range(0, 8, 2)]
    except (TypeError, ValueError):
        return None


def signed_area(points: list[tuple[float, float]]) -> float:
    return sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    ) * 0.5


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(signed_area(points)) if len(points) >= 3 else 0.0


def clip_polygon(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    output = subject
    orientation = 1.0 if signed_area(clip) >= 0 else -1.0
    for edge_start, edge_end in zip(clip, clip[1:] + clip[:1]):
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = is_inside(previous, edge_start, edge_end, orientation)
        for current in input_points:
            current_inside = is_inside(current, edge_start, edge_end, orientation)
            if current_inside != previous_inside:
                output.append(line_intersection(previous, current, edge_start, edge_end))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
    return output


def is_inside(
    point: tuple[float, float], edge_start: tuple[float, float], edge_end: tuple[float, float], orientation: float
) -> bool:
    return orientation * cross(edge_start, edge_end, point) >= -1e-6


def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def line_intersection(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> tuple[float, float]:
    denominator = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if abs(denominator) < 1e-9:
        return b
    determinant_ab = a[0] * b[1] - a[1] * b[0]
    determinant_cd = c[0] * d[1] - c[1] * d[0]
    return (
        (determinant_ab * (c[0] - d[0]) - (a[0] - b[0]) * determinant_cd) / denominator,
        (determinant_ab * (c[1] - d[1]) - (a[1] - b[1]) * determinant_cd) / denominator,
    )


if __name__ == "__main__":
    main()
