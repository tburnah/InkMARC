#!/usr/bin/env python3
"""Run CoTracker3 as a sparse offline oracle over an occlusion-heavy session interval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from annotate_session import Mp4Writer, nearest_marker, read_session_csv, view_point_to_frame
from experiment_occlusion_tracker import corner_step, draw_quad, summary
from experiment_video_tracker import canonical_patch, patch_similarity, validate_quad


ROOT = Path(__file__).resolve().parents[1]
COTRACKER_REPOSITORY = ROOT / ".analysis_tracking" / "co-tracker"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=12.0)
    parser.add_argument("--end", type=float, default=22.0)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    return parser.parse_args()


def sample_video(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    sample_fps: float,
) -> tuple[list[np.ndarray], list[int], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    first_index = max(0, round(start_seconds * source_fps))
    last_index = max(first_index + 1, round(end_seconds * source_fps))
    step = max(1, round(source_fps / sample_fps))
    frames: list[np.ndarray] = []
    times_ns: list[int] = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
    frame_index = first_index
    try:
        while frame_index < last_index:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_index - first_index) % step == 0:
                if frame.shape[1] > 640:
                    scale = 640.0 / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (640, max(1, round(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(frame)
                times_ns.append(round(frame_index * 1_000_000_000 / source_fps))
            frame_index += 1
    finally:
        capture.release()
    if len(frames) < 16:
        raise RuntimeError("The requested interval produced fewer than 16 frames")
    return frames, times_ns, source_fps / step


def full_frame_queries(width: int, height: int, columns: int = 8, rows: int = 6) -> np.ndarray:
    xs = np.linspace(width * 0.08, width * 0.92, columns)
    ys = np.linspace(height * 0.08, height * 0.92, rows)
    return np.asarray([(0.0, x, y) for y in ys for x in xs], dtype=np.float32)


def fit_quad(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    visible: np.ndarray,
    reference_quad: np.ndarray,
    previous_quad: np.ndarray | None,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, int, float | None]:
    usable = visible.astype(bool)
    if int(usable.sum()) < 10:
        return None, int(usable.sum()), None
    transform, mask = cv2.findHomography(
        reference_points[usable],
        current_points[usable],
        cv2.USAC_MAGSAC,
        3.0,
    )
    if transform is None or mask is None or int(mask.sum()) < 8:
        return None, int(usable.sum()), None
    quad = cv2.perspectiveTransform(reference_quad.reshape((1, -1, 2)), transform)[0]
    if validate_quad(quad, previous_quad, width, height) is not None:
        return None, int(mask.sum()), None
    inlier_reference = reference_points[usable][mask.reshape(-1).astype(bool)]
    inlier_current = current_points[usable][mask.reshape(-1).astype(bool)]
    projected = cv2.perspectiveTransform(inlier_reference.reshape((1, -1, 2)), transform)[0]
    error = float(np.linalg.norm(projected - inlier_current, axis=1).mean())
    return quad, len(inlier_reference), error


def main() -> int:
    args = parse_args()
    if not COTRACKER_REPOSITORY.exists():
        raise RuntimeError(f"Clone the official CoTracker repository to {COTRACKER_REPOSITORY}")
    metadata, markers = read_session_csv(args.csv)
    marker_times = [marker.video_time_ns for marker in markers]
    frames, times_ns, output_fps = sample_video(args.video, args.start, args.end, args.sample_fps)
    height, width = frames[0].shape[:2]
    first_marker = nearest_marker(markers, marker_times, times_ns[0])
    if first_marker.quad is None:
        raise RuntimeError("No phone quad exists at the beginning of the selected interval")
    reference_quad = np.asarray(
        [view_point_to_frame(x, y, metadata, width, height) for x, y in first_marker.quad],
        dtype=np.float32,
    )
    queries = full_frame_queries(width, height)
    query_tensor = torch.from_numpy(queries)[None]
    rgb = np.stack([frame[:, :, ::-1].copy() for frame in frames])
    video = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float()

    model = torch.hub.load(str(COTRACKER_REPOSITORY), "cotracker3_online", source="local").eval()
    step = int(model.step)
    required = max(0, step * 2 - video.shape[1])
    remainder = (video.shape[1] + required - step * 2) % step
    padding = required + ((step - remainder) % step)
    if padding:
        video = torch.cat((video, video[:, -1:].repeat(1, padding, 1, 1, 1)), dim=1)

    started = time.perf_counter()
    tracks = visibility = None
    with torch.inference_mode():
        model(video[:, : step * 2], is_first_step=True, queries=query_tensor)
        for index in range(0, video.shape[1] - step, step):
            tracks, visibility = model(video[:, index : index + step * 2])
    inference_seconds = time.perf_counter() - started
    if tracks is None or visibility is None:
        raise RuntimeError("CoTracker did not return tracks")
    tracks_array = tracks[0, : len(frames)].cpu().numpy()
    visibility_array = visibility[0, : len(frames)].cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(args.output, width, height, output_fps)
    result_path = args.output.with_suffix(".csv")
    reference_patch = canonical_patch(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), reference_quad)
    previous_quad: np.ndarray | None = reference_quad.copy()
    step_values: list[float] = []
    similarities: list[float] = []
    visible_ratios: list[float] = []
    supports: list[float] = []
    valid_frames = 0
    catastrophic = 0
    with result_path.open("w", newline="", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(
            ("frame", "time_s", "visible_ratio", "support", "error_px", "step_px", "similarity", "quad")
        )
        for index, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            quad, support, error = fit_quad(
                queries[:, 1:],
                tracks_array[index],
                visibility_array[index],
                reference_quad,
                previous_quad,
                width,
                height,
            )
            step_px = corner_step(previous_quad, quad)
            similarity = patch_similarity(reference_patch, gray, quad)
            visible_ratio = float(visibility_array[index].mean())
            visible_ratios.append(visible_ratio)
            supports.append(float(support))
            valid_frames += int(quad is not None)
            if step_px is not None:
                step_values.append(step_px)
                catastrophic += int(step_px > 100.0)
            if similarity is not None:
                similarities.append(similarity)
            annotated = frame.copy()
            draw_quad(annotated, quad, (0, 255, 80))
            for point, point_visible in zip(tracks_array[index], visibility_array[index]):
                if point_visible:
                    cv2.circle(annotated, tuple(np.rint(point).astype(int)), 3, (255, 200, 0), -1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"CoTracker3 t={times_ns[index] / 1e9:.2f}s visible={visible_ratio:.0%} support={support}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            result_writer.writerow(
                (
                    index,
                    f"{times_ns[index] / 1e9:.6f}",
                    f"{visible_ratio:.6f}",
                    support,
                    "" if error is None else f"{error:.6f}",
                    "" if step_px is None else f"{step_px:.6f}",
                    "" if similarity is None else f"{similarity:.6f}",
                    "" if quad is None else ";".join(f"{x:.3f}:{y:.3f}" for x, y in quad),
                )
            )
            if quad is not None:
                previous_quad = quad.copy()
    writer.close()

    report = {
        "video": str(args.video),
        "interval_seconds": [args.start, args.end],
        "sampled_frames": len(frames),
        "sample_fps": output_fps,
        "query_points": len(queries),
        "inference_seconds_cpu": inference_seconds,
        "seconds_per_sampled_frame": inference_seconds / len(frames),
        "valid_frames": valid_frames,
        "catastrophic_steps": catastrophic,
        "visible_ratio": summary(visible_ratios),
        "support": summary(supports),
        "step_px": summary(step_values),
        "similarity": summary(similarities),
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
