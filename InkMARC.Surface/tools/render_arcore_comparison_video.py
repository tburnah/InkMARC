"""Render the ARCore-versus-fallback overlays onto an exported comparison video.

Example:
  python tools/render_arcore_comparison_video.py <capture>/session.mp4 <capture>/planes.ndjson

The ARCore recording is a raw landscape camera stream.  This tool rotates and crops it into the
portrait SurfaceView coordinate system, then replays the rectangles recorded in ``planes.ndjson``:
white is ARCore and orange is the image fallback.  It deliberately renders the original recorded
coordinates--including an off-screen rectangle--so a mismatch is visible rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np


ARCORE_WHITE = (255, 255, 255)
FALLBACK_ORANGE = (0, 140, 255)  # BGR
PANEL = (20, 20, 20)


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be WIDTHxHEIGHT, for example 1080x2400") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def parse_quad(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 8:
        return None
    try:
        points = np.asarray(value, dtype=np.float32).reshape(4, 2)
    except (TypeError, ValueError):
        return None
    return points if np.isfinite(points).all() else None


def read_log(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at line {line_number}: {error}") from error
    if not rows:
        raise SystemExit("No frame rows were found in the NDJSON log.")
    return rows


def portrait_background(frame: np.ndarray, size: tuple[int, int], rotation: str) -> np.ndarray:
    if rotation == "cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "ccw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    target_width, target_height = size
    scale = max(target_width / frame.shape[1], target_height / frame.shape[0])
    scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    x0 = (scaled.shape[1] - target_width) // 2
    y0 = (scaled.shape[0] - target_height) // 2
    return scaled[y0 : y0 + target_height, x0 : x0 + target_width].copy()


def draw_quad(image: np.ndarray, quad: np.ndarray | None, color: tuple[int, int, int], label: str) -> None:
    if quad is None:
        return
    points = np.rint(quad).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], True, color, 5, cv2.LINE_AA)
    for point in points[:, 0]:
        cv2.circle(image, tuple(point), 8, color, -1, cv2.LINE_AA)
    anchor = tuple(points[0, 0])
    cv2.putText(image, label, (anchor[0] + 12, anchor[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_status(image: np.ndarray, row: dict[str, object], index: int, total: int, seconds: float) -> None:
    fallback = row.get("fallback") if isinstance(row.get("fallback"), dict) else {}
    placement = row.get("placement") if isinstance(row.get("placement"), dict) else {}
    comparison = row.get("comparison") if isinstance(row.get("comparison"), dict) else {}
    lines = [
        f"{seconds:05.1f}s   frame {index + 1}/{total}",
        "ARCore: " + str(row.get("cameraTrackingState", "unknown")) +
        f"  planes={row.get('trackedHorizontalPlaneCount', 0)}" +
        f"  placement={placement.get('source') or '-'}",
        "Fallback: " + str(fallback.get("state") or "-") +
        f"  confidence={float(fallback.get('confidence') or 0):.2f}",
        "Outcome: " + str(comparison.get("outcome") or "unknown"),
        "White = ARCore   Orange = fallback",
    ]
    line_height = 33
    width = min(image.shape[1] - 20, 820)
    cv2.rectangle(image, (10, 10), (10 + width, 20 + line_height * len(lines)), PANEL, -1)
    for line_index, line in enumerate(lines):
        cv2.putText(image, line, (22, 42 + line_index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (245, 245, 245), 2, cv2.LINE_AA)


def open_ffmpeg_writer(output: Path, fps: float, size: tuple[int, int]) -> subprocess.Popen[bytes] | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="session.mp4 exported by the comparison capture")
    parser.add_argument("frames", type=Path, help="matching planes.ndjson file")
    parser.add_argument("--output", type=Path, help="default: arcore_fallback_overlay.mp4 beside the source video")
    parser.add_argument("--view-size", type=parse_size, default=(1080, 2400), help="SurfaceView size; default: 1080x2400")
    parser.add_argument("--rotation", choices=("cw", "ccw", "none"), default="cw", help="rotation from raw MP4 to portrait preview; default: cw")
    parser.add_argument(
        "--offset-ms", type=float, default=0.0,
        help="advance the overlay log by this many milliseconds (positive moves overlays forward); default: 0",
    )
    parser.add_argument("--max-frames", type=int, help="render only this many frames, useful for a quick preview")
    args = parser.parse_args()

    rows = read_log(args.frames)
    log_times_ns = [int(row["frameTimestampNs"]) for row in rows]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_count = min(video_count, args.max_frames) if args.max_frames else video_count
    if target_count <= 0:
        raise SystemExit("No video frames were found.")

    output = args.output or args.video.with_name("arcore_fallback_overlay.mp4")
    width, height = args.view_size
    ffmpeg_writer = open_ffmpeg_writer(output, fps, args.view_size)
    opencv_writer = None
    if ffmpeg_writer is None:
        opencv_writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    first_video_time_ms: float | None = None
    try:
        for video_index in range(target_count):
            ok, frame = capture.read()
            if not ok:
                break
            video_time_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if first_video_time_ms is None:
                first_video_time_ms = video_time_ms
            # Session MP4 and onDrawFrame logging run at different rates.  Do not stretch the
            # log over the video by frame index: select the closest ARCore timestamp instead.
            target_time_ns = log_times_ns[0] + int((video_time_ms - first_video_time_ms + args.offset_ms) * 1_000_000)
            insertion = bisect_left(log_times_ns, target_time_ns)
            if insertion <= 0:
                log_index = 0
            elif insertion >= len(rows):
                log_index = len(rows) - 1
            else:
                before = log_times_ns[insertion - 1]
                after = log_times_ns[insertion]
                log_index = insertion - 1 if target_time_ns - before <= after - target_time_ns else insertion
            row = rows[log_index]
            canvas = portrait_background(frame, args.view_size, args.rotation)
            fallback = row.get("fallback") if isinstance(row.get("fallback"), dict) else {}
            placement = row.get("placement") if isinstance(row.get("placement"), dict) else {}
            draw_quad(canvas, parse_quad(placement.get("cornersView")), ARCORE_WHITE, "ARCore")
            draw_quad(canvas, parse_quad(fallback.get("quadView")), FALLBACK_ORANGE, "Fallback")
            draw_status(canvas, row, video_index, target_count, (video_time_ms - first_video_time_ms) / 1000.0)
            if ffmpeg_writer is not None:
                assert ffmpeg_writer.stdin is not None
                ffmpeg_writer.stdin.write(canvas.tobytes())
            else:
                assert opencv_writer is not None
                opencv_writer.write(canvas)
            if video_index % 120 == 0:
                print(f"Rendered {video_index + 1}/{target_count} frames")
        capture.release()
        if ffmpeg_writer is not None:
            assert ffmpeg_writer.stdin is not None
            ffmpeg_writer.stdin.close()
            if ffmpeg_writer.wait() != 0:
                raise SystemExit("FFmpeg could not encode the overlay video.")
            print(f"Wrote {output}")
        else:
            assert opencv_writer is not None
            opencv_writer.release()
            print(f"Wrote {output} (OpenCV fallback encoder)")
    finally:
        capture.release()
        if ffmpeg_writer is not None and ffmpeg_writer.poll() is None:
            ffmpeg_writer.kill()
        if opencv_writer is not None:
            opencv_writer.release()


if __name__ == "__main__":
    main()
