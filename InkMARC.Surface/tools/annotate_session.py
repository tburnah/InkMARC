#!/usr/bin/env python3
"""Render InkMARC's synchronized tracked quad over a recorded session MP4."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class CameraMarker:
    video_time_ns: int
    quad: list[tuple[float, float]] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an annotated MP4 from an InkMARC raw video and sensor CSV.",
    )
    parser.add_argument("video", type=Path, help="Raw InkMARC MP4")
    parser.add_argument("csv", type=Path, help="Matching *_sensors.csv")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output MP4 (default: <video>_annotated.mp4)",
    )
    return parser.parse_args()


def read_session_csv(path: Path) -> tuple[dict[str, str], list[CameraMarker]]:
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

    if not data_lines:
        raise ValueError(f"No CSV rows found in {path}")

    markers: list[CameraMarker] = []
    for row in csv.DictReader(data_lines):
        if row.get("event") != "camera_frame":
            continue
        quad_values = [row.get(f"quad_{axis}{index}", "") for index in range(4) for axis in ("x", "y")]
        quad = None
        if all(value not in (None, "") for value in quad_values):
            quad = [
                (float(quad_values[index]), float(quad_values[index + 1]))
                for index in range(0, 8, 2)
            ]
        markers.append(CameraMarker(int(row["video_time_ns"]), quad))

    if not markers:
        raise ValueError(f"No camera_frame rows found in {path}")
    markers.sort(key=lambda marker: marker.video_time_ns)
    return metadata, markers


def required_int(metadata: dict[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except KeyError as error:
        raise ValueError(
            f"CSV is missing '# {key}=...'. Record a new session with the updated app.",
        ) from error


def nearest_marker(markers: list[CameraMarker], marker_times: list[int], target_ns: int) -> CameraMarker:
    index = bisect.bisect_left(marker_times, target_ns)
    if index <= 0:
        return markers[0]
    if index >= len(markers):
        return markers[-1]
    before = markers[index - 1]
    after = markers[index]
    return before if target_ns - before.video_time_ns <= after.video_time_ns - target_ns else after


def view_point_to_frame(
    x: float,
    y: float,
    metadata: dict[str, str],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    image_width = required_int(metadata, "analysis_image_width")
    image_height = required_int(metadata, "analysis_image_height")
    view_width = required_int(metadata, "view_width")
    view_height = required_int(metadata, "view_height")
    rotation = required_int(metadata, "analysis_rotation_degrees") % 360

    rotated_width = image_height if rotation in (90, 270) else image_width
    rotated_height = image_width if rotation in (90, 270) else image_height
    scale = max(view_width / rotated_width, view_height / rotated_height)
    offset_x = (view_width - rotated_width * scale) * 0.5
    offset_y = (view_height - rotated_height * scale) * 0.5
    rotated_x = (x - offset_x) / scale
    rotated_y = (y - offset_y) / scale

    frame_aspect = frame_width / frame_height
    rotated_aspect = rotated_width / rotated_height
    raw_aspect = image_width / image_height
    decoder_applied_orientation = abs(math.log(frame_aspect / rotated_aspect)) <= abs(
        math.log(frame_aspect / raw_aspect),
    )

    if decoder_applied_orientation:
        frame_x = rotated_x * frame_width / rotated_width
        frame_y = rotated_y * frame_height / rotated_height
    else:
        if rotation == 90:
            raw_x, raw_y = rotated_y, image_height - rotated_x
        elif rotation == 180:
            raw_x, raw_y = image_width - rotated_x, image_height - rotated_y
        elif rotation == 270:
            raw_x, raw_y = image_width - rotated_y, rotated_x
        else:
            raw_x, raw_y = rotated_x, rotated_y
        frame_x = raw_x * frame_width / image_width
        frame_y = raw_y * frame_height / image_height

    return (
        round(max(0.0, min(frame_width - 1.0, frame_x))),
        round(max(0.0, min(frame_height - 1.0, frame_y))),
    )


class Mp4Writer:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.cv_writer: cv2.VideoWriter | None = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            command = [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.8f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Unable to open MP4 writer; install FFmpeg or an OpenCV MP4 codec")
            self.cv_writer = writer

    def write(self, frame) -> None:
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.write(frame.tobytes())
        else:
            assert self.cv_writer is not None
            self.cv_writer.write(frame)

    def close(self) -> None:
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            return_code = self.process.wait()
            if return_code != 0:
                raise RuntimeError(f"FFmpeg exited with status {return_code}")
        elif self.cv_writer is not None:
            self.cv_writer.release()


def main() -> int:
    args = parse_args()
    output = args.output or args.video.with_name(f"{args.video.stem}_annotated.mp4")
    metadata, markers = read_session_csv(args.csv)
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
        capture.release()
        raise RuntimeError(f"No frames found in {args.video}")
    height, width = frame.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(output, width, height, fps)
    frame_index = 0
    try:
        while ok:
            frame_time_ns = round(frame_index * 1_000_000_000 / fps)
            marker = nearest_marker(markers, marker_times, frame_time_ns)
            if marker.quad is not None:
                points = [view_point_to_frame(x, y, metadata, width, height) for x, y in marker.quad]
                for index, point in enumerate(points):
                    cv2.line(frame, point, points[(index + 1) % len(points)], (0, 179, 255), 5, cv2.LINE_AA)
                    cv2.circle(frame, point, 7, (225, 208, 77), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"InkMARC  t={frame_time_ns / 1e9:.3f}s",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
            ok, frame = capture.read()
    finally:
        capture.release()
        writer.close()

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
