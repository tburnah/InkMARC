#!/usr/bin/env python3
"""Generate publication figures from an InkMARC recording and its synchronized telemetry."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEM = "InkMARC_20260922_122250"
DEFAULT_VIDEO = ROOT / "app" / "src" / "main" / "assets" / f"{DEFAULT_STEM}.mp4"
DEFAULT_CSV = ROOT / "app" / "src" / "main" / "assets" / f"{DEFAULT_STEM}_sensors.csv"
DEFAULT_MODEL = ROOT / "app" / "src" / "main" / "assets" / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT = ROOT / "reports" / "implementation_figures"

INK = (25, 34, 46)
MUTED = (88, 103, 119)
PAPER = (246, 248, 251)
CARD = (255, 255, 255)
LINE = (218, 225, 232)
TEAL = (0, 183, 195)
CYAN = (0, 211, 255)
AMBER = (245, 166, 35)
MAGENTA = (229, 72, 108)
GREEN = (35, 177, 112)
BLUE = (58, 113, 221)


@dataclass(frozen=True)
class SessionData:
    metadata: dict[str, str]
    rows: list[dict[str, str]]
    camera_rows: list[dict[str, str]]
    camera_times_ns: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["seguisb.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def read_session(path: Path) -> SessionData:
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
    rows = list(csv.DictReader(data_lines))
    camera_rows = [row for row in rows if row.get("event") == "camera_frame"]
    camera_times = [int(row["video_time_ns"]) for row in camera_rows]
    return SessionData(metadata, rows, camera_rows, camera_times)


def nearest_camera_row(session: SessionData, seconds: float) -> dict[str, str]:
    target = round(seconds * 1e9)
    index = bisect.bisect_left(session.camera_times_ns, target)
    if index <= 0:
        return session.camera_rows[0]
    if index >= len(session.camera_rows):
        return session.camera_rows[-1]
    before = session.camera_rows[index - 1]
    after = session.camera_rows[index]
    return before if target - int(before["video_time_ns"]) <= int(after["video_time_ns"]) - target else after


def nearest_sensor_row(session: SessionData, seconds: float, sensor_type: str) -> dict[str, str]:
    candidates = [
        row
        for row in session.rows
        if row.get("event") == "sensor" and row.get("sensor_type") == sensor_type
    ]
    target = round(seconds * 1e9)
    return min(candidates, key=lambda row: abs(int(row["video_time_ns"]) - target))


def video_fps(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if math.isfinite(fps) and fps > 0 else 30.0


def raw_frame_at(video: Path, seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {video}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(seconds * fps)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to decode {video} at {seconds:.3f}s")
    return frame


def raw_frame_range(video: Path, start_seconds: float, end_seconds: float) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {video}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = capture.get(cv2.CAP_PROP_FPS)
    start_index = max(0, round(start_seconds * fps))
    end_index = max(start_index, round(end_seconds * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    frames: list[tuple[int, np.ndarray]] = []
    for frame_index in range(start_index, end_index + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append((frame_index, frame))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video}")
    return frames


def rotate_cw(image: np.ndarray) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def raw_to_display_point(point: Sequence[float], raw_height: int) -> tuple[int, int]:
    return round(raw_height - 1 - point[1]), round(point[0])


def view_point_to_raw(
    x: float,
    y: float,
    metadata: dict[str, str],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    image_width = int(metadata["analysis_image_width"])
    image_height = int(metadata["analysis_image_height"])
    view_width = int(metadata["view_width"])
    view_height = int(metadata["view_height"])
    rotation = int(metadata["analysis_rotation_degrees"]) % 360
    rotated_width = image_height if rotation in (90, 270) else image_width
    rotated_height = image_width if rotation in (90, 270) else image_height
    scale = max(view_width / rotated_width, view_height / rotated_height)
    offset_x = (view_width - rotated_width * scale) * 0.5
    offset_y = (view_height - rotated_height * scale) * 0.5
    rotated_x = (x - offset_x) / scale
    rotated_y = (y - offset_y) / scale
    if rotation == 90:
        raw_x, raw_y = rotated_y, image_height - rotated_x
    elif rotation == 180:
        raw_x, raw_y = image_width - rotated_x, image_height - rotated_y
    elif rotation == 270:
        raw_x, raw_y = image_width - rotated_y, rotated_x
    else:
        raw_x, raw_y = rotated_x, rotated_y
    return (
        round(np.clip(raw_x * frame_width / image_width, 0, frame_width - 1)),
        round(np.clip(raw_y * frame_height / image_height, 0, frame_height - 1)),
    )


def row_quad_raw(row: dict[str, str], session: SessionData, width: int, height: int) -> list[tuple[int, int]]:
    return [
        view_point_to_raw(
            float(row[f"quad_x{index}"]),
            float(row[f"quad_y{index}"]),
            session.metadata,
            width,
            height,
        )
        for index in range(4)
    ]


def grid_balanced_points(gray: np.ndarray, maximum: int = 90) -> np.ndarray:
    detected = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=360,
        qualityLevel=0.003,
        minDistance=6.0,
        blockSize=3,
        useHarrisDetector=False,
        k=0.04,
    )
    if detected is None:
        return np.empty((0, 1, 2), np.float32)
    height, width = gray.shape
    per_cell = math.ceil(maximum / 12)
    counts = [0] * 12
    selected: list[np.ndarray] = []
    for point in detected.reshape(-1, 2):
        column = min(3, max(0, int(point[0] / max(1, width) * 4)))
        row = min(2, max(0, int(point[1] / max(1, height) * 3)))
        cell = row * 4 + column
        if counts[cell] >= per_cell:
            continue
        counts[cell] += 1
        selected.append(point)
        if len(selected) >= maximum:
            break
    return np.asarray(selected, np.float32).reshape(-1, 1, 2)


def track_surface_features(
    frames: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = frames[0][1]
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    current = grid_balanced_points(first_gray)
    initial = current.copy()
    previous_gray = first_gray
    for step, (_, frame) in enumerate(frames[1:], start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(current) == 0:
            break
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            current,
            None,
            winSize=(41, 41),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03),
            flags=0,
            minEigThreshold=1e-4,
        )
        keep = status.reshape(-1).astype(bool) & (error.reshape(-1) <= 40.0)
        if step % 3 == 0 and np.count_nonzero(keep):
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                previous_gray,
                next_points,
                current.copy(),
                winSize=(41, 41),
                maxLevel=5,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                minEigThreshold=1e-4,
            )
            forward_backward = np.linalg.norm(back.reshape(-1, 2) - current.reshape(-1, 2), axis=1)
            keep &= back_status.reshape(-1).astype(bool) & (forward_backward <= 7.0)
        initial = initial[keep].reshape(-1, 1, 2)
        current = next_points[keep].reshape(-1, 1, 2)
        previous_gray = gray
    return first, frames[-1][1], initial.reshape(-1, 2), current.reshape(-1, 2)


def choose_display_tracks(initial: np.ndarray, current: np.ndarray, width: int, height: int, maximum: int = 42):
    displacement = np.linalg.norm(current - initial, axis=1)
    order = np.argsort(-displacement)
    per_cell = math.ceil(maximum / 12)
    counts = [0] * 12
    chosen: list[int] = []
    for index in order:
        point = initial[index]
        column = min(3, max(0, int(point[0] / max(1, width) * 4)))
        row = min(2, max(0, int(point[1] / max(1, height) * 3)))
        cell = row * 4 + column
        if counts[cell] >= per_cell:
            continue
        counts[cell] += 1
        chosen.append(int(index))
        if len(chosen) >= maximum:
            break
    return initial[chosen], current[chosen]


def draw_quad_bgr(image: np.ndarray, points: Sequence[Sequence[float]]) -> None:
    polygon = np.rint(np.asarray(points)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], True, (22, 35, 48), 9, cv2.LINE_AA)
    cv2.polylines(image, [polygon], True, (35, 166, 245), 5, cv2.LINE_AA)
    for point in polygon.reshape(-1, 2):
        cv2.circle(image, tuple(point), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(point), 8, (35, 166, 245), 3, cv2.LINE_AA)


def draw_feature_circle(image: np.ndarray, point: Sequence[float]) -> None:
    center = tuple(np.rint(point).astype(int))
    cv2.circle(image, center, 9, (24, 31, 40), 5, cv2.LINE_AA)
    cv2.circle(image, center, 8, (195, 183, 0), 3, cv2.LINE_AA)
    cv2.circle(image, center, 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_motion_arrow(image: np.ndarray, start: Sequence[float], end: Sequence[float]) -> None:
    first = tuple(np.rint(start).astype(int))
    second = tuple(np.rint(end).astype(int))
    cv2.arrowedLine(image, first, second, (24, 31, 40), 8, cv2.LINE_AA, tipLength=0.28)
    cv2.arrowedLine(image, first, second, (35, 166, 245), 4, cv2.LINE_AA, tipLength=0.28)
    cv2.circle(image, second, 7, (24, 31, 40), -1, cv2.LINE_AA)
    cv2.circle(image, second, 4, (255, 230, 90), -1, cv2.LINE_AA)


def pil_from_bgr(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def bgr_from_pil(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def rounded_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 26) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=CARD, outline=LINE, width=2)


def paste_contain(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int], fill=(18, 23, 31)):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), fill)
    panel.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    canvas.paste(panel, (left, top))


def crop_vertical(image: np.ndarray, top: int = 220, bottom: int = 1200) -> np.ndarray:
    top = max(0, min(image.shape[0] - 1, top))
    bottom = max(top + 1, min(image.shape[0], bottom))
    return image[top:bottom]


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))


def make_surface_feature_figure(video: Path, session: SessionData, output: Path) -> None:
    start_seconds = 18.682
    end_seconds = 18.849
    frames = raw_frame_range(video, start_seconds, end_seconds)
    first, last, initial, current = track_surface_features(frames)
    initial, current = choose_display_tracks(initial, current, first.shape[1], first.shape[0])
    first_overlay = first.copy()
    last_overlay = last.copy()
    first_row = nearest_camera_row(session, start_seconds)
    last_row = nearest_camera_row(session, end_seconds)
    draw_quad_bgr(first_overlay, row_quad_raw(first_row, session, first.shape[1], first.shape[0]))
    draw_quad_bgr(last_overlay, row_quad_raw(last_row, session, last.shape[1], last.shape[0]))
    for point in initial:
        draw_feature_circle(first_overlay, point)
    for before, after in zip(initial, current):
        draw_motion_arrow(last_overlay, before, after)
    first_display = crop_vertical(rotate_cw(first_overlay))
    last_display = crop_vertical(rotate_cw(last_overlay))

    canvas = Image.new("RGB", (1880, 1280), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 45), "Track surface features", font=font(54, True), fill=INK)
    draw.text(
        (70, 112),
        "Shi–Tomasi features are seeded across the surface, then propagated with pyramidal Lucas–Kanade flow.",
        font=font(25),
        fill=MUTED,
    )
    boxes = [(60, 178, 915, 1185), (965, 178, 1820, 1185)]
    titles = [
        f"A   Detect + seed  ·  t = {int(first_row['video_time_ns']) / 1e9:.3f} s",
        f"B   Track movement  ·  t = {int(last_row['video_time_ns']) / 1e9:.3f} s",
    ]
    subtitles = [
        f"{len(initial)} representative tracks circled  ·  amber outline = tracked surface",
        "Arrows show measured image-space displacement; dots are current feature positions",
    ]
    for box, image, title, subtitle in zip(boxes, [first_display, last_display], titles, subtitles):
        rounded_card(draw, box)
        draw.text((box[0] + 28, box[1] + 24), title, font=font(29, True), fill=INK)
        draw.text((box[0] + 28, box[1] + 66), subtitle, font=font(20), fill=MUTED)
        paste_contain(canvas, pil_from_bgr(image), (box[0] + 24, box[1] + 108, box[2] - 24, box[3] - 24))
    draw.text(
        (70, 1224),
        "Recreated from the recorded frames with the app’s feature detector and LK parameters.",
        font=font(20),
        fill=MUTED,
    )
    save_png(canvas, output)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (17, 0),
]


def detect_hand_sequence(
    video: Path,
    model: Path,
    target_seconds: float,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    fps = video_fps(video)
    frames = raw_frame_range(video, target_seconds - 1.35, target_seconds)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        # The recorded hand is clipped by the frame edge. A relaxed offline threshold recovers the
        # same 21-landmark geometry so the implementation stages can be visualized.
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    detected: list[np.ndarray] = []
    target_frame = rotate_cw(frames[-1][1])
    with vision.HandLandmarker.create_from_options(options) as detector:
        for frame_index, frame in frames:
            inference_frame = rotate_cw(frame)
            if crop is not None:
                x0, y0, x1, y1 = crop
                inference_frame = inference_frame[y0:y1, x0:x1]
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = round(frame_index * 1000 / fps)
            result = detector.detect_for_video(image, timestamp_ms)
            if result.hand_landmarks:
                points = np.asarray(
                    [
                        [
                            (crop[0] if crop else 0) +
                              np.clip(landmark.x, 0.0, 1.0) * (inference_frame.shape[1] - 1),
                            (crop[1] if crop else 0) +
                              np.clip(landmark.y, 0.0, 1.0) * (inference_frame.shape[0] - 1),
                        ]
                        for landmark in result.hand_landmarks[0]
                    ],
                    dtype=np.float32,
                )
                detected.append(points)
    if not detected:
        raise RuntimeError("MediaPipe did not detect the hand in the selected recording interval")
    return target_frame, detected[-2:]


def extend_hull_past_wrist(mask: np.ndarray, points: np.ndarray) -> None:
    height, width = mask.shape
    wrist = points[0]
    palm_center = points[[5, 9, 13, 17]].mean(axis=0)
    direction = wrist - palm_center
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return
    unit = direction / length
    distances: list[float] = []
    if unit[0] > 1e-6:
        distances.append(((width - 1) - wrist[0]) / unit[0])
    if unit[0] < -1e-6:
        distances.append((0 - wrist[0]) / unit[0])
    if unit[1] > 1e-6:
        distances.append(((height - 1) - wrist[1]) / unit[1])
    if unit[1] < -1e-6:
        distances.append((0 - wrist[1]) / unit[1])
    distance_to_boundary = min(value for value in distances if value >= 0)
    extension = min(distance_to_boundary, min(width, height) * 0.45)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    thickness = min(280, round(max(56.0, palm_width * 0.65)))
    endpoint = np.clip(wrist + unit * extension, [0, 0], [width - 1, height - 1])
    cv2.line(mask, tuple(np.rint(wrist).astype(int)), tuple(np.rint(endpoint).astype(int)), 255, thickness, cv2.LINE_AA)


def hand_hull_and_mask(shape: tuple[int, int], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    hull_mask = np.zeros((height, width), np.uint8)
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.fillConvexPoly(hull_mask, hull, 255, cv2.LINE_AA)
    extend_hull_past_wrist(hull_mask, points)
    padding = int(np.clip(round(min(width, height) * 0.075), 28, 140))
    kernel_size = max(3, padding * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return hull_mask, cv2.dilate(hull_mask, kernel)


def estimate_background_transform(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(previous, 500, 0.01, 8)
    if points is None:
        return np.eye(3, dtype=np.float64)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )
    valid = status.reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    destination = tracked.reshape(-1, 2)[valid]
    if len(source) < 8:
        return np.eye(3, dtype=np.float64)
    homography, _ = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
    return homography if homography is not None else np.eye(3, dtype=np.float64)


def dynamic_residual_mask(frames: Iterable[np.ndarray]) -> np.ndarray:
    previous: np.ndarray | None = None
    persistent: np.ndarray | None = None
    output: np.ndarray | None = None
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    decay_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    full_shape: tuple[int, int] | None = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_shape = gray.shape
        scaled = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if previous is None:
            previous = scaled
            continue
        transform = estimate_background_transform(previous, scaled)
        size = (scaled.shape[1], scaled.shape[0])
        warped_previous = cv2.warpPerspective(previous, transform, size, flags=cv2.INTER_LINEAR)
        valid_source = np.full_like(previous, 255)
        valid = cv2.warpPerspective(valid_source, transform, size, flags=cv2.INTER_NEAREST)
        valid = cv2.threshold(valid, 250, 255, cv2.THRESH_BINARY)[1]
        previous_blur = cv2.GaussianBlur(warped_previous, (7, 7), 0)
        current_blur = cv2.GaussianBlur(scaled, (7, 7), 0)
        valid_pixels = valid.astype(bool)
        brightness_delta = 0.0
        if np.any(valid_pixels):
            brightness_delta = float(np.mean(current_blur[valid_pixels]) - np.mean(previous_blur[valid_pixels]))
        brightness_delta = float(np.clip(brightness_delta, -32.0, 32.0))
        adjusted = np.clip(previous_blur.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)
        difference = cv2.absdiff(current_blur, adjusted)
        raw = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)[1]
        raw = cv2.bitwise_and(raw, valid)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        filtered = np.zeros_like(raw)
        minimum_area = raw.shape[0] * raw.shape[1] * 0.0025
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
                filtered[labels == label] = 255
        filtered = cv2.dilate(filtered, dilate_kernel)
        if persistent is not None:
            warped_persistent = cv2.warpPerspective(persistent, transform, size, flags=cv2.INTER_NEAREST)
            warped_persistent = cv2.erode(warped_persistent, decay_kernel)
            filtered = cv2.bitwise_or(filtered, warped_persistent)
        filtered = cv2.bitwise_and(filtered, valid)
        area_ratio = np.count_nonzero(filtered) / filtered.size
        if 0.0 < area_ratio <= 0.52:
            persistent = filtered.copy()
            output = filtered
        else:
            persistent = None
            output = np.zeros_like(filtered)
        previous = scaled
    if output is None or full_shape is None:
        raise RuntimeError("Unable to reconstruct the dynamic residual mask")
    return cv2.resize(output, (full_shape[1], full_shape[0]), interpolation=cv2.INTER_NEAREST)


def mask_overlay(base: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: float = 0.58) -> np.ndarray:
    result = (base.astype(np.float32) * 0.72).astype(np.uint8)
    color = np.asarray(color_rgb[::-1], np.float32)
    selected = mask > 0
    result[selected] = np.clip(result[selected].astype(np.float32) * (1 - alpha) + color * alpha, 0, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, tuple(int(value) for value in color_rgb[::-1]), 4, cv2.LINE_AA)
    return result


def landmark_overlay(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    result = base.copy()
    for first, second in HAND_CONNECTIONS:
        a = tuple(np.rint(points[first]).astype(int))
        b = tuple(np.rint(points[second]).astype(int))
        cv2.line(result, a, b, (28, 36, 45), 8, cv2.LINE_AA)
        cv2.line(result, a, b, (255, 211, 0), 4, cv2.LINE_AA)
    for index, point in enumerate(points):
        center = tuple(np.rint(point).astype(int))
        radius = 8 if index in (0, 4, 8, 12, 16, 20) else 6
        cv2.circle(result, center, radius + 3, (28, 36, 45), -1, cv2.LINE_AA)
        cv2.circle(result, center, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result, center, radius - 2, (255, 211, 0), -1, cv2.LINE_AA)
    return result


def combined_overlay(base: np.ndarray, hand_mask: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
    result = (base.astype(np.float32) * 0.58).astype(np.uint8)
    hand = hand_mask > 0
    motion = motion_mask > 0
    overlap = hand & motion
    hand_only = hand & ~motion
    motion_only = motion & ~hand
    result[hand_only] = np.asarray(TEAL[::-1], np.uint8)
    result[motion_only] = np.asarray(AMBER[::-1], np.uint8)
    result[overlap] = np.asarray(MAGENTA[::-1], np.uint8)
    union = cv2.bitwise_or(hand_mask, motion_mask)
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
    return result


def make_hand_pipeline_figure(video: Path, model: Path, session: SessionData, output: Path) -> None:
    target_seconds = 22.403
    target_frame, landmark_sets = detect_hand_sequence(
        video,
        model,
        target_seconds,
        crop=(40, 320, 720, 1280),
    )
    points = landmark_sets[-1]
    hull, current_hand_mask = hand_hull_and_mask(target_frame.shape[:2], points)
    hand_mask = current_hand_mask.copy()
    if len(landmark_sets) > 1:
        _, previous_hand_mask = hand_hull_and_mask(target_frame.shape[:2], landmark_sets[-2])
        hand_mask = cv2.bitwise_or(previous_hand_mask, current_hand_mask)
    decoded_dynamic_frames = [
        frame for _, frame in raw_frame_range(video, target_seconds - 0.80, target_seconds)
    ]
    # The live tracker was processing this interval at roughly 10 Hz while the recorder was near
    # 30 fps. Match that cadence so residuals span the same inter-frame motion as the phone path.
    dynamic_frames = decoded_dynamic_frames[::3]
    if dynamic_frames[-1] is not decoded_dynamic_frames[-1]:
        dynamic_frames.append(decoded_dynamic_frames[-1])
    dynamic_frames = [rotate_cw(frame) for frame in dynamic_frames]
    motion_mask = dynamic_residual_mask(dynamic_frames)

    input_display = target_frame
    landmarks_display = landmark_overlay(target_frame, points)
    hull_display = mask_overlay(target_frame, hull, TEAL, 0.60)
    motion_display = mask_overlay(target_frame, motion_mask, AMBER, 0.62)
    combined_display = combined_overlay(target_frame, hand_mask, motion_mask)

    row = nearest_camera_row(session, target_seconds)
    recorded_motion_ratio = float(row["tracker_motion_mask_area_ratio"] or 0)
    canvas = Image.new("RGB", (2320, 1120), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 38), "Hand and shadow exclusion pipeline", font=font(53, True), fill=INK)
    draw.text(
        (64, 105),
        "A recorded frame is converted into landmark, geometric, residual-motion, and final exclusion representations.",
        font=font(25),
        fill=MUTED,
    )
    panels = [input_display, landmarks_display, hull_display, motion_display, combined_display]
    titles = [
        "A  Camera frame",
        "B  MediaPipe landmarks",
        "C  Hand hull + forearm",
        "D  Dynamic residual / shadow",
        "E  Combined exclusion (OR)",
    ]
    subtitles = [
        f"t = {int(row['video_time_ns']) / 1e9:.3f} s",
        "21 tracked keypoints",
        "convex fill before dilation",
        f"recorded mask coverage ≈ {recorded_motion_ratio * 100:.1f}%",
        "cyan = hand  ·  amber = residual",
    ]
    margin, gap, panel_width = 54, 24, 422
    for index, (panel, title, subtitle) in enumerate(zip(panels, titles, subtitles)):
        left = margin + index * (panel_width + gap)
        box = (left, 170, left + panel_width, 1030)
        rounded_card(draw, box, 24)
        draw.text((left + 20, 194), title, font=font(27, True), fill=INK)
        draw.text((left + 20, 234), subtitle, font=font(18), fill=MUTED)
        paste_contain(canvas, pil_from_bgr(panel), (left + 16, 278, left + panel_width - 16, 1008))
    draw.text(
        (64, 1072),
        "MediaPipe was applied to a hand-focused crop of display-oriented recorded frames; the app’s hull, forearm, dilation, threshold, morphology, and temporal-union parameters are retained.",
        font=font(19),
        fill=MUTED,
    )
    save_png(canvas, output)


def gradient_background(width: int, height: int) -> np.ndarray:
    top = np.asarray((23, 31, 45), np.float32)
    bottom = np.asarray((7, 12, 23), np.float32)
    gradient = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        mix = y / max(1, height - 1)
        rgb = top * (1 - mix) + bottom * mix
        gradient[y, :, :] = rgb[::-1]
    return gradient


def warp_phone_screen(screen: np.ndarray, canvas: np.ndarray, quad: np.ndarray) -> None:
    height, width = screen.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped = cv2.warpPerspective(screen, transform, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.warpPerspective(np.full((height, width), 255, np.uint8), transform, (canvas.shape[1], canvas.shape[0]))
    selected = mask > 0
    canvas[selected] = warped[selected]


def draw_dashed_line(image: np.ndarray, start: np.ndarray, end: np.ndarray, color, width: int = 3) -> None:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1:
        return
    direction = vector / length
    distance = 0.0
    while distance < length:
        first = start + direction * distance
        second = start + direction * min(length, distance + 13)
        cv2.line(image, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)), color, width, cv2.LINE_AA)
        distance += 23


def make_gravity_figure(video: Path, session: SessionData, output: Path) -> None:
    seconds = 18.782
    screen = rotate_cw(raw_frame_at(video, seconds))
    gravity_row = nearest_sensor_row(session, seconds, "android.sensor.gravity")
    gravity = np.asarray([float(gravity_row[f"v{index}"]) for index in range(3)], np.float64)
    unit = gravity / np.linalg.norm(gravity)
    canvas = gradient_background(2000, 1200)
    # A shallow extrusion and perspective screen make the device frame explicit.
    outer = np.asarray([[245, 184], [1048, 302], [944, 1052], [132, 891]], np.int32)
    side = np.asarray([[1048, 302], [1078, 324], [974, 1075], [944, 1052]], np.int32)
    bottom = np.asarray([[132, 891], [944, 1052], [974, 1075], [151, 913]], np.int32)
    cv2.fillConvexPoly(canvas, side, (36, 43, 55), cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, bottom, (22, 27, 36), cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, outer, (5, 7, 11), cv2.LINE_AA)
    screen_quad = np.asarray([[276, 221], [1007, 327], [911, 1001], [169, 857]], np.float32)
    warp_phone_screen(screen, canvas, screen_quad)
    cv2.polylines(canvas, [outer.reshape(-1, 1, 2)], True, (76, 88, 105), 4, cv2.LINE_AA)
    cv2.polylines(canvas, [np.rint(screen_quad).astype(np.int32).reshape(-1, 1, 2)], True, (12, 16, 23), 6, cv2.LINE_AA)
    cv2.ellipse(canvas, (643, 286), (34, 6), 8, 0, 360, (18, 22, 29), -1, cv2.LINE_AA)

    origin = np.asarray([610.0, 650.0])
    ex = np.asarray([0.989, 0.148])
    ey_down = np.asarray([-0.135, 0.991])
    ey_up = -ey_down
    ez_out = np.asarray([-0.46, -0.74])
    scale = 430.0
    plane_tip = origin + scale * (unit[0] * ex + unit[1] * ey_up)
    tip = plane_tip + scale * unit[2] * ez_out
    draw_dashed_line(canvas, plane_tip, tip, (165, 178, 195), 3)
    cv2.circle(canvas, tuple(np.rint(plane_tip).astype(int)), 10, (165, 178, 195), 2, cv2.LINE_AA)
    cv2.arrowedLine(canvas, tuple(np.rint(origin).astype(int)), tuple(np.rint(tip).astype(int)), (18, 24, 33), 22, cv2.LINE_AA, tipLength=0.14)
    cv2.arrowedLine(canvas, tuple(np.rint(origin).astype(int)), tuple(np.rint(tip).astype(int)), (35, 166, 245), 12, cv2.LINE_AA, tipLength=0.14)
    cv2.circle(canvas, tuple(np.rint(origin).astype(int)), 17, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(np.rint(origin).astype(int)), 10, (35, 166, 245), -1, cv2.LINE_AA)

    image = pil_from_bgr(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((68, 45), "Gravity direction in the device frame", font=font(52, True), fill=(245, 248, 252))
    draw.text(
        (70, 112),
        "The recorded gravity vector is transformed from phone coordinates into the AR world frame.",
        font=font(25),
        fill=(180, 192, 208),
    )
    draw.rounded_rectangle((1180, 215, 1910, 1015), radius=34, fill=(248, 250, 253), outline=(65, 79, 98), width=2)
    draw.text((1240, 275), "RECORDED SAMPLE", font=font(20, True), fill=BLUE)
    draw.text((1240, 318), f"t = {int(gravity_row['video_time_ns']) / 1e9:.3f} s", font=font(31, True), fill=INK)
    draw.line((1240, 375, 1848, 375), fill=LINE, width=2)
    draw.text((1240, 420), "Device gravity", font=font(24, True), fill=INK)
    draw.text(
        (1240, 466),
        f"g = [{gravity[0]:+.2f}, {gravity[1]:+.2f}, {gravity[2]:+.2f}] m/s²",
        font=font(28),
        fill=INK,
    )
    draw.text((1240, 532), "Normalized direction", font=font(24, True), fill=INK)
    draw.text(
        (1240, 578),
        f"ĝ = [{unit[0]:+.3f}, {unit[1]:+.3f}, {unit[2]:+.3f}]",
        font=font(28),
        fill=INK,
    )
    draw.rounded_rectangle((1238, 662, 1850, 900), radius=22, fill=(234, 240, 248))
    draw.text((1272, 700), "Renderer conversion", font=font(24, True), fill=BLUE)
    draw.text((1272, 752), "worldDown = normalize(R · ĝ)", font=font(27), fill=INK)
    draw.text((1272, 805), "worldUp = −worldDown", font=font(27), fill=INK)
    draw.text((1272, 862), "R = device-to-world rotation", font=font(20), fill=MUTED)
    tip_tuple = tuple(np.rint(tip).astype(int))
    draw.rounded_rectangle((tip_tuple[0] - 48, tip_tuple[1] - 58, tip_tuple[0] + 96, tip_tuple[1] - 8), radius=16, fill=(245, 166, 35))
    draw.text((tip_tuple[0] - 25, tip_tuple[1] - 53), "gravity g", font=font(22, True), fill=(22, 27, 35))
    draw.text((70, 1138), "Phone screen: recorded camera frame. Arrow orientation: synchronized Android gravity sample.", font=font(20), fill=(154, 168, 186))
    save_png(image, output)


def crop_for_confidence(frame: np.ndarray) -> np.ndarray:
    display = rotate_cw(frame)
    return display[215:1215]


def make_tracking_confidence_figure(video: Path, session: SessionData, output: Path) -> None:
    seconds = 18.782
    row = nearest_camera_row(session, seconds)
    actual_seconds = int(row["video_time_ns"]) / 1e9
    raw = raw_frame_at(video, actual_seconds)
    annotated = raw.copy()
    draw_quad_bgr(annotated, row_quad_raw(row, session, raw.shape[1], raw.shape[0]))
    photo = crop_for_confidence(annotated)
    confidence = float(row["tracker_confidence"])
    inliers = int(row["tracker_inliers"])
    candidates = int(row["tracker_candidate_tracks"])
    error = float(row["tracker_reprojection_error_px"])
    count_score = min(1.0, inliers / 10.0)
    ratio_score = min(1.0, inliers / max(1, candidates))
    error_score = 1.0 - min(1.0, error / 6.0)

    canvas = Image.new("RGB", (1900, 1200), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 43), "Tracking confidence", font=font(54, True), fill=INK)
    draw.text(
        (65, 110),
        "A recorded high-confidence frame and the exact confidence components emitted by SurfaceTrackerCore.",
        font=font(25),
        fill=MUTED,
    )
    rounded_card(draw, (55, 170, 1040, 1120), 30)
    draw.text((86, 196), f"Recorded frame  ·  t = {actual_seconds:.3f} s", font=font(28, True), fill=INK)
    draw.text((86, 238), "Amber outline = current tracked surface", font=font(20), fill=MUTED)
    paste_contain(canvas, pil_from_bgr(photo), (80, 284, 1015, 1090))

    rounded_card(draw, (1080, 170, 1845, 1120), 30)
    draw.text((1124, 205), "RECORDED CONFIDENCE", font=font(20, True), fill=GREEN)
    gauge_box = (1283, 250, 1643, 610)
    draw.ellipse(gauge_box, outline=(226, 232, 238), width=32)
    draw.arc(gauge_box, start=-90, end=-90 + 360 * confidence, fill=GREEN, width=32)
    draw.text((1463, 346), f"{confidence:.3f}", font=font(60, True), fill=INK, anchor="mm")
    draw.text((1463, 431), f"{confidence * 100:.1f}%", font=font(31, True), fill=GREEN, anchor="mm")
    draw.text((1463, 476), row["tracker_state"].upper(), font=font(20, True), fill=MUTED, anchor="mm")

    draw.line((1124, 646, 1801, 646), fill=LINE, width=2)
    entries = [
        ("Inlier support", count_score, "50 / minimum 10", 0.40),
        ("Inlier ratio", ratio_score, f"{inliers} / {candidates}", 0.40),
        ("Reprojection quality", error_score, f"1 − {error:.3f} / 6 px", 0.20),
    ]
    y = 684
    for label, score, detail, weight in entries:
        draw.text((1125, y), label, font=font(23, True), fill=INK)
        draw.text((1788, y + 2), f"{score:.3f}", font=font(22, True), fill=INK, anchor="ra")
        draw.rounded_rectangle((1125, y + 42, 1788, y + 62), radius=10, fill=(226, 232, 238))
        draw.rounded_rectangle((1125, y + 42, 1125 + round(663 * score), y + 62), radius=10, fill=GREEN)
        draw.text((1125, y + 73), f"{detail}  ·  weight {weight:.0%}", font=font(18), fill=MUTED)
        y += 116
    draw.rounded_rectangle((1122, 1036, 1802, 1088), radius=16, fill=(233, 245, 239))
    formula = f"0.4×{count_score:.3f} + 0.4×{ratio_score:.3f} + 0.2×{error_score:.3f} = {confidence:.3f}"
    draw.text((1462, 1062), formula, font=font(21, True), fill=(23, 116, 75), anchor="mm")
    draw.text(
        (65, 1150),
        f"Motion model: {row['tracker_motion_model']}  ·  inliers: {inliers}/{candidates}  ·  mean reprojection error: {error:.3f} px  ·  coverage: {float(row['tracker_current_coverage']):.3f}",
        font=font(20),
        fill=MUTED,
    )
    save_png(canvas, output)


def main() -> int:
    args = parse_args()
    session = read_session(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "01_surface_feature_tracking.png",
        args.output_dir / "02_hand_exclusion_pipeline.png",
        args.output_dir / "03_gravity_direction.png",
        args.output_dir / "04_tracking_confidence.png",
    ]
    make_surface_feature_figure(args.video, session, outputs[0])
    make_hand_pipeline_figure(args.video, args.hand_model, session, outputs[1])
    make_gravity_figure(args.video, session, outputs[2])
    make_tracking_confidence_figure(args.video, session, outputs[3])
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
