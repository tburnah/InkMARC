#!/usr/bin/env python3
"""Evaluate LK plus periodic local SIFT keyframes on an InkMARC recording."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from annotate_session import Mp4Writer, read_session_csv, view_point_to_frame
from experiment_video_tracker import (
    KeyframeSiftTracker,
    SeedEvent,
    SparsePlanarTracker,
    TrackResult,
    canonical_patch,
    draw_quad,
    patch_similarity,
    quad_from_row,
    read_seed_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--descriptor-interval", type=int, default=5)
    parser.add_argument("--promotion-interval", type=int, default=15)
    parser.add_argument("--origin-interval", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.video.with_name(f"{args.video.stem}_keyframe_experiment.mp4")
    result_path = output.with_suffix(".csv")
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

    pure_lk = SparsePlanarTracker()
    keyframe = KeyframeSiftTracker(
        args.descriptor_interval,
        args.promotion_interval,
        args.origin_interval,
    )
    counts = {name: [0 for _ in seeds] for name in ("frames", "phone", "lk", "keyframe")}
    score_sums = {name: [0.0 for _ in seeds] for name in ("phone", "lk", "keyframe")}
    score_counts = {name: [0 for _ in seeds] for name in ("phone", "lk", "keyframe")}
    seed_index = 0
    active_seed = -1
    reference_patch: np.ndarray | None = None
    frame_index = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(output, width, height, fps)
    with result_path.open("w", newline="", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(
            (
                "frame",
                "time_s",
                "seed_index",
                "lk_model",
                "lk_support",
                "lk_similarity",
                "keyframe_model",
                "keyframe_support",
                "keyframe_error",
                "keyframe_reason",
                "keyframe_similarity",
                "keyframe_quad",
                "phone_similarity",
            )
        )
        try:
            while ok:
                time_ns = round(frame_index * 1_000_000_000 / fps)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                just_seeded = False
                while seed_index < len(seeds) and time_ns >= seeds[seed_index].time_ns:
                    active_seed = seed_index
                    lk_result = pure_lk.seed(gray, seeds[seed_index].quad)
                    keyframe_result = keyframe.seed(gray, seeds[seed_index].quad)
                    reference_patch = canonical_patch(gray, seeds[seed_index].quad)
                    seed_index += 1
                    just_seeded = True
                if active_seed < 0:
                    lk_result = TrackResult(None, "waiting", 0, None)
                    keyframe_result = TrackResult(None, "waiting", 0, None)
                elif not just_seeded:
                    lk_result = pure_lk.update(gray)
                    keyframe_result = keyframe.update(gray)

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
                scores = {
                    "phone": patch_similarity(reference_patch, gray, phone_frame_quad),
                    "lk": patch_similarity(reference_patch, gray, lk_result.quad),
                    "keyframe": patch_similarity(reference_patch, gray, keyframe_result.quad),
                }
                if active_seed >= 0:
                    counts["frames"][active_seed] += 1
                    counts["phone"][active_seed] += int(phone_frame_quad is not None)
                    counts["lk"][active_seed] += int(lk_result.quad is not None)
                    counts["keyframe"][active_seed] += int(keyframe_result.quad is not None)
                    for name, score in scores.items():
                        if score is not None:
                            score_sums[name][active_seed] += score
                            score_counts[name][active_seed] += 1

                draw_quad(frame, phone_frame_quad, (0, 179, 255), 3)
                draw_quad(frame, lk_result.quad, (80, 220, 80), 3)
                draw_quad(frame, keyframe_result.quad, (255, 120, 40), 3)
                cv2.putText(frame, "phone", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 179, 255), 2)
                cv2.putText(
                    frame,
                    f"LK {lk_result.model} n={lk_result.support}",
                    (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 220, 80),
                    2,
                )
                cv2.putText(
                    frame,
                    f"KEY {keyframe_result.model} n={keyframe_result.support}",
                    (16, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 120, 40),
                    2,
                )
                writer.write(frame)
                quad_text = (
                    ""
                    if keyframe_result.quad is None
                    else ";".join(f"{value:.3f}" for value in keyframe_result.quad.reshape(-1))
                )
                result_writer.writerow(
                    (
                        frame_index,
                        f"{time_ns / 1e9:.6f}",
                        active_seed,
                        lk_result.model,
                        lk_result.support,
                        "" if scores["lk"] is None else f"{scores['lk']:.6f}",
                        keyframe_result.model,
                        keyframe_result.support,
                        "" if keyframe_result.error is None else f"{keyframe_result.error:.6f}",
                        keyframe_result.reason,
                        "" if scores["keyframe"] is None else f"{scores['keyframe']:.6f}",
                        quad_text,
                        "" if scores["phone"] is None else f"{scores['phone']:.6f}",
                    )
                )
                frame_index += 1
                ok, frame = capture.read()
        finally:
            capture.release()
            writer.close()

    print("attempt,frames,phone_valid,lk_valid,keyframe_valid,phone_similarity,lk_similarity,keyframe_similarity")
    for index in range(len(seeds)):
        average = lambda name: (
            score_sums[name][index] / score_counts[name][index]
            if score_counts[name][index]
            else float("nan")
        )
        print(
            f"{index + 1},{counts['frames'][index]},{counts['phone'][index]},"
            f"{counts['lk'][index]},{counts['keyframe'][index]},"
            f"{average('phone'):.3f},{average('lk'):.3f},{average('keyframe'):.3f}"
        )
    print(f"Wrote {output}")
    print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
