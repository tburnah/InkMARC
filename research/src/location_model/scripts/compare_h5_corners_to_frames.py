#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


DRAW_CODE = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1])
data = json.loads((out / "corners.json").read_text())
colors = [(0, 255, 255), (0, 200, 0), (255, 0, 255), (255, 160, 0)]

for seq, frame_idx in enumerate(data["frames"], start=1):
    img = cv2.imread(str(out / f"frame_{seq:03d}.png"))
    if img is None:
        raise RuntimeError(f"Could not read extracted frame_{seq:03d}.png")
    pts = np.asarray(data["corners"][str(frame_idx)], dtype=np.float32)
    pts_i = np.round(pts).astype(np.int32)
    cv2.polylines(img, [pts_i.reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
    for j, p in enumerate(pts_i):
        cv2.circle(img, tuple(p), 6, colors[j % len(colors)], -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"c{j}",
            tuple(p + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colors[j % len(colors)],
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        f"frame {frame_idx} H5 corners",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out / f"frame_{frame_idx}_h5_corners.png"), img)

imgs = [cv2.imread(str(out / f"frame_{i}_h5_corners.png")) for i in data["frames"]]
if all(img is not None for img in imgs) and len(imgs) > 1:
    cv2.imwrite(str(out / "h5_corners_side_by_side.png"), np.concatenate(imgs, axis=1))

if len(data["frames"]) >= 2:
    a_idx, b_idx = data["frames"][0], data["frames"][-1]
    img = cv2.imread(str(out / f"frame_{len(data['frames']):03d}.png"))
    p_a = np.asarray(data["corners"][str(a_idx)], dtype=np.float32)
    p_b = np.asarray(data["corners"][str(b_idx)], dtype=np.float32)
    cv2.polylines(img, [np.round(p_a).astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.round(p_b).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2, cv2.LINE_AA)
    for src, dst in zip(p_a, p_b):
        cv2.arrowedLine(
            img,
            tuple(np.round(src).astype(int)),
            tuple(np.round(dst).astype(int)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    cv2.putText(
        img,
        f"cyan=frame {a_idx}, magenta=frame {b_idx}, red=delta",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out / f"frame_{b_idx}_corner_delta_from_{a_idx}.png"), img)
"""


def parse_frames(raw: str) -> list[int]:
    frames = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not frames:
        raise argparse.ArgumentTypeError("expected at least one frame index")
    if any(frame < 0 for frame in frames):
        raise argparse.ArgumentTypeError("frame indexes must be non-negative")
    return frames


def run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def write_corner_data(h5_path: Path, frames: list[int], out_dir: Path, window: tuple[int, int] | None) -> dict:
    with h5py.File(h5_path, "r") as h5:
        if "corners" not in h5:
            raise KeyError(f"{h5_path} does not contain a 'corners' dataset")
        corners_ds = h5["corners"]
        n_frames = int(corners_ds.shape[0])
        for frame in frames:
            if frame >= n_frames:
                raise IndexError(f"frame {frame} is outside H5 corners length {n_frames}")

        corners = {str(frame): np.asarray(corners_ds[frame], dtype=float).tolist() for frame in frames}
        data: dict = {
            "h5_path": str(h5_path),
            "frames": frames,
            "corners_shape": list(corners_ds.shape),
            "corners": corners,
        }

        if len(frames) >= 2:
            a = np.asarray(corners[str(frames[0])], dtype=np.float32)
            b = np.asarray(corners[str(frames[-1])], dtype=np.float32)
            delta = b - a
            norm = np.linalg.norm(delta, axis=1)
            data["delta_last_minus_first"] = delta.astype(float).tolist()
            data["delta_norm_px"] = norm.astype(float).tolist()
            data["max_delta_px"] = float(norm.max())
            data["mean_delta_px"] = float(norm.mean())

        if window is not None:
            start, end = window
            start = max(0, start)
            end = min(n_frames - 1, end)
            if end <= start:
                raise ValueError("--window end must be greater than start")
            c = np.asarray(corners_ds[start : end + 1], dtype=np.float32)
            step = np.linalg.norm(np.diff(c, axis=0), axis=2)
            data["window"] = {
                "start": start,
                "end": end,
                "frame_to_next_max_delta_px": step.max(axis=1).astype(float).tolist(),
                "frame_to_next_mean_delta_px": step.mean(axis=1).astype(float).tolist(),
            }

    (out_dir / "corners.json").write_text(json.dumps(data, indent=2))
    return data


def write_summary(data: dict, out_dir: Path, jump_threshold: float) -> None:
    lines = [
        f"h5={data['h5_path']}",
        f"corners_shape={tuple(data['corners_shape'])}",
    ]
    for frame in data["frames"]:
        lines.append("")
        lines.append(f"frame {frame}")
        for j, (x, y) in enumerate(data["corners"][str(frame)]):
            lines.append(f"  c{j}: x={x:.3f}, y={y:.3f}")

    if "delta_norm_px" in data:
        lines.append("")
        lines.append(f"delta {data['frames'][-1]}-{data['frames'][0]}")
        for j, ((dx, dy), dn) in enumerate(zip(data["delta_last_minus_first"], data["delta_norm_px"])):
            lines.append(f"  c{j}: dx={dx:.3f}, dy={dy:.3f}, norm={dn:.3f}px")
        lines.append(f"max_delta_px={data['max_delta_px']:.3f}")
        lines.append(f"mean_delta_px={data['mean_delta_px']:.3f}")

    if "window" in data:
        w = data["window"]
        lines.append("")
        lines.append(f"window jumps >= {jump_threshold:.3f}px")
        for offset, (mx, mean) in enumerate(
            zip(w["frame_to_next_max_delta_px"], w["frame_to_next_mean_delta_px"]),
            start=int(w["start"]),
        ):
            if mx >= jump_threshold:
                lines.append(f"  {offset}->{offset + 1}: max={mx:.3f}px mean={mean:.3f}px")

    summary = "\n".join(lines) + "\n"
    (out_dir / "corner_summary.txt").write_text(summary)
    print(summary)


def extract_frames(video_path: Path, frames: list[int], out_dir: Path) -> None:
    expr = "+".join(f"eq(n\\,{frame})" for frame in frames)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select='{expr}'",
            "-vsync",
            "0",
            str(out_dir / "frame_%03d.png"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare H5 surface corners on selected video frames.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=parse_frames, help="Comma-separated frame indexes, e.g. 800,820")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--window", type=parse_frames, help="Two comma-separated frame indexes for temporal jump report.")
    parser.add_argument("--jump-threshold", type=float, default=10.0)
    parser.add_argument("--cv2-python", help="Optional Python executable with cv2 installed for overlay rendering.")
    parser.add_argument("--no-extract", action="store_true", help="Skip ffmpeg frame extraction.")
    args = parser.parse_args()

    if args.window is not None and len(args.window) != 2:
        parser.error("--window must contain exactly two frame indexes")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = write_corner_data(args.h5, args.frames, args.out_dir, tuple(args.window) if args.window else None)
    write_summary(data, args.out_dir, float(args.jump_threshold))

    if not args.no_extract:
        extract_frames(args.video, args.frames, args.out_dir)

    if args.cv2_python:
        run([args.cv2_python, "-c", DRAW_CODE, str(args.out_dir)])
    else:
        print("Overlay rendering skipped. Pass --cv2-python with an interpreter that has cv2 installed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
