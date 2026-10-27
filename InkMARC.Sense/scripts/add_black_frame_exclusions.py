#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Add black-frame exclusion ranges to a video-item manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--black-max",
        type=int,
        default=2,
        help="A frame is black if every grayscale pixel is <= this value.",
    )
    parser.add_argument(
        "--black-mean",
        type=float,
        default=None,
        help="Optional mean grayscale threshold. Only supported by --method raw.",
    )
    parser.add_argument(
        "--black-picture-ratio",
        type=float,
        default=0.999,
        help="Minimum ratio of black pixels for FFmpeg blackdetect to mark a frame black.",
    )
    parser.add_argument(
        "--method",
        choices=["blackdetect", "raw"],
        default="blackdetect",
        help="blackdetect runs inside FFmpeg and is much faster; raw is exact but slower.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1) // 2)),
        help="Number of videos to scan concurrently.",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="Threads per FFmpeg process. Keep this low when --workers is high.",
    )
    parser.add_argument(
        "--variant-kind",
        action="append",
        default=[],
        help="Only scan items with this variant_kind. Repeatable. Default scans all items.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = list(manifest.get("items", []))
    if not items:
        raise ValueError(f"Expected manifest with non-empty 'items': {manifest_path}")

    allowed_kinds = set(args.variant_kind)
    scan_indices = [
        idx
        for idx, item in enumerate(items)
        if not allowed_kinds or str(item.get("variant_kind", "original")) in allowed_kinds
    ]
    if args.max_items is not None:
        scan_indices = scan_indices[: args.max_items]

    summary = Counter()
    failures: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_idx = {
            executor.submit(
                scan_item,
                items[item_idx],
                args.black_max,
                args.black_mean,
                args.black_picture_ratio,
                args.method,
                args.ffmpeg_threads,
            ): item_idx
            for item_idx in scan_indices
        }
        for future in as_completed(future_to_idx):
            item_idx = future_to_idx[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                item = items[item_idx]
                failures.append(
                    {
                        "stem": item.get("stem"),
                        "video_path": str(item.get("output_video_path") or item.get("video_path") or ""),
                        "error": str(exc),
                    }
                )
                summary["scan_failed"] += 1
                continue

            if result["error"] is not None:
                failures.append(result["error"])
                summary[str(result["error"]["error"])] += 1
                continue

            item = dict(items[item_idx])
            ranges = result["ranges"]
            excluded_frames = int(result["excluded_frames"])
            if ranges:
                if args.keep_existing:
                    ranges = merge_ranges([*item.get("exclude_frame_ranges", []), *ranges])
                item["exclude_frame_ranges"] = ranges
                item["black_frame_exclusion"] = {
                    "method": args.method,
                    "black_max": args.black_max,
                    "black_mean": args.black_mean,
                    "black_picture_ratio": args.black_picture_ratio,
                    "excluded_frames": excluded_frames,
                    "ranges": len(ranges),
                }
                items[item_idx] = item
                summary["items_with_black_frames"] += 1
                summary["black_frames"] += excluded_frames
            else:
                if not args.keep_existing:
                    item.pop("exclude_frame_indices", None)
                    item.pop("exclude_frame_ranges", None)
                    item.pop("black_frame_exclusion", None)
                    items[item_idx] = item
                summary["items_without_black_frames"] += 1

            if completed == 1 or completed % 100 == 0:
                print(
                    f"scanned {completed}/{len(scan_indices)} "
                    f"items_with_black_frames={summary['items_with_black_frames']} "
                    f"black_frames={summary['black_frames']}",
                    file=sys.stderr,
                )

    manifest["items"] = items
    manifest["black_frame_exclusion"] = {
        "source_manifest": str(manifest_path),
        "scanned_items": len(scan_indices),
        "variant_kind_filter": sorted(allowed_kinds),
        "method": args.method,
        "workers": int(args.workers),
        "ffmpeg_threads": int(args.ffmpeg_threads),
        "black_max": args.black_max,
        "black_mean": args.black_mean,
        "black_picture_ratio": args.black_picture_ratio,
        "summary": {k: int(v) for k, v in summary.items()},
        "failures": failures,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(manifest["black_frame_exclusion"], indent=2))


def scan_item(
    item: dict[str, Any],
    black_max: int,
    black_mean: float | None,
    black_picture_ratio: float,
    method: str,
    ffmpeg_threads: int,
) -> dict[str, Any]:
    video_path = Path(str(item.get("output_video_path") or item.get("video_path") or ""))
    if not video_path.exists():
        return {
            "ranges": [],
            "excluded_frames": 0,
            "error": {"stem": item.get("stem"), "video_path": str(video_path), "error": "missing_video"},
        }
    if method == "blackdetect":
        ranges = scan_black_ranges_blackdetect(video_path, black_max, black_picture_ratio, ffmpeg_threads)
        return {"ranges": ranges, "excluded_frames": sum(end - start for start, end in ranges), "error": None}
    black_indices = scan_black_frames_raw(video_path, black_max, black_mean, ffmpeg_threads)
    ranges = indices_to_ranges(black_indices)
    return {"ranges": ranges, "excluded_frames": len(black_indices), "error": None}


def scan_black_ranges_blackdetect(
    video_path: Path,
    black_max: int,
    black_picture_ratio: float,
    ffmpeg_threads: int,
) -> list[list[int]]:
    fps = probe_fps(video_path)
    pix_th = max(0.0, min(1.0, black_max / 255.0))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-threads",
        str(max(1, int(ffmpeg_threads))),
        "-i",
        str(video_path),
        "-vf",
        f"blackdetect=d=0:pix_th={pix_th:.8f}:pic_th={black_picture_ratio:.8f}",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    ranges = []
    for match in re.finditer(r"black_start:([0-9.]+)\s+black_end:([0-9.]+)", proc.stderr):
        start_t = float(match.group(1))
        end_t = float(match.group(2))
        start = max(0, round(start_t * fps))
        end = max(start + 1, round(end_t * fps) + 1)
        ranges.append([start, end])
    return merge_ranges(ranges)


def probe_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    info = json.loads(subprocess.check_output(cmd, text=True))
    stream = info["streams"][0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = fraction_to_float(stream.get(key, "0/0"))
        if fps > 0:
            return fps
    raise RuntimeError(f"Could not determine FPS for {video_path}")


def fraction_to_float(value: str) -> float:
    if "/" not in value:
        return float(value)
    num_s, den_s = value.split("/", 1)
    num, den = float(num_s), float(den_s)
    return 0.0 if den == 0 else num / den


def scan_black_frames_raw(
    video_path: Path,
    black_max: int,
    black_mean: float | None,
    ffmpeg_threads: int = 1,
) -> list[int]:
    width, height = probe_dimensions(video_path)
    frame_size = width * height
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        str(max(1, int(ffmpeg_threads))),
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    black_indices: list[int] = []
    frame_idx = 0
    while True:
        frame = proc.stdout.read(frame_size)
        if not frame:
            break
        if len(frame) != frame_size:
            raise RuntimeError(f"short frame read at frame {frame_idx}: {len(frame)} of {frame_size} bytes")
        max_value = max(frame)
        is_black = max_value <= black_max
        if not is_black and black_mean is not None:
            is_black = (sum(frame) / frame_size) <= black_mean
        if is_black:
            black_indices.append(frame_idx)
        frame_idx += 1
    _stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())
    return black_indices


def probe_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    width_s, height_s = out.split("x", 1)
    return int(width_s), int(height_s)


def indices_to_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append([start, prev + 1])
        start = prev = idx
    ranges.append([start, prev + 1])
    return ranges


def merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    if not ranges:
        return []
    ordered = sorted([int(a), int(b)] for a, b in ranges if int(b) > int(a))
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


if __name__ == "__main__":
    main()
