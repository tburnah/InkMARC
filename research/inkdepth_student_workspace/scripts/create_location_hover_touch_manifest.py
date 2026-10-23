#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


DEFAULT_PROBLEM_PARTICIPANTS = {"307", "321"}
STATE_NAME = {1: "hover", 2: "touch"}
STATE_ID = {name: state for state, name in STATE_NAME.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a frame manifest for location fine-tuning from hover/touch frames only."
    )
    parser.add_argument(
        "--manifest",
        default="/data/baseAVIs_432x672/manifest_texture_replacement_v3_no_missing_h5_no_black_participant_splits.json",
        help="Item manifest containing original and texture replacement videos.",
    )
    parser.add_argument("--out", required=True, help="Output frame manifest JSON.")
    parser.add_argument(
        "--variant-kind",
        action="append",
        default=None,
        help="Variant kind to include. Repeatable. Defaults to original and texture_replacement_v3.",
    )
    parser.add_argument(
        "--exclude-participant-id",
        action="append",
        default=[],
        help="Participant id to remove. Defaults also remove 307 and 321 unless --keep-default-problematic is set.",
    )
    parser.add_argument(
        "--keep-default-problematic",
        action="store_true",
        help="Do not automatically remove participants 307 and 321.",
    )
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument(
        "--states",
        nargs="+",
        default=["hover", "touch"],
        help="States to include: hover, touch, 1, or 2.",
    )
    parser.add_argument(
        "--all-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write every kept frame with split=train for final fine-tuning.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Optional stride applied after state filtering. Use 1 for every hover/touch frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional global cap for quick smoke manifests. 0 keeps all frames.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print counts; do not write the manifest.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = [dict(item) for item in manifest.get("items", [])]
    if not items:
        raise ValueError(f"Expected non-empty top-level items in {manifest_path}")

    variant_kinds = set(args.variant_kind or ["original", "texture_replacement_v3"])
    excluded_participants = set(str(value) for value in args.exclude_participant_id)
    if not args.keep_default_problematic:
        excluded_participants |= DEFAULT_PROBLEM_PARTICIPANTS
    include_states = parse_states(args.states)

    selected_items = [
        item
        for item in items
        if str(item.get("variant_kind", "original")) in variant_kinds
        and participant_id(item) not in excluded_participants
    ]

    out_path = Path(args.out)
    writer = None
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = out_path.open("w", encoding="utf-8")
        writer.write('{\n  "frames": [\n')

    summary = {
        "source_manifest": str(manifest_path),
        "variant_kinds": sorted(variant_kinds),
        "excluded_participant_ids": sorted(excluded_participants),
        "states": [STATE_NAME[state] for state in sorted(include_states)],
        "pressure_touch_threshold": float(args.pressure_touch_threshold),
        "all_train": bool(args.all_train),
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
    }
    counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    participant_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    first = True

    try:
        for item_i, item in enumerate(selected_items, start=1):
            for record in iter_hover_touch_records(
                item,
                pressure_touch_threshold=float(args.pressure_touch_threshold),
                include_states=include_states,
                all_train=bool(args.all_train),
                frame_stride=max(1, int(args.frame_stride)),
            ):
                state_name = STATE_NAME[int(record["state_hint"])]
                counts["frames"] += 1
                state_counts[state_name] += 1
                participant_counts[str(record["participant_id"])] += 1
                variant_counts[str(record.get("variant_kind", "original"))] += 1

                if writer is not None:
                    if not first:
                        writer.write(",\n")
                    writer.write("    ")
                    writer.write(json.dumps(record, allow_nan=False, separators=(",", ":")))
                    first = False

                if args.max_frames and counts["frames"] >= int(args.max_frames):
                    break
            if item_i == 1 or item_i % 250 == 0:
                print(f"processed_items={item_i}/{len(selected_items)} frames={counts['frames']}")
            if args.max_frames and counts["frames"] >= int(args.max_frames):
                break
    finally:
        if writer is not None:
            output_summary = {
                **summary,
                "source_items": len(items),
                "selected_items": len(selected_items),
                "frames": int(counts["frames"]),
                "state_counts": dict(sorted(state_counts.items())),
                "participant_counts": dict(sorted(participant_counts.items())),
                "variant_counts": dict(sorted(variant_counts.items())),
            }
            writer.write('\n  ],\n  "location_hover_touch_manifest": ')
            writer.write(json.dumps(output_summary, indent=2, allow_nan=False))
            writer.write("\n}\n")
            writer.close()

    output_summary = {
        **summary,
        "source_items": len(items),
        "selected_items": len(selected_items),
        "frames": int(counts["frames"]),
        "state_counts": dict(sorted(state_counts.items())),
        "participant_counts": dict(sorted(participant_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
    }
    if writer is not None:
        print(f"wrote {out_path}")
    print(json.dumps(output_summary, indent=2, allow_nan=False))


def parse_states(values: Iterable[str]) -> set[int]:
    states: set[int] = set()
    for value in values:
        key = str(value).strip().lower()
        if key.isdigit():
            state = int(key)
        else:
            state = STATE_ID.get(key, -1)
        if state not in STATE_NAME:
            raise ValueError(f"Unsupported state for location fine-tuning: {value}")
        states.add(state)
    return states


def iter_hover_touch_records(
    item: dict[str, Any],
    pressure_touch_threshold: float,
    include_states: set[int],
    all_train: bool,
    frame_stride: int,
) -> Iterable[dict[str, Any]]:
    h5_path = Path(str(item.get("output_h5_path") or item.get("label_h5_path") or ""))
    if not h5_path.exists():
        return
    labels, masks = load_labels_and_masks(str(h5_path))

    n = min(int(item.get("frame_count") or labels.shape[0]), labels.shape[0])
    if n <= 0:
        return
    valid_xy = (
        (masks[:n, 0] != 0)
        & (masks[:n, 1] != 0)
        & np.isfinite(labels[:n, 0])
        & np.isfinite(labels[:n, 1])
    )
    pressure_valid = (masks[:n, 2] != 0) & np.isfinite(labels[:n, 2])
    touch = valid_xy & pressure_valid & (labels[:n, 2] > pressure_touch_threshold)
    states = np.zeros(n, dtype=np.int8)
    states[valid_xy & ~touch] = 1
    states[touch] = 2

    excluded = excluded_frame_indices(item)
    stem = str(item.get("stem") or Path(str(item.get("output_video_path") or item.get("video_path") or "")).stem)
    base_stem = str(item.get("base_stem") or stem)
    video_path = item.get("output_video_path") or item.get("video_path")
    split = "train" if all_train else str(item.get("split", "train"))
    depth_path = item.get("depth_path")
    if depth_path is None:
        depth_path = str(Path(item.get("output_dir", "/data/baseAVIs_432x672")) / "depth" / f"{base_stem}.zarr")
        if not Path(depth_path).exists():
            depth_path = str(Path("/data/baseAVIs_432x672/depth") / f"{base_stem}.zarr")

    for frame_index in range(0, n, frame_stride):
        state = int(states[frame_index])
        if state not in include_states:
            continue
        if frame_index in excluded:
            continue
        yield {
            "video_id": stem,
            "base_stem": base_stem,
            "variant_kind": item.get("variant_kind", "original"),
            "texture_name": item.get("texture_name"),
            "participant_id": participant_id(item),
            "task_id": item.get("task_id"),
            "split": split,
            "frame_index": int(frame_index),
            "depth_index": int(frame_index),
            "video_path": video_path,
            "label_h5_path": str(h5_path),
            "depth_path": depth_path,
            "depth_key": item.get("depth_key", "depth_norm"),
            "state_hint": state,
        }


def participant_id(item: dict[str, Any]) -> str:
    value = item.get("participant_id")
    if value not in {None, ""}:
        return str(value)
    stem = str(item.get("base_stem") or item.get("stem") or Path(str(item.get("output_video_path", ""))).stem)
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "participant":
        return parts[1]
    return "UNKNOWN"


def excluded_frame_indices(item: dict[str, Any]) -> set[int]:
    excluded = {int(idx) for idx in item.get("exclude_frame_indices", []) or []}
    for frame_range in item.get("exclude_frame_ranges", []) or []:
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            raise ValueError(f"exclude_frame_ranges entries must be [start, end): {frame_range}")
        start, end = int(frame_range[0]), int(frame_range[1])
        if end > start:
            excluded.update(range(start, end))
    return excluded


@lru_cache(maxsize=512)
def load_labels_and_masks(h5_path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.float32)
        masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
    return labels, masks


if __name__ == "__main__":
    main()
