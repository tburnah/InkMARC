#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


STATE_NAME = {0: "background", 1: "hover", 2: "touch"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact balanced frame-level manifest from video items.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--train-per-state", type=int, default=120000)
    parser.add_argument("--val-per-state", type=int, default=20000)
    parser.add_argument("--test-per-state", type=int, default=20000)
    parser.add_argument(
        "--max-per-video-state",
        type=int,
        default=0,
        help="Optional cap per video item per state before global sampling. 0 disables.",
    )
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument("--include-splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument(
        "--variant-kinds",
        nargs="*",
        default=None,
        help="Optional item variant_kind filter, e.g. original or texture_replacement_v3.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = list(manifest.get("items", []))
    if not items:
        raise ValueError(f"Expected manifest with non-empty 'items': {manifest_path}")

    targets = {
        "train": int(args.train_per_state),
        "val": int(args.val_per_state),
        "test": int(args.test_per_state),
    }
    include_splits = set(args.include_splits)
    variant_kinds = None if args.variant_kinds is None else {str(value) for value in args.variant_kinds}
    reservoirs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[tuple[str, int]] = Counter()
    skipped = Counter()

    for item_i, item in enumerate(items, start=1):
        split = str(item.get("split", "unknown"))
        if split not in include_splits:
            continue
        if variant_kinds is not None and str(item.get("variant_kind", "original")) not in variant_kinds:
            skipped["variant_kind_filtered"] += 1
            continue
        h5_path = Path(str(item.get("output_h5_path") or item.get("label_h5_path") or ""))
        if not h5_path.exists():
            skipped["missing_h5"] += 1
            continue
        try:
            frame_indices_by_state = frame_indices_for_item(
                item,
                h5_path,
                stride=max(1, int(args.frame_stride)),
                pressure_threshold=float(args.pressure_touch_threshold),
            )
        except Exception:
            skipped["h5_read_failed"] += 1
            continue

        for state, frame_indices in frame_indices_by_state.items():
            if args.max_per_video_state and len(frame_indices) > args.max_per_video_state:
                frame_indices = rng.sample(frame_indices, int(args.max_per_video_state))
            for frame_index in frame_indices:
                record = make_frame_record(item, frame_index, state)
                add_reservoir(
                    reservoirs[(split, state)],
                    seen,
                    (split, state),
                    record,
                    targets.get(split, 0),
                    rng,
                )

        if item_i == 1 or item_i % 500 == 0:
            print(f"processed {item_i}/{len(items)} items seen={format_seen(seen)}")

    frames: list[dict[str, Any]] = []
    for split in sorted(include_splits):
        for state in [0, 1, 2]:
            rows = reservoirs[(split, state)]
            rng.shuffle(rows)
            frames.extend(rows)
    rng.shuffle(frames)

    out_manifest = {
        "frames": frames,
        "balanced_frame_sampling": {
            "source_manifest": str(manifest_path),
            "seed": args.seed,
            "frame_stride": args.frame_stride,
            "targets_per_state": targets,
            "max_per_video_state": args.max_per_video_state,
            "pressure_touch_threshold": args.pressure_touch_threshold,
            "variant_kinds": sorted(variant_kinds) if variant_kinds is not None else None,
            "seen_candidates": {f"{split}/{STATE_NAME[state]}": int(count) for (split, state), count in seen.items()},
            "sampled": summarize_frames(frames),
            "skipped": {k: int(v) for k, v in skipped.items()},
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(out_manifest["balanced_frame_sampling"], indent=2))


def frame_indices_for_item(
    item: dict[str, Any],
    h5_path: Path,
    stride: int,
    pressure_threshold: float,
) -> dict[int, list[int]]:
    with h5py.File(h5_path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.float32)
        masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
    n = min(int(item.get("frame_count") or len(labels)), len(labels))
    indices = np.arange(0, n, stride, dtype=np.int64)
    excluded = excluded_frame_indices(item)
    if excluded:
        keep = np.fromiter((int(idx) not in excluded for idx in indices), dtype=bool, count=len(indices))
        indices = indices[keep]
    if len(indices) == 0:
        return {0: [], 1: [], 2: []}

    selected_labels = labels[indices]
    selected_masks = masks[indices]
    valid_xy = (
        (selected_masks[:, 0] != 0)
        & (selected_masks[:, 1] != 0)
        & np.isfinite(selected_labels[:, 0])
        & np.isfinite(selected_labels[:, 1])
    )
    has_pressure = (selected_masks[:, 2] != 0) & np.isfinite(selected_labels[:, 2])
    touch = has_pressure & (selected_labels[:, 2] > pressure_threshold)
    states = np.zeros(len(indices), dtype=np.int8)
    states[valid_xy & ~touch] = 1
    states[valid_xy & touch] = 2

    return {state: indices[states == state].astype(int).tolist() for state in [0, 1, 2]}


def make_frame_record(item: dict[str, Any], frame_index: int, state: int) -> dict[str, Any]:
    stem = str(item.get("stem") or Path(str(item.get("output_video_path", ""))).stem)
    return {
        "video_id": stem,
        "base_stem": item.get("base_stem", stem),
        "variant_kind": item.get("variant_kind", "original"),
        "texture_name": item.get("texture_name"),
        "participant_id": item.get("participant_id"),
        "task_id": item.get("task_id"),
        "split": item.get("split"),
        "frame_index": int(frame_index),
        "depth_index": int(frame_index),
        "video_path": item.get("output_video_path") or item.get("video_path"),
        "label_h5_path": item.get("output_h5_path") or item.get("label_h5_path"),
        "depth_path": item.get("depth_path"),
        "depth_key": item.get("depth_key", "depth_norm"),
        "state_hint": int(state),
    }


def add_reservoir(
    reservoir: list[dict[str, Any]],
    seen: Counter[tuple[str, int]],
    key: tuple[str, int],
    record: dict[str, Any],
    target: int,
    rng: random.Random,
) -> None:
    seen[key] += 1
    if target <= 0:
        reservoir.append(record)
        return
    if len(reservoir) < target:
        reservoir.append(record)
        return
    j = rng.randrange(seen[key])
    if j < target:
        reservoir[j] = record


def excluded_frame_indices(item: dict[str, Any]) -> set[int]:
    excluded = {int(idx) for idx in item.get("exclude_frame_indices", []) or []}
    for frame_range in item.get("exclude_frame_ranges", []) or []:
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            raise ValueError(f"exclude_frame_ranges entries must be [start, end): {frame_range}")
        start, end = int(frame_range[0]), int(frame_range[1])
        if end > start:
            excluded.update(range(start, end))
    return excluded


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for frame in frames:
        split = str(frame.get("split", "unknown"))
        state = STATE_NAME.get(int(frame.get("state_hint", -1)), "unknown")
        counts[split][state] += 1
        counts[split]["total"] += 1
    return {split: dict(counter) for split, counter in counts.items()}


def format_seen(seen: Counter[tuple[str, int]]) -> dict[str, int]:
    return {f"{split}/{STATE_NAME[state]}": int(count) for (split, state), count in sorted(seen.items())}


if __name__ == "__main__":
    main()
