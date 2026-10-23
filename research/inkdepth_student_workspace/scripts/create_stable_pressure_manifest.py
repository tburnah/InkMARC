#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


STATE_NAME = {0: "background", 1: "hover", 2: "touch"}
STATE_ID = {name: idx for idx, name in STATE_NAME.items()}


def parse_states(values: list[str]) -> set[int]:
    states: set[int] = set()
    for value in values:
        value = str(value).strip().lower()
        if value.isdigit():
            states.add(int(value))
        elif value in STATE_ID:
            states.add(STATE_ID[value])
        else:
            raise ValueError(f"Unknown state: {value}")
    return states


def derive_states(labels: np.ndarray, masks: np.ndarray, pressure_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_xy = (
        (masks[:, 0] != 0)
        & (masks[:, 1] != 0)
        & np.isfinite(labels[:, 0])
        & np.isfinite(labels[:, 1])
    )
    pressure_valid = (masks[:, 2] != 0) & np.isfinite(labels[:, 2])
    touch = valid_xy & pressure_valid & (labels[:, 2] > pressure_threshold)
    states = np.zeros(labels.shape[0], dtype=np.int8)
    states[valid_xy & ~touch] = 1
    states[touch] = 2
    return states, valid_xy, pressure_valid


def transition_distances(states: np.ndarray) -> np.ndarray:
    distances = np.full(states.shape[0], fill_value=10**9, dtype=np.int32)
    transitions = np.flatnonzero(states[1:] != states[:-1]) + 1
    if transitions.size == 0:
        return distances
    indices = np.arange(states.shape[0], dtype=np.int32)
    insert_pos = np.searchsorted(transitions, indices)
    right = np.where(insert_pos < transitions.size, transitions[np.minimum(insert_pos, transitions.size - 1)], 10**9)
    left_pos = np.maximum(insert_pos - 1, 0)
    left = np.where(insert_pos > 0, transitions[left_pos], -10**9)
    return np.minimum(np.abs(indices - left), np.abs(indices - right)).astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced frame manifest excluding ambiguous pressure/transition-boundary labels."
    )
    parser.add_argument("--manifest", required=True, help="Input frame-level manifest with a top-level frames list.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--states", nargs="+", default=["background", "hover", "touch"])
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument(
        "--exclude-transition-window",
        type=int,
        default=3,
        help="Drop frames whose derived state changes within this many frames. 0 drops only exact transition frames.",
    )
    parser.add_argument(
        "--min-touch-pressure",
        type=float,
        default=0.0,
        help="Drop touch frames with pressure <= this value. Use 0 to keep all positive pressure touches.",
    )
    parser.add_argument(
        "--require-state-hint-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows where manifest state_hint disagrees with the H5-derived state.",
    )
    parser.add_argument(
        "--target-per-state",
        type=int,
        default=0,
        help="Rows per split/state after filtering. 0 means auto-balance to the smallest available state count per split.",
    )
    parser.add_argument(
        "--max-per-h5-frame",
        type=int,
        default=0,
        help="Optional cap for repeated texture rows from the same H5/frame before balancing. 0 disables.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = [dict(row) for row in manifest.get("frames", [])]
    if not frames:
        raise ValueError(f"Expected top-level non-empty frames list: {manifest_path}")

    include_splits = {str(split) for split in args.splits}
    include_states = parse_states(args.states)
    frames = [row for row in frames if str(row.get("split", "unknown")) in include_splits]

    by_h5: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        by_h5[str(row.get("label_h5_path") or "")].append(row)

    kept_by_split_state: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen_candidates: Counter[tuple[str, int]] = Counter()
    kept_candidates: Counter[tuple[str, int]] = Counter()
    skipped = Counter()
    per_h5_frame_kept: Counter[tuple[str, int]] = Counter()

    for h5_i, (h5_path, rows) in enumerate(sorted(by_h5.items()), start=1):
        path = Path(h5_path)
        if not h5_path or not path.exists():
            skipped["missing_h5"] += len(rows)
            continue
        try:
            with h5py.File(path, "r") as h5:
                labels = np.asarray(h5["labels"][:], dtype=np.float32)
                masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
        except Exception:
            skipped["unreadable_h5"] += len(rows)
            continue

        states, valid_xy, pressure_valid = derive_states(labels, masks, float(args.pressure_touch_threshold))
        distances = transition_distances(states)

        rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_frame[int(row.get("frame_index", row.get("index", -1)))].append(row)

        for frame_index, frame_rows in rows_by_frame.items():
            if frame_index < 0 or frame_index >= labels.shape[0]:
                skipped["frame_out_of_range"] += len(frame_rows)
                continue
            state = int(states[frame_index])
            split = str(frame_rows[0].get("split", "unknown"))
            key = (split, state)
            seen_candidates[key] += len(frame_rows)

            if state not in include_states:
                skipped["state_excluded"] += len(frame_rows)
                continue
            if int(distances[frame_index]) <= int(args.exclude_transition_window):
                skipped["near_transition"] += len(frame_rows)
                continue
            if state == 2 and float(labels[frame_index, 2]) <= float(args.min_touch_pressure):
                skipped["low_touch_pressure"] += len(frame_rows)
                continue
            if state in {1, 2} and (not bool(valid_xy[frame_index]) or not bool(pressure_valid[frame_index])):
                skipped["invalid_hover_touch_label"] += len(frame_rows)
                continue

            candidate_rows = list(frame_rows)
            if args.require_state_hint_match:
                matched_rows = []
                for row in candidate_rows:
                    hint = row.get("state_hint")
                    if hint is None or int(hint) == state:
                        matched_rows.append(row)
                    else:
                        skipped["state_hint_mismatch"] += 1
                candidate_rows = matched_rows
            if not candidate_rows:
                continue

            if args.max_per_h5_frame and len(candidate_rows) > int(args.max_per_h5_frame):
                candidate_rows = rng.sample(candidate_rows, int(args.max_per_h5_frame))

            for row in candidate_rows:
                out_row = dict(row)
                out_row["state_hint"] = state
                out_row["pressure_filter"] = {
                    "pressure": float(labels[frame_index, 2]) if pressure_valid[frame_index] else None,
                    "transition_distance": int(distances[frame_index]),
                    "exclude_transition_window": int(args.exclude_transition_window),
                    "min_touch_pressure": float(args.min_touch_pressure),
                }
                kept_by_split_state[key].append(out_row)
                kept_candidates[key] += 1
                per_h5_frame_kept[(h5_path, frame_index)] += 1

        if h5_i == 1 or h5_i % 100 == 0:
            print(f"processed {h5_i}/{len(by_h5)} h5 files kept={format_counts(kept_candidates)}")

    sampled: list[dict[str, Any]] = []
    targets: dict[str, dict[str, int]] = {}
    for split in sorted(include_splits):
        available = {state: len(kept_by_split_state[(split, state)]) for state in sorted(include_states)}
        nonzero = [count for count in available.values() if count > 0]
        target = int(args.target_per_state) if args.target_per_state > 0 else (min(nonzero) if nonzero else 0)
        targets[split] = {STATE_NAME[state]: min(target, available[state]) for state in sorted(include_states)}
        for state in sorted(include_states):
            rows = list(kept_by_split_state[(split, state)])
            rng.shuffle(rows)
            sampled.extend(rows[: min(target, len(rows))])
    rng.shuffle(sampled)

    out_manifest = {
        "frames": sampled,
        "stable_pressure_sampling": {
            "source_manifest": str(manifest_path),
            "seed": int(args.seed),
            "splits": sorted(include_splits),
            "states": [STATE_NAME[state] for state in sorted(include_states)],
            "pressure_touch_threshold": float(args.pressure_touch_threshold),
            "exclude_transition_window": int(args.exclude_transition_window),
            "min_touch_pressure": float(args.min_touch_pressure),
            "require_state_hint_match": bool(args.require_state_hint_match),
            "target_per_state": int(args.target_per_state),
            "resolved_targets_per_split_state": targets,
            "seen_candidates": format_counts(seen_candidates),
            "kept_candidates": format_counts(kept_candidates),
            "sampled": summarize_frames(sampled),
            "skipped": {key: int(value) for key, value in skipped.items()},
            "max_repeated_textures_per_h5_frame_after_filter": max(per_h5_frame_kept.values(), default=0),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(out_manifest["stable_pressure_sampling"], indent=2))


def format_counts(counts: Counter[tuple[str, int]]) -> dict[str, int]:
    return {f"{split}/{STATE_NAME[state]}": int(count) for (split, state), count in sorted(counts.items())}


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for frame in frames:
        split = str(frame.get("split", "unknown"))
        state = STATE_NAME.get(int(frame.get("state_hint", -1)), "unknown")
        counts[split][state] += 1
        counts[split]["total"] += 1
    return {split: dict(counter) for split, counter in counts.items()}


if __name__ == "__main__":
    main()
