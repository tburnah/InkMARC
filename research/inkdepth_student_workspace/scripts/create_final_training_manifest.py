#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PROBLEM_PARTICIPANTS = {"307", "321"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a final inference-training manifest with all kept records set to train.")
    parser.add_argument("--input", required=True, help="Input frame/item manifest.")
    parser.add_argument("--output", required=True, help="Output manifest.")
    parser.add_argument("--exclude-participant-id", action="append", default=[], help="Participant id to remove.")
    parser.add_argument("--exclude-video-id", action="append", default=[], help="Exact video_id/base_stem/stem to remove.")
    parser.add_argument("--keep-default-problematic", action="store_true", help="Do not auto-remove participants 307 and 321.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    key, records = manifest_records(manifest)

    excluded_participants = set(str(v) for v in args.exclude_participant_id)
    if not args.keep_default_problematic:
        excluded_participants |= DEFAULT_PROBLEM_PARTICIPANTS
    excluded_participants |= exclusion_participants_from_manifest(manifest)
    excluded_videos = set(str(v) for v in args.exclude_video_id)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    source_split_counts: Counter[str] = Counter()
    for row in records:
        row = dict(row)
        source_split_counts[str(row.get("split", "UNKNOWN"))] += 1
        if participant_id(row) in excluded_participants or video_id(row) in excluded_videos:
            removed.append(row)
            continue
        row["split"] = "train"
        kept.append(row)

    out = dict(manifest) if isinstance(manifest, dict) else {"frames": kept}
    out[key] = kept
    out["final_training_manifest"] = {
        "source_manifest": str(input_path),
        "split_policy": "all_kept_records_train",
        "excluded_participant_ids": sorted(excluded_participants),
        "excluded_video_ids": sorted(excluded_videos),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "kept_records": len(kept),
        "removed_records": len(removed),
        "kept_participants": sorted({participant_id(row) for row in kept}),
        "kept_videos": len({video_id(row) for row in kept}),
        "removed_by_participant": dict(sorted(Counter(participant_id(row) for row in removed).items())),
        "removed_by_video": dict(sorted(Counter(video_id(row) for row in removed).items())),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {output_path}")
    print(json.dumps(out["final_training_manifest"], indent=2, allow_nan=False))


def manifest_records(manifest: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(manifest, list):
        return "frames", [dict(row) for row in manifest]
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a list or object.")
    if isinstance(manifest.get("frames"), list):
        return "frames", [dict(row) for row in manifest["frames"]]
    if isinstance(manifest.get("items"), list):
        return "items", [dict(row) for row in manifest["items"]]
    raise ValueError("Manifest must contain a frames or items list.")


def exclusion_participants_from_manifest(manifest: Any) -> set[str]:
    if not isinstance(manifest, dict):
        return set()
    found: set[str] = set()
    exclusions = manifest.get("exclusions")
    if isinstance(exclusions, dict):
        stack: list[Any] = [exclusions]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                ids = current.get("participant_ids")
                if isinstance(ids, list):
                    found.update(str(value) for value in ids)
                stack.extend(value for value in current.values() if isinstance(value, (dict, list)))
            elif isinstance(current, list):
                stack.extend(value for value in current if isinstance(value, (dict, list)))
    return found


def participant_id(row: dict[str, Any]) -> str:
    value = row.get("participant_id")
    if value not in {None, ""}:
        return str(value)
    video = video_id(row)
    parts = video.split("_")
    if len(parts) >= 2 and parts[0] == "participant":
        return parts[1]
    raise ValueError(f"Could not determine participant_id for record: {row}")


def video_id(row: dict[str, Any]) -> str:
    for key in ("video_id", "base_stem", "stem"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    for key in ("video_path", "output_video_path"):
        value = row.get(key)
        if value not in {None, ""}:
            return Path(str(value)).stem
    return "UNKNOWN"


if __name__ == "__main__":
    main()
