#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def participant_from_stem(stem: str) -> str:
    match = re.match(r"participant_([^_]+)_", stem)
    if not match:
        raise ValueError(f"Could not parse participant id from stem: {stem}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create participant-level train/val/test splits")
    parser.add_argument("--manifest", default="/data/baseAVIs_432x672/manifest.json")
    parser.add_argument("--base-dir", default="/data/baseAVIs_432x672")
    parser.add_argument("--out", default="/data/baseAVIs_432x672/manifest_participant_splits.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--depth-key", default="depth_norm")
    parser.add_argument("--allow-missing-depth", action="store_true")
    args = parser.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")

    manifest_path = Path(args.manifest)
    base_dir = Path(args.base_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = list(manifest.get("items", []))
    if not items:
        raise ValueError(f"Expected manifest with non-empty 'items': {manifest_path}")

    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    missing_depth: list[str] = []
    missing_video: list[str] = []
    missing_h5: list[str] = []

    for item in items:
        item = dict(item)
        stem = str(item["stem"])
        pid = participant_from_stem(stem)
        video_path = Path(item.get("output_video_path") or base_dir / f"{stem}.avi")
        h5_path = Path(item.get("output_h5_path") or base_dir / f"{stem}.h5")
        depth_path = base_dir / "depth" / f"{stem}.zarr"
        if not video_path.exists():
            missing_video.append(stem)
        if not h5_path.exists():
            missing_h5.append(stem)
        if not depth_path.exists():
            missing_depth.append(stem)
            if not args.allow_missing_depth:
                continue
        item["participant_id"] = pid
        item["output_video_path"] = str(video_path)
        item["output_h5_path"] = str(h5_path)
        item["depth_path"] = str(depth_path)
        item["depth_key"] = args.depth_key
        enriched.append(item)
        by_participant[pid].append(item)

    participants = sorted(by_participant)
    rng = random.Random(args.seed)
    rng.shuffle(participants)
    n = len(participants)
    n_train = max(1, round(n * args.train_frac))
    n_val = max(1, round(n * args.val_frac))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_p = set(participants[:n_train])
    val_p = set(participants[n_train : n_train + n_val])
    test_p = set(participants[n_train + n_val :])

    split_for_pid = {pid: "train" for pid in train_p}
    split_for_pid.update({pid: "val" for pid in val_p})
    split_for_pid.update({pid: "test" for pid in test_p})
    for item in enriched:
        item["split"] = split_for_pid[item["participant_id"]]

    out_manifest = dict(manifest)
    out_manifest["items"] = enriched
    out_manifest["split_policy"] = {
        "level": "participant",
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "source_manifest": str(manifest_path),
    }
    out_manifest["split_summary"] = summarize(enriched)
    out_manifest["missing_summary"] = {
        "missing_video": len(missing_video),
        "missing_h5": len(missing_h5),
        "missing_depth": len(missing_depth),
        "excluded_for_missing_depth": 0 if args.allow_missing_depth else len(missing_depth),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(out_manifest["split_summary"], indent=2))
    print(json.dumps(out_manifest["missing_summary"], indent=2))


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter()
    split_frames = Counter()
    split_participants: dict[str, set[str]] = defaultdict(set)
    for item in items:
        split = str(item["split"])
        split_counts[split] += 1
        split_frames[split] += int(item.get("frame_count") or 0)
        split_participants[split].add(str(item["participant_id"]))
    return {
        split: {
            "participants": len(split_participants[split]),
            "videos": int(split_counts[split]),
            "frames": int(split_frames[split]),
            "participant_ids": sorted(split_participants[split]),
        }
        for split in ["train", "val", "test"]
    }


if __name__ == "__main__":
    main()
