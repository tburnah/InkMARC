#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a manifest copy with known outlier videos removed.")
    parser.add_argument("--input", required=True, help="Input JSON manifest.")
    parser.add_argument("--output", required=True, help="Filtered JSON manifest to write.")
    parser.add_argument("--outlier-video-id", action="append", default=[], help="Exact video_id/stem/base_stem to exclude.")
    parser.add_argument("--reason", default="manual_outlier", help="Reason stored in output metadata/report.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    outlier_ids = {str(v) for v in args.outlier_video_id}
    if not outlier_ids:
        raise ValueError("Pass at least one --outlier-video-id")

    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    filtered, removed = filter_manifest(manifest, outlier_ids)
    metadata = filtered.setdefault("outlier_filter", {})
    metadata.update(
        {
            "source_manifest": str(input_path),
            "reason": str(args.reason),
            "outlier_video_ids": sorted(outlier_ids),
            "removed_records": len(removed),
        }
    )

    report = build_report(removed, outlier_ids, args.reason)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered, indent=2, allow_nan=False), encoding="utf-8")
    report_path = Path(args.report) if args.report else output_path.with_suffix(".outliers.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print(f"wrote {output_path}")
    print(f"wrote {report_path}")
    print(f"removed_records={len(removed)}")
    print(f"removed_by_video={dict(report['removed_by_video_id'])}")
    print(f"remaining_counts={report['remaining_counts']}")


def filter_manifest(manifest: Any, outlier_ids: set[str]) -> tuple[Any, list[dict[str, Any]]]:
    removed: list[dict[str, Any]] = []
    if isinstance(manifest, list):
        kept = []
        for row in manifest:
            row = dict(row)
            if is_outlier(row, outlier_ids):
                removed.append(row)
            else:
                kept.append(row)
        return kept, removed

    if not isinstance(manifest, dict):
        raise ValueError("Unsupported manifest JSON root; expected list or object.")

    out = dict(manifest)
    if isinstance(manifest.get("frames"), list):
        kept = []
        for row in manifest["frames"]:
            row = dict(row)
            if is_outlier(row, outlier_ids):
                removed.append(row)
            else:
                kept.append(row)
        out["frames"] = kept
        return out, removed

    if isinstance(manifest.get("items"), list):
        kept = []
        for row in manifest["items"]:
            row = dict(row)
            if is_outlier(row, outlier_ids):
                removed.append(row)
            else:
                kept.append(row)
        out["items"] = kept
        return out, removed

    raise ValueError("Unsupported manifest layout; expected a frames or items list.")


def is_outlier(row: dict[str, Any], outlier_ids: set[str]) -> bool:
    candidates = {
        str(row.get("video_id", "")),
        str(row.get("base_stem", "")),
        str(row.get("stem", "")),
        Path(str(row.get("video_path", row.get("output_video_path", "")))).stem,
    }
    return bool(candidates & outlier_ids)


def build_report(removed: list[dict[str, Any]], outlier_ids: set[str], reason: str) -> dict[str, Any]:
    by_video: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_state: Counter[str] = Counter()
    by_participant: Counter[str] = Counter()
    for row in removed:
        video_id = str(row.get("video_id") or row.get("base_stem") or row.get("stem") or Path(str(row.get("video_path", ""))).stem)
        by_video[video_id] += 1
        by_split[str(row.get("split", "UNKNOWN"))] += 1
        by_state[str(row.get("state_hint", "UNKNOWN"))] += 1
        by_participant[str(row.get("participant_id", "UNKNOWN"))] += 1
    return {
        "reason": reason,
        "outlier_video_ids": sorted(outlier_ids),
        "removed_records": len(removed),
        "removed_by_video_id": dict(sorted(by_video.items())),
        "removed_by_split": dict(sorted(by_split.items())),
        "removed_by_state_hint": dict(sorted(by_state.items())),
        "removed_by_participant_id": dict(sorted(by_participant.items())),
        "remaining_counts": {
            "requested_outliers": len(outlier_ids),
            "matched_outliers": len(by_video),
            "unmatched_outliers": sorted(outlier_ids - set(by_video)),
        },
    }


if __name__ == "__main__":
    main()
