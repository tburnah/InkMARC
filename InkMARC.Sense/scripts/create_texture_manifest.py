#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manifest that adds texture-replacement videos as separate training items."
    )
    parser.add_argument(
        "--manifest",
        default="/data/baseAVIs_432x672/manifest_participant_splits.json",
        help="Source manifest with corrected H5 paths and participant splits.",
    )
    parser.add_argument(
        "--texture-dir",
        default="/data/baseAVIs_432x672/texture_replacement_v3",
        help="Directory containing texture-replaced videos named '<base_stem>_<texture>.<ext>'.",
    )
    parser.add_argument(
        "--out",
        default="/data/baseAVIs_432x672/manifest_texture_replacement_v3_participant_splits.json",
    )
    parser.add_argument("--exclude-originals", action="store_true", help="Only include texture variants.")
    parser.add_argument(
        "--no-prefer-corrected-h5",
        action="store_true",
        help="Do not replace output_h5_path with '<base_stem>_corrected.h5' when that file exists.",
    )
    parser.add_argument(
        "--exclude-missing-h5",
        action="store_true",
        help="Drop source items and variants whose resolved H5 file does not exist.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any source item has no texture variants.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    texture_dir = Path(args.texture_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_items = list(manifest.get("items", []))
    if not source_items:
        raise ValueError(f"Expected manifest with non-empty 'items': {manifest_path}")
    if not texture_dir.exists():
        raise FileNotFoundError(texture_dir)

    corrected_h5_used = 0
    missing_h5_sources: list[str] = []
    resolved_items: list[dict[str, Any]] = []
    for item in source_items:
        item = dict(item)
        stem = str(item["stem"])
        if not args.no_prefer_corrected_h5:
            corrected_h5 = resolved_corrected_h5_path(item)
            if corrected_h5.exists():
                item["original_output_h5_path"] = item.get("output_h5_path")
                item["output_h5_path"] = str(corrected_h5)
                item["h5_variant"] = "corrected"
                corrected_h5_used += 1
        h5_path = Path(str(item.get("output_h5_path", "")))
        if not h5_path.exists():
            missing_h5_sources.append(stem)
            if args.exclude_missing_h5:
                continue
        resolved_items.append(item)

    all_stems = [str(item["stem"]) for item in source_items]
    by_stem = {str(item["stem"]): item for item in resolved_items}
    all_variants = find_variants(texture_dir, all_stems)
    variants = {stem: paths for stem, paths in all_variants.items() if stem in by_stem}

    missing_variants = sorted(set(by_stem) - set(variants))
    if args.strict and missing_variants:
        raise ValueError(f"{len(missing_variants)} source items have no texture variants: {missing_variants[:20]}")

    out_items: list[dict[str, Any]] = []
    if not args.exclude_originals:
        for item in resolved_items:
            item = dict(item)
            item.setdefault("variant_kind", "original")
            item.setdefault("base_stem", str(item["stem"]))
            out_items.append(item)

    for base_stem in sorted(by_stem):
        base_item = by_stem[base_stem]
        for variant_path in variants.get(base_stem, []):
            out_items.append(make_variant_item(base_item, variant_path))

    out_manifest = dict(manifest)
    out_manifest["items"] = out_items
    out_manifest["texture_replacement"] = {
        "version": "v3",
        "source_manifest": str(manifest_path),
        "texture_dir": str(texture_dir),
        "include_originals": not args.exclude_originals,
        "source_items": len(source_items),
        "included_source_items": len(resolved_items),
        "texture_items": sum(len(v) for v in variants.values()),
        "prefer_corrected_h5": not args.no_prefer_corrected_h5,
        "corrected_h5_used": corrected_h5_used,
        "missing_h5_sources": missing_h5_sources,
        "excluded_missing_h5_sources": len(source_items) - len(resolved_items),
        "missing_variant_sources": missing_variants,
    }
    out_manifest["split_summary"] = summarize(out_items)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(json.dumps(out_manifest["texture_replacement"], indent=2))
    print(json.dumps(out_manifest["split_summary"], indent=2))


def find_variants(texture_dir: Path, source_stems: Any) -> dict[str, list[Path]]:
    stems = sorted((str(stem) for stem in source_stems), key=len, reverse=True)
    variants: dict[str, list[Path]] = defaultdict(list)
    unmatched: list[str] = []
    for path in sorted(texture_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        base_stem = None
        for stem in stems:
            if path.stem.startswith(f"{stem}_"):
                base_stem = stem
                break
        if base_stem is None:
            unmatched.append(path.name)
            continue
        variants[base_stem].append(path)
    if unmatched:
        sample = ", ".join(unmatched[:20])
        raise ValueError(f"Found {len(unmatched)} texture videos that do not match manifest stems: {sample}")
    return {stem: paths for stem, paths in variants.items()}


def make_variant_item(base_item: dict[str, Any], variant_path: Path) -> dict[str, Any]:
    base_stem = str(base_item["stem"])
    texture_name = variant_path.stem[len(base_stem) + 1 :]
    item = dict(base_item)
    item["stem"] = f"{base_stem}__tex_{texture_name}"
    item["base_stem"] = base_stem
    item["variant_kind"] = "texture_replacement_v3"
    item["texture_name"] = texture_name
    item["texture_variant_path"] = str(variant_path)
    item["base_output_video_path"] = base_item.get("output_video_path")
    item["output_video_path"] = str(variant_path)
    item["status"] = "texture_replacement_v3"
    return item


def resolved_corrected_h5_path(item: dict[str, Any]) -> Path:
    stem = str(item["stem"])
    current = Path(str(item.get("output_h5_path") or ""))
    parent = current.parent if str(current) else Path(".")
    return parent / f"{stem}_corrected.h5"


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter()
    split_frames = Counter()
    split_participants: dict[str, set[str]] = defaultdict(set)
    split_variants: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        split = str(item.get("split", "unknown"))
        pid = str(item.get("participant_id", "UNKNOWN"))
        split_counts[split] += 1
        split_frames[split] += int(item.get("frame_count") or 0)
        split_participants[split].add(pid)
        split_variants[split][str(item.get("variant_kind", "original"))] += 1
    return {
        split: {
            "participants": len(split_participants[split]),
            "videos": int(split_counts[split]),
            "frames": int(split_frames[split]),
            "participant_ids": sorted(split_participants[split]),
            "variant_distribution": dict(split_variants[split]),
        }
        for split in ["train", "val", "test"]
    }


if __name__ == "__main__":
    main()
