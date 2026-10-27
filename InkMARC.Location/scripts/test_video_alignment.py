#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from data.dataset_core import InkMARCIndexedDataset


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_h5_frame_count(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as h5:
        return int(h5["timestamps_ns"].shape[0])


def get_video_frame_count(video_path: Path) -> Tuple[int | None, str | None]:
    if cv2 is None:
        return None, "cv2 not available (install opencv-python to validate video metadata)"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "cv2.VideoCapture failed to open video"
    try:
        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not fc or fc <= 0:
            return None, "CAP_PROP_FRAME_COUNT returned 0/invalid (codec/backend limitation)"
        return int(fc), None
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="If set, do not require labels (validate all frames).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, treat frame-count mismatches as failures (otherwise warnings).",
    )
    ap.add_argument(
        "--require-all-variants",
        action="store_true",
        help="If set, fail if any expected variant is missing/unreadable (recommended for training augmentation).",
    )
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    indices_dir = dataset_root / "indices"
    manifest_path = indices_dir / f"manifest_{args.version}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest['files'] missing/empty. Rebuild manifest with files map.")

    require_labels = not args.include_unlabeled

    print("InkMARC video alignment audit (multi-variant)")
    print(f"  dataset_root: {dataset_root}")
    print(f"  version: {args.version}")
    print(f"  require_labels: {require_labels}")
    print(f"  strict: {args.strict}")
    print(f"  require_all_variants: {args.require_all_variants}")
    print()

    # Per-file info:
    # file_id -> (n_h5, {variant_rel_path: n_frames_or_None})
    per_file_counts: Dict[str, Tuple[int, Dict[str, int | None]]] = {}

    file_fail = 0
    file_warn = 0

    for file_id, entry in files.items():
        if not isinstance(entry, dict):
            print(f"[FAIL] file_id={file_id}: manifest entry is not a dict")
            file_fail += 1
            continue

        h5_rel = entry.get("h5_rel_path") or entry.get("rel_path")
        variants = entry.get("video_variants")

        if not h5_rel:
            print(f"[FAIL] file_id={file_id}: missing h5_rel_path/rel_path in manifest")
            file_fail += 1
            continue

        h5_path = dataset_root / str(h5_rel)
        if not h5_path.exists():
            print(f"[FAIL] file_id={file_id}: H5 missing: {h5_path}")
            file_fail += 1
            continue

        if not isinstance(variants, list) or len(variants) == 0:
            print(f"[FAIL] file_id={file_id}: missing/empty video_variants in manifest")
            file_fail += 1
            continue

        n_h5 = get_h5_frame_count(h5_path)

        variant_counts: Dict[str, int | None] = {}
        any_variant_missing = False
        any_variant_unreadable = False

        for rel in variants:
            rel = str(rel)
            vid_path = dataset_root / rel
            if not vid_path.exists():
                msg = f"[{'FAIL' if args.require_all_variants else 'WARN'}] file_id={file_id}: video missing: {vid_path}"
                print(msg)
                if args.require_all_variants:
                    file_fail += 1
                else:
                    file_warn += 1
                any_variant_missing = True
                variant_counts[rel] = None
                continue

            n_vid, vid_err = get_video_frame_count(vid_path)
            variant_counts[rel] = n_vid

            if n_vid is None:
                msg = (
                    f"[{'FAIL' if args.require_all_variants else 'WARN'}] file_id={file_id}: "
                    f"cannot read video frame count ({vid_err}) for {rel}"
                )
                print(msg)
                if args.require_all_variants:
                    file_fail += 1
                else:
                    file_warn += 1
                any_variant_unreadable = True
                continue

            # Safety rule: each variant must be at least as long as the H5 rows
            if n_vid < n_h5:
                print(f"[FAIL] file_id={file_id}: variant shorter than H5: {rel} (video={n_vid}, h5={n_h5})")
                file_fail += 1
            elif n_vid != n_h5:
                msg = f"[WARN] file_id={file_id}: variant frame count != H5 rows: {rel} (video={n_vid}, h5={n_h5})"
                if args.strict:
                    print(msg.replace("[WARN]", "[FAIL]"))
                    file_fail += 1
                else:
                    print(msg)
                    file_warn += 1

        per_file_counts[str(file_id)] = (n_h5, variant_counts)

        # Optional: if some variants missing/unreadable and you DON'T require_all_variants,
        # it's still useful to warn once per file.
        if (any_variant_missing or any_variant_unreadable) and not args.require_all_variants:
            file_warn += 0  # already counted per-variant warnings; no extra

    print()
    print(f"File-level results: fails={file_fail}  warnings={file_warn}")
    print()

    # 2) Split-level index bounds audit
    split_fail = 0
    split_warn = 0

    for split in ("train", "val", "test"):
        try:
            ds = InkMARCIndexedDataset(
                dataset_root=dataset_root,
                index_version=args.version,
                split=split,
                require_labels=require_labels,
            )

            oob_h5 = 0
            oob_vid_min = 0
            unknown_vid = 0

            for rec in ds._records:  # pylint: disable=protected-access
                file_id = rec.get("file_id")
                if not file_id:
                    continue
                frame_idx = int(rec.get("frame_idx", -1))

                info = per_file_counts.get(str(file_id))
                if info is None:
                    continue

                n_h5, variant_counts = info

                if frame_idx < 0 or frame_idx >= n_h5:
                    oob_h5 += 1

                # Conservative: require in-bounds for the *shortest readable* variant.
                readable = [n for n in variant_counts.values() if isinstance(n, int)]
                if not readable:
                    unknown_vid += 1
                else:
                    n_min = min(readable)
                    if frame_idx < 0 or frame_idx >= n_min:
                        oob_vid_min += 1

            if oob_h5 > 0:
                print(f"[FAIL] [{split}] out-of-bounds vs H5: {oob_h5}")
                split_fail += 1
            else:
                print(f"[OK]   [{split}] index bounds vs H5: OK")

            if unknown_vid == len(ds):
                print(f"[WARN] [{split}] video bounds not validated (no readable video counts available)")
                split_warn += 1
            elif oob_vid_min > 0:
                print(f"[FAIL] [{split}] out-of-bounds vs min(video_variant_len): {oob_vid_min}")
                split_fail += 1
            else:
                print(f"[OK]   [{split}] index bounds vs video variants: OK")

        except Exception as e:
            print(f"[FAIL] [{split}] dataset load failed: {type(e).__name__}({e})")
            split_fail += 1

        print()

    if file_fail or split_fail:
        print("FAIL: video alignment audit failed.")
        return 2

    if file_warn or split_warn:
        print("PASS with warnings.")
        return 0

    print("PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
