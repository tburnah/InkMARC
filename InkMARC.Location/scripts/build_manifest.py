#!/usr/bin/env python3
"""
build_manifest.py

Scans a dataset directory for .h5 files and writes a manifest JSON describing
each file (participant, task, session_id) plus key HDF5 metadata and shapes.
Search scope includes:
  - dataset_root/*.h5
  - dataset_root/*/*.h5

Extended (Option 2 / Strategy B): also discovers video variants stored alongside
each .h5, expected to be named: <file_id>_*.avi, and emits:
  - manifest["files"][file_id]["video_variants"] = [ ... ]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py


FILENAME_RE = re.compile(
    r"^participant_(?P<participant>\d+?)_(?P<task>[A-Za-z0-9]+?)_(?P<session>\d+?)\.h5$"
)


def find_h5_files(dataset_root: Path) -> List[Path]:
    """
    Return .h5 files found in the dataset root and one directory below it.
    """
    paths = list(dataset_root.glob("*.h5"))
    paths.extend(dataset_root.glob("*/*.h5"))
    return sorted(set(paths))


def find_video_variants(dataset_root: Path, h5_path: Path, file_stem: str) -> List[str]:
    """
    Return a stable, deterministic list of variant video paths for this file_id.
    Videos are expected to be named: <file_stem>_<something>.avi and to live
    alongside the matching H5 file, whether that is in dataset_root or one
    directory below it.
    """
    return sorted(
        p.relative_to(dataset_root).as_posix()
        for p in h5_path.parent.glob(f"{file_stem}_*.avi")
    )


@dataclass(frozen=True)
class H5Record:
    file_id: str
    filename: str
    rel_path: str

    # NEW: list of video variants (may be empty if missing)
    video_variants: List[str]

    participant_id: int
    task: str
    session_id: int

    width: Optional[int]
    height: Optional[int]
    fps: Optional[int]
    channels: Optional[int]
    frame_count_attr: Optional[int]
    attr_count: Optional[int]

    corners_shape: Tuple[int, ...]
    corners_dtype: str
    labels_shape: Tuple[int, ...]
    labels_dtype: str
    label_mask_shape: Tuple[int, ...]
    label_mask_dtype: str
    timestamps_shape: Tuple[int, ...]
    timestamps_dtype: str


def parse_filename(name: str) -> Tuple[int, str, int]:
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(
            f"Filename '{name}' does not match expected pattern "
            f"'participant_<id>_<task>_<session>.h5'"
        )
    return int(m.group("participant")), m.group("task"), int(m.group("session"))


def read_attr_int(h5: h5py.File, key: str) -> Optional[int]:
    if key not in h5.attrs:
        return None
    v = h5.attrs[key]
    try:
        return int(v)
    except Exception:
        return None


def get_ds_info(h5: h5py.File, path: str) -> Tuple[Tuple[int, ...], str]:
    if path not in h5:
        raise KeyError(f"Missing dataset '{path}'")
    ds = h5[path]
    return tuple(ds.shape), str(ds.dtype)


def build_manifest(dataset_root: Path) -> Dict[str, Any]:
    h5_files = find_h5_files(dataset_root)
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files found under: {dataset_root} "
            f"(searched root and one directory below)"
        )

    records: List[Dict[str, Any]] = []
    files_map: Dict[str, Any] = {}
    errors: List[str] = []

    for fp in h5_files:
        try:
            participant_id, task, session_id = parse_filename(fp.name)
            file_id = fp.stem
            rel_path = fp.relative_to(dataset_root).as_posix()

            video_variants = find_video_variants(dataset_root, fp, file_id)
            if not video_variants:
                errors.append(
                    f"{fp.name}: MissingVideoVariants: expected '{file_id}_*.avi' beside the H5, found none"
                )

            with h5py.File(fp, "r") as h5:
                width = read_attr_int(h5, "width")
                height = read_attr_int(h5, "height")
                fps = read_attr_int(h5, "fps")
                channels = read_attr_int(h5, "channels")
                frame_count_attr = read_attr_int(h5, "frame_count")
                attr_count = read_attr_int(h5, "attr_count")

                corners_shape, corners_dtype = get_ds_info(h5, "corners")
                labels_shape, labels_dtype = get_ds_info(h5, "labels")
                label_mask_shape, label_mask_dtype = get_ds_info(h5, "label_mask")
                timestamps_shape, timestamps_dtype = get_ds_info(h5, "timestamps_ns")

            rec = H5Record(
                file_id=file_id,
                filename=fp.name,
                rel_path=rel_path,
                video_variants=video_variants,
                participant_id=participant_id,
                task=task,
                session_id=session_id,
                width=width,
                height=height,
                fps=fps,
                channels=channels,
                frame_count_attr=frame_count_attr,
                attr_count=attr_count,
                corners_shape=corners_shape,
                corners_dtype=corners_dtype,
                labels_shape=labels_shape,
                labels_dtype=labels_dtype,
                label_mask_shape=label_mask_shape,
                label_mask_dtype=label_mask_dtype,
                timestamps_shape=timestamps_shape,
                timestamps_dtype=timestamps_dtype,
            )
            records.append(asdict(rec))

            files_map[file_id] = {
                "h5_rel_path": rel_path,
                "video_variants": video_variants,
                "participant_id": participant_id,
                "task": task,
                "session_id": session_id,
            }

        except Exception as e:
            errors.append(f"{fp.name}: {type(e).__name__}: {e}")

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "dataset_root": dataset_root.as_posix(),
        "file_count": len(records),
        "records": records,
        "files": files_map,
    }

    if errors:
        manifest["errors"] = errors

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/LocationData"),
        help=(
            "Root directory containing participant_*.h5 files and <file_id>_*.avi variants. "
            "Scans both the root and one directory below it."
        ),
    )
    ap.add_argument(
        "--indices-dir",
        type=Path,
        default=None,
        help="Where to write manifest/index/splits (default: <dataset-root>/indices)",
    )
    ap.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag for outputs (default: v1)",
    )
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    indices_dir: Path = args.indices_dir or (dataset_root / "indices")
    indices_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(dataset_root)

    out_path = indices_dir / f"manifest_{args.version}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {out_path}")
    print(f"Files: {manifest['file_count']}")
    if "errors" in manifest:
        print(f"Warnings: {len(manifest['errors'])} (see manifest['errors'])")


if __name__ == "__main__":
    main()
