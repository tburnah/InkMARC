#!/usr/bin/env python3
"""
build_index.py

Reads manifest_*.json and writes index_*.jsonl with one line per frame/sample.

Each sample points to:
  - which H5 file (rel_path / file_id)
  - which frame index within that file
  - timestamp_ns for convenience

This is intentionally lightweight: no corners/labels arrays are duplicated into index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py


def iter_samples(dataset_root: Path, record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    rel_path = record["rel_path"]
    file_id = record["file_id"]
    participant_id = record["participant_id"]
    task = record["task"]
    session_id = record["session_id"]

    fp = dataset_root / rel_path
    with h5py.File(fp, "r") as h5:
        ts = h5["timestamps_ns"]
        n = int(ts.shape[0])

        # basic sanity checks if present
        if "corners" in h5:
            if int(h5["corners"].shape[0]) != n:
                raise ValueError(f"{fp.name}: corners first dim != timestamps_ns")
        if "labels" in h5:
            if int(h5["labels"].shape[0]) != n:
                raise ValueError(f"{fp.name}: labels first dim != timestamps_ns")
        if "label_mask" in h5:
            if int(h5["label_mask"].shape[0]) != n:
                raise ValueError(f"{fp.name}: label_mask first dim != timestamps_ns")

        # emit one record per frame
        for i in range(n):
            sample_id = f"{file_id}:{i:06d}"
            yield {
                "schema_version": 1,
                "sample_id": sample_id,
                "file_id": file_id,
                "rel_path": rel_path,
                "frame_idx": i,
                "participant_id": participant_id,
                "task": task,
                "session_id": session_id,
                "timestamp_ns": int(ts[i]),
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("/data/LocationData"))
    ap.add_argument("--indices-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON (default: <indices-dir>/manifest_<version>.json)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output index JSONL (default: <indices-dir>/index_<version>.jsonl)",
    )
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    indices_dir: Path = args.indices_dir or (dataset_root / "indices")
    indices_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (indices_dir / f"manifest_{args.version}.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run build_manifest.py first."
        )

    output_path = args.output or (indices_dir / f"index_{args.version}.jsonl")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = manifest["records"]

    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            for sample in iter_samples(dataset_root, r):
                f.write(json.dumps(sample) + "\n")
                total += 1

    print(f"Wrote index: {output_path}")
    print(f"Samples (frames): {total}")


if __name__ == "__main__":
    main()
