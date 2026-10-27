#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# --- add src/ to PYTHONPATH ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.dataset_core import InkMARCIndexedDataset
from data.label_schema import extract_xy_touch



def check_split(dataset_root: str, version: str, split: str, max_samples: int | None):
    ds = InkMARCIndexedDataset(
        dataset_root=dataset_root,
        index_version=version,
        split=split,
        require_labels=True,
        transform=None,  # IMPORTANT: raw labels only
    )

    counts = Counter()
    pressure_values = []

    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    for i in range(n):
        sample = ds[i]
        labels = np.asarray(sample["labels"], dtype=np.float32)
        mask = np.asarray(sample["label_mask"], dtype=np.uint8)

        touch, touch_mask, x, y, xy_mask = extract_xy_touch(labels, mask)

        # Count touch labels actually evaluated
        if touch_mask > 0.5:
            counts["touch_total"] += 1
            if touch >= 0.5:
                counts["touch_1"] += 1
            else:
                counts["touch_0"] += 1

        # Collect pressure values where present
        if mask[2] > 0:  # PRESSURE index = 2 (schema)
            pressure_values.append(float(labels[2]))

    ds.close()

    print(f"\nSplit: {split}")
    print(f"  samples_checked: {n}")
    print(f"  touch_total (mask==1): {counts['touch_total']}")
    print(f"  touch_1: {counts['touch_1']}")
    print(f"  touch_0: {counts['touch_0']}")

    if counts["touch_total"] > 0:
        frac = counts["touch_1"] / counts["touch_total"]
        print(f"  touch_1 fraction: {frac:.4f}")

    if pressure_values:
        pv = np.asarray(pressure_values)
        print(
            f"  pressure stats (mask==1): "
            f"min={pv.min():.6f}, "
            f"p1={np.percentile(pv,1):.6f}, "
            f"p5={np.percentile(pv,5):.6f}, "
            f"p50={np.percentile(pv,50):.6f}, "
            f"p95={np.percentile(pv,95):.6f}, "
            f"max={pv.max():.6f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Optional cap for quick checks")
    args = ap.parse_args()

    print("Checking touch balance using extract_xy_touch()")

    for split in ["train", "val", "test"]:
        try:
            check_split(args.dataset_root, args.version, split, args.max_samples)
        except Exception as e:
            print(f"\nSplit: {split}  (skipped: {e})")


if __name__ == "__main__":
    main()
