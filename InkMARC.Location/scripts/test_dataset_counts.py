#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset_core import InkMARCIndexedDataset


def count_split(ds: InkMARCIndexedDataset) -> tuple[int, int]:
    """
    Returns:
      total_count: number of index records in this split
      usable_count: number of records that pass ds filtering (e.g., require_labels)
                   In our setup, ds is already filtered, so usable==len(ds).
                   We compute both for reporting clarity.
    """
    total = getattr(ds, "total_records", None)
    if total is None:
        # If dataset_core doesn't expose total pre-filter count,
        # treat total as usable for now.
        total = len(ds)
    usable = len(ds)
    return int(total), int(usable)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="If set, do not require labels (count all frames).")
    args = ap.parse_args()

    require_labels = not args.include_unlabeled

    print("InkMARC dataset split counts")
    print(f"  dataset_root: {args.dataset_root}")
    print(f"  index_version: {args.version}")
    print(f"  require_labels: {require_labels}")
    print()

    any_ok = False
    for split in ("train", "val", "test"):
        try:
            ds = InkMARCIndexedDataset(
                dataset_root=args.dataset_root,
                index_version=args.version,
                split=split,
                require_labels=require_labels,
            )
            total, usable = count_split(ds)
            any_ok = True
            print(f"[{split}] total_records={total:,}  usable_records={usable:,}")
        except Exception as e:
            print(f"[{split}] failed: {type(e).__name__}({e})")

        print()

    if not any_ok:
        print("No samples counted (check index/splits paths and filtering).")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
