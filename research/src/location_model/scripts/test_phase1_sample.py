#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from data.dataset_core import InkMARCIndexedDataset
from data.transforms.phase1_transform import Phase1Transform


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="runs/phase1_sample.png")
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--margin-frac", type=float, default=0.10)
    args = ap.parse_args()

    tf = Phase1Transform(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split=args.split,
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
    )

    ds = InkMARCIndexedDataset(
        args.dataset_root, index_version=args.version, split=args.split, require_labels=True, transform=tf
    )

    idx = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]

    img = sample["image"].permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_bgr = img[:, :, ::-1].copy()

    if float(sample["xy_mask"].item()) > 0.0:
        xy = sample["xy"].numpy()
        x = int(round(float(xy[0]) * args.out_w))
        y = int(round(float(xy[1]) * args.out_h))
        x = max(0, min(x, args.out_w - 1))
        y = max(0, min(y, args.out_h - 1))
        cv2.circle(img_bgr, (x, y), 5, (0, 0, 255), -1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img_bgr)
    tf.close()
    ds.close()

    if not ok:
        print(f"Failed to write image to {out_path}")
        return 2
    print(f"Wrote sample to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
