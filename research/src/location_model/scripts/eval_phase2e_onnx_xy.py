#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader

# Make src imports work when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset_history import InkMARCIndexedHistoryDataset
from data.transforms.phase2e_transform_v1 import Phase2eTransform_v1
from train_phase2e import collate_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--max-samples", type=int, default=0, help="If >0, evaluate only the first N samples.")
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument(
        "--require-touch-for-xy",
        action="store_true",
        help="Evaluate only on samples where touch is present (touch_mask>0.5 and touch>0.01).",
    )
    ap.add_argument(
        "--touch-only",
        action="store_true",
        help="Alias for --require-touch-for-xy.",
    )
    args = ap.parse_args()
    require_touch_for_xy = bool(args.require_touch_for_xy or args.touch_only)

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(str(onnx_path))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name

    tf = Phase2eTransform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split=args.split,
        out_w=args.out_w,
        out_h=args.out_h,
        normalize=True,
        history_len=3,
        require_touch_for_xy=require_touch_for_xy,
    )

    ds = InkMARCIndexedHistoryDataset(
        args.dataset_root,
        index_version=args.version,
        split=args.split,
        require_labels=True,
        history_len=3,
        transform=tf,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    total_abs_px = 0.0
    total_mask = 0.0
    scale = np.array([args.out_w, args.out_h], dtype=np.float32)
    processed = 0
    max_samples = max(0, int(args.max_samples))

    for batch in loader:
        if max_samples and processed >= max_samples:
            break

        img = batch["image"].numpy().astype(np.float32)
        aux = batch["aux14"].numpy().astype(np.float32)
        xy = batch["xy"].numpy().astype(np.float32)
        m = batch["xy_mask"].numpy().astype(np.float32)

        if max_samples:
            remaining = max_samples - processed
            if remaining <= 0:
                break
            if img.shape[0] > remaining:
                img = img[:remaining]
                aux = aux[:remaining]
                xy = xy[:remaining]
                m = m[:remaining]

        pred = sess.run([out_name], {in_names[0]: img, in_names[1]: aux})[0]
        abs_err = np.abs(pred - xy)
        abs_err_px = (abs_err * scale).mean(axis=1)  # [B]
        total_abs_px += float((abs_err_px * m).sum())
        total_mask += float(m.sum())
        processed += int(img.shape[0])

    xy_mae_px = total_abs_px / max(total_mask, 1.0)
    print(f"onnx={onnx_path}")
    print(
        f"split={args.split} require_touch_for_xy={require_touch_for_xy} "
        f"samples={processed} xy_mae_px={xy_mae_px:.4f}"
    )

    if hasattr(ds, "close"):
        ds.close()
    tf.close()


if __name__ == "__main__":
    main()
