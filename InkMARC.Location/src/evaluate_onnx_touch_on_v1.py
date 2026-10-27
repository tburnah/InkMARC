#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# --------- robust imports (your repo may use either style) ----------
try:
    from data.dataset_core import InkMARCIndexedDataset  # type: ignore
    from data.video_io import VideoFrameReader  # type: ignore
except ModuleNotFoundError:
    from dataset_core import InkMARCIndexedDataset  # type: ignore
    from video_io import VideoFrameReader  # type: ignore


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_frame(
    frame_bgr: np.ndarray,
    *,
    img_size: int,
    mode: str,
) -> np.ndarray:
    """
    frame_bgr: HxWx3 uint8 (OpenCV BGR)
    returns: 1x3ximg_sizeximg_size float32 (NCHW) normalized like torchvision.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR, got {frame_bgr.shape}")

    # BGR -> RGB
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0  # HWC in [0,1]
    h, w, _ = rgb.shape

    if mode == "warp":
        # simple resize to square (matches typical old pipeline if it resized)
        import cv2
        rgb_rs = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    elif mode == "letterbox":
        # aspect-preserving resize + pad to square
        import cv2
        scale = img_size / max(h, w)
        nh = int(round(h * scale))
        nw = int(round(w * scale))
        rgb_small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((img_size, img_size, 3), dtype=np.float32)
        top = (img_size - nh) // 2
        left = (img_size - nw) // 2
        canvas[top : top + nh, left : left + nw, :] = rgb_small
        rgb_rs = canvas

    else:
        raise ValueError(f"Unknown mode={mode}. Use warp|letterbox")

    # HWC -> NCHW
    x = np.transpose(rgb_rs, (2, 0, 1))[None, ...]  # 1x3xHxW

    # torchvision normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    x = (x - mean) / std
    return x.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to old touch ONNX model")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--mode", default="warp", choices=["warp", "letterbox"])
    ap.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = evaluate all supervised frames")
    ap.add_argument("--stride", type=int, default=1, help="Evaluate every Nth frame for speed")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(str(onnx_path))

    # Load ONNX
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Load dataset + video reader
    ds = InkMARCIndexedDataset(
        dataset_root=args.dataset_root,
        index_version=args.version,
        split=args.split,
        require_labels=True,
        transform=None,
    )
    vr = VideoFrameReader(
        dataset_root=args.dataset_root,
        manifest=ds.manifest,
        cache_size=2,
    )

    y_true: List[int] = []
    y_pred: List[int] = []

    n_total = 0
    n_supervised = 0

    # Iterate samples
    limit = args.max_samples if args.max_samples and args.max_samples > 0 else None
    taken = 0

    for i in range(0, len(ds), max(1, args.stride)):
        s: Dict[str, Any] = ds[i]
        labels = np.array(s["labels"], dtype=np.float64)
        mask = np.array(s["label_mask"], dtype=np.float64)

        # Your confirmed schema:
        # idx0=x, idx1=y, idx2=pressure
        pressure = float(labels[2])
        touch = 1 if pressure > float(args.pressure_touch_threshold) else 0

        # supervised if pressure channel present
        # (you reported label_mask all ones; still keep it correct)
        touch_mask = 1 if float(mask[2]) != 0.0 else 0

        n_total += 1
        if touch_mask == 0:
            continue
        n_supervised += 1

        meta = s["meta"]
        frame = vr.get_frame(str(meta["file_id"]), int(meta["frame_idx"]), split=args.split)

        x = preprocess_frame(frame, img_size=args.img_size, mode=args.mode)

        out = session.run([output_name], {input_name: x})[0]
        out = np.asarray(out).reshape(-1)

        # Old models sometimes output sigmoid prob, sometimes logits.
        # If values are outside [0,1], treat as logits and apply sigmoid.
        if np.any(out < 0.0) or np.any(out > 1.0):
            prob = sigmoid(out)[0]
        else:
            prob = float(out[0])

        pred = 1 if prob >= 0.5 else 0

        y_true.append(touch)
        y_pred.append(pred)

        taken += 1
        if limit is not None and taken >= limit:
            break

    ds.close()
    try:
        vr.release()
    except Exception:
        pass

    if len(y_true) == 0:
        print("No supervised samples found (touch_mask==0 for all).")
        return

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Evaluated split={args.split} supervised={len(y_true)} (total={n_total}, supervised={n_supervised})")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Touch", "Touch"]))

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
