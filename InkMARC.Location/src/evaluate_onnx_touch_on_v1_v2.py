#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# --------- robust imports (repo may use either style) ----------
try:
    from data.dataset_core import InkMARCIndexedDataset  # type: ignore
    from data.video_io import VideoFrameReader  # type: ignore
except ModuleNotFoundError:
    from dataset_core import InkMARCIndexedDataset  # type: ignore
    from video_io import VideoFrameReader  # type: ignore


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_frame(frame_bgr: np.ndarray, *, img_size: int, mode: str) -> np.ndarray:
    """
    frame_bgr: HxWx3 uint8 (OpenCV BGR)
    returns: 1x3ximg_sizeximg_size float32 (NCHW) normalized like torchvision
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR, got {frame_bgr.shape}")

    import cv2

    # BGR -> RGB float [0,1]
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    h, w, _ = rgb.shape

    if mode == "warp":
        rgb_rs = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    elif mode == "letterbox":
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
    x = np.transpose(rgb_rs, (2, 0, 1))[None, ...]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    x = (x - mean) / std
    return x.astype(np.float32)


def infer_prob(session: ort.InferenceSession, input_name: str, output_name: str, x: np.ndarray) -> float:
    out = session.run([output_name], {input_name: x})[0]
    out = np.asarray(out).reshape(-1)

    # If outside [0,1], treat as logits
    if np.any(out < 0.0) or np.any(out > 1.0):
        return float(sigmoid(out)[0])
    return float(out[0])


def report_block(title: str, y_true: List[int], y_pred: List[int]) -> None:
    if len(y_true) == 0:
        print(f"\n[{title}] No samples.")
        return
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n[{title}] n={len(y_true)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Touch", "Touch"]))
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--mode", default="letterbox", choices=["warp", "letterbox"])
    ap.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    ap.add_argument("--include-no-pressure-negatives", action="store_true",
                    help="Include samples where label_mask[2]==0 as No-Touch negatives.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(str(onnx_path))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    ds = InkMARCIndexedDataset(
        dataset_root=args.dataset_root,
        index_version=args.version,
        split=args.split,
        require_labels=False,
        transform=None,
    )
    vr = VideoFrameReader(args.dataset_root, ds.manifest, cache_size=2)

    # Overall (optional), plus subsets
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    y_true_near: List[int] = []   # label_mask[2]==1 (pressure stream present; near-surface domain)
    y_pred_near: List[int] = []

    y_true_far: List[int] = []    # label_mask[2]==0 (no pressure reading; far-hover easy negatives)
    y_pred_far: List[int] = []

    n_total = 0
    n_used = 0
    n_near = 0
    n_far = 0

    cap = args.max_samples if args.max_samples and args.max_samples > 0 else None

    for i in range(0, len(ds), max(1, args.stride)):
        s: Dict[str, Any] = ds[i]
        labels = np.array(s["labels"], dtype=np.float64)
        mask = np.array(s["label_mask"], dtype=np.float64)

        n_total += 1

        # Your confirmed schema: idx2 = pressure
        pressure = float(labels[2])
        has_pressure = (float(mask[2]) != 0.0)

        if not has_pressure and not args.include_no_pressure_negatives:
            continue

        # Ground truth label
        if has_pressure:
            touch = 1 if pressure > float(args.pressure_touch_threshold) else 0
        else:
            # no pressure reading -> define as far-hover negative for this experiment
            touch = 0

        meta = s["meta"]
        frame = vr.get_frame(str(meta["file_id"]), int(meta["frame_idx"]), split=args.split)
        x = preprocess_frame(frame, img_size=args.img_size, mode=args.mode)
        prob = infer_prob(session, input_name, output_name, x)
        pred = 1 if prob >= 0.5 else 0

        # Record overall + subset
        y_true_all.append(touch)
        y_pred_all.append(pred)
        n_used += 1

        if has_pressure:
            y_true_near.append(touch)
            y_pred_near.append(pred)
            n_near += 1
        else:
            y_true_far.append(touch)
            y_pred_far.append(pred)
            n_far += 1

        if cap is not None and n_used >= cap:
            break

    ds.close()
    try:
        vr.release()
    except Exception:
        pass

    print(f"Evaluated split={args.split} total={n_total} used={n_used} near(has_pressure)={n_near} far(no_pressure)={n_far}")
    print(f"Mode={args.mode} img_size={args.img_size} threshold={args.pressure_touch_threshold} include_no_pressure_negatives={args.include_no_pressure_negatives}")

    if args.include_no_pressure_negatives:
        report_block("OVERALL (near + far)", y_true_all, y_pred_all)

    report_block("NEAR-SURFACE ONLY (label_mask[2]==1)", y_true_near, y_pred_near)

    if args.include_no_pressure_negatives:
        report_block("FAR-HOVER ONLY (label_mask[2]==0, forced No-Touch)", y_true_far, y_pred_far)


if __name__ == "__main__":
    main()
