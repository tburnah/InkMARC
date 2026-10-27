#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import zarr
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_temporal_touch import (  # noqa: E402
    convert_xy,
    crop_around_xy,
    derive_states,
    load_items,
    normalize_crop,
    transition_distances,
    valid_xy_pressure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fast touch classifiers from local depth features.")
    parser.add_argument("--manifest", default="/data/baseAVIs_432x672/manifest_texture_replacement_v3_no_missing_h5_no_black_participant_splits.json")
    parser.add_argument("--output-dir", default="outputs/depth_feature_touch_original_w5_stride2")
    parser.add_argument("--variant-kind", default="original")
    parser.add_argument("--offsets", default="-6,-3,-1,0,1,3,6")
    parser.add_argument("--center-stride", type=int, default=2)
    parser.add_argument("--exclude-transition-window", type=int, default=5)
    parser.add_argument("--min-touch-pressure", type=float, default=1000.0)
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--crop-normalize", choices=["none", "local_zscore", "center_median"], default="local_zscore")
    parser.add_argument("--preload-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=120000)
    parser.add_argument("--max-val-samples", type=int, default=30000)
    parser.add_argument("--models", nargs="+", default=["hgb", "extra", "logreg"])
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = parse_offsets(args.offsets)
    masks = build_radial_masks(int(args.crop_size))

    cfg = vars(args).copy()
    cfg["offsets"] = offsets
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    x_train, y_train, train_meta = build_matrix(args, "train", offsets, masks, rng, int(args.max_train_samples))
    x_val, y_val, val_meta = build_matrix(args, "val", offsets, masks, rng, int(args.max_val_samples))
    np.savez_compressed(output_dir / "features_train.npz", x=x_train, y=y_train)
    np.savez_compressed(output_dir / "features_val.npz", x=x_val, y=y_val)
    write_meta(output_dir / "val_meta.csv", val_meta)

    rows: list[dict[str, float | str]] = []
    for name in args.models:
        model = make_model(name, int(args.seed))
        weights = balanced_weights(y_train)
        if name == "logreg":
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=weights)
        probs = predict_proba(model, x_val)
        metrics = binary_metrics(y_val, probs, 0.5)
        metrics.update(best_threshold_metrics(y_val, probs))
        row: dict[str, float | str] = {"model": name, **metrics}
        rows.append(row)
        joblib.dump(model, output_dir / f"{name}.joblib")
        np.save(output_dir / f"{name}_val_probs.npy", probs.astype(np.float32))
        print(format_row(row))

    write_rows(output_dir / "metrics.csv", rows)
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def build_matrix(
    args: argparse.Namespace,
    split: str,
    offsets: list[int],
    masks: dict[str, np.ndarray],
    rng: np.random.Generator,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    items = load_items(Path(args.manifest), split, args.variant_kind)
    rng.shuffle(items)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta: list[dict[str, Any]] = []
    min_offset = min(offsets)
    max_offset = max(offsets)
    for item_i, item in enumerate(items, start=1):
        h5_path = Path(str(item.get("output_h5_path") or item.get("label_h5_path") or ""))
        depth_path = Path(str(item.get("depth_path") or ""))
        if not h5_path.exists() or not depth_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as h5:
                labels = np.asarray(h5["labels"][:], dtype=np.float32)
                label_masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
                corners = np.asarray(h5["corners"][:], dtype=np.float32) if "corners" in h5 else None
                resize_attrs = dict(h5.get("preprocessing/resize_transform", {}).attrs) if "preprocessing/resize_transform" in h5 else {}
            if corners is None:
                continue
            depth_store = zarr.open(depth_path, mode="r")
            depth_arr = depth_store if hasattr(depth_store, "shape") else depth_store[str(item.get("depth_key", "depth_norm"))]
            if bool(args.preload_depth):
                depth_arr = np.asarray(depth_arr[:], dtype=np.float32)
        except Exception as exc:
            print(f"{split}: skip {h5_path}: {exc}")
            continue
        n = min(int(item.get("frame_count") or len(labels)), len(labels), int(depth_arr.shape[0]))
        states = derive_states(labels[:n], label_masks[:n], float(args.pressure_touch_threshold))
        distances = transition_distances(states)
        frame_indices = range(max(0, -min_offset), min(n, n - max_offset), max(1, int(args.center_stride)))
        frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        for frame_index in frame_indices:
            state = int(states[frame_index])
            if state not in {1, 2}:
                continue
            if int(distances[frame_index]) <= int(args.exclude_transition_window):
                continue
            pressure = float(labels[frame_index, 2])
            if state == 2 and pressure <= float(args.min_touch_pressure):
                continue
            if not valid_xy_pressure(labels[frame_index], label_masks[frame_index]):
                continue
            xy = convert_xy(labels[frame_index], corners[frame_index], resize_attrs)
            if xy is None:
                continue
            features = extract_features(
                depth_arr,
                frame_cache,
                frame_index,
                offsets,
                xy.astype(np.float32),
                int(args.crop_size),
                str(args.crop_normalize),
                masks,
            )
            xs.append(features)
            ys.append(1.0 if state == 2 else 0.0)
            meta.append(
                {
                    "video_id": str(item.get("stem") or item.get("base_stem") or h5_path.stem),
                    "frame_index": int(frame_index),
                    "pressure": pressure,
                    "touch": int(state == 2),
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                }
            )
            if max_samples > 0 and len(xs) >= max_samples:
                break
        if item_i == 1 or item_i % 10 == 0:
            print(f"{split}: processed {item_i}/{len(items)} samples={len(xs)}")
        if max_samples > 0 and len(xs) >= max_samples:
            break
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    if max_samples > 0 and len(y) > max_samples:
        idx = stratified_subset(y, max_samples, rng)
        x = x[idx]
        y = y[idx]
        meta = [meta[int(i)] for i in idx]
    print(f"{split}: matrix={x.shape} touch={int(y.sum())} hover={int((y < 0.5).sum())}")
    return x, y, meta


def extract_features(
    depth_arr: Any,
    frame_cache: OrderedDict[int, np.ndarray],
    frame_index: int,
    offsets: list[int],
    xy: np.ndarray,
    crop_size: int,
    crop_normalize: str,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    feats: list[float] = [float(xy[0]), float(xy[1])]
    center_stats: np.ndarray | None = None
    for offset in offsets:
        depth = get_depth_frame(depth_arr, frame_cache, int(frame_index) + int(offset))
        crop = crop_around_xy(depth, xy, crop_size, jitter_pixels=0)
        raw_stats = crop_stats(crop, masks)
        norm_stats = crop_stats(normalize_crop(crop, crop_normalize), masks)
        stats = np.concatenate([raw_stats, norm_stats]).astype(np.float32)
        if offset == 0:
            center_stats = stats
        feats.extend(stats.tolist())
    if center_stats is not None:
        for offset in offsets:
            if offset == 0:
                continue
            depth = get_depth_frame(depth_arr, frame_cache, int(frame_index) + int(offset))
            crop = crop_around_xy(depth, xy, crop_size, jitter_pixels=0)
            stats = np.concatenate([crop_stats(crop, masks), crop_stats(normalize_crop(crop, crop_normalize), masks)]).astype(np.float32)
            feats.extend((stats - center_stats).tolist())
    return np.asarray(feats, dtype=np.float32)


def get_depth_frame(depth_arr: Any, cache: OrderedDict[int, np.ndarray], frame_index: int) -> np.ndarray:
    frame = cache.get(frame_index)
    if frame is None:
        frame = np.asarray(depth_arr[frame_index], dtype=np.float32).squeeze()
        cache[frame_index] = frame
        while len(cache) > 96:
            cache.popitem(last=False)
    else:
        cache.move_to_end(frame_index)
    return frame


def crop_stats(crop: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    crop = np.nan_to_num(np.asarray(crop, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    out: list[float] = []
    for mask in masks.values():
        vals = crop[mask]
        if vals.size == 0:
            out.extend([0.0] * 5)
            continue
        out.extend(
            [
                float(vals.mean()),
                float(vals.std()),
                float(vals.min()),
                float(vals.max()),
                float(vals.max() - vals.min()),
            ]
        )
    return np.asarray(out, dtype=np.float32)


def build_radial_masks(crop_size: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:crop_size, 0:crop_size]
    cx = (crop_size - 1) / 2.0
    cy = (crop_size - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    masks: dict[str, np.ndarray] = {}
    last = np.zeros((crop_size, crop_size), dtype=bool)
    for radius in (3, 5, 9, 15, 25, 40, 60):
        current = rr <= min(radius, crop_size // 2)
        masks[f"disc_{radius}"] = current
        ring = current & ~last
        masks[f"ring_{radius}"] = ring
        last = current
    return masks


def make_model(name: str, seed: int) -> Any:
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    if name == "extra":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
    raise ValueError(f"Unsupported model: {name}")


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    return np.asarray(model.decision_function(x), dtype=np.float32)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    y_bool = y >= 0.5
    pos = max(1, int(y_bool.sum()))
    neg = max(1, int((~y_bool).sum()))
    weights = np.where(y_bool, 0.5 / pos, 0.5 / neg)
    return weights * len(y)


def stratified_subset(y: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y >= 0.5)
    neg = np.flatnonzero(y < 0.5)
    half = count // 2
    pos_n = min(len(pos), half)
    neg_n = min(len(neg), count - pos_n)
    pos_sel = rng.choice(pos, size=pos_n, replace=False)
    neg_sel = rng.choice(neg, size=neg_n, replace=False)
    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return idx


def binary_metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probs >= threshold
    truth = y >= 0.5
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count": float(len(y)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def best_threshold_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    best_acc: dict[str, float] | None = None
    best_f1: dict[str, float] | None = None
    for threshold in np.linspace(0.01, 0.99, 99):
        row = binary_metrics(y, probs, float(threshold))
        row["threshold"] = float(threshold)
        if best_acc is None or row["accuracy"] > best_acc["accuracy"]:
            best_acc = row
        if best_f1 is None or row["f1"] > best_f1["f1"]:
            best_f1 = row
    assert best_acc is not None and best_f1 is not None
    return {
        "best_accuracy": best_acc["accuracy"],
        "best_accuracy_threshold": best_acc["threshold"],
        "best_accuracy_precision": best_acc["precision"],
        "best_accuracy_recall": best_acc["recall"],
        "best_f1": best_f1["f1"],
        "best_f1_threshold": best_f1["threshold"],
        "best_f1_precision": best_f1["precision"],
        "best_f1_recall": best_f1["recall"],
    }


def write_meta(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_row(row: dict[str, float | str]) -> str:
    keys = ["model", "accuracy", "f1", "precision", "recall", "best_accuracy", "best_f1", "best_accuracy_threshold", "best_f1_threshold"]
    parts = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def parse_offsets(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


if __name__ == "__main__":
    main()
