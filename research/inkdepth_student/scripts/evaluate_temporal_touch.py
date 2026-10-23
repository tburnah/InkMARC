#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import InkDepthStudent
from inkdepth.training.validate import _binary_metrics_from_arrays
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary touch predictions with temporal smoothing.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--smooth-windows", default="1,3,5,9,15,21")
    parser.add_argument("--ema-alphas", default="0.15,0.25,0.35,0.50")
    parser.add_argument("--hysteresis-pairs", default="0.35:0.65,0.40:0.60,0.45:0.55")
    parser.add_argument(
        "--sequence-key",
        choices=["video_id", "base_stem"],
        default="video_id",
        help="video_id keeps texture variants separate; base_stem groups texture variants from the same H5.",
    )
    parser.add_argument("--save-frame-predictions", action="store_true")
    parser.add_argument(
        "--stable-only",
        action="store_true",
        help="Report metrics only on frames outside transition window and above min touch pressure.",
    )
    parser.add_argument("--exclude-transition-window", type=int, default=5)
    parser.add_argument("--min-touch-pressure", type=float, default=1000.0)
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    output_dir = Path(args.output_dir or Path(cfg.output_dir) / f"temporal_touch_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InkDepthStudent.from_config(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    ds = StylusDataset(cfg, split=args.split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stylus_collate_fn,
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", False)) and int(cfg.training.num_workers) > 0,
    )

    rows = collect_predictions(model, loader, device)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows were produced. Check manifest/split/config.")
    df = add_sequence_keys(df, args.sequence_key)
    df = add_pressure_audit_columns(df, ds.records, args)
    df = df.sort_values(["sequence_id", "frame_index", "video_id"]).reset_index(drop=True)
    if args.stable_only:
        before = len(df)
        df = df[stable_mask(df, args)].copy().reset_index(drop=True)
        print(f"stable_only kept {len(df)}/{before} rows")
    valid = df["has_touch"] & df["touch_true"].notna() & df["touch_prob"].notna()
    if not valid.any():
        raise RuntimeError("No valid touch labels were found. Check label config and split.")

    thresholds = parse_float_list(args.thresholds)
    smooth_windows = parse_int_list(args.smooth_windows)
    ema_alphas = parse_float_list(args.ema_alphas)
    hysteresis_pairs = parse_pairs(args.hysteresis_pairs)

    metric_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metric_rows.append(evaluate_probs(df, df["touch_prob"].to_numpy(float), f"raw_t{threshold:.2f}", threshold))

    for window in smooth_windows:
        if window <= 1:
            continue
        probs = grouped_centered_rolling_mean(df, "touch_prob", window)
        for threshold in thresholds:
            metric_rows.append(evaluate_probs(df, probs, f"mean_w{window}_t{threshold:.2f}", threshold))

    for alpha in ema_alphas:
        probs = grouped_ema(df, "touch_prob", alpha)
        for threshold in thresholds:
            metric_rows.append(evaluate_probs(df, probs, f"ema_a{alpha:.2f}_t{threshold:.2f}", threshold))

    for low, high in hysteresis_pairs:
        preds = grouped_hysteresis(df, "touch_prob", low, high)
        metric_rows.append(evaluate_preds(df, preds, f"hyst_{low:.2f}_{high:.2f}"))

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["touch_f1", "touch_accuracy"], ascending=[False, False]
    )
    metrics_csv = output_dir / "temporal_touch_metrics.csv"
    metrics_json = output_dir / "temporal_touch_metrics.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(metrics_df.to_dict("records"), indent=2, allow_nan=False), encoding="utf-8")

    by_sequence = summarize_by_sequence(df)
    by_sequence.to_csv(output_dir / "temporal_touch_by_sequence.csv", index=False)
    if args.save_frame_predictions:
        df.to_csv(output_dir / "temporal_touch_frame_predictions.csv", index=False)

    best = metrics_df.iloc[0].to_dict()
    print(f"saved {metrics_csv}")
    print(f"saved {metrics_json}")
    print(f"sequences={df['sequence_id'].nunique()} frames={len(df)} valid_touch={int(valid.sum())}")
    print(
        "best "
        f"{best['method']}: acc={best['touch_accuracy']:.4f} "
        f"f1={best['touch_f1']:.4f} precision={best['touch_precision']:.4f} recall={best['touch_recall']:.4f}"
    )


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="predict"):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(batch["image"])
        logits = outputs.get("touch_logits")
        if logits is None:
            raise RuntimeError("Model produced no touch_logits. Use a binary touch config/checkpoint.")
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        bsz = len(probs)
        for i in range(bsz):
            has_touch = bool(batch["has_touch"][i].detach().cpu())
            rows.append(
                {
                    "video_id": str(batch["video_id"][i]),
                    "participant_id": str(batch["participant_id"][i]),
                    "task_id": str(batch["task_id"][i]),
                    "frame_index": int(batch["frame_index"][i].detach().cpu()),
                    "has_touch": has_touch,
                    "touch_true": int(float(batch["touch"][i].detach().cpu()) > 0.5) if has_touch else None,
                    "touch_prob": float(probs[i]),
                    "valid_xy": bool(batch["valid_xy"][i].detach().cpu()),
                    "pressure": float(batch["pressure"][i].detach().cpu()),
                }
            )
    return rows


def add_sequence_keys(df: pd.DataFrame, sequence_key: str) -> pd.DataFrame:
    out = df.copy()
    if sequence_key == "base_stem":
        out["sequence_id"] = out["video_id"].astype(str).str.split("__tex_", n=1, expand=True)[0]
    else:
        out["sequence_id"] = out["video_id"].astype(str)
    return out


def add_pressure_audit_columns(df: pd.DataFrame, records: list[dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    rec_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in records:
        rec_lookup[(str(rec.get("video_id", rec.get("video", ""))), int(rec.get("frame_index", rec.get("index", -1))))] = rec
    h5_to_indices: dict[str, set[int]] = defaultdict(set)
    row_meta: list[tuple[str | None, int]] = []
    for row in out.itertuples(index=False):
        rec = rec_lookup.get((str(row.video_id), int(row.frame_index)))
        h5_path = str(rec.get("label_h5_path") or "") if rec is not None else ""
        frame_index = int(row.frame_index)
        row_meta.append((h5_path or None, frame_index))
        if h5_path:
            h5_to_indices[h5_path].add(frame_index)

    state_by_key: dict[tuple[str, int], int] = {}
    distance_by_key: dict[tuple[str, int], int | None] = {}
    pressure_by_key: dict[tuple[str, int], float | None] = {}
    for h5_path in h5_to_indices:
        path = Path(h5_path)
        if not path.exists():
            continue
        try:
            with h5py.File(path, "r") as h5:
                labels = np.asarray(h5["labels"][:], dtype=np.float32)
                masks = np.asarray(h5["label_mask"][:], dtype=np.uint8)
        except Exception:
            continue
        states, pressures = derive_states_and_pressures(labels, masks, float(args.pressure_touch_threshold))
        distances = transition_distances(states)
        for frame_index in h5_to_indices[h5_path]:
            if 0 <= frame_index < len(states):
                state_by_key[(h5_path, frame_index)] = int(states[frame_index])
                distance_by_key[(h5_path, frame_index)] = int(distances[frame_index])
                pressure_by_key[(h5_path, frame_index)] = pressures[frame_index]

    out["derived_state"] = [
        state_by_key.get((h5_path, frame_index), np.nan) if h5_path is not None else np.nan
        for h5_path, frame_index in row_meta
    ]
    out["transition_distance"] = [
        distance_by_key.get((h5_path, frame_index), np.nan) if h5_path is not None else np.nan
        for h5_path, frame_index in row_meta
    ]
    out["h5_pressure"] = [
        pressure_by_key.get((h5_path, frame_index), np.nan) if h5_path is not None else np.nan
        for h5_path, frame_index in row_meta
    ]
    return out


def derive_states_and_pressures(labels: np.ndarray, masks: np.ndarray, pressure_threshold: float) -> tuple[np.ndarray, list[float | None]]:
    valid_xy = (
        (masks[:, 0] != 0)
        & (masks[:, 1] != 0)
        & np.isfinite(labels[:, 0])
        & np.isfinite(labels[:, 1])
    )
    pressure_valid = (masks[:, 2] != 0) & np.isfinite(labels[:, 2])
    touch = valid_xy & pressure_valid & (labels[:, 2] > pressure_threshold)
    states = np.zeros(labels.shape[0], dtype=np.int8)
    states[valid_xy & ~touch] = 1
    states[touch] = 2
    pressures: list[float | None] = [float(p) if ok else None for p, ok in zip(labels[:, 2], pressure_valid)]
    return states, pressures


def transition_distances(states: np.ndarray) -> np.ndarray:
    distances = np.full(states.shape[0], fill_value=10**9, dtype=np.int32)
    transitions = np.flatnonzero(states[1:] != states[:-1]) + 1
    if transitions.size == 0:
        return distances
    indices = np.arange(states.shape[0], dtype=np.int32)
    insert_pos = np.searchsorted(transitions, indices)
    right = np.where(insert_pos < transitions.size, transitions[np.minimum(insert_pos, transitions.size - 1)], 10**9)
    left_pos = np.maximum(insert_pos - 1, 0)
    left = np.where(insert_pos > 0, transitions[left_pos], -10**9)
    return np.minimum(np.abs(indices - left), np.abs(indices - right)).astype(np.int32)


def stable_mask(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    valid_touch = df["has_touch"].to_numpy(bool) & df["touch_true"].notna().to_numpy()
    transition = df["transition_distance"].fillna(-1).astype(float).to_numpy()
    pressure = df["h5_pressure"].fillna(np.nan).astype(float).to_numpy()
    y = df["touch_true"].fillna(-1).astype(int).to_numpy()
    far = transition > int(args.exclude_transition_window)
    pressure_ok = (y == 0) | (pressure > float(args.min_touch_pressure))
    return valid_touch & far & pressure_ok


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_pairs(value: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        low, high = part.split(":", 1)
        pairs.append((float(low), float(high)))
    return pairs


def grouped_centered_rolling_mean(df: pd.DataFrame, column: str, window: int) -> np.ndarray:
    values = np.empty(len(df), dtype=np.float64)
    for _, idx in df.groupby("sequence_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        series = pd.Series(df.loc[idx_arr, column].to_numpy(float))
        values[idx_arr] = series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return values


def grouped_ema(df: pd.DataFrame, column: str, alpha: float) -> np.ndarray:
    values = np.empty(len(df), dtype=np.float64)
    for _, idx in df.groupby("sequence_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        raw = df.loc[idx_arr, column].to_numpy(float)
        smoothed = np.empty_like(raw)
        prev = raw[0]
        for i, value in enumerate(raw):
            prev = alpha * value + (1.0 - alpha) * prev
            smoothed[i] = prev
        values[idx_arr] = smoothed
    return values


def grouped_hysteresis(df: pd.DataFrame, column: str, low: float, high: float) -> np.ndarray:
    preds = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("sequence_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        probs = df.loc[idx_arr, column].to_numpy(float)
        state = int(probs[0] >= high)
        seq_preds = np.empty(len(probs), dtype=np.int64)
        for i, prob in enumerate(probs):
            if state == 0 and prob >= high:
                state = 1
            elif state == 1 and prob <= low:
                state = 0
            seq_preds[i] = state
        preds[idx_arr] = seq_preds
    return preds


def evaluate_probs(df: pd.DataFrame, probs: np.ndarray, method: str, threshold: float) -> dict[str, Any]:
    return evaluate_preds(df, (probs >= threshold).astype(int), method)


def evaluate_preds(df: pd.DataFrame, preds: np.ndarray, method: str) -> dict[str, Any]:
    valid = df["has_touch"].to_numpy(bool) & df["touch_true"].notna().to_numpy()
    y = df.loc[valid, "touch_true"].astype(int).to_numpy()
    p = preds[valid].astype(int)
    metrics = _binary_metrics_from_arrays(y, p, "touch")
    return {
        "method": method,
        "touch_count": metrics["touch_count"],
        "touch_accuracy": metrics["touch_accuracy"],
        "touch_precision": metrics["touch_precision"],
        "touch_recall": metrics["touch_recall"],
        "touch_f1": metrics["touch_f1"],
        "tn": metrics["touch_confusion"]["tn"],
        "fp": metrics["touch_confusion"]["fp"],
        "fn": metrics["touch_confusion"]["fn"],
        "tp": metrics["touch_confusion"]["tp"],
    }


def summarize_by_sequence(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence_id, group in df.groupby("sequence_id", sort=False):
        valid = group["has_touch"] & group["touch_true"].notna()
        y = group.loc[valid, "touch_true"].astype(int).to_numpy()
        p = (group.loc[valid, "touch_prob"].astype(float).to_numpy() >= 0.5).astype(int)
        row = {
            "sequence_id": sequence_id,
            "frames": int(len(group)),
            "valid_touch": int(valid.sum()),
            "touch_rate": float(y.mean()) if len(y) else None,
        }
        if len(y):
            row.update(_binary_metrics_from_arrays(y, p, "touch"))
            row.pop("touch_confusion", None)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
