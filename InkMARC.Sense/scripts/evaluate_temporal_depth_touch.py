#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_temporal_touch import (  # noqa: E402
    TemporalDepthTouchContextNet,
    TemporalDepthTouchNet,
    TemporalTouchDepthDataset,
    binary_metrics,
    parse_int_list,
    temporal_collate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a temporal local-depth touch checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(payload.get("config") or {})
    if not cfg:
        raise RuntimeError(f"Checkpoint has no saved config: {checkpoint_path}")
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / f"eval_{args.split}_{checkpoint_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    args_ns = argparse.Namespace(**cfg)
    offsets = cfg.get("offsets")
    if offsets is None:
        offsets = parse_int_list(str(cfg["offsets"]))
    offsets = [int(x) for x in offsets]

    ds = TemporalTouchDepthDataset(args_ns, split=args.split, offsets=offsets, training=False)
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg.get("batch_size", 128)),
        "shuffle": False,
        "num_workers": int(cfg.get("num_workers", 8)),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": temporal_collate,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(ds, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kind = str(cfg.get("model_kind", "local"))
    model = (
        TemporalDepthTouchContextNet(in_channels=len(offsets))
        if model_kind == "local_context"
        else TemporalDepthTouchNet(in_channels=len(offsets))
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    rows = collect_predictions(model, loader, device)
    ds.close()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No predictions were produced.")
    thresholds = np.linspace(0.05, 0.95, 91)
    metrics_df = threshold_sweep(df, thresholds)
    if args.threshold is None:
        threshold = float(metrics_df.sort_values(["touch_accuracy", "touch_f1"], ascending=False).iloc[0]["threshold"])
    else:
        threshold = float(args.threshold)
    df["touch_pred"] = (df["touch_prob"] >= threshold).astype(int)

    metrics_df.to_csv(output_dir / "threshold_metrics.csv", index=False)
    group_metrics(df, "participant_id", threshold).to_csv(output_dir / "metrics_by_participant.csv", index=False)
    group_metrics(df, "task_id", threshold).to_csv(output_dir / "metrics_by_task.csv", index=False)
    group_metrics(df, "video_id", threshold).to_csv(output_dir / "metrics_by_video.csv", index=False)
    if args.save_predictions:
        df.to_csv(output_dir / "frame_predictions.csv", index=False)

    best_acc = metrics_df.sort_values(["touch_accuracy", "touch_f1"], ascending=False).iloc[0].to_dict()
    best_f1 = metrics_df.sort_values(["touch_f1", "touch_accuracy"], ascending=False).iloc[0].to_dict()
    print(f"saved {output_dir}")
    print(f"samples={len(df)} threshold={threshold:.2f}")
    print(
        "best_accuracy "
        f"acc={best_acc['touch_accuracy']:.4f} f1={best_acc['touch_f1']:.4f} "
        f"precision={best_acc['touch_precision']:.4f} recall={best_acc['touch_recall']:.4f} "
        f"threshold={best_acc['threshold']:.2f}"
    )
    print(
        "best_f1 "
        f"acc={best_f1['touch_accuracy']:.4f} f1={best_f1['touch_f1']:.4f} "
        f"precision={best_f1['touch_precision']:.4f} recall={best_f1['touch_recall']:.4f} "
        f"threshold={best_f1['threshold']:.2f}"
    )


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="predict"):
        x = batch["x"].to(device, non_blocking=True)
        context = batch.get("context")
        if context is not None:
            context = context.to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True)
        logits = model(x, context, xy) if context is not None else model(x)
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        y = batch["y"].detach().float().cpu().numpy()
        for i, prob in enumerate(probs):
            video_id = str(batch["video_id"][i])
            rows.append(
                {
                    "video_id": video_id,
                    "participant_id": participant_from_video_id(video_id),
                    "task_id": task_from_video_id(video_id),
                    "frame_index": int(batch["frame_index"][i]),
                    "touch_true": int(y[i] >= 0.5),
                    "touch_prob": float(prob),
                    "xy_x": float(batch["xy"][i][0]),
                    "xy_y": float(batch["xy"][i][1]),
                    "pressure": float(batch["pressure"][i]),
                }
            )
    return rows


def threshold_sweep(df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    y = df["touch_true"].astype(float).to_numpy()
    probs = df["touch_prob"].astype(float).to_numpy()
    for threshold in thresholds:
        metrics = binary_metrics(y, probs, float(threshold))
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return pd.DataFrame(rows)


def group_metrics(df: pd.DataFrame, group_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        y = group["touch_true"].astype(float).to_numpy()
        probs = group["touch_prob"].astype(float).to_numpy()
        row = {
            group_col: value,
            "samples": int(len(group)),
            "touch": int((y >= 0.5).sum()),
            "hover": int((y < 0.5).sum()),
            **binary_metrics(y, probs, threshold),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["samples", group_col], ascending=[False, True])


def participant_from_video_id(video_id: str) -> str:
    parts = video_id.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "participant" else "UNKNOWN"


def task_from_video_id(video_id: str) -> str:
    parts = video_id.split("_")
    return parts[2] if len(parts) >= 3 and parts[0] == "participant" else "UNKNOWN"


if __name__ == "__main__":
    main()
