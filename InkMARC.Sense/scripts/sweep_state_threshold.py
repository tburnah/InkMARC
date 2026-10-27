#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import build_model_from_config
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


def parse_thresholds(spec: str) -> np.ndarray:
    if ":" in spec:
        start, stop, step = (float(part) for part in spec.split(":"))
        return np.round(np.arange(start, stop + step * 0.5, step), 10)
    return np.asarray([float(part) for part in spec.split(",") if part.strip()], dtype=float)


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(1, fp + tn),
        "false_negative_rate": fn / max(1, fn + tp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def state_target_mode(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "loss", {}), "state_target", "state")).lower()


def state_interaction_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=1)[:, 1]


@torch.inference_mode()
def collect_state_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    target_mode = state_target_mode(cfg)
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc="collect", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs.get("state_logits")
        if logits is None:
            raise RuntimeError("Model did not return state_logits")
        valid = batch["has_state"].bool()
        if not bool(valid.any()):
            continue
        truth = batch["state"][valid].long()
        if target_mode in {"presence", "interaction", "binary_presence", "background_interaction"}:
            truth = (truth > 0).long()
        score = state_interaction_prob(logits.detach().cpu())[valid]
        ys.append(truth.cpu().numpy().astype(np.int64))
        probs.append(score.numpy().astype(np.float64))
    if not ys:
        raise RuntimeError("No valid state labels were found")
    return np.concatenate(ys), np.concatenate(probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep state interaction threshold on a validation/test split")
    parser.add_argument("--config", default=str(ROOT / "configs/inkdepth_mobilenetv3.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", default="0.01:0.99:0.01")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "latest.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)

    ds = StylusDataset(cfg, split=args.split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stylus_collate_fn,
    )
    y, probs = collect_state_scores(model, loader, cfg, device)
    rows = []
    for threshold in parse_thresholds(args.thresholds):
        pred = (probs >= threshold).astype(np.int64)
        row = {"threshold": float(threshold), **binary_metrics(y, pred)}
        rows.append(row)

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"samples={len(y)} positives={int(y.sum())} negatives={int((y == 0).sum())}")
    print("Best by recall, then precision:")
    for row in sorted(rows, key=lambda r: (r["recall"], r["precision"], r["f1"]), reverse=True)[: args.top_k]:
        print(format_row(row))
    print("Best by F1:")
    print(format_row(max(rows, key=lambda r: (r["f1"], r["recall"]))))
    print("Best by accuracy:")
    print(format_row(max(rows, key=lambda r: (r["accuracy"], r["recall"]))))


def format_row(row: dict[str, float | int]) -> str:
    return (
        f"threshold={row['threshold']:.2f} "
        f"recall={row['recall']:.4f} precision={row['precision']:.4f} "
        f"f1={row['f1']:.4f} accuracy={row['accuracy']:.4f} "
        f"fn={row['fn']} fp={row['fp']} tp={row['tp']} tn={row['tn']} "
        f"fnr={row['false_negative_rate']:.4f} fpr={row['false_positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
