#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate touch/location metrics by participant, defaulting to original-only data.",
    )
    parser.add_argument("--config", default=None, help="Config to evaluate. Required unless --predictions-csv is used.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate. Required unless --predictions-csv is used.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions-csv", default=None, help="Optional existing touch_location_frame_predictions.csv to summarise.")
    parser.add_argument("--include-textured", action="store_true", help="Do not force data.variant_kind_filter=[original].")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--raw-threshold", type=float, default=0.30)
    parser.add_argument("--hyst-on", type=float, default=0.15)
    parser.add_argument("--hyst-off", type=float, default=0.05)
    parser.add_argument("--hyst-min-on", type=int, default=1)
    parser.add_argument("--hyst-min-off", type=int, default=1)
    parser.add_argument("--max-episode-gap", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions_csv:
        rows = read_prediction_rows(Path(args.predictions_csv))
    else:
        if args.config is None:
            raise ValueError("--config is required unless --predictions-csv is provided.")
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --predictions-csv is provided.")
        from inkdepth.utils.config import load_config, parse_overrides

        cfg = load_config(args.config, parse_overrides(args.override))
        if not args.include_textured:
            cfg.data.variant_kind_filter = ["original"]
        rows = run_model(cfg, args.checkpoint, args.split)

    rows.sort(key=lambda row: (str(row["participant_id"]), str(row["video_id"]), int(row["frame_index"])))
    write_csv(output_dir / "touch_location_frame_predictions.csv", rows)

    overall = evaluate_rows(rows, args)
    participant_rows = participant_metrics(rows, args, args.split)

    write_csv(output_dir / "participant_metrics.csv", participant_rows)
    (output_dir / "participant_metrics.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "original_only": not args.include_textured,
                "variant_kind_filter": None if args.include_textured else ["original"],
                "participants": len(participant_rows),
                "overall": overall,
                "by_participant": participant_rows,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(f"wrote {len(participant_rows)} participant rows to {output_dir / 'participant_metrics.csv'}")
    print(f"total_rows={len(rows)} valid_touch_rows={overall['valid_touch_rows']}")


def run_model(cfg: Any, checkpoint: str, split: str) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    from evaluate_touch_location_report import collect_rows
    from inkdepth.datasets import StylusDataset, stylus_collate_fn
    from inkdepth.models import build_model_from_config
    from inkdepth.utils.checkpoints import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)

    ds = StylusDataset(cfg, split=split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stylus_collate_fn,
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", False)) and int(cfg.training.num_workers) > 0,
    )
    with torch.no_grad():
        return collect_rows(model, loader, cfg, device)


def evaluate_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    y = [int(row["touch_true"]) for row in rows if row["has_touch"]]
    raw_pred = [int(float(row["touch_prob"]) >= float(args.raw_threshold)) for row in rows if row["has_touch"]]
    raw_metrics = binary_metrics(y, raw_pred)
    raw_metrics["threshold"] = float(args.raw_threshold)

    hyst_pred_all = apply_hysteresis(
        rows,
        on_threshold=float(args.hyst_on),
        off_threshold=float(args.hyst_off),
        min_on_samples=int(args.hyst_min_on),
        min_off_samples=int(args.hyst_min_off),
    )
    hyst_pred = [pred for row, pred in zip(rows, hyst_pred_all) if row["has_touch"]]
    hyst_metrics = binary_metrics(y, hyst_pred)
    hyst_metrics.update(episode_metrics(rows, hyst_pred_all, max_gap=int(args.max_episode_gap)))
    hyst_metrics.update(
        {
            "on_threshold": float(args.hyst_on),
            "off_threshold": float(args.hyst_off),
            "min_on_samples": int(args.hyst_min_on),
            "min_off_samples": int(args.hyst_min_off),
        }
    )

    return {
        "rows": len(rows),
        "valid_touch_rows": len(y),
        "valid_xy_rows": sum(1 for row in rows if row["valid_xy"]),
        "xy_by_touch": {
            "touch_true": xy_summary(rows, touch_true=1),
            "touch_false": xy_summary(rows, touch_true=0),
        },
        "raw_threshold": raw_metrics,
        "hysteresis": hyst_metrics,
    }


def participant_metrics(rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["participant_id"])].append(row)

    out: list[dict[str, Any]] = []
    for participant_id in sorted(grouped):
        participant_rows = grouped[participant_id]
        metrics = evaluate_rows(participant_rows, args)
        raw = metrics["raw_threshold"]
        hyst = metrics["hysteresis"]
        touch_true = metrics["xy_by_touch"]["touch_true"]
        touch_false = metrics["xy_by_touch"]["touch_false"]
        out.append(
            {
                "participant_id": participant_id,
                "split": split,
                "rows": metrics["rows"],
                "valid_touch_rows": metrics["valid_touch_rows"],
                "valid_xy_rows": metrics["valid_xy_rows"],
                "raw_threshold": raw["threshold"],
                "raw_accuracy": raw["accuracy"],
                "raw_precision": raw["precision"],
                "raw_recall": raw["recall"],
                "raw_f1": raw["f1"],
                "raw_tn": raw["tn"],
                "raw_fp": raw["fp"],
                "raw_fn": raw["fn"],
                "raw_tp": raw["tp"],
                "hyst_on_threshold": hyst["on_threshold"],
                "hyst_off_threshold": hyst["off_threshold"],
                "hyst_min_on": hyst["min_on_samples"],
                "hyst_min_off": hyst["min_off_samples"],
                "hyst_accuracy": hyst["accuracy"],
                "hyst_precision": hyst["precision"],
                "hyst_recall": hyst["recall"],
                "hyst_frame_f1": hyst["f1"],
                "hyst_episode_precision": hyst["episode_precision"],
                "hyst_episode_recall": hyst["episode_recall"],
                "hyst_episode_f1": hyst["episode_f1"],
                "hyst_true_episodes": hyst["true_episodes"],
                "hyst_pred_episodes": hyst["pred_episodes"],
                **prefixed_xy("touch_true", touch_true),
                **prefixed_xy("touch_false", touch_false),
            }
        )
    return out


def prefixed_xy(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_xy_count": values.get("count", 0),
        f"{prefix}_mean_euclidean_px": values.get("mean_euclidean_px", ""),
        f"{prefix}_median_euclidean_px": values.get("median_euclidean_px", ""),
        f"{prefix}_normalized_l1": values.get("normalized_l1", ""),
        f"{prefix}_normalized_l2": values.get("normalized_l2", ""),
    }


def xy_summary(rows: list[dict[str, Any]], touch_true: int) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["has_touch"] and row["touch_true"] == touch_true and row["valid_xy"] and row["xy_l2_norm"] != ""
    ]
    if not selected:
        return {"count": 0}
    pix = [float(row["xy_err_pix"]) for row in selected]
    l1 = [float(row["xy_l1_norm"]) for row in selected]
    l2 = [float(row["xy_l2_norm"]) for row in selected]
    return {
        "count": len(selected),
        "mean_euclidean_px": mean(pix),
        "median_euclidean_px": median(pix),
        "normalized_l1": mean(l1),
        "normalized_l2": mean(l2),
    }


def binary_metrics(y: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 1)
    tn = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 0)
    fp = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 1)
    fn = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count": len(y),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def apply_hysteresis(
    rows: list[dict[str, Any]],
    on_threshold: float,
    off_threshold: float,
    min_on_samples: int,
    min_off_samples: int,
) -> list[int]:
    preds = [0] * len(rows)
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[str(row["video_id"])].append(i)
    for indices in grouped.values():
        active = False
        on_count = 0
        off_count = 0
        for index in indices:
            prob = float(rows[index]["touch_prob"])
            if active:
                if prob < off_threshold:
                    off_count += 1
                else:
                    off_count = 0
                if off_count >= min_off_samples:
                    active = False
                    off_count = 0
                on_count = 0
            else:
                if prob >= on_threshold:
                    on_count += 1
                else:
                    on_count = 0
                if on_count >= min_on_samples:
                    active = True
                    on_count = 0
                off_count = 0
            preds[index] = int(active)
    return preds


def episode_metrics(rows: list[dict[str, Any]], pred: list[int], max_gap: int) -> dict[str, Any]:
    truth = [int(row["touch_true"]) if row["has_touch"] else 0 for row in rows]
    truth_episodes = collect_episodes(rows, truth, max_gap)
    pred_episodes = collect_episodes(rows, pred, max_gap)
    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    truth_by_video: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, episode in enumerate(truth_episodes):
        truth_by_video[episode["video_id"]].append((i, episode))
    for pred_i, pred_episode in enumerate(pred_episodes):
        for truth_i, truth_episode in truth_by_video.get(pred_episode["video_id"], []):
            if intervals_overlap(pred_episode, truth_episode):
                matched_pred.add(pred_i)
                matched_truth.add(truth_i)
                break
    precision = len(matched_pred) / max(1, len(pred_episodes))
    recall = len(matched_truth) / max(1, len(truth_episodes))
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "episode_precision": precision,
        "episode_recall": recall,
        "episode_f1": f1,
        "true_episodes": len(truth_episodes),
        "pred_episodes": len(pred_episodes),
    }


def collect_episodes(rows: list[dict[str, Any]], values: list[int], max_gap: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row, value in zip(rows, values):
        video_id = str(row["video_id"])
        frame_index = int(row["frame_index"])
        if value:
            if current is None or current["video_id"] != video_id or frame_index - int(current["end"]) > max_gap:
                if current is not None:
                    episodes.append(current)
                current = {"video_id": video_id, "start": frame_index, "end": frame_index}
            else:
                current["end"] = frame_index
        elif current is not None:
            episodes.append(current)
            current = None
    if current is not None:
        episodes.append(current)
    return episodes


def intervals_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return str(a["video_id"]) == str(b["video_id"]) and int(a["start"]) <= int(b["end"]) and int(b["start"]) <= int(a["end"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            row["participant_id"] = str(row.get("participant_id", "UNKNOWN"))
            row["video_id"] = str(row.get("video_id", ""))
            row["task_id"] = str(row.get("task_id", ""))
            row["frame_index"] = int(row.get("frame_index") or 0)
            row["has_touch"] = parse_bool(row.get("has_touch"))
            row["touch_true"] = None if row.get("touch_true") in {"", None} else int(float(row["touch_true"]))
            row["touch_prob"] = float(row.get("touch_prob") or 0.0)
            row["valid_xy"] = parse_bool(row.get("valid_xy"))
            for key in ["xy_pred_x", "xy_pred_y", "xy_true_x", "xy_true_y", "xy_l1_norm", "xy_l2_norm", "xy_err_pix"]:
                if row.get(key) not in {"", None}:
                    row[key] = float(row[key])
                else:
                    row[key] = ""
            rows.append(row)
    return rows


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
