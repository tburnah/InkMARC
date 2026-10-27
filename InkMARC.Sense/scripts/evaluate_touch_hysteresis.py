#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep temporal hysteresis over saved touch probabilities.")
    parser.add_argument("--predictions", required=True, help="CSV from evaluate_temporal_depth_touch.py --save-predictions")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--prob-column",
        default="touch_prob",
        help="Probability column or fusion method, e.g. logistic_fusion or blend_w21_depth0.58.",
    )
    parser.add_argument("--group-column", default=None, help="Defaults to sequence_id when present, else video_id.")
    parser.add_argument("--valid-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--on-thresholds", default="0.05:0.95:0.05")
    parser.add_argument("--off-thresholds", default="0.05:0.95:0.05")
    parser.add_argument("--min-on-samples", default="1,2,3")
    parser.add_argument("--min-off-samples", default="1,2,3,5")
    parser.add_argument(
        "--max-episode-gap",
        type=int,
        default=4,
        help="Maximum frame-index gap for adjacent sampled positives to count as the same episode.",
    )
    parser.add_argument("--save-best-predictions", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent / "hysteresis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_predictions(predictions_path, str(args.prob_column), bool(args.valid_only))
    if not rows:
        raise RuntimeError(f"No rows in {predictions_path}")
    group_column = str(args.group_column or ("sequence_id" if "sequence_id" in rows[0] else "video_id"))
    rows.sort(key=lambda r: (str(r.get(group_column, "")), r["frame_index"], r["video_id"]))

    on_thresholds = parse_float_list(args.on_thresholds)
    off_thresholds = parse_float_list(args.off_thresholds)
    min_on_samples = parse_int_list(args.min_on_samples)
    min_off_samples = parse_int_list(args.min_off_samples)

    sweep_rows: list[dict[str, Any]] = []
    best_predictions_by_key: dict[str, list[int]] = {}
    best_by_accuracy: dict[str, Any] | None = None
    best_by_f1: dict[str, Any] | None = None
    for on_threshold in on_thresholds:
        for off_threshold in off_thresholds:
            if off_threshold > on_threshold:
                continue
            for min_on in min_on_samples:
                for min_off in min_off_samples:
                    preds = apply_hysteresis(rows, on_threshold, off_threshold, min_on, min_off, group_column)
                    metrics = binary_metrics([r["touch_true"] for r in rows], preds)
                    episode = episode_metrics(rows, preds, int(args.max_episode_gap))
                    record = {
                        "on_threshold": on_threshold,
                        "off_threshold": off_threshold,
                        "min_on_samples": min_on,
                        "min_off_samples": min_off,
                        **metrics,
                        **episode,
                    }
                    sweep_rows.append(record)
                    if best_by_accuracy is None or better(record, best_by_accuracy, "touch_accuracy"):
                        best_by_accuracy = record
                        best_predictions_by_key["accuracy"] = preds
                    if best_by_f1 is None or better(record, best_by_f1, "touch_f1"):
                        best_by_f1 = record
                        best_predictions_by_key["f1"] = preds

    if best_by_accuracy is None or best_by_f1 is None:
        raise RuntimeError("No hysteresis combinations were evaluated.")

    write_csv(output_dir / "hysteresis_metrics.csv", sweep_rows)
    write_json(output_dir / "best_accuracy.json", best_by_accuracy)
    write_json(output_dir / "best_f1.json", best_by_f1)
    write_csv(
        output_dir / "metrics_by_video_best_accuracy.csv",
        grouped_binary_metrics(rows, best_predictions_by_key["accuracy"], "video_id"),
    )
    write_csv(
        output_dir / "metrics_by_task_best_accuracy.csv",
        grouped_binary_metrics(rows, best_predictions_by_key["accuracy"], "task_id"),
    )
    write_csv(
        output_dir / "metrics_by_participant_best_accuracy.csv",
        grouped_binary_metrics(rows, best_predictions_by_key["accuracy"], "participant_id"),
    )
    if args.save_best_predictions:
        write_predictions(output_dir / "frame_predictions_best_accuracy.csv", rows, best_predictions_by_key["accuracy"])

    print(f"saved {output_dir}")
    print(f"samples={len(rows)} combinations={len(sweep_rows)} prob_column={args.prob_column} group_column={group_column}")
    print("best_accuracy " + format_record(best_by_accuracy))
    print("best_f1 " + format_record(best_by_f1))


def load_predictions(path: Path, prob_column: str, valid_only: bool) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if valid_only and not valid_touch_row(row):
                continue
            touch_true = row.get("touch_true", "")
            if touch_true == "":
                continue
            video_id = row.get("video_id", "")
            rows.append(
                {
                    **row,
                    "video_id": video_id,
                    "participant_id": row.get("participant_id") or participant_from_video_id(video_id),
                    "task_id": row.get("task_id") or task_from_video_id(video_id),
                    "frame_index": int(row["frame_index"]),
                    "touch_true": int(float(touch_true) >= 0.5),
                    "touch_prob": method_prob(row, prob_column),
                }
            )
    return rows


def valid_touch_row(row: dict[str, str]) -> bool:
    if row.get("touch_true", "") == "":
        return False
    has_touch = row.get("has_touch")
    if has_touch is not None and has_touch != "":
        return has_touch.strip().lower() in {"1", "true", "yes"}
    return True


def method_prob(row: dict[str, str], method: str) -> float:
    if method in row and row[method] != "":
        return float(row[method])
    if method.startswith("blend_w"):
        left, weight_s = method.rsplit("_depth", 1)
        window = int(left.removeprefix("blend_w"))
        weight = float(weight_s)
        depth = float(row[f"depth_mean_w{window}"])
        rgb = float(row[f"rgb_mean_w{window}"])
        return weight * depth + (1.0 - weight) * rgb
    if method.startswith("depth_uncertain_blend_w"):
        left, margin_s = method.rsplit("_m", 1)
        window = int(left.removeprefix("depth_uncertain_blend_w"))
        margin = float(margin_s)
        depth = float(row[f"depth_mean_w{window}"])
        rgb = float(row[f"rgb_mean_w{window}"])
        return 0.5 * depth + 0.5 * rgb if abs(depth - 0.5) <= margin else depth
    raise KeyError(f"Cannot find probability column or reconstruct fusion method: {method}")


def parse_float_list(spec: str) -> list[float]:
    if ":" in spec:
        parts = [float(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Range spec must be start:end:step: {spec}")
        start, end, step = parts
        values = []
        value = start
        while value <= end + step * 0.5:
            values.append(round(value, 10))
            value += step
        return values
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def apply_hysteresis(
    rows: list[dict[str, Any]],
    on_threshold: float,
    off_threshold: float,
    min_on_samples: int,
    min_off_samples: int,
    group_column: str,
) -> list[int]:
    preds = [0] * len(rows)
    grouped_indices = group_indices(rows, group_column)
    for _, indices in grouped_indices.items():
        active = False
        on_count = 0
        off_count = 0
        for index in indices:
            prob = rows[index]["touch_prob"]
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


def binary_metrics(y: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 1)
    tn = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 0)
    fp = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 1)
    fn = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "touch_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "touch_precision": precision,
        "touch_recall": recall,
        "touch_f1": f1,
        "touch_count": len(y),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def episode_metrics(rows: list[dict[str, Any]], pred: list[int], max_gap: int) -> dict[str, Any]:
    truth_episodes = collect_episodes(rows, [r["touch_true"] for r in rows], max_gap)
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
        "matched_true_episodes": len(matched_truth),
        "matched_pred_episodes": len(matched_pred),
    }


def collect_episodes(rows: list[dict[str, Any]], values: list[int], max_gap: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    grouped_indices = group_indices(rows, "video_id")
    for video_id, indices in grouped_indices.items():
        current: dict[str, Any] | None = None
        previous_frame: int | None = None
        for index in indices:
            frame_index = rows[index]["frame_index"]
            if values[index] == 1:
                if current is None or previous_frame is None or frame_index - previous_frame > max_gap:
                    if current is not None:
                        episodes.append(current)
                    current = {"video_id": video_id, "start": frame_index, "end": frame_index, "samples": 1}
                else:
                    current["end"] = frame_index
                    current["samples"] += 1
                previous_frame = frame_index
            else:
                if current is not None:
                    episodes.append(current)
                    current = None
                previous_frame = frame_index
        if current is not None:
            episodes.append(current)
    return episodes


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["start"] <= right["end"] and right["start"] <= left["end"]


def group_indices(rows: list[dict[str, Any]], key: str) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row[key])].append(index)
    return grouped


def grouped_binary_metrics(rows: list[dict[str, Any]], pred: list[int], group_key: str) -> list[dict[str, Any]]:
    output = []
    for value, indices in group_indices(rows, group_key).items():
        y = [rows[i]["touch_true"] for i in indices]
        p = [pred[i] for i in indices]
        output.append(
            {
                group_key: value,
                "samples": len(indices),
                "touch": sum(y),
                "hover": len(y) - sum(y),
                **binary_metrics(y, p),
            }
        )
    return sorted(output, key=lambda r: (-int(r["samples"]), str(r[group_key])))


def better(candidate: dict[str, Any], incumbent: dict[str, Any], primary_key: str) -> bool:
    tie_key = "touch_f1" if primary_key == "touch_accuracy" else "touch_accuracy"
    return (candidate[primary_key], candidate[tie_key]) > (incumbent[primary_key], incumbent[tie_key])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def write_predictions(path: Path, rows: list[dict[str, Any]], pred: list[int]) -> None:
    output_rows = []
    for row, guess in zip(rows, pred):
        output_rows.append({**row, "hysteresis_touch_pred": guess})
    write_csv(path, output_rows)


def format_record(record: dict[str, Any]) -> str:
    return (
        f"acc={record['touch_accuracy']:.4f} f1={record['touch_f1']:.4f} "
        f"precision={record['touch_precision']:.4f} recall={record['touch_recall']:.4f} "
        f"episodes_f1={record['episode_f1']:.4f} "
        f"on={record['on_threshold']:.2f} off={record['off_threshold']:.2f} "
        f"min_on={record['min_on_samples']} min_off={record['min_off_samples']}"
    )


def participant_from_video_id(video_id: str) -> str:
    parts = video_id.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "participant" else "UNKNOWN"


def task_from_video_id(video_id: str) -> str:
    parts = video_id.split("_")
    return parts[2] if len(parts) >= 3 and parts[0] == "participant" else "UNKNOWN"


if __name__ == "__main__":
    main()
