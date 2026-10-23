#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PROBLEM_PARTICIPANTS = {"307", "321"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run participant-level cross-validation with problematic videos excluded.",
    )
    parser.add_argument("--config", default=str(ROOT / "configs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/crossval/final_combined_live_da3_geometry_heatmap_touch_presence"))
    parser.add_argument("--manifest", default=None, help="Override config data.manifest_path before fold creation.")
    parser.add_argument("--folds", default="5", help="Number of folds, or 'leave-one-participant-out'.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--exclude-participant-id", action="append", default=[], help="Additional participant id to exclude.")
    parser.add_argument("--exclude-video-id", action="append", default=[], help="Additional exact video_id/base_stem/stem to exclude.")
    parser.add_argument("--keep-default-problematic", action="store_true", help="Do not auto-exclude known problematic participants 307 and 321.")
    parser.add_argument("--thresholds", default="0.01:0.99:0.01", help="State threshold sweep spec passed to sweep_state_threshold.py.")
    parser.add_argument("--checkpoint-name", default="latest.pt", help="Fold checkpoint to sweep after training.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for child train/sweep commands.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write fold manifests and commands.")
    parser.add_argument("--skip-train", action="store_true", help="Do not run training subprocesses.")
    parser.add_argument("--skip-sweep", action="store_true", help="Do not run state threshold sweeps.")
    parser.add_argument("--override", nargs="*", default=[], help="Extra key=value overrides passed to each training run.")
    args = parser.parse_args()

    config_path = Path(args.config)
    base_cfg = load_yaml(config_path)
    source_manifest = Path(args.manifest or base_cfg["data"]["manifest_path"])
    if not source_manifest.is_absolute():
        source_manifest = ROOT / source_manifest
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(source_manifest)
    records_key, records = manifest_records(manifest)
    excluded_participants = set(args.exclude_participant_id)
    if not args.keep_default_problematic:
        excluded_participants |= DEFAULT_PROBLEM_PARTICIPANTS
    excluded_participants |= exclusion_participants_from_manifest(manifest)
    excluded_videos = set(args.exclude_video_id)

    filtered, excluded = filter_records(records, excluded_participants, excluded_videos)
    if not filtered:
        raise ValueError("No records remain after exclusions.")

    participants = sorted({participant_id(row) for row in filtered})
    folds = make_folds(filtered, parse_folds(args.folds, len(participants)), args.seed)
    commands: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_index, val_participants in enumerate(folds):
        fold_name = f"fold_{fold_index:02d}"
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_manifest = fold_dir / "manifest.json"
        fold_records = []
        val_set = set(val_participants)
        for row in filtered:
            row = dict(row)
            row["split"] = "val" if participant_id(row) in val_set else "train"
            row["cv_fold"] = fold_index
            fold_records.append(row)
        fold_manifest_payload = dict(manifest)
        fold_manifest_payload[records_key] = fold_records
        fold_manifest_payload["cross_validation"] = {
            "fold": fold_index,
            "folds": len(folds),
            "source_manifest": str(source_manifest),
            "seed": args.seed,
            "split_level": "participant",
            "train_participant_ids": sorted(set(participants) - val_set),
            "val_participant_ids": sorted(val_set),
            "excluded_participant_ids": sorted(excluded_participants),
            "excluded_video_ids": sorted(excluded_videos),
            "excluded_records": len(excluded),
        }
        write_json(fold_manifest, fold_manifest_payload)
        summary = summarize_fold(fold_records)
        summary.update({"fold": fold_index, "manifest": str(fold_manifest), "output_dir": str(fold_dir)})
        fold_summaries.append(summary)

        train_cmd = [
            args.python,
            str(ROOT / "scripts/train_student.py"),
            "--config",
            str(config_path),
            "--override",
            f"data.manifest_path={fold_manifest}",
            f"output_dir={fold_dir}",
            f"seed={args.seed + fold_index}",
            "training.copy_manifest=false",
            *args.override,
        ]
        sweep_csv = fold_dir / f"state_threshold_sweep_{Path(args.checkpoint_name).stem}_val.csv"
        sweep_cmd = [
            args.python,
            str(ROOT / "scripts/sweep_state_threshold.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(fold_dir / args.checkpoint_name),
            "--split",
            "val",
            "--thresholds",
            args.thresholds,
            "--output-csv",
            str(sweep_csv),
            "--override",
            f"data.manifest_path={fold_manifest}",
            f"output_dir={fold_dir}",
            *args.override,
        ]
        commands.append({"fold": fold_index, "train": train_cmd, "sweep": sweep_cmd})

    write_json(output_dir / "folds.json", {"folds": fold_summaries, "excluded": exclusion_report(excluded)})
    write_json(output_dir / "commands.json", commands)
    write_fold_summary_csv(output_dir / "folds.csv", fold_summaries)

    if args.prepare_only:
        print(f"prepared {len(folds)} folds under {output_dir}")
        print(f"excluded_records={len(excluded)} excluded_participants={sorted(excluded_participants)}")
        return

    for command in commands:
        fold = command["fold"]
        if not args.skip_train:
            print(f"running train fold={fold}")
            run(command["train"], ROOT)
        if not args.skip_sweep:
            print(f"running state threshold sweep fold={fold}")
            run(command["sweep"], ROOT)

    aggregate_results(output_dir, len(folds), args.checkpoint_name)
    print(f"cross-validation complete: {output_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def manifest_records(manifest: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(manifest, list):
        return "frames", [dict(row) for row in manifest]
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a list or object.")
    if isinstance(manifest.get("frames"), list):
        return "frames", [dict(row) for row in manifest["frames"]]
    if isinstance(manifest.get("items"), list):
        return "items", [dict(row) for row in manifest["items"]]
    raise ValueError("Manifest must contain a frames or items list.")


def exclusion_participants_from_manifest(manifest: Any) -> set[str]:
    if not isinstance(manifest, dict):
        return set()
    found: set[str] = set()
    exclusions = manifest.get("exclusions")
    if isinstance(exclusions, dict):
        stack = [exclusions]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                ids = current.get("participant_ids")
                if isinstance(ids, list):
                    found.update(str(value) for value in ids)
                stack.extend(value for value in current.values() if isinstance(value, (dict, list)))
            elif isinstance(current, list):
                stack.extend(value for value in current if isinstance(value, (dict, list)))
    return found


def filter_records(
    records: list[dict[str, Any]],
    excluded_participants: set[str],
    excluded_videos: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    excluded = []
    for row in records:
        if participant_id(row) in excluded_participants or video_id(row) in excluded_videos:
            excluded.append(row)
        else:
            kept.append(row)
    return kept, excluded


def participant_id(row: dict[str, Any]) -> str:
    value = row.get("participant_id")
    if value not in {None, ""}:
        return str(value)
    video = video_id(row)
    parts = video.split("_")
    if len(parts) >= 2 and parts[0] == "participant":
        return parts[1]
    raise ValueError(f"Could not determine participant_id for record: {row}")


def video_id(row: dict[str, Any]) -> str:
    for key in ("video_id", "base_stem", "stem"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    for key in ("video_path", "output_video_path"):
        value = row.get(key)
        if value not in {None, ""}:
            return Path(str(value)).stem
    return "UNKNOWN"


def parse_folds(spec: str, participant_count: int) -> int:
    normalized = str(spec).strip().lower()
    if normalized in {"loo", "loocv", "leave-one-out", "leave-one-participant-out"}:
        return participant_count
    folds = int(spec)
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    if folds > participant_count:
        raise ValueError(f"--folds={folds} exceeds participant_count={participant_count}")
    return folds


def make_folds(records: list[dict[str, Any]], fold_count: int, seed: int) -> list[list[str]]:
    counts = Counter(participant_id(row) for row in records)
    participants = list(counts)
    rng = random.Random(seed)
    rng.shuffle(participants)
    participants.sort(key=lambda pid: counts[pid], reverse=True)
    fold_parts: list[list[str]] = [[] for _ in range(fold_count)]
    fold_sizes = [0 for _ in range(fold_count)]
    for pid in participants:
        idx = min(range(fold_count), key=lambda i: (fold_sizes[i], len(fold_parts[i])))
        fold_parts[idx].append(pid)
        fold_sizes[idx] += counts[pid]
    return [sorted(parts) for parts in fold_parts]


def summarize_fold(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_records = Counter(str(row.get("split", "UNKNOWN")) for row in records)
    split_participants: dict[str, set[str]] = defaultdict(set)
    split_videos: dict[str, set[str]] = defaultdict(set)
    split_states: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        split = str(row.get("split", "UNKNOWN"))
        split_participants[split].add(participant_id(row))
        split_videos[split].add(video_id(row))
        split_states[split][str(row.get("state_hint", "UNKNOWN"))] += 1
    out: dict[str, Any] = {}
    for split in ("train", "val"):
        out[f"{split}_records"] = int(split_records[split])
        out[f"{split}_participants"] = len(split_participants[split])
        out[f"{split}_videos"] = len(split_videos[split])
        out[f"{split}_participant_ids"] = sorted(split_participants[split])
        out[f"{split}_state_counts"] = dict(sorted(split_states[split].items()))
    return out


def exclusion_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "by_participant": dict(sorted(Counter(participant_id(row) for row in records).items())),
        "by_video": dict(sorted(Counter(video_id(row) for row in records).items())),
    }


def write_fold_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "fold",
        "train_records",
        "train_participants",
        "train_videos",
        "val_records",
        "val_participants",
        "val_videos",
        "manifest",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def run(command: list[str], cwd: Path) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def aggregate_results(output_dir: Path, fold_count: int, checkpoint_name: str) -> None:
    rows = []
    threshold_name = Path(checkpoint_name).stem
    for fold in range(fold_count):
        fold_dir = output_dir / f"fold_{fold:02d}"
        metrics_path = fold_dir / "metrics_latest.json"
        sweep_path = fold_dir / f"state_threshold_sweep_{threshold_name}_val.csv"
        row: dict[str, Any] = {"fold": fold}
        if metrics_path.exists():
            metrics = load_json(metrics_path).get("frame", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    row[f"argmax_{key}"] = value
        if sweep_path.exists():
            sweep_rows = read_csv_numeric(sweep_path)
            if sweep_rows:
                best_f1 = max(sweep_rows, key=lambda item: (item["f1"], item["recall"]))
                best_acc = max(sweep_rows, key=lambda item: (item["accuracy"], item["recall"]))
                row.update(prefix_row(best_f1, "best_f1"))
                row.update(prefix_row(best_acc, "best_accuracy"))
                for floor in (0.95, 0.97, 0.99):
                    eligible = [item for item in sweep_rows if item["recall"] >= floor]
                    if eligible:
                        selected = max(eligible, key=lambda item: (item["precision"], item["f1"]))
                        row.update(prefix_row(selected, f"recall_ge_{str(floor).replace('.', '_')}"))
        rows.append(row)
    write_aggregate_csv(output_dir / "aggregate_metrics.csv", rows)
    write_json(output_dir / "aggregate_metrics.json", rows)


def read_csv_numeric(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                parsed[key] = float(value)
            rows.append(parsed)
        return rows


def prefix_row(row: dict[str, float], prefix: str) -> dict[str, float]:
    keys = ("threshold", "accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "tn", "fp", "fn", "tp")
    return {f"{prefix}_{key}": row[key] for key in keys if key in row}


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
