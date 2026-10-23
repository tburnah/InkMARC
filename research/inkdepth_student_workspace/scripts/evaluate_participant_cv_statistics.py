#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_touch_location_by_participant import (  # noqa: E402
    apply_hysteresis,
    read_prediction_rows,
    run_model,
    write_csv,
)


LOCATION_FIELDS = [
    "participant_id",
    "fold",
    "n_contact_location_frames",
    "n_noncontact_location_frames",
    "contact_mean_euclidean_error_px",
    "noncontact_mean_euclidean_error_px",
    "contact_median_euclidean_error_px",
    "noncontact_median_euclidean_error_px",
    "contact_normalised_l1",
    "noncontact_normalised_l1",
    "contact_normalised_l2",
    "noncontact_normalised_l2",
    "mean_error_difference_px",
    "median_error_difference_px",
]

TOUCH_METHOD_FIELDS = [
    "participant_id",
    "fold",
    "method",
    "threshold_settings_used",
    "n_frames",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
]

TOUCH_PAIRED_FIELDS = [
    "participant_id",
    "fold",
    "raw_accuracy",
    "hysteresis_accuracy",
    "accuracy_difference",
    "raw_precision",
    "hysteresis_precision",
    "precision_difference",
    "raw_recall",
    "hysteresis_recall",
    "recall_difference",
    "raw_f1",
    "hysteresis_f1",
    "f1_difference",
    "raw_balanced_accuracy",
    "hysteresis_balanced_accuracy",
    "balanced_accuracy_difference",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute participant-level out-of-fold CV statistics for InkMARC-Sense touch/location predictions.",
    )
    parser.add_argument("--fold-root", required=True, help="Directory containing fold_XX subdirectories.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/cv_participant_statistics"))
    parser.add_argument("--prediction-relative-path", default="touch_location_frame_predictions.csv")
    parser.add_argument("--split", default="val", help="Held-out split to evaluate/generate for each fold.")
    parser.add_argument("--config", default=None, help="Required with --generate-missing-predictions.")
    parser.add_argument("--checkpoint-name", default="latest.pt")
    parser.add_argument("--generate-missing-predictions", action="store_true")
    parser.add_argument("--include-textured", action="store_true", help="Do not force data.variant_kind_filter=[original].")
    parser.add_argument("--raw-threshold", type=float, default=0.30)
    parser.add_argument("--hyst-on", type=float, default=0.15)
    parser.add_argument("--hyst-off", type=float, default=0.05)
    parser.add_argument("--hyst-min-on", type=int, default=1)
    parser.add_argument("--hyst-min-off", type=int, default=1)
    parser.add_argument("--max-episode-gap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    fold_root = Path(args.fold_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = load_or_generate_fold_predictions(fold_root, args)
    if not fold_rows:
        raise RuntimeError("No out-of-fold prediction rows were loaded.")

    participant_folds = assert_participants_in_one_fold(fold_rows)
    location_rows, location_excluded = build_location_rows(fold_rows)
    touch_rows, paired_touch_rows, touch_excluded = build_touch_rows(fold_rows, args)

    write_csv_with_fields(output_dir / "participant_location_contact_vs_noncontact.csv", location_rows, LOCATION_FIELDS)
    write_csv_with_fields(output_dir / "participant_touch_metrics_by_method.csv", touch_rows, TOUCH_METHOD_FIELDS)
    write_csv_with_fields(output_dir / "participant_touch_threshold_paired_comparison.csv", paired_touch_rows, TOUCH_PAIRED_FIELDS)

    location_summary = build_location_summary(location_rows, location_excluded, args)
    touch_summary = build_touch_summary(paired_touch_rows, touch_excluded, args)
    write_json(output_dir / "location_contact_vs_noncontact_summary.json", location_summary)
    write_json(output_dir / "touch_threshold_comparison_summary.json", touch_summary)
    write_markdown_summary(output_dir / "statistical_tests_summary.md", location_summary, touch_summary, participant_folds, args)

    if not args.skip_plots:
        plot_outputs(output_dir, location_rows, paired_touch_rows)

    print(f"wrote participant CV statistics to {output_dir}")
    print(f"participants={len(participant_folds)} location_test_n={location_summary['paired_test']['n']} touch_test_n={touch_summary['f1_paired_test']['n']}")
    for warning in location_excluded + touch_excluded:
        print(f"warning: {warning}")


def load_or_generate_fold_predictions(fold_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    fold_dirs = sorted(path for path in fold_root.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {fold_root}")

    all_rows: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        fold = parse_fold_id(fold_dir)
        pred_path = fold_dir / args.prediction_relative_path
        if not pred_path.exists():
            found = sorted(fold_dir.glob(f"**/{Path(args.prediction_relative_path).name}"))
            if found:
                pred_path = found[0]
        if pred_path.exists():
            rows = read_prediction_rows(pred_path)
        elif args.generate_missing_predictions:
            rows = generate_fold_predictions(fold_dir, args)
            pred_path = fold_dir / args.prediction_relative_path
            write_csv(pred_path, rows)
        else:
            raise FileNotFoundError(
                f"Missing predictions for {fold_dir}. Expected {fold_dir / args.prediction_relative_path}. "
                "Run this script with --generate-missing-predictions, or first create fold prediction CSVs with "
                "scripts/evaluate_touch_location_by_participant.py."
            )

        if not args.include_textured:
            before = len(rows)
            rows = [row for row in rows if is_original_prediction(row)]
            removed = before - len(rows)
            if removed:
                print(f"warning: fold_{fold:02d} filtered {removed} textured prediction rows")

        validate_fold_predictions_against_manifest(fold_dir, rows, fold, args.split)
        for row in rows:
            row["fold"] = fold
            row["source_prediction_csv"] = str(pred_path)
        all_rows.extend(rows)
    return all_rows


def generate_fold_predictions(fold_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config is None:
        raise ValueError("--config is required when --generate-missing-predictions is used.")
    checkpoint = fold_dir / args.checkpoint_name
    manifest = fold_dir / "manifest.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint}")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing fold manifest: {manifest}")

    from inkdepth.utils.config import load_config, parse_overrides

    cfg = load_config(
        args.config,
        parse_overrides(
            [
                f"data.manifest_path={manifest}",
                f"output_dir={fold_dir}",
                "training.copy_manifest=false",
            ]
        ),
    )
    if not args.include_textured:
        cfg.data.variant_kind_filter = ["original"]
    rows = run_model(cfg, str(checkpoint), args.split)
    rows.sort(key=lambda row: (str(row["participant_id"]), str(row["video_id"]), int(row["frame_index"])))
    return rows


def validate_fold_predictions_against_manifest(fold_dir: Path, rows: list[dict[str, Any]], fold: int, split: str) -> None:
    manifest_path = fold_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"warning: fold_{fold:02d} has no manifest.json; cannot verify held-out participant IDs")
        return
    manifest = load_json(manifest_path)
    cv = manifest.get("cross_validation") if isinstance(manifest, dict) else None
    if isinstance(cv, dict) and isinstance(cv.get("val_participant_ids"), list):
        expected = {str(value) for value in cv["val_participant_ids"]}
    else:
        records = manifest_records(manifest)
        expected = {participant_id(row) for row in records if str(row.get("split", split)) == split}
    observed = {str(row["participant_id"]) for row in rows}
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(
            f"fold_{fold:02d} prediction CSV contains participants not marked held out in manifest: {unexpected}"
        )


def assert_participants_in_one_fold(rows: list[dict[str, Any]]) -> dict[str, int]:
    participant_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        participant_folds[str(row["participant_id"])].add(int(row["fold"]))
    duplicates = {pid: sorted(folds) for pid, folds in participant_folds.items() if len(folds) > 1}
    if duplicates:
        raise ValueError(f"Participants appear in more than one held-out fold: {duplicates}")
    return {pid: next(iter(folds)) for pid, folds in participant_folds.items()}


def build_location_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped = group_by_participant(rows)
    out: list[dict[str, Any]] = []
    excluded: list[str] = []
    for (fold, pid), participant_rows in sorted(grouped.items()):
        contact = [row for row in participant_rows if is_valid_location(row) and row["touch_true"] == 1]
        noncontact = [row for row in participant_rows if is_valid_location(row) and row["touch_true"] == 0]
        contact_stats = location_stats(contact)
        noncontact_stats = location_stats(noncontact)
        if not contact or not noncontact:
            excluded.append(
                f"participant {pid} fold {fold} excluded from paired location test: "
                f"contact_frames={len(contact)} noncontact_frames={len(noncontact)}"
            )
        out.append(
            {
                "participant_id": pid,
                "fold": fold,
                "n_contact_location_frames": len(contact),
                "n_noncontact_location_frames": len(noncontact),
                "contact_mean_euclidean_error_px": contact_stats["mean_px"],
                "noncontact_mean_euclidean_error_px": noncontact_stats["mean_px"],
                "contact_median_euclidean_error_px": contact_stats["median_px"],
                "noncontact_median_euclidean_error_px": noncontact_stats["median_px"],
                "contact_normalised_l1": contact_stats["mean_l1"],
                "noncontact_normalised_l1": noncontact_stats["mean_l1"],
                "contact_normalised_l2": contact_stats["mean_l2"],
                "noncontact_normalised_l2": noncontact_stats["mean_l2"],
                "mean_error_difference_px": diff_or_blank(noncontact_stats["mean_px"], contact_stats["mean_px"]),
                "median_error_difference_px": diff_or_blank(noncontact_stats["median_px"], contact_stats["median_px"]),
            }
        )
    return out, excluded


def build_touch_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    grouped = group_by_participant(rows)
    by_method: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    excluded: list[str] = []
    for (fold, pid), participant_rows in sorted(grouped.items()):
        participant_rows = sorted(participant_rows, key=lambda row: (str(row["video_id"]), int(row["frame_index"])))
        y_rows = [row for row in participant_rows if row["has_touch"]]
        if not y_rows:
            excluded.append(f"participant {pid} fold {fold} excluded from touch comparison: no valid touch labels")
            continue
        y = [int(row["touch_true"]) for row in y_rows]
        raw_pred = [int(float(row["touch_prob"]) >= args.raw_threshold) for row in y_rows]
        hyst_all = apply_hysteresis(
            participant_rows,
            on_threshold=args.hyst_on,
            off_threshold=args.hyst_off,
            min_on_samples=args.hyst_min_on,
            min_off_samples=args.hyst_min_off,
        )
        hyst_pred = [pred for row, pred in zip(participant_rows, hyst_all) if row["has_touch"]]
        raw = classification_metrics(y, raw_pred)
        hyst = classification_metrics(y, hyst_pred)
        by_method.append(touch_method_row(pid, fold, "Raw Threshold", f"threshold={args.raw_threshold:.6g}", raw))
        by_method.append(
            touch_method_row(
                pid,
                fold,
                "Hysteresis",
                (
                    f"on={args.hyst_on:.6g}; off={args.hyst_off:.6g}; "
                    f"min_on={args.hyst_min_on}; min_off={args.hyst_min_off}"
                ),
                hyst,
            )
        )
        paired.append(
            {
                "participant_id": pid,
                "fold": fold,
                "raw_accuracy": raw["accuracy"],
                "hysteresis_accuracy": hyst["accuracy"],
                "accuracy_difference": hyst["accuracy"] - raw["accuracy"],
                "raw_precision": raw["precision"],
                "hysteresis_precision": hyst["precision"],
                "precision_difference": hyst["precision"] - raw["precision"],
                "raw_recall": raw["recall"],
                "hysteresis_recall": hyst["recall"],
                "recall_difference": hyst["recall"] - raw["recall"],
                "raw_f1": raw["f1"],
                "hysteresis_f1": hyst["f1"],
                "f1_difference": hyst["f1"] - raw["f1"],
                "raw_balanced_accuracy": raw["balanced_accuracy"],
                "hysteresis_balanced_accuracy": hyst["balanced_accuracy"],
                "balanced_accuracy_difference": hyst["balanced_accuracy"] - raw["balanced_accuracy"],
            }
        )
    return by_method, paired, excluded


def build_location_summary(rows: list[dict[str, Any]], excluded: list[str], args: argparse.Namespace) -> dict[str, Any]:
    paired_rows = [
        row
        for row in rows
        if to_float_or_none(row["mean_error_difference_px"]) is not None
        and int(row["n_contact_location_frames"]) > 0
        and int(row["n_noncontact_location_frames"]) > 0
    ]
    diffs = [float(row["mean_error_difference_px"]) for row in paired_rows]
    median_diffs = [float(row["median_error_difference_px"]) for row in paired_rows]
    return {
        "positive_direction": "positive means non-contact/hover frames had higher error than contact/touch frames",
        "participant_rows": len(rows),
        "paired_participants": len(paired_rows),
        "excluded": excluded,
        "contact_mean_error_px": describe([float(row["contact_mean_euclidean_error_px"]) for row in paired_rows]),
        "noncontact_mean_error_px": describe([float(row["noncontact_mean_euclidean_error_px"]) for row in paired_rows]),
        "mean_error_difference_px": describe_paired_difference(diffs, args),
        "median_error_difference_px": describe_paired_difference(median_diffs, args),
        "paired_test": paired_test(diffs, args, preferred_name="noncontact - contact mean Euclidean error"),
    }


def build_touch_summary(rows: list[dict[str, Any]], excluded: list[str], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "positive_direction": "positive means hysteresis performed better than raw threshold",
        "paired_participants": len(rows),
        "excluded": excluded,
        "raw_f1": describe([float(row["raw_f1"]) for row in rows]),
        "hysteresis_f1": describe([float(row["hysteresis_f1"]) for row in rows]),
        "f1_difference": describe_paired_difference([float(row["f1_difference"]) for row in rows], args),
        "accuracy_difference": describe_paired_difference([float(row["accuracy_difference"]) for row in rows], args),
        "precision_difference": describe_paired_difference([float(row["precision_difference"]) for row in rows], args),
        "recall_difference": describe_paired_difference([float(row["recall_difference"]) for row in rows], args),
        "balanced_accuracy_difference": describe_paired_difference([float(row["balanced_accuracy_difference"]) for row in rows], args),
        "f1_paired_test": paired_test([float(row["f1_difference"]) for row in rows], args, preferred_name="hysteresis - raw F1"),
    }


def paired_test(diffs: list[float], args: argparse.Namespace, preferred_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "comparison": preferred_name,
        "n": len(diffs),
        "mean_difference": mean(diffs),
        "sd_difference": sample_sd(diffs),
        "cohens_dz": cohens_dz(diffs),
        "bootstrap_mean_ci_95": bootstrap_mean_ci(diffs, args.bootstrap_iterations, args.seed),
        "normality": {"test": "Shapiro-Wilk", "available": False, "w": None, "p": None, "alpha": 0.05},
        "test": {"name": None, "statistic": None, "p": None, "effect_size": None},
    }
    if len(diffs) < 2:
        out["test"]["note"] = "not enough paired participants for inferential test"
        return out
    scipy_stats = import_scipy_stats()
    if scipy_stats is None:
        out["test"]["note"] = "scipy is not available; bootstrap CI is reported but Shapiro/t-test/Wilcoxon were not run"
        return out

    normal = False
    if len(diffs) >= 3:
        try:
            shapiro = scipy_stats.shapiro(diffs)
            out["normality"] = {
                "test": "Shapiro-Wilk",
                "available": True,
                "w": finite_or_none(shapiro.statistic),
                "p": finite_or_none(shapiro.pvalue),
                "alpha": 0.05,
            }
            normal = bool(math.isfinite(float(shapiro.pvalue)) and shapiro.pvalue >= 0.05)
        except Exception as exc:  # pragma: no cover - defensive for scipy edge cases
            out["normality"]["note"] = str(exc)

    if normal:
        result = scipy_stats.ttest_1samp(diffs, popmean=0.0)
        out["test"] = {
            "name": "paired t-test on participant-level differences",
            "statistic": finite_or_none(result.statistic),
            "p": finite_or_none(result.pvalue),
            "effect_size": {"name": "Cohen's dz", "value": cohens_dz(diffs)},
        }
    else:
        try:
            result = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            out["test"] = {
                "name": "Wilcoxon signed-rank test on participant-level differences",
                "statistic": finite_or_none(result.statistic),
                "p": finite_or_none(result.pvalue),
                "effect_size": {"name": "rank-biserial correlation", "value": rank_biserial(diffs)},
            }
        except ValueError as exc:
            out["test"] = {
                "name": "Wilcoxon signed-rank test on participant-level differences",
                "statistic": None,
                "p": None,
                "effect_size": {"name": "rank-biserial correlation", "value": rank_biserial(diffs)},
                "note": str(exc),
            }
    return out


def classification_metrics(y: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 1)
    tn = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 0)
    fp = sum(1 for truth, guess in zip(y, pred) if truth == 0 and guess == 1)
    fn = sum(1 for truth, guess in zip(y, pred) if truth == 1 and guess == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "n_frames": len(y),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def touch_method_row(pid: str, fold: int, method: str, settings: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "participant_id": pid,
        "fold": fold,
        "method": method,
        "threshold_settings_used": settings,
        "n_frames": metrics["n_frames"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_negatives": metrics["false_negatives"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }


def location_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"mean_px": "", "median_px": "", "mean_l1": "", "mean_l2": ""}
    return {
        "mean_px": mean([float(row["xy_err_pix"]) for row in rows]),
        "median_px": median([float(row["xy_err_pix"]) for row in rows]),
        "mean_l1": mean([float(row["xy_l1_norm"]) for row in rows]),
        "mean_l2": mean([float(row["xy_l2_norm"]) for row in rows]),
    }


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sd": sample_sd(values),
        "median": median(values),
        "q1": percentile(values, 25),
        "q3": percentile(values, 75),
        "iqr": percentile(values, 75) - percentile(values, 25) if values else None,
    }


def describe_paired_difference(values: list[float], args: argparse.Namespace) -> dict[str, Any]:
    out = describe(values)
    out["bootstrap_mean_ci_95"] = bootstrap_mean_ci(values, args.bootstrap_iterations, args.seed)
    return out


def bootstrap_mean_ci(values: list[float], iterations: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"low": None, "high": None, "iterations": iterations}
    rng = random.Random(seed)
    n = len(values)
    sampled_means = []
    for _ in range(max(0, iterations)):
        sampled_means.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    return {
        "low": percentile(sampled_means, 2.5),
        "high": percentile(sampled_means, 97.5),
        "iterations": iterations,
    }


def cohens_dz(values: list[float]) -> float | None:
    sd = sample_sd(values)
    if sd in {None, 0.0}:
        return None
    return mean(values) / sd


def rank_biserial(values: list[float]) -> float | None:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return None
    abs_values = [abs(value) for value in nonzero]
    ranks = average_ranks(abs_values)
    positive = sum(rank for value, rank in zip(nonzero, ranks) if value > 0)
    negative = sum(rank for value, rank in zip(nonzero, ranks) if value < 0)
    total = positive + negative
    if total == 0:
        return None
    return (positive - negative) / total


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def plot_outputs(output_dir: Path, location_rows: list[dict[str, Any]], touch_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable; plots skipped: {exc}")
        return

    loc = [
        row
        for row in location_rows
        if to_float_or_none(row["mean_error_difference_px"]) is not None
        and int(row["n_contact_location_frames"]) > 0
        and int(row["n_noncontact_location_frames"]) > 0
    ]
    loc_diffs = [float(row["mean_error_difference_px"]) for row in loc]
    save_qq_plot(plt, loc_diffs, output_dir / "location_mean_error_difference_qq.png", "Location mean-error differences")
    save_paired_plot(
        plt,
        [(str(row["participant_id"]), float(row["contact_mean_euclidean_error_px"]), float(row["noncontact_mean_euclidean_error_px"])) for row in loc],
        output_dir / "location_contact_vs_noncontact_paired.png",
        "Contact",
        "Non-contact",
        "Mean Euclidean error (px)",
    )
    save_difference_plot(
        plt,
        [(str(row["participant_id"]), float(row["mean_error_difference_px"])) for row in loc],
        output_dir / "location_mean_error_difference.png",
        "Non-contact - contact mean error (px)",
    )

    f1_diffs = [float(row["f1_difference"]) for row in touch_rows]
    save_qq_plot(plt, f1_diffs, output_dir / "touch_f1_difference_qq.png", "Touch F1 differences")
    save_paired_plot(
        plt,
        [(str(row["participant_id"]), float(row["raw_f1"]), float(row["hysteresis_f1"])) for row in touch_rows],
        output_dir / "touch_raw_vs_hysteresis_f1_paired.png",
        "Raw",
        "Hysteresis",
        "F1",
    )
    save_difference_plot(
        plt,
        [(str(row["participant_id"]), float(row["f1_difference"])) for row in touch_rows],
        output_dir / "touch_f1_difference.png",
        "Hysteresis - raw F1",
    )


def save_qq_plot(plt: Any, values: list[float], path: Path, title: str) -> None:
    if not values:
        return
    ordered = sorted(values)
    n = len(ordered)
    mu = mean(ordered)
    sd = sample_sd(ordered) or 1.0
    normal = NormalDist()
    theoretical = [mu + sd * normal.inv_cdf((i - 0.5) / n) for i in range(1, n + 1)]
    plt.figure(figsize=(5, 4))
    plt.scatter(theoretical, ordered)
    lo = min(theoretical + ordered)
    hi = max(theoretical + ordered)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Observed quantiles")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_paired_plot(plt: Any, values: list[tuple[str, float, float]], path: Path, left: str, right: str, ylabel: str) -> None:
    if not values:
        return
    plt.figure(figsize=(5, 4))
    for _, a, b in values:
        plt.plot([0, 1], [a, b], marker="o", color="#4c78a8", alpha=0.75)
    plt.xticks([0, 1], [left, right])
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_difference_plot(plt: Any, values: list[tuple[str, float]], path: Path, ylabel: str) -> None:
    if not values:
        return
    plt.figure(figsize=(max(5, len(values) * 0.35), 4))
    xs = list(range(len(values)))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.scatter(xs, [value for _, value in values], color="#f58518")
    plt.xticks(xs, [pid for pid, _ in values], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_markdown_summary(
    path: Path,
    location: dict[str, Any],
    touch: dict[str, Any],
    participant_folds: dict[str, int],
    args: argparse.Namespace,
) -> None:
    loc_test = location["paired_test"]
    touch_test = touch["f1_paired_test"]
    loc_ci = loc_test["bootstrap_mean_ci_95"]
    touch_ci = touch_test["bootstrap_mean_ci_95"]
    text = f"""# Participant-Level CV Statistical Tests

All inferential tests in this report use participant-level out-of-fold metrics. Individual frames are not treated as independent observations because frames are temporally correlated. Mann-Whitney is not used because both comparisons are paired; Wilcoxon signed-rank is the non-parametric paired alternative.

## Inputs

- Fold root: `{args.fold_root}`
- Participants: {len(participant_folds)}
- Raw threshold: {args.raw_threshold}
- Hysteresis: on={args.hyst_on}, off={args.hyst_off}, min_on={args.hyst_min_on}, min_off={args.hyst_min_off}
- Positive location difference: non-contact minus contact error, so positive means contact frames had lower error.
- Positive touch difference: hysteresis minus raw threshold, so positive means hysteresis performed better.

## Location: Contact vs Non-Contact

Across {location['paired_participants']} participants, the participant-level mean contact error was {fmt(location['contact_mean_error_px']['mean'])} px (SD {fmt(location['contact_mean_error_px']['sd'])}), and the participant-level mean non-contact error was {fmt(location['noncontact_mean_error_px']['mean'])} px (SD {fmt(location['noncontact_mean_error_px']['sd'])}). The mean paired difference was {fmt(loc_test['mean_difference'])} px, with bootstrap 95% CI [{fmt(loc_ci['low'])}, {fmt(loc_ci['high'])}].

Normality of participant-level paired differences was assessed using Shapiro-Wilk (W={fmt(loc_test['normality'].get('w'))}, p={fmt(loc_test['normality'].get('p'))}). The selected paired test was {loc_test['test'].get('name') or 'not run'} (statistic={fmt(loc_test['test'].get('statistic'))}, p={fmt(loc_test['test'].get('p'))}, effect size={format_effect(loc_test['test'].get('effect_size'))}).

## Touch Classification: Raw Threshold vs Hysteresis

Across {touch['paired_participants']} participants, participant-level raw-threshold F1 was {fmt(touch['raw_f1']['mean'])} (SD {fmt(touch['raw_f1']['sd'])}), and participant-level hysteresis F1 was {fmt(touch['hysteresis_f1']['mean'])} (SD {fmt(touch['hysteresis_f1']['sd'])}). The mean paired F1 difference was {fmt(touch_test['mean_difference'])}, with bootstrap 95% CI [{fmt(touch_ci['low'])}, {fmt(touch_ci['high'])}].

Normality of participant-level paired F1 differences was assessed using Shapiro-Wilk (W={fmt(touch_test['normality'].get('w'))}, p={fmt(touch_test['normality'].get('p'))}). The selected paired test was {touch_test['test'].get('name') or 'not run'} (statistic={fmt(touch_test['test'].get('statistic'))}, p={fmt(touch_test['test'].get('p'))}, effect size={format_effect(touch_test['test'].get('effect_size'))}).

## Exclusions

Location exclusions:
{bullet_list(location['excluded'])}

Touch exclusions:
{bullet_list(touch['excluded'])}
"""
    path.write_text(text, encoding="utf-8")


def group_by_participant(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["fold"]), str(row["participant_id"]))].append(row)
    return grouped


def is_valid_location(row: dict[str, Any]) -> bool:
    return bool(row["has_touch"]) and row["touch_true"] in {0, 1} and bool(row["valid_xy"]) and row.get("xy_err_pix") not in {"", None}


def is_original_prediction(row: dict[str, Any]) -> bool:
    variant = row.get("variant_kind")
    if variant not in {None, ""}:
        return str(variant) == "original"
    return "__tex_" not in str(row.get("video_id", ""))


def manifest_records(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        return [dict(row) for row in manifest]
    if isinstance(manifest, dict):
        if isinstance(manifest.get("frames"), list):
            return [dict(row) for row in manifest["frames"]]
        if isinstance(manifest.get("items"), list):
            return [dict(row) for row in manifest["items"]]
    return []


def participant_id(row: dict[str, Any]) -> str:
    value = row.get("participant_id")
    if value not in {None, ""}:
        return str(value)
    for key in ("video_id", "base_stem", "stem"):
        value = row.get(key)
        if value not in {None, ""}:
            parts = str(value).split("_")
            if len(parts) >= 2 and parts[0] == "participant":
                return parts[1]
    return "UNKNOWN"


def parse_fold_id(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse fold index from {path}") from exc


def import_scipy_stats() -> Any | None:
    try:
        from scipy import stats

        return stats
    except Exception:
        return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def write_csv_with_fields(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def diff_or_blank(a: Any, b: Any) -> float | str:
    if a == "" or b == "":
        return ""
    return float(a) - float(b)


def to_float_or_none(value: Any) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = mean(values)
    assert mu is not None
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return percentile(values, 50)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def format_effect(effect: Any) -> str:
    if not isinstance(effect, dict):
        return "NA"
    return f"{effect.get('name')}={fmt(effect.get('value'))}"


def bullet_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


if __name__ == "__main__":
    main()
