#!/usr/bin/env python
"""Reaggregate saved presence CV sweeps at one prespecified threshold."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.46)
    parser.add_argument("--checkpoint-name", default="latest.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between zero and one")

    with (args.fold_root / "folds.csv").open(newline="") as handle:
        folds = list(csv.DictReader(handle))
    if len(folds) < 2:
        raise ValueError("At least two folds are required for sample standard deviation")
    rows = []
    metric_names = ("accuracy", "precision", "recall", "f1")
    for fold in folds:
        path = args.fold_root / f"fold_{int(fold['fold']):02d}" / f"state_threshold_sweep_{Path(args.checkpoint_name).stem}_val.csv"
        with path.open(newline="") as handle:
            matches = [row for row in csv.DictReader(handle)
                       if math.isclose(float(row["threshold"]), args.threshold, abs_tol=1e-10, rel_tol=0)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one threshold={args.threshold} row in {path}")
        saved = matches[0]
        counts = {key: int(saved[key]) for key in ("tp", "tn", "fp", "fn")}
        tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
        n = sum(counts.values())
        if n != int(fold["val_records"]):
            raise ValueError(f"Evaluated count does not match fold validation records: {path}")
        metrics = {
            "accuracy": (tp + tn) / max(1, n),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
        for key, value in metrics.items():
            if not math.isclose(value, float(saved[key]), abs_tol=1e-12):
                raise ValueError(f"Saved {key} disagrees with confusion counts: {path}")
        rows.append({"fold": int(fold["fold"]), "threshold": args.threshold,
                     "n_frames": n, **counts, **metrics, "source_csv": str(path.resolve())})

    summary = {
        "threshold": args.threshold,
        "positive_class": "interaction",
        "checkpoint_name": args.checkpoint_name,
        "method": "Recomputed from confusion counts in existing held-out fold threshold sweeps; no new inference or threshold selection.",
        "n_folds": len(rows),
        "n_evaluated_frames": sum(row["n_frames"] for row in rows),
        "aggregation": "Unweighted mean and sample standard deviation across folds (ddof=1).",
        "metrics": {key: {"mean": statistics.mean(row[key] for row in rows),
                          "std": statistics.stdev(row[key] for row in rows)} for key in metric_names},
        "folds": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"Presence CV at fixed threshold {args.threshold:g}",
             f"Evaluated frames: {summary['n_evaluated_frames']:,}", summary["aggregation"]]
    for key in metric_names:
        metric = summary["metrics"][key]
        lines.append(f"{key.title()}: {100 * metric['mean']:.1f} +/- {100 * metric['std']:.1f}%")
    report = "\n".join(lines) + "\n"
    (args.output_dir / "results.txt").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
