#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.training.validate import _binary_metrics_from_arrays


KEYS = ["video_id", "frame_index"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse depth and RGB binary touch predictions.")
    parser.add_argument("--depth-val", required=True)
    parser.add_argument("--rgb-val", required=True)
    parser.add_argument("--depth-test", required=True)
    parser.add_argument("--rgb-test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", default="1,3,5,9,15,21")
    parser.add_argument("--thresholds", default="0.05:0.95:0.01")
    parser.add_argument("--weights", default="0.00:1.00:0.02")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = parse_int_list(args.windows)
    thresholds = parse_range(args.thresholds)
    weights = parse_range(args.weights)

    val = add_features(load_pair(args.depth_val, args.rgb_val), windows)
    test = add_features(load_pair(args.depth_test, args.rgb_test), windows)

    val_valid = valid_mask(val)
    test_valid = valid_mask(test)
    if not val_valid.any() or not test_valid.any():
        raise RuntimeError("Fusion needs valid touch labels in both val and test prediction CSVs.")

    feature_cols = make_feature_cols(windows)
    val_metrics: list[dict[str, Any]] = []
    test_metrics: list[dict[str, Any]] = []

    for column in probability_columns(windows):
        add_threshold_sweep(val_metrics, val, val[column].to_numpy(float), column, thresholds, "val")
        add_threshold_sweep(test_metrics, test, test[column].to_numpy(float), column, thresholds, "test")

    for window in windows:
        d_col = f"depth_mean_w{window}"
        r_col = f"rgb_mean_w{window}"
        for weight in weights:
            val_probs = weight * val[d_col].to_numpy(float) + (1.0 - weight) * val[r_col].to_numpy(float)
            test_probs = weight * test[d_col].to_numpy(float) + (1.0 - weight) * test[r_col].to_numpy(float)
            name = f"blend_w{window}_depth{weight:.2f}"
            add_threshold_sweep(val_metrics, val, val_probs, name, thresholds, "val")
            add_threshold_sweep(test_metrics, test, test_probs, name, thresholds, "test")

    for window in windows:
        d_col = f"depth_mean_w{window}"
        r_col = f"rgb_mean_w{window}"
        depth = val[d_col].to_numpy(float)
        rgb = val[r_col].to_numpy(float)
        test_depth = test[d_col].to_numpy(float)
        test_rgb = test[r_col].to_numpy(float)
        for margin in np.arange(0.02, 0.31, 0.02):
            val_probs = np.where(np.abs(depth - 0.5) <= margin, 0.5 * depth + 0.5 * rgb, depth)
            test_probs = np.where(np.abs(test_depth - 0.5) <= margin, 0.5 * test_depth + 0.5 * test_rgb, test_depth)
            name = f"depth_uncertain_blend_w{window}_m{margin:.2f}"
            add_threshold_sweep(val_metrics, val, val_probs, name, thresholds, "val")
            add_threshold_sweep(test_metrics, test, test_probs, name, thresholds, "test")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0),
    )
    model.fit(val.loc[val_valid, feature_cols].to_numpy(float), val.loc[val_valid, "touch_true"].astype(int).to_numpy())
    val_lr = model.predict_proba(val[feature_cols].to_numpy(float))[:, 1]
    test_lr = model.predict_proba(test[feature_cols].to_numpy(float))[:, 1]
    val["logistic_fusion"] = val_lr
    test["logistic_fusion"] = test_lr
    add_threshold_sweep(val_metrics, val, val_lr, "logistic_fusion", thresholds, "val")
    add_threshold_sweep(test_metrics, test, test_lr, "logistic_fusion", thresholds, "test")

    val_df = pd.DataFrame(val_metrics)
    test_df = pd.DataFrame(test_metrics)
    selected = select_on_val_apply_to_test(val_df, test_df)
    test_oracle = best_rows(test_df, ["touch_accuracy", "touch_f1"]).assign(selection="test_sweep_oracle")
    val_best = best_rows(val_df, ["touch_accuracy", "touch_f1"]).assign(selection="val_best")

    summary = pd.concat([selected, val_best, test_oracle], ignore_index=True)
    summary = summary.sort_values(["selection", "touch_accuracy", "touch_f1"], ascending=[True, False, False])

    val.to_csv(output_dir / "joined_val_predictions.csv", index=False)
    test.to_csv(output_dir / "joined_test_predictions.csv", index=False)
    val_df.to_csv(output_dir / "fusion_val_sweep.csv", index=False)
    test_df.to_csv(output_dir / "fusion_test_sweep.csv", index=False)
    summary.to_csv(output_dir / "fusion_summary.csv", index=False)
    participant_breakdown(test, selected, output_dir)

    payload = {
        "val_rows": int(len(val)),
        "val_valid_touch": int(val_valid.sum()),
        "test_rows": int(len(test)),
        "test_valid_touch": int(test_valid.sum()),
        "feature_cols": feature_cols,
        "selected": selected.to_dict("records"),
        "test_oracle": test_oracle.to_dict("records"),
    }
    (output_dir / "fusion_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"saved {output_dir / 'fusion_summary.csv'}")
    print(f"saved {output_dir / 'fusion_summary.json'}")
    print("selected on val, evaluated on test:")
    print(selected[["method", "threshold", "touch_accuracy", "touch_f1", "touch_precision", "touch_recall", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print("\nbest test sweep rows, diagnostic only:")
    print(test_oracle[["method", "threshold", "touch_accuracy", "touch_f1", "touch_precision", "touch_recall", "tn", "fp", "fn", "tp"]].to_string(index=False))


def load_pair(depth_path: str, rgb_path: str) -> pd.DataFrame:
    depth = pd.read_csv(depth_path)
    rgb = pd.read_csv(rgb_path)
    keep_depth = [
        "video_id",
        "participant_id",
        "task_id",
        "frame_index",
        "has_touch",
        "touch_true",
        "touch_prob",
        "valid_xy",
        "pressure",
        "sequence_id",
        "derived_state",
        "transition_distance",
        "h5_pressure",
    ]
    keep_depth = [c for c in keep_depth if c in depth.columns]
    depth = depth[keep_depth].rename(columns={"touch_prob": "depth_prob"})
    rgb = rgb[KEYS + ["touch_prob"]].rename(columns={"touch_prob": "rgb_prob"})
    out = depth.merge(rgb, on=KEYS, how="inner", validate="one_to_one")
    if len(out) != len(depth) or len(out) != len(rgb):
        print(f"warning: merged {len(out)} rows from depth={len(depth)} rgb={len(rgb)}")
    out = out.sort_values(["sequence_id", "frame_index", "video_id"]).reset_index(drop=True)
    return out


def add_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["depth_logit"] = logit(out["depth_prob"].to_numpy(float))
    out["rgb_logit"] = logit(out["rgb_prob"].to_numpy(float))
    out["prob_diff"] = out["depth_prob"].to_numpy(float) - out["rgb_prob"].to_numpy(float)
    out["prob_mean"] = 0.5 * (out["depth_prob"].to_numpy(float) + out["rgb_prob"].to_numpy(float))
    out["prob_product"] = out["depth_prob"].to_numpy(float) * out["rgb_prob"].to_numpy(float)
    for window in windows:
        out[f"depth_mean_w{window}"] = grouped_centered_rolling_mean(out, "depth_prob", window)
        out[f"rgb_mean_w{window}"] = grouped_centered_rolling_mean(out, "rgb_prob", window)
        out[f"blend_equal_w{window}"] = 0.5 * (out[f"depth_mean_w{window}"] + out[f"rgb_mean_w{window}"])
        out[f"depth_logit_w{window}"] = logit(out[f"depth_mean_w{window}"].to_numpy(float))
        out[f"rgb_logit_w{window}"] = logit(out[f"rgb_mean_w{window}"].to_numpy(float))
        out[f"diff_w{window}"] = out[f"depth_mean_w{window}"] - out[f"rgb_mean_w{window}"]
    return out


def probability_columns(windows: list[int]) -> list[str]:
    cols = ["depth_prob", "rgb_prob", "prob_mean"]
    for window in windows:
        cols.extend([f"depth_mean_w{window}", f"rgb_mean_w{window}", f"blend_equal_w{window}"])
    return list(dict.fromkeys(cols))


def make_feature_cols(windows: list[int]) -> list[str]:
    cols = ["depth_prob", "rgb_prob", "depth_logit", "rgb_logit", "prob_diff", "prob_mean", "prob_product"]
    for window in windows:
        cols.extend(
            [
                f"depth_mean_w{window}",
                f"rgb_mean_w{window}",
                f"blend_equal_w{window}",
                f"depth_logit_w{window}",
                f"rgb_logit_w{window}",
                f"diff_w{window}",
            ]
        )
    return list(dict.fromkeys(cols))


def add_threshold_sweep(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    probs: np.ndarray,
    method: str,
    thresholds: np.ndarray,
    split: str,
) -> None:
    for threshold in thresholds:
        row = evaluate_probs(df, probs, method, float(threshold))
        row["split"] = split
        row["threshold"] = float(threshold)
        rows.append(row)


def evaluate_probs(df: pd.DataFrame, probs: np.ndarray, method: str, threshold: float) -> dict[str, Any]:
    valid = valid_mask(df)
    y = df.loc[valid, "touch_true"].astype(int).to_numpy()
    pred = (probs[valid] >= threshold).astype(int)
    metrics = _binary_metrics_from_arrays(y, pred, "touch")
    return {
        "method": method,
        "touch_count": int(metrics["touch_count"]),
        "touch_accuracy": float(metrics["touch_accuracy"]),
        "touch_precision": float(metrics["touch_precision"]),
        "touch_recall": float(metrics["touch_recall"]),
        "touch_f1": float(metrics["touch_f1"]),
        "tn": int(metrics["touch_confusion"]["tn"]),
        "fp": int(metrics["touch_confusion"]["fp"]),
        "fn": int(metrics["touch_confusion"]["fn"]),
        "tp": int(metrics["touch_confusion"]["tp"]),
    }


def select_on_val_apply_to_test(val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective in ["accuracy", "f1"]:
        sort_cols = ["touch_accuracy", "touch_f1"] if objective == "accuracy" else ["touch_f1", "touch_accuracy"]
        best = val_df.sort_values(sort_cols, ascending=[False, False]).iloc[0]
        mask = (test_df["method"] == best["method"]) & np.isclose(test_df["threshold"], best["threshold"])
        test_row = test_df.loc[mask].iloc[0].copy()
        test_row["selection"] = f"selected_by_val_{objective}"
        test_row["val_touch_accuracy"] = float(best["touch_accuracy"])
        test_row["val_touch_f1"] = float(best["touch_f1"])
        rows.append(test_row)
    return pd.DataFrame(rows)


def best_rows(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    first = df.sort_values(sort_cols, ascending=[False, False]).iloc[0].copy()
    return pd.DataFrame([first])


def participant_breakdown(test: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for selected_row in selected.itertuples(index=False):
        probs = method_probs(test, str(selected_row.method))
        threshold = float(selected_row.threshold)
        for participant_id, group in test.groupby("participant_id", sort=True):
            valid = valid_mask(group)
            if not valid.any():
                continue
            idx = group.index.to_numpy()
            y = group.loc[valid, "touch_true"].astype(int).to_numpy()
            pred = (probs[idx][valid.to_numpy()] >= threshold).astype(int)
            metrics = _binary_metrics_from_arrays(y, pred, "touch")
            rows.append(
                {
                    "selection": str(selected_row.selection),
                    "method": str(selected_row.method),
                    "threshold": threshold,
                    "participant_id": participant_id,
                    "touch_count": int(metrics["touch_count"]),
                    "touch_accuracy": float(metrics["touch_accuracy"]),
                    "touch_precision": float(metrics["touch_precision"]),
                    "touch_recall": float(metrics["touch_recall"]),
                    "touch_f1": float(metrics["touch_f1"]),
                    "tn": int(metrics["touch_confusion"]["tn"]),
                    "fp": int(metrics["touch_confusion"]["fp"]),
                    "fn": int(metrics["touch_confusion"]["fn"]),
                    "tp": int(metrics["touch_confusion"]["tp"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "fusion_test_by_participant.csv", index=False)


def method_probs(df: pd.DataFrame, method: str) -> np.ndarray:
    if method in df.columns:
        return df[method].to_numpy(float)
    if method.startswith("blend_w"):
        left, weight_s = method.rsplit("_depth", 1)
        window = int(left.removeprefix("blend_w"))
        weight = float(weight_s)
        return weight * df[f"depth_mean_w{window}"].to_numpy(float) + (1.0 - weight) * df[f"rgb_mean_w{window}"].to_numpy(float)
    if method.startswith("depth_uncertain_blend_w"):
        left, margin_s = method.rsplit("_m", 1)
        window = int(left.removeprefix("depth_uncertain_blend_w"))
        margin = float(margin_s)
        depth = df[f"depth_mean_w{window}"].to_numpy(float)
        rgb = df[f"rgb_mean_w{window}"].to_numpy(float)
        return np.where(np.abs(depth - 0.5) <= margin, 0.5 * depth + 0.5 * rgb, depth)
    raise KeyError(f"Cannot reconstruct probabilities for method {method}")


def grouped_centered_rolling_mean(df: pd.DataFrame, column: str, window: int) -> np.ndarray:
    values = np.empty(len(df), dtype=np.float64)
    for _, idx in df.groupby("sequence_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        series = pd.Series(df.loc[idx_arr, column].to_numpy(float))
        values[idx_arr] = series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return values


def valid_mask(df: pd.DataFrame) -> pd.Series:
    return df["has_touch"].astype(bool) & df["touch_true"].notna()


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_range(value: str) -> np.ndarray:
    parts = [float(v) for v in value.split(":")]
    if len(parts) == 1:
        return np.asarray(parts, dtype=np.float64)
    if len(parts) != 3:
        raise ValueError(f"Range must be start:stop:step, got {value!r}")
    start, stop, step = parts
    return np.round(np.arange(start, stop + 0.5 * step, step), 10)


if __name__ == "__main__":
    main()
