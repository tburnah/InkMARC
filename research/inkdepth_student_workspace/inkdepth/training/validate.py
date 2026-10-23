from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from inkdepth.models.losses import MultiTaskLoss, gradient_matching_loss, scale_and_shift_invariant_loss
from inkdepth.training.metrics import MetricTracker, binary_touch_metrics, state_accuracy, xy_error_metrics, xy_metric_spec, xy_valid_mask
from inkdepth.utils.visualization import save_debug_panel


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: Any,
    device: torch.device,
    epoch: int = 0,
    save_visuals: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    model.eval()
    criterion = MultiTaskLoss(cfg)
    tracker = MetricTracker()
    image_size = tuple(int(x) for x in cfg.data.input_size)
    xy_scale, xy_metric_name = xy_metric_spec(cfg)
    max_visuals = int(cfg.training.max_visuals)
    saved = 0
    rows: list[dict[str, Any]] = []
    visual_dir = Path(output_dir or cfg.output_dir) / "visuals" / f"epoch_{epoch:04d}"

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _to_device(batch, device)
        outputs = model(batch["image"])
        loss, logs = criterion(outputs, batch)
        n = batch["image"].shape[0]
        tracker.update(logs, n=n)
        tracker.update({"val_loss": loss.detach().item()}, n=n)

        depth = outputs.get("depth")
        if depth is not None:
            has_depth = batch.get("has_depth")
            if has_depth is None or bool(has_depth.any()):
                tracker.update(
                    {
                        "depth_l1": _masked_depth_l1(depth, batch["depth"], has_depth).item(),
                        "depth_ssi": scale_and_shift_invariant_loss(depth, batch["depth"], has_depth).item(),
                        "depth_grad": gradient_matching_loss(depth, batch["depth"], has_depth).item(),
                    },
                    n=n,
                )
        if outputs.get("touch_logits") is not None:
            tracker.update(binary_touch_metrics(outputs["touch_logits"], batch["touch"], batch["has_touch"]), n=n)
        if outputs.get("state_logits") is not None:
            tracker.update(
                state_accuracy(
                    outputs["state_logits"],
                    batch["state"],
                    batch["has_state"],
                    _state_target_mode(cfg),
                ),
                n=n,
            )
        if outputs.get("xy") is not None:
            valid_xy = xy_valid_mask(batch, cfg)
            tracker.update(
                xy_error_metrics(outputs["xy"], batch["xy"], valid_xy, xy_scale, xy_metric_name),
                n=int(valid_xy.sum().detach().cpu()),
            )
        _append_rows(rows, outputs, batch, image_size, xy_scale, xy_metric_name, xy_valid_mask(batch, cfg))

        if save_visuals and saved < max_visuals and depth is not None:
            bsz = batch["image"].shape[0]
            for i in range(min(bsz, max_visuals - saved)):
                heatmap = outputs.get("location_heatmap")
                xy = outputs.get("xy")
                save_debug_panel(
                    visual_dir / f"sample_{saved:03d}.jpg",
                    batch["image"][i],
                    batch["depth"][i],
                    depth[i],
                    None if heatmap is None else heatmap[i],
                    None if xy is None else xy[i],
                )
                saved += 1
    metrics = tracker.compute()
    if output_dir is not None:
        metrics = _numeric_metrics(_save_grouped_metrics(rows, metrics, Path(output_dir), epoch))
    elif rows:
        metrics = _numeric_metrics(_frame_metrics_from_rows(rows, metrics))
    return metrics


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _masked_depth_l1(pred: torch.Tensor, target: torch.Tensor, has_depth: torch.Tensor | None) -> torch.Tensor:
    mask = torch.isfinite(target)
    if has_depth is not None:
        mask = mask & has_depth[:, None, None, None].bool()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return torch.abs(pred - target)[mask].mean()


def _append_rows(
    rows: list[dict[str, Any]],
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, Any],
    image_size: tuple[int, int],
    xy_scale: tuple[float, float],
    xy_metric_name: str,
    xy_valid: torch.Tensor,
) -> None:
    bsz = batch["image"].shape[0]
    touch_logits = outputs.get("touch_logits")
    state_logits = outputs.get("state_logits")
    xy_pred = outputs.get("xy")
    h, w = image_size
    for i in range(bsz):
        row: dict[str, Any] = {
            "video_id": batch["video_id"][i],
            "participant_id": batch["participant_id"][i],
            "task_id": batch["task_id"][i],
            "frame_index": int(batch["frame_index"][i].detach().cpu()),
            "has_depth": bool(batch["has_depth"][i].detach().cpu()),
            "has_labels": bool(batch["has_labels"][i].detach().cpu()),
            "has_touch": bool(batch["has_touch"][i].detach().cpu()),
            "has_state": bool(batch["has_state"][i].detach().cpu()),
            "valid_xy": bool(xy_valid[i].detach().cpu()),
            "touch_true": None,
            "touch_pred": None,
            "touch_prob": None,
            "state_true": None,
            "state_pred": None,
            "xy_err_norm": None,
            "xy_err_pix": None,
            "xy_err_surface": None,
        }
        if row["has_touch"] and touch_logits is not None:
            true = int(float(batch["touch"][i].detach().cpu()) > 0.5)
            prob = float(torch.sigmoid(touch_logits[i]).detach().cpu())
            pred = int(prob >= 0.5)
            row["touch_true"] = true
            row["touch_pred"] = pred
            row["touch_prob"] = prob
        if row["has_state"] and state_logits is not None:
            true_state = int(batch["state"][i].detach().cpu())
            if state_logits.shape[1] == 2 and true_state >= 0:
                true_state = int(true_state > 0)
            row["state_true"] = true_state
            row["state_pred"] = int(state_logits[i].detach().argmax().cpu())
        if row["valid_xy"] and xy_pred is not None:
            diff = (xy_pred[i].detach().cpu() - batch["xy"][i].detach().cpu()).numpy()
            row["xy_err_norm"] = float(np.linalg.norm(diff))
            scaled_err = float(np.linalg.norm(diff * np.asarray(xy_scale, dtype=np.float32)))
            if xy_metric_name == "xy_surface":
                row["xy_err_surface"] = scaled_err
            else:
                row["xy_err_pix"] = scaled_err
        rows.append(row)


def _binary_metrics_from_df(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    true_col = f"{prefix}_true"
    pred_col = f"{prefix}_pred"
    prob_col = f"{prefix}_prob"
    valid = df[true_col].notna() & df[pred_col].notna()
    if not valid.any():
        return {}
    y = df.loc[valid, true_col].astype(int).to_numpy()
    p = df.loc[valid, pred_col].astype(int).to_numpy()
    metrics = _binary_metrics_from_arrays(y, p, prefix)
    if prob_col in df:
        prob_valid = df[true_col].notna() & df[prob_col].notna()
        if prob_valid.any():
            y_prob = df.loc[prob_valid, true_col].astype(int).to_numpy()
            probs = df.loc[prob_valid, prob_col].astype(float).to_numpy()
            metrics.update(_threshold_sweep_metrics(y_prob, probs, prefix))
    return metrics


def _binary_metrics_from_arrays(y: np.ndarray, p: np.ndarray, prefix: str) -> dict[str, Any]:
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        f"{prefix}_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        f"{prefix}_count": int(len(y)),
    }


def _threshold_sweep_metrics(y: np.ndarray, probs: np.ndarray, prefix: str) -> dict[str, Any]:
    best_f1: dict[str, Any] | None = None
    best_acc: dict[str, Any] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        p = (probs >= threshold).astype(int)
        metrics = _binary_metrics_from_arrays(y, p, prefix)
        metrics[f"{prefix}_threshold"] = float(threshold)
        if best_f1 is None or metrics[f"{prefix}_f1"] > best_f1[f"{prefix}_f1"]:
            best_f1 = metrics
        if best_acc is None or metrics[f"{prefix}_accuracy"] > best_acc[f"{prefix}_accuracy"]:
            best_acc = metrics
    if best_f1 is None or best_acc is None:
        return {}
    return {
        f"{prefix}_best_f1": best_f1[f"{prefix}_f1"],
        f"{prefix}_best_f1_threshold": best_f1[f"{prefix}_threshold"],
        f"{prefix}_best_f1_precision": best_f1[f"{prefix}_precision"],
        f"{prefix}_best_f1_recall": best_f1[f"{prefix}_recall"],
        f"{prefix}_best_accuracy": best_acc[f"{prefix}_accuracy"],
        f"{prefix}_best_accuracy_threshold": best_acc[f"{prefix}_threshold"],
        f"{prefix}_best_accuracy_precision": best_acc[f"{prefix}_precision"],
        f"{prefix}_best_accuracy_recall": best_acc[f"{prefix}_recall"],
    }


def _state_metrics_from_df(df: pd.DataFrame) -> dict[str, Any]:
    valid = df["state_true"].notna() & df["state_pred"].notna()
    if not valid.any():
        return {}
    y = df.loc[valid, "state_true"].astype(int).to_numpy()
    p = df.loc[valid, "state_pred"].astype(int).to_numpy()
    num_classes = 2 if max(int(np.max(y)), int(np.max(p))) <= 1 else 3
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y, p):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true, pred] += 1
    precisions = []
    recalls = []
    f1s = []
    class_names = ["background", "interaction"] if num_classes == 2 else ["background", "hover", "touch"]
    class_metrics: dict[str, float] = {}
    for cls in range(num_classes):
        tp = matrix[cls, cls]
        fp = matrix[:, cls].sum() - tp
        fn = matrix[cls, :].sum() - tp
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        f1 = float(2 * precision * recall / max(1e-8, precision + recall))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        name = class_names[cls]
        class_metrics[f"state_precision_{name}"] = precision
        class_metrics[f"state_recall_{name}"] = recall
        class_metrics[f"state_f1_{name}"] = f1
    return {
        "state_accuracy": float((y == p).mean()),
        "state_precision_macro": float(np.mean(precisions)),
        "state_recall_macro": float(np.mean(recalls)),
        "state_f1_macro": float(np.mean(f1s)),
        "state_confusion": matrix.tolist(),
        "state_count": int(valid.sum()),
        **class_metrics,
    }


def _state_target_mode(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "loss", {}), "state_target", "state")).lower()


def _xy_metrics_from_df(df: pd.DataFrame) -> dict[str, Any]:
    valid = df["xy_err_norm"].notna()
    if not valid.any():
        return {}
    norm = df.loc[valid, "xy_err_norm"].astype(float).to_numpy()
    out = {
        "xy_mean_norm": float(np.mean(norm)),
        "xy_median_norm": float(np.median(norm)),
        "xy_count": int(valid.sum()),
    }
    pix_valid = valid & df["xy_err_pix"].notna()
    if pix_valid.any():
        pix = df.loc[pix_valid, "xy_err_pix"].astype(float).to_numpy()
        out.update({"xy_mean_pix": float(np.mean(pix)), "xy_median_pix": float(np.median(pix))})
    surface_valid = valid & df["xy_err_surface"].notna()
    if surface_valid.any():
        surface = df.loc[surface_valid, "xy_err_surface"].astype(float).to_numpy()
        out.update({"xy_mean_surface": float(np.mean(surface)), "xy_median_surface": float(np.median(surface))})
    return out


def _group_summary(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_key, dropna=False):
        metrics = {
            group_key: key,
            "frames": int(len(group)),
            "has_labels": int(group["has_labels"].sum()),
            "has_touch": int(group["has_touch"].sum()),
            "has_state": int(group["has_state"].sum()),
            "valid_xy": int(group["valid_xy"].sum()),
        }
        metrics.update(_binary_metrics_from_df(group, "touch"))
        metrics.update(_state_metrics_from_df(group))
        metrics.update(_xy_metrics_from_df(group))
        rows.append(metrics)
    return pd.DataFrame(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _frame_metrics_from_rows(rows: list[dict[str, Any]], metrics: dict[str, float]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return dict(metrics)
    frame_metrics: dict[str, Any] = dict(metrics)
    frame_metrics.update(_binary_metrics_from_df(df, "touch"))
    frame_metrics.update(_state_metrics_from_df(df))
    frame_metrics.update(_xy_metrics_from_df(df))
    return frame_metrics


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            numeric[key] = float(value)
    return numeric


def _save_grouped_metrics(rows: list[dict[str, Any]], metrics: dict[str, float], output_dir: Path, epoch: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return dict(metrics)
    frame_metrics = _frame_metrics_from_rows(rows, metrics)
    by_video = _group_summary(df, "video_id")
    by_participant = _group_summary(df, "participant_id")
    by_task = _group_summary(df, "task_id")
    by_video.to_csv(output_dir / "metrics_by_video.csv", index=False)
    by_participant.to_csv(output_dir / "metrics_by_participant.csv", index=False)
    by_task.to_csv(output_dir / "metrics_by_task.csv", index=False)
    payload = {
        "epoch": epoch,
        "frame": frame_metrics,
        "by_video": by_video.to_dict("records"),
        "by_participant": by_participant.to_dict("records"),
        "by_task": by_task.to_dict("records"),
    }
    text = json.dumps(_jsonable(payload), indent=2, allow_nan=False)
    (output_dir / f"metrics_epoch_{epoch}.json").write_text(text, encoding="utf-8")
    (output_dir / "metrics_latest.json").write_text(text, encoding="utf-8")
    return frame_metrics
