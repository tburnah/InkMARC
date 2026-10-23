from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from ml.inkloc.training.forward import model_forward


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _binary_metrics(score: torch.Tensor, target: torch.Tensor, threshold: float) -> Dict[str, float]:
    pred = score >= float(threshold)
    truth = target.bool()
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    tn = int((~pred & ~truth).sum().item())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _best_threshold(score: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    best = _binary_metrics(score, target, 0.5)
    for threshold in torch.linspace(0.01, 0.99, 99):
        metrics = _binary_metrics(score, target, float(threshold.item()))
        if metrics["f1"] > best["f1"]:
            best = metrics
    return best


@torch.no_grad()
def evaluate_calibration(
    model,
    loss_fn,
    loader,
    device: torch.device,
    pixel_size: Tuple[int, int] = (480, 320),
    max_batches: int = 0,
) -> Dict:
    model.eval()
    loss_sums: Dict[str, float] = {}
    seen = 0
    xy_chunks = []
    xy_target_chunks = []
    has_xy_chunks = []
    state_chunks = []
    pred_state_chunks = []
    valid_logit_chunks = []
    touch_score_chunks = []
    participant_chunks = []
    sequence_chunks = []

    for bi, batch in enumerate(loader, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model_forward(model, batch)
        losses = loss_fn(outputs, batch)
        bs = int(batch["image"].shape[0])
        seen += bs
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * bs

        xy_chunks.append(outputs["xy"].detach().cpu())
        xy_target_chunks.append(batch["xy"].detach().cpu())
        has_xy_chunks.append(batch["has_xy"].detach().cpu().bool())
        state_chunks.append(batch["state"].detach().cpu().long())
        pred_state_chunks.append(outputs["state_probs"].argmax(dim=1).detach().cpu().long())
        valid_logit_chunks.append(outputs["valid_logit"].detach().cpu())
        touch_score_chunks.append(outputs["state_probs"][:, 2].detach().cpu())
        if "participant_id" in batch:
            participant_chunks.append(batch["participant_id"].detach().cpu().long())
        if "sequence_id" in batch:
            sequence_chunks.append(batch["sequence_id"].detach().cpu().long())
        if max_batches > 0 and bi >= max_batches:
            break

    den = max(seen, 1)
    losses_out = {k: v / den for k, v in loss_sums.items()}
    xy = torch.cat(xy_chunks, dim=0)
    xy_target = torch.cat(xy_target_chunks, dim=0)
    has_xy = torch.cat(has_xy_chunks, dim=0)
    state = torch.cat(state_chunks, dim=0)
    pred_state = torch.cat(pred_state_chunks, dim=0)
    valid_score = torch.sigmoid(torch.cat(valid_logit_chunks, dim=0))
    touch_score = torch.cat(touch_score_chunks, dim=0)
    participant_id = torch.cat(participant_chunks, dim=0) if participant_chunks else None
    sequence_id = torch.cat(sequence_chunks, dim=0) if sequence_chunks else None
    touch_true = state == 2
    valid_true = state > 0

    confusion = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(state, pred_state):
        confusion[int(t), int(p)] += 1

    width, height = pixel_size
    xy_delta = xy - xy_target
    xy_norm_l2 = torch.linalg.vector_norm(xy_delta, dim=1)
    xy_px_l2 = torch.linalg.vector_norm(xy_delta * torch.tensor([width, height]), dim=1)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].mean().item()) if bool(mask.any()) else float("nan")

    fixed_touch = _binary_metrics(touch_score, touch_true, 0.5)
    best_touch = _best_threshold(touch_score, touch_true)
    valid_pred = valid_score >= 0.5

    by_state = {}
    for state_id, name in [(1, "hover"), (2, "touch")]:
        mask = state == state_id
        xy_mask = mask & has_xy
        by_state[name] = {
            "count": int(mask.sum().item()),
            "state_accuracy": masked_mean((pred_state == state).float(), mask),
            "xy_mean_l2": masked_mean(xy_norm_l2, xy_mask),
            "xy_mean_pixel_error": masked_mean(xy_px_l2, xy_mask),
            "mean_touch_probability": masked_mean(touch_score, mask),
        }

    def grouped_metrics(group_values: torch.Tensor | None) -> Dict[str, Dict[str, float]]:
        if group_values is None:
            return {}
        out = {}
        for value in torch.unique(group_values):
            mask = group_values == value
            if not bool(mask.any()):
                continue
            group_touch = _binary_metrics(touch_score[mask], touch_true[mask], float(best_touch["threshold"]))
            group_xy_mask = has_xy[mask]
            out[str(int(value.item()))] = {
                "count": int(mask.sum().item()),
                "state_accuracy": masked_mean((pred_state[mask] == state[mask]).float(), torch.ones_like(state[mask], dtype=torch.bool)),
                "valid_accuracy": masked_mean(((valid_score[mask] >= 0.5) == valid_true[mask]).float(), torch.ones_like(state[mask], dtype=torch.bool)),
                "xy_mean_pixel_error": masked_mean(xy_px_l2[mask], group_xy_mask),
                "touch_precision": group_touch["precision"],
                "touch_recall": group_touch["recall"],
                "touch_f1": group_touch["f1"],
            }
        return out

    error_modes = {
        "true_background_pred_hover": int(((state == 0) & (pred_state == 1)).sum().item()),
        "true_background_pred_touch": int(((state == 0) & (pred_state == 2)).sum().item()),
        "true_hover_pred_touch": int(((state == 1) & (pred_state == 2)).sum().item()),
        "true_touch_pred_hover": int(((state == 2) & (pred_state == 1)).sum().item()),
        "true_touch_pred_background": int(((state == 2) & (pred_state == 0)).sum().item()),
    }

    histograms = {}
    bins = torch.linspace(0.0, 1.0, 11)
    for state_id, name in [(0, "background"), (1, "hover"), (2, "touch")]:
        values = touch_score[state == state_id]
        histograms[name] = torch.histc(values, bins=10, min=0.0, max=1.0).long().tolist() if values.numel() else [0] * 10

    return {
        "num_samples": int(state.numel()),
        "losses": losses_out,
        "state_accuracy": float((pred_state == state).float().mean().item()),
        "valid_accuracy": float((valid_pred == valid_true).float().mean().item()),
        "xy_mean_l2": masked_mean(xy_norm_l2, has_xy),
        "xy_mean_pixel_error": masked_mean(xy_px_l2, has_xy),
        "touch_at_0_5": fixed_touch,
        "touch_best": best_touch,
        "confusion_matrix": confusion.tolist(),
        "class_counts": torch.bincount(state, minlength=3).tolist(),
        "predicted_class_counts": torch.bincount(pred_state, minlength=3).tolist(),
        "mean_touch_probability_true_touch": masked_mean(touch_score, touch_true),
        "mean_touch_probability_non_touch": masked_mean(touch_score, ~touch_true),
        "by_state": by_state,
        "error_modes": error_modes,
        "touch_probability_histogram_bins": bins.tolist(),
        "touch_probability_histogram_by_state": histograms,
        "by_participant": grouped_metrics(participant_id),
        "by_sequence": grouped_metrics(sequence_id),
    }


def write_calibration_report(report: Dict, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_thresholds(report: Dict, out_path: str | Path) -> None:
    payload = {
        "touch_probability": "state_probs[:, 2]",
        "touch_threshold": float(report["touch_best"]["threshold"]),
        "touch_precision": float(report["touch_best"]["precision"]),
        "touch_recall": float(report["touch_best"]["recall"]),
        "touch_f1": float(report["touch_best"]["f1"]),
        "source": "validation",
        "num_validation_samples": int(report["num_samples"]),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
