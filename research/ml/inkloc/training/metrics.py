from __future__ import annotations

from typing import Dict, Tuple

import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1.0)


@torch.no_grad()
def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], pixel_size: Tuple[int, int] = (480, 320)) -> Dict[str, float]:
    valid_t = batch["valid"].bool()
    touch_t = batch["touch"].bool()
    state_t = batch["state"].long()
    valid_p = torch.sigmoid(outputs["valid_logit"]) >= 0.5
    touch_p = outputs["state_probs"][:, 2] >= 0.5
    state_p = outputs["state_probs"].argmax(dim=1)

    tp = (touch_p & touch_t).sum().float()
    fp = (touch_p & ~touch_t).sum().float()
    fn = (~touch_p & touch_t).sum().float()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    touch_score = outputs["state_probs"][:, 2].float()
    best_threshold = touch_score.new_tensor(0.5)
    best_precision = precision
    best_recall = recall
    best_f1 = f1
    for threshold in torch.linspace(0.05, 0.95, 91, device=touch_score.device):
        pred = touch_score >= threshold
        b_tp = (pred & touch_t).sum().float()
        b_fp = (pred & ~touch_t).sum().float()
        b_fn = (~pred & touch_t).sum().float()
        b_precision = _safe_div(b_tp, b_tp + b_fp)
        b_recall = _safe_div(b_tp, b_tp + b_fn)
        b_f1 = _safe_div(2 * b_precision * b_recall, b_precision + b_recall)
        if b_f1 > best_f1:
            best_threshold = threshold
            best_precision = b_precision
            best_recall = b_recall
            best_f1 = b_f1

    has_xy = batch["has_xy"].bool()
    xy_err = torch.linalg.vector_norm(outputs["xy"] - batch["xy"].float(), dim=1)
    width, height = pixel_size
    xy_pix = torch.linalg.vector_norm((outputs["xy"] - batch["xy"].float()) * outputs["xy"].new_tensor([width, height]), dim=1)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values[mask].mean() if mask.any() else values.new_tensor(float("nan"))

    return {
        "valid_accuracy": (valid_p == valid_t).float().mean().item(),
        "touch_accuracy": (touch_p == touch_t).float().mean().item(),
        "touch_precision": precision.item(),
        "touch_recall": recall.item(),
        "touch_f1": f1.item(),
        "touch_best_threshold": best_threshold.item(),
        "touch_best_precision": best_precision.item(),
        "touch_best_recall": best_recall.item(),
        "touch_best_f1": best_f1.item(),
        "state_accuracy": (state_p == state_t).float().mean().item(),
        "xy_mean_l2": masked_mean(xy_err, has_xy).item(),
        "xy_mean_pixel_error": masked_mean(xy_pix, has_xy).item(),
        "xy_hover_l2": masked_mean(xy_err, state_t == 1).item(),
        "xy_touch_l2": masked_mean(xy_err, state_t == 2).item(),
    }
