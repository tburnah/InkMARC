from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        if n <= 0:
            return
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


@dataclass
class MetricTracker:
    meters: dict[str, AverageMeter] = field(default_factory=dict)

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            if value is None:
                continue
            self.meters.setdefault(key, AverageMeter()).update(float(value), n)

    def compute(self) -> dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}


def binary_touch_metrics(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    if valid is None:
        valid = torch.isfinite(targets)
    valid = valid.bool()
    if valid.sum() == 0:
        return {}
    probs = torch.sigmoid(logits.detach().flatten()[valid.flatten()])
    y = targets.detach().flatten()[valid.flatten()].float()
    pred = probs >= 0.5
    truth = y >= 0.5
    tp = (pred & truth).sum().item()
    tn = (~pred & ~truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "touch_acc": (tp + tn) / total,
        "touch_precision": precision,
        "touch_recall": recall,
        "touch_f1": f1,
    }


def state_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None = None,
    target_mode: str = "state",
) -> dict[str, float]:
    if valid is None:
        valid = targets >= 0
    valid = valid.bool()
    if valid.sum() == 0:
        return {}
    pred = logits.detach().argmax(dim=1)[valid]
    truth = targets.detach()[valid].long()
    if target_mode in {"presence", "interaction", "binary_presence", "background_interaction"}:
        truth = (truth > 0).long()
    return {"state_acc": (pred == truth).float().mean().item()}


def xy_error_metrics(
    xy_pred: torch.Tensor,
    xy_target: torch.Tensor,
    valid: torch.Tensor,
    scale: Iterable[float],
    scaled_metric_name: str = "xy_pix",
) -> dict[str, float]:
    valid = valid.bool()
    if valid.sum() == 0:
        return {}
    diff = torch.abs(xy_pred.detach()[valid] - xy_target.detach()[valid])
    l2 = torch.linalg.norm(diff, dim=1)
    scale_x, scale_y = list(scale)
    metric_scale = torch.tensor([scale_x, scale_y], device=diff.device, dtype=diff.dtype)
    scaled_l2 = torch.linalg.norm(diff * metric_scale, dim=1)
    return {
        "xy_l1": diff.mean().item(),
        "xy_l2": l2.mean().item(),
        scaled_metric_name: scaled_l2.mean().item(),
    }


def xy_metric_spec(cfg: Any) -> tuple[tuple[float, float], str]:
    label_cfg = cfg.data.labels
    mode = str(getattr(label_cfg, "coordinate_mode", "surface_to_image")).lower()
    if mode == "surface_pixels":
        surface_w = float(getattr(label_cfg, "surface_width", getattr(label_cfg, "tablet_width", 44800.0)))
        surface_h = float(getattr(label_cfg, "surface_height", getattr(label_cfg, "tablet_height", 29600.0)))
        return (surface_w, surface_h), "xy_surface"
    h, w = [float(x) for x in cfg.data.input_size]
    return (w, h), "xy_pix"


def xy_valid_mask(batch: dict[str, torch.Tensor], cfg: Any) -> torch.Tensor:
    valid = batch["valid_xy"].bool()
    target = str(getattr(getattr(cfg, "loss", {}), "xy_target", "valid_xy")).lower()
    if target in {"touch", "touch_only", "contact", "contact_only"}:
        has_touch = batch.get("has_touch", batch.get("valid_touch"))
        if has_touch is not None:
            valid = valid & has_touch.bool()
        valid = valid & (batch["touch"].float() > 0.5)
    elif target not in {"valid_xy", "all_valid_xy", "all"}:
        raise ValueError(f"Unknown loss.xy_target={target}")
    return valid
