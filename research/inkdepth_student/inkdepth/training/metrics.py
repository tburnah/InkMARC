from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

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


def state_accuracy(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    if valid is None:
        valid = targets >= 0
    valid = valid.bool()
    if valid.sum() == 0:
        return {}
    pred = logits.detach().argmax(dim=1)[valid]
    truth = targets.detach()[valid].long()
    return {"state_acc": (pred == truth).float().mean().item()}


def xy_error_metrics(
    xy_pred: torch.Tensor,
    xy_target: torch.Tensor,
    valid: torch.Tensor,
    image_size: Iterable[int],
) -> dict[str, float]:
    valid = valid.bool()
    if valid.sum() == 0:
        return {}
    diff = torch.abs(xy_pred.detach()[valid] - xy_target.detach()[valid])
    l2 = torch.linalg.norm(diff, dim=1)
    h, w = list(image_size)
    pixel_scale = torch.tensor([w, h], device=diff.device, dtype=diff.dtype)
    pix_l2 = torch.linalg.norm(diff * pixel_scale, dim=1)
    return {
        "xy_l1": diff.mean().item(),
        "xy_l2": l2.mean().item(),
        "xy_pix": pix_l2.mean().item(),
    }
