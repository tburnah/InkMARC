from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class LocTouchLoss(nn.Module):
    def __init__(
        self,
        valid_weight: float = 1.0,
        touch_weight: float = 1.0,
        state_weight: float = 0.5,
        xy_weight: float = 5.0,
        heatmap_weight: float = 0.0,
        pressure_weight: float = 0.0,
        transition_weight: float = 0.0,
        touch_pos_weight: float = 0.0,
        touch_focal_gamma: float = 0.0,
        hover_false_positive_weight: float = 0.0,
        state_class_weights: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.weights = {
            "valid": valid_weight,
            "touch": touch_weight,
            "state": state_weight,
            "xy": xy_weight,
            "heatmap": heatmap_weight,
            "pressure": pressure_weight,
            "transition": transition_weight,
        }
        self.touch_pos_weight = float(touch_pos_weight)
        self.touch_focal_gamma = float(touch_focal_gamma)
        self.hover_false_positive_weight = float(hover_false_positive_weight)
        self.state_class_weights = state_class_weights

    @classmethod
    def from_config(cls, cfg: Dict) -> "LocTouchLoss":
        weights = cfg.get("loss_weights", cfg.get("training", {}).get("loss_weights", {}))
        return cls(
            valid_weight=float(weights.get("valid", 1.0)),
            touch_weight=float(weights.get("touch", 1.0)),
            state_weight=float(weights.get("state", 0.5)),
            xy_weight=float(weights.get("xy", 5.0)),
            heatmap_weight=float(weights.get("heatmap", 0.0)),
            pressure_weight=float(weights.get("pressure", 0.0)),
            transition_weight=float(weights.get("transition", 0.0)),
            touch_pos_weight=float(weights.get("touch_pos_weight", 0.0)),
            touch_focal_gamma=float(weights.get("touch_focal_gamma", 0.0)),
            hover_false_positive_weight=float(weights.get("hover_false_positive_weight", 0.0)),
            state_class_weights=tuple(float(x) for x in weights["state_class_weights"]) if "state_class_weights" in weights else None,
        )

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = batch["valid"].float()
        touch = batch["touch"].float()
        state = batch["state"].long()
        has_xy = batch["has_xy"].float()
        xy_target = batch["xy"].float()

        valid_loss = F.binary_cross_entropy_with_logits(outputs["valid_logit"], valid)
        pos_weight = None
        if self.touch_pos_weight > 0.0:
            pos_weight = outputs["touch_logit"].new_tensor(self.touch_pos_weight)
        touch_raw = F.binary_cross_entropy_with_logits(outputs["touch_logit"], touch, reduction="none", pos_weight=pos_weight)
        if self.touch_focal_gamma > 0.0:
            prob = torch.sigmoid(outputs["touch_logit"])
            pt = torch.where(touch > 0.5, prob, 1.0 - prob)
            touch_raw = touch_raw * (1.0 - pt).clamp_min(1e-6).pow(self.touch_focal_gamma)
        touch_mask = valid.clamp(0.0, 1.0)
        if self.hover_false_positive_weight > 0.0:
            hover_mask = (state == 1).float()
            touch_mask = touch_mask + hover_mask * self.hover_false_positive_weight
        touch_loss = (touch_raw * touch_mask).sum() / touch_mask.sum().clamp_min(1.0)
        state_weights = None
        if self.state_class_weights is not None:
            state_weights = outputs["state_logits"].new_tensor(self.state_class_weights)
        state_loss = F.cross_entropy(outputs["state_logits"], state, weight=state_weights)

        xy_raw = F.smooth_l1_loss(outputs["xy"], xy_target, reduction="none").sum(dim=1)
        xy_loss = (xy_raw * has_xy).sum() / has_xy.sum().clamp_min(1.0)

        heatmap_loss = outputs["xy"].new_tensor(0.0)
        if self.weights["heatmap"] > 0.0 and "heatmap_target" in batch:
            target = batch["heatmap_target"].float()
            raw = F.mse_loss(outputs["heatmap_prob"], target, reduction="none").mean(dim=(1, 2, 3))
            if "heatmap_mask" in batch:
                mask = batch["heatmap_mask"].float().reshape(-1)
            else:
                mask = has_xy
            heatmap_loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)

        pressure_loss = outputs["xy"].new_tensor(0.0)
        if self.weights["pressure"] > 0.0 and "pressure" in batch and "pressure_pred" in outputs:
            pressure = batch["pressure"].float()
            mask = touch.clamp(0.0, 1.0)
            raw = F.smooth_l1_loss(outputs["pressure_pred"], pressure, reduction="none")
            pressure_loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)

        transition_loss = outputs["xy"].new_tensor(0.0)
        if self.weights["transition"] > 0.0 and "transition" in batch and "transition_logits" in outputs:
            transition_loss = F.cross_entropy(outputs["transition_logits"], batch["transition"].long())

        total = (
            self.weights["valid"] * valid_loss
            + self.weights["touch"] * touch_loss
            + self.weights["state"] * state_loss
            + self.weights["xy"] * xy_loss
            + self.weights["heatmap"] * heatmap_loss
            + self.weights["pressure"] * pressure_loss
            + self.weights["transition"] * transition_loss
        )
        return {
            "loss": total,
            "valid_loss": valid_loss.detach(),
            "touch_loss": touch_loss.detach(),
            "state_loss": state_loss.detach(),
            "xy_loss": xy_loss.detach(),
            "heatmap_loss": heatmap_loss.detach(),
            "pressure_loss": pressure_loss.detach(),
            "transition_loss": transition_loss.detach(),
        }
