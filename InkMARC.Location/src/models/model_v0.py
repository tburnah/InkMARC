# src/models/model_v0.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelV0Config:
    """
    Phase 0 baseline:
      - Input: corners (4,2) -> flatten to 8
      - Output: 5 channels (logits for touch + 4 regression heads)
    """
    in_dim: int = 8
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.1

    # Loss weights
    w_touch: float = 1.0
    w_reg: float = 1.0

    # If True, only apply regression loss when touched==1 as well as mask==1.
    # This is often scientifically reasonable if x/y/pressure are only meaningful on-touch.
    regress_only_if_touched: bool = True


class InkMARCPhase0MLP(nn.Module):
    """
    Minimal MLP baseline for Phase 0.

    Input:
      corners: (B, 4, 2) float32
    Output:
      logits: (B, 5)
        logits[:,0] = touch logit (use sigmoid for probability)
        logits[:,1:] = regression outputs (raw)
    """

    def __init__(self, cfg: ModelV0Config):
        super().__init__()
        self.cfg = cfg

        layers = []
        dim = cfg.in_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(dim, cfg.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            dim = cfg.hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(dim, 5)

        # Optional: small init stability
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _flatten_corners(corners: torch.Tensor) -> torch.Tensor:
        # corners: (B,4,2) -> (B,8)
        return corners.reshape(corners.shape[0], -1)

    @staticmethod
    def _normalize_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Per-sample normalization (z-score) to reduce scale sensitivity across sessions.
        This is a reasonable baseline normalization for geometry-only inputs.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(eps)
        return (x - mean) / std

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        x = self._flatten_corners(corners)
        x = self._normalize_per_sample(x)
        x = self.backbone(x)
        out = self.head(x)  # (B,5)
        return out

    def compute_loss_and_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        logits: (B,5)
        labels: (B,5) float
        label_mask: (B,5) uint8/bool/int
        """
        cfg = self.cfg

        # Ensure dtype/shape
        labels = labels.float()
        mask = label_mask.to(dtype=torch.float32)

        # Touch is channel 0 (binary)
        touch_logits = logits[:, 0]
        touch_target = labels[:, 0].clamp(0.0, 1.0)

        # Apply touch mask if present (some datasets always mask touch=1, but we keep it general)
        touch_mask = mask[:, 0].clamp(0.0, 1.0)
        # BCE with logits per sample, then masked average
        bce_per = F.binary_cross_entropy_with_logits(touch_logits, touch_target, reduction="none")
        touch_den = touch_mask.sum().clamp_min(1.0)
        touch_loss = (bce_per * touch_mask).sum() / touch_den

        # Regression channels 1..4
        reg_pred = logits[:, 1:]
        reg_tgt = labels[:, 1:]
        reg_mask = mask[:, 1:]  # (B,4)

        if cfg.regress_only_if_touched:
            # Only count regression where the ground truth says touched == 1 (and mask==1)
            # We use touch_target as gating (you could also gate on predicted touch, but that's usually worse early).
            gate = touch_target.unsqueeze(1)  # (B,1)
            reg_mask = reg_mask * gate

        mse_per = (reg_pred - reg_tgt) ** 2  # (B,4)
        reg_den = reg_mask.sum().clamp_min(1.0)
        reg_loss = (mse_per * reg_mask).sum() / reg_den

        loss = cfg.w_touch * touch_loss + cfg.w_reg * reg_loss

        # Metrics (simple + stable)
        with torch.no_grad():
            touch_prob = torch.sigmoid(touch_logits)
            touch_pred = (touch_prob >= 0.5).float()

            # Masked accuracy for touch
            touch_acc = ((touch_pred == touch_target).float() * touch_mask).sum() / touch_den

            # Masked regression RMSE
            reg_rmse = torch.sqrt((mse_per * reg_mask).sum() / reg_den)

        metrics = {
            "loss": float(loss.item()),
            "touch_loss": float(touch_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "touch_acc": float(touch_acc.item()),
            "reg_rmse": float(reg_rmse.item()),
            "touch_den": float(touch_den.item()),
            "reg_den": float(reg_den.item()),
        }
        return loss, metrics
