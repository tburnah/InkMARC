from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _valid_depth_mask(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target) & (target > 0)


def scale_and_shift_invariant_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_sample: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = _valid_depth_mask(target)
    if valid_sample is not None:
        mask = mask & valid_sample[:, None, None, None].bool()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    p = pred[:, 0]
    t = target[:, 0]
    m = mask[:, 0].float()
    a00 = (m * p * p).sum(dim=(1, 2))
    a01 = (m * p).sum(dim=(1, 2))
    a11 = m.sum(dim=(1, 2))
    b0 = (m * p * t).sum(dim=(1, 2))
    b1 = (m * t).sum(dim=(1, 2))
    det = a00 * a11 - a01 * a01
    valid = det.abs() > 1e-6
    scale = torch.ones_like(det)
    shift = torch.zeros_like(det)
    scale[valid] = (a11[valid] * b0[valid] - a01[valid] * b1[valid]) / det[valid]
    shift[valid] = (-a01[valid] * b0[valid] + a00[valid] * b1[valid]) / det[valid]
    aligned = scale[:, None, None] * p + shift[:, None, None]
    err = torch.abs(aligned - t) * m
    return err.sum() / m.sum().clamp_min(1.0)


def gradient_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_sample: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = _valid_depth_mask(target).float()
    if valid_sample is not None:
        mask = mask * valid_sample[:, None, None, None].float()
    pred = pred * mask
    target = target * mask
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    mx = mask[..., :, 1:] * mask[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    my = mask[..., 1:, :] * mask[..., :-1, :]
    loss_x = (torch.abs(dx_p - dx_t) * mx).sum() / mx.sum().clamp_min(1.0)
    loss_y = (torch.abs(dy_p - dy_t) * my).sum() / my.sum().clamp_min(1.0)
    return loss_x + loss_y


def masked_depth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_sample: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = _valid_depth_mask(target)
    if valid_sample is not None:
        mask = mask & valid_sample[:, None, None, None].bool()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return torch.abs(pred - target)[mask].mean()


def xy_weighted_depth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    xy: torch.Tensor,
    valid_xy: torch.Tensor,
    valid_sample: torch.Tensor | None = None,
    sigma_pixels: float = 12.0,
    radius_pixels: float = 36.0,
) -> torch.Tensor:
    mask = _valid_depth_mask(target)
    sample_valid = valid_xy.bool()
    if valid_sample is not None:
        sample_valid = sample_valid & valid_sample.bool()
    mask = mask & sample_valid[:, None, None, None]
    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    b, _, h, w = target.shape
    xs = torch.arange(w, device=target.device, dtype=target.dtype).view(1, 1, 1, w)
    ys = torch.arange(h, device=target.device, dtype=target.dtype).view(1, 1, h, 1)
    x0 = xy[:, 0].to(dtype=target.dtype).view(b, 1, 1, 1) * max(1, w - 1)
    y0 = xy[:, 1].to(dtype=target.dtype).view(b, 1, 1, 1) * max(1, h - 1)
    dist2 = (xs - x0).square() + (ys - y0).square()
    sigma = max(float(sigma_pixels), 1e-6)
    weights = torch.exp(-dist2 / (2.0 * sigma * sigma))
    if radius_pixels and radius_pixels > 0:
        weights = weights * (dist2 <= float(radius_pixels) ** 2).to(weights.dtype)
    weights = weights * mask.to(weights.dtype)

    weighted_error = torch.abs(pred - target) * weights
    denom = weights.flatten(1).sum(dim=1)
    per_sample = weighted_error.flatten(1).sum(dim=1) / denom.clamp_min(1.0)
    valid = denom > 0
    if not valid.any():
        return pred.new_tensor(0.0)
    return per_sample[valid].mean()


def make_gaussian_heatmap(xy: torch.Tensor, valid: torch.Tensor, size: tuple[int, int], sigma: float = 2.0) -> torch.Tensor:
    b = xy.shape[0]
    h, w = size
    ys = torch.linspace(0.0, 1.0, h, device=xy.device, dtype=xy.dtype)
    xs = torch.linspace(0.0, 1.0, w, device=xy.device, dtype=xy.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist2 = (xx[None] - xy[:, 0, None, None]) ** 2 + (yy[None] - xy[:, 1, None, None]) ** 2
    sigma_norm = sigma / max(h, w)
    heatmap = torch.exp(-dist2 / (2 * sigma_norm**2)).unsqueeze(1)
    return heatmap * valid[:, None, None, None].float()


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        state_label_smoothing = float(getattr(cfg.loss, "state_label_smoothing", 0.0))
        self.ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=state_label_smoothing)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def weights(self) -> dict[str, float]:
        w = {
            "depth_ssi": float(self.cfg.loss.lambda_depth_ssi),
            "depth_l1": float(getattr(self.cfg.loss, "lambda_depth_l1", 0.0)),
            "depth_roi_l1": float(getattr(self.cfg.loss, "lambda_depth_roi_l1", 0.0)),
            "depth_grad": float(self.cfg.loss.lambda_depth_grad),
            "touch": float(self.cfg.loss.lambda_touch),
            "state": float(self.cfg.loss.lambda_state),
            "xy": float(self.cfg.loss.lambda_xy),
            "heatmap": float(self.cfg.loss.lambda_heatmap),
        }
        phase = str(self.cfg.training.phase)
        if phase == "depth_pretrain":
            w.update({"touch": 0.0, "state": 0.0, "xy": 0.0, "heatmap": 0.0})
        elif phase == "task_finetune":
            w["depth_ssi"] *= 0.1
            w["depth_grad"] *= 0.1
        return w

    def forward(self, outputs: dict[str, torch.Tensor | None], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        weights = self.weights()
        total = batch["image"].new_tensor(0.0)
        logs: dict[str, float] = {}

        depth_pred = outputs.get("depth")
        has_depth = batch.get("has_depth")
        if depth_pred is not None and weights["depth_ssi"] > 0:
            l_depth = scale_and_shift_invariant_loss(depth_pred, batch["depth"], has_depth)
            total = total + weights["depth_ssi"] * l_depth
            logs["loss_depth_ssi"] = l_depth.detach().item()
        if depth_pred is not None and weights["depth_l1"] > 0:
            l_l1 = masked_depth_l1_loss(depth_pred, batch["depth"], has_depth)
            total = total + weights["depth_l1"] * l_l1
            logs["loss_depth_l1"] = l_l1.detach().item()
        if depth_pred is not None and weights["depth_roi_l1"] > 0:
            l_roi = xy_weighted_depth_l1_loss(
                depth_pred,
                batch["depth"],
                batch["xy"],
                batch["valid_xy"],
                has_depth,
                sigma_pixels=float(getattr(self.cfg.loss, "depth_roi_sigma_pixels", 12.0)),
                radius_pixels=float(getattr(self.cfg.loss, "depth_roi_radius_pixels", 36.0)),
            )
            total = total + weights["depth_roi_l1"] * l_roi
            logs["loss_depth_roi_l1"] = l_roi.detach().item()
        if depth_pred is not None and weights["depth_grad"] > 0:
            l_grad = gradient_matching_loss(depth_pred, batch["depth"], has_depth)
            total = total + weights["depth_grad"] * l_grad
            logs["loss_depth_grad"] = l_grad.detach().item()

        touch_logits = outputs.get("touch_logits")
        if touch_logits is not None and weights["touch"] > 0:
            valid = batch.get("has_touch", batch["valid_touch"]).bool()
            if valid.sum() > 0:
                l_touch = self.bce(touch_logits[valid], batch["touch"][valid].float()).mean()
                total = total + weights["touch"] * l_touch
                logs["loss_touch"] = l_touch.detach().item()

        state_logits = outputs.get("state_logits")
        if state_logits is not None and weights["state"] > 0:
            valid = batch.get("has_state", batch["valid_state"]).bool()
            if valid.sum() > 0:
                l_state = self.ce(state_logits[valid], batch["state"][valid].long()).mean()
                total = total + weights["state"] * l_state
                logs["loss_state"] = l_state.detach().item()

        xy = outputs.get("xy")
        valid_xy = batch["valid_xy"].bool()
        if xy is not None and weights["xy"] > 0 and valid_xy.sum() > 0:
            err = torch.abs(xy[valid_xy] - batch["xy"][valid_xy])
            if str(self.cfg.loss.xy_loss) == "smooth_l1":
                l_xy = self.smooth_l1(xy[valid_xy], batch["xy"][valid_xy]).mean()
            else:
                l_xy = err.mean()
            total = total + weights["xy"] * l_xy
            logs["loss_xy"] = l_xy.detach().item()

        heatmap = outputs.get("location_heatmap")
        if heatmap is not None and weights["heatmap"] > 0 and valid_xy.sum() > 0:
            target = make_gaussian_heatmap(batch["xy"], valid_xy, heatmap.shape[-2:])
            if str(self.cfg.loss.heatmap_loss) == "bce":
                l_heatmap = F.binary_cross_entropy_with_logits(heatmap, target, reduction="none")
            else:
                l_heatmap = F.mse_loss(torch.sigmoid(heatmap), target, reduction="none")
            l_heatmap = l_heatmap[valid_xy].mean()
            total = total + weights["heatmap"] * l_heatmap
            logs["loss_heatmap"] = l_heatmap.detach().item()

        logs["loss_total"] = total.detach().item()
        return total, logs
