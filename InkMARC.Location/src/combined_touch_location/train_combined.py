from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from combined_touch_location.combined_model import CombinedTouchLocationModel, SUPPORTED_BACKBONES
from combined_touch_location.combined_transform import CombinedTouchLocationTransform
from data.dataset_core_v2 import InkMARCIndexedDataset
from data.label_schema import extract_xy_touch


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "state": torch.stack([b["state"] for b in batch], dim=0),
        "touch": torch.stack([b["touch"] for b in batch], dim=0),
        "xy": torch.stack([b["xy"] for b in batch], dim=0),
        "xy_mask": torch.stack([b["xy_mask"] for b in batch], dim=0),
        "heatmap": torch.stack([b["heatmap"] for b in batch], dim=0),
        "heatmap_mask": torch.stack([b["heatmap_mask"] for b in batch], dim=0),
        "meta": [b.get("meta", {}) for b in batch],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    import cv2  # type: ignore

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(1)


def maybe_subset(ds: Any, n: int) -> Any:
    if n <= 0 or n >= len(ds):
        return ds
    return Subset(ds, list(range(n)))


def subset_indices_random(length: int, n: int, seed: int) -> list[int]:
    if n <= 0 or n >= length:
        return list(range(length))
    rng = np.random.default_rng(seed)
    idx = np.arange(length, dtype=np.int64)
    rng.shuffle(idx)
    return idx[:n].tolist()


def select_loc_positive_indices(
    dataset_root: Path,
    version: str,
    split: str,
    n: int,
    seed: int,
    touch_pressure_threshold: float,
    require_labels: bool,
) -> list[int]:
    raw_ds = InkMARCIndexedDataset(
        dataset_root,
        index_version=version,
        split=split,
        require_labels=require_labels,
        transform=None,
    )
    try:
        positives: list[int] = []
        negatives: list[int] = []
        for i in range(len(raw_ds)):
            sample = raw_ds[i]
            touch, _touch_mask, _x, _y, xy_mask = extract_xy_touch(
                sample["labels"],
                sample["label_mask"],
                pressure_touch_threshold=touch_pressure_threshold,
            )
            if touch > 0.5 and xy_mask > 0.5:
                positives.append(i)
            else:
                negatives.append(i)

        if not positives:
            raise RuntimeError(
                f"No touch-positive xy-valid samples found in split='{split}' for threshold={touch_pressure_threshold}"
            )

        rng = np.random.default_rng(seed)
        rng.shuffle(positives)
        rng.shuffle(negatives)

        if n <= len(positives):
            chosen = positives[:n]
        else:
            extra = max(0, n - len(positives))
            chosen = positives + negatives[:extra]
        rng.shuffle(chosen)
        return chosen
    finally:
        raw_ds.close()


def classify_interaction_state(touch: float, xy_mask: float) -> str:
    """
    Three-state view of the data:
      - background: no location supervision and no touch
      - hover:      location supervision present, no touch
      - touch:      location supervision present, touch

    Any inconsistent edge case falls back to background for sampling purposes.
    """
    if touch > 0.5 and xy_mask > 0.5:
        return "touch"
    if touch <= 0.5 and xy_mask > 0.5:
        return "hover"
    return "background"


def summarize_subset_labels(
    dataset_root: Path,
    version: str,
    split: str,
    indices: list[int] | None,
    touch_pressure_threshold: float,
    require_labels: bool,
) -> Dict[str, int]:
    raw_ds = InkMARCIndexedDataset(
        dataset_root,
        index_version=version,
        split=split,
        require_labels=require_labels,
        transform=None,
    )
    try:
        if indices is None:
            selected = range(len(raw_ds))
        else:
            selected = indices

        counts = {
            "samples": 0,
            "touch_positive": 0,
            "xy_valid": 0,
            "loc_valid": 0,
            "background": 0,
            "hover": 0,
            "touch_state": 0,
        }
        for i in selected:
            sample = raw_ds[int(i)]
            touch, _touch_mask, _x, _y, xy_mask = extract_xy_touch(
                sample["labels"],
                sample["label_mask"],
                pressure_touch_threshold=touch_pressure_threshold,
            )
            counts["samples"] += 1
            if touch > 0.5:
                counts["touch_positive"] += 1
            if xy_mask > 0.5:
                counts["xy_valid"] += 1
            if touch > 0.5 and xy_mask > 0.5:
                counts["loc_valid"] += 1
            state = classify_interaction_state(touch, xy_mask)
            if state == "touch":
                counts["touch_state"] += 1
            else:
                counts[state] += 1
        return counts
    finally:
        raw_ds.close()


def build_three_state_sampler(
    dataset_root: Path,
    version: str,
    split: str,
    indices: list[int] | None,
    touch_pressure_threshold: float,
    require_labels: bool,
    ratio_background: float,
    ratio_hover: float,
    ratio_touch: float,
    generator: torch.Generator,
) -> tuple[WeightedRandomSampler, Dict[str, int]]:
    raw_ds = InkMARCIndexedDataset(
        dataset_root,
        index_version=version,
        split=split,
        require_labels=require_labels,
        transform=None,
    )
    try:
        if indices is None:
            selected = list(range(len(raw_ds)))
        else:
            selected = [int(i) for i in indices]

        state_by_pos: list[str] = []
        state_counts: Dict[str, int] = {"background": 0, "hover": 0, "touch": 0}
        for i in selected:
            sample = raw_ds[i]
            touch, _touch_mask, _x, _y, xy_mask = extract_xy_touch(
                sample["labels"],
                sample["label_mask"],
                pressure_touch_threshold=touch_pressure_threshold,
            )
            state = classify_interaction_state(touch, xy_mask)
            state_by_pos.append(state)
            state_counts[state] += 1

        target_sum = float(ratio_background + ratio_hover + ratio_touch)
        if target_sum <= 0.0:
            raise ValueError("Three-state sampler ratios must sum to > 0.")
        target_probs = {
            "background": float(ratio_background) / target_sum,
            "hover": float(ratio_hover) / target_sum,
            "touch": float(ratio_touch) / target_sum,
        }

        weights: list[float] = []
        for state in state_by_pos:
            count = state_counts[state]
            if count <= 0 or target_probs[state] <= 0.0:
                weights.append(0.0)
            else:
                weights.append(target_probs[state] / count)

        if not any(w > 0.0 for w in weights):
            raise RuntimeError("Three-state sampler produced all-zero weights.")

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(selected),
            replacement=True,
            generator=generator,
        )
        return sampler, state_counts
    finally:
        raw_ds.close()


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "state": batch["state"].to(device, non_blocking=True).long(),
        "touch": batch["touch"].to(device, non_blocking=True).float(),
        "xy": batch["xy"].to(device, non_blocking=True).float(),
        "xy_mask": batch["xy_mask"].to(device, non_blocking=True).float(),
        "heatmap": batch["heatmap"].to(device, non_blocking=True).float(),
        "heatmap_mask": batch["heatmap_mask"].to(device, non_blocking=True).float(),
        "meta": batch.get("meta", []),
    }


def masked_smoothl1_xy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_sample = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
    mask = mask.reshape(-1).float()
    den = mask.sum().clamp_min(1.0)
    return (per_sample * mask).sum() / den


def masked_kl_heatmap(logits: torch.Tensor, target_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.flatten(2), dim=-1)
    target = target_prob.flatten(2)
    per_sample = F.kl_div(logp, target, reduction="none").sum(dim=-1).squeeze(1)
    mask = mask.reshape(-1).float()
    den = mask.sum().clamp_min(1.0)
    return (per_sample * mask).sum() / den


def compute_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    *,
    lambda_state: float,
    lambda_binary_touch: float,
    lambda_xy: float,
    lambda_hm: float,
    binary_touch_pos_weight: float,
) -> Dict[str, torch.Tensor]:
    state = batch["state"].reshape(-1).long()
    touch = batch["touch"].reshape(-1).float()
    xy_mask = batch["xy_mask"].reshape(-1).float()
    heatmap_mask = batch["heatmap_mask"].reshape(-1).float()

    # State loss applies to all samples. Location supervision applies to
    # hover/touch samples, which are exactly the samples with valid xy labels.
    loc_mask = xy_mask
    hm_mask = heatmap_mask

    loss_state = F.cross_entropy(out["state_logits"], state)
    pos_weight = None
    if binary_touch_pos_weight > 0.0:
        pos_weight = torch.tensor(binary_touch_pos_weight, device=touch.device, dtype=touch.dtype)
    loss_binary_touch = F.binary_cross_entropy_with_logits(out["touch_logits"].reshape(-1), touch, pos_weight=pos_weight)
    loss_xy = masked_smoothl1_xy(out["xy"], batch["xy"], loc_mask)
    loss_hm = masked_kl_heatmap(out["heatmap_logits"], batch["heatmap"], hm_mask)
    loss = (
        float(lambda_state) * loss_state
        + float(lambda_binary_touch) * loss_binary_touch
        + float(lambda_xy) * loss_xy
        + float(lambda_hm) * loss_hm
    )

    return {
        "loss": loss,
        "loss_state": loss_state,
        "loss_touch": loss_binary_touch,
        "loss_binary_touch": loss_binary_touch,
        "loss_xy": loss_xy,
        "loss_hm": loss_hm,
        "loc_mask": loc_mask,
        "hm_mask": hm_mask,
    }


def train_one_epoch(
    *,
    model: CombinedTouchLocationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = {}
    seen = 0
    start = perf_counter()
    num_batches = len(loader) if hasattr(loader, "__len__") else 0

    for bi, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch["image"])
            warmup_active = int(args.touch_warmup_epochs) > 0 and epoch <= int(args.touch_warmup_epochs)
            losses = compute_losses(
                out,
                batch,
                lambda_state=args.touch_warmup_lambda_state if warmup_active else args.lambda_state,
                lambda_binary_touch=args.lambda_binary_touch,
                lambda_xy=args.touch_warmup_lambda_xy if warmup_active else args.lambda_xy,
                lambda_hm=args.touch_warmup_lambda_hm if warmup_active else args.lambda_hm,
                binary_touch_pos_weight=args.binary_touch_pos_weight,
            )

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = int(batch["image"].shape[0])
        seen += bs
        for key in ("loss", "loss_state", "loss_touch", "loss_binary_touch", "loss_xy", "loss_hm"):
            sums[key] = sums.get(key, 0.0) + float(losses[key].detach().item()) * bs

        if args.log_every > 0 and num_batches > 0 and (
            bi == 1 or bi % args.log_every == 0 or bi == num_batches
        ):
            avg = sums["loss"] / max(seen, 1)
            print(f"[train] epoch={epoch:03d} batch={bi}/{num_batches} avg_loss={avg:.4f} elapsed={perf_counter() - start:.1f}s")

    return {k: v / max(seen, 1) for k, v in sums.items()}


@torch.no_grad()
def recalibrate_batchnorm(
    *,
    model: CombinedTouchLocationModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
) -> None:
    """
    Recompute BatchNorm running statistics from the current training subset.

    MobileNetV3 uses BatchNorm throughout the shared backbone. On tiny overfit
    runs, the learned weights can depend on live batch statistics while the
    running statistics lag behind, making eval-mode metrics look collapsed.
    This pass updates only BatchNorm buffers; it does not run backward or change
    trainable parameters.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return

    was_training = model.training
    old_momenta = {m: m.momentum for m in bn_modules}
    for module in bn_modules:
        module.reset_running_stats()
        module.momentum = None

    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

    for bi, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            model(batch["image"])
        if max_batches > 0 and bi >= max_batches:
            break

    for module, momentum in old_momenta.items():
        module.momentum = momentum
    model.train(was_training)


@torch.no_grad()
def evaluate(
    *,
    model: CombinedTouchLocationModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    seen = 0
    touch_tp = touch_fp = touch_tn = touch_fn = 0
    state_touch_tp = state_touch_fp = state_touch_tn = state_touch_fn = 0
    state_cm = torch.zeros(3, 3, dtype=torch.long)
    interaction_tp = interaction_fp = interaction_tn = interaction_fn = 0
    loc_count = 0
    xy_mae_sum = 0.0
    xy_mae_px_sum = 0.0
    nde_chunks: List[torch.Tensor] = []
    touch_prob_chunks: List[torch.Tensor] = []
    touch_true_chunks: List[torch.Tensor] = []
    scale = torch.tensor([args.out_w, args.out_h], device=device, dtype=torch.float32)

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch["image"])
            losses = compute_losses(
                out,
                batch,
                lambda_state=args.lambda_state,
                lambda_binary_touch=args.lambda_binary_touch,
                lambda_xy=args.lambda_xy,
                lambda_hm=args.lambda_hm,
                binary_touch_pos_weight=args.binary_touch_pos_weight,
            )

        bs = int(batch["image"].shape[0])
        seen += bs
        for key in ("loss", "loss_state", "loss_touch", "loss_binary_touch", "loss_xy", "loss_hm"):
            sums[key] = sums.get(key, 0.0) + float(losses[key].detach().item()) * bs

        state_true = batch["state"].reshape(-1).long()
        state_pred = out["state_logits"].argmax(dim=1).reshape(-1)
        for t, p in zip(state_true.detach().cpu(), state_pred.detach().cpu()):
            state_cm[int(t), int(p)] += 1

        touch_true = batch["touch"].reshape(-1).float() > 0.5
        touch_prob = torch.sigmoid(out["touch_logits"].reshape(-1).float())
        touch_pred = touch_prob >= float(args.touch_threshold)
        touch_prob_chunks.append(touch_prob.detach().cpu())
        touch_true_chunks.append(touch_true.detach().cpu())
        touch_tp += int((touch_pred & touch_true).sum().item())
        touch_fp += int((touch_pred & ~touch_true).sum().item())
        touch_tn += int((~touch_pred & ~touch_true).sum().item())
        touch_fn += int((~touch_pred & touch_true).sum().item())

        state_touch_true = state_true == 2
        state_touch_pred = state_pred == 2
        state_touch_tp += int((state_touch_pred & state_touch_true).sum().item())
        state_touch_fp += int((state_touch_pred & ~state_touch_true).sum().item())
        state_touch_tn += int((~state_touch_pred & ~state_touch_true).sum().item())
        state_touch_fn += int((~state_touch_pred & state_touch_true).sum().item())

        interaction_true = state_true > 0
        interaction_pred = state_pred > 0
        interaction_tp += int((interaction_pred & interaction_true).sum().item())
        interaction_fp += int((interaction_pred & ~interaction_true).sum().item())
        interaction_tn += int((~interaction_pred & ~interaction_true).sum().item())
        interaction_fn += int((~interaction_pred & interaction_true).sum().item())

        loc_mask = losses["loc_mask"] > 0.5
        if loc_mask.any():
            abs_err = (out["xy"] - batch["xy"]).abs().float()
            xy_mae_sum += float(abs_err.mean(dim=1)[loc_mask].sum().item())
            xy_mae_px_sum += float((abs_err * scale).mean(dim=1)[loc_mask].sum().item())
            eu = torch.sqrt(((out["xy"] - batch["xy"]).float() ** 2).sum(dim=1))
            nde_chunks.append(eu[loc_mask].detach().cpu())
            loc_count += int(loc_mask.sum().item())

    total_touch = touch_tp + touch_fp + touch_tn + touch_fn
    precision = touch_tp / max(touch_tp + touch_fp, 1)
    recall = touch_tp / max(touch_tp + touch_fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    state_touch_total = state_touch_tp + state_touch_fp + state_touch_tn + state_touch_fn
    state_touch_precision = state_touch_tp / max(state_touch_tp + state_touch_fp, 1)
    state_touch_recall = state_touch_tp / max(state_touch_tp + state_touch_fn, 1)
    state_touch_f1 = (
        2.0 * state_touch_precision * state_touch_recall / max(state_touch_precision + state_touch_recall, 1e-12)
    )
    interaction_precision = interaction_tp / max(interaction_tp + interaction_fp, 1)
    interaction_recall = interaction_tp / max(interaction_tp + interaction_fn, 1)
    interaction_f1 = (
        2.0 * interaction_precision * interaction_recall / max(interaction_precision + interaction_recall, 1e-12)
    )
    best_threshold = float(args.touch_threshold)
    best_precision = precision
    best_recall = recall
    best_f1 = f1
    best_pred_positive_rate = (touch_tp + touch_fp) / max(total_touch, 1)
    if touch_prob_chunks:
        probs = torch.cat(touch_prob_chunks)
        truth = torch.cat(touch_true_chunks).bool()
        for threshold in torch.linspace(0.05, 0.95, 91):
            pred = probs >= threshold
            b_tp = int((pred & truth).sum().item())
            b_fp = int((pred & ~truth).sum().item())
            b_fn = int((~pred & truth).sum().item())
            b_precision = b_tp / max(b_tp + b_fp, 1)
            b_recall = b_tp / max(b_tp + b_fn, 1)
            b_f1 = 2.0 * b_precision * b_recall / max(b_precision + b_recall, 1e-12)
            if b_f1 > best_f1:
                best_threshold = float(threshold.item())
                best_precision = b_precision
                best_recall = b_recall
                best_f1 = b_f1
                best_pred_positive_rate = (b_tp + b_fp) / max(total_touch, 1)

    per_state: Dict[str, float] = {}
    state_names = ["background", "hover", "touch"]
    macro_f1_sum = 0.0
    for si, name in enumerate(state_names):
        s_tp = int(state_cm[si, si].item())
        s_fp = int(state_cm[:, si].sum().item()) - s_tp
        s_fn = int(state_cm[si, :].sum().item()) - s_tp
        s_precision = s_tp / max(s_tp + s_fp, 1)
        s_recall = s_tp / max(s_tp + s_fn, 1)
        s_f1 = 2.0 * s_precision * s_recall / max(s_precision + s_recall, 1e-12)
        macro_f1_sum += s_f1
        per_state[f"state_{name}_precision"] = s_precision
        per_state[f"state_{name}_recall"] = s_recall
        per_state[f"state_{name}_f1"] = s_f1

    metrics = {k: v / max(seen, 1) for k, v in sums.items()}
    metrics.update(
        {
            "state_accuracy": float(state_cm.diag().sum().item()) / max(int(state_cm.sum().item()), 1),
            "state_macro_f1": macro_f1_sum / 3.0,
            "interaction_precision": interaction_precision,
            "interaction_recall": interaction_recall,
            "interaction_f1": interaction_f1,
            "touch_accuracy": (touch_tp + touch_tn) / max(total_touch, 1),
            "touch_precision": precision,
            "touch_recall": recall,
            "touch_f1": f1,
            "touch_true_positive_rate": (touch_tp + touch_fn) / max(total_touch, 1),
            "touch_pred_positive_rate": (touch_tp + touch_fp) / max(total_touch, 1),
            "touch_best_f1": best_f1,
            "touch_best_threshold": best_threshold,
            "touch_best_precision": best_precision,
            "touch_best_recall": best_recall,
            "touch_best_pred_positive_rate": best_pred_positive_rate,
            "touch_tp": float(touch_tp),
            "touch_fp": float(touch_fp),
            "touch_tn": float(touch_tn),
            "touch_fn": float(touch_fn),
            "state_touch_accuracy": (state_touch_tp + state_touch_tn) / max(state_touch_total, 1),
            "state_touch_precision": state_touch_precision,
            "state_touch_recall": state_touch_recall,
            "state_touch_f1": state_touch_f1,
            "state_touch_tp": float(state_touch_tp),
            "state_touch_fp": float(state_touch_fp),
            "state_touch_tn": float(state_touch_tn),
            "state_touch_fn": float(state_touch_fn),
            "loc_valid_samples": float(loc_count),
            "xy_mae": xy_mae_sum / loc_count if loc_count > 0 else float("nan"),
            "xy_mae_px": xy_mae_px_sum / loc_count if loc_count > 0 else float("nan"),
            "nde_median": float(torch.cat(nde_chunks).median().item()) if nde_chunks else float("nan"),
        }
    )
    metrics.update(per_state)
    return metrics


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def torch_load_checkpoint(path: Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(requires_grad)


def configure_trainable_parameters(model: CombinedTouchLocationModel, args: argparse.Namespace) -> Dict[str, int]:
    if args.freeze_backbone:
        set_requires_grad(model.features, False)
        if args.unfreeze_last_backbone_blocks > 0:
            children = list(model.features.children())
            for block in children[-int(args.unfreeze_last_backbone_blocks):]:
                set_requires_grad(block, True)
    if args.freeze_heatmap_head:
        set_requires_grad(model.heatmap_head, False)
    if args.freeze_state_head:
        set_requires_grad(model.state_head, False)
    if args.freeze_touch_head:
        set_requires_grad(model.touch_head, False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def build_loader(
    *,
    dataset_root: Path,
    version: str,
    split: str,
    args: argparse.Namespace,
    shuffle: bool,
    generator: torch.Generator,
    forced_indices: list[int] | None = None,
    summary_label: str | None = None,
) -> tuple[DataLoader, CombinedTouchLocationTransform, list[int] | None]:
    transform = CombinedTouchLocationTransform(
        dataset_root=dataset_root,
        index_version=version,
        split=split,
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        touch_pressure_threshold=args.touch_pressure_threshold,
        video_cache_size=args.video_cache_size,
    )
    ds = InkMARCIndexedDataset(
        dataset_root,
        index_version=version,
        split=split,
        require_labels=args.require_any_label,
        transform=transform,
    )
    selected_indices: list[int] | None = forced_indices
    if forced_indices is not None:
        ds = Subset(ds, forced_indices)
    elif split == "train":
        if args.overfit_small > 0:
            if args.overfit_loc_positive_only:
                selected_indices = select_loc_positive_indices(
                    dataset_root=dataset_root,
                    version=version,
                    split=split,
                    n=args.overfit_small,
                    seed=args.seed,
                    touch_pressure_threshold=args.touch_pressure_threshold,
                    require_labels=args.require_any_label,
                )
            else:
                selected_indices = subset_indices_random(len(ds), args.overfit_small, args.seed)
            ds = Subset(ds, selected_indices)
    elif split == "val":
        val_cap = args.val_max_samples if args.val_max_samples > 0 else args.overfit_small
        if val_cap > 0:
            selected_indices = subset_indices_random(len(ds), val_cap, args.seed + 1)
            ds = Subset(ds, selected_indices)

    persistent = bool(args.persistent_workers and args.num_workers > 0)
    prefetch = args.prefetch_factor if args.num_workers > 0 else None
    sampler = None
    sampler_state_counts = None
    if split == "train" and shuffle and args.train_sampler == "three_state":
        sampler, sampler_state_counts = build_three_state_sampler(
            dataset_root=dataset_root,
            version=version,
            split=split,
            indices=selected_indices,
            touch_pressure_threshold=args.touch_pressure_threshold,
            require_labels=args.require_any_label,
            ratio_background=args.sampler_ratio_background,
            ratio_hover=args.sampler_ratio_hover,
            ratio_touch=args.sampler_ratio_touch,
            generator=generator,
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(torch.cuda.is_available() and str(args.device).startswith("cuda")),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    label_summary = summarize_subset_labels(
        dataset_root=dataset_root,
        version=version,
        split=split,
        indices=selected_indices,
        touch_pressure_threshold=args.touch_pressure_threshold,
        require_labels=args.require_any_label,
    )
    label = summary_label or split
    print(
        f"Subset summary [{label}] | "
        f"samples={label_summary['samples']} "
        f"touch_positive={label_summary['touch_positive']} "
        f"xy_valid={label_summary['xy_valid']} "
        f"loc_valid={label_summary['loc_valid']} "
        f"background={label_summary['background']} "
        f"hover={label_summary['hover']} "
        f"touch_state={label_summary['touch_state']}"
    )
    if sampler_state_counts is not None:
        print(
            f"Sampler summary [{label}] | "
            f"mode=three_state "
            f"source_background={sampler_state_counts['background']} "
            f"source_hover={sampler_state_counts['hover']} "
            f"source_touch={sampler_state_counts['touch']} "
            f"target_background={args.sampler_ratio_background:.3f} "
            f"target_hover={args.sampler_ratio_hover:.3f} "
            f"target_touch={args.sampler_ratio_touch:.3f}"
        )

    return loader, transform, selected_indices


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train combined touch/location MobileNetV3 model.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out-dir", default="runs/combined_touch_location")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--hm-w", type=int, default=60)
    ap.add_argument("--hm-h", type=int, default=40)
    ap.add_argument("--hm-sigma", type=float, default=1.5)
    ap.add_argument("--touch-pressure-threshold", type=float, default=0.0)
    ap.add_argument("--lambda-touch", type=float, default=1.0, help="Deprecated alias for --lambda-state.")
    ap.add_argument("--lambda-state", type=float, default=None, help="State classification loss weight.")
    ap.add_argument("--lambda-binary-touch", type=float, default=1.0, help="Binary touch head BCE loss weight.")
    ap.add_argument(
        "--binary-touch-pos-weight",
        type=float,
        default=0.0,
        help="Optional BCE positive-class weight for the binary touch head. Default 0 disables pos_weight.",
    )
    ap.add_argument("--touch-threshold", type=float, default=0.5, help="Probability threshold for binary touch metrics.")
    ap.add_argument(
        "--touch-decision-threshold",
        dest="touch_threshold",
        type=float,
        help="Deprecated alias for --touch-threshold.",
    )
    ap.add_argument("--lambda-xy", type=float, default=1.0)
    ap.add_argument("--lambda-hm", type=float, default=1.0)
    ap.add_argument(
        "--touch-warmup-epochs",
        type=int,
        default=0,
        help="Train the binary touch head with reduced auxiliary losses for the first N epochs.",
    )
    ap.add_argument("--touch-warmup-lambda-state", type=float, default=0.0)
    ap.add_argument("--touch-warmup-lambda-xy", type=float, default=0.0)
    ap.add_argument("--touch-warmup-lambda-hm", type=float, default=0.0)
    ap.add_argument("--backbone", default="mobilenet_v3_large", choices=SUPPORTED_BACKBONES)
    ap.add_argument("--pretrained-backbone", action="store_true", help="Initialize MobileNetV3 from ImageNet weights.")
    ap.add_argument(
        "--imagenet-normalize",
        action="store_true",
        help="Apply ImageNet mean/std normalization inside the model. Intended for --pretrained-backbone.",
    )
    ap.add_argument("--freeze-backbone", action="store_true", help="Freeze MobileNet feature extractor parameters.")
    ap.add_argument("--freeze-heatmap-head", action="store_true", help="Freeze heatmap/location head parameters.")
    ap.add_argument("--freeze-state-head", action="store_true", help="Freeze 3-class state head parameters.")
    ap.add_argument("--freeze-touch-head", action="store_true", help="Freeze binary touch head parameters.")
    ap.add_argument(
        "--unfreeze-last-backbone-blocks",
        type=int,
        default=0,
        help="When --freeze-backbone is set, re-enable gradients for the last N MobileNet feature blocks.",
    )
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--video-cache-size", type=int, default=16)
    ap.add_argument(
        "--require-any-label",
        action="store_true",
        help="Filter to frames with any label_mask entry. Default keeps unlabeled background frames for three-state training.",
    )
    ap.add_argument(
        "--train-sampler",
        default="uniform",
        choices=["uniform", "three_state"],
        help="Training sampler mode. 'three_state' rebalances background/hover/touch with replacement.",
    )
    ap.add_argument("--sampler-ratio-background", type=float, default=0.25)
    ap.add_argument("--sampler-ratio-hover", type=float, default=0.35)
    ap.add_argument("--sampler-ratio-touch", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--onnx-path", default="")
    ap.add_argument("--onnx-opset", type=int, default=17)
    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--overfit-small", nargs="?", const=64, type=int, default=0)
    ap.add_argument(
        "--overfit-loc-positive-only",
        action="store_true",
        help="When using --overfit-small, choose touch-positive xy-valid training samples first.",
    )
    ap.add_argument(
        "--overfit-eval-train-subset",
        action="store_true",
        help="When using --overfit-small, evaluate on the exact selected train subset instead of val.",
    )
    ap.add_argument("--val-max-samples", type=int, default=0)
    ap.add_argument("--resume", default="", help="Resume training from a checkpoint path.")
    ap.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore optimizer state when --resume is used.",
    )
    ap.add_argument(
        "--recalibrate-bn-before-eval",
        action="store_true",
        help="Recompute BatchNorm running stats from the training loader before each validation pass.",
    )
    ap.add_argument(
        "--bn-recalibration-max-batches",
        type=int,
        default=0,
        help="Limit BatchNorm recalibration batches. Default 0 uses the full training loader.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.lambda_state is None:
        args.lambda_state = args.lambda_touch
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "training_config.json", vars(args))

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))

    train_tf: CombinedTouchLocationTransform | None = None
    val_tf: CombinedTouchLocationTransform | None = None
    metrics_history: List[Dict[str, Any]] = []
    best_xy_mae_px = float("inf")
    best_touch_f1 = float("-inf")
    best_path = out_dir / "checkpoint_best.pt"
    best_touch_path = out_dir / "checkpoint_best_touch.pt"
    last_ckpt_path: Path | None = None
    start_epoch = 1

    try:
        train_loader, train_tf, train_indices = build_loader(
            dataset_root=Path(args.dataset_root),
            version=args.version,
            split="train",
            args=args,
            shuffle=True,
            generator=generator,
            summary_label="train",
        )
        if args.overfit_eval_train_subset and args.overfit_small > 0:
            val_loader, val_tf, _ = build_loader(
                dataset_root=Path(args.dataset_root),
                version=args.version,
                split="train",
                args=args,
                shuffle=False,
                generator=generator,
                forced_indices=train_indices,
                summary_label="eval_train_subset",
            )
        else:
            val_loader, val_tf, _ = build_loader(
                dataset_root=Path(args.dataset_root),
                version=args.version,
                split="val",
                args=args,
                shuffle=False,
                generator=generator,
                summary_label="val",
            )

        print(
            "Data summary | "
            f"dataset_root={Path(args.dataset_root).resolve()} version={args.version} "
            f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
            f"batch_size={args.batch_size}"
        )
        print(
            "Optimization summary | "
            f"epochs={args.epochs} train_batches_per_epoch={len(train_loader)} "
            f"total_optimizer_steps={args.epochs * len(train_loader)}"
        )

        model = CombinedTouchLocationModel(
            backbone=args.backbone,
            hm_h=args.hm_h,
            hm_w=args.hm_w,
            pretrained_backbone=args.pretrained_backbone,
            imagenet_normalize=args.imagenet_normalize,
        ).to(device)

        if args.resume:
            resume_path = Path(args.resume)
            ckpt = torch_load_checkpoint(resume_path, map_location=device)
            if not isinstance(ckpt, dict) or "model_state" not in ckpt:
                raise RuntimeError(f"Unsupported checkpoint format for --resume: {resume_path}")
            model.load_state_dict(ckpt["model_state"], strict=True)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_xy_mae_px = float(ckpt.get("best_val_xy_mae_px", best_xy_mae_px))
            best_touch_f1 = float(ckpt.get("best_val_touch_f1", best_touch_f1))
            history_path = out_dir / "metrics_history.json"
            if history_path.exists():
                loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(loaded_history, list):
                    metrics_history = loaded_history
            print(
                "Resume summary | "
                f"checkpoint={resume_path} start_epoch={start_epoch} "
                f"best_val_xy_mae_px={best_xy_mae_px:.4f} "
                f"best_val_touch_f1={best_touch_f1:.4f} "
                f"optimizer_restore_requested={bool(args.resume_optimizer and 'optimizer_state' in ckpt)}"
            )

        trainable_summary = configure_trainable_parameters(model, args)
        print(
            "Trainable summary | "
            f"trainable_params={trainable_summary['trainable_params']} "
            f"total_params={trainable_summary['total_params']} "
            f"freeze_backbone={args.freeze_backbone} "
            f"unfreeze_last_backbone_blocks={args.unfreeze_last_backbone_blocks}"
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters remain after applying freeze options.")
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

        if args.resume and args.resume_optimizer:
            resume_path = Path(args.resume)
            ckpt = torch_load_checkpoint(resume_path, map_location=device)
            if isinstance(ckpt, dict) and "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except ValueError as exc:
                    print(f"Warning: optimizer state was not restored after freeze configuration: {exc}")

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                args=args,
                epoch=epoch,
            )
            if args.recalibrate_bn_before_eval:
                recalibrate_batchnorm(
                    model=model,
                    loader=train_loader,
                    device=device,
                    use_amp=use_amp,
                    max_batches=args.bn_recalibration_max_batches,
                )
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                args=args,
            )

            val_xy = float(val_metrics.get("xy_mae_px", float("nan")))
            is_best = math.isfinite(val_xy) and val_xy < best_xy_mae_px
            if is_best:
                best_xy_mae_px = val_xy
            val_touch = float(val_metrics.get("touch_f1", float("nan")))
            is_best_touch = math.isfinite(val_touch) and val_touch > best_touch_f1
            if is_best_touch:
                best_touch_f1 = val_touch

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_xy_mae_px": best_xy_mae_px,
                "best_val_touch_f1": best_touch_f1,
                "args": vars(args),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            last_ckpt_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(ckpt, last_ckpt_path)
            if is_best:
                torch.save(ckpt, best_path)
            if is_best_touch:
                torch.save(ckpt, best_touch_path)

            row = {
                "epoch": epoch,
                "is_best": is_best,
                "is_best_touch": is_best_touch,
                "best_val_xy_mae_px": best_xy_mae_px,
                "best_val_touch_f1": best_touch_f1,
                "train": train_metrics,
                "val": val_metrics,
            }
            metrics_history.append(row)
            write_json(out_dir / "metrics_history.json", metrics_history)

            print(
                f"epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.4f} "
                f"(state={train_metrics['loss_state']:.4f} touch={train_metrics['loss_touch']:.4f} "
                f"xy={train_metrics['loss_xy']:.4f} hm={train_metrics['loss_hm']:.4f}) | "
                f"val loss={val_metrics['loss']:.4f} "
                f"state_acc={val_metrics['state_accuracy']:.3f} state_macro_f1={val_metrics['state_macro_f1']:.3f} "
                f"interaction_f1={val_metrics['interaction_f1']:.3f} "
                f"touch_f1={val_metrics['touch_f1']:.3f} touch_p={val_metrics['touch_precision']:.3f} "
                f"touch_r={val_metrics['touch_recall']:.3f} touch_true_pos={val_metrics['touch_true_positive_rate']:.3f} "
                f"touch_pred_pos={val_metrics['touch_pred_positive_rate']:.3f} "
                f"touch_best_f1={val_metrics['touch_best_f1']:.3f}@{val_metrics['touch_best_threshold']:.2f} "
                f"state_touch_f1={val_metrics['state_touch_f1']:.3f} "
                f"xy_mae_px={val_metrics['xy_mae_px']:.2f} nde_med={val_metrics['nde_median']:.4f} "
                f"{'BEST_XY' if is_best else ''} {'BEST_TOUCH' if is_best_touch else ''}"
            )

        if not best_path.exists():
            print("Warning: checkpoint_best.pt was not written because validation xy_mae_px was never finite.")
        if not best_touch_path.exists():
            print("Warning: checkpoint_best_touch.pt was not written because validation touch_f1 was never finite.")

        if args.export_onnx:
            from combined_touch_location.export_combined_onnx import export_onnx_from_checkpoint

            ckpt_to_export = best_path if best_path.exists() else last_ckpt_path
            if ckpt_to_export is None:
                raise RuntimeError("No checkpoint available for ONNX export.")
            onnx_path = Path(args.onnx_path) if args.onnx_path else out_dir / "combined_touch_location.onnx"
            export_onnx_from_checkpoint(
                checkpoint_path=ckpt_to_export,
                onnx_path=onnx_path,
                device=device,
                out_h=args.out_h,
                out_w=args.out_w,
                onnx_opset=args.onnx_opset,
                include_heatmap=False,
                backbone=args.backbone,
                hm_h=args.hm_h,
                hm_w=args.hm_w,
            )

    finally:
        if train_tf is not None:
            train_tf.close()
        if val_tf is not None:
            val_tf.close()


if __name__ == "__main__":
    main()
