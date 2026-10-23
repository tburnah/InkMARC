from __future__ import annotations

import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import cv2

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import InkDepthStudent
from inkdepth.models.losses import MultiTaskLoss
from inkdepth.training.metrics import MetricTracker, binary_touch_metrics, state_accuracy, xy_error_metrics
from inkdepth.training.validate import validate
from inkdepth.utils.checkpoints import load_checkpoint, load_model_weights, save_checkpoint
from inkdepth.utils.config import ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loaders(cfg: Any) -> tuple[DataLoader, DataLoader]:
    train_ds = StylusDataset(cfg, split="train", training=True)
    val_ds = StylusDataset(cfg, split="val", training=False)
    sampler = build_sampler(train_ds, cfg)
    num_workers = int(cfg.training.num_workers)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": stylus_collate_fn,
        "worker_init_fn": _worker_init_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(cfg.training, "persistent_workers", False))
        prefetch_factor = getattr(cfg.training, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        **loader_kwargs,
        drop_last=False,
    )
    return train_loader, val_loader


def _worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_sampler(dataset: StylusDataset, cfg: Any) -> WeightedRandomSampler | None:
    mode = str(getattr(cfg.training, "sampler", "uniform"))
    if mode == "uniform":
        return None
    weights: list[float] = []
    counts = {
        "background": 0,
        "hover": 0,
        "touch": 0,
        "unlabeled": 0,
        "valid_xy": 0,
        "valid_touch": 0,
        "touch_positive": 0,
        "touch_negative": 0,
    }

    touch_target = str(getattr(cfg.data.labels, "touch_target", "pressure")).lower()

    def label_for_record(rec: dict[str, Any]) -> dict[str, Any]:
        if dataset.synthetic:
            return {"state": 1, "touch": 1.0, "valid_xy": True}
        if "state_hint" in rec:
            state = rec.get("state_hint")
            valid_xy = state in {1, 2}
            if touch_target in {"interaction", "presence", "present", "stylus_present"}:
                has_touch = state in {0, 1, 2}
                touch = 1.0 if state in {1, 2} else 0.0
            else:
                has_touch = state in {1, 2}
                touch = 1.0 if state == 2 else 0.0
            return {"state": state, "touch": touch, "valid_xy": valid_xy, "has_touch": has_touch}
        return dataset._lookup_label(rec)

    labels_by_index = [label_for_record(rec) for rec in dataset.records]
    for label in labels_by_index:
        state = label.get("state")
        valid_xy = bool(label.get("valid_xy", False))
        touch = label.get("touch")
        has_touch = bool(label.get("has_touch", touch is not None))
        if state == 0:
            counts["background"] += 1
        elif state == 1:
            counts["hover"] += 1
        elif state == 2:
            counts["touch"] += 1
        else:
            counts["unlabeled"] += 1
        counts["valid_xy"] += int(valid_xy)
        counts["valid_touch"] += int(has_touch)
        counts["touch_positive"] += int(has_touch and touch is not None and float(touch) > 0.5)
        counts["touch_negative"] += int(has_touch and touch is not None and float(touch) <= 0.5)

    class_counts = {
        0: max(1, counts["background"]),
        1: max(1, counts["hover"]),
        2: max(1, counts["touch"]),
        None: max(1, counts["unlabeled"]),
    }
    group_touch_counts = _group_touch_counts(dataset.records, labels_by_index, mode)
    for rec, label in zip(dataset.records, labels_by_index):
        state = label.get("state")
        valid_xy = bool(label.get("valid_xy", False))
        touch = label.get("touch")
        has_touch = bool(label.get("has_touch", touch is not None))
        if mode == "class_balanced":
            weights.append(1.0 / class_counts.get(state, class_counts[None]))
        elif mode == "valid_xy_only":
            weights.append(1.0 if valid_xy else 0.0)
        elif mode == "valid_xy_balanced":
            pos = max(1, counts["valid_xy"])
            neg = max(1, len(dataset.records) - counts["valid_xy"])
            weights.append(1.0 / (pos if valid_xy else neg))
        elif mode == "touch_balanced":
            pos = max(1, counts["touch_positive"])
            neg = max(1, counts["touch_negative"])
            if not has_touch or touch is None:
                weights.append(0.0)
            else:
                weights.append(1.0 / (pos if float(touch) > 0.5 else neg))
        elif mode in {"task_touch", "video_touch"}:
            if not has_touch or touch is None:
                weights.append(0.0)
            else:
                group = _sampler_group_key(rec, mode)
                group_counts = group_touch_counts[group]
                key = "positive" if float(touch) > 0.5 else "negative"
                weights.append(1.0 / max(1, group_counts[key]))
        elif mode == "touch_oversample":
            weights.append(5.0 if touch is not None and float(touch) > 0.5 else 1.0)
        else:
            raise ValueError(f"Unknown training.sampler={mode}")
    if sum(weights) <= 0:
        raise ValueError(f"Sampler {mode} has no eligible samples. Check label validity.")
    print(f"sampler={mode} label_counts={counts}")
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def _group_touch_counts(
    records: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    mode: str,
) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    if mode not in {"task_touch", "video_touch"}:
        return out
    for rec, label in zip(records, labels):
        touch = label.get("touch")
        has_touch = bool(label.get("has_touch", touch is not None))
        if not has_touch or touch is None:
            continue
        group = _sampler_group_key(rec, mode)
        out[group]["positive" if float(touch) > 0.5 else "negative"] += 1
    return out


def _sampler_group_key(rec: dict[str, Any], mode: str) -> str:
    if mode == "task_touch":
        task_id = rec.get("task_id")
        if task_id not in {None, "", "None"}:
            return str(task_id)
        return _task_from_video_id(str(rec.get("base_stem") or rec.get("video_id") or rec.get("video") or "UNKNOWN"))
    return str(rec.get("base_stem") or rec.get("video_id") or rec.get("video") or "UNKNOWN")


def _task_from_video_id(video_id: str) -> str:
    parts = video_id.split("_")
    return parts[2] if len(parts) >= 3 and parts[0] == "participant" else "UNKNOWN"


def build_optimizer(model: InkDepthStudent, cfg: Any) -> torch.optim.Optimizer:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(cfg.optimizer.backbone_lr)})
    if head_params:
        param_groups.append({"params": head_params, "lr": float(cfg.optimizer.head_lr)})
    if not param_groups:
        raise ValueError("No trainable parameters. Check model.freeze_* settings.")
    return torch.optim.AdamW(param_groups, weight_decay=float(cfg.optimizer.weight_decay))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Any) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not bool(cfg.scheduler.enabled):
        return None
    if str(cfg.scheduler.name) != "cosine":
        raise ValueError("Only scheduler.name=cosine is implemented")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.epochs),
        eta_min=float(cfg.scheduler.min_lr),
    )


def train(cfg: Any) -> dict[str, float]:
    set_seed(int(cfg.seed))
    output_dir = ensure_dir(cfg.output_dir)
    _check_coordinate_mode(cfg)
    _copy_manifest(cfg, output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)
    model = InkDepthStudent.from_config(cfg).to(device)
    init_checkpoint = getattr(cfg.training, "init_checkpoint", None)
    if init_checkpoint:
        load_model_weights(
            init_checkpoint,
            model,
            map_location=device,
            strict=bool(getattr(cfg.training, "init_strict", True)),
            skip_mismatched=bool(getattr(cfg.training, "init_skip_mismatched", False)),
        )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.training.amp) and device.type == "cuda")
    criterion = MultiTaskLoss(cfg)
    start_epoch = 0

    if cfg.training.resume:
        start_epoch = load_checkpoint(cfg.training.resume, model, optimizer, scaler, scheduler, map_location=device)

    writer = _build_summary_writer(cfg, output_dir)
    csv_path = output_dir / "metrics.csv"
    best = {"val_loss": float("inf"), "touch_f1": -float("inf"), "state_f1": -float("inf"), "xy_pix": float("inf")}
    last_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        model.train()
        tracker = MetricTracker()
        sampled_counts: Counter[str] = Counter()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.training.epochs}")
        for step, batch in enumerate(pbar):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(batch["image"])
                loss, logs = criterion(outputs, batch)
            if not loss.requires_grad:
                n = batch["image"].shape[0]
                _update_sampled_counts(sampled_counts, batch)
                tracker.update(logs, n=n)
                sampled_counts["empty_loss_batches"] += 1
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            n = batch["image"].shape[0]
            _update_sampled_counts(sampled_counts, batch)
            tracker.update(logs, n=n)
            if outputs.get("touch_logits") is not None:
                tracker.update(binary_touch_metrics(outputs["touch_logits"], batch["touch"], batch["valid_touch"]), n=n)
            if outputs.get("state_logits") is not None:
                tracker.update(state_accuracy(outputs["state_logits"], batch["state"], batch["valid_state"]), n=n)
            if outputs.get("xy") is not None:
                tracker.update(xy_error_metrics(outputs["xy"], batch["xy"], batch["valid_xy"], cfg.data.input_size), n=n)

            if step % int(cfg.training.log_every) == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in tracker.compute().items() if k.startswith("loss_")})

        if scheduler is not None:
            scheduler.step()

        train_metrics = {f"train/{k}": v for k, v in tracker.compute().items()}
        train_metrics.update({f"train/sampled_{k}": float(v) for k, v in sampled_counts.items()})
        val_metrics: dict[str, float] = {}
        if (epoch + 1) % int(cfg.training.val_every) == 0:
            raw_val = validate(
                model,
                val_loader,
                cfg,
                device,
                epoch=epoch + 1,
                save_visuals=bool(cfg.training.save_visuals),
                output_dir=output_dir,
            )
            val_metrics = {f"val/{k}": v for k, v in raw_val.items()}

        last_metrics = {**train_metrics, **val_metrics}
        _write_logs(writer, csv_path, epoch + 1, last_metrics)
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, last_metrics, scaler, scheduler, dict(cfg))
        _save_bests(output_dir, model, optimizer, epoch, last_metrics, scaler, scheduler, cfg, best)

    writer.close()
    return last_metrics


def _save_bests(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Any,
    cfg: Any,
    best: dict[str, float],
) -> None:
    val_loss = metrics.get("val/val_loss")
    if val_loss is not None and val_loss < best["val_loss"]:
        best["val_loss"] = val_loss
        save_checkpoint(output_dir / "best_val_loss.pt", model, optimizer, epoch, metrics, scaler, scheduler, dict(cfg))
    touch_f1 = metrics.get("val/touch_f1")
    if touch_f1 is not None and touch_f1 > best["touch_f1"]:
        best["touch_f1"] = touch_f1
        save_checkpoint(output_dir / "best_touch.pt", model, optimizer, epoch, metrics, scaler, scheduler, dict(cfg))
    state_f1 = metrics.get("val/state_f1_macro")
    if state_f1 is not None and state_f1 > best["state_f1"]:
        best["state_f1"] = state_f1
        save_checkpoint(output_dir / "best_state.pt", model, optimizer, epoch, metrics, scaler, scheduler, dict(cfg))
    xy_pix = metrics.get("val/xy_pix")
    if xy_pix is not None and xy_pix < best["xy_pix"]:
        best["xy_pix"] = xy_pix
        save_checkpoint(output_dir / "best_xy.pt", model, optimizer, epoch, metrics, scaler, scheduler, dict(cfg))


def _write_logs(writer: Any, csv_path: Path, epoch: int, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        writer.add_scalar(key, value, epoch)
    row = {"epoch": epoch, **metrics}
    exists = csv_path.exists()
    fieldnames = list(row.keys())
    if exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            old_fields = next(csv.reader(f), [])
        fieldnames = list(dict.fromkeys([*old_fields, *fieldnames]))
    rows = []
    if exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append({k: row.get(k, "") for k in fieldnames})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class _NullSummaryWriter:
    def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


def _build_summary_writer(cfg: Any, output_dir: Path) -> Any:
    if not bool(getattr(cfg.training, "tensorboard", False)):
        return _NullSummaryWriter()
    # TensorBoard can import TensorFlow in this environment, which emits noisy
    # XLA factory registration messages. Import lazily after setting the log
    # level so normal training output stays readable.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(output_dir / "tb"))


def _copy_manifest(cfg: Any, output_dir: Path) -> None:
    if not bool(getattr(cfg.training, "copy_manifest", True)):
        return
    manifest = Path(cfg.data.manifest_path)
    if manifest.exists():
        shutil.copy2(manifest, output_dir / manifest.name)


def _check_coordinate_mode(cfg: Any) -> None:
    if bool(getattr(cfg.model, "train_location", True)) and str(getattr(cfg.data.labels, "coordinate_mode", "")) == "auto":
        raise ValueError(
            "coordinate_mode=auto is audit-only for location training. "
            "Set data.labels.coordinate_mode to normalized, image_pixels, surface_pixels, or surface_to_image."
        )


def _update_sampled_counts(counts: Counter[str], batch: dict[str, Any]) -> None:
    counts["frames"] += int(batch["image"].shape[0])
    counts["has_labels"] += int(batch["has_labels"].sum().detach().cpu())
    counts["valid_xy"] += int(batch["valid_xy"].sum().detach().cpu())
    counts["has_touch"] += int(batch["has_touch"].sum().detach().cpu())
    states = batch["state"].detach().cpu()
    has_state = batch["has_state"].detach().cpu().bool()
    for state_id, name in [(0, "background"), (1, "hover"), (2, "touch")]:
        counts[f"state_{name}"] += int(((states == state_id) & has_state).sum())
    counts["state_unlabeled"] += int((~has_state).sum())
