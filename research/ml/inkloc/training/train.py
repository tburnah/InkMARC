from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from ml.inkloc.data.dataset import ManifestInteractionDataset
from ml.inkloc.data.inkmarc import InkMARCCombinedDataset
from ml.inkloc.data.synthetic import SyntheticInteractionDataset
from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.training.calibration import evaluate_calibration
from ml.inkloc.training.forward import model_forward
from ml.inkloc.training.losses import LocTouchLoss
from ml.inkloc.training.metrics import compute_metrics
from ml.inkloc.utils.config import load_config
from ml.inkloc.utils.seed import set_seed


def _maybe_subset(ds, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(ds):
        return ds
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:max_samples].tolist()
    return Subset(ds, idx)


def build_dataset(cfg, split: str = "train"):
    data_cfg = cfg.get("data", {})
    if data_cfg.get("type") == "inkmarc" or data_cfg.get("dataset_root"):
        temporal_cfg = cfg.get("temporal", {})
        ds = InkMARCCombinedDataset(
            dataset_root=data_cfg.get("dataset_root", "/data/LocationData"),
            index_version=str(data_cfg.get("index_version", "v1")),
            split=str(data_cfg.get(f"{split}_split", split)),
            inkmarc_src=data_cfg.get("inkmarc_src", "/home/t/tburnah/inkmarc/src"),
            input_height=cfg["model"]["input_height"],
            input_width=cfg["model"]["input_width"],
            margin_frac=float(data_cfg.get("margin_frac", 0.10)),
            heatmap_height=cfg["model"]["heatmap_height"],
            heatmap_width=cfg["model"]["heatmap_width"],
            heatmap_sigma=float(data_cfg.get("heatmap_sigma", 1.5)),
            touch_pressure_threshold=float(data_cfg.get("touch_pressure_threshold", 0.0)),
            require_labels=bool(data_cfg.get("require_labels", True)),
            video_cache_size=int(data_cfg.get("video_cache_size", 16)),
            temporal_enabled=bool(temporal_cfg.get("enabled", False)),
            temporal_input_mode=str(temporal_cfg.get("input_mode", "motion_map")),
            motion_channels=int(temporal_cfg.get("motion_channels", temporal_cfg.get("num_motion_maps", 1))),
            motion_type=str(temporal_cfg.get("motion_type", "abs_gray_diff")),
        )
        training_cfg = cfg.get("training", {})
        cap_key = "train_max_samples" if split == "train" else "val_max_samples"
        return _maybe_subset(ds, int(training_cfg.get(cap_key, 0)), int(cfg.get("seed", 7)) + (0 if split == "train" else 1))
    if data_cfg.get("manifest"):
        return ManifestInteractionDataset(data_cfg["manifest"], data_cfg.get("image_root"), cfg["model"]["input_height"], cfg["model"]["input_width"])
    ds = SyntheticInteractionDataset(size=int(data_cfg.get("synthetic_size", 64)), height=cfg["model"]["input_height"], width=cfg["model"]["input_width"])
    return ds


def move_batch(batch, device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _load_hard_records(path: str | None) -> dict[int, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    records = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[int(rec["dataset_pos"])] = rec
    return records


def build_sampler(ds, cfg: dict) -> WeightedRandomSampler | None:
    train_cfg = cfg.get("training", {})
    mode = str(train_cfg.get("sampler", "normal"))
    if mode == "normal":
        return None
    hard_records = _load_hard_records(train_cfg.get("hard_examples_path"))
    if mode in {"state_balanced", "hard_weighted", "touch_as_hover_weighted", "touch_as_hover_capped"} and not hard_records:
        raise ValueError(f"sampler={mode} requires training.hard_examples_path from mine_hard_examples.py")
    weights_cfg = train_cfg.get("sampler_weights", {})
    base_by_state = {
        0: float(weights_cfg.get("background", 1.0)),
        1: float(weights_cfg.get("hover", 1.0)),
        2: float(weights_cfg.get("touch", 1.0)),
    }
    hard_by_type = {
        "touch_as_hover": float(weights_cfg.get("touch_as_hover", 4.0)),
        "hover_as_touch": float(weights_cfg.get("hover_as_touch", 4.0)),
        "background_as_valid": float(weights_cfg.get("background_as_valid", 2.0)),
        "high_xy_error": float(weights_cfg.get("high_xy_error", 2.0)),
        "low_conf_touch": float(weights_cfg.get("low_conf_touch", 2.0)),
    }
    weights = torch.ones(len(ds), dtype=torch.double)
    known_state_counts = {0: 0, 1: 0, 2: 0}
    for rec in hard_records.values():
        state = int(rec["true_state"])
        if state in known_state_counts:
            known_state_counts[state] += 1
    if mode == "state_balanced":
        target = {s: 1.0 / max(c, 1) for s, c in known_state_counts.items()}
        for pos, rec in hard_records.items():
            if 0 <= pos < len(weights):
                weights[pos] = target.get(int(rec["true_state"]), 1.0)
    elif mode == "hard_weighted":
        for pos, rec in hard_records.items():
            if 0 <= pos < len(weights):
                weights[pos] = base_by_state.get(int(rec["true_state"]), 1.0) * hard_by_type.get(str(rec["error_type"]), 1.0)
    elif mode in {"touch_as_hover_weighted", "touch_as_hover_capped"}:
        hard_positions = [
            pos for pos, rec in hard_records.items()
            if str(rec["error_type"]) == "touch_as_hover" and 0 <= pos < len(weights)
        ]
        if mode == "touch_as_hover_capped":
            cap_fraction = float(train_cfg.get("hard_cap_fraction", 0.12))
            cap = max(1, int(len(weights) * cap_fraction))
            hard_positions = hard_positions[:cap]
        multiplier = float(weights_cfg.get("touch_as_hover", 2.0))
        for pos in hard_positions:
            rec = hard_records[pos]
            weights[pos] = base_by_state.get(int(rec["true_state"]), 1.0) * multiplier
    else:
        raise ValueError(f"Unknown sampler mode: {mode}")
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 7)))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def batch_composition(batch: dict, hard_by_dataset_index: dict[int, str]) -> str:
    state = batch["state"].detach().cpu().long()
    total = max(1, int(state.numel()))
    bg = int((state == 0).sum().item())
    hover = int((state == 1).sum().item())
    touch = int((state == 2).sum().item())
    tah = 0
    hat = 0
    if "dataset_index" in batch:
        for idx in batch["dataset_index"].detach().cpu().tolist():
            err = hard_by_dataset_index.get(int(idx))
            if err == "touch_as_hover":
                tah += 1
            elif err == "hover_as_touch":
                hat += 1
    return (
        f"comp_bg={bg / total:.2f} comp_hover={hover / total:.2f} comp_touch={touch / total:.2f} "
        f"comp_touch_as_hover={tah / total:.2f} comp_hover_as_touch={hat / total:.2f}"
    )


@torch.no_grad()
def evaluate(model, loss_fn, loader, device: torch.device, max_batches: int = 0):
    model.eval()
    totals = {}
    metric_totals = {}
    seen_batches = 0
    seen = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model_forward(model, batch)
        losses = loss_fn(outputs, batch)
        metrics = compute_metrics(outputs, batch, pixel_size=(480, 320))
        bs = int(batch["image"].shape[0])
        seen += bs
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * bs
        for k, v in metrics.items():
            if v == v:
                metric_totals[k] = metric_totals.get(k, 0.0) + float(v) * bs
        seen_batches += 1
        if max_batches > 0 and seen_batches >= max_batches:
            break
    den = max(seen, 1)
    losses_out = {k: v / den for k, v in totals.items()}
    metrics_out = {k: v / den for k, v in metric_totals.items()}
    return losses_out, metrics_out, seen_batches


def format_metrics(prefix: str, losses, metrics) -> str:
    touch_fixed = metrics.get("touch_at_0_5", {})
    touch_best = metrics.get("touch_best", {})
    pieces = [
        f"{prefix}_loss={losses.get('loss', float('nan')):.4f}",
        f"state_acc={metrics.get('state_accuracy', float('nan')):.3f}",
        f"valid_acc={metrics.get('valid_accuracy', float('nan')):.3f}",
        f"touch_f1={touch_fixed.get('f1', float('nan')):.3f}",
        f"touch_best_f1={touch_best.get('f1', float('nan')):.3f}",
        f"touch_best_th={touch_best.get('threshold', float('nan')):.2f}",
        f"xy_px={metrics.get('xy_mean_pixel_error', float('nan')):.1f}",
    ]
    return " ".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/loctouch_lite.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--calibration-out", default="ml/artifacts/calibration/loctouch_lite_report.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    set_seed(int(cfg.get("seed", 7)))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model = LocTouchNetLite.from_config(cfg).to(device)
    loss_fn = LocTouchLoss.from_config(cfg)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint.get("step", 0))
    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="eval")
    print(f"dataset train_size={len(train_ds)} val_size={len(val_ds)}")
    num_workers = int(train_cfg.get("num_workers", 0))
    sampler = build_sampler(train_ds, cfg)
    hard_by_dataset_index = {
        int(rec["dataset_index"]): str(rec["error_type"])
        for rec in _load_hard_records(train_cfg.get("hard_examples_path")).values()
    }
    loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    lr = float(args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg["weight_decay"]))
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "ml/artifacts/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        report = evaluate_calibration(model, loss_fn, val_loader, device, max_batches=int(train_cfg.get("val_max_batches", 0)))
        print(format_metrics("val", report["losses"], report))
        return 0

    steps = start_step
    best_val = float("inf")
    best_touch_f1 = float("-inf")
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        existing = torch.load(best_path, map_location="cpu", weights_only=False)
        best_val = float(existing.get("val_loss", best_val))
    best_touch_path = ckpt_dir / "best_touch_f1.pt"
    if best_touch_path.exists():
        existing = torch.load(best_touch_path, map_location="cpu", weights_only=False)
        existing_report = existing.get("calibration", {})
        best_touch_f1 = float(existing_report.get("touch_best", {}).get("f1", best_touch_f1))
    running_loss = 0.0
    running_seen = 0
    eval_every = int(train_cfg.get("eval_every_steps", 0))
    log_every = int(train_cfg.get("log_every", 10))
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model_forward(model, batch)
            losses = loss_fn(outputs, batch)
            if not args.dry_run:
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                opt.step()
            steps += 1
            bs = int(batch["image"].shape[0])
            running_loss += float(losses["loss"].detach().cpu()) * bs
            running_seen += bs
            if steps % log_every == 0 or args.dry_run:
                metrics = compute_metrics(outputs, batch)
                avg_loss = running_loss / max(running_seen, 1)
                print(
                    f"epoch={epoch} step={steps} train_loss={avg_loss:.4f} "
                    f"batch_loss={losses['loss'].item():.4f} state_acc={metrics['state_accuracy']:.3f} "
                    f"valid_acc={metrics['valid_accuracy']:.3f} touch_f1={metrics['touch_f1']:.3f} "
                    f"touch_best_f1={metrics['touch_best_f1']:.3f} "
                    f"xy_px={metrics['xy_mean_pixel_error']:.1f} "
                    f"{batch_composition(batch, hard_by_dataset_index)}"
                )
                running_loss = 0.0
                running_seen = 0
            if eval_every > 0 and steps % eval_every == 0:
                report = evaluate_calibration(model, loss_fn, val_loader, device, max_batches=int(train_cfg.get("val_max_batches", 0)))
                val_loss = float(report["losses"]["loss"])
                touch_f1 = float(report["touch_best"]["f1"])
                print(f"epoch={epoch} step={steps} {format_metrics('val', report['losses'], report)}")
                payload = {"model": model.state_dict(), "config": cfg, "step": steps, "val_loss": val_loss, "calibration": report}
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(payload, ckpt_dir / "best.pt")
                if touch_f1 > best_touch_f1:
                    best_touch_f1 = touch_f1
                    torch.save(payload, ckpt_dir / "best_touch_f1.pt")
                model.train()
            if args.dry_run or (args.max_steps is not None and steps >= args.max_steps):
                if args.dry_run:
                    torch.save({"model": model.state_dict(), "config": cfg, "step": steps}, ckpt_dir / "last.pt")
                    return 0
                report = evaluate_calibration(model, loss_fn, val_loader, device, max_batches=int(train_cfg.get("val_max_batches", 0)))
                print(f"final {format_metrics('val', report['losses'], report)}")
                payload = {"model": model.state_dict(), "config": cfg, "step": steps, "val_loss": report["losses"]["loss"], "calibration": report}
                torch.save(payload, ckpt_dir / "final.pt")
                torch.save(payload, ckpt_dir / "last.pt")
                return 0
    report = evaluate_calibration(model, loss_fn, val_loader, device, max_batches=int(train_cfg.get("val_max_batches", 0)))
    print(f"final {format_metrics('val', report['losses'], report)}")
    torch.save({"model": model.state_dict(), "config": cfg, "step": steps, "val_loss": report["losses"]["loss"], "calibration": report}, ckpt_dir / "final.pt")
    torch.save({"model": model.state_dict(), "config": cfg, "step": steps, "val_loss": report["losses"]["loss"], "calibration": report}, ckpt_dir / "last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
