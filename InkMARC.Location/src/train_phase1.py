from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset_core import InkMARCIndexedDataset
from data.transforms.phase1_transform import Phase1Transform
from models.model_v1 import ResNet18MultiTask


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    touch = torch.stack([b["touch"] for b in batch], dim=0)
    touch_mask = torch.stack([b["touch_mask"] for b in batch], dim=0)
    xy = torch.stack([b["xy"] for b in batch], dim=0)
    xy_mask = torch.stack([b["xy_mask"] for b in batch], dim=0)
    meta = [b.get("meta", {}) for b in batch]
    return {
        "image": images,
        "touch": touch,
        "touch_mask": touch_mask,
        "xy": xy,
        "xy_mask": xy_mask,
        "meta": meta,
    }


def _masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    den = mask.sum().clamp_min(1.0)
    return (bce * mask).sum() / den


def _masked_smoothl1_loss(pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = torch.nn.functional.smooth_l1_loss(pred, targets, reduction="none")  # (B,2)
    per = per.mean(dim=1)  # (B,)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den


@torch.no_grad()
def evaluate(
    model: ResNet18MultiTask,
    loader: DataLoader,
    device: torch.device,
    out_w: int,
    out_h: int,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        touch = batch["touch"].to(device)
        touch_mask = batch["touch_mask"].to(device)
        xy = batch["xy"].to(device)
        xy_mask = batch["xy_mask"].to(device)

        out = model(images)
        touch_logit = out["touch_logit"].squeeze(1)
        xy_pred = out["xy"]

        touch_loss = _masked_bce_loss(touch_logit, touch, touch_mask)
        xy_loss = _masked_smoothl1_loss(xy_pred, xy, xy_mask)
        loss = touch_loss + xy_loss

        touch_prob = torch.sigmoid(touch_logit)
        touch_pred = (touch_prob >= 0.5).float()

        touch_den = touch_mask.sum().clamp_min(1.0)
        touch_acc = ((touch_pred == touch).float() * touch_mask).sum() / touch_den

        tp = ((touch_pred == 1.0) & (touch == 1.0) & (touch_mask == 1.0)).sum().float()
        fp = ((touch_pred == 1.0) & (touch == 0.0) & (touch_mask == 1.0)).sum().float()
        fn = ((touch_pred == 0.0) & (touch == 1.0) & (touch_mask == 1.0)).sum().float()
        denom = (2 * tp + fp + fn).clamp_min(1.0)
        touch_f1 = (2 * tp) / denom

        abs_err = (xy_pred - xy).abs()
        mae = abs_err.mean(dim=1)
        xy_den = xy_mask.sum().clamp_min(1.0)
        xy_mae = (mae * xy_mask).sum() / xy_den

        scale = torch.tensor([out_w, out_h], device=device, dtype=abs_err.dtype)
        abs_err_px = (abs_err * scale).mean(dim=1)
        xy_mae_px = (abs_err_px * xy_mask).sum() / xy_den

        bs = images.shape[0]
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item()) * bs
        sums["touch_loss"] = sums.get("touch_loss", 0.0) + float(touch_loss.item()) * bs
        sums["xy_loss"] = sums.get("xy_loss", 0.0) + float(xy_loss.item()) * bs
        sums["touch_acc"] = sums.get("touch_acc", 0.0) + float(touch_acc.item()) * bs
        sums["touch_f1"] = sums.get("touch_f1", 0.0) + float(touch_f1.item()) * bs
        sums["xy_mae"] = sums.get("xy_mae", 0.0) + float(xy_mae.item()) * bs
        sums["xy_mae_px"] = sums.get("xy_mae_px", 0.0) + float(xy_mae_px.item()) * bs
        n += bs

    if n == 0:
        return {}
    return {k: v / n for k, v in sums.items()}


def _maybe_subset(ds, n: int):
    if n <= 0 or n >= len(ds):
        return ds
    return Subset(ds, list(range(n)))

def _close_dataset(ds) -> None:
    if hasattr(ds, "close"):
        ds.close()
        return
    if isinstance(ds, Subset):
        inner = getattr(ds, "dataset", None)
        if inner is not None and hasattr(inner, "close"):
            inner.close()

def _resolve_ckpt_path(ckpt_arg: str, save_dir: Path) -> Path:
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
    else:
        best = save_dir / "checkpoint_best.pt"
        if best.exists():
            ckpt_path = best
        else:
            epoch_ckpts = sorted(save_dir.glob("checkpoint_epoch*.pt"))
            if not epoch_ckpts:
                raise FileNotFoundError(f"No checkpoints found in {save_dir}")
            ckpt_path = epoch_ckpts[-1]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _percentile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = p * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


@torch.inference_mode()
def run_latency_report(
    model: ResNet18MultiTask,
    loader: DataLoader,
    device: torch.device,
    runs: int,
    warmup_runs: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    image_ms = []
    batch_ms = []

    total_runs = max(1, int(runs)) + max(0, int(warmup_runs))
    for ri in range(total_runs):
        is_warmup = ri < warmup_runs
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if is_warmup:
                continue
            bs = images.shape[0]
            batch_ms.append(float(dt_ms))
            image_ms.append(float(dt_ms) / max(bs, 1))

    if not image_ms:
        return {}

    image_ms_sorted = sorted(image_ms)
    batch_ms_sorted = sorted(batch_ms)
    n = len(image_ms_sorted)
    mean_image = sum(image_ms_sorted) / n
    var_image = sum((x - mean_image) ** 2 for x in image_ms_sorted) / n

    return {
        "num_batches": float(len(batch_ms_sorted)),
        "num_images_profiled": float(n),
        "image_ms_mean": mean_image,
        "image_ms_std": var_image ** 0.5,
        "image_ms_min": image_ms_sorted[0],
        "image_ms_p50": _percentile(image_ms_sorted, 0.50),
        "image_ms_p90": _percentile(image_ms_sorted, 0.90),
        "image_ms_p95": _percentile(image_ms_sorted, 0.95),
        "image_ms_max": image_ms_sorted[-1],
        "batch_ms_mean": sum(batch_ms_sorted) / max(len(batch_ms_sorted), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/phase1")
    ap.add_argument("--ckpt", default="", help="Checkpoint to evaluate (default: save-dir/checkpoint_best.pt)")
    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--eval-test", action="store_true", help="Load checkpoint and run eval + latency report")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"], help="Split to evaluate with --eval-test")
    ap.add_argument("--test-batch-size", type=int, default=0, help="Override batch size for --eval-test")
    ap.add_argument("--test-num-workers", type=int, default=-1, help="Override workers for --eval-test")
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit --eval-test to first N test samples")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured inference passes over test set")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup passes over test set (not measured)")
    ap.add_argument("--amp", action="store_true", help="Use CUDA AMP in latency profiling")
    ap.add_argument(
        "--overfit-small",
        nargs="?",
        const=64,
        type=int,
        default=0,
        help="If set, restrict to first N samples (default 64) for quick overfit.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_test:
        test_tf = Phase1Transform(
            dataset_root=Path(args.dataset_root),
            index_version=args.version,
            split=args.eval_split,
            out_w=args.out_w,
            out_h=args.out_h,
            margin_frac=args.margin_frac,
            normalize=True,
        )
        test_ds = InkMARCIndexedDataset(
            args.dataset_root, index_version=args.version, split=args.eval_split, require_labels=True, transform=test_tf
        )
        eval_cap = args.eval_max_samples if args.eval_max_samples > 0 else args.overfit_small
        if eval_cap > 0:
            test_ds = _maybe_subset(test_ds, eval_cap)

        test_loader = DataLoader(
            test_ds,
            batch_size=(args.test_batch_size if args.test_batch_size > 0 else args.batch_size),
            shuffle=False,
            num_workers=(args.test_num_workers if args.test_num_workers >= 0 else args.num_workers),
            pin_memory=True,
            collate_fn=collate_fn,
        )

        model = ResNet18MultiTask().to(device)
        ckpt_path = _resolve_ckpt_path(args.ckpt, save_dir)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        print("Phase 1 evaluation + latency")
        print(f"  device: {device}")
        print(f"  split: {args.eval_split}")
        print(f"  samples: {len(test_ds)}")
        print(f"  checkpoint: {ckpt_path}")

        test_metrics = evaluate(model, test_loader, device, args.out_w, args.out_h)
        print(
            f"test metrics | "
            f"loss={test_metrics.get('loss', float('nan')):.4f} "
            f"touch_acc={test_metrics.get('touch_acc', float('nan')):.4f} "
            f"touch_f1={test_metrics.get('touch_f1', float('nan')):.4f} "
            f"xy_mae={test_metrics.get('xy_mae', float('nan')):.4f} "
            f"xy_mae_px={test_metrics.get('xy_mae_px', float('nan')):.2f}"
        )

        latency = run_latency_report(
            model=model,
            loader=test_loader,
            device=device,
            runs=args.latency_runs,
            warmup_runs=args.latency_warmup,
            use_amp=bool(args.amp and device.type == "cuda"),
        )
        print(
            f"latency per image (ms) | "
            f"mean={latency.get('image_ms_mean', float('nan')):.3f} "
            f"std={latency.get('image_ms_std', float('nan')):.3f} "
            f"p50={latency.get('image_ms_p50', float('nan')):.3f} "
            f"p90={latency.get('image_ms_p90', float('nan')):.3f} "
            f"p95={latency.get('image_ms_p95', float('nan')):.3f} "
            f"min={latency.get('image_ms_min', float('nan')):.3f} "
            f"max={latency.get('image_ms_max', float('nan')):.3f}"
        )
        print(
            f"latency summary | "
            f"batch_mean_ms={latency.get('batch_ms_mean', float('nan')):.3f} "
            f"images={int(latency.get('num_images_profiled', 0.0))} "
            f"batches={int(latency.get('num_batches', 0.0))}"
        )

        _close_dataset(test_ds)
        test_tf.close()
        return

    train_tf = Phase1Transform(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="train",
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
    )
    val_tf = Phase1Transform(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="val",
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
    )

    train_ds = InkMARCIndexedDataset(
        args.dataset_root, index_version=args.version, split="train", require_labels=True, transform=train_tf
    )
    val_ds = InkMARCIndexedDataset(
        args.dataset_root, index_version=args.version, split="val", require_labels=True, transform=val_tf
    )

    train_ds = _maybe_subset(train_ds, args.overfit_small)
    val_ds = _maybe_subset(val_ds, args.overfit_small)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = ResNet18MultiTask().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    (save_dir / "config.json").write_text(
        __import__("json").dumps({"model": "model_v1", "args": vars(args)}, indent=2)
    )

    print("Phase 1 training (RGB + homography)")
    print(f"  device: {device}")
    print(f"  train samples: {len(train_ds)}  val samples: {len(val_ds)}")
    print(f"  save_dir: {save_dir}")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running: Dict[str, float] = {}

        for batch in train_loader:
            images = batch["image"].to(device)
            touch = batch["touch"].to(device)
            touch_mask = batch["touch_mask"].to(device)
            xy = batch["xy"].to(device)
            xy_mask = batch["xy_mask"].to(device)

            out = model(images)
            touch_logit = out["touch_logit"].squeeze(1)
            xy_pred = out["xy"]

            touch_loss = _masked_bce_loss(touch_logit, touch, touch_mask)
            xy_loss = _masked_smoothl1_loss(xy_pred, xy, xy_mask)
            loss = touch_loss + xy_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            touch_prob = torch.sigmoid(touch_logit)
            touch_pred = (touch_prob >= 0.5).float()
            touch_den = touch_mask.sum().clamp_min(1.0)
            touch_acc = ((touch_pred == touch).float() * touch_mask).sum() / touch_den

            tp = ((touch_pred == 1.0) & (touch == 1.0) & (touch_mask == 1.0)).sum().float()
            fp = ((touch_pred == 1.0) & (touch == 0.0) & (touch_mask == 1.0)).sum().float()
            fn = ((touch_pred == 0.0) & (touch == 1.0) & (touch_mask == 1.0)).sum().float()
            denom = (2 * tp + fp + fn).clamp_min(1.0)
            touch_f1 = (2 * tp) / denom

            abs_err = (xy_pred - xy).abs()
            mae = abs_err.mean(dim=1)
            xy_den = xy_mask.sum().clamp_min(1.0)
            xy_mae = (mae * xy_mask).sum() / xy_den

            scale = torch.tensor([args.out_w, args.out_h], device=device, dtype=abs_err.dtype)
            abs_err_px = (abs_err * scale).mean(dim=1)
            xy_mae_px = (abs_err_px * xy_mask).sum() / xy_den

            bs = images.shape[0]
            running["loss"] = running.get("loss", 0.0) + float(loss.item()) * bs
            running["touch_loss"] = running.get("touch_loss", 0.0) + float(touch_loss.item()) * bs
            running["xy_loss"] = running.get("xy_loss", 0.0) + float(xy_loss.item()) * bs
            running["touch_acc"] = running.get("touch_acc", 0.0) + float(touch_acc.item()) * bs
            running["touch_f1"] = running.get("touch_f1", 0.0) + float(touch_f1.item()) * bs
            running["xy_mae"] = running.get("xy_mae", 0.0) + float(xy_mae.item()) * bs
            running["xy_mae_px"] = running.get("xy_mae_px", 0.0) + float(xy_mae_px.item()) * bs

        train_metrics = {k: v / len(train_ds) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, args.out_w, args.out_h)

        print(
            f"epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} touch_acc={train_metrics['touch_acc']:.4f} "
            f"touch_f1={train_metrics['touch_f1']:.4f} xy_mae={train_metrics['xy_mae']:.4f} "
            f"xy_mae_px={train_metrics['xy_mae_px']:.2f} | "
            f"val loss={val_metrics.get('loss', float('nan')):.4f} "
            f"touch_acc={val_metrics.get('touch_acc', float('nan')):.4f} "
            f"touch_f1={val_metrics.get('touch_f1', float('nan')):.4f} "
            f"xy_mae={val_metrics.get('xy_mae', float('nan')):.4f} "
            f"xy_mae_px={val_metrics.get('xy_mae_px', float('nan')):.2f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / f"checkpoint_epoch{epoch:02d}.pt")
        v = val_metrics.get("loss", None)
        if v is not None and v < best_val:
            best_val = float(v)
            torch.save(ckpt, save_dir / "checkpoint_best.pt")

    _close_dataset(train_ds)
    _close_dataset(val_ds)
    train_tf.close()
    val_tf.close()


if __name__ == "__main__":
    main()
