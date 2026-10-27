# train_phase1a.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset_core import InkMARCIndexedDataset
from data.transforms.phase1a_transform_v1 import Phase1aTransform_v1
from eval.inference_latency import resolve_ckpt_path, run_loader_latency
from models.model_v3 import MobileNetV3_RawPlusCorners_XY


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    corners = torch.stack([b["corners"] for b in batch], dim=0)
    xy = torch.stack([b["xy"] for b in batch], dim=0)
    xy_mask = torch.stack([b["xy_mask"] for b in batch], dim=0)
    meta = [b.get("meta", {}) for b in batch]
    return {"image": images, "corners": corners, "xy": xy, "xy_mask": xy_mask, "meta": meta}


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


def _masked_smoothl1_loss(pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = torch.nn.functional.smooth_l1_loss(pred, targets, reduction="none")  # (B,2)
    per = per.mean(dim=1)  # (B,)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den


@torch.no_grad()
def evaluate(
    model: MobileNetV3_RawPlusCorners_XY,
    loader: DataLoader,
    device: torch.device,
    out_w: int,
    out_h: int,
) -> Dict[str, float]:
    model.eval()

    sums: Dict[str, float] = {}
    n = 0
    nde_vals: List[float] = []

    for batch in loader:
        images = batch["image"].to(device)
        corners = batch["corners"].to(device)
        xy = batch["xy"].to(device)
        xy_mask = batch["xy_mask"].to(device)

        out = model(images, corners)
        xy_pred = out["xy"]

        loss = _masked_smoothl1_loss(xy_pred, xy, xy_mask)

        abs_err = (xy_pred - xy).abs()
        mae = abs_err.mean(dim=1)  # (B,)
        den = xy_mask.sum().clamp_min(1.0)
        xy_mae = (mae * xy_mask).sum() / den

        scale = torch.tensor([out_w, out_h], device=device, dtype=abs_err.dtype)
        abs_err_px = (abs_err * scale).mean(dim=1)
        xy_mae_px = (abs_err_px * xy_mask).sum() / den

        eu = torch.sqrt(((xy_pred - xy) ** 2).sum(dim=1))  # (B,)
        for v, m in zip(eu.tolist(), xy_mask.tolist()):
            if m > 0.5:
                nde_vals.append(float(v))

        bs = images.shape[0]
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item()) * bs
        sums["xy_mae"] = sums.get("xy_mae", 0.0) + float(xy_mae.item()) * bs
        sums["xy_mae_px"] = sums.get("xy_mae_px", 0.0) + float(xy_mae_px.item()) * bs
        n += bs

    if n == 0:
        return {}

    outm = {k: v / n for k, v in sums.items()}

    if nde_vals:
        nde_vals.sort()
        mid = len(nde_vals) // 2
        outm["nde_median"] = nde_vals[mid] if (len(nde_vals) % 2 == 1) else 0.5 * (nde_vals[mid - 1] + nde_vals[mid])
    else:
        outm["nde_median"] = float("nan")

    return outm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/phase1a_raw_plus_corners_xy")
    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation and latency, then exit")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"], help="Split to evaluate in eval-only mode")
    ap.add_argument("--ckpt", default="", help="Checkpoint to evaluate (default: best, then latest epoch)")
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit eval-only to first N samples")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured eval passes for latency")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup eval passes for latency")
    ap.add_argument("--amp", action="store_true", help="Use CUDA AMP for eval-only latency")
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--overfit-small", nargs="?", const=64, type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)

    train_tf = Phase1aTransform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="train",
        out_w=args.out_w,
        out_h=args.out_h,
        normalize=True,
    )
    val_tf = Phase1aTransform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="val",
        out_w=args.out_w,
        out_h=args.out_h,
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
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    model = MobileNetV3_RawPlusCorners_XY().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        eval_tf = Phase1aTransform_v1(
            dataset_root=Path(args.dataset_root),
            index_version=args.version,
            split=args.eval_split,
            out_w=args.out_w,
            out_h=args.out_h,
            normalize=True,
        )
        eval_ds = InkMARCIndexedDataset(
            args.dataset_root, index_version=args.version, split=args.eval_split, require_labels=True, transform=eval_tf
        )
        eval_cap = args.eval_max_samples if args.eval_max_samples > 0 else args.overfit_small
        if eval_cap > 0:
            eval_ds = _maybe_subset(eval_ds, eval_cap)
        eval_loader = DataLoader(
            eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
        )

        ckpt_path = resolve_ckpt_path(args.ckpt, save_dir)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        metrics = evaluate(model, eval_loader, device, args.out_w, args.out_h)
        print(f"Evaluated {ckpt_path} on split={args.eval_split}")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        use_amp = bool(args.amp and device.type == "cuda")

        def _infer_batch(batch):
            images = batch["image"].to(device, non_blocking=True)
            corners = batch["corners"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(images, corners)

        latency = run_loader_latency(
            loader=eval_loader,
            device=device,
            runs=args.latency_runs,
            warmup_runs=args.latency_warmup,
            infer_batch=_infer_batch,
        )
        print(
            f"latency per image (ms) | mean={latency.get('image_ms_mean', float('nan')):.3f} "
            f"std={latency.get('image_ms_std', float('nan')):.3f} p50={latency.get('image_ms_p50', float('nan')):.3f} "
            f"p90={latency.get('image_ms_p90', float('nan')):.3f} p95={latency.get('image_ms_p95', float('nan')):.3f} "
            f"min={latency.get('image_ms_min', float('nan')):.3f} max={latency.get('image_ms_max', float('nan')):.3f}"
        )
        _close_dataset(eval_ds)
        eval_tf.close()
        return

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running: Dict[str, float] = {}

        for batch in train_loader:
            images = batch["image"].to(device)
            corners = batch["corners"].to(device)
            xy = batch["xy"].to(device)
            xy_mask = batch["xy_mask"].to(device)

            out = model(images, corners)
            xy_pred = out["xy"]

            loss = _masked_smoothl1_loss(xy_pred, xy, xy_mask)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            abs_err = (xy_pred - xy).abs()
            mae = abs_err.mean(dim=1)
            den = xy_mask.sum().clamp_min(1.0)
            xy_mae = (mae * xy_mask).sum() / den

            scale = torch.tensor([args.out_w, args.out_h], device=device, dtype=abs_err.dtype)
            abs_err_px = (abs_err * scale).mean(dim=1)
            xy_mae_px = (abs_err_px * xy_mask).sum() / den

            bs = images.shape[0]
            running["loss"] = running.get("loss", 0.0) + float(loss.item()) * bs
            running["xy_mae"] = running.get("xy_mae", 0.0) + float(xy_mae.item()) * bs
            running["xy_mae_px"] = running.get("xy_mae_px", 0.0) + float(xy_mae_px.item()) * bs

        train_metrics = {k: v / len(train_ds) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, args.out_w, args.out_h)

        print(
            f"epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} xy_mae={train_metrics['xy_mae']:.4f} xy_mae_px={train_metrics['xy_mae_px']:.2f} | "
            f"val loss={val_metrics.get('loss', float('nan')):.4f} nde_med={val_metrics.get('nde_median', float('nan')):.4f} "
            f"xy_mae={val_metrics.get('xy_mae', float('nan')):.4f} xy_mae_px={val_metrics.get('xy_mae_px', float('nan')):.2f}"
        )

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optim_state": optim.state_dict(), "args": vars(args)}
        torch.save(ckpt, save_dir / f"checkpoint_epoch{epoch:02d}.pt")

        v = val_metrics.get("xy_mae_px", None)
        if v is not None and v < best_val:
            best_val = float(v)
            torch.save(ckpt, save_dir / "checkpoint_best.pt")

    _close_dataset(train_ds)
    _close_dataset(val_ds)
    train_tf.close()
    val_tf.close()


if __name__ == "__main__":
    main()
