# src/train_phase0.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset_core import InkMARCIndexedDataset
from eval.inference_latency import resolve_ckpt_path, run_loader_latency
from models.model_v0 import InkMARCPhase0MLP, ModelV0Config

import numpy as np


def collate_fn(batch):
    corners = torch.from_numpy(
        np.stack([b["corners"] for b in batch], axis=0)
    ).float()   # (B,4,2)

    labels = torch.from_numpy(
        np.stack([b["labels"] for b in batch], axis=0)
    ).float()   # (B,5)

    mask = torch.from_numpy(
        np.stack([b["label_mask"] for b in batch], axis=0)
    ).float()   # (B,5)

    return {
        "corners": corners,
        "labels": labels,
        "label_mask": mask,
    }

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    sums = {}
    n = 0
    for batch in loader:
        corners = batch["corners"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["label_mask"].to(device)

        logits = model(corners)
        loss, metrics = model.compute_loss_and_metrics(logits, labels, mask)

        bs = corners.shape[0]
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v * bs
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/phase0")
    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation and latency, then exit")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"], help="Split to evaluate in eval-only mode")
    ap.add_argument("--ckpt", default="", help="Checkpoint to evaluate (default: best, then latest epoch)")
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit eval-only to first N samples")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured eval passes for latency")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup eval passes for latency")
    ap.add_argument("--amp", action="store_true", help="Use CUDA AMP for eval-only latency")
    args = ap.parse_args()

    device = torch.device(args.device)

    cfg = ModelV0Config()
    model = InkMARCPhase0MLP(cfg).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        eval_ds = InkMARCIndexedDataset(
            args.dataset_root, index_version=args.version, split=args.eval_split, require_labels=True
        )
        eval_ds = _maybe_subset(eval_ds, args.eval_max_samples)
        eval_loader = DataLoader(
            eval_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
        )
        ckpt_path = resolve_ckpt_path(args.ckpt, save_dir)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        metrics = evaluate(model, eval_loader, device)
        print(f"Evaluated {ckpt_path} on split={args.eval_split}")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        use_amp = bool(args.amp and device.type == "cuda")
        def _infer_batch(batch):
            corners = batch["corners"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(corners)

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
        return

    # Datasets (require_labels=True ensures we only train on usable supervision)
    train_ds = InkMARCIndexedDataset(args.dataset_root, index_version=args.version, split="train", require_labels=True)
    val_ds = InkMARCIndexedDataset(args.dataset_root, index_version=args.version, split="val", require_labels=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Save config for later writing/repro
    (save_dir / "config.json").write_text(
        __import__("json").dumps({"model": "model_v0", **asdict(cfg), "args": vars(args)}, indent=2)
    )

    print("Phase 0 baseline training")
    print(f"  device: {device}")
    print(f"  train samples: {len(train_ds)}  val samples: {len(val_ds)}")
    print(f"  save_dir: {save_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}

        for batch in train_loader:
            corners = batch["corners"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["label_mask"].to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(corners)
            loss, metrics = model.compute_loss_and_metrics(logits, labels, mask)
            loss.backward()
            optim.step()

            bs = corners.shape[0]
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v * bs

        train_metrics = {k: v / len(train_ds) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} touch_acc={train_metrics['touch_acc']:.4f} reg_rmse={train_metrics['reg_rmse']:.4f} | "
            f"val loss={val_metrics.get('loss', float('nan')):.4f} touch_acc={val_metrics.get('touch_acc', float('nan')):.4f} reg_rmse={val_metrics.get('reg_rmse', float('nan')):.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "cfg": asdict(cfg),
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / f"checkpoint_epoch{epoch:02d}.pt")

    # Close cached H5 handles
    _close_dataset(train_ds)
    _close_dataset(val_ds)


if __name__ == "__main__":
    main()
