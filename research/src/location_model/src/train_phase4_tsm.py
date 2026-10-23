from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.dataset_clip import InkMARCIndexedClipDataset
from data.transforms.phase4_transform_clip_v2 import Phase4TransformClip_v2
from eval.inference_latency import resolve_ckpt_path, run_loader_latency
from models.model_phase4_tsm import MobileNetV3_TSM_HeatmapDSNT


def collate_fn(batch):
    clip = torch.stack([b["image_clip"] for b in batch], dim=0)  # (B,T,C,H,W)
    xy = torch.stack([b["xy"] for b in batch], dim=0)
    xy_mask = torch.stack([b["xy_mask"] for b in batch], dim=0)
    heatmap = torch.stack([b["heatmap"] for b in batch], dim=0)
    heatmap_mask = torch.stack([b["heatmap_mask"] for b in batch], dim=0)
    meta = [b.get("meta", {}) for b in batch]
    return {
        "image_clip": clip,
        "xy": xy,
        "xy_mask": xy_mask,
        "heatmap": heatmap,
        "heatmap_mask": heatmap_mask,
        "meta": meta,
    }


def _maybe_subset(ds, n: int):
    if n <= 0 or n >= len(ds):
        return ds
    return Subset(ds, list(range(n)))


def _masked_smoothl1_xy(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=1)  # (B,)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den


def _masked_kl_heatmap(logits: torch.Tensor, target_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b = logits.shape[0]
    logp = F.log_softmax(logits.view(b, 1, -1), dim=-1).view_as(logits)
    per = F.kl_div(logp, target_prob, reduction="none").sum(dim=(1, 2, 3))  # (B,)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den


@torch.no_grad()
def evaluate(model, loader, device, out_w: int, out_h: int) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    nde_vals: List[float] = []

    for batch in loader:
        clip = batch["image_clip"].to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True)
        xy_mask = batch["xy_mask"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True)

        out = model(clip)
        xy_pred = out["xy"]
        logits = out["heatmap_logits"]

        loss_xy = _masked_smoothl1_xy(xy_pred, xy, xy_mask)
        loss_hm = _masked_kl_heatmap(logits, heatmap, heatmap_mask)
        loss = loss_xy + loss_hm

        abs_err = (xy_pred - xy).abs()
        mae = abs_err.mean(dim=1)
        den = xy_mask.sum().clamp_min(1.0)
        xy_mae = (mae * xy_mask).sum() / den

        scale = torch.tensor([out_w, out_h], device=device, dtype=abs_err.dtype)
        abs_err_px = (abs_err * scale).mean(dim=1)
        xy_mae_px = (abs_err_px * xy_mask).sum() / den

        eu = torch.sqrt(((xy_pred - xy) ** 2).sum(dim=1))
        for v, m in zip(eu.tolist(), xy_mask.tolist()):
            if m > 0.5:
                nde_vals.append(float(v))

        bs = clip.shape[0]
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item()) * bs
        sums["loss_xy"] = sums.get("loss_xy", 0.0) + float(loss_xy.item()) * bs
        sums["loss_hm"] = sums.get("loss_hm", 0.0) + float(loss_hm.item()) * bs
        sums["xy_mae"] = sums.get("xy_mae", 0.0) + float(xy_mae.item()) * bs
        sums["xy_mae_px"] = sums.get("xy_mae_px", 0.0) + float(xy_mae_px.item()) * bs
        n += bs

    outm = {k: v / max(n, 1) for k, v in sums.items()}
    if nde_vals:
        nde_vals.sort()
        mid = len(nde_vals) // 2
        outm["nde_median"] = nde_vals[mid] if (len(nde_vals) % 2 == 1) else 0.5 * (nde_vals[mid - 1] + nde_vals[mid])
    else:
        outm["nde_median"] = float("nan")

    return outm


def _make_loader(
    ds,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    collate_fn,
    prefetch_factor: int,
    persistent_workers: bool,
):
    # DataLoader only accepts prefetch_factor/persistent_workers if num_workers>0
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(ds, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/phase4_tsm")
    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation and latency, then exit")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"], help="Split to evaluate in eval-only mode")
    ap.add_argument("--ckpt", default="", help="Checkpoint to evaluate (default: best, then latest epoch)")
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit eval-only to first N clips")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured eval passes for latency")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup eval passes for latency")

    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)

    ap.add_argument("--hm-w", type=int, default=60)
    ap.add_argument("--hm-h", type=int, default=40)
    ap.add_argument("--hm-sigma", type=float, default=1.5)

    ap.add_argument("--clip-len", type=int, default=5)
    ap.add_argument("--tsm-fold-div", type=int, default=8)

    ap.add_argument("--overfit-small", nargs="?", const=64, type=int, default=0)

    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")

    # Progress + performance knobs
    ap.add_argument("--log-every", type=int, default=50, help="Log every N train steps")
    ap.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch_factor when num_workers>0")
    ap.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs")
    ap.add_argument("--benchmark", action="store_true", help="Enable cudnn benchmark (speed)")
    ap.add_argument("--train-eval", action="store_true", help="Also evaluate on train set each epoch (slow, for sanity)")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_tf = Phase4TransformClip_v2(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="train",
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=False,
    )
    val_tf = Phase4TransformClip_v2(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="val",
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=False,
    )
    eval_tf = Phase4TransformClip_v2(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split=args.eval_split,
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=False,
    )

    try:
        train_ds = InkMARCIndexedClipDataset(
            args.dataset_root,
            index_version=args.version,
            split="train",
            clip_len=args.clip_len,
            require_labels_last=True,
            transform=train_tf,
        )
        val_ds = InkMARCIndexedClipDataset(
            args.dataset_root,
            index_version=args.version,
            split="val",
            clip_len=args.clip_len,
            require_labels_last=True,
            transform=val_tf,
        )
        eval_ds = InkMARCIndexedClipDataset(
            args.dataset_root,
            index_version=args.version,
            split=args.eval_split,
            clip_len=args.clip_len,
            require_labels_last=True,
            transform=eval_tf,
        )

        train_ds = _maybe_subset(train_ds, args.overfit_small)
        val_ds = _maybe_subset(val_ds, args.overfit_small)
        eval_cap = args.eval_max_samples if args.eval_max_samples > 0 else args.overfit_small
        eval_ds = _maybe_subset(eval_ds, eval_cap)

        print(f"train clips: {len(train_ds)} | val clips: {len(val_ds)}")

        train_loader = _make_loader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=bool(args.persistent_workers),
        )
        val_loader = _make_loader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=bool(args.persistent_workers),
        )
        eval_loader = _make_loader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=bool(args.persistent_workers),
        )

        train_loader_eval: Optional[DataLoader] = None
        if args.train_eval:
            train_loader_eval = _make_loader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=max(0, args.num_workers // 2),
                pin_memory=True,
                collate_fn=collate_fn,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=False,
            )

        model = MobileNetV3_TSM_HeatmapDSNT(
            hm_h=args.hm_h,
            hm_w=args.hm_w,
            tsm_fold_div=args.tsm_fold_div,
        ).to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")

        if args.eval_only:
            ckpt_path = resolve_ckpt_path(args.ckpt, save_dir)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)

            metrics = evaluate(model, eval_loader, device, args.out_w, args.out_h)
            print(f"Evaluated {ckpt_path} on split={args.eval_split}")
            for k, v in metrics.items():
                print(f"{k}: {v}")

            def _infer_batch(batch):
                clip = batch["image_clip"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _ = model(clip)

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
            return

        for epoch in range(1, args.epochs + 1):
            t_epoch_start = perf_counter()
            model.train()
            running: Dict[str, float] = {}

            log_every = max(1, int(args.log_every))
            steps_per_epoch = len(train_loader)

            # timing
            t_epoch0 = perf_counter()
            t_last_log = t_epoch0
            data_time = 0.0
            compute_time = 0.0
            end_iter = perf_counter()

            for i, batch in enumerate(train_loader):
                # Time spent waiting for dataloader to yield batch
                t_got = perf_counter()
                data_time += (t_got - end_iter)

                clip = batch["image_clip"].to(device, non_blocking=True)
                xy = batch["xy"].to(device, non_blocking=True)
                xy_mask = batch["xy_mask"].to(device, non_blocking=True)
                heatmap = batch["heatmap"].to(device, non_blocking=True)
                heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)

                t_comp0 = perf_counter()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(clip)
                    xy_pred = out["xy"]
                    logits = out["heatmap_logits"]

                    loss_xy = _masked_smoothl1_xy(xy_pred, xy, xy_mask)
                    loss_hm = _masked_kl_heatmap(logits, heatmap, heatmap_mask)
                    loss = loss_xy + loss_hm

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                compute_time += (perf_counter() - t_comp0)

                bs = clip.shape[0]
                running["loss"] = running.get("loss", 0.0) + float(loss.item()) * bs
                running["loss_xy"] = running.get("loss_xy", 0.0) + float(loss_xy.item()) * bs
                running["loss_hm"] = running.get("loss_hm", 0.0) + float(loss_hm.item()) * bs

                # Progress logging
                if (i % log_every) == 0:
                    now = perf_counter()
                    dt_log = now - t_last_log
                    elapsed = now - t_epoch0

                    # throughput over the last log window
                    # use actual samples processed since last log: approximate as log_every * batch_size except for i=0
                    window_steps = log_every if i > 0 else 1
                    ips = (window_steps * args.batch_size) / max(dt_log, 1e-6)

                    frac = (i + 1) / max(steps_per_epoch, 1)
                    eta = (elapsed / max(frac, 1e-6)) - elapsed

                    avg_data = data_time / max(i + 1, 1)
                    avg_comp = compute_time / max(i + 1, 1)

                    print(
                        f"epoch {epoch:02d} step {i+1:05d}/{steps_per_epoch} "
                        f"loss={loss.item():.4f} (xy={loss_xy.item():.4f} hm={loss_hm.item():.4f}) "
                        f"{ips:.1f} samp/s eta={eta/60:.1f}m | "
                        f"avg data={avg_data:.3f}s avg compute={avg_comp:.3f}s"
                    )
                    t_last_log = now

                end_iter = perf_counter()

            train_metrics = {k: v / len(train_ds) for k, v in running.items()}
            val_metrics = evaluate(model, val_loader, device, args.out_w, args.out_h)

            msg = (
                f"epoch {epoch:02d} | "
                f"train loss={train_metrics['loss']:.4f} (xy={train_metrics['loss_xy']:.4f} hm={train_metrics['loss_hm']:.4f}) | "
                f"val loss={val_metrics.get('loss', float('nan')):.4f} nde_med={val_metrics.get('nde_median', float('nan')):.4f} "
                f"xy_mae={val_metrics.get('xy_mae', float('nan')):.4f} xy_mae_px={val_metrics.get('xy_mae_px', float('nan')):.2f}"
            )

            if train_loader_eval is not None:
                train_eval_metrics = evaluate(model, train_loader_eval, device, args.out_w, args.out_h)
                msg += (
                    f" | train_eval nde_med={train_eval_metrics.get('nde_median', float('nan')):.4f} "
                    f"xy_mae_px={train_eval_metrics.get('xy_mae_px', float('nan')):.2f}"
                )

            print(msg)

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / f"checkpoint_epoch{epoch:02d}.pt")
            t_epoch_end = perf_counter()
            print(f"epoch {epoch:02d} wall_time={(t_epoch_end - t_epoch_start):.1f}s")

            v = val_metrics.get("xy_mae_px", None)
            if v is not None and v < best_val:
                best_val = float(v)
                torch.save(ckpt, save_dir / "checkpoint_best.pt")

    finally:
        # Ensure VideoFrameReader is released even if training crashes
        train_tf.close()
        val_tf.close()
        eval_tf.close()


if __name__ == "__main__":
    main()
