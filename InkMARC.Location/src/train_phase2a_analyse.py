from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
from analysis.error_dashboard_phase2a import Rec, write_dashboard

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from data.dataset_core import InkMARCIndexedDataset
from data.transforms.phase2_transform_v1 import Phase2Transform_v1
from eval.inference_latency import run_loader_latency
from models.model_phase2 import MobileNetV3_HeatmapDSNT

import time
from tqdm.auto import tqdm

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    xy = torch.stack([b["xy"] for b in batch], dim=0)
    xy_mask = torch.stack([b["xy_mask"] for b in batch], dim=0)
    heatmap = torch.stack([b["heatmap"] for b in batch], dim=0)
    heatmap_mask = torch.stack([b["heatmap_mask"] for b in batch], dim=0)
    meta = [b.get("meta", {}) for b in batch]
    return {
        "image": images,
        "xy": xy,
        "xy_mask": xy_mask,
        "heatmap": heatmap,
        "heatmap_mask": heatmap_mask,
        "meta": meta,
    }

class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v: float, k: int):
        self.sum += float(v) * int(k)
        self.n += int(k)

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)

def _maybe_subset(ds, n: int):
    if n <= 0 or n >= len(ds):
        return ds
    return Subset(ds, list(range(n)))


def _masked_smoothl1_xy(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=1)  # (B,)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den


def _masked_kl_heatmap(logits: torch.Tensor, target_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits:       (B, 1, H, W)
    target_prob:  (B, 1, H, W)  (probabilities; sum over H*W = 1 for valid samples)
    mask:         (B,) float {0,1}
    """
    # Flatten spatial dims once: (B, 1, HW)
    logp = F.log_softmax(logits.flatten(2), dim=-1)
    tgt = target_prob.flatten(2)

    # KL per element: (B, 1, HW) -> sum over HW -> (B, 1) -> (B,)
    per = F.kl_div(logp, tgt, reduction="none").sum(dim=-1).squeeze(1)

    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den

@torch.no_grad()
def evaluate(model, loader, device, out_w: int, out_h: int, use_amp: bool = False, desc: str = "val") -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    nde_chunks: List[torch.Tensor] = []

    scale_f32 = torch.tensor([out_w, out_h], device=device, dtype=torch.float32)
    scale_f16 = torch.tensor([out_w, out_h], device=device, dtype=torch.float16)
    scale_bf16 = torch.tensor([out_w, out_h], device=device, dtype=torch.bfloat16)  # optional

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True).detach()
        xy_mask = batch["xy_mask"].to(device, non_blocking=True).detach()
        heatmap = batch["heatmap"].to(device, non_blocking=True).detach()
        heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True).detach()

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(images)
            xy_pred = out["xy"]
            logits = out["heatmap_logits"]

            loss_xy = _masked_smoothl1_xy(xy_pred, xy, xy_mask)
            loss_hm = _masked_kl_heatmap(logits, heatmap, heatmap_mask)
            loss = loss_xy + loss_hm

        abs_err = (xy_pred - xy).abs()
        mae = abs_err.mean(dim=1)
        den = xy_mask.sum().clamp_min(1.0)
        xy_mae = (mae * xy_mask).sum() / den

        if abs_err.dtype == torch.float16:
            scale = scale_f16
        elif abs_err.dtype == torch.bfloat16:
            scale = scale_bf16
        else:
            scale = scale_f32

        abs_err_px = (abs_err * scale).mean(dim=1)
        xy_mae_px = (abs_err_px * xy_mask).sum() / den

        eu = torch.sqrt(((xy_pred - xy) ** 2).sum(dim=1))
        mask = (xy_mask > 0.5)
        sel = eu[mask]
        if sel.numel() > 0:
            nde_chunks.append(sel.detach().float().cpu())

        bs = images.shape[0]
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item()) * bs
        sums["loss_xy"] = sums.get("loss_xy", 0.0) + float(loss_xy.item()) * bs
        sums["loss_hm"] = sums.get("loss_hm", 0.0) + float(loss_hm.item()) * bs
        sums["xy_mae"] = sums.get("xy_mae", 0.0) + float(xy_mae.item()) * bs
        sums["xy_mae_px"] = sums.get("xy_mae_px", 0.0) + float(xy_mae_px.item()) * bs
        n += bs

    outm = {k: v / max(n, 1) for k, v in sums.items()}
    if nde_chunks:
        nde_all = torch.cat(nde_chunks)
        outm["nde_median"] = float(nde_all.median().item())
    else:
        outm["nde_median"] = float("nan")

    return outm

@torch.no_grad()
def evaluate_with_dashboard(
    model,
    loader,
    device,
    out_w: int,
    out_h: int,
    save_dir: Path,
    epoch: int,
    use_amp: bool = False,
    desc: str = "val",
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0

    loss_m = AvgMeter()
    loss_xy_m = AvgMeter()
    loss_hm_m = AvgMeter()
    xy_mae_px_m = AvgMeter()

    # For nde median without Python list/sort
    nde_chunks: List[torch.Tensor] = []

    # Dashboard records
    recs: List[Rec] = []

    # Precreate scale tensors once
    scale_f32 = torch.tensor([out_w, out_h], device=device, dtype=torch.float32)
    scale_f16 = torch.tensor([out_w, out_h], device=device, dtype=torch.float16)
    scale_bf16 = torch.tensor([out_w, out_h], device=device, dtype=torch.bfloat16)

    pbar = tqdm(loader, desc=f"{desc} e{epoch:02d}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True).detach()
        xy_mask = batch["xy_mask"].to(device, non_blocking=True).detach()
        heatmap = batch["heatmap"].to(device, non_blocking=True).detach()
        heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True).detach()

        meta = batch["meta"]  # list[dict] per collate

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(images)
            xy_pred = out["xy"]
            logits = out["heatmap_logits"]

            loss_xy = _masked_smoothl1_xy(xy_pred, xy, xy_mask)
            loss_hm = _masked_kl_heatmap(logits, heatmap, heatmap_mask)
            loss = loss_xy + loss_hm

        # --- Metrics in torch (once) ---
        diff = (xy_pred - xy)                       # (B,2)
        abs_err = diff.abs()                        # (B,2)

        mae = abs_err.mean(dim=1)                   # (B,)
        den = xy_mask.sum().clamp_min(1.0)
        xy_mae = (mae * xy_mask).sum() / den

        # dtype-matched scale selection
        if abs_err.dtype == torch.float16:
            scale = scale_f16
        elif abs_err.dtype == torch.bfloat16:
            scale = scale_bf16
        else:
            scale = scale_f32

        abs_err_px = (abs_err * scale).mean(dim=1)  # (B,)
        xy_mae_px = (abs_err_px * xy_mask).sum() / den

        eu = torch.sqrt((diff * diff).sum(dim=1))   # (B,)

        # NDE collection (no tolist)
        valid = (xy_mask > 0.5)
        sel = eu[valid]
        if sel.numel() > 0:
            nde_chunks.append(sel.detach().float().cpu())

        # --- Aggregate sums ---
        bs = images.shape[0]

        # update meters
        loss_m.update(loss.item(), bs)
        loss_xy_m.update(loss_xy.item(), bs)
        loss_hm_m.update(loss_hm.item(), bs)
        xy_mae_px_m.update(xy_mae_px.item(), bs)

        # show rolling metrics
        pbar.set_postfix(
            loss=f"{loss_m.avg:.4f}",
            xy=f"{loss_xy_m.avg:.4f}",
            hm=f"{loss_hm_m.avg:.4f}",
            mae_px=f"{xy_mae_px_m.avg:.2f}",
        )

        sums["loss"] = sums.get("loss", 0.0) + float(loss.item()) * bs
        sums["loss_xy"] = sums.get("loss_xy", 0.0) + float(loss_xy.item()) * bs
        sums["loss_hm"] = sums.get("loss_hm", 0.0) + float(loss_hm.item()) * bs
        sums["xy_mae"] = sums.get("xy_mae", 0.0) + float(xy_mae.item()) * bs
        sums["xy_mae_px"] = sums.get("xy_mae_px", 0.0) + float(xy_mae_px.item()) * bs
        n += bs

        # --- Dashboard vectors: compute in torch, transfer once ---
        # (keep same definitions as your NumPy dashboard)
        err_l1_norm_t = mae
        err_eu_norm_t = eu
        err_l1_px_t = abs_err_px

        # Euclidean px (anisotropic scaling)
        dx_px_t = diff[:, 0] * float(out_w)
        dy_px_t = diff[:, 1] * float(out_h)
        err_eu_px_t = torch.sqrt(dx_px_t * dx_px_t + dy_px_t * dy_px_t)

        # Transfer minimal arrays to CPU once per batch
        xy_cpu = xy.detach().cpu().numpy()
        pred_cpu = xy_pred.detach().cpu().numpy()
        xy_mask_cpu = xy_mask.detach().cpu().numpy()
        hm_mask_cpu = heatmap_mask.detach().cpu().numpy()

        l1_cpu = err_l1_norm_t.detach().float().cpu().numpy()
        eu_cpu = err_eu_norm_t.detach().float().cpu().numpy()
        l1px_cpu = err_l1_px_t.detach().float().cpu().numpy()
        eupx_cpu = err_eu_px_t.detach().float().cpu().numpy()

        for i in range(bs):
            m = meta[i] if isinstance(meta, list) else {}

            recs.append(Rec(
                file_id=str(m.get("file_id", "")),
                rel_path=str(m.get("rel_path", "")),
                split=str(m.get("split", "val")),
                participant_id=int(m.get("participant_id", -1)),
                task=str(m.get("task", "unknown")),
                session_id=str(m.get("session_id", "")),
                frame_idx=int(m.get("frame_idx", -1)),
                sample_id=str(m.get("sample_id", "")),
                xy_mask=int(xy_mask_cpu[i] > 0.5),
                heatmap_mask=int(hm_mask_cpu[i] > 0.5),
                gt_x=float(xy_cpu[i, 0]),
                gt_y=float(xy_cpu[i, 1]),
                pred_x=float(pred_cpu[i, 0]),
                pred_y=float(pred_cpu[i, 1]),
                err_l1_norm=float(l1_cpu[i]),
                err_eu_norm=float(eu_cpu[i]),
                err_l1_px=float(l1px_cpu[i]),
                err_eu_px=float(eupx_cpu[i]),
            ))

    outm = {k: v / max(n, 1) for k, v in sums.items()}

    if nde_chunks:
        nde_all = torch.cat(nde_chunks)
        outm["nde_median"] = float(nde_all.median().item())
    else:
        outm["nde_median"] = float("nan")

    dash_dir = save_dir / "dashboard"
    write_dashboard(recs, dash_dir / f"epoch_{epoch:02d}", out_w=out_w, out_h=out_h, prefix="val")

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
    ap.add_argument("--save-dir", default="runs/phase2_heatmap_dsnt")
    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--hm-w", type=int, default=60)
    ap.add_argument("--hm-h", type=int, default=40)
    ap.add_argument("--hm-sigma", type=float, default=1.5)
    ap.add_argument("--overfit-small", nargs="?", const=64, type=int, default=0)

    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--require-touch-for-xy", action="store_true",
                    help="Train/eval localisation only on touch frames (touch_mask=1 & touch=1).")

    ap.add_argument("--resume", type=str, default="",
                    help="Path to a checkpoint (.pt) to resume from. If empty, start fresh.")
    ap.add_argument("--resume-best", action="store_true",
                    help="Resume from <save-dir>/checkpoint_best.pt if it exists.")
    ap.add_argument("--reset-optim", action="store_true",
                    help="When resuming, do not load optimizer/scaler state (fresh LR schedule).")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only run evaluation (requires --resume or --resume-best).")
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit eval-only to first N samples")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured eval passes for latency")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup eval passes for latency")
    ap.add_argument("--export-onnx", action="store_true",
                    help="Export ONNX (requires --resume or --resume-best) and exit.")
    ap.add_argument("--onnx-path", type=str, default="",
                    help="Output path for ONNX (default: <save-dir>/model.onnx).")
    ap.add_argument("--onnx-h", type=int, default=320,
                    help="ONNX export input height (fixed).")
    ap.add_argument("--onnx-w", type=int, default=480,
                    help="ONNX export input width (fixed).")

    args = ap.parse_args()

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_tf = Phase2Transform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="train",
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=args.require_touch_for_xy,
    )
    val_tf = Phase2Transform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="val",
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=args.require_touch_for_xy,
    )

    train_ds = InkMARCIndexedDataset(args.dataset_root, index_version=args.version, split="train", require_labels=True, transform=train_tf)
    val_ds = InkMARCIndexedDataset(args.dataset_root, index_version=args.version, split="val", require_labels=True, transform=val_tf)

    train_ds = _maybe_subset(train_ds, args.overfit_small)
    val_cap = args.eval_max_samples if args.eval_max_samples > 0 else args.overfit_small
    val_ds = _maybe_subset(val_ds, val_cap)

    persistent = bool(args.persistent_workers and args.num_workers > 0)
    prefetch = args.prefetch_factor if args.num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
    )

    model = MobileNetV3_HeatmapDSNT(hm_h=args.hm_h, hm_w=args.hm_w, with_touch=False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1

    ckpt_path = None
    if args.resume_best:
        cand = save_dir / "checkpoint_best.pt"
        if cand.exists():
            ckpt_path = cand
    elif args.resume:
        cand = Path(args.resume)
        if cand.exists():
            ckpt_path = cand

    if ckpt_path is not None:
        print(f"[resume] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state"], strict=True)

        if not args.reset_optim:
            if "optim_state" in ckpt:
                optim.load_state_dict(ckpt["optim_state"])
            # If you want to resume AMP scaler exactly:
            if use_amp and "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])

        # continue from next epoch
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        # keep best_val sensible so "best" continues to work
        # If ckpt was best, you can seed best_val from current val later.
        best_val = float("inf")
    else:
        best_val = float("inf")    

    if args.export_onnx:
        if ckpt_path is None:
            raise SystemExit("export-onnx requires --resume or --resume-best to load a checkpoint.")

        class _ONNXWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)
                return out["xy"], out["heatmap_logits"]

        model.eval()
        wrapper = _ONNXWrapper(model)

        onnx_path = Path(args.onnx_path) if args.onnx_path else (save_dir / "model.onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.zeros(1, 3, args.onnx_h, args.onnx_w, device=device)
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["xy", "heatmap_logits"],
            dynamic_axes={"image": {0: "batch"}, "xy": {0: "batch"}, "heatmap_logits": {0: "batch"}},
        )
        print(f"[onnx] Exported to: {onnx_path}")
        return

    if args.eval_only:
        if ckpt_path is None:
            raise SystemExit("eval-only requires --resume or --resume-best to load a checkpoint.")
        val_metrics = evaluate_with_dashboard(
            model,
            val_loader,
            device,
            args.out_w,
            args.out_h,
            save_dir=save_dir,
            epoch=0,
            use_amp=use_amp,
            desc="val",
        )
        print(
            "eval-only | "
            f"val loss={val_metrics.get('loss', float('nan')):.4f} nde_med={val_metrics.get('nde_median', float('nan')):.4f} "
            f"xy_mae={val_metrics.get('xy_mae', float('nan')):.4f} xy_mae_px={val_metrics.get('xy_mae_px', float('nan')):.2f}"
        )

        def _infer_batch(batch):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _ = model(images)

        latency = run_loader_latency(
            loader=val_loader,
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

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running: Dict[str, float] = {}

            loss_m = AvgMeter()
            loss_xy_m = AvgMeter()
            loss_hm_m = AvgMeter()

            pbar = tqdm(train_loader, desc=f"train e{epoch:02d}", leave=True, dynamic_ncols=True)
            for batch in pbar:
                images = batch["image"].to(device, non_blocking=True)
                xy = batch["xy"].to(device, non_blocking=True).detach()
                xy_mask = batch["xy_mask"].to(device, non_blocking=True).detach()
                heatmap = batch["heatmap"].to(device, non_blocking=True).detach()
                heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True).detach()

                optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(images)
                    xy_pred = out["xy"]
                    logits = out["heatmap_logits"]

                    loss_xy = _masked_smoothl1_xy(xy_pred, xy, xy_mask)
                    loss_hm = _masked_kl_heatmap(logits, heatmap, heatmap_mask)
                    loss = loss_xy + loss_hm

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                bs = images.shape[0]
                running["loss"] = running.get("loss", 0.0) + float(loss.item()) * bs
                running["loss_xy"] = running.get("loss_xy", 0.0) + float(loss_xy.item()) * bs
                running["loss_hm"] = running.get("loss_hm", 0.0) + float(loss_hm.item()) * bs

                # rolling meters for bar
                loss_m.update(loss.item(), bs)
                loss_xy_m.update(loss_xy.item(), bs)
                loss_hm_m.update(loss_hm.item(), bs)

                pbar.set_postfix(
                    loss=f"{loss_m.avg:.4f}",
                    xy=f"{loss_xy_m.avg:.4f}",
                    hm=f"{loss_hm_m.avg:.4f}",
                    lr=f"{optim.param_groups[0]['lr']:.2e}",
                )

            train_metrics = {k: v / len(train_ds) for k, v in running.items()}
            val_metrics = evaluate_with_dashboard(
                model,
                val_loader,
                device,
                args.out_w,
                args.out_h,
                save_dir=save_dir,
                epoch=epoch,
                use_amp=use_amp,
                desc="val",
            )


            print(
                f"epoch {epoch:02d} | "
                f"train loss={train_metrics['loss']:.4f} (xy={train_metrics['loss_xy']:.4f} hm={train_metrics['loss_hm']:.4f}) | "
                f"val loss={val_metrics.get('loss', float('nan')):.4f} nde_med={val_metrics.get('nde_median', float('nan')):.4f} "
                f"xy_mae={val_metrics.get('xy_mae', float('nan')):.4f} xy_mae_px={val_metrics.get('xy_mae_px', float('nan')):.2f}"
            )

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "args": vars(args),
            }
            if use_amp:
                ckpt["scaler_state"] = scaler.state_dict()
            torch.save(ckpt, save_dir / f"checkpoint_epoch{epoch:02d}.pt")

            v = val_metrics.get("xy_mae_px", None)
            if v is not None and v < best_val:
                best_val = float(v)
                torch.save(ckpt, save_dir / "checkpoint_best.pt")

    finally:
        train_tf.close()
        val_tf.close()


if __name__ == "__main__":
    main()
