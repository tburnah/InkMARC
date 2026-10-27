from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

from data.dataset_clip import InkMARCIndexedClipDataset
from data.transforms.phase4_transform_clip_v2 import Phase4TransformClip_v2
from eval.inference_latency import resolve_ckpt_path, run_loader_latency
from models.model_phase2 import dsnt, spatial_softmax


class MobileNetV3_XYHistoryDSNT(nn.Module):
    """
    Autoregressive temporal model:
      - runs MobileNetV3 features per frame
      - predicts xy for each frame
      - feeds the last K predicted xy pairs into the next frame step
      - returns prediction for the last frame in the clip
    """

    def __init__(self, hm_h: int = 40, hm_w: int = 60, history_len: int = 2) -> None:
        super().__init__()
        self.hm_h = int(hm_h)
        self.hm_w = int(hm_w)
        self.history_len = max(1, int(history_len))

        backbone = torchvision.models.mobilenet_v3_large(weights=None)
        self.features = backbone.features

        last_ch = 960
        hist_dim = 2 * self.history_len
        cond_dim = 256

        self.frame_token = nn.Sequential(
            nn.Linear(last_ch, cond_dim),
            nn.ReLU(inplace=True),
        )
        self.hist_token = nn.Sequential(
            nn.Linear(hist_dim, cond_dim),
            nn.ReLU(inplace=True),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.ReLU(inplace=True),
        )
        self.film = nn.Linear(cond_dim, 2 * last_ch)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, x_clip: torch.Tensor, detach_history: bool = False) -> Dict[str, torch.Tensor]:
        b, t, c, h, w = x_clip.shape
        x = x_clip.view(b * t, c, h, w)

        feats = self.features(x)  # (B*T, Cf, Hf, Wf)
        cf, hf, wf = feats.shape[1], feats.shape[2], feats.shape[3]
        feats = feats.view(b, t, cf, hf, wf)

        hist = torch.zeros(b, self.history_len, 2, dtype=feats.dtype, device=feats.device)

        logits_last = None
        prob_last = None
        xy_last = None

        for ti in range(t):
            f_t = feats[:, ti, :, :, :]              # (B,Cf,Hf,Wf)
            pooled_t = f_t.mean(dim=(2, 3))          # (B,Cf)

            hist_flat = hist.reshape(b, -1)          # (B, 2*K)
            cond = torch.cat([self.frame_token(pooled_t), self.hist_token(hist_flat)], dim=1)
            cond = self.cond_fuse(cond)

            gamma, beta = self.film(cond).chunk(2, dim=1)
            f_t = f_t * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

            logits_t = self.heatmap_head(f_t)
            logits_t = F.interpolate(
                logits_t,
                size=(self.hm_h, self.hm_w),
                mode="bilinear",
                align_corners=False,
            )
            prob_t = spatial_softmax(logits_t)
            xy_t = dsnt(prob_t)

            hist = torch.roll(hist, shifts=-1, dims=1)
            hist_new = xy_t.detach() if detach_history else xy_t
            hist[:, -1, :] = hist_new

            logits_last = logits_t
            prob_last = prob_t
            xy_last = xy_t

        if logits_last is None or prob_last is None or xy_last is None:
            raise RuntimeError("Empty clip passed to model.")

        return {
            "heatmap_logits": logits_last,
            "heatmap_prob": prob_last,
            "xy": xy_last,
        }


class _Phase2cOnnxWrapper(nn.Module):
    def __init__(self, model: MobileNetV3_XYHistoryDSNT) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_clip: torch.Tensor):
        out = self.model(image_clip, detach_history=True)
        return out["xy"], out["heatmap_logits"]


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
    logp = F.log_softmax(logits.flatten(2), dim=-1)
    tgt = target_prob.flatten(2)
    per = F.kl_div(logp, tgt, reduction="none").sum(dim=-1).squeeze(1)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den

def evaluate(
    model: MobileNetV3_XYHistoryDSNT,
    loader: DataLoader,
    device: torch.device,
    out_w: int,
    out_h: int,
    use_amp: bool = False,
    detach_history: bool = True,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    nde_chunks: List[torch.Tensor] = []

    scale_f32 = torch.tensor([out_w, out_h], device=device, dtype=torch.float32)
    scale_f16 = torch.tensor([out_w, out_h], device=device, dtype=torch.float16)
    scale_bf16 = torch.tensor([out_w, out_h], device=device, dtype=torch.bfloat16)

    for batch in loader:
        clip = batch["image_clip"].to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True)
        xy_mask = batch["xy_mask"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(clip, detach_history=detach_history)
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
        sel = eu[xy_mask > 0.5]
        if sel.numel() > 0:
            nde_chunks.append(sel.detach().float().cpu())

        bs = clip.shape[0]
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
) -> DataLoader:
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


def export_best_checkpoint_to_onnx(
    model: MobileNetV3_XYHistoryDSNT,
    ckpt_arg: str,
    save_dir: Path,
    out_path: Path,
    device: torch.device,
    clip_len: int,
    out_h: int,
    out_w: int,
    onnx_opset: int,
) -> Path:
    ckpt_path = resolve_ckpt_path(ckpt_arg, save_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    wrapper = _Phase2cOnnxWrapper(model).to(device).eval()
    dummy = torch.zeros(1, clip_len, 3, out_h, out_w, device=device, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["image_clip"],
        output_names=["xy", "heatmap_logits"],
        dynamic_axes={
            "image_clip": {0: "batch"},
            "xy": {0: "batch"},
            "heatmap_logits": {0: "batch"},
        },
        opset_version=onnx_opset,
    )
    print(f"ONNX exported: {out_path} (from {ckpt_path})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/phase2c_xy_history")

    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--hm-w", type=int, default=60)
    ap.add_argument("--hm-h", type=int, default=40)
    ap.add_argument("--hm-sigma", type=float, default=1.5)

    ap.add_argument("--clip-len", type=int, default=5)
    ap.add_argument("--history-len", type=int, default=2)
    ap.add_argument(
        "--init-from-ckpt",
        default="",
        help="Warm-start model from a checkpoint (e.g. phase2a checkpoint_best.pt). Shared keys are loaded with strict=False.",
    )
    ap.add_argument(
        "--detach-history",
        action="store_true",
        help="Detach previous predicted xy before feeding history to next frame step.",
    )

    ap.add_argument("--require-touch-for-xy", action="store_true")
    ap.add_argument("--overfit-small", nargs="?", const=64, type=int, default=0)

    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation and exit")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"], help="Split to evaluate on")
    ap.add_argument("--ckpt", default="", help="Checkpoint to evaluate (defaults to save-dir/checkpoint_best.pt)")
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit eval split to first N samples")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured eval passes for latency")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup eval passes for latency")
    ap.add_argument(
        "--export-best-onnx",
        default="",
        help="If set, load best previous checkpoint (or --ckpt) and export ONNX to this path, then exit.",
    )
    ap.add_argument("--onnx-opset", type=int, default=17)

    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--amp-dtype", default="bf16", choices=["fp16", "bf16"])

    args = ap.parse_args()

    if args.history_len < 1:
        raise ValueError("--history-len must be >= 1")
    if args.clip_len < 2:
        raise ValueError("--clip-len must be >= 2 for temporal prediction.")

    export_onnx_only = bool(args.export_best_onnx)
    if not export_onnx_only and not args.dataset_root:
        raise ValueError("--dataset-root is required unless --export-best-onnx is used.")

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    # Only needed for training (backprop). Skip on eval-only.
    scaler = None
    if use_amp and not args.eval_only:
        scaler = torch.amp.GradScaler("cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = MobileNetV3_XYHistoryDSNT(
        hm_h=args.hm_h,
        hm_w=args.hm_w,
        history_len=args.history_len,
    ).to(device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if export_onnx_only:
        export_best_checkpoint_to_onnx(
            model=model,
            ckpt_arg=args.ckpt,
            save_dir=save_dir,
            out_path=Path(args.export_best_onnx),
            device=device,
            clip_len=args.clip_len,
            out_h=args.out_h,
            out_w=args.out_w,
            onnx_opset=args.onnx_opset,
        )
        return

    train_tf = Phase4TransformClip_v2(
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
    val_tf = Phase4TransformClip_v2(
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

    eval_tf = Phase4TransformClip_v2(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split=args.eval_split,
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=args.require_touch_for_xy,
    )

    eval_ds = InkMARCIndexedClipDataset(
        args.dataset_root,
        index_version=args.version,
        split=args.eval_split,
        clip_len=args.clip_len,
        require_labels_last=True,
        transform=eval_tf,
    )

    eval_cap = args.eval_max_samples if args.eval_max_samples > 0 else args.overfit_small
    eval_ds = _maybe_subset(eval_ds, eval_cap)

    train_ds = _maybe_subset(train_ds, args.overfit_small)
    val_ds = _maybe_subset(val_ds, args.overfit_small)

    persistent = bool(args.persistent_workers and args.num_workers > 0)

    train_loader = _make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent,
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
        persistent_workers=persistent,
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
        persistent_workers=persistent,
    )


    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.eval_only:
        ckpt_path = resolve_ckpt_path(args.ckpt, save_dir)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)

        metrics = evaluate(
            model,
            eval_loader,
            device,
            args.out_w,
            args.out_h,
            use_amp=use_amp,
            detach_history=True,  # typically True for eval
        )

        print(f"Evaluated {ckpt_path} on split={args.eval_split}")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        def _infer_batch(batch):
            clip = batch["image_clip"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(clip, detach_history=True)

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
        # make sure transforms get closed
        train_tf.close()
        eval_tf.close()
        return

    if args.init_from_ckpt:
        ckpt_path = Path(args.init_from_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--init-from-ckpt not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint has unsupported model state type: {type(state)}")

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"warm-started from {ckpt_path} | "
            f"loaded={len(model.state_dict()) - len(missing)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running: Dict[str, float] = {}

            for batch in train_loader:
                clip = batch["image_clip"].to(device, non_blocking=True)
                xy = batch["xy"].to(device, non_blocking=True)
                xy_mask = batch["xy_mask"].to(device, non_blocking=True)
                heatmap = batch["heatmap"].to(device, non_blocking=True)
                heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    out = model(clip, detach_history=args.detach_history)
                    xy_pred = out["xy"]
                    logits = out["heatmap_logits"]

                    loss_xy = _masked_smoothl1_xy(xy_pred, xy, xy_mask)
                    loss_hm = _masked_kl_heatmap(logits, heatmap, heatmap_mask)
                    loss = loss_xy + loss_hm

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                bs = clip.shape[0]
                running["loss"] = running.get("loss", 0.0) + float(loss.item()) * bs
                running["loss_xy"] = running.get("loss_xy", 0.0) + float(loss_xy.item()) * bs
                running["loss_hm"] = running.get("loss_hm", 0.0) + float(loss_hm.item()) * bs

            train_metrics = {k: v / len(train_ds) for k, v in running.items()}
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.out_w,
                args.out_h,
                use_amp=use_amp,
                detach_history=args.detach_history,
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
            torch.save(ckpt, save_dir / f"checkpoint_epoch{epoch:02d}.pt")

            v = val_metrics.get("xy_mae_px", None)
            if v is not None and v < best_val:
                best_val = float(v)
                torch.save(ckpt, save_dir / "checkpoint_best.pt")

    finally:
        train_tf.close()
        val_tf.close()
        eval_tf.close()


if __name__ == "__main__":
    main()
