from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.dataset_core_v2 import InkMARCIndexedDataset
from data.transforms.phase2_transform_v2 import Phase2Transform_v2
from eval.inference_latency import resolve_ckpt_path, run_loader_latency
from models.model_phase2 import MobileNetV3_HeatmapDSNT


class _Phase2aOnnxWrapper(nn.Module):
    def __init__(self, model: MobileNetV3_HeatmapDSNT) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        out = self.model(image)
        return out["xy"], out["heatmap_logits"]


class _Phase2fXyOnlyOnnxWrapper(nn.Module):
    def __init__(self, model: MobileNetV3_HeatmapDSNT) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.model(image)
        return out["xy"]


def _export_model_to_onnx(
    model: MobileNetV3_HeatmapDSNT,
    out_path: Path,
    device: torch.device,
    out_h: int,
    out_w: int,
    onnx_opset: int,
) -> Path:
    model.eval()
    wrapper = _Phase2fXyOnlyOnnxWrapper(model).to(device).eval()
    dummy_img = torch.zeros(1, 3, out_h, out_w, device=device, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_img,
        str(out_path),
        input_names=["image"],
        output_names=["xy"],
        dynamic_axes={
            "image": {0: "batch"},
            "xy": {0: "batch"},
        },
        opset_version=onnx_opset,
    )
    print(f"ONNX exported: {out_path} | outputs=['xy']")
    return out_path


def _quantize_onnx_dynamic(src_onnx: Path, dst_onnx: Path) -> Path:
    try:
        from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic
    except ImportError as e:
        raise RuntimeError(
            "ONNX quantization requires onnxruntime. Install it (e.g. pip install onnxruntime)."
        ) from e

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_onnx = dst_onnx.with_name(f"{dst_onnx.stem}.preprocessed.onnx")

    quant_input = src_onnx
    try:
        quant_pre_process(
            input_model_path=str(src_onnx),
            output_model_path=str(preprocessed_onnx),
        )
        quant_input = preprocessed_onnx
        print(f"Preprocessed ONNX for quantization: {preprocessed_onnx}")
    except Exception as e:
        print(f"ONNX pre-processing skipped due to error; quantizing source model directly. error={e}")

    quantize_dynamic(
        model_input=str(quant_input),
        model_output=str(dst_onnx),
        weight_type=QuantType.QInt8,
        # ConvInteger dynamic quantization is often not executable on GPU builds
        # and can fail even when CPU fallback is enabled. Keep this export runtime-safe.
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"Quantized ONNX exported: {dst_onnx} (from {src_onnx})")
    return dst_onnx


class _Phase2CalibrationDataReader:
    def __init__(self, loader: DataLoader, input_name: str, max_batches: int, log_every: int = 10) -> None:
        self.loader = loader
        self.input_name = input_name
        self.max_batches = max(0, int(max_batches))
        self.log_every = max(1, int(log_every))
        self._iter: Optional[Iterator[Dict[str, torch.Tensor]]] = None
        self._seen = 0
        self.rewind()

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iter is None or self._seen >= self.max_batches:
            return None
        try:
            batch = next(self._iter)
        except StopIteration:
            return None

        image = batch["image"]
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        image_np = image.detach().cpu().to(torch.float32).numpy()
        self._seen += 1
        if self._seen == 1 or self._seen % self.log_every == 0 or self._seen == self.max_batches:
            print(f"[calib] prepared batch {self._seen}/{self.max_batches}")
        return {self.input_name: image_np}

    def rewind(self) -> None:
        self._iter = iter(self.loader)
        self._seen = 0


def _build_calibration_loader(args: argparse.Namespace) -> tuple[DataLoader, Phase2Transform_v2]:
    calib_tf = Phase2Transform_v2(
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
    calib_ds = InkMARCIndexedDataset(
        args.dataset_root,
        index_version=args.version,
        split="train",
        require_labels=True,
        transform=calib_tf,
    )
    calib_loader = DataLoader(
        calib_ds,
        batch_size=max(1, int(args.calib_batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    return calib_loader, calib_tf


def _inspect_onnx_op_types(path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print(f"ONNX op inspection skipped (onnx not installed): {path}")
        return

    model = onnx.load(str(path))
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    keys = ["Conv", "QuantizeLinear", "DequantizeLinear", "QLinearConv", "MatMul", "Gemm"]
    summary = ", ".join(f"{k}={counts.get(k, 0)}" for k in keys)
    print(f"ONNX op counts ({path}): {summary}")


def _quantize_onnx_qdq_int8_static(
    src_onnx: Path,
    dst_onnx: Path,
    calib_reader: _Phase2CalibrationDataReader,
) -> Path:
    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quant_pre_process,
            quantize_static,
        )
    except ImportError as e:
        raise RuntimeError(
            "Static ONNX quantization requires onnxruntime. Install it (e.g. pip install onnxruntime)."
        ) from e

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_onnx = dst_onnx.with_name(f"{dst_onnx.stem}.preprocessed.onnx")
    quant_input = src_onnx
    print(f"Starting static QDQ quantization: src={src_onnx} dst={dst_onnx}")
    try:
        quant_pre_process(
            input_model_path=str(src_onnx),
            output_model_path=str(preprocessed_onnx),
        )
        quant_input = preprocessed_onnx
        print(f"Preprocessed ONNX for static quantization: {preprocessed_onnx}")
    except Exception as e:
        print(f"ONNX pre-processing skipped; quantizing source model directly. error={e}")

    quantize_static(
        model_input=str(quant_input),
        model_output=str(dst_onnx),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"Static QDQ INT8 ONNX exported: {dst_onnx} (from {src_onnx})")
    _inspect_onnx_op_types(dst_onnx)
    return dst_onnx


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
def evaluate(
    model,
    loader,
    device,
    out_w: int,
    out_h: int,
    use_amp: bool = False,
    log_every: int = 0,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    nde_chunks: List[torch.Tensor] = []

    scale_f32 = torch.tensor([out_w, out_h], device=device, dtype=torch.float32)
    scale_f16 = torch.tensor([out_w, out_h], device=device, dtype=torch.float16)
    scale_bf16 = torch.tensor([out_w, out_h], device=device, dtype=torch.bfloat16)  # optional

    num_batches = len(loader) if hasattr(loader, "__len__") else 0
    eval_start = perf_counter()
    for bi, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        xy = batch["xy"].to(device, non_blocking=True)
        xy_mask = batch["xy_mask"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True)

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

        if log_every > 0 and num_batches > 0 and (bi == 1 or bi % log_every == 0 or bi == num_batches):
            elapsed = perf_counter() - eval_start
            print(f"[eval] batch {bi}/{num_batches} elapsed={elapsed:.1f}s")

    outm = {k: v / max(n, 1) for k, v in sums.items()}
    if nde_chunks:
        nde_all = torch.cat(nde_chunks)
        outm["nde_median"] = float(nde_all.median().item())
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
    ap.add_argument("--save-dir", default="runs/phase2_heatmap_dsnt")
    ap.add_argument("--resume-best", action="store_true", help="Resume training from save-dir/checkpoint_best.pt")
    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation and latency, then exit")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"], help="Split to evaluate in eval-only mode")
    ap.add_argument("--ckpt", default="", help="Checkpoint to evaluate (default: best, then latest epoch)")
    ap.add_argument("--export-ckpt", default="", help="Checkpoint to export as ONNX (default: best, then latest epoch)")
    ap.add_argument("--save-onnx", default="", help="(Legacy) Export selected checkpoint to ONNX path")
    ap.add_argument("--save-onnx-quantized", default="", help="(Legacy) Export dynamic-quantized ONNX to this path")
    ap.add_argument("--export-onnx", default="", help="Export selected checkpoint to fp32 ONNX (xy output only)")
    ap.add_argument("--quantize-qdq-int8", default="", help="Write static QDQ INT8 ONNX to this path")
    ap.add_argument("--calib-batches", type=int, default=50, help="Calibration batch count for static QDQ quantization")
    ap.add_argument("--calib-batch-size", type=int, default=1, help="Calibration batch size for static QDQ quantization")
    ap.add_argument("--log-every", type=int, default=50, help="Print progress every N batches")
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if val xy_mae_px does not improve for N consecutive epochs (0 disables)",
    )
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.1,
        help="Minimum xy_mae_px improvement (in px) to reset early-stop patience",
    )
    ap.add_argument(
        "--early-stop-warmup-epochs",
        type=int,
        default=5,
        help="Do not trigger early stopping before this epoch number",
    )
    ap.add_argument(
        "--restore-best-on-stop",
        action="store_true",
        help="When early stopping triggers, load save-dir/checkpoint_best.pt into the model",
    )
    ap.add_argument("--onnx-opset", type=int, default=17)
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit eval-only to first N samples")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured eval passes for latency")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup eval passes for latency")
    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--hm-w", type=int, default=60)
    ap.add_argument("--hm-h", type=int, default=40)
    ap.add_argument("--hm-sigma", type=float, default=1.5)
    ap.add_argument("--overfit-small", nargs="?", const=64, type=int, default=0)
    ap.add_argument("--require-touch-for-xy", action="store_true",
                    help="Train/eval localisation only on touch frames (touch_mask=1 & touch=1).")

    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=4)

    args = ap.parse_args()

    if not args.export_onnx and args.save_onnx:
        args.export_onnx = args.save_onnx

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_tf = Phase2Transform_v2(
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
    val_tf = Phase2Transform_v2(
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
    eval_tf = Phase2Transform_v2(
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

    train_ds = InkMARCIndexedDataset(args.dataset_root, index_version=args.version, split="train", require_labels=True, transform=train_tf)
    val_ds = InkMARCIndexedDataset(args.dataset_root, index_version=args.version, split="val", require_labels=True, transform=val_tf)
    eval_ds = InkMARCIndexedDataset(
        args.dataset_root, index_version=args.version, split=args.eval_split, require_labels=True, transform=eval_tf
    )

    train_ds = _maybe_subset(train_ds, args.overfit_small)
    val_ds = _maybe_subset(val_ds, args.overfit_small)
    eval_cap = args.eval_max_samples if args.eval_max_samples > 0 else args.overfit_small
    eval_ds = _maybe_subset(eval_ds, eval_cap)

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
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
    )

    train_batches = len(train_loader) if hasattr(train_loader, "__len__") else -1
    val_batches = len(val_loader) if hasattr(val_loader, "__len__") else -1
    eval_batches = len(eval_loader) if hasattr(eval_loader, "__len__") else -1
    print(
        "Data summary | "
        f"dataset_root={Path(args.dataset_root).resolve()} version={args.version} "
        f"split(train/val/eval)={train_tf.split}/{val_tf.split}/{eval_tf.split} "
        f"overfit_small={args.overfit_small} eval_max_samples={args.eval_max_samples}"
    )
    print(
        "Loader summary | "
        f"batch_size={args.batch_size} num_workers={args.num_workers} "
        f"train_samples={len(train_ds)} train_batches={train_batches} "
        f"val_samples={len(val_ds)} val_batches={val_batches} "
        f"eval_samples={len(eval_ds)} eval_batches={eval_batches}"
    )

    model = MobileNetV3_HeatmapDSNT(hm_h=args.hm_h, hm_w=args.hm_w, with_touch=False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    start_epoch = 1
    epochs_since_improve = 0

    try:
        if args.resume_best:
            resume_path = save_dir / "checkpoint_best.pt"
            if not resume_path.exists():
                raise FileNotFoundError(f"--resume-best was set, but checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)
            optim_state = ckpt.get("optim_state")
            if isinstance(optim_state, dict):
                optim.load_state_dict(optim_state)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if "best_val_xy_mae_px" in ckpt:
                best_val = float(ckpt["best_val_xy_mae_px"])
            if "epochs_since_improve" in ckpt:
                epochs_since_improve = int(ckpt["epochs_since_improve"])
            else:
                resume_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    args.out_w,
                    args.out_h,
                    use_amp=use_amp,
                    log_every=max(0, args.log_every),
                )
                resume_xy_mae_px = resume_metrics.get("xy_mae_px")
                if resume_xy_mae_px is not None and not math.isnan(float(resume_xy_mae_px)):
                    best_val = float(resume_xy_mae_px)
                epochs_since_improve = 0
            print(f"Resumed from best checkpoint: {resume_path} (next epoch: {start_epoch})")

        if args.eval_only:
            ckpt_path = resolve_ckpt_path(args.ckpt, save_dir)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)

            metrics = evaluate(
                model,
                eval_loader,
                device,
                args.out_w,
                args.out_h,
                use_amp=use_amp,
                log_every=max(0, args.log_every),
            )
            print(f"Evaluated {ckpt_path} on split={args.eval_split}")
            for k, v in metrics.items():
                print(f"{k}: {v}")

            def _infer_batch(batch):
                images = batch["image"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _ = model(images)

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

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running: Dict[str, float] = {}
            train_start = perf_counter()
            num_train_batches = len(train_loader) if hasattr(train_loader, "__len__") else 0
            seen_train = 0

            for bi, batch in enumerate(train_loader, start=1):
                images = batch["image"].to(device, non_blocking=True)
                xy = batch["xy"].to(device, non_blocking=True)
                xy_mask = batch["xy_mask"].to(device, non_blocking=True)
                heatmap = batch["heatmap"].to(device, non_blocking=True)
                heatmap_mask = batch["heatmap_mask"].to(device, non_blocking=True)

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
                seen_train += bs

                if args.log_every > 0 and num_train_batches > 0 and (
                    bi == 1 or bi % args.log_every == 0 or bi == num_train_batches
                ):
                    elapsed = perf_counter() - train_start
                    avg_loss = running["loss"] / max(1, seen_train)
                    print(
                        f"[train] epoch {epoch:02d} batch {bi}/{num_train_batches} "
                        f"elapsed={elapsed:.1f}s avg_loss={avg_loss:.4f}"
                    )

            train_metrics = {k: v / len(train_ds) for k, v in running.items()}
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.out_w,
                args.out_h,
                use_amp=use_amp,
                log_every=max(0, args.log_every),
            )

            print(
                f"epoch {epoch:02d} | "
                f"train loss={train_metrics['loss']:.4f} (xy={train_metrics['loss_xy']:.4f} hm={train_metrics['loss_hm']:.4f}) | "
                f"val loss={val_metrics.get('loss', float('nan')):.4f} nde_med={val_metrics.get('nde_median', float('nan')):.4f} "
                f"xy_mae={val_metrics.get('xy_mae', float('nan')):.4f} xy_mae_px={val_metrics.get('xy_mae_px', float('nan')):.2f}"
            )

            v = val_metrics.get("xy_mae_px", None)
            is_best = False
            has_valid_val = v is not None and not math.isnan(float(v))
            improved = has_valid_val and float(v) < (best_val - args.early_stop_min_delta)
            if improved:
                best_val = float(v)
                is_best = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "best_val_xy_mae_px": best_val,
                "epochs_since_improve": epochs_since_improve,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / f"checkpoint_epoch{epoch:02d}.pt")
            if is_best:
                torch.save(ckpt, save_dir / "checkpoint_best.pt")

            if args.early_stop_patience > 0 and epoch >= args.early_stop_warmup_epochs:
                if epochs_since_improve >= args.early_stop_patience:
                    print(
                        "Early stopping triggered "
                        f"(epoch={epoch}, patience={args.early_stop_patience}, "
                        f"epochs_since_improve={epochs_since_improve}, best_xy_mae_px={best_val:.4f})"
                    )
                    if args.restore_best_on_stop:
                        best_ckpt_path = save_dir / "checkpoint_best.pt"
                        if best_ckpt_path.exists():
                            best_ckpt = torch.load(best_ckpt_path, map_location=device)
                            model.load_state_dict(best_ckpt["model_state"], strict=True)
                            print(f"Loaded best checkpoint after early stop: {best_ckpt_path}")
                        else:
                            print(
                                "Warning: --restore-best-on-stop was set, but checkpoint_best.pt was not found."
                            )
                    break

        if args.export_onnx or args.quantize_qdq_int8 or args.save_onnx_quantized:
            ckpt_path = resolve_ckpt_path(args.export_ckpt, save_dir)
            print(f"Loading checkpoint for export: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)

            onnx_out = Path(args.export_onnx) if args.export_onnx else (save_dir / "phase2f_xy_fp32.onnx")
            _export_model_to_onnx(
                model=model,
                out_path=onnx_out,
                device=device,
                out_h=args.out_h,
                out_w=args.out_w,
                onnx_opset=args.onnx_opset,
            )
            print(f"ONNX export output is xy only: {onnx_out}")
            if args.save_onnx_quantized:
                _quantize_onnx_dynamic(
                    src_onnx=onnx_out,
                    dst_onnx=Path(args.save_onnx_quantized),
                )
            if args.quantize_qdq_int8:
                print(
                    "Preparing calibration loader "
                    f"(split=train, batches={args.calib_batches}, batch_size={args.calib_batch_size})"
                )
                calib_loader, calib_tf = _build_calibration_loader(args)
                try:
                    calib_reader = _Phase2CalibrationDataReader(
                        loader=calib_loader,
                        input_name="image",
                        max_batches=args.calib_batches,
                        log_every=max(1, min(args.log_every, 10)),
                    )
                    _quantize_onnx_qdq_int8_static(
                        src_onnx=onnx_out,
                        dst_onnx=Path(args.quantize_qdq_int8),
                        calib_reader=calib_reader,
                    )
                    print(f"Calibration batches used: {args.calib_batches} (batch_size={args.calib_batch_size})")
                finally:
                    calib_tf.close()

    finally:
        train_tf.close()
        val_tf.close()
        eval_tf.close()


if __name__ == "__main__":
    main()
