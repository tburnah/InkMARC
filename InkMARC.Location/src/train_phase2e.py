from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset_history import InkMARCIndexedHistoryDataset
from data.transforms.phase2e_transform_v1 import Phase2eTransform_v1
from eval.inference_latency import resolve_ckpt_path, run_loader_latency
from models.model_v3 import MobileNetV3_RawPlusCorners_XY


class _Phase2eOnnxWrapper(nn.Module):
    def __init__(self, model: MobileNetV3_RawPlusCorners_XY) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, aux14: torch.Tensor) -> torch.Tensor:
        out = self.model(image, aux14)
        return out["xy"]


def _export_model_to_onnx(
    model: MobileNetV3_RawPlusCorners_XY,
    out_path: Path,
    device: torch.device,
    out_h: int,
    out_w: int,
    onnx_opset: int,
) -> Path:
    model.eval()
    wrapper = _Phase2eOnnxWrapper(model).to(device).eval()
    dummy_img = torch.zeros(1, 3, out_h, out_w, device=device, dtype=torch.float32)
    dummy_aux = torch.zeros(1, 14, device=device, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_aux),
        str(out_path),
        input_names=["image", "aux14"],
        output_names=["xy"],
        dynamic_axes={
            "image": {0: "batch"},
            "aux14": {0: "batch"},
            "xy": {0: "batch"},
        },
        opset_version=onnx_opset,
    )
    print(f"ONNX exported: {out_path}")
    return out_path


def _quantize_onnx_dynamic(src_onnx: Path, dst_onnx: Path) -> Path:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise RuntimeError(
            "ONNX quantization requires onnxruntime. Install it (e.g. pip install onnxruntime)."
        ) from e

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src_onnx),
        model_output=str(dst_onnx),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized ONNX exported: {dst_onnx} (from {src_onnx})")
    return dst_onnx


class _Phase2eCalibrationDataReader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._it: Iterator | None = None

    def get_next(self):
        if self._it is None:
            self._it = iter(self.loader)
        try:
            batch = next(self._it)
        except StopIteration:
            return None
        return {
            "image": batch["image"].cpu().numpy().astype(np.float32, copy=False),
            "aux14": batch["aux14"].cpu().numpy().astype(np.float32, copy=False),
        }

    def rewind(self):
        self._it = None


def _quantize_onnx_static_qdq(
    src_onnx: Path,
    dst_onnx: Path,
    dataset_root: str,
    version: str,
    split: str,
    out_w: int,
    out_h: int,
    history_len: int,
    require_touch_for_xy: bool,
    calib_samples: int,
    calib_batch_size: int,
    calib_num_workers: int,
) -> Path:
    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    except ImportError as e:
        raise RuntimeError(
            "ONNX static quantization requires onnxruntime. Install it (e.g. pip install onnxruntime)."
        ) from e

    if not dataset_root:
        raise ValueError("--dataset-root is required for --onnx-quantization-mode static_qdq.")

    calib_tf = Phase2eTransform_v1(
        dataset_root=Path(dataset_root),
        index_version=version,
        split=split,
        out_w=out_w,
        out_h=out_h,
        normalize=True,
        history_len=history_len,
        require_touch_for_xy=require_touch_for_xy,
    )
    calib_ds = InkMARCIndexedHistoryDataset(
        dataset_root,
        index_version=version,
        split=split,
        require_labels=True,
        history_len=history_len,
        transform=calib_tf,
    )
    if calib_samples > 0:
        calib_ds = _maybe_subset(calib_ds, calib_samples)
    calib_loader = DataLoader(
        calib_ds,
        batch_size=calib_batch_size,
        shuffle=False,
        num_workers=calib_num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    reader = _Phase2eCalibrationDataReader(calib_loader)

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(src_onnx),
        model_output=str(dst_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"Static QDQ quantized ONNX exported: {dst_onnx} (from {src_onnx}, split={split}, samples={calib_samples})")

    _close_dataset(calib_ds)
    calib_tf.close()
    return dst_onnx


def _quantize_onnx(
    src_onnx: Path,
    dst_onnx: Path,
    mode: str,
    dataset_root: str,
    version: str,
    split: str,
    out_w: int,
    out_h: int,
    history_len: int,
    require_touch_for_xy: bool,
    calib_samples: int,
    calib_batch_size: int,
    calib_num_workers: int,
) -> Path:
    if mode == "dynamic":
        return _quantize_onnx_dynamic(src_onnx=src_onnx, dst_onnx=dst_onnx)
    if mode == "static_qdq":
        return _quantize_onnx_static_qdq(
            src_onnx=src_onnx,
            dst_onnx=dst_onnx,
            dataset_root=dataset_root,
            version=version,
            split=split,
            out_w=out_w,
            out_h=out_h,
            history_len=history_len,
            require_touch_for_xy=require_touch_for_xy,
            calib_samples=calib_samples,
            calib_batch_size=calib_batch_size,
            calib_num_workers=calib_num_workers,
        )
    raise ValueError(f"Unsupported --onnx-quantization-mode: {mode}")


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    corners = torch.stack([b["corners"] for b in batch], dim=0)
    history = torch.stack([b["history"] for b in batch], dim=0)
    aux14 = torch.cat([corners, history], dim=1)
    xy = torch.stack([b["xy"] for b in batch], dim=0)
    xy_mask = torch.stack([b["xy_mask"] for b in batch], dim=0)
    meta = [b.get("meta", {}) for b in batch]
    return {
        "image": images,
        "corners": corners,
        "history": history,
        "aux14": aux14,
        "xy": xy,
        "xy_mask": xy_mask,
        "meta": meta,
    }


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
    per = torch.nn.functional.smooth_l1_loss(pred, targets, reduction="none")
    per = per.mean(dim=1)
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
        aux14 = batch["aux14"].to(device)
        xy = batch["xy"].to(device)
        xy_mask = batch["xy_mask"].to(device)

        out = model(images, aux14)
        xy_pred = out["xy"]
        loss = _masked_smoothl1_loss(xy_pred, xy, xy_mask)

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


def _scheduled_sampling_prob(epoch: int, epochs: int, p0: float, p1: float) -> float:
    if epochs <= 1:
        return float(p1)
    t = float(epoch - 1) / float(epochs - 1)
    return float(p0 + (p1 - p0) * t)


def _build_aux14_for_training(
    model: MobileNetV3_RawPlusCorners_XY,
    images: torch.Tensor,
    corners: torch.Tensor,
    history_gt: torch.Tensor,
    history_mode: str,
    ss_pred_prob: float,
) -> torch.Tensor:
    if history_mode == "gt":
        return torch.cat([corners, history_gt], dim=1)

    if history_mode != "scheduled_sampling":
        raise ValueError(f"Unsupported history_mode: {history_mode}")

    b = history_gt.shape[0]
    hist_len = history_gt.shape[1] // 2
    hist = history_gt.view(b, hist_len, 2)
    aux_gt = torch.cat([corners, history_gt], dim=1)

    with torch.no_grad():
        seed_pred = model(images, aux_gt)["xy"].detach()

    hist_pred = torch.roll(hist.clone(), shifts=-1, dims=1)
    hist_pred[:, -1, :] = seed_pred

    use_pred = (torch.rand(b, device=images.device) < float(ss_pred_prob)).to(dtype=hist.dtype)
    use_pred = use_pred.view(b, 1, 1)
    hist_mix = hist * (1.0 - use_pred) + hist_pred * use_pred
    return torch.cat([corners, hist_mix.reshape(b, hist_len * 2)], dim=1)


def export_best_checkpoint_to_onnx(
    model: MobileNetV3_RawPlusCorners_XY,
    ckpt_arg: str,
    save_dir: Path,
    out_path: Path,
    device: torch.device,
    out_h: int,
    out_w: int,
    onnx_opset: int,
) -> Path:
    ckpt_path = resolve_ckpt_path(ckpt_arg, save_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    _export_model_to_onnx(
        model=model,
        out_path=out_path,
        device=device,
        out_h=out_h,
        out_w=out_w,
        onnx_opset=onnx_opset,
    )
    print(f"Export source checkpoint: {ckpt_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="runs/phase2e_aux14_xy")

    ap.add_argument("--history-len", type=int, default=3)
    ap.add_argument("--history-mode", default="gt", choices=["gt", "scheduled_sampling"])
    ap.add_argument("--require-touch-for-xy", action="store_true")
    ap.add_argument("--ss-start-pred-prob", type=float, default=0.0)
    ap.add_argument("--ss-end-pred-prob", type=float, default=0.5)
    ap.add_argument(
        "--load-best-model",
        action="store_true",
        help="Warm-start training by loading <save-dir>/checkpoint_best.pt before epoch 1.",
    )

    ap.add_argument("--eval-only", action="store_true", help="Only run evaluation and latency, then exit")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"], help="Split to evaluate in eval-only mode")
    ap.add_argument("--ckpt", default="", help="Checkpoint to evaluate (default: best, then latest epoch)")
    ap.add_argument("--eval-max-samples", type=int, default=0, help="Limit eval-only to first N samples")
    ap.add_argument("--latency-runs", type=int, default=5, help="Measured eval passes for latency")
    ap.add_argument("--latency-warmup", type=int, default=1, help="Warmup eval passes for latency")
    ap.add_argument("--amp", action="store_true", help="Use CUDA AMP for eval-only latency")

    ap.add_argument("--export-best-onnx", default="", help="Load best checkpoint and export ONNX to this path, then exit.")
    ap.add_argument(
        "--export-best-onnx-quantized",
        default="",
        help="Load best checkpoint and export quantized ONNX to this path, then exit.",
    )
    ap.add_argument("--save-onnx", default="", help="After training/eval, export current model to this ONNX path.")
    ap.add_argument(
        "--save-onnx-quantized",
        default="",
        help="After training/eval, export quantized ONNX of current model to this path.",
    )
    ap.add_argument(
        "--onnx-quantization-mode",
        default="dynamic",
        choices=["dynamic", "static_qdq"],
        help="dynamic uses ConvInteger (may not run on all CPU EPs); static_qdq is more CPU-portable.",
    )
    ap.add_argument("--onnx-calib-split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--onnx-calib-samples", type=int, default=512, help="Calibration sample cap for static_qdq.")
    ap.add_argument("--onnx-calib-batch-size", type=int, default=64)
    ap.add_argument("--onnx-calib-num-workers", type=int, default=0)
    ap.add_argument("--onnx-opset", type=int, default=17)

    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--overfit-small", nargs="?", const=64, type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=50, help="Print training status every N batches")
    args = ap.parse_args()

    if args.history_len != 3:
        raise ValueError("Phase2e is fixed to history_len=3 (aux14 = 8 corners + 6 history).")
    if not (0.0 <= args.ss_start_pred_prob <= 1.0 and 0.0 <= args.ss_end_pred_prob <= 1.0):
        raise ValueError("scheduled sampling probabilities must be in [0,1].")

    export_onnx_only = bool(args.export_best_onnx or args.export_best_onnx_quantized)
    if not export_onnx_only and not args.dataset_root:
        raise ValueError(
            "--dataset-root is required unless --export-best-onnx or --export-best-onnx-quantized is used."
        )

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = MobileNetV3_RawPlusCorners_XY(corners_dim=14).to(device)

    if export_onnx_only:
        onnx_out = Path(args.export_best_onnx) if args.export_best_onnx else Path(args.export_best_onnx_quantized).with_suffix(".fp32.onnx")
        export_best_checkpoint_to_onnx(
            model=model,
            ckpt_arg=args.ckpt,
            save_dir=save_dir,
            out_path=onnx_out,
            device=device,
            out_h=args.out_h,
            out_w=args.out_w,
            onnx_opset=args.onnx_opset,
        )
        if args.export_best_onnx_quantized:
            _quantize_onnx(
                src_onnx=onnx_out,
                dst_onnx=Path(args.export_best_onnx_quantized),
                mode=args.onnx_quantization_mode,
                dataset_root=args.dataset_root,
                version=args.version,
                split=args.onnx_calib_split,
                out_w=args.out_w,
                out_h=args.out_h,
                history_len=args.history_len,
                require_touch_for_xy=args.require_touch_for_xy,
                calib_samples=args.onnx_calib_samples,
                calib_batch_size=args.onnx_calib_batch_size,
                calib_num_workers=args.onnx_calib_num_workers,
            )
        return

    train_tf = Phase2eTransform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="train",
        out_w=args.out_w,
        out_h=args.out_h,
        normalize=True,
        history_len=args.history_len,
        require_touch_for_xy=args.require_touch_for_xy,
    )
    val_tf = Phase2eTransform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split="val",
        out_w=args.out_w,
        out_h=args.out_h,
        normalize=True,
        history_len=args.history_len,
        require_touch_for_xy=args.require_touch_for_xy,
    )

    train_ds = InkMARCIndexedHistoryDataset(
        args.dataset_root,
        index_version=args.version,
        split="train",
        require_labels=True,
        history_len=args.history_len,
        transform=train_tf,
    )
    val_ds = InkMARCIndexedHistoryDataset(
        args.dataset_root,
        index_version=args.version,
        split="val",
        require_labels=True,
        history_len=args.history_len,
        transform=val_tf,
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

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.load_best_model and not args.eval_only:
        best_ckpt = save_dir / "checkpoint_best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"--load-best-model requested but checkpoint not found: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Warm-started model from best checkpoint: {best_ckpt}")

    if args.eval_only:
        eval_tf = Phase2eTransform_v1(
            dataset_root=Path(args.dataset_root),
            index_version=args.version,
            split=args.eval_split,
            out_w=args.out_w,
            out_h=args.out_h,
            normalize=True,
            history_len=args.history_len,
            require_touch_for_xy=args.require_touch_for_xy,
        )
        eval_ds = InkMARCIndexedHistoryDataset(
            args.dataset_root,
            index_version=args.version,
            split=args.eval_split,
            require_labels=True,
            history_len=args.history_len,
            transform=eval_tf,
        )
        eval_cap = args.eval_max_samples if args.eval_max_samples > 0 else args.overfit_small
        if eval_cap > 0:
            eval_ds = _maybe_subset(eval_ds, eval_cap)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        ckpt_path = resolve_ckpt_path(args.ckpt, save_dir)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)

        metrics = evaluate(model, eval_loader, device, args.out_w, args.out_h)
        print(f"Evaluated {ckpt_path} on split={args.eval_split}")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        use_amp = bool(args.amp and device.type == "cuda")

        def _infer_batch(batch):
            images = batch["image"].to(device, non_blocking=True)
            aux14 = batch["aux14"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(images, aux14)

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
        if args.save_onnx or args.save_onnx_quantized:
            onnx_out = Path(args.save_onnx) if args.save_onnx else (save_dir / "phase2e_eval_fp32.onnx")
            _export_model_to_onnx(
                model=model,
                out_path=onnx_out,
                device=device,
                out_h=args.out_h,
                out_w=args.out_w,
                onnx_opset=args.onnx_opset,
            )
            if args.save_onnx_quantized:
                _quantize_onnx(
                    src_onnx=onnx_out,
                    dst_onnx=Path(args.save_onnx_quantized),
                    mode=args.onnx_quantization_mode,
                    dataset_root=args.dataset_root,
                    version=args.version,
                    split=args.onnx_calib_split,
                    out_w=args.out_w,
                    out_h=args.out_h,
                    history_len=args.history_len,
                    require_touch_for_xy=args.require_touch_for_xy,
                    calib_samples=args.onnx_calib_samples,
                    calib_batch_size=args.onnx_calib_batch_size,
                    calib_num_workers=args.onnx_calib_num_workers,
                )
        _close_dataset(eval_ds)
        _close_dataset(train_ds)
        _close_dataset(val_ds)
        eval_tf.close()
        train_tf.close()
        val_tf.close()
        return

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running: Dict[str, float] = {}
        ss_prob = _scheduled_sampling_prob(
            epoch=epoch,
            epochs=args.epochs,
            p0=args.ss_start_pred_prob,
            p1=args.ss_end_pred_prob,
        )
        num_batches = max(1, len(train_loader))
        t_epoch = perf_counter()

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            corners = batch["corners"].to(device)
            history_gt = batch["history"].to(device)
            xy = batch["xy"].to(device)
            xy_mask = batch["xy_mask"].to(device)

            aux14 = _build_aux14_for_training(
                model=model,
                images=images,
                corners=corners,
                history_gt=history_gt,
                history_mode=args.history_mode,
                ss_pred_prob=ss_prob,
            )

            out = model(images, aux14)
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
            running["n_seen"] = running.get("n_seen", 0.0) + float(bs)

            should_log = (
                batch_idx == 1
                or batch_idx == num_batches
                or (args.log_interval > 0 and (batch_idx % args.log_interval == 0))
            )
            if should_log:
                seen = max(1.0, running["n_seen"])
                avg_loss = running["loss"] / seen
                avg_xy_mae = running["xy_mae"] / seen
                avg_xy_mae_px = running["xy_mae_px"] / seen
                elapsed = perf_counter() - t_epoch
                print(
                    f"epoch {epoch:02d} batch {batch_idx:04d}/{num_batches:04d} | "
                    f"loss={avg_loss:.4f} xy_mae={avg_xy_mae:.4f} xy_mae_px={avg_xy_mae_px:.2f} | "
                    f"history_mode={args.history_mode} ss_pred_prob={ss_prob:.3f} | "
                    f"elapsed={elapsed:.1f}s"
                )

        running.pop("n_seen", None)
        train_metrics = {k: v / len(train_ds) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, args.out_w, args.out_h)

        print(
            f"epoch {epoch:02d} | history_mode={args.history_mode} ss_pred_prob={ss_prob:.3f} | "
            f"train loss={train_metrics['loss']:.4f} xy_mae={train_metrics['xy_mae']:.4f} xy_mae_px={train_metrics['xy_mae_px']:.2f} | "
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

    _close_dataset(train_ds)
    _close_dataset(val_ds)
    train_tf.close()
    val_tf.close()

    if args.save_onnx or args.save_onnx_quantized:
        onnx_out = Path(args.save_onnx) if args.save_onnx else (save_dir / "phase2e_final_fp32.onnx")
        _export_model_to_onnx(
            model=model,
            out_path=onnx_out,
            device=device,
            out_h=args.out_h,
            out_w=args.out_w,
            onnx_opset=args.onnx_opset,
        )
        if args.save_onnx_quantized:
            _quantize_onnx(
                src_onnx=onnx_out,
                dst_onnx=Path(args.save_onnx_quantized),
                mode=args.onnx_quantization_mode,
                dataset_root=args.dataset_root,
                version=args.version,
                split=args.onnx_calib_split,
                out_w=args.out_w,
                out_h=args.out_h,
                history_len=args.history_len,
                require_touch_for_xy=args.require_touch_for_xy,
                calib_samples=args.onnx_calib_samples,
                calib_batch_size=args.onnx_calib_batch_size,
                calib_num_workers=args.onnx_calib_num_workers,
            )


if __name__ == "__main__":
    main()
