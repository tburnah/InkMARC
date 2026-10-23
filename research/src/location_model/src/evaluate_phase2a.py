#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.dataset_core import InkMARCIndexedDataset
from data.transforms.phase2_transform_v1 import Phase2Transform_v1


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


def _resolve_provider(mode: str) -> List[str]:
    mode = mode.lower()
    available = set(ort.get_available_providers())
    if mode == "cpu":
        return ["CPUExecutionProvider"]
    if mode == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in this onnxruntime build. "
                f"Available providers: {sorted(available)}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if mode == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    raise ValueError(f"Unknown --provider={mode}. Use auto|cpu|cuda.")


def _create_session_with_fallback(onnx_path: Path, providers: List[str], provider_mode: str) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        msg = str(e)
        convinteger_issue = ("ConvInteger" in msg) or ("NOT_IMPLEMENTED" in msg)
        wants_gpu = provider_mode in {"auto", "cuda"} and ("CUDAExecutionProvider" in providers)
        if convinteger_issue and wants_gpu:
            print(
                "ONNX Runtime could not run this graph on CUDA (likely int8 ConvInteger). "
                "Falling back to CPUExecutionProvider."
            )
            try:
                return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            except Exception as cpu_e:
                cpu_msg = str(cpu_e)
                if ("ConvInteger" in cpu_msg) or ("NOT_IMPLEMENTED" in cpu_msg):
                    fp32_sibling = None
                    n = onnx_path.name
                    if "_int8" in n:
                        candidate = onnx_path.with_name(n.replace("_int8", "_fp32"))
                        if candidate.exists():
                            fp32_sibling = candidate
                    hint = (
                        "This int8 ONNX uses ConvInteger ops that are not supported by the current "
                        "onnxruntime build/providers on this machine. "
                        "Use the fp32 ONNX for evaluation, or regenerate a compatible quantized model "
                        "from fp32 by quantizing only MatMul/Gemm ops."
                    )
                    if fp32_sibling is not None:
                        hint += f" Detected fp32 sibling: {fp32_sibling}"
                    raise RuntimeError(hint) from cpu_e
                raise
        raise


def _select_outputs(session: ort.InferenceSession) -> Tuple[str, str]:
    outs = session.get_outputs()
    out_names = [o.name for o in outs]

    if "xy" in out_names and "heatmap_logits" in out_names:
        return "xy", "heatmap_logits"

    xy_name = None
    logits_name = None
    for o in outs:
        shape = o.shape
        if len(shape) == 2 and shape[-1] == 2:
            xy_name = o.name
        if len(shape) == 4:
            logits_name = o.name

    if xy_name is None or logits_name is None:
        raise RuntimeError(
            "Could not infer ONNX output names for phase2a model. "
            f"Expected outputs with names ['xy','heatmap_logits'] or shapes like (B,2) and (B,1,H,W). "
            f"Found outputs: {[(o.name, o.shape) for o in outs]}"
        )
    return xy_name, logits_name


def _masked_smoothl1_xy(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=1)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den


def _masked_kl_heatmap(logits: torch.Tensor, target_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.flatten(2), dim=-1)
    tgt = target_prob.flatten(2)
    per = F.kl_div(logp, tgt, reduction="none").sum(dim=-1).squeeze(1)
    den = mask.sum().clamp_min(1.0)
    return (per * mask).sum() / den


@torch.no_grad()
def evaluate_onnx(
    session: ort.InferenceSession,
    input_name: str,
    xy_name: str,
    logits_name: str,
    loader: DataLoader,
    out_w: int,
    out_h: int,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    n = 0
    nde_chunks: List[torch.Tensor] = []
    scale = torch.tensor([out_w, out_h], dtype=torch.float32)

    for batch in loader:
        images = batch["image"].to(torch.float32)
        xy = batch["xy"].to(torch.float32)
        xy_mask = batch["xy_mask"].to(torch.float32)
        heatmap = batch["heatmap"].to(torch.float32)
        heatmap_mask = batch["heatmap_mask"].to(torch.float32)

        outputs = session.run(
            [xy_name, logits_name],
            {input_name: images.cpu().numpy()},
        )
        xy_pred = torch.from_numpy(outputs[0]).to(torch.float32)
        logits = torch.from_numpy(outputs[1]).to(torch.float32)

        loss_xy = _masked_smoothl1_xy(xy_pred, xy, xy_mask)
        loss_hm = _masked_kl_heatmap(logits, heatmap, heatmap_mask)
        loss = loss_xy + loss_hm

        abs_err = (xy_pred - xy).abs()
        mae = abs_err.mean(dim=1)
        den = xy_mask.sum().clamp_min(1.0)
        xy_mae = (mae * xy_mask).sum() / den

        abs_err_px = (abs_err * scale).mean(dim=1)
        xy_mae_px = (abs_err_px * xy_mask).sum() / den

        eu = torch.sqrt(((xy_pred - xy) ** 2).sum(dim=1))
        sel = eu[xy_mask > 0.5]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to a phase2a ONNX model.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max-samples", type=int, default=0, help="0 = evaluate all samples in split.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--hm-w", type=int, default=60)
    ap.add_argument("--hm-h", type=int, default=40)
    ap.add_argument("--hm-sigma", type=float, default=1.5)
    ap.add_argument(
        "--require-touch-for-xy",
        action="store_true",
        help="Evaluate localization only on touch frames.",
    )
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    providers = _resolve_provider(args.provider)
    session = _create_session_with_fallback(onnx_path, providers, args.provider)
    input_name = session.get_inputs()[0].name
    xy_name, logits_name = _select_outputs(session)

    tf = Phase2Transform_v1(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split=args.split,
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        normalize=True,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=args.require_touch_for_xy,
    )

    try:
        ds = InkMARCIndexedDataset(
            args.dataset_root,
            index_version=args.version,
            split=args.split,
            require_labels=True,
            transform=tf,
        )
        ds = _maybe_subset(ds, args.max_samples)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=bool(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

        metrics = evaluate_onnx(
            session=session,
            input_name=input_name,
            xy_name=xy_name,
            logits_name=logits_name,
            loader=loader,
            out_w=args.out_w,
            out_h=args.out_h,
        )

        print(
            f"Evaluated ONNX={onnx_path} split={args.split} "
            f"samples={len(ds)} provider={providers[0]}"
        )
        print(f"ONNX input={input_name}, outputs=(xy={xy_name}, heatmap_logits={logits_name})")
        for k in ["loss", "loss_xy", "loss_hm", "xy_mae", "xy_mae_px", "nde_median"]:
            print(f"{k}: {metrics.get(k, float('nan'))}")
    finally:
        tf.close()


if __name__ == "__main__":
    main()
