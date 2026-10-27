from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.dataset_clip import InkMARCIndexedClipDataset
from data.transforms.phase4_transform_clip_v2 import Phase4TransformClip_v2
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--clip-len", type=int, default=5)

    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt (e.g., checkpoint_best.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--margin-frac", type=float, default=0.10)
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--hm-w", type=int, default=60)
    ap.add_argument("--hm-h", type=int, default=40)
    ap.add_argument("--hm-sigma", type=float, default=1.5)

    ap.add_argument("--subset", type=int, default=0, help="If >0, evaluate only first N clips of the split")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Build transform + dataset
    tf = Phase4TransformClip_v2(
        dataset_root=Path(args.dataset_root),
        index_version=args.version,
        split=args.split,
        out_w=args.out_w,
        out_h=args.out_h,
        margin_frac=args.margin_frac,
        hm_w=args.hm_w,
        hm_h=args.hm_h,
        hm_sigma=args.hm_sigma,
        require_touch_for_xy=False,
    )

    try:
        ds = InkMARCIndexedClipDataset(
            args.dataset_root,
            index_version=args.version,
            split=args.split,
            clip_len=args.clip_len,
            require_labels_last=True,
            transform=tf,
        )
        ds = _maybe_subset(ds, args.subset)

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

        # Load checkpoint
        ckpt_path = Path(args.checkpoint)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        hm_h = int(ckpt_args.get("hm_h", args.hm_h))
        hm_w = int(ckpt_args.get("hm_w", args.hm_w))
        tsm_fold_div = int(ckpt_args.get("tsm_fold_div", 8))

        model = MobileNetV3_TSM_HeatmapDSNT(
            hm_h=hm_h,
            hm_w=hm_w,
            tsm_fold_div=tsm_fold_div,
        ).to(device)

        # Support both key names if you ever change them
        state = None
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state", None) or ckpt.get("state_dict", None)
        if state is None:
            raise ValueError(f"Checkpoint {ckpt_path} does not contain model_state/state_dict keys.")

        model.load_state_dict(state, strict=True)

        metrics = evaluate(model, loader, device, args.out_w, args.out_h)

        epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
        print(f"checkpoint: {ckpt_path}")
        if epoch is not None:
            print(f"epoch: {epoch}")
        print(f"split: {args.split} | clips: {len(ds)}")
        print(
            "metrics | "
            f"loss={metrics.get('loss', float('nan')):.4f} "
            f"(xy={metrics.get('loss_xy', float('nan')):.4f} hm={metrics.get('loss_hm', float('nan')):.4f}) | "
            f"nde_med={metrics.get('nde_median', float('nan')):.4f} "
            f"xy_mae={metrics.get('xy_mae', float('nan')):.4f} "
            f"xy_mae_px={metrics.get('xy_mae_px', float('nan')):.2f}"
        )

    finally:
        tf.close()


if __name__ == "__main__":
    main()
