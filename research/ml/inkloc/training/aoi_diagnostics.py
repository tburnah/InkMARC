from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.training.forward import model_forward
from ml.inkloc.training.train import build_dataset
from ml.inkloc.utils.config import load_config


def _binary_metrics(score: torch.Tensor, target: torch.Tensor, threshold: float) -> Dict[str, float]:
    pred = score >= threshold
    truth = target.bool()
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    tn = int((~pred & ~truth).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _aoi_mask(image: torch.Tensor, xy: torch.Tensor, radius: int) -> torch.Tensor:
    b, _c, h, w = image.shape
    cx = (xy[:, 0] * (w - 1)).round().long().clamp(0, w - 1)
    cy = (xy[:, 1] * (h - 1)).round().long().clamp(0, h - 1)
    yy = torch.arange(h, device=image.device).view(1, h, 1)
    xx = torch.arange(w, device=image.device).view(1, 1, w)
    mask = (xx - cx.view(b, 1, 1)).abs() <= radius
    mask = mask & ((yy - cy.view(b, 1, 1)).abs() <= radius)
    return mask.unsqueeze(1).to(image.dtype)


def make_variant(image: torch.Tensor, xy: torch.Tensor, radius: int, mode: str, fill: float = 0.0) -> torch.Tensor:
    if mode == "normal":
        return image
    mask = _aoi_mask(image, xy, radius)
    fill_t = image.new_full(image.shape, fill)
    if mode == "aoi_only":
        return image * mask + fill_t * (1.0 - mask)
    if mode in ("aoi_masked", "background_only"):
        return image * (1.0 - mask) + fill_t * mask
    raise ValueError(f"Unknown AOI mode: {mode}")


def summarize(
    *,
    model: LocTouchNetLite,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    mode: str,
    radius: int,
    coord_source: str,
    ablate_local: bool = False,
    ablate_global: bool = False,
    ablate_motion: bool = False,
    touch_ablation_mode: str = "rgb_motion",
    max_batches: int = 0,
) -> Dict:
    state_chunks = []
    pred_state_chunks = []
    touch_score_chunks = []
    xy_chunks = []
    xy_target_chunks = []
    has_xy_chunks = []
    with torch.no_grad():
        for bi, batch in enumerate(loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            image = batch["image"]
            base_batch = dict(batch)
            base_batch["image"] = image
            normal = model_forward(model, base_batch, touch_ablation_mode=touch_ablation_mode)
            if coord_source == "gt":
                xy_for_aoi = torch.where(batch["has_xy"].view(-1, 1) > 0.5, batch["xy"], normal["xy"].detach())
            elif coord_source == "pred":
                xy_for_aoi = normal["xy"].detach()
            else:
                raise ValueError("--coord-source must be gt or pred")
            variant = make_variant(image, xy_for_aoi, radius, mode)
            if mode == "normal" and not ablate_local and not ablate_global and not ablate_motion:
                outputs = normal
            else:
                variant_batch = dict(batch)
                variant_batch["image"] = variant
                outputs = model_forward(
                    model,
                    variant_batch,
                    touch_ablation_mode=touch_ablation_mode,
                    ablate_local_token=ablate_local,
                    ablate_global_token=ablate_global,
                    ablate_motion_token=ablate_motion,
                )
            state_chunks.append(batch["state"].detach().cpu().long())
            pred_state_chunks.append(outputs["state_probs"].argmax(dim=1).detach().cpu().long())
            touch_score_chunks.append(outputs["state_probs"][:, 2].detach().cpu())
            xy_chunks.append(outputs["xy"].detach().cpu())
            xy_target_chunks.append(batch["xy"].detach().cpu())
            has_xy_chunks.append(batch["has_xy"].detach().cpu().bool())
            if max_batches > 0 and bi >= max_batches:
                break

    state = torch.cat(state_chunks)
    pred_state = torch.cat(pred_state_chunks)
    touch_score = torch.cat(touch_score_chunks)
    xy = torch.cat(xy_chunks)
    xy_target = torch.cat(xy_target_chunks)
    has_xy = torch.cat(has_xy_chunks)
    touch_true = state == 2
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(state, pred_state):
        confusion[int(t), int(p)] += 1
    xy_px = torch.linalg.vector_norm((xy - xy_target) * torch.tensor([480.0, 320.0]), dim=1)
    touch = _binary_metrics(touch_score, touch_true, threshold)
    return {
        "mode": mode,
        "radius": int(radius),
        "coord_source": coord_source,
        "ablate_local_token": bool(ablate_local),
        "ablate_global_token": bool(ablate_global),
        "ablate_motion_token": bool(ablate_motion),
        "touch_ablation_mode": touch_ablation_mode,
        "num_samples": int(state.numel()),
        "state_accuracy": float((state == pred_state).float().mean().item()),
        "touch_precision": touch["precision"],
        "touch_recall": touch["recall"],
        "touch_f1": touch["f1"],
        "confusion_matrix": confusion.tolist(),
        "xy_mean_pixel_error": float(xy_px[has_xy].mean().item()) if bool(has_xy.any()) else float("nan"),
        "mean_touch_probability_true_touch": float(touch_score[touch_true].mean().item()) if bool(touch_true.any()) else float("nan"),
        "mean_touch_probability_non_touch": float(touch_score[~touch_true].mean().item()) if bool((~touch_true).any()) else float("nan"),
    }


def add_deltas(reports: Iterable[Dict], normal: Dict) -> list[Dict]:
    out = []
    for report in reports:
        item = dict(report)
        item["delta_touch_f1_vs_normal"] = item["touch_f1"] - normal["touch_f1"]
        item["delta_xy_px_vs_normal"] = item["xy_mean_pixel_error"] - normal["xy_mean_pixel_error"]
        item["delta_state_acc_vs_normal"] = item["state_accuracy"] - normal["state_accuracy"]
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/loctouch_lite.yaml")
    parser.add_argument("--checkpoint", default="ml/artifacts/checkpoints/selected.pt")
    parser.add_argument("--thresholds", default="ml/artifacts/calibration/loctouch_lite_thresholds.json")
    parser.add_argument("--out", default="ml/artifacts/diagnostics/aoi_diagnostics_selected.json")
    parser.add_argument("--radii", nargs="+", type=int, default=[32, 48, 64, 96])
    parser.add_argument("--coord-source", choices=["gt", "pred"], default="gt")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    threshold_doc = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    threshold = float(threshold_doc["touch_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model = LocTouchNetLite.from_config(cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_ds = build_dataset(cfg, split="eval")
    loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(train_cfg.get("num_workers", 0)) > 0,
    )

    reports = []
    normal = summarize(
        model=model,
        loader=loader,
        device=device,
        threshold=threshold,
        mode="normal",
        radius=0,
        coord_source=args.coord_source,
        max_batches=args.max_batches,
    )
    reports.append(normal)
    for radius in args.radii:
        for mode in ["aoi_only", "aoi_masked", "background_only"]:
            reports.append(summarize(
                model=model,
                loader=loader,
                device=device,
                threshold=threshold,
                mode=mode,
                radius=radius,
                coord_source=args.coord_source,
                max_batches=args.max_batches,
            ))
    for ablate_local, ablate_global, ablate_motion, name in [
        (True, False, False, "token_no_local"),
        (False, True, False, "token_no_global"),
        (False, False, True, "token_no_motion"),
    ]:
        report = summarize(
            model=model,
            loader=loader,
            device=device,
            threshold=threshold,
            mode="normal",
            radius=0,
            coord_source=args.coord_source,
            ablate_local=ablate_local,
            ablate_global=ablate_global,
            ablate_motion=ablate_motion,
            max_batches=args.max_batches,
        )
        report["mode"] = name
        reports.append(report)

    payload = {
        "checkpoint": args.checkpoint,
        "threshold": threshold,
        "coord_source": args.coord_source,
        "reports": add_deltas(reports, normal),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
