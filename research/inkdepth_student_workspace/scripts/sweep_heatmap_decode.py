#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import build_model_from_config
from inkdepth.training.metrics import xy_metric_spec, xy_valid_mask
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep XY decoding methods from location heatmap logits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--temperatures", default="1.0,0.75,0.5,0.35,0.25,0.15,0.10,0.075,0.05")
    parser.add_argument("--topk", default="4,9,16,25,49,81,121")
    parser.add_argument("--local-radii", default="1,2,3,4,5,7")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    if args.batch_size is not None:
        cfg.training.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.training.num_workers = int(args.num_workers)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    ds = StylusDataset(cfg, split=args.split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stylus_collate_fn,
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", False)) and int(cfg.training.num_workers) > 0,
    )

    temps = parse_float_list(args.temperatures)
    topks = parse_int_list(args.topk)
    radii = parse_int_list(args.local_radii)
    methods = build_method_names(temps, topks, radii)

    xy_scale, _ = xy_metric_spec(cfg)
    accum = {name: {"touch_true": [], "touch_false": [], "all_valid": []} for name in methods}
    prediction_rows: list[dict[str, Any]] = []

    for batch_i, batch in enumerate(tqdm(loader, desc="predict")):
        if args.max_batches and batch_i >= int(args.max_batches):
            break
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(batch["image"])
        heatmap = outputs.get("location_heatmap")
        if heatmap is None:
            raise RuntimeError("Model produced no location_heatmap. Set model.train_location=true.")
        model_xy = outputs.get("xy")
        if model_xy is None:
            raise RuntimeError("Model produced no xy output. Set model.train_location=true.")

        decoded: dict[str, torch.Tensor] = {"model_softargmax": model_xy.detach()}
        for temp in temps:
            decoded[f"softargmax_t{fmt_num(temp)}"] = soft_argmax(heatmap, temperature=temp)
        decoded["argmax"] = argmax_xy(heatmap)
        for k in topks:
            decoded[f"top{k}_com_t1"] = topk_com(heatmap, k=k, temperature=1.0)
        for radius in radii:
            decoded[f"local_r{radius}_com_t1"] = local_com(heatmap, radius=radius, temperature=1.0)

        valid = xy_valid_mask(batch, cfg)
        touch_valid = batch.get("has_touch", batch.get("valid_touch")).bool()
        touch_true = batch["touch"].float() > 0.5
        target = batch["xy"]
        scale = torch.tensor(xy_scale, device=device, dtype=target.dtype).view(1, 2)

        for name, xy in decoded.items():
            err = torch.linalg.norm((xy - target) * scale, dim=1)
            all_mask = valid
            true_mask = valid & touch_valid & touch_true
            false_mask = valid & touch_valid & ~touch_true
            accum[name]["all_valid"].extend(err[all_mask].detach().cpu().tolist())
            accum[name]["touch_true"].extend(err[true_mask].detach().cpu().tolist())
            accum[name]["touch_false"].extend(err[false_mask].detach().cpu().tolist())

        if args.write_predictions:
            append_prediction_rows(prediction_rows, batch, decoded, target, valid, touch_valid, touch_true, xy_scale)

    summary = []
    for name in methods:
        row = {"method": name}
        for group in ("touch_true", "touch_false", "all_valid"):
            row.update(prefix(group, summarize(accum[name][group])))
        summary.append(row)
    summary.sort(key=lambda row: (row.get("touch_true_median_px", float("inf")), row.get("touch_true_mean_px", float("inf"))))

    write_csv(out_dir / "heatmap_decode_sweep.csv", summary)
    (out_dir / "heatmap_decode_sweep.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if args.write_predictions:
        write_csv(out_dir / "heatmap_decode_predictions.csv", prediction_rows)

    best = summary[0]
    print(
        "best "
        f"{best['method']} touch_true mean={best['touch_true_mean_px']:.4f} "
        f"median={best['touch_true_median_px']:.4f} count={best['touch_true_count']}"
    )
    print("top methods:")
    for row in summary[:12]:
        print(
            f"{row['method']}: "
            f"touch_true mean={row['touch_true_mean_px']:.4f} median={row['touch_true_median_px']:.4f}; "
            f"touch_false mean={row['touch_false_mean_px']:.4f} median={row['touch_false_median_px']:.4f}"
        )


def build_method_names(temps: list[float], topks: list[int], radii: list[int]) -> list[str]:
    names = ["model_softargmax"]
    names.extend(f"softargmax_t{fmt_num(temp)}" for temp in temps)
    names.append("argmax")
    names.extend(f"top{k}_com_t1" for k in topks)
    names.extend(f"local_r{radius}_com_t1" for radius in radii)
    return list(dict.fromkeys(names))


def soft_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    b, _, h, w = logits.shape
    flat = (logits / max(float(temperature), 1e-6)).reshape(b, -1)
    prob = torch.softmax(flat, dim=1).reshape(b, h, w)
    xs, ys = coordinate_grids(logits)
    x = (prob * xs.view(1, 1, w)).sum(dim=(1, 2))
    y = (prob * ys.view(1, h, 1)).sum(dim=(1, 2))
    return torch.stack([x, y], dim=1)


def argmax_xy(logits: torch.Tensor) -> torch.Tensor:
    b, _, h, w = logits.shape
    index = logits.flatten(1).argmax(dim=1)
    y = torch.div(index, w, rounding_mode="floor").to(dtype=logits.dtype) / max(1, h - 1)
    x = (index % w).to(dtype=logits.dtype) / max(1, w - 1)
    return torch.stack([x, y], dim=1)


def topk_com(logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    b, _, h, w = logits.shape
    flat = logits.flatten(1)
    k = min(max(1, int(k)), flat.shape[1])
    vals, indices = torch.topk(flat, k=k, dim=1)
    weights = torch.softmax(vals / max(float(temperature), 1e-6), dim=1)
    y_idx = torch.div(indices, w, rounding_mode="floor").to(dtype=logits.dtype)
    x_idx = (indices % w).to(dtype=logits.dtype)
    x = (weights * (x_idx / max(1, w - 1))).sum(dim=1)
    y = (weights * (y_idx / max(1, h - 1))).sum(dim=1)
    return torch.stack([x, y], dim=1)


def local_com(logits: torch.Tensor, radius: int, temperature: float) -> torch.Tensor:
    b, _, h, w = logits.shape
    flat = logits.flatten(1)
    peak = flat.argmax(dim=1)
    peak_y = torch.div(peak, w, rounding_mode="floor")
    peak_x = peak % w
    xs, ys = coordinate_grids(logits)
    yy = torch.arange(h, device=logits.device).view(1, h, 1)
    xx = torch.arange(w, device=logits.device).view(1, 1, w)
    mask = (yy - peak_y.view(b, 1, 1)).abs().le(radius) & (xx - peak_x.view(b, 1, 1)).abs().le(radius)
    masked = logits[:, 0].masked_fill(~mask, torch.finfo(logits.dtype).min)
    prob = torch.softmax((masked / max(float(temperature), 1e-6)).flatten(1), dim=1).reshape(b, h, w)
    x = (prob * xs.view(1, 1, w)).sum(dim=(1, 2))
    y = (prob * ys.view(1, h, 1)).sum(dim=(1, 2))
    return torch.stack([x, y], dim=1)


def coordinate_grids(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, h, w = logits.shape
    xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    return xs, ys


def append_prediction_rows(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
    decoded: dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    touch_valid: torch.Tensor,
    touch_true: torch.Tensor,
    xy_scale: tuple[float, float],
) -> None:
    scale = torch.tensor(xy_scale, device=target.device, dtype=target.dtype).view(1, 2)
    decoded_cpu = {name: xy.detach().cpu().numpy() for name, xy in decoded.items()}
    target_np = target.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    touch_valid_np = touch_valid.detach().cpu().numpy().astype(bool)
    touch_true_np = touch_true.detach().cpu().numpy().astype(bool)
    for i in range(target_np.shape[0]):
        row: dict[str, Any] = {
            "video_id": str(batch["video_id"][i]),
            "frame_index": int(batch["frame_index"][i].detach().cpu()),
            "has_touch": bool(touch_valid_np[i]),
            "touch_true": int(touch_true_np[i]) if touch_valid_np[i] else "",
            "valid_xy": bool(valid_np[i]),
            "true_x": float(target_np[i, 0]) if valid_np[i] else "",
            "true_y": float(target_np[i, 1]) if valid_np[i] else "",
        }
        for name, xy_np in decoded_cpu.items():
            row[f"{name}_x"] = float(xy_np[i, 0])
            row[f"{name}_y"] = float(xy_np[i, 1])
            if valid_np[i]:
                diff = (torch.tensor(xy_np[i], device=target.device, dtype=target.dtype) - target[i]) * scale.squeeze(0)
                row[f"{name}_err_px"] = float(torch.linalg.norm(diff).detach().cpu())
            else:
                row[f"{name}_err_px"] = ""
        rows.append(row)


def summarize(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"count": 0, "mean_px": 0.0, "median_px": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"count": int(arr.size), "mean_px": float(arr.mean()), "median_px": float(np.median(arr))}


def prefix(name: str, values: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{name}_{key}": value for key, value in values.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields = list(dict.fromkeys([*fields, *row.keys()]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def fmt_num(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}


if __name__ == "__main__":
    main()
