#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import build_model_from_config
from inkdepth.models.losses import MultiTaskLoss
from inkdepth.utils.config import load_config, parse_overrides
from inkdepth.utils.visualization import save_debug_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one StylusDataset batch")
    parser.add_argument("--config", default=str(ROOT / "configs/inkdepth_mobilenetv3.yaml"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--all-splits", action="store_true", help="Inspect one batch from train, val, and test")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--forward", action="store_true", help="Also run a model forward pass")
    parser.add_argument("--compute-loss", action="store_true")
    parser.add_argument("--save-panel", default=None, help="Optional debug panel path")
    parser.add_argument("--check-missing-h5", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    if args.synthetic:
        overrides["data.synthetic.enabled"] = True
        overrides.setdefault("model.pretrained", False)
        overrides.setdefault("training.num_workers", 0)
    cfg = load_config(args.config, overrides)
    splits = ["train", "val", "test"] if args.all_splits else [args.split]
    for split in splits:
        inspect_split(cfg, split, args)


def inspect_split(cfg, split: str, args: argparse.Namespace) -> None:
    ds = StylusDataset(cfg, split=split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=stylus_collate_fn,
    )
    batch = next(iter(loader))
    print(f"\n[{split}] samples: {len(ds)}")
    for key, value in batch.items():
        if torch.is_tensor(value):
            extra = ""
            if value.dtype == torch.bool:
                extra = f" true={int(value.sum())}/{value.numel()}"
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}{extra}")
        else:
            print(f"{key}: {type(value).__name__} len={len(value)}")
    if args.check_missing_h5 and not ds.synthetic:
        missing = [i for i, r in enumerate(ds.records) if r.get("label_h5_path") and not Path(r["label_h5_path"]).exists()]
        if missing:
            sample = ds[missing[0]]
            print(
                "missing_h5_check: ok "
                f"video={sample['video_id']} frame={sample['frame_index']} "
                f"has_labels={sample['has_labels']} has_depth={sample['has_depth']}"
            )
        else:
            print("missing_h5_check: no missing H5 records in this split/subset")
    if args.forward or args.compute_loss or args.save_panel:
        model = build_model_from_config(cfg).eval()
        with torch.no_grad():
            out = model(batch["image"])
        for key, value in out.items():
            if torch.is_tensor(value):
                print(f"out.{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            else:
                print(f"out.{key}: {value}")
        if args.compute_loss:
            loss, logs = MultiTaskLoss(cfg)(out, batch)
            print(f"loss: {float(loss):.6f}")
            for key, value in sorted(logs.items()):
                print(f"{key}: {value:.6f}")
        if args.save_panel:
            panel_path = Path(args.save_panel)
            if len(splits := (["train", "val", "test"] if args.all_splits else [args.split])) > 1:
                panel_path = panel_path.with_name(f"{panel_path.stem}_{split}{panel_path.suffix}")
            save_debug_panel(
                panel_path,
                batch["image"][0],
                batch["depth"][0],
                out["depth"][0],
                None if out.get("location_heatmap") is None else out["location_heatmap"][0],
                None if out.get("xy") is None else out["xy"][0],
            )
            print(f"saved panel: {panel_path}")


if __name__ == "__main__":
    main()
