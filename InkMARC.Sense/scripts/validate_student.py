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
from inkdepth.training.validate import validate
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate InkMARC-Sense model")
    parser.add_argument("--config", default=str(ROOT / "configs/inkdepth_mobilenetv3.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, parse_overrides(args.override))
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "latest.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)
    ds = StylusDataset(cfg, split=args.split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=stylus_collate_fn,
    )
    metrics = validate(model, loader, cfg, device, save_visuals=args.save_visuals, output_dir=cfg.output_dir)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
