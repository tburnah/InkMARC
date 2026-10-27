#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.training.train import train
from inkdepth.utils.config import load_config
from inkdepth.utils.export_onnx import export_onnx


LOCATION_CONFIG = ROOT / "configs/final_rgb_geometry_location_base_plus_texture.yaml"
TOUCH_CONFIG = ROOT / "configs/final_rgb_geometry_touch_base_only.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export the depth-free RGB+geometry location/touch model."
    )
    parser.add_argument("--location-config", default=str(LOCATION_CONFIG))
    parser.add_argument("--touch-config", default=str(TOUCH_CONFIG))
    parser.add_argument("--skip-location", action="store_true")
    parser.add_argument("--skip-touch", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run two batches per stage on synthetic data.")
    parser.add_argument("--output", default=None, help="Final ONNX path.")
    args = parser.parse_args()

    location_cfg = load_config(args.location_config)
    touch_cfg = load_config(args.touch_config)
    if args.smoke_test:
        _make_smoke_config(location_cfg, ROOT / "outputs/smoke_fast_rgb_location")
        _make_smoke_config(touch_cfg, ROOT / "outputs/smoke_fast_rgb_touch")
        touch_cfg.training.init_checkpoint = str(Path(location_cfg.output_dir) / "best_xy.pt")

    if not args.skip_location:
        train(location_cfg)

    location_checkpoint = Path(location_cfg.output_dir) / "best_xy.pt"
    touch_cfg.training.init_checkpoint = str(location_checkpoint)
    if not args.skip_touch:
        if not location_checkpoint.exists():
            raise FileNotFoundError(
                f"Location checkpoint not found: {location_checkpoint}. Run without --skip-location first."
            )
        train(touch_cfg)

    if args.skip_export:
        return
    final_checkpoint = _first_existing(
        Path(touch_cfg.output_dir) / "best_touch.pt",
        Path(touch_cfg.output_dir) / "best_val_loss.pt",
        Path(touch_cfg.output_dir) / "latest.pt",
    )
    output = Path(args.output) if args.output else Path(touch_cfg.output_dir) / "inktrack_rgb_geometry.onnx"
    diffs = export_onnx(
        touch_cfg,
        final_checkpoint,
        output,
        include_depth=False,
        include_heatmap=False,
    )
    print(f"exported: {output}")
    for key, value in sorted(diffs.items()):
        print(f"{key}: {value:.8f}")


def _make_smoke_config(cfg, output_dir: Path) -> None:
    cfg.output_dir = str(output_dir)
    cfg.data.synthetic.enabled = True
    cfg.data.synthetic.length = 8
    cfg.model.pretrained = False
    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.prefetch_factor = None
    cfg.training.max_steps_per_epoch = 2
    cfg.training.copy_manifest = False


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"No final checkpoint found in: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
