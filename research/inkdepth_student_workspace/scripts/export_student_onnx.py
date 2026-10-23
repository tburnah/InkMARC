#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.utils.config import load_config, parse_overrides
from inkdepth.utils.export_onnx import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export InkDepthStudent to ONNX")
    parser.add_argument("--config", default=str(ROOT / "configs/inkdepth_mobilenetv3.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-depth", action="store_true", help="Do not include low-res depth output")
    parser.add_argument("--include-heatmap", action="store_true", help="Include location heatmap output")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, parse_overrides(args.override))
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best_val_loss.pt")
    output = args.output or str(Path(cfg.output_dir) / "inkdepth_student.onnx")
    diffs = export_onnx(
        cfg,
        checkpoint,
        output,
        opset=args.opset,
        include_depth=not args.no_depth,
        include_heatmap=args.include_heatmap,
    )
    print(f"exported: {output}")
    for key, value in sorted(diffs.items()):
        print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
