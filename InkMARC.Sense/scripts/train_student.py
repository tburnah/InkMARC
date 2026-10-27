#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.training.train import train
from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InkMARCSense")
    parser.add_argument("--config", default=str(ROOT / "configs/inkdepth_mobilenetv3.yaml"))
    parser.add_argument("--override", nargs="*", default=[], help="Dotted YAML overrides, e.g. training.phase=depth_pretrain")
    args = parser.parse_args()
    cfg = load_config(args.config, parse_overrides(args.override))
    metrics = train(cfg)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
