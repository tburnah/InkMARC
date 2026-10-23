from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.training.calibration import evaluate_calibration, write_calibration_report, write_thresholds
from ml.inkloc.training.losses import LocTouchLoss
from ml.inkloc.training.train import build_dataset, format_metrics
from ml.inkloc.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/loctouch_lite.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="ml/artifacts/calibration/loctouch_lite_report.json")
    parser.add_argument("--thresholds-out", default="ml/artifacts/calibration/loctouch_lite_thresholds.json")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model = LocTouchNetLite.from_config(cfg).to(device).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    num_workers = int(train_cfg.get("num_workers", 0))
    val_ds = build_dataset(cfg, split="eval")
    loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    loss_fn = LocTouchLoss.from_config(cfg)
    report = evaluate_calibration(model, loss_fn, loader, device, max_batches=args.max_batches)
    report["checkpoint"] = args.checkpoint
    report["config"] = args.config
    report["val_size"] = len(val_ds)
    write_calibration_report(report, args.out)
    write_thresholds(report, args.thresholds_out)
    print(format_metrics("val", report["losses"], report))
    print(json.dumps({
        "confusion_matrix": report["confusion_matrix"],
        "touch_best": report["touch_best"],
        "thresholds_out": args.thresholds_out,
        "report_out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
