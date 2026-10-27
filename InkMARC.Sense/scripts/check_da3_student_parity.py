#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset, stylus_collate_fn
from inkdepth.models import build_model_from_config
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DA3 wrapper outputs against deployable RGBD+geometry student outputs.")
    parser.add_argument("--combined-config", required=True)
    parser.add_argument("--combined-checkpoint", required=True)
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    combined_cfg = load_config(args.combined_config, overrides)
    student_cfg = load_config(args.student_config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combined = build_model_from_config(combined_cfg).to(device).eval()
    student = build_model_from_config(student_cfg).to(device).eval()
    load_checkpoint(args.combined_checkpoint, combined, map_location=device)
    load_checkpoint(args.student_checkpoint, student, map_location=device)

    ds = StylusDataset(combined_cfg, split=args.split, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(combined_cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=stylus_collate_fn,
    )

    diffs: dict[str, list[float]] = {"state_logits": [], "touch_logits": [], "xy": []}
    seen = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        combined_out = combined(image)
        student_input = torch.cat([image[:, :3], combined_out["depth"], image[:, 3:]], dim=1)
        student_out = student(student_input)
        for key in diffs:
            a = combined_out[key]
            b = student_out[key]
            if a is None or b is None:
                raise RuntimeError(f"Missing output {key}: combined={a is not None} student={b is not None}")
            diffs[key].append(float((a - b).abs().max().detach().cpu()))
        seen += 1
        if seen >= args.batches:
            break

    print(f"checked_batches={seen}")
    for key, values in diffs.items():
        tensor = torch.tensor(values)
        print(f"{key}/max_abs_diff_max: {float(tensor.max()):.8f}")
        print(f"{key}/max_abs_diff_mean: {float(tensor.mean()):.8f}")


if __name__ == "__main__":
    main()
