#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.models import build_model_from_config
from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract InkDepthStudent weights from an InkDepthDA3Student checkpoint.")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--student-config", required=True, help="Deployable InkDepthStudent config used to validate shapes.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.student_config, parse_overrides(args.override))
    model = build_model_from_config(cfg)
    current = model.state_dict()
    payload = torch.load(args.source_checkpoint, map_location="cpu")
    source_state = payload.get("model", payload)
    extracted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source_state.items():
        if not key.startswith("student."):
            continue
        student_key = key.removeprefix("student.")
        if student_key in current and current[student_key].shape == value.shape:
            extracted[student_key] = value
        else:
            skipped.append(key)
    missing = sorted(set(current) - set(extracted))
    output_payload: dict[str, Any] = {
        "epoch": int(payload.get("epoch", -1)),
        "model": extracted,
        "metrics": payload.get("metrics", {}),
        "config": dict(cfg),
        "source_checkpoint": str(args.source_checkpoint),
        "extraction": {
            "loaded_tensors": len(extracted),
            "skipped_tensors": skipped,
            "missing_tensors": missing,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output)
    print(f"wrote {output}")
    print(f"loaded_tensors={len(extracted)} skipped_tensors={len(skipped)} missing_tensors={len(missing)}")
    if missing:
        print("missing:")
        for key in missing:
            print(f"  {key}")


if __name__ == "__main__":
    main()
