#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SENSE_ROOT = ROOT.parent / "InkMARC.Sense"
# Shared dataset/configuration helpers; keep the Depth model independently packaged.
sys.path.insert(0, str(SENSE_ROOT))
sys.path.insert(0, str(ROOT))

from inkmarc_depth import build_model_from_config
from inkdepth.utils.checkpoints import load_checkpoint
from inkdepth.utils.config import load_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed-shape DA3 depth proxy to ONNX.")
    parser.add_argument("--config", default=str(ROOT / "configs/da3_depth_proxy_504x322.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort

    cfg = load_config(args.config, parse_overrides(args.override))
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best_val_loss.pt")
    output = Path(args.output or str(Path(cfg.output_dir) / "depth_proxy_504x322.onnx")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(cfg).cpu().eval()
    load_checkpoint(checkpoint, model, map_location="cpu")
    h, w = tuple(int(x) for x in cfg.model.proxy_input_size)
    dummy = torch.rand(1, 3, h, w)
    with torch.no_grad():
        torch_out = model(dummy)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["image"],
        output_names=["depth"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamic_axes=None,
    )

    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    ort_out = session.run(["depth"], {"image": dummy.numpy().astype(np.float32)})[0]
    op_counts = collections.Counter(node.op_type for node in onnx_model.graph.node)
    summary = {
        "output": str(output),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "input_shape": [1, 3, h, w],
        "output_shape": list(torch_out.shape),
        "max_abs_diff": float(np.max(np.abs(torch_out.detach().numpy() - ort_out))),
        "mean_abs_diff": float(np.mean(np.abs(torch_out.detach().numpy() - ort_out))),
        "op_counts": dict(sorted(op_counts.items())),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
