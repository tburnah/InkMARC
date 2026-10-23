from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from inkdepth.models import InkDepthStudent
from inkdepth.utils.checkpoints import load_checkpoint


class ONNXWrapper(nn.Module):
    def __init__(self, model: InkDepthStudent, state_mode: str = "binary", include_depth: bool = True) -> None:
        super().__init__()
        self.model = model
        self.state_mode = state_mode
        self.include_depth = include_depth

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(x)
        tensors: list[torch.Tensor] = []
        if self.state_mode == "binary":
            tensors.append(out["touch_logits"])
        else:
            tensors.append(out["state_logits"])
        tensors.append(out["xy"])
        if self.include_depth:
            tensors.append(out["depth_lowres"])
        return tuple(tensors)


def export_onnx(
    cfg: Any,
    checkpoint: str | Path,
    output_path: str | Path,
    opset: int = 17,
    include_depth: bool = True,
) -> dict[str, float]:
    device = torch.device("cpu")
    model = InkDepthStudent.from_config(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)
    wrapper = ONNXWrapper(model, state_mode=str(cfg.model.state_mode), include_depth=include_depth).eval()
    h, w = tuple(int(x) for x in cfg.data.input_size)
    dummy = torch.rand(1, 3, h, w, device=device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_names = ["touch_logits" if cfg.model.state_mode == "binary" else "state_logits", "xy"]
    if include_depth:
        output_names.append("depth_lowres")
    with torch.no_grad():
        torch_outputs = wrapper(dummy)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    return validate_onnx(output_path, dummy, torch_outputs, output_names)


def validate_onnx(
    onnx_path: str | Path,
    dummy: torch.Tensor,
    torch_outputs: tuple[torch.Tensor, ...],
    output_names: list[str],
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"onnxruntime_available": 0.0}
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(output_names, {"image": dummy.numpy()})
    diffs = {}
    for name, pt, ort_out in zip(output_names, torch_outputs, ort_outputs):
        diffs[f"max_abs_diff/{name}"] = float(np.max(np.abs(pt.detach().numpy() - ort_out)))
    return diffs
