from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from inkdepth.models import InkMARCSense
from inkdepth.utils.checkpoints import load_checkpoint


class ONNXWrapper(nn.Module):
    def __init__(self, model: InkMARCSense, include_depth: bool = True, include_heatmap: bool = False) -> None:
        super().__init__()
        self.model = model
        self.include_depth = include_depth
        self.include_heatmap = include_heatmap

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(x)
        tensors: list[torch.Tensor] = []
        if self.model.train_state:
            state_logits = out["state_logits"]
            if state_logits is None:
                raise RuntimeError("model.train_state=True but state_logits is None")
            tensors.append(state_logits)
        if self.model.train_touch:
            touch_logits = out["touch_logits"]
            if touch_logits is None:
                raise RuntimeError("model.train_touch=True but touch_logits is None")
            tensors.append(touch_logits)
        if self.model.train_location:
            xy = out["xy"]
            if xy is None:
                raise RuntimeError("model.train_location=True but xy is None")
            tensors.append(xy)
        if self.include_heatmap and self.model.train_location:
            tensors.append(out["location_heatmap"])
        if self.include_depth:
            tensors.append(out["depth_lowres"])
        return tuple(tensors)

    def output_names(self) -> list[str]:
        names: list[str] = []
        if self.model.train_state:
            names.append("state_logits")
        if self.model.train_touch:
            names.append("touch_logits")
        if self.model.train_location:
            names.append("xy")
            if self.include_heatmap:
                names.append("location_heatmap")
        if self.include_depth:
            names.append("depth_lowres")
        return names


def export_onnx(
    cfg: Any,
    checkpoint: str | Path,
    output_path: str | Path,
    opset: int = 17,
    include_depth: bool = True,
    include_heatmap: bool = False,
) -> dict[str, float]:
    device = torch.device("cpu")
    model = InkMARCSense.from_config(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)
    wrapper = ONNXWrapper(model, include_depth=include_depth, include_heatmap=include_heatmap).eval()
    h, w = tuple(int(x) for x in cfg.data.input_size)
    dummy = torch.rand(1, int(getattr(cfg.model, "input_channels", 3)), h, w, device=device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_names = wrapper.output_names()
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
