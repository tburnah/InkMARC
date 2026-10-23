from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ml.inkloc.models.loctouch_lite import LocTouchNetLite, LocTouchNetLiteONNX
from ml.inkloc.utils.config import load_config


def export_onnx(config_path: str, out_path: str, checkpoint_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    export_cfg = cfg.get("export", {})
    temporal_cfg = cfg.get("temporal", {})
    torch.manual_seed(int(cfg.get("seed", 7)))
    model = LocTouchNetLite.from_config(cfg).eval().cpu()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    include_heatmap = bool(export_cfg.get("include_heatmap_output", False))
    temporal_input_mode = str(temporal_cfg.get("input_mode", "motion_map"))
    wrapper = LocTouchNetLiteONNX(
        model,
        include_heatmap_output=include_heatmap,
        temporal_input_mode=temporal_input_mode,
    ).eval().cpu()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy_image = torch.rand(1, 3, cfg["model"]["input_height"], cfg["model"]["input_width"], dtype=torch.float32)
    input_names = ["image"]
    dummy_inputs: tuple[torch.Tensor, ...]
    dynamic_axes = None
    if bool(temporal_cfg.get("enabled", False)):
        if temporal_input_mode == "image_t_minus_1":
            dummy_temporal = torch.rand(1, 3, cfg["model"]["input_height"], cfg["model"]["input_width"], dtype=torch.float32)
            input_names.append("image_t_minus_1")
        elif temporal_input_mode == "motion_map":
            motion_channels = max(1, int(temporal_cfg.get("motion_channels", temporal_cfg.get("num_motion_maps", 0))))
            dummy_temporal = torch.rand(1, motion_channels, cfg["model"]["input_height"], cfg["model"]["input_width"], dtype=torch.float32)
            input_names.append("motion_map")
        else:
            raise ValueError(f"Unknown temporal.input_mode: {temporal_input_mode}")
        dummy_inputs = (dummy_image, dummy_temporal)
    else:
        dummy_inputs = (dummy_image,)
    output_names = ["xy", "loc_conf", "valid_logit", "touch_logit", "state_logits"]
    if include_heatmap:
        output_names.append("heatmap_logits")
    if bool(export_cfg.get("dynamic_batch", True)):
        dynamic_axes = {name: {0: "batch"} for name in input_names}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(out),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=int(export_cfg.get("opset", 17)),
        do_constant_folding=True,
    )
    print(f"exported {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/loctouch_lite.yaml")
    parser.add_argument("--out", default="ml/artifacts/loctouch_lite.onnx")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    export_onnx(args.config, args.out, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
