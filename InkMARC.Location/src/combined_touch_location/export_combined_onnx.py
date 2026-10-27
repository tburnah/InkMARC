from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from combined_touch_location.combined_model import CombinedTouchLocationModel, SUPPORTED_BACKBONES


def torch_load_checkpoint(path: Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class CombinedOnnxWrapper(nn.Module):
    def __init__(self, model: CombinedTouchLocationModel, include_heatmap: bool = False) -> None:
        super().__init__()
        self.model = model
        self.include_heatmap = bool(include_heatmap)

    def forward(self, image: torch.Tensor):
        out = self.model(image)
        if self.include_heatmap:
            return out["state_logits"], out["touch_logits"], out["xy"], out["heatmap_logits"], out["heatmap_prob"]
        return out["state_logits"], out["touch_logits"], out["xy"]


def _checkpoint_args(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    raw = ckpt.get("args", {})
    return raw if isinstance(raw, dict) else {}


def _resolve_model_arg(cli_value: Any, ckpt_args: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value not in (None, "", 0):
        return cli_value
    value = ckpt_args.get(key, default)
    return value if value not in (None, "") else default


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    backbone: str | None = None,
    hm_h: int | None = None,
    hm_w: int | None = None,
    pretrained_backbone: bool | None = None,
    imagenet_normalize: bool | None = None,
) -> tuple[CombinedTouchLocationModel, Dict[str, Any]]:
    ckpt = torch_load_checkpoint(checkpoint_path, map_location=device)
    ckpt_args = _checkpoint_args(ckpt) if isinstance(ckpt, dict) else {}

    resolved_backbone = str(backbone or ckpt_args.get("backbone", "mobilenet_v3_large"))
    resolved_hm_h = int(hm_h or ckpt_args.get("hm_h", 40))
    resolved_hm_w = int(hm_w or ckpt_args.get("hm_w", 60))
    resolved_pretrained = bool(ckpt_args.get("pretrained_backbone", False) if pretrained_backbone is None else pretrained_backbone)
    resolved_imagenet_normalize = bool(
        ckpt_args.get("imagenet_normalize", False) if imagenet_normalize is None else imagenet_normalize
    )

    model = CombinedTouchLocationModel(
        backbone=resolved_backbone,
        hm_h=resolved_hm_h,
        hm_w=resolved_hm_w,
        pretrained_backbone=resolved_pretrained,
        imagenet_normalize=resolved_imagenet_normalize,
    ).to(device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt_args


def export_onnx_from_checkpoint(
    *,
    checkpoint_path: Path,
    onnx_path: Path,
    device: torch.device,
    out_h: int,
    out_w: int,
    onnx_opset: int = 17,
    include_heatmap: bool = False,
    backbone: str | None = None,
    hm_h: int | None = None,
    hm_w: int | None = None,
) -> Path:
    model, _ = load_model_from_checkpoint(
        checkpoint_path,
        device=device,
        backbone=backbone,
        hm_h=hm_h,
        hm_w=hm_w,
    )
    wrapper = CombinedOnnxWrapper(model, include_heatmap=include_heatmap).to(device).eval()

    dummy = torch.zeros(1, 3, int(out_h), int(out_w), device=device, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    output_names = ["state_logits", "touch_logits", "xy"]
    dynamic_axes: Dict[str, Dict[int, str]] = {
        "image": {0: "batch"},
        "state_logits": {0: "batch"},
        "touch_logits": {0: "batch"},
        "xy": {0: "batch"},
    }
    if include_heatmap:
        output_names.extend(["heatmap_logits", "heatmap_prob"])
        dynamic_axes["heatmap_logits"] = {0: "batch"}
        dynamic_axes["heatmap_prob"] = {0: "batch"}

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=int(onnx_opset),
    )
    print(f"ONNX exported: {onnx_path} | outputs={output_names}")
    return onnx_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export combined touch/location model to ONNX.")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt or an epoch checkpoint.")
    ap.add_argument("--onnx-path", default="", help="Output ONNX path. Defaults beside checkpoint.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-w", type=int, default=0)
    ap.add_argument("--out-h", type=int, default=0)
    ap.add_argument("--hm-w", type=int, default=0)
    ap.add_argument("--hm-h", type=int, default=0)
    ap.add_argument("--backbone", default="", choices=("", *SUPPORTED_BACKBONES))
    ap.add_argument("--onnx-opset", type=int, default=17)
    ap.add_argument("--include-heatmap", action="store_true")
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    ckpt_args = _checkpoint_args(ckpt) if isinstance(ckpt, dict) else {}

    out_w = int(_resolve_model_arg(args.out_w, ckpt_args, "out_w", 480))
    out_h = int(_resolve_model_arg(args.out_h, ckpt_args, "out_h", 320))
    hm_w = int(_resolve_model_arg(args.hm_w, ckpt_args, "hm_w", 60))
    hm_h = int(_resolve_model_arg(args.hm_h, ckpt_args, "hm_h", 40))
    backbone = str(_resolve_model_arg(args.backbone, ckpt_args, "backbone", "mobilenet_v3_large"))

    if args.onnx_path:
        onnx_path = Path(args.onnx_path)
    else:
        onnx_path = checkpoint_path.with_suffix(".onnx")

    export_onnx_from_checkpoint(
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        device=torch.device(args.device),
        out_h=out_h,
        out_w=out_w,
        onnx_opset=args.onnx_opset,
        include_heatmap=args.include_heatmap,
        backbone=backbone,
        hm_h=hm_h,
        hm_w=hm_w,
    )


if __name__ == "__main__":
    main()
