#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PATCH_SIZE = 14


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a fixed-shape depth-only Depth Anything 3 Small ONNX model."
    )
    parser.add_argument(
        "--repo-root",
        default="/home/t/tburnah/src/depth_model/Depth-Anything-3",
        help="Depth-Anything-3 checkout containing src/depth_anything_3.",
    )
    parser.add_argument("--model-id", default="depth-anything/DA3-SMALL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=322)
    parser.add_argument("--width", type=int, default=322)
    parser.add_argument(
        "--from-frame-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "Compute the ONNX input shape from an original frame size using DA3 preprocessing. "
            "For this repo's 672x432 input and process_res=322, use: --from-frame-size 672 432."
        ),
    )
    parser.add_argument("--process-res", type=int, default=322)
    parser.add_argument(
        "--process-res-method",
        default="lower_bound_resize",
        choices=[
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ],
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--keep-pose-head",
        action="store_true",
        help="Keep DA3 camera pose head in the traced graph. Slower and unnecessary for depth-only use.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX checker and ONNX Runtime parity validation.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    sys.path.insert(0, str(repo_root / "src"))

    import torch
    import onnx
    patch_da3_onnx_export()

    if args.from_frame_size is not None:
        height, width = processed_hw_from_frame(
            frame_h=int(args.from_frame_size[0]),
            frame_w=int(args.from_frame_size[1]),
            process_res=int(args.process_res),
            process_res_method=str(args.process_res_method),
        )
    else:
        height, width = int(args.height), int(args.width)

    validate_shape(height, width)

    from depth_anything_3.api import DepthAnything3  # noqa: WPS433

    device = torch.device(args.device)
    da3 = DepthAnything3.from_pretrained(str(args.model_id)).to(device).eval()
    model = DA3DepthOnlyExport(da3, skip_pose_head=not bool(args.keep_pose_head)).to(device).eval()

    dummy = torch.randn(1, 1, 3, height, width, device=device)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["image"],
            output_names=["depth"],
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    print(f"exported: {output}")
    print(f"input image: [1,1,3,{height},{width}]")
    print("output depth: [1,1,H,W]")

    if not args.no_verify:
        verify_onnx(model, dummy, output)


class DA3DepthOnlyExport:
    def __new__(cls, *args: Any, **kwargs: Any):
        import torch

        class _Module(torch.nn.Module):
            def __init__(self, da3: Any, skip_pose_head: bool = True) -> None:
                super().__init__()
                self.net = da3.model
                if skip_pose_head and hasattr(self.net, "cam_dec"):
                    self.net.cam_dec = None

            def forward(self, image):
                output = self.net(
                    image,
                    None,
                    None,
                    [],
                    False,
                    False,
                    "saddle_balanced",
                )
                depth = output["depth"]
                if depth.ndim == 5 and depth.shape[-1] == 1:
                    depth = depth.permute(0, 1, 4, 2, 3).reshape(
                        depth.shape[0], 1, depth.shape[2], depth.shape[3]
                    )
                elif depth.ndim == 4:
                    depth = depth.reshape(depth.shape[0], 1, depth.shape[-2], depth.shape[-1])
                else:
                    raise RuntimeError(f"Unexpected DA3 depth shape: {tuple(depth.shape)}")
                return torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)

        return _Module(*args, **kwargs)


def processed_hw_from_frame(
    frame_h: int,
    frame_w: int,
    process_res: int,
    process_res_method: str,
) -> tuple[int, int]:
    if process_res_method.startswith("upper_bound"):
        scale = process_res / float(max(frame_h, frame_w))
    elif process_res_method.startswith("lower_bound"):
        scale = process_res / float(min(frame_h, frame_w))
    else:
        raise ValueError(f"Unsupported process_res_method: {process_res_method}")

    resized_h = max(1, int(round(frame_h * scale)))
    resized_w = max(1, int(round(frame_w * scale)))

    if process_res_method.endswith("resize"):
        return nearest_multiple(resized_h, PATCH_SIZE), nearest_multiple(resized_w, PATCH_SIZE)
    if process_res_method.endswith("crop"):
        return (resized_h // PATCH_SIZE) * PATCH_SIZE, (resized_w // PATCH_SIZE) * PATCH_SIZE
    raise ValueError(f"Unsupported process_res_method: {process_res_method}")


def nearest_multiple(value: int, multiple: int) -> int:
    down = (value // multiple) * multiple
    up = down + multiple
    return max(multiple, up if abs(up - value) <= abs(value - down) else down)


def validate_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"Input shape must be positive, got {height}x{width}")
    if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
        raise ValueError(
            f"DA3 input height/width must be divisible by {PATCH_SIZE}; got {height}x{width}"
        )


def verify_onnx(model: Any, dummy: Any, output: Path) -> None:
    import onnxruntime as ort
    import torch

    with torch.inference_mode():
        torch_depth = model(dummy).detach().cpu().numpy()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(output),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    ort_depth = session.run(None, {"image": dummy.detach().cpu().numpy().astype(np.float32)})[0]
    max_abs_diff = float(np.max(np.abs(torch_depth - ort_depth)))
    mean_abs_diff = float(np.mean(np.abs(torch_depth - ort_depth)))
    print(f"verify max_abs_diff: {max_abs_diff:.6g}")
    print(f"verify mean_abs_diff: {mean_abs_diff:.6g}")


def patch_da3_onnx_export() -> None:
    import torch

    from depth_anything_3.model.dinov2.layers.rope import PositionGetter  # noqa: WPS433

    def onnx_friendly_position_getter(
        self: Any,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            positions = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    PositionGetter.__call__ = onnx_friendly_position_getter


if __name__ == "__main__":
    main()
