from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from config import TrainingConfig
from model import build_model
from utils import NormalizationStats, load_checkpoint, save_json


class OnnxExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, imu_window: torch.Tensor) -> tuple[torch.Tensor, ...]:
        lengths = torch.ones(
            (imu_window.size(0),),
            dtype=torch.int64,
            device=imu_window.device,
        ) * imu_window.size(-1)
        outputs = self.model(imu_window, lengths=lengths)
        exported: list[torch.Tensor] = [outputs["regression"]]
        if outputs.get("log_variance") is not None:
            exported.append(outputs["log_variance"])
        if outputs.get("motion_logits") is not None:
            exported.append(outputs["motion_logits"])
        if outputs.get("bias_correction") is not None:
            exported.append(outputs["bias_correction"])
        return tuple(exported)


def _normalization_to_dict(normalization: NormalizationStats | dict[str, Any] | None) -> dict[str, Any] | None:
    if normalization is None:
        return None
    if isinstance(normalization, NormalizationStats):
        return normalization.to_dict()
    return dict(normalization)


def checkpoint_to_training_config(checkpoint: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**checkpoint["config"])


def load_model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    train_config = checkpoint_to_training_config(checkpoint)
    model = build_model(
        model_type=checkpoint["model_type"],
        input_channels=checkpoint["input_channels"],
        output_dim=checkpoint["output_dim"],
        conv_channels=checkpoint["conv_channels"],
        kernel_size=checkpoint["kernel_size"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
        activation=checkpoint["activation"],
        use_residual=checkpoint["use_residual"],
        motion_head_enabled=checkpoint.get("motion_head_enabled", train_config.motion_head_enabled),
        hybrid_head_enabled=checkpoint.get("hybrid_head_enabled", train_config.hybrid_head_enabled),
        hybrid_bias_correction_enabled=checkpoint.get(
            "hybrid_bias_correction_enabled",
            train_config.hybrid_bias_correction_enabled,
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def infer_output_names(model: nn.Module, train_config: TrainingConfig) -> list[str]:
    output_names = ["regression"]
    hybrid_enabled = getattr(train_config, "hybrid_head_enabled", False)
    motion_enabled = getattr(train_config, "motion_head_enabled", False)
    bias_enabled = getattr(train_config, "hybrid_bias_correction_enabled", False)
    if hybrid_enabled:
        output_names.append("log_variance")
    if motion_enabled:
        output_names.append("motion_logits")
    if bias_enabled:
        output_names.append("bias_correction")
    return output_names


def build_onnx_manifest(
    train_config: TrainingConfig,
    checkpoint: dict[str, Any],
    normalization: NormalizationStats | dict[str, Any] | None,
    output_path: Path,
    output_names: list[str],
    opset_version: int,
) -> dict[str, Any]:
    return {
        "artifact_type": "imu_odometry_onnx_export",
        "onnx_file": output_path.name,
        "opset_version": opset_version,
        "input": {
            "name": "imu_window",
            "shape": [1, checkpoint["input_channels"], train_config.window_size],
            "layout": "NCT",
            "dtype": "float32",
        },
        "outputs": output_names,
        "model": {
            "model_type": checkpoint["model_type"],
            "target_mode": train_config.target_mode,
            "target_names": checkpoint["target_names"],
            "delta_xy_parameterization": train_config.delta_xy_parameterization,
            "hybrid_head_enabled": checkpoint.get("hybrid_head_enabled", train_config.hybrid_head_enabled),
            "hybrid_bias_correction_enabled": checkpoint.get(
                "hybrid_bias_correction_enabled",
                train_config.hybrid_bias_correction_enabled,
            ),
            "motion_head_enabled": checkpoint.get("motion_head_enabled", train_config.motion_head_enabled),
        },
        "preprocessing": {
            "window_size": train_config.window_size,
            "stride": train_config.stride,
            "feature_names": train_config.resolved_features,
            "quaternion_frame_mode": train_config.quaternion_frame_mode,
            "normalization": _normalization_to_dict(normalization),
        },
    }


def verify_onnx_export(
    output_path: Path,
    model: nn.Module,
    checkpoint: dict[str, Any],
    train_config: TrainingConfig,
    output_names: list[str],
) -> dict[str, float]:
    wrapped_model = OnnxExportWrapper(copy.deepcopy(model).cpu().eval())
    dummy_input = torch.randn(1, checkpoint["input_channels"], train_config.window_size, dtype=torch.float32)
    with torch.no_grad():
        torch_outputs = wrapped_model(dummy_input)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"imu_window": dummy_input.numpy()})
    max_abs_diffs: dict[str, float] = {}
    for name, torch_output, ort_output in zip(output_names, torch_outputs, ort_outputs):
        max_abs_diffs[name] = float(np.max(np.abs(torch_output.detach().cpu().numpy() - ort_output)))
    return max_abs_diffs


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset_version: int = 17,
    verify: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    train_config = checkpoint_to_training_config(checkpoint)
    model = load_model_from_checkpoint(checkpoint)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped_model = OnnxExportWrapper(copy.deepcopy(model).cpu().eval())
    output_names = infer_output_names(model, train_config)
    dummy_input = torch.randn(1, checkpoint["input_channels"], train_config.window_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_input,),
            str(output_path),
            input_names=["imu_window"],
            output_names=output_names,
            export_params=True,
            do_constant_folding=True,
            opset_version=opset_version,
        )

    normalization = checkpoint["normalization"]
    manifest = build_onnx_manifest(
        train_config=train_config,
        checkpoint=checkpoint,
        normalization=normalization,
        output_path=output_path,
        output_names=output_names,
        opset_version=opset_version,
    )
    save_json(manifest, output_path.with_suffix(".manifest.json"))

    verification: dict[str, float] | None = None
    if verify:
        verification = verify_onnx_export(
            output_path=output_path,
            model=model,
            checkpoint=checkpoint,
            train_config=train_config,
            output_names=output_names,
        )
        save_json(
            {
                "onnx_file": output_path.name,
                "max_abs_diff_by_output": verification,
            },
            output_path.with_suffix(".verification.json"),
        )

    return {
        "onnx_path": str(output_path),
        "manifest_path": str(output_path.with_suffix(".manifest.json")),
        "verification_path": (
            str(output_path.with_suffix(".verification.json")) if verification is not None else None
        ),
        "output_names": output_names,
        "max_abs_diff_by_output": verification,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an IMU odometry checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
        opset_version=args.opset_version,
        verify=not args.no_verify,
    )
    print(f"Exported ONNX: {result['onnx_path']}")
    print(f"Manifest: {result['manifest_path']}")
    if result["verification_path"] is not None:
        print(f"Verification: {result['verification_path']}")
        print(f"Max abs diff by output: {result['max_abs_diff_by_output']}")


if __name__ == "__main__":
    main()
