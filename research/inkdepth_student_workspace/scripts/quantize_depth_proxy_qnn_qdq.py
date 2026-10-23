#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from onnxruntime.quantization import CalibrationDataReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset
from inkdepth.utils.config import load_config, parse_overrides


class ProxyCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        cfg: Any,
        model_path: str | Path,
        split: str,
        calibration_size: int,
        input_size: tuple[int, int],
    ) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(Path(model_path).expanduser().resolve()), providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name
        self.input_shape = [int(dim) for dim in session.get_inputs()[0].shape]
        dataset = StylusDataset(cfg, split=split, training=False)
        sample_count = min(int(calibration_size), len(dataset))
        indices = np.linspace(0, len(dataset) - 1, num=sample_count, dtype=np.int64).tolist()
        self.data_list: list[dict[str, np.ndarray]] = []
        for item_idx, dataset_idx in enumerate(indices, start=1):
            rgb = dataset[int(dataset_idx)]["image"][:3].float().unsqueeze(0)
            if tuple(rgb.shape[-2:]) != input_size:
                rgb = F.interpolate(rgb, size=input_size, mode="bilinear", align_corners=False)
            arr = np.ascontiguousarray(rgb.numpy().astype(np.float32))
            if list(arr.shape) != self.input_shape:
                raise ValueError(f"Calibration shape mismatch: got {list(arr.shape)}, expected {self.input_shape}")
            self.data_list.append({self.input_name: arr})
            if item_idx == 1 or item_idx % 25 == 0 or item_idx == sample_count:
                print(f"prepared calibration sample {item_idx}/{sample_count}", flush=True)
        self._iter: Any = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self.data_list)
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = None


def parse_quant_type(name: str):
    from onnxruntime.quantization import QuantType

    return {
        "quint8": QuantType.QUInt8,
        "qint8": QuantType.QInt8,
        "quint16": QuantType.QUInt16,
        "qint16": QuantType.QInt16,
    }[name.lower()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize fixed-shape depth proxy ONNX to QNN QDQ format.")
    parser.add_argument("--config", default=str(ROOT / "configs/da3_depth_proxy_504x322.yaml"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--activation-type", default="QUInt16", choices=["QUInt8", "QInt8", "QUInt16", "QInt16"])
    parser.add_argument("--weight-type", default="QUInt8", choices=["QUInt8", "QInt8", "QUInt16", "QInt16"])
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    from onnxruntime.quantization import CalibrationMethod, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

    cfg = load_config(args.config, parse_overrides(args.override))
    input_size = tuple(int(x) for x in cfg.model.proxy_input_size)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preproc_path = output_path.with_suffix(".preproc.onnx")

    reader = ProxyCalibrationReader(cfg, input_path, args.split, args.calibration_size, input_size)
    changed = qnn_preprocess_model(input_path, preproc_path, fuse_layernorm=False)
    model_to_quantize = preproc_path if changed else input_path
    qnn_config = get_qnn_qdq_config(
        model_to_quantize,
        reader,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=parse_quant_type(args.activation_type),
        weight_type=parse_quant_type(args.weight_type),
        calibration_providers=["CPUExecutionProvider"],
        op_types_to_quantize=["Conv", "ConvTranspose"],
    )
    quantize(model_to_quantize, output_path, qnn_config)
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "preprocessed_model": str(preproc_path) if changed else None,
        "model_changed_by_qnn_preprocess": bool(changed),
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "calibration_size": len(reader.data_list),
        "input_shape": reader.input_shape,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
