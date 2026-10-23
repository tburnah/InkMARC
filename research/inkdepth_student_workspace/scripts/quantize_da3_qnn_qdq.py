#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset
from inkdepth.utils.config import load_config, parse_overrides


def tensor_to_uint8_frames(rgb: torch.Tensor) -> list[np.ndarray]:
    arr = rgb.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(frame) for frame in arr]


class DA3CalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        cfg: Any,
        model_path: str | Path,
        split: str,
        calibration_size: int,
        da3_repo_root: str | Path,
        max_records: int = 0,
    ) -> None:
        import onnxruntime as ort

        repo_src = str(Path(da3_repo_root).expanduser().resolve() / "src")
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: WPS433

        session = ort.InferenceSession(str(Path(model_path).expanduser().resolve()), providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name
        self.input_shape = [int(dim) for dim in session.get_inputs()[0].shape]
        self.process_res = int(getattr(cfg.model, "da3_process_res", 322))
        self.process_res_method = str(getattr(cfg.model, "da3_process_res_method", "lower_bound_resize"))
        self.input_processor = InputProcessor()

        dataset = StylusDataset(cfg, split=split, training=False)
        if max_records > 0:
            dataset.records = dataset.records[:max_records]
        if len(dataset) == 0:
            raise ValueError(f"No calibration records found for split={split!r}")

        sample_count = min(int(calibration_size), len(dataset))
        indices = np.linspace(0, len(dataset) - 1, num=sample_count, dtype=np.int64).tolist()
        self.data_list: list[dict[str, np.ndarray]] = []
        for item_idx, dataset_idx in enumerate(indices, start=1):
            sample = dataset[int(dataset_idx)]
            image = sample["image"].float()
            rgb = image[:3].unsqueeze(0)
            calib_input = self._preprocess_rgb(rgb)
            if list(calib_input.shape) != self.input_shape:
                raise ValueError(
                    f"Calibration input shape mismatch at sample {dataset_idx}: "
                    f"got {list(calib_input.shape)}, expected {self.input_shape}"
                )
            self.data_list.append({self.input_name: calib_input})
            if item_idx == 1 or item_idx % 25 == 0 or item_idx == sample_count:
                print(f"prepared calibration sample {item_idx}/{sample_count}", flush=True)

        self._iter: Any = None

    def _preprocess_rgb(self, rgb: torch.Tensor) -> np.ndarray:
        frames = tensor_to_uint8_frames(rgb)
        imgs_cpu, _, _ = self.input_processor(
            frames,
            extrinsics=None,
            intrinsics=None,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
            num_workers=1,
            sequential=True,
            print_progress=False,
        )
        return np.ascontiguousarray(imgs_cpu.unsqueeze(0).numpy().astype(np.float32))

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self.data_list)
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = None


def parse_quant_type(name: str):
    from onnxruntime.quantization import QuantType

    normalized = name.lower()
    by_name = {
        "quint8": QuantType.QUInt8,
        "qint8": QuantType.QInt8,
        "quint16": QuantType.QUInt16,
        "qint16": QuantType.QInt16,
    }
    if normalized not in by_name:
        raise ValueError(f"Unsupported quant type {name!r}; choose one of {sorted(by_name)}")
    return by_name[normalized]


def parse_calibration_method(name: str):
    from onnxruntime.quantization import CalibrationMethod

    normalized = name.lower()
    by_name = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    if normalized not in by_name:
        raise ValueError(f"Unsupported calibration method {name!r}; choose one of {sorted(by_name)}")
    return by_name[normalized]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize fixed-shape DA3 ONNX to QNN-compatible QDQ ONNX.")
    parser.add_argument("--config", default=str(ROOT / "configs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml"))
    parser.add_argument("--input", default=str(ROOT / "outputs/onnx/da3_small_depth_504x322.onnx"))
    parser.add_argument("--output", default=str(ROOT / "outputs/onnx/da3_small_depth_504x322_qnn_qdq_u8.onnx"))
    parser.add_argument("--preprocessed-output", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--activation-type", default="QUInt8", choices=["QUInt8", "QInt8", "QUInt16", "QInt16"])
    parser.add_argument("--weight-type", default="QUInt8", choices=["QUInt8", "QInt8", "QUInt16", "QInt16"])
    parser.add_argument("--calibrate-method", default="MinMax", choices=["MinMax", "Entropy", "Percentile", "Distribution"])
    parser.add_argument(
        "--op-types-to-quantize",
        nargs="*",
        default=None,
        help="Optional ONNX op types to quantize, for example: --op-types-to-quantize Conv MatMul",
    )
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--fuse-layernorm", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    from onnxruntime.quantization import quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

    cfg = load_config(args.config, parse_overrides(args.override))
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preproc_path = (
        Path(args.preprocessed_output).expanduser().resolve()
        if args.preprocessed_output
        else output_path.with_suffix(".preproc.onnx")
    )

    reader = DA3CalibrationDataReader(
        cfg=cfg,
        model_path=input_path,
        split=args.split,
        calibration_size=args.calibration_size,
        da3_repo_root=str(getattr(cfg.model, "da3_repo_root", "/home/t/tburnah/src/depth_model/Depth-Anything-3")),
        max_records=args.max_records,
    )

    print(f"preprocessing model for QNN: {input_path}", flush=True)
    model_changed = qnn_preprocess_model(
        input_path,
        preproc_path,
        fuse_layernorm=bool(args.fuse_layernorm),
    )
    model_to_quantize = preproc_path if model_changed else input_path
    print(f"model_changed={model_changed}; quantizing {model_to_quantize}", flush=True)

    qnn_config = get_qnn_qdq_config(
        model_to_quantize,
        reader,
        calibrate_method=parse_calibration_method(args.calibrate_method),
        activation_type=parse_quant_type(args.activation_type),
        weight_type=parse_quant_type(args.weight_type),
        per_channel=bool(args.per_channel),
        calibration_providers=["CPUExecutionProvider"],
        op_types_to_quantize=args.op_types_to_quantize,
    )
    quantize(model_to_quantize, output_path, qnn_config)

    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "preprocessed_model": str(preproc_path) if model_changed else None,
        "model_changed_by_qnn_preprocess": bool(model_changed),
        "config": str(Path(args.config).expanduser().resolve()),
        "split": args.split,
        "calibration_size": len(reader.data_list),
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "calibrate_method": args.calibrate_method,
        "per_channel": bool(args.per_channel),
        "op_types_to_quantize": args.op_types_to_quantize,
        "input_name": reader.input_name,
        "input_shape": reader.input_shape,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
