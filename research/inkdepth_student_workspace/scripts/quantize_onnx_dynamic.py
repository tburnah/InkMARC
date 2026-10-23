#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ONNX Runtime dynamic INT8 quantization.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    parser.add_argument(
        "--op-types",
        nargs="+",
        default=None,
        help="Optional ONNX op types to quantize, for example: --op-types MatMul Gemm",
    )
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=weight_type,
        op_types_to_quantize=args.op_types,
    )
    onnx.checker.check_model(str(dst))
    print(f"wrote: {dst}")


if __name__ == "__main__":
    main()
