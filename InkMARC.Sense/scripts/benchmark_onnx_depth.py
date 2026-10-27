#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a fixed-shape DA3 depth ONNX model.")
    parser.add_argument("model")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.threads > 0:
        options.intra_op_num_threads = int(args.threads)

    providers = [args.provider]
    session = ort.InferenceSession(str(Path(args.model)), sess_options=options, providers=providers)
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    shape = [int(dim) for dim in input_meta.shape]
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

    for _ in range(max(0, args.warmup)):
        session.run([output_meta.name], {input_meta.name: x})

    times = []
    for _ in range(max(1, args.runs)):
        start = time.perf_counter()
        session.run([output_meta.name], {input_meta.name: x})
        times.append((time.perf_counter() - start) * 1000.0)

    result = {
        "model": str(Path(args.model).resolve()),
        "provider": args.provider,
        "input": input_meta.name,
        "input_shape": shape,
        "output": output_meta.name,
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "runs": len(times),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
