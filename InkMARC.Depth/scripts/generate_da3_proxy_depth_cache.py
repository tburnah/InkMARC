#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SENSE_ROOT = ROOT.parent / "InkMARC.Sense"
# Shared dataset/configuration helpers; keep the Depth model independently packaged.
sys.path.insert(0, str(SENSE_ROOT))
sys.path.insert(0, str(ROOT))

from inkdepth.datasets import StylusDataset
from inkdepth.utils.config import load_config, parse_overrides


def tensor_to_uint8_frames(rgb: torch.Tensor) -> list[np.ndarray]:
    arr = rgb.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return [np.ascontiguousarray(frame) for frame in arr]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-shape DA3 teacher depth targets for proxy training.")
    parser.add_argument("--config", default=str(SENSE_ROOT / "configs/final_combined_live_da3_geometry_heatmap_touch_presence_from_best_xy.yaml"))
    parser.add_argument("--da3-onnx", default=str(ROOT / "outputs/onnx/da3_small_depth_504x322.onnx"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/da3_proxy_depth_cache_504x322"))
    parser.add_argument("--split", action="append", default=None, help="Dataset split to process. Repeatable. Defaults to train and val.")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    import onnxruntime as ort

    repo_root = str(Path(load_config(args.config).model.da3_repo_root).expanduser().resolve() / "src")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from depth_anything_3.utils.io.input_processor import InputProcessor  # noqa: WPS433

    cfg = load_config(args.config, parse_overrides(args.override))
    output_dir = Path(args.output_dir).expanduser().resolve()
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    splits = args.split or ["train", "val"]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(Path(args.da3_onnx).expanduser().resolve()), sess_options=session_options, providers=[args.provider])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_shape = [int(dim) for dim in input_meta.shape]
    input_processor = InputProcessor()
    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    all_entries: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        dataset = StylusDataset(cfg, split=split, training=False)
        total = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
        entries: list[dict[str, Any]] = []
        split_dir = sample_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in tqdm(range(total), desc=f"cache {split}"):
            out_path = split_dir / f"{idx:08d}.npz"
            if args.resume and out_path.exists():
                entries.append({"index": idx, "path": str(out_path.relative_to(output_dir))})
                continue
            sample = dataset[idx]
            rgb = sample["image"][:3].unsqueeze(0).float()
            frames = tensor_to_uint8_frames(rgb)
            da3_input, _, _ = input_processor(
                frames,
                extrinsics=None,
                intrinsics=None,
                process_res=int(getattr(cfg.model, "da3_process_res", 322)),
                process_res_method=str(getattr(cfg.model, "da3_process_res_method", "lower_bound_resize")),
                num_workers=1,
                sequential=True,
                print_progress=False,
            )
            da3_input_np = np.ascontiguousarray(da3_input.unsqueeze(0).numpy().astype(np.float32))
            if list(da3_input_np.shape) != input_shape:
                raise ValueError(f"DA3 input shape mismatch: got {list(da3_input_np.shape)}, expected {input_shape}")
            depth = session.run([output_meta.name], {input_meta.name: da3_input_np})[0].astype(np.float32)
            if depth.ndim == 3:
                depth = depth[:, None]
            np.savez_compressed(out_path, depth=depth[0].astype(out_dtype, copy=False))
            entries.append({"index": idx, "path": str(out_path.relative_to(output_dir))})
        all_entries[split] = entries

    index = {
        "source_config": str(Path(args.config).expanduser().resolve()),
        "da3_onnx": str(Path(args.da3_onnx).expanduser().resolve()),
        "provider": args.provider,
        "dtype": args.dtype,
        "input_name": input_meta.name,
        "input_shape": input_shape,
        "output_name": output_meta.name,
        "splits": all_entries,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'index.json'}")


if __name__ == "__main__":
    main()
