from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.utils.config import load_config


def check_onnx(onnx_path: str, config_path: str = "ml/configs/loctouch_lite.yaml", checkpoint_path: str | None = None) -> bool:
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print("onnx_checker: ok")
    except ImportError:
        print("onnx_checker: skipped; onnx is not installed")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime_parity: skipped; onnxruntime is not installed")
        return True

    cfg = load_config(config_path)
    temporal_cfg = cfg.get("temporal", {})
    torch.manual_seed(int(cfg.get("seed", 7)))
    torch_model = LocTouchNetLite.from_config(cfg).eval().cpu()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        torch_model.load_state_dict(checkpoint["model"])
    image = torch.rand(2, 3, cfg["model"]["input_height"], cfg["model"]["input_width"], dtype=torch.float32)
    ort_inputs = {"image": image.numpy()}
    if bool(temporal_cfg.get("enabled", False)):
        input_mode = str(temporal_cfg.get("input_mode", "motion_map"))
        if input_mode == "image_t_minus_1":
            temporal = torch.rand(2, 3, cfg["model"]["input_height"], cfg["model"]["input_width"], dtype=torch.float32)
            torch_kwargs = {"image_t_minus_1": temporal}
            ort_inputs["image_t_minus_1"] = temporal.numpy()
        elif input_mode == "motion_map":
            motion_channels = max(1, int(temporal_cfg.get("motion_channels", temporal_cfg.get("num_motion_maps", 0))))
            temporal = torch.rand(2, motion_channels, cfg["model"]["input_height"], cfg["model"]["input_width"], dtype=torch.float32)
            torch_kwargs = {"motion_map": temporal}
            ort_inputs["motion_map"] = temporal.numpy()
        else:
            raise ValueError(f"Unknown temporal.input_mode: {input_mode}")
    else:
        torch_kwargs = {}
    with torch.no_grad():
        torch_out = torch_model(image, **torch_kwargs)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, ort_inputs)
    names = [o.name for o in sess.get_outputs()]
    checks = {
        "xy": torch_out["xy"].numpy(),
        "loc_conf": torch_out["loc_conf"].numpy(),
        "valid_logit": torch_out["valid_logit"].numpy(),
        "touch_logit": torch_out["touch_logit"].numpy(),
        "state_logits": torch_out["state_logits"].numpy(),
    }
    for name, arr in zip(names, ort_out):
        if name in checks:
            np.testing.assert_allclose(arr, checks[name], rtol=1e-4, atol=1e-4)
    print("onnxruntime_parity: ok")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--config", default="ml/configs/loctouch_lite.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    check_onnx(args.onnx, args.config, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
