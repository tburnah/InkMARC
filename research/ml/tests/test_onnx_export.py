import importlib.util
import json

import pytest

from ml.inkloc.export.check_onnx import check_onnx
from ml.inkloc.export.export_onnx import export_onnx
from ml.inkloc.utils.config import load_config


def test_onnx_export_and_check(tmp_path):
    pytest.importorskip("onnx")
    out = tmp_path / "loctouch_lite.onnx"
    export_onnx("ml/configs/loctouch_lite.yaml", str(out))
    assert out.exists()
    if importlib.util.find_spec("onnxruntime") is None:
        check_onnx(str(out))
    else:
        check_onnx(str(out))


def test_temporal_motion_onnx_export_and_check(tmp_path):
    pytest.importorskip("onnx")
    cfg = load_config("ml/configs/loctouch_lite.yaml")
    cfg["temporal"] = {
        "enabled": True,
        "num_motion_maps": 1,
        "input_mode": "motion_map",
        "motion_hidden_dim": 16,
    }
    config_path = tmp_path / "temporal.yaml"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "loctouch_lite_temporal.onnx"
    export_onnx(str(config_path), str(out))
    assert out.exists()
    check_onnx(str(out), str(config_path))
