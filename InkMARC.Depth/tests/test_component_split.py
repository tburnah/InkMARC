"""Run from the suite root: python -m unittest discover -s InkMARC.Depth/tests -v."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "InkMARC.Sense"))
sys.path.insert(0, str(ROOT / "InkMARC.Depth"))

import torch

from inkmarc_depth import InkMARCDepth, build_model_from_config as build_depth
from inkdepth.models import (
    DepthProxyMobileNet, InkDepthDA3Student, InkDepthStudent,
    InkMARCSense, InkMARCSenseDA3, build_model_from_config as build_sense,
)
from inkdepth.utils.config import load_config


class ComponentSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_imports_and_configuration_names(self):
        self.assertIs(InkDepthStudent, InkMARCSense)
        self.assertIs(InkDepthDA3Student, InkMARCSenseDA3)
        self.assertIs(DepthProxyMobileNet, InkMARCDepth)
        cfg = load_config(ROOT / "InkMARC.Depth/configs/da3_depth_proxy_504x322.yaml")
        for names, model in [
            (("InkDepthStudent", "InkMARCSense"), InkMARCSense),
            (("InkDepthDA3Student", "InkMARCSenseDA3"), InkMARCSenseDA3),
            (("DepthProxyMobileNet", "InkMARCDepth"), InkMARCDepth),
        ]:
            with patch.object(model, "from_config", return_value=object()) as factory:
                for name in names:
                    cfg.model.name = name
                    self.assertIs(build_sense(cfg), factory.return_value)
                    if model is InkMARCDepth:
                        self.assertIs(build_depth(cfg), factory.return_value)

    def test_existing_depth_checkpoint_and_output_contract(self):
        cfg = load_config(
            ROOT / "InkMARC.Depth/configs/da3_depth_proxy_504x322.yaml",
            {"model.pretrained": False},
        )
        model = build_depth(cfg).eval()
        checkpoint = ROOT / "research/inkdepth_student/outputs/da3_depth_proxy_504x322_mnv2_mixed_full/best_val_loss.pt"
        # Historical checkpoints pickle the dict-based config. Decode it as a
        # plain dict; only the state_dict is needed for this compatibility check.
        with torch.serialization.safe_globals([(dict, "inkdepth.utils.config.Config")]):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"], strict=True)
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 504, 322))
        self.assertEqual(tuple(output.shape), (1, 1, 504, 322))
        self.assertTrue(torch.isfinite(output).all())

    def test_existing_sense_checkpoint(self):
        cfg = load_config(ROOT / "InkMARC.Sense/configs/inkmarc_rgbd_geometry_export_best_strict.yaml")
        model = build_sense(cfg).eval()
        checkpoint = ROOT / "research/inkdepth_student/outputs/inkmarc_rgbd_geometry_export_best_strict/inkmarc_rgbd_geometry_best_strict.pt"
        with torch.serialization.safe_globals([(dict, "inkdepth.utils.config.Config")]):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload.get("model", payload), strict=True)
        with torch.inference_mode():
            output = model(torch.zeros(1, 7, 672, 432))
        self.assertEqual(tuple(output["xy"].shape), (1, 2))
        self.assertTrue(torch.isfinite(output["xy"]).all())

    def test_shared_artifact_links(self):
        target = ROOT / "research/inkdepth_student/outputs"
        for component in ("InkMARC.Sense", "InkMARC.Depth"):
            self.assertEqual((ROOT / component / "outputs").resolve(), target.resolve())


if __name__ == "__main__":
    unittest.main()
