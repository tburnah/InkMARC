from __future__ import annotations

from typing import Any

from .depth_proxy import DepthProxyMobileNet
from .inkdepth_da3_student import InkDepthDA3Student
from .inkdepth_student import InkDepthStudent


def build_model_from_config(cfg: Any):
    name = str(getattr(cfg.model, "name", "InkDepthStudent"))
    if name == "InkDepthStudent":
        return InkDepthStudent.from_config(cfg)
    if name == "InkDepthDA3Student":
        return InkDepthDA3Student.from_config(cfg)
    if name == "DepthProxyMobileNet":
        return DepthProxyMobileNet.from_config(cfg)
    raise ValueError(f"Unsupported model.name={name}")


__all__ = ["InkDepthStudent", "InkDepthDA3Student", "DepthProxyMobileNet", "build_model_from_config"]
