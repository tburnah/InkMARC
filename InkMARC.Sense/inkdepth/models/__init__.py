from __future__ import annotations

from typing import Any

from .depth_proxy import DepthProxyMobileNet, InkMARCDepth
from .inkdepth_da3_student import InkDepthDA3Student, InkMARCSenseDA3
from .inkdepth_student import InkDepthStudent, InkMARCSense


def build_model_from_config(cfg: Any):
    name = str(getattr(cfg.model, "name", "InkMARCSense"))
    if name in {"InkMARCSense", "InkDepthStudent"}:
        return InkMARCSense.from_config(cfg)
    if name in {"InkMARCSenseDA3", "InkDepthDA3Student"}:
        return InkMARCSenseDA3.from_config(cfg)
    if name in {"InkMARCDepth", "DepthProxyMobileNet"}:
        return InkMARCDepth.from_config(cfg)
    raise ValueError(f"Unsupported model.name={name}")


__all__ = [
    "InkMARCSense", "InkMARCSenseDA3", "InkMARCDepth",
    "InkDepthStudent", "InkDepthDA3Student", "DepthProxyMobileNet",
    "build_model_from_config",
]
