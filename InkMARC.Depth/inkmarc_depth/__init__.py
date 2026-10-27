"""InkMARC-Depth: the distilled relative-depth provider for InkMARC-Sense."""

from .model import DepthProxyMobileNet, InkMARCDepth


def build_model_from_config(cfg):
    """Accept the suite name and the original checkpoint/configuration name."""
    name = str(getattr(cfg.model, "name", "InkMARCDepth"))
    if name not in {"InkMARCDepth", "DepthProxyMobileNet"}:
        raise ValueError(f"Unsupported model.name={name}")
    return InkMARCDepth.from_config(cfg)


__all__ = ["InkMARCDepth", "DepthProxyMobileNet", "build_model_from_config"]
