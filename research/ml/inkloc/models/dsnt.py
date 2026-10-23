import torch


def spatial_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over HxW while preserving [B, C, H, W]."""

    if not torch.jit.is_tracing() and logits.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(logits.shape)}")
    b, c, h, w = logits.shape
    flat = logits.reshape(b, c, h * w) / temperature
    return torch.softmax(flat, dim=-1).reshape(b, c, h, w)


def dsnt_xy(prob: torch.Tensor) -> torch.Tensor:
    """Return normalized [0, 1] xy coordinates from heatmap probabilities."""

    if not torch.jit.is_tracing() and prob.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(prob.shape)}")
    b, c, h, w = prob.shape
    if not torch.jit.is_tracing() and c != 1:
        raise ValueError("LocTouchNetLite expects a single heatmap channel")
    xs = torch.linspace(0.0, 1.0, w, dtype=prob.dtype, device=prob.device).view(1, 1, 1, w)
    ys = torch.linspace(0.0, 1.0, h, dtype=prob.dtype, device=prob.device).view(1, 1, h, 1)
    x = (prob * xs).sum(dim=(2, 3))
    y = (prob * ys).sum(dim=(2, 3))
    return torch.cat([x, y], dim=1).reshape(b, 2)


def heatmap_peak_confidence(prob: torch.Tensor) -> torch.Tensor:
    """Use heatmap peakiness as a location confidence in [0, 1]."""

    b = prob.shape[0]
    flat = prob.reshape(b, -1)
    peak = flat.max(dim=1).values
    uniform = torch.ones_like(peak) / flat.size(1)
    denom = (1.0 - uniform).clamp_min(1e-6)
    return ((peak - uniform) / denom).clamp(0.0, 1.0)
