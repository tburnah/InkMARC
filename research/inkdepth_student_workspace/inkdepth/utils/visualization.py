from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], [2, 98])
    out = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _colorize_gray(arr: np.ndarray) -> Image.Image:
    gray = _normalize(arr)
    rgb = np.stack([gray, np.flipud(gray), 255 - gray], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


def save_debug_panel(
    path: str | Path,
    image: torch.Tensor,
    target_depth: torch.Tensor,
    pred_depth: torch.Tensor,
    heatmap: torch.Tensor | None,
    xy: torch.Tensor | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    rgb = Image.fromarray(img_np)
    h, w = img_np.shape[:2]

    tgt = target_depth.detach().cpu().squeeze().numpy()
    pred = pred_depth.detach().cpu().squeeze().numpy()
    err = np.abs(pred - tgt)
    panels = [rgb, _colorize_gray(tgt), _colorize_gray(pred), _colorize_gray(err)]

    if heatmap is not None:
        hm = heatmap.detach().cpu()
        if hm.ndim == 2:
            hm = hm[None, None]
        elif hm.ndim == 3:
            hm = hm[None]
        elif hm.ndim == 4:
            hm = hm[:1]
        else:
            raise ValueError(f"Expected heatmap with 2-4 dims, got shape {tuple(hm.shape)}")
        hm = torch.softmax(hm.flatten(2), dim=-1).reshape_as(hm)
        hm = F.interpolate(hm, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
        hm_img = _colorize_gray(hm)
    else:
        hm_img = Image.new("RGB", (w, h), (0, 0, 0))
    if xy is not None:
        draw = ImageDraw.Draw(hm_img)
        x = float(xy.detach().cpu()[0]) * (w - 1)
        y = float(xy.detach().cpu()[1]) * (h - 1)
        r = max(4, min(h, w) // 80)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=2)
        draw.line((x - 2 * r, y, x + 2 * r, y), fill=(255, 255, 255), width=1)
        draw.line((x, y - 2 * r, x, y + 2 * r), fill=(255, 255, 255), width=1)
    panels.append(hm_img)

    canvas = Image.new("RGB", (w * len(panels), h), (20, 20, 20))
    for i, panel in enumerate(panels):
        canvas.paste(panel.resize((w, h)), (i * w, 0))
    canvas.save(path)
