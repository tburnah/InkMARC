from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def load_rgb_image(path: str, height: int = 320, width: int = 480) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
