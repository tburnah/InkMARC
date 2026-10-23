from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


class SyntheticInteractionDataset(Dataset):
    """Tiny learnability dataset with simple state-colored interaction markers."""

    def __init__(self, size: int = 64, height: int = 320, width: int = 480, seed: int = 7, noise: float = 0.02):
        self.size = size
        self.height = height
        self.width = width
        self.seed = seed
        self.noise = noise

    def __len__(self) -> int:
        return self.size

    def _draw_marker(
        self,
        image: torch.Tensor,
        *,
        cx: int,
        cy: int,
        state: int,
        radius_override: Optional[int] = None,
        color_override: Optional[torch.Tensor] = None,
    ) -> None:
        radius = radius_override if radius_override is not None else (5 if state == 1 else 8)
        color = color_override if color_override is not None else (
            torch.tensor([0.15, 0.85, 0.25]) if state == 1 else torch.tensor([0.95, 0.2, 0.15])
        )
        y0, y1 = max(0, cy - radius), min(self.height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(self.width, cx + radius + 1)
        yy = torch.arange(y0, y1).view(-1, 1)
        xx = torch.arange(x0, x1).view(1, -1)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        patch = image[:, y0:y1, x0:x1]
        patch[:, mask] = color.view(3, 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        image = torch.rand(3, self.height, self.width, generator=gen) * self.noise
        prev_image = image.clone()
        state = idx % 3
        valid = 1.0 if state in (1, 2) else 0.0
        touch = 1.0 if state == 2 else 0.0
        has_xy = valid
        if valid:
            margin_x = max(8, self.width // 20)
            margin_y = max(8, self.height // 20)
            cx = int(torch.randint(margin_x, self.width - margin_x, (1,), generator=gen).item())
            cy = int(torch.randint(margin_y, self.height - margin_y, (1,), generator=gen).item())
            self._draw_marker(image, cx=cx, cy=cy, state=state)
            prev_cx = max(margin_x, min(self.width - margin_x - 1, cx - 6))
            prev_cy = max(margin_y, min(self.height - margin_y - 1, cy - 4))
            prev_state = 1 if state == 2 else state
            self._draw_marker(
                prev_image,
                cx=prev_cx,
                cy=prev_cy,
                state=prev_state,
                radius_override=4 if state == 1 else 5,
            )
            xy = torch.tensor([cx / (self.width - 1), cy / (self.height - 1)], dtype=torch.float32)
        else:
            xy = torch.zeros(2, dtype=torch.float32)
        motion_map = torch.mean(torch.abs(image - prev_image), dim=0, keepdim=True)
        return {
            "image": image.clamp(0.0, 1.0),
            "image_t_minus_1": prev_image.clamp(0.0, 1.0),
            "motion_map": motion_map.clamp(0.0, 1.0),
            "dt_prev": torch.tensor(1.0 / 30.0, dtype=torch.float32),
            "has_prev_frame": torch.tensor(1.0 if valid else 0.0, dtype=torch.float32),
            "valid": torch.tensor(valid, dtype=torch.float32),
            "touch": torch.tensor(touch, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.long),
            "xy": xy,
            "has_xy": torch.tensor(has_xy, dtype=torch.float32),
        }


def make_synthetic_batch(batch_size: int = 4, height: int = 320, width: int = 480, seed: int = 11) -> Dict[str, torch.Tensor]:
    ds = SyntheticInteractionDataset(size=batch_size, height=height, width=width, seed=seed)
    samples = [ds[i] for i in range(batch_size)]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0].keys()}
