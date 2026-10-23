from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


X_IDX = 0
Y_IDX = 1
P_IDX = 2
TX_IDX = 3
TY_IDX = 4


def extract_xy_touch(
    labels: np.ndarray,
    label_mask: np.ndarray,
    *,
    pressure_touch_threshold: float = 0,
    require_xy_for_touch: bool = False,
) -> Tuple[float, float, float, float, float]:
    """
    Returns:
      touch: float in {0,1}
      touch_mask: float in {0,1}  (whether touch supervision is valid)
      x: float (tablet coord)
      y: float (tablet coord)
      xy_mask: float in {0,1}  (whether x/y are valid)
    """

    labels = np.asarray(labels)
    label_mask = np.asarray(label_mask)

    x_ok = float(label_mask[X_IDX] != 0)
    y_ok = float(label_mask[Y_IDX] != 0)
    p_ok = float(label_mask[P_IDX] != 0)

    x = float(labels[X_IDX]) if x_ok > 0.5 else 0.0
    y = float(labels[Y_IDX]) if y_ok > 0.5 else 0.0

    p = float(labels[P_IDX]) if p_ok > 0.5 else 0.0

    # touch is derived from pressure MAGNITUDE (not presence)
    if p_ok > 0.5:
        touch = 1.0 if p > float(pressure_touch_threshold) else 0.0
        touch_mask = 1.0
    else:
        touch = 0.0
        touch_mask = 0.0  # no supervision if pressure not recorded

    xy_mask = 1.0 if (x_ok > 0.5 and y_ok > 0.5) else 0.0

    # Optional: only consider touch-labeled when xy exists (prevents weird supervision)
    if require_xy_for_touch and xy_mask < 0.5:
        touch_mask = 0.0

    return touch, touch_mask, x, y, xy_mask
