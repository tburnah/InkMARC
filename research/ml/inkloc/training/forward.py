from __future__ import annotations

from typing import Dict


def model_forward(model, batch: Dict, touch_ablation_mode: str = "rgb_motion", **extra_kwargs):
    image = batch["image_t"] if "image_t" in batch else batch["image"]
    kwargs = {"touch_ablation_mode": touch_ablation_mode}
    input_mode = getattr(model, "temporal_input_mode", "motion_map")
    if input_mode == "image_t_minus_1":
        if "image_t_minus_1" in batch:
            kwargs["image_t_minus_1"] = batch["image_t_minus_1"]
        elif "prev_image" in batch:
            kwargs["image_t_minus_1"] = batch["prev_image"]
    else:
        if "motion_map" in batch:
            kwargs["motion_map"] = batch["motion_map"]
        elif "image_t_minus_1" in batch:
            kwargs["image_t_minus_1"] = batch["image_t_minus_1"]
        elif "prev_image" in batch:
            kwargs["image_t_minus_1"] = batch["prev_image"]
    kwargs.update(extra_kwargs)
    return model(image, **kwargs)
