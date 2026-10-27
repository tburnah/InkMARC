from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2  # type: ignore
import numpy as np
import torch

from data.label_schema import extract_xy_touch
from data.transforms.homography_warp import warp_to_canonical
from data.video_io_v2 import VideoFrameReader


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class CombinedTouchLocationTransform:
    """
    Transform an indexed location sample into combined touch/location supervision.

    State is derived from xy validity plus pressure in labels[2]:
      0 = background (no valid xy)
      1 = hover      (valid xy, pressure <= threshold)
      2 = touch      (valid xy, pressure > threshold)
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        index_version: str,
        split: str,
        out_w: int = 480,
        out_h: int = 320,
        margin_frac: float = 0.10,
        normalize: bool = True,
        hm_w: int = 60,
        hm_h: int = 40,
        hm_sigma: float = 1.5,
        touch_pressure_threshold: float = 0.0,
        video_cache_size: int = 16,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = str(index_version)
        self.split = str(split)
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.margin_frac = float(margin_frac)
        self.normalize = bool(normalize)
        self.hm_w = int(hm_w)
        self.hm_h = int(hm_h)
        self.hm_sigma = float(hm_sigma)
        self.touch_pressure_threshold = float(touch_pressure_threshold)
        self.video_cache_size = max(1, int(video_cache_size))

        self._hm_xx, self._hm_yy = np.meshgrid(
            np.arange(self.hm_w, dtype=np.float32),
            np.arange(self.hm_h, dtype=np.float32),
        )

        manifest_path = self.dataset_root / "indices" / f"manifest_{self.index_version}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest = _read_json(manifest_path)
        self.video_reader = VideoFrameReader(
            self.dataset_root,
            self.manifest,
            cache_size=self.video_cache_size,
        )

    def _make_gaussian_heatmap(self, cx: float, cy: float) -> np.ndarray:
        g = np.exp(
            -(
                (self._hm_xx - cx) ** 2
                + (self._hm_yy - cy) ** 2
            )
            / (2.0 * self.hm_sigma * self.hm_sigma)
        )
        s = float(g.sum())
        if s > 0.0:
            g /= s
        return g[None, :, :].astype(np.float32)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        corners = np.asarray(sample["corners"], dtype=np.float32)
        labels = np.asarray(sample["labels"], dtype=np.float32)
        label_mask = np.asarray(sample["label_mask"], dtype=np.uint8)
        meta = sample.get("meta", {})

        file_id = meta.get("file_id")
        frame_idx = meta.get("frame_idx")
        if file_id is None or frame_idx is None:
            raise KeyError("sample['meta'] missing file_id or frame_idx")

        variant_rel = meta.get("video_rel_path")
        if variant_rel:
            frame_bgr = self.video_reader.get_frame_from_variant(str(variant_rel), int(frame_idx))
        else:
            frame_bgr = self.video_reader.get_frame(str(file_id), int(frame_idx), self.split)

        warp = warp_to_canonical(
            frame_bgr,
            corners,
            out_w=self.out_w,
            out_h=self.out_h,
            margin_frac=self.margin_frac,
        )
        img_surf_bgr = warp["img_surf_bgr"]

        # Pressure-derived touch label. labels[2] is pressure; if pressure is
        # not valid in label_mask, extract_xy_touch returns touch=0.
        touch, _touch_mask, x, y, xy_mask = extract_xy_touch(
            labels,
            label_mask,
            pressure_touch_threshold=self.touch_pressure_threshold,
        )

        tab_w = 44800.0
        tab_h = 29600.0
        xy_valid = xy_mask > 0.5
        if xy_valid:
            x_norm = float(np.clip(x / tab_w, 0.0, 1.0))
            y_norm = float(np.clip(y / tab_h, 0.0, 1.0))
            xy_norm = (x_norm, y_norm)
            xy_mask_out = 1.0
            state = 2 if touch > 0.5 else 1
        else:
            xy_norm = (0.0, 0.0)
            xy_mask_out = 0.0
            state = 0

        img_rgb = cv2.cvtColor(img_surf_bgr, cv2.COLOR_BGR2RGB)
        image_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
        if self.normalize:
            image_t = image_t.to(torch.float32).div_(255.0)

        if xy_mask_out > 0.5:
            cx = xy_norm[0] * (self.hm_w - 1)
            cy = xy_norm[1] * (self.hm_h - 1)
            heatmap = self._make_gaussian_heatmap(cx, cy)
            heatmap_mask = 1.0
        else:
            heatmap = np.zeros((1, self.hm_h, self.hm_w), dtype=np.float32)
            heatmap_mask = 0.0

        return {
            "image": image_t,
            "state": torch.tensor(int(state), dtype=torch.long),
            "touch": torch.tensor(float(touch), dtype=torch.float32),
            "xy": torch.tensor(xy_norm, dtype=torch.float32),
            "xy_mask": torch.tensor(float(xy_mask_out), dtype=torch.float32),
            "heatmap": torch.from_numpy(heatmap),
            "heatmap_mask": torch.tensor(float(heatmap_mask), dtype=torch.float32),
            "meta": meta,
        }

    def close(self) -> None:
        self.video_reader.release()
