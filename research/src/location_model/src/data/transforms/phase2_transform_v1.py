from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2  # type: ignore
import numpy as np
import torch

from data.transforms.homography_warp import warp_to_canonical
from data.video_io import VideoFrameReader
from data.label_schema import extract_xy_touch


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

class Phase2Transform_v1:
    def __init__(
        self,
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
        require_touch_for_xy: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = index_version
        self.split = split
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.margin_frac = float(margin_frac)
        self.normalize = bool(normalize)

        self.hm_w = int(hm_w)
        self.hm_h = int(hm_h)
        self.hm_sigma = float(hm_sigma)
        self.require_touch_for_xy = bool(require_touch_for_xy)
        self._hm_xx, self._hm_yy = np.meshgrid(np.arange(self.hm_w, dtype=np.float32), np.arange(self.hm_h, dtype=np.float32))
        
        manifest_path = self.dataset_root / "indices" / f"manifest_{self.index_version}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest: Dict[str, Any] = _read_json(manifest_path)

        self.video_reader = VideoFrameReader(self.dataset_root, self.manifest)

    def _make_gaussian_heatmap(self, cx: float, cy: float, sigma: float) -> np.ndarray:
        """
        Returns (1,H,W) float32 heatmap, normalized to sum=1.
        cx,cy in heatmap pixel coordinates (0..w-1, 0..h-1).
        """
        xx, yy = self._hm_xx, self._hm_yy
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        s = float(g.sum())
        if s > 0:
            g /= s
        return g[None, :, :].astype(np.float32)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        corners = sample["corners"]
        labels = sample["labels"]
        label_mask = sample["label_mask"]
        meta = sample.get("meta", {})

        file_id = meta.get("file_id")
        frame_idx = meta.get("frame_idx")
        if file_id is None or frame_idx is None:
            raise KeyError("sample['meta'] missing file_id or frame_idx")

        frame_bgr = self.video_reader.get_frame(str(file_id), int(frame_idx), self.split)
        warp = warp_to_canonical(
            frame_bgr,
            corners,
            out_w=self.out_w,
            out_h=self.out_h,
            margin_frac=self.margin_frac,
        )
        img_surf_bgr = warp["img_surf_bgr"]

        # Touch + raw tablet xy
        touch, touch_mask, x, y, xy_mask = extract_xy_touch(labels, label_mask)

        TAB_W = 44800.0
        TAB_H = 29600.0

        xy_valid = (xy_mask > 0.5)
        if self.require_touch_for_xy:
            xy_valid = xy_valid and (float(touch_mask) > 0.5) and (float(touch) > 0.01)

        if xy_valid:
            x_norm = float(np.clip(x / TAB_W, 0.0, 1.0))
            y_norm = float(np.clip(y / TAB_H, 0.0, 1.0))
            xy_norm = (x_norm, y_norm)
            xy_mask_out = 1.0
        else:
            xy_norm = (0.0, 0.0)
            xy_mask_out = 0.0

        # Image tensor
        img_rgb = cv2.cvtColor(img_surf_bgr, cv2.COLOR_BGR2RGB) 
        image_t = torch.from_numpy(img_rgb).permute(2,0,1).contiguous()
        if self.normalize:
            image_t = image_t.to(torch.float32).div_(255.0)         

        # Heatmap target
        if xy_mask_out > 0.5:
            cx = xy_norm[0] * (self.hm_w - 1)
            cy = xy_norm[1] * (self.hm_h - 1)
            hm = self._make_gaussian_heatmap(cx, cy, self.hm_sigma)
            hm_mask = 1.0
        else:
            hm = np.zeros((1, self.hm_h, self.hm_w), dtype=np.float32)
            hm_mask = 0.0

        return {
            "image": image_t,
            "touch": torch.tensor(touch, dtype=torch.float32),
            "touch_mask": torch.tensor(touch_mask, dtype=torch.float32),
            "xy": torch.tensor(xy_norm, dtype=torch.float32),
            "xy_mask": torch.tensor(xy_mask_out, dtype=torch.float32),
            "heatmap": torch.from_numpy(hm),
            "heatmap_mask": torch.tensor(hm_mask, dtype=torch.float32),
            "meta": meta,
        }

    def close(self) -> None:
        self.video_reader.release()
