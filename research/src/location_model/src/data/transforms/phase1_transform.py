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


def _apply_homography(H: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    pt = np.array([[xy]], dtype=np.float64)  # (1,1,2)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


class Phase1Transform:
    def __init__(
        self,
        dataset_root: Path,
        index_version: str,
        split: str,
        out_w: int = 480,
        out_h: int = 320,
        margin_frac: float = 0.10,
        normalize: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = index_version
        self.split = split
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.margin_frac = float(margin_frac)
        self.normalize = bool(normalize)

        manifest_path = self.dataset_root / "indices" / f"manifest_{self.index_version}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest: Dict[str, Any] = _read_json(manifest_path)

        self.video_reader = VideoFrameReader(self.dataset_root, self.manifest)

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
        H_crop_to_surf = warp["H_crop_to_surf"]
        crop_origin = warp["crop_origin"]

        # Touch: treat "pressure present" as touch=1, absent as touch=0
        # Touch + raw tablet xy from schema helper
        touch, touch_mask, x, y, xy_mask = extract_xy_touch(labels, label_mask)

        # Decide when xy is valid:
        # Use xy_mask from the labels, and optionally require touch
        xy_valid = (xy_mask > 0.5)   # or just (xy_mask > 0.5)

        TAB_W = 44800.0
        TAB_H = 29600.0

        if xy_valid:
            # Tablet -> normalized plane coords in [0,1]
            x_norm = x / TAB_W
            y_norm = y / TAB_H

            # Optional: clamp to a small tolerance to avoid wild labels
            x_norm = float(np.clip(x_norm, -0.1, 1.1))
            y_norm = float(np.clip(y_norm, -0.1, 1.1))

            xy_norm = (x_norm, y_norm)
            xy_mask = 1.0
        else:
            xy_norm = (0.0, 0.0)
            xy_mask = 0.0

        img_rgb = img_surf_bgr[:, :, ::-1]
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.float32)
        if self.normalize:
            img_rgb = img_rgb / 255.0

        image_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()

        return {
            "image": image_t,
            "touch": torch.tensor(touch, dtype=torch.float32),
            "touch_mask": torch.tensor(touch_mask, dtype=torch.float32),
            "xy": torch.tensor(xy_norm, dtype=torch.float32),
            "xy_mask": torch.tensor(xy_mask, dtype=torch.float32),
            "meta": meta,
        }

    def close(self) -> None:
        self.video_reader.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
