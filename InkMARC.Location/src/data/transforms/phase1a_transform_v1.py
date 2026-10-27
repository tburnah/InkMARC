# data/transforms/phase1a_transform_v1.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2  # type: ignore
import numpy as np
import torch

from data.video_io import VideoFrameReader
from data.label_schema import extract_xy_touch


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase1aTransform_v1:
    """
    Phase 1a:
      - No homography.
      - Feed raw RGB (resized to out_w/out_h) + normalised corners.
      - Labels: touch + xy (normalised to [0,1] in tablet space), consistent with Phase 1.
    """

    def __init__(
        self,
        dataset_root: Path,
        index_version: str,
        split: str,
        out_w: int = 480,
        out_h: int = 320,
        normalize: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = index_version
        self.split = split
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.normalize = bool(normalize)

        manifest_path = self.dataset_root / "indices" / f"manifest_{self.index_version}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest: Dict[str, Any] = _read_json(manifest_path)

        self.video_reader = VideoFrameReader(self.dataset_root, self.manifest)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        corners = sample["corners"]          # expected shape-ish: 4 points (x,y)
        labels = sample["labels"]
        label_mask = sample["label_mask"]
        meta = sample.get("meta", {})

        file_id = meta.get("file_id")
        frame_idx = meta.get("frame_idx")
        if file_id is None or frame_idx is None:
            raise KeyError("sample['meta'] missing file_id or frame_idx")

        frame_bgr = self.video_reader.get_frame(str(file_id), int(frame_idx), self.split)
        orig_h, orig_w = frame_bgr.shape[:2]

        # Resize raw frame to fixed net input size (no crop/warp)
        frame_bgr_rs = cv2.resize(frame_bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

        # Convert corners to float32 array shape (4,2)
        c = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if c.shape[0] != 4:
            raise ValueError(f"Expected 4 corners, got shape={c.shape}")

        # Normalise to image space using ORIGINAL frame dims (ratio invariant under resize)
        # NOTE: corners may be negative or > image bounds; keep them (do not clamp).
        c_norm = np.zeros((4, 2), dtype=np.float32)
        c_norm[:, 0] = c[:, 0] / float(orig_w)
        c_norm[:, 1] = c[:, 1] / float(orig_h)

        corners_t = torch.from_numpy(c_norm.reshape(8)).to(dtype=torch.float32)

        # Touch + xy from schema helper
        touch, touch_mask, x, y, xy_mask = extract_xy_touch(labels, label_mask)

        # XY validity rule: identical to Phase 1 v2 (no gating based on touch).
        xy_valid = (xy_mask > 0.5)

        TAB_W = 44800.0
        TAB_H = 29600.0

        if xy_valid:
            x_norm = float(np.clip(x / TAB_W, 0.0, 1.0))
            y_norm = float(np.clip(y / TAB_H, 0.0, 1.0))
            xy_out = (x_norm, y_norm)
            xy_mask_out = 1.0
        else:
            xy_out = (0.0, 0.0)
            xy_mask_out = 0.0

        img_rgb = frame_bgr_rs[:, :, ::-1]
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.float32)
        if self.normalize:
            img_rgb = img_rgb / 255.0
        image_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()

        return {
            "image": image_t,
            "corners": corners_t,
            "xy": torch.tensor(xy_out, dtype=torch.float32),
            "xy_mask": torch.tensor(xy_mask_out, dtype=torch.float32),
            "meta": meta,
        }

    def close(self) -> None:
        self.video_reader.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
