from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2  # type: ignore
import numpy as np
import torch

from data.label_schema import extract_xy_touch
from data.video_io import VideoFrameReader


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase2eTransform_v1:
    """
    Single-frame transform for Phase2e:
      - image: raw RGB resized to (out_w, out_h), normalized optionally.
      - corners: current 4 corners as normalized 8-float aux chunk.
      - history: previous XY points from sample["history_xy"], flattened.
      - target: current normalized XY.
    """

    def __init__(
        self,
        dataset_root: Path,
        index_version: str,
        split: str,
        out_w: int = 480,
        out_h: int = 320,
        normalize: bool = True,
        history_len: int = 3,
        require_touch_for_xy: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = index_version
        self.split = split
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.normalize = bool(normalize)
        self.history_len = int(history_len)
        self.require_touch_for_xy = bool(require_touch_for_xy)
        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")

        manifest_path = self.dataset_root / "indices" / f"manifest_{self.index_version}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest: Dict[str, Any] = _read_json(manifest_path)
        self.video_reader = VideoFrameReader(self.dataset_root, self.manifest)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        corners = sample["corners"]
        labels = sample["labels"]
        label_mask = sample["label_mask"]
        history_xy = sample.get("history_xy", np.zeros((self.history_len, 2), dtype=np.float32))
        meta = sample.get("meta", {})

        file_id = meta.get("file_id")
        frame_idx = meta.get("frame_idx")
        if file_id is None or frame_idx is None:
            raise KeyError("sample['meta'] missing file_id or frame_idx")

        frame_bgr = self.video_reader.get_frame(str(file_id), int(frame_idx), self.split)
        orig_h, orig_w = frame_bgr.shape[:2]
        frame_bgr_rs = cv2.resize(frame_bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

        c = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if c.shape[0] != 4:
            raise ValueError(f"Expected 4 corners, got shape={c.shape}")
        c_norm = np.zeros((4, 2), dtype=np.float32)
        c_norm[:, 0] = c[:, 0] / float(orig_w)
        c_norm[:, 1] = c[:, 1] / float(orig_h)
        corners_t = torch.from_numpy(c_norm.reshape(8)).to(dtype=torch.float32)

        h = np.asarray(history_xy, dtype=np.float32).reshape(-1, 2)
        if h.shape[0] < self.history_len:
            pad = np.zeros((self.history_len - h.shape[0], 2), dtype=np.float32)
            h = np.concatenate([pad, h], axis=0)
        if h.shape[0] > self.history_len:
            h = h[-self.history_len :]
        h = np.clip(h, 0.0, 1.0)
        history_t = torch.from_numpy(h.reshape(self.history_len * 2)).to(dtype=torch.float32)

        touch, touch_mask, x, y, xy_mask = extract_xy_touch(labels, label_mask)
        xy_valid = (xy_mask > 0.5)
        if self.require_touch_for_xy:
            xy_valid = xy_valid and (float(touch_mask) > 0.5) and (float(touch) > 0.01)

        tab_w = 44800.0
        tab_h = 29600.0
        if xy_valid:
            x_norm = float(np.clip(x / tab_w, 0.0, 1.0))
            y_norm = float(np.clip(y / tab_h, 0.0, 1.0))
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
            "history": history_t,
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
