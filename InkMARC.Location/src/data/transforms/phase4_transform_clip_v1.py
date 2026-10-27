from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.transforms.homography_warp import warp_to_canonical  # :contentReference[oaicite:7]{index=7}
from data.video_io import VideoFrameReader                     # :contentReference[oaicite:8]{index=8}
from data.label_schema import extract_xy_touch                 # we'll ignore touch output :contentReference[oaicite:9]{index=9}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    s = float(g.sum())
    if s > 0:
        g /= s
    return g[None, :, :].astype(np.float32)


class Phase4TransformClip_v1:
    """
    Produces a causal clip of warped frames: image_clip (T,C,H,W)
    Supervision is for the LAST frame only: xy, xy_mask, heatmap, heatmap_mask.

    Requires the dataset __getitem__ to provide:
      - sample["meta"]["file_id"], sample["meta"]["frame_idx"]
      - sample["corners"] for the LAST frame
      - sample["labels"], sample["label_mask"] for the LAST frame
    """

    def __init__(
        self,
        dataset_root: Path,
        index_version: str,
        split: str,
        clip_len: int = 5,
        out_w: int = 480,
        out_h: int = 320,
        margin_frac: float = 0.10,
        normalize: bool = True,
        hm_w: int = 60,
        hm_h: int = 40,
        hm_sigma: float = 1.5,
        require_touch_for_xy: bool = False,  # keep for parity, but you likely want False now
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = index_version
        self.split = split

        self.clip_len = int(clip_len)
        if self.clip_len <= 0:
            raise ValueError("clip_len must be >= 1")

        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.margin_frac = float(margin_frac)
        self.normalize = bool(normalize)

        self.hm_w = int(hm_w)
        self.hm_h = int(hm_h)
        self.hm_sigma = float(hm_sigma)
        self.require_touch_for_xy = bool(require_touch_for_xy)

        manifest_path = self.dataset_root / "indices" / f"manifest_{self.index_version}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest: Dict[str, Any] = _read_json(manifest_path)

        self.video_reader = VideoFrameReader(self.dataset_root, self.manifest)

    def _get_frame_bgr(self, file_id: str, frame_idx: int) -> np.ndarray:
        # VideoFrameReader already selects random variant on train split :contentReference[oaicite:10]{index=10}
        return self.video_reader.get_frame(str(file_id), int(frame_idx), self.split)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        meta = sample.get("meta", {})
        file_id = meta.get("file_id")
        frame_idx = meta.get("frame_idx")
        if file_id is None or frame_idx is None:
            raise KeyError("sample['meta'] missing file_id or frame_idx")

        # Labels/corners are only for LAST frame (current idx)
        corners_last = sample["corners"]
        labels_last = sample["labels"]
        label_mask_last = sample["label_mask"]

        # Build causal indices [t-T+1 ... t], clamped at 0
        t = int(frame_idx)
        idxs = [max(0, t - (self.clip_len - 1) + k) for k in range(self.clip_len)]

        # NOTE: We warp each frame using the LAST frame corners.
        # This keeps Phase 4 window “temporal factor only” and avoids needing historical corners.
        # It is a deliberate systems choice; we can later upgrade to per-frame corners if desired.
        frames_t: List[torch.Tensor] = []
        for fi in idxs:
            frame_bgr = self._get_frame_bgr(str(file_id), fi)
            warp = warp_to_canonical(
                frame_bgr,
                corners_last,
                out_w=self.out_w,
                out_h=self.out_h,
                margin_frac=self.margin_frac,
            )
            img_surf_bgr = warp["img_surf_bgr"]

            img_rgb = img_surf_bgr[:, :, ::-1]
            img_rgb = np.ascontiguousarray(img_rgb, dtype=np.float32)
            if self.normalize:
                img_rgb = img_rgb / 255.0
            image_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()  # (C,H,W)
            frames_t.append(image_t)

        image_clip = torch.stack(frames_t, dim=0)  # (T,C,H,W)

        # Extract xy supervision (last frame)
        touch, touch_mask, x, y, xy_mask = extract_xy_touch(labels_last, label_mask_last)

        TAB_W = 44800.0
        TAB_H = 29600.0

        xy_valid = (xy_mask > 0.5)
        if self.require_touch_for_xy:
            xy_valid = xy_valid and (touch_mask > 0.5) and (touch > 0.5)

        if xy_valid:
            x_norm = float(np.clip(x / TAB_W, 0.0, 1.0))
            y_norm = float(np.clip(y / TAB_H, 0.0, 1.0))
            xy_norm = (x_norm, y_norm)
            xy_mask_out = 1.0
        else:
            xy_norm = (0.0, 0.0)
            xy_mask_out = 0.0

        # Heatmap target (last frame)
        if xy_mask_out > 0.5:
            cx = xy_norm[0] * (self.hm_w - 1)
            cy = xy_norm[1] * (self.hm_h - 1)
            hm = _make_gaussian_heatmap(self.hm_h, self.hm_w, cx, cy, self.hm_sigma)
            hm_mask = 1.0
        else:
            hm = np.zeros((1, self.hm_h, self.hm_w), dtype=np.float32)
            hm_mask = 0.0

        return {
            "image_clip": image_clip,  # (T,C,H,W)
            "xy": torch.tensor(xy_norm, dtype=torch.float32),
            "xy_mask": torch.tensor(xy_mask_out, dtype=torch.float32),
            "heatmap": torch.from_numpy(hm),
            "heatmap_mask": torch.tensor(hm_mask, dtype=torch.float32),
            "meta": meta,
        }

    def close(self) -> None:
        self.video_reader.release()
