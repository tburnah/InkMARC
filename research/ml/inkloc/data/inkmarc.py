from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


INKMARC_SRC_DEFAULT = Path(__file__).resolve().parents[4] / "InkMARC.Location" / "src"


def _ensure_inkmarc_imports(inkmarc_src: str | Path = INKMARC_SRC_DEFAULT):
    src = Path(inkmarc_src)
    if not src.exists():
        raise FileNotFoundError(f"Missing InkMARC source tree: {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    from combined_touch_location.combined_transform import CombinedTouchLocationTransform
    from data.dataset_core_v2 import InkMARCIndexedDataset

    return InkMARCIndexedDataset, CombinedTouchLocationTransform


def sequence_key_for_record(rec: Dict) -> str:
    variant = rec.get("video_rel_path")
    if variant:
        return str(variant)
    return str(rec.get("file_id") or rec.get("rel_path") or "")


def build_prev_index(records: List[Dict]) -> list[Optional[int]]:
    prev_index: list[Optional[int]] = [None] * len(records)
    by_sequence: Dict[str, list[tuple[int, int]]] = {}
    for idx, rec in enumerate(records):
        key = sequence_key_for_record(rec)
        by_sequence.setdefault(key, []).append((int(rec.get("frame_idx", 0)), idx))
    for items in by_sequence.values():
        items.sort()
        last_frame_idx = None
        last_pos = None
        for frame_idx, pos in items:
            if last_frame_idx is not None and frame_idx == last_frame_idx + 1:
                prev_index[pos] = last_pos
            last_frame_idx = frame_idx
            last_pos = pos
    return prev_index


def _gray(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (image.unsqueeze(0) * weights).sum(dim=1, keepdim=True).squeeze(0)


def compute_motion_map(current: torch.Tensor, previous: torch.Tensor, motion_type: str) -> torch.Tensor:
    if motion_type == "abs_gray_diff":
        return (_gray(current) - _gray(previous)).abs()
    if motion_type == "signed_gray_diff":
        return _gray(current) - _gray(previous)
    if motion_type == "rgb_diff":
        return current - previous
    raise ValueError(f"Unknown temporal.motion_type: {motion_type}")


class InkMARCCombinedDataset(Dataset):
    """Adapter for indexed LocationData with optional temporal inputs.

    Location supervision remains single-frame. Optional temporal fields are:
      - image_t_minus_1
      - motion_map
      - dt_prev
      - has_prev_frame
    """

    def __init__(
        self,
        dataset_root: str | Path = "/data/LocationData",
        index_version: str = "v1",
        split: str = "train",
        inkmarc_src: str | Path = INKMARC_SRC_DEFAULT,
        input_height: int = 320,
        input_width: int = 480,
        margin_frac: float = 0.10,
        heatmap_height: int = 40,
        heatmap_width: int = 60,
        heatmap_sigma: float = 1.5,
        touch_pressure_threshold: float = 0.0,
        require_labels: bool = True,
        video_cache_size: int = 16,
        temporal_enabled: bool = False,
        temporal_input_mode: str = "motion_map",
        motion_channels: int = 1,
        motion_type: str = "abs_gray_diff",
    ) -> None:
        InkMARCIndexedDataset, CombinedTouchLocationTransform = _ensure_inkmarc_imports(inkmarc_src)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_input_mode = str(temporal_input_mode)
        self.motion_channels = int(motion_channels)
        self.motion_type = str(motion_type)
        self.transform = CombinedTouchLocationTransform(
            dataset_root=Path(dataset_root),
            index_version=index_version,
            split=split,
            out_w=input_width,
            out_h=input_height,
            margin_frac=margin_frac,
            normalize=True,
            hm_w=heatmap_width,
            hm_h=heatmap_height,
            hm_sigma=heatmap_sigma,
            touch_pressure_threshold=touch_pressure_threshold,
            video_cache_size=video_cache_size,
        )
        self.dataset = InkMARCIndexedDataset(
            dataset_root,
            index_version=index_version,
            split=split,
            require_labels=require_labels,
            transform=None,
        )
        self.prev_index = build_prev_index(self.dataset._records)

    def __len__(self) -> int:
        return len(self.dataset)

    def _transform(self, raw_sample: Dict) -> Dict:
        return self.transform(raw_sample)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_sample = self.dataset[idx]
        sample = self._transform(raw_sample)
        state = sample["state"].long()
        has_xy = sample["xy_mask"].float()
        valid = (state > 0).to(torch.float32)
        meta = sample.get("meta", {})
        participant_id = meta.get("participant_id")
        file_id = str(meta.get("file_id") or meta.get("rel_path") or "")
        sequence_hash = int(hashlib.sha1(file_id.encode("utf-8")).hexdigest()[:8], 16)
        frame_idx = int(meta.get("frame_idx") or 0)

        item = {
            "image": sample["image"].float(),
            "image_t": sample["image"].float(),
            "valid": valid,
            "touch": sample["touch"].float(),
            "state": state,
            "xy": sample["xy"].float(),
            "has_xy": has_xy,
            "heatmap_target": sample["heatmap"].float(),
            "heatmap_mask": sample["heatmap_mask"].float(),
            "participant_id": torch.tensor(int(participant_id) if participant_id is not None else -1, dtype=torch.long),
            "sequence_id": torch.tensor(sequence_hash, dtype=torch.long),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
            "dataset_index": torch.tensor(int(idx), dtype=torch.long),
        }

        if self.temporal_enabled:
            prev_pos = self.prev_index[idx]
            if prev_pos is None:
                prev_image = item["image"].clone()
                motion_map = torch.zeros(
                    self.motion_channels,
                    item["image"].shape[1],
                    item["image"].shape[2],
                    dtype=item["image"].dtype,
                )
                dt_prev = 0.0
                has_prev_frame = 0.0
            else:
                prev_raw = self.dataset[prev_pos]
                prev_sample = self._transform(prev_raw)
                prev_image = prev_sample["image"].float()
                base_motion = compute_motion_map(item["image"], prev_image, self.motion_type)
                if self.motion_type != "rgb_diff" and self.motion_channels != 1:
                    raise ValueError(f"motion_type={self.motion_type} requires motion_channels=1")
                if self.motion_type == "rgb_diff" and self.motion_channels != 3:
                    raise ValueError("motion_type=rgb_diff requires motion_channels=3")
                motion_map = base_motion
                ts_cur = float(raw_sample.get("timestamps_ns", 0))
                ts_prev = float(prev_raw.get("timestamps_ns", ts_cur))
                dt_prev = max(0.0, (ts_cur - ts_prev) / 1e9)
                has_prev_frame = 1.0

            item["image_t_minus_1"] = prev_image
            item["motion_map"] = motion_map
            item["dt_prev"] = torch.tensor(dt_prev, dtype=torch.float32)
            item["has_prev_frame"] = torch.tensor(has_prev_frame, dtype=torch.float32)

        return item

    def close(self) -> None:
        close_dataset = getattr(self.dataset, "close", None)
        if callable(close_dataset):
            close_dataset()
        close_transform = getattr(self.transform, "close", None)
        if callable(close_transform):
            close_transform()
