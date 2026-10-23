from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import cv2  # type: ignore


class VideoFrameReader:
    """
    Simple LRU-cached video frame reader for InkMARC AVI variants.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: Dict[str, Any],
        cache_size: int = 4,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest = manifest
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[str, cv2.VideoCapture]" = OrderedDict()
        self._chosen_variant_by_file: Dict[str, str] = {}

        files = self.manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("manifest['files'] missing or not a dict")
        self._files: Dict[str, Any] = files

    def _select_variant(self, file_id: str, variants: List[str], split: str) -> str:
        if not variants:
            raise ValueError("video_variants is empty")

        if split != "train":
            return variants[0]

        # Train: choose once per file_id per reader instance (per process/worker)
        chosen = self._chosen_variant_by_file.get(file_id)
        if chosen is None:
            chosen = random.choice(variants)
            self._chosen_variant_by_file[file_id] = chosen
        return chosen

    def _get_capture(self, rel_path: str) -> cv2.VideoCapture:
        if rel_path in self._cache:
            cap = self._cache.pop(rel_path)
            self._cache[rel_path] = cap
            return cap

        abs_path = self.dataset_root / rel_path
        cap = cv2.VideoCapture(str(abs_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cv2.VideoCapture failed to open: {abs_path}")

        self._cache[rel_path] = cap
        if len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            try:
                old.release()
            except Exception:
                pass
        return cap

    def get_frame(self, file_id: str, frame_idx: int, split: str) -> Any:
        entry = self._files.get(str(file_id))
        if not isinstance(entry, dict):
            raise KeyError(f"file_id '{file_id}' not found or not a dict in manifest['files']")

        variants = entry.get("video_variants")
        if not isinstance(variants, list) or len(variants) == 0:
            raise KeyError(f"file_id '{file_id}' missing video_variants list in manifest")

        variant_rel = self._select_variant(str(file_id), [str(v) for v in variants], split)
        cap = self._get_capture(variant_rel)

        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        if not ok:
            # Some backends return False even if they seek; still try read.
            pass

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from {variant_rel} (file_id={file_id}, split={split})"
            )
        return frame

    def release(self) -> None:
        for cap in self._cache.values():
            try:
                cap.release()
            except Exception:
                pass
        self._cache.clear()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
