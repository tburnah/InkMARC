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
        max_read_retries: int = 2,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest = manifest
        self.cache_size = max(1, int(cache_size))
        self.max_read_retries = max(0, int(max_read_retries))
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

    def _drop_capture(self, rel_path: str) -> None:
        cap = self._cache.pop(rel_path, None)
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    def _read_frame_with_retries(self, rel_path: str, frame_idx: int, *, context: str) -> Any:
        last_error: str | None = None
        total_attempts = self.max_read_retries + 1
        for attempt in range(1, total_attempts + 1):
            cap = self._get_capture(rel_path)
            ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            if not ok:
                # Some backends return False even if seek works; still try read.
                pass

            ret, frame = cap.read()
            if ret and frame is not None:
                return frame

            last_error = (
                f"read failed attempt {attempt}/{total_attempts} "
                f"(frame_idx={frame_idx}, rel_path={rel_path}, context={context})"
            )
            self._drop_capture(rel_path)

        raise RuntimeError(last_error or f"Failed to read frame {frame_idx} from {rel_path}")

    def get_frame_from_variant(self, variant_rel: str, frame_idx: int) -> Any:
        """
        Read a frame from an explicit variant path (relative to dataset_root).
        Use this when the dataset record already specifies which variant to use
        (e.g., sample["meta"]["video_rel_path"]).
        """
        return self._read_frame_with_retries(
            str(variant_rel),
            int(frame_idx),
            context="get_frame_from_variant",
        )

    def get_frame(self, file_id: str, frame_idx: int, split: str) -> Any:
        """
        Backwards-compatible API:
          - val/test => variants[0]
          - train    => random choice (cached per file per reader instance)
        """
        entry = self._files.get(str(file_id))
        if not isinstance(entry, dict):
            raise KeyError(f"file_id '{file_id}' not found or not a dict in manifest['files']")

        variants = entry.get("video_variants")
        if not isinstance(variants, list) or len(variants) == 0:
            raise KeyError(f"file_id '{file_id}' missing video_variants list in manifest")

        variant_rel = self._select_variant(str(file_id), [str(v) for v in variants], split)
        return self._read_frame_with_retries(
            variant_rel,
            int(frame_idx),
            context=f"get_frame(file_id={file_id}, split={split})",
        )

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
