from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:
    Dataset = object  # type: ignore

from data.dataset_core import InkMARCIndexedDataset


TAB_W = 44800.0
TAB_H = 29600.0
X_IDX = 0
Y_IDX = 1


class InkMARCIndexedHistoryDataset(Dataset):
    """
    Single-frame dataset with temporal XY history from previous frames.

    Returns base sample fields from InkMARCIndexedDataset, plus:
      - history_xy: (history_len, 2) float32 in [0,1], oldest -> newest.
      - history_valid: (history_len,) float32 (1 when XY label exists).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        index_version: str = "v1",
        split: Optional[str] = None,
        require_labels: bool = True,
        history_len: int = 3,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache_open_files: bool = True,
    ) -> None:
        self.history_len = int(history_len)
        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")

        self.transform = transform
        self.base = InkMARCIndexedDataset(
            dataset_root=dataset_root,
            index_version=index_version,
            split=split,
            require_labels=require_labels,
            transform=None,
            cache_open_files=cache_open_files,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        meta = sample.get("meta", {})

        rel_path = meta.get("rel_path")
        frame_idx = int(meta.get("frame_idx", -1))
        if rel_path is None or frame_idx < 0:
            raise KeyError("Sample metadata missing rel_path/frame_idx.")

        abs_path = self.base.dataset_root / str(rel_path)
        h5 = self.base._open_h5(abs_path)
        labels_ds = self.base._get_ds(h5, abs_path, "labels")
        mask_ds = self.base._get_ds(h5, abs_path, "label_mask")

        hist = np.zeros((self.history_len, 2), dtype=np.float32)
        valid = np.zeros((self.history_len,), dtype=np.float32)

        # Oldest -> newest history ordering.
        for hi, fi in enumerate(range(frame_idx - self.history_len, frame_idx)):
            if fi < 0:
                continue
            lbl = np.asarray(labels_ds[fi], dtype=np.float32)
            msk = np.asarray(mask_ds[fi], dtype=np.uint8)
            x_ok = bool(msk[X_IDX] != 0)
            y_ok = bool(msk[Y_IDX] != 0)
            if not (x_ok and y_ok):
                continue

            x_norm = float(np.clip(float(lbl[X_IDX]) / TAB_W, 0.0, 1.0))
            y_norm = float(np.clip(float(lbl[Y_IDX]) / TAB_H, 0.0, 1.0))
            hist[hi, 0] = x_norm
            hist[hi, 1] = y_norm
            valid[hi] = 1.0

        sample["history_xy"] = hist
        sample["history_valid"] = valid
        sample["meta"] = {
            **meta,
            "history_frame_idxs": list(range(frame_idx - self.history_len, frame_idx)),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def close(self) -> None:
        self.base.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
