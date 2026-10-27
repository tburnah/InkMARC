# src/data/dataset_clip.py
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    Dataset = object  # type: ignore


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _frame_index_from_record(rec: Dict[str, Any]) -> int:
    if "frame_idx" in rec:
        return int(rec["frame_idx"])
    if "frame" in rec:
        return int(rec["frame"])
    if "index" in rec:
        return int(rec["index"])
    raise KeyError("Index record missing frame index (expected 'frame_idx' or legacy 'frame')")


def _resolve_h5_rel_path(dataset_root: Path, rec: Dict[str, Any], manifest_files: Dict[str, Any] | None) -> str:
    """
    Mirrors dataset_core resolution logic but returns RELATIVE path under dataset_root.
    Preferred: rec['rel_path'].
    Fallback: manifest lookup by file_id.
    """
    rel_path = rec.get("rel_path")
    if isinstance(rel_path, str) and rel_path:
        return rel_path

    file_id = rec.get("file_id")
    if file_id and manifest_files:
        entry = manifest_files.get(str(file_id))
        if entry is None:
            raise KeyError(f"file_id '{file_id}' not found in manifest['files']")
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            rp = entry.get("h5_rel_path") or entry.get("rel_path") or entry.get("path") or entry.get("file")
            if not rp:
                raise KeyError(f"manifest['files'][{file_id!r}] missing rel_path/path/file")
            return str(rp)

    raise KeyError("Could not resolve H5 rel path: expected rel_path or (file_id + manifest)")


def _sample_id_from_record(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("sample_id") or rec.get("id")
    return str(v) if v is not None else None


class InkMARCIndexedClipDataset(Dataset):
    """
    Clip dataset built on the same manifest/index/splits structure as InkMARCIndexedDataset :contentReference[oaicite:2]{index=2}.

    Yields dict with:
      - corners_clip: (T,4,2) float32
      - labels_last: (5,) float32
      - label_mask_last: (5,) uint8
      - timestamps_last_ns: uint64 scalar
      - meta: includes file_id, rel_path, frame_idx, clip_frame_idxs (List[int])
    """

    def __init__(
        self,
        dataset_root: str | Path,
        index_version: str = "v1",
        split: Optional[str] = None,
        clip_len: int = 5,
        require_labels_last: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache_open_files: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = index_version
        self.split = split
        self.clip_len = int(clip_len)
        if self.clip_len <= 0:
            raise ValueError("clip_len must be >= 1")

        self.require_labels_last = bool(require_labels_last)
        self.transform = transform
        self.cache_open_files = cache_open_files

        indices_dir = self.dataset_root / "indices"
        self.manifest_path = indices_dir / f"manifest_{index_version}.json"
        self.index_path = indices_dir / f"index_{index_version}.jsonl"
        self.splits_path = indices_dir / f"splits_{index_version}.json"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing index: {self.index_path}")

        self.manifest: Dict[str, Any] = _read_json(self.manifest_path)
        self._manifest_files: Dict[str, Any] | None = None
        if isinstance(self.manifest, dict):
            mf = self.manifest.get("files")
            if isinstance(mf, dict):
                self._manifest_files = mf

        # H5 caches
        self._ds_cache: Dict[tuple[str, str], h5py.Dataset] = {}
        self._h5_cache = OrderedDict()  # type: ignore[var-annotated]
        self.max_open_files = 32

        records_all = _read_jsonl(self.index_path)

        # Apply split (same supported formats as dataset_core :contentReference[oaicite:3]{index=3})
        if split is not None:
            if not self.splits_path.exists():
                raise FileNotFoundError(f"split='{split}' requested but missing splits file: {self.splits_path}")

            splits_doc = _read_json(self.splits_path)
            splits_map = splits_doc.get("splits") if isinstance(splits_doc, dict) and isinstance(splits_doc.get("splits"), dict) else splits_doc
            if not isinstance(splits_map, dict) or split not in splits_map:
                raise ValueError(f"Unknown split '{split}'. Available: {list(splits_map.keys()) if isinstance(splits_map, dict) else '??'}")

            split_list = splits_map[split]
            if not isinstance(split_list, list):
                raise ValueError(f"splits['{split}'] must be a list")

            if len(split_list) > 0:
                first = split_list[0]
                if isinstance(first, int):
                    allowed = set(int(x) for x in split_list)
                    records_all = [r for i, r in enumerate(records_all) if i in allowed]
                elif isinstance(first, str):
                    allowed = set(str(x) for x in split_list)
                    records_all = [r for r in records_all if (_sample_id_from_record(r) in allowed)]
                elif isinstance(first, dict):
                    allowed_pairs = set()
                    for d in split_list:
                        if not isinstance(d, dict):
                            continue
                        rp = d.get("rel_path") or d.get("file")
                        fi = d.get("frame_idx") if "frame_idx" in d else d.get("frame")
                        if rp is None or fi is None:
                            continue
                        allowed_pairs.add((str(rp), int(fi)))

                    def _rec_pair(rr: Dict[str, Any]) -> tuple[str, int]:
                        rp2 = rr.get("rel_path") or rr.get("file")
                        if rp2 is None:
                            rp2 = _resolve_h5_rel_path(self.dataset_root, rr, self._manifest_files)
                        return (str(rp2), _frame_index_from_record(rr))

                    records_all = [r for r in records_all if _rec_pair(r) in allowed_pairs]
                else:
                    raise ValueError(f"Unsupported splits entry type: {type(first)}")

        # Build clip-valid index: ensure we can form [t-T+1..t] within same file
        self._records: List[Dict[str, Any]] = self._make_clip_records(records_all)

        if len(self._records) == 0:
            raise RuntimeError("Clip dataset has 0 samples after filtering. Check clip_len and index integrity.")

    def _make_clip_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Group by H5 rel path, sort by frame_idx
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            rp = _resolve_h5_rel_path(self.dataset_root, r, self._manifest_files)
            by_file.setdefault(rp, []).append(r)

        kept: List[Dict[str, Any]] = []

        # label_schema indices :contentReference[oaicite:2]{index=2}
        X_IDX, Y_IDX = 0, 1

        for rp, recs in by_file.items():
            recs_sorted = sorted(recs, key=_frame_index_from_record)
            idx_map = {_frame_index_from_record(r): r for r in recs_sorted}
            frames = sorted(idx_map.keys())
            if not frames:
                continue

            abs_path = self.dataset_root / rp
            if not abs_path.exists():
                continue

            # Open once per file to check label_mask cheaply (same approach as canonical loader) :contentReference[oaicite:3]{index=3}
            with h5py.File(abs_path, "r") as h5:
                if "label_mask" not in h5:
                    continue
                mask_ds = h5["label_mask"]  # (N,5) uint8

                n_frames = int(mask_ds.shape[0])

                for t in frames:
                    start = t - (self.clip_len - 1)
                    if start < 0:
                        continue

                    # Ensure clip frames exist in index
                    ok = True
                    for fi in range(start, t + 1):
                        if fi not in idx_map:
                            ok = False
                            break
                    if not ok:
                        continue

                    # Ensure target frame index is within H5 bounds
                    if t < 0 or t >= n_frames:
                        continue

                    # Enforce target has XY labels (not just "any label")
                    m = np.asarray(mask_ds[t], dtype=np.uint8)
                    xy_ok = (m[X_IDX] != 0) and (m[Y_IDX] != 0)
                    if self.require_labels_last and not xy_ok:
                        continue

                    kept.append(idx_map[t])

        return kept

    def __len__(self) -> int:
        return len(self._records)

    def _open_h5(self, abs_path: Path) -> h5py.File:
        key = str(abs_path)
        if not self.cache_open_files:
            return h5py.File(abs_path, "r")

        h5 = self._h5_cache.get(key)
        if h5 is not None:
            self._h5_cache.move_to_end(key)
            return h5

        h5 = h5py.File(abs_path, "r", rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)
        self._h5_cache[key] = h5
        self._h5_cache.move_to_end(key)

        while len(self._h5_cache) > self.max_open_files:
            old_key, old_h5 = self._h5_cache.popitem(last=False)
            try:
                old_h5.close()
            except Exception:
                pass
            for k in list(self._ds_cache.keys()):
                if k[0] == old_key:
                    self._ds_cache.pop(k, None)

        return h5

    def _get_ds(self, h5: h5py.File, abs_path: Path, key: str) -> h5py.Dataset:
        k = (str(abs_path), key)
        ds = self._ds_cache.get(k)
        if ds is None:
            ds = h5[key]
            self._ds_cache[k] = ds
        return ds

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_t = self._records[idx]
        rp = _resolve_h5_rel_path(self.dataset_root, rec_t, self._manifest_files)
        abs_path = self.dataset_root / rp
        h5 = self._open_h5(abs_path)

        t = _frame_index_from_record(rec_t)
        frame_idxs = list(range(t - (self.clip_len - 1), t + 1))

        corners_ds = self._get_ds(h5, abs_path, "corners")
        labels_ds  = self._get_ds(h5, abs_path, "labels")
        mask_ds    = self._get_ds(h5, abs_path, "label_mask")
        ts_ds      = self._get_ds(h5, abs_path, "timestamps_ns")

        corners_clip = np.asarray(corners_ds[frame_idxs]).astype(np.float32, copy=False)  # (T,4,2)
        labels_last = np.asarray(labels_ds[t]).astype(np.float32, copy=False)            # (5,)
        mask_last   = np.asarray(mask_ds[t]).astype(np.uint8, copy=False)                # (5,)
        ts_last     = np.asarray(ts_ds[t]).astype(np.uint64, copy=False).item()

        # Enforce require_labels_last here (cheap)
        if self.require_labels_last and not np.any(mask_last != 0):
            # fall back: pick the next sample (rare) or raise
            raise IndexError("Selected clip target has no labels (label_mask all zero). Consider regenerating clip index or set require_labels_last=False.")

        sample: Dict[str, Any] = {
            "corners_clip": corners_clip,
            "labels_last": labels_last,
            "label_mask_last": mask_last,
            "timestamps_last_ns": ts_last,
            "meta": {
                "schema_version": rec_t.get("schema_version"),
                "sample_id": rec_t.get("sample_id"),
                "file_id": rec_t.get("file_id"),
                "rel_path": rp,
                "frame_idx": t,
                "clip_frame_idxs": frame_idxs,
                "participant_id": rec_t.get("participant_id"),
                "task": rec_t.get("task"),
                "session_id": rec_t.get("session_id"),
                "index_timestamp_ns": rec_t.get("timestamp_ns"),
                "index_version": self.index_version,
                "split": self.split,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def close(self) -> None:
        for h5 in self._h5_cache.values():
            try:
                h5.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
