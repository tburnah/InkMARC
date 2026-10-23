# ~/inkmarc/src/data/dataset_core.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from collections import OrderedDict

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    # Allows non-torch tooling (e.g., scripts) to import this module.
    Dataset = object  # type: ignore


# -----------------------------
# Parsing utilities
# -----------------------------

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


def _resolve_h5_path(dataset_root: Path, rec: Dict[str, Any], manifest_files: Dict[str, Any] | None) -> Path:
    """
    Resolve the HDF5 file path for an index record.

    Priority:
      1) rec["rel_path"] (preferred; already relative to dataset root)
      2) rec["file"] (legacy key, if you ever generate that)
      3) rec["file_id"] -> manifest_files[file_id]["rel_path"] (or str)
    """
    rel_path = rec.get("rel_path")
    if isinstance(rel_path, str) and rel_path:
        return dataset_root / rel_path

    legacy_file = rec.get("file")
    if isinstance(legacy_file, str) and legacy_file:
        return dataset_root / legacy_file

    file_id = rec.get("file_id")
    if file_id and manifest_files:
        entry = manifest_files.get(str(file_id))
        if entry is None:
            raise KeyError(f"file_id '{file_id}' not found in manifest['files']")
        if isinstance(entry, str):
            return dataset_root / entry
        if isinstance(entry, dict):
            rp = entry.get("h5_rel_path") or entry.get("rel_path") or entry.get("path") or entry.get("file")
            if not rp:
                raise KeyError(f"manifest['files'][{file_id!r}] missing rel_path/path/file")
            return dataset_root / str(rp)

    raise KeyError("Could not resolve H5 path: expected rel_path or file (or file_id + manifest)")


def _frame_index_from_record(rec: Dict[str, Any]) -> int:
    """
    Return the per-file frame index for this record.
    Supports current schema ("frame_idx") and a couple of legacy fallbacks.
    """
    if "frame_idx" in rec:
        return int(rec["frame_idx"])
    if "frame" in rec:
        return int(rec["frame"])
    if "index" in rec:
        return int(rec["index"])
    raise KeyError("Index record missing frame index (expected 'frame_idx' or legacy 'frame')")


def _sample_id_from_record(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("sample_id") or rec.get("id")
    return str(v) if v is not None else None


# -----------------------------
# Core dataset
# -----------------------------

class InkMARCIndexedDataset(Dataset):
    """
    Canonical dataset loader that is stable across experiments/models.

    It reads:
      - indices/manifest_<version>.json
      - indices/index_<version>.jsonl
      - indices/splits_<version>.json (optional)

    And yields per-sample dicts with:
      - corners: (4,2) float32
      - labels: (5,) float32
      - label_mask: (5,) uint8
      - timestamps_ns: uint64 scalar
      - meta: record metadata

    Filtering:
      - require_labels=True means we only include frames where label_mask has any non-zero value.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        index_version: str = "v1",
        split: Optional[str] = None,  # "train" | "val" | "test" | None
        require_labels: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache_open_files: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.index_version = index_version
        self.split = split
        self.require_labels = require_labels
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

        self._ds_cache: Dict[tuple[str, str], h5py.Dataset] = {}

        self._h5_cache = OrderedDict()  # type: ignore[var-annotated]
        self.max_open_files = 32

        index_rows = _read_jsonl(self.index_path)  # list[dict]
        records: List[Dict[str, Any]] = list(index_rows)

        # Apply split if requested
        if split is not None:
            if not self.splits_path.exists():
                raise FileNotFoundError(
                    f"split='{split}' requested but missing splits file: {self.splits_path}"
                )
            splits_doc = _read_json(self.splits_path)
            if not isinstance(splits_doc, dict):
                raise ValueError(f"Splits file must be a dict, got {type(splits_doc)}")

            # Support both formats:
            #  A) { "train": [...], "val": [...], "test": [...] }
            #  B) { "schema_version": ..., "meta": ..., "splits": { "train": [...], ... } }
            splits_map = splits_doc.get("splits") if isinstance(splits_doc.get("splits"), dict) else splits_doc

            if split not in splits_map:
                raise ValueError(f"Unknown split '{split}'. Available: {list(splits_map.keys())}")

            split_list = splits_map[split]
            if not isinstance(split_list, list):
                raise ValueError(f"splits['{split}'] must be a list, got {type(split_list)}")

            if not isinstance(split_list, list):
                raise ValueError(f"splits['{split}'] must be a list, got {type(split_list)}")

            # Support splits stored as:
            #  A) list of integer index-row ids
            #  B) list of sample_id strings
            #  C) list of dicts with {"rel_path","frame_idx"} (or legacy {"file","frame"})
            if len(split_list) > 0:
                first = split_list[0]

                if isinstance(first, int):
                    allowed_idx = set(int(x) for x in split_list)
                    records = [r for i, r in enumerate(records) if i in allowed_idx]

                elif isinstance(first, str):
                    allowed_ids = set(str(x) for x in split_list)
                    records = [r for r in records if (_sample_id_from_record(r) in allowed_ids)]

                elif isinstance(first, dict):
                    allowed_pairs = set()
                    for d in split_list:
                        if not isinstance(d, dict):
                            continue
                        # accept rel_path/frame_idx OR file/frame
                        rp = d.get("rel_path") or d.get("file")
                        fi = d.get("frame_idx") if "frame_idx" in d else d.get("frame")
                        if rp is None or fi is None:
                            continue
                        allowed_pairs.add((str(rp), int(fi)))

                    def _rec_pair(rr: Dict[str, Any]) -> tuple[str, int]:
                        rp2 = rr.get("rel_path") or rr.get("file")
                        if rp2 is None:
                            # fall back to manifest resolution to a rel path if possible
                            p = _resolve_h5_path(self.dataset_root, rr, self._manifest_files)
                            rp2 = str(p.relative_to(self.dataset_root))
                        return (str(rp2), _frame_index_from_record(rr))

                    records = [r for r in records if _rec_pair(r) in allowed_pairs]

                else:
                    raise ValueError(
                        f"Unsupported splits entry type: {type(first)}. "
                        "Expected int, str, or dict."
                    )

        self._records_all: List[Dict[str, Any]] = records

        # Filter to labeled frames
        if self.require_labels:
            self._records: List[Dict[str, Any]] = self._filter_records_with_labels(self._records_all)
        else:
            self._records = list(self._records_all)

        if len(self._records) == 0:
            raise RuntimeError(
                "After filtering, dataset has 0 samples. "
                "Either require_labels=False or verify label_mask contents."
            )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._records[idx]  # dict from index jsonl
        path = _resolve_h5_path(self.dataset_root, rec, self._manifest_files)
        h5 = self._open_h5(path)

        i = _frame_index_from_record(rec)

        # Read row i from each dataset
        corners_ds = self._get_ds(h5, path, "corners")
        labels_ds = self._get_ds(h5, path, "labels")
        mask_ds   = self._get_ds(h5, path, "label_mask")
        ts_ds     = self._get_ds(h5, path, "timestamps_ns")

        corners = np.asarray(corners_ds[i])
        if corners.dtype != np.float32:
            corners = corners.astype(np.float32, copy=False)

        labels = np.asarray(labels_ds[i])
        if labels.dtype != np.float32:
            labels = labels.astype(np.float32, copy=False)

        label_mask = np.asarray(mask_ds[i])
        if label_mask.dtype != np.uint8:
            label_mask = label_mask.astype(np.uint8, copy=False)

        ts = int(ts_ds[i])

        sample: Dict[str, Any] = {
            "corners": corners,
            "labels": labels,
            "label_mask": label_mask,
            "timestamps_ns": ts,
            "meta": {
                # Index identity
                "schema_version": rec.get("schema_version"),
                "sample_id": rec.get("sample_id"),
                "file_id": rec.get("file_id"),
                "rel_path": rec.get("rel_path"),
                "frame_idx": i,
                # Higher-level metadata (if present in index)
                "participant_id": rec.get("participant_id"),
                "task": rec.get("task"),
                "session_id": rec.get("session_id"),
                "index_timestamp_ns": rec.get("timestamp_ns"),
                # Loader context
                "index_version": self.index_version,
                "split": self.split,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _get_ds(self, h5: h5py.File, path: Path, key: str) -> h5py.Dataset:
        k = (str(path), key)
        ds = self._ds_cache.get(k)
        if ds is None:
            ds = h5[key]
            self._ds_cache[k] = ds
        return ds

    # -----------------------------
    # Internals
    # -----------------------------

    def _open_h5(self, path: Path) -> h5py.File:
        key = str(path)
        if not self.cache_open_files:
            return h5py.File(path, "r")

        h5 = self._h5_cache.get(key)
        if h5 is not None:
            # mark as recently used
            self._h5_cache.move_to_end(key)
            return h5

        # Optional: increase HDF5 chunk cache a bit (safe, no output change)
        h5 = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)
        self._h5_cache[key] = h5
        self._h5_cache.move_to_end(key)

        # evict if needed
        while len(self._h5_cache) > self.max_open_files:
            old_key, old_h5 = self._h5_cache.popitem(last=False)
            try:
                old_h5.close()
            except Exception:
                pass
            # also evict any dataset handles tied to that file
            for k in list(self._ds_cache.keys()):
                if k[0] == old_key:
                    self._ds_cache.pop(k, None)

        return h5

    def close(self) -> None:
        for h5 in self._h5_cache.values():
            try:
                h5.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def _filter_records_with_labels(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep only records where label_mask has at least one non-zero element.
        Reads only label_mask rows (cheap).

        Notes:
        - Groups by resolved absolute H5 path to minimize opens.
        - If a referenced H5 is missing, records for it are skipped.
        """
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            path = _resolve_h5_path(self.dataset_root, r, self._manifest_files)
            by_file.setdefault(str(path), []).append(r)

        kept: List[Dict[str, Any]] = []

        for abs_path, recs in by_file.items():
            path = Path(abs_path)
            if not path.exists():
                continue

            with h5py.File(path, "r") as h5:
                if "label_mask" not in h5:
                    continue
                mask_ds = h5["label_mask"]  # (N,5) uint8

                for r in recs:
                    frame = _frame_index_from_record(r)
                    if frame < 0 or frame >= mask_ds.shape[0]:
                        continue
                    m = np.asarray(mask_ds[frame], dtype=np.uint8)
                    if np.any(m != 0):
                        kept.append(r)

        return kept

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_h5_cache"] = OrderedDict()
        d["_ds_cache"] = {}
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._h5_cache = OrderedDict()
        self._ds_cache = {}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
