from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


def _empty_label() -> dict[str, Any]:
    return {
        "touch": None,
        "state": None,
        "xy": None,
        "valid_xy": False,
        "pressure": None,
        "has_labels": False,
        "has_state": False,
        "has_touch": False,
    }


def _needs_labels(cfg: Any) -> bool:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None and any(
        bool(getattr(model_cfg, key, False)) for key in ("train_touch", "train_state", "train_location")
    ):
        return True
    loss_cfg = getattr(cfg, "loss", None)
    if loss_cfg is not None and any(
        float(getattr(loss_cfg, key, 0.0)) > 0.0 for key in ("lambda_touch", "lambda_state", "lambda_xy", "lambda_heatmap")
    ):
        return True
    return False


def _touch_target(cfg: Any) -> str:
    labels = getattr(getattr(cfg, "data", {}), "labels", {})
    return str(getattr(labels, "touch_target", "pressure")).lower()


def _touch_from_state(state: int | None, pressure_touch: float | None, target: str) -> tuple[float | None, bool]:
    if target in {"interaction", "presence", "present", "stylus_present"}:
        if state is None:
            return None, False
        return (1.0 if int(state) in {1, 2} else 0.0), True
    return pressure_touch, pressure_touch is not None


class StylusDataset(Dataset):
    """Single-frame stylus dataset with pseudo-depth supervision.

    The manifest reader accepts common layouts: a list of frame records, a dict with
    a "frames" list, or a dict with videos containing per-frame records. For your
    label H5 format, adapt `_load_external_h5_labels` and `_lookup_label`.
    """

    def __init__(
        self,
        cfg: Any,
        split: str = "train",
        training: bool = True,
        synthetic: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.training = training
        self.input_size = tuple(int(x) for x in cfg.data.input_size)
        self.synthetic = bool(cfg.data.synthetic.enabled if synthetic is None else synthetic)
        self.augment_cfg = getattr(cfg, "augment", {})
        self._depth_store: Any | None = None
        self._depth_store_cache: OrderedDict[str, Any] = OrderedDict()
        self._label_h5_cache: dict[str, h5py.File] = {}
        self._label_table: dict[tuple[str, int], dict[str, Any]] = {}
        self.depth_cache_size = int(getattr(cfg.data.depth, "cache_size", 64))
        label_loading = getattr(cfg.data.labels, "load", "auto")
        if str(label_loading).lower() == "auto":
            self.load_labels = _needs_labels(cfg)
        elif isinstance(label_loading, str):
            self.load_labels = label_loading.lower() in {"1", "true", "yes", "on"}
        else:
            self.load_labels = bool(label_loading)

        if self.synthetic:
            self.records = self._build_synthetic_records(int(cfg.data.synthetic.length))
        else:
            self.records = self._load_manifest(Path(cfg.data.manifest_path), split)
            if self.load_labels:
                self._load_external_labels()

        self.jitter = ColorJitter(
            brightness=float(getattr(self.augment_cfg, "brightness", 0.0)),
            contrast=float(getattr(self.augment_cfg, "contrast", 0.0)),
            saturation=float(getattr(self.augment_cfg, "saturation", 0.0)),
            hue=float(getattr(self.augment_cfg, "hue", 0.0)),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        has_depth = True
        if self.synthetic:
            image, depth, label = self._make_synthetic_sample(index)
        else:
            try:
                depth = self._read_depth(rec, index)
            except (FileNotFoundError, KeyError, IndexError, ValueError):
                depth = np.zeros(self.input_size, dtype=np.float32)
                has_depth = False
            if self._input_source() in {"depth", "depth_as_rgb", "depth_gray", "depth_grayscale"}:
                image = self._depth_to_rgb(depth)
            else:
                image = self._read_image(rec)
            label = self._lookup_label(rec) if self.load_labels else _empty_label()

        image = self._resize_image(image)
        depth = self._resize_depth(depth)
        image = self._augment_image(image)

        depth_t = torch.from_numpy(depth[None].astype(np.float32))
        image_t = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        if self._input_source() in {"rgbd", "rgb_depth", "rgb+depth"}:
            image_t = torch.cat([image_t, depth_t.clamp(0.0, 1.0)], dim=0)

        xy = label.get("xy")
        xy_t = torch.tensor(xy, dtype=torch.float32) if xy is not None else None
        state = label.get("state")
        touch = label.get("touch")
        pressure = label.get("pressure")
        has_labels = bool(label.get("has_labels", state is not None or touch is not None or xy is not None))
        has_state = bool(label.get("has_state", state is not None))
        has_touch = bool(label.get("has_touch", touch is not None))
        valid_xy = bool(label.get("valid_xy", xy is not None and np.isfinite(xy).all()))
        video_id = str(rec.get("video_id", rec.get("video", "synthetic")))

        return {
            "image": image_t,
            "depth": depth_t,
            "has_depth": bool(has_depth),
            "state": None if not has_state or state is None else int(state),
            "has_state": has_state,
            "touch": None if not has_touch or touch is None else float(touch),
            "has_touch": has_touch,
            "xy": xy_t,
            "valid_xy": valid_xy,
            "pressure": None if pressure is None else float(pressure),
            "has_labels": has_labels,
            "participant_id": str(rec.get("participant_id", _participant_from_stem(video_id))),
            "task_id": str(rec.get("task_id", _task_from_stem(video_id))),
            "video_id": video_id,
            "frame_index": int(rec.get("frame_index", rec.get("index", index))),
        }

    def _input_source(self) -> str:
        return str(getattr(self.cfg.data, "input_source", getattr(self.cfg.data, "rgb_source", "rgb"))).lower()

    def _depth_to_rgb(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
        depth = np.clip(depth, 0.0, 1.0)
        gray = (depth * 255.0).round().astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def _build_synthetic_records(self, length: int) -> list[dict[str, Any]]:
        return [{"video_id": "synthetic", "frame_index": i} for i in range(length)]

    def _make_synthetic_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        h, w = self.input_size
        yy, xx = np.mgrid[0:h, 0:w]
        cx = (0.2 + 0.6 * ((index * 37) % 101) / 100.0) * w
        cy = (0.2 + 0.6 * ((index * 53) % 101) / 100.0) * h
        sigma = 18.0
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))).astype(np.float32)
        depth = np.clip(0.25 + 0.65 * (yy / max(1, h - 1)) + 0.25 * blob, 0, 1).astype(np.float32)
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[..., 0] = np.clip(70 + 100 * depth, 0, 255).astype(np.uint8)
        image[..., 1] = np.clip(50 + 160 * blob, 0, 255).astype(np.uint8)
        image[..., 2] = np.clip(110 + 60 * (xx / max(1, w - 1)), 0, 255).astype(np.uint8)
        touch = float(index % 3 != 0)
        state = 2 if touch else 1
        label = {
            "touch": touch,
            "state": state,
            "xy": np.array([cx / w, cy / h], dtype=np.float32),
            "valid_xy": True,
            "pressure": touch,
            "has_labels": True,
            "has_state": True,
            "has_touch": True,
        }
        return image, depth, label

    def _load_manifest(self, path: Path, split: str) -> list[dict[str, Any]]:
        manifest = _read_json(path)
        records: list[dict[str, Any]] = []
        if isinstance(manifest, list):
            records = [dict(item) for item in manifest]
        elif isinstance(manifest, dict) and isinstance(manifest.get("frames"), list):
            records = [dict(item) for item in manifest["frames"] if str(item.get("split", split)) == split]
        elif isinstance(manifest, dict) and isinstance(manifest.get("videos"), list):
            for video in manifest["videos"]:
                video_id = str(video.get("video_id", video.get("id", video.get("path", ""))))
                video_path = video.get("path") or video.get("video_path")
                frame_dir = video.get("frame_dir")
                for frame in video.get("frames", []):
                    rec = dict(frame)
                    rec.setdefault("video_id", video_id)
                    rec.setdefault("video_path", video_path)
                    rec.setdefault("frame_dir", frame_dir)
                    records.append(rec)
        elif isinstance(manifest, dict) and isinstance(manifest.get("items"), list):
            videos_seen = 0
            max_videos = getattr(self.cfg.data, "max_videos_per_split", None)
            variant_filter = _variant_kind_filter(self.cfg)
            for item in manifest["items"]:
                item_split = str(item.get("split", split))
                if item_split != split:
                    continue
                if variant_filter is not None and str(item.get("variant_kind", "original")) not in variant_filter:
                    continue
                if max_videos is not None and videos_seen >= int(max_videos):
                    break
                videos_seen += 1
                stem = str(item.get("stem") or Path(item.get("output_video_path", "")).stem)
                video_path = item.get("output_video_path") or item.get("video_path")
                h5_path = item.get("output_h5_path") or item.get("label_h5_path")
                depth_path = item.get("depth_path")
                if depth_path is None:
                    depth_path = str(Path(item.get("output_dir", "/data/baseAVIs_432x672")) / "depth" / f"{stem}.zarr")
                    if not Path(depth_path).exists():
                        depth_path = str(Path("/data/baseAVIs_432x672/depth") / f"{stem}.zarr")
                frame_count = int(item.get("frame_count") or 0)
                if frame_count <= 0:
                    frame_count = self._infer_frame_count(video_path, depth_path, item.get("depth_key", getattr(self.cfg.data.depth, "key", "depth_norm")))
                stride = max(1, int(getattr(self.cfg.data, "frame_stride", 1)))
                max_frames = getattr(self.cfg.data, "max_frames_per_video", None)
                frame_indices = list(range(0, frame_count, stride))
                excluded_frames = _excluded_frame_indices(item)
                if excluded_frames:
                    frame_indices = [idx for idx in frame_indices if idx not in excluded_frames]
                if max_frames is not None:
                    frame_indices = frame_indices[: int(max_frames)]
                for frame_index in frame_indices:
                    records.append(
                        {
                            "video_id": stem,
                            "participant_id": item.get("participant_id") or _participant_from_stem(stem),
                            "task_id": item.get("task_id") or _task_from_stem(stem),
                            "frame_index": frame_index,
                            "depth_index": frame_index,
                            "video_path": video_path,
                            "label_h5_path": h5_path,
                            "depth_path": depth_path,
                            "depth_key": item.get("depth_key", getattr(self.cfg.data.depth, "key", "depth_norm")),
                            "split": item_split,
                        }
                    )
        else:
            raise ValueError(f"Unsupported manifest layout in {path}")

        filtered = [r for r in records if str(r.get("split", split)) == split]
        return filtered if filtered else records

    def _load_external_labels(self) -> None:
        label_cfg = self.cfg.data.labels
        fmt = str(getattr(label_cfg, "format", "manifest"))
        path = _as_path(getattr(label_cfg, "path", None))
        if fmt in {"none", "manifest", "inkmarc_h5"} or path is None:
            return
        if fmt == "json":
            rows = _read_json(path)
            if isinstance(rows, dict) and "frames" in rows:
                rows = rows["frames"]
            for row in rows:
                self._label_table[(str(row["video_id"]), int(row["frame_index"]))] = row
        elif fmt == "csv":
            df = pd.read_csv(path)
            for row in df.to_dict("records"):
                self._label_table[(str(row["video_id"]), int(row["frame_index"]))] = row
        elif fmt == "h5":
            self._load_external_h5_labels(path)
        else:
            raise ValueError(f"Unsupported label format: {fmt}")

    def _load_external_h5_labels(self, path: Path) -> None:
        # Placeholder adapter: replace this with your actual H5 schema.
        # Expected simple schema for now:
        #   video_id: fixed/variable string dataset, shape [N]
        #   frame_index: int dataset, shape [N]
        #   optional touch/state/x/y datasets, shape [N]
        with h5py.File(path, "r") as h5:
            if "video_id" not in h5 or "frame_index" not in h5:
                raise ValueError(
                    "Label H5 adapter expects datasets video_id and frame_index. "
                    "Edit StylusDataset._load_external_h5_labels for your schema."
                )
            video_ids = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in h5["video_id"][:]]
            frame_indices = h5["frame_index"][:].astype(int)
            keys = [k for k in ["touch", "state", "x", "y"] if k in h5]
            for i, (video_id, frame_index) in enumerate(zip(video_ids, frame_indices)):
                row = {"video_id": video_id, "frame_index": int(frame_index)}
                row.update({key: h5[key][i].item() for key in keys})
                self._label_table[(video_id, int(frame_index))] = row

    def _lookup_label(self, rec: dict[str, Any]) -> dict[str, Any]:
        video_id = str(rec.get("video_id", rec.get("video", "")))
        frame_index = int(rec.get("frame_index", rec.get("index", -1)))
        if str(getattr(self.cfg.data.labels, "format", "manifest")) == "inkmarc_h5" and rec.get("label_h5_path"):
            return self._lookup_inkmarc_h5_label(Path(rec["label_h5_path"]), frame_index)
        row = {**rec, **self._label_table.get((video_id, frame_index), {})}
        label_cfg = self.cfg.data.labels
        touch = row.get(getattr(label_cfg, "touch_key", "touch"))
        state = row.get(getattr(label_cfg, "state_key", "state"))
        x = row.get(getattr(label_cfg, "x_key", "x"))
        y = row.get(getattr(label_cfg, "y_key", "y"))
        xy = None
        if x is not None and y is not None and np.isfinite([x, y]).all():
            xy = np.array([float(x), float(y)], dtype=np.float32)
            if not bool(getattr(label_cfg, "xy_normalized", True)):
                h, w = self.input_size
                xy = xy / np.array([w, h], dtype=np.float32)
            xy = np.clip(xy, 0.0, 1.0)
        if state is not None:
            touch, has_touch = _touch_from_state(int(state), None if touch is None else float(touch), _touch_target(self.cfg))
        else:
            has_touch = touch is not None
        return {
            "touch": touch,
            "state": state,
            "xy": xy,
            "valid_xy": xy is not None,
            "pressure": row.get("pressure"),
            "has_labels": any(v is not None for v in [touch, state, xy]),
            "has_state": state is not None,
            "has_touch": has_touch,
        }

    def _lookup_inkmarc_h5_label(self, path: Path, frame_index: int) -> dict[str, Any]:
        if not path.exists():
            return _empty_label()
        h5 = self._label_h5_cache.get(str(path))
        if h5 is None:
            try:
                h5 = h5py.File(path, "r")
            except OSError:
                return _empty_label()
            self._label_h5_cache[str(path)] = h5
        labels = np.asarray(h5["labels"][frame_index], dtype=np.float32)
        label_mask = np.asarray(h5["label_mask"][frame_index], dtype=np.uint8)
        x_ok = bool(label_mask[0] != 0 and np.isfinite(labels[0]))
        y_ok = bool(label_mask[1] != 0 and np.isfinite(labels[1]))
        p_ok = bool(label_mask[2] != 0 and np.isfinite(labels[2]))
        pressure = float(labels[2]) if p_ok else None
        pressure_touch = None
        if p_ok:
            threshold = float(getattr(self.cfg.data.labels, "pressure_touch_threshold", 0.0))
            pressure_touch = float(pressure > threshold)
        xy = None
        if x_ok and y_ok:
            corners = np.asarray(h5["corners"][frame_index], dtype=np.float32) if "corners" in h5 else None
            resize_attrs = dict(h5.get("preprocessing/resize_transform", {}).attrs) if "preprocessing/resize_transform" in h5 else {}
            xy = self._convert_xy(float(labels[0]), float(labels[1]), corners=corners, resize_attrs=resize_attrs)
        valid_xy = bool(xy is not None and np.isfinite(xy).all())
        has_labels = True
        if valid_xy and pressure_touch is not None and pressure_touch > 0.5:
            state = 2
        elif valid_xy:
            state = 1
        elif str(getattr(self.cfg.data.labels, "background_policy", "invalid_xy_is_background")) in {
            "missing_xy_is_background",
            "invalid_xy_is_background",
        }:
            state = 0
        else:
            state = None
        touch, has_touch = _touch_from_state(state, pressure_touch, _touch_target(self.cfg))
        return {
            "touch": touch,
            "state": state,
            "xy": xy,
            "valid_xy": valid_xy,
            "pressure": pressure,
            "has_labels": has_labels,
            "has_state": state is not None,
            "has_touch": has_touch,
        }

    def _convert_xy(
        self,
        x: float,
        y: float,
        corners: np.ndarray | None = None,
        resize_attrs: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        label_cfg = self.cfg.data.labels
        mode = str(getattr(label_cfg, "coordinate_mode", "surface_to_image"))
        image_w = float(getattr(label_cfg, "image_width", self.input_size[1]))
        image_h = float(getattr(label_cfg, "image_height", self.input_size[0]))
        surface_w = float(getattr(label_cfg, "surface_width", getattr(label_cfg, "tablet_width", 44800.0)) or 0.0)
        surface_h = float(getattr(label_cfg, "surface_height", getattr(label_cfg, "tablet_height", 29600.0)) or 0.0)
        if mode == "normalized":
            xy = np.array([x, y], dtype=np.float32)
        elif mode == "image_pixels":
            xy = np.array([x / image_w, y / image_h], dtype=np.float32)
        elif mode == "surface_pixels":
            if surface_w <= 0 or surface_h <= 0:
                raise ValueError("coordinate_mode=surface_pixels requires data.labels.surface_width/surface_height")
            xy = np.array([x / surface_w, y / surface_h], dtype=np.float32)
        elif mode == "surface_to_image":
            if corners is None:
                raise ValueError("coordinate_mode=surface_to_image requires H5 corners")
            xy = surface_to_image_xy(
                x,
                y,
                corners,
                surface_w=surface_w,
                surface_h=surface_h,
                image_w=image_w,
                image_h=image_h,
                surface_model_w=float(getattr(label_cfg, "surface_model_width", 480)),
                surface_model_h=float(getattr(label_cfg, "surface_model_height", 320)),
                corners_coordinate_mode=str(getattr(label_cfg, "corners_coordinate_mode", "resized")),
                resize_attrs=resize_attrs or {},
            )
        elif mode == "auto":
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                xy = np.array([x, y], dtype=np.float32)
            elif 0.0 <= x <= image_w and 0.0 <= y <= image_h:
                xy = np.array([x / image_w, y / image_h], dtype=np.float32)
            elif surface_w > 0 and surface_h > 0 and 0.0 <= x <= surface_w and 0.0 <= y <= surface_h:
                xy = np.array([x / surface_w, y / surface_h], dtype=np.float32)
            else:
                return None
        else:
            raise ValueError(f"Unknown coordinate_mode: {mode}")
        if xy is None:
            return None
        if not np.isfinite(xy).all():
            return None
        if (xy < 0).any() or (xy > 1).any():
            return None
        return xy.astype(np.float32)

    def _read_image(self, rec: dict[str, Any]) -> np.ndarray:
        image_path = rec.get("image_path") or rec.get("frame_path") or rec.get("path")
        if image_path is not None:
            path = Path(image_path)
            if not path.is_absolute() and self.cfg.data.frame_root:
                path = Path(self.cfg.data.frame_root) / path
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        video_path = rec.get("video_path") or rec.get("video")
        if video_path is None:
            raise ValueError(f"Record has no image/frame/video path: {rec}")
        path = Path(video_path)
        if not path.is_absolute() and self.cfg.data.video_root:
            path = Path(self.cfg.data.video_root) / path
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(rec.get("frame_index", rec.get("index", 0))))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame from {path}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def _open_depth_store(self) -> Any | None:
        if self._depth_store is not None:
            return self._depth_store
        depth_cfg = self.cfg.data.depth
        path = _as_path(getattr(depth_cfg, "path", None))
        if path is None:
            return None
        fmt = str(getattr(depth_cfg, "format", "zarr"))
        self._depth_store = zarr.open(path, mode="r") if fmt == "zarr" else h5py.File(path, "r")
        return self._depth_store

    def _read_depth(self, rec: dict[str, Any], index: int) -> np.ndarray:
        depth_path = rec.get("depth_path")
        depth_key = rec.get("depth_key", getattr(self.cfg.data.depth, "key", "depth"))
        depth_index = int(rec.get("depth_index", index))
        if depth_path:
            path = Path(depth_path)
            fmt = str(getattr(self.cfg.data.depth, "format", "zarr"))
            if fmt == "zarr":
                store = self._depth_store_cache.get(str(path))
                if store is None:
                    store = zarr.open(path, mode="r")
                    self._depth_store_cache[str(path)] = store
                    self._trim_depth_cache()
                else:
                    self._depth_store_cache.move_to_end(str(path))
                arr = store if hasattr(store, "shape") else store[depth_key]
                depth = arr[depth_index] if getattr(arr, "ndim", 0) >= 3 else arr[:]
            else:
                with h5py.File(path, "r") as h5:
                    arr = h5[depth_key]
                    depth = arr[depth_index] if arr.ndim >= 3 else arr[:]
            return np.asarray(depth, dtype=np.float32).squeeze()

        store = self._open_depth_store()
        if store is None:
            return np.zeros(self.input_size, dtype=np.float32)
        arr = store if hasattr(store, "shape") else store[depth_key]
        return np.asarray(arr[depth_index], dtype=np.float32).squeeze()

    def _trim_depth_cache(self) -> None:
        if self.depth_cache_size <= 0:
            self._depth_store_cache.clear()
            return
        while len(self._depth_store_cache) > self.depth_cache_size:
            _, store = self._depth_store_cache.popitem(last=False)
            close = getattr(store, "close", None)
            if callable(close):
                close()

    def _infer_frame_count(self, video_path: str | None, depth_path: str | None, depth_key: str) -> int:
        if depth_path and Path(depth_path).exists():
            try:
                store = zarr.open(depth_path, mode="r")
                arr = store if hasattr(store, "shape") else store[depth_key]
                if len(arr.shape) >= 3:
                    return int(arr.shape[0])
            except Exception:
                pass
        if video_path and Path(video_path).exists():
            cap = cv2.VideoCapture(str(video_path))
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if count > 0:
                    return count
            finally:
                cap.release()
        return 0

    def close(self) -> None:
        for h5 in self._label_h5_cache.values():
            h5.close()
        self._label_h5_cache.clear()
        if isinstance(self._depth_store, h5py.File):
            self._depth_store.close()
        self._depth_store_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _resize_image(self, image: np.ndarray) -> Image.Image:
        h, w = self.input_size
        return Image.fromarray(image).resize((w, h), resample=Image.BILINEAR)

    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        h, w = self.input_size
        depth_t = torch.from_numpy(depth.astype(np.float32))[None, None]
        depth_t = F.interpolate(depth_t, size=(h, w), mode="bilinear", align_corners=False)
        return depth_t[0, 0].numpy()

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if not self.training:
            return image
        if bool(getattr(self.augment_cfg, "color_jitter", False)):
            image = self.jitter(image)
        if random.random() < float(getattr(self.augment_cfg, "blur_prob", 0.0)):
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))
        noise_std = float(getattr(self.augment_cfg, "noise_std", 0.0))
        if noise_std > 0:
            arr = np.asarray(image).astype(np.float32) / 255.0
            arr += np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
            image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        return image


def stylus_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    depths = torch.stack([b["depth"] for b in batch])
    has_depth = torch.tensor([b.get("has_depth", True) for b in batch], dtype=torch.bool)
    states = torch.tensor([(-1 if b.get("state") is None else b["state"]) for b in batch], dtype=torch.long)
    has_state = torch.tensor([b.get("has_state", b.get("state") is not None) for b in batch], dtype=torch.bool)
    touches = torch.tensor([(float("nan") if b.get("touch") is None else b["touch"]) for b in batch], dtype=torch.float32)
    has_touch = torch.tensor([b.get("has_touch", b.get("touch") is not None) for b in batch], dtype=torch.bool)
    xy = torch.stack([b["xy"] if b["xy"] is not None else torch.zeros(2) for b in batch]).float()
    valid_xy = torch.tensor([b["valid_xy"] for b in batch], dtype=torch.bool)
    pressure = torch.tensor([(float("nan") if b.get("pressure") is None else b["pressure"]) for b in batch], dtype=torch.float32)
    has_labels = torch.tensor([b.get("has_labels", False) for b in batch], dtype=torch.bool)
    return {
        "image": images,
        "depth": depths,
        "has_depth": has_depth,
        "state": states,
        "has_state": has_state,
        "valid_state": has_state,
        "touch": touches.nan_to_num(0.0),
        "has_touch": has_touch,
        "valid_touch": has_touch,
        "xy": xy,
        "valid_xy": valid_xy,
        "pressure": pressure.nan_to_num(0.0),
        "has_labels": has_labels,
        "participant_id": [str(b.get("participant_id", "UNKNOWN")) for b in batch],
        "task_id": [str(b.get("task_id", "UNKNOWN")) for b in batch],
        "video_id": [b["video_id"] for b in batch],
        "frame_index": torch.tensor([b["frame_index"] for b in batch], dtype=torch.long),
    }


def _participant_from_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "participant":
        return parts[1]
    return "UNKNOWN"


def _task_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return parts[2] if len(parts) >= 3 and parts[0] == "participant" else "UNKNOWN"


def _excluded_frame_indices(item: dict[str, Any]) -> set[int]:
    excluded: set[int] = set()
    for value in item.get("exclude_frame_indices", []) or []:
        excluded.add(int(value))
    for frame_range in item.get("exclude_frame_ranges", []) or []:
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            raise ValueError(f"exclude_frame_ranges entries must be [start, end): {frame_range}")
        start, end = int(frame_range[0]), int(frame_range[1])
        if end > start:
            excluded.update(range(start, end))
    return excluded


def _variant_kind_filter(cfg: Any) -> set[str] | None:
    value = getattr(cfg.data, "variant_kind_filter", None)
    if value is None:
        value = getattr(cfg.data, "variant_kinds", None)
    if value is None:
        return None
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part) for part in value}


def surface_to_image_xy(
    x: float,
    y: float,
    corners: np.ndarray,
    surface_w: float,
    surface_h: float,
    image_w: float,
    image_h: float,
    surface_model_w: float = 480.0,
    surface_model_h: float = 320.0,
    corners_coordinate_mode: str = "resized",
    resize_attrs: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Project stylus surface coordinates through the per-frame surface quadrilateral.

    This mirrors the C# viewer: normalized stylus coordinates are first placed
    into a canonical 480x320 rectangle, then projected to the image surface quad
    with a perspective homography.

    H5 `corners` are expected in order top-left, top-right, bottom-right,
    bottom-left. For the current /data/baseAVIs_432x672 files they appear to
    already be in resized-video coordinates; set corners_coordinate_mode=original
    to apply H5 resize metadata before normalization.
    """

    if surface_w <= 0 or surface_h <= 0:
        raise ValueError("surface_to_image_xy requires positive surface_w/surface_h")
    c = np.asarray(corners, dtype=np.float32)
    if c.shape != (4, 2) or not np.isfinite(c).all():
        return None
    u = float(x) / float(surface_w)
    v = float(y) / float(surface_h)
    if not np.isfinite([u, v]).all() or u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return None
    if corners_coordinate_mode == "original":
        c = _resize_points(c, resize_attrs or {})
    elif corners_coordinate_mode != "resized":
        raise ValueError(f"Unknown corners_coordinate_mode: {corners_coordinate_mode}")

    src = np.array(
        [
            [0.0, 0.0],
            [surface_model_w - 1.0, 0.0],
            [surface_model_w - 1.0, surface_model_h - 1.0],
            [0.0, surface_model_h - 1.0],
        ],
        dtype=np.float32,
    )
    h_mat = cv2.getPerspectiveTransform(src, c.astype(np.float32))
    src_pt = np.array([u * (surface_model_w - 1.0), v * (surface_model_h - 1.0), 1.0], dtype=np.float32)
    dst_pt = h_mat @ src_pt
    if abs(float(dst_pt[2])) < 1e-8:
        return None
    point = np.array([dst_pt[0] / dst_pt[2], dst_pt[1] / dst_pt[2]], dtype=np.float32)
    xy = np.array([point[0] / image_w, point[1] / image_h], dtype=np.float32)
    if not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
        return None
    return xy


def _resize_points(points: np.ndarray, attrs: dict[str, Any]) -> np.ndarray:
    if not attrs:
        return points
    scale_x = float(attrs.get("scale_x", 1.0))
    scale_y = float(attrs.get("scale_y", 1.0))
    crop_left = float(attrs.get("crop_left", 0.0))
    crop_top = float(attrs.get("crop_top", 0.0))
    pad_left = float(attrs.get("pad_left", 0.0))
    pad_top = float(attrs.get("pad_top", 0.0))
    out = points.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - crop_left) * scale_x + pad_left
    out[:, 1] = (out[:, 1] - crop_top) * scale_y + pad_top
    return out
