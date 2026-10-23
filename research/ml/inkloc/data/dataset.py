from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .transforms import load_rgb_image


REQUIRED_FIELDS = {"image_path", "state", "touch", "valid", "x_norm", "y_norm", "has_xy"}


class ManifestInteractionDataset(Dataset):
    """JSONL/CSV dataset for canonical warped RGB interaction frames."""

    def __init__(self, manifest_path: str | Path, image_root: str | Path | None = None, height: int = 320, width: int = 480):
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root is not None else self.manifest_path.parent
        self.height = height
        self.width = width
        self.rows = self._read_manifest(self.manifest_path)
        if not self.rows:
            raise ValueError(f"Manifest has no rows: {self.manifest_path}")

    def _read_manifest(self, path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if line.strip():
                        row = json.loads(line)
                        self._validate_row(row, line_no)
                        rows.append(row)
            return rows
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            for idx, row in enumerate(rows, start=2):
                self._validate_row(row, idx)
            return rows
        raise ValueError("Manifest must be .jsonl or .csv")

    def _validate_row(self, row: Dict, line_no: int) -> None:
        missing = REQUIRED_FIELDS - set(row.keys())
        if missing:
            raise ValueError(f"{self.manifest_path}:{line_no} missing fields: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        image = load_rgb_image(str(image_path), self.height, self.width)
        state = int(row["state"])
        valid = float(row["valid"])
        touch = float(row["touch"])
        has_xy = float(row["has_xy"])
        xy = torch.tensor([float(row["x_norm"]), float(row["y_norm"])], dtype=torch.float32).clamp(0.0, 1.0)
        sample: Dict[str, torch.Tensor] = {
            "image": image,
            "valid": torch.tensor(valid, dtype=torch.float32),
            "touch": torch.tensor(touch, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.long),
            "xy": xy,
            "has_xy": torch.tensor(has_xy, dtype=torch.float32),
        }
        if row.get("pressure") not in (None, ""):
            sample["pressure"] = torch.tensor(float(row["pressure"]), dtype=torch.float32)
        if row.get("transition") not in (None, ""):
            sample["transition"] = torch.tensor(int(row["transition"]), dtype=torch.long)
        return sample
