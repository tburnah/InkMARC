from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """Dictionary with attribute access for nested YAML configs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = deepcopy(cfg)
        for dotted_key, value in overrides.items():
            set_by_dotted_key(cfg, dotted_key, value)
    return _to_config(cfg)


def set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    current: dict[str, Any] = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
