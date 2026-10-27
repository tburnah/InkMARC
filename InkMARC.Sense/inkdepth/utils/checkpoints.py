from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("epoch", -1)) + 1


def load_model_weights(
    path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    skip_mismatched: bool = False,
) -> None:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload)
    if skip_mismatched:
        current = model.state_dict()
        skipped = []
        filtered = {}
        for key, value in state.items():
            if key in current and current[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        state = filtered
        if skipped:
            print(f"Skipped {len(skipped)} mismatched checkpoint tensors")
    model.load_state_dict(state, strict=strict)
