from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, Any

import torch


def resolve_ckpt_path(ckpt_arg: str, save_dir: Path) -> Path:
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
    else:
        best = save_dir / "checkpoint_best.pt"
        if best.exists():
            ckpt_path = best
        else:
            epoch_ckpts = sorted(
                save_dir.glob("checkpoint_epoch*.pt"),
                key=lambda p: _extract_epoch_num(p.name),
            )
            if not epoch_ckpts:
                raise FileNotFoundError(f"No checkpoints found in {save_dir}")
            ckpt_path = epoch_ckpts[-1]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _extract_epoch_num(name: str) -> int:
    m = re.search(r"checkpoint_epoch(\d+)\.pt$", name)
    if not m:
        return -1
    return int(m.group(1))


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = p * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for key in ("image", "image_clip", "corners", "xy", "labels"):
            x = batch.get(key)
            if torch.is_tensor(x) and x.ndim > 0:
                return int(x.shape[0])
        for v in batch.values():
            if torch.is_tensor(v) and v.ndim > 0:
                return int(v.shape[0])
    if torch.is_tensor(batch) and batch.ndim > 0:
        return int(batch.shape[0])
    return 1


@torch.inference_mode()
def run_loader_latency(
    loader: Iterable[Any],
    device: torch.device,
    runs: int,
    warmup_runs: int,
    infer_batch: Callable[[Any], None],
) -> Dict[str, float]:
    image_ms: list[float] = []
    batch_ms: list[float] = []

    total_runs = max(1, int(runs)) + max(0, int(warmup_runs))
    for ri in range(total_runs):
        is_warmup = ri < warmup_runs
        for batch in loader:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = perf_counter()
            infer_batch(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt_ms = (perf_counter() - t0) * 1000.0

            if is_warmup:
                continue
            bs = max(_infer_batch_size(batch), 1)
            batch_ms.append(float(dt_ms))
            image_ms.append(float(dt_ms) / bs)

    if not image_ms:
        return {}

    image_ms_sorted = sorted(image_ms)
    batch_ms_sorted = sorted(batch_ms)
    n = len(image_ms_sorted)
    mean_image = sum(image_ms_sorted) / n
    var_image = sum((x - mean_image) ** 2 for x in image_ms_sorted) / n

    return {
        "num_batches": float(len(batch_ms_sorted)),
        "num_images_profiled": float(n),
        "image_ms_mean": mean_image,
        "image_ms_std": var_image**0.5,
        "image_ms_min": image_ms_sorted[0],
        "image_ms_p50": _percentile(image_ms_sorted, 0.50),
        "image_ms_p90": _percentile(image_ms_sorted, 0.90),
        "image_ms_p95": _percentile(image_ms_sorted, 0.95),
        "image_ms_max": image_ms_sorted[-1],
        "batch_ms_mean": sum(batch_ms_sorted) / max(len(batch_ms_sorted), 1),
    }
