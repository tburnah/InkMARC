#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Robust imports (repo layout)
# ----------------------------
try:
    from data.dataset_core import InkMARCIndexedDataset  # type: ignore
    from data.video_io import VideoFrameReader  # type: ignore
except ModuleNotFoundError:
    from dataset_core import InkMARCIndexedDataset  # type: ignore
    from video_io import VideoFrameReader  # type: ignore

from torchvision import models


# ----------------------------
# Repro + speed
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_torch_perf_knobs() -> None:
    # These are safe defaults for CNN inference/training speed.
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def worker_init_fn(_: int) -> None:
    """
    Critical performance fix: prevent OpenCV from spawning threads inside each DataLoader worker.
    Without this, 12 workers * 8 OpenCV threads can thrash your 16-core VM.
    """
    try:
        import cv2  # type: ignore

        cv2.setNumThreads(0)
    except Exception:
        pass


# ----------------------------
# Preprocess: letterbox or warp
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_frame_bgr_to_tensor(
    frame_bgr: np.ndarray,
    img_size: int,
    mode: str,
) -> torch.Tensor:
    """
    frame_bgr: HxWx3 uint8 (OpenCV BGR)
    returns: 3ximg_sizeximg_size float32 normalized like torchvision
    """
    import cv2  # type: ignore

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR, got {frame_bgr.shape}")

    # BGR -> RGB float [0,1]
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    h, w, _ = rgb.shape

    if mode == "warp":
        rgb_rs = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    elif mode == "letterbox":
        scale = img_size / max(h, w)
        nh = int(round(h * scale))
        nw = int(round(w * scale))
        rgb_small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((img_size, img_size, 3), dtype=np.float32)
        top = (img_size - nh) // 2
        left = (img_size - nw) // 2
        canvas[top : top + nh, left : left + nw, :] = rgb_small
        rgb_rs = canvas
    else:
        raise ValueError(f"Unknown mode={mode}. Use warp|letterbox")

    # HWC -> CHW
    x = np.transpose(rgb_rs, (2, 0, 1))  # 3xHxW

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    x = (x - mean) / std

    return torch.from_numpy(x).float()


# ----------------------------
# Batch + Collate
# ----------------------------
@dataclass
class Batch:
    images: torch.Tensor  # (B,3,H,W)
    y: torch.Tensor       # (B,) float {0,1}
    meta: List[Dict[str, Any]]


def collate_fn(items: List[Dict[str, Any]]) -> Batch:
    images = torch.stack([it["image"] for it in items], dim=0)
    y = torch.tensor([it["y"] for it in items], dtype=torch.float32)
    meta = [it["meta"] for it in items]
    return Batch(images=images, y=y, meta=meta)


# ----------------------------
# Dataset wrapper
# ----------------------------
class InteractionDataset(torch.utils.data.Dataset):
    """
    Stage A:
      y=1 if label_mask[2]==1 (pressure channel exists => interaction regime)
      y=0 if label_mask[2]==0 (no pressure channel => far)
    Must use require_labels=False to include far frames.
    """

    def __init__(
        self,
        dataset_root: str,
        version: str,
        split: str,
        img_size: int,
        mode: str,
        stride: int,
        max_samples: int,
        cache_size: int,
        fixed_variant: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.version = version
        self.split = split
        self.img_size = img_size
        self.mode = mode
        self.fixed_variant = fixed_variant
        self.rng = random.Random(seed + (0 if split == "train" else 999))

        # CRITICAL: require_labels=False so we can see far-hover frames (mask[2]==0).
        self.ds = InkMARCIndexedDataset(
            dataset_root=dataset_root,
            index_version=version,
            split=split,
            require_labels=False,
            transform=None,
        )

        # Increase cache size to reduce reopen/seek overhead
        self.vr = VideoFrameReader(dataset_root, self.ds.manifest, cache_size=cache_size)

        # Optional downsampling for speed
        self.indices = list(range(0, len(self.ds), max(1, stride)))
        if max_samples and max_samples > 0:
            self.indices = self.indices[:max_samples]

        # If fixed_variant: choose one deterministically per dataset instance to maximize cache hits
        self._fixed_variant_id: Optional[int] = None
        if fixed_variant and len(self.ds.manifest.get("variants", [])) > 0:
            self._fixed_variant_id = 0

    def __len__(self) -> int:
        return len(self.indices)

    def _get_frame(self, file_id: str, frame_idx: int) -> np.ndarray:
        # Try to keep variant selection stable for throughput when fixed_variant is enabled.
        # This assumes VideoFrameReader internally uses manifest variants for training.
        # If VideoFrameReader doesn't expose variant selection, fixed_variant simply won't change behavior.
        return self.vr.get_frame(file_id, frame_idx, split=self.split)

    def __getitem__(self, j: int) -> Dict[str, Any]:
        i = self.indices[j]
        s: Dict[str, Any] = self.ds[i]

        labels = np.asarray(s.get("labels", [0, 0, 0, 0, 0]), dtype=np.float32)
        mask = np.asarray(s.get("label_mask", [0, 0, 0, 0, 0]), dtype=np.float32)

        # Stage A label: interaction if pressure channel present
        has_pressure = 1.0 if float(mask[2]) != 0.0 else 0.0
        y = has_pressure

        meta = s["meta"]
        frame = self._get_frame(str(meta["file_id"]), int(meta["frame_idx"]))
        x = preprocess_frame_bgr_to_tensor(frame, img_size=self.img_size, mode=self.mode)

        return {"image": x, "y": y, "meta": meta}

    def close(self) -> None:
        try:
            self.ds.close()
        except Exception:
            pass
        try:
            self.vr.release()
        except Exception:
            pass


# ----------------------------
# Model factory (logits output)
# ----------------------------
class BinaryHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dropout > 0:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 1))
        else:
            self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(1)  # (B,)


def make_model(arch: str, dropout: float) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_dim = base.fc.in_features
        base.fc = nn.Linear(in_dim, 1)  # logits
        # ensure output shape (B,)
        return nn.Sequential(base, nn.Flatten(start_dim=0, end_dim=0))  # harmless wrapper
    elif arch == "mobilenet_v3_small":
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # classifier: Sequential(..., Linear(in_features, out_features))
        in_dim = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_dim, 1)  # logits
        return base
    else:
        raise ValueError(f"Unknown arch={arch}. Use resnet18|mobilenet_v3_small")


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    out = out.view(-1)  # (B,)
    return out


# ----------------------------
# Metrics / eval
# ----------------------------
@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(torch.int64)
    yt = y.to(torch.int64)

    tp = int(((preds == 1) & (yt == 1)).sum().item())
    tn = int(((preds == 0) & (yt == 0)).sum().item())
    fp = int(((preds == 1) & (yt == 0)).sum().item())
    fn = int(((preds == 0) & (yt == 1)).sum().item())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return float(acc), float(f1)


@torch.no_grad()
def estimate_pos_weight(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 300,
) -> float:
    pos = 0.0
    neg = 0.0
    for b_idx, batch in enumerate(loader):
        y = batch.y.to(device, non_blocking=True)
        pos += float((y == 1).sum().item())
        neg += float((y == 0).sum().item())
        if (b_idx + 1) >= max_batches:
            break
    if pos <= 0:
        return 1.0
    return float(neg / pos)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    return_raw: bool = False,
) -> Tuple[float, float, float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[Dict[str, Any]]]]:
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss = 0.0
    n = 0

    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    meta_all: List[Dict[str, Any]] = []

    for batch in loader:
        x = batch.images.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = forward_logits(model, x)
            loss = crit(logits, y)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs

        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
        meta_all.extend(batch.meta)   # <-- add this

    logits_cat = torch.cat(logits_all, dim=0)
    y_cat = torch.cat(y_all, dim=0)

    acc, f1 = metrics_from_logits(logits_cat, y_cat)

    if return_raw:
        return total_loss / max(1, n), acc, f1, logits_cat, y_cat, meta_all
    return total_loss / max(1, n), acc, f1, None, None, None

def hysteresis_sequence_report(
    *,
    logits: torch.Tensor,
    y: torch.Tensor,
    meta: List[Dict[str, Any]],
    tau_enter: float,
    tau_exit: float,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits).numpy()
    yt = y.numpy().astype(np.int32)

    # Build per-frame records
    records = []
    for p, gt, m in zip(probs, yt, meta):
        file_id = str(m["file_id"])
        frame_idx = int(m["frame_idx"])
        records.append((file_id, frame_idx, float(p), int(gt)))

    # Group by file_id and sort by time
    from collections import defaultdict
    groups = defaultdict(list)
    for file_id, frame_idx, p, gt in records:
        groups[file_id].append((frame_idx, p, gt))

    total_frames = 0
    total_flips = 0
    total_fp_frames = 0  # predicted 1 while gt 0
    total_fn_frames = 0  # predicted 0 while gt 1

    # Segment stats
    interaction_segment_lengths = []
    false_open_segment_lengths = []
    false_close_segment_lengths = []

    def segment_lengths(mask: np.ndarray) -> List[int]:
        # mask is bool (T,)
        out = []
        run = 0
        for v in mask:
            if v:
                run += 1
            else:
                if run > 0:
                    out.append(run)
                    run = 0
        if run > 0:
            out.append(run)
        return out

    for file_id, seq in groups.items():
        seq.sort(key=lambda t: t[0])  # sort by frame_idx
        p_seq = np.array([t[1] for t in seq], dtype=np.float32)
        gt_seq = np.array([t[2] for t in seq], dtype=np.int32)

        pred_seq = apply_hysteresis(p_seq, tau_enter=tau_enter, tau_exit=tau_exit, initial_state=0)

        # flips
        flips = int(np.sum(pred_seq[1:] != pred_seq[:-1]))
        total_flips += flips
        total_frames += len(pred_seq)

        # per-frame fp/fn
        total_fp_frames += int(np.sum((pred_seq == 1) & (gt_seq == 0)))
        total_fn_frames += int(np.sum((pred_seq == 0) & (gt_seq == 1)))

        # segment lengths
        interaction_segment_lengths.extend(segment_lengths(pred_seq == 1))
        false_open_segment_lengths.extend(segment_lengths((pred_seq == 1) & (gt_seq == 0)))
        false_close_segment_lengths.extend(segment_lengths((pred_seq == 0) & (gt_seq == 1)))

    flips_per_1k = 1000.0 * total_flips / max(1, total_frames)
    fp_rate = total_fp_frames / max(1, int(np.sum(yt == 0)))  # "false open" rate on FAR frames
    fn_rate = total_fn_frames / max(1, int(np.sum(yt == 1)))  # "miss" rate on INTERACTION frames

    def mean_or_zero(xs: List[int]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "frames": float(total_frames),
        "flips_per_1k_frames": float(flips_per_1k),
        "false_open_rate_on_far": float(fp_rate),
        "miss_rate_on_interaction": float(fn_rate),
        "mean_interaction_segment_len": mean_or_zero(interaction_segment_lengths),
        "mean_false_open_segment_len": mean_or_zero(false_open_segment_lengths),
        "mean_false_close_segment_len": mean_or_zero(false_close_segment_lengths),
        "num_interaction_segments": float(len(interaction_segment_lengths)),
        "num_false_open_segments": float(len(false_open_segment_lengths)),
        "num_false_close_segments": float(len(false_close_segment_lengths)),
    }

def apply_hysteresis(
    probs: np.ndarray,
    tau_enter: float,
    tau_exit: float,
    initial_state: int = 0,  # 0=FAR, 1=INTERACTION
) -> np.ndarray:
    """
    probs: (T,) probability of INTERACTION
    returns: (T,) states {0,1} after hysteresis.
    """
    state = int(initial_state)
    out = np.zeros_like(probs, dtype=np.int32)

    for t, p in enumerate(probs):
        if state == 0:
            if p >= tau_enter:
                state = 1
        else:
            if p <= tau_exit:
                state = 0
        out[t] = state

    return out


# ----------------------------
# Export
# ----------------------------
def export_onnx_logits(model: nn.Module, out_path: Path, img_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )


def save_metadata_json(
    out_path: Path,
    *,
    arch: str,
    img_size: int,
    mode: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    tau_enter: float,
    tau_exit: float,
) -> None:
    meta = {
        "task": "interaction_gate_stage_a",
        "arch": arch,
        "input": {
            "img_size": img_size,
            "mode": mode,
            "layout": "NCHW",
            "color": "RGB",
            "normalize": {"mean": list(mean), "std": list(std)},
        },
        "output": {"name": "logits", "activation": "sigmoid_external"},
        "recommended_hysteresis": {"tau_enter": tau_enter, "tau_exit": tau_exit},
        "notes": [
            "Stage A label: y=1 iff label_mask[2]==1, else y=0.",
            "Apply sigmoid(logits) on-device; use hysteresis thresholds to avoid flicker.",
        ],
    }
    out_path.write_text(json.dumps(meta, indent=2))

def sweep_thresholds(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    # logits,y on CPU
    probs = torch.sigmoid(logits).numpy()
    yt = y.numpy().astype(np.int32)

    best = {"thr": 0.5, "f1": -1.0, "acc": 0.0, "prec": 0.0, "rec": 0.0}

    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs >= thr).astype(np.int32)

        tp = int(((pred == 1) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())

        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)

        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1), "acc": float(acc),
                    "prec": float(prec), "rec": float(rec)}
    return best

def confusion_at_threshold(logits: torch.Tensor, y: torch.Tensor, thr: float) -> Dict[str, float]:
    probs = torch.sigmoid(logits).numpy()
    yt = y.numpy().astype(np.int32)
    pred = (probs >= thr).astype(np.int32)

    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())

    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    fpr  = fp / max(1, fp + tn)   # false-open rate on FAR
    fnr  = fn / max(1, fn + tp)   # miss rate on INTERACTION

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "fpr": fpr, "fnr": fnr}


# ----------------------------
# Training
# ----------------------------
def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable


def main() -> None:
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--save-dir", required=True)

    # Model / input
    ap.add_argument("--arch", choices=["mobilenet_v3_small", "resnet18"], default="mobilenet_v3_small")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--mode", choices=["letterbox", "warp"], default="letterbox")
    ap.add_argument("--dropout", type=float, default=0.0)

    # Train
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--auto-pos-weight", action="store_true", default=True)

    # Loader / speed
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--cache-size", type=int, default=8)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=2)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--fixed-variant", action="store_true", help="Use a stable video variant for speed (if supported).")

    # Warm start (learning speed)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1,
                    help="Freeze pretrained backbone for first N epochs (linear probe), then unfreeze.")

    # Gate hysteresis (saved to JSON)
    ap.add_argument("--tau-enter", type=float, default=0.15)
    ap.add_argument("--tau-exit", type=float, default=0.08)

    ap.add_argument("--resume", type=str, default="", help="Path to a .pt checkpoint to load")
    ap.add_argument("--eval-only", action="store_true", help="Only run val/eval then exit")

    args = ap.parse_args()

    seed_everything(args.seed)
    set_torch_perf_knobs()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and (device.type == "cuda")

    # Datasets
    train_ds = InteractionDataset(
        dataset_root=args.dataset_root,
        version=args.version,
        split="train",
        img_size=args.img_size,
        mode=args.mode,
        stride=args.train_stride,
        max_samples=args.max_train_samples,
        cache_size=args.cache_size,
        fixed_variant=args.fixed_variant,
        seed=args.seed,
    )
    val_ds = InteractionDataset(
        dataset_root=args.dataset_root,
        version=args.version,
        split="val",
        img_size=args.img_size,
        mode=args.mode,
        stride=args.val_stride,
        max_samples=args.max_val_samples,
        cache_size=args.cache_size,
        fixed_variant=True,  # always stable for val
        seed=args.seed,
    )

    pin = (device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # Model
    model = make_model(args.arch, dropout=args.dropout).to(device)

    start_epoch = 1
    best_f1 = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        print(f"[resume] loaded {args.resume} (epoch={start_epoch-1}, best_f1={best_f1:.4f})")

    if args.eval_only:
        val_loss, val_acc, val_f1, val_logits, val_y, val_meta = evaluate(
            model, val_loader, device=device, use_amp=use_amp, return_raw=True
        )

        # Choose your hysteresis (e.g. from calibration)
        tau_enter = 0.10
        tau_exit = 0.05

        rep = hysteresis_sequence_report(
            logits=val_logits,
            y=val_y,
            meta=val_meta,
            tau_enter=tau_enter,
            tau_exit=tau_exit,
        )

        print(f"[hysteresis] tau_enter={tau_enter:.2f} tau_exit={tau_exit:.2f}")
        for k, v in rep.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # val_loss, val_acc, val_f1, val_logits, val_y = evaluate(
        #     model, val_loader, device=device, use_amp=use_amp, return_raw=True
        # )
        # cal = sweep_thresholds(val_logits, val_y)
        # tau_enter = min(0.95, cal["thr"] + 0.10)
        # tau_exit  = max(0.05, cal["thr"] - 0.10)

        # print(f"[eval] val_loss={val_loss:.4f} acc={val_acc:.4f} f1@0.5={val_f1:.4f}")
        # print(f"[calibration] best_thr={cal['thr']:.2f} f1={cal['f1']:.4f} "
        #     f"prec={cal['prec']:.4f} rec={cal['rec']:.4f}")
        # print(f"[calibration] suggested hysteresis tau_enter={tau_enter:.2f} tau_exit={tau_exit:.2f}")

        # for t in [0.05, 0.15]:
        #     c = confusion_at_threshold(val_logits, val_y, t)
        #     print(f"[thr={t:.2f}] prec={c['prec']:.4f} rec={c['rec']:.4f} fpr={c['fpr']:.4f} fnr={c['fnr']:.4f} "
        #         f"(fp={c['fp']}, fn={c['fn']})")
        #     pos_rate = float((val_y == 1).float().mean().item())
        #     print(f"[val] positive_rate={pos_rate:.4f}")

        return

    # Optional: freeze backbone for a quick “linear probe”
    # Implemented simply as: freeze everything except last classifier layer.
    def freeze_for_linear_probe(m: nn.Module) -> None:
        for p in m.parameters():
            p.requires_grad = False
        # Unfreeze classifier head depending on arch
        if args.arch == "resnet18":
            # base.fc exists, but our wrapper is sequential in this branch
            # safer: unfreeze all Linear layers
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    for p in module.parameters():
                        p.requires_grad = True
        else:
            # MobileNetV3: classifier[-1]
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    for p in module.parameters():
                        p.requires_grad = True

    if args.freeze_backbone_epochs > 0:
        freeze_for_linear_probe(model)

    # Loss weighting
    pos_weight = 1.0
    if args.auto_pos_weight:
        pos_weight = estimate_pos_weight(train_loader, device=device, max_batches=300)
    print(f"[pos_weight] using auto pos_weight={pos_weight:.4f} (neg/pos)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze after warm-up
        if args.freeze_backbone_epochs > 0 and epoch == (args.freeze_backbone_epochs + 1):
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model.train()
        running = 0.0
        n = 0

        for batch in train_loader:
            x = batch.images.to(device, non_blocking=True)
            y = batch.y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = forward_logits(model, x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.shape[0]
            running += float(loss.item()) * bs
            n += bs

        train_loss = running / max(1, n)
        val_loss, val_acc, val_f1, val_logits, val_y = evaluate(
            model, val_loader, device=device, use_amp=use_amp, return_raw=True
        )

        cal = sweep_thresholds(val_logits, val_y)
        print(f"[calibration] best_thr={cal['thr']:.2f} f1={cal['f1']:.4f} "
            f"prec={cal['prec']:.4f} rec={cal['rec']:.4f}")

        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} "
            f"(val_total={len(val_ds)})"
        )

        # Save best by F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_f1": best_f1,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "best.pt")

    # Export deployment artifacts
    onnx_path = save_dir / "interaction_stageA_logits.onnx"
    export_onnx_logits(model, onnx_path, args.img_size, device)
    print(f"ONNX exported: {onnx_path}")

    meta_path = save_dir / "interaction_stageA_meta.json"
    save_metadata_json(
        meta_path,
        arch=args.arch,
        img_size=args.img_size,
        mode=args.mode,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        tau_enter=float(args.tau_enter),
        tau_exit=float(args.tau_exit),
    )
    print(f"Meta exported: {meta_path}")

    # Cleanup
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
