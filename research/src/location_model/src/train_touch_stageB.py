#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

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
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def worker_init_fn(_: int) -> None:
    # prevent OpenCV spawning threads inside each DataLoader worker
    try:
        import cv2  # type: ignore
        cv2.setNumThreads(0)
    except Exception:
        pass


# ----------------------------
# Preprocess
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_frame_bgr_to_tensor(frame_bgr: np.ndarray, img_size: int, mode: str) -> torch.Tensor:
    import cv2  # type: ignore

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR, got {frame_bgr.shape}")

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
        canvas[top:top + nh, left:left + nw, :] = rgb_small
        rgb_rs = canvas
    else:
        raise ValueError(f"Unknown mode={mode}. Use warp|letterbox")

    x = np.transpose(rgb_rs, (2, 0, 1))  # CHW

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    x = (x - mean) / std
    return torch.from_numpy(x).float()


# ----------------------------
# Batch + Collate
# ----------------------------
@dataclass
class Batch:
    images: torch.Tensor
    y: torch.Tensor
    meta: List[Dict[str, Any]]


def collate_fn(items: List[Dict[str, Any]]) -> Batch:
    images = torch.stack([it["image"] for it in items], dim=0)
    y = torch.tensor([it["y"] for it in items], dtype=torch.float32)
    meta = [it["meta"] for it in items]
    return Batch(images=images, y=y, meta=meta)


# ----------------------------
# Stage B Dataset
# ----------------------------
class TouchDataset(torch.utils.data.Dataset):
    """
    Stage B:
      Only include interaction-regime frames (label_mask[2] != 0).
      y=1 if "touch" (labels[touch_index] >= touch_threshold), else y=0 (hover-close).

    Notes:
      - We use require_labels=True by default to focus on frames with labels.
      - We also explicitly filter by mask[interaction_mask_index] to guarantee we're in regime.
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
        seed: int,
        *,
        interaction_mask_index: int = 2,
        touch_index: int = 2,
        touch_threshold: float = 0.5,
        hover_max: float = 0.01,
        touch_min: float = 0.03,
        require_labels: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.img_size = img_size
        self.mode = mode
        self.touch_index = int(touch_index)
        self.touch_threshold = float(touch_threshold)
        self.interaction_mask_index = int(interaction_mask_index)
        self.hover_max = hover_max
        self.touch_min = touch_min

        self.ds = InkMARCIndexedDataset(
            dataset_root=dataset_root,
            index_version=version,
            split=split,
            require_labels=require_labels,
            transform=None,
        )
        self.vr = VideoFrameReader(dataset_root, self.ds.manifest, cache_size=cache_size)

        # Build filtered indices: only frames in interaction regime
        raw = list(range(0, len(self.ds), max(1, stride)))
        keep: List[int] = []

        desc = f"[index] {split} filter interaction"
        for i in tqdm(raw, desc=desc, mininterval=1.0, smoothing=0.0):
            s: Dict[str, Any] = self.ds[i]
            mask = np.asarray(s.get("label_mask", [0, 0, 0, 0, 0]), dtype=np.float32)
            if float(mask[self.interaction_mask_index]) != 0.0:
                keep.append(i)

        if max_samples and max_samples > 0:
            keep = keep[:max_samples]

        self.indices = keep
        print(f"[index] {split}: kept {len(self.indices)}/{len(raw)} frames ({len(self.indices)/max(1,len(raw)):.3f})")

        if max_samples and max_samples > 0:
            keep = keep[:max_samples]

        self.indices = keep
        # deterministic split-specific RNG if you later add augmentation
        self.rng = random.Random(seed + (0 if split == "train" else 999))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int) -> Dict[str, Any]:
        i = self.indices[j]
        s: Dict[str, Any] = self.ds[i]

        labels = np.asarray(s.get("labels", [0, 0, 0, 0, 0]), dtype=np.float32)
        mask = np.asarray(s.get("label_mask", [0, 0, 0, 0, 0]), dtype=np.float32)

        # Safety check: should always hold due to filtering
        if float(mask[self.interaction_mask_index]) == 0.0:
            # Treat as hover (or raise). We'll be strict to avoid silent label pollution.
            raise RuntimeError("Stage B got a non-interaction frame after filtering.")

        pressure = float(labels[self.touch_index])

        if pressure <= self.hover_max:
            y = 0.0
        elif pressure >= self.touch_min:
            y = 1.0
        else:
            # ambiguous: resample a different index
            return self.__getitem__(self.rng.randrange(len(self)))

        meta = s["meta"]
        frame = self.vr.get_frame(str(meta["file_id"]), int(meta["frame_idx"]), split=self.split)
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
# Model
# ----------------------------
def make_model(arch: str) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_dim = base.fc.in_features
        base.fc = nn.Linear(in_dim, 1)
        return base
    if arch == "mobilenet_v3_small":
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_dim = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_dim, 1)
        return base
    raise ValueError(f"Unknown arch={arch}. Use resnet18|mobilenet_v3_small")

def pressure_stats(ds: TouchDataset, n: int = 50000) -> None:
    ps = []
    for j in range(min(len(ds.indices), n)):
        i = ds.indices[j]
        s = ds.ds[i]
        labels = np.asarray(s.get("labels", [0,0,0,0,0]), dtype=np.float32)
        ps.append(float(labels[ds.touch_index]))
    p = np.array(ps, dtype=np.float32)
    print("[pressure] min/max", float(p.min()), float(p.max()))
    for q in [0,1,5,10,25,50,75,90,95,99,100]:
        print(f"[pressure] p{q:02d}={float(np.percentile(p,q)):.6f}")
    print("[pressure] <=hover_max", float(np.mean(p <= ds.hover_max)))
    print("[pressure] >=touch_min", float(np.mean(p >= ds.touch_min)))
    print("[pressure] ambiguous", float(np.mean((p > ds.hover_max) & (p < ds.touch_min))))

def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).view(-1)


# ----------------------------
# Metrics / evaluation
# ----------------------------
@torch.no_grad()
def metrics_from_logits_at_thr(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).to(torch.int64)
    yt = y.to(torch.int64)

    tp = int(((pred == 1) & (yt == 1)).sum().item())
    tn = int(((pred == 0) & (yt == 0)).sum().item())
    fp = int(((pred == 1) & (yt == 0)).sum().item())
    fn = int(((pred == 0) & (yt == 1)).sum().item())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)

    # FPR/FNR relative to negative/positive class
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr),
            "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


def sweep_thresholds(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
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
            best = {"thr": float(thr), "f1": float(f1), "acc": float(acc), "prec": float(prec), "rec": float(rec)}
    return best


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    return_raw: bool = False,
) -> Tuple[float, Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[Dict[str, Any]]]]:
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
        meta_all.extend(batch.meta)

    logits_cat = torch.cat(logits_all, dim=0)
    y_cat = torch.cat(y_all, dim=0)

    m05 = metrics_from_logits_at_thr(logits_cat, y_cat, thr=0.5)

    if return_raw:
        return total_loss / max(1, n), m05, logits_cat, y_cat, meta_all
    return total_loss / max(1, n), m05, None, None, None


@torch.no_grad()
def estimate_pos_weight(loader: torch.utils.data.DataLoader, device: torch.device, max_batches: int = 300) -> float:
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


# ----------------------------
# Export metadata
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
    tau_enter: float,
    tau_exit: float,
    touch_index: int,
    touch_threshold: float,
) -> None:
    meta = {
        "task": "touch_vs_hover_stage_b",
        "arch": arch,
        "input": {
            "img_size": img_size,
            "mode": mode,
            "layout": "NCHW",
            "color": "RGB",
            "normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        },
        "output": {"name": "logits", "activation": "sigmoid_external"},
        "label_rule": {
            "only_if_interaction_mask_index": 2,
            "y_touch": f"labels[{touch_index}] >= {touch_threshold}",
        },
        "recommended_hysteresis": {"tau_enter": tau_enter, "tau_exit": tau_exit},
    }
    out_path.write_text(json.dumps(meta, indent=2))


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--save-dir", required=True)

    ap.add_argument("--arch", choices=["mobilenet_v3_small", "resnet18"], default="mobilenet_v3_small")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--mode", choices=["letterbox", "warp"], default="letterbox")

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--auto-pos-weight", action="store_true", default=True)

    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--cache-size", type=int, default=8)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=2)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)

    # Stage B label config
    ap.add_argument("--touch-index", type=int, default=2, help="labels[index] used for touch signal")
    ap.add_argument("--touch-threshold", type=float, default=0.5, help="touch if labels[touch_index] >= threshold")
    ap.add_argument("--interaction-mask-index", type=int, default=2, help="label_mask index defining interaction regime")

    # Hysteresis for Stage B output (optional; you can calibrate later)
    ap.add_argument("--tau-enter", type=float, default=0.60)
    ap.add_argument("--tau-exit", type=float, default=0.40)

    ap.add_argument("--resume", type=str, default="", help="Path to a .pt checkpoint to load")
    ap.add_argument("--eval-only", action="store_true", help="Only run val/eval then exit")

    ap.add_argument("--hover-max", type=float, default=0.01, help="pressure <= hover_max => hover label (0)")
    ap.add_argument("--touch-min", type=float, default=0.03, help="pressure >= touch_min => touch label (1)")


    args = ap.parse_args()

    seed_everything(args.seed)
    set_torch_perf_knobs()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and (device.type == "cuda")

    # Datasets
    train_ds = TouchDataset(
        dataset_root=args.dataset_root,
        version=args.version,
        split="train",
        img_size=args.img_size,
        mode=args.mode,
        stride=args.train_stride,
        max_samples=args.max_train_samples,
        cache_size=args.cache_size,
        seed=args.seed,
        interaction_mask_index=args.interaction_mask_index,
        touch_index=args.touch_index,
        touch_threshold=args.touch_threshold,
        hover_max=args.hover_max,
        touch_min=args.touch_min,
        require_labels=True,
    )
    val_ds = TouchDataset(
        dataset_root=args.dataset_root,
        version=args.version,
        split="val",
        img_size=args.img_size,
        mode=args.mode,
        stride=args.val_stride,
        max_samples=args.max_val_samples,
        cache_size=args.cache_size,
        seed=args.seed,
        interaction_mask_index=args.interaction_mask_index,
        touch_index=args.touch_index,
        touch_threshold=args.touch_threshold,
        hover_max=args.hover_max,
        touch_min=args.touch_min,        
        require_labels=True,
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

    # Model + resume
    model = make_model(args.arch).to(device)
    best_f1 = -1.0
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        print(f"[resume] loaded {args.resume} (epoch={start_epoch-1}, best_f1={best_f1:.4f})")

    # Eval-only
    if args.eval_only:
        print("\n=== Pressure statistics (validation set) ===")
        pressure_stats(val_ds)

        val_loss, m05, logits, y, _meta = evaluate(
            model, val_loader, device=device, use_amp=use_amp, return_raw=True
        )
        cal = sweep_thresholds(logits, y)
        pos_rate = float((y == 1).float().mean().item())

        print(f"[eval] val_loss={val_loss:.4f} "
            f"acc@0.5={m05['acc']:.4f} f1@0.5={m05['f1']:.4f} "
            f"pos_rate={pos_rate:.4f}")
        print(f"[calibration] best_thr={cal['thr']:.2f} "
            f"f1={cal['f1']:.4f} prec={cal['prec']:.4f} rec={cal['rec']:.4f}")

        train_ds.close()
        val_ds.close()
        return

    # Loss weighting
    pos_weight = 1.0
    if args.auto_pos_weight:
        pos_weight = estimate_pos_weight(train_loader, device=device, max_batches=300)
    print(f"[pos_weight] using pos_weight={pos_weight:.4f} (neg/pos)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d} [train]", mininterval=1.0, smoothing=0.0)
        for batch in pbar:
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

            # Update occasionally (tqdm throttles this)
            pbar.set_postfix(loss=f"{loss.item():.4f}", seen=n)

        train_loss = running / max(1, n)
        print(f"[train] epoch {epoch:02d} loss={train_loss:.4f} time={(time.time()-t0):.1f}s")

        val_loss, m05, val_logits, val_y, _meta = evaluate(model, val_loader, device=device, use_amp=use_amp, return_raw=True)
        cal = sweep_thresholds(val_logits, val_y)

        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"acc@0.5={m05['acc']:.4f} f1@0.5={m05['f1']:.4f} | "
            f"best_thr={cal['thr']:.2f} f1={cal['f1']:.4f}"
            f" (val_total={len(val_ds)})"
        )

        # save best by calibrated F1 (or switch to m05['f1'] if you prefer)
        if cal["f1"] > best_f1:
            best_f1 = cal["f1"]
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_f1": best_f1,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "best.pt")

    # Export artifacts
    onnx_path = save_dir / "touch_stageB_logits.onnx"
    export_onnx_logits(model, onnx_path, args.img_size, device)
    print(f"ONNX exported: {onnx_path}")

    meta_path = save_dir / "touch_stageB_meta.json"
    save_metadata_json(
        meta_path,
        arch=args.arch,
        img_size=args.img_size,
        mode=args.mode,
        tau_enter=float(args.tau_enter),
        tau_exit=float(args.tau_exit),
        touch_index=int(args.touch_index),
        touch_threshold=float(args.touch_threshold),
    )
    print(f"Meta exported: {meta_path}")

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
