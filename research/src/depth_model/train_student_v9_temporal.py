# train_student_v9_temporal.py
import csv
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.student_depth_net_v9_temporal import StudentDepthNetV9TemporalLite


DATASET_ROOT = Path("/data/depth/pseudo_depth_dataset")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
INPUT_SIZE = 256
BATCH_SIZE = 8
SEQUENCE_LENGTH = 4
SEQUENCE_STRIDE = 1
EPOCHS = 1
LR = 3e-4
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILOG_WEIGHT = 0.1
TEMPORAL_SMOOTHNESS_WEIGHT = 0.02
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-6
DEPTH_CLIP_MIN = 0.45
DEPTH_CLIP_MAX = 15.0
DEFAULT_INIT_CKPT = DATASET_ROOT / "student_v8_best.pt"
DEFAULT_ONNX_PATH = DATASET_ROOT / "student_v9_temporal_best.onnx"
DEFAULT_QUANT_ONNX_PATH = DATASET_ROOT / "student_v9_temporal_best.int8.onnx"
VAL_SPLITS = {"test", "validate"}
SAMPLE_ID_PATTERN = re.compile(r"^(.*?)(\d+)$")


class TemporalPseudoDepthDataset(Dataset):
    def __init__(self, root: Path, windows):
        self.root = root
        self.windows = list(windows)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.windows)

    def _load_row(self, row):
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        image = self.to_tensor(image)

        depth = np.load(self.root / row["depth_path"]).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = depth.clamp(min=DEPTH_CLIP_MIN, max=DEPTH_CLIP_MAX)
        depth = (depth - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN)
        return image, depth

    def __getitem__(self, idx):
        rows = self.windows[idx]
        images = []
        depths = []
        for row in rows:
            image, depth = self._load_row(row)
            images.append(image)
            depths.append(depth)

        return torch.stack(images, dim=0), torch.stack(depths, dim=0)


class ExportableStudentDepthNetV9TemporalLite(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image, prev_state):
        logits, next_state = self.model(image, prev_state, return_state=True)
        return F.softplus(logits) + 1e-4, next_state


def load_manifest_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_rows(rows):
    train_rows = []
    val_rows = []

    for row in rows:
        split = row.get("split", "").strip().lower()
        sample_id = row.get("sample_id", "")
        is_avi = split == "avi_black" or sample_id.startswith("avi_")

        if is_avi:
            train_rows.append(row)
        elif split in VAL_SPLITS:
            val_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows:
        raise RuntimeError("No training rows were found in the manifest.")
    if not val_rows:
        raise RuntimeError(
            "No validation rows were found. Expected non-AVI rows with split=test/validate."
        )

    return train_rows, val_rows


def temporal_group_key(row):
    sample_id = row.get("sample_id", "")
    match = SAMPLE_ID_PATTERN.match(sample_id)
    if match:
        prefix = match.group(1)
    else:
        prefix = sample_id
    return row.get("split", "").strip().lower(), prefix


def build_temporal_windows(rows, sequence_length: int, sequence_stride: int):
    grouped = defaultdict(list)
    for row in rows:
        grouped[temporal_group_key(row)].append(row)

    windows = []
    for group_rows in grouped.values():
        if len(group_rows) < sequence_length:
            continue
        for start in range(0, len(group_rows) - sequence_length + 1, sequence_stride):
            windows.append(group_rows[start : start + sequence_length])

    if not windows:
        raise RuntimeError(
            f"No temporal windows were produced. sequence_length={sequence_length}, "
            f"sequence_stride={sequence_stride}"
        )

    return windows


def configure_runtime():
    if DEVICE != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def move_images(images, use_channels_last: bool):
    if use_channels_last and DEVICE == "cuda":
        return images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
    return images.to(DEVICE, non_blocking=True)


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, prefetch_factor: int):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**loader_kwargs)


def positive_depth_from_logits(pred_logits):
    return F.softplus(pred_logits) + 1e-4


def si_log_loss(pred, target, eps=1e-6):
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    g = torch.log(pred) - torch.log(target)
    return torch.sqrt((g ** 2).mean() - 0.5 * (g.mean() ** 2) + eps)


def compute_step_metrics(pred_logits, depths, prev_pred_depth=None):
    pred_depth = positive_depth_from_logits(pred_logits)
    l1 = F.l1_loss(pred_depth, depths)
    silog = si_log_loss(pred_depth, depths)

    temporal = pred_depth.new_zeros(())
    if prev_pred_depth is not None:
        temporal = F.l1_loss(pred_depth, prev_pred_depth)

    loss = l1 + SILOG_WEIGHT * silog + TEMPORAL_SMOOTHNESS_WEIGHT * temporal
    return {
        "loss": loss,
        "l1": l1,
        "silog": silog,
        "temporal": temporal,
        "pred_depth": pred_depth,
    }


def run_sequence(model, images, depths, use_channels_last: bool):
    state = None
    prev_pred_depth = None
    totals = {
        "loss": torch.zeros((), device=DEVICE),
        "l1": torch.zeros((), device=DEVICE),
        "silog": torch.zeros((), device=DEVICE),
        "temporal": torch.zeros((), device=DEVICE),
    }

    steps = images.size(1)
    for step in range(steps):
        frame = move_images(images[:, step], use_channels_last)
        target = depths[:, step].to(DEVICE, non_blocking=True)
        pred_logits, state = model(frame, state, return_state=True)
        metrics = compute_step_metrics(pred_logits, target, prev_pred_depth)

        for key in totals:
            totals[key] = totals[key] + metrics[key]
        prev_pred_depth = metrics["pred_depth"].detach()

    denom = float(steps)
    return {key: value / denom for key, value in totals.items()}


def train_one_epoch(model, loader, optimizer, use_channels_last: bool):
    model.train()
    totals = {"loss": 0.0, "l1": 0.0, "silog": 0.0, "temporal": 0.0}
    total_frames = 0

    for images, depths in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            metrics = run_sequence(model, images, depths, use_channels_last)

        metrics["loss"].backward()
        optimizer.step()

        frame_count = images.size(0) * images.size(1)
        total_frames += frame_count
        for key in totals:
            totals[key] += metrics[key].item() * frame_count

    return {key: value / total_frames for key, value in totals.items()}


@torch.no_grad()
def eval_one_epoch(model, loader, use_channels_last: bool):
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "silog": 0.0, "temporal": 0.0}
    total_frames = 0

    for images, depths in tqdm(loader, desc="val", leave=False):
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            metrics = run_sequence(model, images, depths, use_channels_last)

        frame_count = images.size(0) * images.size(1)
        total_frames += frame_count
        for key in totals:
            totals[key] += metrics[key].item() * frame_count

    return {key: value / total_frames for key, value in totals.items()}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--resume-best",
        action="store_true",
        default=True,
        help="Resume training from student_v9_temporal_best.pt if it exists.",
    )
    parser.add_argument(
        "--no-resume-best",
        action="store_false",
        dest="resume_best",
        help="Start from a warm-start checkpoint or scratch even if student_v9_temporal_best.pt exists.",
    )
    parser.add_argument(
        "--init-from-path",
        type=Path,
        default=DEFAULT_INIT_CKPT,
        help="Warm-start checkpoint to load when no V9 checkpoint exists.",
    )
    parser.add_argument(
        "--no-init-from",
        action="store_true",
        help="Start from random initialization when no V9 checkpoint exists.",
    )
    parser.add_argument(
        "--save-onnx",
        action="store_true",
        help="Export the best checkpoint to ONNX after training.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Path for the exported ONNX model.",
    )
    parser.add_argument(
        "--save-quantized-onnx",
        action="store_true",
        help="Export an int8-quantized ONNX model after training.",
    )
    parser.add_argument(
        "--quantized-onnx-path",
        type=Path,
        default=DEFAULT_QUANT_ONNX_PATH,
        help="Path for the quantized ONNX model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training and validation batch size in sequence windows.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Frames per temporal training window.",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=SEQUENCE_STRIDE,
        help="Sliding-window stride for temporal windows.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=PREFETCH_FACTOR,
        help="Per-worker DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels-last memory format for the model and input images on CUDA.",
    )
    parser.add_argument(
        "--temporal-smoothness-weight",
        type=float,
        default=TEMPORAL_SMOOTHNESS_WEIGHT,
        help="Weight for consecutive-frame depth smoothness within each sequence.",
    )
    return parser.parse_args()


def load_compatible_warmstart(model, checkpoint_path: Path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model_state = model.state_dict()

    loaded_keys = []
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)
    print(
        f"Initialized compatible weights from {checkpoint_path} | "
        f"loaded={len(loaded_keys)} | skipped={len(skipped_keys)}"
    )


def maybe_load_initial_weights(
    model,
    resume_path: Path,
    init_from_path: Path,
    resume_best: bool,
    no_init_from: bool,
):
    if resume_best and resume_path.exists():
        state_dict = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {resume_path}")
        return "resume"

    if (not no_init_from) and init_from_path.exists():
        load_compatible_warmstart(model, init_from_path)
        return "warm_start"

    return "scratch"


def export_onnx(model_path: Path, output_path: Path, use_channels_last: bool = False):
    export_model = StudentDepthNetV9TemporalLite()
    state_dict = torch.load(model_path, map_location="cpu")
    export_model.load_state_dict(state_dict)
    export_model.eval()
    if use_channels_last:
        export_model = export_model.to(memory_format=torch.channels_last)

    wrapped = ExportableStudentDepthNetV9TemporalLite(export_model)
    dummy_image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    dummy_state = export_model.init_state(1, INPUT_SIZE, INPUT_SIZE, device="cpu")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        (dummy_image, dummy_state),
        output_path,
        input_names=["image", "prev_state"],
        output_names=["depth", "next_state"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "prev_state": {0: "batch", 2: "state_height", 3: "state_width"},
            "depth": {0: "batch", 2: "height", 3: "width"},
            "next_state": {0: "batch", 2: "state_height", 3: "state_width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


def export_quantized_onnx(onnx_path: Path, quantized_path: Path):
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "Quantized ONNX export requires onnxruntime to be installed in the training environment."
        ) from exc

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Exported quantized ONNX model to {quantized_path}")


def main():
    global TEMPORAL_SMOOTHNESS_WEIGHT

    args = parse_args()
    TEMPORAL_SMOOTHNESS_WEIGHT = args.temporal_smoothness_weight
    configure_runtime()

    manifest_rows = load_manifest_rows(MANIFEST_PATH)
    train_rows, val_rows = split_rows(manifest_rows)
    train_windows = build_temporal_windows(
        train_rows,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
    )
    val_windows = build_temporal_windows(
        val_rows,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
    )
    train_ds = TemporalPseudoDepthDataset(DATASET_ROOT, train_windows)
    val_ds = TemporalPseudoDepthDataset(DATASET_ROOT, val_windows)
    print(
        "Temporal dataset | "
        f"train_windows={len(train_windows)} | "
        f"val_windows={len(val_windows)} | "
        f"sequence_length={args.sequence_length} | "
        f"sequence_stride={args.sequence_stride}"
    )

    train_loader = build_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = StudentDepthNetV9TemporalLite().to(DEVICE)
    if args.channels_last and DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ckpt_best = DATASET_ROOT / "student_v9_temporal_best.pt"
    ckpt_last = DATASET_ROOT / "student_v9_temporal_last.pt"
    init_source = maybe_load_initial_weights(
        model,
        ckpt_best,
        args.init_from_path,
        args.resume_best,
        args.no_init_from,
    )

    baseline_metrics = eval_one_epoch(model, val_loader, args.channels_last)
    print(
        "Initial val | "
        f"source={init_source} | "
        f"val_loss={baseline_metrics['loss']:.6f} | "
        f"val_l1={baseline_metrics['l1']:.6f} | "
        f"val_silog={baseline_metrics['silog']:.6f} | "
        f"val_temporal={baseline_metrics['temporal']:.6f}"
    )

    best_val = baseline_metrics["loss"]
    if init_source != "resume":
        torch.save(model.state_dict(), ckpt_best)
        print(f"Saved initial best model to {ckpt_best}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, args.channels_last)
        val_metrics = eval_one_epoch(model, val_loader, args.channels_last)
        scheduler.step(val_metrics["loss"])
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"train_l1={train_metrics['l1']:.6f} | "
            f"train_silog={train_metrics['silog']:.6f} | "
            f"train_temporal={train_metrics['temporal']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_l1={val_metrics['l1']:.6f} | "
            f"val_silog={val_metrics['silog']:.6f} | "
            f"val_temporal={val_metrics['temporal']:.6f}"
        )

        torch.save(model.state_dict(), ckpt_last)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), ckpt_best)
            print(f"Saved best model to {ckpt_best}")

    if args.save_onnx or args.save_quantized_onnx:
        export_onnx(ckpt_best, args.onnx_path, use_channels_last=args.channels_last)

    if args.save_quantized_onnx:
        export_quantized_onnx(args.onnx_path, args.quantized_onnx_path)


if __name__ == "__main__":
    main()
