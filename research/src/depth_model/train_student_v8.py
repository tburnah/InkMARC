# train_student_v8.py
import csv
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.student_depth_net_v8 import StudentDepthNetV8


DATASET_ROOT = Path("/data/depth/pseudo_depth_dataset")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
INPUT_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILOG_WEIGHT = 0.1
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-6
DEPTH_CLIP_MIN = 0.45
DEPTH_CLIP_MAX = 15.0
DEFAULT_ONNX_PATH = DATASET_ROOT / "student_v8_best.onnx"
DEFAULT_QUANT_ONNX_PATH = DATASET_ROOT / "student_v8_best.int8.onnx"
VAL_SPLITS = {"test", "validate"}


class PseudoDepthDataset(Dataset):
    def __init__(self, root: Path, rows):
        self.root = root
        self.rows = list(rows)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        img = Image.open(self.root / row["image_path"]).convert("RGB")
        img = self.to_tensor(img)

        depth = np.load(self.root / row["depth_path"]).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)

        depth = depth.clamp(min=DEPTH_CLIP_MIN, max=DEPTH_CLIP_MAX)
        depth = (depth - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN)

        return img, depth


def load_manifest_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_fixed_train_val_rows(rows):
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


class ExportableStudentDepthNetV8(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.softplus(self.model(x)) + 1e-4


def si_log_loss(pred, target, eps=1e-6):
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    g = torch.log(pred) - torch.log(target)
    return torch.sqrt((g ** 2).mean() - 0.5 * (g.mean() ** 2) + eps)


def configure_runtime():
    if DEVICE != "cuda":
        return

    # A40 is an Ampere GPU; TF32 and cuDNN autotuning are usually free wins
    # for fixed-shape conv-heavy training like this.
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


def train_one_epoch(model, loader, optimizer, use_channels_last: bool):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_silog = 0.0

    for images, depths in tqdm(loader, desc="train", leave=False):
        images = move_images(images, use_channels_last)
        depths = depths.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(images)
            pred = F.softplus(pred) + 1e-4

            l1 = F.l1_loss(pred, depths)
            silog = si_log_loss(pred, depths)
            loss = l1 + SILOG_WEIGHT * silog

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_l1 += l1.item() * batch_size
        total_silog += silog.item() * batch_size

    denom = len(loader.dataset)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "silog": total_silog / denom,
    }


@torch.no_grad()
def eval_one_epoch(model, loader, use_channels_last: bool):
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_silog = 0.0

    for images, depths in tqdm(loader, desc="val", leave=False):
        images = move_images(images, use_channels_last)
        depths = depths.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(images)
            pred = F.softplus(pred) + 1e-4
            l1 = F.l1_loss(pred, depths)
            silog = si_log_loss(pred, depths)
            loss = l1 + SILOG_WEIGHT * silog

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_l1 += l1.item() * batch_size
        total_silog += silog.item() * batch_size

    denom = len(loader.dataset)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "silog": total_silog / denom,
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--resume-best",
        action="store_true",
        default=True,
        help="Initialize training from student_v8_best.pt if it exists.",
    )
    parser.add_argument(
        "--no-resume-best",
        action="store_false",
        dest="resume_best",
        help="Start training from scratch even if student_v8_best.pt exists.",
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
        help="Training and validation batch size.",
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
    return parser.parse_args()


def maybe_load_resume_weights(model, resume_path: Path, resume_best: bool):
    if not resume_best or not resume_path.exists():
        return

    state_dict = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Resumed model weights from {resume_path}")


def export_onnx(model_path: Path, output_path: Path):
    export_model = StudentDepthNetV8()
    state_dict = torch.load(model_path, map_location="cpu")
    export_model.load_state_dict(state_dict)
    export_model.eval()

    wrapped = ExportableStudentDepthNetV8(export_model)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
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
    args = parse_args()
    configure_runtime()
    manifest_rows = load_manifest_rows(MANIFEST_PATH)
    train_rows, val_rows = build_fixed_train_val_rows(manifest_rows)
    train_ds = PseudoDepthDataset(DATASET_ROOT, train_rows)
    val_ds = PseudoDepthDataset(DATASET_ROOT, val_rows)
    print(
        "Dataset split | "
        f"train_rows={len(train_rows)} | "
        f"val_rows={len(val_rows)} | "
        "val_source=non_avi_test_validate"
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

    model = StudentDepthNetV8().to(DEVICE)
    if args.channels_last and DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    ckpt_best = DATASET_ROOT / "student_v8_best.pt"
    ckpt_last = DATASET_ROOT / "student_v8_last.pt"
    maybe_load_resume_weights(model, ckpt_best, args.resume_best)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    best_val = float("inf")

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
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_l1={val_metrics['l1']:.6f} | "
            f"val_silog={val_metrics['silog']:.6f}"
        )

        torch.save(model.state_dict(), ckpt_last)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), ckpt_best)
            print(f"Saved best model to {ckpt_best}")

    if args.save_onnx or args.save_quantized_onnx:
        export_onnx(ckpt_best, args.onnx_path)

    if args.save_quantized_onnx:
        export_quantized_onnx(args.onnx_path, args.quantized_onnx_path)


if __name__ == "__main__":
    main()
