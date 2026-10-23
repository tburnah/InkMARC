# train_student_v2.py
import csv
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from models.student_depth_net_v2 import StudentDepthNetV2


DATASET_ROOT = Path("/data/depth/pseudo_depth_dataset")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILOG_WEIGHT = 0.1
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-6
DEPTH_CLIP_MIN = 0.45
DEPTH_CLIP_MAX = 15.0
DEFAULT_ONNX_PATH = DATASET_ROOT / "student_v2_best.onnx"
DEFAULT_QUANT_ONNX_PATH = DATASET_ROOT / "student_v2_best.int8.onnx"


class PseudoDepthDataset(Dataset):
    def __init__(self, root: Path, manifest_path: Path):
        self.root = root
        self.rows = []
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        img = Image.open(self.root / row["image_path"]).convert("RGB")
        img = self.to_tensor(img)  # [3,H,W], 0..1

        depth = np.load(self.root / row["depth_path"]).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1,H,W]

        # Fixed clipped normalization preserves a consistent scale across images.
        depth = depth.clamp(min=DEPTH_CLIP_MIN, max=DEPTH_CLIP_MAX)
        depth = (depth - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN)

        return img, depth


class ExportableStudentDepthNetV2(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.softplus(self.model(x)) + 1e-4


def si_log_loss(pred, target, eps=1e-6):
    # simple scale-invariant-ish loss variant
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    g = torch.log(pred) - torch.log(target)
    return torch.sqrt((g ** 2).mean() - 0.5 * (g.mean() ** 2) + eps)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_silog = 0.0

    for images, depths in tqdm(loader, desc="train", leave=False):
        images = images.to(DEVICE, non_blocking=True)
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

        total_loss += loss.item() * images.size(0)
        total_l1 += l1.item() * images.size(0)
        total_silog += silog.item() * images.size(0)

    denom = len(loader.dataset)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "silog": total_silog / denom,
    }


@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_silog = 0.0

    for images, depths in tqdm(loader, desc="val", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        depths = depths.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(images)
            pred = F.softplus(pred) + 1e-4
            l1 = F.l1_loss(pred, depths)
            silog = si_log_loss(pred, depths)
            loss = l1 + SILOG_WEIGHT * silog

        total_loss += loss.item() * images.size(0)
        total_l1 += l1.item() * images.size(0)
        total_silog += silog.item() * images.size(0)

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
        help="Initialize training from student_v2_best.pt if it exists.",
    )
    parser.add_argument(
        "--no-resume-best",
        action="store_false",
        dest="resume_best",
        help="Start training from scratch even if student_v2_best.pt exists.",
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
    return parser.parse_args()


def maybe_load_resume_weights(model, resume_path: Path, resume_best: bool):
    if not resume_best or not resume_path.exists():
        return

    state_dict = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Resumed model weights from {resume_path}")


def export_onnx(model_path: Path, output_path: Path):
    export_model = StudentDepthNetV2()
    state_dict = torch.load(model_path, map_location="cpu")
    export_model.load_state_dict(state_dict)
    export_model.eval()

    wrapped = ExportableStudentDepthNetV2(export_model)
    dummy = torch.randn(1, 3, 256, 256, dtype=torch.float32)

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
    dataset = PseudoDepthDataset(DATASET_ROOT, MANIFEST_PATH)

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = StudentDepthNetV2().to(DEVICE)
    ckpt_best = DATASET_ROOT / "student_v2_best.pt"
    ckpt_last = DATASET_ROOT / "student_v2_last.pt"
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
        train_metrics = train_one_epoch(model, train_loader, optimizer)
        val_metrics = eval_one_epoch(model, val_loader)
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
