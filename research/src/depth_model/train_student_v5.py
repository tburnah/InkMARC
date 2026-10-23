# train_student_v5.py
import csv
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from models.student_depth_net_v5 import StudentDepthNetV5

DATASET_ROOT = Path("/data/depth/pseudo_depth_dataset_320")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILOG_WEIGHT = 0.05
GRADIENT_WEIGHT = 0.5
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6
DEPTH_CLIP_MIN = 0.45
DEPTH_CLIP_MAX = 15.0
ORIGINAL_BEST_CKPT = DATASET_ROOT / "student_best.pt"
DEFAULT_ONNX_PATH = DATASET_ROOT / "student_v5_best.onnx"
DEFAULT_QUANT_ONNX_PATH = DATASET_ROOT / "student_v5_best.int8.onnx"


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
        img = self.to_tensor(img)

        depth = np.load(self.root / row["depth_path"]).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)

        # Fixed clipped normalization preserves a consistent scale across images.
        depth = depth.clamp(min=DEPTH_CLIP_MIN, max=DEPTH_CLIP_MAX)
        depth = (depth - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN)

        return img, depth


class ExportableStudentDepthNetV5(torch.nn.Module):
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


def gradient_loss(pred, target):
    # Encourage sharper depth discontinuities around silhouettes and tool edges
    # without changing inference behavior or adding an auxiliary head.
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)
    return 0.5 * (loss_x + loss_y)


def compute_losses(pred_logits, depths):
    # The model still emits a single raw depth map. Losses operate on positive
    # depth values after softplus so V5 stays aligned with earlier exports.
    pred_depth = F.softplus(pred_logits) + 1e-4

    l1 = F.l1_loss(pred_depth, depths)
    silog = si_log_loss(pred_depth, depths)
    grad = gradient_loss(pred_depth, depths)
    loss = l1 + SILOG_WEIGHT * silog + GRADIENT_WEIGHT * grad

    return loss, l1, silog, grad


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_silog = 0.0
    total_grad = 0.0

    for images, depths in tqdm(loader, desc="train", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        depths = depths.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred_logits = model(images)
            loss, l1, silog, grad = compute_losses(pred_logits, depths)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_l1 += l1.item() * batch_size
        total_silog += silog.item() * batch_size
        total_grad += grad.item() * batch_size

    denom = len(loader.dataset)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "silog": total_silog / denom,
        "grad": total_grad / denom,
    }


@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_silog = 0.0
    total_grad = 0.0

    for images, depths in tqdm(loader, desc="val", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        depths = depths.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred_logits = model(images)
            loss, l1, silog, grad = compute_losses(pred_logits, depths)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_l1 += l1.item() * batch_size
        total_silog += silog.item() * batch_size
        total_grad += grad.item() * batch_size

    denom = len(loader.dataset)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "silog": total_silog / denom,
        "grad": total_grad / denom,
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--resume-best",
        action="store_true",
        default=True,
        help="Resume training from student_v5_best.pt if it exists.",
    )
    parser.add_argument(
        "--no-resume-best",
        action="store_false",
        dest="resume_best",
        help="Do not resume from student_v5_best.pt.",
    )
    parser.add_argument(
        "--init-from-original-best",
        action="store_true",
        default=True,
        help="If no V5 checkpoint exists, initialize from student_best.pt when available.",
    )
    parser.add_argument(
        "--no-init-from-original-best",
        action="store_false",
        dest="init_from_original_best",
        help="Start from random initialization when no V5 checkpoint exists.",
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


def maybe_load_initial_weights(
    model,
    v5_resume_path: Path,
    original_best_path: Path,
    resume_best: bool,
    init_from_original_best: bool,
):
    if resume_best and v5_resume_path.exists():
        state_dict = torch.load(v5_resume_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {v5_resume_path}")
        return

    if init_from_original_best and original_best_path.exists():
        state_dict = torch.load(original_best_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Initialized model weights from original best checkpoint {original_best_path}")


def export_onnx(model_path: Path, output_path: Path):
    export_model = StudentDepthNetV5()
    state_dict = torch.load(model_path, map_location="cpu")
    export_model.load_state_dict(state_dict)
    export_model.eval()

    wrapped = ExportableStudentDepthNetV5(export_model)
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

    model = StudentDepthNetV5().to(DEVICE)
    ckpt_best = DATASET_ROOT / "student_v5_best.pt"
    ckpt_last = DATASET_ROOT / "student_v5_last.pt"
    maybe_load_initial_weights(
        model,
        ckpt_best,
        ORIGINAL_BEST_CKPT,
        args.resume_best,
        args.init_from_original_best,
    )

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
            f"train_grad={train_metrics['grad']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_l1={val_metrics['l1']:.6f} | "
            f"val_silog={val_metrics['silog']:.6f} | "
            f"val_grad={val_metrics['grad']:.6f}"
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
