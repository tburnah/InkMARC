# train_student_v6_edgefocus.py
import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.student_depth_net_v6 import StudentDepthNetV6


DATASET_ROOT = Path("/data/depth/pseudo_depth_dataset_320")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-6
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILOG_WEIGHT = 0.1
GRADIENT_WEIGHT = 0.2
EDGE_L1_WEIGHT = 0.25
EDGE_GRAD_FACTOR = 4.0
EDGE_PERCENTILE = 90.0
EDGE_MIN_MAGNITUDE = 0.01
EDGE_DILATION_RADIUS = 2
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 6
DEPTH_CLIP_MIN = 0.45
DEPTH_CLIP_MAX = 15.0
POSITIVE_DEPTH_EPS = 1e-4
V6_BASE_BEST_CKPT = DATASET_ROOT / "student_v6_best.pt"
DEFAULT_ONNX_PATH = DATASET_ROOT / "student_v6_edgefocus_best.onnx"
DEFAULT_QUANT_ONNX_PATH = DATASET_ROOT / "student_v6_edgefocus_best.int8.onnx"
DEFAULT_QUANT_CALIB_BATCHES = 32
DEFAULT_QUANT_CALIB_BATCH_SIZE = 1
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

        # Fixed clipped normalization preserves a consistent scale across images.
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

        # Keep AVI-derived samples train-only so edge-focus fine-tuning is still
        # judged on the original non-AVI validation anchor.
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


class ExportableStudentDepthNetV6EdgeFocus(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.softplus(self.model(x)) + POSITIVE_DEPTH_EPS


class OnnxCalibrationDataReader:
    def __init__(self, loader: DataLoader, input_name: str, max_batches: int):
        self.loader = loader
        self.input_name = input_name
        self.max_batches = max(1, int(max_batches))
        self._iter: Optional[Iterator] = None
        self._seen = 0
        self.rewind()

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iter is None or self._seen >= self.max_batches:
            return None
        try:
            images, _ = next(self._iter)
        except StopIteration:
            return None

        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        image_np = images.detach().cpu().to(torch.float32).numpy()
        self._seen += 1
        if self._seen == 1 or self._seen == self.max_batches:
            print(f"[calib] prepared batch {self._seen}/{self.max_batches}")
        return {self.input_name: image_np}

    def rewind(self) -> None:
        self._iter = iter(self.loader)
        self._seen = 0


def positive_depth_from_logits(pred_logits):
    return F.softplus(pred_logits) + POSITIVE_DEPTH_EPS


def si_log_loss(pred, target, eps=1e-6):
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    g = torch.log(pred) - torch.log(target)
    return torch.sqrt((g ** 2).mean() - 0.5 * (g.mean() ** 2) + eps)


def gradient_map(depth):
    dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    edge_mag = torch.zeros_like(depth)
    edge_mag[:, :, :, 1:] += dx
    edge_mag[:, :, :, :-1] += dx
    edge_mag[:, :, 1:, :] += dy
    edge_mag[:, :, :-1, :] += dy
    return edge_mag


def build_edge_mask(target):
    # The edge mask isolates strong target-depth boundaries so the extra
    # supervision spends capacity near silhouettes and sharp geometry changes.
    edge_mag = gradient_map(target)
    flat = edge_mag.flatten(start_dim=2)
    threshold = torch.quantile(flat, EDGE_PERCENTILE / 100.0, dim=2, keepdim=True)
    threshold = threshold.view(target.size(0), target.size(1), 1, 1)
    threshold = torch.maximum(threshold, torch.full_like(threshold, EDGE_MIN_MAGNITUDE))

    edge_mask = edge_mag >= threshold
    if EDGE_DILATION_RADIUS > 0:
        kernel_size = 2 * EDGE_DILATION_RADIUS + 1
        edge_mask = F.max_pool2d(
            edge_mask.float(),
            kernel_size=kernel_size,
            stride=1,
            padding=EDGE_DILATION_RADIUS,
        ) > 0.5

    return edge_mask.float()


def masked_l1_loss(pred, target, mask):
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() + 1e-6)


def edge_focused_gradient_loss(pred, target, edge_mask):
    # This term makes the optimizer care more about matching depth transitions
    # where the target boundary structure is strongest.
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    edge_mask_x = torch.maximum(edge_mask[:, :, :, 1:], edge_mask[:, :, :, :-1])
    edge_mask_y = torch.maximum(edge_mask[:, :, 1:, :], edge_mask[:, :, :-1, :])

    diff_x = (pred_dx - target_dx).abs()
    diff_y = (pred_dy - target_dy).abs()

    weight_x = edge_mask_x * (1.0 + EDGE_GRAD_FACTOR * target_dx.abs())
    weight_y = edge_mask_y * (1.0 + EDGE_GRAD_FACTOR * target_dy.abs())

    loss_x = (weight_x * diff_x).sum() / (weight_x.sum() + 1e-6)
    loss_y = (weight_y * diff_y).sum() / (weight_y.sum() + 1e-6)
    return 0.5 * (loss_x + loss_y)


def full_gradient_metric(pred, target):
    # Log a full-image gradient metric separately so runs can still be compared
    # on global edge behavior without using it directly for optimization.
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)
    return 0.5 * (loss_x + loss_y)


def compute_losses(pred_logits, depths):
    pred_depth = positive_depth_from_logits(pred_logits)
    edge_mask = build_edge_mask(depths)

    l1 = F.l1_loss(pred_depth, depths)
    silog = si_log_loss(pred_depth, depths)
    edge_l1 = masked_l1_loss(pred_depth, depths, edge_mask)
    grad_focus = edge_focused_gradient_loss(pred_depth, depths, edge_mask)
    grad_full = full_gradient_metric(pred_depth, depths)
    edge_coverage = edge_mask.mean()

    loss = l1 + SILOG_WEIGHT * silog + GRADIENT_WEIGHT * grad_focus + EDGE_L1_WEIGHT * edge_l1

    return {
        "loss": loss,
        "l1": l1,
        "silog": silog,
        "edge_l1": edge_l1,
        "grad_focus": grad_focus,
        "grad_full": grad_full,
        "edge_coverage": edge_coverage,
    }


def train_one_epoch(model, loader, optimizer):
    model.train()
    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "silog": 0.0,
        "edge_l1": 0.0,
        "grad_focus": 0.0,
        "grad_full": 0.0,
        "edge_coverage": 0.0,
    }

    for images, depths in tqdm(loader, desc="train", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        depths = depths.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred_logits = model(images)
            metrics = compute_losses(pred_logits, depths)

        metrics["loss"].backward()
        optimizer.step()

        batch_size = images.size(0)
        for key in totals:
            totals[key] += metrics[key].item() * batch_size

    denom = len(loader.dataset)
    return {key: value / denom for key, value in totals.items()}


@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "silog": 0.0,
        "edge_l1": 0.0,
        "grad_focus": 0.0,
        "grad_full": 0.0,
        "edge_coverage": 0.0,
    }

    for images, depths in tqdm(loader, desc="val", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        depths = depths.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred_logits = model(images)
            metrics = compute_losses(pred_logits, depths)

        batch_size = images.size(0)
        for key in totals:
            totals[key] += metrics[key].item() * batch_size

    denom = len(loader.dataset)
    return {key: value / denom for key, value in totals.items()}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--resume-best",
        action="store_true",
        default=True,
        help="Resume training from student_v6_edgefocus_best.pt if it exists.",
    )
    parser.add_argument(
        "--no-resume-best",
        action="store_false",
        dest="resume_best",
        help="Do not resume from student_v6_edgefocus_best.pt.",
    )
    parser.add_argument(
        "--init-from-v6-best",
        action="store_true",
        default=True,
        help="If no V6 edge-focus checkpoint exists, initialize from student_v6_best.pt when available.",
    )
    parser.add_argument(
        "--no-init-from-v6-best",
        action="store_false",
        dest="init_from_v6_best",
        help="Start from random initialization when no compatible V6 checkpoint exists.",
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
        help="Export a static QDQ int8 ONNX model after training.",
    )
    parser.add_argument(
        "--quantized-onnx-path",
        type=Path,
        default=DEFAULT_QUANT_ONNX_PATH,
        help="Path for the quantized ONNX model.",
    )
    parser.add_argument(
        "--quant-calib-batches",
        type=int,
        default=DEFAULT_QUANT_CALIB_BATCHES,
        help="Calibration batch count for static QDQ ONNX quantization.",
    )
    parser.add_argument(
        "--quant-calib-batch-size",
        type=int,
        default=DEFAULT_QUANT_CALIB_BATCH_SIZE,
        help="Calibration batch size for static QDQ ONNX quantization.",
    )
    return parser.parse_args()


def maybe_load_initial_weights(
    model,
    resume_path: Path,
    v6_best_path: Path,
    resume_best: bool,
    init_from_v6_best: bool,
):
    if resume_best and resume_path.exists():
        state_dict = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {resume_path}")
        return "resume"

    if init_from_v6_best and v6_best_path.exists():
        state_dict = torch.load(v6_best_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Initialized model weights from compatible V6 checkpoint {v6_best_path}")
        return "v6_best"

    return "scratch"


def export_onnx(model_path: Path, output_path: Path):
    export_model = StudentDepthNetV6()
    state_dict = torch.load(model_path, map_location="cpu")
    export_model.load_state_dict(state_dict)
    export_model.eval()

    wrapped = ExportableStudentDepthNetV6EdgeFocus(export_model)
    dummy = torch.randn(1, 3, 320, 320, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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


def build_quant_calibration_loader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def inspect_onnx_op_types(path: Path):
    try:
        import onnx
    except ImportError:
        print(f"ONNX op inspection skipped (onnx not installed): {path}")
        return

    model = onnx.load(str(path))
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1

    keys = [
        "Conv",
        "ConvInteger",
        "QuantizeLinear",
        "DequantizeLinear",
        "QLinearConv",
        "MatMul",
        "Gemm",
    ]
    summary = ", ".join(f"{key}={counts.get(key, 0)}" for key in keys)
    print(f"ONNX op counts ({path}): {summary}")


def export_quantized_onnx(
    onnx_path: Path,
    quantized_path: Path,
    calibration_dataset: Dataset,
    calib_batches: int,
    calib_batch_size: int,
):
    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quant_pre_process,
            quantize_static,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Quantized ONNX export requires onnxruntime to be installed in the training environment."
        ) from exc

    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_onnx = quantized_path.with_name(f"{quantized_path.stem}.preprocessed.onnx")
    quant_input = onnx_path
    try:
        quant_pre_process(
            input_model_path=str(onnx_path),
            output_model_path=str(preprocessed_onnx),
        )
        quant_input = preprocessed_onnx
        print(f"Preprocessed ONNX for static quantization: {preprocessed_onnx}")
    except Exception as exc:
        print(f"ONNX pre-processing skipped; quantizing source model directly. error={exc}")

    calib_loader = build_quant_calibration_loader(calibration_dataset, calib_batch_size)
    calib_reader = OnnxCalibrationDataReader(
        loader=calib_loader,
        input_name="image",
        max_batches=calib_batches,
    )

    quantize_static(
        model_input=str(quant_input),
        model_output=str(quantized_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    print(f"Exported static QDQ int8 ONNX model to {quantized_path}")
    inspect_onnx_op_types(quantized_path)


def main():
    args = parse_args()
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

    model = StudentDepthNetV6().to(DEVICE)
    ckpt_best = DATASET_ROOT / "student_v6_edgefocus_best.pt"
    ckpt_last = DATASET_ROOT / "student_v6_edgefocus_last.pt"
    init_source = maybe_load_initial_weights(
        model,
        ckpt_best,
        V6_BASE_BEST_CKPT,
        args.resume_best,
        args.init_from_v6_best,
    )

    baseline_metrics = eval_one_epoch(model, val_loader)
    print(
        "Initial val | "
        f"source={init_source} | "
        f"val_loss={baseline_metrics['loss']:.6f} | "
        f"val_l1={baseline_metrics['l1']:.6f} | "
        f"val_silog={baseline_metrics['silog']:.6f} | "
        f"val_edge_l1={baseline_metrics['edge_l1']:.6f} | "
        f"val_grad_focus={baseline_metrics['grad_focus']:.6f} | "
        f"val_grad_full={baseline_metrics['grad_full']:.6f} | "
        f"val_edge_cov={baseline_metrics['edge_coverage']:.4f}"
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
    epochs_without_improvement = 0

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
            f"train_edge_l1={train_metrics['edge_l1']:.6f} | "
            f"train_grad_focus={train_metrics['grad_focus']:.6f} | "
            f"train_grad_full={train_metrics['grad_full']:.6f} | "
            f"train_edge_cov={train_metrics['edge_coverage']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_l1={val_metrics['l1']:.6f} | "
            f"val_silog={val_metrics['silog']:.6f} | "
            f"val_edge_l1={val_metrics['edge_l1']:.6f} | "
            f"val_grad_focus={val_metrics['grad_focus']:.6f} | "
            f"val_grad_full={val_metrics['grad_full']:.6f} | "
            f"val_edge_cov={val_metrics['edge_coverage']:.4f}"
        )

        torch.save(model.state_dict(), ckpt_last)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), ckpt_best)
            epochs_without_improvement = 0
            print(f"Saved best model to {ckpt_best}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping | "
                f"best_val={best_val:.6f} | "
                f"patience={EARLY_STOPPING_PATIENCE}"
            )
            break

    if args.save_onnx or args.save_quantized_onnx:
        export_onnx(ckpt_best, args.onnx_path)

    if args.save_quantized_onnx:
        export_quantized_onnx(
            args.onnx_path,
            args.quantized_onnx_path,
            calibration_dataset=train_ds,
            calib_batches=args.quant_calib_batches,
            calib_batch_size=args.quant_calib_batch_size,
        )


if __name__ == "__main__":
    main()
