from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.student_depth_net_v6 import (
    StudentDepthNetV6,
    optimize_student_depth_net_v6_for_inference,
)


DATASET_ROOT = Path("/data/depth/pseudo_depth_dataset_320")
CHECKPOINT_PATH = DATASET_ROOT / "student_v6_edgefocus_best.pt"
DEPTH_CLIP_MIN = 0.45
DEPTH_CLIP_MAX = 15.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--sample",
        type=str,
        default="test_001000",
        help="Sample stem matching images/<sample>.png and depth/<sample>.npy.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to the trained V6 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for saved comparison images.",
    )
    return parser.parse_args()


def si_log_loss(pred, target, eps=1e-6):
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    g = torch.log(pred) - torch.log(target)
    return torch.sqrt((g ** 2).mean() - 0.5 * (g.mean() ** 2) + eps)


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)
    return 0.5 * (loss_x + loss_y)


def colorize_depth(depth_2d: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    depth_norm = (depth_2d - min_val) / (max_val - min_val + 1e-8)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)


def resolve_sample_paths(sample: str):
    candidates = [sample]
    if not sample.startswith("test_"):
        candidates.append(f"test_{sample}")

    for stem in candidates:
        image_path = DATASET_ROOT / "images" / f"{stem}.png"
        depth_path = DATASET_ROOT / "depth" / f"{stem}.npy"
        if image_path.exists() and depth_path.exists():
            return stem, image_path, depth_path

    raise FileNotFoundError(
        f"Missing sample files for '{sample}'. Tried: "
        + ", ".join(candidates)
    )


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_stem, image_path, depth_path = resolve_sample_paths(args.sample)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    if DEVICE == "cuda":
        image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)
    image_tensor = image_tensor.to(DEVICE, non_blocking=True)

    target_raw = np.load(depth_path).astype(np.float32)
    target_clipped = np.clip(target_raw, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
    target_norm = (target_clipped - DEPTH_CLIP_MIN) / (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN)
    target_tensor = torch.from_numpy(target_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    model = StudentDepthNetV6().to(DEVICE)
    state_dict = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    optimize_student_depth_net_v6_for_inference(
        model,
        use_channels_last=(DEVICE == "cuda"),
    )

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred_logits = model(image_tensor)
            pred_norm = F.softplus(pred_logits) + 1e-4

    l1 = F.l1_loss(pred_norm, target_tensor).item()
    silog = si_log_loss(pred_norm, target_tensor).item()
    grad = gradient_loss(pred_norm, target_tensor).item()

    pred_norm_np = pred_norm.squeeze().detach().cpu().numpy()
    pred_raw = pred_norm_np * (DEPTH_CLIP_MAX - DEPTH_CLIP_MIN) + DEPTH_CLIP_MIN

    target_viz = colorize_depth(target_clipped, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
    pred_viz = colorize_depth(pred_raw, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)

    diff_abs = np.abs(pred_norm_np - target_norm)
    diff_viz = colorize_depth(diff_abs, 0.0, max(1e-6, float(diff_abs.max())))

    cv2.imwrite(str(output_dir / f"{sample_stem}_target_depth.png"), target_viz)
    cv2.imwrite(str(output_dir / f"{sample_stem}_pred_depth.png"), pred_viz)
    cv2.imwrite(str(output_dir / f"{sample_stem}_abs_diff.png"), diff_viz)

    side_by_side = np.hstack([target_viz, pred_viz, diff_viz])
    cv2.imwrite(str(output_dir / f"{sample_stem}_comparison.png"), side_by_side)

    print(f"sample: {sample_stem}")
    print(f"image: {image_path}")
    print(f"depth: {depth_path}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"target_raw shape: {target_raw.shape}")
    print(f"target_raw min/max/mean: {target_raw.min():.6f} / {target_raw.max():.6f} / {target_raw.mean():.6f}")
    print(f"pred_raw min/max/mean: {pred_raw.min():.6f} / {pred_raw.max():.6f} / {pred_raw.mean():.6f}")
    print(f"normalized metrics | l1={l1:.6f} | silog={silog:.6f} | grad={grad:.6f}")
    print(f"saved: {output_dir / f'{sample_stem}_target_depth.png'}")
    print(f"saved: {output_dir / f'{sample_stem}_pred_depth.png'}")
    print(f"saved: {output_dir / f'{sample_stem}_abs_diff.png'}")
    print(f"saved: {output_dir / f'{sample_stem}_comparison.png'}")


if __name__ == "__main__":
    main()
