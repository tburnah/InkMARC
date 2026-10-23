from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose

from models.student_depth_net_v6 import (
    StudentDepthNetV6,
    optimize_student_depth_net_v6_for_inference,
)


REPO_ROOT = Path(__file__).resolve().parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


DATASET_TARGET_SIZE = 320
TEACHER_INPUT_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

IMAGE_PATH = Path(__file__).resolve().parent / "PenHand320.jpg"
STUDENT_CHECKPOINT = Path("/data/depth/pseudo_depth_dataset_320/student_v6_edgefocus_best.pt")
TEACHER_CHECKPOINT = Path("/home/t/tburnah/inkmarc/depth/depth_anything_v2_vitb.pth")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
TEACHER_ENCODER = "vitb"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=IMAGE_PATH,
        help="Path to the RGB image to compare.",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=STUDENT_CHECKPOINT,
        help="Path to the trained V6 checkpoint.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=TEACHER_CHECKPOINT,
        help="Path to the Depth Anything V2 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for saved comparison images.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DATASET_TARGET_SIZE,
        help="Square size used for student inference and teacher output comparison.",
    )
    return parser.parse_args()


def remove_letterbox(img: np.ndarray, threshold: int = 12, min_fraction: float = 0.02) -> np.ndarray:
    gray = img.mean(axis=2)
    mask = gray > threshold

    row_fraction = mask.mean(axis=1)
    col_fraction = mask.mean(axis=0)

    ys = np.where(row_fraction > min_fraction)[0]
    xs = np.where(col_fraction > min_fraction)[0]

    if len(xs) == 0 or len(ys) == 0:
        return img

    y0, y1 = ys[0], ys[-1]
    x0, x1 = xs[0], xs[-1]
    return img[y0:y1 + 1, x0:x1 + 1]


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((size, size), Image.Resampling.LANCZOS))


def build_teacher_transform(input_size: int = TEACHER_INPUT_SIZE):
    return Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def colorize_map(values_2d: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    values_norm = (values_2d - min_val) / (max_val - min_val + 1e-8)
    values_norm = np.clip(values_norm, 0.0, 1.0)
    values_u8 = (values_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(values_u8, cv2.COLORMAP_INFERNO)


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


def preprocess_image(image_path: Path, target_size: int) -> np.ndarray:
    rgb = np.array(Image.open(image_path).convert("RGB"))
    rgb = remove_letterbox(rgb)
    rgb = center_crop_square(rgb)
    return resize_square(rgb, target_size)


def load_teacher(checkpoint_path: Path) -> DepthAnythingV2:
    model = DepthAnythingV2(**MODEL_CONFIGS[TEACHER_ENCODER])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_student(checkpoint_path: Path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model = StudentDepthNetV6().to(DEVICE)
    model.load_state_dict(state_dict)
    return optimize_student_depth_net_v6_for_inference(
        model,
        use_channels_last=(DEVICE == "cuda"),
    )


def run_teacher(model: DepthAnythingV2, rgb: np.ndarray, teacher_transform) -> np.ndarray:
    teacher_image = rgb.astype(np.float32) / 255.0
    teacher_tensor = teacher_transform({"image": teacher_image})["image"]
    teacher_tensor = torch.from_numpy(teacher_tensor).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            depth = model(teacher_tensor)
            depth = F.interpolate(
                depth[:, None],
                rgb.shape[:2],
                mode="bilinear",
                align_corners=True,
            )[:, 0]

    return depth.squeeze(0).detach().cpu().numpy()


def run_student(model, rgb: np.ndarray) -> np.ndarray:
    image_tensor = transforms.ToTensor()(Image.fromarray(rgb)).unsqueeze(0)
    if DEVICE == "cuda":
        image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)
    image_tensor = image_tensor.to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred_logits = model(image_tensor)
            pred = F.softplus(pred_logits) + 1e-4

    return pred.squeeze().detach().cpu().numpy()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")
    if not args.student_checkpoint.exists():
        raise FileNotFoundError(f"Missing student checkpoint: {args.student_checkpoint}")
    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {args.teacher_checkpoint}")

    rgb = preprocess_image(args.image, args.target_size)
    teacher_transform = build_teacher_transform()
    teacher_model = load_teacher(args.teacher_checkpoint)
    student_model = load_student(args.student_checkpoint)

    teacher_depth = run_teacher(teacher_model, rgb, teacher_transform)
    student_depth = run_student(student_model, rgb)

    teacher_min = float(teacher_depth.min())
    teacher_max = float(teacher_depth.max())
    if teacher_max - teacher_min < 1e-8:
        teacher_norm = np.zeros_like(teacher_depth, dtype=np.float32)
    else:
        teacher_norm = (teacher_depth - teacher_min) / (teacher_max - teacher_min)

    student_norm = np.clip(student_depth, 0.0, 1.0)
    diff_abs = np.abs(student_norm - teacher_norm)

    teacher_tensor = torch.from_numpy(teacher_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    student_tensor = torch.from_numpy(student_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    l1 = F.l1_loss(student_tensor, teacher_tensor).item()
    silog = si_log_loss(student_tensor, teacher_tensor).item()
    grad = gradient_loss(student_tensor, teacher_tensor).item()

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    teacher_viz = colorize_map(teacher_norm, 0.0, 1.0)
    student_viz = colorize_map(student_norm, 0.0, 1.0)
    diff_viz = colorize_map(diff_abs, 0.0, max(1e-6, float(diff_abs.max())))

    image_stem = args.image.stem
    cv2.imwrite(str(output_dir / f"{image_stem}_rgb.png"), rgb_bgr)
    cv2.imwrite(str(output_dir / f"{image_stem}_teacher_depth.png"), teacher_viz)
    cv2.imwrite(str(output_dir / f"{image_stem}_student_depth.png"), student_viz)
    cv2.imwrite(str(output_dir / f"{image_stem}_abs_diff.png"), diff_viz)

    comparison = np.hstack([rgb_bgr, teacher_viz, student_viz, diff_viz])
    cv2.imwrite(str(output_dir / f"{image_stem}_comparison.png"), comparison)

    print(f"image: {args.image}")
    print(f"student_checkpoint: {args.student_checkpoint}")
    print(f"teacher_checkpoint: {args.teacher_checkpoint}")
    print(f"target_size: {args.target_size}")
    print(f"teacher depth min/max: {teacher_depth.min():.6f} / {teacher_depth.max():.6f}")
    print(f"student depth min/max: {student_depth.min():.6f} / {student_depth.max():.6f}")
    print(f"normalized metrics | l1={l1:.6f} | silog={silog:.6f} | grad={grad:.6f}")
    print(f"saved: {output_dir / f'{image_stem}_rgb.png'}")
    print(f"saved: {output_dir / f'{image_stem}_teacher_depth.png'}")
    print(f"saved: {output_dir / f'{image_stem}_student_depth.png'}")
    print(f"saved: {output_dir / f'{image_stem}_abs_diff.png'}")
    print(f"saved: {output_dir / f'{image_stem}_comparison.png'}")


if __name__ == "__main__":
    main()
