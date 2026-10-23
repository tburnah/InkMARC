from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


REPO_ROOT = Path(__file__).resolve().parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


IMAGE_PATH = Path(__file__).resolve().parent / "PenHand.jpg"
TEACHER_CHECKPOINT = Path("/home/t/tburnah/inkmarc/depth/depth_anything_v2_vitb.pth")
TARGET_SIZE = 256
TEACHER_INPUT_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        help="Path to the RGB image to analyze.",
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
        help="Directory for saved visualization images.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help="Square size used before teacher inference and mask generation.",
    )
    parser.add_argument(
        "--edge-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold used to convert edge magnitude into a binary mask.",
    )
    parser.add_argument(
        "--dilate-radius",
        type=int,
        default=2,
        help="Radius for dilating the binary edge mask.",
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


def build_edge_magnitude(depth_norm: np.ndarray) -> np.ndarray:
    dx = np.abs(depth_norm[:, 1:] - depth_norm[:, :-1])
    dy = np.abs(depth_norm[1:, :] - depth_norm[:-1, :])

    edge_mag = np.zeros_like(depth_norm, dtype=np.float32)
    edge_mag[:, 1:] += dx
    edge_mag[:, :-1] += dx
    edge_mag[1:, :] += dy
    edge_mag[:-1, :] += dy
    return edge_mag


def build_edge_mask(edge_mag: np.ndarray, percentile: float, dilate_radius: int) -> tuple[np.ndarray, float]:
    threshold = float(np.percentile(edge_mag, percentile))
    mask = edge_mag >= threshold

    if dilate_radius > 0:
        kernel_size = 2 * dilate_radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return mask, threshold


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")
    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {args.teacher_checkpoint}")

    rgb = preprocess_image(args.image, args.target_size)
    teacher_transform = build_teacher_transform()
    teacher_model = load_teacher(args.teacher_checkpoint)
    teacher_depth = run_teacher(teacher_model, rgb, teacher_transform)

    depth_min = float(teacher_depth.min())
    depth_max = float(teacher_depth.max())
    depth_norm = (teacher_depth - depth_min) / (depth_max - depth_min + 1e-8)

    edge_mag = build_edge_magnitude(depth_norm)
    edge_mask, threshold = build_edge_mask(
        edge_mag,
        percentile=args.edge_percentile,
        dilate_radius=args.dilate_radius,
    )

    depth_viz = colorize_map(teacher_depth, depth_min, depth_max)
    edge_mag_viz = colorize_map(edge_mag, 0.0, max(1e-6, float(edge_mag.max())))
    mask_u8 = (edge_mask.astype(np.uint8) * 255)
    mask_viz = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = rgb_bgr.copy()
    overlay[edge_mask] = (0, 255, 255)
    overlay = cv2.addWeighted(rgb_bgr, 0.65, overlay, 0.35, 0.0)

    stem = args.image.stem
    cv2.imwrite(str(output_dir / f"{stem}_teacher_depth.png"), depth_viz)
    cv2.imwrite(str(output_dir / f"{stem}_edge_magnitude.png"), edge_mag_viz)
    cv2.imwrite(str(output_dir / f"{stem}_edge_mask.png"), mask_viz)
    cv2.imwrite(str(output_dir / f"{stem}_edge_overlay.png"), overlay)

    comparison = np.hstack([rgb_bgr, depth_viz, edge_mag_viz, overlay])
    cv2.imwrite(str(output_dir / f"{stem}_edge_comparison.png"), comparison)

    coverage = float(edge_mask.mean())
    print(f"image: {args.image}")
    print(f"preprocessed_shape: {rgb.shape}")
    print(f"edge_percentile: {args.edge_percentile}")
    print(f"dilate_radius: {args.dilate_radius}")
    print(f"edge_threshold: {threshold:.6f}")
    print(f"mask_coverage: {coverage:.4%}")
    print(f"saved: {output_dir / f'{stem}_teacher_depth.png'}")
    print(f"saved: {output_dir / f'{stem}_edge_magnitude.png'}")
    print(f"saved: {output_dir / f'{stem}_edge_mask.png'}")
    print(f"saved: {output_dir / f'{stem}_edge_overlay.png'}")
    print(f"saved: {output_dir / f'{stem}_edge_comparison.png'}")


if __name__ == "__main__":
    main()
