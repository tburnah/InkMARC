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

from models.student_depth_net_v9_temporal import StudentDepthNetV9TemporalLite


REPO_ROOT = Path(__file__).resolve().parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


TARGET_SIZE = 256
TEACHER_INPUT_SIZE = 518
SEQUENCE_LENGTH = 4
TEMPORAL_FRAME_STRIDE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

VIDEO_PATH = Path("/data/LocationData/participant_31_T6a_1765950891032_Black.avi")
STUDENT_CHECKPOINT = Path("/data/depth/pseudo_depth_dataset/student_v9_temporal_best.pt")
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
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Path to the AVI video to sample.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="0-based target frame index if --time-sec is not given.",
    )
    parser.add_argument(
        "--time-sec",
        type=float,
        default=None,
        help="Optional target timestamp in seconds. Overrides --frame-index.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Number of temporal frames to feed the V9 model, ending at the target frame.",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=TEMPORAL_FRAME_STRIDE,
        help="Raw-video frame gap between temporal context frames. Default matches dataset AVI sampling.",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=STUDENT_CHECKPOINT,
        help="Path to the trained V9 temporal checkpoint.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=TEACHER_CHECKPOINT,
        help="Path to the Depth Anything V2 checkpoint.",
    )
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help="Run only the teacher and save teacher outputs without student comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for saved images.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help="Square size used for student inference and teacher output comparison.",
    )
    return parser.parse_args()


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


def resize_short_side_then_center_crop(img: np.ndarray, target_size: int) -> np.ndarray:
    h, w, _ = img.shape
    short_side = min(h, w)
    scale = target_size / float(short_side)
    new_w = max(target_size, int(round(w * scale)))
    new_h = max(target_size, int(round(h * scale)))

    resized = np.array(
        Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS)
    )

    y0 = (new_h - target_size) // 2
    x0 = (new_w - target_size) // 2
    return resized[y0:y0 + target_size, x0:x0 + target_size]


def resolve_target_frame_index(
    capture: cv2.VideoCapture,
    frame_index: int | None = None,
    time_sec: float | None = None,
):
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    if time_sec is not None:
        if fps <= 0.0:
            raise RuntimeError("Video FPS is unavailable, so --time-sec cannot be resolved.")
        resolved_index = int(round(time_sec * fps))
    else:
        resolved_index = 0 if frame_index is None else frame_index

    if resolved_index < 0 or (frame_count > 0 and resolved_index >= frame_count):
        raise IndexError(
            f"Frame index {resolved_index} is out of range for video with {frame_count} frames."
        )

    return resolved_index, fps, frame_count


def build_sequence_indices(target_frame_index: int, sequence_length: int, temporal_stride: int):
    if sequence_length < 1:
        raise ValueError("--sequence-length must be at least 1.")
    if temporal_stride < 1:
        raise ValueError("--temporal-stride must be at least 1.")

    start_index = target_frame_index - (sequence_length - 1) * temporal_stride
    if start_index < 0:
        raise ValueError(
            "Not enough earlier frames to build the requested temporal context. "
            f"target_frame={target_frame_index}, sequence_length={sequence_length}, "
            f"temporal_stride={temporal_stride}"
        )

    return [
        start_index + step_idx * temporal_stride
        for step_idx in range(sequence_length)
    ]


def extract_frame_sequence(
    video_path: Path,
    frame_indices: list[int],
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames_rgb = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
            frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    return frames_rgb


def load_teacher(checkpoint_path: Path) -> DepthAnythingV2:
    model = DepthAnythingV2(**MODEL_CONFIGS[TEACHER_ENCODER])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_student(checkpoint_path: Path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model = StudentDepthNetV9TemporalLite().to(DEVICE)
    model.load_state_dict(state_dict)
    if DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


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


def rgb_to_student_tensor(rgb: np.ndarray) -> torch.Tensor:
    image_tensor = transforms.ToTensor()(Image.fromarray(rgb)).unsqueeze(0)
    if DEVICE == "cuda":
        image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)
    return image_tensor.to(DEVICE, non_blocking=True)


def run_student_temporal(model, rgbs: list[np.ndarray]) -> np.ndarray:
    state = None
    pred = None

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            for rgb in rgbs:
                image_tensor = rgb_to_student_tensor(rgb)
                pred_logits, state = model(image_tensor, state, return_state=True)
                pred = F.softplus(pred_logits) + 1e-4

    if pred is None:
        raise RuntimeError("No frames were provided for temporal inference.")

    return pred.squeeze().detach().cpu().numpy()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise FileNotFoundError(f"Missing video: {args.video}")
    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {args.teacher_checkpoint}")
    if not args.teacher_only and not args.student_checkpoint.exists():
        raise FileNotFoundError(f"Missing student checkpoint: {args.student_checkpoint}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    try:
        resolved_index, fps, frame_count = resolve_target_frame_index(
            capture,
            frame_index=args.frame_index,
            time_sec=args.time_sec,
        )
    finally:
        capture.release()

    sequence_indices = build_sequence_indices(
        resolved_index,
        sequence_length=args.sequence_length,
        temporal_stride=args.temporal_stride,
    )
    frames_rgb = extract_frame_sequence(args.video, sequence_indices)
    processed_rgbs = [
        resize_short_side_then_center_crop(frame_rgb, args.target_size)
        for frame_rgb in frames_rgb
    ]
    processed_rgb = processed_rgbs[-1]
    raw_target_rgb = frames_rgb[-1]

    teacher_transform = build_teacher_transform()
    teacher_model = load_teacher(args.teacher_checkpoint)
    teacher_depth = run_teacher(teacher_model, processed_rgb, teacher_transform)

    teacher_min = float(teacher_depth.min())
    teacher_max = float(teacher_depth.max())
    if teacher_max - teacher_min < 1e-8:
        teacher_norm = np.zeros_like(teacher_depth, dtype=np.float32)
    else:
        teacher_norm = (teacher_depth - teacher_min) / (teacher_max - teacher_min)

    frame_tag = f"{args.video.stem}_frame_{resolved_index:06d}"
    raw_bgr = cv2.cvtColor(raw_target_rgb, cv2.COLOR_RGB2BGR)
    processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
    teacher_viz = colorize_map(teacher_norm, 0.0, 1.0)
    context_strip = np.hstack([
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for rgb in processed_rgbs
    ])

    cv2.imwrite(str(output_dir / f"{frame_tag}_raw.png"), raw_bgr)
    cv2.imwrite(str(output_dir / f"{frame_tag}_rgb.png"), processed_bgr)
    cv2.imwrite(str(output_dir / f"{frame_tag}_teacher_depth.png"), teacher_viz)
    cv2.imwrite(str(output_dir / f"{frame_tag}_temporal_context.png"), context_strip)

    print(f"video: {args.video}")
    print(f"frame_index: {resolved_index}")
    print(f"fps: {fps:.6f}")
    print(f"frame_count: {frame_count}")
    print(f"sequence_length: {args.sequence_length}")
    print(f"temporal_stride: {args.temporal_stride}")
    print(f"sequence_frame_indices: {sequence_indices}")
    print(f"teacher_checkpoint: {args.teacher_checkpoint}")
    print(f"target_size: {args.target_size}")
    print(f"teacher depth min/max: {teacher_depth.min():.6f} / {teacher_depth.max():.6f}")
    print(f"saved: {output_dir / f'{frame_tag}_raw.png'}")
    print(f"saved: {output_dir / f'{frame_tag}_rgb.png'}")
    print(f"saved: {output_dir / f'{frame_tag}_teacher_depth.png'}")
    print(f"saved: {output_dir / f'{frame_tag}_temporal_context.png'}")

    if args.teacher_only:
        teacher_panel = np.hstack([processed_bgr, teacher_viz])
        cv2.imwrite(str(output_dir / f"{frame_tag}_teacher_comparison.png"), teacher_panel)
        print(f"saved: {output_dir / f'{frame_tag}_teacher_comparison.png'}")
        return

    student_model = load_student(args.student_checkpoint)
    student_depth = run_student_temporal(student_model, processed_rgbs)
    student_norm = np.clip(student_depth, 0.0, 1.0)
    diff_abs = np.abs(student_norm - teacher_norm)

    teacher_tensor = torch.from_numpy(teacher_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    student_tensor = torch.from_numpy(student_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    l1 = F.l1_loss(student_tensor, teacher_tensor).item()
    silog = si_log_loss(student_tensor, teacher_tensor).item()
    grad = gradient_loss(student_tensor, teacher_tensor).item()

    student_viz = colorize_map(student_norm, 0.0, 1.0)
    diff_viz = colorize_map(diff_abs, 0.0, max(1e-6, float(diff_abs.max())))

    cv2.imwrite(str(output_dir / f"{frame_tag}_student_depth.png"), student_viz)
    cv2.imwrite(str(output_dir / f"{frame_tag}_abs_diff.png"), diff_viz)

    comparison = np.hstack([processed_bgr, teacher_viz, student_viz, diff_viz])
    cv2.imwrite(str(output_dir / f"{frame_tag}_comparison.png"), comparison)

    print(f"student_checkpoint: {args.student_checkpoint}")
    print(f"student depth min/max: {student_depth.min():.6f} / {student_depth.max():.6f}")
    print(f"normalized metrics | l1={l1:.6f} | silog={silog:.6f} | grad={grad:.6f}")
    print(f"saved: {output_dir / f'{frame_tag}_student_depth.png'}")
    print(f"saved: {output_dir / f'{frame_tag}_abs_diff.png'}")
    print(f"saved: {output_dir / f'{frame_tag}_comparison.png'}")


if __name__ == "__main__":
    main()
