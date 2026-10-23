import csv
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm
import sys
from torchvision.transforms import Compose

REPO_ROOT = Path(__file__).resolve().parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

try:
    # TensorFlow is only used for TFRecord reading in this script. Keep it off
    # the GPU so PyTorch can use the full device for teacher inference.
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


# ----------------------------
# Config
# ----------------------------
TFRECORD_SOURCES = [
    ("train", "/data/TouchData/train_split.tfrecord"),
    ("test", "/data/TouchData/test_split.tfrecord"),
    ("validate", "/data/TouchData/validate_split.tfrecord"),
]
CHECKPOINT_PATH = "/home/t/tburnah/inkmarc/depth/depth_anything_v2_vitb.pth"
DEFAULT_OUTPUT_PARENT = Path("/data/depth")
DEFAULT_DATASET_PREFIX = "pseudo_depth_dataset"
DEFAULT_TARGET_SIZE = 256
DEFAULT_TEACHER_INPUT_SIZE = 518
DEFAULT_BATCH_SIZE = 8
DEFAULT_CPU_WORKERS = 8
DEFAULT_AVI_ROOT = Path("/data/LocationData")
DEFAULT_AVI_FRAME_STRIDE = 10
SAVE_VIS = True
MAX_RECORDS = None   # set e.g. 500 for testing


# ----------------------------
# Depth Anything V2 config
# ----------------------------
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
ENCODER = "vitb"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Square size to save teacher RGB/depth pairs at.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory to write the dataset to. Defaults to "
            "/data/depth/pseudo_depth_dataset_<target_size>."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of preprocessed samples to batch for teacher inference.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="Worker threads for CPU preprocessing and saving.",
    )
    parser.add_argument(
        "--skip-tfrecords",
        action="store_true",
        help="Skip the TFRecord sources and do not regenerate those samples.",
    )
    parser.add_argument(
        "--include-black-avi",
        action="store_true",
        help="Append frames from *_Black.avi videos under --avi-root.",
    )
    parser.add_argument(
        "--avi-root",
        type=Path,
        default=DEFAULT_AVI_ROOT,
        help="Directory containing AVI videos to sample from.",
    )
    parser.add_argument(
        "--avi-frame-stride",
        type=int,
        default=DEFAULT_AVI_FRAME_STRIDE,
        help="Keep every Nth frame from *_Black.avi videos.",
    )
    parser.add_argument(
        "--avi-max-frames-per-video",
        type=int,
        default=None,
        help="Optional cap on saved frames per AVI after stride sampling.",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append new samples to an existing manifest instead of rewriting it.",
    )
    return parser.parse_args()


def resolve_output_root(target_size: int, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root
    return DEFAULT_OUTPUT_PARENT / f"{DEFAULT_DATASET_PREFIX}_{target_size}"


def ensure_dirs(output_root: Path) -> None:
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "depth").mkdir(parents=True, exist_ok=True)
    (output_root / "depth_vis").mkdir(parents=True, exist_ok=True)


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


def resize_short_side_then_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w, _ = img.shape
    short_side = min(h, w)
    scale = size / float(short_side)
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))

    resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS))
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return resized[y0:y0 + size, x0:x0 + size]


def normalize_depth_to_u8(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    dmin = float(depth.min())
    dmax = float(depth.max())
    if dmax - dmin < 1e-8:
        return np.zeros_like(depth, dtype=np.uint8)
    depth_norm = (depth - dmin) / (dmax - dmin)
    return (depth_norm * 255.0).clip(0, 255).astype(np.uint8)


def build_teacher_transform(input_size: int = DEFAULT_TEACHER_INPUT_SIZE):
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


def preprocess_from_tfrecord_bytes(image_bytes: bytes, target_size: int) -> np.ndarray:
    # stored as 448x448x3 raw bytes, likely BGR
    img = np.frombuffer(image_bytes, dtype=np.uint8).reshape(448, 448, 3)
    img = img[:, :, ::-1]  # BGR -> RGB
    img = remove_letterbox(img)
    img = center_crop_square(img)
    img = resize_square(img, target_size)
    return img


def preprocess_from_video_frame(frame_bgr: np.ndarray, target_size: int) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return resize_short_side_then_center_crop(frame_rgb, target_size)


def load_teacher() -> DepthAnythingV2:
    model = DepthAnythingV2(**MODEL_CONFIGS[ENCODER])
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    return model


def iter_examples(path: str):
    dataset = tf.data.TFRecordDataset(path)
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        yield example


def iter_black_avi_paths(root: Path):
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() == ".avi" and path.name.lower().endswith("_black.avi"):
            yield path


def chunked(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def preprocess_example(item, target_size: int, teacher_transform):
    split_idx, example = item
    image_bytes = example.features.feature["image"].bytes_list.value[0]
    label = int(example.features.feature["label"].int64_list.value[0])
    sample_id = f"{split_idx['split_name']}_{split_idx['index']:06d}"

    rgb = preprocess_from_tfrecord_bytes(image_bytes, target_size)
    bgr = rgb[:, :, ::-1]
    teacher_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    teacher_tensor = teacher_transform({"image": teacher_image})["image"]

    return {
        "sample_id": sample_id,
        "split_name": split_idx["split_name"],
        "label": label,
        "rgb": rgb,
        "teacher_tensor": torch.from_numpy(teacher_tensor),
    }


def preprocess_video_frame(item, target_size: int, teacher_transform):
    video_path, frame_index, frame_bgr = item
    sample_id = f"avi_{video_path.stem}_f{frame_index:06d}"

    rgb = preprocess_from_video_frame(frame_bgr, target_size)
    teacher_image = rgb.astype(np.float32) / 255.0
    teacher_tensor = teacher_transform({"image": teacher_image})["image"]

    return {
        "sample_id": sample_id,
        "split_name": "avi_black",
        "label": "",
        "rgb": rgb,
        "teacher_tensor": torch.from_numpy(teacher_tensor),
    }


def infer_depth_batch(model: DepthAnythingV2, samples):
    images = torch.stack([sample["teacher_tensor"] for sample in samples], dim=0).to(DEVICE)
    target_hw = samples[0]["rgb"].shape[:2]

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            depth = model(images)
            depth = torch.nn.functional.interpolate(
                depth[:, None],
                target_hw,
                mode="bilinear",
                align_corners=True,
            )[:, 0]

    return depth.cpu().numpy()


def save_sample(output_root: Path, sample, depth: np.ndarray):
    sample_id = sample["sample_id"]
    image_path = output_root / "images" / f"{sample_id}.png"
    depth_path = output_root / "depth" / f"{sample_id}.npy"
    depth_vis_path = output_root / "depth_vis" / f"{sample_id}.png"

    Image.fromarray(sample["rgb"]).save(image_path)
    np.save(depth_path, depth.astype(np.float32))

    if SAVE_VIS:
        depth_u8 = normalize_depth_to_u8(depth)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(depth_vis_path), depth_color)
        depth_vis_rel = f"depth_vis/{sample_id}.png"
    else:
        depth_vis_rel = ""

    return {
        "sample_id": sample_id,
        "split": sample["split_name"],
        "image_path": f"images/{sample_id}.png",
        "depth_path": f"depth/{sample_id}.npy",
        "depth_vis_path": depth_vis_rel,
        "label": sample["label"],
    }


def load_existing_sample_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["sample_id"] for row in reader}


def iter_video_frame_items(video_path: Path, frame_stride: int, max_frames_per_video: int | None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Warning: failed to open video {video_path}")
        return

    kept = 0
    frame_index = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            break

        if frame_index % frame_stride == 0:
            yield video_path, frame_index, frame_bgr
            kept += 1
            if max_frames_per_video is not None and kept >= max_frames_per_video:
                break

        frame_index += 1

    capture.release()


def process_tfrecord_sources(
    writer,
    existing_sample_ids: set[str],
    output_root: Path,
    teacher_transform,
    model: DepthAnythingV2,
    cpu_pool: ThreadPoolExecutor,
    args,
):
    total_written = 0

    for split_name, tfrecord_path in TFRECORD_SOURCES:
        desc = f"Teacher pass [{split_name}]"
        example_iter = (
            ({"split_name": split_name, "index": idx}, example)
            for idx, example in enumerate(iter_examples(tfrecord_path))
        )

        progress = tqdm(desc=desc, unit="sample")
        for raw_batch in chunked(example_iter, args.batch_size):
            if MAX_RECORDS is not None and total_written >= MAX_RECORDS:
                break

            remaining = None if MAX_RECORDS is None else MAX_RECORDS - total_written
            if remaining is not None and remaining <= 0:
                break

            filtered_batch = []
            for sample_meta, example in raw_batch:
                sample_id = f"{sample_meta['split_name']}_{sample_meta['index']:06d}"
                if sample_id not in existing_sample_ids:
                    filtered_batch.append((sample_meta, example))

            if remaining is not None:
                filtered_batch = filtered_batch[:remaining]

            if not filtered_batch:
                progress.update(len(raw_batch))
                continue

            samples = list(
                cpu_pool.map(
                    lambda item: preprocess_example(item, args.target_size, teacher_transform),
                    filtered_batch,
                )
            )
            depth_batch = infer_depth_batch(model, samples)
            rows = list(
                cpu_pool.map(
                    lambda pair: save_sample(output_root, pair[0], pair[1]),
                    zip(samples, depth_batch),
                )
            )

            for row in rows:
                writer.writerow(row)
                existing_sample_ids.add(row["sample_id"])

            total_written += len(rows)
            progress.update(len(raw_batch))

        progress.close()
        if MAX_RECORDS is not None and total_written >= MAX_RECORDS:
            break


def process_black_avi_sources(
    writer,
    existing_sample_ids: set[str],
    output_root: Path,
    teacher_transform,
    model: DepthAnythingV2,
    cpu_pool: ThreadPoolExecutor,
    args,
):
    avi_paths = list(iter_black_avi_paths(args.avi_root))
    if not avi_paths:
        print(f"No *_Black.avi files found under {args.avi_root}")
        return

    progress = tqdm(avi_paths, desc="Teacher pass [avi_black]", unit="video")
    for video_path in progress:
        pending_items = []
        for item in iter_video_frame_items(
            video_path,
            frame_stride=args.avi_frame_stride,
            max_frames_per_video=args.avi_max_frames_per_video,
        ):
            sample_id = f"avi_{video_path.stem}_f{item[1]:06d}"
            if sample_id in existing_sample_ids:
                continue

            pending_items.append(item)
            if len(pending_items) == args.batch_size:
                samples = list(
                    cpu_pool.map(
                        lambda raw_item: preprocess_video_frame(raw_item, args.target_size, teacher_transform),
                        pending_items,
                    )
                )
                depth_batch = infer_depth_batch(model, samples)
                rows = list(
                    cpu_pool.map(
                        lambda pair: save_sample(output_root, pair[0], pair[1]),
                        zip(samples, depth_batch),
                    )
                )

                for row in rows:
                    writer.writerow(row)
                    existing_sample_ids.add(row["sample_id"])

                pending_items = []

        if pending_items:
            samples = list(
                cpu_pool.map(
                    lambda raw_item: preprocess_video_frame(raw_item, args.target_size, teacher_transform),
                    pending_items,
                )
            )
            depth_batch = infer_depth_batch(model, samples)
            rows = list(
                cpu_pool.map(
                    lambda pair: save_sample(output_root, pair[0], pair[1]),
                    zip(samples, depth_batch),
                )
            )

            for row in rows:
                writer.writerow(row)
                existing_sample_ids.add(row["sample_id"])


def main():
    args = parse_args()
    output_root = resolve_output_root(args.target_size, args.output_root)

    ensure_dirs(output_root)
    model = load_teacher()
    teacher_transform = build_teacher_transform()

    manifest_path = output_root / "manifest.csv"
    fieldnames = ["sample_id", "split", "image_path", "depth_path", "depth_vis_path", "label"]

    existing_sample_ids = load_existing_sample_ids(manifest_path) if args.append_existing else set()
    manifest_mode = "a" if args.append_existing and manifest_path.exists() else "w"

    with manifest_path.open(manifest_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if manifest_mode == "w":
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.cpu_workers) as cpu_pool:
            if not args.skip_tfrecords:
                process_tfrecord_sources(
                    writer,
                    existing_sample_ids,
                    output_root,
                    teacher_transform,
                    model,
                    cpu_pool,
                    args,
                )

            if args.include_black_avi:
                process_black_avi_sources(
                    writer,
                    existing_sample_ids,
                    output_root,
                    teacher_transform,
                    model,
                    cpu_pool,
                    args,
                )

    print(f"Done. Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
