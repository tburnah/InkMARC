#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

SPLITS = ["train", "val", "test"]
MAGIC = b"\x93NUMPY"

def try_numeric_load(p):
    try:
        return np.load(p, mmap_mode="r")
    except Exception:
        return None

def try_object_load(p):
    try:
        return np.load(p, allow_pickle=True)
    except Exception:
        return None

def detect_raw_dtype(filesize, shape):
    n_elems = int(np.prod(shape))
    for dtype in (np.uint8, np.float16, np.float32):
        if filesize == n_elems * np.dtype(dtype).itemsize:
            return dtype
    raise ValueError(
        f"Cannot infer raw dtype from size={filesize} for shape={shape} "
        f"(tried uint8/float16/float32)."
    )

def memmap_as_array(p: Path, shape, dtype):
    mm = np.memmap(p, dtype=dtype, mode="r", shape=shape)
    return np.asarray(mm)  # materialize so we can save proper .npy

def to_uint8(img):
    """Convert numeric array to uint8 0..255 sensibly."""
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        # If seems normalized -> scale; otherwise cast
        mx = float(img.max())
        mn = float(img.min())
        if mx <= 1.0 + 1e-6 and mn >= -1e-6:
            img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img
    # ints other than uint8
    return np.clip(img, 0, 255).astype(np.uint8)

def repair_one(images_path: Path, labels_path: Path, out_path: Path, H, W, C):
    y = np.load(labels_path, mmap_mode="r")
    N = int(y.shape[0])
    shape = (N, H, W, C)

    print(f"- Repairing {images_path.name}  N={N}, shape={shape}")

    # 1) Try normal .npy load
    arr = try_numeric_load(images_path)
    if arr is not None:
        if arr.dtype == object:
            print("  object-array .npy -> stacking elements...")
            obj = try_object_load(images_path)
            if obj is not None:
                arr = np.stack(obj)
            else:
                arr = None  # fall through
        if arr is not None:
            if arr.shape != shape:
                raise ValueError(f"  numeric .npy has shape {arr.shape}, expected {shape}")
            arr_u8 = to_uint8(arr)
            np.save(out_path, arr_u8)
            print(f"  wrote {out_path.name}: {arr_u8.shape} {arr_u8.dtype}")
            return

    # 2) Raw memmap (no .npy header or unpickling failed)
    filesize = images_path.stat().st_size
    raw_dtype = detect_raw_dtype(filesize, shape)
    print(f"  treating as RAW MEMMAP with dtype={raw_dtype}")
    arr_raw = memmap_as_array(images_path, shape, raw_dtype)
    arr_u8 = to_uint8(arr_raw)
    np.save(out_path, arr_u8)
    print(f"  wrote {out_path.name}: {arr_u8.shape} {arr_u8.dtype}")

def main():
    ap = argparse.ArgumentParser("Repair NEW *_images.npy into real uint8 arrays matching old data")
    ap.add_argument("--in-dir", required=True, help="Dir with NEW *_images.npy (possibly object/raw)")
    ap.add_argument("--labels-dir", required=True, help="Dir with NEW *_labels.npy (for N)")
    ap.add_argument("--out-dir", required=True, help="Output dir for repaired *_images.npy")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--channels", type=int, default=3)
    args = ap.parse_args()

    in_dir, lbl_dir = Path(args.in_dir), Path(args.labels_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        img_p = in_dir / f"{split}_images.npy"
        lbl_p = lbl_dir / f"{split}_labels.npy"
        if not img_p.exists():
            print(f"Skip {split}: missing {img_p}")
            continue
        if not lbl_p.exists():
            print(f"Skip {split}: missing {lbl_p}")
            continue
        repair_one(img_p, lbl_p, out_dir / img_p.name, args.height, args.width, args.channels)

if __name__ == "__main__":
    main()
