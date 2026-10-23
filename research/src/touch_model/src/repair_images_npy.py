#!/usr/bin/env python3
# repair_images_npy.py
import argparse, os, struct
from pathlib import Path
import numpy as np

SPLITS = ["train", "val", "test"]

MAGIC = b"\x93NUMPY"

def npy_header_ok(path: Path):
    try:
        with open(path, "rb") as f:
            magic = f.read(6)
            return magic == MAGIC
    except Exception:
        return False

def try_load_numeric(path: Path):
    try:
        arr = np.load(path, mmap_mode="r")
        return arr  # numeric or object; dtype tells us
    except Exception:
        return None

def try_load_object(path: Path):
    try:
        arr = np.load(path, allow_pickle=True)
        return arr
    except Exception:
        return None

def guess_raw_memmap(path: Path, shape, dtype):
    # Interpret the file as raw contiguous array of given dtype/shape.
    # Validate by file size.
    H, W, C, N = shape[1], shape[2], shape[3], shape[0]
    itemsize = np.dtype(dtype).itemsize
    expected_bytes = N * H * W * C * itemsize
    filesize = path.stat().st_size
    if filesize != expected_bytes:
        raise ValueError(
            f"{path.name}: size mismatch for raw memmap.\n"
            f"  expected {expected_bytes} bytes for shape {shape} dtype {dtype}, "
            f"found {filesize} bytes."
        )
    arr = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    return np.asarray(arr)  # materialize to a proper ndarray in RAM for saving

def main():
    ap = argparse.ArgumentParser("Repair image npy files into proper numeric ndarrays")
    ap.add_argument("--in-dir", required=True, help="Directory containing *_images.npy")
    ap.add_argument("--labels-dir", required=True, help="Directory containing *_labels.npy (to infer N)")
    ap.add_argument("--out-dir", required=True, help="Output directory for repaired *_images.npy")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--dtype", choices=["uint8","float32"], default="uint8",
                    help="Assumed dtype if file is raw memmap (and target dtype when converting)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    lbl_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = np.uint8 if args.dtype == "uint8" else np.float32

    for split in SPLITS:
        img_path = in_dir / f"{split}_images.npy"
        lbl_path = lbl_dir / f"{split}_labels.npy"
        if not img_path.exists():
            print(f"Skip {split}: {img_path} missing")
            continue
        if not lbl_path.exists():
            print(f"Skip {split}: {lbl_path} missing (needed to determine N)")
            continue

        # Get N from labels
        y = np.load(lbl_path, mmap_mode="r")
        N = int(y.shape[0])
        shape = (N, args.height, args.width, args.channels)

        print(f"\n--- {split.upper()} ---")
        print(f"Input: {img_path}")
        print(f"Using labels count N={N}, target shape={shape}, target dtype={target_dtype}")

        arr = try_load_numeric(img_path)
        if arr is not None:
            if arr.dtype == object:
                print("Detected object-array .npy → loading with allow_pickle and stacking...")
                arr_obj = try_load_object(img_path)
                if arr_obj is None:
                    print("  allow_pickle load failed; will treat as raw memmap")
                else:
                    try:
                        arr_fixed = np.stack(arr_obj)  # (N,H,W,C)
                    except Exception as e:
                        raise RuntimeError(f"{img_path.name}: cannot stack object array elements: {e}")
                    if arr_fixed.shape != shape:
                        raise ValueError(f"{img_path.name}: stacked shape {arr_fixed.shape} != expected {shape}")
                    if arr_fixed.dtype != target_dtype:
                        arr_fixed = arr_fixed.astype(target_dtype, copy=False)
                    np.save(out_dir / img_path.name, arr_fixed)
                    print(f"Wrote repaired: {(out_dir / img_path.name).resolve()}  dtype={arr_fixed.dtype}")
                    continue
            else:
                # numeric and readable already → just ensure dtype and shape match
                if arr.shape != shape:
                    raise ValueError(f"{img_path.name}: numeric .npy shape {arr.shape} != expected {shape}")
                if arr.dtype != target_dtype:
                    print(f"Casting dtype {arr.dtype} -> {target_dtype}")
                    arr2 = np.asarray(arr, dtype=target_dtype)
                else:
                    arr2 = arr
                np.save(out_dir / img_path.name, arr2)
                print(f"Wrote repaired: {(out_dir / img_path.name).resolve()}  dtype={arr2.dtype}")
                continue

        # If we got here, either numeric load failed or object load failed → try raw memmap
        print("No valid .npy header or unpickling failed → treating as raw memmap...")
        arr_mm = guess_raw_memmap(img_path, shape, target_dtype)
        np.save(out_dir / img_path.name, arr_mm)
        print(f"Wrote repaired from raw memmap: {(out_dir / img_path.name).resolve()}  dtype={arr_mm.dtype}")

if __name__ == "__main__":
    main()
