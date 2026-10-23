#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path

FILES = ["train_images.npy","val_images.npy","test_images.npy"]

def load_object_array(path: Path) -> np.ndarray:
    # Load trusted object array
    arr = np.load(path, allow_pickle=True)
    if arr.dtype != object:
        return arr  # already numeric
    # Expect arr to be a sequence of equally-shaped frames
    try:
        arr2 = np.stack(arr)  # (N,H,W,C)
    except Exception as e:
        raise RuntimeError(f"{path.name}: cannot stack object array elements: {e}")
    return arr2

def main():
    ap = argparse.ArgumentParser("Convert pickled/object image .npy -> numeric ndarray")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--to-dtype", choices=["uint8","float32"], default="uint8",
                    help="Output dtype for images")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_dtype = np.uint8 if args.to_dtype == "uint8" else np.float32

    for fname in FILES:
        src = in_dir / fname
        if not src.exists():
            print(f"Skipping missing: {fname}")
            continue
        arr = load_object_array(src)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"{fname}: expected (N,H,W,3), got {arr.shape}")
        if out_dtype is np.uint8 and arr.dtype != np.uint8:
            # If values are 0..255 ints, cast; else if 0..1 floats, scale up
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255.0 + 0.5).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8, copy=False)
        elif out_dtype is np.float32 and arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        np.save(out_dir / fname, arr)
        print(f"Wrote {fname}: shape={arr.shape} dtype={arr.dtype}")

if __name__ == "__main__":
    main()
