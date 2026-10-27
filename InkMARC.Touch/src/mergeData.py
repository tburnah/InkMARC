#!/usr/bin/env python3
# merge_npy_splits.py
import argparse
from pathlib import Path
import numpy as np

SPLITS = ["train", "val", "test"]

def load_npy(path, mmap=True):
    mode = "r" if mmap else None
    return np.load(path, mmap_mode=mode)

def assert_compatible_images(a, b, target_dtype=None):
    # both should be 4D: (N,H,W,C)
    if a.ndim != 4 or b.ndim != 4:
        raise ValueError(f"Images must be 4D, got {a.shape} and {b.shape}")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"Image shapes differ: {a.shape[1:]} vs {b.shape[1:]}")
    if target_dtype is None and a.dtype != b.dtype:
        raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype} (use --to-dtype to coerce)")
    return a.shape[1:], (target_dtype or a.dtype)

def assert_compatible_labels(a, b, target_dtype=None):
    # both should be 1D: (N,)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Labels must be 1D, got {a.shape} and {b.shape}")
    if target_dtype is None and a.dtype != b.dtype:
        raise ValueError(f"Label dtype mismatch: {a.dtype} vs {b.dtype} (use --labels-to-dtype)")
    return (target_dtype or a.dtype)

def write_concat_images(out_path, img_a, img_b, dtype, shuffle=False, seed=42):
    n_a, n_b = img_a.shape[0], img_b.shape[0]
    H, W, C = img_a.shape[1:]
    total = n_a + n_b

    mm = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total, H, W, C))
    # Copy A then B with optional dtype conversion
    mm[:n_a] = img_a.astype(dtype, copy=False)
    mm[n_a:] = img_b.astype(dtype, copy=False)

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = np.arange(total)
        rng.shuffle(idx)
        # in-place shuffle via permutation index (create a temp copy on disk-efficient basis)
        tmp = mm.copy()  # creates a RAM copy; for huge sets, do blockwise shuffle instead
        mm[:] = tmp[idx]
        del tmp
    mm.flush()

def write_concat_labels(out_path, y_a, y_b, dtype, shuffle=False, seed=42, same_shuffle_index=None):
    n_a, n_b = y_a.shape[0], y_b.shape[0]
    total = n_a + n_b
    y_full = np.empty((total,), dtype=dtype)
    y_full[:n_a] = y_a.astype(dtype, copy=False)
    y_full[n_a:] = y_b.astype(dtype, copy=False)

    if shuffle:
        if same_shuffle_index is None:
            rng = np.random.default_rng(seed)
            idx = np.arange(total)
            rng.shuffle(idx)
        else:
            idx = same_shuffle_index
        y_full = y_full[idx]
        return y_full, idx
    return y_full, None

def main():
    ap = argparse.ArgumentParser("Merge two sets of npy splits (train/val/test) safely")
    ap.add_argument("--orig-dir", required=True, help="Directory with original .npy files")
    ap.add_argument("--new-dir", required=True, help="Directory with new .npy files")
    ap.add_argument("--out-dir", required=True, help="Directory to write merged .npy files")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle merged sets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--to-dtype", choices=["uint8","float32"], default=None,
                    help="Coerce image dtype (helps if orig/new dtypes differ)")
    ap.add_argument("--labels-to-dtype", choices=["int32","int64"], default=None,
                    help="Coerce label dtype")
    args = ap.parse_args()

    orig = Path(args.orig_dir)
    new  = Path(args.new_dir)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # dtype coercion
    img_dtype = None
    if args.to_dtype == "uint8":
        img_dtype = np.uint8
    elif args.to_dtype == "float32":
        img_dtype = np.float32

    lbl_dtype = None
    if args.labels_to_dtype == "int32":
        lbl_dtype = np.int32
    elif args.labels_to_dtype == "int64":
        lbl_dtype = np.int64

    for split in SPLITS:
        print(f"=== {split.upper()} ===")
        orig_img_p = orig / f"{split}_images.npy"
        orig_lbl_p = orig / f"{split}_labels.npy"
        new_img_p  = new  / f"{split}_images.npy"
        new_lbl_p  = new  / f"{split}_labels.npy"

        if not (orig_img_p.exists() and orig_lbl_p.exists() and new_img_p.exists() and new_lbl_p.exists()):
            print(f"Skipping {split}: one or more files missing.")
            continue

        A_img = load_npy(orig_img_p)
        B_img = load_npy(new_img_p)
        shape, merged_img_dtype = assert_compatible_images(A_img, B_img, img_dtype)
        print(f"Images: {A_img.shape} + {B_img.shape} -> dtype {merged_img_dtype}")

        A_lbl = load_npy(orig_lbl_p)
        B_lbl = load_npy(new_lbl_p)
        merged_lbl_dtype = assert_compatible_labels(A_lbl, B_lbl, lbl_dtype)
        print(f"Labels: {A_lbl.shape} + {B_lbl.shape} -> dtype {merged_lbl_dtype}")

        if A_img.shape[0] != A_lbl.shape[0] or B_img.shape[0] != B_lbl.shape[0]:
            raise ValueError("Image/label counts differ within a split")

        # First, merge labels and produce a shuffle index if requested
        merged_labels, shuffle_idx = write_concat_labels(
            out / f"{split}_labels.npy",
            A_lbl, B_lbl, merged_lbl_dtype,
            shuffle=args.shuffle, seed=args.seed
        )
        np.save(out / f"{split}_labels.npy", merged_labels)  # writes a compact array

        # Now write merged images; if shuffling, apply the same permutation
        merged_img_path = out / f"{split}_images.npy"
        n_total = A_img.shape[0] + B_img.shape[0]
        H, W, C = shape
        mm = np.memmap(merged_img_path, dtype=merged_img_dtype, mode="w+", shape=(n_total, H, W, C))
        # Copy then (optional) shuffle without holding everything in RAM
        mm[:A_img.shape[0]] = A_img.astype(merged_img_dtype, copy=False)
        mm[A_img.shape[0]:] = B_img.astype(merged_img_dtype, copy=False)
        if args.shuffle:
            if shuffle_idx is None:
                # Shouldn't happen, but guard
                rng = np.random.default_rng(args.seed)
                idx = np.arange(n_total); rng.shuffle(idx)
            else:
                idx = shuffle_idx
            # Blockwise apply permutation to reduce peak memory:
            block = 8192
            tmp = np.memmap(out / f"._tmp_{split}_images.npy",
                            dtype=merged_img_dtype, mode="w+", shape=(n_total, H, W, C))
            # Write permuted view into tmpcd 
            for start in range(0, n_total, block):
                end = min(end := start + block, n_total)
                tmp[start:end] = mm[idx[start:end]]
            tmp.flush()
            del mm
            # Replace mm with tmp content (atomic-ish swap)
            Path(merged_img_path).unlink(missing_ok=True)
            Path(out / f"._tmp_{split}_images.npy").rename(merged_img_path)
        else:
            mm.flush()
            del mm

        print(f"Wrote: {merged_img_path.name}, {split}_labels.npy")

    print(f"Done. Merged files in: {out.resolve()}")

if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as e:
        print("!! ERROR:", e)
        sys.exit(1)