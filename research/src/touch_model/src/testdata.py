#!/usr/bin/env python3
# build_splits_streaming_verbose.py
import argparse, os, sys, h5py, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Optional tqdm
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def list_h5(input_dir):
    paths = sorted(Path(input_dir).glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")
    return [str(p) for p in paths]

def read_labels_from_h5(path):
    with h5py.File(path, "r") as f:
        if "/touch" in f:
            y = np.array(f["/touch"]).reshape(-1)
        elif "/Attributes" in f:
            a = np.array(f["/Attributes"])
            if a.ndim != 2 or a.shape[1] < 1:
                raise ValueError(f"{path}: /Attributes shape {a.shape} invalid")
            y = a[:, 0]
        else:
            raise KeyError(f"{path}: no /touch or /Attributes")
    return (y > 0).astype(np.int32)

def image_count_and_shape(path):
    with h5py.File(path, "r") as f:
        if "/images_reshaped" in f: X = f["/images_reshaped"]
        elif "/Images" in f:        X = f["/Images"]
        else: raise KeyError(f"{path}: no images dataset")
        return X.shape[0], X.shape[1:]  # N, (H,W,C)

def read_image(path, idx):
    with h5py.File(path, "r") as f:
        if "/images_reshaped" in f: X = f["/images_reshaped"]
        else:                        X = f["/Images"]
        return X[idx]  # uint8 slice

def main():
    ap = argparse.ArgumentParser("Streaming H5 → npy splits (with progress)")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true", help="Save float32 [0,1] instead of uint8")
    ap.add_argument("--quiet", action="store_true", help="Minimal logging")
    ap.add_argument("--log-interval", type=int, default=1000, help="Print every N images if tqdm not available")
    args = ap.parse_args()

    paths = list_h5(args.input_dir)
    if not args.quiet:
        print(f"Found {len(paths)} .h5 files")

    # 1) Collect labels and pointers (only metadata in RAM)
    pointers = []
    labels = []
    total = 0
    for fi, p in enumerate(paths):
        y = read_labels_from_h5(p)
        labels.append(y)
        n = len(y)
        pointers.extend((fi, i) for i in range(n))
        total += n
        if not args.quiet:
            pos = int((y == 1).sum()); neg = n - pos
            print(f"[{os.path.basename(p)}] samples={n}  pos={pos}  neg={neg}")

    labels = np.concatenate(labels)
    N = len(labels)

    # 2) Check image shape consistency
    _, shape0 = image_count_and_shape(paths[0])
    H, W, C = shape0
    for p in paths[1:]:
        _, shp = image_count_and_shape(p)
        if shp != shape0:
            raise ValueError(f"Image shapes differ: {shape0} vs {shp} in {p}")

    if not args.quiet:
        print(f"Total samples: {N}  shape: (H,W,C)=({H},{W},{C})")
        print(f"Overall class balance: pos={(labels==1).sum()}  neg={(labels==0).sum()}")

    # 3) Stratified splits
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    test_size = args.test_ratio
    val_size  = args.val_ratio / (1.0 - test_size)

    idx_all = np.arange(N)
    idx_tmp, idx_test, y_tmp, y_test = train_test_split(
        idx_all, labels, test_size=test_size, random_state=args.seed, stratify=labels
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tmp, y_tmp, test_size=val_size, random_state=args.seed, stratify=y_tmp
    )

    if not args.quiet:
        print(f"Split sizes → train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    # 4) Prepare outputs
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if args.normalize else np.uint8

    def create_memmap(out_name, count):
        path = out / f"{out_name}.npy"
        mm = np.memmap(path, mode="w+", dtype=dtype, shape=(count, H, W, C))
        return mm, path

    Xtr, p_tr = create_memmap("train_images", len(idx_train))
    Xva, p_va = create_memmap("val_images",   len(idx_val))
    Xte, p_te = create_memmap("test_images",  len(idx_test))

    np.save(out / "train_labels.npy", y_train.astype(np.int32))
    np.save(out / "val_labels.npy",   y_val.astype(np.int32))
    np.save(out / "test_labels.npy",  y_test.astype(np.int32))
    if not args.quiet:
        print("Labels saved.")

    # 5) Streamed writers with progress
    def write_split(memmap_arr, split_indices, desc):
        n = len(split_indices)
        if tqdm is not None and not args.quiet:
            iterator = tqdm(split_indices, desc=desc, unit="img")
            step_log = None
        else:
            iterator = split_indices
            step_log = args.log_interval

        w = 0
        for k in iterator:
            fi, li = pointers[k]
            img = read_image(paths[fi], li)  # uint8
            if args.normalize:
                memmap_arr[w] = (img.astype(np.float32) / 255.0)
            else:
                memmap_arr[w] = img
            w += 1
            if step_log and not args.quiet and (w % step_log == 0):
                print(f"{desc}: wrote {w}/{n}")
                sys.stdout.flush()
        memmap_arr.flush()
        if not args.quiet:
            print(f"{desc}: done ({w} images)")

    write_split(Xtr, idx_train, "train_images")
    write_split(Xva, idx_val,   "val_images")
    write_split(Xte, idx_test,  "test_images")

    if not args.quiet:
        print("All done. Files saved to:", out.resolve())

if __name__ == "__main__":
    # Make Python flush prints immediately if needed
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
