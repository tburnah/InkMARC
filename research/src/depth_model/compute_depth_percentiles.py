import argparse
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/data/depth/pseudo_depth_dataset")
DEFAULT_PERCENTILES = (1, 2, 90, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset-wide percentiles from saved teacher depth .npy files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset root containing the depth/ directory.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Optional explicit path to the directory containing depth .npy files.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_PERCENTILES),
        help="Percentiles to compute. Example: --percentiles 1 2 90 99",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on number of .npy files to scan for a quick estimate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_dir = args.depth_dir if args.depth_dir is not None else args.root / "depth"

    depth_paths = sorted(depth_dir.glob("*.npy"))
    if args.sample_limit is not None:
        depth_paths = depth_paths[:args.sample_limit]

    if not depth_paths:
        raise SystemExit(f"No .npy files found in {depth_dir}")

    values = []
    total_pixels = 0

    for idx, path in enumerate(depth_paths, start=1):
        depth = np.load(path).astype(np.float32, copy=False)
        flat = depth.reshape(-1)
        values.append(flat)
        total_pixels += flat.size

        if idx % 500 == 0 or idx == len(depth_paths):
            print(f"Loaded {idx}/{len(depth_paths)} files")

    all_values = np.concatenate(values, axis=0)
    percentile_values = np.percentile(all_values, args.percentiles)

    print()
    print(f"Depth directory: {depth_dir}")
    print(f"Files scanned: {len(depth_paths)}")
    print(f"Total pixels: {total_pixels}")
    print(f"Min: {all_values.min():.6f}")
    print(f"Max: {all_values.max():.6f}")
    print(f"Mean: {all_values.mean():.6f}")
    print()

    for percentile, value in zip(args.percentiles, percentile_values):
        label = f"p{percentile:g}"
        print(f"{label}: {value:.6f}")


if __name__ == "__main__":
    main()
