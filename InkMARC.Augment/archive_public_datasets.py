#!/usr/bin/env python3
"""Create ZIP archives of the public dataset, preferring corrected HDF5 files.

The two base directories are archived non-recursively so that unrelated derived
output directories are not swept up.  texture_replacement_v3 is archived as a
separate, recursive dataset.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    source: Path
    recursive: bool


DATASETS = (
    Dataset("baseAVIs", Path("/data/baseAVIs"), False),
    Dataset("baseAVIs_432x672", Path("/data/baseAVIs_432x672"), False),
    Dataset(
        "texture_replacement_v3",
        Path("/data/baseAVIs_432x672/texture_replacement_v3"),
        True,
    ),
)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def source_files(dataset: Dataset) -> list[Path]:
    iterator = dataset.source.rglob("*") if dataset.recursive else dataset.source.iterdir()
    return sorted(path for path in iterator if path.is_file())


def select_files(dataset: Dataset) -> tuple[list[Path], list[Path]]:
    """Return selected files and ordinary H5 files superseded by corrected ones."""
    files = source_files(dataset)
    file_set = set(files)
    skipped: list[Path] = []
    selected: list[Path] = []

    for path in files:
        if path.suffix.lower() == ".h5" and not path.stem.endswith("_corrected"):
            corrected = path.with_name(f"{path.stem}_corrected{path.suffix}")
            if corrected in file_set:
                skipped.append(path)
                continue
        selected.append(path)

    return selected, skipped


def archive_name(dataset: Dataset, path: Path) -> str:
    # Each ZIP has one top-level directory, making extraction tidy.
    return (Path(dataset.name) / path.relative_to(dataset.source)).as_posix()


def make_archive(
    dataset: Dataset,
    files: list[Path],
    destination: Path,
    compression: int,
    overwrite: bool,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists (use --overwrite)")
    if partial.exists():
        if not overwrite:
            raise FileExistsError(f"{partial} already exists (use --overwrite)")
        partial.unlink()

    total_bytes = sum(path.stat().st_size for path in files)
    written_bytes = 0
    last_report = 0.0
    started = time.monotonic()

    try:
        with zipfile.ZipFile(
            partial,
            mode="w",
            compression=compression,
            allowZip64=True,
            compresslevel=6 if compression == zipfile.ZIP_DEFLATED else None,
        ) as archive:
            for number, path in enumerate(files, start=1):
                archive.write(path, archive_name(dataset, path))
                written_bytes += path.stat().st_size
                now = time.monotonic()
                if now - last_report >= 10 or number == len(files):
                    percent = 100 * written_bytes / total_bytes if total_bytes else 100
                    elapsed = max(now - started, 0.001)
                    rate = written_bytes / elapsed
                    print(
                        f"\r  {number}/{len(files)} files, {percent:5.1f}%, "
                        f"{human_size(rate)}/s",
                        end="",
                        flush=True,
                    )
                    last_report = now
        print()
        os.replace(partial, destination)
    except BaseException:
        print(f"\nIncomplete archive retained at: {partial}", file=sys.stderr)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/dataset_archives"),
        help="archive destination (default: /data/dataset_archives)",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=[dataset.name for dataset in DATASETS],
        help="archive only this dataset; repeat to select more than one",
    )
    parser.add_argument(
        "--compression",
        choices=("stored", "deflated"),
        default="stored",
        help="stored is much faster for already-compressed media (default: stored)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing completed/partial ZIPs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show selections and space needs only"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wanted = set(args.only) if args.only else {dataset.name for dataset in DATASETS}
    datasets = [dataset for dataset in DATASETS if dataset.name in wanted]

    plans: list[tuple[Dataset, list[Path], list[Path], int]] = []
    for dataset in datasets:
        if not dataset.source.is_dir():
            print(f"Missing source directory: {dataset.source}", file=sys.stderr)
            return 2
        files, skipped = select_files(dataset)
        total_bytes = sum(path.stat().st_size for path in files)
        plans.append((dataset, files, skipped, total_bytes))
        scope = "recursive" if dataset.recursive else "top-level files only"
        print(f"{dataset.name} ({scope})")
        print(f"  include: {len(files)} files, {human_size(total_bytes)}")
        print(f"  omit:    {len(skipped)} superseded .h5 files")

    required = sum(plan[3] for plan in plans)
    print(f"Total upper-bound space required: {human_size(required)}")
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(args.output_dir).free
    # Stored ZIPs need roughly the input size. Deflated ZIPs cannot be larger by
    # much, so the input size is a conservative and easy-to-understand check.
    if available < required:
        print(
            f"Not enough free space in {args.output_dir}: "
            f"need about {human_size(required)}, have {human_size(available)}",
            file=sys.stderr,
        )
        return 2

    compression = (
        zipfile.ZIP_STORED if args.compression == "stored" else zipfile.ZIP_DEFLATED
    )
    for dataset, files, _skipped, _total_bytes in plans:
        destination = args.output_dir / f"{dataset.name}.zip"
        print(f"Creating {destination}")
        make_archive(dataset, files, destination, compression, args.overwrite)

    print("Created:")
    for dataset in datasets:
        destination = args.output_dir / f"{dataset.name}.zip"
        print(f"  {destination} ({human_size(destination.stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
