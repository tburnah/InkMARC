#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def _run(cmd: list[str], repo_root: Path) -> None:
    print(f"$ {' '.join(cmd)}")
    env = dict(**dict(**__import__("os").environ))
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(cmd, check=True, env=env)

def _has_repo_marker(path: Path) -> bool:
    return (path / "scripts" / "build_manifest.py").exists()


def _search_up(start: Path) -> Optional[Path]:
    for p in [start] + list(start.parents):
        if _has_repo_marker(p):
            return p
    return None


def _find_repo_root(hints: Iterable[Path]) -> Path:
    for h in hints:
        found = _search_up(h)
        if found is not None:
            return found
    fallback = Path.home() / "inkmarc"
    if _has_repo_marker(fallback):
        return fallback
    raise FileNotFoundError(
        "Could not locate repo root containing scripts/build_manifest.py. "
        "Pass --repo-root explicitly."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--dataset-root", default="/data/LocationData")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--sample-split", default="train")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--sample-out", default="runs/phase1_sample.png")
    ap.add_argument("--out-w", type=int, default=480)
    ap.add_argument("--out-h", type=int, default=320)
    ap.add_argument("--margin-frac", type=float, default=0.10)
    args = ap.parse_args()

    dataset_root = args.dataset_root
    version = args.version

    if args.repo_root:
        repo_root = Path(args.repo_root).expanduser().resolve()
    else:
        repo_root = _find_repo_root(
            [
                Path.cwd(),
                Path(__file__).resolve().parent,
            ]
        )
    py = sys.executable

    _run(
        [
            py,
            str(repo_root / "scripts" / "build_manifest.py"),
            "--dataset-root",
            dataset_root,
            "--version",
            version,
        ],
        repo_root,
    )
    _run(
        [
            py,
            str(repo_root / "scripts" / "test_video_alignment.py"),
            "--dataset-root",
            dataset_root,
            "--version",
            version,
        ],
        repo_root,
    )
    _run(
        [
            py,
            str(repo_root / "scripts" / "test_phase1_sample.py"),
            "--dataset-root",
            dataset_root,
            "--version",
            version,
            "--split",
            args.sample_split,
            "--index",
            str(args.sample_index),
            "--out",
            args.sample_out,
            "--out-w",
            str(args.out_w),
            "--out-h",
            str(args.out_h),
            "--margin-frac",
            str(args.margin_frac),
        ],
        repo_root,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
