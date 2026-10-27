#!/usr/bin/env python3
"""
build_splits.py

Reads index_*.jsonl and writes splits_*.json with train/val/test sample_id lists.

Default strategy is 'participant' to avoid leakage: each participant is assigned to exactly
one split, and all of their samples follow.

Alternative: --strategy file (assign each H5 file to one split)

Ratios are configurable and deterministic via --seed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_index(index_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Index is empty: {index_path}")
    return rows


def normalize_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    s = train + val + test
    if s <= 0:
        raise ValueError("Ratios must sum to > 0")
    return train / s, val / s, test / s


def assign_groups_to_splits(
    group_keys: List[Any],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, set]:
    r_train, r_val, r_test = ratios
    rng = random.Random(seed)
    keys = list(group_keys)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    # ensure all assigned
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    assert len(train_keys) + len(val_keys) + len(test_keys) == n
    return {"train": train_keys, "val": val_keys, "test": test_keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("/data/LocationData"))
    ap.add_argument("--indices-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1")
    ap.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to index JSONL (default: <indices-dir>/index_<version>.jsonl)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output splits JSON (default: <indices-dir>/splits_<version>.json)",
    )
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--strategy",
        type=str,
        default="participant",
        choices=["participant", "file"],
        help="Leakage control strategy: participant (default) or file",
    )
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    indices_dir: Path = args.indices_dir or (dataset_root / "indices")
    indices_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.index or (indices_dir / f"index_{args.version}.jsonl")
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            f"Run build_index.py first."
        )

    output_path = args.output or (indices_dir / f"splits_{args.version}.json")

    rows = load_index(index_path)
    ratios = normalize_ratios(args.train, args.val, args.test)

    # Group samples according to strategy
    groups: Dict[Any, List[str]] = defaultdict(list)
    if args.strategy == "participant":
        for r in rows:
            groups[int(r["participant_id"])].append(r["sample_id"])
    else:  # file
        for r in rows:
            groups[str(r["file_id"])].append(r["sample_id"])

    group_keys = list(groups.keys())
    split_keys = assign_groups_to_splits(group_keys, ratios, seed=args.seed)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for split_name, keys in split_keys.items():
        for k in keys:
            splits[split_name].extend(groups[k])

    # Deterministic ordering helps diffing/traceability
    splits["train"].sort()
    splits["val"].sort()
    splits["test"].sort()

    out = {
        "schema_version": 1,
        "meta": {
            "index_version": args.version,
            "strategy": args.strategy,
            "seed": args.seed,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "group_count": len(group_keys),
            "sample_counts": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
                "total": len(rows),
            },
        },
        "splits": splits,
    }

    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote splits: {output_path}")
    print(json.dumps(out["meta"]["sample_counts"], indent=2))


if __name__ == "__main__":
    main()
