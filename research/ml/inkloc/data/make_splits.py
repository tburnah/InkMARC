from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_index_version(indices_dir: Path, source_version: str, out_version: str) -> None:
    for prefix, suffix in [("index", "jsonl"), ("manifest", "json")]:
        src = indices_dir / f"{prefix}_{source_version}.{suffix}"
        dst = indices_dir / f"{prefix}_{out_version}.{suffix}"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)


def ids_for_participants(rows: Iterable[Dict], participants: set[int]) -> list[str]:
    return [str(r["sample_id"]) for r in rows if int(r.get("participant_id", -1)) in participants]


def participant_held_out(rows: list[Dict], val_participants: set[int], test_participants: set[int]) -> Dict[str, list[str]]:
    train_ids = []
    val_ids = []
    test_ids = []
    held_out = val_participants | test_participants
    for row in rows:
        participant = int(row.get("participant_id", -1))
        sample_id = str(row["sample_id"])
        if participant in test_participants:
            test_ids.append(sample_id)
        if participant in val_participants:
            val_ids.append(sample_id)
        if participant not in held_out:
            train_ids.append(sample_id)
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def sequence_held_out(rows: list[Dict], seed: int, val_frac: float, test_frac: float) -> Dict[str, list[str]]:
    by_seq = defaultdict(list)
    for row in rows:
        by_seq[str(row.get("file_id") or row.get("rel_path"))].append(str(row["sample_id"]))
    seqs = np.array(sorted(by_seq.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n = len(seqs)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_seqs = set(seqs[:n_test].tolist())
    val_seqs = set(seqs[n_test:n_test + n_val].tolist())
    train_seqs = set(seqs[n_test + n_val:].tolist())
    return {
        "train": [sid for seq in sorted(train_seqs) for sid in by_seq[seq]],
        "val": [sid for seq in sorted(val_seqs) for sid in by_seq[seq]],
        "test": [sid for seq in sorted(test_seqs) for sid in by_seq[seq]],
    }


def parse_participants(value: str) -> set[int]:
    if not value:
        return set()
    return {int(v.strip()) for v in value.split(",") if v.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data/LocationData")
    parser.add_argument("--source-version", default="focused")
    parser.add_argument("--out-version", default=None)
    parser.add_argument("--mode", choices=["sequence_held_out", "participant_held_out"], required=True)
    parser.add_argument("--val-participants", default="")
    parser.add_argument("--test-participants", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--make-lopo", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    indices_dir = root / "indices"
    rows = read_jsonl(indices_dir / f"index_{args.source_version}.jsonl")
    participants = sorted({int(r.get("participant_id", -1)) for r in rows if r.get("participant_id") is not None})
    outputs = []

    if args.make_lopo:
        if len(participants) < 2:
            raise RuntimeError(f"Need at least two participants for leave-one-participant-out, got {participants}")
        for held_out in participants:
            out_version = f"{args.source_version}_lopo_{held_out}"
            splits = participant_held_out(rows, {held_out}, {held_out})
            copy_index_version(indices_dir, args.source_version, out_version)
            write_json(indices_dir / f"splits_{out_version}.json", {
                "mode": "participant_held_out",
                "source_version": args.source_version,
                "held_out_participant": held_out,
                "limitation": "val and test are the same held-out participant set; use val for calibration and reserve test only after creating a separate split.",
                "splits": splits,
            })
            outputs.append(out_version)
    else:
        out_version = args.out_version or f"{args.source_version}_{args.mode}"
        if args.mode == "participant_held_out":
            splits = participant_held_out(rows, parse_participants(args.val_participants), parse_participants(args.test_participants))
        else:
            splits = sequence_held_out(rows, args.seed, args.val_frac, args.test_frac)
        copy_index_version(indices_dir, args.source_version, out_version)
        write_json(indices_dir / f"splits_{out_version}.json", {
            "mode": args.mode,
            "source_version": args.source_version,
            "seed": args.seed,
            "participants": participants,
            "splits": splits,
        })
        outputs.append(out_version)

    for out_version in outputs:
        doc = json.loads((indices_dir / f"splits_{out_version}.json").read_text(encoding="utf-8"))
        splits = doc["splits"]
        print(json.dumps({
            "out_version": out_version,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "participants": participants,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
