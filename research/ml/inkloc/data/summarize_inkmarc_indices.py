from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.inkloc.data.inkmarc import INKMARC_SRC_DEFAULT


def ensure_inkmarc_imports():
    src = str(INKMARC_SRC_DEFAULT)
    if src not in sys.path:
        sys.path.insert(0, src)
    from data.dataset_core_v2 import InkMARCIndexedDataset
    from data.label_schema import P_IDX, extract_xy_touch

    return InkMARCIndexedDataset, P_IDX, extract_xy_touch


def summarize_split(dataset_root: str, index_version: str, split: str, pressure_touch_threshold: float) -> dict:
    InkMARCIndexedDataset, P_IDX, extract_xy_touch = ensure_inkmarc_imports()
    ds = InkMARCIndexedDataset(
        dataset_root,
        index_version=index_version,
        split=split,
        require_labels=False,
        transform=None,
    )
    records = ds._records
    state_counts = Counter()
    touch_counts = Counter()
    pressure_availability = Counter()
    has_xy_counts = Counter()
    by_participant = defaultdict(lambda: Counter())
    by_sequence = defaultdict(lambda: Counter())
    sequence_to_participant: dict[str, int] = {}

    for idx, rec in enumerate(records):
        sample = ds[idx]
        labels = np.asarray(sample["labels"], dtype=np.float32)
        label_mask = np.asarray(sample["label_mask"], dtype=np.uint8)
        touch, _touch_mask, _x, _y, xy_mask = extract_xy_touch(
            labels,
            label_mask,
            pressure_touch_threshold=pressure_touch_threshold,
        )
        has_pressure = int(label_mask[P_IDX] != 0)
        state = 2 if touch > 0.5 and xy_mask > 0.5 else 1 if xy_mask > 0.5 else 0
        participant_id = int(rec.get("participant_id", -1) or -1)
        sequence_id = str(rec.get("file_id") or rec.get("rel_path"))

        state_counts[state] += 1
        touch_counts[int(touch > 0.5)] += 1
        pressure_availability[has_pressure] += 1
        has_xy_counts[int(xy_mask > 0.5)] += 1
        by_participant[participant_id][state] += 1
        by_participant[participant_id]["total"] += 1
        by_sequence[sequence_id][state] += 1
        by_sequence[sequence_id]["total"] += 1
        sequence_to_participant[sequence_id] = participant_id

    participant_summary = {}
    for pid in sorted(by_participant):
        counts = by_participant[pid]
        participant_summary[str(pid)] = {
            "total": int(counts["total"]),
            "background": int(counts[0]),
            "hover": int(counts[1]),
            "touch": int(counts[2]),
        }

    sequence_summary = {}
    for seq in sorted(by_sequence):
        counts = by_sequence[seq]
        sequence_summary[seq] = {
            "participant_id": int(sequence_to_participant[seq]),
            "total": int(counts["total"]),
            "background": int(counts[0]),
            "hover": int(counts[1]),
            "touch": int(counts[2]),
        }

    return {
        "split": split,
        "num_samples": len(ds),
        "num_participants": len(participant_summary),
        "num_sequences": len(sequence_summary),
        "state_counts": {
            "background": int(state_counts[0]),
            "hover": int(state_counts[1]),
            "touch": int(state_counts[2]),
        },
        "touch_vs_non_touch": {
            "non_touch": int(touch_counts[0]),
            "touch": int(touch_counts[1]),
        },
        "pressure_availability": {
            "missing": int(pressure_availability[0]),
            "present": int(pressure_availability[1]),
        },
        "has_xy": {
            "missing": int(has_xy_counts[0]),
            "present": int(has_xy_counts[1]),
        },
        "participants": participant_summary,
        "sequences": sequence_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data/LocationData")
    parser.add_argument("--versions", nargs="+", default=["focused", "v1"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--pressure-touch-threshold", type=float, default=0.0)
    parser.add_argument("--out", default="/home/t/tburnah/ml/artifacts/data_summaries/inkmarc_index_summary.json")
    args = parser.parse_args()

    payload = {}
    for version in args.versions:
        payload[version] = {}
        for split in args.splits:
            payload[version][split] = summarize_split(
                args.dataset_root,
                version,
                split,
                args.pressure_touch_threshold,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "versions": list(payload.keys()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
