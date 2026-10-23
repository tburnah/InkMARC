#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# --- add src/ to PYTHONPATH ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.dataset_core import InkMARCIndexedDataset


# Matches your C# writer:
# 0:X, 1:Y, 2:Pressure, 3:TiltX, 4:TiltY
X_IDX = 0
Y_IDX = 1
P_IDX = 2
TX_IDX = 3
TY_IDX = 4


def touch_rules(labels: np.ndarray, mask: np.ndarray, thr: float) -> Dict[str, float]:
    """Compute several 'touch' interpretations for debugging."""
    p_present = float(mask[P_IDX] != 0)
    p = float(labels[P_IDX]) if p_present > 0.5 else 0.0

    return {
        "p_present": p_present,
        "p_value": p,
        "touch_pressure_present": 1.0 if p_present > 0.5 else 0.0,
        "touch_pressure_gt_thr": 1.0 if (p_present > 0.5 and p > thr) else 0.0,
        "touch_pressure_eq0": 1.0 if (p_present > 0.5 and abs(p) <= 1e-12) else 0.0,  # diagnostic
    }


def fmt_arr(a: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):.6g}" for x in a.tolist()) + "]"


def dump_sample(i: int, sample: Dict[str, Any], thr: float) -> None:
    labels = np.asarray(sample["labels"], dtype=np.float64)
    mask = np.asarray(sample["label_mask"], dtype=np.uint8)
    meta = sample.get("meta", {})

    rules = touch_rules(labels, mask, thr)

    print(
        f"idx={i} file_id={meta.get('file_id')} rel_path={meta.get('rel_path')} frame_idx={meta.get('frame_idx')} "
        f"task={meta.get('task')} participant={meta.get('participant_id')}"
    )
    print(f"  label_mask: {mask.tolist()}  (X,Y,P,Tx,Ty)")
    print(f"  labels    : {fmt_arr(labels)}")
    print(
        f"  X={labels[X_IDX]:.3f} Y={labels[Y_IDX]:.3f} P={labels[P_IDX]:.6f} Tx={labels[TX_IDX]:.3f} Ty={labels[TY_IDX]:.3f}"
    )
    print(
        f"  p_present={int(rules['p_present'])}  "
        f"touch(present)={int(rules['touch_pressure_present'])}  "
        f"touch(p>{thr})={int(rules['touch_pressure_gt_thr'])}  "
        f"p==0={int(rules['touch_pressure_eq0'])}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--count", type=int, default=20, help="How many samples to print (general dump).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--threshold", type=float, default=0.01, help="Pressure threshold to test touch.")
    ap.add_argument(
        "--find",
        choices=["any", "pressure_missing", "pressure_zero", "xy_missing"],
        default="any",
        help="Filter mode for the targeted dump section.",
    )
    ap.add_argument("--find-count", type=int, default=20, help="How many matching samples to print for --find.")
    args = ap.parse_args()

    ds = InkMARCIndexedDataset(
        dataset_root=args.dataset_root,
        index_version=args.version,
        split=args.split,
        require_labels=True,
        transform=None,
    )

    rng = np.random.default_rng(args.seed)

    print(f"Dataset: split={args.split}  n={len(ds)}  version={args.version}")
    print("\n--- Random sample dump ---")
    for _ in range(min(args.count, len(ds))):
        idx = int(rng.integers(0, len(ds)))
        dump_sample(idx, ds[idx], thr=float(args.threshold))
        print()

    # Targeted search for interesting cases
    print(f"\n--- Targeted dump: find={args.find} (up to {args.find_count}) ---")
    found = 0
    # Scan in index order (deterministic)
    for i in range(len(ds)):
        s = ds[i]
        labels = np.asarray(s["labels"], dtype=np.float64)
        mask = np.asarray(s["label_mask"], dtype=np.uint8)

        x_ok = mask[X_IDX] != 0
        y_ok = mask[Y_IDX] != 0
        p_ok = mask[P_IDX] != 0
        p_val = float(labels[P_IDX]) if p_ok else 0.0

        match = False
        if args.find == "any":
            match = True
        elif args.find == "pressure_missing":
            match = (not p_ok)
        elif args.find == "pressure_zero":
            match = (p_ok and abs(p_val) <= 1e-12)
        elif args.find == "xy_missing":
            match = (not x_ok) or (not y_ok)

        if match:
            dump_sample(i, s, thr=float(args.threshold))
            print()
            found += 1
            if found >= int(args.find_count):
                break

    if found == 0:
        print("No matching samples found for the requested --find filter.")
        if args.find in ("pressure_missing", "pressure_zero"):
            print(
                "This strongly suggests your exported dataset has pressure present and non-zero for all labeled frames, "
                "so 'touch from pressure' will have no negatives unless you add them explicitly or use a different signal."
            )

    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
