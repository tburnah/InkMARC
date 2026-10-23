from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageDraw

from ml.inkloc.data.inkmarc import INKMARC_SRC_DEFAULT
from ml.inkloc.utils.config import load_config


def ensure_inkmarc_imports():
    src = str(INKMARC_SRC_DEFAULT)
    if src not in sys.path:
        sys.path.insert(0, src)
    from data.dataset_core_v2 import InkMARCIndexedDataset
    from data.label_schema import P_IDX, extract_xy_touch
    from data.transforms.homography_warp import warp_to_canonical
    from data.video_io_v2 import VideoFrameReader

    return InkMARCIndexedDataset, P_IDX, extract_xy_touch, warp_to_canonical, VideoFrameReader


def derive_state(
    labels: np.ndarray,
    label_mask: np.ndarray,
    *,
    pressure_touch_threshold: float,
    extract_xy_touch,
) -> dict:
    touch, _touch_mask, x, y, xy_mask = extract_xy_touch(
        labels,
        label_mask,
        pressure_touch_threshold=pressure_touch_threshold,
    )
    state = 2 if touch > 0.5 and xy_mask > 0.5 else 1 if xy_mask > 0.5 else 0
    return {
        "state": int(state),
        "touch": float(touch),
        "x": float(x),
        "y": float(y),
        "xy_mask": float(xy_mask),
    }


def fetch_images(raw_sample: dict, *, cfg: dict, split: str, video_reader, warp_to_canonical) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(raw_sample["corners"], dtype=np.float32)
    meta = raw_sample["meta"]
    frame_idx = int(meta.get("frame_idx", 0))
    variant_rel = meta.get("video_rel_path")
    file_id = meta.get("file_id")
    data_cfg = cfg["data"]

    if variant_rel:
        frame_bgr = video_reader.get_frame_from_variant(str(variant_rel), frame_idx)
    else:
        frame_bgr = video_reader.get_frame(str(file_id), frame_idx, split)

    warp = warp_to_canonical(
        frame_bgr,
        corners,
        out_w=int(cfg["model"]["input_width"]),
        out_h=int(cfg["model"]["input_height"]),
        margin_frac=float(data_cfg.get("margin_frac", 0.10)),
    )
    original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(warp["img_surf_bgr"], cv2.COLOR_BGR2RGB)
    return original_rgb, warped_rgb


def image_diff_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a16 = a.astype(np.int16)
    b16 = b.astype(np.int16)
    abs_diff = np.abs(a16 - b16)
    return {
        "exact_match": bool(np.array_equal(a, b)),
        "mae": float(abs_diff.mean()),
        "max_abs_diff": int(abs_diff.max()),
        "rmse": float(np.sqrt(np.mean((a16 - b16) ** 2))),
        "equal_pixel_ratio": float(np.mean(np.all(a == b, axis=2))),
    }


def annotate(image_rgb: np.ndarray, text_lines: list[str]) -> Image.Image:
    img = Image.fromarray(image_rgb)
    pad_h = 18 * len(text_lines) + 8
    canvas = Image.new("RGB", (img.width, img.height + pad_h), color=(0, 0, 0))
    canvas.paste(img, (0, pad_h))
    draw = ImageDraw.Draw(canvas)
    y = 4
    for line in text_lines:
        draw.text((4, y), line, fill=(255, 255, 255))
        y += 18
    return canvas


def make_pair_sheet(row: dict, prev_original: np.ndarray, curr_original: np.ndarray, prev_warped: np.ndarray, curr_warped: np.ndarray, out_path: Path) -> None:
    left_top = annotate(
        prev_original,
        [
            f"prev frame={row['prev_frame_idx']} state={row['prev_state_name']} pressure={row['prev_pressure']:.1f}",
            f"xy=({row['prev_x']:.1f},{row['prev_y']:.1f})",
        ],
    )
    right_top = annotate(
        curr_original,
        [
            f"curr frame={row['curr_frame_idx']} state={row['curr_state_name']} pressure={row['curr_pressure']:.1f}",
            f"xy=({row['curr_x']:.1f},{row['curr_y']:.1f})",
        ],
    )
    left_bottom = annotate(
        prev_warped,
        [
            f"warped prev mae={row['warped_mae']:.4f}",
            f"orig mae={row['original_mae']:.4f}",
        ],
    )
    right_bottom = annotate(
        curr_warped,
        [
            f"exact warped={row['warped_exact_match']}",
            f"exact orig={row['original_exact_match']}",
        ],
    )

    cell_w = max(img.width for img in (left_top, right_top, left_bottom, right_bottom))
    cell_h = max(img.height for img in (left_top, right_top, left_bottom, right_bottom))
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 2), color=(24, 24, 24))
    for idx, img in enumerate((left_top, right_top, left_bottom, right_bottom)):
        x = (idx % 2) * cell_w
        y = (idx // 2) * cell_h
        sheet.paste(img, (x, y))
    sheet.save(out_path)


def summarize_thresholds(rows: list[dict], key: str, thresholds: list[float]) -> dict[str, int]:
    values = np.asarray([float(r[key]) for r in rows], dtype=np.float32)
    return {f"<= {thr:g}": int(np.sum(values <= thr)) for thr in thresholds}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/t/tburnah/ml/configs/loctouch_lite.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--index-version", default=None)
    parser.add_argument("--out-dir", default="/home/t/tburnah/ml/artifacts/transition_duplicate_audit")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--near-duplicate-mae", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    split = str(args.split or data_cfg.get("eval_split", "val"))
    index_version = str(args.index_version or data_cfg.get("index_version", "focused"))
    out_dir = Path(args.out_dir)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    InkMARCIndexedDataset, P_IDX, extract_xy_touch, warp_to_canonical, VideoFrameReader = ensure_inkmarc_imports()
    raw_ds = InkMARCIndexedDataset(
        data_cfg["dataset_root"],
        index_version=index_version,
        split=split,
        require_labels=bool(data_cfg.get("require_labels", False)),
        transform=None,
    )
    records_meta = raw_ds._records

    manifest_path = Path(data_cfg["dataset_root"]) / "indices" / f"manifest_{index_version}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_reader = VideoFrameReader(Path(data_cfg["dataset_root"]), manifest, cache_size=int(data_cfg.get("video_cache_size", 16)))

    by_sequence: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for raw_pos, rec in enumerate(records_meta):
        seq_key = str(rec.get("file_id") or rec.get("rel_path"))
        by_sequence[seq_key].append((int(rec.get("frame_idx", 0)), raw_pos))

    transition_pairs: list[tuple[str, int, int]] = []
    state_counter = Counter()
    for seq_key, items in by_sequence.items():
        items.sort()
        prev = None
        for frame_idx, raw_pos in items:
            sample = raw_ds[raw_pos]
            state_info = derive_state(
                np.asarray(sample["labels"], dtype=np.float32),
                np.asarray(sample["label_mask"], dtype=np.uint8),
                pressure_touch_threshold=float(data_cfg.get("touch_pressure_threshold", 0.0)),
                extract_xy_touch=extract_xy_touch,
            )
            state = int(state_info["state"])
            state_counter[state] += 1
            if prev is not None and frame_idx == prev["frame_idx"] + 1:
                if (prev["state"], state) in ((1, 2), (2, 1)):
                    transition_pairs.append((seq_key, prev["raw_pos"], raw_pos))
            prev = {"frame_idx": frame_idx, "raw_pos": raw_pos, "state": state}

    rows: list[dict] = []
    for pair_idx, (seq_key, prev_pos, curr_pos) in enumerate(transition_pairs):
        prev_sample = raw_ds[prev_pos]
        curr_sample = raw_ds[curr_pos]
        prev_labels = np.asarray(prev_sample["labels"], dtype=np.float32)
        prev_mask = np.asarray(prev_sample["label_mask"], dtype=np.uint8)
        curr_labels = np.asarray(curr_sample["labels"], dtype=np.float32)
        curr_mask = np.asarray(curr_sample["label_mask"], dtype=np.uint8)

        prev_info = derive_state(
            prev_labels,
            prev_mask,
            pressure_touch_threshold=float(data_cfg.get("touch_pressure_threshold", 0.0)),
            extract_xy_touch=extract_xy_touch,
        )
        curr_info = derive_state(
            curr_labels,
            curr_mask,
            pressure_touch_threshold=float(data_cfg.get("touch_pressure_threshold", 0.0)),
            extract_xy_touch=extract_xy_touch,
        )

        prev_pressure = float(prev_labels[P_IDX]) if prev_mask[P_IDX] != 0 else float("nan")
        curr_pressure = float(curr_labels[P_IDX]) if curr_mask[P_IDX] != 0 else float("nan")

        prev_original, prev_warped = fetch_images(prev_sample, cfg=cfg, split=split, video_reader=video_reader, warp_to_canonical=warp_to_canonical)
        curr_original, curr_warped = fetch_images(curr_sample, cfg=cfg, split=split, video_reader=video_reader, warp_to_canonical=warp_to_canonical)

        orig_metrics = image_diff_metrics(prev_original, curr_original)
        warped_metrics = image_diff_metrics(prev_warped, curr_warped)

        if prev_info["xy_mask"] > 0.5 and curr_info["xy_mask"] > 0.5:
            xy_delta = float(np.linalg.norm(np.asarray([curr_info["x"] - prev_info["x"], curr_info["y"] - prev_info["y"]])))
        else:
            xy_delta = float("nan")

        row = {
            "pair_index": pair_idx,
            "sequence_id": seq_key,
            "participant_id": int(curr_sample["meta"].get("participant_id", -1) or -1),
            "prev_dataset_index": prev_pos,
            "curr_dataset_index": curr_pos,
            "prev_frame_idx": int(prev_sample["meta"].get("frame_idx", 0)),
            "curr_frame_idx": int(curr_sample["meta"].get("frame_idx", 0)),
            "transition_type": f"{'hover' if prev_info['state'] == 1 else 'touch'}_to_{'hover' if curr_info['state'] == 1 else 'touch'}",
            "prev_state": int(prev_info["state"]),
            "curr_state": int(curr_info["state"]),
            "prev_state_name": "hover" if prev_info["state"] == 1 else "touch",
            "curr_state_name": "hover" if curr_info["state"] == 1 else "touch",
            "prev_pressure": prev_pressure,
            "curr_pressure": curr_pressure,
            "pressure_delta": curr_pressure - prev_pressure,
            "prev_x": prev_info["x"],
            "prev_y": prev_info["y"],
            "curr_x": curr_info["x"],
            "curr_y": curr_info["y"],
            "xy_delta_tablet": xy_delta,
            "original_exact_match": orig_metrics["exact_match"],
            "warped_exact_match": warped_metrics["exact_match"],
            "original_mae": orig_metrics["mae"],
            "warped_mae": warped_metrics["mae"],
            "original_rmse": orig_metrics["rmse"],
            "warped_rmse": warped_metrics["rmse"],
            "original_max_abs_diff": orig_metrics["max_abs_diff"],
            "warped_max_abs_diff": warped_metrics["max_abs_diff"],
            "original_equal_pixel_ratio": orig_metrics["equal_pixel_ratio"],
            "warped_equal_pixel_ratio": warped_metrics["equal_pixel_ratio"],
            "likely_duplicate": bool(
                warped_metrics["exact_match"]
                or orig_metrics["exact_match"]
                or warped_metrics["mae"] <= float(args.near_duplicate_mae)
            ),
        }
        rows.append(row)

        row_dir = examples_dir / f"{pair_idx:04d}_{row['transition_type']}_{row['participant_id']}_{row['prev_frame_idx']}_{row['curr_frame_idx']}"
        row_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prev_original).save(row_dir / "prev_original.png")
        Image.fromarray(curr_original).save(row_dir / "curr_original.png")
        Image.fromarray(prev_warped).save(row_dir / "prev_warped.png")
        Image.fromarray(curr_warped).save(row_dir / "curr_warped.png")
        make_pair_sheet(row, prev_original, curr_original, prev_warped, curr_warped, row_dir / "pair_sheet.png")
        (row_dir / "meta.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    rows.sort(key=lambda item: (item["warped_mae"], item["original_mae"]))
    summary = {
        "split": split,
        "index_version": index_version,
        "dataset_size": len(raw_ds),
        "sequence_count": len(by_sequence),
        "state_counts": {str(k): int(v) for k, v in sorted(state_counter.items())},
        "hover_touch_adjacent_transition_count": len(rows),
        "transition_type_counts": dict(Counter(r["transition_type"] for r in rows)),
        "warped_exact_match_count": int(sum(bool(r["warped_exact_match"]) for r in rows)),
        "original_exact_match_count": int(sum(bool(r["original_exact_match"]) for r in rows)),
        "likely_duplicate_count": int(sum(bool(r["likely_duplicate"]) for r in rows)),
        "warped_mae_threshold_counts": summarize_thresholds(rows, "warped_mae", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]),
        "original_mae_threshold_counts": summarize_thresholds(rows, "original_mae", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]),
        "top_suspicious_pairs": rows[: min(int(args.top_k), len(rows))],
    }

    csv_path = out_dir / "transition_duplicate_summary.csv"
    jsonl_path = out_dir / "transition_duplicate_summary.jsonl"
    report_path = out_dir / "transition_duplicate_report.md"
    summary_json_path = out_dir / "transition_duplicate_report.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["pair_index"])
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report_lines = [
        "# Transition Duplicate Audit",
        "",
        f"- Split: `{split}`",
        f"- Index version: `{index_version}`",
        f"- Dataset size: {len(raw_ds)}",
        f"- Sequences: {len(by_sequence)}",
        f"- Hover/touch adjacent transitions inspected: {len(rows)}",
        f"- Warped exact matches: {summary['warped_exact_match_count']}",
        f"- Original exact matches: {summary['original_exact_match_count']}",
        f"- Likely duplicates (warped mae <= {args.near_duplicate_mae} or exact): {summary['likely_duplicate_count']}",
        "",
        "## Threshold counts",
        f"- Warped mae: {summary['warped_mae_threshold_counts']}",
        f"- Original mae: {summary['original_mae_threshold_counts']}",
        "",
        "## Transition types",
        f"- Counts: {summary['transition_type_counts']}",
        "",
        "## Most suspicious pairs",
    ]
    for row in rows[: min(int(args.top_k), len(rows))]:
        report_lines.append(
            f"- pair {row['pair_index']}: seq={row['sequence_id']} frames={row['prev_frame_idx']}->{row['curr_frame_idx']} "
            f"{row['transition_type']} warped_mae={row['warped_mae']:.4f} original_mae={row['original_mae']:.4f} "
            f"pressures={row['prev_pressure']:.1f}->{row['curr_pressure']:.1f}"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "report": str(report_path),
        "summary_json": str(summary_json_path),
        "transition_count": len(rows),
        "likely_duplicate_count": summary["likely_duplicate_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
