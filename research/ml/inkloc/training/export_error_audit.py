from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2  # type: ignore
import numpy as np
import onnxruntime as ort
import torch
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

    return InkMARCIndexedDataset, extract_xy_touch, P_IDX, warp_to_canonical, VideoFrameReader


def maybe_subset_indices(length: int, max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or max_samples >= length:
        return list(range(length))
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=gen)[:max_samples].tolist()


def overlay_xy(image_rgb: np.ndarray, xy: tuple[float, float] | None, color: tuple[int, int, int]) -> np.ndarray:
    img = image_rgb.copy()
    if xy is None:
        return img
    h, w = img.shape[:2]
    x = int(round(float(xy[0]) * (w - 1)))
    y = int(round(float(xy[1]) * (h - 1)))
    cv2.circle(img, (x, y), 8, color, 2)
    cv2.line(img, (x - 10, y), (x + 10, y), color, 1)
    cv2.line(img, (x, y - 10), (x, y + 10), color, 1)
    return img


def make_strip(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    pil_images = []
    for img, label in zip(images, labels):
        canvas = Image.fromarray(img)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 18), fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 255))
        pil_images.append(canvas)
    width = sum(img.width for img in pil_images)
    height = max(img.height for img in pil_images)
    out = Image.new("RGB", (width, height))
    x = 0
    for img in pil_images:
        out.paste(img, (x, 0))
        x += img.width
    return out


def make_contact_sheet(records: list[dict], image_key: str, title: str, out_path: Path, cols: int = 8) -> None:
    if not records:
        return
    thumbs = []
    for rec in records:
        img = Image.open(rec[image_key]).convert("RGB").resize((160, 120))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 160, 18), fill=(0, 0, 0))
        draw.text((4, 3), rec["sheet_label"], fill=(255, 255, 255))
        thumbs.append(img)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 160, rows * 120 + 24), color=(24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), title, fill=(255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 160
        y = 24 + (idx // cols) * 120
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)


def state_name(state: int) -> str:
    return ["background", "hover", "touch"][int(state)]


def decision_rule(state_probs: np.ndarray, rule_doc: dict) -> int:
    p_bg, p_hover, p_touch = state_probs.tolist()
    rule = rule_doc["rule"]
    params = rule_doc["params"]
    if rule != "A_simple":
        raise ValueError(f"Audit currently supports A_simple decision config, got {rule}")
    if p_touch >= float(params["touch_threshold"]):
        return 2
    return 1 if p_hover >= p_bg else 0


def summarize_distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/t/tburnah/ml/configs/loctouch_lite.yaml")
    parser.add_argument("--checkpoint", default="/home/t/tburnah/ml/artifacts/checkpoints/decision_selected.pt")
    parser.add_argument("--onnx", default="/home/t/tburnah/ml/artifacts/loctouch_lite_decision_selected.onnx")
    parser.add_argument("--decision-config", default="/home/t/tburnah/ml/artifacts/calibration/android_decision_config.json")
    parser.add_argument("--out-dir", default="/home/t/tburnah/ml/artifacts/audit")
    parser.add_argument("--transition-window", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    decision_doc = json.loads(Path(args.decision_config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    examples_dir = out_dir / "examples"
    contact_dir = out_dir / "contact_sheets"
    examples_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    InkMARCIndexedDataset, extract_xy_touch, P_IDX, warp_to_canonical, VideoFrameReader = ensure_inkmarc_imports()
    data_cfg = cfg["data"]
    split = str(data_cfg.get("eval_split", "val"))
    index_version = str(data_cfg.get("index_version", "focused"))
    raw_ds = InkMARCIndexedDataset(
        data_cfg["dataset_root"],
        index_version=index_version,
        split=split,
        require_labels=bool(data_cfg.get("require_labels", False)),
        transform=None,
    )
    subset_indices = maybe_subset_indices(len(raw_ds), int(cfg["training"].get("val_max_samples", 0)), int(cfg.get("seed", 7)) + 1)
    records_meta = raw_ds._records  # upstream dataset record metadata, no IO
    manifest_path = Path(data_cfg["dataset_root"]) / "indices" / f"manifest_{index_version}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_reader = VideoFrameReader(Path(data_cfg["dataset_root"]), manifest, cache_size=int(data_cfg.get("video_cache_size", 16)))

    seq_to_positions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for raw_pos in subset_indices:
        rec = records_meta[raw_pos]
        seq_key = str(rec.get("file_id") or rec.get("rel_path"))
        seq_to_positions[seq_key].append((int(rec.get("frame_idx", 0)), raw_pos))
    seq_sorted = {k: sorted(v) for k, v in seq_to_positions.items()}

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    summary_rows = []
    transition_touch_frames = 0
    transition_error_frames = 0
    correct_touch_pressures = []
    missed_touch_pressures = []
    hover_as_touch_pressures = []
    category_buckets: dict[str, list[dict]] = defaultdict(list)
    correct_hover = []
    correct_touch = []

    def fetch_warped(raw_sample: dict) -> tuple[np.ndarray, np.ndarray]:
        corners = np.asarray(raw_sample["corners"], dtype=np.float32)
        meta = raw_sample["meta"]
        file_id = meta.get("file_id")
        frame_idx = int(meta.get("frame_idx", 0))
        variant_rel = meta.get("video_rel_path")
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
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(warp["img_surf_bgr"], cv2.COLOR_BGR2RGB)

    for rank, raw_pos in enumerate(subset_indices):
        raw_sample = raw_ds[raw_pos]
        labels = np.asarray(raw_sample["labels"], dtype=np.float32)
        label_mask = np.asarray(raw_sample["label_mask"], dtype=np.uint8)
        touch, _touch_mask, x, y, xy_mask = extract_xy_touch(labels, label_mask, pressure_touch_threshold=float(data_cfg.get("touch_pressure_threshold", 0.0)))
        pressure = float(labels[P_IDX]) if label_mask[P_IDX] != 0 else float("nan")
        original_rgb, warped_rgb = fetch_warped(raw_sample)
        image = warped_rgb.astype(np.float32) / 255.0
        ort_out = sess.run(None, {"image": image.transpose(2, 0, 1)[None, ...].astype(np.float32)})
        xy_pred = ort_out[0][0]
        state_logits = ort_out[4][0]
        state_probs = np.exp(state_logits - np.max(state_logits))
        state_probs = state_probs / np.clip(state_probs.sum(), 1e-8, None)
        pred_state = decision_rule(state_probs, decision_doc)

        if xy_mask > 0.5:
            gt_xy = (float(np.clip(x / 44800.0, 0.0, 1.0)), float(np.clip(y / 29600.0, 0.0, 1.0)))
            xy_error_px = float(np.linalg.norm((np.asarray(xy_pred) - np.asarray(gt_xy)) * np.asarray([480.0, 320.0])))
        else:
            gt_xy = None
            xy_error_px = float("nan")

        true_state = 2 if touch > 0.5 and xy_mask > 0.5 else 1 if xy_mask > 0.5 else 0
        seq_key = str(raw_sample["meta"].get("file_id") or raw_sample["meta"].get("rel_path"))
        frame_idx = int(raw_sample["meta"].get("frame_idx", 0))
        neighbor_map = {f: p for f, p in seq_sorted[seq_key]}
        prev_pos = neighbor_map.get(frame_idx - 1)
        next_pos = neighbor_map.get(frame_idx + 1)

        def transition_adjacent() -> bool:
            state_here = true_state
            for delta in range(1, args.transition_window + 1):
                for other_pos in (neighbor_map.get(frame_idx - delta), neighbor_map.get(frame_idx + delta)):
                    if other_pos is None:
                        continue
                    other = raw_ds[other_pos]
                    olab = np.asarray(other["labels"], dtype=np.float32)
                    omask = np.asarray(other["label_mask"], dtype=np.uint8)
                    otouch, _m, _x, _y, oxy = extract_xy_touch(olab, omask, pressure_touch_threshold=float(data_cfg.get("touch_pressure_threshold", 0.0)))
                    ostate = 2 if otouch > 0.5 and oxy > 0.5 else 1 if oxy > 0.5 else 0
                    if ostate != state_here:
                        return True
            return False

        is_transition_adj = transition_adjacent()
        if true_state == 2 and is_transition_adj:
            transition_touch_frames += 1

        row = {
            "dataset_index": raw_pos,
            "participant_id": int(raw_sample["meta"].get("participant_id", -1) or -1),
            "sequence_id": seq_key,
            "frame_idx": frame_idx,
            "pressure": pressure,
            "true_state": true_state,
            "pred_state": pred_state,
            "touch_probability": float(state_probs[2]),
            "p_hover": float(state_probs[1]),
            "p_background": float(state_probs[0]),
            "decision_threshold": float(decision_doc["params"]["touch_threshold"]),
            "xy_error_px": xy_error_px,
            "has_xy": int(xy_mask > 0.5),
            "transition_adjacent": is_transition_adj,
        }

        error_type = None
        if true_state == 1 and pred_state == 2:
            error_type = "hover_as_touch"
            if not math.isnan(pressure):
                hover_as_touch_pressures.append(pressure)
        elif true_state == 2 and pred_state == 1:
            error_type = "touch_as_hover"
            if not math.isnan(pressure):
                missed_touch_pressures.append(pressure)
        elif true_state == 2 and pred_state == 0:
            error_type = "touch_as_background"
            if not math.isnan(pressure):
                missed_touch_pressures.append(pressure)
        elif true_state == 0 and pred_state in (1, 2):
            error_type = "background_as_valid"
        elif xy_mask > 0.5 and xy_error_px > 96.0:
            error_type = "high_xy_error"
        elif true_state == 2 and pred_state == 2 and state_probs[2] < 0.8:
            error_type = "low_conf_correct_touch"
            if not math.isnan(pressure):
                correct_touch_pressures.append(pressure)
        elif pred_state == 2 and true_state != 2 and state_probs[2] > 0.8:
            error_type = "high_conf_wrong_touch"
        elif true_state == 2 and pred_state == 2 and not math.isnan(pressure):
            correct_touch_pressures.append(pressure)

        row["error_type"] = error_type or ""
        if error_type and is_transition_adj:
            transition_error_frames += 1

        summary_rows.append(row)
        if true_state == 1 and pred_state == 1:
            correct_hover.append(row)
        if true_state == 2 and pred_state == 2:
            correct_touch.append(row)
        if error_type:
            category_buckets[error_type].append(row)

        if error_type:
            item_dir = examples_dir / f"{error_type}_{raw_pos:06d}"
            item_dir.mkdir(parents=True, exist_ok=True)
            gt_overlay = overlay_xy(warped_rgb, gt_xy, (0, 255, 0))
            pred_overlay = overlay_xy(warped_rgb, (float(xy_pred[0]), float(xy_pred[1])), (255, 0, 0))
            Image.fromarray(warped_rgb).save(item_dir / "warped.png")
            Image.fromarray(gt_overlay).save(item_dir / "overlay_gt.png")
            Image.fromarray(pred_overlay).save(item_dir / "overlay_pred.png")
            Image.fromarray(original_rgb).save(item_dir / "original.png")
            strip_images = []
            strip_labels = []
            for label, pos in [("prev", prev_pos), ("curr", raw_pos), ("next", next_pos)]:
                if pos is None:
                    continue
                sample = raw_ds[pos]
                _orig, warp_img = fetch_warped(sample)
                strip_images.append(warp_img)
                strip_labels.append(label)
            if strip_images:
                make_strip(strip_images, strip_labels).save(item_dir / "contact_strip.png")
            meta_doc = dict(row)
            meta_doc["gt_xy"] = gt_xy
            meta_doc["pred_xy"] = [float(xy_pred[0]), float(xy_pred[1])]
            meta_doc["decision_rule"] = decision_doc
            (item_dir / "meta.json").write_text(json.dumps(meta_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            row["example_dir"] = str(item_dir)
            row["warped_path"] = str(item_dir / "warped.png")
            row["overlay_gt_path"] = str(item_dir / "overlay_gt.png")
            row["overlay_pred_path"] = str(item_dir / "overlay_pred.png")

    # Export summaries
    csv_path = out_dir / "error_audit_summary.csv"
    jsonl_path = out_dir / "error_audit_summary.jsonl"
    fieldnames = sorted({k for row in summary_rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Contact sheets
    random.seed(int(cfg.get("seed", 7)))
    top_specs = [
        ("hover_as_touch", category_buckets["hover_as_touch"], "overlay_pred_path"),
        ("touch_as_hover", category_buckets["touch_as_hover"], "overlay_pred_path"),
        ("high_xy_error", sorted(category_buckets["high_xy_error"], key=lambda r: r["xy_error_px"], reverse=True), "overlay_pred_path"),
    ]
    for name, rows, image_key in top_specs:
        ranked = sorted(rows, key=lambda r: (r["touch_probability"], r.get("xy_error_px", 0.0)), reverse=True)[: args.top_k]
        for row in ranked:
            row["sheet_label"] = f"{row['participant_id']}:{row['frame_idx']} p={row['touch_probability']:.2f}"
        make_contact_sheet(ranked, image_key, name, contact_dir / f"{name}_top_{args.top_k}.png")
    for name, rows in [("correct_hover", correct_hover), ("correct_touch", correct_touch)]:
        picks = random.sample(rows, min(args.top_k, len(rows)))
        enriched = []
        for row in picks:
            example_dir = examples_dir / f"{name}_{row['dataset_index']:06d}"
            if not example_dir.exists():
                raw_sample = raw_ds[row["dataset_index"]]
                _orig, warp_img = fetch_warped(raw_sample)
                example_dir.mkdir(parents=True, exist_ok=True)
                overlay = overlay_xy(warp_img, None, (255, 255, 255))
                Image.fromarray(overlay).save(example_dir / "overlay_pred.png")
            entry = dict(row)
            entry["overlay_pred_path"] = str(example_dir / "overlay_pred.png")
            entry["sheet_label"] = f"{row['participant_id']}:{row['frame_idx']}"
            enriched.append(entry)
        make_contact_sheet(enriched, "overlay_pred_path", name, contact_dir / f"{name}_random_{args.top_k}.png")

    # Analysis
    by_participant = Counter(row["participant_id"] for row in summary_rows if row["error_type"])
    by_sequence = Counter(row["sequence_id"] for row in summary_rows if row["error_type"])
    hover_as_touch_rows = category_buckets["hover_as_touch"]
    touch_as_hover_rows = category_buckets["touch_as_hover"] + category_buckets["touch_as_background"]
    high_xy_rows = category_buckets["high_xy_error"]
    transition_error_count = sum(1 for row in summary_rows if row["error_type"] and row["transition_adjacent"])
    stable_error_count = sum(1 for row in summary_rows if row["error_type"] and not row["transition_adjacent"])
    report = {
        "num_samples": len(summary_rows),
        "error_counts": {k: len(v) for k, v in category_buckets.items()},
        "pressure_correct_touch": summarize_distribution(correct_touch_pressures),
        "pressure_touch_as_hover": summarize_distribution(missed_touch_pressures),
        "pressure_hover_as_touch": summarize_distribution(hover_as_touch_pressures),
        "transition_touch_frames": transition_touch_frames,
        "transition_error_frames": transition_error_frames,
        "transition_adjacent_errors": transition_error_count,
        "stable_errors": stable_error_count,
        "top_participants_by_errors": by_participant.most_common(8),
        "top_sequences_by_errors": by_sequence.most_common(8),
        "hover_as_touch_touch_prob_mean": float(np.mean([r["touch_probability"] for r in hover_as_touch_rows])) if hover_as_touch_rows else float("nan"),
        "touch_as_hover_touch_prob_mean": float(np.mean([r["touch_probability"] for r in touch_as_hover_rows])) if touch_as_hover_rows else float("nan"),
        "high_xy_error_touch_overlap": sum(1 for r in high_xy_rows if r["true_state"] == 2),
    }

    transition_ratio = transition_error_count / max(transition_error_count + stable_error_count, 1)
    missed_touch_pressure = report["pressure_touch_as_hover"].get("median")
    correct_touch_pressure = report["pressure_correct_touch"].get("median")
    hover_fp_mean = report["hover_as_touch_touch_prob_mean"]
    recommendation = "E. proceed to Android benchmark"
    findings = []
    if transition_ratio > 0.4:
        findings.append("Errors are concentrated near transition boundaries.")
        recommendation = "B. exclude/soft-label transition boundary frames"
    if missed_touch_pressure and correct_touch_pressure and missed_touch_pressure < correct_touch_pressure:
        findings.append("Missed touches are lower-pressure than correct touches.")
        recommendation = "A. adjust pressure/touch label threshold"
    if hover_fp_mean == hover_fp_mean and hover_fp_mean > 0.6:
        findings.append("Hover false positives are near-contact hovers with high touch confidence.")
        recommendation = "D. collect more natural hover/touch data"
    if report["high_xy_error_touch_overlap"] > max(1, len(high_xy_rows) // 2):
        findings.append("High xy error often co-occurs with touch errors.")
    if not findings:
        findings.append("Errors are mixed rather than dominated by a single obvious label pathology.")
        recommendation = "C. add temporal AOI motion"

    md_path = out_dir / "error_audit_report.md"
    md = [
        "# Error Audit Report",
        "",
        f"- Samples audited: {len(summary_rows)}",
        f"- Hover->touch errors: {len(hover_as_touch_rows)}",
        f"- Touch->hover errors: {len(category_buckets['touch_as_hover'])}",
        f"- Touch->background errors: {len(category_buckets['touch_as_background'])}",
        f"- High xy error rows: {len(high_xy_rows)}",
        "",
        "## Findings",
    ]
    for item in findings:
        md.append(f"- {item}")
    md += [
        "",
        "## Pressure",
        f"- Correct touch median pressure: {report['pressure_correct_touch'].get('median')}",
        f"- Missed touch median pressure: {report['pressure_touch_as_hover'].get('median')}",
        f"- Hover->touch median pressure: {report['pressure_hover_as_touch'].get('median')}",
        "",
        "## Transition Adjacency",
        f"- Transition-adjacent error frames: {transition_error_count}",
        f"- Stable error frames: {stable_error_count}",
        "",
        "## Dominant Sources",
        f"- Top participants by error count: {report['top_participants_by_errors']}",
        f"- Top sequences by error count: {report['top_sequences_by_errors']}",
        "",
        "## Recommendation",
        f"- {recommendation}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "report": str(md_path),
        "recommendation": recommendation,
        "findings": findings,
    }, indent=2))

    raw_ds.close()
    video_reader.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
