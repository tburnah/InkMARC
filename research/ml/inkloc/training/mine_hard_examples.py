from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.training.forward import model_forward
from ml.inkloc.training.train import build_dataset
from ml.inkloc.utils.config import load_config


def classify_error(state: int, pred_state: int, touch_prob: float, xy_error: float, xy_threshold: float) -> str | None:
    if state == 2 and pred_state == 1:
        return "touch_as_hover"
    if state == 1 and pred_state == 2:
        return "hover_as_touch"
    if state == 0 and pred_state > 0:
        return "background_as_valid"
    if xy_error > xy_threshold and state > 0:
        return "high_xy_error"
    if state == 2 and touch_prob < 0.5:
        return "low_conf_touch"
    return None


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/loctouch_lite.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--out", default="ml/artifacts/hard_examples/train_hard_examples.jsonl")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--xy-threshold-px", type=float, default=96.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model = LocTouchNetLite.from_config(cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    ds = build_dataset(cfg, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(train_cfg.get("num_workers", 0)) > 0,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    written = 0
    seen = 0
    with out_path.open("w", encoding="utf-8") as f:
        for bi, batch in enumerate(loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model_forward(model, batch)
            pred_state = outputs["state_probs"].argmax(dim=1)
            touch_prob = outputs["state_probs"][:, 2]
            xy_error = torch.linalg.vector_norm((outputs["xy"] - batch["xy"]) * outputs["xy"].new_tensor([480.0, 320.0]), dim=1)
            bs = int(batch["image"].shape[0])
            for i in range(bs):
                seen += 1
                state = int(batch["state"][i].item())
                pred = int(pred_state[i].item())
                err = classify_error(state, pred, float(touch_prob[i].item()), float(xy_error[i].item()), args.xy_threshold_px)
                if err is None:
                    continue
                counts[err] = counts.get(err, 0) + 1
                rec = {
                    "dataset_pos": seen - 1,
                    "dataset_index": int(batch.get("dataset_index", torch.full((bs,), -1, device=device))[i].item()),
                    "participant_id": int(batch.get("participant_id", torch.full((bs,), -1, device=device))[i].item()),
                    "sequence_id": int(batch.get("sequence_id", torch.full((bs,), -1, device=device))[i].item()),
                    "frame_idx": int(batch.get("frame_idx", torch.full((bs,), -1, device=device))[i].item()),
                    "true_state": state,
                    "predicted_state": pred,
                    "touch_probability": float(touch_prob[i].item()),
                    "xy_error_px": float(xy_error[i].item()),
                    "error_type": err,
                }
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                written += 1
            if args.max_batches > 0 and bi >= args.max_batches:
                break
    print(json.dumps({"out": str(out_path), "seen": seen, "written": written, "counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
