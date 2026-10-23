from collections import defaultdict
import torch
import numpy as np
from .temporal_stabilizer import XYStabilizer

@torch.no_grad()
def evaluate_with_phase3(model, loader, device, out_w, out_h):
    model.eval()

    dt_vals = []        # normalized Δt
    dt_px_vals = []     # pixel Δt (optional)

    # Collect per frame
    per_file = defaultdict(list)
    for batch in loader:
        images = batch["image"].to(device)
        xy_gt = batch["xy"].cpu().numpy()         # (B,2) normalized
        xy_mask = batch["xy_mask"].cpu().numpy()  # (B,) 0/1 float
        meta = batch["meta"]

        out = model(images)
        xy_pred = out["xy"].detach().cpu().numpy()  # (B,2) normalized

        for i in range(len(meta)):
            # Prefer rel_path (stable, non-null); fall back to file_id if needed
            key = meta[i].get("rel_path") or meta[i].get("file_id")
            frame_idx = meta[i].get("frame_idx")

            if key is None or frame_idx is None:
                # Fail fast and loudly during debugging; in production you could 'continue'
                # but you do NOT want these silently mixing streams.
                continue

            per_file[str(key)].append({
                "frame_idx": int(frame_idx),
                "xy_pred": xy_pred[i].astype(np.float32),
                "xy_gt": xy_gt[i].astype(np.float32),
                "xy_mask": float(xy_mask[i]),
            })

    # Apply stabiliser per file in time order
    abs_err_sum = 0.0
    abs_err_px_sum = 0.0
    denom = 0.0

    scale = np.array([out_w, out_h], dtype=np.float32)

    for _, rows in per_file.items():
        rows.sort(key=lambda r: r["frame_idx"])
        stab = XYStabilizer()

        prev_frame_idx = None
        prev_xy = None

        for r in rows:
            pred = r["xy_pred"]
            gt = r["xy_gt"]
            m = r["xy_mask"]
            fi = r["frame_idx"]

            # 1) Reset on invalid frames (do not carry state across gaps)
            if m <= 0.5:
                stab.reset()
                prev_xy = None
                prev_frame_idx = None
                continue

            # 2) Reset if frame index jumps (dropped frames, shuffled frames, etc.)
            if prev_frame_idx is not None and fi != prev_frame_idx + 1:
                stab.reset()
                prev_xy = None

            # 3) Step stabiliser using only valid frames
            stable_xy, _ = stab.step(pred, 1.0)

            # --- Δt computation ---
            if prev_xy is not None:
                dt = np.linalg.norm(stable_xy - prev_xy)
                dt_vals.append(float(dt))

                dt_px = np.linalg.norm((stable_xy - prev_xy) * scale)
                dt_px_vals.append(float(dt_px))

            prev_xy = stable_xy
            prev_frame_idx = fi

            # Evaluate ONLY where mask is valid (comparability with Phase 2)
            abs_err = np.abs(stable_xy - gt).mean()
            abs_err_px = (np.abs(stable_xy - gt) * scale).mean()

            abs_err_sum += float(abs_err)
            abs_err_px_sum += float(abs_err_px)
            denom += 1.0

            prev_frame_idx = fi

    xy_mae = abs_err_sum / max(denom, 1.0)
    xy_mae_px = abs_err_px_sum / max(denom, 1.0)

    def _percentile(vals, p):
        if not vals:
            return float("nan")
        return float(np.percentile(np.asarray(vals), p))

    metrics = {
        "dt_median": _percentile(dt_vals, 50),
        "dt_p95": _percentile(dt_vals, 95),
        "dt_px_median": _percentile(dt_px_vals, 50),
        "dt_px_p95": _percentile(dt_px_vals, 95),
    }
    
    return {
        "xy_mae_phase3": xy_mae,
        "xy_mae_px_phase3": xy_mae_px,
        "dt_median": metrics["dt_median"],
        "dt_p95": metrics["dt_p95"],
        "dt_px_median": metrics["dt_px_median"],
        "dt_px_p95": metrics["dt_px_p95"],
        "n_frames_eval": denom,
        "n_files": len(per_file),
    }
