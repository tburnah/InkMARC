#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inkdepth.datasets.stylus_dataset import StylusDataset  # noqa: E402
from inkdepth.models import build_model_from_config  # noqa: E402
from inkdepth.utils.checkpoints import load_checkpoint  # noqa: E402
from inkdepth.utils.config import load_config, parse_overrides  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RGB, DA3 depth, surface geometry, and heatmap for one frame.")
    parser.add_argument(
        "--frame",
        default="/home/t/tburnah/inkdepth_student/outputs/model_input_visuals/5dDBiG_frame_1741067569731.jpg",
    )
    parser.add_argument(
        "--corners",
        default="/home/t/tburnah/inkdepth_student/outputs/model_input_visuals/5dDBiG_frame_1741067569731_plausible_corners_resized.npy",
        help="4x2 corners in model-resized 432x672 image coordinates.",
    )
    parser.add_argument("--config", default=str(ROOT / "configs/touch_live_da3_small_geometry_original_stride4_stable_w5_p1000.yaml"))
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "outputs/location_tuning_experiments_stride2_no307_no321/best_strict_no_touch_regression_checkpoint.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/model_input_visuals/5dDBiG_frame_1741067569731_plausible_corners_da3_geometry_heatmap_softmax_aspect_restored.jpg"),
    )
    parser.add_argument("--draw-pred-xy", action="store_true")
    parser.add_argument("--draw-heatmap-argmax", action="store_true")
    parser.add_argument(
        "--hide-heatmap-below-touch",
        type=float,
        default=None,
        help="If set, blank the heatmap panels when touch probability is below this threshold.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    frame_rgb = read_rgb(Path(args.frame))
    original_h, original_w = frame_rgb.shape[:2]
    model_h, model_w = tuple(int(x) for x in cfg.data.input_size)
    rgb_model = cv2.resize(frame_rgb, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    corners_model = np.load(args.corners).astype(np.float32)
    geometry = build_geometry_maps(cfg, corners_model)

    image_np = np.concatenate([rgb_model.astype(np.float32) / 255.0, geometry.transpose(1, 2, 0)], axis=2)
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        out = model(image_t)

    touch_prob = float(torch.sigmoid(out["touch_logits"])[0].detach().cpu()) if out.get("touch_logits") is not None else float("nan")
    presence_prob = None
    state_logits = out.get("state_logits")
    if state_logits is not None:
        if state_logits.shape[-1] == 2:
            presence_prob = float(torch.softmax(state_logits, dim=-1)[0, 1].detach().cpu())
        else:
            presence_prob = float(torch.sigmoid(state_logits.reshape(-1)[0]).detach().cpu())
    xy = out.get("xy")
    xy_np = None if xy is None else xy[0].detach().cpu().numpy().astype(np.float32)

    depth = out["depth"].detach().cpu()[0, 0].numpy()
    heatmap = out["location_heatmap"].detach().cpu()[0:1] if out.get("location_heatmap") is not None else None
    heatmap_prob = heatmap_to_display_probability(heatmap, (model_h, model_w)) if heatmap is not None else np.zeros((model_h, model_w), dtype=np.float32)
    heatmap_diag = heatmap_diagnostics(heatmap, xy_np) if heatmap is not None else {}
    show_heatmap = args.hide_heatmap_below_touch is None or touch_prob >= float(args.hide_heatmap_below_touch)

    display_rgb = frame_rgb
    display_corners = draw_corners(display_rgb.copy(), scale_corners(corners_model, model_w, model_h, original_w, original_h))
    display_prediction = display_rgb.copy()
    if args.draw_pred_xy and xy_np is not None:
        display_prediction = draw_xy(display_prediction, xy_np)
    if args.draw_heatmap_argmax and heatmap_diag:
        arg_xy = np.array([heatmap_diag["argmax_x_norm"], heatmap_diag["argmax_y_norm"]], dtype=np.float32)
        display_prediction = draw_xy(display_prediction, arg_xy, color=(255, 80, 80))

    heatmap_panel = restore_aspect(colorize(heatmap_prob, cv2.COLORMAP_MAGMA), original_w, original_h)
    overlay_panel = overlay_heatmap(display_rgb, restore_aspect(heatmap_prob, original_w, original_h))
    heatmap_title = "XY head heatmap (softmax)"
    overlay_title = "Heatmap overlay"
    if not show_heatmap:
        heatmap_panel = inactive_panel(original_w, original_h, f"hidden: touch={touch_prob:.2f}")
        overlay_panel = inactive_panel(original_w, original_h, "inactive frame")
        heatmap_title = "XY heatmap hidden"
        overlay_title = "Overlay hidden"

    panels = [
        ("RGB input", display_rgb),
        ("Plausible surface corners", display_corners),
        (prediction_title(touch_prob, presence_prob), display_prediction),
        ("Live DA3 Small depth", restore_aspect(colorize(depth, cv2.COLORMAP_VIRIDIS), original_w, original_h)),
        ("Geometry: surface mask", restore_aspect(gray_rgb(geometry[0]), original_w, original_h)),
        ("Geometry: surface u", restore_aspect(colorize(geometry[1], cv2.COLORMAP_TURBO), original_w, original_h)),
        ("Geometry: surface v", restore_aspect(colorize(geometry[2], cv2.COLORMAP_TURBO), original_w, original_h)),
        (heatmap_title, heatmap_panel),
        (overlay_title, overlay_panel),
    ]

    canvas = make_grid(panels, cols=3, panel_w=original_w // 2, panel_h=original_h // 2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"saved {output}")
    print(f"source_size={original_w}x{original_h} model_size={model_w}x{model_h} output_size={canvas.shape[1]}x{canvas.shape[0]}")
    print(f"touch_prob={touch_prob:.6f}" + ("" if presence_prob is None else f" presence_prob={presence_prob:.6f}"))
    if xy_np is not None:
        print(f"xy_norm={xy_np[0]:.6f},{xy_np[1]:.6f} xy_px_original={xy_np[0] * original_w:.2f},{xy_np[1] * original_h:.2f}")
    if heatmap_diag:
        print(
            "heatmap_argmax_norm="
            f"{heatmap_diag['argmax_x_norm']:.6f},{heatmap_diag['argmax_y_norm']:.6f} "
            f"heatmap_argmax_prob={heatmap_diag['argmax_prob']:.6f} "
            f"heatmap_entropy={heatmap_diag['entropy']:.3f}"
        )


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_geometry_maps(cfg, corners: np.ndarray) -> np.ndarray:
    ds = object.__new__(StylusDataset)
    ds.cfg = cfg
    ds.input_size = tuple(int(x) for x in cfg.data.input_size)
    return ds._surface_geometry_maps({"corners": corners, "resize_attrs": {}})


def heatmap_to_display_probability(heatmap: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    hm = torch.softmax(heatmap.flatten(2), dim=-1).reshape_as(heatmap)
    hm = F.interpolate(hm, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
    return hm.astype(np.float32)


def heatmap_diagnostics(heatmap: torch.Tensor, xy: np.ndarray | None) -> dict[str, float]:
    hm = heatmap[0, 0].float()
    prob = torch.softmax(hm.flatten(), dim=0).reshape_as(hm)
    h, w = prob.shape
    max_idx = int(prob.argmax())
    y, x = divmod(max_idx, w)
    entropy = float(-(prob * prob.clamp_min(1e-12).log()).sum())
    out = {
        "argmax_x_norm": float(x / max(1, w - 1)),
        "argmax_y_norm": float(y / max(1, h - 1)),
        "argmax_prob": float(prob[y, x]),
        "entropy": entropy,
    }
    if xy is not None:
        out["softargmax_x_norm"] = float(xy[0])
        out["softargmax_y_norm"] = float(xy[1])
    return out


def normalize(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [low, high])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)


def colorize(arr: np.ndarray, cmap: int) -> np.ndarray:
    gray = (normalize(arr) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cmap), cv2.COLOR_BGR2RGB)


def gray_rgb(arr: np.ndarray) -> np.ndarray:
    gray = (normalize(arr, 0.0, 100.0) * 255.0).round().astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def restore_aspect(panel: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_LINEAR)


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    hm_rgb = colorize(heatmap, cv2.COLORMAP_MAGMA)
    alpha = np.clip(normalize(heatmap, 85.0, 100.0), 0.0, 0.75)[:, :, None]
    return np.clip(rgb.astype(np.float32) * (1.0 - alpha) + hm_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def scale_corners(corners: np.ndarray, model_w: int, model_h: int, original_w: int, original_h: int) -> np.ndarray:
    out = corners.copy().astype(np.float32)
    out[:, 0] *= float(original_w) / float(model_w)
    out[:, 1] *= float(original_h) / float(model_h)
    return out


def draw_corners(rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    pts = corners.round().astype(np.int32).reshape((-1, 1, 2))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.polylines(bgr, [pts], isClosed=True, color=(40, 255, 40), thickness=2, lineType=cv2.LINE_AA)
    for x, y in corners.round().astype(np.int32):
        cv2.circle(bgr, (int(x), int(y)), 4, (40, 255, 40), -1, lineType=cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_xy(rgb: np.ndarray, xy: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h, w = rgb.shape[:2]
    x = int(round(float(xy[0]) * (w - 1)))
    y = int(round(float(xy[1]) * (h - 1)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.drawMarker(bgr, (x, y), color[::-1], markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def inactive_panel(width: int, height: int, text: str) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = 24
    cv2.putText(
        panel,
        text,
        (18, max(28, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return panel


def prediction_title(touch_prob: float, presence_prob: float | None) -> str:
    if presence_prob is None:
        return f"Model prediction: touch={touch_prob:.2f}"
    return f"Model prediction: touch={touch_prob:.2f} presence={presence_prob:.2f}"


def make_grid(panels: list[tuple[str, np.ndarray]], cols: int, panel_w: int, panel_h: int) -> np.ndarray:
    title_h = 22
    rows = int(np.ceil(len(panels) / cols))
    canvas = np.zeros((rows * (panel_h + title_h), cols * panel_w, 3), dtype=np.uint8)
    canvas[:] = 8
    for idx, (title, image) in enumerate(panels):
        row, col = divmod(idx, cols)
        x0 = col * panel_w
        y0 = row * (panel_h + title_h)
        panel = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        cv2.putText(
            canvas,
            title,
            (x0 + 6, y0 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        canvas[y0 + title_h : y0 + title_h + panel_h, x0 : x0 + panel_w] = panel
    return canvas


if __name__ == "__main__":
    main()
