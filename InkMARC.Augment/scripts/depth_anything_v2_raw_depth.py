#!/usr/bin/env python3
"""Generate raw-frame depth maps with Depth Anything V2 or V3 Small."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    print(
        f"[ERROR] Missing Python package '{exc.name}'. Install Depth-Anything-V2 "
        "requirements in the Python environment passed via --raw-depth-python.",
        file=sys.stderr,
    )
    raise SystemExit(1)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

STYLUS_MAX_X = 44800.0
STYLUS_MAX_Y = 29600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--frames", nargs="*", type=int, default=None)
    parser.add_argument("--repo-root", default=os.environ.get("DEPTH_ANYTHING_V2_ROOT", ""))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--model-id", default="depth-anything/DA3-SMALL")
    parser.add_argument("--video-mode", choices=("per-frame", "streaming"), default="per-frame")
    parser.add_argument("--streaming-chunk-size", type=int, default=60)
    parser.add_argument("--streaming-overlap", type=int, default=30)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--input-size", type=int, default=322)
    parser.add_argument("--device", default="")
    parser.add_argument("--gradient-mask-threshold", type=int, default=16)
    parser.add_argument("--tilt-gradient-mask-threshold", type=int, default=8)
    parser.add_argument("--secondary-left-strip-width", type=int, default=10)
    parser.add_argument("--secondary-left-profile-smooth", type=int, default=15)
    parser.add_argument("--dynamic-gradient-threshold-min-gap", type=int, default=24)
    parser.add_argument("--dynamic-gradient-threshold-min-pixels", type=int, default=512)
    parser.add_argument("--fallback-local-vertical-radius", type=int, default=150)
    parser.add_argument("--fallback-local-threshold", type=int, default=36)
    parser.add_argument("--h5-sidecar", default="")
    parser.add_argument("--mask-output-dir", default="")
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--mask-raw-stream", action="store_true")
    parser.add_argument("--mask-difference-raw-stream", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


class RawDepthTiming:
    def __init__(self, enabled: bool, device: str):
        self.enabled = enabled
        self.device = device
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def sync(self) -> None:
        if self.enabled and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, name: str, sync_cuda: bool = False):
        if not self.enabled:
            yield
            return
        if sync_cuda:
            self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                self.sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.totals[name] = self.totals.get(name, 0.0) + elapsed_ms
            self.counts[name] = self.counts.get(name, 0) + 1

    def print_summary(self, label: str, frames: int) -> None:
        if not self.enabled:
            return
        total_ms = sum(self.totals.values())
        print(
            f"[PY_TIMING] {label} frames={frames} total={total_ms:.3f} ms "
            f"avg_frame={(total_ms / max(frames, 1)):.3f} ms",
            file=sys.stderr,
            flush=True,
        )
        for name, total in sorted(self.totals.items(), key=lambda item: item[1], reverse=True):
            count = self.counts.get(name, 1)
            print(
                f"[PY_TIMING]   {name} total={total:.3f} ms "
                f"avg_call={(total / max(count, 1)):.3f} ms calls={count}",
                file=sys.stderr,
                flush=True,
            )


class Mp4VideoWriter:
    def __init__(self, output: Path, fps: float, size: tuple[int, int]):
        self.output = output
        self.width, self.height = size
        self.proc: subprocess.Popen[bytes] | None = None
        self.fallback: cv2.VideoWriter | None = None
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(fps if fps > 0.0 else 30.0),
            "-i",
            "-",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        except FileNotFoundError:
            self.fallback = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps if fps > 0.0 else 30.0,
                (self.width, self.height),
                True,
            )
            if not self.fallback.isOpened():
                raise RuntimeError(f"Could not open MP4 video for writing: {output}")

    def write(self, frame_bgr: np.ndarray) -> None:
        frame = frame_bgr
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        if self.proc is not None:
            if self.proc.stdin is None:
                raise RuntimeError(f"MP4 writer is not writable: {self.output}")
            self.proc.stdin.write(frame.tobytes())
        else:
            assert self.fallback is not None
            self.fallback.write(frame)

    def release(self) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            status = self.proc.wait()
            self.proc = None
            if status != 0:
                raise RuntimeError(f"ffmpeg failed while writing MP4 video: {self.output}")
        if self.fallback is not None:
            self.fallback.release()
            self.fallback = None


def select_device(requested: str) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_repo_root(raw_root: str, model_version: str) -> Path:
    if not raw_root:
        raise RuntimeError(
            "A Depth Anything checkout is required. Pass --depth-anything-root."
        )
    root = Path(raw_root).expanduser().resolve()
    expected = root / ("src/depth_anything_3/api.py" if model_version == "v3" else "depth_anything_v2/dpt.py")
    if not expected.exists():
        raise RuntimeError(f"Depth Anything {model_version} checkout not found at {root}")
    return root


def load_model(repo_root: Path, checkpoint: Path, encoder: str, device: str):
    sys.path.insert(0, str(repo_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


class DA3SmallAdapter:
    """Expose DA3 Small through the infer_image contract used below."""

    def __init__(self, repo_root: Path, model_id: str, device: str):
        sys.path.insert(0, str(repo_root / "src"))
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(model_id).to(device).eval()
        self.device = torch.device(device)

    def infer_image(self, frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imgs_cpu, _, _ = self.model.input_processor(
            [frame_rgb],
            extrinsics=None,
            intrinsics=None,
            process_res=input_size,
            process_res_method="lower_bound_resize",
            num_workers=1,
            sequential=True,
            print_progress=False,
        )
        imgs = imgs_cpu[:, None].to(self.device, non_blocking=True).float()
        output = self.model.forward(
            imgs,
            None,
            None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        )
        depth = output["depth"]
        if depth.ndim == 4:
            depth = depth.reshape(depth.shape[0], 1, depth.shape[-2], depth.shape[-1])
        elif depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.permute(0, 1, 4, 2, 3).reshape(
                depth.shape[0], 1, depth.shape[2], depth.shape[3])
        else:
            raise RuntimeError(f"Unexpected DA3 depth shape: {tuple(depth.shape)}")
        depth = F.interpolate(
            depth.float(),
            size=frame_bgr.shape[:2],
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        return depth.cpu().numpy()


class DA3StreamingDepthAdapter:
    """Read temporally aligned per-frame depths produced by official DA3-Streaming."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.frame_index = 0

    def infer_image(self, frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
        del input_size
        path = self.results_dir / f"frame_{self.frame_index}.npz"
        self.frame_index += 1
        if not path.exists():
            raise RuntimeError(f"Missing DA3 Streaming depth result: {path}")
        with np.load(path) as result:
            depth = np.asarray(result["depth"], dtype=np.float32)
        if depth.shape != frame_bgr.shape[:2]:
            depth = cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return depth


def prepare_da3_streaming(args: argparse.Namespace, repo_root: Path) -> DA3StreamingDepthAdapter:
    """Extract frames, run official DA3-Streaming, and expose its aligned depth sequence."""
    if args.frames:
        raise RuntimeError("DA3 Streaming only supports full-video runs without --frames")
    if args.model_version != "v3":
        raise RuntimeError("DA3 Streaming requires --model-version v3")
    chunk_size = int(args.streaming_chunk_size)
    overlap = int(args.streaming_overlap)
    if chunk_size < 2 or overlap < 1 or overlap >= chunk_size:
        raise RuntimeError("DA3 Streaming requires 0 < overlap < chunk size")

    from huggingface_hub import snapshot_download
    import yaml

    model_dir = Path(snapshot_download(args.model_id, local_files_only=True)).resolve()
    model_weights = model_dir / "model.safetensors"
    model_config = model_dir / "config.json"
    if not model_weights.exists() or not model_config.exists():
        raise RuntimeError(f"Cached DA3 model is incomplete: {model_dir}")

    stem = Path(args.video).stem
    # DA3-Streaming must run with its own repository as cwd because it imports
    # sibling modules. Resolve our work path first so changing cwd cannot make
    # the generated config, frames, or output paths disappear.
    work_dir = Path(args.outdir).expanduser().resolve() / f"_{stem}_da3_streaming"
    frames_dir = work_dir / "frames"
    results_root = work_dir / "streaming_output"
    results_dir = results_root / "results_output"
    marker = work_dir / "complete.txt"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    existing_results = len(list(results_dir.glob("frame_*.npz"))) if results_dir.exists() else 0
    if marker.exists() and expected_frames > 0 and existing_results == expected_frames:
        print(f"[INFO] reusing DA3 Streaming results: {results_dir}", file=sys.stderr, flush=True)
        return DA3StreamingDepthAdapter(results_dir)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    frames_dir.mkdir(parents=True)
    cap = cv2.VideoCapture(args.video)
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        path = frames_dir / f"frame_{written:08d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write DA3 Streaming input frame: {path}")
        written += 1
        if written == 1 or written % 250 == 0:
            print(f"[PROGRESS] extracted {written} DA3 Streaming frames", file=sys.stderr, flush=True)
    cap.release()
    if written == 0:
        raise RuntimeError("No video frames were extracted for DA3 Streaming")

    config = {
        "Weights": {"DA3": str(model_weights), "DA3_CONFIG": str(model_config), "SALAD": ""},
        "Model": {
            "chunk_size": chunk_size, "overlap": overlap, "loop_chunk_size": 20,
            "loop_enable": False, "useDBoW": False, "delete_temp_files": True,
            "align_lib": "torch", "align_method": "sim3", "scale_compute_method": "auto",
            "align_type": "dense", "ref_view_strategy": "middle",
            "ref_view_strategy_loop": "saddle_balanced", "depth_threshold": 15.0,
            "save_depth_conf_result": True, "save_debug_info": False,
            "Sparse_Align": {"keypoint_select": "orb", "keypoint_num": 5000},
            "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1.0e-9"},
            "Pointcloud_Save": {"sample_ratio": 0.0001, "conf_threshold_coef": 0.75},
        },
        "Loop": {
            "SALAD": {"image_size": [336, 336], "batch_size": 32,
                      "similarity_threshold": 0.85, "top_k": 5,
                      "use_nms": True, "nms_threshold": 25},
            "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30, "lambda_init": "1.0e-6"},
        },
    }
    config_path = work_dir / "streaming_small.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    streaming_dir = repo_root / "da3_streaming"
    command = [
        sys.executable, str(streaming_dir / "da3_streaming.py"),
        "--image_dir", str(frames_dir), "--config", str(config_path),
        "--output_dir", str(results_root),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[INFO] running official DA3 Streaming: chunk={chunk_size} overlap={overlap}", file=sys.stderr, flush=True)
    # In mask-stream mode stdout is a binary protocol consumed by the C++
    # pipeline. Keep the official streamer's verbose progress/alignment logs
    # off stdout or they corrupt the header and make the reader close its pipe.
    subprocess.run(
        command,
        cwd=streaming_dir,
        env=env,
        check=True,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    result_count = len(list(results_dir.glob("frame_*.npz")))
    if result_count != written:
        raise RuntimeError(f"DA3 Streaming produced {result_count} depths for {written} video frames")
    # The official output stores a copy of every RGB frame alongside depth. The
    # shadow pipeline only needs depth, so compact the cache and remove extracted
    # PNGs/point-cloud products before processing the output video.
    for result_path in sorted(results_dir.glob("frame_*.npz")):
        with np.load(result_path) as result:
            depth = np.asarray(result["depth"], dtype=np.float32)
        compact_path = result_path.with_suffix(".compact.npz")
        np.savez_compressed(compact_path, depth=depth)
        compact_path.replace(result_path)
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(results_root / "pcd", ignore_errors=True)
    marker.write_text(f"frames={written}\n", encoding="utf-8")
    return DA3StreamingDepthAdapter(results_dir)


def depth_to_u8(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros(depth.shape, dtype=np.uint8)
    lo = float(np.min(depth[finite]))
    hi = float(np.max(depth[finite]))
    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.uint8)
    scaled = (depth - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def depth_to_u8_percentile(depth: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Match InkDepth's DA3 per-frame percentile normalization."""
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros(depth.shape, dtype=np.uint8)
    lo, hi = np.percentile(depth[finite], [low, high])
    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.uint8)
    scaled = (depth - float(lo)) * (255.0 / float(hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def frame_prefix(index: int) -> str:
    return f"frame_{index:06d}"


def infer_depth_u8(model, frame_bgr: np.ndarray, input_size: int) -> np.ndarray:
    with torch.inference_mode():
        depth = model.infer_image(frame_bgr, input_size)
    if isinstance(model, (DA3SmallAdapter, DA3StreamingDepthAdapter)):
        return depth_to_u8_percentile(depth)
    return depth_to_u8(depth)


def extend_left_depth_gradient(depth_u8: np.ndarray, columns: int = 3) -> np.ndarray:
    sample_width = min(columns, depth_u8.shape[1])
    if sample_width <= 0:
        return np.zeros_like(depth_u8)
    row_values = np.mean(depth_u8[:, :sample_width], axis=1).astype(np.uint8)
    return np.repeat(row_values[:, None], depth_u8.shape[1], axis=1)


def smooth_vertical_profile(profile: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return profile.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(profile.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.full(window, 1.0 / float(window), dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_left_tilt_gradient(
    depth_u8: np.ndarray,
    strip_width: int = 10,
    profile_smooth: int = 15,
    surface_mask: np.ndarray | None = None,
) -> np.ndarray:
    height, width = depth_u8.shape[:2]
    sample_width = min(max(2, int(strip_width)), width)
    strip = depth_u8[:, :sample_width].astype(np.float32)

    reliable_rows = np.ones(height, dtype=bool)
    if surface_mask is not None and surface_mask.shape[:2] == (height, width):
        # A sloping surface edge may not reach the left sampling strip in the
        # first/last surface rows. Sampling the scene outside the surface there
        # creates the dark horizontal bands formerly visible in the gradient.
        # Fit only rows whose whole strip is on the surface, then extend the
        # nearest fitted value through both vertical ends of the surface.
        strip_on_surface = surface_mask[:, :sample_width] != 0
        reliable_rows = np.all(strip_on_surface, axis=1)
        if not np.any(reliable_rows):
            # Retain a useful surface-aware fit for unusually narrow/cropped
            # surfaces which never cover the complete strip.
            required_pixels = max(1, (sample_width + 1) // 2)
            reliable_rows = np.count_nonzero(strip_on_surface, axis=1) >= required_pixels

    if np.any(reliable_rows):
        reliable_y = np.flatnonzero(reliable_rows)
        reliable_profile = np.median(strip[reliable_rows], axis=1).astype(np.float32)
        profile = np.interp(
            np.arange(height, dtype=np.float32),
            reliable_y.astype(np.float32),
            reliable_profile,
        ).astype(np.float32)
    else:
        # No usable surface intersection: preserve the original left-strip
        # behaviour rather than manufacturing an arbitrary reference field.
        reliable_rows[:] = True
        profile = np.median(strip, axis=1).astype(np.float32)
    profile = smooth_vertical_profile(profile, profile_smooth)

    x_offsets = np.arange(sample_width, dtype=np.float32) - ((sample_width - 1.0) * 0.5)
    residuals = strip - profile[:, None]
    valid = reliable_rows[:, None] & (np.abs(x_offsets[None, :]) > 1.0e-6)
    slope_samples = residuals[valid] / np.broadcast_to(x_offsets, strip.shape)[valid]
    if slope_samples.size == 0:
        global_slope = 0.0
    else:
        slope_median = float(np.median(slope_samples))
        slope_mad = float(np.median(np.abs(slope_samples - slope_median)))
        max_deviation = max(0.25, 3.0 * 1.4826 * slope_mad)
        inliers = np.abs(slope_samples - slope_median) <= max_deviation
        global_slope = float(np.median(slope_samples[inliers])) if np.any(inliers) else slope_median

    full_x_offsets = np.arange(width, dtype=np.float32) - ((sample_width - 1.0) * 0.5)
    gradient = profile[:, None] + (global_slope * full_x_offsets[None, :])
    return np.clip(gradient, 0, 255).astype(np.uint8)


def depth_gradient_difference(depth_u8: np.ndarray, gradient_u8: np.ndarray) -> np.ndarray:
    diff = depth_u8.astype(np.int16) - gradient_u8.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)


def depth_gradient_mask(
    difference_u8: np.ndarray,
    threshold: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    threshold = int(np.clip(threshold, 0, 255))
    if valid_mask is None:
        return np.where(difference_u8 > threshold, 255, 0).astype(np.uint8)
    out = np.zeros(difference_u8.shape, dtype=np.uint8)
    valid = valid_mask != 0
    out[valid & (difference_u8 > threshold)] = 255
    return out


def threshold_pixel_count(
    difference_u8: np.ndarray,
    threshold: int,
    valid_mask: np.ndarray | None,
) -> int:
    selected = difference_u8 > int(np.clip(threshold, 0, 255))
    if valid_mask is not None:
        selected &= valid_mask != 0
    return int(np.count_nonzero(selected))


def dynamic_gradient_threshold(
    difference_u8: np.ndarray,
    base_threshold: int,
    valid_mask: np.ndarray | None,
    min_gap: int,
    min_pixels: int,
) -> tuple[int, bool]:
    if valid_mask is None or min_gap <= 0:
        return base_threshold, False
    values = difference_u8[valid_mask != 0]
    if values.size == 0:
        return base_threshold, False

    percentiles = np.array([90, 92, 94, 95, 96, 97, 98, 99], dtype=np.float32)
    levels = np.percentile(values, percentiles)
    best: tuple[int, float] | None = None
    for i in range(len(levels) - 1):
        gap = float(levels[i + 1] - levels[i])
        if gap < min_gap or levels[i + 1] < base_threshold + min_gap:
            continue
        candidate = int(round(float(levels[i] + levels[i + 1]) * 0.5))
        if threshold_pixel_count(difference_u8, candidate, valid_mask) < min_pixels:
            continue
        if best is None or gap > best[1]:
            best = (candidate, gap)

    if best is None:
        return base_threshold, False
    return max(base_threshold, best[0]), True


def use_fallback_local_gate(
    frame_index: int,
    image_shape: tuple[int, int],
    stylus_trace: "StylusTrace | None",
    previous_centroid: tuple[float, float] | None,
) -> bool:
    if previous_centroid is None or stylus_trace is None:
        return False
    return not stylus_trace.frame_points(frame_index, image_shape)


def fallback_local_threshold(
    threshold: int,
    difference_u8: np.ndarray,
    valid_mask: np.ndarray | None,
    frame_index: int,
    image_shape: tuple[int, int],
    stylus_trace: "StylusTrace | None",
    previous_centroid: tuple[float, float] | None,
    local_threshold: int,
    min_pixels: int,
) -> int:
    if not use_fallback_local_gate(frame_index, image_shape, stylus_trace, previous_centroid):
        return threshold
    candidate = max(threshold, int(np.clip(local_threshold, 0, 255)))
    if threshold_pixel_count(difference_u8, candidate, valid_mask) < min_pixels:
        return threshold
    return candidate


def apply_fallback_local_vertical_gate(
    mask_u8: np.ndarray,
    frame_index: int,
    stylus_trace: "StylusTrace | None",
    previous_centroid: tuple[float, float] | None,
    vertical_radius: int,
    min_pixels: int,
) -> np.ndarray:
    if vertical_radius <= 0:
        return mask_u8
    if not use_fallback_local_gate(frame_index, mask_u8.shape, stylus_trace, previous_centroid):
        return mask_u8
    center_y = int(round(previous_centroid[1]))
    if center_y < 0 or center_y >= mask_u8.shape[0]:
        return mask_u8
    lo = max(0, center_y - vertical_radius)
    hi = min(mask_u8.shape[0], center_y + vertical_radius + 1)
    gated = np.zeros_like(mask_u8)
    gated[lo:hi, :] = mask_u8[lo:hi, :]
    if np.count_nonzero(gated) < min_pixels:
        return mask_u8
    return gated


def make_depth_gradient_mask_for_frame(
    difference_u8: np.ndarray,
    difference_source: str,
    frame_index: int,
    stylus_trace: "StylusTrace | None",
    previous_centroid: tuple[float, float] | None,
    surface_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    min_pixels = max(1, int(args.dynamic_gradient_threshold_min_pixels))
    if difference_source == "tilt":
        threshold = int(np.clip(args.tilt_gradient_mask_threshold, 0, 255))
        skip_fallback_gate = True
        dynamic_used = False
    else:
        skip_fallback_gate = False
        base_threshold = int(np.clip(args.gradient_mask_threshold, 0, 255))
        threshold, dynamic_used = dynamic_gradient_threshold(
            difference_u8,
            base_threshold,
            surface_mask,
            int(args.dynamic_gradient_threshold_min_gap),
            min_pixels,
        )
        if not dynamic_used:
            threshold = fallback_local_threshold(
                threshold,
                difference_u8,
                surface_mask,
                frame_index,
                difference_u8.shape,
                stylus_trace,
                previous_centroid,
                args.fallback_local_threshold,
                min_pixels,
            )
    mask_u8 = depth_gradient_mask(difference_u8, threshold, surface_mask)
    if not dynamic_used and not skip_fallback_gate:
        mask_u8 = apply_fallback_local_vertical_gate(
            mask_u8,
            frame_index,
            stylus_trace,
            previous_centroid,
            args.fallback_local_vertical_radius,
            min_pixels,
        )
    return mask_u8


def choose_active_difference(
    primary_difference_u8: np.ndarray,
    fallback_difference_u8: np.ndarray,
    surface_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    min_pixels = max(1, int(args.dynamic_gradient_threshold_min_pixels))
    tilt_threshold = int(np.clip(args.tilt_gradient_mask_threshold, 0, 255))
    if threshold_pixel_count(primary_difference_u8, tilt_threshold, surface_mask) >= min_pixels:
        return primary_difference_u8, "tilt"
    return fallback_difference_u8, "legacy"


def depth_mask_inputs(
    depth_u8: np.ndarray,
    args: argparse.Namespace,
    surface_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, str]:
    secondary_gradient_u8 = robust_left_tilt_gradient(
        depth_u8,
        args.secondary_left_strip_width,
        args.secondary_left_profile_smooth,
        surface_mask,
    )
    secondary_difference_u8 = depth_gradient_difference(depth_u8, secondary_gradient_u8)
    if args.fast:
        return None, None, secondary_gradient_u8, secondary_difference_u8, "tilt"

    gradient_u8 = extend_left_depth_gradient(depth_u8)
    legacy_difference_u8 = depth_gradient_difference(depth_u8, gradient_u8)
    return gradient_u8, legacy_difference_u8, secondary_gradient_u8, secondary_difference_u8, ""


class StylusTrace:
    def __init__(
        self,
        points: np.ndarray,
        pressure: np.ndarray,
        source_size: tuple[float, float],
        corners: np.ndarray | None = None,
    ):
        self.points = points
        self.pressure = pressure
        self.source_size = source_size
        self.corners = corners

    def frame_points(self, frame_index: int, image_shape: tuple[int, int]) -> list[tuple[float, float]]:
        if frame_index < 0 or frame_index >= len(self.points) or frame_index >= len(self.pressure):
            return []
        if self.pressure[frame_index] <= 0:
            return []
        x, y = self.points[frame_index]
        if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0:
            return []

        height, width = image_shape
        candidates: list[tuple[float, float]] = []
        source_w, source_h = self.source_size
        if source_w > 0 and source_h > 0:
            candidates.append((x * width / source_w, y * height / source_h))
        candidates.append((x * width / STYLUS_MAX_X, y * height / STYLUS_MAX_Y))

        valid: list[tuple[float, float]] = []
        for px, py in candidates:
            if np.isfinite(px) and np.isfinite(py) and -width <= px <= 2 * width and -height <= py <= 2 * height:
                if not any(abs(px - qx) < 1.0 and abs(py - qy) < 1.0 for qx, qy in valid):
                    valid.append((px, py))
        return valid

    def surface_mask(self, frame_index: int, image_shape: tuple[int, int]) -> np.ndarray | None:
        if self.corners is None or frame_index < 0 or frame_index >= len(self.corners):
            return None
        height, width = image_shape
        source_w, source_h = self.source_size
        if source_w <= 0 or source_h <= 0:
            return None
        corners = np.asarray(self.corners[frame_index], dtype=np.float32)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return None
        if abs(float(cv2.contourArea(corners))) <= 1.0:
            return None
        scaled = corners.copy()
        scaled[:, 0] *= float(width) / float(source_w)
        scaled[:, 1] *= float(height) / float(source_h)
        if not np.all(np.isfinite(scaled)):
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(scaled).astype(np.int32), 255, lineType=cv2.LINE_8)
        return mask


def load_stylus_trace(video_path: str, explicit_sidecar: str = "") -> StylusTrace | None:
    if not explicit_sidecar:
        return None
    sidecar = Path(explicit_sidecar)
    if not sidecar.exists():
        return None
    try:
        import h5py
    except ModuleNotFoundError:
        print(f"[WARN] H5 sidecar exists but h5py is not installed; skipping stylus selection: {sidecar}", file=sys.stderr)
        return None

    try:
        with h5py.File(sidecar, "r") as h5:
            source_w = float(h5.attrs.get("width", 0.0))
            source_h = float(h5.attrs.get("height", 0.0))
            resize_transform = h5.get("preprocessing/resize_transform")
            if resize_transform is not None:
                output_w = float(resize_transform.attrs.get("output_width", 0.0))
                output_h = float(resize_transform.attrs.get("output_height", 0.0))
                if output_w > 0 and output_h > 0:
                    source_w, source_h = output_w, output_h
            if source_w <= 0 or source_h <= 0:
                source_w, source_h = 720.0, 1120.0
            corners = None
            if "corners" in h5:
                raw_corners = np.asarray(h5["corners"], dtype=np.float32)
                if raw_corners.ndim == 3 and raw_corners.shape[1:] == (4, 2):
                    corners = raw_corners
            if "labels" not in h5 or "label_mask" not in h5:
                if corners is None:
                    return None
                points = np.full((len(corners), 2), -1.0, dtype=np.float32)
                pressure = np.zeros(len(corners), dtype=np.float32)
                return StylusTrace(points, pressure, (source_w, source_h), corners)
            labels = np.asarray(h5["labels"], dtype=np.float32)
            label_mask = np.asarray(h5["label_mask"])
            if labels.ndim != 2 or label_mask.ndim != 2 or labels.shape[0] != label_mask.shape[0] or labels.shape[1] < 3:
                if corners is None:
                    return None
                points = np.full((len(corners), 2), -1.0, dtype=np.float32)
                pressure = np.zeros(len(corners), dtype=np.float32)
                return StylusTrace(points, pressure, (source_w, source_h), corners)
            points = np.full((labels.shape[0], 2), -1.0, dtype=np.float32)
            pressure = np.zeros(labels.shape[0], dtype=np.float32)
            valid = (label_mask[:, 0] != 0) & (label_mask[:, 1] != 0) & (label_mask[:, 2] != 0)
            valid &= np.isfinite(labels[:, 0]) & np.isfinite(labels[:, 1]) & np.isfinite(labels[:, 2])
            valid &= labels[:, 2] > 0
            points[valid, 0] = labels[valid, 0]
            points[valid, 1] = labels[valid, 1]
            pressure[valid] = labels[valid, 2]
            if not np.any(pressure > 0):
                if corners is None:
                    return None
            print(f"[INFO] Loaded H5 stylus sidecar: {sidecar}", file=sys.stderr, flush=True)
            return StylusTrace(points, pressure, (source_w, source_h), corners)
    except Exception as exc:
        print(f"[WARN] Could not read H5 stylus sidecar {sidecar}: {exc}", file=sys.stderr)
        return None


def point_to_component_distance(point: tuple[float, float], stat: np.ndarray, centroid: np.ndarray) -> float:
    px, py = point
    left = float(stat[cv2.CC_STAT_LEFT])
    top = float(stat[cv2.CC_STAT_TOP])
    right = left + float(stat[cv2.CC_STAT_WIDTH]) - 1.0
    bottom = top + float(stat[cv2.CC_STAT_HEIGHT]) - 1.0
    dx = max(left - px, 0.0, px - right)
    dy = max(top - py, 0.0, py - bottom)
    box_distance = float(np.hypot(dx, dy))
    centroid_distance = float(np.hypot(px - centroid[0], py - centroid[1]))
    return box_distance + 0.05 * centroid_distance


def keep_hand_component(
    mask_u8: np.ndarray,
    difference_u8: np.ndarray,
    frame_index: int,
    stylus_trace: StylusTrace | None,
    previous_centroid: tuple[float, float] | None,
    surface_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    if surface_mask is None and stylus_trace is not None:
        surface_mask = stylus_trace.surface_mask(frame_index, mask_u8.shape)
    if surface_mask is not None:
        surface_candidate = cv2.bitwise_and(mask_u8, surface_mask)
        if np.count_nonzero(surface_candidate) > 0:
            mask_u8 = surface_candidate

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if component_count <= 1:
        return np.zeros_like(mask_u8), previous_centroid
    if component_count == 2:
        return np.where(labels == 1, 255, 0).astype(np.uint8), (float(centroids[1][0]), float(centroids[1][1]))

    stylus_points = stylus_trace.frame_points(frame_index, mask_u8.shape) if stylus_trace is not None else []
    best_label = 1
    best_score = float("inf")
    for label in range(1, component_count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        centroid = centroids[label]
        values = difference_u8[labels == label]
        mean_value = float(np.mean(values)) if values.size else 0.0
        p90_value = float(np.percentile(values, 90)) if values.size else 0.0
        intensity_score = mean_value + (0.5 * p90_value) + (min(area, 20000.0) * 0.001)
        if stylus_points:
            score = min(point_to_component_distance(point, stats[label], centroid) for point in stylus_points)
            score -= intensity_score * 0.15
        elif previous_centroid is not None:
            score = float(np.hypot(centroid[0] - previous_centroid[0], centroid[1] - previous_centroid[1]))
            score -= intensity_score * 3.0
        else:
            score = -intensity_score
        if score < best_score:
            best_score = score
            best_label = label

    refined = np.where(labels == best_label, 255, 0).astype(np.uint8)
    return refined, (float(centroids[best_label][0]), float(centroids[best_label][1]))


def connect_stylus_tip_to_mask(
    mask_u8: np.ndarray,
    frame_index: int,
    stylus_trace: StylusTrace | None,
    tip_radius: int = 2,
    wedge_half_width: float = 8.0,
    max_bridge_distance: float = 90.0,
) -> np.ndarray:
    if stylus_trace is None:
        return mask_u8
    points = stylus_trace.frame_points(frame_index, mask_u8.shape)
    if not points:
        return mask_u8

    marked = mask_u8.copy()
    px, py = points[0]
    tip = np.array([float(px), float(py)], dtype=np.float32)
    tip_i = (int(round(px)), int(round(py)))

    ys, xs = np.nonzero(mask_u8)
    if len(xs) == 0:
        cv2.circle(marked, tip_i, tip_radius, 255, -1, lineType=cv2.LINE_8)
        return marked

    dx = xs.astype(np.float32) - tip[0]
    dy = ys.astype(np.float32) - tip[1]
    distances2 = (dx * dx) + (dy * dy)
    nearest_index = int(np.argmin(distances2))
    nearest_distance = float(np.sqrt(distances2[nearest_index]))
    if nearest_distance > max_bridge_distance:
        cv2.circle(marked, tip_i, tip_radius, 255, -1, lineType=cv2.LINE_8)
        return marked

    nearest = np.array([float(xs[nearest_index]), float(ys[nearest_index])], dtype=np.float32)
    direction = nearest - tip
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1.0:
        cv2.circle(marked, tip_i, tip_radius, 255, -1, lineType=cv2.LINE_8)
        return marked

    unit = direction / direction_norm
    normal = np.array([-unit[1], unit[0]], dtype=np.float32)
    half_width = min(wedge_half_width, max(2.0, nearest_distance * 0.35))
    base_left = nearest + (normal * half_width)
    base_right = nearest - (normal * half_width)
    polygon = np.rint(np.array([tip, base_left, base_right], dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(marked, polygon, 255, lineType=cv2.LINE_8)
    cv2.circle(marked, tip_i, tip_radius, 255, -1, lineType=cv2.LINE_8)
    marked = cv2.morphologyEx(
        marked,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return marked


def write_selected_frames(args: argparse.Namespace, model, timing: RawDepthTiming) -> int:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    written = 0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stylus_trace = load_stylus_trace(args.video, args.h5_sidecar)
    previous_centroid: tuple[float, float] | None = None
    for frame_index in sorted(set(args.frames or [])):
        with timing.measure("video_seek"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        with timing.measure("video_read"):
            ok, frame = cap.read()
        if not ok or frame is None:
            continue
        with timing.measure("inference", sync_cuda=True):
            depth_u8 = infer_depth_u8(model, frame, args.input_size)
        with timing.measure("surface_mask"):
            surface_mask = stylus_trace.surface_mask(frame_index, depth_u8.shape) if stylus_trace is not None else None
        with timing.measure("gradient_difference"):
            gradient_u8, legacy_difference_u8, secondary_gradient_u8, secondary_difference_u8, fast_source = (
                depth_mask_inputs(depth_u8, args, surface_mask)
            )
        with timing.measure("mask_threshold"):
            if fast_source:
                difference_u8, difference_source = secondary_difference_u8, fast_source
            else:
                difference_u8, difference_source = choose_active_difference(
                    secondary_difference_u8, legacy_difference_u8, surface_mask, args)
            mask_u8 = make_depth_gradient_mask_for_frame(
                difference_u8, difference_source, frame_index, stylus_trace, previous_centroid, surface_mask, args)
        with timing.measure("component_selection"):
            mask_u8, previous_centroid = keep_hand_component(
                mask_u8, difference_u8, frame_index, stylus_trace, previous_centroid, surface_mask)
        depth_output = outdir / f"{frame_prefix(frame_index)}_raw_depth.png"
        gradient_output = outdir / f"{frame_prefix(frame_index)}_raw_depth_left_gradient.png"
        difference_output = outdir / f"{frame_prefix(frame_index)}_raw_depth_gradient_difference.png"
        active_difference_output = outdir / f"{frame_prefix(frame_index)}_raw_depth_active_gradient_difference.png"
        secondary_gradient_output = outdir / f"{frame_prefix(frame_index)}_raw_depth_left_tilt_gradient.png"
        secondary_difference_output = outdir / f"{frame_prefix(frame_index)}_raw_depth_left_tilt_gradient_difference.png"
        mask_output = outdir / f"{frame_prefix(frame_index)}_raw_depth_gradient_mask.png"
        with timing.measure("write_images"):
            if not args.mask_only and not args.fast:
                if not cv2.imwrite(str(depth_output), depth_u8):
                    raise RuntimeError(f"Could not write depth map: {depth_output}")
                if not cv2.imwrite(str(gradient_output), gradient_u8):
                    raise RuntimeError(f"Could not write depth gradient map: {gradient_output}")
                if not cv2.imwrite(str(difference_output), legacy_difference_u8):
                    raise RuntimeError(f"Could not write depth gradient difference map: {difference_output}")
                if not cv2.imwrite(str(active_difference_output), difference_u8):
                    raise RuntimeError(f"Could not write active depth gradient difference map: {active_difference_output}")
                if not cv2.imwrite(str(secondary_gradient_output), secondary_gradient_u8):
                    raise RuntimeError(f"Could not write secondary depth gradient map: {secondary_gradient_output}")
                if not cv2.imwrite(str(secondary_difference_output), secondary_difference_u8):
                    raise RuntimeError(f"Could not write secondary depth gradient difference map: {secondary_difference_output}")
            if not cv2.imwrite(str(mask_output), mask_u8):
                raise RuntimeError(f"Could not write depth gradient mask: {mask_output}")
        written += 1
    cap.release()
    return written


def write_video(args: argparse.Namespace, model, timing: RawDepthTiming) -> int:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = None
    gradient_writer = None
    difference_writer = None
    mask_writer = None
    if not args.mask_only and not args.fast:
        output = outdir / f"{Path(args.video).stem}_raw_depth.mp4"
        gradient_output = outdir / f"{Path(args.video).stem}_raw_depth_left_gradient.mp4"
        difference_output = outdir / f"{Path(args.video).stem}_raw_depth_gradient_difference.mp4"
        mask_output = outdir / f"{Path(args.video).stem}_raw_depth_gradient_mask.mp4"
        try:
            writer = Mp4VideoWriter(output, fps, (width, height))
            gradient_writer = Mp4VideoWriter(gradient_output, fps, (width, height))
            difference_writer = Mp4VideoWriter(difference_output, fps, (width, height))
            mask_writer = Mp4VideoWriter(mask_output, fps, (width, height))
        except Exception:
            for opened_writer in (writer, gradient_writer, difference_writer, mask_writer):
                if opened_writer is not None:
                    opened_writer.release()
            raise

    mask_output_dir = Path(args.mask_output_dir) if args.mask_output_dir else None
    if mask_output_dir is not None:
        mask_output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    stylus_trace = load_stylus_trace(args.video, args.h5_sidecar)
    previous_centroid: tuple[float, float] | None = None
    while True:
        with timing.measure("video_read"):
            ok, frame = cap.read()
        if not ok or frame is None:
            break
        with timing.measure("inference", sync_cuda=True):
            depth_u8 = infer_depth_u8(model, frame, args.input_size)
        with timing.measure("resize"):
            if depth_u8.shape[:2] != (height, width):
                depth_u8 = cv2.resize(depth_u8, (width, height), interpolation=cv2.INTER_LINEAR)
        with timing.measure("surface_mask"):
            surface_mask = stylus_trace.surface_mask(written, depth_u8.shape) if stylus_trace is not None else None
        with timing.measure("gradient_difference"):
            gradient_u8, legacy_difference_u8, secondary_gradient_u8, secondary_difference_u8, fast_source = (
                depth_mask_inputs(depth_u8, args, surface_mask)
            )
        with timing.measure("mask_threshold"):
            if fast_source:
                difference_u8, difference_source = secondary_difference_u8, fast_source
            else:
                difference_u8, difference_source = choose_active_difference(
                    secondary_difference_u8, legacy_difference_u8, surface_mask, args)
            mask_u8 = make_depth_gradient_mask_for_frame(
                difference_u8, difference_source, written, stylus_trace, previous_centroid, surface_mask, args)
        with timing.measure("component_selection"):
            mask_u8, previous_centroid = keep_hand_component(
                mask_u8, difference_u8, written, stylus_trace, previous_centroid, surface_mask)
        with timing.measure("write_outputs"):
            if mask_output_dir is not None:
                mask_path = mask_output_dir / f"{frame_prefix(written)}_raw_depth_gradient_mask.png"
                if not cv2.imwrite(str(mask_path), mask_u8):
                    raise RuntimeError(f"Could not write depth gradient mask: {mask_path}")
                difference_path = mask_output_dir / f"{frame_prefix(written)}_raw_depth_active_gradient_difference.png"
                if not cv2.imwrite(str(difference_path), difference_u8):
                    raise RuntimeError(f"Could not write active depth difference: {difference_path}")
            if not args.mask_only and not args.fast:
                writer.write(depth_u8)
                gradient_writer.write(gradient_u8)
                difference_writer.write(legacy_difference_u8)
                mask_writer.write(mask_u8)
        written += 1
        if written == 1 or written % 25 == 0:
            print(f"[PROGRESS] raw depth processed {written} frames", file=sys.stderr, flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    if gradient_writer is not None:
        gradient_writer.release()
    if difference_writer is not None:
        difference_writer.release()
    if mask_writer is not None:
        mask_writer.release()
    return written


def write_mask_raw_stream(args: argparse.Namespace, model, timing: RawDepthTiming) -> int:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    selected_frames = sorted(set(args.frames or []))
    frame_count = len(selected_frames) if selected_frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = sys.stdout.buffer
    out.write(f"SHADOW_DEPTH_MASK_STREAM {width} {height} {frame_count}\n".encode("ascii"))
    out.flush()

    written = 0
    stylus_trace = load_stylus_trace(args.video, args.h5_sidecar)
    previous_centroid: tuple[float, float] | None = None
    while True:
        if selected_frames:
            if written >= len(selected_frames):
                break
            frame_index = selected_frames[written]
            with timing.measure("video_seek"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            frame_index = written
        with timing.measure("video_read"):
            ok, frame = cap.read()
        if not ok or frame is None:
            if selected_frames:
                empty = np.zeros((height, width), dtype=np.uint8)
                with timing.measure("write_stream"):
                    out.write(np.ascontiguousarray(empty).tobytes())
                    out.flush()
                written += 1
                continue
            break
        with timing.measure("inference", sync_cuda=True):
            depth_u8 = infer_depth_u8(model, frame, args.input_size)
        with timing.measure("resize"):
            if depth_u8.shape[:2] != (height, width):
                depth_u8 = cv2.resize(depth_u8, (width, height), interpolation=cv2.INTER_LINEAR)
        with timing.measure("surface_mask"):
            surface_mask = stylus_trace.surface_mask(frame_index, depth_u8.shape) if stylus_trace is not None else None
        with timing.measure("gradient_difference"):
            _, legacy_difference_u8, _, secondary_difference_u8, fast_source = depth_mask_inputs(
                depth_u8, args, surface_mask)
        with timing.measure("mask_threshold"):
            if fast_source:
                difference_u8, difference_source = secondary_difference_u8, fast_source
            else:
                difference_u8, difference_source = choose_active_difference(
                    secondary_difference_u8, legacy_difference_u8, surface_mask, args)
            mask_u8 = make_depth_gradient_mask_for_frame(
                difference_u8, difference_source, frame_index, stylus_trace, previous_centroid, surface_mask, args)
        with timing.measure("component_selection"):
            mask_u8, previous_centroid = keep_hand_component(
                mask_u8, difference_u8, frame_index, stylus_trace, previous_centroid, surface_mask)
        with timing.measure("write_stream"):
            out.write(np.ascontiguousarray(mask_u8).tobytes())
            out.flush()
        written += 1
        if written == 1 or written % 25 == 0:
            print(f"[PROGRESS] raw depth streamed {written} masks", file=sys.stderr, flush=True)
    cap.release()
    return written


def write_mask_difference_raw_stream(args: argparse.Namespace, model, timing: RawDepthTiming) -> int:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    selected_frames = sorted(set(args.frames or []))
    frame_count = len(selected_frames) if selected_frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = sys.stdout.buffer
    out.write(f"SHADOW_DEPTH_MASK_DIFFERENCE_STREAM {width} {height} {frame_count}\n".encode("ascii"))
    out.flush()

    written = 0
    stylus_trace = load_stylus_trace(args.video, args.h5_sidecar)
    previous_centroid: tuple[float, float] | None = None
    while True:
        if selected_frames:
            if written >= len(selected_frames):
                break
            frame_index = selected_frames[written]
            with timing.measure("video_seek"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            frame_index = written
        with timing.measure("video_read"):
            ok, frame = cap.read()
        if not ok or frame is None:
            if selected_frames:
                empty = np.zeros((height, width), dtype=np.uint8)
                with timing.measure("write_stream"):
                    out.write(np.ascontiguousarray(empty).tobytes())
                    out.write(np.ascontiguousarray(empty).tobytes())
                    out.flush()
                written += 1
                continue
            break
        with timing.measure("inference", sync_cuda=True):
            depth_u8 = infer_depth_u8(model, frame, args.input_size)
        with timing.measure("resize"):
            if depth_u8.shape[:2] != (height, width):
                depth_u8 = cv2.resize(depth_u8, (width, height), interpolation=cv2.INTER_LINEAR)
        with timing.measure("surface_mask"):
            surface_mask = stylus_trace.surface_mask(frame_index, depth_u8.shape) if stylus_trace is not None else None
        with timing.measure("gradient_difference"):
            _, legacy_difference_u8, _, secondary_difference_u8, fast_source = depth_mask_inputs(
                depth_u8, args, surface_mask)
        with timing.measure("mask_threshold"):
            if fast_source:
                difference_u8, difference_source = secondary_difference_u8, fast_source
            else:
                difference_u8, difference_source = choose_active_difference(
                    secondary_difference_u8, legacy_difference_u8, surface_mask, args)
            mask_u8 = make_depth_gradient_mask_for_frame(
                difference_u8, difference_source, frame_index, stylus_trace, previous_centroid, surface_mask, args)
        with timing.measure("component_selection"):
            mask_u8, previous_centroid = keep_hand_component(
                mask_u8, difference_u8, frame_index, stylus_trace, previous_centroid, surface_mask)
        with timing.measure("write_stream"):
            out.write(np.ascontiguousarray(mask_u8).tobytes())
            out.write(np.ascontiguousarray(difference_u8).tobytes())
            out.flush()
        written += 1
        if written == 1 or written % 25 == 0:
            print(f"[PROGRESS] raw depth streamed {written} mask/difference frames", file=sys.stderr, flush=True)
    cap.release()
    return written


def main() -> int:
    args = parse_args()
    repo_root = require_repo_root(args.repo_root, args.model_version)
    device = select_device(args.device)
    timing = RawDepthTiming(args.timing, device)
    with timing.measure("model_load", sync_cuda=True):
        if args.video_mode == "streaming":
            model = prepare_da3_streaming(args, repo_root)
        elif args.model_version == "v3":
            model = DA3SmallAdapter(repo_root, args.model_id, device)
        else:
            if args.checkpoint:
                checkpoint = Path(args.checkpoint).expanduser().resolve()
            else:
                checkpoint_name = f"depth_anything_v2_{args.encoder}.pth"
                checkpoint = repo_root / "checkpoints" / checkpoint_name
                if not checkpoint.exists():
                    checkpoint = repo_root.parent / checkpoint_name
            if not checkpoint.exists():
                raise RuntimeError(f"Depth Anything V2 checkpoint not found: {checkpoint}")
            model = load_model(repo_root, checkpoint, args.encoder, device)
    if args.mask_difference_raw_stream:
        written = write_mask_difference_raw_stream(args, model, timing)
        print(f"[DONE] raw depth mask/difference stream frames processed: {written}", file=sys.stderr, flush=True)
        timing.print_summary("mask_difference_raw_stream", written)
    elif args.mask_raw_stream:
        written = write_mask_raw_stream(args, model, timing)
        print(f"[DONE] raw depth mask stream frames processed: {written}", file=sys.stderr, flush=True)
        timing.print_summary("mask_raw_stream", written)
    elif args.frames:
        written = write_selected_frames(args, model, timing)
        print(f"[DONE] raw depth diagnostic frames processed: {written}", file=sys.stderr, flush=True)
        timing.print_summary("selected_frames", written)
    else:
        written = write_video(args, model, timing)
        print(f"[DONE] raw depth video frames processed: {written}", file=sys.stderr, flush=True)
        timing.print_summary("video", written)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
