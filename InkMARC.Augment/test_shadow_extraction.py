
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


_SRGB_TO_LINEAR_LUT = np.array(
    [
        (v / 255.0) / 12.92 if (v / 255.0) <= 0.04045 else ((v / 255.0 + 0.055) / 1.055) ** 2.4
        for v in range(256)
    ],
    dtype=np.float32,
)
_APRILTAG_DETECTOR = None
_CUDA_RUNTIME_ENABLED = False
_CUDA_RUNTIME_AVAILABLE = None
_CUDA_GAUSSIAN_FILTER_CACHE = {}

SURFACE_WIDTH_M = 0.224
SURFACE_HEIGHT_M = 0.148
TAG_SIZE_M = 0.03
TAG_COORD_SCALE = 2.0 / TAG_SIZE_M
HALF_SURFACE_WIDTH_M = SURFACE_WIDTH_M / 2.0
HALF_SURFACE_HEIGHT_M = SURFACE_HEIGHT_M / 2.0
TAG_CENTER_LOCATIONS = {
    8: np.array([-0.0920,  0.0540], dtype=np.float32),
    9: np.array([-0.0307,  0.0540], dtype=np.float32),
    10: np.array([ 0.0307,  0.0540], dtype=np.float32),
    11: np.array([ 0.0920,  0.0540], dtype=np.float32),
    4: np.array([-0.0920,  0.0000], dtype=np.float32),
    5: np.array([-0.0307,  0.0000], dtype=np.float32),
    6: np.array([ 0.0307,  0.0000], dtype=np.float32),
    7: np.array([ 0.0920,  0.0000], dtype=np.float32),
    0: np.array([-0.0920, -0.0540], dtype=np.float32),
    1: np.array([-0.0307, -0.0540], dtype=np.float32),
    2: np.array([ 0.0307, -0.0540], dtype=np.float32),
    3: np.array([ 0.0920, -0.0540], dtype=np.float32),
}
SURFACE_CORNERS_M = np.array(
    [
        [-HALF_SURFACE_WIDTH_M,  HALF_SURFACE_HEIGHT_M],  # TL
        [ HALF_SURFACE_WIDTH_M,  HALF_SURFACE_HEIGHT_M],  # TR
        [ HALF_SURFACE_WIDTH_M, -HALF_SURFACE_HEIGHT_M],  # BR
        [-HALF_SURFACE_WIDTH_M, -HALF_SURFACE_HEIGHT_M],  # BL
    ],
    dtype=np.float32,
)
TAG_LOCAL_CORNERS = np.array(
    [
        [-1.0,  1.0],  # TL
        [ 1.0,  1.0],  # TR
        [ 1.0, -1.0],  # BR
        [-1.0, -1.0],  # BL
    ],
    dtype=np.float32,
)
TAG_HALF_SIZE_M = TAG_SIZE_M / 2.0
OCCLUSION_DIFF_THRESHOLD = 43.0
OCCLUSION_THIN_LINE_KERNEL_SIZE = 9
MIN_OCCLUSION_COMPONENT_AREA = 100


def cuda_runtime_available() -> bool:
    global _CUDA_RUNTIME_AVAILABLE
    if _CUDA_RUNTIME_AVAILABLE is not None:
        return _CUDA_RUNTIME_AVAILABLE
    try:
        _CUDA_RUNTIME_AVAILABLE = (
            hasattr(cv2, "cuda")
            and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
            and cv2.cuda.getCudaEnabledDeviceCount() > 0
            and hasattr(cv2.cuda, "createGaussianFilter")
        )
    except Exception:
        _CUDA_RUNTIME_AVAILABLE = False
    return _CUDA_RUNTIME_AVAILABLE


def set_cuda_runtime_enabled(enabled: bool) -> bool:
    global _CUDA_RUNTIME_ENABLED
    _CUDA_RUNTIME_ENABLED = bool(enabled and cuda_runtime_available())
    return _CUDA_RUNTIME_ENABLED


def use_cuda_runtime() -> bool:
    return _CUDA_RUNTIME_ENABLED and cuda_runtime_available()


def opencv_type_for_array(image: np.ndarray) -> int:
    if image.dtype == np.uint8:
        depth = cv2.CV_8U
    elif image.dtype == np.float32:
        depth = cv2.CV_32F
    else:
        raise TypeError(f"Unsupported CUDA image dtype: {image.dtype}")

    channels = 1 if image.ndim == 2 else int(image.shape[2])
    if channels < 1 or channels > 4:
        raise TypeError(f"Unsupported CUDA channel count: {channels}")
    return depth + ((channels - 1) << 3)


def normalize_gaussian_ksize(ksize, sigma: float) -> tuple[int, int]:
    if isinstance(ksize, int):
        kx = ky = int(ksize)
    else:
        kx, ky = int(ksize[0]), int(ksize[1])
    if kx == 0 or ky == 0:
        auto = max(3, int(round(float(sigma) * 6.0 + 1.0)))
        if auto % 2 == 0:
            auto += 1
        kx = auto if kx == 0 else kx
        ky = auto if ky == 0 else ky
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    return max(1, kx), max(1, ky)


def cuda_gaussian_filter_for(image_type: int, ksize, sigma: float):
    filter_ksize = normalize_gaussian_ksize(ksize, sigma)
    # OpenCV's CUDA separable Gaussian implementation rejects kernels wider
    # than 32 taps, so cap the auto-sized CPU-equivalent kernel here.
    filter_ksize = (min(filter_ksize[0], 31), min(filter_ksize[1], 31))
    cache_key = (image_type, filter_ksize, float(sigma))
    gaussian_filter = _CUDA_GAUSSIAN_FILTER_CACHE.get(cache_key)
    if gaussian_filter is None:
        gaussian_filter = cv2.cuda.createGaussianFilter(
            image_type,
            image_type,
            filter_ksize,
            float(sigma),
        )
        _CUDA_GAUSSIAN_FILTER_CACHE[cache_key] = gaussian_filter
    return gaussian_filter


def gaussian_blur_gpu_mat(src_gpu, image_type: int, ksize, sigma: float):
    gaussian_filter = cuda_gaussian_filter_for(image_type, ksize, sigma)
    return gaussian_filter.apply(src_gpu)


def cuda_gaussian_blur(image: np.ndarray, ksize, sigma: float) -> np.ndarray:
    src = np.ascontiguousarray(image)
    image_type = opencv_type_for_array(src)
    gpu_src = cv2.cuda_GpuMat()
    gpu_src.upload(src)
    gpu_dst = gaussian_blur_gpu_mat(gpu_src, image_type, ksize, sigma)
    return gpu_dst.download()


def warp_perspective_gpu_mat(
    src_gpu,
    H: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value=0,
):
    return cv2.cuda.warpPerspective(
        src_gpu,
        np.asarray(H, dtype=np.float32),
        dsize,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value,
    )


def gaussian_blur_runtime(image: np.ndarray, ksize, sigma: float) -> np.ndarray:
    if use_cuda_runtime():
        try:
            return cuda_gaussian_blur(image, ksize, sigma)
        except Exception as exc:
            print(f"[WARN] CUDA GaussianBlur failed, falling back to CPU: {exc}")
            set_cuda_runtime_enabled(False)
    return cv2.GaussianBlur(image, ksize, sigma)


def warp_perspective_runtime(
    image: np.ndarray,
    H: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value=0,
) -> np.ndarray:
    if use_cuda_runtime():
        try:
            src = np.ascontiguousarray(image)
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src)
            gpu_dst = cv2.cuda.warpPerspective(
                gpu_src,
                np.asarray(H, dtype=np.float32),
                dsize,
                flags=flags,
                borderMode=border_mode,
                borderValue=border_value,
            )
            return gpu_dst.download()
        except Exception as exc:
            print(f"[WARN] CUDA warpPerspective failed, falling back to CPU: {exc}")
            set_cuda_runtime_enabled(False)
    return cv2.warpPerspective(
        image,
        H,
        dsize,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value,
    )


def gpu_mat_from_array(image: np.ndarray):
    gpu = cv2.cuda_GpuMat()
    gpu.upload(np.ascontiguousarray(image))
    return gpu


def gpu_mask_to_float(valid_mask_gpu) -> any:
    mask_gpu = cv2.cuda.compareWithScalar(valid_mask_gpu, 0, cv2.CMP_GT)
    return mask_gpu.convertTo(cv2.CV_32F, None, 1.0 / 255.0, 0.0)


def clip_gpu_field(field_gpu, min_value: float = 0.0, max_value: float = 255.0):
    clipped = cv2.cuda.maxWithScalar(field_gpu, float(min_value))
    clipped = cv2.cuda.minWithScalar(clipped, float(max_value))
    return clipped


def compute_linear_luminance_gpu(image_bgr_gpu):
    planes = cv2.cuda.split(image_bgr_gpu)
    weights = [0.0722, 0.7152, 0.2126]
    luminance_gpu = None
    for plane_gpu, weight in zip(planes, weights):
        channel_gpu = plane_gpu.convertTo(cv2.CV_32F, None, 1.0 / 255.0, 0.0)
        low_mask_gpu = cv2.cuda.compareWithScalar(channel_gpu, 0.04045, cv2.CMP_LE)
        high_mask_gpu = cv2.cuda.bitwise_not(low_mask_gpu)

        low_gpu = cv2.cuda.divideWithScalar(channel_gpu, 12.92)
        high_gpu = cv2.cuda.addWithScalar(channel_gpu, 0.055)
        high_gpu = cv2.cuda.divideWithScalar(high_gpu, 1.055)
        high_gpu = cv2.cuda.pow(high_gpu, 2.4)

        low_mask_f_gpu = low_mask_gpu.convertTo(cv2.CV_32F, None, 1.0 / 255.0, 0.0)
        high_mask_f_gpu = high_mask_gpu.convertTo(cv2.CV_32F, None, 1.0 / 255.0, 0.0)
        linear_gpu = cv2.cuda.add(
            cv2.cuda.multiply(low_gpu, low_mask_f_gpu),
            cv2.cuda.multiply(high_gpu, high_mask_f_gpu),
        )
        weighted_gpu = cv2.cuda.multiplyWithScalar(linear_gpu, weight * 255.0)
        luminance_gpu = weighted_gpu if luminance_gpu is None else cv2.cuda.add(luminance_gpu, weighted_gpu)
    return luminance_gpu


def smooth_masked_field_gpu(field_gpu, valid_mask_gpu, sigma: float):
    weight_gpu = gpu_mask_to_float(valid_mask_gpu)
    weighted_field_gpu = cv2.cuda.multiply(field_gpu, weight_gpu)
    blurred_num_gpu = gaussian_blur_gpu_mat(weighted_field_gpu, cv2.CV_32FC1, (0, 0), sigma)
    blurred_den_gpu = gaussian_blur_gpu_mat(weight_gpu, cv2.CV_32FC1, (0, 0), sigma)
    safe_den_gpu = cv2.cuda.maxWithScalar(blurred_den_gpu, 1e-3)
    out_gpu = cv2.cuda.divide(blurred_num_gpu, safe_den_gpu)
    out_gpu = cv2.cuda.multiply(out_gpu, gpu_mask_to_float(blurred_den_gpu))
    return out_gpu


def compute_fast_shadow_projection_gpu(
    frame_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    board_mask: np.ndarray,
    H: np.ndarray,
    ref_shape: tuple[int, int],
    blur_ksize: int = 31,
    min_ref_lightness: float = 40.0,
    frame_gpu=None,
    board_mask_gpu=None,
):
    ref_h, ref_w = ref_shape[:2]
    frame_h, frame_w = frame_bgr.shape[:2]
    H_f32 = np.asarray(H, dtype=np.float32)
    H_inv = np.linalg.inv(H_f32).astype(np.float32)

    frame_gpu = gpu_mat_from_array(frame_bgr) if frame_gpu is None else frame_gpu
    ref_gpu = gpu_mat_from_array(ref_bgr)
    board_mask_gpu = gpu_mat_from_array(board_mask) if board_mask_gpu is None else board_mask_gpu

    frame_rectified_gpu = warp_perspective_gpu_mat(frame_gpu, H_inv, (ref_w, ref_h))
    rectified_valid_mask_gpu = warp_perspective_gpu_mat(board_mask_gpu, H_inv, (ref_w, ref_h))
    rectified_valid_mask = rectified_valid_mask_gpu.download()
    rectified_valid_mask = cv2.morphologyEx(
        rectified_valid_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    rectified_valid_mask_gpu = gpu_mat_from_array(rectified_valid_mask)
    rectified_valid_weight_gpu = gpu_mask_to_float(rectified_valid_mask_gpu)

    ref_l_raw_gpu = compute_linear_luminance_gpu(ref_gpu)
    frame_l_raw_gpu = compute_linear_luminance_gpu(frame_rectified_gpu)
    ref_l_blur_gpu = gaussian_blur_gpu_mat(ref_l_raw_gpu, cv2.CV_32FC1, (blur_ksize, blur_ksize), 0)
    frame_l_blur_gpu = gaussian_blur_gpu_mat(frame_l_raw_gpu, cv2.CV_32FC1, (blur_ksize, blur_ksize), 0)

    absolute_drop_blur_gpu = cv2.cuda.subtract(ref_l_blur_gpu, frame_l_blur_gpu)
    absolute_drop_blur_gpu = clip_gpu_field(absolute_drop_blur_gpu)
    absolute_drop_blur_gpu = cv2.cuda.multiply(absolute_drop_blur_gpu, rectified_valid_weight_gpu)

    raw_absolute_drop_gpu = cv2.cuda.subtract(ref_l_raw_gpu, frame_l_raw_gpu)
    raw_absolute_drop_gpu = clip_gpu_field(raw_absolute_drop_gpu)
    raw_absolute_drop_gpu = cv2.cuda.multiply(raw_absolute_drop_gpu, rectified_valid_weight_gpu)

    denom_gpu = cv2.cuda.maxWithScalar(ref_l_blur_gpu, float(min_ref_lightness))
    residual_gpu = cv2.cuda.divide(absolute_drop_blur_gpu, denom_gpu)
    residual_gpu = cv2.cuda.multiplyWithScalar(residual_gpu, 255.0)
    residual_gpu = clip_gpu_field(residual_gpu)
    residual_gpu = cv2.cuda.multiply(residual_gpu, rectified_valid_weight_gpu)

    frame_shadow_residual_gpu = warp_perspective_gpu_mat(
        residual_gpu,
        H_f32,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
    )
    frame_shadow_drop_raw_gpu = warp_perspective_gpu_mat(
        raw_absolute_drop_gpu,
        H_f32,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
    )

    return (
        rectified_valid_mask,
        frame_shadow_residual_gpu.download().astype(np.float32),
        frame_shadow_drop_raw_gpu.download().astype(np.float32),
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def open_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    codec: str = "MJPG",
):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    attempts = []
    backend_candidates = [
        ("FFMPEG", getattr(cv2, "CAP_FFMPEG", None)),
        ("OPENCV_MJPEG", getattr(cv2, "CAP_OPENCV_MJPEG", None)),
        ("DEFAULT", None),
    ]
    for backend_name, backend_id in backend_candidates:
        try:
            if backend_id is None:
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), frame_size)
            else:
                writer = cv2.VideoWriter(str(output_path), int(backend_id), fourcc, float(fps), frame_size)
        except cv2.error as exc:
            attempts.append(f"{backend_name}: {exc}")
            continue
        if writer.isOpened():
            return writer, backend_name
        writer.release()
        attempts.append(f"{backend_name}: writer did not open")

    details = "\n".join(f"  - {item}" for item in attempts)
    raise RuntimeError(
        f"Could not open output video for writing: {output_path}\n"
        f"Tried codec={codec}, fps={fps:.3f}, frame_size={frame_size}.\n"
        f"Backend attempts:\n{details}"
    )


def list_texture_paths(texture_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not texture_dir.is_dir():
        raise NotADirectoryError(f"Texture directory not found: {texture_dir}")
    paths = [p for p in sorted(texture_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No texture images found in: {texture_dir}")
    return paths


def load_fitted_texture(texture_path: Path, ref_shape: tuple[int, int]) -> np.ndarray:
    texture_bgr = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    if texture_bgr is None:
        raise FileNotFoundError(f"Could not read surface texture image: {texture_path}")
    return fit_texture_to_reference(texture_bgr, ref_shape)


def clear_previous_outputs(output_dir: Path, prefix: str) -> None:
    for path in output_dir.glob(f"{prefix}_*"):
        if path.is_file():
            path.unlink()


def sample_frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        return []
    if count <= 1:
        return [min(total_frames - 1, max(0, total_frames // 2))]
    vals = np.linspace(0, total_frames - 1, count)
    return sorted({int(round(v)) for v in vals})


def add_profile_time(profile: dict | None, name: str, elapsed: float) -> None:
    if profile is None:
        return
    profile[name] = float(profile.get(name, 0.0)) + float(elapsed)


def print_profile_summary(profile: dict, frames: int, prefix: str = "[PROFILE]") -> None:
    if frames <= 0:
        return
    process_total = float(profile.get("total", 0.0))
    total = (
        process_total
        + float(profile.get("video_read", 0.0))
        + float(profile.get("video_write", 0.0))
        + float(profile.get("multi_texture_warp", 0.0))
        + float(profile.get("multi_texture_overlay", 0.0))
    )
    print(f"{prefix} processed_frames={frames}, total={total:.2f}s, avg={total / frames * 1000.0:.2f}ms/frame")
    if process_total > 0:
        print(f"{prefix}   process_frame_total: {process_total / frames * 1000.0:.2f}ms/frame ({process_total:.2f}s)")
    for name, elapsed in sorted(profile.items(), key=lambda item: item[1], reverse=True):
        if name == "total":
            continue
        print(f"{prefix}   {name}: {elapsed / frames * 1000.0:.2f}ms/frame ({elapsed:.2f}s)")


def get_apriltag_detector():
    global _APRILTAG_DETECTOR
    if _APRILTAG_DETECTOR is not None:
        return _APRILTAG_DETECTOR
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    _APRILTAG_DETECTOR = aruco.ArucoDetector(dictionary, params)
    return _APRILTAG_DETECTOR


def build_reference_cache(ref_bgr: np.ndarray) -> dict:
    ref_detections, ref_rejected = detect_apriltags(ref_bgr)
    ref_tag_artifact_mask = make_reference_tag_artifact_mask(
        ref_bgr,
        ref_detections=ref_detections,
    )
    ref_green_mask, ref_blue_mask, ref_hsv = build_reference_color_masks(ref_bgr, None)
    ref_tag_cache = {}
    for tag_id, corners in ref_detections.items():
        ref_patch = warp_tag_patch(ref_bgr, np.asarray(corners, dtype=np.float32))
        green_mask, blue_mask, mask_debug = build_tag_color_masks(ref_patch)
        ref_tag_cache[int(tag_id)] = {
            "ref_patch": ref_patch,
            "green_mask": green_mask,
            "blue_mask": blue_mask,
            "mask_debug": mask_debug,
        }
    return {
        "ref_detections": ref_detections,
        "ref_rejected": ref_rejected,
        "ref_tag_artifact_mask": ref_tag_artifact_mask,
        "ref_green_mask": ref_green_mask,
        "ref_blue_mask": ref_blue_mask,
        "ref_hsv": ref_hsv,
        "ref_tag_cache": ref_tag_cache,
    }


def detect_apriltags(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = get_apriltag_detector()
    corners, ids, rejected = detector.detectMarkers(gray)
    detections = {}
    if ids is None:
        return detections, rejected
    ids = ids.ravel().tolist()
    for tag_id, corner_set in zip(ids, corners):
        pts = np.asarray(corner_set, dtype=np.float32).reshape(4, 2)
        detections[int(tag_id)] = pts
    return detections, rejected


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {pts.shape[0]}")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def validate_homography(
    H: np.ndarray,
    ref_corners: np.ndarray,
    frame_corners: np.ndarray,
    ref_shape: tuple[int, int],
    frame_shape: tuple[int, int],
    max_reprojection_error: float = 12.0,
):
    if H is None or not np.all(np.isfinite(H)):
        return False, "Homography contains non-finite values."

    det = float(np.linalg.det(H[:2, :2]))
    if abs(det) < 1e-8:
        return False, f"Homography is near-singular (det={det:.3e})."

    projected = cv2.perspectiveTransform(ref_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(projected - frame_corners, axis=1)))
    if not np.isfinite(reprojection_error) or reprojection_error > max_reprojection_error:
        return False, f"Reprojection error too high ({reprojection_error:.2f}px)."

    return True, {
        "det": det,
        "reprojection_error": reprojection_error,
    }


def validate_homography_from_matches(
    H: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    max_reprojection_error: float = 8.0,
):
    if H is None or not np.all(np.isfinite(H)):
        return False, "Homography contains non-finite values."

    det = float(np.linalg.det(H[:2, :2]))
    if abs(det) < 1e-8:
        return False, f"Homography is near-singular (det={det:.3e})."

    projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(projected - dst_pts, axis=1)))
    if not np.isfinite(reprojection_error) or reprojection_error > max_reprojection_error:
        return False, f"Match reprojection error too high ({reprojection_error:.2f}px)."

    return True, {
        "det": det,
        "reprojection_error": reprojection_error,
    }


def preprocess_green_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_lo = np.array([35, 45, 45], dtype=np.uint8)
    green_hi = np.array([95, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, green_lo, green_hi)

    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return green_mask


def detect_board_quad(frame_bgr: np.ndarray):
    green_mask = preprocess_green_mask(frame_bgr)
    board_mask = green_mask.copy()

    kernel = np.ones((7, 7), np.uint8)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, "No board contour found."

    h, w = frame_bgr.shape[:2]
    min_area = 0.05 * (h * w)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None, board_mask, green_mask, f"Board contour too small ({area:.0f}px^2)."

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) >= 4:
        hull = cv2.convexHull(approx)
        if len(hull) >= 4:
            approx = hull

    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = order_points(approx.reshape(-1, 2))
    else:
        rect = cv2.minAreaRect(contour)
        quad = order_points(cv2.boxPoints(rect))

    return quad, board_mask, green_mask, {
        "contour_area": area,
        "approx_points": int(len(approx)),
    }


def find_homography_from_common_tags(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    ref_detections: dict[int, np.ndarray] | None = None,
    frame_detections: dict[int, np.ndarray] | None = None,
    ref_rejected=None,
    frame_rejected=None,
):
    if ref_detections is None or ref_rejected is None:
        ref_detections, ref_rejected = detect_apriltags(ref_bgr)
    if frame_detections is None or frame_rejected is None:
        frame_detections, frame_rejected = detect_apriltags(frame_bgr)

    common_ids = sorted(set(ref_detections.keys()) & set(frame_detections.keys()))
    if not common_ids:
        return None, {
            "ref_detections": ref_detections,
            "frame_detections": frame_detections,
            "ref_rejected": ref_rejected,
            "frame_rejected": frame_rejected,
            "common_ids": common_ids,
            "reason": "No common AprilTags between reference and frame.",
        }

    src_pts = []
    dst_pts = []
    center_matches = []
    for tag_id in common_ids:
        ref_corners = np.asarray(ref_detections[tag_id], dtype=np.float32)
        frame_corners = np.asarray(frame_detections[tag_id], dtype=np.float32)
        src_pts.extend(ref_corners.tolist())
        dst_pts.extend(frame_corners.tolist())
        center_matches.append(
            {
                "id": int(tag_id),
                "ref_center": np.mean(ref_corners, axis=0),
                "image_center": np.mean(frame_corners, axis=0),
            }
        )

    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    if len(src_pts) == 4:
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        inliers = None
    else:
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)

    if H is None or not np.all(np.isfinite(H)):
        return None, {
            "ref_detections": ref_detections,
            "frame_detections": frame_detections,
            "ref_rejected": ref_rejected,
            "frame_rejected": frame_rejected,
            "common_ids": common_ids,
            "center_matches": center_matches,
            "reason": "Failed to solve homography from common tag corners.",
        }

    validation_ok, validation = validate_homography_from_matches(H, src_pts, dst_pts)
    if not validation_ok:
        return None, {
            "ref_detections": ref_detections,
            "frame_detections": frame_detections,
            "ref_rejected": ref_rejected,
            "frame_rejected": frame_rejected,
            "common_ids": common_ids,
            "center_matches": center_matches,
            "src_pts": src_pts,
            "dst_pts": dst_pts,
            "reason": validation,
        }

    ref_h, ref_w = ref_bgr.shape[:2]
    ref_corners = np.array(
        [
            [0.0, 0.0],
            [ref_w - 1.0, 0.0],
            [ref_w - 1.0, ref_h - 1.0],
            [0.0, ref_h - 1.0],
        ],
        dtype=np.float32,
    )
    quad = cv2.perspectiveTransform(ref_corners.reshape(-1, 1, 2), H).reshape(-1, 2)

    debug = {
        "mode": "common_tag_homography",
        "quad": quad,
        "ref_detections": ref_detections,
        "frame_detections": frame_detections,
        "ref_rejected": ref_rejected,
        "frame_rejected": frame_rejected,
        "common_ids": common_ids,
        "center_matches": center_matches,
        "src_pts": src_pts,
        "dst_pts": dst_pts,
        **validation,
    }
    if inliers is not None:
        debug["corner_inliers"] = np.asarray(inliers).ravel().astype(bool).tolist()
    return H, debug


def make_projected_board_mask(H: np.ndarray, ref_shape: tuple[int, int], frame_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ref_h, ref_w = ref_shape[:2]
    frame_h, frame_w = frame_shape[:2]
    ref_corners = np.array(
        [
            [0.0, 0.0],
            [ref_w - 1.0, 0.0],
            [ref_w - 1.0, ref_h - 1.0],
            [0.0, ref_h - 1.0],
        ],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(ref_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(projected).astype(np.int32), 255)
    return mask, projected


def make_projected_tag_structure_mask(
    ref_bgr: np.ndarray,
    H: np.ndarray,
    frame_shape: tuple[int, int],
    dilate_size: int = 25,
    ref_detections: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    frame_h, frame_w = frame_shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if ref_detections is None:
        ref_detections, _ = detect_apriltags(ref_bgr)
    if not ref_detections:
        return mask

    for corners in ref_detections.values():
        projected = cv2.perspectiveTransform(
            np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            H,
        ).reshape(-1, 2)
        cv2.fillConvexPoly(mask, np.round(projected).astype(np.int32), 255)

    if dilate_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def make_reference_tag_structure_mask(
    ref_bgr: np.ndarray,
    dilate_size: int = 15,
    ref_detections: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    mask = np.zeros(ref_bgr.shape[:2], dtype=np.uint8)
    if ref_detections is None:
        ref_detections, _ = detect_apriltags(ref_bgr)
    if not ref_detections:
        return mask

    for corners in ref_detections.values():
        cv2.fillConvexPoly(mask, np.round(np.asarray(corners, dtype=np.float32)).astype(np.int32), 255)

    if dilate_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def make_reference_tag_artifact_mask(
    ref_bgr: np.ndarray,
    patch_size: int = 96,
    structure_dilate: int = 3,
    halo_dilate: int = 2,
    ref_detections: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    mask = np.zeros(ref_bgr.shape[:2], dtype=np.uint8)
    if ref_detections is None:
        ref_detections, _ = detect_apriltags(ref_bgr)
    if not ref_detections:
        return mask

    dst = np.array(
        [
            [0.0, 0.0],
            [patch_size - 1.0, 0.0],
            [patch_size - 1.0, patch_size - 1.0],
            [0.0, patch_size - 1.0],
        ],
        dtype=np.float32,
    )

    for corners in ref_detections.values():
        corners_f = np.asarray(corners, dtype=np.float32)
        tag_patch = warp_tag_patch(ref_bgr, corners_f, size=patch_size)
        gray = cv2.cvtColor(tag_patch, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(tag_patch, cv2.COLOR_BGR2HSV)

        threshold, _ = cv2.threshold(gray.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_mask = gray <= threshold
        blue_mask = (
            (hsv[..., 1] >= 35) &
            (hsv[..., 2] >= 20) &
            (hsv[..., 0] >= 90) &
            (hsv[..., 0] <= 150)
        )
        structure_mask = (dark_mask | blue_mask).astype(np.uint8) * 255

        if structure_dilate > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_dilate, structure_dilate))
            structure_mask = cv2.dilate(structure_mask, kernel, iterations=1)
        structure_mask = cv2.morphologyEx(
            structure_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        if halo_dilate > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_dilate, halo_dilate))
            structure_mask = cv2.dilate(structure_mask, kernel, iterations=1)

        H_back = cv2.getPerspectiveTransform(dst, corners_f)
        warped_mask = cv2.warpPerspective(
            structure_mask,
            H_back,
            (ref_bgr.shape[1], ref_bgr.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        mask = cv2.bitwise_or(mask, warped_mask)

    return mask


def find_homography_from_board_quad(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    ref_detections: dict[int, np.ndarray] | None = None,
    frame_detections: dict[int, np.ndarray] | None = None,
    ref_rejected=None,
    frame_rejected=None,
):
    H, tag_debug = find_homography_from_common_tags(
        ref_bgr,
        frame_bgr,
        ref_detections=ref_detections,
        frame_detections=frame_detections,
        ref_rejected=ref_rejected,
        frame_rejected=frame_rejected,
    )
    if H is not None:
        projected_mask, projected_quad = make_projected_board_mask(H, ref_bgr.shape[:2], frame_bgr.shape[:2])
        debug_dict = {
            "board_mask": projected_mask,
            "green_mask": None,
            "quad": projected_quad,
            **tag_debug,
        }
        return H, projected_mask, None, debug_dict, None

    green_mask = preprocess_green_mask(frame_bgr)
    board_mask = green_mask.copy()
    kernel = np.ones((7, 7), np.uint8)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if frame_detections is None or frame_rejected is None:
        detections, rejected = detect_apriltags(frame_bgr)
    else:
        detections, rejected = frame_detections, frame_rejected
    quad, quad_debug = estimate_board_quad_from_tags(frame_bgr, detections)

    if quad is None:
        failure_debug = {
            "board_mask": board_mask,
            "green_mask": green_mask,
            "detections": detections,
            "rejected": rejected,
            "tag_match_debug": tag_debug,
            **quad_debug,
        }
        reason = quad_debug.get("reason", "Board homography failed.")
        if isinstance(tag_debug, dict) and tag_debug.get("reason"):
            reason = f"{tag_debug['reason']} | {reason}"
        return None, board_mask, green_mask, failure_debug, reason

    ref_h, ref_w = ref_bgr.shape[:2]
    ref_corners = np.array(
        [
            [0.0, 0.0],
            [ref_w - 1.0, 0.0],
            [ref_w - 1.0, ref_h - 1.0],
            [0.0, ref_h - 1.0],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(ref_corners, quad.astype(np.float32))
    ok, validation = validate_homography(H, ref_corners, quad.astype(np.float32), ref_bgr.shape[:2], frame_bgr.shape[:2])
    if not ok:
        debug_dict = {
            "quad": quad,
            "board_mask": board_mask,
            "green_mask": green_mask,
            "detections": detections,
            "rejected": rejected,
            **quad_debug,
        }
        return None, board_mask, green_mask, debug_dict, validation

    projected_mask, projected_quad = make_projected_board_mask(H, ref_bgr.shape[:2], frame_bgr.shape[:2])
    debug_dict = {
        "quad": projected_quad,
        "board_mask": projected_mask,
        "green_mask": green_mask,
        "detections": detections,
        "rejected": rejected,
        **quad_debug,
        **validation,
        "ref_corners": ref_corners,
        "fit_mode": quad_debug.get("mode"),
    }
    return H, projected_mask, green_mask, debug_dict, None


def draw_board_debug(frame_bgr: np.ndarray, debug: dict, out_path: Path):
    if not isinstance(debug, dict) or "quad" not in debug:
        return

    viz = frame_bgr.copy()
    quad = np.asarray(debug["quad"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(viz, [quad], True, (0, 255, 255), 3, cv2.LINE_AA)
    for idx, pt in enumerate(quad.reshape(-1, 2)):
        cv2.circle(viz, tuple(pt.tolist()), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            viz,
            str(idx),
            (int(pt[0]) + 8, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), viz)


def draw_tag_center_debug(frame_bgr: np.ndarray, debug: dict, out_path: Path):
    if not isinstance(debug, dict) or not debug.get("center_matches"):
        return

    viz = frame_bgr.copy()
    for match in debug["center_matches"]:
        pt = np.asarray(match["image_center"], dtype=np.int32).ravel()
        bid = int(match["id"])
        cv2.circle(viz, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            viz,
            str(bid),
            (int(pt[0]) + 8, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), viz)


def tag_image_homography(tag_corners_img: np.ndarray) -> np.ndarray | None:
    if tag_corners_img is None or len(tag_corners_img) != 4:
        return None
    H = cv2.getPerspectiveTransform(TAG_LOCAL_CORNERS.astype(np.float32), np.asarray(tag_corners_img, dtype=np.float32))
    if H is None or not np.all(np.isfinite(H)):
        return None
    return H


def project_surface_corner_from_tag(
    H_tag: np.ndarray,
    surface_corner_m: np.ndarray,
    tag_center_m: np.ndarray,
) -> np.ndarray:
    tag_local = (surface_corner_m - tag_center_m) * TAG_COORD_SCALE
    p = np.array([[tag_local[0], tag_local[1]]], dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(p, H_tag).reshape(2)
    return projected


def tag_board_corners(tag_center_m: np.ndarray) -> np.ndarray:
    cx, cy = float(tag_center_m[0]), float(tag_center_m[1])
    return np.array(
        [
            [cx - TAG_HALF_SIZE_M, cy + TAG_HALF_SIZE_M],
            [cx + TAG_HALF_SIZE_M, cy + TAG_HALF_SIZE_M],
            [cx + TAG_HALF_SIZE_M, cy - TAG_HALF_SIZE_M],
            [cx - TAG_HALF_SIZE_M, cy - TAG_HALF_SIZE_M],
        ],
        dtype=np.float32,
    )


def estimate_board_quad_from_tags(frame_bgr: np.ndarray, detections: dict[int, np.ndarray]):
    visible_ids = [tag_id for tag_id in detections.keys() if tag_id in TAG_CENTER_LOCATIONS]
    dominant_column = None
    sparse_ids = visible_ids
    if 0 < len(visible_ids) < 4:
        column_counts = {}
        for tag_id in visible_ids:
            col = tag_id % 4
            column_counts[col] = column_counts.get(col, 0) + 1
        dominant_column = min(
            column_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[0]
        sparse_ids = [tag_id for tag_id in visible_ids if tag_id % 4 == dominant_column]
        if not sparse_ids:
            sparse_ids = visible_ids

    per_corner_samples = [[] for _ in range(4)]
    per_corner_weights = [[] for _ in range(4)]
    tag_debug = []
    center_debug = []
    corner_src = []
    corner_dst = []
    corner_ids = []

    tag_items = [(tag_id, TAG_CENTER_LOCATIONS[tag_id]) for tag_id in TAG_CENTER_LOCATIONS.keys() if tag_id in sparse_ids]

    for tag_id, tag_center in tag_items:
        corners = detections.get(tag_id)
        if corners is None:
            continue

        corners = np.asarray(corners, dtype=np.float32)
        board_corners = tag_board_corners(tag_center)
        center_debug.append(
            {
                "id": int(tag_id),
                "board_center": tag_center.astype(np.float32),
                "image_center": np.mean(corners, axis=0),
            }
        )
        for bpt, ipt in zip(board_corners, corners):
            corner_src.append(bpt)
            corner_dst.append(ipt)
            corner_ids.append(int(tag_id))

        H_tag = tag_image_homography(corners)
        if H_tag is None:
            continue

        weight = max(float(cv2.contourArea(corners)), 1.0)
        projected_corners = []
        for idx, surface_corner in enumerate(SURFACE_CORNERS_M):
            img_pt = project_surface_corner_from_tag(H_tag, surface_corner, tag_center)
            per_corner_samples[idx].append(img_pt)
            per_corner_weights[idx].append(weight)
            projected_corners.append(img_pt)

        tag_debug.append(
            {
                "id": int(tag_id),
                "weight": weight,
                "corners_img": np.asarray(projected_corners, dtype=np.float32),
                "detected_corners": corners.astype(np.float32),
            }
        )

    if len(corner_src) >= 4:
        src_pts = np.asarray(corner_src, dtype=np.float32)
        dst_pts = np.asarray(corner_dst, dtype=np.float32)
        if len(corner_src) == 4:
            H_board = cv2.getPerspectiveTransform(src_pts, dst_pts)
            inliers = None
        else:
            H_board, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        if H_board is not None and np.all(np.isfinite(H_board)):
            quad = cv2.perspectiveTransform(SURFACE_CORNERS_M.reshape(-1, 1, 2), H_board).reshape(-1, 2)
            quad = order_points(quad)
            inlier_mask = None if inliers is None else np.asarray(inliers).ravel().astype(bool).tolist()
            return quad, {
                "mode": "tag_corner_homography",
                "tags": tag_debug,
                "center_matches": center_debug,
                "dominant_column": dominant_column,
                "sparse_ids": sparse_ids,
                "corner_src": src_pts,
                "corner_dst": dst_pts,
                "corner_tag_ids": corner_ids,
                "corner_homography": H_board,
                "corner_inliers": inlier_mask,
            }

    if not any(len(samples) for samples in per_corner_samples):
        return None, {
            "tags": tag_debug,
            "center_matches": center_debug,
            "dominant_column": dominant_column,
            "sparse_ids": sparse_ids,
            "reason": "No usable tags detected.",
        }

    quad = np.zeros((4, 2), dtype=np.float32)
    for idx, samples in enumerate(per_corner_samples):
        if not samples:
            return None, {
                "tags": tag_debug,
                "center_matches": center_debug,
                "dominant_column": dominant_column,
                "sparse_ids": sparse_ids,
                "reason": f"Missing surface corner {idx}.",
            }
        pts = np.asarray(samples, dtype=np.float32)
        ws = np.asarray(per_corner_weights[idx], dtype=np.float32)
        if len(pts) >= 3:
            med = np.median(pts, axis=0)
            dists = np.linalg.norm(pts - med[None, :], axis=1)
            keep = dists <= max(15.0, np.median(dists) * 2.0 if len(dists) else 15.0)
            if np.any(keep):
                pts = pts[keep]
                ws = ws[keep]
        quad[idx] = np.average(pts, axis=0, weights=ws)

    return order_points(quad), {
        "mode": "tag_corner_projection",
        "tags": tag_debug,
        "center_matches": center_debug,
        "dominant_column": dominant_column,
        "sparse_ids": sparse_ids,
    }


def detect_blue_tile_centers(frame_bgr: np.ndarray, board_mask: np.ndarray | None = None):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue_lo = np.array([85, 50, 30], dtype=np.uint8)
    blue_hi = np.array([150, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lo, blue_hi)

    if board_mask is not None:
        blue_mask = cv2.bitwise_and(blue_mask, board_mask)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    boxes = []
    h, w = frame_bgr.shape[:2]
    min_area = max(150.0, 0.0002 * (h * w))
    max_area = 0.25 * (h * w)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue
        aspect = bw / float(bh)
        if aspect < 0.35 or aspect > 3.0:
            continue
        m = cv2.moments(contour)
        if m["m00"] > 1e-6:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
        else:
            cx = float(x + bw / 2.0)
            cy = float(y + bh / 2.0)
        centers.append((cx, cy))
        boxes.append((x, y, bw, bh))

    if not centers:
        return np.empty((0, 2), dtype=np.float32), blue_mask, []

    centers = np.asarray(centers, dtype=np.float32)
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    return centers[order], blue_mask, [boxes[i] for i in order]


def estimate_local_spacing(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    dists = []
    for i in range(len(pts)):
        delta = pts - pts[i]
        dist = np.sqrt((delta[:, 0] ** 2) + (delta[:, 1] ** 2))
        dist = dist[dist > 1e-3]
        if len(dist):
            dists.append(float(np.min(dist)))
    if not dists:
        return 0.0
    return float(np.median(dists))


def refine_homography_with_tile_centers(
    H_initial: np.ndarray,
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    board_mask: np.ndarray,
    min_matches: int = 4,
):
    ref_centers, ref_blue_mask, _ = detect_blue_tile_centers(ref_bgr, None)
    frame_centers, frame_blue_mask, _ = detect_blue_tile_centers(frame_bgr, board_mask)

    if len(ref_centers) < min_matches or len(frame_centers) < min_matches:
        return H_initial, None, {
            "ref_tile_count": int(len(ref_centers)),
            "frame_tile_count": int(len(frame_centers)),
            "tile_matches": 0,
            "ref_blue_mask": ref_blue_mask,
            "frame_blue_mask": frame_blue_mask,
        }

    projected = cv2.perspectiveTransform(ref_centers.reshape(-1, 1, 2), H_initial).reshape(-1, 2)
    spacing = estimate_local_spacing(frame_centers)
    max_match_dist = float(np.clip(spacing * 0.6 if spacing > 0 else 90.0, 45.0, 220.0))

    matches_src = []
    matches_dst = []
    used = set()
    for i, p in enumerate(projected):
        dists = np.linalg.norm(frame_centers - p, axis=1)
        if len(dists) == 0:
            continue
        j = int(np.argmin(dists))
        if j in used:
            continue
        if dists[j] <= max_match_dist:
            matches_src.append(ref_centers[i])
            matches_dst.append(frame_centers[j])
            used.add(j)

    if len(matches_src) < min_matches:
        return H_initial, None, {
            "ref_tile_count": int(len(ref_centers)),
            "frame_tile_count": int(len(frame_centers)),
            "tile_matches": int(len(matches_src)),
            "tile_spacing": spacing,
            "max_match_dist": max_match_dist,
            "ref_blue_mask": ref_blue_mask,
            "frame_blue_mask": frame_blue_mask,
        }

    src_pts = np.asarray(matches_src, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.asarray(matches_dst, dtype=np.float32).reshape(-1, 1, 2)
    H_refined, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)
    if H_refined is None or not np.all(np.isfinite(H_refined)):
        return H_initial, None, {
            "ref_tile_count": int(len(ref_centers)),
            "frame_tile_count": int(len(frame_centers)),
            "tile_matches": int(len(matches_src)),
            "tile_spacing": spacing,
            "max_match_dist": max_match_dist,
            "ref_blue_mask": ref_blue_mask,
            "frame_blue_mask": frame_blue_mask,
        }

    debug = {
        "ref_tile_count": int(len(ref_centers)),
        "frame_tile_count": int(len(frame_centers)),
        "tile_matches": int(len(matches_src)),
        "tile_inliers": int(inlier_mask.sum()) if inlier_mask is not None else 0,
        "tile_spacing": spacing,
        "max_match_dist": max_match_dist,
        "ref_blue_mask": ref_blue_mask,
        "frame_blue_mask": frame_blue_mask,
        "ref_tile_centers": ref_centers,
        "frame_tile_centers": frame_centers,
        "tile_match_src": src_pts,
        "tile_match_dst": dst_pts,
    }
    return H_refined, H_initial, debug


def make_alignment_panel(
    frame_bgr: np.ndarray,
    warped_ref: np.ndarray,
    shadow_intensity: np.ndarray,
    shadow_mask: np.ndarray,
    overlay: np.ndarray,
):
    def label(img, text, size=(640, 360)):
        img2 = cv2.resize(img.copy(), size, interpolation=cv2.INTER_AREA)
        cv2.putText(img2, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return img2

    shadow_intensity_bgr = cv2.cvtColor(shadow_intensity, cv2.COLOR_GRAY2BGR)
    shadow_mask_bgr = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
    top = np.hstack([
        label(frame_bgr, "Frame"),
        label(warped_ref.astype(np.uint8), "Warped ref"),
        label(shadow_intensity_bgr, "Shadow intensity"),
    ])
    bottom = np.hstack([
        label(shadow_mask_bgr, "Shadow mask"),
        label(overlay, "Overlay"),
        label(warped_ref.astype(np.uint8), "Warped ref"),
    ])
    return np.vstack([top, bottom])


def make_tag_subtraction_debug_image(
    calibrated_ref_patch: np.ndarray,
    frame_patch: np.ndarray,
) -> np.ndarray:
    def label(img, text, size=(256, 256)):
        img2 = cv2.resize(img.copy(), size, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img2, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img2

    signed_drop = cv2.subtract(calibrated_ref_patch, frame_patch)
    abs_diff = cv2.absdiff(calibrated_ref_patch, frame_patch)
    return np.hstack([
        label(calibrated_ref_patch, "Calibrated ref tag"),
        label(frame_patch, "Frame tag"),
        label(signed_drop, "Ref - frame"),
        label(abs_diff, "Abs diff"),
    ])


def circular_hue_delta(target_h: np.ndarray, source_h: np.ndarray) -> np.ndarray:
    delta = target_h.astype(np.float32) - source_h.astype(np.float32)
    return (delta + 90.0) % 180.0 - 90.0


def select_top_left_common_tag(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    ref_detections: dict[int, np.ndarray] | None = None,
    frame_detections: dict[int, np.ndarray] | None = None,
):
    if ref_detections is None:
        ref_detections, _ = detect_apriltags(ref_bgr)
    if frame_detections is None:
        frame_detections, _ = detect_apriltags(frame_bgr)
    common_ids = [tag_id for tag_id in ref_detections.keys() if tag_id in frame_detections]
    if not common_ids:
        return None

    ranked = []
    for tag_id in common_ids:
        center = np.mean(np.asarray(frame_detections[tag_id], dtype=np.float32), axis=0)
        ranked.append((float(center[1]), float(center[0]), int(tag_id)))
    ranked.sort()
    tag_id = ranked[0][2]
    return {
        "id": tag_id,
        "ref_corners": np.asarray(ref_detections[tag_id], dtype=np.float32),
        "frame_corners": np.asarray(frame_detections[tag_id], dtype=np.float32),
        "frame_center": np.mean(np.asarray(frame_detections[tag_id], dtype=np.float32), axis=0),
    }


def warp_tag_patch(image_bgr: np.ndarray, tag_corners: np.ndarray, size: int = 96) -> np.ndarray:
    dst = np.array(
        [
            [0.0, 0.0],
            [size - 1.0, 0.0],
            [size - 1.0, size - 1.0],
            [0.0, size - 1.0],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(np.asarray(tag_corners, dtype=np.float32), dst)
    return cv2.warpPerspective(image_bgr, H, (size, size))


def make_tag_dark_light_masks(tag_patch_bgr: np.ndarray, margin: int = 8):
    # Deprecated compatibility helper. Tag calibration now uses semantic green/blue masks.
    gray = cv2.cvtColor(tag_patch_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    inner = np.zeros_like(gray, dtype=np.uint8)
    inner[margin:max(h - margin, margin), margin:max(w - margin, margin)] = 255

    inner_vals = gray[inner > 0]
    if inner_vals.size < 100:
        return None, None, None

    threshold, _ = cv2.threshold(inner_vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = (gray <= threshold) & (inner > 0)
    light_mask = (gray > threshold) & (inner > 0)
    if int(np.count_nonzero(dark_mask)) < 80 or int(np.count_nonzero(light_mask)) < 80:
        return None, None, None
    return dark_mask, light_mask, float(threshold)


def estimate_tag_color_adjustments(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    hue_weight: float = 1.0,
    hue_max_shift: float = 20.0,
    ref_detections: dict[int, np.ndarray] | None = None,
    frame_detections: dict[int, np.ndarray] | None = None,
    ref_tag_cache: dict[int, dict] | None = None,
):
    selected = select_top_left_common_tag(
        ref_bgr,
        frame_bgr,
        ref_detections=ref_detections,
        frame_detections=frame_detections,
    )
    if selected is None:
        return None

    cache_entry = None if ref_tag_cache is None else ref_tag_cache.get(int(selected["id"]))
    if cache_entry is None:
        ref_patch = warp_tag_patch(ref_bgr, selected["ref_corners"])
        green_mask, blue_mask, mask_debug = build_tag_color_masks(ref_patch)
    else:
        ref_patch = cache_entry["ref_patch"]
        green_mask = cache_entry["green_mask"]
        blue_mask = cache_entry["blue_mask"]
        mask_debug = cache_entry["mask_debug"]

    frame_patch = warp_tag_patch(frame_bgr, selected["frame_corners"])
    if int(np.count_nonzero(green_mask)) < 1 and int(np.count_nonzero(blue_mask)) < 1:
        return None

    green_adj, blue_adj, diagnostics = estimate_tag_class_hsv_adjustments(
        ref_patch,
        frame_patch,
        green_mask,
        blue_mask,
        hue_weight=hue_weight,
        hue_max_shift=hue_max_shift,
    )

    return {
        "selected_tag_id": int(selected["id"]),
        "selected_tag_center": selected["frame_center"].astype(np.float32),
        "auto_blue": blue_adj,
        "auto_green": green_adj,
        "tag_green_mask": green_mask.astype(np.uint8) * 255,
        "tag_blue_mask": blue_mask.astype(np.uint8) * 255,
        "tag_ref_patch": ref_patch.copy(),
        "tag_frame_patch": frame_patch,
        "mask_debug": mask_debug,
        "diagnostics": diagnostics,
        "tag_green_pixel_count": int(np.count_nonzero(green_mask)),
        "tag_blue_pixel_count": int(np.count_nonzero(blue_mask)),
        "tag_green_reliable": bool(diagnostics["green"]["reliable"]),
        "tag_blue_reliable": bool(diagnostics["blue"]["reliable"]),
    }


def estimate_luminance_adjustment(
    ref_l: np.ndarray,
    frame_l: np.ndarray,
    sample_mask: np.ndarray,
    min_count: int = 80,
    selected_min_count: int = 60,
    gain_limits: tuple[float, float] = (0.4, 2.5),
    offset_limits: tuple[float, float] = (-80.0, 80.0),
):
    count = int(np.count_nonzero(sample_mask))
    if count < min_count:
        return {
            "count": count,
            "selected_count": count,
            "gain": 1.0,
            "offset": 0.0,
            "ref_l_median": 0.0,
            "frame_l_median": 0.0,
        }

    ref_vals = ref_l[sample_mask].astype(np.float32)
    frame_vals = frame_l[sample_mask].astype(np.float32)

    lo = np.percentile(frame_vals, 10)
    hi = np.percentile(frame_vals, 90)
    keep = (frame_vals >= lo) & (frame_vals <= hi)
    if int(np.count_nonzero(keep)) < selected_min_count:
        keep = np.ones_like(frame_vals, dtype=bool)

    ref_sel = ref_vals[keep]
    frame_sel = frame_vals[keep]
    A = np.stack([ref_sel, np.ones_like(ref_sel)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, frame_sel, rcond=None)
    gain = float(np.clip(sol[0], gain_limits[0], gain_limits[1]))
    offset = float(np.clip(sol[1], offset_limits[0], offset_limits[1]))
    return {
        "count": count,
        "selected_count": int(np.count_nonzero(keep)),
        "gain": gain,
        "offset": offset,
        "ref_l_median": float(np.median(ref_sel)),
        "frame_l_median": float(np.median(frame_sel)),
    }


def apply_class_luminance_adjustment(
    image_bgr: np.ndarray,
    green_params: dict,
    blue_params: dict,
    valid_mask: np.ndarray | None = None,
):
    green_mask, blue_mask, _ = build_reference_color_masks(image_bgr, valid_mask)
    out = image_bgr.astype(np.float32).copy()
    luminance = compute_linear_luminance(image_bgr).astype(np.float32)
    target_luminance = luminance.copy()

    if np.any(green_mask):
        target_luminance[green_mask] = np.clip(
            (luminance[green_mask] * green_params["gain"]) + green_params["offset"],
            0,
            255,
        )
    if np.any(blue_mask):
        target_luminance[blue_mask] = np.clip(
            (luminance[blue_mask] * blue_params["gain"]) + blue_params["offset"],
            0,
            255,
        )

    scale = np.ones_like(luminance, dtype=np.float32)
    bright = luminance > 1e-3
    scale[bright] = target_luminance[bright] / luminance[bright]
    out *= scale[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), green_mask.astype(np.uint8) * 255, blue_mask.astype(np.uint8) * 255


def estimate_tag_class_luminance_adjustments(
    ref_tag_patch_bgr: np.ndarray,
    frame_tag_patch_bgr: np.ndarray,
    green_mask: np.ndarray,
    blue_mask: np.ndarray,
    min_count: int = 24,
    selected_min_count: int = 20,
):
    ref_l = compute_linear_luminance(ref_tag_patch_bgr)
    frame_l = compute_linear_luminance(frame_tag_patch_bgr)
    green_params = estimate_luminance_adjustment(
        ref_l,
        frame_l,
        green_mask,
        min_count=min_count,
        selected_min_count=selected_min_count,
    )
    blue_params = estimate_luminance_adjustment(
        ref_l,
        frame_l,
        blue_mask,
        min_count=min_count,
        selected_min_count=selected_min_count,
    )
    diagnostics = {
        "green": {
            "pixel_count": int(np.count_nonzero(green_mask)),
            "selected_count": int(green_params["selected_count"]),
            "reliable": bool(green_params["selected_count"] >= min_count),
        },
        "blue": {
            "pixel_count": int(np.count_nonzero(blue_mask)),
            "selected_count": int(blue_params["selected_count"]),
            "reliable": bool(blue_params["selected_count"] >= min_count),
        },
        "min_count": int(min_count),
        "selected_min_count": int(selected_min_count),
    }
    return green_params, blue_params, diagnostics


def estimate_tag_luminance_adjustments(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    ref_detections: dict[int, np.ndarray] | None = None,
    frame_detections: dict[int, np.ndarray] | None = None,
):
    selected = select_top_left_common_tag(
        ref_bgr,
        frame_bgr,
        ref_detections=ref_detections,
        frame_detections=frame_detections,
    )
    if selected is None:
        return None

    ref_patch = warp_tag_patch(ref_bgr, selected["ref_corners"])
    frame_patch = warp_tag_patch(frame_bgr, selected["frame_corners"])
    green_mask, blue_mask, mask_debug = build_tag_color_masks(ref_patch)
    if int(np.count_nonzero(green_mask)) < 1 and int(np.count_nonzero(blue_mask)) < 1:
        return None

    green_params, blue_params, diagnostics = estimate_tag_class_luminance_adjustments(
        ref_patch,
        frame_patch,
        green_mask,
        blue_mask,
    )

    return {
        "selected_tag_id": int(selected["id"]),
        "selected_tag_center": selected["frame_center"].astype(np.float32),
        "green_luminance_adjustment": green_params,
        "blue_luminance_adjustment": blue_params,
        "tag_green_mask": green_mask.astype(np.uint8) * 255,
        "tag_blue_mask": blue_mask.astype(np.uint8) * 255,
        "tag_ref_patch": ref_patch,
        "tag_frame_patch": frame_patch,
        "mask_debug": mask_debug,
        "diagnostics": diagnostics,
        "tag_green_pixel_count": int(np.count_nonzero(green_mask)),
        "tag_blue_pixel_count": int(np.count_nonzero(blue_mask)),
        "tag_green_reliable": bool(diagnostics["green"]["reliable"]),
        "tag_blue_reliable": bool(diagnostics["blue"]["reliable"]),
    }


def refine_reference_color_masks(
    green_mask: np.ndarray,
    blue_mask: np.ndarray,
    valid_mask: np.ndarray,
    green_erode_size: int = 5,
    blue_dilate_size: int = 5,
):
    valid_u8 = valid_mask.astype(np.uint8) * 255
    green_u8 = green_mask.astype(np.uint8) * 255
    blue_u8 = blue_mask.astype(np.uint8) * 255

    if green_erode_size > 1:
        green_kernel = np.ones((green_erode_size, green_erode_size), np.uint8)
        green_u8 = cv2.erode(green_u8, green_kernel, iterations=1)
    if blue_dilate_size > 1:
        blue_kernel = np.ones((blue_dilate_size, blue_dilate_size), np.uint8)
        blue_u8 = cv2.dilate(blue_u8, blue_kernel, iterations=1)

    green_u8 = cv2.bitwise_and(green_u8, valid_u8)
    blue_u8 = cv2.bitwise_and(blue_u8, valid_u8)

    # Let blue own the narrow boundary band so printed tile edges are less likely
    # to leak into the green class during board flattening.
    green_u8[blue_u8 > 0] = 0
    return green_u8 > 0, blue_u8 > 0


def build_reference_color_masks(image_bgr: np.ndarray, valid_mask: np.ndarray | None = None):
    hsv = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    valid = np.ones(hsv.shape[:2], dtype=bool) if valid_mask is None else (valid_mask > 0)
    sat = hsv[..., 1] >= 25
    val = hsv[..., 2] >= 25

    green_mask = valid & sat & val & (hsv[..., 0] >= 35) & (hsv[..., 0] <= 95)
    blue_mask = valid & sat & val & (hsv[..., 0] >= 96) & (hsv[..., 0] <= 150)
    green_mask, blue_mask = refine_reference_color_masks(green_mask, blue_mask, valid)
    return green_mask, blue_mask, hsv


def build_tag_color_masks(
    ref_tag_patch_bgr: np.ndarray,
    margin: int = 6,
    erode_size: int = 1,
):
    h, w = ref_tag_patch_bgr.shape[:2]
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    inner_mask[margin:max(h - margin, margin), margin:max(w - margin, margin)] = 255
    green_mask, blue_mask, _ = build_reference_color_masks(ref_tag_patch_bgr, inner_mask)

    if erode_size > 0:
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.erode(green_mask.astype(np.uint8) * 255, kernel, iterations=erode_size) > 0
        blue_mask = cv2.erode(blue_mask.astype(np.uint8) * 255, kernel, iterations=erode_size) > 0

    green_mask &= inner_mask > 0
    blue_mask &= inner_mask > 0
    debug = {
        "margin": int(margin),
        "erode_size": int(erode_size),
        "green_count": int(np.count_nonzero(green_mask)),
        "blue_count": int(np.count_nonzero(blue_mask)),
        "inner_count": int(np.count_nonzero(inner_mask)),
    }
    return green_mask, blue_mask, debug


def sample_top_left_tag_class_colors(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    ref_detections: dict[int, np.ndarray] | None = None,
    frame_detections: dict[int, np.ndarray] | None = None,
):
    selected = select_top_left_common_tag(
        ref_bgr,
        frame_bgr,
        ref_detections=ref_detections,
        frame_detections=frame_detections,
    )
    if selected is None:
        return None

    ref_patch = warp_tag_patch(ref_bgr, selected["ref_corners"])
    frame_patch = warp_tag_patch(frame_bgr, selected["frame_corners"])
    green_mask, blue_mask, mask_debug = build_tag_color_masks(ref_patch)
    if int(np.count_nonzero(green_mask)) < 1 and int(np.count_nonzero(blue_mask)) < 1:
        return None

    green_bgr = None
    blue_bgr = None
    if int(np.count_nonzero(green_mask)) > 0:
        green_bgr = np.median(frame_patch[green_mask].astype(np.float32), axis=0).astype(np.uint8)
    if int(np.count_nonzero(blue_mask)) > 0:
        blue_bgr = np.median(frame_patch[blue_mask].astype(np.float32), axis=0).astype(np.uint8)

    return {
        "selected_tag_id": int(selected["id"]),
        "selected_tag_center": selected["frame_center"].astype(np.float32),
        "green_bgr": green_bgr,
        "blue_bgr": blue_bgr,
        "green_count": int(np.count_nonzero(green_mask)),
        "blue_count": int(np.count_nonzero(blue_mask)),
        "mask_debug": mask_debug,
    }


def estimate_hsv_adjustment(
    ref_hsv: np.ndarray,
    frame_hsv: np.ndarray,
    sample_mask: np.ndarray,
    hue_weight: float = 0.35,
    hue_max_shift: float = 8.0,
    min_count: int = 150,
    selected_min_count: int = 100,
    s_scale_limits: tuple[float, float] = (0.5, 1.8),
    v_scale_limits: tuple[float, float] = (0.7, 1.3),
):
    count = int(np.count_nonzero(sample_mask))
    if count < min_count:
        return {
            "count": count,
            "selected_count": count,
            "raw_h_shift": 0.0,
            "h_shift": 0.0,
            "s_scale": 1.0,
            "v_scale": 1.0,
            "ref_hsv_median": [0.0, 0.0, 0.0],
            "frame_hsv_median": [0.0, 0.0, 0.0],
        }

    ref_h = ref_hsv[..., 0][sample_mask].astype(np.float32)
    ref_s = ref_hsv[..., 1][sample_mask].astype(np.float32)
    ref_v = ref_hsv[..., 2][sample_mask].astype(np.float32)
    frm_h = frame_hsv[..., 0][sample_mask].astype(np.float32)
    frm_s = frame_hsv[..., 1][sample_mask].astype(np.float32)
    frm_v = frame_hsv[..., 2][sample_mask].astype(np.float32)

    # Use brighter, more saturated samples so shadows and low-confidence edges do not
    # drag the calibration away from the board's base appearance.
    sat_floor = np.percentile(frm_s, 25)
    val_floor = np.percentile(frm_v, 60)
    keep = (frm_s >= sat_floor) & (frm_v >= val_floor)
    if int(np.count_nonzero(keep)) < selected_min_count:
        keep = np.ones_like(frm_h, dtype=bool)

    ref_h_sel = ref_h[keep]
    ref_s_sel = ref_s[keep]
    ref_v_sel = ref_v[keep]
    frm_h_sel = frm_h[keep]
    frm_s_sel = frm_s[keep]
    frm_v_sel = frm_v[keep]

    raw_h_shift = float(np.median(circular_hue_delta(frm_h_sel, ref_h_sel)))
    h_shift = float(np.clip(raw_h_shift * hue_weight, -hue_max_shift, hue_max_shift))
    s_scale = float(np.clip(np.median(frm_s_sel) / max(np.median(ref_s_sel), 1.0), s_scale_limits[0], s_scale_limits[1]))
    v_scale = float(np.clip(np.percentile(frm_v_sel, 75) / max(np.percentile(ref_v_sel, 75), 1.0), v_scale_limits[0], v_scale_limits[1]))
    return {
        "count": count,
        "selected_count": int(np.count_nonzero(keep)),
        "raw_h_shift": raw_h_shift,
        "h_shift": h_shift,
        "s_scale": s_scale,
        "v_scale": v_scale,
        "ref_hsv_median": [
            float(np.median(ref_h_sel)),
            float(np.median(ref_s_sel)),
            float(np.median(ref_v_sel)),
        ],
        "frame_hsv_median": [
            float(np.median(frm_h_sel)),
            float(np.median(frm_s_sel)),
            float(np.median(frm_v_sel)),
        ],
    }


def estimate_tag_class_hsv_adjustments(
    ref_tag_patch_bgr: np.ndarray,
    frame_tag_patch_bgr: np.ndarray,
    green_mask: np.ndarray,
    blue_mask: np.ndarray,
    hue_weight: float = 1.0,
    hue_max_shift: float = 20.0,
    min_saturation: int = 48,
    min_value: int = 48,
    min_count: int = 24,
    selected_min_count: int = 20,
):
    ref_hsv = cv2.cvtColor(ref_tag_patch_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.cvtColor(frame_tag_patch_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    reliable_pixels = (
        (ref_hsv[..., 1] >= min_saturation) &
        (frame_hsv[..., 1] >= min_saturation) &
        (ref_hsv[..., 2] >= min_value) &
        (frame_hsv[..., 2] >= min_value)
    )

    green_sample_mask = green_mask & reliable_pixels
    blue_sample_mask = blue_mask & reliable_pixels
    green_adj = estimate_hsv_adjustment(
        ref_hsv,
        frame_hsv,
        green_sample_mask,
        hue_weight=hue_weight,
        hue_max_shift=hue_max_shift,
        min_count=min_count,
        selected_min_count=selected_min_count,
        s_scale_limits=(0.5, 2.0),
        v_scale_limits=(0.6, 1.6),
    )
    blue_adj = estimate_hsv_adjustment(
        ref_hsv,
        frame_hsv,
        blue_sample_mask,
        hue_weight=hue_weight,
        hue_max_shift=hue_max_shift,
        min_count=min_count,
        selected_min_count=selected_min_count,
        s_scale_limits=(0.5, 2.0),
        v_scale_limits=(0.6, 1.6),
    )
    diagnostics = {
        "green": {
            "pixel_count": int(np.count_nonzero(green_mask)),
            "eligible_count": int(np.count_nonzero(green_sample_mask)),
            "selected_count": int(green_adj["selected_count"]),
            "reliable": bool(green_adj["selected_count"] >= min_count),
        },
        "blue": {
            "pixel_count": int(np.count_nonzero(blue_mask)),
            "eligible_count": int(np.count_nonzero(blue_sample_mask)),
            "selected_count": int(blue_adj["selected_count"]),
            "reliable": bool(blue_adj["selected_count"] >= min_count),
        },
        "min_saturation": int(min_saturation),
        "min_value": int(min_value),
        "min_count": int(min_count),
        "selected_min_count": int(selected_min_count),
    }
    return green_adj, blue_adj, diagnostics


def apply_hsv_adjustment(
    image_bgr: np.ndarray,
    green_adj: dict,
    blue_adj: dict,
    valid_mask: np.ndarray | None = None,
):
    green_mask, blue_mask, hsv = build_reference_color_masks(image_bgr, valid_mask)
    return apply_hsv_adjustment_with_masks(hsv, green_mask, blue_mask, green_adj, blue_adj)


def apply_hsv_adjustment_with_masks(
    hsv: np.ndarray,
    green_mask: np.ndarray,
    blue_mask: np.ndarray,
    green_adj: dict,
    blue_adj: dict,
):
    hsv = hsv.astype(np.float32)

    if np.any(green_mask):
        hsv[..., 0][green_mask] = (hsv[..., 0][green_mask] + green_adj["h_shift"]) % 180.0
        hsv[..., 1][green_mask] = np.clip(hsv[..., 1][green_mask] * green_adj["s_scale"], 0, 255)
        hsv[..., 2][green_mask] = np.clip(hsv[..., 2][green_mask] * green_adj["v_scale"], 0, 255)

    if np.any(blue_mask):
        hsv[..., 0][blue_mask] = (hsv[..., 0][blue_mask] + blue_adj["h_shift"]) % 180.0
        hsv[..., 1][blue_mask] = np.clip(hsv[..., 1][blue_mask] * blue_adj["s_scale"], 0, 255)
        hsv[..., 2][blue_mask] = np.clip(hsv[..., 2][blue_mask] * blue_adj["v_scale"], 0, 255)

    calibrated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return calibrated, green_mask.astype(np.uint8) * 255, blue_mask.astype(np.uint8) * 255


def make_class_color_from_reference(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    hsv_adj: dict,
    lum_adj: dict | None = None,
):
    if int(np.count_nonzero(mask)) < 20:
        return None

    hsv = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    vals = hsv[mask > 0]
    h = float(np.median(vals[:, 0]))
    s = float(np.median(vals[:, 1]))
    v = float(np.median(vals[:, 2]))

    h = float((h + hsv_adj["h_shift"]) % 180.0)
    s = float(np.clip(s * hsv_adj["s_scale"], 0, 255))
    v = float(np.clip(v * hsv_adj["v_scale"], 0, 255))
    swatch_hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    swatch_bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR).astype(np.float32)[0, 0]

    if lum_adj is not None:
        lum = compute_linear_luminance(swatch_bgr.reshape(1, 1, 3))[0, 0]
        target_lum = float(np.clip((lum * lum_adj["gain"]) + lum_adj["offset"], 0, 255))
        if lum > 1e-3:
            swatch_bgr *= target_lum / lum

    return np.clip(swatch_bgr, 0, 255).astype(np.uint8)


def compute_tag_classwise_error(
    ref_patch_bgr: np.ndarray,
    frame_patch_bgr: np.ndarray,
    green_mask: np.ndarray | None = None,
    blue_mask: np.ndarray | None = None,
) -> dict:
    ref_lab = cv2.cvtColor(ref_patch_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    frame_lab = cv2.cvtColor(frame_patch_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.mean(np.abs(ref_lab - frame_lab), axis=2)

    def masked_error(mask: np.ndarray | None):
        if mask is None:
            return None, 0
        mask_bool = mask > 0
        count = int(np.count_nonzero(mask_bool))
        if count < 1:
            return None, 0
        return float(np.mean(diff[mask_bool])), count

    green_error, green_count = masked_error(green_mask)
    blue_error, blue_count = masked_error(blue_mask)
    total_count = green_count + blue_count
    if total_count > 0:
        total_error = 0.0
        if green_error is not None:
            total_error += green_error * green_count
        if blue_error is not None:
            total_error += blue_error * blue_count
        total_error /= total_count
    else:
        total_error = float(np.mean(diff))
    return {
        "green_error": green_error,
        "blue_error": blue_error,
        "total_error": float(total_error),
        "green_count": int(green_count),
        "blue_count": int(blue_count),
    }


def compute_tag_patch_error(
    ref_patch_bgr: np.ndarray,
    frame_patch_bgr: np.ndarray,
    green_mask: np.ndarray | None = None,
    blue_mask: np.ndarray | None = None,
) -> float:
    return float(
        compute_tag_classwise_error(
            ref_patch_bgr,
            frame_patch_bgr,
            green_mask,
            blue_mask,
        )["total_error"]
    )


def choose_blue_hue_shift_from_tag(
    ref_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    green_adj: dict,
    blue_adj: dict,
):
    tag_debug = estimate_tag_color_adjustments(ref_bgr, frame_bgr)
    if not isinstance(tag_debug, dict):
        return blue_adj["h_shift"], None

    tag_blue_adj = tag_debug["auto_blue"]
    source = "tag" if tag_debug.get("tag_blue_reliable") else "board_fallback"
    chosen_shift = float(tag_blue_adj["h_shift"]) if source == "tag" else float(blue_adj["h_shift"])
    return chosen_shift, {
        "selected_tag_id": int(tag_debug["selected_tag_id"]),
        "compatibility_wrapper": True,
        "source": source,
        "chosen_h_shift": chosen_shift,
        "tag_blue_adjustment": tag_blue_adj,
        "tag_green_adjustment": tag_debug["auto_green"],
    }


def snap_reference_to_class_colors(
    image_bgr: np.ndarray,
    green_mask: np.ndarray,
    blue_mask: np.ndarray,
    board_mask: np.ndarray | None = None,
    canonical_green_bgr: np.ndarray | None = None,
    canonical_blue_bgr: np.ndarray | None = None,
):
    out = image_bgr.astype(np.uint8).copy()
    valid = np.ones(image_bgr.shape[:2], dtype=bool) if board_mask is None else (board_mask > 0)
    green = (green_mask > 0) & valid
    blue = (blue_mask > 0) & valid
    board = valid.copy()

    if int(np.count_nonzero(green)) < 20 or int(np.count_nonzero(blue)) < 20:
        return out, None

    green_color = (
        canonical_green_bgr.astype(np.float32)
        if canonical_green_bgr is not None
        else np.median(out[green].astype(np.float32), axis=0)
    )
    blue_color = (
        canonical_blue_bgr.astype(np.float32)
        if canonical_blue_bgr is not None
        else np.median(out[blue].astype(np.float32), axis=0)
    )

    out[green] = np.clip(green_color, 0, 255).astype(np.uint8)
    out[blue] = np.clip(blue_color, 0, 255).astype(np.uint8)

    undecided = board & ~(green | blue)
    if np.any(undecided):
        undecided_vals = out[undecided].astype(np.float32)
        green_dist = np.sum((undecided_vals - green_color[None, :]) ** 2, axis=1)
        blue_dist = np.sum((undecided_vals - blue_color[None, :]) ** 2, axis=1)
        choose_green = green_dist <= blue_dist
        coords = np.column_stack(np.where(undecided))
        if coords.size:
            out[coords[choose_green, 0], coords[choose_green, 1]] = np.clip(green_color, 0, 255).astype(np.uint8)
            out[coords[~choose_green, 0], coords[~choose_green, 1]] = np.clip(blue_color, 0, 255).astype(np.uint8)

    return out, {
        "green_bgr": [float(green_color[0]), float(green_color[1]), float(green_color[2])],
        "blue_bgr": [float(blue_color[0]), float(blue_color[1]), float(blue_color[2])],
        "green_count": int(np.count_nonzero(green)),
        "blue_count": int(np.count_nonzero(blue)),
        "undecided_count": int(np.count_nonzero(undecided)),
    }


def combine_hsv_adjustments(
    auto_adjustment: dict,
    manual_adjustment: dict,
    hue_limit: float,
    s_scale_limits: tuple[float, float] | None = None,
    v_scale_limits: tuple[float, float] | None = None,
):
    s_scale = float(auto_adjustment["s_scale"] * manual_adjustment["s_scale"])
    v_scale = float(auto_adjustment["v_scale"] * manual_adjustment["v_scale"])
    if s_scale_limits is not None:
        s_scale = float(np.clip(s_scale, s_scale_limits[0], s_scale_limits[1]))
    if v_scale_limits is not None:
        v_scale = float(np.clip(v_scale, v_scale_limits[0], v_scale_limits[1]))
    return {
        "h_shift": float(np.clip(auto_adjustment["h_shift"] + manual_adjustment["h_shift"], -hue_limit, hue_limit)),
        "s_scale": s_scale,
        "v_scale": v_scale,
    }


def make_neutral_luminance_adjustment() -> dict:
    return {
        "count": 0,
        "selected_count": 0,
        "gain": 1.0,
        "offset": 0.0,
        "ref_l_median": 0.0,
        "frame_l_median": 0.0,
    }


def tag_adjustment_is_reliable(adjustment: dict | None, min_selected_count: int = 24) -> bool:
    return bool(isinstance(adjustment, dict) and adjustment.get("selected_count", 0) >= min_selected_count)


def calibrate_reference_colors(
    frame_bgr: np.ndarray,
    warped_ref_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    board_mask: np.ndarray,
    manual_adjustments: dict,
    auto_settings: dict,
    ref_detections: dict[int, np.ndarray] | None = None,
    frame_detections: dict[int, np.ndarray] | None = None,
    reference_cache: dict | None = None,
    fast_mode: bool = False,
    fast_color_calibration: bool = False,
    eyedropper_tag_colors: bool = False,
    profile: dict | None = None,
):
    stage_t = time.perf_counter()
    green_mask, blue_mask, ref_hsv = build_reference_color_masks(warped_ref_bgr, board_mask)
    frame_hsv = cv2.cvtColor(frame_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    add_profile_time(profile, "color.build_masks_and_frame_hsv", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    auto_green = estimate_hsv_adjustment(
        ref_hsv,
        frame_hsv,
        green_mask,
        hue_weight=auto_settings["hue_weight"],
        hue_max_shift=auto_settings["hue_max_shift"],
    )
    auto_blue = estimate_hsv_adjustment(
        ref_hsv,
        frame_hsv,
        blue_mask,
        hue_weight=auto_settings["hue_weight"],
        hue_max_shift=auto_settings["hue_max_shift"],
    )
    add_profile_time(profile, "color.board_hsv_estimate", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    board_green_adj = combine_hsv_adjustments(auto_green, manual_adjustments["green"], hue_limit=8.0)
    board_blue_adj = combine_hsv_adjustments(auto_blue, manual_adjustments["blue"], hue_limit=3.0)

    tag_hsv_debug = None
    if not (fast_mode and fast_color_calibration):
        tag_hsv_debug = estimate_tag_color_adjustments(
            ref_bgr,
            frame_bgr,
            ref_detections=ref_detections,
            frame_detections=frame_detections,
            ref_tag_cache=None if reference_cache is None else reference_cache.get("ref_tag_cache"),
        )
    add_profile_time(profile, "color.tag_hsv_estimate", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    tag_green_adj = dict(board_green_adj)
    tag_blue_adj = dict(board_blue_adj)
    tag_green_source = "board_fallback"
    tag_blue_source = "board_fallback"
    tag_green_pixel_count = 0
    tag_blue_pixel_count = 0
    if isinstance(tag_hsv_debug, dict):
        tag_green_pixel_count = int(tag_hsv_debug.get("tag_green_pixel_count", 0))
        tag_blue_pixel_count = int(tag_hsv_debug.get("tag_blue_pixel_count", 0))
        tag_green_candidate = combine_hsv_adjustments(
            tag_hsv_debug["auto_green"],
            manual_adjustments["green"],
            hue_limit=20.0,
            s_scale_limits=(0.5, 2.0),
            v_scale_limits=(0.6, 1.6),
        )
        tag_blue_candidate = combine_hsv_adjustments(
            tag_hsv_debug["auto_blue"],
            manual_adjustments["blue"],
            hue_limit=20.0,
            s_scale_limits=(0.5, 2.0),
            v_scale_limits=(0.6, 1.6),
        )
        if tag_adjustment_is_reliable(tag_hsv_debug.get("auto_green")):
            tag_green_adj = tag_green_candidate
            tag_green_source = "tag"
        if tag_adjustment_is_reliable(tag_hsv_debug.get("auto_blue")):
            tag_blue_adj = tag_blue_candidate
            tag_blue_source = "tag"

    green_adj = dict(tag_green_adj)
    blue_adj = dict(tag_blue_adj)
    blue_hue_choice_debug = None if not isinstance(tag_hsv_debug, dict) else {
        "compatibility_wrapper": False,
        "source": tag_blue_source,
        "chosen_h_shift": float(blue_adj["h_shift"]),
        "tag_blue_adjustment": tag_hsv_debug["auto_blue"],
        "tag_green_adjustment": tag_hsv_debug["auto_green"],
    }
    add_profile_time(profile, "color.choose_adjustments", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    hsv_only_warped_ref, warped_green_mask, warped_blue_mask = apply_hsv_adjustment_with_masks(
        ref_hsv,
        green_mask,
        blue_mask,
        green_adj,
        blue_adj,
    )
    add_profile_time(profile, "color.apply_warped_ref", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    if reference_cache is not None and all(k in reference_cache for k in ("ref_hsv", "ref_green_mask", "ref_blue_mask")):
        hsv_only_ref, ref_green_mask, ref_blue_mask = apply_hsv_adjustment_with_masks(
            reference_cache["ref_hsv"],
            reference_cache["ref_green_mask"],
            reference_cache["ref_blue_mask"],
            green_adj,
            blue_adj,
        )
    else:
        hsv_only_ref, ref_green_mask, ref_blue_mask = apply_hsv_adjustment(
            ref_bgr,
            green_adj,
            blue_adj,
            None,
        )
    add_profile_time(profile, "color.apply_reference", time.perf_counter() - stage_t)
    calibrated_warped_ref = hsv_only_warped_ref.copy()
    calibrated_ref = hsv_only_ref.copy()
    calibration_source = "board_hsv_only"
    if tag_green_source == "tag" or tag_blue_source == "tag":
        calibration_source = "board_hsv_plus_semantic_tag_hsv"

    stage_t = time.perf_counter()
    tag_color_debug = None if fast_mode else estimate_tag_luminance_adjustments(
        hsv_only_ref,
        frame_bgr,
        ref_detections=ref_detections,
        frame_detections=frame_detections,
    )
    add_profile_time(profile, "color.tag_luminance_estimate", time.perf_counter() - stage_t)
    green_luminance_adjustment = None
    blue_luminance_adjustment = None
    tag_error_before = None
    tag_error_after = None
    tag_error_before_by_class = None
    tag_error_after_by_class = None
    tag_gate_passed = False
    tag_luminance_applied = False
    tag_luminance_rejected = False
    if isinstance(tag_color_debug, dict):
        stage_t = time.perf_counter()
        tag_green_mask = tag_color_debug.get("tag_green_mask")
        tag_blue_mask = tag_color_debug.get("tag_blue_mask")
        tag_error_before_by_class = compute_tag_classwise_error(
            tag_color_debug["tag_ref_patch"],
            tag_color_debug["tag_frame_patch"],
            tag_green_mask,
            tag_blue_mask,
        )
        tag_error_before = float(tag_error_before_by_class["total_error"])
        tag_error_after = tag_error_before

        green_lum_reliable = tag_adjustment_is_reliable(tag_color_debug.get("green_luminance_adjustment"))
        blue_lum_reliable = tag_adjustment_is_reliable(tag_color_debug.get("blue_luminance_adjustment"))
        green_lum_candidate = tag_color_debug["green_luminance_adjustment"] if green_lum_reliable else make_neutral_luminance_adjustment()
        blue_lum_candidate = tag_color_debug["blue_luminance_adjustment"] if blue_lum_reliable else make_neutral_luminance_adjustment()
        candidate_available = green_lum_reliable or blue_lum_reliable

        tag_color_debug["green_luminance_adjustment"] = None
        tag_color_debug["blue_luminance_adjustment"] = None
        tag_color_debug["tag_calibrated_ref_patch"] = tag_color_debug["tag_ref_patch"].copy()

        if candidate_available:
            lum_warped_ref, lum_warped_green_mask, lum_warped_blue_mask = apply_class_luminance_adjustment(
                hsv_only_warped_ref,
                green_lum_candidate,
                blue_lum_candidate,
                board_mask,
            )
            lum_ref, lum_ref_green_mask, lum_ref_blue_mask = apply_class_luminance_adjustment(
                hsv_only_ref,
                green_lum_candidate,
                blue_lum_candidate,
                None,
            )
            lum_tag_patch, _, _ = apply_class_luminance_adjustment(
                tag_color_debug["tag_ref_patch"],
                green_lum_candidate,
                blue_lum_candidate,
                None,
            )
            tag_error_after_by_class = compute_tag_classwise_error(
                lum_tag_patch,
                tag_color_debug["tag_frame_patch"],
                tag_green_mask,
                tag_blue_mask,
            )
            tag_error_after = float(tag_error_after_by_class["total_error"])
            tag_gate_passed = bool(tag_error_after <= tag_error_before * 1.02)
            if tag_gate_passed:
                calibrated_warped_ref = lum_warped_ref
                warped_green_mask = lum_warped_green_mask
                warped_blue_mask = lum_warped_blue_mask
                calibrated_ref = lum_ref
                ref_green_mask = lum_ref_green_mask
                ref_blue_mask = lum_ref_blue_mask
                tag_color_debug["tag_calibrated_ref_patch"] = lum_tag_patch
                green_luminance_adjustment = None if not green_lum_reliable else green_lum_candidate
                blue_luminance_adjustment = None if not blue_lum_reliable else blue_lum_candidate
                tag_color_debug["green_luminance_adjustment"] = green_luminance_adjustment
                tag_color_debug["blue_luminance_adjustment"] = blue_luminance_adjustment
                tag_luminance_applied = True
                calibration_source = f"{calibration_source}_plus_tag_luminance"
            else:
                tag_luminance_rejected = True
                calibration_source = f"{calibration_source}_tag_gate"
        else:
            tag_error_after_by_class = dict(tag_error_before_by_class)
        add_profile_time(profile, "color.tag_luminance_apply_gate", time.perf_counter() - stage_t)

    canonical_green_bgr = None
    canonical_blue_bgr = None
    warped_snap_debug = None
    ref_snap_debug = None
    if not fast_mode:
        stage_t = time.perf_counter()
        canonical_green_bgr = make_class_color_from_reference(
            ref_bgr,
            ref_green_mask,
            green_adj,
            green_luminance_adjustment,
        )
        canonical_blue_bgr = make_class_color_from_reference(
            ref_bgr,
            ref_blue_mask,
            blue_adj,
            blue_luminance_adjustment,
        )

        calibrated_warped_ref, warped_snap_debug = snap_reference_to_class_colors(
            calibrated_warped_ref,
            warped_green_mask,
            warped_blue_mask,
            board_mask,
            canonical_green_bgr=canonical_green_bgr,
            canonical_blue_bgr=canonical_blue_bgr,
        )
        calibrated_ref, ref_snap_debug = snap_reference_to_class_colors(
            calibrated_ref,
            ref_green_mask,
            ref_blue_mask,
            None,
            canonical_green_bgr=canonical_green_bgr,
            canonical_blue_bgr=canonical_blue_bgr,
        )
        if isinstance(tag_color_debug, dict) and tag_color_debug.get("tag_calibrated_ref_patch") is not None:
            tag_patch_green_mask, tag_patch_blue_mask, _ = build_tag_color_masks(tag_color_debug["tag_calibrated_ref_patch"])
            snapped_tag_patch, _ = snap_reference_to_class_colors(
                tag_color_debug["tag_calibrated_ref_patch"],
                tag_patch_green_mask.astype(np.uint8) * 255,
                tag_patch_blue_mask.astype(np.uint8) * 255,
                None,
                canonical_green_bgr=canonical_green_bgr,
                canonical_blue_bgr=canonical_blue_bgr,
            )
            tag_color_debug["tag_calibrated_ref_patch"] = snapped_tag_patch
        add_profile_time(profile, "color.snap_classes", time.perf_counter() - stage_t)
    else:
        stage_t = time.perf_counter()
        calibration_source = f"{calibration_source}_fast"
        if fast_color_calibration:
            calibration_source = f"{calibration_source}_board_color"
        if isinstance(tag_hsv_debug, dict):
            tag_color_debug = {
                "selected_tag_id": int(tag_hsv_debug["selected_tag_id"]),
                "selected_tag_center": np.asarray(tag_hsv_debug["selected_tag_center"], dtype=np.float32),
                "tag_green_adjustment": green_adj,
                "tag_blue_adjustment": blue_adj,
                "tag_green_source": tag_green_source,
                "tag_blue_source": tag_blue_source,
                "tag_green_pixel_count": tag_green_pixel_count,
                "tag_blue_pixel_count": tag_blue_pixel_count,
                "tag_error_before": None,
                "tag_error_after": None,
                "tag_error_before_by_class": None,
                "tag_error_after_by_class": None,
                "tag_gate_passed": False,
                "tag_luminance_applied": False,
                "tag_luminance_rejected": False,
                "fast_mode_skipped_luminance": True,
            }
        elif fast_color_calibration:
            tag_color_debug = {
                "selected_tag_id": None,
                "tag_green_adjustment": green_adj,
                "tag_blue_adjustment": blue_adj,
                "tag_green_source": tag_green_source,
                "tag_blue_source": tag_blue_source,
                "tag_green_pixel_count": tag_green_pixel_count,
                "tag_blue_pixel_count": tag_blue_pixel_count,
                "tag_error_before": None,
                "tag_error_after": None,
                "tag_error_before_by_class": None,
                "tag_error_after_by_class": None,
                "tag_gate_passed": False,
                "tag_luminance_applied": False,
                "tag_luminance_rejected": False,
                "fast_mode_skipped_luminance": True,
                "fast_color_calibration": True,
            }
        add_profile_time(profile, "color.fast_debug_packaging", time.perf_counter() - stage_t)

    eyedropper_debug = None
    if eyedropper_tag_colors:
        stage_t = time.perf_counter()
        eyedropper_debug = sample_top_left_tag_class_colors(
            ref_bgr,
            frame_bgr,
            ref_detections=ref_detections,
            frame_detections=frame_detections,
        )
        if isinstance(eyedropper_debug, dict):
            sampled_green_bgr = eyedropper_debug.get("green_bgr")
            sampled_blue_bgr = eyedropper_debug.get("blue_bgr")
            canonical_green_bgr = sampled_green_bgr if sampled_green_bgr is not None else canonical_green_bgr
            canonical_blue_bgr = sampled_blue_bgr if sampled_blue_bgr is not None else canonical_blue_bgr
            calibrated_warped_ref, warped_snap_debug = snap_reference_to_class_colors(
                calibrated_warped_ref,
                warped_green_mask,
                warped_blue_mask,
                board_mask,
                canonical_green_bgr=canonical_green_bgr,
                canonical_blue_bgr=canonical_blue_bgr,
            )
            calibrated_ref, ref_snap_debug = snap_reference_to_class_colors(
                calibrated_ref,
                ref_green_mask,
                ref_blue_mask,
                None,
                canonical_green_bgr=canonical_green_bgr,
                canonical_blue_bgr=canonical_blue_bgr,
            )
            calibration_source = f"{calibration_source}_eyedropper_tag_colors"
        add_profile_time(profile, "color.eyedropper_tag_colors", time.perf_counter() - stage_t)

    if isinstance(tag_color_debug, dict):
        stage_t = time.perf_counter()
        tag_color_debug["tag_green_adjustment"] = green_adj
        tag_color_debug["tag_blue_adjustment"] = blue_adj
        tag_color_debug["tag_green_source"] = tag_green_source
        tag_color_debug["tag_blue_source"] = tag_blue_source
        tag_color_debug["tag_green_pixel_count"] = tag_green_pixel_count
        tag_color_debug["tag_blue_pixel_count"] = tag_blue_pixel_count
        tag_color_debug["tag_error_before"] = tag_error_before
        tag_color_debug["tag_error_after"] = tag_error_after
        tag_color_debug["tag_error_before_by_class"] = tag_error_before_by_class
        tag_color_debug["tag_error_after_by_class"] = tag_error_after_by_class
        tag_color_debug["tag_gate_passed"] = bool(tag_gate_passed)
        tag_color_debug["tag_luminance_applied"] = bool(tag_luminance_applied)
        tag_color_debug["tag_luminance_rejected"] = bool(tag_luminance_rejected)
        add_profile_time(profile, "color.debug_packaging", time.perf_counter() - stage_t)

    return calibrated_warped_ref, calibrated_ref, {
        "auto_green": auto_green,
        "auto_blue": auto_blue,
        "green_adjustment": green_adj,
        "blue_adjustment": blue_adj,
        "green_luminance_adjustment": green_luminance_adjustment,
        "blue_luminance_adjustment": blue_luminance_adjustment,
        "auto_settings": auto_settings,
        "calibration_source": calibration_source,
        "warped_snap_debug": warped_snap_debug,
        "ref_snap_debug": ref_snap_debug,
        "blue_hue_choice_debug": blue_hue_choice_debug,
        "tag_green_adjustment": tag_green_adj,
        "tag_blue_adjustment": tag_blue_adj,
        "tag_green_source": tag_green_source,
        "tag_blue_source": tag_blue_source,
        "tag_green_pixel_count": tag_green_pixel_count,
        "tag_blue_pixel_count": tag_blue_pixel_count,
        "tag_error_before": tag_error_before,
        "tag_error_after": tag_error_after,
        "tag_gate_passed": bool(tag_gate_passed),
        "tag_luminance_applied": bool(tag_luminance_applied),
        "tag_luminance_rejected": bool(tag_luminance_rejected),
        "eyedropper_tag_colors": eyedropper_debug,
        "canonical_green_bgr": None if canonical_green_bgr is None else [int(v) for v in canonical_green_bgr.tolist()],
        "canonical_blue_bgr": None if canonical_blue_bgr is None else [int(v) for v in canonical_blue_bgr.tolist()],
        "warped_green_mask": warped_green_mask,
        "warped_blue_mask": warped_blue_mask,
        "ref_green_mask": ref_green_mask,
        "ref_blue_mask": ref_blue_mask,
        "tag_color_debug": tag_color_debug,
    }


def robust_channel_gain(frame_bgr: np.ndarray, warped_ref_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    aligned = warped_ref_bgr.astype(np.float32).copy()
    valid = mask > 0
    if valid.sum() < 100:
        return aligned

    eps = 1e-6
    for c in range(3):
        frm_vals = frame_bgr[..., c][valid].astype(np.float32)
        ref_vals = warped_ref_bgr[..., c][valid].astype(np.float32)

        # Exclude very dark values for a more stable brightness match.
        keep = ref_vals > 10
        if keep.sum() < 100:
            continue

        frm_med = np.median(frm_vals[keep])
        ref_med = np.median(ref_vals[keep])
        gain = frm_med / max(ref_med, eps)
        gain = float(np.clip(gain, 0.5, 2.0))
        aligned[..., c] *= gain

    return np.clip(aligned, 0, 255)


def compute_linear_luminance(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.dtype == np.uint8:
        b = _SRGB_TO_LINEAR_LUT[image_bgr[..., 0]]
        g = _SRGB_TO_LINEAR_LUT[image_bgr[..., 1]]
        r = _SRGB_TO_LINEAR_LUT[image_bgr[..., 2]]
    else:
        bgr = np.clip(image_bgr.astype(np.float32), 0.0, 255.0) / 255.0
        b = np.where(
            bgr[..., 0] <= 0.04045,
            bgr[..., 0] / 12.92,
            ((bgr[..., 0] + 0.055) / 1.055) ** 2.4,
        )
        g = np.where(
            bgr[..., 1] <= 0.04045,
            bgr[..., 1] / 12.92,
            ((bgr[..., 1] + 0.055) / 1.055) ** 2.4,
        )
        r = np.where(
            bgr[..., 2] <= 0.04045,
            bgr[..., 2] / 12.92,
            ((bgr[..., 2] + 0.055) / 1.055) ** 2.4,
        )

    luminance = (0.0722 * b) + (0.7152 * g) + (0.2126 * r)
    return luminance.astype(np.float32) * 255.0


def blur_lightness(image_bgr: np.ndarray, ksize: int = 31) -> np.ndarray:
    l = compute_linear_luminance(image_bgr)
    if ksize % 2 == 0:
        ksize += 1
    return gaussian_blur_runtime(l, (ksize, ksize), 0)


def warp_frame_to_reference_space(
    frame_bgr: np.ndarray,
    board_mask: np.ndarray,
    H: np.ndarray,
    ref_shape: tuple[int, int],
):
    ref_h, ref_w = ref_shape[:2]
    H_inv = np.linalg.inv(H)
    frame_rectified = warp_perspective_runtime(frame_bgr, H_inv, (ref_w, ref_h))
    valid_mask = warp_perspective_runtime(board_mask, H_inv, (ref_w, ref_h))
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return frame_rectified, valid_mask


def fit_texture_to_reference(texture_bgr: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    ref_h, ref_w = ref_shape[:2]
    tex_h, tex_w = texture_bgr.shape[:2]
    if tex_h <= 0 or tex_w <= 0:
        raise ValueError("Texture image has invalid dimensions.")

    scale = max(ref_w / tex_w, ref_h / tex_h)
    resized_w = max(1, int(round(tex_w * scale)))
    resized_h = max(1, int(round(tex_h * scale)))
    resized = cv2.resize(texture_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (resized_w - ref_w) // 2)
    y0 = max(0, (resized_h - ref_h) // 2)
    fitted = resized[y0:y0 + ref_h, x0:x0 + ref_w]
    if fitted.shape[0] != ref_h or fitted.shape[1] != ref_w:
        fitted = cv2.resize(fitted, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    return fitted


def compute_rectified_shadow_residual(
    ref_bgr: np.ndarray,
    frame_rectified_bgr: np.ndarray,
    valid_mask: np.ndarray,
    blur_ksize: int = 31,
    min_ref_lightness: float = 40.0,
):
    ref_l_raw = compute_linear_luminance(ref_bgr)
    frame_l_raw = compute_linear_luminance(frame_rectified_bgr)
    ref_l_blur = blur_lightness(ref_bgr, blur_ksize)
    frame_l_blur = blur_lightness(frame_rectified_bgr, blur_ksize)

    absolute_drop_blur = np.zeros_like(ref_l_blur, dtype=np.float32)
    residual = np.zeros_like(ref_l_blur, dtype=np.float32)
    raw_absolute_drop = np.zeros_like(ref_l_raw, dtype=np.float32)
    valid = valid_mask > 0
    absolute_drop_blur[valid] = np.clip(ref_l_blur[valid] - frame_l_blur[valid], 0, 255)
    raw_absolute_drop[valid] = np.clip(ref_l_raw[valid] - frame_l_raw[valid], 0, 255)

    # Use only the blurred luminance drop as the primary shadow signal so
    # high-frequency tag edges and warp/compression mismatches do not drive
    # the residual used for shadow extraction.
    denom = np.maximum(ref_l_blur, min_ref_lightness)
    residual[valid] = np.clip((absolute_drop_blur[valid] / denom[valid]) * 255.0, 0, 255)
    return residual, raw_absolute_drop


def filter_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, np.ndarray, int]:
    filtered_occlusion_mask = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    removed_component_count = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered_occlusion_mask[labels == label_id] = 255
        else:
            removed_component_count += 1
    removed_mask = cv2.bitwise_and(mask, cv2.bitwise_not(filtered_occlusion_mask))
    return filtered_occlusion_mask, removed_mask, removed_component_count


def compute_hand_occlusion_mask(
    frame_bgr: np.ndarray,
    aligned_ref_bgr: np.ndarray,
    board_mask: np.ndarray,
    binary_threshold: float = OCCLUSION_DIFF_THRESHOLD,
    thin_line_kernel_size: int = OCCLUSION_THIN_LINE_KERNEL_SIZE,
    min_component_area: int = MIN_OCCLUSION_COMPONENT_AREA,
    protected_surface_mask: np.ndarray | None = None,
):
    frame_f = frame_bgr.astype(np.float32)
    ref_f = aligned_ref_bgr.astype(np.float32)
    signed_bgr_diff = frame_f - ref_f
    abs_bgr_diff = np.abs(signed_bgr_diff)
    bgr_abs_diff = np.mean(abs_bgr_diff, axis=2)
    chroma_diff = np.linalg.norm(
        signed_bgr_diff - np.mean(signed_bgr_diff, axis=2, keepdims=True),
        axis=2,
    )
    frame_l = compute_linear_luminance(frame_bgr)
    ref_l = compute_linear_luminance(aligned_ref_bgr)
    luminance_drop = ref_l - frame_l
    hsv = cv2.cvtColor(frame_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame_bgr.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycrcb)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    skin_seed = (
        (board_mask > 0)
        & (cr_chan >= 135)
        & (cr_chan <= 175)
        & (cb_chan >= 80)
        & (cb_chan <= 135)
        & (y_chan >= 45)
        & (((h_chan <= 28) | (h_chan >= 165)))
        & (s_chan >= 15)
        & (s_chan <= 155)
        & (v_chan >= 50)
    ).astype(np.uint8) * 255
    skin_seed = cv2.morphologyEx(
        skin_seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    skin_seed = cv2.morphologyEx(
        skin_seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )

    white_mask = np.zeros_like(board_mask, dtype=np.uint8)
    candidate = (board_mask > 0) & (bgr_abs_diff >= binary_threshold)

    # Shadows are usually a moderate, fairly neutral darkening of the reference.
    # Hands/pens tend to have either stronger chroma direction or much larger
    # absolute difference. Exclude only the neutral/moderate darkening so shadows
    # can remain in the shadow mask instead of becoming occlusion.
    max_signed_channel = np.max(signed_bgr_diff, axis=2)
    shadow_like = (
        (board_mask > 0)
        & (luminance_drop >= float(binary_threshold) * 0.35)
        & (bgr_abs_diff <= float(binary_threshold) * 2.0)
        & (chroma_diff <= np.maximum(22.0, bgr_abs_diff * 0.42))
        & (max_signed_channel <= 12.0)
    )
    candidate &= ~shadow_like

    ref_gray = cv2.cvtColor(aligned_ref_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ref_edges = cv2.Canny(ref_gray, 24, 72)
    ref_edges = cv2.dilate(ref_edges, np.ones((5, 5), np.uint8), iterations=1) > 0

    protected = np.zeros_like(candidate, dtype=bool)
    if protected_surface_mask is not None:
        protected = protected_surface_mask > 0
        strong_protected = (
            protected
            & (bgr_abs_diff >= float(binary_threshold) * 1.45)
            & (chroma_diff >= float(binary_threshold) * 0.45)
            & ~ref_edges
        )
        candidate = (candidate & ~protected) | strong_protected
    else:
        candidate &= ~ref_edges

    skin_area = int(np.count_nonzero(skin_seed))
    if skin_area >= max(1200, int(min_component_area) * 4):
        skin_neighborhood = cv2.dilate(
            skin_seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (155, 155)),
            iterations=1,
        ) > 0
        dark_tool_candidate = (
            (board_mask > 0)
            & skin_neighborhood
            & (v_chan <= 110)
            & (bgr_abs_diff >= max(10.0, float(binary_threshold) * 0.2))
        ).astype(np.uint8) * 255
        dark_tool_candidate = cv2.morphologyEx(
            dark_tool_candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        dark_tool_seed = np.zeros_like(board_mask, dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_tool_candidate, connectivity=8)
        max_tool_area = max(8000, int(min_component_area) * 20)
        max_tool_width = max(24, int(round(board_mask.shape[1] * 0.25)))
        max_tool_height = max(24, int(round(board_mask.shape[0] * 0.20)))
        max_tool_y = int(round(board_mask.shape[0] * 0.65))
        for label_id in range(1, num_labels):
            x, y, comp_w, comp_h, area = stats[label_id]
            if (
                area >= 20
                and area <= max_tool_area
                and comp_w <= max_tool_width
                and comp_h <= max_tool_height
                and y <= max_tool_y
            ):
                dark_tool_seed[labels == label_id] = 255
        dark_tool_seed = cv2.morphologyEx(
            dark_tool_seed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        tool_neighborhood = cv2.dilate(
            dark_tool_seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
            iterations=1,
        ) > 0
        dark_tip_candidate = (
            (board_mask > 0)
            & tool_neighborhood
            & (v_chan <= 120)
        ).astype(np.uint8) * 255
        dark_tip_seed = np.zeros_like(board_mask, dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_tip_candidate, connectivity=8)
        for label_id in range(1, num_labels):
            x, y, comp_w, comp_h, area = stats[label_id]
            if (
                area >= 10
                and area <= max_tool_area
                and comp_w <= max_tool_width
                and comp_h <= max_tool_height
                and y <= max_tool_y
            ):
                dark_tip_seed[labels == label_id] = 255
        dark_tip_seed = cv2.morphologyEx(
            dark_tip_seed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        dark_tool_seed = cv2.bitwise_or(dark_tool_seed, dark_tip_seed)
        candidate = (skin_seed > 0) | (dark_tool_seed > 0)
    else:
        dark_tool_seed = np.zeros_like(board_mask, dtype=np.uint8)

    white_mask[candidate] = 255

    thin_kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (int(thin_line_kernel_size), int(thin_line_kernel_size)),
    )
    opened_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, thin_kernel, iterations=1)
    if int(np.count_nonzero(skin_seed)) >= max(1200, int(min_component_area) * 4):
        opened_mask = cv2.bitwise_or(opened_mask, skin_seed)
        opened_mask = cv2.bitwise_or(opened_mask, dark_tool_seed)
    thin_line_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(opened_mask))
    occlusion_mask, small_region_mask, removed_small_region_count = filter_small_components(
        opened_mask,
        min_component_area,
    )

    return occlusion_mask, {
        "contrast_brightness_difference": white_mask,
        "bgr_abs_diff": bgr_abs_diff,
        "abs_bgr_diff": abs_bgr_diff,
        "signed_bgr_diff": signed_bgr_diff,
        "chroma_diff": chroma_diff,
        "luminance_drop": luminance_drop,
        "skin_seed_mask": skin_seed,
        "dark_tool_seed_mask": dark_tool_seed,
        "shadow_like_exclusion_mask": shadow_like.astype(np.uint8) * 255,
        "reference_edge_mask": ref_edges.astype(np.uint8) * 255,
        "protected_surface_mask": protected.astype(np.uint8) * 255,
        "white_mask_before_thin_line_removal": white_mask,
        "thin_line_mask": thin_line_mask,
        "small_region_mask": small_region_mask,
        "thin_line_debug": {
            "removed_pixel_count": int(np.count_nonzero(thin_line_mask)),
            "removed_small_region_pixel_count": int(np.count_nonzero(small_region_mask)),
            "removed_small_region_count": removed_small_region_count,
            "binary_threshold": float(binary_threshold),
            "thin_line_kernel_size": int(thin_line_kernel_size),
            "min_component_area": int(min_component_area),
        },
    }


def make_overlay(
    frame_bgr: np.ndarray,
    board_mask: np.ndarray,
    shadow_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    shadow_intensity: np.ndarray | None = None,
    board_color_mask: np.ndarray | None = None,
    board_fill_bgr: np.ndarray | None = None,
) -> np.ndarray:
    overlay = frame_bgr.copy().astype(np.float32)
    frame_float = frame_bgr.astype(np.float32)

    board_fill = board_fill_bgr if board_fill_bgr is not None else np.full_like(frame_bgr, 255, dtype=np.uint8)
    board_fill_float = board_fill.astype(np.float32)
    shadow_color = np.zeros_like(overlay)

    board_u8 = (board_mask > 0).astype(np.uint8) * 255
    board_color_u8 = np.zeros_like(board_u8)
    if board_color_mask is not None:
        board_color_u8 = cv2.bitwise_and((board_color_mask > 0).astype(np.uint8) * 255, board_u8)
    board_render_mask = cv2.morphologyEx(
        board_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    board_render_mask = cv2.dilate(
        board_render_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    board_render_mask = cv2.bitwise_or(board_render_mask, board_color_u8)
    occlusion_render_mask = (occlusion_mask > 0).astype(np.uint8) * 255
    occlusion_render_mask = cv2.dilate(
        occlusion_render_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    occlusion_render_mask = cv2.bitwise_and(occlusion_render_mask, board_render_mask)

    board_idx = board_render_mask > 0

    # Render onto the synthetic board surface before compositing occlusion and shadow.
    overlay[board_idx] = board_fill_float[board_idx]

    # Keep occluded regions as the original frame so the hand/object remains visible.
    occl_idx = occlusion_render_mask > 0
    overlay[occl_idx] = frame_float[occl_idx]

    shadow_idx = shadow_mask > 0

    if shadow_intensity is None:
        shadow_alpha = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
        shadow_alpha[shadow_idx] = 0.08
    else:
        shadow_alpha = shadow_intensity.astype(np.float32) / 255.0
        shadow_alpha = np.clip(shadow_alpha, 0.0, 0.35)
        shadow_alpha[~shadow_idx] = 0.0
        shadow_alpha[occl_idx] = 0.0

    # Feather the rendered shadow edge so the overlay reads more like a soft
    # shadow field than a hard binary cutout.
    shadow_alpha = smooth_masked_field(shadow_alpha, board_mask, sigma=6.0)
    shadow_alpha[occl_idx] = 0.0
    shadow_alpha = np.clip(shadow_alpha, 0.0, 0.35)

    # Blend shadow pixels as transparent black with opacity tied to shadow intensity.
    shadow_alpha_3 = shadow_alpha[..., None]
    overlay = overlay * (1.0 - shadow_alpha_3) + shadow_color * shadow_alpha_3

    return np.clip(overlay, 0, 255).astype(np.uint8)


def normalize_to_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if np.allclose(x.max(), x.min()):
        return np.zeros_like(x, dtype=np.uint8)
    x = x - x.min()
    x = x / max(x.max(), 1e-6)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def smooth_masked_field(field: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    if use_cuda_runtime():
        try:
            field_gpu = gpu_mat_from_array(field.astype(np.float32))
            valid_mask_gpu = gpu_mat_from_array(valid_mask.astype(np.uint8))
            return smooth_masked_field_gpu(field_gpu, valid_mask_gpu, sigma).download()
        except Exception as exc:
            print(f"[WARN] CUDA smooth_masked_field failed, falling back to CPU: {exc}")
            set_cuda_runtime_enabled(False)
    weight = (valid_mask > 0).astype(np.float32)
    blurred_num = gaussian_blur_runtime(field.astype(np.float32) * weight, (0, 0), sigma)
    blurred_den = gaussian_blur_runtime(weight, (0, 0), sigma)
    out = np.zeros_like(field, dtype=np.float32)
    keep = blurred_den > 1e-3
    out[keep] = blurred_num[keep] / blurred_den[keep]
    return out


def smooth_masked_field_with_weight(
    field: np.ndarray,
    weight: np.ndarray,
    blurred_den: np.ndarray,
    sigma: float,
) -> np.ndarray:
    blurred_num = gaussian_blur_runtime(field.astype(np.float32) * weight, (0, 0), sigma)
    out = np.zeros_like(field, dtype=np.float32)
    keep = blurred_den > 1e-3
    out[keep] = blurred_num[keep] / blurred_den[keep]
    return out


def feather_replace_masked_field(
    field: np.ndarray,
    replace_mask: np.ndarray,
    valid_mask: np.ndarray,
    sigma_fill: float = 24.0,
    feather_radius: int = 9,
) -> np.ndarray:
    if replace_mask is None or np.count_nonzero(replace_mask) == 0:
        return field.astype(np.float32)

    fill_support = ((valid_mask > 0) & ~(replace_mask > 0)).astype(np.uint8) * 255
    if int(np.count_nonzero(fill_support)) < 50:
        return field.astype(np.float32)

    fill = smooth_masked_field(field.astype(np.float32), fill_support, sigma=sigma_fill)
    return feather_replace_in_mask(field.astype(np.float32), fill, replace_mask, feather_radius=feather_radius)


def feather_replace_with_alpha(field: np.ndarray, fill: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = field.astype(np.float32).copy()
    out = out * (1.0 - alpha) + fill.astype(np.float32) * alpha
    return out.astype(np.float32)


def fill_masked_regions_adaptive(
    field: np.ndarray,
    replace_mask: np.ndarray,
    valid_mask: np.ndarray,
    min_sigma: float = 24.0,
    sigma_scale: float = 0.6,
    feather_radius: int = 11,
    min_component_area: int = 50,
) -> np.ndarray:
    if replace_mask is None or np.count_nonzero(replace_mask) == 0:
        return field.astype(np.float32)

    replace_u8 = cv2.bitwise_and((replace_mask > 0).astype(np.uint8) * 255, (valid_mask > 0).astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(replace_u8, connectivity=8)
    if num_labels <= 1:
        return field.astype(np.float32)

    out = field.astype(np.float32).copy()
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        sigma = float(max(min_sigma, sigma_scale * max(w, h)))
        padding = int(max(feather_radius * 2, np.ceil(3.0 * sigma)))

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(out.shape[1], x + w + padding)
        y1 = min(out.shape[0], y + h + padding)

        local_field = out[y0:y1, x0:x1]
        local_component_mask = (labels[y0:y1, x0:x1] == label_id).astype(np.uint8) * 255
        local_fill_support = (((valid_mask[y0:y1, x0:x1] > 0) & ~(local_component_mask > 0)).astype(np.uint8) * 255)
        if int(np.count_nonzero(local_fill_support)) < 50:
            continue

        local_fill = smooth_masked_field(local_field, local_fill_support, sigma=sigma)
        out[y0:y1, x0:x1] = feather_replace_in_mask(
            local_field,
            local_fill,
            local_component_mask,
            feather_radius=feather_radius,
        )

    return out.astype(np.float32)


def fill_masked_regions_adaptive_pair(
    field_a: np.ndarray,
    field_b: np.ndarray,
    replace_mask: np.ndarray,
    valid_mask: np.ndarray,
    min_sigma: float = 24.0,
    sigma_scale: float = 0.6,
    feather_radius: int = 11,
    min_component_area: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    if replace_mask is None or np.count_nonzero(replace_mask) == 0:
        return field_a.astype(np.float32), field_b.astype(np.float32)

    replace_u8 = cv2.bitwise_and((replace_mask > 0).astype(np.uint8) * 255, (valid_mask > 0).astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(replace_u8, connectivity=8)
    if num_labels <= 1:
        return field_a.astype(np.float32), field_b.astype(np.float32)

    out_a = field_a.astype(np.float32).copy()
    out_b = field_b.astype(np.float32).copy()
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        sigma = float(max(min_sigma, sigma_scale * max(w, h)))
        padding = int(max(feather_radius * 2, np.ceil(3.0 * sigma)))

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(out_a.shape[1], x + w + padding)
        y1 = min(out_a.shape[0], y + h + padding)

        local_component_mask = (labels[y0:y1, x0:x1] == label_id).astype(np.uint8) * 255
        local_fill_support = (((valid_mask[y0:y1, x0:x1] > 0) & ~(local_component_mask > 0)).astype(np.uint8) * 255)
        if int(np.count_nonzero(local_fill_support)) < 50:
            continue

        weight = (local_fill_support > 0).astype(np.float32)
        blurred_den = gaussian_blur_runtime(weight, (0, 0), sigma)

        mask_u8 = (local_component_mask > 0).astype(np.uint8) * 255
        kernel_size = feather_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(mask_u8, kernel, iterations=1)
        dist = cv2.distanceTransform(cv2.bitwise_not(expanded), cv2.DIST_L2, 3)
        alpha = np.clip(1.0 - (dist / max(float(feather_radius), 1.0)), 0.0, 1.0)
        alpha[expanded == 0] = 0.0

        local_a = out_a[y0:y1, x0:x1]
        fill_a = smooth_masked_field_with_weight(local_a, weight, blurred_den, sigma)
        out_a[y0:y1, x0:x1] = feather_replace_with_alpha(local_a, fill_a, alpha)

        local_b = out_b[y0:y1, x0:x1]
        fill_b = smooth_masked_field_with_weight(local_b, weight, blurred_den, sigma)
        out_b[y0:y1, x0:x1] = feather_replace_with_alpha(local_b, fill_b, alpha)

    return out_a.astype(np.float32), out_b.astype(np.float32)


def suppress_rectified_structure_edges(
    shadow_field: np.ndarray,
    ref_bgr: np.ndarray,
    frame_rectified_bgr: np.ndarray,
    valid_mask: np.ndarray,
    edge_sigma: float = 1.2,
    grad_percentile: float = 75.0,
    suppress_strength: float = 1.0,
    dilate_size: int = 5,
):
    valid = valid_mask > 0
    if int(np.count_nonzero(valid)) < 100:
        return shadow_field.astype(np.float32), np.zeros_like(valid_mask, dtype=np.uint8), {
            "applied": False,
            "edge_sigma": float(edge_sigma),
            "grad_percentile": float(grad_percentile),
            "suppress_strength": float(suppress_strength),
            "dilate_size": int(dilate_size),
            "edge_mask_pixel_count": 0,
            "ref_grad_threshold": 0.0,
            "frame_grad_threshold": 0.0,
        }

    ref_l = compute_linear_luminance(ref_bgr).astype(np.float32)
    frame_l = compute_linear_luminance(frame_rectified_bgr).astype(np.float32)
    if edge_sigma > 0:
        ref_l = gaussian_blur_runtime(ref_l, (0, 0), edge_sigma)
        frame_l = gaussian_blur_runtime(frame_l, (0, 0), edge_sigma)

    def gradient_magnitude(luminance: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def percentile_threshold(vals: np.ndarray) -> tuple[float, bool]:
        positive = vals[vals > 1e-3]
        if positive.size == 0:
            return 0.0, False
        return float(max(np.percentile(positive, grad_percentile), 1.0)), True

    ref_grad = gradient_magnitude(ref_l)
    frame_grad = gradient_magnitude(frame_l)
    ref_vals = ref_grad[valid]
    frame_vals = frame_grad[valid]
    ref_thresh, ref_active = percentile_threshold(ref_vals)
    frame_thresh, frame_active = percentile_threshold(frame_vals)

    structure_mask = valid & (
        ((ref_grad > ref_thresh) if ref_active else np.zeros_like(valid, dtype=bool)) |
        ((frame_grad > frame_thresh) if frame_active else np.zeros_like(valid, dtype=bool))
    )
    if dilate_size > 1 and int(np.count_nonzero(structure_mask)) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        structure_mask = cv2.dilate(structure_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
        structure_mask &= valid

    edge_mask_u8 = structure_mask.astype(np.uint8) * 255
    if int(np.count_nonzero(edge_mask_u8)) == 0:
        return shadow_field.astype(np.float32), edge_mask_u8, {
            "applied": False,
            "edge_sigma": float(edge_sigma),
            "grad_percentile": float(grad_percentile),
            "suppress_strength": float(suppress_strength),
            "dilate_size": int(dilate_size),
            "edge_mask_pixel_count": 0,
            "ref_grad_threshold": ref_thresh,
            "frame_grad_threshold": frame_thresh,
        }

    fill = feather_replace_masked_field(
        shadow_field.astype(np.float32),
        edge_mask_u8,
        valid_mask,
        sigma_fill=24.0,
        feather_radius=9,
    )
    suppressed = shadow_field.astype(np.float32).copy()
    alpha = (edge_mask_u8 > 0).astype(np.float32) * float(np.clip(suppress_strength, 0.0, 1.0))
    suppressed = suppressed * (1.0 - alpha) + fill * alpha
    suppressed[~valid] = 0
    return suppressed.astype(np.float32), edge_mask_u8, {
        "applied": True,
        "edge_sigma": float(edge_sigma),
        "grad_percentile": float(grad_percentile),
        "suppress_strength": float(suppress_strength),
        "dilate_size": int(dilate_size),
        "edge_mask_pixel_count": int(np.count_nonzero(edge_mask_u8)),
        "ref_grad_threshold": ref_thresh,
        "frame_grad_threshold": frame_thresh,
    }


def harmonize_class_boundary_residual(
    field: np.ndarray,
    green_mask: np.ndarray | None,
    blue_mask: np.ndarray | None,
    board_mask: np.ndarray,
    green_erode_size: int = 5,
    blue_dilate_size: int = 7,
    sigma: float = 3.0,
) -> np.ndarray:
    if green_mask is None or blue_mask is None:
        return field

    board_u8 = (board_mask > 0).astype(np.uint8) * 255
    green_u8 = cv2.bitwise_and((green_mask > 0).astype(np.uint8) * 255, board_u8)
    blue_u8 = cv2.bitwise_and((blue_mask > 0).astype(np.uint8) * 255, board_u8)
    if cv2.countNonZero(green_u8) == 0 or cv2.countNonZero(blue_u8) == 0:
        return field

    green_kernel = np.ones((green_erode_size, green_erode_size), np.uint8)
    blue_kernel = np.ones((blue_dilate_size, blue_dilate_size), np.uint8)
    green_core = cv2.erode(green_u8, green_kernel, iterations=1)
    blue_outer = cv2.dilate(blue_u8, blue_kernel, iterations=1)
    blue_outer = cv2.bitwise_and(blue_outer, board_u8)

    seam = cv2.bitwise_and(green_u8, cv2.bitwise_not(green_core))
    seam = cv2.bitwise_and(seam, blue_outer)
    if cv2.countNonZero(seam) == 0:
        return field

    blue_fill = smooth_masked_field(field, blue_u8, sigma=sigma)
    out = field.copy()
    seam_idx = seam > 0
    out[seam_idx] = blue_fill[seam_idx]
    return out


def center_class_residual_baselines(
    field: np.ndarray,
    green_mask: np.ndarray | None,
    blue_mask: np.ndarray | None,
    board_mask: np.ndarray,
    trim_percent: float = 15.0,
):
    if green_mask is None or blue_mask is None:
        return field, None

    valid = board_mask > 0
    green = (green_mask > 0) & valid
    blue = (blue_mask > 0) & valid
    if int(np.count_nonzero(green)) < 100 or int(np.count_nonzero(blue)) < 100:
        return field, None

    def trimmed_median(vals: np.ndarray):
        if vals.size == 0:
            return 0.0
        lo = np.percentile(vals, trim_percent)
        hi = np.percentile(vals, 100.0 - trim_percent)
        keep = vals[(vals >= lo) & (vals <= hi)]
        if keep.size == 0:
            keep = vals
        return float(np.median(keep))

    green_med = trimmed_median(field[green].astype(np.float32))
    blue_med = trimmed_median(field[blue].astype(np.float32))
    target = min(green_med, blue_med)

    out = field.copy()
    out[green] = np.clip(out[green] - (green_med - target), 0, 255)
    out[blue] = np.clip(out[blue] - (blue_med - target), 0, 255)
    return out, {
        "green_median": green_med,
        "blue_median": blue_med,
        "target": target,
    }


def harmonize_blue_tiles_to_surround(
    field: np.ndarray,
    blue_mask: np.ndarray | None,
    board_mask: np.ndarray,
    outer_radius: int = 9,
    inner_radius: int = 2,
    edge_band: int = 10,
):
    if blue_mask is None:
        return field, None

    blue_u8 = cv2.bitwise_and((blue_mask > 0).astype(np.uint8) * 255, (board_mask > 0).astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_u8, connectivity=8)
    if num_labels <= 1:
        return field, {"tile_count": 0, "tiles": []}

    out = field.copy()
    debug_tiles = []
    outer_kernel = np.ones((outer_radius * 2 + 1, outer_radius * 2 + 1), np.uint8)
    inner_kernel = np.ones((inner_radius * 2 + 1, inner_radius * 2 + 1), np.uint8)

    def robust_median(vals: np.ndarray):
        if vals.size < 20:
            return None
        lo = np.percentile(vals, 15.0)
        hi = np.percentile(vals, 85.0)
        keep = vals[(vals >= lo) & (vals <= hi)]
        if keep.size < 10:
            keep = vals
        return float(np.median(keep))

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 200:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x0 = x
        y0 = y
        x1 = x + w
        y1 = y + h
        pad = max(outer_radius + 2, edge_band + 2)
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(field.shape[1], x1 + pad)
        ry1 = min(field.shape[0], y1 + pad)

        tile_mask = labels[ry0:ry1, rx0:rx1] == label_id
        tile_mask_u8 = tile_mask.astype(np.uint8) * 255
        local_board = board_mask[ry0:ry1, rx0:rx1] > 0
        local_blue = blue_u8[ry0:ry1, rx0:rx1] > 0
        local_field = field[ry0:ry1, rx0:rx1].astype(np.float32)

        dilated_outer = cv2.dilate(tile_mask_u8, outer_kernel, iterations=1) > 0
        dilated_inner = cv2.dilate(tile_mask_u8, inner_kernel, iterations=1) > 0
        surround = dilated_outer & (~dilated_inner) & local_board & ~local_blue
        if int(np.count_nonzero(surround)) < 80:
            continue

        local_h, local_w = tile_mask.shape
        local_x0 = x0 - rx0
        local_y0 = y0 - ry0
        local_x1 = local_x0 + w - 1
        local_y1 = local_y0 + h - 1
        bw = max(1, min(edge_band, w // 3 if w >= 3 else 1))
        bh = max(1, min(edge_band, h // 3 if h >= 3 else 1))

        yy = np.arange(local_h)[:, None]
        xx = np.arange(local_w)[None, :]
        top_in = tile_mask & (yy <= local_y0 + bh - 1)
        bottom_in = tile_mask & (yy >= local_y1 - bh + 1)
        left_in = tile_mask & (xx <= local_x0 + bw - 1)
        right_in = tile_mask & (xx >= local_x1 - bw + 1)

        top_out = surround & (yy < local_y0)
        bottom_out = surround & (yy > local_y1)
        left_out = surround & (xx < local_x0)
        right_out = surround & (xx > local_x1)

        top_diff = None
        bottom_diff = None
        left_diff = None
        right_diff = None

        top_in_med = robust_median(local_field[top_in])
        top_out_med = robust_median(local_field[top_out])
        if top_in_med is not None and top_out_med is not None:
            top_diff = top_in_med - top_out_med

        bottom_in_med = robust_median(local_field[bottom_in])
        bottom_out_med = robust_median(local_field[bottom_out])
        if bottom_in_med is not None and bottom_out_med is not None:
            bottom_diff = bottom_in_med - bottom_out_med

        left_in_med = robust_median(local_field[left_in])
        left_out_med = robust_median(local_field[left_out])
        if left_in_med is not None and left_out_med is not None:
            left_diff = left_in_med - left_out_med

        right_in_med = robust_median(local_field[right_in])
        right_out_med = robust_median(local_field[right_out])
        if right_in_med is not None and right_out_med is not None:
            right_diff = right_in_med - right_out_med

        diffs = [d for d in [top_diff, bottom_diff, left_diff, right_diff] if d is not None]
        if not diffs:
            continue

        top_diff = float(np.median(diffs) if top_diff is None else top_diff)
        bottom_diff = float(np.median(diffs) if bottom_diff is None else bottom_diff)
        left_diff = float(np.median(diffs) if left_diff is None else left_diff)
        right_diff = float(np.median(diffs) if right_diff is None else right_diff)

        xn = np.zeros((h, w), dtype=np.float32)
        yn = np.zeros((h, w), dtype=np.float32)
        if w > 1:
            xn[:] = (np.arange(w, dtype=np.float32)[None, :]) / float(w - 1)
        if h > 1:
            yn[:] = (np.arange(h, dtype=np.float32)[:, None]) / float(h - 1)

        offset_x = (1.0 - xn) * left_diff + xn * right_diff
        offset_y = (1.0 - yn) * top_diff + yn * bottom_diff
        offset = 0.5 * (offset_x + offset_y)
        out_roi = out[y0:y1, x0:x1]
        out_roi[tile_mask[local_y0:local_y0 + h, local_x0:local_x0 + w]] = np.clip(
            out_roi[tile_mask[local_y0:local_y0 + h, local_x0:local_x0 + w]] -
            offset[tile_mask[local_y0:local_y0 + h, local_x0:local_x0 + w]],
            0,
            255,
        )
        out[y0:y1, x0:x1] = out_roi

        debug_tiles.append(
            {
                "bbox": [x0, y0, x1 - 1, y1 - 1],
                "area": area,
                "top_diff": top_diff,
                "bottom_diff": bottom_diff,
                "left_diff": left_diff,
                "right_diff": right_diff,
            }
        )

    return out, {
        "tile_count": len(debug_tiles),
        "tiles": debug_tiles,
    }


def feather_replace_in_mask(
    field: np.ndarray,
    fill: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 12,
) -> np.ndarray:
    if mask is None or np.count_nonzero(mask) == 0:
        return field

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if feather_radius <= 0:
        out = field.copy()
        out[mask_u8 > 0] = fill[mask_u8 > 0]
        return out

    kernel_size = feather_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(mask_u8, kernel, iterations=1)

    dist = cv2.distanceTransform(cv2.bitwise_not(expanded), cv2.DIST_L2, 3)
    alpha = np.clip(1.0 - (dist / max(float(feather_radius), 1.0)), 0.0, 1.0)
    alpha[expanded == 0] = 0.0

    out = field.astype(np.float32).copy()
    out = out * (1.0 - alpha) + fill.astype(np.float32) * alpha
    return out.astype(np.float32)


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    h, w = filled.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, holes)


def fill_contours(mask: np.ndarray, min_area: float = 500.0) -> np.ndarray:
    out = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(out, [contour], -1, 255, thickness=cv2.FILLED)
    return out


def build_shadow_mask(
    shadow_field: np.ndarray,
    board_mask: np.ndarray,
    threshold: float,
    open_kernel_size: int = 3,
    close_kernel_size: int = 61,
    min_area: float = 2000.0,
) -> np.ndarray:
    shadow_mask = np.zeros(shadow_field.shape[:2], dtype=np.uint8)
    shadow_mask[
        (board_mask > 0) &
        (shadow_field > float(threshold))
    ] = 255
    kernel_small = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=1)
    shadow_mask = fill_binary_holes(shadow_mask)
    shadow_mask = fill_contours(shadow_mask, min_area=min_area)
    return shadow_mask


def make_shadow_intensity_image(shadow_residual: np.ndarray, board_mask: np.ndarray) -> np.ndarray:
    valid = board_mask > 0
    out = np.zeros_like(shadow_residual, dtype=np.uint8)
    if int(np.count_nonzero(valid)) < 50:
        return out

    smoothed = smooth_masked_field(shadow_residual, board_mask, sigma=14.0)
    out[valid] = np.clip(smoothed[valid], 0, 255).astype(np.uint8)
    return out


def make_frame_space_shadow_intensity(
    frame_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    board_mask: np.ndarray,
    sigma: float = 20.0,
    min_ref_lightness: float = 40.0,
    frame_gpu=None,
    mask_gpu=None,
) -> np.ndarray:
    if use_cuda_runtime():
        try:
            frame_gpu = gpu_mat_from_array(frame_bgr) if frame_gpu is None else frame_gpu
            ref_gpu = gpu_mat_from_array(ref_bgr.astype(np.uint8))
            mask_gpu = gpu_mat_from_array(board_mask.astype(np.uint8)) if mask_gpu is None else mask_gpu
            valid = board_mask > 0

            frame_l_gpu = compute_linear_luminance_gpu(frame_gpu)
            ref_l_gpu = compute_linear_luminance_gpu(ref_gpu)
            frame_s_gpu = smooth_masked_field_gpu(frame_l_gpu, mask_gpu, sigma=sigma)
            ref_s_gpu = smooth_masked_field_gpu(ref_l_gpu, mask_gpu, sigma=sigma)

            denom_gpu = cv2.cuda.maxWithScalar(ref_s_gpu, float(min_ref_lightness))
            drop_gpu = cv2.cuda.subtract(ref_s_gpu, frame_s_gpu)
            drop_gpu = clip_gpu_field(drop_gpu)
            drop_gpu = cv2.cuda.divide(drop_gpu, denom_gpu)
            drop_gpu = cv2.cuda.multiplyWithScalar(drop_gpu, 255.0)
            drop_gpu = clip_gpu_field(drop_gpu)

            drop = drop_gpu.download().astype(np.float32)
            out = np.zeros_like(drop, dtype=np.uint8)
            if int(np.count_nonzero(valid)) < 50:
                return out

            smoothed = smooth_masked_field_gpu(drop_gpu, mask_gpu, sigma=14.0).download()
            out[valid] = np.clip(smoothed[valid], 0, 255).astype(np.uint8)
            return out
        except Exception as exc:
            print(f"[WARN] CUDA frame-space intensity failed, falling back to CPU: {exc}")
            set_cuda_runtime_enabled(False)
    frame_l = compute_linear_luminance(frame_bgr)
    ref_l = compute_linear_luminance(ref_bgr)
    valid = board_mask > 0

    frame_s = smooth_masked_field(frame_l, board_mask, sigma=sigma)
    ref_s = smooth_masked_field(ref_l, board_mask, sigma=sigma)

    drop = np.zeros_like(frame_s, dtype=np.float32)
    denom = np.maximum(ref_s, min_ref_lightness)
    drop[valid] = np.clip(((ref_s[valid] - frame_s[valid]) / denom[valid]) * 255.0, 0, 255)
    return make_shadow_intensity_image(drop, board_mask)


def write_panel(
    out_path: Path,
    frame_bgr: np.ndarray,
    warped_ref: np.ndarray,
    aligned_ref: np.ndarray,
    shadow_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    overlay: np.ndarray,
):
    h, w = frame_bgr.shape[:2]

    def label(img, text):
        img2 = img.copy()
        cv2.putText(img2, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return img2

    sm = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
    om = cv2.cvtColor(occlusion_mask, cv2.COLOR_GRAY2BGR)

    top = np.hstack([
        label(frame_bgr, "Frame"),
        label(warped_ref.astype(np.uint8), "Warped ref"),
        label(aligned_ref.astype(np.uint8), "Aligned ref"),
    ])
    bottom = np.hstack([
        label(sm, "Shadow mask"),
        label(om, "Occlusion mask"),
        label(overlay, "Overlay"),
    ])

    panel = np.vstack([top, bottom])
    cv2.imwrite(str(out_path), panel)


def process_frame(
    frame_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    output_dir: Path,
    prefix: str,
    manual_color_adjustments: dict,
    auto_color_settings: dict,
    shadow_mask_threshold: float,
    occlusion_diff_threshold: float = OCCLUSION_DIFF_THRESHOLD,
    occlusion_thin_line_kernel_size: int = OCCLUSION_THIN_LINE_KERNEL_SIZE,
    min_occlusion_component_area: int = MIN_OCCLUSION_COMPONENT_AREA,
    fast_mode: bool = False,
    reference_cache: dict | None = None,
    minimal_outputs: bool = False,
    fast_color_calibration: bool = False,
    eyedropper_tag_colors: bool = False,
    surface_texture_ref_bgr: np.ndarray | None = None,
    write_outputs: bool = True,
    verbose: bool = True,
    profile: dict | None = None,
    return_render_data: bool = False,
):
    total_t = time.perf_counter()
    stage_t = total_t
    if write_outputs:
        clear_previous_outputs(output_dir, prefix)
    if reference_cache is None:
        ref_detections, ref_rejected = detect_apriltags(ref_bgr)
        ref_tag_artifact_mask = make_reference_tag_artifact_mask(ref_bgr, ref_detections=ref_detections)
    else:
        ref_detections = reference_cache["ref_detections"]
        ref_rejected = reference_cache["ref_rejected"]
        ref_tag_artifact_mask = reference_cache["ref_tag_artifact_mask"]
    frame_detections, frame_rejected = detect_apriltags(frame_bgr)
    add_profile_time(profile, "apriltag_detection", time.perf_counter() - stage_t)
    stage_t = time.perf_counter()
    H, board_mask, green_mask, debug, error = find_homography_from_board_quad(
        ref_bgr,
        frame_bgr,
        ref_detections=ref_detections,
        frame_detections=frame_detections,
        ref_rejected=ref_rejected,
        frame_rejected=frame_rejected,
    )
    add_profile_time(profile, "homography_and_board_mask", time.perf_counter() - stage_t)
    if H is None:
        if verbose:
            print(f"[WARN] {prefix}: homography failed: {error}")
        if write_outputs and isinstance(debug, dict):
            draw_board_debug(frame_bgr, debug, output_dir / f"{prefix}_board_quad.jpg")
            draw_tag_center_debug(frame_bgr, debug, output_dir / f"{prefix}_tag_centers.jpg")
            if debug.get("board_mask") is not None:
                cv2.imwrite(str(output_dir / f"{prefix}_board_mask.png"), debug["board_mask"])
            if debug.get("green_mask") is not None:
                cv2.imwrite(str(output_dir / f"{prefix}_green_mask.png"), debug["green_mask"])
            if debug.get("frame_blue_mask") is not None:
                cv2.imwrite(str(output_dir / f"{prefix}_frame_blue_mask.png"), debug["frame_blue_mask"])
            if debug.get("ref_blue_mask") is not None:
                cv2.imwrite(str(output_dir / f"{prefix}_ref_blue_mask.png"), debug["ref_blue_mask"])
        add_profile_time(profile, "total", time.perf_counter() - total_t)
        return None

    h, w = frame_bgr.shape[:2]

    stage_t = time.perf_counter()
    warped_ref = warp_perspective_runtime(
        ref_bgr,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
    )
    tag_structure_mask = make_projected_tag_structure_mask(ref_bgr, H, frame_bgr.shape, ref_detections=ref_detections)
    board_mask = board_mask if board_mask is not None else np.full((h, w), 255, dtype=np.uint8)
    add_profile_time(profile, "reference_warp_and_masks", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    calibrated_warped_ref, calibrated_ref, color_debug = calibrate_reference_colors(
        frame_bgr,
        warped_ref,
        ref_bgr,
        board_mask,
        manual_color_adjustments,
        auto_color_settings,
        ref_detections=ref_detections,
        frame_detections=frame_detections,
        reference_cache=reference_cache,
        fast_mode=fast_mode,
        fast_color_calibration=fast_color_calibration,
        eyedropper_tag_colors=eyedropper_tag_colors,
        profile=profile,
    )
    if eyedropper_tag_colors:
        aligned_ref = calibrated_warped_ref.copy()
    else:
        aligned_ref = robust_channel_gain(frame_bgr, calibrated_warped_ref, board_mask)
    add_profile_time(profile, "color_calibration", time.perf_counter() - stage_t)
    rectified_blur_ksize = 31
    use_fast_gpu_projection = fast_mode and use_cuda_runtime()
    frame_gpu = None
    board_mask_gpu = None
    if use_fast_gpu_projection:
        frame_gpu = gpu_mat_from_array(frame_bgr)
        board_mask_gpu = gpu_mat_from_array(board_mask)
    frame_rectified = None
    rectified_valid_mask = np.zeros(ref_bgr.shape[:2], dtype=np.uint8)
    if not use_fast_gpu_projection:
        stage_t = time.perf_counter()
        frame_rectified, rectified_valid_mask = warp_frame_to_reference_space(frame_bgr, board_mask, H, ref_bgr.shape[:2])
        add_profile_time(profile, "rectify_frame", time.perf_counter() - stage_t)
    rectified_structure_edge_mask = np.zeros_like(rectified_valid_mask, dtype=np.uint8)
    rectified_cleanup_debug = {
        "blur_ksize": rectified_blur_ksize,
        "edge_suppression_applied": False,
        "edge_mask_pixel_count": 0,
        "tag_structure_fill_applied": False,
        "rectified_tag_artifact_mask_pixel_count": 0,
        "fast_mode": bool(fast_mode),
    }
    if use_fast_gpu_projection:
        stage_t = time.perf_counter()
        (
            rectified_valid_mask,
            frame_shadow_residual,
            frame_shadow_drop_raw,
        ) = compute_fast_shadow_projection_gpu(
            frame_bgr,
            calibrated_ref,
            board_mask,
            H,
            ref_bgr.shape[:2],
            blur_ksize=rectified_blur_ksize,
            frame_gpu=frame_gpu,
            board_mask_gpu=board_mask_gpu,
        )
        add_profile_time(profile, "shadow_projection_gpu", time.perf_counter() - stage_t)
        rectified_tag_artifact_mask = cv2.bitwise_and(ref_tag_artifact_mask, rectified_valid_mask)
        rectified_cleanup_debug["rectified_tag_artifact_mask_pixel_count"] = int(np.count_nonzero(rectified_tag_artifact_mask))
        rectified_cleanup_debug["gpu_fast_projection"] = True
        rectified_cleanup_debug["fast_mode_skips"] = [
            "suppress_rectified_structure_edges",
            "feather_replace_masked_field",
            "fill_masked_regions_adaptive",
        ]
    else:
        stage_t = time.perf_counter()
        (
            rectified_shadow_residual,
            rectified_shadow_drop_raw,
        ) = compute_rectified_shadow_residual(
            calibrated_ref,
            frame_rectified,
            rectified_valid_mask,
            blur_ksize=rectified_blur_ksize,
        )
        add_profile_time(profile, "rectified_shadow_residual", time.perf_counter() - stage_t)
        rectified_shadow_residual_clean = rectified_shadow_residual.astype(np.float32).copy()
        rectified_shadow_drop_raw_clean = rectified_shadow_drop_raw.astype(np.float32).copy()
        rectified_tag_artifact_mask = cv2.bitwise_and(ref_tag_artifact_mask, rectified_valid_mask)
        rectified_cleanup_debug["rectified_tag_artifact_mask_pixel_count"] = int(np.count_nonzero(rectified_tag_artifact_mask))
    if not fast_mode:
        stage_t = time.perf_counter()
        (
            rectified_shadow_residual_clean,
            rectified_structure_edge_mask,
            edge_debug,
        ) = suppress_rectified_structure_edges(
            rectified_shadow_residual_clean,
            calibrated_ref,
            frame_rectified,
            rectified_valid_mask,
        )
        rectified_cleanup_debug.update({
            "edge_suppression_applied": bool(edge_debug.get("applied")),
            "edge_mask_pixel_count": int(edge_debug.get("edge_mask_pixel_count", 0)),
            "edge_debug": edge_debug,
        })
        if int(np.count_nonzero(rectified_structure_edge_mask)) > 0:
            rectified_shadow_drop_raw_clean = feather_replace_masked_field(
                rectified_shadow_drop_raw_clean,
                rectified_structure_edge_mask,
                rectified_valid_mask,
                sigma_fill=24.0,
                feather_radius=9,
            )

        if int(np.count_nonzero(rectified_tag_artifact_mask)) > 0:
            rectified_shadow_residual_clean, rectified_shadow_drop_raw_clean = fill_masked_regions_adaptive_pair(
                rectified_shadow_residual_clean,
                rectified_shadow_drop_raw_clean,
                rectified_tag_artifact_mask,
                rectified_valid_mask,
                min_sigma=24.0,
                sigma_scale=0.6,
                feather_radius=11,
            )
            rectified_cleanup_debug["tag_structure_fill_applied"] = True
        add_profile_time(profile, "rectified_cleanup", time.perf_counter() - stage_t)
    else:
        rectified_cleanup_debug.update({
            "edge_suppression_applied": False,
            "edge_mask_pixel_count": 0,
            "edge_debug": {
                "applied": False,
                "fast_mode_skipped": True,
            },
            "fast_mode_skips": rectified_cleanup_debug.get("fast_mode_skips", [
                "suppress_rectified_structure_edges",
                "feather_replace_masked_field",
                "fill_masked_regions_adaptive",
            ]),
        })
    if use_fast_gpu_projection:
        frame_shadow_residual = frame_shadow_residual.astype(np.float32)
        frame_shadow_drop_raw = frame_shadow_drop_raw.astype(np.float32)
    else:
        stage_t = time.perf_counter()
        frame_shadow_residual = warp_perspective_runtime(
            rectified_shadow_residual_clean,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
        ).astype(np.float32)
        frame_shadow_drop_raw = warp_perspective_runtime(
            rectified_shadow_drop_raw_clean,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
        ).astype(np.float32)
        add_profile_time(profile, "shadow_warp_to_frame", time.perf_counter() - stage_t)
    if not fast_mode:
        stage_t = time.perf_counter()
        frame_shadow_residual = harmonize_class_boundary_residual(
            frame_shadow_residual,
            color_debug.get("warped_green_mask"),
            color_debug.get("warped_blue_mask"),
            board_mask,
        )
        frame_shadow_drop_raw = harmonize_class_boundary_residual(
            frame_shadow_drop_raw,
            color_debug.get("warped_green_mask"),
            color_debug.get("warped_blue_mask"),
            board_mask,
        )
        frame_shadow_residual, _ = center_class_residual_baselines(
            frame_shadow_residual,
            color_debug.get("warped_green_mask"),
            color_debug.get("warped_blue_mask"),
            board_mask,
        )
        frame_shadow_drop_raw, _ = center_class_residual_baselines(
            frame_shadow_drop_raw,
            color_debug.get("warped_green_mask"),
            color_debug.get("warped_blue_mask"),
            board_mask,
        )
        frame_shadow_residual, _ = harmonize_blue_tiles_to_surround(
            frame_shadow_residual,
            color_debug.get("warped_blue_mask"),
            board_mask,
        )
        frame_shadow_drop_raw, _ = harmonize_blue_tiles_to_surround(
            frame_shadow_drop_raw,
            color_debug.get("warped_blue_mask"),
            board_mask,
        )
        add_profile_time(profile, "frame_shadow_harmonization", time.perf_counter() - stage_t)
    frame_shadow_residual[board_mask == 0] = 0
    frame_shadow_drop_raw[board_mask == 0] = 0
    stage_t = time.perf_counter()
    frame_shadow_smooth = smooth_masked_field(frame_shadow_residual, board_mask, sigma=18.0)
    frame_shadow_for_mask = frame_shadow_smooth.copy()
    frame_shadow_for_intensity = frame_shadow_smooth.copy()
    blue_tag_mask = color_debug.get("warped_blue_mask")
    if blue_tag_mask is not None and not fast_mode:
        tag_mask = cv2.dilate(blue_tag_mask, np.ones((21, 21), np.uint8), iterations=1)
        tag_fill = smooth_masked_field(frame_shadow_residual, board_mask, sigma=34.0)
        frame_shadow_for_mask[tag_mask > 0] = tag_fill[tag_mask > 0]
    if not fast_mode and tag_structure_mask is not None and np.count_nonzero(tag_structure_mask) > 0:
        non_tag_board_mask = board_mask.copy()
        non_tag_board_mask[tag_structure_mask > 0] = 0
        structure_fill = smooth_masked_field(frame_shadow_residual, non_tag_board_mask, sigma=36.0)
        frame_shadow_for_intensity = feather_replace_in_mask(
            frame_shadow_for_intensity,
            structure_fill,
            tag_structure_mask,
            feather_radius=14,
        )
    add_profile_time(profile, "shadow_smoothing_and_fills", time.perf_counter() - stage_t)
    stage_t = time.perf_counter()
    shadow_intensity = make_frame_space_shadow_intensity(
        frame_bgr,
        aligned_ref,
        board_mask,
        sigma=18.0,
        min_ref_lightness=1.0,
        frame_gpu=frame_gpu,
        mask_gpu=board_mask_gpu,
    )
    add_profile_time(profile, "shadow_intensity", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    shadow_mask = build_shadow_mask(frame_shadow_for_mask, board_mask, shadow_mask_threshold)
    add_profile_time(profile, "shadow_mask_cleanup", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    protected_surface_mask = np.zeros_like(board_mask, dtype=np.uint8)
    if color_debug.get("warped_green_mask") is not None:
        protected_surface_mask[color_debug["warped_green_mask"] > 0] = 255
    if color_debug.get("warped_blue_mask") is not None:
        protected_surface_mask[color_debug["warped_blue_mask"] > 0] = 255
    if tag_structure_mask is not None:
        protected_surface_mask[tag_structure_mask > 0] = 255
    occlusion_mask, occlusion_debug = compute_hand_occlusion_mask(
        frame_bgr,
        aligned_ref,
        board_mask,
        binary_threshold=occlusion_diff_threshold,
        thin_line_kernel_size=occlusion_thin_line_kernel_size,
        min_component_area=min_occlusion_component_area,
        protected_surface_mask=protected_surface_mask,
    )
    add_profile_time(profile, "occlusion_mask", time.perf_counter() - stage_t)
    if write_outputs and not minimal_outputs:
        cv2.imwrite(str(output_dir / f"{prefix}_raw_frame.png"), frame_bgr)
        cv2.imwrite(str(output_dir / f"{prefix}_aligned_ref.png"), aligned_ref.astype(np.uint8))
        raw_color_difference = np.zeros_like(frame_bgr, dtype=np.uint8)
        raw_color_difference[board_mask > 0] = np.clip(
            occlusion_debug["abs_bgr_diff"][board_mask > 0],
            0,
            255,
        ).astype(np.uint8)
        signed_color_difference = np.zeros_like(frame_bgr, dtype=np.uint8)
        signed_color_difference[board_mask > 0] = np.clip(
            128.0 + (occlusion_debug["signed_bgr_diff"][board_mask > 0] * 0.75),
            0,
            255,
        ).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{prefix}_raw_color_difference.png"), raw_color_difference)
        cv2.imwrite(str(output_dir / f"{prefix}_signed_color_difference.png"), signed_color_difference)
        raw_difference = np.zeros_like(board_mask, dtype=np.uint8)
        raw_difference_values = np.clip(occlusion_debug["bgr_abs_diff"], 0, 255).astype(np.uint8)
        raw_difference[board_mask > 0] = raw_difference_values[board_mask > 0]
        cv2.imwrite(str(output_dir / f"{prefix}_raw_difference.png"), raw_difference)
    if write_outputs:
        cv2.imwrite(str(output_dir / f"{prefix}_occlusion_mask.png"), occlusion_mask)
    if write_outputs and not minimal_outputs:
        simple_score = np.zeros_like(board_mask, dtype=np.uint8)
        simple_score_values = occlusion_debug["contrast_brightness_difference"]
        simple_score[board_mask > 0] = simple_score_values[board_mask > 0]
        cv2.imwrite(str(output_dir / f"{prefix}_contrast_brightness_difference.png"), simple_score)
        cv2.imwrite(str(output_dir / f"{prefix}_white_mask_before_thin_line_removal.png"), occlusion_debug["white_mask_before_thin_line_removal"])
        cv2.imwrite(str(output_dir / f"{prefix}_thin_line_exclusion_mask.png"), occlusion_debug["thin_line_mask"])
        cv2.imwrite(str(output_dir / f"{prefix}_small_region_exclusion_mask.png"), occlusion_debug["small_region_mask"])
        cv2.imwrite(str(output_dir / f"{prefix}_reference_edge_mask.png"), occlusion_debug["reference_edge_mask"])
        cv2.imwrite(str(output_dir / f"{prefix}_protected_surface_mask.png"), occlusion_debug["protected_surface_mask"])
        cv2.imwrite(str(output_dir / f"{prefix}_skin_seed_mask.png"), occlusion_debug["skin_seed_mask"])
        cv2.imwrite(str(output_dir / f"{prefix}_dark_tool_seed_mask.png"), occlusion_debug["dark_tool_seed_mask"])
        cv2.imwrite(str(output_dir / f"{prefix}_shadow_like_exclusion_mask.png"), occlusion_debug["shadow_like_exclusion_mask"])

    occlusion_hard_mask = occlusion_mask > 0
    frame_shadow_residual[occlusion_hard_mask] = 0
    frame_shadow_smooth[occlusion_hard_mask] = 0
    frame_shadow_for_mask[occlusion_hard_mask] = 0
    frame_shadow_for_intensity[occlusion_hard_mask] = 0
    shadow_mask[occlusion_hard_mask] = 0
    shadow_intensity[occlusion_hard_mask] = 0
    board_color_mask = None
    warped_green_mask = color_debug.get("warped_green_mask")
    warped_blue_mask = color_debug.get("warped_blue_mask")
    if warped_green_mask is not None or warped_blue_mask is not None:
        board_color_mask = np.zeros_like(board_mask, dtype=np.uint8)
        if warped_green_mask is not None:
            board_color_mask[warped_green_mask > 0] = 255
        if warped_blue_mask is not None:
            board_color_mask[warped_blue_mask > 0] = 255
    if return_render_data:
        add_profile_time(profile, "total", time.perf_counter() - total_t)
        return {
            "frame_bgr": frame_bgr,
            "H": H,
            "board_mask": board_mask,
            "shadow_mask": shadow_mask,
            "occlusion_mask": occlusion_mask,
            "shadow_intensity": shadow_intensity,
            "board_color_mask": board_color_mask,
        }
    warped_surface_texture = None
    if surface_texture_ref_bgr is not None:
        stage_t = time.perf_counter()
        warped_surface_texture = warp_perspective_runtime(
            surface_texture_ref_bgr,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
        )
        add_profile_time(profile, "surface_texture_warp", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    overlay = make_overlay(
        frame_bgr,
        board_mask,
        shadow_mask,
        occlusion_mask,
        shadow_intensity,
        board_color_mask=board_color_mask,
        board_fill_bgr=warped_surface_texture,
    )
    add_profile_time(profile, "overlay", time.perf_counter() - stage_t)

    if write_outputs:
        stage_t = time.perf_counter()
        cv2.imwrite(str(output_dir / f"{prefix}_overlay.png"), overlay)
        add_profile_time(profile, "debug_output_writes", time.perf_counter() - stage_t)
    tag_color_debug = color_debug.get("tag_color_debug")

    if verbose and isinstance(debug, dict):
        print(
            f"[INFO] {prefix}: contour_area={debug.get('contour_area', 0):.0f}, "
            f"reprojection_error={debug.get('reprojection_error', float('nan')):.2f}, "
            f"tile_matches={debug.get('tile_matches', 0)}, "
            f"tile_inliers={debug.get('tile_inliers', 0)}, "
            f"color_source={color_debug.get('calibration_source')}, "
            f"tag_id={tag_color_debug.get('selected_tag_id') if isinstance(tag_color_debug, dict) else 'none'}, "
            f"thin_line_px={occlusion_debug.get('thin_line_debug', {}).get('removed_pixel_count', 0)}, "
            f"tag_green_source={color_debug.get('tag_green_source')}, "
            f"tag_blue_source={color_debug.get('tag_blue_source')}, "
            f"rect_edge_px={rectified_cleanup_debug.get('edge_mask_pixel_count', 0)}, "
            f"rect_tag_px={rectified_cleanup_debug.get('rectified_tag_artifact_mask_pixel_count', 0)}, "
            f"green_h={color_debug['green_adjustment']['h_shift']:.1f} "
            f"(raw {color_debug['auto_green']['raw_h_shift']:.1f}), "
            f"green_s={color_debug['green_adjustment']['s_scale']:.2f}, "
            f"green_v={color_debug['green_adjustment']['v_scale']:.2f}, "
            f"blue_h={color_debug['blue_adjustment']['h_shift']:.1f} "
            f"(raw {color_debug['auto_blue']['raw_h_shift']:.1f}), "
            f"blue_s={color_debug['blue_adjustment']['s_scale']:.2f}, "
            f"blue_v={color_debug['blue_adjustment']['v_scale']:.2f}, "
            f"mode={'fast' if fast_mode else 'full'}, "
            f"shadow_mask_threshold={shadow_mask_threshold:.1f}, "
            f"{'written outputs to ' + str(output_dir) if write_outputs else 'video frame processed'}"
        )
    add_profile_time(profile, "total", time.perf_counter() - total_t)
    return overlay


def render_texture_overlay(
    render_data: dict,
    surface_texture_ref_bgr: np.ndarray | None,
    profile: dict | None = None,
) -> np.ndarray:
    frame_bgr = render_data["frame_bgr"]
    h, w = frame_bgr.shape[:2]
    warped_surface_texture = None
    if surface_texture_ref_bgr is not None:
        stage_t = time.perf_counter()
        warped_surface_texture = warp_perspective_runtime(
            surface_texture_ref_bgr,
            render_data["H"],
            (w, h),
            flags=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
        )
        add_profile_time(profile, "multi_texture_warp", time.perf_counter() - stage_t)

    stage_t = time.perf_counter()
    overlay = make_overlay(
        frame_bgr,
        render_data["board_mask"],
        render_data["shadow_mask"],
        render_data["occlusion_mask"],
        render_data["shadow_intensity"],
        board_color_mask=render_data["board_color_mask"],
        board_fill_bgr=warped_surface_texture,
    )
    add_profile_time(profile, "multi_texture_overlay", time.perf_counter() - stage_t)
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Test shadow extraction from a known planar surface in a video."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--reference", required=True, help="Path to reference surface image.")
    parser.add_argument("--outdir", required=True, help="Directory for debug outputs.")
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        default=None,
        help="Specific frame indices to test. If omitted, evenly sampled frames are used.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of evenly sampled frames if --frames is not provided.",
    )
    parser.add_argument("--green-h-shift", type=float, default=0.0, help="Manual hue shift added to the auto green correction.")
    parser.add_argument("--green-s-scale", type=float, default=1.0, help="Manual saturation scale multiplied into the auto green correction.")
    parser.add_argument("--green-v-scale", type=float, default=1.0, help="Manual value scale multiplied into the auto green correction.")
    parser.add_argument("--blue-h-shift", type=float, default=0.0, help="Manual hue shift added to the auto blue correction.")
    parser.add_argument("--blue-s-scale", type=float, default=1.0, help="Manual saturation scale multiplied into the auto blue correction.")
    parser.add_argument("--blue-v-scale", type=float, default=1.0, help="Manual value scale multiplied into the auto blue correction.")
    parser.add_argument("--auto-hue-weight", type=float, default=0.55, help="How strongly to apply the sampled hue correction before the manual slider offset.")
    parser.add_argument("--auto-hue-max", type=float, default=12.0, help="Maximum absolute hue shift applied automatically to each board color.")
    parser.add_argument(
        "--shadow-mask-threshold",
        type=float,
        default=24.0,
        help="Threshold applied to the smoothed shadow field before mask cleanup.",
    )
    parser.add_argument(
        "--occlusion-diff-threshold",
        type=float,
        default=OCCLUSION_DIFF_THRESHOLD,
        help="Mean BGR absolute-difference threshold for hand/occlusion masking.",
    )
    parser.add_argument(
        "--occlusion-thin-line-kernel-size",
        type=int,
        default=OCCLUSION_THIN_LINE_KERNEL_SIZE,
        help="Morphological open kernel size used to remove thin line artifacts from the occlusion mask.",
    )
    parser.add_argument(
        "--min-occlusion-component-area",
        type=int,
        default=MIN_OCCLUSION_COMPONENT_AREA,
        help="Minimum connected-component area retained in the occlusion mask.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the slow cleanup and harmonization stages to preview a faster output path.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA-backed blur and warp operations when the current OpenCV build supports them.",
    )
    parser.add_argument(
        "--minimal-outputs",
        action="store_true",
        help="Write only overlay and final occlusion mask outputs.",
    )
    parser.add_argument(
        "--fast-color-calibration",
        action="store_true",
        help="In --fast mode, skip semantic tag HSV calibration and use board-level HSV calibration only.",
    )
    parser.add_argument(
        "--eyedropper-tag-colors",
        action="store_true",
        help="Sample green/blue from the top-left visible tag in the raw frame and fill all reference class colors with those samples.",
    )
    parser.add_argument(
        "--surface-texture",
        type=str,
        default=None,
        help="Optional texture image used to fill the reference surface before warping into the frame.",
    )
    parser.add_argument(
        "--surface-textures-dir",
        type=str,
        default=None,
        help="Optional directory of texture images. With --output-video, writes one AVI per texture in one shared pass.",
    )
    parser.add_argument(
        "--reference-scale",
        type=float,
        default=1.0,
        help="Optional scale applied to the rotated reference image before processing. Default preserves original behavior.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional AVI output path. When set, frames are processed into a rendered video instead of only writing per-frame PNGs.",
    )
    parser.add_argument(
        "--verbose-video",
        action="store_true",
        help="Print per-frame diagnostics while rendering --output-video.",
    )
    parser.add_argument(
        "--profile-video",
        action="store_true",
        help="Print accumulated per-stage timings while rendering --output-video.",
    )
    parser.add_argument(
        "--video-progress-interval",
        type=int,
        default=100,
        help="Frame interval for video progress logs.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    ref_path = Path(args.reference)
    outdir = Path(args.outdir)

    ensure_dir(outdir)

    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_path}")
    # The reference asset is stored rotated 90 degrees clockwise relative to the board.
    # Rotate it back so the homography is computed in the board's actual orientation.
    ref_bgr = cv2.rotate(ref_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if args.reference_scale <= 0:
        raise ValueError("--reference-scale must be > 0")
    if abs(float(args.reference_scale) - 1.0) > 1e-6:
        ref_h, ref_w = ref_bgr.shape[:2]
        scaled_w = max(1, int(round(ref_w * float(args.reference_scale))))
        scaled_h = max(1, int(round(ref_h * float(args.reference_scale))))
        interpolation = cv2.INTER_AREA if args.reference_scale < 1.0 else cv2.INTER_LINEAR
        ref_bgr = cv2.resize(ref_bgr, (scaled_w, scaled_h), interpolation=interpolation)
        print(
            f"[INFO] Reference scale: {args.reference_scale:.4f} "
            f"({ref_w}x{ref_h} -> {scaled_w}x{scaled_h})"
        )
    surface_texture_ref_bgr = None
    surface_texture_entries = []
    if args.surface_texture and args.surface_textures_dir:
        raise ValueError("Use either --surface-texture or --surface-textures-dir, not both.")
    if args.surface_textures_dir:
        if not args.output_video:
            raise ValueError("--surface-textures-dir requires --output-video so output AVI names can be derived.")
        texture_dir = Path(args.surface_textures_dir)
        for texture_path in list_texture_paths(texture_dir):
            surface_texture_entries.append(
                {
                    "path": texture_path,
                    "texture_ref_bgr": load_fitted_texture(texture_path, ref_bgr.shape[:2]),
                }
            )
    elif args.surface_texture:
        surface_texture_path = Path(args.surface_texture)
        surface_texture_ref_bgr = load_fitted_texture(surface_texture_path, ref_bgr.shape[:2])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = args.frames if args.frames else sample_frame_indices(total_frames, args.num_samples)
    cuda_enabled = set_cuda_runtime_enabled(args.cuda)

    print(f"[INFO] Total frames: {total_frames}")
    if args.output_video:
        print("[INFO] Rendering all frames to output video")
    else:
        print(f"[INFO] Testing frames: {frame_indices}")
    if args.cuda:
        print(f"[INFO] CUDA runtime {'enabled' if cuda_enabled else 'unavailable; using CPU fallback'}")
    if args.surface_texture:
        print(f"[INFO] Surface texture: {args.surface_texture}")
    if args.surface_textures_dir:
        print(f"[INFO] Surface textures: {len(surface_texture_entries)} from {args.surface_textures_dir}")
    if args.output_video:
        print(f"[INFO] Output video: {args.output_video}")

    manual_color_adjustments = {
        "green": {
            "h_shift": args.green_h_shift,
            "s_scale": args.green_s_scale,
            "v_scale": args.green_v_scale,
        },
        "blue": {
            "h_shift": args.blue_h_shift,
            "s_scale": args.blue_s_scale,
            "v_scale": args.blue_v_scale,
        },
    }
    auto_color_settings = {
        "hue_weight": args.auto_hue_weight,
        "hue_max_shift": args.auto_hue_max,
    }
    reference_cache = build_reference_cache(ref_bgr)

    if args.output_video:
        out_video_path = Path(args.output_video)
        ensure_dir(out_video_path.parent)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writers = []
        if surface_texture_entries:
            suffix = out_video_path.suffix or ".avi"
            for entry in surface_texture_entries:
                texture_path = entry["path"]
                texture_out_path = out_video_path.with_name(
                    f"{out_video_path.stem}_{texture_path.stem}{suffix}"
                )
                try:
                    writer, writer_backend = open_video_writer(
                        texture_out_path,
                        float(fps),
                        (frame_w, frame_h),
                        codec="MJPG",
                    )
                except RuntimeError:
                    cap.release()
                    for opened_writer in writers:
                        opened_writer["writer"].release()
                    raise
                writers.append(
                    {
                        "writer": writer,
                        "backend": writer_backend,
                        "output_path": texture_out_path,
                        "texture_path": texture_path,
                        "texture_ref_bgr": entry["texture_ref_bgr"],
                    }
                )
            backend_names = sorted({item["backend"] for item in writers})
            print(f"[INFO] Video writer backends: {', '.join(backend_names)}")
            for item in writers:
                print(f"[INFO] Texture output: {item['texture_path'].name} -> {item['output_path']}")
        else:
            try:
                writer, writer_backend = open_video_writer(
                    out_video_path,
                    float(fps),
                    (frame_w, frame_h),
                    codec="MJPG",
                )
            except RuntimeError:
                cap.release()
                raise
            writers.append(
                {
                    "writer": writer,
                    "backend": writer_backend,
                    "output_path": out_video_path,
                    "texture_path": None,
                    "texture_ref_bgr": surface_texture_ref_bgr,
                }
            )
            print(f"[INFO] Video writer backend: {writer_backend}")

        processed_count = 0
        fallback_count = 0
        video_profile = {} if args.profile_video else None
        progress_interval = max(1, int(args.video_progress_interval))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for idx in range(total_frames):
            read_t = time.perf_counter()
            ok, frame = cap.read()
            add_profile_time(video_profile, "video_read", time.perf_counter() - read_t)
            if not ok or frame is None:
                print(f"[WARN] Could not read frame {idx}; stopping video render.")
                break
            prefix = f"frame_{idx:06d}"
            render_data_or_overlay = process_frame(
                frame,
                ref_bgr,
                outdir,
                prefix,
                manual_color_adjustments,
                auto_color_settings,
                args.shadow_mask_threshold,
                occlusion_diff_threshold=args.occlusion_diff_threshold,
                occlusion_thin_line_kernel_size=args.occlusion_thin_line_kernel_size,
                min_occlusion_component_area=args.min_occlusion_component_area,
                fast_mode=args.fast,
                reference_cache=reference_cache,
                minimal_outputs=args.minimal_outputs,
                fast_color_calibration=args.fast_color_calibration,
                eyedropper_tag_colors=args.eyedropper_tag_colors,
                surface_texture_ref_bgr=None if surface_texture_entries else surface_texture_ref_bgr,
                write_outputs=False,
                verbose=args.verbose_video,
                profile=video_profile,
                return_render_data=bool(surface_texture_entries),
            )
            if render_data_or_overlay is None:
                fallback_count += 1
                write_t = time.perf_counter()
                for item in writers:
                    item["writer"].write(frame)
                add_profile_time(video_profile, "video_write", time.perf_counter() - write_t)
            elif surface_texture_entries:
                for item in writers:
                    overlay = render_texture_overlay(
                        render_data_or_overlay,
                        item["texture_ref_bgr"],
                        profile=video_profile,
                    )
                    write_t = time.perf_counter()
                    item["writer"].write(overlay)
                    add_profile_time(video_profile, "video_write", time.perf_counter() - write_t)
            else:
                write_t = time.perf_counter()
                writers[0]["writer"].write(render_data_or_overlay)
                add_profile_time(video_profile, "video_write", time.perf_counter() - write_t)
            processed_count += 1
            if processed_count % progress_interval == 0:
                print(f"[INFO] Video progress: {processed_count}/{total_frames} frames")
                if video_profile is not None:
                    print_profile_summary(video_profile, processed_count)

        for item in writers:
            item["writer"].release()
        cap.release()
        if video_profile is not None:
            print_profile_summary(video_profile, processed_count)
        if surface_texture_entries:
            print(
                f"[DONE] Finished texture video render: {len(writers)} outputs "
                f"({processed_count} frames, {fallback_count} fallback frames)."
            )
        else:
            print(
                f"[DONE] Finished video render: {out_video_path} "
                f"({processed_count} frames, {fallback_count} fallback frames)."
            )
        return

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] Could not read frame {idx}")
            continue
        prefix = f"frame_{idx:06d}"
        process_frame(
            frame,
            ref_bgr,
            outdir,
            prefix,
            manual_color_adjustments,
            auto_color_settings,
            args.shadow_mask_threshold,
            occlusion_diff_threshold=args.occlusion_diff_threshold,
            occlusion_thin_line_kernel_size=args.occlusion_thin_line_kernel_size,
            min_occlusion_component_area=args.min_occlusion_component_area,
            fast_mode=args.fast,
            reference_cache=reference_cache,
            minimal_outputs=args.minimal_outputs,
            fast_color_calibration=args.fast_color_calibration,
            eyedropper_tag_colors=args.eyedropper_tag_colors,
            surface_texture_ref_bgr=surface_texture_ref_bgr,
            write_outputs=True,
            verbose=True,
            profile=None,
        )

    cap.release()
    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
