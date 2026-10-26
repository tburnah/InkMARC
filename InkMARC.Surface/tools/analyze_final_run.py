#!/usr/bin/env python3
"""Summarise a recorded InkMARC session without re-running the tracker.

The script treats the synchronized CSV as the authoritative tracker telemetry.
It reports observable continuity, latency, support, masking, and sensor quality
metrics.  It intentionally does not label those proxies as absolute pose error,
because a monocular session has no independent ground-truth trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAMERA_EVENT = "camera_frame"


def read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            key, separator, value = line[1:].strip().partition("=")
            if separator:
                metadata[key] = value
    return metadata


def numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def quantiles(values: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {"n": 0, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(series.size),
        "median": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def finite_quad(row: pd.Series) -> np.ndarray | None:
    values = []
    for point in range(4):
        x = pd.to_numeric(row.get(f"quad_x{point}"), errors="coerce")
        y = pd.to_numeric(row.get(f"quad_y{point}"), errors="coerce")
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        values.append((float(x), float(y)))
    return np.asarray(values, dtype=np.float64)


def run_durations(camera: pd.DataFrame, mask: pd.Series) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    start = None
    previous = None
    frames = 0
    for time_s, active in zip(camera["time_s"], mask.fillna(False).astype(bool)):
        if active:
            if start is None:
                start = float(time_s)
                frames = 0
            previous = float(time_s)
            frames += 1
        elif start is not None and previous is not None:
            result.append({"start_s": start, "end_s": previous, "duration_s": previous - start, "frames": frames})
            start = None
            previous = None
            frames = 0
    if start is not None and previous is not None:
        result.append({"start_s": start, "end_s": previous, "duration_s": previous - start, "frames": frames})
    return sorted(result, key=lambda item: item["duration_s"], reverse=True)


def classify_reason(reason: str) -> str:
    lower = reason.lower()
    checks = [
        ("waiting_for_seed", "awaiting placement"),
        ("auto_reseed", "automatic reseed"),
        ("innovation_rejected", "innovation gate"),
        ("insufficient_effective", "effective support gate"),
        ("weighted_plane_coverage", "weighted coverage gate"),
        ("insufficient_plane", "plane support gate"),
        ("predicted_only", "IMU/prediction hold"),
        ("relock=blend", "controlled relock"),
        ("no_transform", "visual transform unavailable"),
        ("motion_mask", "dynamic-mask rejection"),
        ("hand_mask", "hand-mask rejection"),
    ]
    for token, label in checks:
        if token in lower:
            return label
    return "accepted/other"


def reason_scalar(reason: str, label: str) -> float:
    match = re.search(rf"(?:^|:){re.escape(label)}=(-?\d+(?:\.\d+)?)", reason)
    return float(match.group(1)) if match else np.nan


def analyse(csv_path: Path) -> tuple[dict, pd.DataFrame]:
    metadata = read_metadata(csv_path)
    all_rows = pd.read_csv(csv_path, comment="#", low_memory=False)
    camera = all_rows[all_rows["event"].eq(CAMERA_EVENT)].copy()
    camera["video_time_ns"] = numeric(camera, "video_time_ns")
    camera = camera.sort_values("video_time_ns").reset_index(drop=True)
    camera["time_s"] = camera["video_time_ns"] / 1e9

    for column in [
        "camera_delivery_latency_ms", "processing_latency_ms", "total_latency_ms",
        "tracker_active_tracks", "tracker_candidate_tracks", "tracker_stable_tracks",
        "tracker_inliers", "tracker_inlier_ratio", "tracker_reprojection_error_px",
        "tracker_reference_coverage", "tracker_current_coverage", "tracker_transform_condition",
        "tracker_confidence", "tracker_hand_mask_age_ms", "tracker_hand_mask_area_ratio",
        "tracker_motion_mask_area_ratio", "tracker_combined_mask_area_ratio",
        "tracker_mask_rejected_tracks", "tracker_reacquired_tracks",
        "tracker_effective_plane_support", "tracker_weighted_plane_coverage",
    ]:
        camera[column] = numeric(camera, column)

    camera["held"] = camera.get("tracker_held", False).astype(str).str.lower().isin(["true", "1"])
    camera["reason_class"] = camera.get("tracker_rejection_reason", "").fillna("").astype(str).map(classify_reason)
    camera["state"] = camera.get("tracker_state", "").fillna("missing").astype(str)
    reasons = camera.get("tracker_rejection_reason", "").fillna("").astype(str)
    for label in [
        "fit_rmse_px", "overall_support", "filter_alpha", "camera_imu_offset_ms",
        "exposure_ms", "rolling_shutter_ms", "prediction_ms",
    ]:
        camera[f"reason_{label}"] = reasons.map(lambda value, key=label: reason_scalar(value, key))

    quads = [finite_quad(row) for _, row in camera.iterrows()]
    camera["quad_valid"] = [quad is not None for quad in quads]
    camera["quad_center_x"] = [float(quad[:, 0].mean()) if quad is not None else np.nan for quad in quads]
    camera["quad_center_y"] = [float(quad[:, 1].mean()) if quad is not None else np.nan for quad in quads]
    camera["quad_area_px2"] = [
        float(abs(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))) / 2.0)
        if quad is not None else np.nan
        for quad in quads
    ]

    corner_step = np.full(len(camera), np.nan)
    center_step = np.full(len(camera), np.nan)
    area_ratio = np.full(len(camera), np.nan)
    frame_interval_ms = camera["time_s"].diff().to_numpy() * 1000.0
    for index in range(1, len(camera)):
        previous, current = quads[index - 1], quads[index]
        if previous is None or current is None or frame_interval_ms[index] > 80.0:
            continue
        corner_step[index] = float(np.linalg.norm(current - previous, axis=1).max())
        center_step[index] = float(np.linalg.norm(current.mean(axis=0) - previous.mean(axis=0)))
        previous_area = float(camera.loc[index - 1, "quad_area_px2"])
        current_area = float(camera.loc[index, "quad_area_px2"])
        if previous_area > 0 and current_area > 0:
            area_ratio[index] = max(previous_area / current_area, current_area / previous_area)
    camera["corner_step_px"] = corner_step
    camera["center_step_px"] = center_step
    camera["area_ratio_step"] = area_ratio

    source_ns = numeric(camera, "source_timestamp_ns")
    start_ns = float(metadata.get("video_start_elapsed_realtime_ns", "nan"))
    # The CSV's video_time_ns is based on the exposure-centre timestamp, while source_timestamp_ns
    # identifies the first-row exposure start on this device.  Their difference is therefore a
    # deliberate timing correction, not an observed synchronization error.
    camera["source_to_effective_offset_ms"] = (camera["video_time_ns"] - (source_ns - start_ns)) / 1e6

    hand_present = camera["tracker_hand_mask_area_ratio"].fillna(0) >= 0.005
    dynamic_present = camera["tracker_motion_mask_area_ratio"].fillna(0) >= 0.02
    occlusion_present = camera["tracker_combined_mask_area_ratio"].fillna(0) >= 0.02
    normal_motion = camera["quad_valid"] & ~camera["held"] & ~camera["state"].eq("seeded")
    catastrophic = normal_motion & (camera["corner_step_px"] > 100.0)

    sensor_rows = all_rows[all_rows["event"].eq("sensor")].copy()
    sensor_summary: dict[str, dict] = {}
    for sensor_type, rows in sensor_rows.groupby("sensor_type"):
        times = pd.to_numeric(rows["source_timestamp_ns"], errors="coerce").dropna().sort_values()
        intervals = times.diff().dropna() / 1e9
        duration = (times.iloc[-1] - times.iloc[0]) / 1e9 if len(times) > 1 else np.nan
        rate = (len(times) - 1) / duration if duration and duration > 0 else np.nan
        values = rows[["v0", "v1", "v2"]].apply(pd.to_numeric, errors="coerce")
        magnitude = np.sqrt((values * values).sum(axis=1))
        sensor_summary[str(sensor_type)] = {
            "samples": int(len(rows)),
            "estimated_rate_hz": None if not np.isfinite(rate) else float(rate),
            "interval_ms": quantiles(intervals * 1000.0),
            "magnitude": quantiles(magnitude),
        }

    valid_count = int(camera["quad_valid"].sum())
    tracked_or_seeded = camera["state"].isin(["tracking", "seeded"])
    first_quad_index = camera.index[camera["quad_valid"]].min() if valid_count else None
    operating = camera.loc[first_quad_index:].copy() if first_quad_index is not None else camera.iloc[0:0].copy()
    operating_tracking = operating["state"].isin(["tracking", "seeded"])
    accepted_visual = operating["tracker_effective_plane_support"].notna()
    report = {
        "session": metadata.get("session_id", csv_path.stem),
        "metadata": metadata,
        "rows": {"all": int(len(all_rows)), "camera": int(len(camera)), "sensor": int(len(sensor_rows))},
        "duration_s": float(camera["time_s"].max() - camera["time_s"].min()),
        "frame_interval_ms": quantiles(camera["time_s"].diff() * 1000.0),
        "state_counts": {str(k): int(v) for k, v in camera["state"].value_counts().items()},
        "reason_counts": {str(k): int(v) for k, v in camera["reason_class"].value_counts().items()},
        "availability": {
            "valid_quad_frames": valid_count,
            "valid_quad_ratio": valid_count / len(camera) if len(camera) else None,
            "tracking_or_seeded_frames": int(tracked_or_seeded.sum()),
            "tracking_or_seeded_ratio": float(tracked_or_seeded.mean()) if len(camera) else None,
            "held_frames": int(camera["held"].sum()),
            "held_ratio": float(camera["held"].mean()) if len(camera) else None,
            "longest_non_tracking_runs": run_durations(camera, ~tracked_or_seeded)[:8],
            "longest_hold_runs": run_durations(camera, camera["held"])[:8],
        },
        "post_placement": {
            "start_s": float(operating["time_s"].min()) if len(operating) else None,
            "duration_s": float(operating["time_s"].max() - operating["time_s"].min()) if len(operating) else None,
            "frames": int(len(operating)),
            "valid_quad_frames": int(operating["quad_valid"].sum()),
            "valid_quad_ratio": float(operating["quad_valid"].mean()) if len(operating) else None,
            "tracking_or_seeded_ratio": float(operating_tracking.mean()) if len(operating) else None,
            "accepted_visual_corrections": int(accepted_visual.sum()),
            "accepted_visual_correction_ratio": float(accepted_visual.mean()) if len(operating) else None,
            "held_frames": int(operating["held"].sum()),
            "held_ratio": float(operating["held"].mean()) if len(operating) else None,
            "seeded_or_reseeded_frames": int(operating["state"].eq("seeded").sum()),
            "longest_hold_s": max((run["duration_s"] for run in run_durations(operating, operating["held"])), default=0.0),
        },
        "latency_ms": {
            "delivery": quantiles(camera["camera_delivery_latency_ms"]),
            "processing": quantiles(camera["processing_latency_ms"]),
            "total": quantiles(camera["total_latency_ms"]),
            "source_to_effective_timestamp_offset": quantiles(camera["source_to_effective_offset_ms"]),
            "exposure": quantiles(numeric(camera, "exposure_time_ns") / 1e6),
            "rolling_shutter_skew": quantiles(numeric(camera, "rolling_shutter_skew_ns") / 1e6),
            "output_prediction_horizon": quantiles(camera["reason_prediction_ms"]),
            "online_camera_imu_offset": quantiles(camera["reason_camera_imu_offset_ms"]),
        },
        "visual_support": {
            "active_tracks": quantiles(camera["tracker_active_tracks"]),
            "candidate_tracks": quantiles(camera["tracker_candidate_tracks"]),
            "stable_tracks": quantiles(camera["tracker_stable_tracks"]),
            "inliers": quantiles(camera["tracker_inliers"]),
            "inlier_ratio": quantiles(camera["tracker_inlier_ratio"]),
            "reprojection_error_px": quantiles(camera["tracker_reprojection_error_px"]),
            "constrained_fit_rmse_px": quantiles(camera["reason_fit_rmse_px"]),
            "overall_support_ratio": quantiles(camera["reason_overall_support"]),
            "measurement_filter_alpha": quantiles(camera["reason_filter_alpha"]),
            "effective_plane_support": quantiles(camera["tracker_effective_plane_support"]),
            "weighted_plane_coverage": quantiles(camera["tracker_weighted_plane_coverage"]),
            "confidence": quantiles(camera["tracker_confidence"]),
            "reacquired_tracks_total": int(camera["tracker_reacquired_tracks"].fillna(0).sum()),
        },
        "occlusion": {
            "semantic_hand_frames": int(hand_present.sum()),
            "semantic_hand_ratio": float(hand_present.mean()) if len(camera) else None,
            "dynamic_mask_frames": int(dynamic_present.sum()),
            "dynamic_mask_ratio": float(dynamic_present.mean()) if len(camera) else None,
            "combined_occlusion_frames": int(occlusion_present.sum()),
            "combined_occlusion_ratio": float(occlusion_present.mean()) if len(camera) else None,
            "hand_mask_area_ratio": quantiles(camera.loc[hand_present, "tracker_hand_mask_area_ratio"]),
            "motion_mask_area_ratio": quantiles(camera.loc[dynamic_present, "tracker_motion_mask_area_ratio"]),
            "mask_rejected_tracks": quantiles(camera["tracker_mask_rejected_tracks"]),
            "hand_intervals": run_durations(camera, hand_present)[:8],
        },
        "motion_continuity_proxies": {
            "corner_step_px": quantiles(camera.loc[normal_motion, "corner_step_px"]),
            "center_step_px": quantiles(camera.loc[normal_motion, "center_step_px"]),
            "area_ratio_step": quantiles(camera.loc[normal_motion, "area_ratio_step"]),
            "corner_steps_over_100_px": int(catastrophic.sum()),
            "corner_steps_over_100_px_times_s": [float(value) for value in camera.loc[catastrophic, "time_s"].tolist()],
        },
        "conditioned_on_occlusion": {},
        "sensors": sensor_summary,
        "interpretation_limits": [
            "No independent camera pose or planar fiducial ground truth was recorded.",
            "Overlay step size includes intentional camera motion and is a continuity proxy, not geometric error.",
            "The semantic mask indicates detector output, not exhaustive hand visibility.",
        ],
    }

    for label, mask in [("occlusion", occlusion_present), ("clear", ~occlusion_present)]:
        subset = camera[mask]
        report["conditioned_on_occlusion"][label] = {
            "frames": int(len(subset)),
            "held_ratio": float(subset["held"].mean()) if len(subset) else None,
            "valid_quad_ratio": float(subset["quad_valid"].mean()) if len(subset) else None,
            "confidence": quantiles(subset["tracker_confidence"]),
            "inliers": quantiles(subset["tracker_inliers"]),
            "reprojection_error_px": quantiles(subset["tracker_reprojection_error_px"]),
            "effective_plane_support": quantiles(subset["tracker_effective_plane_support"]),
            "corner_step_px": quantiles(subset["corner_step_px"]),
        }
    return report, camera


def plot_timeline(camera: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    time = camera["time_s"]

    axes[0].plot(time, camera["tracker_confidence"], color="#1167b1", lw=1.2, label="confidence")
    axes[0].fill_between(time, 0, 1, where=camera["held"], color="#ef8354", alpha=0.25, label="held")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Confidence")
    axes[0].legend(loc="upper right", ncols=2, frameon=False)

    axes[1].plot(time, camera["tracker_inliers"], color="#2a9d8f", lw=1.0, label="inliers")
    axes[1].plot(time, camera["tracker_effective_plane_support"], color="#264653", lw=1.0, label="effective support")
    axes[1].plot(time, camera["tracker_weighted_plane_coverage"] * 100, color="#e9c46a", lw=1.0, label="weighted coverage x100")
    axes[1].set_ylabel("Count / scaled")
    axes[1].legend(loc="upper right", ncols=3, frameon=False)

    axes[2].plot(time, camera["tracker_hand_mask_area_ratio"] * 100, color="#8e44ad", lw=1.0, label="hand")
    axes[2].plot(time, camera["tracker_motion_mask_area_ratio"] * 100, color="#d35400", lw=1.0, label="dynamic")
    axes[2].plot(time, camera["tracker_combined_mask_area_ratio"] * 100, color="#c0392b", lw=1.0, label="combined")
    axes[2].set_ylabel("Masked area (%)")
    axes[2].legend(loc="upper right", ncols=3, frameon=False)

    axes[3].plot(time, camera["processing_latency_ms"], color="#5c677d", lw=1.0, label="processing")
    axes[3].plot(time, camera["total_latency_ms"], color="#003049", lw=1.0, label="total")
    axes[3].set_ylabel("Latency (ms)")
    axes[3].set_xlabel("Recorded video time (s)")
    axes[3].legend(loc="upper right", ncols=2, frameon=False)

    for axis in axes:
        axis.grid(axis="y", color="#d7dce2", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("InkMARC final run: tracker evidence timeline", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def post_placement_camera_rows(camera: pd.DataFrame) -> pd.DataFrame:
    """Return the camera records from the first rendered surface overlay onward.

    A valid recorded quad is the observable indication that placement completed.  Keeping this
    selection separate makes the thesis figure explicit about its population without changing
    the diagnostic timeline, which intentionally includes setup and placement.
    """
    first_quad_index = camera.index[camera["quad_valid"]].min()
    return camera.loc[first_quad_index:].copy() if pd.notna(first_quad_index) else camera.iloc[0:0].copy()


def tracker_output_category(camera: pd.DataFrame) -> pd.Series:
    """Classify recorded post-placement output using tracker_state and tracker_held only."""
    category = pd.Series("unclassified", index=camera.index, dtype="object")
    category.loc[camera["held"]] = "Held"
    category.loc[camera["state"].eq("tracking") & ~camera["held"]] = "Fresh"
    category.loc[camera["state"].eq("seeded") & ~camera["held"]] = "Seeded"
    return category


def add_hand_visible_band(axes: list[plt.Axes], start_s: float = 20.0, end_s: float = 27.0) -> None:
    """Add the manually reviewed hand-visible interval to every thesis-figure panel."""
    for axis in axes:
        axis.axvspan(start_s, end_s, color="#d9d9d9", alpha=0.55, zorder=0, lw=0)
    axes[0].text(
        (start_s + end_s) / 2,
        1.03,
        "Hand visible",
        ha="center",
        va="bottom",
        transform=axes[0].get_xaxis_transform(),
        color="#4d4d4d",
        fontsize=10,
    )


def plot_surface_occlusion_evidence(camera: pd.DataFrame, output: Path) -> dict[str, int]:
    """Plot the thesis-oriented evidence timeline from recorded post-placement telemetry.

    This intentionally omits confidence, raw inliers, combined masking, and latency.  Those
    diagnostics remain available in :func:`plot_timeline` and the JSON report.
    """
    operating = post_placement_camera_rows(camera)
    if operating.empty:
        raise ValueError("No post-placement camera rows (valid overlay quadrilaterals) were recorded.")

    category = tracker_output_category(operating)
    unexpected = category.eq("unclassified")
    if unexpected.any():
        states = operating.loc[unexpected, "state"].value_counts().to_dict()
        raise ValueError(f"Post-placement records cannot be classified from state/held fields: {states}")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.5), sharex=True, constrained_layout=True)
    time = operating["time_s"]
    state_colours = {"Fresh": "#0072B2", "Held": "#D55E00", "Seeded": "#009E73"}
    state_y = {"Fresh": 2, "Held": 1, "Seeded": 0}

    add_hand_visible_band(list(axes))

    # A three-row categorical strip reads at small print sizes without implying a numeric state.
    for name in ("Fresh", "Held"):
        selected = category.eq(name)
        axes[0].fill_between(
            time,
            state_y[name] - 0.36,
            state_y[name] + 0.36,
            where=selected,
            step="mid",
            color=state_colours[name],
            lw=0,
            zorder=2,
        )
    # Seeded records in this session are mostly isolated camera frames.  Markers retain their
    # categorical meaning when a one-frame band would disappear in a reduced thesis print.
    seeded = category.eq("Seeded")
    axes[0].scatter(
        time.loc[seeded],
        np.full(int(seeded.sum()), state_y["Seeded"]),
        s=24,
        marker="o",
        color=state_colours["Seeded"],
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    axes[0].set_yticks([2, 1, 0], ["Fresh", "Held", "Seeded"])
    axes[0].set_ylim(-0.65, 2.65)
    axes[0].text(0.0, 1.03, "(a) Tracker output", transform=axes[0].transAxes, ha="left", va="bottom", fontweight="bold")

    support_line = axes[1].plot(
        time, operating["tracker_effective_plane_support"], color="#0072B2", lw=1.55, label="Effective support", zorder=2
    )[0]
    axes[1].axhline(2.75, color="#0072B2", lw=0.7, ls="--", alpha=0.65, zorder=1)
    axes[1].annotate("Minimum support (2.75)", xy=(0.0, 2.75), xycoords=("axes fraction", "data"),
                     xytext=(4, 3), textcoords="offset points", color="#006A9F", fontsize=8, va="bottom",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.35})
    axes[1].set_ylabel("Effective support")
    coverage_axis = axes[1].twinx()
    coverage_line = coverage_axis.plot(
        time, operating["tracker_weighted_plane_coverage"] * 100,
        color="#CC79A7", lw=1.55, label="Weighted coverage", zorder=2,
    )[0]
    coverage_axis.axhline(20, color="#CC79A7", lw=0.7, ls="--", alpha=0.65, zorder=1)
    coverage_axis.annotate("Minimum coverage (20%)", xy=(1.0, 20), xycoords=("axes fraction", "data"),
                           xytext=(-4, 3), textcoords="offset points", color="#8C4E73", fontsize=8,
                           ha="right", va="bottom",
                           bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.35})
    coverage_axis.set_ylabel("Weighted coverage (%)")
    coverage_axis.set_ylim(bottom=0, top=max(100, float((operating["tracker_weighted_plane_coverage"] * 100).max()) * 1.08))
    axes[1].legend([support_line, coverage_line], ["Effective support", "Weighted coverage"],
                   loc="upper left", frameon=False, fontsize=8)
    axes[1].yaxis.label.set_color("#006A9F")
    coverage_axis.yaxis.label.set_color("#8C4E73")
    axes[1].text(0.0, 1.03, "(b) Visual evidence", transform=axes[1].transAxes, ha="left", va="bottom", fontweight="bold")

    axes[2].plot(time, operating["tracker_motion_mask_area_ratio"] * 100,
                 color="#E69F00", lw=1.55, label="Dynamic mask", zorder=2)
    axes[2].plot(time, operating["tracker_hand_mask_area_ratio"] * 100,
                 color="#009E73", lw=1.55, label="Semantic hand mask", zorder=2)
    axes[2].set_ylabel("Masked image area (%)")
    axes[2].set_xlabel("Recorded video time (s)")
    axes[2].legend(loc="upper left", frameon=False, fontsize=8)
    axes[2].text(0.0, 1.03, "(c) Occlusion masking", transform=axes[2].transAxes, ha="left", va="bottom", fontweight="bold")

    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#d9d9d9", lw=0.55, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    coverage_axis.spines["top"].set_visible(False)
    coverage_axis.grid(False)
    axes[2].set_xlim(float(time.min()), float(time.max()))
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return {name.lower(): int(category.eq(name).sum()) for name in ("Fresh", "Held", "Seeded")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--thesis-figure", type=Path, help="Optional thesis-oriented surface-occlusion figure (PNG, SVG, or PDF).")
    args = parser.parse_args()
    report, camera = analyse(args.csv)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.timeline.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_timeline(camera, args.timeline)
    if args.thesis_figure:
        args.thesis_figure.parent.mkdir(parents=True, exist_ok=True)
        counts = plot_surface_occlusion_evidence(camera, args.thesis_figure)
        print(f"Thesis figure state counts: {counts}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
