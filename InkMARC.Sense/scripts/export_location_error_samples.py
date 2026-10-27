#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def display_dimensions_mm(diagonal_inches: float, width_px: int, height_px: int) -> tuple[float, float]:
    diagonal_mm = diagonal_inches * 25.4
    scale = diagonal_mm / math.hypot(width_px, height_px)
    return width_px * scale, height_px * scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and plot sample-level stylus location errors.")
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-id", default=None, help="Optional single-video filter; omit to export all valid rows.")
    parser.add_argument("--canonical-width", type=float, default=432.0)
    parser.add_argument("--canonical-height", type=float, default=672.0)
    parser.add_argument("--display-diagonal-inches", type=float, default=6.55)
    parser.add_argument("--display-width-px", type=int, default=1080)
    parser.add_argument("--display-height-px", type=int, default=2400)
    args = parser.parse_args()

    display_width_mm, display_height_mm = display_dimensions_mm(
        args.display_diagonal_inches, args.display_width_px, args.display_height_px
    )
    selected: list[dict[str, str]] = []
    with args.predictions_csv.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            if args.video_id and source.get("video_id") != args.video_id:
                continue
            if source.get("valid_xy", "").lower() != "true":
                continue
            selected.append(source)

    selected.sort(key=lambda row: (str(row.get("participant_id", "")), str(row.get("video_id", "")), int(row["frame_index"])))
    exported: list[dict[str, object]] = []
    participant_sample_counts: dict[str, int] = {}
    for source in selected:
            try:
                dx_norm = float(source["xy_pred_x"]) - float(source["xy_true_x"])
                dy_norm = float(source["xy_pred_y"]) - float(source["xy_true_y"])
            except (KeyError, TypeError, ValueError):
                continue
            participant_id = source.get("participant_id", "")
            participant_sample_counts[participant_id] = participant_sample_counts.get(participant_id, 0) + 1
            dx_px = dx_norm * args.canonical_width
            dy_px = dy_norm * args.canonical_height
            dx_mm = dx_norm * display_width_mm
            dy_mm = dy_norm * display_height_mm
            exported.append(
                {
                    "sample_index": len(exported) + 1,
                    "participant_sample_index": participant_sample_counts[participant_id],
                    "video_id": source["video_id"],
                    "participant_id": source.get("participant_id", ""),
                    "task_id": source.get("task_id", ""),
                    "frame_index": source.get("frame_index", ""),
                    "touch_true": source.get("touch_true", ""),
                    "touch_probability": source.get("touch_prob", ""),
                    "predicted_x_normalised": source["xy_pred_x"],
                    "predicted_y_normalised": source["xy_pred_y"],
                    "ground_truth_x_normalised": source["xy_true_x"],
                    "ground_truth_y_normalised": source["xy_true_y"],
                    "x_error_px": dx_px,
                    "y_error_px": dy_px,
                    "euclidean_error_px": math.hypot(dx_px, dy_px),
                    "x_error_mm_oneplus8": dx_mm,
                    "y_error_mm_oneplus8": dy_mm,
                    "euclidean_error_mm_oneplus8": math.hypot(dx_mm, dy_mm),
                }
            )

    if not exported:
        raise RuntimeError(f"No valid XY rows found (video_id={args.video_id!r})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "location_error_by_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    plot(exported, args.output_dir / "location_error_by_sample.png", single_video=bool(args.video_id))
    plot_mm_strip(exported, args.output_dir / "location_error_mm_strip.png")
    plot_mm_strip(exported, args.output_dir / "location_error_mm_strip.svg")
    errors_px = sorted(float(row["euclidean_error_px"]) for row in exported)
    midpoint = len(errors_px) // 2
    median_px = errors_px[midpoint] if len(errors_px) % 2 else (errors_px[midpoint - 1] + errors_px[midpoint]) / 2
    print(f"rows={len(exported)} median_error_px={median_px:.12f}")
    print(f"display_mm={display_width_mm:.6f}x{display_height_mm:.6f}")
    print(csv_path)


def plot(rows: list[dict[str, object]], output_path: Path, single_video: bool) -> None:
    import matplotlib.pyplot as plt

    sample = [int(row["sample_index"]) for row in rows]
    frame = [int(row["frame_index"]) for row in rows]
    error_px = [float(row["euclidean_error_px"]) for row in rows]
    error_mm = [float(row["euclidean_error_mm_oneplus8"]) for row in rows]
    sorted_px = sorted(error_px)
    midpoint = len(sorted_px) // 2
    median_px = sorted_px[midpoint] if len(sorted_px) % 2 else (sorted_px[midpoint - 1] + sorted_px[midpoint]) / 2
    participants = [str(row["participant_id"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    colours = {participant: f"C{index}" for index, participant in enumerate(sorted(set(participants)))}
    for participant in sorted(colours):
        indices = [index for index, value in enumerate(participants) if value == participant]
        axes[0].scatter(
            [sample[index] for index in indices], [error_px[index] for index in indices],
            s=5 if len(rows) > 500 else 14, alpha=0.55, color=colours[participant], label=f"Participant {participant}"
        )
    axes[0].axhline(median_px, color="#c43c35", linestyle="--", linewidth=1.4, label=f"Median = {median_px:.2f} px")
    axes[0].set_ylabel("Euclidean error (px)")
    axes[0].set_yscale("log")
    axes[0].set_title("Location error by sample" if single_video else "Six-participant held-out location error by sample")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)

    for participant in sorted(colours):
        indices = [index for index, value in enumerate(participants) if value == participant]
        axes[1].scatter(
            [sample[index] for index in indices], [error_mm[index] for index in indices],
            s=5 if len(rows) > 500 else 14, alpha=0.55, color=colours[participant]
        )
    for threshold in (1, 2, 3, 5):
        axes[1].axhline(threshold, color="#777777", linestyle=":" if threshold < 5 else "--", linewidth=0.9)
        axes[1].text(sample[-1] + 0.25, threshold, f"{threshold} mm", va="center", fontsize=8)
    axes[1].set_xlabel("Sample (ordered by frame index)")
    axes[1].set_ylabel("OnePlus 8 display-space error (mm)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)

    if single_video:
        for index, frame_index in zip(sample, frame):
            if index == 1 or index == sample[-1] or index % 5 == 0:
                axes[1].annotate(str(frame_index), (index, 0), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=7)
        fig.text(0.5, 0.005, "Small labels show source frame index", ha="center", fontsize=8, color="#555555")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mm_strip(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    errors_mm = np.asarray([float(row["euclidean_error_mm_oneplus8"]) for row in rows])
    participants = np.asarray([str(row["participant_id"]) for row in rows])
    participant_palette = (
        "#2457c5",
        "#2864dc",
        "#2f70d0",
        "#367bc5",
        "#4385bd",
        "#528fb5",
    )
    participant_colours = {
        participant: participant_palette[index % len(participant_palette)]
        for index, participant in enumerate(sorted(set(participants)))
    }
    bin_width_mm = 0.10
    upper_bound_mm = 10.0
    bin_edges = np.arange(0.0, upper_bound_mm + bin_width_mm, bin_width_mm)
    counts, _ = np.histogram(errors_mm, bins=bin_edges)
    overflow_count = int(np.count_nonzero(errors_mm > upper_bound_mm))
    overflow_x = upper_bound_mm + 0.25
    samples_per_dot = 10
    columns_per_bin = 3
    row_height_percentage = columns_per_bin * samples_per_dot * 100.0 / len(errors_mm)
    full_marker_size = 3.2
    packed_x_parts: list[np.ndarray] = []
    packed_y_parts: list[np.ndarray] = []
    packed_participant_parts: list[np.ndarray] = []
    marker_size_parts: list[np.ndarray] = []
    rng = np.random.default_rng(1337)

    # Aggregate consecutive groups of samples into representative dots, then
    # pack those dots into unique cells. A final partial group is drawn smaller.
    groups: list[tuple[np.ndarray, float]] = []
    for bin_index in range(len(counts)):
        left = bin_edges[bin_index]
        right = bin_edges[bin_index + 1]
        if bin_index == len(counts) - 1:
            indices = np.flatnonzero((errors_mm >= left) & (errors_mm <= right))
        else:
            indices = np.flatnonzero((errors_mm >= left) & (errors_mm < right))
        groups.append((indices, left))
    groups.append((np.flatnonzero(errors_mm > upper_bound_mm), overflow_x - bin_width_mm / 2.0))

    cell_width_mm = bin_width_mm / columns_per_bin
    for indices, left in groups:
        indices = rng.permutation(indices)
        ranks = np.arange(math.ceil(len(indices) / samples_per_dot))
        representative_indices = indices[ranks * samples_per_dot]
        dot_weights = np.minimum(samples_per_dot, len(indices) - ranks * samples_per_dot)
        columns = ranks % columns_per_bin
        grid_rows = ranks // columns_per_bin
        row_offsets = np.where(grid_rows % 2 == 0, -0.12, 0.12)
        horizontal_jitter = rng.uniform(-0.08, 0.08, size=len(ranks))
        vertical_jitter = rng.uniform(-0.08, 0.08, size=len(ranks))
        packed_x_parts.append(
            left + (columns + 0.5 + row_offsets + horizontal_jitter) * cell_width_mm
        )
        packed_y_parts.append((grid_rows + 0.5 + vertical_jitter) * row_height_percentage)
        packed_participant_parts.append(participants[representative_indices])
        marker_size_parts.append(full_marker_size * dot_weights / samples_per_dot)

    packed_x = np.concatenate(packed_x_parts)
    packed_y = np.concatenate(packed_y_parts)
    packed_participants = np.concatenate(packed_participant_parts)
    marker_sizes = np.concatenate(marker_size_parts)

    fig, ax = plt.subplots(figsize=(13, 2.5), constrained_layout=True)
    for participant in sorted(participant_colours):
        participant_selected = packed_participants == participant
        for marker_size in np.unique(marker_sizes[participant_selected]):
            selected = participant_selected & (marker_sizes == marker_size)
            ax.scatter(
                packed_x[selected],
                packed_y[selected],
                s=float(marker_size),
                alpha=0.9,
                color=participant_colours[participant],
                edgecolors="none",
                rasterized=output_path.suffix.lower() != ".svg",
            )
    ax.set_xlabel("OnePlus 8 display-space error (mm)", fontsize=12)
    ax.set_ylabel("Samples per bin (%)", fontsize=11)
    ax.set_xlim(0, overflow_x + 0.15)
    maximum_count = max(int(counts.max()), overflow_count)
    ax.set_ylim(0, maximum_count * 100.0 / len(errors_mm) * 1.12)
    ax.set_xticks([*range(0, 11), overflow_x])
    ax.set_xticklabels([*[str(value) for value in range(0, 11)], ">10"])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4,
            markerfacecolor=participant_colours[participant],
            markeredgewidth=0,
            label=participant,
        )
        for participant in sorted(participant_colours)
    ]
    ax.legend(
        handles=legend_handles,
        title="Participant",
        loc="upper right",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        handletextpad=0.25,
        columnspacing=0.8,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
