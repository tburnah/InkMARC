#include "shadow_pipeline_v3.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --video VIDEO --reference IMAGE --outdir DIR [options]\n\n"
        << "Options:\n"
        << "  --frames N [N ...]           Specific diagnostic frames to process.\n"
        << "  --surface-tracking-only      Stage 1: match the tag sheet, write corrected H5 corners,\n"
        << "                               and a compact tag-sheet overlay MP4.\n"
        << "  --corrected-sidecar-geometry-only\n"
        << "                               Stage 2: use only the verified _corrected.h5 geometry;\n"
        << "                               do not run tag-sheet matching.\n"
        << "  --center-consensus           Experimental: fit partial tags by their centers and\n"
        << "                               exclude seed-derived tags from geometry fitting.\n"
        << "  --partial-cell-pattern      Experimental: validate partial IDs from visible 6x6\n"
        << "                               inner cells and fit their centers, not damaged corners.\n"
        << "  --stuck-corner-frame-limit N Warn when an unchanged run exceeds this many frames. Default: 2\n"
        << "  --stuck-corner-epsilon PX    Maximum corner movement treated as unchanged. Default: 0.01\n"
        << "  --reference-scale SCALE      Scale after rotating reference. Default: 1.0\n"
        << "  --overlay-alpha VALUE        Reference-sheet overlay alpha. Default: 0.35\n"
        << "  --write-overlay              Write tag-sheet overlay PNGs/video. Default off.\n"
        << "  --no-write-overlay           Disable tag-sheet overlay output.\n"
        << "  --write-board-mask           Write projected homography board-mask PNGs/video. Default off.\n"
        << "  --no-write-board-mask        Disable projected board-mask output.\n"
        << "  --write-colormatch           Write aligned-ref color-difference and hand/stylus mask artifacts.\n"
        << "  --no-write-colormatch        Disable aligned-ref color-difference artifacts.\n"
        << "  --write-hand-stylus-mask     Write the refined pre-shadow hand/stylus mask video.\n"
        << "  --no-write-hand-stylus-mask  Disable refined hand/stylus mask video output.\n"
        << "  --write-hand-stylus-mask-stages  Write MP4s for every hand-mask refinement stage.\n"
        << "  --write-raw-depth            Write raw-frame Depth Anything depth maps.\n"
        << "  --no-write-raw-depth         Disable raw-frame depth output.\n"
        << "  --surface-texture IMAGE      Fill the board with this texture and reapply extracted shadow.\n"
        << "  --surface-texture-dir DIR    Fill the board with each texture image in DIR and write one video per texture.\n"
        << "  --shadow-strength VALUE      Multiplier for transferred shadow strength. Default: 1.0\n"
        << "  --final-overlay-only         Only write final_overlay outputs from surface texture rendering.\n"
        << "  --fast                       Final-overlay production mode: skip debug outputs and legacy depth fallback.\n"
        << "  --h5-geometry-fallback      Use H5 corners only if tag-based surface alignment fails. Default on.\n"
        << "  --no-h5-geometry-fallback   Never use H5 corners for surface geometry.\n"
        << "  --pipeline-video-format NAME Generated debug video format: mp4, legacy, mkv-ffv1. Default: mp4\n"
        << "  --final-overlay-video-codec NAME  Final overlay video codec: lossless, h264-high, mjpeg-high. Default: h264-high\n"
        << "  --write-preview-mp4         Also write browser-compatible preview MP4s for final overlays.\n"
        << "  --foreground-keep-mask-dilate-kernel N     Ellipse size for compositing keep-mask dilation. Default: 7\n"
        << "  --foreground-keep-mask-dilate-iterations N Iterations for compositing keep-mask dilation. Default: 1\n"
        << "  --foreground-keep-mask-erode-kernel N      Ellipse size for compositing keep-mask erosion. Default: 7\n"
        << "  --foreground-keep-mask-erode-iterations N  Iterations for compositing keep-mask erosion. Default: 1\n"
        << "  --no-foreground-keep-mask-erode            Disable compositing keep-mask erosion.\n"
        << "  --write-foreground-keep-mask-debug         Write original/adjusted foreground keep-mask PNGs for frame runs.\n"
        << "  --raw-depth-python PATH      Python executable for raw depth output. Default: python3\n"
        << "  --raw-depth-script PATH      Python helper for raw depth output. Default: scripts/depth_anything_v2_raw_depth.py\n"
        << "  --depth-anything-root PATH   Local Depth-Anything-V2 (or V3) checkout.\n"
        << "  --depth-model-version NAME   v2 or v3. Default: v2\n"
        << "  --depth-model-id ID          DA3 model ID/path. Default: depth-anything/DA3-SMALL\n"
        << "  --depth-video-mode NAME      per-frame or streaming. Default: per-frame\n"
        << "  --depth-streaming-chunk-size N  DA3 Streaming chunk size. Default: 60\n"
        << "  --depth-streaming-overlap N     DA3 Streaming overlap. Default: 30\n"
        << "  --depth-checkpoint PATH      V2 checkpoint (only used with model version v2).\n"
        << "  --depth-encoder NAME         V2 encoder: vits, vitb, vitl, or vitg.\n"
        << "  --depth-input-size N         V2 input size (DA3 process resolution). Default: 322\n"
        << "  --depth-device DEVICE        cuda, mps, or cpu. Default: helper auto-selects.\n"
        << "  --gradient-mask-threshold N  Legacy raw-depth difference mask threshold. Default: 16\n"
        << "  --tilt-gradient-mask-threshold N  Tilt raw-depth difference mask threshold. Default: 8\n"
        << "  --sheet-guided-depth-threshold  Raise the depth threshold until visible-sheet mismatch is small. Default on\n"
        << "  --no-sheet-guided-depth-threshold  Disable visible-sheet-guided depth threshold selection.\n"
        << "  --sheet-guided-max-threshold N  Highest threshold considered. Default: 32\n"
        << "  --sheet-guided-mismatch-percent P  Maximum candidate removed by sheet exclusion. Default: 3.0\n"
        << "  --sheet-guided-min-retention-percent P  Minimum cleaned baseline retained. Default: 99.5\n"
        << "  --outputs LIST               Compatibility alias: all, overlay, board-mask, colormatch, raw-depth.\n"
        << "  --timing                     Print stage timing summary to stderr.\n"
        << "  --prefer-seeded-fallback     Try previous-seed fallback first in full-video fallback frames. Default on.\n"
        << "  --no-prefer-seeded-fallback  Use old direct-first full-video fallback ordering.\n"
        << "  --report-tag-failures        Print CSV rows for frames with fewer than 2 matching tags.\n"
        << "  --tag-failure-report PATH    Write tag failure CSV to PATH instead of stdout.\n";
}

int intArg(const std::string& value, int fallback) {
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

double doubleArg(const std::string& value, double fallback) {
    try {
        return std::stod(value);
    } catch (...) {
        return fallback;
    }
}

void applyOutputsList(const std::string& value, shadow_v3::Options* options) {
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item == "all") {
            options->write_overlay = true;
            options->write_board_mask = true;
            options->write_colormatch = true;
            options->write_raw_depth = true;
        } else if (item == "overlay") {
            options->write_overlay = true;
        } else if (item == "board-mask" || item == "board_mask") {
            options->write_board_mask = true;
        } else if (item == "colormatch" || item == "color-match") {
            options->write_colormatch = true;
        } else if (item == "raw-depth" || item == "raw_depth" || item == "depth") {
            options->write_raw_depth = true;
        } else if (item == "none") {
            options->write_overlay = false;
            options->write_board_mask = false;
            options->write_colormatch = false;
            options->write_raw_depth = false;
        } else if (!item.empty()) {
            throw std::runtime_error("Unknown --outputs entry: " + item);
        }
    }
}

shadow_v3::Options parseArgs(int argc, char** argv) {
    shadow_v3::Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--video") {
            options.video_path = needValue("--video");
        } else if (arg == "--reference") {
            options.reference_path = needValue("--reference");
        } else if (arg == "--outdir") {
            options.outdir = needValue("--outdir");
        } else if (arg == "--frames") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                options.frames.push_back(intArg(argv[++i], 0));
            }
        } else if (arg == "--surface-tracking-only") {
            options.surface_tracking_only = true;
        } else if (arg == "--corrected-sidecar-geometry-only" ||
                   arg == "--corrected_sidecar_geometry_only") {
            options.corrected_sidecar_geometry_only = true;
        } else if (arg == "--center-consensus" || arg == "--center_consensus") {
            options.center_consensus = true;
        } else if (arg == "--partial-cell-pattern" || arg == "--partial_cell_pattern") {
            options.partial_cell_pattern = true;
        } else if (arg == "--stuck-corner-frame-limit") {
            options.stuck_corner_frame_limit = intArg(
                needValue("--stuck-corner-frame-limit"), options.stuck_corner_frame_limit);
        } else if (arg == "--stuck-corner-epsilon") {
            options.stuck_corner_epsilon = doubleArg(
                needValue("--stuck-corner-epsilon"), options.stuck_corner_epsilon);
        } else if (arg == "--reference-scale") {
            options.reference_scale = doubleArg(needValue("--reference-scale"), options.reference_scale);
        } else if (arg == "--overlay-alpha") {
            options.overlay_alpha = doubleArg(needValue("--overlay-alpha"), options.overlay_alpha);
        } else if (arg == "--write-overlay") {
            options.write_overlay = true;
        } else if (arg == "--no-write-overlay") {
            options.write_overlay = false;
        } else if (arg == "--write-board-mask") {
            options.write_board_mask = true;
        } else if (arg == "--no-write-board-mask") {
            options.write_board_mask = false;
        } else if (arg == "--write-colormatch") {
            options.write_colormatch = true;
        } else if (arg == "--no-write-colormatch") {
            options.write_colormatch = false;
        } else if (arg == "--write-hand-stylus-mask") {
            options.write_hand_stylus_mask_video = true;
        } else if (arg == "--no-write-hand-stylus-mask") {
            options.write_hand_stylus_mask_video = false;
        } else if (arg == "--write-hand-stylus-mask-stages") {
            options.write_hand_stylus_mask_stage_videos = true;
            options.write_hand_stylus_mask_video = true;
        } else if (arg == "--write-raw-depth") {
            options.write_raw_depth = true;
        } else if (arg == "--no-write-raw-depth") {
            options.write_raw_depth = false;
        } else if (arg == "--surface-texture" || arg == "--surface_texture") {
            options.surface_texture_path = needValue(arg.c_str());
            options.write_colormatch = true;
        } else if (arg == "--surface-texture-dir" || arg == "--surface_texture_dir" ||
                   arg == "--surface-textures-dir" || arg == "--surface_textures_dir") {
            options.surface_texture_dir = needValue(arg.c_str());
            options.write_colormatch = true;
        } else if (arg == "--shadow-strength" || arg == "--shadow_strength") {
            options.shadow_strength = doubleArg(needValue(arg.c_str()), options.shadow_strength);
        } else if (arg == "--final-overlay-only") {
            options.final_overlay_only = true;
        } else if (arg == "--fast") {
            options.fast = true;
            options.final_overlay_only = true;
            options.write_overlay = false;
            options.write_board_mask = false;
            options.write_raw_depth = false;
            options.write_foreground_keep_mask_debug = false;
            options.write_preview_mp4 = false;
        } else if (arg == "--h5-geometry-fallback" || arg == "--h5_geometry_fallback" ||
                   arg == "--use-corrected-h5-geometry" || arg == "--use_corrected_h5_geometry") {
            options.h5_geometry_fallback = true;
        } else if (arg == "--no-h5-geometry-fallback" || arg == "--no_h5_geometry_fallback" ||
                   arg == "--no-use-corrected-h5-geometry" || arg == "--no_use_corrected_h5_geometry") {
            options.h5_geometry_fallback = false;
        } else if (arg == "--pipeline-video-format" || arg == "--pipeline_video_format") {
            options.pipeline_video_format = needValue(arg.c_str());
            if (options.pipeline_video_format != "mp4" &&
                    options.pipeline_video_format != "legacy" &&
                    options.pipeline_video_format != "mkv-ffv1") {
                throw std::runtime_error("--pipeline-video-format must be one of: mp4, legacy, mkv-ffv1");
            }
        } else if (arg == "--final-overlay-video-codec" || arg == "--final_overlay_video_codec") {
            options.final_overlay_video_codec = needValue(arg.c_str());
            if (options.final_overlay_video_codec != "lossless" &&
                    options.final_overlay_video_codec != "h264-high" &&
                    options.final_overlay_video_codec != "mjpeg-high") {
                throw std::runtime_error(
                    "--final-overlay-video-codec must be one of: lossless, h264-high, mjpeg-high");
            }
        } else if (arg == "--write-preview-mp4" || arg == "--write_preview_mp4") {
            options.write_preview_mp4 = true;
        } else if (arg == "--no-write-preview-mp4" || arg == "--no_write_preview_mp4") {
            options.write_preview_mp4 = false;
        } else if (arg == "--foreground-keep-mask-erode" || arg == "--foreground_keep_mask_erode") {
            options.foreground_keep_mask_erode_enabled = true;
        } else if (arg == "--foreground-keep-mask-dilate-kernel" || arg == "--foreground_keep_mask_dilate_kernel") {
            options.foreground_keep_mask_dilate_kernel_size =
                intArg(needValue(arg.c_str()), options.foreground_keep_mask_dilate_kernel_size);
        } else if (arg == "--foreground-keep-mask-dilate-iterations" || arg == "--foreground_keep_mask_dilate_iterations") {
            options.foreground_keep_mask_dilate_iterations =
                intArg(needValue(arg.c_str()), options.foreground_keep_mask_dilate_iterations);
        } else if (arg == "--no-foreground-keep-mask-erode" || arg == "--no_foreground_keep_mask_erode") {
            options.foreground_keep_mask_erode_enabled = false;
        } else if (arg == "--foreground-keep-mask-erode-kernel" || arg == "--foreground_keep_mask_erode_kernel") {
            options.foreground_keep_mask_erode_kernel_size =
                intArg(needValue(arg.c_str()), options.foreground_keep_mask_erode_kernel_size);
        } else if (arg == "--foreground-keep-mask-erode-iterations" || arg == "--foreground_keep_mask_erode_iterations") {
            options.foreground_keep_mask_erode_iterations =
                intArg(needValue(arg.c_str()), options.foreground_keep_mask_erode_iterations);
        } else if (arg == "--write-foreground-keep-mask-debug" || arg == "--write_foreground_keep_mask_debug") {
            options.write_foreground_keep_mask_debug = true;
        } else if (arg == "--no-write-foreground-keep-mask-debug" || arg == "--no_write_foreground_keep_mask_debug") {
            options.write_foreground_keep_mask_debug = false;
        } else if (arg == "--raw-depth-python") {
            options.raw_depth_python = needValue("--raw-depth-python");
        } else if (arg == "--raw-depth-script") {
            options.raw_depth_script = needValue("--raw-depth-script");
        } else if (arg == "--depth-anything-root") {
            options.depth_anything_root = needValue("--depth-anything-root");
        } else if (arg == "--depth-model-version") {
            options.depth_model_version = needValue("--depth-model-version");
        } else if (arg == "--depth-model-id") {
            options.depth_model_id = needValue("--depth-model-id");
        } else if (arg == "--depth-video-mode") {
            options.depth_video_mode = needValue("--depth-video-mode");
        } else if (arg == "--depth-streaming-chunk-size") {
            options.depth_streaming_chunk_size = intArg(
                needValue("--depth-streaming-chunk-size"), options.depth_streaming_chunk_size);
        } else if (arg == "--depth-streaming-overlap") {
            options.depth_streaming_overlap = intArg(
                needValue("--depth-streaming-overlap"), options.depth_streaming_overlap);
        } else if (arg == "--depth-checkpoint") {
            options.depth_checkpoint = needValue("--depth-checkpoint");
        } else if (arg == "--depth-encoder") {
            options.depth_encoder = needValue("--depth-encoder");
        } else if (arg == "--depth-input-size") {
            options.depth_input_size = intArg(needValue("--depth-input-size"), options.depth_input_size);
        } else if (arg == "--depth-device") {
            options.depth_device = needValue("--depth-device");
        } else if (arg == "--gradient-mask-threshold") {
            options.depth_gradient_mask_threshold =
                intArg(needValue("--gradient-mask-threshold"), options.depth_gradient_mask_threshold);
        } else if (arg == "--tilt-gradient-mask-threshold") {
            options.tilt_gradient_mask_threshold =
                intArg(needValue("--tilt-gradient-mask-threshold"), options.tilt_gradient_mask_threshold);
        } else if (arg == "--sheet-guided-depth-threshold") {
            options.sheet_guided_depth_threshold = true;
        } else if (arg == "--no-sheet-guided-depth-threshold") {
            options.sheet_guided_depth_threshold = false;
        } else if (arg == "--sheet-guided-max-threshold") {
            options.sheet_guided_max_threshold = intArg(
                needValue("--sheet-guided-max-threshold"), options.sheet_guided_max_threshold);
        } else if (arg == "--sheet-guided-mismatch-percent") {
            options.sheet_guided_mismatch_percent = doubleArg(
                needValue("--sheet-guided-mismatch-percent"), options.sheet_guided_mismatch_percent);
        } else if (arg == "--sheet-guided-min-retention-percent") {
            options.sheet_guided_min_retention_percent = doubleArg(
                needValue("--sheet-guided-min-retention-percent"), options.sheet_guided_min_retention_percent);
        } else if (arg == "--outputs") {
            applyOutputsList(needValue("--outputs"), &options);
        } else if (arg == "--timing") {
            options.enable_timing = true;
        } else if (arg == "--prefer-seeded-fallback") {
            options.prefer_seeded_fallback = true;
        } else if (arg == "--no-prefer-seeded-fallback") {
            options.prefer_seeded_fallback = false;
        } else if (arg == "--report-tag-failures") {
            options.report_tag_failures = true;
        } else if (arg == "--tag-failure-report") {
            options.report_tag_failures = true;
            options.tag_failure_report_path = needValue("--tag-failure-report");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.video_path.empty() || options.reference_path.empty() || options.outdir.empty()) {
        usage(argv[0]);
        throw std::runtime_error("--video, --reference, and --outdir are required");
    }
    if (!options.surface_texture_path.empty() && !options.surface_texture_dir.empty()) {
        throw std::runtime_error("Use either --surface-texture or --surface-texture-dir, not both.");
    }
    if (options.fast) {
        options.final_overlay_only = true;
        options.write_overlay = false;
        options.write_board_mask = false;
        options.write_raw_depth = false;
        options.write_foreground_keep_mask_debug = false;
        options.write_preview_mp4 = false;
    }
    if (options.foreground_keep_mask_erode_kernel_size < 1) {
        throw std::runtime_error("--foreground-keep-mask-erode-kernel must be >= 1");
    }
    if (options.foreground_keep_mask_erode_iterations < 0) {
        throw std::runtime_error("--foreground-keep-mask-erode-iterations must be >= 0");
    }
    if (options.shadow_strength < 0.0) {
        throw std::runtime_error("--shadow-strength must be >= 0");
    }
    if (options.foreground_keep_mask_dilate_kernel_size < 1) {
        throw std::runtime_error("--foreground-keep-mask-dilate-kernel must be >= 1");
    }
    if (options.foreground_keep_mask_dilate_iterations < 0) {
        throw std::runtime_error("--foreground-keep-mask-dilate-iterations must be >= 0");
    }
    if (options.stuck_corner_frame_limit < 0 || options.stuck_corner_epsilon < 0.0) {
        throw std::runtime_error("stuck-corner limits must be non-negative");
    }
    if (options.sheet_guided_max_threshold < 0 || options.sheet_guided_max_threshold > 255) {
        throw std::runtime_error("--sheet-guided-max-threshold must be between 0 and 255");
    }
    if (options.sheet_guided_mismatch_percent < 0.0 ||
            options.sheet_guided_mismatch_percent > 100.0) {
        throw std::runtime_error("--sheet-guided-mismatch-percent must be between 0 and 100");
    }
    if (options.sheet_guided_min_retention_percent < 0.0 ||
            options.sheet_guided_min_retention_percent > 100.0) {
        throw std::runtime_error("--sheet-guided-min-retention-percent must be between 0 and 100");
    }
    if (options.surface_tracking_only) {
        if (!options.frames.empty()) {
            throw std::runtime_error("--surface-tracking-only requires a full-video run without --frames");
        }
        // The tracking stage is deliberately reviewable: its only video output
        // is the compact tag-sheet overlay MP4.
        options.write_overlay = true;
        options.pipeline_video_format = "mp4";
        options.write_board_mask = false;
        options.write_colormatch = false;
        options.write_raw_depth = false;
        options.write_hand_stylus_mask_video = false;
        options.write_hand_stylus_mask_stage_videos = false;
        options.final_overlay_only = false;
        options.surface_texture_path.clear();
        options.surface_texture_dir.clear();
        options.h5_geometry_fallback = false;
    }
    if (options.surface_tracking_only && options.corrected_sidecar_geometry_only) {
        throw std::runtime_error(
            "--surface-tracking-only and --corrected-sidecar-geometry-only are separate stages");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        shadow_v3::Options options = parseArgs(argc, argv);
        int exit_code = options.report_tag_failures
            ? shadow_v3::reportTagFailuresV3(options)
            : shadow_v3::runVideoV3(options);
        if (exit_code == 0) {
            std::ostream& done_stream = options.report_tag_failures ? std::cerr : std::cout;
            done_stream << "[DONE] v3 run completed\n";
        }
        return exit_code;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
