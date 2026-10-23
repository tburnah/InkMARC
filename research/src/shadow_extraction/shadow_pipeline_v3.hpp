#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace shadow_v3 {

namespace fs = std::filesystem;

struct DetectionResult {
    std::map<int, std::vector<cv::Point2f>> detections;
    std::vector<std::vector<cv::Point2f>> rejected;
};

struct SoftTag {
    int id = -1;
    std::vector<cv::Point2f> frame_corners;
    double score = 0.0;
    double visible_fraction = 0.0;
    std::string source;
};

struct Options {
    std::string video_path;
    std::string reference_path;
    fs::path outdir;
    fs::path tag_failure_report_path;
    std::vector<int> frames;
    double reference_scale = 1.0;
    double overlay_alpha = 0.35;
    double shadow_strength = 1.0;
    bool report_tag_failures = false;
    bool enable_timing = false;
    bool fast = false;
    bool surface_tracking_only = false;
    bool corrected_sidecar_geometry_only = false;
    bool center_consensus = false;
    bool partial_cell_pattern = false;
    int stuck_corner_frame_limit = 2;
    double stuck_corner_epsilon = 0.01;
    bool prefer_seeded_fallback = true;
    bool write_overlay = false;
    bool write_board_mask = false;
    bool write_colormatch = false;
    bool write_hand_stylus_mask_video = false;
    bool write_hand_stylus_mask_stage_videos = false;
    bool write_raw_depth = false;
    bool final_overlay_only = false;
    bool h5_geometry_fallback = true;
    bool foreground_keep_mask_erode_enabled = true;
    bool write_foreground_keep_mask_debug = false;
    bool write_preview_mp4 = false;
    fs::path surface_texture_path;
    fs::path surface_texture_dir;
    std::string pipeline_video_format = "mp4";
    std::string final_overlay_video_codec = "h264-high";
    std::string raw_depth_python = "python3";
    fs::path raw_depth_script = "scripts/depth_anything_v2_raw_depth.py";
    fs::path depth_anything_root;
    fs::path depth_checkpoint;
    std::string depth_model_version = "v2";
    std::string depth_model_id = "depth-anything/DA3-SMALL";
    std::string depth_video_mode = "per-frame";
    int depth_streaming_chunk_size = 60;
    int depth_streaming_overlap = 30;
    std::string depth_encoder = "vits";
    std::string depth_device;
    int depth_input_size = 322;
    int depth_gradient_mask_threshold = 16;
    int tilt_gradient_mask_threshold = 8;
    bool sheet_guided_depth_threshold = true;
    int sheet_guided_max_threshold = 32;
    double sheet_guided_mismatch_percent = 3.0;
    double sheet_guided_min_retention_percent = 99.5;
    int foreground_keep_mask_dilate_kernel_size = 7;
    int foreground_keep_mask_dilate_iterations = 1;
    int foreground_keep_mask_erode_kernel_size = 7;
    int foreground_keep_mask_erode_iterations = 1;
};

struct ReferenceTagPattern {
    cv::Mat blue_mask;
    cv::Mat green_mask;
    cv::Mat valid_mask;
    int valid_pixels = 0;
    std::array<unsigned char, 36> inner_cell_labels{};
    std::array<cv::Point2f, 36> ref_inner_cell_centers{};
    int inner_cell_count = 0;
};

struct ReferenceModel {
    cv::Mat reference_bgr;
    cv::Mat reference_blue_mask;
    cv::Mat reference_green_mask;
    cv::Mat reference_visual_label_mask;
    bool reference_visual_masks_exclusive = false;
    DetectionResult reference_tags;
    std::map<int, ReferenceTagPattern> reference_tag_patterns;
};

struct GeometryResult {
    int frame_index = -1;
    cv::Mat H_ref_to_frame;
    cv::Mat H_seed_ref_to_frame;
    cv::Mat board_mask;
    std::vector<cv::Point2f> board_quad;
    DetectionResult frame_tags;
    int common_tag_count = 0;
    int soft_tag_count = 0;
    std::vector<SoftTag> soft_tags;
    std::vector<SoftTag> soft_tag_candidates;
    bool used_single_tag_fallback = false;
    bool used_previous_homography_seed = false;
    bool used_corrected_h5_corners = false;
    bool ok = false;
};

struct FrameResult {
    int frame_index = -1;
    cv::Mat output_bgr;
    GeometryResult geometry;
    bool ok = false;
};

ReferenceModel loadReferenceModel(const Options& options);

GeometryResult estimateTagSheetGeometry(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options* options = nullptr,
    const cv::Mat& seed_H_ref_to_frame = cv::Mat(),
    const DetectionResult* precomputed_frame_tags = nullptr);

FrameResult renderTagSheetOverlayV3(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    const cv::Mat& seed_H_ref_to_frame = cv::Mat(),
    const DetectionResult* precomputed_frame_tags = nullptr);

int runVideoV3(const Options& options);

int reportTagFailuresV3(const Options& options);

}  // namespace shadow_v3
