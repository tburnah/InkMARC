#pragma once

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace shadow_v2 {

namespace fs = std::filesystem;

struct DetectionResult {
    std::map<int, std::vector<cv::Point2f>> detections;
    std::vector<std::vector<cv::Point2f>> rejected;
};

struct Options {
    std::string video_path;
    std::string reference_path;
    std::string texture_path;
    std::string texture_dir;
    fs::path outdir;
    std::vector<int> frames;
    std::string outputs;
    double reference_scale = 1.0;
    int frame_workers = 8;
    int texture_workers = 1;
    int tag_detect_interval = 5;
    bool direct_sheet_refine = true;
    bool projected_tag_cell_colormatch = true;
    bool write_debug_frames = false;
};

struct ReferenceModel {
    cv::Mat reference_bgr;
    cv::Mat reference_hsv;
    cv::Mat green_mask_ref;
    cv::Mat blue_mask_ref;
    cv::Mat surface_mask_ref;
    DetectionResult reference_tags;

    // Precomputed once from the reference tag layout.
    // These replace per-frame reconstruction of the same cell polygons.
    struct TagCell {
        int tag_id = -1;
        int cx = 0;
        int cy = 0;
        bool green = false;
        cv::Rect roi;
        cv::Mat expected_mask_roi;
        int expected_pixels = 0;
    };
    std::vector<TagCell> tag_cells_8x8;
};

struct TrackerState {
    cv::Mat H_ref_to_prev;
    cv::Mat H_ref_to_prevprev;
    cv::Mat prev_frame_bgr;
    cv::Mat prev_tag_contrast;
    std::vector<cv::Point2f> reference_feature_points;
    int held_homography_frames = 0;
    bool valid = false;
};

struct GeometryResult {
    int frame_index = -1;
    cv::Mat H_ref_to_frame;
    cv::Mat H_frame_to_ref;
    cv::Mat board_mask_frame;
    DetectionResult frame_tags;
    bool used_tags = false;
    bool used_tracker = false;
    bool used_direct_refine = false;
    bool ok = false;
};

struct FrameAnalysis {
    int frame_index = -1;
    cv::Mat frame_bgr;
    GeometryResult geometry;

    // Canonical reference-space products. These are computed once per frame and
    // reused for every texture output.
    cv::Mat flat_frame_bgr;
    cv::Mat flat_color_ref_bgr;
    cv::Mat flat_hand_stylus_mask;
    cv::Mat flat_shadow_factor_bgr;  // v2 target: reference-space CV_32FC3 multiplier.
    cv::Mat shadow_factor_bgr_frame; // current native bridge: frame-space CV_32FC3 multiplier.
    cv::Mat aligned_ref_bgr_frame;

    // Frame-space masks used only for final compositing and diagnostics.
    cv::Mat hand_stylus_mask_frame;
    cv::Mat board_render_mask_frame;
    cv::Mat occlusion_render_mask_frame;
    bool ok = false;
};

struct TextureModel {
    fs::path path;
    cv::Mat texture_ref_bgr;
    cv::Mat texture_ref_linear;  // Optional cached linear-light texture.
};

struct TextureRenderResult {
    int frame_index = -1;
    fs::path texture_path;
    cv::Mat overlay_bgr;
    bool ok = false;
};

ReferenceModel loadReferenceModel(const Options& options);
std::vector<TextureModel> loadTextures(const Options& options, const cv::Size& reference_size);

GeometryResult estimateGeometryV2(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    TrackerState* tracker);

FrameAnalysis analyzeFrameV2(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    TrackerState* tracker);

TextureRenderResult renderTextureV2(
    const FrameAnalysis& analysis,
    const TextureModel& texture,
    const ReferenceModel& reference);

int runVideoV2(const Options& options);

}  // namespace shadow_v2
