#include "shadow_pipeline_v2.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cctype>
#include <cstdlib>
#include <deque>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

// The v2 runner owns the video loop, per-frame analysis, and per-texture render
// fanout. It includes the draft implementation only as a library of proven
// helper algorithms while we keep the original draft file untouched.
#define main shadow_draft_main
#include "test_shadow_extraction_aligned_ref.cpp"
#undef main

namespace shadow_v2 {
namespace {

cv::Mat fitTextureToReferenceLocal(const cv::Mat& texture_bgr, const cv::Size& reference_size) {
    cv::Mat out;
    cv::resize(texture_bgr, out, reference_size, 0.0, 0.0, cv::INTER_AREA);
    return out;
}

::DetectionResult toDraftDetection(const DetectionResult& src) {
    ::DetectionResult out;
    out.detections = src.detections;
    out.rejected = src.rejected;
    return out;
}

DetectionResult fromDraftDetection(const ::DetectionResult& src) {
    DetectionResult out;
    out.detections = src.detections;
    out.rejected = src.rejected;
    return out;
}

::Args makeDraftArgs(const Options& options) {
    ::Args args;
    args.video = options.video_path;
    args.reference = options.reference_path;
    args.outdir = options.outdir.string();
    args.surface_texture = options.texture_path;
    args.surface_texture_dir = options.texture_dir;
    args.frames = options.frames;
    args.reference_scale = options.reference_scale;
    args.write_colormatch = options.outputs == "colormatch" || options.outputs == "all";
    args.write_combined_quality = options.outputs == "quality" || options.outputs == "all";
    args.write_raw_frame = options.outputs == "raw" || options.outputs == "all";
    args.write_aligned_ref = options.outputs == "aligned" || options.outputs == "all";
    args.write_raw_frame_flat = options.outputs == "flat_raw" || options.outputs == "all";
    args.write_aligned_ref_flat = options.outputs == "flat_aligned" || options.outputs == "all";
    return args;
}

::SheetTrackerConfig makeDraftTrackerConfig(const Options& options) {
    ::SheetTrackerConfig config;
    ::applyTrackerEnvOverrides(&config);
    config.directSheetRefineEnabled = options.direct_sheet_refine;
    config.tagDetectInterval = std::max(1, options.tag_detect_interval);
    return config;
}

::SheetTrackerState toDraftTrackerState(const TrackerState& src) {
    ::SheetTrackerState out;
    out.H_ref_to_prev = src.H_ref_to_prev;
    out.H_ref_to_prevprev = src.H_ref_to_prevprev;
    out.prevFrameBgr = src.prev_frame_bgr;
    out.prevTagContrast = src.prev_tag_contrast;
    out.referenceFeaturePoints = src.reference_feature_points;
    out.heldHomographyFrames = src.held_homography_frames;
    out.hasValidH = src.valid;
    return out;
}

void copyDraftTrackerState(const ::SheetTrackerState& src, TrackerState* dst) {
    if (dst == nullptr) {
        return;
    }
    dst->H_ref_to_prev = src.H_ref_to_prev;
    dst->H_ref_to_prevprev = src.H_ref_to_prevprev;
    dst->prev_frame_bgr = src.prevFrameBgr;
    dst->prev_tag_contrast = src.prevTagContrast;
    dst->reference_feature_points = src.referenceFeaturePoints;
    dst->held_homography_frames = src.heldHomographyFrames;
    dst->valid = src.hasValidH;
}

GeometryResult fromDraftGeometry(const ::FrameGeometry& src) {
    GeometryResult out;
    out.frame_index = src.index;
    out.H_ref_to_frame = src.H_ref_to_frame;
    if (!out.H_ref_to_frame.empty()) {
        out.H_frame_to_ref = out.H_ref_to_frame.inv();
    }
    out.board_mask_frame = src.board_mask;
    out.frame_tags = fromDraftDetection(src.frame_tags);
    out.ok = src.ok;
    out.used_tags = !src.frame_tags.detections.empty();
    return out;
}

::FrameGeometry toDraftGeometry(const GeometryResult& src, const cv::Mat& frame_bgr) {
    ::FrameGeometry out;
    out.index = src.frame_index;
    out.frame_bgr = frame_bgr;
    out.frame_tags = toDraftDetection(src.frame_tags);
    out.H_ref_to_frame = src.H_ref_to_frame;
    out.board_mask = src.board_mask_frame;
    out.ok = src.ok;
    return out;
}

cv::VideoWriter openVideoWriter(
    const fs::path& output_path,
    int width,
    int height,
    double fps) {
    cv::VideoWriter writer(
        output_path.string(),
        cv::CAP_FFMPEG,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps > 0.0 ? fps : 30.0,
        cv::Size(width, height));
    if (!writer.isOpened()) {
        writer.open(
            output_path.string(),
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps > 0.0 ? fps : 30.0,
            cv::Size(width, height));
    }
    if (!writer.isOpened()) {
        throw std::runtime_error("Could not open output video: " + output_path.string());
    }
    return writer;
}

int envIntLocal(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return default_value;
    }
}

std::string framePrefixLocal(int index) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "frame_%06d", index);
    return buffer;
}

}  // namespace

ReferenceModel loadReferenceModel(const Options& options) {
    ReferenceModel model;
    model.reference_bgr = cv::imread(options.reference_path, cv::IMREAD_COLOR);
    if (model.reference_bgr.empty()) {
        throw std::runtime_error("Could not read reference image: " + options.reference_path);
    }
    cv::rotate(model.reference_bgr, model.reference_bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
    if (std::abs(options.reference_scale - 1.0) > 1e-6) {
        cv::Size scaled(
            std::max(1, static_cast<int>(std::lround(model.reference_bgr.cols * options.reference_scale))),
            std::max(1, static_cast<int>(std::lround(model.reference_bgr.rows * options.reference_scale))));
        cv::resize(
            model.reference_bgr,
            model.reference_bgr,
            scaled,
            0.0,
            0.0,
            options.reference_scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    }
    cv::cvtColor(model.reference_bgr, model.reference_hsv, cv::COLOR_BGR2HSV);

    ::DetectionResult draft_tags = ::detectAprilTags(model.reference_bgr);
    for (const auto& [id, corners] : draft_tags.detections) {
        model.reference_tags.detections[id] = corners;
    }
    model.reference_tags.rejected = draft_tags.rejected;

    ::buildStrictReferenceColorMasks(
        model.reference_bgr,
        &model.green_mask_ref,
        &model.blue_mask_ref);
    model.surface_mask_ref = ::makeSurfaceMaskFromAlignedRef(model.reference_bgr);
    return model;
}

std::vector<TextureModel> loadTextures(const Options& options, const cv::Size& reference_size) {
    std::vector<TextureModel> textures;
    auto addTexture = [&](const fs::path& path) {
        cv::Mat texture = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (!texture.empty()) {
            textures.push_back(TextureModel{path, fitTextureToReferenceLocal(texture, reference_size), cv::Mat()});
        }
    };

    if (!options.texture_path.empty()) {
        addTexture(options.texture_path);
    }
    if (!options.texture_dir.empty()) {
        for (const auto& entry : fs::directory_iterator(options.texture_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (::isTextureImagePath(entry.path())) {
                addTexture(entry.path());
            }
        }
        std::sort(textures.begin(), textures.end(), [](const TextureModel& a, const TextureModel& b) {
            return a.path.string() < b.path.string();
        });
    }
    if (textures.empty()) {
        throw std::runtime_error("No readable texture input found");
    }
    return textures;
}

GeometryResult estimateGeometryV2(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    TrackerState* tracker) {
    ::FrameGeometry draft_geometry;
    ::Args draft_args = makeDraftArgs(options);
    ::SheetTrackerConfig draft_config = makeDraftTrackerConfig(options);
    ::SheetTrackerState draft_tracker = tracker != nullptr
        ? toDraftTrackerState(*tracker)
        : ::SheetTrackerState();
    ::DetectionResult draft_ref_tags = toDraftDetection(reference.reference_tags);
    cv::Mat reference_tag_contrast = ::makeTagContrastImage(reference.reference_bgr);

    bool ok = ::computeFrameGeometry(
        frame_index,
        frame_bgr.clone(),
        reference.reference_bgr,
        draft_ref_tags,
        reference_tag_contrast,
        &draft_config,
        &draft_tracker,
        draft_args,
        options.outdir,
        framePrefixLocal(frame_index),
        false,
        &draft_geometry,
        nullptr);
    copyDraftTrackerState(draft_tracker, tracker);
    if (!ok) {
        return GeometryResult{frame_index};
    }
    return fromDraftGeometry(draft_geometry);
}

FrameAnalysis analyzeFrameV2(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    TrackerState* tracker) {
    FrameAnalysis out;
    out.frame_index = frame_index;
    out.frame_bgr = frame_bgr.clone();

    out.geometry = estimateGeometryV2(frame_index, frame_bgr, reference, options, tracker);
    if (!out.geometry.ok) {
        return out;
    }

    ::FrameGeometry draft_geometry = toDraftGeometry(out.geometry, frame_bgr);
    ::FrameRenderData draft_render_data;
    ::Args draft_args = makeDraftArgs(options);
    ::DetectionResult draft_ref_tags = toDraftDetection(reference.reference_tags);
    cv::Mat reference_tag_contrast = ::makeTagContrastImage(reference.reference_bgr);

    bool ok = ::processFrame(
        frame_index,
        frame_bgr,
        reference.reference_bgr,
        cv::Mat(),
        draft_ref_tags,
        reference_tag_contrast,
        nullptr,
        nullptr,
        draft_args,
        options.outdir,
        framePrefixLocal(frame_index),
        false,
        nullptr,
        nullptr,
        &draft_render_data,
        &draft_geometry);
    if (!ok || draft_render_data.shadow_factor_bgr.empty()) {
        return out;
    }

    out.aligned_ref_bgr_frame = draft_render_data.aligned_ref_bgr;
    out.hand_stylus_mask_frame = draft_render_data.hand_stylus_mask;
    out.board_render_mask_frame = draft_render_data.board_render_mask;
    out.occlusion_render_mask_frame = draft_render_data.occlusion_render_mask;
    out.shadow_factor_bgr_frame = draft_render_data.shadow_factor_bgr;
    out.ok = true;
    return out;
}

TextureRenderResult renderTextureV2(
    const FrameAnalysis& analysis,
    const TextureModel& texture,
    const ReferenceModel& reference) {
    (void)reference;
    TextureRenderResult result;
    result.frame_index = analysis.frame_index;
    result.texture_path = texture.path;
    if (!analysis.ok || texture.texture_ref_bgr.empty()) {
        return result;
    }

    ::FrameRenderData draft_render_data;
    draft_render_data.frame_bgr = analysis.frame_bgr;
    draft_render_data.aligned_ref_bgr = analysis.aligned_ref_bgr_frame;
    draft_render_data.H_ref_to_frame = analysis.geometry.H_ref_to_frame;
    draft_render_data.board_mask = analysis.geometry.board_mask_frame;
    draft_render_data.hand_stylus_mask = analysis.hand_stylus_mask_frame;
    draft_render_data.board_render_mask = analysis.board_render_mask_frame;
    draft_render_data.occlusion_render_mask = analysis.occlusion_render_mask_frame;
    draft_render_data.shadow_factor_bgr = analysis.shadow_factor_bgr_frame;

    cv::Mat warped_texture;
    ::renderTextureOverlayFused(
        draft_render_data,
        texture.texture_ref_bgr,
        &warped_texture,
        &result.overlay_bgr);
    result.ok = !result.overlay_bgr.empty();
    return result;
}

int runVideoV2(const Options& options) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    std::string frame_workers = std::to_string(std::max(1, options.frame_workers));
    std::string tag_interval = std::to_string(std::max(1, options.tag_detect_interval));
    setenv("SHADOW_FRAME_WORKERS", frame_workers.c_str(), 0);
    setenv("SHADOW_TAG_DETECT_INTERVAL", tag_interval.c_str(), 0);
    setenv("SHADOW_DIRECT_SHEET_REFINE", options.direct_sheet_refine ? "1" : "0", 0);
    setenv("SHADOW_PROJECTED_TAG_CELL_COLORMATCH", options.projected_tag_cell_colormatch ? "1" : "0", 0);

    bool saving_video = options.frames.empty();
    if (saving_video) {
        ::g_info_enabled = false;
        ::g_profile_enabled = false;
    } else {
        ::g_info_enabled = std::getenv("SHADOW_INFO") != nullptr && std::string(std::getenv("SHADOW_INFO")) != "0";
        ::g_profile_enabled = std::getenv("SHADOW_PROFILE") != nullptr && std::string(std::getenv("SHADOW_PROFILE")) != "0";
    }

    fs::create_directories(options.outdir);
    ReferenceModel reference = loadReferenceModel(options);
    std::vector<TextureModel> textures = loadTextures(options, reference.reference_bgr.size());

    cv::VideoCapture cap(options.video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video: " + options.video_path);
    }
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    struct WriterEntry {
        fs::path texture_path;
        cv::VideoWriter writer;
    };
    std::vector<WriterEntry> writers;
    if (saving_video) {
        fs::path video_stem = fs::path(options.video_path).stem();
        if (textures.size() == 1 && !options.texture_path.empty()) {
            writers.push_back(WriterEntry{
                textures.front().path,
                openVideoWriter(options.outdir / "final_overlay.avi", frame_width, frame_height, fps)
            });
        } else {
            for (const TextureModel& texture : textures) {
                fs::path output = options.outdir /
                    (video_stem.string() + "_" + texture.path.stem().string() + ".avi");
                writers.push_back(WriterEntry{texture.path, openVideoWriter(output, frame_width, frame_height, fps)});
            }
        }
    }

    std::vector<int> target_frames = options.frames;
    std::sort(target_frames.begin(), target_frames.end());
    TrackerState tracker;
    int ok_count = 0;
    int written_count = 0;
    int progress_interval = std::clamp(envIntLocal("SHADOW_VIDEO_PROGRESS_INTERVAL", 25), 1, 1000000);

    auto renderDiagnosticFrame = [&](const FrameAnalysis& analysis) {
        std::vector<TextureRenderResult> rendered;
        rendered.reserve(textures.size());
        for (const TextureModel& texture : textures) {
            rendered.push_back(renderTextureV2(analysis, texture, reference));
        }
        for (const TextureRenderResult& result : rendered) {
            if (!result.ok) {
                continue;
            }
            fs::path output = options.outdir / (framePrefixLocal(analysis.frame_index) + "_v2_final_overlay.png");
            if (rendered.size() > 1) {
                output = options.outdir /
                    (framePrefixLocal(analysis.frame_index) + "_" + result.texture_path.stem().string() + "_v2_final_overlay.png");
            }
            ::writePngFast(output, result.overlay_bgr);
        }
    };

    if (!target_frames.empty()) {
        for (int target : target_frames) {
            int preroll_start = std::max(0, target - 30);
            cap.set(cv::CAP_PROP_POS_FRAMES, preroll_start);
            tracker = TrackerState();
            for (int frame_index = preroll_start; frame_index <= target; ++frame_index) {
                cv::Mat frame;
                if (!cap.read(frame) || frame.empty()) {
                    break;
                }
                if (frame_index < target) {
                    estimateGeometryV2(frame_index, frame, reference, options, &tracker);
                    continue;
                }
                FrameAnalysis analysis = analyzeFrameV2(frame_index, frame, reference, options, &tracker);
                if (analysis.ok) {
                    renderDiagnosticFrame(analysis);
                    ++ok_count;
                }
            }
        }
        std::cout << "[DONE] v2 diagnostic frames rendered: " << ok_count << "\n";
        return 0;
    }

    for (int frame_index = 0; ; ++frame_index) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        FrameAnalysis analysis = analyzeFrameV2(frame_index, frame, reference, options, &tracker);
        if (!analysis.ok) {
            if (saving_video) {
                for (auto& writer : writers) {
                    writer.writer.write(cv::Mat(frame_height, frame_width, CV_8UC3, cv::Scalar(0, 0, 0)));
                }
                ++written_count;
            }
            continue;
        }

        std::vector<TextureRenderResult> rendered;
        rendered.reserve(textures.size());
        if (textures.size() > 1 && options.texture_workers > 1) {
            std::vector<std::future<TextureRenderResult>> futures;
            for (const TextureModel& texture : textures) {
                futures.push_back(std::async(std::launch::async, [&analysis, &texture, &reference] {
                    return renderTextureV2(analysis, texture, reference);
                }));
            }
            for (auto& future : futures) {
                rendered.push_back(future.get());
            }
        } else {
            for (const TextureModel& texture : textures) {
                rendered.push_back(renderTextureV2(analysis, texture, reference));
            }
        }

        if (saving_video) {
            for (size_t i = 0; i < writers.size() && i < rendered.size(); ++i) {
                if (rendered[i].ok) {
                    writers[i].writer.write(rendered[i].overlay_bgr);
                } else {
                    writers[i].writer.write(cv::Mat(frame_height, frame_width, CV_8UC3, cv::Scalar(0, 0, 0)));
                }
            }
            ++written_count;
            if (written_count == 1 || written_count % progress_interval == 0) {
                std::cout << "[PROGRESS] v2 wrote " << written_count << " frames\n" << std::flush;
            }
        }
        ++ok_count;
    }

    for (auto& writer : writers) {
        writer.writer.release();
    }
    if (saving_video) {
        std::cout << "[DONE] v2 video frames written: " << written_count << "\n";
    } else {
        std::cout << "[DONE] v2 diagnostic frames rendered: " << ok_count << "\n";
    }
    return 0;
}

}  // namespace shadow_v2
