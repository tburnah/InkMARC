#include "shadow_pipeline_v3.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <unistd.h>

namespace shadow_v3 {
namespace {

constexpr int kTagPatternPatchSize = 72;
constexpr float kStylusMaxX = 44800.0f;
constexpr float kStylusMaxY = 29600.0f;
constexpr int kStylusModelWidth = 480;
constexpr int kStylusModelHeight = 320;

struct H5CornerTrace {
    std::vector<std::array<cv::Point2f, 4>> frame_corners;
    std::vector<cv::Point2f> stylus_points;
    std::vector<float> stylus_pressure;
    std::vector<cv::Point2f> stylus_tilt;
    cv::Size2f source_size;
    fs::path path;
    bool available = false;
    bool corrected = false;
    bool stylus_available = false;
    bool tilt_available = false;
};

struct StylusWedgeTemporalState {
    bool valid = false;
    int frame_index = -1;
    cv::Point2f tip;
    cv::Point2f vertex;
    cv::Point2f endpoint0;
    cv::Point2f endpoint1;
    bool rgb_valid = false;
    int rgb_frame_index = -1;
    cv::Point2f rgb_tip;
    cv::Point2f rgb_centroid;
    int rgb_area = 0;
};

struct SurfaceTextureEntry {
    fs::path path;
    cv::Mat texture_ref_bgr;
};

herr_t closeH5(hid_t id, herr_t (*closer)(hid_t)) {
    return id >= 0 ? closer(id) : 0;
}

const cv::Mat& morphKernel3() {
    static const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    return kernel;
}

const cv::Mat& morphKernel5() {
    static const cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
    return kernel;
}

int envInt(const char* name, int default_value);

void configureOpenCvRuntime() {
    cv::setUseOptimized(true);
    const int cpu_count = std::max(1, cv::getNumberOfCPUs());
    const int requested_threads = envInt("SHADOW_OPENCV_THREADS", cpu_count);
    cv::setNumThreads(std::clamp(requested_threads, 1, cpu_count));
}

DetectionResult detectAprilTags(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(gray, &min_value, &max_value);
    if (max_value <= 0.0) {
        return DetectionResult();
    }

    static const cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    static const cv::aruco::ArucoDetector detector = [] {
        cv::aruco::DetectorParameters params;
        params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
        params.adaptiveThreshWinSizeMax = 51;
        params.adaptiveThreshWinSizeStep = 8;
        params.aprilTagMinWhiteBlackDiff = 3;
        params.aprilTagDeglitch = 1;
        return cv::aruco::ArucoDetector(dictionary, params);
    }();

    DetectionResult result;

    auto mergeDetections = [&](const cv::Mat& input_gray, double point_scale) {
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> rejected;
        detector.detectMarkers(input_gray, corners, ids, rejected);
        for (std::vector<cv::Point2f> rejected_quad : rejected) {
            if (std::abs(point_scale - 1.0) > 1e-6) {
                for (cv::Point2f& point : rejected_quad) {
                    point.x = static_cast<float>(point.x / point_scale);
                    point.y = static_cast<float>(point.y / point_scale);
                }
            }
            result.rejected.push_back(std::move(rejected_quad));
        }

        for (size_t i = 0; i < ids.size(); ++i) {
            int id = ids[i];
            if (result.detections.count(id) != 0) {
                continue;
            }
            std::vector<cv::Point2f> scaled_corners = corners[i];
            if (std::abs(point_scale - 1.0) > 1e-6) {
                for (cv::Point2f& point : scaled_corners) {
                    point.x = static_cast<float>(point.x / point_scale);
                    point.y = static_cast<float>(point.y / point_scale);
                }
            }
            result.detections[id] = std::move(scaled_corners);
        }
    };

    mergeDetections(gray, 1.0);

    if (result.detections.size() < 4) {
        cv::Mat equalized;
        cv::equalizeHist(gray, equalized);
        mergeDetections(equalized, 1.0);
    }

    if (result.detections.size() < 4) {
        static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
        cv::Mat clahe_gray;
        clahe->apply(gray, clahe_gray);
        mergeDetections(clahe_gray, 1.0);
    }

    if (result.detections.size() < 4) {
        cv::Mat normalized;
        cv::normalize(gray, normalized, 0, 255, cv::NORM_MINMAX);
        mergeDetections(normalized, 1.0);
    }

    if (result.detections.size() < 4) {
        cv::Mat bright;
        gray.convertTo(bright, CV_8U, 1.8, 24.0);
        mergeDetections(bright, 1.0);
    }

    if (result.detections.size() < 4) {
        constexpr double fallback_scale = 0.75;
        cv::Mat small_gray;
        cv::resize(gray, small_gray, cv::Size(), fallback_scale, fallback_scale, cv::INTER_AREA);
        mergeDetections(small_gray, fallback_scale);
    }

    return result;
}

bool writePngFast(const fs::path& path, const cv::Mat& image) {
    return cv::imwrite(path.string(), image);
}

void writeBoardMaskPng(const fs::path& path, const cv::Mat& board_mask) {
    if (board_mask.empty()) {
        return;
    }
    writePngFast(path, board_mask);
}

std::string framePrefix(int index) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "frame_%06d", index);
    return buffer;
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

std::string shellQuote(const std::string& value);

std::string fpsArgument(double fps) {
    std::ostringstream out;
    out << (fps > 0.0 ? fps : 30.0);
    return out.str();
}

struct FinalOverlayVideoWriter {
    FILE* pipe = nullptr;
    cv::VideoWriter fallback;
    int width = 0;
    int height = 0;

    void open(
        const fs::path& output_path,
        int output_width,
        int output_height,
        double fps,
        const std::string& codec = "lossless") {
        width = output_width;
        height = output_height;
        if (codec == "legacy-mjpg") {
            fallback = openVideoWriter(output_path, width, height, fps);
            return;
        }
        std::string encoder_args;
        std::string codec_label = codec;
        if (codec == "h264-high") {
            encoder_args = "-an -c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv444p ";
        } else if (codec == "mjpeg-high") {
            encoder_args = "-an -c:v mjpeg -q:v 2 -pix_fmt yuvj444p ";
        } else if (codec == "preview-mp4") {
            encoder_args = "-vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" "
                "-an -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p ";
        } else {
            codec_label = "lossless";
            encoder_args = "-an -c:v ffv1 -level 3 -g 1 ";
        }
        const std::string command =
            "ffmpeg -hide_banner -loglevel error -y "
            "-f rawvideo -pix_fmt bgr24 " +
            std::string("-s ") + std::to_string(width) + "x" + std::to_string(height) + " " +
            "-r " + fpsArgument(fps) + " -i - "
            + encoder_args +
            shellQuote(output_path.string());
        pipe = popen(command.c_str(), "w");
        if (pipe == nullptr) {
            std::cerr << "[WARN] Could not start ffmpeg final overlay writer (codec="
                      << codec_label << "), falling back to MJPG: "
                      << output_path << "\n";
            fallback = openVideoWriter(output_path, width, height, fps);
        }
    }

    void write(const cv::Mat& frame_bgr) {
        if (pipe == nullptr) {
            fallback.write(frame_bgr);
            return;
        }
        cv::Mat output = frame_bgr;
        if (output.empty()) {
            output = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        if (output.size() != cv::Size(width, height)) {
            cv::resize(output, output, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
        }
        if (output.channels() == 1) {
            cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
        } else if (output.channels() == 4) {
            cv::cvtColor(output, output, cv::COLOR_BGRA2BGR);
        }
        if (!output.isContinuous()) {
            output = output.clone();
        }
        const size_t bytes = output.total() * output.elemSize();
        const size_t written = std::fwrite(output.data, 1, bytes, pipe);
        if (written != bytes) {
            std::cerr << "[WARN] Short write to ffmpeg final overlay stream\n";
        }
    }

    void release() {
        if (pipe != nullptr) {
            const int status = pclose(pipe);
            pipe = nullptr;
            if (status != 0) {
                std::cerr << "[WARN] ffmpeg final overlay writer exited with status "
                          << status << "\n";
            }
        }
        if (fallback.isOpened()) {
            fallback.release();
        }
    }
};

bool useMkvFfv1PipelineVideos(const Options& options) {
    return options.pipeline_video_format == "mkv-ffv1";
}

fs::path pipelineVideoOutputPath(
    const fs::path& outdir,
    const std::string& stem,
    const std::string& suffix,
    const Options& options) {
    std::string extension = ".mp4";
    if (useMkvFfv1PipelineVideos(options)) {
        extension = ".mkv";
    } else if (options.pipeline_video_format == "legacy") {
        extension = ".avi";
    }
    return outdir / (stem + suffix + extension);
}

fs::path finalOverlayVideoOutputPath(
    const fs::path& outdir,
    const std::string& stem,
    const std::string& suffix,
    const Options& options) {
    std::string extension = ".avi";
    if (useMkvFfv1PipelineVideos(options)) {
        extension = ".mkv";
    } else if (options.final_overlay_video_codec == "h264-high") {
        extension = ".mp4";
    }
    return outdir / (stem + suffix + extension);
}

std::string pipelineVideoCodec(const Options& options) {
    if (useMkvFfv1PipelineVideos(options)) {
        return "lossless";
    }
    if (options.pipeline_video_format == "legacy") {
        return "legacy-mjpg";
    }
    return "preview-mp4";
}

std::string finalOverlayVideoCodec(const Options& options) {
    return useMkvFfv1PipelineVideos(options) ? "lossless" : options.final_overlay_video_codec;
}

void drawFrameNumberLabel(cv::Mat* frame_bgr, int frame_index) {
    if (frame_bgr == nullptr || frame_bgr->empty()) {
        return;
    }
    const std::string label = "Frame " + std::to_string(frame_index);
    constexpr int font = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double font_scale = 0.8;
    constexpr int thickness = 2;
    constexpr int padding = 7;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        label, font, font_scale, thickness, &baseline);
    const cv::Rect background(
        0,
        0,
        std::min(frame_bgr->cols, text_size.width + 2 * padding),
        std::min(frame_bgr->rows, text_size.height + baseline + 2 * padding));
    cv::rectangle(*frame_bgr, background, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(
        *frame_bgr,
        label,
        cv::Point(padding, padding + text_size.height),
        font,
        font_scale,
        cv::Scalar(255, 255, 255),
        thickness,
        cv::LINE_AA);
}

fs::path previewVideoPath(const fs::path& outdir, const std::string& stem, const std::string& suffix) {
    return outdir / (stem + suffix + "_preview.mp4");
}

struct SurfaceTextureVideoOutput {
    fs::path texture_path;
    fs::path output_path;
    cv::Mat texture_ref_bgr;
    FinalOverlayVideoWriter final_overlay_writer;
    FinalOverlayVideoWriter preview_writer;
};

bool hasSurfaceTextureOutput(const Options& options) {
    return !options.surface_texture_path.empty() || !options.surface_texture_dir.empty();
}

fs::path h5SidecarPathForVideo(const std::string& video_path) {
    fs::path sidecar(video_path);
    sidecar.replace_extension(".h5");
    return sidecar;
}

fs::path correctedH5SidecarPathForVideo(const std::string& video_path) {
    fs::path corrected = h5SidecarPathForVideo(video_path);
    corrected.replace_filename(corrected.stem().string() + "_corrected" + corrected.extension().string());
    return corrected;
}

fs::path preferredH5SidecarPathForVideo(const std::string& video_path) {
    fs::path corrected = correctedH5SidecarPathForVideo(video_path);
    if (fs::exists(corrected)) {
        return corrected;
    }
    fs::path original = h5SidecarPathForVideo(video_path);
    if (fs::exists(original)) {
        return original;
    }
    return fs::path();
}

bool readH5IntAttribute(hid_t file, const char* name, int* value) {
    if (value == nullptr || H5Aexists(file, name) <= 0) {
        return false;
    }
    hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
    if (attr < 0) {
        return false;
    }
    herr_t status = H5Aread(attr, H5T_NATIVE_INT, value);
    closeH5(attr, H5Aclose);
    return status >= 0;
}

bool readH5ResizeOutputSize(hid_t file, int* width, int* height) {
    if (width == nullptr || height == nullptr) {
        return false;
    }
    hid_t group = H5Gopen2(file, "preprocessing/resize_transform", H5P_DEFAULT);
    if (group < 0) {
        return false;
    }
    int output_width = 0;
    int output_height = 0;
    bool ok =
        readH5IntAttribute(group, "output_width", &output_width) &&
        readH5IntAttribute(group, "output_height", &output_height) &&
        output_width > 0 &&
        output_height > 0;
    closeH5(group, H5Gclose);
    if (!ok) {
        return false;
    }
    *width = output_width;
    *height = output_height;
    return true;
}

bool copyH5AttributeByName(hid_t src, hid_t dst, const char* name) {
    if (H5Aexists(src, name) <= 0) {
        return true;
    }
    hid_t attr = H5Aopen(src, name, H5P_DEFAULT);
    if (attr < 0) {
        return false;
    }
    hid_t type = H5Aget_type(attr);
    hid_t space = H5Aget_space(attr);
    bool ok = type >= 0 && space >= 0;
    std::vector<unsigned char> data;
    if (ok) {
        hssize_t point_count = H5Sget_simple_extent_npoints(space);
        size_t type_size = H5Tget_size(type);
        ok = point_count >= 0 && type_size > 0;
        if (ok) {
            data.resize(static_cast<size_t>(point_count) * type_size);
            ok = H5Aread(attr, type, data.data()) >= 0;
        }
    }
    if (ok) {
        hid_t dst_attr = H5Acreate2(dst, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
        ok = dst_attr >= 0 && H5Awrite(dst_attr, type, data.data()) >= 0;
        closeH5(dst_attr, H5Aclose);
    }
    closeH5(space, H5Sclose);
    closeH5(type, H5Tclose);
    closeH5(attr, H5Aclose);
    return ok;
}

void copyKnownH5RootAttributes(hid_t src, hid_t dst) {
    const char* names[] = {
        "attr_count",
        "channels",
        "fps",
        "frame_count",
        "height",
        "width",
    };
    for (const char* name : names) {
        if (!copyH5AttributeByName(src, dst, name)) {
            std::cerr << "[WARN] Could not copy H5 root attribute: " << name << "\n";
        }
    }
}

struct H5RootCopyContext {
    hid_t src_file = -1;
    hid_t dst_file = -1;
};

herr_t copyH5RootLinkCallback(
    hid_t,
    const char* name,
    const H5L_info_t*,
    void* op_data) {
    if (std::strcmp(name, "corners") == 0) {
        return 0;
    }
    H5RootCopyContext* context = static_cast<H5RootCopyContext*>(op_data);
    return H5Ocopy(
        context->src_file,
        name,
        context->dst_file,
        name,
        H5P_DEFAULT,
        H5P_DEFAULT) >= 0 ? 0 : -1;
}

bool h5CornerArrayUsable(const std::array<cv::Point2f, 4>& corners) {
    std::vector<cv::Point2f> points(corners.begin(), corners.end());
    for (const cv::Point2f& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
    }
    return std::abs(cv::contourArea(points)) > 1.0;
}

std::vector<cv::Point2f> projectQuad(const cv::Mat& H, const std::vector<cv::Point2f>& quad);
std::vector<cv::Point2f> referenceImageCorners(cv::Size size);

bool correctedH5CornersFromGeometry(
    const GeometryResult& geometry,
    const ReferenceModel& reference,
    cv::Size frame_size,
    cv::Size2f h5_source_size,
    std::array<cv::Point2f, 4>* out) {
    if (out == nullptr ||
            !geometry.ok ||
            frame_size.width <= 0 ||
            frame_size.height <= 0 ||
            h5_source_size.width <= 0.0f ||
            h5_source_size.height <= 0.0f) {
        return false;
    }

    std::vector<cv::Point2f> frame_corners = geometry.board_quad;
    if (frame_corners.size() != 4 && !geometry.H_ref_to_frame.empty()) {
        frame_corners = projectQuad(
            geometry.H_ref_to_frame,
            referenceImageCorners(reference.reference_bgr.size()));
    }
    if (frame_corners.size() != 4) {
        return false;
    }

    const float scale_x = h5_source_size.width / static_cast<float>(frame_size.width);
    const float scale_y = h5_source_size.height / static_cast<float>(frame_size.height);
    for (size_t i = 0; i < 4; ++i) {
        const cv::Point2f& p = frame_corners[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            return false;
        }
        (*out)[i] = cv::Point2f(p.x * scale_x, p.y * scale_y);
    }
    return h5CornerArrayUsable(*out);
}

cv::Mat makeStylusRgbCandidateMask(
    const cv::Mat& frame_bgr,
    const cv::Mat& prior_mask,
    cv::Mat* core_debug_mask,
    cv::Mat* yellow_debug_mask,
    cv::Mat* dark_debug_mask);

bool writeCorrectedH5Sidecar(
    const std::string& video_path,
    const std::vector<std::array<cv::Point2f, 4>>& corrected_corners,
    int updated_count) {
    if (corrected_corners.empty() || updated_count <= 0) {
        return false;
    }

    const fs::path source_path = h5SidecarPathForVideo(video_path);
    const fs::path output_path = correctedH5SidecarPathForVideo(video_path);
    const fs::path temp_path = output_path.string() + ".tmp";
    if (!fs::exists(source_path)) {
        return false;
    }

    std::error_code fs_error;
    fs::remove(temp_path, fs_error);
    fs_error.clear();
    hid_t src_file = H5Fopen(source_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (src_file < 0) {
        std::cerr << "[WARN] Could not open source H5 for corrected write: " << source_path << "\n";
        return false;
    }
    hid_t dst_file = H5Fcreate(temp_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (dst_file < 0) {
        closeH5(src_file, H5Fclose);
        std::cerr << "[WARN] Could not create corrected H5: " << temp_path << "\n";
        return false;
    }

    bool ok = true;
    H5RootCopyContext context{src_file, dst_file};
    hsize_t idx = 0;
    if (H5Literate(src_file, H5_INDEX_NAME, H5_ITER_INC, &idx, copyH5RootLinkCallback, &context) < 0) {
        std::cerr << "[WARN] Could not copy all H5 datasets while writing corrected sidecar\n";
        ok = false;
    }
    copyKnownH5RootAttributes(src_file, dst_file);

    std::vector<float> raw(corrected_corners.size() * 8);
    for (size_t frame = 0; frame < corrected_corners.size(); ++frame) {
        for (size_t corner = 0; corner < 4; ++corner) {
            const size_t base = frame * 8 + corner * 2;
            raw[base] = corrected_corners[frame][corner].x;
            raw[base + 1] = corrected_corners[frame][corner].y;
        }
    }

    hsize_t dims[3] = {
        static_cast<hsize_t>(corrected_corners.size()),
        4,
        2,
    };
    hid_t space = H5Screate_simple(3, dims, nullptr);
    hid_t dataset = space >= 0
        ? H5Dcreate2(dst_file, "corners", H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
        : -1;
    if (dataset < 0 || H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()) < 0) {
        std::cerr << "[WARN] Could not write corrected H5 /corners dataset\n";
        ok = false;
    }
    closeH5(dataset, H5Dclose);
    closeH5(space, H5Sclose);
    closeH5(dst_file, H5Fclose);
    closeH5(src_file, H5Fclose);

    if (!ok) {
        fs::remove(temp_path);
        return false;
    }

    fs::remove(output_path, fs_error);
    fs_error.clear();
    fs::rename(temp_path, output_path, fs_error);
    if (fs_error) {
        std::cerr << "[WARN] Could not move corrected H5 into place: "
                  << temp_path << " -> " << output_path
                  << " (" << fs_error.message() << ")\n";
        fs_error.clear();
        fs::remove(temp_path, fs_error);
        return false;
    }
    std::cout << "[INFO] Wrote corrected H5 corners: " << output_path
              << " updated_frames=" << updated_count << "\n";
    return true;
}

H5CornerTrace loadH5CornerTraceForVideo(
    const std::string& video_path,
    bool prefer_corrected = true) {
    H5CornerTrace trace;
    fs::path original_sidecar = h5SidecarPathForVideo(video_path);
    fs::path corrected_sidecar = correctedH5SidecarPathForVideo(video_path);
    fs::path sidecar = (prefer_corrected && fs::exists(corrected_sidecar))
        ? corrected_sidecar
        : original_sidecar;
    if (!fs::exists(sidecar)) {
        return trace;
    }
    trace.path = sidecar;
    trace.corrected = sidecar == corrected_sidecar;

    H5E_auto2_t old_error_func = nullptr;
    void* old_error_data = nullptr;
    H5Eget_auto2(H5E_DEFAULT, &old_error_func, &old_error_data);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    hid_t file = H5Fopen(sidecar.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        H5Eset_auto2(H5E_DEFAULT, old_error_func, old_error_data);
        std::cerr << "[WARN] Could not open H5 corner sidecar, skipping red outline: "
                  << sidecar << "\n";
        return trace;
    }

    hid_t dataset = H5Dopen2(file, "corners", H5P_DEFAULT);
    if (dataset < 0) {
        closeH5(file, H5Fclose);
        H5Eset_auto2(H5E_DEFAULT, old_error_func, old_error_data);
        std::cerr << "[WARN] H5 sidecar has no /corners dataset, skipping red outline: "
                  << sidecar << "\n";
        return trace;
    }

    hid_t dataspace = H5Dget_space(dataset);
    int rank = dataspace >= 0 ? H5Sget_simple_extent_ndims(dataspace) : -1;
    hsize_t dims[3] = {0, 0, 0};
    if (rank != 3 || H5Sget_simple_extent_dims(dataspace, dims, nullptr) < 0 ||
            dims[1] != 4 || dims[2] != 2) {
        closeH5(dataspace, H5Sclose);
        closeH5(dataset, H5Dclose);
        closeH5(file, H5Fclose);
        H5Eset_auto2(H5E_DEFAULT, old_error_func, old_error_data);
        std::cerr << "[WARN] H5 /corners dataset has unexpected shape, skipping red outline: "
                  << sidecar << "\n";
        return trace;
    }

    std::vector<float> raw(static_cast<size_t>(dims[0] * dims[1] * dims[2]));
    herr_t read_status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data());
    closeH5(dataspace, H5Sclose);
    closeH5(dataset, H5Dclose);

    int attr_width = 0;
    int attr_height = 0;
    readH5IntAttribute(file, "width", &attr_width);
    readH5IntAttribute(file, "height", &attr_height);
    int resize_output_width = 0;
    int resize_output_height = 0;
    const bool have_resize_output_size =
        readH5ResizeOutputSize(file, &resize_output_width, &resize_output_height);

    hid_t labels_dataset = H5Dopen2(file, "labels", H5P_DEFAULT);
    hid_t mask_dataset = H5Dopen2(file, "label_mask", H5P_DEFAULT);
    if (labels_dataset >= 0 && mask_dataset >= 0) {
        hid_t labels_space = H5Dget_space(labels_dataset);
        hid_t mask_space = H5Dget_space(mask_dataset);
        hsize_t label_dims[2] = {0, 0};
        hsize_t mask_dims[2] = {0, 0};
        int label_rank = labels_space >= 0 ? H5Sget_simple_extent_ndims(labels_space) : -1;
        int mask_rank = mask_space >= 0 ? H5Sget_simple_extent_ndims(mask_space) : -1;
        if (label_rank == 2 &&
                mask_rank == 2 &&
                H5Sget_simple_extent_dims(labels_space, label_dims, nullptr) >= 0 &&
                H5Sget_simple_extent_dims(mask_space, mask_dims, nullptr) >= 0 &&
                label_dims[0] == mask_dims[0] &&
                label_dims[1] >= 3 &&
                mask_dims[1] >= 2) {
            std::vector<float> labels(static_cast<size_t>(label_dims[0] * label_dims[1]));
            std::vector<unsigned char> mask(static_cast<size_t>(mask_dims[0] * mask_dims[1]));
            if (H5Dread(labels_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, labels.data()) >= 0 &&
                    H5Dread(mask_dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask.data()) >= 0) {
                trace.stylus_points.assign(static_cast<size_t>(label_dims[0]), cv::Point2f(-1.0f, -1.0f));
                trace.stylus_pressure.assign(static_cast<size_t>(label_dims[0]), 0.0f);
                trace.stylus_tilt.assign(
                    static_cast<size_t>(label_dims[0]),
                    cv::Point2f(
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()));
                const size_t label_stride = static_cast<size_t>(label_dims[1]);
                const size_t mask_stride = static_cast<size_t>(mask_dims[1]);
                for (size_t frame = 0; frame < trace.stylus_points.size(); ++frame) {
                    if (mask[frame * mask_stride] == 0 ||
                            mask[frame * mask_stride + 1] == 0) {
                        continue;
                    }
                    float x = labels[frame * label_stride];
                    float y = labels[frame * label_stride + 1];
                    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0f || y < 0.0f) {
                        continue;
                    }
                    float pressure = 0.0f;
                    if (mask_stride >= 3 && mask[frame * mask_stride + 2] != 0) {
                        pressure = labels[frame * label_stride + 2];
                        if (!std::isfinite(pressure)) {
                            pressure = 0.0f;
                        }
                    }
                    trace.stylus_points[frame] = cv::Point2f(x, y);
                    trace.stylus_pressure[frame] = pressure;
                    if (label_dims[1] >= 5 &&
                            mask_dims[1] >= 5 &&
                            mask[frame * mask_stride + 3] != 0 &&
                            mask[frame * mask_stride + 4] != 0) {
                        float tilt_x = labels[frame * label_stride + 3];
                        float tilt_y = labels[frame * label_stride + 4];
                        if (std::isfinite(tilt_x) && std::isfinite(tilt_y)) {
                            trace.stylus_tilt[frame] = cv::Point2f(tilt_x, tilt_y);
                            trace.tilt_available = true;
                        }
                    }
                    trace.stylus_available = true;
                }
            }
        }
        closeH5(labels_space, H5Sclose);
        closeH5(mask_space, H5Sclose);
    }
    closeH5(labels_dataset, H5Dclose);
    closeH5(mask_dataset, H5Dclose);

    closeH5(file, H5Fclose);
    H5Eset_auto2(H5E_DEFAULT, old_error_func, old_error_data);

    if (read_status < 0) {
        std::cerr << "[WARN] Could not read H5 /corners dataset, skipping red outline: "
                  << sidecar << "\n";
        return trace;
    }

    trace.frame_corners.resize(static_cast<size_t>(dims[0]));
    for (size_t frame = 0; frame < trace.frame_corners.size(); ++frame) {
        for (size_t corner = 0; corner < 4; ++corner) {
            size_t base = frame * 8 + corner * 2;
            trace.frame_corners[frame][corner] = cv::Point2f(raw[base], raw[base + 1]);
        }
    }

    if (have_resize_output_size) {
        trace.source_size = cv::Size2f(
            static_cast<float>(resize_output_width),
            static_cast<float>(resize_output_height));
    } else if (attr_width > 0 && attr_height > 0) {
        trace.source_size = cv::Size2f(static_cast<float>(attr_width), static_cast<float>(attr_height));
    }
    if (trace.source_size.width <= 0.0f || trace.source_size.height <= 0.0f) {
        trace.source_size = cv::Size2f(720.0f, 1120.0f);
    }
    trace.available = true;
    std::cout << "[INFO] Loaded H5 corner sidecar: " << sidecar
              << " frames=" << trace.frame_corners.size()
              << " source=" << trace.source_size.width << "x" << trace.source_size.height
              << (have_resize_output_size ? " resize_output=yes" : " resize_output=no")
              << " corrected=" << (trace.corrected ? "yes" : "no")
              << " stylus=" << (trace.stylus_available ? "yes" : "no")
              << " tilt=" << (trace.tilt_available ? "yes" : "no") << "\n";
    return trace;
}

cv::Mat makeBlueMaskFromHsv(const cv::Mat& hsv) {
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(85, 30, 20), cv::Scalar(135, 255, 255), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, morphKernel3());
    return mask;
}

cv::Mat makeGreenMaskFromHsv(const cv::Mat& hsv) {
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(35, 25, 20), cv::Scalar(85, 255, 255), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, morphKernel3());
    return mask;
}

cv::Mat makeBlueMask(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    return makeBlueMaskFromHsv(hsv);
}

cv::Mat makeGreenMask(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    return makeGreenMaskFromHsv(hsv);
}

int envInt(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    try {
        return std::stoi(raw);
    } catch (...) {
        return default_value;
    }
}

std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

void appendShellArg(std::ostringstream* command, const std::string& value) {
    *command << " " << shellQuote(value);
}

void appendRawDepthCommonArgs(
    const Options& options,
    std::ostringstream* command) {
    *command << " --video";
    appendShellArg(command, options.video_path);
    fs::path h5_sidecar = preferredH5SidecarPathForVideo(options.video_path);
    if (!h5_sidecar.empty()) {
        *command << " --h5-sidecar";
        appendShellArg(command, h5_sidecar.string());
    }
    *command << " --outdir";
    appendShellArg(command, options.outdir.string());
    *command << " --encoder";
    appendShellArg(command, options.depth_encoder);
    *command << " --model-version";
    appendShellArg(command, options.depth_model_version);
    *command << " --model-id";
    appendShellArg(command, options.depth_model_id);
    *command << " --video-mode";
    appendShellArg(command, options.depth_video_mode);
    *command << " --streaming-chunk-size " << options.depth_streaming_chunk_size;
    *command << " --streaming-overlap " << options.depth_streaming_overlap;
    *command << " --input-size " << options.depth_input_size;
    if (!options.depth_anything_root.empty()) {
        *command << " --repo-root";
        appendShellArg(command, options.depth_anything_root.string());
    }
    if (!options.depth_checkpoint.empty()) {
        *command << " --checkpoint";
        appendShellArg(command, options.depth_checkpoint.string());
    }
    if (!options.depth_device.empty()) {
        *command << " --device";
        appendShellArg(command, options.depth_device);
    }
    *command << " --gradient-mask-threshold " << options.depth_gradient_mask_threshold;
    *command << " --tilt-gradient-mask-threshold " << options.tilt_gradient_mask_threshold;
    if (options.fast) {
        *command << " --fast";
    }
    if (options.enable_timing) {
        *command << " --timing";
    }
}

void validateRawDepthCommandOptions(const Options& options) {
    if (options.raw_depth_python.empty()) {
        throw std::runtime_error(
            "--raw-depth-python is empty. If you used --raw-depth-python \"$PY\", "
            "define PY first or pass the full Python path.");
    }
    if (!fs::exists(options.raw_depth_script)) {
        throw std::runtime_error(
            "Raw depth helper script does not exist: " +
            options.raw_depth_script.string());
    }

    const fs::path python_path(options.raw_depth_python);
    if (python_path.has_parent_path()) {
        if (!fs::exists(python_path)) {
            throw std::runtime_error(
                "Raw depth Python executable does not exist: " +
                options.raw_depth_python);
        }
        if (access(options.raw_depth_python.c_str(), X_OK) != 0) {
            throw std::runtime_error(
                "Raw depth Python path is not executable: " +
                options.raw_depth_python);
        }
    }
}

void runRawDepthOutput(
    const Options& options,
    const std::vector<int>& target_frames) {
    if (!options.write_raw_depth) {
        return;
    }
    if (options.depth_input_size <= 0) {
        throw std::runtime_error("--depth-input-size must be positive");
    }
    if (options.depth_model_version != "v2" && options.depth_model_version != "v3") {
        throw std::runtime_error("--depth-model-version must be v2 or v3");
    }
    if (options.depth_video_mode != "per-frame" && options.depth_video_mode != "streaming") {
        throw std::runtime_error("--depth-video-mode must be per-frame or streaming");
    }
    if (options.depth_video_mode == "streaming") {
        if (options.depth_model_version != "v3") {
            throw std::runtime_error("DA3 Streaming requires --depth-model-version v3");
        }
        if (!options.frames.empty()) {
            throw std::runtime_error("DA3 Streaming is only available for full-video runs without --frames");
        }
        if (options.depth_streaming_chunk_size < 2 || options.depth_streaming_overlap < 1 ||
                options.depth_streaming_overlap >= options.depth_streaming_chunk_size) {
            throw std::runtime_error("DA3 Streaming requires 0 < overlap < chunk size");
        }
    }
    if (options.depth_encoder != "vits" &&
            options.depth_encoder != "vitb" &&
            options.depth_encoder != "vitl" &&
            options.depth_encoder != "vitg") {
        throw std::runtime_error("--depth-encoder must be one of: vits, vitb, vitl, vitg");
    }
    validateRawDepthCommandOptions(options);

    std::ostringstream command;
    appendShellArg(&command, options.raw_depth_python);
    appendShellArg(&command, options.raw_depth_script.string());
    appendRawDepthCommonArgs(options, &command);
    if (options.final_overlay_only && hasSurfaceTextureOutput(options)) {
        command << " --mask-only";
    }
    if (target_frames.empty() &&
            (options.write_colormatch || options.write_hand_stylus_mask_video)) {
        command << " --mask-output-dir";
        appendShellArg(&command, (options.outdir / "_raw_depth_gradient_mask_frames").string());
    }
    if (!target_frames.empty()) {
        command << " --frames";
        for (int frame_index : target_frames) {
            command << " " << frame_index;
        }
    }

    std::cout << "[INFO] running raw Depth Anything "
              << (options.depth_model_version == "v3" ? "V3 Small" : "V2")
              << " output\n" << std::flush;
    int status = std::system(command.str().c_str());
    if (status != 0) {
        throw std::runtime_error("Raw Depth Anything output failed with status " + std::to_string(status));
    }
}

class RawDepthMaskStream {
public:
    RawDepthMaskStream() = default;
    RawDepthMaskStream(const RawDepthMaskStream&) = delete;
    RawDepthMaskStream& operator=(const RawDepthMaskStream&) = delete;

    ~RawDepthMaskStream() {
        close();
    }

    void open(
        const Options& options,
        cv::Size expected_size,
        const std::vector<int>& target_frames = {}) {
        if (pipe_ != nullptr) {
            close();
        }
        validateRawDepthCommandOptions(options);
        std::ostringstream command;
        appendShellArg(&command, options.raw_depth_python);
        appendShellArg(&command, options.raw_depth_script.string());
        appendRawDepthCommonArgs(options, &command);
        command << " --mask-difference-raw-stream";
        if (!target_frames.empty()) {
            command << " --frames";
            for (int frame_index : target_frames) {
                command << " " << frame_index;
            }
        }

        pipe_ = popen(command.str().c_str(), "r");
        if (pipe_ == nullptr) {
            throw std::runtime_error("Could not start raw depth mask stream");
        }

        char header[256];
        if (std::fgets(header, sizeof(header), pipe_) == nullptr) {
            close();
            throw std::runtime_error("Raw depth mask stream did not provide a header");
        }
        char magic[64] = {0};
        int width = 0;
        int height = 0;
        int frame_count = 0;
        if (std::sscanf(header, "%63s %d %d %d", magic, &width, &height, &frame_count) != 4 ||
                (std::strcmp(magic, "SHADOW_DEPTH_MASK_STREAM") != 0 &&
                 std::strcmp(magic, "SHADOW_DEPTH_MASK_DIFFERENCE_STREAM") != 0) ||
                width <= 0 ||
                height <= 0) {
            close();
            throw std::runtime_error(std::string("Unexpected raw depth mask stream header: ") + header);
        }
        stream_difference_ = std::strcmp(magic, "SHADOW_DEPTH_MASK_DIFFERENCE_STREAM") == 0;
        stream_size_ = cv::Size(width, height);
        frame_count_ = frame_count;
        if (stream_size_ != expected_size) {
            std::cerr << "[WARN] Raw depth mask stream size " << stream_size_.width << "x" << stream_size_.height
                      << " differs from video size " << expected_size.width << "x" << expected_size.height
                      << "; masks will be resized\n";
        }
        std::cout << "[INFO] Raw depth mask stream started frames=" << frame_count_
                  << " size=" << stream_size_.width << "x" << stream_size_.height << "\n";
    }

    bool isOpen() const {
        return pipe_ != nullptr;
    }

    cv::Mat readMask(cv::Size frame_size) {
        if (pipe_ == nullptr || stream_size_.empty()) {
            return cv::Mat::zeros(frame_size, CV_8UC1);
        }
        const size_t bytes = static_cast<size_t>(stream_size_.area());
        mask_buffer_.resize(bytes);
        size_t read = std::fread(mask_buffer_.data(), 1, bytes, pipe_);
        if (read != bytes) {
            std::cerr << "[WARN] Raw depth mask stream ended early at frame " << frames_read_ << "\n";
            close();
            return cv::Mat::zeros(frame_size, CV_8UC1);
        }
        ++frames_read_;
        cv::Mat mask(stream_size_, CV_8UC1, mask_buffer_.data());
        mask.copyTo(mask_frame_);
        last_difference_.release();
        if (stream_difference_) {
            difference_buffer_.resize(bytes);
            size_t difference_read = std::fread(difference_buffer_.data(), 1, bytes, pipe_);
            if (difference_read != bytes) {
                std::cerr << "[WARN] Raw depth difference stream ended early at frame " << frames_read_ << "\n";
                close();
            } else {
                cv::Mat difference(stream_size_, CV_8UC1, difference_buffer_.data());
                difference.copyTo(last_difference_);
                if (last_difference_.size() != frame_size) {
                    cv::resize(last_difference_, last_difference_, frame_size, 0.0, 0.0, cv::INTER_LINEAR);
                }
            }
        }
        if (mask_frame_.size() != frame_size) {
            cv::resize(mask_frame_, mask_frame_, frame_size, 0.0, 0.0, cv::INTER_NEAREST);
        }
        cv::threshold(mask_frame_, mask_frame_, 0, 255, cv::THRESH_BINARY);
        return mask_frame_;
    }

    cv::Mat lastDifference() const {
        return last_difference_;
    }

    void close() {
        if (pipe_ != nullptr) {
            int status = pclose(pipe_);
            pipe_ = nullptr;
            if (status != 0) {
                std::cerr << "[WARN] Raw depth mask stream exited with status " << status << "\n";
            }
        }
    }

private:
    FILE* pipe_ = nullptr;
    cv::Size stream_size_;
    int frame_count_ = 0;
    int frames_read_ = 0;
    bool stream_difference_ = false;
    cv::Mat last_difference_;
    cv::Mat mask_frame_;
    std::vector<uchar> mask_buffer_;
    std::vector<uchar> difference_buffer_;
};

cv::Mat readRawDepthGradientMaskPng(const fs::path& outdir, const std::string& prefix, cv::Size frame_size) {
    fs::path path = outdir / (prefix + "_raw_depth_gradient_mask.png");
    cv::Mat mask = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        std::cerr << "[WARN] Missing raw depth gradient mask: " << path << "\n";
        return cv::Mat::zeros(frame_size, CV_8UC1);
    }
    if (mask.size() != frame_size) {
        cv::resize(mask, mask, frame_size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

cv::Mat readRawDepthGradientDifferencePng(const fs::path& outdir, const std::string& prefix, cv::Size frame_size) {
    fs::path active_path = outdir / (prefix + "_raw_depth_active_gradient_difference.png");
    cv::Mat active = cv::imread(active_path.string(), cv::IMREAD_GRAYSCALE);
    if (!active.empty()) {
        if (active.size() != frame_size) {
            cv::resize(active, active, frame_size, 0.0, 0.0, cv::INTER_LINEAR);
        }
        return active;
    }
    fs::path tilted_path = outdir / (prefix + "_raw_depth_left_tilt_gradient_difference.png");
    cv::Mat tilted = cv::imread(tilted_path.string(), cv::IMREAD_GRAYSCALE);
    if (!tilted.empty()) {
        if (tilted.size() != frame_size) {
            cv::resize(tilted, tilted, frame_size, 0.0, 0.0, cv::INTER_LINEAR);
        }
        return tilted;
    }
    fs::path path = outdir / (prefix + "_raw_depth_gradient_difference.png");
    cv::Mat difference = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (difference.empty()) {
        return cv::Mat();
    }
    if (difference.size() != frame_size) {
        cv::resize(difference, difference, frame_size, 0.0, 0.0, cv::INTER_LINEAR);
    }
    return difference;
}

cv::Mat readRawDepthGradientMaskFramePng(const fs::path& mask_dir, int frame_index, cv::Size frame_size) {
    fs::path path = mask_dir / (framePrefix(frame_index) + "_raw_depth_gradient_mask.png");
    cv::Mat mask = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        return cv::Mat::zeros(frame_size, CV_8UC1);
    }
    if (mask.size() != frame_size) {
        cv::resize(mask, mask, frame_size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

cv::Mat maskFromVideoFrame(const cv::Mat& mask_frame, cv::Size frame_size) {
    cv::Mat gray;
    if (mask_frame.empty()) {
        return cv::Mat::zeros(frame_size, CV_8UC1);
    }
    if (mask_frame.channels() == 1) {
        gray = mask_frame.clone();
    } else {
        cv::cvtColor(mask_frame, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.size() != frame_size) {
        cv::resize(gray, gray, frame_size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY);
    return gray;
}

struct FrameMaskCache {
    cv::Mat hsv;
    cv::Mat blue_mask;
    cv::Mat green_mask;
    cv::Mat valid_mask;
};

cv::Mat getCachedHsv(const cv::Mat& bgr, FrameMaskCache* cache) {
    if (cache == nullptr) {
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        return hsv;
    }
    if (cache->hsv.empty()) {
        cv::cvtColor(bgr, cache->hsv, cv::COLOR_BGR2HSV);
    }
    return cache->hsv;
}

cv::Mat getCachedBlueMask(const cv::Mat& bgr, FrameMaskCache* cache) {
    if (cache == nullptr) {
        return makeBlueMask(bgr);
    }
    if (cache->blue_mask.empty()) {
        cache->blue_mask = makeBlueMaskFromHsv(getCachedHsv(bgr, cache));
    }
    return cache->blue_mask;
}

cv::Mat getCachedGreenMask(const cv::Mat& bgr, FrameMaskCache* cache) {
    if (cache == nullptr) {
        return makeGreenMask(bgr);
    }
    if (cache->green_mask.empty()) {
        cache->green_mask = makeGreenMaskFromHsv(getCachedHsv(bgr, cache));
    }
    return cache->green_mask;
}

cv::Mat getCachedValidMask(const cv::Mat& bgr, FrameMaskCache* cache) {
    if (cache == nullptr) {
        cv::Mat valid_mask;
        cv::bitwise_or(makeBlueMask(bgr), makeGreenMask(bgr), valid_mask);
        return valid_mask;
    }
    if (cache->valid_mask.empty()) {
        cv::bitwise_or(getCachedBlueMask(bgr, cache), getCachedGreenMask(bgr, cache), cache->valid_mask);
    }
    return cache->valid_mask;
}

struct VisualAlignmentScratch {
    cv::Mat warped_blue;
    cv::Mat warped_green;
    cv::Mat warped_label;
    cv::Mat label_blue;
    cv::Mat label_green;
    cv::Mat visible_blue;
    cv::Mat blue_hits;
    cv::Mat visible_green;
    cv::Mat green_hits;
};

struct RenderScratch {
    cv::Mat ref_mask;
    cv::Mat warped_ref;
    cv::Mat warped_mask;
    cv::Mat blended;
};

struct StageTiming {
    double ms = 0.0;
    int calls = 0;
};

struct GeometryTimingStats {
    StageTiming april_tag_detection;
    StageTiming normal_homography;
    StageTiming sparse_soft_fallback;
    StageTiming previous_seed_fallback;
    StageTiming fallback_seed_setup;
    StageTiming fallback_homography_fit;
    StageTiming sheet_edge_refinement;
    StageTiming visual_search;
    StageTiming tag_pattern_shift;
    StageTiming soft_recovery_rejected;
    StageTiming soft_recovery_color_masks;
    StageTiming seed_pattern_support;
    StageTiming seed_color_support;
};

struct GeometryDecisionStats {
    int seeded_fast_accept_total = 0;
    int seeded_fast_accept_ok = 0;
    int seeded_fast_accept_ok_3plus = 0;
    int seeded_fast_reject_not_ok = 0;
    int seeded_fast_reject_not_previous_seed = 0;
    int seeded_fast_reject_soft_count = 0;
    int seeded_fast_reject_edge = 0;
    int seeded_fast_reject_score = 0;
    int chooser_total = 0;
    int chooser_direct = 0;
    int chooser_seeded = 0;
    int chooser_seeded_direct_bad = 0;
    int chooser_direct_seeded_bad = 0;
    int chooser_seeded_weak_direct = 0;
    int chooser_seeded_soft_count = 0;
    int chooser_seeded_spread = 0;
    int chooser_seeded_score = 0;
};

struct RenderTimingStats {
    StageTiming raw_frame_clone;
    StageTiming aligned_ref_warp;
    StageTiming aligned_ref_clone;
    StageTiming tag_color_match;
    StageTiming depth_mask_read;
    StageTiming stylus_mask_refinement;
    StageTiming depth_mask_prepare;
    StageTiming empty_mask_fallback;
    StageTiming foreground_keep_mask;
    StageTiming masked_difference;
    StageTiming clean_difference;
    StageTiming surface_texture;
    StageTiming shadow_factor;
    StageTiming apply_shadow_factor;
    StageTiming debug_artifacts;
    StageTiming texture_variant_surface;
    StageTiming texture_variant_apply_shadow;
};

GeometryTimingStats g_geometry_timing;
GeometryDecisionStats g_geometry_decisions;
RenderTimingStats g_render_timing;
bool g_geometry_timing_enabled = false;

struct ScopedStageTimer {
    StageTiming* stage = nullptr;
    std::chrono::steady_clock::time_point start;

    explicit ScopedStageTimer(StageTiming* stage_in)
        : stage(g_geometry_timing_enabled ? stage_in : nullptr),
          start(std::chrono::steady_clock::now()) {}

    ~ScopedStageTimer() {
        if (stage == nullptr) {
            return;
        }
        stage->ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        ++stage->calls;
    }
};

struct TimingStats {
    using Stage = StageTiming;

    Stage direct_geometry;
    Stage direct_fallback_geometry;
    Stage seeded_prefilter;
    Stage propagation;
    Stage seeded_geometry;
    Stage seeded_fallback_geometry;
    Stage seeded_fast_accept_check;
    Stage choose_geometry;
    Stage render;
    Stage write;
    Stage overlay_render;
    Stage colormatch_artifacts;
    Stage texture_variants;
    Stage png_write;
    Stage video_write;

    static double elapsedMs(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }

    static void add(Stage* stage, double ms) {
        if (stage == nullptr) {
            return;
        }
        stage->ms += ms;
        ++stage->calls;
    }

    void print(const char* label) const {
        auto printStage = [](const char* name, const Stage& stage) {
            if (stage.calls == 0) {
                return;
            }
            std::ostringstream line;
            line << "[TIMING] " << name
                 << " calls=" << stage.calls
                 << " total_ms=" << stage.ms
                 << " avg_ms=" << (stage.ms / static_cast<double>(stage.calls))
                 << "\n";
            std::cerr << line.str();
        };
        auto printCount = [](const char* name, int count) {
            if (count == 0) {
                return;
            }
            std::cerr << "[TIMING] " << name << " count=" << count << "\n";
        };
        std::cerr << "[TIMING] " << label << "\n";
        printStage("direct_geometry", direct_geometry);
        printStage("direct_fallback_geometry", direct_fallback_geometry);
        printStage("seeded_prefilter", seeded_prefilter);
        printStage("propagation", propagation);
        printStage("seeded_geometry", seeded_geometry);
        printStage("seeded_fallback_geometry", seeded_fallback_geometry);
        printStage("seeded_fast_accept_check", seeded_fast_accept_check);
        printStage("choose_geometry", choose_geometry);
        printStage("render", render);
        printStage("write", write);
        printStage("overlay_render", overlay_render);
        printStage("colormatch_artifacts", colormatch_artifacts);
        printStage("render.raw_frame_clone", g_render_timing.raw_frame_clone);
        printStage("render.aligned_ref_warp", g_render_timing.aligned_ref_warp);
        printStage("render.aligned_ref_clone", g_render_timing.aligned_ref_clone);
        printStage("render.tag_color_match", g_render_timing.tag_color_match);
        printStage("render.depth_mask_read", g_render_timing.depth_mask_read);
        printStage("render.stylus_mask_refinement", g_render_timing.stylus_mask_refinement);
        printStage("render.depth_mask_prepare", g_render_timing.depth_mask_prepare);
        printStage("render.empty_mask_fallback", g_render_timing.empty_mask_fallback);
        printStage("render.foreground_keep_mask", g_render_timing.foreground_keep_mask);
        printStage("render.masked_difference", g_render_timing.masked_difference);
        printStage("render.clean_difference", g_render_timing.clean_difference);
        printStage("render.surface_texture", g_render_timing.surface_texture);
        printStage("render.shadow_factor", g_render_timing.shadow_factor);
        printStage("render.apply_shadow_factor", g_render_timing.apply_shadow_factor);
        printStage("render.debug_artifacts", g_render_timing.debug_artifacts);
        printStage("render.texture_variant_surface", g_render_timing.texture_variant_surface);
        printStage("render.texture_variant_apply_shadow", g_render_timing.texture_variant_apply_shadow);
        printStage("texture_variants", texture_variants);
        printStage("png_write", png_write);
        printStage("video_write", video_write);
        printStage("geometry.april_tag_detection", g_geometry_timing.april_tag_detection);
        printStage("geometry.normal_3plus_homography", g_geometry_timing.normal_homography);
        printStage("geometry.sparse_soft_fallback", g_geometry_timing.sparse_soft_fallback);
        printStage("geometry.previous_seed_fallback", g_geometry_timing.previous_seed_fallback);
        printStage("geometry.fallback_seed_setup", g_geometry_timing.fallback_seed_setup);
        printStage("geometry.fallback_homography_fit", g_geometry_timing.fallback_homography_fit);
        printStage("geometry.sheet_edge_refinement", g_geometry_timing.sheet_edge_refinement);
        printStage("geometry.visual_search", g_geometry_timing.visual_search);
        printStage("geometry.tag_pattern_shift", g_geometry_timing.tag_pattern_shift);
        printStage("geometry.soft_recovery_rejected", g_geometry_timing.soft_recovery_rejected);
        printStage("geometry.soft_recovery_color_masks", g_geometry_timing.soft_recovery_color_masks);
        printStage("geometry.seed_pattern_support", g_geometry_timing.seed_pattern_support);
        printStage("geometry.seed_color_support", g_geometry_timing.seed_color_support);
        printCount("decision.seeded_fast_accept.total", g_geometry_decisions.seeded_fast_accept_total);
        printCount("decision.seeded_fast_accept.ok", g_geometry_decisions.seeded_fast_accept_ok);
        printCount("decision.seeded_fast_accept.ok_3plus", g_geometry_decisions.seeded_fast_accept_ok_3plus);
        printCount("decision.seeded_fast_reject.not_ok", g_geometry_decisions.seeded_fast_reject_not_ok);
        printCount("decision.seeded_fast_reject.not_previous_seed", g_geometry_decisions.seeded_fast_reject_not_previous_seed);
        printCount("decision.seeded_fast_reject.soft_count", g_geometry_decisions.seeded_fast_reject_soft_count);
        printCount("decision.seeded_fast_reject.edge", g_geometry_decisions.seeded_fast_reject_edge);
        printCount("decision.seeded_fast_reject.score", g_geometry_decisions.seeded_fast_reject_score);
        printCount("decision.chooser.total", g_geometry_decisions.chooser_total);
        printCount("decision.chooser.direct", g_geometry_decisions.chooser_direct);
        printCount("decision.chooser.seeded", g_geometry_decisions.chooser_seeded);
        printCount("decision.chooser.seeded_direct_bad", g_geometry_decisions.chooser_seeded_direct_bad);
        printCount("decision.chooser.direct_seeded_bad", g_geometry_decisions.chooser_direct_seeded_bad);
        printCount("decision.chooser.seeded_weak_direct", g_geometry_decisions.chooser_seeded_weak_direct);
        printCount("decision.chooser.seeded_soft_count", g_geometry_decisions.chooser_seeded_soft_count);
        printCount("decision.chooser.seeded_spread", g_geometry_decisions.chooser_seeded_spread);
        printCount("decision.chooser.seeded_score", g_geometry_decisions.chooser_seeded_score);
    }
};

ReferenceTagPattern buildReferenceTagPattern(
    const cv::Mat& reference_bgr,
    const std::vector<cv::Point2f>& ref_quad) {
    ReferenceTagPattern pattern;
    if (reference_bgr.empty() || ref_quad.size() != 4) {
        return pattern;
    }

    std::vector<cv::Point2f> canonical = {
        {0.0f, 0.0f},
        {static_cast<float>(kTagPatternPatchSize - 1), 0.0f},
        {static_cast<float>(kTagPatternPatchSize - 1), static_cast<float>(kTagPatternPatchSize - 1)},
        {0.0f, static_cast<float>(kTagPatternPatchSize - 1)},
    };
    cv::Mat H_ref = cv::getPerspectiveTransform(ref_quad, canonical);
    if (H_ref.empty()) {
        return pattern;
    }

    cv::Mat ref_patch;
    cv::warpPerspective(
        reference_bgr,
        ref_patch,
        H_ref,
        cv::Size(kTagPatternPatchSize, kTagPatternPatchSize),
        cv::INTER_LINEAR,
        cv::BORDER_REPLICATE);

    cv::Mat ref_hsv;
    cv::cvtColor(ref_patch, ref_hsv, cv::COLOR_BGR2HSV);
    pattern.blue_mask = makeBlueMaskFromHsv(ref_hsv);
    pattern.green_mask = makeGreenMaskFromHsv(ref_hsv);
    cv::bitwise_or(pattern.blue_mask, pattern.green_mask, pattern.valid_mask);
    pattern.valid_pixels = cv::countNonZero(pattern.valid_mask);

    // AprilTag 36h11 has a 6x6 payload inside its one-cell border. Record the
    // expected blue/green label and reference-image center for each payload
    // cell so partially visible tags can validate a whole-sheet homography
    // without trusting an occluded tag's four corners.
    constexpr int grid_size = 8;
    constexpr int payload_size = 6;
    constexpr int cell_size = kTagPatternPatchSize / grid_size;
    std::vector<cv::Point2f> canonical_centers;
    canonical_centers.reserve(payload_size * payload_size);
    int cell_index = 0;
    for (int row = 1; row <= payload_size; ++row) {
        for (int col = 1; col <= payload_size; ++col) {
            const cv::Rect sample(
                col * cell_size + 2,
                row * cell_size + 2,
                cell_size - 4,
                cell_size - 4);
            const int blue = cv::countNonZero(pattern.blue_mask(sample));
            const int green = cv::countNonZero(pattern.green_mask(sample));
            pattern.inner_cell_labels[static_cast<size_t>(cell_index)] =
                blue > green ? 1 : (green > blue ? 2 : 0);
            canonical_centers.emplace_back(
                (static_cast<float>(col) + 0.5f) * cell_size,
                (static_cast<float>(row) + 0.5f) * cell_size);
            ++cell_index;
        }
    }
    cv::Mat H_canonical_to_ref = H_ref.inv();
    std::vector<cv::Point2f> ref_centers;
    cv::perspectiveTransform(canonical_centers, ref_centers, H_canonical_to_ref);
    pattern.inner_cell_count = std::min(
        static_cast<int>(ref_centers.size()),
        payload_size * payload_size);
    for (int i = 0; i < pattern.inner_cell_count; ++i) {
        pattern.ref_inner_cell_centers[static_cast<size_t>(i)] =
            ref_centers[static_cast<size_t>(i)];
    }
    return pattern;
}

double quadArea(const std::vector<cv::Point2f>& quad) {
    if (quad.size() != 4) {
        return 0.0;
    }
    return std::abs(cv::contourArea(quad));
}

cv::Point2f quadCenter(const std::vector<cv::Point2f>& quad) {
    cv::Point2f center(0.0f, 0.0f);
    if (quad.empty()) {
        return center;
    }
    for (const cv::Point2f& point : quad) {
        center += point;
    }
    center.x /= static_cast<float>(quad.size());
    center.y /= static_cast<float>(quad.size());
    return center;
}

std::vector<cv::Point2f> orderCornersTLTRBRBL(const std::vector<cv::Point2f>& quad) {
    if (quad.size() != 4) {
        return quad;
    }

    std::vector<cv::Point2f> ordered(4);
    std::vector<float> sum(4), diff(4);
    for (int i = 0; i < 4; ++i) {
        sum[i] = quad[i].x + quad[i].y;
        diff[i] = quad[i].x - quad[i].y;
    }
    ordered[0] = quad[static_cast<size_t>(std::min_element(sum.begin(), sum.end()) - sum.begin())];
    ordered[2] = quad[static_cast<size_t>(std::max_element(sum.begin(), sum.end()) - sum.begin())];
    ordered[1] = quad[static_cast<size_t>(std::max_element(diff.begin(), diff.end()) - diff.begin())];
    ordered[3] = quad[static_cast<size_t>(std::min_element(diff.begin(), diff.end()) - diff.begin())];
    return ordered;
}

std::vector<cv::Point2f> projectQuad(const cv::Mat& H, const std::vector<cv::Point2f>& quad) {
    std::vector<cv::Point2f> projected;
    if (H.empty() || quad.size() != 4) {
        return projected;
    }
    cv::perspectiveTransform(quad, projected, H);
    return projected;
}

std::vector<cv::Point2f> referenceImageCorners(cv::Size size) {
    return {
        {0.0f, 0.0f},
        {static_cast<float>(size.width - 1), 0.0f},
        {static_cast<float>(size.width - 1), static_cast<float>(size.height - 1)},
        {0.0f, static_cast<float>(size.height - 1)},
    };
}

std::pair<cv::Mat, std::vector<cv::Point2f>> makeProjectedBoardMask(
    const cv::Mat& H_ref_to_frame,
    cv::Size reference_size,
    cv::Size frame_size) {
    std::vector<cv::Point2f> projected = projectQuad(
        H_ref_to_frame,
        referenceImageCorners(reference_size));
    cv::Mat mask(frame_size, CV_8UC1, cv::Scalar(0));
    if (projected.size() != 4) {
        return {mask, projected};
    }

    std::vector<cv::Point> poly;
    poly.reserve(projected.size());
    for (const cv::Point2f& point : projected) {
        poly.emplace_back(cvRound(point.x), cvRound(point.y));
    }
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return {mask, projected};
}

void attachProjectedBoardMask(
    GeometryResult* result,
    cv::Size reference_size,
    cv::Size frame_size) {
    if (result == nullptr || !result->ok || result->H_ref_to_frame.empty()) {
        return;
    }
    auto mask_and_quad = makeProjectedBoardMask(
        result->H_ref_to_frame,
        reference_size,
        frame_size);
    result->board_mask = std::move(mask_and_quad.first);
    result->board_quad = std::move(mask_and_quad.second);
}

bool h5FrameCornersInFrame(
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size frame_size,
    std::vector<cv::Point2f>* frame_corners) {
    if (frame_corners == nullptr ||
            h5_corners == nullptr ||
            !h5_corners->available ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->frame_corners.size() ||
            h5_corners->source_size.width <= 0.0f ||
            h5_corners->source_size.height <= 0.0f) {
        return false;
    }
    const std::array<cv::Point2f, 4>& raw_corners =
        h5_corners->frame_corners[static_cast<size_t>(frame_index)];
    if (!h5CornerArrayUsable(raw_corners)) {
        return false;
    }
    const float scale_x = static_cast<float>(frame_size.width) / h5_corners->source_size.width;
    const float scale_y = static_cast<float>(frame_size.height) / h5_corners->source_size.height;
    frame_corners->clear();
    frame_corners->reserve(4);
    for (const cv::Point2f& corner : raw_corners) {
        if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
            frame_corners->clear();
            return false;
        }
        frame_corners->emplace_back(corner.x * scale_x, corner.y * scale_y);
    }
    return std::abs(cv::contourArea(*frame_corners)) > 100.0;
}

GeometryResult geometryFromH5Corners(
    int frame_index,
    const ReferenceModel& reference,
    const H5CornerTrace* h5_corners,
    cv::Size frame_size) {
    GeometryResult result;
    result.frame_index = frame_index;
    std::vector<cv::Point2f> frame_corners;
    if (!h5FrameCornersInFrame(h5_corners, frame_index, frame_size, &frame_corners)) {
        return result;
    }
    std::vector<cv::Point2f> ref_corners = referenceImageCorners(reference.reference_bgr.size());
    result.H_ref_to_frame = cv::getPerspectiveTransform(ref_corners, frame_corners);
    result.ok = !result.H_ref_to_frame.empty();
    result.used_corrected_h5_corners = result.ok;
    result.board_quad = frame_corners;
    if (result.ok) {
        attachProjectedBoardMask(&result, reference.reference_bgr.size(), frame_size);
        result.board_quad = frame_corners;
    }
    return result;
}

std::vector<cv::Point2f> orderCandidateLikeExpected(
    const std::vector<cv::Point2f>& candidate,
    const std::vector<cv::Point2f>& expected) {
    if (candidate.size() != 4 || expected.size() != 4) {
        return candidate;
    }

    std::array<int, 4> perm = {0, 1, 2, 3};
    std::array<int, 4> best_perm = perm;
    double best_cost = std::numeric_limits<double>::max();
    do {
        double cost = 0.0;
        for (int i = 0; i < 4; ++i) {
            cost += cv::norm(candidate[static_cast<size_t>(perm[static_cast<size_t>(i)])] - expected[static_cast<size_t>(i)]);
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    std::vector<cv::Point2f> ordered(4);
    for (int i = 0; i < 4; ++i) {
        ordered[static_cast<size_t>(i)] = candidate[static_cast<size_t>(best_perm[static_cast<size_t>(i)])];
    }
    return ordered;
}

double projectedTagSize(const std::vector<cv::Point2f>& quad) {
    if (quad.size() != 4) {
        return 0.0;
    }
    double e0 = cv::norm(quad[1] - quad[0]);
    double e1 = cv::norm(quad[2] - quad[1]);
    double e2 = cv::norm(quad[3] - quad[2]);
    double e3 = cv::norm(quad[0] - quad[3]);
    return 0.25 * (e0 + e1 + e2 + e3);
}

bool quadInFrame(const std::vector<cv::Point2f>& quad, cv::Size frame_size) {
    if (quad.size() != 4) {
        return false;
    }
    cv::Rect2f frame_bounds(0.0f, 0.0f, static_cast<float>(frame_size.width), static_cast<float>(frame_size.height));
    cv::Rect2f bounds = cv::boundingRect(quad);
    return (bounds & frame_bounds).area() > 0.0f;
}

double projectedQuadMaxJump(const cv::Mat& H_a, const cv::Mat& H_b, cv::Size reference_size) {
    if (H_a.empty() || H_b.empty()) {
        return std::numeric_limits<double>::max();
    }
    std::vector<cv::Point2f> ref_corners = referenceImageCorners(reference_size);
    std::vector<cv::Point2f> a = projectQuad(H_a, ref_corners);
    std::vector<cv::Point2f> b = projectQuad(H_b, ref_corners);
    if (a.size() != 4 || b.size() != 4) {
        return std::numeric_limits<double>::max();
    }
    double max_jump = 0.0;
    for (int i = 0; i < 4; ++i) {
        max_jump = std::max(max_jump, cv::norm(a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]));
    }
    return max_jump;
}

double meanHardTagReprojectionError(
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const std::vector<int>& hard_tag_ids,
    const cv::Mat& H) {
    if (H.empty() || hard_tag_ids.empty()) {
        return std::numeric_limits<double>::max();
    }

    double total_error = 0.0;
    int point_count = 0;
    for (int id : hard_tag_ids) {
        auto ref_it = reference.reference_tags.detections.find(id);
        auto frame_it = frame_tags.detections.find(id);
        if (ref_it == reference.reference_tags.detections.end() ||
                frame_it == frame_tags.detections.end()) {
            continue;
        }
        std::vector<cv::Point2f> projected = projectQuad(H, ref_it->second);
        if (projected.size() != 4 || frame_it->second.size() != 4) {
            continue;
        }
        for (int i = 0; i < 4; ++i) {
            total_error += cv::norm(projected[static_cast<size_t>(i)] - frame_it->second[static_cast<size_t>(i)]);
            ++point_count;
        }
    }
    return point_count > 0
        ? total_error / static_cast<double>(point_count)
        : std::numeric_limits<double>::max();
}

double meanCorrespondenceReprojectionError(
    const std::vector<cv::Point2f>& src_pts,
    const std::vector<cv::Point2f>& dst_pts,
    const cv::Mat& H) {
    if (H.empty() || src_pts.size() < 4 || src_pts.size() != dst_pts.size()) {
        return std::numeric_limits<double>::max();
    }

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(src_pts, projected, H);
    if (projected.size() != dst_pts.size()) {
        return std::numeric_limits<double>::max();
    }

    double total_error = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        total_error += cv::norm(projected[i] - dst_pts[i]);
    }
    return total_error / static_cast<double>(projected.size());
}

bool maskAt(const cv::Mat& mask, const cv::Point2f& point) {
    int x = cvRound(point.x);
    int y = cvRound(point.y);
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) {
        return false;
    }
    return mask.at<uchar>(y, x) != 0;
}

struct SheetEdgeAlignment {
    bool usable = false;
    bool ok = true;
    int usable_edges = 0;
    double max_abs_offset = 0.0;
};

struct TagPatternShift {
    bool ok = false;
    cv::Point2f delta;
    int support = 0;
};

struct VisualAlignmentScore {
    double score = -std::numeric_limits<double>::max();
    double blue_score = 0.0;
    double green_score = 0.0;
    double cell_pattern_score = 0.0;
    double edge_penalty = 0.0;
    int visible_blue = 0;
    int visible_green = 0;
    int visible_cells = 0;
    int matched_cells = 0;
    int scored_tags = 0;
};

cv::Mat translateHomography(const cv::Mat& H, const cv::Point2f& delta) {
    if (H.empty()) {
        return H;
    }
    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0, 2) = static_cast<double>(delta.x);
    T.at<double>(1, 2) = static_cast<double>(delta.y);
    return T * H;
}

cv::Mat rotateScaleTranslateHomography(
    const cv::Mat& H,
    const cv::Point2f& center,
    double angle_degrees,
    const cv::Point2f& delta) {
    if (H.empty()) {
        return H;
    }
    cv::Mat A = cv::getRotationMatrix2D(center, angle_degrees, 1.0);
    A.at<double>(0, 2) += static_cast<double>(delta.x);
    A.at<double>(1, 2) += static_cast<double>(delta.y);
    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    A.copyTo(T(cv::Rect(0, 0, 3, 2)));
    return T * H;
}

double medianValue(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid - 1), values.end());
        median = 0.5 * (median + values[mid - 1]);
    }
    return median;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    p = std::clamp(p, 0.0, 1.0);
    double pos = p * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    std::nth_element(values.begin(), values.begin() + static_cast<long>(lo), values.end());
    double lo_value = values[lo];
    if (hi == lo) {
        return lo_value;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<long>(hi), values.end());
    double hi_value = values[hi];
    return lo_value + (hi_value - lo_value) * (pos - static_cast<double>(lo));
}

double hueDistanceOpenCv(double a, double b) {
    double d = std::abs(a - b);
    return std::min(d, 180.0 - d);
}

cv::Mat warpTagPatch(const cv::Mat& image_bgr, const std::vector<cv::Point2f>& tag_corners, int size = 96) {
    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(size - 1), 0.0f},
        {static_cast<float>(size - 1), static_cast<float>(size - 1)},
        {0.0f, static_cast<float>(size - 1)},
    };
    cv::Mat H = cv::getPerspectiveTransform(tag_corners, dst);
    cv::Mat patch;
    cv::warpPerspective(image_bgr, patch, H, cv::Size(size, size));
    return patch;
}

void buildReferenceColorMasks(
    const cv::Mat& image_bgr,
    const cv::Mat& valid_mask,
    cv::Mat* green_mask,
    cv::Mat* blue_mask) {
    cv::Mat hsv;
    cv::cvtColor(image_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat valid = valid_mask.empty()
        ? cv::Mat(image_bgr.size(), CV_8UC1, cv::Scalar(255))
        : valid_mask.clone();
    cv::threshold(valid, valid, 0, 255, cv::THRESH_BINARY);
    cv::inRange(hsv, cv::Scalar(35, 25, 25), cv::Scalar(95, 255, 255), *green_mask);
    cv::inRange(hsv, cv::Scalar(96, 25, 25), cv::Scalar(150, 255, 255), *blue_mask);
    cv::bitwise_and(*green_mask, valid, *green_mask);
    cv::bitwise_and(*blue_mask, valid, *blue_mask);
    cv::erode(*green_mask, *green_mask, morphKernel5(), cv::Point(-1, -1), 1);
    cv::dilate(*blue_mask, *blue_mask, morphKernel5(), cv::Point(-1, -1), 1);
    cv::bitwise_and(*green_mask, valid, *green_mask);
    cv::bitwise_and(*blue_mask, valid, *blue_mask);
    green_mask->setTo(0, *blue_mask);
}

void buildStrictReferenceColorMasks(
    const cv::Mat& image_bgr,
    const cv::Mat& valid_mask,
    cv::Mat* green_mask,
    cv::Mat* blue_mask) {
    cv::Mat hsv;
    cv::cvtColor(image_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat valid = valid_mask.empty()
        ? cv::Mat(image_bgr.size(), CV_8UC1, cv::Scalar(255))
        : valid_mask.clone();
    cv::threshold(valid, valid, 0, 255, cv::THRESH_BINARY);

    cv::inRange(hsv, cv::Scalar(35, 25, 25), cv::Scalar(95, 255, 255), *green_mask);
    cv::inRange(hsv, cv::Scalar(96, 25, 25), cv::Scalar(150, 255, 255), *blue_mask);
    cv::bitwise_and(*green_mask, valid, *green_mask);
    cv::bitwise_and(*blue_mask, valid, *blue_mask);
    blue_mask->setTo(0, *green_mask);
}

void buildTagColorMasks(const cv::Mat& ref_tag_patch_bgr, cv::Mat* green_mask, cv::Mat* blue_mask) {
    cv::Mat inner(ref_tag_patch_bgr.size(), CV_8UC1, cv::Scalar(0));
    const int margin = std::max(4, ref_tag_patch_bgr.cols / 8);
    inner(cv::Rect(
        margin,
        margin,
        std::max(1, ref_tag_patch_bgr.cols - 2 * margin),
        std::max(1, ref_tag_patch_bgr.rows - 2 * margin))).setTo(255);
    buildReferenceColorMasks(ref_tag_patch_bgr, inner, green_mask, blue_mask);
    cv::erode(*green_mask, *green_mask, morphKernel3(), cv::Point(-1, -1), 1);
    cv::erode(*blue_mask, *blue_mask, morphKernel3(), cv::Point(-1, -1), 1);
    cv::bitwise_and(*green_mask, inner, *green_mask);
    cv::bitwise_and(*blue_mask, inner, *blue_mask);
}

void buildStrictTagColorMasks(const cv::Mat& ref_tag_patch_bgr, cv::Mat* green_mask, cv::Mat* blue_mask) {
    cv::Mat inner(ref_tag_patch_bgr.size(), CV_8UC1, cv::Scalar(0));
    const int margin = std::max(4, ref_tag_patch_bgr.cols / 8);
    inner(cv::Rect(
        margin,
        margin,
        std::max(1, ref_tag_patch_bgr.cols - 2 * margin),
        std::max(1, ref_tag_patch_bgr.rows - 2 * margin))).setTo(255);
    buildStrictReferenceColorMasks(ref_tag_patch_bgr, inner, green_mask, blue_mask);
    cv::bitwise_and(*green_mask, inner, *green_mask);
    cv::bitwise_and(*blue_mask, inner, *blue_mask);
}

cv::Vec3b medianColor(const cv::Mat& image_bgr, const cv::Mat& mask) {
    std::vector<double> b;
    std::vector<double> g;
    std::vector<double> r;
    int count = cv::countNonZero(mask);
    b.reserve(count);
    g.reserve(count);
    r.reserve(count);
    for (int y = 0; y < image_bgr.rows; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        const cv::Vec3b* image_row = image_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image_bgr.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            b.push_back(image_row[x][0]);
            g.push_back(image_row[x][1]);
            r.push_back(image_row[x][2]);
        }
    }
    return cv::Vec3b(
        cv::saturate_cast<uchar>(medianValue(b)),
        cv::saturate_cast<uchar>(medianValue(g)),
        cv::saturate_cast<uchar>(medianValue(r)));
}

bool plausibleCellDonorColor(const cv::Vec3b& color, bool green) {
    const int b = color[0];
    const int g = color[1];
    const int r = color[2];
    const int maxc = std::max({b, g, r});
    const int minc = std::min({b, g, r});
    const int chroma = maxc - minc;
    const int min_donor_max = green ?
        std::clamp(envInt("SHADOW_COLORMATCH_GREEN_MIN_DONOR_MAX", 112), 40, 180) :
        std::clamp(envInt("SHADOW_COLORMATCH_BLUE_MIN_DONOR_MAX", 100), 40, 180);
    if (maxc < min_donor_max || chroma < 18) {
        return false;
    }
    if (green) {
        return g >= r + 6 && g >= b - 30;
    }
    return b >= r + 14 && b >= g - 24;
}

bool selectTopLeftCommonTag(
    const DetectionResult& ref_tags,
    const DetectionResult& frame_tags,
    int* selected_id) {
    bool found = false;
    double best_y = 0.0;
    double best_x = 0.0;
    int best_id = -1;
    for (const auto& [id, ref_corners] : ref_tags.detections) {
        (void)ref_corners;
        auto frame_it = frame_tags.detections.find(id);
        if (frame_it == frame_tags.detections.end() || frame_it->second.empty()) {
            continue;
        }
        cv::Point2f center = quadCenter(frame_it->second);
        if (!found || center.y < best_y || (center.y == best_y && center.x < best_x)) {
            found = true;
            best_y = center.y;
            best_x = center.x;
            best_id = id;
        }
    }
    if (!found) {
        return false;
    }
    *selected_id = best_id;
    return true;
}

cv::Mat makeVisibleColorSampleMask(const cv::Mat& frame_patch_bgr, const cv::Mat& class_mask, bool green_class) {
    cv::Mat hsv;
    cv::cvtColor(frame_patch_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat hsv_mask;
    if (green_class) {
        cv::inRange(hsv, cv::Scalar(30, 25, 25), cv::Scalar(105, 255, 255), hsv_mask);
    } else {
        cv::inRange(hsv, cv::Scalar(80, 35, 120), cv::Scalar(165, 255, 255), hsv_mask);
    }
    cv::Mat dominance(frame_patch_bgr.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < frame_patch_bgr.rows; ++y) {
        const cv::Vec3b* row = frame_patch_bgr.ptr<cv::Vec3b>(y);
        uchar* out = dominance.ptr<uchar>(y);
        for (int x = 0; x < frame_patch_bgr.cols; ++x) {
            int b = row[x][0];
            int g = row[x][1];
            int r = row[x][2];
            int maxc = std::max({b, g, r});
            int minc = std::min({b, g, r});
            int chroma = maxc - minc;
            if (green_class) {
                if (chroma >= 18 && g >= r + 6 && g >= b - 18) {
                    out[x] = 255;
                }
            } else if (maxc >= 155 && chroma >= 45 && b >= g + 30 && b >= r + 35) {
                out[x] = 255;
            }
        }
    }
    cv::Mat visible_mask;
    cv::bitwise_and(hsv_mask, dominance, visible_mask);
    cv::Mat sample_mask;
    cv::bitwise_and(class_mask, visible_mask, sample_mask);
    return sample_mask;
}

bool sampleTagClassColors(
    const cv::Mat& ref_bgr,
    const cv::Mat& frame_bgr,
    const std::vector<cv::Point2f>& ref_corners,
    const std::vector<cv::Point2f>& frame_corners,
    cv::Vec3b* sampled_green,
    cv::Vec3b* sampled_blue,
    bool* have_green,
    bool* have_blue) {
    const cv::Mat ref_patch = warpTagPatch(ref_bgr, ref_corners);
    const cv::Mat frame_patch = warpTagPatch(frame_bgr, frame_corners);
    cv::Mat tag_green_mask;
    cv::Mat tag_blue_mask;
    buildStrictTagColorMasks(ref_patch, &tag_green_mask, &tag_blue_mask);
    cv::Mat green_sample_mask = makeVisibleColorSampleMask(frame_patch, tag_green_mask, true);
    cv::Mat blue_sample_mask = makeVisibleColorSampleMask(frame_patch, tag_blue_mask, false);
    *have_green = cv::countNonZero(green_sample_mask) >= 20;
    *have_blue = cv::countNonZero(blue_sample_mask) >= 20;
    if (!*have_green && !*have_blue) {
        return false;
    }
    if (*have_green) {
        *sampled_green = medianColor(frame_patch, green_sample_mask);
    }
    if (*have_blue) {
        *sampled_blue = medianColor(frame_patch, blue_sample_mask);
    }
    return true;
}

bool applyTopLeftTagColorMatch(
    const cv::Mat& ref_bgr,
    const cv::Mat& frame_bgr,
    const DetectionResult& ref_tags,
    const DetectionResult& frame_tags,
    const cv::Mat& board_mask,
    cv::Mat* warped_ref_bgr) {
    int selected_id = -1;
    if (!selectTopLeftCommonTag(ref_tags, frame_tags, &selected_id)) {
        return false;
    }
    cv::Vec3b sampled_green(0, 0, 0);
    cv::Vec3b sampled_blue(0, 0, 0);
    bool have_green = false;
    bool have_blue = false;
    if (!sampleTagClassColors(
            ref_bgr,
            frame_bgr,
            ref_tags.detections.at(selected_id),
            frame_tags.detections.at(selected_id),
            &sampled_green,
            &sampled_blue,
            &have_green,
            &have_blue)) {
        return false;
    }
    cv::Mat warped_green_mask;
    cv::Mat warped_blue_mask;
    buildStrictReferenceColorMasks(*warped_ref_bgr, board_mask, &warped_green_mask, &warped_blue_mask);
    if (have_green) {
        warped_ref_bgr->setTo(sampled_green, warped_green_mask);
    }
    if (have_blue) {
        warped_ref_bgr->setTo(sampled_blue, warped_blue_mask);
    }
    return true;
}

std::vector<cv::Point> makeCellPoly(
    const std::vector<cv::Point2f>& tag_corners,
    int cells,
    int cx,
    int cy) {
    auto interp = [&](float x, float y) {
        const cv::Point2f top = tag_corners[0] + (tag_corners[1] - tag_corners[0]) * x;
        const cv::Point2f bottom = tag_corners[3] + (tag_corners[2] - tag_corners[3]) * x;
        return top + (bottom - top) * y;
    };
    const float x0 = static_cast<float>(cx) / static_cast<float>(cells);
    const float x1 = static_cast<float>(cx + 1) / static_cast<float>(cells);
    const float y0 = static_cast<float>(cy) / static_cast<float>(cells);
    const float y1 = static_cast<float>(cy + 1) / static_cast<float>(cells);
    auto roundPoint = [](const cv::Point2f& point) {
        return cv::Point(cvRound(point.x), cvRound(point.y));
    };
    return {
        roundPoint(interp(x0, y0)),
        roundPoint(interp(x1, y0)),
        roundPoint(interp(x1, y1)),
        roundPoint(interp(x0, y1)),
    };
}

bool makeLocalCellMask(
    const std::vector<cv::Point>& poly,
    cv::Size image_size,
    cv::Rect* roi,
    cv::Mat* local_mask,
    std::vector<cv::Point>* local_poly = nullptr) {
    if (poly.size() != 4 || roi == nullptr || local_mask == nullptr) {
        return false;
    }
    cv::Rect bounds = cv::boundingRect(poly) & cv::Rect(0, 0, image_size.width, image_size.height);
    if (bounds.empty()) {
        return false;
    }
    std::vector<cv::Point> shifted;
    shifted.reserve(poly.size());
    for (const cv::Point& point : poly) {
        shifted.emplace_back(point.x - bounds.x, point.y - bounds.y);
    }
    *roi = bounds;
    *local_mask = cv::Mat(bounds.size(), CV_8UC1, cv::Scalar(0));
    cv::fillConvexPoly(*local_mask, shifted, cv::Scalar(255));
    if (local_poly != nullptr) {
        *local_poly = std::move(shifted);
    }
    return cv::countNonZero(*local_mask) > 0;
}

struct FlatColorMatchCellDefinition {
    cv::Rect roi;
    cv::Mat expected_mask;
    cv::Mat paint_mask;
    cv::Point2f center;
    bool green = false;
    int cells = 8;
    int expected_count = 0;
};

struct FlatColorMatchCellRecord {
    cv::Rect roi;
    cv::Mat expected_mask;
    cv::Mat paint_mask;
    cv::Point2f center;
    bool green = false;
    int cells = 8;
    bool painted = false;
    int expected_count = 0;
    int accepted_count = 0;
    cv::Vec3b color = {0, 0, 0};
};

struct QualityCellDefinition {
    cv::Rect roi;
    std::vector<cv::Point> local_poly;
    cv::Mat analysis_mask;
    cv::Mat expected_mask;
    double expected_hue = 0.0;
    bool valid = false;
};

double circularMeanHueRoi(const cv::Mat& hsv_image, const cv::Mat& local_mask, const cv::Rect& roi) {
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    int count = 0;
    const cv::Mat hsv_roi = hsv_image(roi);
    for (int y = 0; y < hsv_roi.rows; ++y) {
        const uchar* mask_row = local_mask.ptr<uchar>(y);
        const cv::Vec3b* hsv_row = hsv_roi.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv_roi.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            double angle = static_cast<double>(hsv_row[x][0]) * (CV_PI / 90.0);
            sum_cos += std::cos(angle);
            sum_sin += std::sin(angle);
            ++count;
        }
    }
    if (count == 0) {
        return 0.0;
    }
    double angle = std::atan2(sum_sin, sum_cos);
    if (angle < 0.0) {
        angle += 2.0 * CV_PI;
    }
    return angle * (90.0 / CV_PI);
}

uchar homogeneityIntensityForMaskRoi(const cv::Mat& hsv_image, const cv::Mat& local_mask, const cv::Rect& roi) {
    int count = cv::countNonZero(local_mask);
    if (count < 12) {
        return 0;
    }
    std::vector<double> sats;
    std::vector<double> vals;
    sats.reserve(count);
    vals.reserve(count);
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    const cv::Mat hsv_roi = hsv_image(roi);
    for (int y = 0; y < hsv_roi.rows; ++y) {
        const uchar* mask_row = local_mask.ptr<uchar>(y);
        const cv::Vec3b* hsv_row = hsv_roi.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv_roi.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            const double h = hsv_row[x][0];
            const double angle = h * (CV_PI / 90.0);
            sum_cos += std::cos(angle);
            sum_sin += std::sin(angle);
            sats.push_back(hsv_row[x][1]);
            vals.push_back(hsv_row[x][2]);
        }
    }
    double n = static_cast<double>(sats.size());
    double resultant = std::sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / std::max(1.0, n);
    resultant = std::clamp(resultant, 1e-6, 1.0);
    double hue_std = std::sqrt(-2.0 * std::log(resultant)) * (90.0 / CV_PI);
    double sat_range = percentile(sats, 0.90) - percentile(sats, 0.10);
    double val_range = percentile(vals, 0.90) - percentile(vals, 0.10);
    double score = hue_std * 5.0 + sat_range * 1.6 + val_range * 1.6;
    int intensity = 255 - static_cast<int>(std::lround(score));
    return cv::saturate_cast<uchar>(std::clamp(intensity, 0, 255));
}

double brightnessScoreForMaskRoi(const cv::Mat& hsv_image, const cv::Mat& local_mask, const cv::Rect& roi) {
    int count = cv::countNonZero(local_mask);
    if (count < 12) {
        return 0.0;
    }
    std::vector<double> vals;
    vals.reserve(count);
    const cv::Mat hsv_roi = hsv_image(roi);
    for (int y = 0; y < hsv_roi.rows; ++y) {
        const uchar* mask_row = local_mask.ptr<uchar>(y);
        const cv::Vec3b* hsv_row = hsv_roi.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv_roi.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            vals.push_back(hsv_row[x][2]);
        }
    }
    double high_value = percentile(vals, 0.90);
    return std::clamp((high_value - 45.0) / 75.0, 0.0, 1.0);
}

std::vector<QualityCellDefinition> buildInnerQualityCellDefinitions(
    const DetectionResult& ref_tags,
    cv::Size image_size,
    const cv::Mat& ref_hsv,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask) {
    constexpr int cells = 8;
    std::vector<QualityCellDefinition> definitions;
    for (const auto& [tag_id, ref_corners] : ref_tags.detections) {
        (void)tag_id;
        if (ref_corners.size() != 4) {
            continue;
        }
        for (int cy = 0; cy < cells; ++cy) {
            for (int cx = 0; cx < cells; ++cx) {
                cv::Rect roi;
                cv::Mat local_mask;
                std::vector<cv::Point> local_poly;
                if (!makeLocalCellMask(makeCellPoly(ref_corners, cells, cx, cy), image_size, &roi, &local_mask, &local_poly)) {
                    continue;
                }
                cv::Mat analysis_mask;
                cv::erode(local_mask, analysis_mask, morphKernel3(), cv::Point(-1, -1), 1);
                if (cv::countNonZero(analysis_mask) < 12) {
                    continue;
                }
                cv::Mat expected_green;
                cv::Mat expected_blue;
                cv::bitwise_and(analysis_mask, ref_green_mask(roi), expected_green);
                cv::bitwise_and(analysis_mask, ref_blue_mask(roi), expected_blue);
                int green_count = cv::countNonZero(expected_green);
                int blue_count = cv::countNonZero(expected_blue);
                cv::Mat expected_mask;
                if (blue_count >= green_count && blue_count >= 12) {
                    expected_mask = expected_blue;
                } else if (green_count >= 12) {
                    expected_mask = expected_green;
                } else {
                    continue;
                }
                QualityCellDefinition definition;
                definition.roi = roi;
                definition.local_poly = std::move(local_poly);
                definition.analysis_mask = std::move(analysis_mask);
                definition.expected_mask = expected_mask.clone();
                definition.expected_hue = circularMeanHueRoi(ref_hsv, definition.expected_mask, definition.roi);
                definition.valid = true;
                definitions.push_back(std::move(definition));
            }
        }
    }
    return definitions;
}

std::vector<QualityCellDefinition> buildOuterGreenQualityCellDefinitions(
    const DetectionResult& ref_tags,
    cv::Size image_size,
    const cv::Mat& ref_hsv,
    const cv::Mat& ref_green_mask) {
    constexpr int cells = 8;
    constexpr int rings = 4;
    std::vector<QualityCellDefinition> definitions;
    auto addCell = [&](const std::vector<cv::Point2f>& ref_corners, int cx, int cy) {
        cv::Rect roi;
        cv::Mat local_mask;
        std::vector<cv::Point> local_poly;
        if (!makeLocalCellMask(makeCellPoly(ref_corners, cells, cx, cy), image_size, &roi, &local_mask, &local_poly)) {
            return;
        }
        cv::Mat expected_green;
        cv::bitwise_and(local_mask, ref_green_mask(roi), expected_green);
        int green_count = cv::countNonZero(expected_green);
        int cell_count = cv::countNonZero(local_mask);
        if (green_count < 12 || green_count < static_cast<int>(std::lround(cell_count * 0.35))) {
            return;
        }
        QualityCellDefinition definition;
        definition.roi = roi;
        definition.local_poly = std::move(local_poly);
        definition.analysis_mask = expected_green.clone();
        definition.expected_mask = std::move(expected_green);
        definition.expected_hue = circularMeanHueRoi(ref_hsv, definition.expected_mask, definition.roi);
        definition.valid = true;
        definitions.push_back(std::move(definition));
    };
    for (const auto& [tag_id, ref_corners] : ref_tags.detections) {
        (void)tag_id;
        if (ref_corners.size() != 4) {
            continue;
        }
        for (int cy = -rings; cy < cells + rings; ++cy) {
            for (int cx = -rings; cx < cells + rings; ++cx) {
                if (cx >= 0 && cx < cells && cy >= 0 && cy < cells) {
                    continue;
                }
                addCell(ref_corners, cx, cy);
            }
        }
        for (int cy = -rings; cy < cells + rings; ++cy) {
            addCell(ref_corners, -rings - 1, cy);
        }
    }
    return definitions;
}

cv::Mat makeTagCellCombinedQualityGrayImage(
    const cv::Mat& flat_hsv,
    const cv::Mat& ref_hsv,
    const DetectionResult& ref_tags,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask) {
    cv::Mat combined(flat_hsv.size(), CV_8UC1, cv::Scalar(0));
    std::vector<QualityCellDefinition> cells = buildInnerQualityCellDefinitions(
        ref_tags,
        flat_hsv.size(),
        ref_hsv,
        ref_green_mask,
        ref_blue_mask);
    for (const QualityCellDefinition& cell : cells) {
        if (!cell.valid) {
            continue;
        }
        uchar homogeneity = homogeneityIntensityForMaskRoi(flat_hsv, cell.analysis_mask, cell.roi);
        double observed_hue = circularMeanHueRoi(flat_hsv, cell.expected_mask, cell.roi);
        double hue_distance = hueDistanceOpenCv(observed_hue, cell.expected_hue);
        double hue_score = static_cast<double>(
            cv::saturate_cast<uchar>(std::clamp(255 - static_cast<int>(std::lround(hue_distance * 5.0)), 0, 255))) / 255.0;
        double homogeneity_score = static_cast<double>(homogeneity) / 255.0;
        double brightness_score = brightnessScoreForMaskRoi(flat_hsv, cell.expected_mask, cell.roi);
        uchar value = cv::saturate_cast<uchar>(
            std::lround(std::sqrt(homogeneity_score * hue_score) * brightness_score * 255.0));
        cv::fillConvexPoly(combined(cell.roi), cell.local_poly, cv::Scalar(value));
    }
    return combined;
}

void addOuterGreenCellQuality(
    const cv::Mat& flat_hsv,
    const cv::Mat& ref_hsv,
    const DetectionResult& ref_tags,
    const cv::Mat& ref_green_mask,
    cv::Mat* combined_gray) {
    std::vector<QualityCellDefinition> cells = buildOuterGreenQualityCellDefinitions(
        ref_tags,
        flat_hsv.size(),
        ref_hsv,
        ref_green_mask);
    for (const QualityCellDefinition& cell : cells) {
        if (!cell.valid) {
            continue;
        }
        uchar homogeneity = homogeneityIntensityForMaskRoi(flat_hsv, cell.analysis_mask, cell.roi);
        double observed_hue = circularMeanHueRoi(flat_hsv, cell.expected_mask, cell.roi);
        double hue_distance = hueDistanceOpenCv(observed_hue, cell.expected_hue);
        double hue_score = static_cast<double>(
            cv::saturate_cast<uchar>(std::clamp(255 - static_cast<int>(std::lround(hue_distance * 5.0)), 0, 255))) / 255.0;
        double homogeneity_score = static_cast<double>(homogeneity) / 255.0;
        double brightness_score = brightnessScoreForMaskRoi(flat_hsv, cell.expected_mask, cell.roi);
        uchar value = cv::saturate_cast<uchar>(
            std::lround(std::sqrt(homogeneity_score * hue_score) * brightness_score * 255.0));
        cv::fillConvexPoly((*combined_gray)(cell.roi), cell.local_poly, cv::Scalar(value));
    }
}

cv::Mat makeFlatOccluderPenaltyMask(const cv::Mat& flat_raw_frame_bgr, const cv::Mat& reference_bgr) {
    cv::Mat hsv;
    cv::cvtColor(flat_raw_frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat ycrcb;
    cv::cvtColor(flat_raw_frame_bgr, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::Mat gray;
    cv::cvtColor(flat_raw_frame_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 35, 110);
    cv::dilate(edges, edges, morphKernel3(), cv::Point(-1, -1), 1);

    cv::Mat penalty(flat_raw_frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, flat_raw_frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* bgr_row = flat_raw_frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* ref_row = reference_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* hsv_row = hsv.ptr<cv::Vec3b>(y);
            const cv::Vec3b* ycrcb_row = ycrcb.ptr<cv::Vec3b>(y);
            const uchar* edge_row = edges.ptr<uchar>(y);
            uchar* out = penalty.ptr<uchar>(y);
            for (int x = 0; x < flat_raw_frame_bgr.cols; ++x) {
                int b = bgr_row[x][0];
                int g = bgr_row[x][1];
                int r = bgr_row[x][2];
                int maxc = std::max({b, g, r});
                int minc = std::min({b, g, r});
                int chroma = maxc - minc;
                bool skin_ycrcb =
                    ycrcb_row[x][0] >= 35 &&
                    ycrcb_row[x][1] >= 132 && ycrcb_row[x][1] <= 182 &&
                    ycrcb_row[x][2] >= 72 && ycrcb_row[x][2] <= 142;
                bool skin_hsv =
                    ((hsv_row[x][0] <= 25 || hsv_row[x][0] >= 165) &&
                     hsv_row[x][1] >= 18 && hsv_row[x][1] <= 190 &&
                     hsv_row[x][2] >= 55);
                bool skin_like = skin_ycrcb && skin_hsv;
                bool dark_tool_like = maxc <= 85 && chroma <= 35 && edge_row[x] > 0;
                int rb = ref_row[x][0];
                int rg = ref_row[x][1];
                int rr = ref_row[x][2];
                int ref_max = std::max({rb, rg, rr});
                int ref_min = std::min({rb, rg, rr});
                int ref_chroma = ref_max - ref_min;
                bool ref_green = rg >= rr + 4 && rg >= rb - 35;
                bool ref_blue = rb >= rr + 10 && rb >= rg - 35;
                bool ref_sheet_color = ref_chroma >= 12 && (ref_green || ref_blue);
                bool observed_green =
                    (g >= r + 4 && g >= b - 35) ||
                    (hsv_row[x][0] >= 30 && hsv_row[x][0] <= 105 &&
                     hsv_row[x][1] >= 16 && hsv_row[x][2] >= 22);
                bool observed_blue =
                    (b >= r + 10 && b >= g - 35) ||
                    (hsv_row[x][0] >= 82 && hsv_row[x][0] <= 140 &&
                     hsv_row[x][1] >= 18 && hsv_row[x][2] >= 18);
                bool observed_expected_sheet_color =
                    (ref_green && observed_green) || (ref_blue && observed_blue);
                bool strong_mismatch = ref_sheet_color && chroma < 10 && maxc < 120 && edge_row[x] > 0;
                bool dark_neutral_occluder =
                    ref_sheet_color &&
                    !observed_expected_sheet_color &&
                    maxc <= 92 &&
                    chroma <= 26;
                if (skin_like || dark_tool_like || strong_mismatch || dark_neutral_occluder) {
                    out[x] = 255;
                }
            }
        }
    });
    cv::morphologyEx(penalty, penalty, cv::MORPH_OPEN, morphKernel3(), cv::Point(-1, -1), 1);
    cv::dilate(penalty, penalty, morphKernel5(), cv::Point(-1, -1), 1);
    return penalty;
}

cv::Mat makeUnexpectedEdgeQualityPenaltyMask(
    const cv::Mat& flat_raw_frame_bgr,
    const cv::Mat& reference_bgr) {
    cv::Mat flat_gray;
    cv::cvtColor(flat_raw_frame_bgr, flat_gray, cv::COLOR_BGR2GRAY);
    cv::Mat ref_gray;
    cv::cvtColor(reference_bgr, ref_gray, cv::COLOR_BGR2GRAY);

    cv::Mat flat_edges;
    cv::Canny(flat_gray, flat_edges, 32, 105);
    cv::dilate(flat_edges, flat_edges, morphKernel3(), cv::Point(-1, -1), 1);

    cv::Mat ref_edges;
    cv::Canny(ref_gray, ref_edges, 32, 105);
    cv::dilate(ref_edges, ref_edges, morphKernel5(), cv::Point(-1, -1), 2);

    cv::Mat flat_hsv;
    cv::cvtColor(flat_raw_frame_bgr, flat_hsv, cv::COLOR_BGR2HSV);
    cv::Mat penalty(flat_raw_frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, flat_raw_frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const uchar* flat_edge_row = flat_edges.ptr<uchar>(y);
            const uchar* ref_edge_row = ref_edges.ptr<uchar>(y);
            const cv::Vec3b* bgr_row = flat_raw_frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* ref_row = reference_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* hsv_row = flat_hsv.ptr<cv::Vec3b>(y);
            uchar* out = penalty.ptr<uchar>(y);
            for (int x = 0; x < flat_raw_frame_bgr.cols; ++x) {
                if (flat_edge_row[x] == 0 || ref_edge_row[x] != 0) {
                    continue;
                }

                const int b = bgr_row[x][0];
                const int g = bgr_row[x][1];
                const int r = bgr_row[x][2];
                const int maxc = std::max({b, g, r});
                const int minc = std::min({b, g, r});
                const int chroma = maxc - minc;

                const int rb = ref_row[x][0];
                const int rg = ref_row[x][1];
                const int rr = ref_row[x][2];
                const int ref_max = std::max({rb, rg, rr});
                const int ref_min = std::min({rb, rg, rr});
                const int ref_chroma = ref_max - ref_min;
                const bool ref_green = rg >= rr + 4 && rg >= rb - 35;
                const bool ref_blue = rb >= rr + 10 && rb >= rg - 35;
                const bool ref_sheet_color = ref_chroma >= 12 && (ref_green || ref_blue);
                if (!ref_sheet_color) {
                    continue;
                }

                const bool observed_green =
                    (g >= r + 4 && g >= b - 35) ||
                    (hsv_row[x][0] >= 30 && hsv_row[x][0] <= 105 &&
                     hsv_row[x][1] >= 16 && hsv_row[x][2] >= 22);
                const bool observed_blue =
                    (b >= r + 10 && b >= g - 35) ||
                    (hsv_row[x][0] >= 82 && hsv_row[x][0] <= 140 &&
                     hsv_row[x][1] >= 18 && hsv_row[x][2] >= 18);
                const bool observed_expected_sheet_color =
                    (ref_green && observed_green) || (ref_blue && observed_blue);
                const bool dark_neutral = maxc <= 110 && chroma <= 32;
                const bool wrong_class =
                    (ref_green && !observed_green) ||
                    (ref_blue && !observed_blue);
                if (wrong_class || (dark_neutral && !observed_expected_sheet_color)) {
                    out[x] = 255;
                }
            }
        }
    });
    cv::dilate(penalty, penalty, morphKernel5(), cv::Point(-1, -1), 1);
    return penalty;
}

cv::Mat makeGreenShadowBoostMask(
    const cv::Mat& flat_raw_frame_bgr,
    const cv::Mat& reference_bgr,
    const cv::Mat& strict_reference_green_mask) {
    cv::Mat flat_hsv;
    cv::cvtColor(flat_raw_frame_bgr, flat_hsv, cv::COLOR_BGR2HSV);
    cv::Mat ref_hsv;
    cv::cvtColor(reference_bgr, ref_hsv, cv::COLOR_BGR2HSV);
    cv::Mat boost(flat_raw_frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, flat_raw_frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const uchar* green_row = strict_reference_green_mask.ptr<uchar>(y);
            const cv::Vec3b* flat_row = flat_hsv.ptr<cv::Vec3b>(y);
            const cv::Vec3b* ref_row = ref_hsv.ptr<cv::Vec3b>(y);
            uchar* out = boost.ptr<uchar>(y);
            for (int x = 0; x < flat_raw_frame_bgr.cols; ++x) {
                if (green_row[x] == 0) {
                    continue;
                }
                int hue_diff = std::abs(static_cast<int>(flat_row[x][0]) - static_cast<int>(ref_row[x][0]));
                float hue_delta = static_cast<float>(std::min(hue_diff, 180 - hue_diff));
                int sat = flat_row[x][1];
                int val = flat_row[x][2];
                int ref_val = std::max(1, static_cast<int>(ref_row[x][2]));
                float value_ratio = static_cast<float>(val) / static_cast<float>(ref_val);
                bool green_compatible =
                    flat_row[x][0] >= 30 && flat_row[x][0] <= 115 &&
                    hue_delta <= 24.0f &&
                    sat >= 18 &&
                    val >= 28;
                bool shadow_like = value_ratio >= 0.28f && value_ratio <= 0.95f;
                if (green_compatible && shadow_like) {
                    out[x] = 255;
                }
            }
        }
    });
    cv::morphologyEx(boost, boost, cv::MORPH_OPEN, morphKernel3(), cv::Point(-1, -1), 1);
    return boost;
}

cv::Mat makeSoftTagSupportBoostMask(
    const ReferenceModel& reference,
    const GeometryResult& geometry) {
    cv::Mat boost(reference.reference_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat reference_color_mask;
    cv::bitwise_or(reference.reference_green_mask, reference.reference_blue_mask, reference_color_mask);
    constexpr int cells = 8;
    for (const SoftTag& soft : geometry.soft_tags) {
        auto ref_it = reference.reference_tags.detections.find(soft.id);
        if (ref_it == reference.reference_tags.detections.end() || ref_it->second.size() != 4) {
            continue;
        }
        for (int cy = 0; cy < cells; ++cy) {
            for (int cx = 0; cx < cells; ++cx) {
                cv::Rect roi;
                cv::Mat local_mask;
                if (!makeLocalCellMask(
                        makeCellPoly(ref_it->second, cells, cx, cy),
                        reference.reference_bgr.size(),
                        &roi,
                        &local_mask)) {
                    continue;
                }
                cv::Mat expected_color;
                cv::bitwise_and(local_mask, reference_color_mask(roi), expected_color);
                const int expected_count = cv::countNonZero(expected_color);
                const int cell_count = cv::countNonZero(local_mask);
                if (expected_count < 12 ||
                        expected_count < static_cast<int>(std::lround(cell_count * 0.20))) {
                    continue;
                }
                boost(roi).setTo(cv::Scalar(255), local_mask);
            }
        }
    }
    return boost;
}

bool h5StylusPointFromCornerHomography(
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size frame_size,
    cv::Point2f* frame_point,
    bool require_touch = true) {
    if (frame_point == nullptr ||
            h5_corners == nullptr ||
            !h5_corners->available ||
            !h5_corners->stylus_available ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->frame_corners.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_points.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_pressure.size() ||
            h5_corners->source_size.width <= 0.0f ||
            h5_corners->source_size.height <= 0.0f) {
        return false;
    }
    if (require_touch && h5_corners->stylus_pressure[static_cast<size_t>(frame_index)] <= 0.0f) {
        return false;
    }

    const cv::Point2f stylus = h5_corners->stylus_points[static_cast<size_t>(frame_index)];
    if (!std::isfinite(stylus.x) || !std::isfinite(stylus.y) ||
            stylus.x < 0.0f || stylus.y < 0.0f) {
        return false;
    }

    const std::array<cv::Point2f, 4>& corners =
        h5_corners->frame_corners[static_cast<size_t>(frame_index)];
    std::vector<cv::Point2f> frame_corners;
    frame_corners.reserve(4);
    const float scale_x = static_cast<float>(frame_size.width) / h5_corners->source_size.width;
    const float scale_y = static_cast<float>(frame_size.height) / h5_corners->source_size.height;
    for (const cv::Point2f& corner : corners) {
        if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
            return false;
        }
        frame_corners.emplace_back(corner.x * scale_x, corner.y * scale_y);
    }
    if (std::abs(cv::contourArea(frame_corners)) <= 1.0) {
        return false;
    }

    std::vector<cv::Point2f> model_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(kStylusModelWidth - 1), 0.0f},
        {static_cast<float>(kStylusModelWidth - 1), static_cast<float>(kStylusModelHeight - 1)},
        {0.0f, static_cast<float>(kStylusModelHeight - 1)},
    };
    cv::Mat H_model_to_frame = cv::getPerspectiveTransform(model_corners, frame_corners);
    if (H_model_to_frame.empty()) {
        return false;
    }

    const float nx = stylus.x / kStylusMaxX;
    const float ny = stylus.y / kStylusMaxY;
    if (!std::isfinite(nx) || !std::isfinite(ny)) {
        return false;
    }
    cv::Point2f model_point(
        nx * static_cast<float>(kStylusModelWidth - 1),
        ny * static_cast<float>(kStylusModelHeight - 1));
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(std::vector<cv::Point2f>{model_point}, projected, H_model_to_frame);
    if (projected.size() != 1 || !std::isfinite(projected[0].x) || !std::isfinite(projected[0].y)) {
        return false;
    }
    *frame_point = projected[0];
    return true;
}

bool h5StylusTiltRayFromCornerHomography(
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size frame_size,
    cv::Point2f* ray_start,
    cv::Point2f* ray_end,
    bool require_touch = true) {
    if (ray_start == nullptr ||
            ray_end == nullptr ||
            h5_corners == nullptr ||
            !h5_corners->available ||
            !h5_corners->stylus_available ||
            !h5_corners->tilt_available ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->frame_corners.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_points.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_pressure.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_tilt.size() ||
            h5_corners->source_size.width <= 0.0f ||
            h5_corners->source_size.height <= 0.0f) {
        return false;
    }
    if (require_touch && h5_corners->stylus_pressure[static_cast<size_t>(frame_index)] <= 0.0f) {
        return false;
    }

    const cv::Point2f stylus = h5_corners->stylus_points[static_cast<size_t>(frame_index)];
    const cv::Point2f tilt = h5_corners->stylus_tilt[static_cast<size_t>(frame_index)];
    if (!std::isfinite(stylus.x) || !std::isfinite(stylus.y) ||
            !std::isfinite(tilt.x) || !std::isfinite(tilt.y) ||
            stylus.x < 0.0f || stylus.y < 0.0f) {
        return false;
    }
    const float tilt_magnitude = std::hypot(tilt.x, tilt.y);
    if (tilt_magnitude < 1.0f) {
        return false;
    }

    const std::array<cv::Point2f, 4>& corners =
        h5_corners->frame_corners[static_cast<size_t>(frame_index)];
    std::vector<cv::Point2f> frame_corners;
    frame_corners.reserve(4);
    const float scale_x = static_cast<float>(frame_size.width) / h5_corners->source_size.width;
    const float scale_y = static_cast<float>(frame_size.height) / h5_corners->source_size.height;
    for (const cv::Point2f& corner : corners) {
        if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
            return false;
        }
        frame_corners.emplace_back(corner.x * scale_x, corner.y * scale_y);
    }
    if (std::abs(cv::contourArea(frame_corners)) <= 1.0) {
        return false;
    }

    std::vector<cv::Point2f> model_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(kStylusModelWidth - 1), 0.0f},
        {static_cast<float>(kStylusModelWidth - 1), static_cast<float>(kStylusModelHeight - 1)},
        {0.0f, static_cast<float>(kStylusModelHeight - 1)},
    };
    cv::Mat H_model_to_frame = cv::getPerspectiveTransform(model_corners, frame_corners);
    if (H_model_to_frame.empty()) {
        return false;
    }

    cv::Point2f model_start(
        (stylus.x / kStylusMaxX) * static_cast<float>(kStylusModelWidth - 1),
        (stylus.y / kStylusMaxY) * static_cast<float>(kStylusModelHeight - 1));
    if (!std::isfinite(model_start.x) || !std::isfinite(model_start.y)) {
        return false;
    }
    const float ray_length = 20.0f + 70.0f * std::clamp(tilt_magnitude / 60.0f, 0.0f, 1.0f);
    cv::Point2f tilt_unit(tilt.x / tilt_magnitude, tilt.y / tilt_magnitude);
    cv::Point2f model_end = model_start + tilt_unit * ray_length;

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(std::vector<cv::Point2f>{model_start, model_end}, projected, H_model_to_frame);
    if (projected.size() != 2 ||
            !std::isfinite(projected[0].x) || !std::isfinite(projected[0].y) ||
            !std::isfinite(projected[1].x) || !std::isfinite(projected[1].y)) {
        return false;
    }
    *ray_start = projected[0];
    *ray_end = projected[1];
    return true;
}

bool h5StylusPointInReferenceSurface(
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size reference_size,
    cv::Point2f* ref_point,
    bool require_touch = true) {
    if (ref_point == nullptr ||
            h5_corners == nullptr ||
            !h5_corners->stylus_available ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_points.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_pressure.size()) {
        return false;
    }
    if (require_touch && h5_corners->stylus_pressure[static_cast<size_t>(frame_index)] <= 0.0f) {
        return false;
    }
    const cv::Point2f stylus = h5_corners->stylus_points[static_cast<size_t>(frame_index)];
    if (!std::isfinite(stylus.x) || !std::isfinite(stylus.y) ||
            stylus.x < 0.0f || stylus.y < 0.0f) {
        return false;
    }

    const float nx = stylus.x / kStylusMaxX;
    const float ny = stylus.y / kStylusMaxY;
    if (!std::isfinite(nx) || !std::isfinite(ny)) {
        return false;
    }
    *ref_point = cv::Point2f(
        nx * static_cast<float>(reference_size.width - 1),
        ny * static_cast<float>(reference_size.height - 1));
    return std::isfinite(ref_point->x) && std::isfinite(ref_point->y);
}

bool h5StylusTiltRayInReferenceSurface(
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size reference_size,
    cv::Point2f* ref_start,
    cv::Point2f* ref_end) {
    if (ref_start == nullptr ||
            ref_end == nullptr ||
            h5_corners == nullptr ||
            !h5_corners->stylus_available ||
            !h5_corners->tilt_available ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_points.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_pressure.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_tilt.size() ||
            h5_corners->stylus_pressure[static_cast<size_t>(frame_index)] <= 0.0f) {
        return false;
    }
    const cv::Point2f stylus = h5_corners->stylus_points[static_cast<size_t>(frame_index)];
    const cv::Point2f tilt = h5_corners->stylus_tilt[static_cast<size_t>(frame_index)];
    if (!std::isfinite(stylus.x) || !std::isfinite(stylus.y) ||
            !std::isfinite(tilt.x) || !std::isfinite(tilt.y) ||
            stylus.x < 0.0f || stylus.y < 0.0f) {
        return false;
    }
    const float tilt_magnitude = std::hypot(tilt.x, tilt.y);
    if (tilt_magnitude < 1.0f) {
        return false;
    }

    cv::Point2f model_start(
        (stylus.x / kStylusMaxX) * static_cast<float>(kStylusModelWidth - 1),
        (stylus.y / kStylusMaxY) * static_cast<float>(kStylusModelHeight - 1));
    if (!std::isfinite(model_start.x) || !std::isfinite(model_start.y)) {
        return false;
    }
    const float ray_length = 20.0f + 70.0f * std::clamp(tilt_magnitude / 60.0f, 0.0f, 1.0f);
    cv::Point2f tilt_unit(tilt.x / tilt_magnitude, tilt.y / tilt_magnitude);
    cv::Point2f model_end = model_start + tilt_unit * ray_length;

    std::vector<cv::Point2f> model_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(kStylusModelWidth - 1), 0.0f},
        {static_cast<float>(kStylusModelWidth - 1), static_cast<float>(kStylusModelHeight - 1)},
        {0.0f, static_cast<float>(kStylusModelHeight - 1)},
    };
    cv::Mat H_model_to_ref = cv::getPerspectiveTransform(
        model_corners,
        referenceImageCorners(reference_size));
    if (H_model_to_ref.empty()) {
        return false;
    }
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(std::vector<cv::Point2f>{model_start, model_end}, projected, H_model_to_ref);
    if (projected.size() != 2 ||
            !std::isfinite(projected[0].x) || !std::isfinite(projected[0].y) ||
            !std::isfinite(projected[1].x) || !std::isfinite(projected[1].y)) {
        return false;
    }
    *ref_start = projected[0];
    *ref_end = projected[1];
    return true;
}

cv::Mat makeFlatStylusTipPenaltyMask(
    int frame_index,
    cv::Size frame_size,
    const cv::Mat& H_frame_to_ref,
    const H5CornerTrace* h5_corners,
    cv::Size reference_size) {
    cv::Mat mask(reference_size, CV_8UC1, cv::Scalar(0));
    if (H_frame_to_ref.empty()) {
        return mask;
    }
    (void)frame_size;
    int radius = std::clamp(envInt("SHADOW_H5_STYLUS_EXCLUDE_RADIUS", 38), 4, 120);
    int frame_radius = std::clamp(envInt("SHADOW_H5_STYLUS_EXCLUDE_FRAME_RADIUS", 1), 0, 4);
    double ray_cells = static_cast<double>(std::clamp(envInt("SHADOW_H5_STYLUS_EXCLUDE_RAY_CELLS", 3), 0, 10));
    for (int offset = -frame_radius; offset <= frame_radius; ++offset) {
        int sample_frame = frame_index + offset;
        cv::Point2f ref_point;
        if (!h5StylusPointInReferenceSurface(h5_corners, sample_frame, reference_size, &ref_point)) {
            continue;
        }
        if (!std::isfinite(ref_point.x) || !std::isfinite(ref_point.y)) {
            continue;
        }
        cv::circle(
            mask,
            cv::Point(cvRound(ref_point.x), cvRound(ref_point.y)),
            radius,
            cv::Scalar(255),
            cv::FILLED,
            cv::LINE_AA);

        cv::Point2f ref_ray_start;
        cv::Point2f ref_ray_end;
        if (ray_cells <= 0.0 ||
                !h5StylusTiltRayInReferenceSurface(
                    h5_corners,
                    sample_frame,
                    reference_size,
                    &ref_ray_start,
                    &ref_ray_end)) {
            continue;
        }
        cv::Point2f ray_dir = ref_ray_end - ref_ray_start;
        float ray_norm = cv::norm(ray_dir);
        if (ray_norm <= 1e-3f || !std::isfinite(ray_norm)) {
            continue;
        }
        ray_dir *= 1.0f / ray_norm;
        cv::Point2f ray_tip = ref_point + ray_dir * static_cast<float>(static_cast<double>(radius) * ray_cells);
        cv::line(
            mask,
            cv::Point(cvRound(ref_point.x), cvRound(ref_point.y)),
            cv::Point(cvRound(ray_tip.x), cvRound(ray_tip.y)),
            cv::Scalar(255),
            radius * 2,
            cv::LINE_AA);
    }
    return mask;
}

std::vector<FlatColorMatchCellDefinition> buildFlatColorMatchCellDefinitions(
    const DetectionResult& ref_tags,
    cv::Size image_size,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask) {
    constexpr int inner_cells = 8;
    constexpr int outer_cells = 8;
    constexpr int rings = 4;
    std::vector<FlatColorMatchCellDefinition> definitions;
    cv::Mat green_only = ref_green_mask.clone();
    cv::Mat blue_only = ref_blue_mask.clone();
    green_only.setTo(0, ref_blue_mask);
    blue_only.setTo(0, ref_green_mask);

    auto addCell = [&](
        const std::vector<cv::Point2f>& ref_corners,
        int cells,
        int cx,
        int cy,
        bool green_only_cell) {
        std::vector<cv::Point> poly = makeCellPoly(ref_corners, cells, cx, cy);
        cv::Rect roi;
        cv::Mat local_mask;
        if (!makeLocalCellMask(poly, image_size, &roi, &local_mask)) {
            return;
        }

        cv::Point2f center(0.0f, 0.0f);
        for (const cv::Point& point : poly) {
            center += cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y));
        }
        center *= 1.0f / static_cast<float>(poly.size());

        auto addDefinition = [&](
            const cv::Mat& expected_mask,
            bool green,
            int expected_count) {
            if (expected_count < 12) {
                return;
            }
            FlatColorMatchCellDefinition definition;
            definition.roi = roi;
            definition.expected_mask = expected_mask.clone();
            definition.paint_mask = local_mask.clone();
            definition.center = center;
            definition.green = green;
            definition.cells = cells;
            definition.expected_count = expected_count;
            definitions.push_back(std::move(definition));
        };

        cv::Mat expected_green;
        cv::bitwise_and(local_mask, green_only(roi), expected_green);
        int green_count = cv::countNonZero(expected_green);
        if (green_only_cell) {
            addDefinition(expected_green, true, green_count);
            return;
        }

        cv::Mat expected_blue;
        cv::bitwise_and(local_mask, blue_only(roi), expected_blue);
        int blue_count = cv::countNonZero(expected_blue);
        if (blue_count >= green_count) {
            addDefinition(expected_blue, false, blue_count);
        } else {
            addDefinition(expected_green, true, green_count);
        }
    };

    for (const auto& [tag_id, ref_corners] : ref_tags.detections) {
        (void)tag_id;
        if (ref_corners.size() != 4) {
            continue;
        }
        for (int cy = 0; cy < inner_cells; ++cy) {
            for (int cx = 0; cx < inner_cells; ++cx) {
                addCell(ref_corners, inner_cells, cx, cy, false);
            }
        }
        for (int cy = -rings; cy < outer_cells + rings; ++cy) {
            for (int cx = -rings; cx < outer_cells + rings; ++cx) {
                if (cx >= 0 && cx < outer_cells && cy >= 0 && cy < outer_cells) {
                    continue;
                }
                addCell(ref_corners, outer_cells, cx, cy, true);
            }
        }
        for (int cy = -rings; cy < outer_cells + rings; ++cy) {
            addCell(ref_corners, outer_cells, -rings - 1, cy, true);
        }
    }
    return definitions;
}

cv::Mat makeCellQualityMask(
    const cv::Mat& flat_quality_gray,
    const cv::Mat& flat_hsv,
    const cv::Mat& expected_mask,
    const cv::Rect& roi,
    bool green) {
    constexpr int kColormatchQualityThreshold = 89;
    cv::Mat quality_ok;
    cv::threshold(flat_quality_gray(roi), quality_ok, kColormatchQualityThreshold, 255, cv::THRESH_BINARY);
    cv::Mat accepted;
    cv::bitwise_and(expected_mask, quality_ok, accepted);
    cv::Mat class_ok(accepted.size(), CV_8UC1, cv::Scalar(0));
    const cv::Mat hsv_roi = flat_hsv(roi);
    for (int y = 0; y < hsv_roi.rows; ++y) {
        const uchar* accepted_row = accepted.ptr<uchar>(y);
        const cv::Vec3b* hsv_row = hsv_roi.ptr<cv::Vec3b>(y);
        uchar* out = class_ok.ptr<uchar>(y);
        for (int x = 0; x < hsv_roi.cols; ++x) {
            if (accepted_row[x] == 0) {
                continue;
            }
            const int h = hsv_row[x][0];
            const int s = hsv_row[x][1];
            const int v = hsv_row[x][2];
            bool ok = false;
            if (green) {
                ok = h >= 30 && h <= 105 && s >= 16 && v >= 22;
            } else {
                ok = h >= 82 && h <= 140 && s >= 18 && v >= 18;
            }
            if (ok) {
                out[x] = 255;
            }
        }
    }
    cv::bitwise_and(accepted, class_ok, accepted);
    return accepted;
}

cv::Mat makeFlatStylusCellPenaltyMask(
    const std::vector<FlatColorMatchCellDefinition>& cell_definitions,
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size reference_size) {
    cv::Mat mask(reference_size, CV_8UC1, cv::Scalar(0));
    cv::Point2f ref_point;
    if (!h5StylusPointInReferenceSurface(h5_corners, frame_index, reference_size, &ref_point)) {
        return mask;
    }

    cv::Point2f ref_ray_start;
    cv::Point2f ref_ray_end;
    cv::Point2f ray_dir(0.0f, 0.0f);
    bool have_ray_dir = h5StylusTiltRayInReferenceSurface(
        h5_corners,
        frame_index,
        reference_size,
        &ref_ray_start,
        &ref_ray_end);
    if (have_ray_dir) {
        ray_dir = ref_ray_end - ref_ray_start;
        const float ray_norm = cv::norm(ray_dir);
        if (ray_norm > 1e-3f && std::isfinite(ray_norm)) {
            ray_dir *= 1.0f / ray_norm;
        } else {
            have_ray_dir = false;
        }
    }

    int anchor_index = -1;
    double best_dist2 = std::numeric_limits<double>::max();
    for (size_t i = 0; i < cell_definitions.size(); ++i) {
        const FlatColorMatchCellDefinition& cell = cell_definitions[i];
        if (cell.roi.empty() || cell.paint_mask.empty()) {
            continue;
        }
        if (cell.roi.contains(cv::Point(cvFloor(ref_point.x), cvFloor(ref_point.y)))) {
            int lx = cvFloor(ref_point.x) - cell.roi.x;
            int ly = cvFloor(ref_point.y) - cell.roi.y;
            if (lx >= 0 && ly >= 0 && lx < cell.paint_mask.cols && ly < cell.paint_mask.rows &&
                    cell.paint_mask.at<uchar>(ly, lx) != 0) {
                anchor_index = static_cast<int>(i);
                break;
            }
        }
        const double dx = static_cast<double>(cell.center.x - ref_point.x);
        const double dy = static_cast<double>(cell.center.y - ref_point.y);
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            anchor_index = static_cast<int>(i);
        }
    }
    if (anchor_index < 0) {
        return mask;
    }

    const FlatColorMatchCellDefinition& anchor = cell_definitions[static_cast<size_t>(anchor_index)];
    const int anchor_pixels = std::max(1, cv::countNonZero(anchor.paint_mask));
    const double cell_size = std::max(4.0, std::sqrt(static_cast<double>(anchor_pixels)));
    const double neighbor_radius = cell_size * 1.65;
    const double ray_length = cell_size * static_cast<double>(std::clamp(envInt("SHADOW_H5_STYLUS_CELL_RAY_CELLS", 4), 0, 10));
    const double ray_half_width = cell_size * 1.15;

    for (const FlatColorMatchCellDefinition& cell : cell_definitions) {
        if (cell.roi.empty() || cell.paint_mask.empty()) {
            continue;
        }
        const cv::Point2f delta = cell.center - ref_point;
        const double dist = cv::norm(delta);
        bool exclude = dist <= neighbor_radius;
        if (!exclude && have_ray_dir && ray_length > 0.0) {
            const double ahead = static_cast<double>(delta.x * ray_dir.x + delta.y * ray_dir.y);
            const double lateral = std::abs(static_cast<double>(delta.x * ray_dir.y - delta.y * ray_dir.x));
            exclude = ahead > 0.0 && ahead <= ray_length && lateral <= ray_half_width;
        }
        if (exclude) {
            mask(cell.roi).setTo(cv::Scalar(255), cell.paint_mask);
        }
    }
    return mask;
}

void applyCellPenaltyToQuality(
    const std::vector<FlatColorMatchCellDefinition>& cell_definitions,
    const cv::Mat& penalty_mask,
    cv::Mat* flat_quality_gray) {
    if (penalty_mask.empty() || flat_quality_gray == nullptr || flat_quality_gray->empty()) {
        return;
    }
    for (const FlatColorMatchCellDefinition& cell : cell_definitions) {
        if (cell.roi.empty() || cell.paint_mask.empty()) {
            continue;
        }
        const int paint_count = cv::countNonZero(cell.paint_mask);
        if (paint_count < 12) {
            continue;
        }
        cv::Mat penalized;
        cv::bitwise_and(penalty_mask(cell.roi), cell.paint_mask, penalized);
        const int penalty_count = cv::countNonZero(penalized);
        if (penalty_count >= std::max(4, static_cast<int>(std::lround(paint_count * 0.12)))) {
            (*flat_quality_gray)(cell.roi).setTo(cv::Scalar(0), cell.paint_mask);
        }
    }
}

void applyGreenCellFloorToQuality(
    const std::vector<FlatColorMatchCellDefinition>& cell_definitions,
    const cv::Mat& green_shadow_boost,
    uchar floor_value,
    cv::Mat* flat_quality_gray) {
    if (green_shadow_boost.empty() || flat_quality_gray == nullptr || flat_quality_gray->empty()) {
        return;
    }
    for (const FlatColorMatchCellDefinition& cell : cell_definitions) {
        if (!cell.green || cell.roi.empty() || cell.paint_mask.empty()) {
            continue;
        }
        const int paint_count = cv::countNonZero(cell.paint_mask);
        if (paint_count < 12) {
            continue;
        }
        cv::Mat boosted;
        cv::bitwise_and(green_shadow_boost(cell.roi), cell.paint_mask, boosted);
        const int boost_count = cv::countNonZero(boosted);
        if (boost_count < std::max(4, static_cast<int>(std::lround(paint_count * 0.20)))) {
            continue;
        }
        cv::Mat low;
        cv::threshold((*flat_quality_gray)(cell.roi), low, floor_value - 1, 255, cv::THRESH_BINARY_INV);
        cv::bitwise_and(low, cell.paint_mask, low);
        (*flat_quality_gray)(cell.roi).setTo(cv::Scalar(floor_value), low);
    }
}

void fillNearestMaskedFlatColorMatchCells(
    const std::vector<FlatColorMatchCellRecord>& records,
    cv::Mat* flat_overlay_bgr,
    cv::Mat* flat_overlay_mask,
    const cv::Point2f& preferred_fill_direction = cv::Point2f(0.0f, 0.0f)) {
    struct Group {
        bool green = false;
        int cells = 8;
    };
    const std::vector<Group> groups = {
        {false, 8},
        {true, 8},
    };

    for (const Group& group : groups) {
        bool have_expected = false;
        for (const FlatColorMatchCellRecord& record : records) {
            if (record.green == group.green && record.cells == group.cells && record.expected_count >= 12) {
                have_expected = true;
                break;
            }
        }
        if (!have_expected) {
            continue;
        }

        for (const FlatColorMatchCellRecord& target : records) {
            if (target.green != group.green || target.cells != group.cells || target.expected_count < 12) {
                continue;
            }

            cv::Mat already_filled;
            cv::bitwise_and((*flat_overlay_mask)(target.roi), target.paint_mask, already_filled);
            cv::Mat not_filled;
            cv::bitwise_not(already_filled, not_filled);
            cv::Mat missing_mask;
            cv::bitwise_and(target.paint_mask, not_filled, missing_mask);
            if (cv::countNonZero(missing_mask) < 1) {
                continue;
            }

            const float preferred_norm = cv::norm(preferred_fill_direction);
            const bool have_preferred_direction =
                preferred_norm > 1e-3f && std::isfinite(preferred_norm);
            cv::Point2f preferred_unit = have_preferred_direction
                ? preferred_fill_direction * (1.0f / preferred_norm)
                : cv::Point2f(0.0f, 0.0f);

            auto chooseBest = [&](bool require_preferred_direction) {
                const FlatColorMatchCellRecord* best = nullptr;
                double best_dist2 = std::numeric_limits<double>::max();
                for (const FlatColorMatchCellRecord& candidate : records) {
                    if (!candidate.painted ||
                            candidate.green != target.green ||
                            candidate.cells != target.cells) {
                        continue;
                    }
                    const double dx = static_cast<double>(candidate.center.x - target.center.x);
                    const double dy = static_cast<double>(candidate.center.y - target.center.y);
                    if (require_preferred_direction) {
                        const double along = dx * static_cast<double>(preferred_unit.x) +
                            dy * static_cast<double>(preferred_unit.y);
                        if (along <= 0.0) {
                            continue;
                        }
                    }
                    const double dist2 = dx * dx + dy * dy;
                    if (dist2 < best_dist2) {
                        best_dist2 = dist2;
                        best = &candidate;
                    }
                }
                return best;
            };

            const FlatColorMatchCellRecord* best = have_preferred_direction
                ? chooseBest(true)
                : nullptr;
            if (best == nullptr) {
                best = chooseBest(false);
            }
            if (best == nullptr) {
                continue;
            }
            (*flat_overlay_bgr)(target.roi).setTo(best->color, missing_mask);
            (*flat_overlay_mask)(target.roi).setTo(255, missing_mask);
        }
    }
}

cv::Point2f preferredOppositeStylusFillDirection(
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size reference_size) {
    cv::Point2f ref_start;
    cv::Point2f ref_end;
    if (!h5StylusTiltRayInReferenceSurface(
            h5_corners,
            frame_index,
            reference_size,
            &ref_start,
            &ref_end)) {
        return cv::Point2f(0.0f, 0.0f);
    }
    cv::Point2f direction = ref_start - ref_end;
    const float norm = cv::norm(direction);
    if (norm <= 1e-3f || !std::isfinite(norm)) {
        return cv::Point2f(0.0f, 0.0f);
    }
    return direction * (1.0f / norm);
}

cv::Mat makeAlignedRefColorMatchImageFromTagCells(
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& flat_raw_frame_bgr,
    const cv::Mat& flat_hsv,
    const cv::Mat& flat_quality_gray,
    const std::vector<FlatColorMatchCellDefinition>& cell_definitions,
    const cv::Mat& H_ref_to_frame,
    const cv::Point2f& preferred_fill_direction = cv::Point2f(0.0f, 0.0f)) {
    cv::Mat flat_overlay_bgr(flat_raw_frame_bgr.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat flat_overlay_mask(flat_raw_frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    std::vector<FlatColorMatchCellRecord> records;
    records.reserve(cell_definitions.size());
    for (const FlatColorMatchCellDefinition& definition : cell_definitions) {
        FlatColorMatchCellRecord record;
        record.roi = definition.roi;
        record.expected_mask = definition.expected_mask;
        record.paint_mask = definition.paint_mask;
        record.center = definition.center;
        record.green = definition.green;
        record.cells = definition.cells;
        record.expected_count = definition.expected_count;
        cv::Mat accepted = makeCellQualityMask(
            flat_quality_gray,
            flat_hsv,
            definition.expected_mask,
            definition.roi,
            definition.green);
        record.accepted_count = cv::countNonZero(accepted);
        if (definition.expected_count < 12 ||
                record.accepted_count < std::max(6, static_cast<int>(std::lround(definition.expected_count * 0.35)))) {
            records.push_back(std::move(record));
            continue;
        }
        cv::Vec3b sampled_color = medianColor(flat_raw_frame_bgr(definition.roi), accepted);
        if (!plausibleCellDonorColor(sampled_color, definition.green)) {
            records.push_back(std::move(record));
            continue;
        }
        record.color = sampled_color;
        record.painted = true;
        flat_overlay_bgr(definition.roi).setTo(record.color, definition.paint_mask);
        flat_overlay_mask(definition.roi).setTo(255, definition.paint_mask);
        records.push_back(std::move(record));
    }
    fillNearestMaskedFlatColorMatchCells(
        records,
        &flat_overlay_bgr,
        &flat_overlay_mask,
        preferred_fill_direction);

    cv::Mat frame_overlay_bgr;
    cv::Mat frame_overlay_mask;
    cv::warpPerspective(
        flat_overlay_bgr,
        frame_overlay_bgr,
        H_ref_to_frame,
        aligned_ref_bgr.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
    cv::warpPerspective(
        flat_overlay_mask,
        frame_overlay_mask,
        H_ref_to_frame,
        aligned_ref_bgr.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));
    cv::Mat out = aligned_ref_bgr.clone();
    frame_overlay_bgr.copyTo(out, frame_overlay_mask);
    return out;
}

cv::Mat robustChannelGain(const cv::Mat& frame_bgr, const cv::Mat& warped_ref_bgr, const cv::Mat& mask) {
    cv::Mat aligned_f;
    warped_ref_bgr.convertTo(aligned_f, CV_32FC3);
    int valid_count = mask.empty() ? static_cast<int>(frame_bgr.total()) : cv::countNonZero(mask);
    if (valid_count < 100) {
        cv::Mat out;
        aligned_f.convertTo(out, CV_8UC3);
        return out;
    }
    std::vector<cv::Mat> aligned_channels;
    cv::split(aligned_f, aligned_channels);
    for (int c = 0; c < 3; ++c) {
        std::vector<double> frame_vals;
        std::vector<double> ref_vals;
        frame_vals.reserve(valid_count);
        ref_vals.reserve(valid_count);
        for (int y = 0; y < frame_bgr.rows; ++y) {
            const uchar* mask_row = mask.empty() ? nullptr : mask.ptr<uchar>(y);
            const cv::Vec3b* frame_row = frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* ref_row = warped_ref_bgr.ptr<cv::Vec3b>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                if ((mask_row != nullptr && mask_row[x] == 0) || ref_row[x][c] <= 10) {
                    continue;
                }
                frame_vals.push_back(frame_row[x][c]);
                ref_vals.push_back(ref_row[x][c]);
            }
        }
        if (frame_vals.size() < 100 || ref_vals.size() < 100) {
            continue;
        }
        double gain = std::clamp(medianValue(frame_vals) / std::max(medianValue(ref_vals), 1e-6), 0.5, 2.0);
        aligned_channels[c] *= gain;
    }
    cv::merge(aligned_channels, aligned_f);
    cv::Mat out;
    aligned_f.convertTo(out, CV_8UC3);
    return out;
}

cv::Mat makeSurfaceMaskFromAlignedRef(const cv::Mat& aligned_ref_bgr) {
    cv::Mat gray;
    cv::cvtColor(aligned_ref_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat raw;
    cv::threshold(gray, raw, 5, 255, cv::THRESH_BINARY);
    cv::morphologyEx(raw, raw, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)), cv::Point(-1, -1), 2);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(raw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat surface = cv::Mat::zeros(raw.size(), CV_8UC1);
    if (contours.empty()) {
        return surface;
    }
    size_t best_index = 0;
    double best_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > best_area) {
            best_area = area;
            best_index = i;
        }
    }
    cv::drawContours(surface, contours, static_cast<int>(best_index), cv::Scalar(255), cv::FILLED);
    return surface;
}

cv::Mat makeVisibleBoardColorProtectionMask(const cv::Mat& frame_bgr) {
    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat hsv_mask;
    cv::inRange(hsv, cv::Scalar(35, 12, 25), cv::Scalar(125, 255, 255), hsv_mask);
    cv::Mat dominance(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* row = frame_bgr.ptr<cv::Vec3b>(y);
            uchar* out = dominance.ptr<uchar>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                int b = row[x][0];
                int g = row[x][1];
                int r = row[x][2];
                int maxc = std::max({b, g, r});
                int minc = std::min({b, g, r});
                int chroma = maxc - minc;
                bool skin_like = r >= b + 8 && r >= g - 12 && maxc >= 70;
                bool dark_neutral = maxc <= 90 && chroma <= 28;
                bool green_or_aqua = chroma >= 10 && g >= r + 4 && g >= b - 35;
                bool blue_or_cyan = chroma >= 15 && b >= r + 12 && b >= g - 30;
                if ((green_or_aqua || blue_or_cyan) && !skin_like && !dark_neutral) {
                    out[x] = 255;
                }
            }
        }
    });
    cv::Mat visible_board;
    cv::bitwise_and(hsv_mask, dominance, visible_board);
    return visible_board;
}

cv::Mat excludeVisibleSheetColorFromHandStylusMask(
    const cv::Mat& hand_stylus_mask,
    const cv::Mat& observed_bgr,
    const cv::Mat& board_mask,
    const cv::Point& stylus_roi_center = cv::Point(-1, -1)) {
    if (hand_stylus_mask.empty()) {
        return cv::Mat();
    }

    cv::Mat refined;
    cv::threshold(hand_stylus_mask, refined, 0, 255, cv::THRESH_BINARY);
    if (envInt("SHADOW_HAND_MASK_EXCLUDE_VISIBLE_SHEET", 1) == 0 ||
            observed_bgr.empty() ||
            observed_bgr.channels() != 3 ||
            observed_bgr.size() != refined.size()) {
        return refined;
    }

    cv::Mat visible_sheet = makeVisibleBoardColorProtectionMask(observed_bgr);
    if (!board_mask.empty() && board_mask.size() == refined.size()) {
        cv::bitwise_and(visible_sheet, board_mask, visible_sheet);
    }
    cv::bitwise_and(visible_sheet, refined, visible_sheet);
    if (cv::countNonZero(visible_sheet) == 0) {
        return refined;
    }

    if (envInt("SHADOW_HAND_MASK_EXCLUDE_VISIBLE_SHEET_KEEP_STYLUS_ROI", 1) != 0 &&
            stylus_roi_center.x >= 0 &&
            stylus_roi_center.y >= 0 &&
            stylus_roi_center.x < refined.cols &&
            stylus_roi_center.y < refined.rows) {
        const int radius = std::clamp(envInt("SHADOW_STYLUS_AOI_RADIUS", 35), 20, 80);
        cv::Mat stylus_roi(refined.size(), CV_8UC1, cv::Scalar(0));
        cv::circle(stylus_roi, stylus_roi_center, radius, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
        cv::Mat outside_stylus_roi;
        cv::bitwise_not(stylus_roi, outside_stylus_roi);
        cv::bitwise_and(visible_sheet, outside_stylus_roi, visible_sheet);
        if (cv::countNonZero(visible_sheet) == 0) {
            return refined;
        }
    }

    int dilate_size = std::clamp(envInt("SHADOW_HAND_MASK_EXCLUDE_VISIBLE_SHEET_DILATE", 3), 1, 15);
    if (dilate_size % 2 == 0) {
        ++dilate_size;
    }
    if (dilate_size > 1) {
        cv::dilate(
            visible_sheet,
            visible_sheet,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_size, dilate_size)),
            cv::Point(-1, -1),
            1);
        cv::bitwise_and(visible_sheet, refined, visible_sheet);
    }

    refined.setTo(0, visible_sheet);
    if (!board_mask.empty() && board_mask.size() == refined.size()) {
        cv::bitwise_and(refined, board_mask, refined);
    }
    return refined;
}

cv::Mat fillBinaryMaskHoles(const cv::Mat& mask) {
    cv::Mat binary;
    cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat flood = binary.clone();
    cv::Mat flood_mask = cv::Mat::zeros(binary.rows + 2, binary.cols + 2, CV_8UC1);
    cv::floodFill(flood, flood_mask, cv::Point(0, 0), cv::Scalar(255));
    cv::Mat holes;
    cv::bitwise_not(flood, holes);
    cv::Mat filled;
    cv::bitwise_or(binary, holes, filled);
    return filled;
}

cv::Mat largestComponentMask(const cv::Mat& mask, int min_area) {
    cv::Mat binary;
    cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8UC1);
    int best_label = -1;
    int best_area = 0;
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > best_area) {
            best_area = area;
            best_label = label;
        }
    }
    if (best_label >= 0 && best_area >= min_area) {
        out.setTo(255, labels == best_label);
    }
    return out;
}

bool fastApproxEnabled() {
    return envInt("SHADOW_FAST_APPROX", 1) != 0;
}

int normalizedOddMorphKernelSize(int kernel_size) {
    kernel_size = std::clamp(kernel_size, 1, 101);
    if (kernel_size % 2 == 0) {
        ++kernel_size;
    }
    return kernel_size;
}

cv::Mat fitTextureToReference(const cv::Mat& texture_bgr, cv::Size ref_size) {
    if (texture_bgr.empty() || ref_size.width <= 0 || ref_size.height <= 0) {
        return cv::Mat();
    }
    double scale = std::max(
        static_cast<double>(ref_size.width) / static_cast<double>(texture_bgr.cols),
        static_cast<double>(ref_size.height) / static_cast<double>(texture_bgr.rows));
    cv::Size resized_size(
        std::max(1, static_cast<int>(std::lround(texture_bgr.cols * scale))),
        std::max(1, static_cast<int>(std::lround(texture_bgr.rows * scale))));

    cv::Mat resized;
    cv::resize(texture_bgr, resized, resized_size, 0.0, 0.0, cv::INTER_LINEAR);
    int x0 = std::max(0, (resized.cols - ref_size.width) / 2);
    int y0 = std::max(0, (resized.rows - ref_size.height) / 2);
    cv::Rect crop(
        x0,
        y0,
        std::min(ref_size.width, resized.cols - x0),
        std::min(ref_size.height, resized.rows - y0));
    cv::Mat fitted = resized(crop).clone();
    if (fitted.size() != ref_size) {
        cv::resize(fitted, fitted, ref_size, 0.0, 0.0, cv::INTER_LINEAR);
    }
    return fitted;
}

std::string lowerString(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool isTextureImagePath(const fs::path& path) {
    if (!fs::is_regular_file(path)) {
        return false;
    }
    const std::string ext = lowerString(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".bmp" || ext == ".tif" || ext == ".tiff" ||
           ext == ".webp";
}

std::vector<fs::path> listTexturePaths(const fs::path& texture_dir) {
    if (!fs::exists(texture_dir) || !fs::is_directory(texture_dir)) {
        throw std::runtime_error("Surface texture directory does not exist: " + texture_dir.string());
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(texture_dir)) {
        if (isTextureImagePath(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

cv::Mat addYellowStylusBodyToHandMask(
    const cv::Mat& hand_stylus_mask,
    const cv::Mat& frame_bgr,
    const cv::Mat& board_mask) {
    if (envInt("SHADOW_HAND_MASK_INCLUDE_YELLOW_STYLUS", 1) == 0 ||
            hand_stylus_mask.empty() ||
            frame_bgr.empty() ||
            frame_bgr.channels() != 3 ||
            hand_stylus_mask.size() != frame_bgr.size()) {
        return hand_stylus_mask.clone();
    }

    cv::Mat hand;
    cv::threshold(hand_stylus_mask, hand, 0, 255, cv::THRESH_BINARY);

    cv::Mat prior;
    if (!board_mask.empty() && board_mask.size() == hand.size()) {
        cv::threshold(board_mask, prior, 0, 255, cv::THRESH_BINARY);
    } else {
        prior = cv::Mat(hand.size(), CV_8UC1, cv::Scalar(255));
    }

    cv::Mat core;
    cv::Mat yellow;
    cv::Mat dark;
    cv::Mat candidate = makeStylusRgbCandidateMask(frame_bgr, prior, &core, &yellow, &dark);
    if (candidate.empty() || cv::countNonZero(candidate) == 0) {
        return hand;
    }

    cv::Mat near_hand;
    const int near_kernel = normalizedOddMorphKernelSize(envInt("SHADOW_YELLOW_STYLUS_NEAR_HAND_KERNEL", 61));
    cv::dilate(
        hand,
        near_hand,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(near_kernel, near_kernel)),
        cv::Point(-1, -1),
        1);
    cv::bitwise_and(near_hand, prior, near_hand);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::Mat binary;
    cv::threshold(candidate, binary, 0, 255, cv::THRESH_BINARY);
    const int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    cv::Mat selected = cv::Mat::zeros(candidate.size(), CV_8UC1);
    const int min_area = std::clamp(envInt("SHADOW_YELLOW_STYLUS_MIN_AREA", 8), 1, 200);
    const int max_area = std::clamp(envInt("SHADOW_YELLOW_STYLUS_MAX_AREA", 6000), 100, 20000);
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_area || area > max_area) {
            continue;
        }
        cv::Mat component = labels == label;
        component.convertTo(component, CV_8UC1);
        cv::Mat overlap;
        cv::bitwise_and(component, near_hand, overlap);
        if (cv::countNonZero(overlap) == 0) {
            continue;
        }
        selected.setTo(255, component);
    }
    if (cv::countNonZero(selected) == 0) {
        return hand;
    }

    const int close_kernel = normalizedOddMorphKernelSize(envInt("SHADOW_YELLOW_STYLUS_CLOSE_KERNEL", 9));
    cv::morphologyEx(
        selected,
        selected,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close_kernel, close_kernel)),
        cv::Point(-1, -1),
        1);
    const int dilate_kernel = normalizedOddMorphKernelSize(envInt("SHADOW_YELLOW_STYLUS_DILATE_KERNEL", 5));
    cv::dilate(
        selected,
        selected,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_kernel, dilate_kernel)),
        cv::Point(-1, -1),
        1);
    cv::bitwise_and(selected, prior, selected);

    cv::Mat out;
    cv::bitwise_or(hand, selected, out);
    return out;
}

cv::Mat makeForegroundKeepMaskForCompositing(
    const cv::Mat& hand_stylus_mask,
    const Options& options,
    const cv::Mat& board_mask = cv::Mat(),
    cv::Mat* original_keep_mask = nullptr) {
    if (hand_stylus_mask.empty()) {
        if (original_keep_mask != nullptr) {
            original_keep_mask->release();
        }
        return cv::Mat();
    }

    cv::Mat keep_mask;
    cv::threshold(hand_stylus_mask, keep_mask, 0, 255, cv::THRESH_BINARY);
    if (!board_mask.empty() && board_mask.size() == keep_mask.size()) {
        cv::bitwise_and(keep_mask, board_mask, keep_mask);
    }

    // Historical behavior: expand the foreground keep-mask before copying the
    // raw hand/stylus back over the warped texture. The optional erosion below
    // is applied to this actual compositing mask, so disabling erosion preserves
    // the old output exactly.
    if (options.foreground_keep_mask_dilate_iterations > 0) {
        const int kernel_size = normalizedOddMorphKernelSize(
            options.foreground_keep_mask_dilate_kernel_size);
        cv::dilate(
            keep_mask,
            keep_mask,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size)),
            cv::Point(-1, -1),
            options.foreground_keep_mask_dilate_iterations);
    }
    if (!board_mask.empty() && board_mask.size() == keep_mask.size()) {
        cv::bitwise_and(keep_mask, board_mask, keep_mask);
    }

    if (original_keep_mask != nullptr) {
        *original_keep_mask = keep_mask.clone();
    }

    // This compensates for slight over-preservation of the original green/blue
    // tag surface around the hand/stylus boundary after texture replacement.
    // The default 7x7 ellipse erodes the foreground keep-mask by roughly 3 px,
    // matching the typical 3-5 px visible halo at the current processing size.
    //
    // At this point v3 has a combined hand+stylus mask, not a reliable separate
    // stylus mask. Keeping this configurable makes 5x5, 7x7, and 9x9 one-pass
    // tests easy while avoiding a hard-coded erosion that could damage thin
    // stylus regions in difficult frames.
    if (options.foreground_keep_mask_erode_enabled &&
            options.foreground_keep_mask_erode_iterations > 0) {
        const int kernel_size = normalizedOddMorphKernelSize(
            options.foreground_keep_mask_erode_kernel_size);
        cv::erode(
            keep_mask,
            keep_mask,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size)),
            cv::Point(-1, -1),
            options.foreground_keep_mask_erode_iterations);
    }
    if (!board_mask.empty() && board_mask.size() == keep_mask.size()) {
        cv::bitwise_and(keep_mask, board_mask, keep_mask);
    }
    if (envInt("SHADOW_FOREGROUND_KEEP_FILL_HOLES", 1) != 0) {
        keep_mask = fillBinaryMaskHoles(keep_mask);
        if (!board_mask.empty() && board_mask.size() == keep_mask.size()) {
            cv::bitwise_and(keep_mask, board_mask, keep_mask);
        }
    }

    return keep_mask;
}

cv::Mat makeSurfaceTextureOutput(
    const cv::Mat& frame_bgr,
    const cv::Mat& surface_texture_ref_bgr,
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& board_mask,
    const cv::Mat& foreground_keep_mask,
    cv::Mat* texture_only_bgr = nullptr) {
    cv::Mat board_render_mask;
    cv::threshold(board_mask, board_render_mask, 0, 255, cv::THRESH_BINARY);
    cv::morphologyEx(
        board_render_mask,
        board_render_mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    const int render_dilate = std::clamp(envInt("SHADOW_SURFACE_TEXTURE_BOARD_DILATE", 0), 0, 9);
    if (render_dilate > 0) {
        const int kernel_size = normalizedOddMorphKernelSize(render_dilate);
        cv::dilate(
            board_render_mask,
            board_render_mask,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size)),
            cv::Point(-1, -1),
            1);
    }
    const cv::Rect board_roi = cv::boundingRect(board_render_mask);

    cv::Mat occlusion_render_mask;
    cv::threshold(foreground_keep_mask, occlusion_render_mask, 0, 255, cv::THRESH_BINARY);
    cv::bitwise_and(occlusion_render_mask, board_render_mask, occlusion_render_mask);

    cv::Mat output = frame_bgr.clone();
    if (!board_roi.empty()) {
        cv::Mat roi_H = H_ref_to_frame.clone();
        cv::Mat translate = (cv::Mat_<double>(3, 3) <<
            1.0, 0.0, -static_cast<double>(board_roi.x),
            0.0, 1.0, -static_cast<double>(board_roi.y),
            0.0, 0.0, 1.0);
        roi_H = translate * roi_H;
        cv::Mat warped_texture_roi;
        cv::warpPerspective(
            surface_texture_ref_bgr,
            warped_texture_roi,
            roi_H,
            board_roi.size(),
            cv::INTER_LINEAR,
            cv::BORDER_REFLECT_101);
        warped_texture_roi.copyTo(output(board_roi), board_render_mask(board_roi));
        if (texture_only_bgr != nullptr) {
            *texture_only_bgr = output.clone();
        }
        frame_bgr(board_roi).copyTo(output(board_roi), occlusion_render_mask(board_roi));
    } else if (texture_only_bgr != nullptr) {
        *texture_only_bgr = output.clone();
    }
    return output;
}

cv::Mat suppressGreenBlueMaskEdgeFringe(
    const cv::Mat& composited_bgr,
    const cv::Mat& texture_only_bgr,
    const cv::Mat& keep_mask,
    const cv::Mat& board_mask,
    cv::Mat* cleanup_candidate_mask = nullptr) {
    if (cleanup_candidate_mask != nullptr) {
        cleanup_candidate_mask->release();
    }
    if (composited_bgr.empty() ||
            texture_only_bgr.empty() ||
            keep_mask.empty() ||
            composited_bgr.size() != texture_only_bgr.size() ||
            composited_bgr.type() != CV_8UC3 ||
            texture_only_bgr.type() != CV_8UC3) {
        return composited_bgr.clone();
    }
    if (envInt("SHADOW_EDGE_CHROMA_CLEANUP_ENABLE", 1) == 0) {
        return composited_bgr.clone();
    }

    cv::Mat keep;
    cv::threshold(keep_mask, keep, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(keep) == 0) {
        return composited_bgr.clone();
    }

    const int dilate_px = std::clamp(envInt("SHADOW_EDGE_CHROMA_CLEANUP_DILATE", 7), 1, 31) | 1;
    const int erode_px = std::clamp(envInt("SHADOW_EDGE_CHROMA_CLEANUP_ERODE", 5), 1, 31) | 1;
    const int candidate_dilate_px = std::clamp(envInt("SHADOW_EDGE_CHROMA_CLEANUP_CANDIDATE_DILATE", 3), 0, 15);
    const int alpha_blur_px = std::clamp(envInt("SHADOW_EDGE_CHROMA_CLEANUP_ALPHA_BLUR", 3), 0, 15);
    const float strength = static_cast<float>(
        std::clamp(envInt("SHADOW_EDGE_CHROMA_CLEANUP_STRENGTH", 85), 0, 100)) / 100.0f;
    const bool inner_only = envInt("SHADOW_EDGE_CHROMA_CLEANUP_INNER_ONLY", 1) != 0;

    cv::Mat dilated;
    cv::Mat eroded;
    cv::dilate(
        keep,
        dilated,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_px, dilate_px)),
        cv::Point(-1, -1),
        1);
    cv::erode(
        keep,
        eroded,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_px, erode_px)),
        cv::Point(-1, -1),
        1);
    cv::Mat edge_band;
    cv::absdiff(dilated, eroded, edge_band);
    if (inner_only) {
        cv::bitwise_and(edge_band, keep, edge_band);
    }
    if (!board_mask.empty() && board_mask.size() == edge_band.size()) {
        cv::Mat board;
        cv::threshold(board_mask, board, 0, 255, cv::THRESH_BINARY);
        cv::bitwise_and(edge_band, board, edge_band);
    }
    if (cv::countNonZero(edge_band) == 0) {
        return composited_bgr.clone();
    }

    cv::Mat blue = makeBlueMask(composited_bgr);
    cv::Mat green = makeGreenMask(composited_bgr);
    cv::Mat green_blue;
    cv::bitwise_or(blue, green, green_blue);

    cv::Mat candidate;
    cv::bitwise_and(edge_band, green_blue, candidate);
    if (candidate_dilate_px > 0 && cv::countNonZero(candidate) > 0) {
        const int k = candidate_dilate_px | 1;
        cv::dilate(
            candidate,
            candidate,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)),
            cv::Point(-1, -1),
            1);
        cv::bitwise_and(candidate, edge_band, candidate);
    }
    if (cleanup_candidate_mask != nullptr) {
        *cleanup_candidate_mask = candidate.clone();
    }
    if (cv::countNonZero(candidate) == 0 || strength <= 0.0f) {
        return composited_bgr.clone();
    }

    cv::Mat alpha;
    candidate.convertTo(alpha, CV_32F, 1.0 / 255.0);
    if (alpha_blur_px > 0) {
        const int k = alpha_blur_px | 1;
        cv::GaussianBlur(alpha, alpha, cv::Size(k, k), 0.0);
        cv::Mat edge_f;
        edge_band.convertTo(edge_f, CV_32F, 1.0 / 255.0);
        alpha = alpha.mul(edge_f);
    }
    cv::Mat changed_mask;
    cv::threshold(alpha, changed_mask, 1e-6, 255.0, cv::THRESH_BINARY);
    changed_mask.convertTo(changed_mask, CV_8UC1);
    cv::Rect changed_roi = cv::boundingRect(changed_mask);
    if (changed_roi.empty()) {
        return composited_bgr.clone();
    }

    cv::Mat out_lab;
    cv::Mat tex_lab;
    cv::cvtColor(composited_bgr(changed_roi), out_lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(texture_only_bgr(changed_roi), tex_lab, cv::COLOR_BGR2Lab);
    cv::Mat alpha_roi = alpha(changed_roi);
    cv::parallel_for_(cv::Range(0, out_lab.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            cv::Vec3b* out_row = out_lab.ptr<cv::Vec3b>(y);
            const cv::Vec3b* tex_row = tex_lab.ptr<cv::Vec3b>(y);
            const float* alpha_row = alpha_roi.ptr<float>(y);
            for (int x = 0; x < out_lab.cols; ++x) {
                const float a = std::clamp(alpha_row[x] * strength, 0.0f, 1.0f);
                if (a <= 0.0f) {
                    continue;
                }
                out_row[x][1] = cv::saturate_cast<uchar>(
                    std::lround((1.0f - a) * static_cast<float>(out_row[x][1]) + a * static_cast<float>(tex_row[x][1])));
                out_row[x][2] = cv::saturate_cast<uchar>(
                    std::lround((1.0f - a) * static_cast<float>(out_row[x][2]) + a * static_cast<float>(tex_row[x][2])));
            }
        }
    });

    cv::Mat adjusted_bgr;
    cv::cvtColor(out_lab, adjusted_bgr, cv::COLOR_Lab2BGR);
    cv::Mat out_bgr = composited_bgr.clone();
    adjusted_bgr.copyTo(out_bgr(changed_roi), changed_mask(changed_roi));
    return out_bgr;
}

cv::Mat makeAlignedRefMaskedDifference(
    const cv::Mat& frame_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& board_mask,
    const cv::Mat& hand_stylus_mask) {
    cv::Mat difference(frame_bgr.size(), frame_bgr.type(), cv::Scalar(0, 0, 0));
    if (!board_mask.empty() && board_mask.size() == frame_bgr.size()) {
        cv::Mat board_binary;
        cv::threshold(board_mask, board_binary, 0, 255, cv::THRESH_BINARY);
        const cv::Rect roi = cv::boundingRect(board_binary);
        if (!roi.empty()) {
            cv::Mat local_difference;
            cv::absdiff(frame_bgr(roi), aligned_ref_bgr(roi), local_difference);
            local_difference.setTo(cv::Scalar(0, 0, 0), board_binary(roi) == 0);
            local_difference.copyTo(difference(roi));
        }
    } else {
        cv::absdiff(frame_bgr, aligned_ref_bgr, difference);
    }
    if (!hand_stylus_mask.empty()) {
        cv::Mat exclusion;
        cv::threshold(hand_stylus_mask, exclusion, 0, 255, cv::THRESH_BINARY);
        difference.setTo(cv::Scalar(0, 0, 0), exclusion);
    }
    return difference;
}

cv::Mat makeTagMaskFromColoredAlignedRef(const cv::Mat& aligned_ref_bgr) {
    cv::Mat tag_mask(aligned_ref_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, aligned_ref_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* row = aligned_ref_bgr.ptr<cv::Vec3b>(y);
            uchar* mask_row = tag_mask.ptr<uchar>(y);
            for (int x = 0; x < aligned_ref_bgr.cols; ++x) {
                int b = row[x][0];
                int g = row[x][1];
                int r = row[x][2];
                if (b > g + 20 && b > r + 20) {
                    mask_row[x] = 255;
                }
            }
        }
    });
    cv::dilate(
        tag_mask,
        tag_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17)),
        cv::Point(-1, -1),
        1);
    return tag_mask;
}

cv::Mat fillMaskedByNeighborAverages(
    const cv::Mat& image_bgr,
    const cv::Mat& fill_mask,
    const cv::Mat& valid_source_mask) {
    cv::Mat fill_binary;
    cv::threshold(fill_mask, fill_binary, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(fill_binary) == 0) {
        return image_bgr.clone();
    }

    const int max_fill_iterations = fastApproxEnabled()
        ? std::clamp(envInt("SHADOW_FAST_FILL_ITERS", 40), 1, 80)
        : 80;
    cv::Rect fill_bounds = cv::boundingRect(fill_binary);
    cv::Rect work_roi(
        std::max(0, fill_bounds.x - max_fill_iterations),
        std::max(0, fill_bounds.y - max_fill_iterations),
        std::min(image_bgr.cols, fill_bounds.x + fill_bounds.width + max_fill_iterations) -
            std::max(0, fill_bounds.x - max_fill_iterations),
        std::min(image_bgr.rows, fill_bounds.y + fill_bounds.height + max_fill_iterations) -
            std::max(0, fill_bounds.y - max_fill_iterations));

    cv::Mat image_roi = image_bgr(work_roi);
    cv::Mat fill_binary_roi = fill_binary(work_roi);
    cv::Mat valid_source_roi;
    if (!valid_source_mask.empty()) {
        valid_source_roi = valid_source_mask(work_roi);
    }

    cv::Mat filled_f;
    image_roi.convertTo(filled_f, CV_32FC3);

    cv::Mat valid_u8;
    if (valid_source_roi.empty()) {
        cv::bitwise_not(fill_binary_roi, valid_u8);
    } else {
        cv::threshold(valid_source_roi, valid_u8, 0, 255, cv::THRESH_BINARY);
        valid_u8.setTo(0, fill_binary_roi);
    }
    cv::Mat valid_f;
    valid_u8.convertTo(valid_f, CV_32F, 1.0 / 255.0);
    cv::Mat unknown = fill_binary_roi.clone();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat expanded_valid;
    cv::Mat candidates;
    cv::Mat weighted = cv::Mat::zeros(filled_f.size(), filled_f.type());
    filled_f.copyTo(weighted, valid_u8);
    cv::Mat sum_bgr;
    cv::Mat weight_sum;
    std::vector<cv::Point> candidate_points;
    int unknown_remaining = cv::countNonZero(unknown);
    for (int iter = 0; iter < max_fill_iterations && unknown_remaining > 0; ++iter) {
        cv::dilate(valid_u8, expanded_valid, kernel, cv::Point(-1, -1), 1);
        cv::bitwise_and(expanded_valid, unknown, candidates);
        cv::findNonZero(candidates, candidate_points);
        if (candidate_points.empty()) {
            break;
        }
        const int candidate_count = static_cast<int>(candidate_points.size());

        cv::boxFilter(weighted, sum_bgr, -1, cv::Size(3, 3), cv::Point(-1, -1), false, cv::BORDER_REPLICATE);

        cv::boxFilter(valid_f, weight_sum, -1, cv::Size(3, 3), cv::Point(-1, -1), false, cv::BORDER_REPLICATE);

        for (const cv::Point& point : candidate_points) {
            const float weight = weight_sum.at<float>(point);
            if (weight > 1e-6f) {
                const cv::Vec3f filled_value = sum_bgr.at<cv::Vec3f>(point) * (1.0f / weight);
                filled_f.at<cv::Vec3f>(point) = filled_value;
                weighted.at<cv::Vec3f>(point) = filled_value;
            }
        }

        valid_u8.setTo(255, candidates);
        valid_f.setTo(1.0f, candidates);
        unknown.setTo(0, candidates);
        unknown_remaining -= candidate_count;
    }

    cv::Mat filled_bgr;
    filled_f.convertTo(filled_bgr, CV_8UC3);
    cv::Mat out = image_bgr.clone();
    filled_bgr.copyTo(out(work_roi));
    return out;
}

cv::Mat makeCleanAlignedRefMaskedDifferenceFull(
    const cv::Mat& aligned_ref_masked_difference_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& hand_stylus_mask) {
    if (aligned_ref_masked_difference_bgr.empty() || aligned_ref_bgr.empty()) {
        return cv::Mat();
    }

    cv::Mat surface_mask = makeSurfaceMaskFromAlignedRef(aligned_ref_bgr);
    cv::Mat tag_mask = makeTagMaskFromColoredAlignedRef(aligned_ref_bgr);
    cv::bitwise_and(tag_mask, surface_mask, tag_mask);

    cv::Mat inner_surface_mask;
    cv::erode(
        surface_mask,
        inner_surface_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)),
        cv::Point(-1, -1),
        1);
    cv::Mat surface_border_mask;
    cv::subtract(surface_mask, inner_surface_mask, surface_border_mask);

    cv::Mat fill_mask;
    cv::bitwise_or(tag_mask, surface_border_mask, fill_mask);

    cv::Mat source_mask = surface_mask.clone();
    cv::Mat hand_mask;
    if (!hand_stylus_mask.empty()) {
        cv::threshold(hand_stylus_mask, hand_mask, 0, 255, cv::THRESH_BINARY);
        fill_mask.setTo(0, hand_mask);
        source_mask.setTo(0, hand_mask);
    }

    cv::Mat clean_bgr = fillMaskedByNeighborAverages(
        aligned_ref_masked_difference_bgr,
        fill_mask,
        source_mask);
    cv::Mat blur_mask;
    cv::dilate(
        fill_mask,
        blur_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)),
        cv::Point(-1, -1),
        1);
    cv::bitwise_and(blur_mask, surface_mask, blur_mask);
    if (!hand_mask.empty()) {
        blur_mask.setTo(0, hand_mask);
    }

    cv::Mat out = cv::Mat::zeros(clean_bgr.size(), clean_bgr.type());
    clean_bgr.copyTo(out, surface_mask);
    if (fastApproxEnabled()) {
        cv::Rect blur_bounds = cv::boundingRect(blur_mask);
        if (!blur_bounds.empty()) {
            constexpr int blur_radius = 4;
            cv::Rect source_roi(
                std::max(0, blur_bounds.x - blur_radius),
                std::max(0, blur_bounds.y - blur_radius),
                std::min(clean_bgr.cols, blur_bounds.x + blur_bounds.width + blur_radius) -
                    std::max(0, blur_bounds.x - blur_radius),
                std::min(clean_bgr.rows, blur_bounds.y + blur_bounds.height + blur_radius) -
                    std::max(0, blur_bounds.y - blur_radius));

            cv::Mat blurred_roi;
            cv::GaussianBlur(clean_bgr(source_roi), blurred_roi, cv::Size(9, 9), 0.0);
            cv::Rect local_blur_bounds(
                blur_bounds.x - source_roi.x,
                blur_bounds.y - source_roi.y,
                blur_bounds.width,
                blur_bounds.height);
            blurred_roi(local_blur_bounds).copyTo(out(blur_bounds), blur_mask(blur_bounds));
        }
    } else {
        cv::Mat blurred_bgr;
        cv::GaussianBlur(clean_bgr, blurred_bgr, cv::Size(9, 9), 0.0);
        blurred_bgr.copyTo(out, blur_mask);
    }
    if (!hand_mask.empty()) {
        out.setTo(cv::Scalar(0, 0, 0), hand_mask);
    }
    return out;
}

cv::Mat makeCleanAlignedRefMaskedDifference(
    const cv::Mat& aligned_ref_masked_difference_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& hand_stylus_mask) {
    if (aligned_ref_masked_difference_bgr.empty() || aligned_ref_bgr.empty()) {
        return cv::Mat();
    }
    if (envInt("SHADOW_CLEAN_DIFF_ROI_ENABLE", 1) == 0 ||
            aligned_ref_masked_difference_bgr.size() != aligned_ref_bgr.size()) {
        return makeCleanAlignedRefMaskedDifferenceFull(
            aligned_ref_masked_difference_bgr,
            aligned_ref_bgr,
            hand_stylus_mask);
    }

    cv::Mat gray;
    cv::cvtColor(aligned_ref_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat nonzero;
    cv::threshold(gray, nonzero, 5, 255, cv::THRESH_BINARY);
    cv::Rect roi = cv::boundingRect(nonzero);
    if (roi.empty()) {
        return cv::Mat::zeros(aligned_ref_masked_difference_bgr.size(), aligned_ref_masked_difference_bgr.type());
    }

    const int pad = std::clamp(envInt("SHADOW_CLEAN_DIFF_ROI_PAD", 16), 16, 160);
    const cv::Rect image_rect(0, 0, aligned_ref_bgr.cols, aligned_ref_bgr.rows);
    roi.x -= pad;
    roi.y -= pad;
    roi.width += pad * 2;
    roi.height += pad * 2;
    roi &= image_rect;
    if (roi == image_rect) {
        return makeCleanAlignedRefMaskedDifferenceFull(
            aligned_ref_masked_difference_bgr,
            aligned_ref_bgr,
            hand_stylus_mask);
    }

    cv::Mat hand_roi;
    if (!hand_stylus_mask.empty() && hand_stylus_mask.size() == aligned_ref_bgr.size()) {
        hand_roi = hand_stylus_mask(roi);
    }
    cv::Mat clean_roi = makeCleanAlignedRefMaskedDifferenceFull(
        aligned_ref_masked_difference_bgr(roi),
        aligned_ref_bgr(roi),
        hand_roi);
    cv::Mat out = cv::Mat::zeros(aligned_ref_masked_difference_bgr.size(), aligned_ref_masked_difference_bgr.type());
    clean_roi.copyTo(out(roi));
    return out;
}

cv::Mat toFloat01Bgr(const cv::Mat& image_bgr) {
    cv::Mat out;
    if (image_bgr.depth() == CV_8U) {
        image_bgr.convertTo(out, CV_32FC3, 1.0 / 255.0);
    } else {
        image_bgr.convertTo(out, CV_32FC3);
        cv::min(out, cv::Scalar(1.0, 1.0, 1.0), out);
        cv::max(out, cv::Scalar(0.0, 0.0, 0.0), out);
    }
    return out;
}

float srgbToLinearScalar(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

float linearToSrgbScalar(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x <= 0.0031308f ? 12.92f * x : (1.055f * std::pow(x, 1.0f / 2.4f)) - 0.055f;
}

cv::Mat srgbToLinearBgr(const cv::Mat& srgb_bgr) {
    cv::Mat linear(srgb_bgr.size(), CV_32FC3);
    cv::parallel_for_(cv::Range(0, srgb_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* src_row = srgb_bgr.ptr<cv::Vec3f>(y);
            cv::Vec3f* dst_row = linear.ptr<cv::Vec3f>(y);
            for (int x = 0; x < srgb_bgr.cols; ++x) {
                dst_row[x] = cv::Vec3f(
                    srgbToLinearScalar(src_row[x][0]),
                    srgbToLinearScalar(src_row[x][1]),
                    srgbToLinearScalar(src_row[x][2]));
            }
        }
    });
    return linear;
}

const std::array<float, 256>& srgbU8ToLinearFastLut() {
    static const std::array<float, 256> lut = [] {
        std::array<float, 256> values{};
        for (int i = 0; i < 256; ++i) {
            values[static_cast<size_t>(i)] = srgbToLinearScalar(static_cast<float>(i) / 255.0f);
        }
        return values;
    }();
    return lut;
}

cv::Mat srgbU8ToLinearFastBgr(const cv::Mat& srgb_bgr) {
    const auto& lut = srgbU8ToLinearFastLut();
    cv::Mat linear(srgb_bgr.size(), CV_32FC3);
    cv::parallel_for_(cv::Range(0, srgb_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* src_row = srgb_bgr.ptr<cv::Vec3b>(y);
            cv::Vec3f* dst_row = linear.ptr<cv::Vec3f>(y);
            for (int x = 0; x < srgb_bgr.cols; ++x) {
                dst_row[x] = cv::Vec3f(
                    lut[src_row[x][0]],
                    lut[src_row[x][1]],
                    lut[src_row[x][2]]);
            }
        }
    });
    return linear;
}

cv::Mat maxChannelFloat01BgrU8(const cv::Mat& image_bgr) {
    cv::Mat score(image_bgr.size(), CV_32F);
    constexpr float inv255 = 1.0f / 255.0f;
    cv::parallel_for_(cv::Range(0, image_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* src_row = image_bgr.ptr<cv::Vec3b>(y);
            float* dst_row = score.ptr<float>(y);
            for (int x = 0; x < image_bgr.cols; ++x) {
                dst_row[x] = static_cast<float>(std::max({src_row[x][0], src_row[x][1], src_row[x][2]})) * inv255;
            }
        }
    });
    return score;
}

cv::Mat linearToSrgbBgrU8(const cv::Mat& linear_bgr) {
    cv::Mat srgb(linear_bgr.size(), CV_8UC3);
    cv::parallel_for_(cv::Range(0, linear_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* src_row = linear_bgr.ptr<cv::Vec3f>(y);
            cv::Vec3b* dst_row = srgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < linear_bgr.cols; ++x) {
                dst_row[x] = cv::Vec3b(
                    cv::saturate_cast<uchar>(std::lround(linearToSrgbScalar(src_row[x][0]) * 255.0f)),
                    cv::saturate_cast<uchar>(std::lround(linearToSrgbScalar(src_row[x][1]) * 255.0f)),
                    cv::saturate_cast<uchar>(std::lround(linearToSrgbScalar(src_row[x][2]) * 255.0f)));
            }
        }
    });
    return srgb;
}

cv::Mat luminanceLinearBgr(const cv::Mat& linear_bgr) {
    cv::Mat luminance(linear_bgr.size(), CV_32F);
    cv::parallel_for_(cv::Range(0, linear_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* src_row = linear_bgr.ptr<cv::Vec3f>(y);
            float* dst_row = luminance.ptr<float>(y);
            for (int x = 0; x < linear_bgr.cols; ++x) {
                dst_row[x] = (0.0722f * src_row[x][0]) + (0.7152f * src_row[x][1]) + (0.2126f * src_row[x][2]);
            }
        }
    });
    return luminance;
}

cv::Mat divideWeightedBlurNumerator(const cv::Mat& numerator, const cv::Mat& denominator) {
    cv::Mat out(numerator.size(), CV_32F, cv::Scalar(0.0f));
    cv::parallel_for_(cv::Range(0, numerator.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* num_row = numerator.ptr<float>(y);
            const float* den_row = denominator.ptr<float>(y);
            float* out_row = out.ptr<float>(y);
            for (int x = 0; x < numerator.cols; ++x) {
                out_row[x] = num_row[x] / (den_row[x] + 1e-6f);
            }
        }
    });
    return out;
}

cv::Mat weightedBlurFloat(const cv::Mat& value, const cv::Mat& weight, double sigma) {
    cv::Mat weighted = value.mul(weight);
    cv::Mat numerator;
    cv::GaussianBlur(weighted, numerator, cv::Size(0, 0), sigma);
    cv::Mat denominator;
    cv::GaussianBlur(weight, denominator, cv::Size(0, 0), sigma);
    return divideWeightedBlurNumerator(numerator, denominator);
}

cv::Mat weightedBlurFloatApproxDownsample(const cv::Mat& value, const cv::Mat& weight, double sigma, double scale) {
    cv::Size small_size(
        std::max(1, static_cast<int>(std::lround(value.cols * scale))),
        std::max(1, static_cast<int>(std::lround(value.rows * scale))));
    cv::Mat value_small;
    cv::Mat weight_small;
    cv::resize(value, value_small, small_size, 0.0, 0.0, cv::INTER_AREA);
    cv::resize(weight, weight_small, small_size, 0.0, 0.0, cv::INTER_AREA);
    cv::Mat blurred_small = weightedBlurFloat(value_small, weight_small, std::max(0.1, sigma * scale));
    cv::Mat out;
    cv::resize(blurred_small, out, value.size(), 0.0, 0.0, cv::INTER_LINEAR);
    return out;
}

void clipFloatMat(cv::Mat* mat, float low, float high) {
    cv::max(*mat, cv::Scalar(low), *mat);
    cv::min(*mat, cv::Scalar(high), *mat);
}

cv::Mat computeShadowFactorBgrFull(
    const cv::Mat& working_src_bgr,
    const cv::Mat& raw_src_bgr,
    const cv::Mat& clean_difference_bgr,
    float mask_low = 0.02f,
    float mask_high = 0.35f,
    double blur_sigma_mask = 9.0,
    double blur_sigma_field = 15.0,
    float tint_strength = 0.12f,
    float darkness_min = 0.35f,
    float darkness_max = 1.0f,
    bool fast_approx = fastApproxEnabled()) {
    cv::Mat working_lin;
    cv::Mat raw_lin;
    cv::Mat score;

    if (fast_approx &&
            working_src_bgr.depth() == CV_8U &&
            raw_src_bgr.depth() == CV_8U &&
            clean_difference_bgr.depth() == CV_8U) {
        working_lin = srgbU8ToLinearFastBgr(working_src_bgr);
        raw_lin = srgbU8ToLinearFastBgr(raw_src_bgr);
        score = maxChannelFloat01BgrU8(clean_difference_bgr);
    } else {
        cv::Mat working = toFloat01Bgr(working_src_bgr);
        cv::Mat raw = toFloat01Bgr(raw_src_bgr);
        cv::Mat clean_diff = toFloat01Bgr(clean_difference_bgr);
        working_lin = srgbToLinearBgr(working);
        raw_lin = srgbToLinearBgr(raw);
        score = cv::Mat(clean_diff.size(), CV_32F);
        cv::parallel_for_(cv::Range(0, clean_diff.rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const cv::Vec3f* diff_row = clean_diff.ptr<cv::Vec3f>(y);
                float* score_row = score.ptr<float>(y);
                for (int x = 0; x < clean_diff.cols; ++x) {
                    score_row[x] = std::max({diff_row[x][0], diff_row[x][1], diff_row[x][2]});
                }
            }
        });
    }

    cv::Mat mask(score.size(), CV_32F);
    float mask_range = std::max(1e-6f, mask_high - mask_low);
    cv::parallel_for_(cv::Range(0, score.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* score_row = score.ptr<float>(y);
            float* mask_row = mask.ptr<float>(y);
            for (int x = 0; x < score.cols; ++x) {
                mask_row[x] = std::clamp((score_row[x] - mask_low) / mask_range, 0.0f, 1.0f);
            }
        }
    });

    cv::Mat hard;
    cv::threshold(score, hard, mask_low, 255.0, cv::THRESH_BINARY);
    hard.convertTo(hard, CV_8UC1);
    hard = largestComponentMask(hard, 1);
    cv::dilate(
        hard,
        hard,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)),
        cv::Point(-1, -1),
        1);
    cv::parallel_for_(cv::Range(0, mask.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const uchar* hard_row = hard.ptr<uchar>(y);
            float* mask_row = mask.ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (hard_row[x] == 0) {
                    mask_row[x] = 0.0f;
                }
            }
        }
    });
    cv::GaussianBlur(mask, mask, cv::Size(0, 0), blur_sigma_mask);
    clipFloatMat(&mask, 0.0f, 1.0f);

    cv::Mat lum_working = luminanceLinearBgr(working_lin);
    cv::Mat lum_raw = luminanceLinearBgr(raw_lin);
    cv::Mat darkness(score.size(), CV_32F);
    cv::parallel_for_(cv::Range(0, score.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* lum_w_row = lum_working.ptr<float>(y);
            const float* lum_r_row = lum_raw.ptr<float>(y);
            float* darkness_row = darkness.ptr<float>(y);
            for (int x = 0; x < score.cols; ++x) {
                darkness_row[x] = std::clamp(lum_r_row[x] / std::max(lum_w_row[x], 1e-4f), darkness_min, darkness_max);
            }
        }
    });
    darkness = fast_approx
        ? weightedBlurFloatApproxDownsample(darkness, mask, blur_sigma_field, 0.35)
        : weightedBlurFloat(darkness, mask, blur_sigma_field);
    clipFloatMat(&darkness, darkness_min, darkness_max);

    cv::Mat ratio_rgb(raw_lin.size(), CV_32FC3);
    cv::parallel_for_(cv::Range(0, raw_lin.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* raw_row = raw_lin.ptr<cv::Vec3f>(y);
            const cv::Vec3f* working_row = working_lin.ptr<cv::Vec3f>(y);
            cv::Vec3f* ratio_row = ratio_rgb.ptr<cv::Vec3f>(y);
            for (int x = 0; x < raw_lin.cols; ++x) {
                ratio_row[x] = cv::Vec3f(
                    raw_row[x][0] / std::max(working_row[x][0], 1e-4f),
                    raw_row[x][1] / std::max(working_row[x][1], 1e-4f),
                    raw_row[x][2] / std::max(working_row[x][2], 1e-4f));
            }
        }
    });
    cv::Mat ratio_lum = luminanceLinearBgr(ratio_rgb);
    cv::Mat tint(ratio_rgb.size(), CV_32FC3);
    cv::parallel_for_(cv::Range(0, ratio_rgb.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* ratio_row = ratio_rgb.ptr<cv::Vec3f>(y);
            const float* lum_row = ratio_lum.ptr<float>(y);
            cv::Vec3f* tint_row = tint.ptr<cv::Vec3f>(y);
            for (int x = 0; x < ratio_rgb.cols; ++x) {
                float inv_lum = 1.0f / std::max(lum_row[x], 1e-4f);
                tint_row[x] = ratio_row[x] * inv_lum;
            }
        }
    });

    std::vector<cv::Mat> tint_channels;
    cv::split(tint, tint_channels);
    for (cv::Mat& channel : tint_channels) {
        channel = fast_approx
            ? weightedBlurFloatApproxDownsample(channel, mask, blur_sigma_field, 0.35)
            : weightedBlurFloat(channel, mask, blur_sigma_field);
    }
    cv::Mat tint_smooth;
    cv::merge(tint_channels, tint_smooth);

    cv::Mat factor(tint_smooth.size(), CV_32FC3);
    cv::parallel_for_(cv::Range(0, factor.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* tint_row = tint_smooth.ptr<cv::Vec3f>(y);
            const float* mask_row = mask.ptr<float>(y);
            const float* darkness_row = darkness.ptr<float>(y);
            cv::Vec3f* factor_row = factor.ptr<cv::Vec3f>(y);
            for (int x = 0; x < factor.cols; ++x) {
                cv::Vec3f tint_pixel;
                for (int c = 0; c < 3; ++c) {
                    float clipped_tint = std::clamp(tint_row[x][c], 0.8f, 1.25f);
                    tint_pixel[c] = 1.0f + (tint_strength * (clipped_tint - 1.0f));
                }
                cv::Vec3f shadow_multiplier;
                for (int c = 0; c < 3; ++c) {
                    shadow_multiplier[c] = std::clamp(darkness_row[x] * tint_pixel[c], darkness_min, 1.05f);
                }
                float m = mask_row[x];
                factor_row[x] = cv::Vec3f(
                    (1.0f - m) + (m * shadow_multiplier[0]),
                    (1.0f - m) + (m * shadow_multiplier[1]),
                    (1.0f - m) + (m * shadow_multiplier[2]));
            }
        }
    });

    return factor;
}

cv::Mat computeShadowFactorBgr(
    const cv::Mat& working_src_bgr,
    const cv::Mat& raw_src_bgr,
    const cv::Mat& clean_difference_bgr,
    float mask_low = 0.02f,
    float mask_high = 0.35f,
    double blur_sigma_mask = 9.0,
    double blur_sigma_field = 15.0,
    float tint_strength = 0.12f,
    float darkness_min = 0.35f,
    float darkness_max = 1.0f,
    bool fast_approx = fastApproxEnabled()) {
    if (envInt("SHADOW_FACTOR_ROI_ENABLE", 1) == 0 ||
            working_src_bgr.empty() ||
            raw_src_bgr.empty() ||
            clean_difference_bgr.empty() ||
            working_src_bgr.size() != raw_src_bgr.size() ||
            working_src_bgr.size() != clean_difference_bgr.size()) {
        return computeShadowFactorBgrFull(
            working_src_bgr,
            raw_src_bgr,
            clean_difference_bgr,
            mask_low,
            mask_high,
            blur_sigma_mask,
            blur_sigma_field,
            tint_strength,
            darkness_min,
            darkness_max,
            fast_approx);
    }

    cv::Mat active;
    if (clean_difference_bgr.channels() == 1) {
        cv::threshold(clean_difference_bgr, active, 0, 255, cv::THRESH_BINARY);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(clean_difference_bgr, channels);
        cv::max(channels[0], channels[1], active);
        cv::max(active, channels[2], active);
        cv::threshold(active, active, 0, 255, cv::THRESH_BINARY);
    }
    cv::Rect roi = cv::boundingRect(active);
    if (roi.empty()) {
        return cv::Mat(working_src_bgr.size(), CV_32FC3, cv::Scalar(1.0f, 1.0f, 1.0f));
    }
    const int pad = std::clamp(envInt("SHADOW_FACTOR_ROI_PAD", 96), 32, 240);
    const cv::Rect image_rect(0, 0, working_src_bgr.cols, working_src_bgr.rows);
    roi.x -= pad;
    roi.y -= pad;
    roi.width += pad * 2;
    roi.height += pad * 2;
    roi &= image_rect;
    if (roi == image_rect) {
        return computeShadowFactorBgrFull(
            working_src_bgr,
            raw_src_bgr,
            clean_difference_bgr,
            mask_low,
            mask_high,
            blur_sigma_mask,
            blur_sigma_field,
            tint_strength,
            darkness_min,
            darkness_max,
            fast_approx);
    }

    cv::Mat factor = cv::Mat(working_src_bgr.size(), CV_32FC3, cv::Scalar(1.0f, 1.0f, 1.0f));
    cv::Mat factor_roi = computeShadowFactorBgrFull(
        working_src_bgr(roi),
        raw_src_bgr(roi),
        clean_difference_bgr(roi),
        mask_low,
        mask_high,
        blur_sigma_mask,
        blur_sigma_field,
        tint_strength,
        darkness_min,
        darkness_max,
        fast_approx);
    factor_roi.copyTo(factor(roi));
    return factor;
}

cv::Mat applyShadowFactorToBaseBgr(
    const cv::Mat& new_base_bgr,
    const cv::Mat& shadow_factor_bgr,
    double shadow_strength,
    bool fast_approx) {
    cv::Mat base_lin = fast_approx && new_base_bgr.depth() == CV_8U
        ? srgbU8ToLinearFastBgr(new_base_bgr)
        : srgbToLinearBgr(toFloat01Bgr(new_base_bgr));

    cv::Mat result_lin(base_lin.size(), CV_32FC3);
    const float strength = static_cast<float>(std::max(0.0, shadow_strength));
    cv::parallel_for_(cv::Range(0, base_lin.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* base_row = base_lin.ptr<cv::Vec3f>(y);
            const cv::Vec3f* factor_row = shadow_factor_bgr.ptr<cv::Vec3f>(y);
            cv::Vec3f* out_row = result_lin.ptr<cv::Vec3f>(y);
            for (int x = 0; x < base_lin.cols; ++x) {
                cv::Vec3f factor;
                for (int c = 0; c < 3; ++c) {
                    factor[c] = std::clamp(1.0f + ((factor_row[x][c] - 1.0f) * strength), 0.0f, 1.25f);
                }
                out_row[x] = cv::Vec3f(
                    base_row[x][0] * factor[0],
                    base_row[x][1] * factor[1],
                    base_row[x][2] * factor[2]);
            }
        }
    });
    return linearToSrgbBgrU8(result_lin);
}

cv::Mat applyShadowFactorToBaseBgr(
    const cv::Mat& new_base_bgr,
    const cv::Mat& shadow_factor_bgr,
    double shadow_strength = 1.0) {
    const bool fast_approx = fastApproxEnabled() &&
        new_base_bgr.depth() == CV_8U &&
        shadow_factor_bgr.depth() == CV_32F;
    return applyShadowFactorToBaseBgr(new_base_bgr, shadow_factor_bgr, shadow_strength, fast_approx);
}

cv::Mat transferableShadowOverlayBgr(
    const cv::Mat& working_src_bgr,
    const cv::Mat& raw_src_bgr,
    const cv::Mat& clean_difference_bgr,
    const cv::Mat& new_base_bgr) {
    const bool fast_approx = fastApproxEnabled() &&
        working_src_bgr.depth() == CV_8U &&
        raw_src_bgr.depth() == CV_8U &&
        clean_difference_bgr.depth() == CV_8U &&
        new_base_bgr.depth() == CV_8U;
    cv::Mat shadow_factor_bgr = computeShadowFactorBgr(
        working_src_bgr,
        raw_src_bgr,
        clean_difference_bgr,
        0.02f,
        0.35f,
        9.0,
        15.0,
        0.12f,
        0.35f,
        1.0f,
        fast_approx);
    return applyShadowFactorToBaseBgr(new_base_bgr, shadow_factor_bgr, 1.0, fast_approx);
}

cv::Mat makeHandPenMaskFromColormatchDifference(
    const cv::Mat& difference_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& observed_bgr,
    const cv::Mat& projected_tag_protection_mask,
    double* otsu_threshold,
    int* used_threshold) {
    cv::Mat surface = makeSurfaceMaskFromAlignedRef(aligned_ref_bgr);
    cv::Mat value(difference_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::Rect surface_roi;
    if (cv::countNonZero(surface) > 0) {
        surface_roi = cv::boundingRect(surface);
    }
    if (!surface_roi.empty()) {
        cv::parallel_for_(cv::Range(surface_roi.y, surface_roi.y + surface_roi.height), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* src_row = difference_bgr.ptr<cv::Vec3b>(y);
            uchar* value_row = value.ptr<uchar>(y);
            const uchar* surface_row = surface.ptr<uchar>(y);
            for (int x = surface_roi.x; x < surface_roi.x + surface_roi.width; ++x) {
                if (surface_row[x] == 0) {
                    continue;
                }
                value_row[x] = std::max({src_row[x][0], src_row[x][1], src_row[x][2]});
            }
        }
        });
    }
    int surface_count = cv::countNonZero(surface);
    double otsu_t = 45.0;
    if (surface_count > 0) {
        cv::Mat values(surface_count, 1, CV_8UC1);
        int idx = 0;
        for (int y = surface_roi.y; y < surface_roi.y + surface_roi.height; ++y) {
            const uchar* surface_row = surface.ptr<uchar>(y);
            const uchar* value_row = value.ptr<uchar>(y);
            for (int x = surface_roi.x; x < surface_roi.x + surface_roi.width; ++x) {
                if (surface_row[x] > 0) {
                    values.at<uchar>(idx++, 0) = value_row[x];
                }
            }
        }
        cv::Mat thresholded;
        otsu_t = cv::threshold(values, thresholded, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    const double min_otsu = static_cast<double>(envInt("SHADOW_HAND_MIN_OTSU", 35));
    if (otsu_t < min_otsu) {
        if (otsu_threshold != nullptr) {
            *otsu_threshold = otsu_t;
        }
        if (used_threshold != nullptr) {
            *used_threshold = 0;
        }
        return cv::Mat::zeros(difference_bgr.size(), CV_8UC1);
    }
    int threshold = static_cast<int>(std::clamp(std::lround(otsu_t - 10.0), 45L, 60L));
    cv::Mat candidate;
    cv::threshold(value, candidate, threshold, 255, cv::THRESH_BINARY);
    cv::bitwise_and(candidate, surface, candidate);
    cv::Mat visible_board;
    if (envInt("SHADOW_HAND_PROTECT_VISIBLE_BOARD", 1) != 0) {
        visible_board = makeVisibleBoardColorProtectionMask(observed_bgr);
        cv::bitwise_and(visible_board, surface, visible_board);
        if (!projected_tag_protection_mask.empty() && projected_tag_protection_mask.size() == difference_bgr.size()) {
            cv::Mat projected_tag_protection;
            cv::bitwise_and(projected_tag_protection_mask, surface, projected_tag_protection);
            cv::bitwise_or(visible_board, projected_tag_protection, visible_board);
        }
    }
    cv::morphologyEx(candidate, candidate, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)), cv::Point(-1, -1), 1);
    if (!visible_board.empty()) {
        candidate.setTo(0, visible_board);
    }
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31)), cv::Point(-1, -1), 1);
    if (!visible_board.empty()) {
        candidate.setTo(0, visible_board);
    }
    candidate = fillBinaryMaskHoles(candidate);
    cv::bitwise_and(candidate, surface, candidate);
    if (!visible_board.empty()) {
        candidate.setTo(0, visible_board);
    }
    cv::Mat component = largestComponentMask(candidate, 1000);
    cv::bitwise_and(component, surface, component);
    if (otsu_threshold != nullptr) {
        *otsu_threshold = otsu_t;
    }
    if (used_threshold != nullptr) {
        *used_threshold = threshold;
    }
    return component;
}

bool estimateGreenEdgeOffset(
    const cv::Mat& green_mask,
    const cv::Point2f& p0,
    const cv::Point2f& p1,
    const cv::Point2f& sheet_center,
    double* offset_out,
    double* support_out) {
    cv::Point2f edge = p1 - p0;
    double length = cv::norm(edge);
    if (length < 20.0) {
        return false;
    }

    cv::Point2f normal(-edge.y / static_cast<float>(length), edge.x / static_cast<float>(length));
    cv::Point2f mid = (p0 + p1) * 0.5f;
    if ((sheet_center - mid).dot(normal) < 0.0f) {
        normal *= -1.0f;
    }

    std::vector<double> offsets;
    int samples = std::clamp(static_cast<int>(std::lround(length / 8.0)), 20, 90);
    constexpr int search_radius = 32;
    constexpr float probe = 4.0f;
    for (int i = 0; i < samples; ++i) {
        float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(samples);
        cv::Point2f base = p0 + edge * t;

        double best_score = -1.0;
        int best_offset = 0;
        for (int offset = -search_radius; offset <= search_radius; ++offset) {
            cv::Point2f outside = base + normal * (static_cast<float>(offset) - probe);
            cv::Point2f inside = base + normal * (static_cast<float>(offset) + probe);
            bool outside_green = maskAt(green_mask, outside);
            bool inside_green = maskAt(green_mask, inside);

            double score = 0.0;
            if (!outside_green) {
                score += 1.0;
            }
            if (inside_green) {
                score += 1.0;
            }
            score -= 0.01 * std::abs(offset);
            if (score > best_score) {
                best_score = score;
                best_offset = offset;
            }
        }
        if (best_score >= 1.65) {
            offsets.push_back(static_cast<double>(best_offset));
        }
    }

    if (offsets.size() < static_cast<size_t>(std::max(8, samples / 4))) {
        return false;
    }
    std::nth_element(offsets.begin(), offsets.begin() + static_cast<long>(offsets.size() / 2), offsets.end());
    if (offset_out != nullptr) {
        *offset_out = offsets[offsets.size() / 2];
    }
    if (support_out != nullptr) {
        *support_out = static_cast<double>(offsets.size()) / static_cast<double>(samples);
    }
    return true;
}

bool estimateGreenEdgeEndpointOffsets(
    const cv::Mat& green_mask,
    const cv::Point2f& p0,
    const cv::Point2f& p1,
    const cv::Point2f& sheet_center,
    double* offset0_out,
    double* offset1_out,
    double* support_out) {
    cv::Point2f edge = p1 - p0;
    double length = cv::norm(edge);
    if (length < 20.0) {
        return false;
    }

    cv::Point2f normal(-edge.y / static_cast<float>(length), edge.x / static_cast<float>(length));
    cv::Point2f mid = (p0 + p1) * 0.5f;
    if ((sheet_center - mid).dot(normal) < 0.0f) {
        normal *= -1.0f;
    }

    std::vector<double> left_offsets;
    std::vector<double> right_offsets;
    int samples = std::clamp(static_cast<int>(std::lround(length / 8.0)), 20, 90);
    constexpr int search_radius = 32;
    constexpr float probe = 4.0f;
    for (int i = 0; i < samples; ++i) {
        float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(samples);
        cv::Point2f base = p0 + edge * t;

        double best_score = -1.0;
        int best_offset = 0;
        for (int offset = -search_radius; offset <= search_radius; ++offset) {
            cv::Point2f outside = base + normal * (static_cast<float>(offset) - probe);
            cv::Point2f inside = base + normal * (static_cast<float>(offset) + probe);
            bool outside_green = maskAt(green_mask, outside);
            bool inside_green = maskAt(green_mask, inside);

            double score = 0.0;
            if (!outside_green) {
                score += 1.0;
            }
            if (inside_green) {
                score += 1.0;
            }
            score -= 0.01 * std::abs(offset);
            if (score > best_score) {
                best_score = score;
                best_offset = offset;
            }
        }
        if (best_score >= 1.65) {
            if (t < 0.5f) {
                left_offsets.push_back(static_cast<double>(best_offset));
            } else {
                right_offsets.push_back(static_cast<double>(best_offset));
            }
        }
    }

    int min_side_support = std::max(3, samples / 12);
    if (left_offsets.size() < static_cast<size_t>(min_side_support) ||
            right_offsets.size() < static_cast<size_t>(min_side_support)) {
        return false;
    }

    double offset0 = medianValue(left_offsets);
    double offset1 = medianValue(right_offsets);
    if (std::abs(offset0 - offset1) > 24.0) {
        return false;
    }
    if (offset0_out != nullptr) {
        *offset0_out = offset0;
    }
    if (offset1_out != nullptr) {
        *offset1_out = offset1;
    }
    if (support_out != nullptr) {
        *support_out =
            static_cast<double>(left_offsets.size() + right_offsets.size()) /
            static_cast<double>(samples);
    }
    return true;
}

bool estimateBlueTagEdgeOffset(
    const cv::Mat& blue_mask,
    const cv::Point2f& p0,
    const cv::Point2f& p1,
    const cv::Point2f& tag_center,
    cv::Point2f* delta_out) {
    cv::Point2f edge = p1 - p0;
    double length = cv::norm(edge);
    if (length < 12.0) {
        return false;
    }

    cv::Point2f normal(-edge.y / static_cast<float>(length), edge.x / static_cast<float>(length));
    cv::Point2f mid = (p0 + p1) * 0.5f;
    if ((tag_center - mid).dot(normal) < 0.0f) {
        normal *= -1.0f;
    }

    std::vector<double> offsets;
    int samples = std::clamp(static_cast<int>(std::lround(length / 5.0)), 10, 50);
    constexpr int search_radius = 26;
    constexpr float probe = 3.0f;
    for (int i = 0; i < samples; ++i) {
        float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(samples);
        cv::Point2f base = p0 + edge * t;

        double best_score = -1.0;
        int best_offset = 0;
        for (int offset = -search_radius; offset <= search_radius; ++offset) {
            cv::Point2f outside = base + normal * (static_cast<float>(offset) - probe);
            cv::Point2f inside = base + normal * (static_cast<float>(offset) + probe);
            bool outside_blue = maskAt(blue_mask, outside);
            bool inside_blue = maskAt(blue_mask, inside);

            double score = 0.0;
            if (!outside_blue) {
                score += 1.0;
            }
            if (inside_blue) {
                score += 1.0;
            }
            score -= 0.01 * std::abs(offset);
            if (score > best_score) {
                best_score = score;
                best_offset = offset;
            }
        }
        if (best_score >= 1.65) {
            offsets.push_back(static_cast<double>(best_offset));
        }
    }

    if (offsets.size() < static_cast<size_t>(std::max(5, samples / 4))) {
        return false;
    }
    double offset = medianValue(offsets);
    if (std::abs(offset) < 1.0) {
        return false;
    }
    if (delta_out != nullptr) {
        *delta_out = normal * static_cast<float>(offset);
    }
    return true;
}

TagPatternShift estimateDominantBlueTagEdgeShift(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& H_ref_to_frame,
    FrameMaskCache* frame_masks = nullptr) {
    TagPatternShift result;
    if (frame_bgr.empty() || H_ref_to_frame.empty()) {
        return result;
    }

    cv::Mat blue_mask = getCachedBlueMask(frame_bgr, frame_masks);
    std::vector<std::vector<cv::Point2f>> tag_deltas(reference.reference_tags.detections.size());
    std::vector<std::vector<cv::Point2f>> ref_tag_corners;
    ref_tag_corners.reserve(reference.reference_tags.detections.size());
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        (void)id;
        ref_tag_corners.push_back(ref_corners);
    }

    cv::parallel_for_(cv::Range(0, static_cast<int>(ref_tag_corners.size())), [&](const cv::Range& range) {
        for (int index = range.start; index < range.end; ++index) {
            const std::vector<cv::Point2f>& ref_corners = ref_tag_corners[static_cast<size_t>(index)];
            std::vector<cv::Point2f> projected = projectQuad(H_ref_to_frame, ref_corners);
            if (!quadInFrame(projected, frame_bgr.size()) || projected.size() != 4) {
                continue;
            }
            if (projectedTagSize(projected) < 12.0) {
                continue;
            }

            cv::Point2f center = quadCenter(projected);
            cv::Point2f left_delta;
            if (estimateBlueTagEdgeOffset(blue_mask, projected[0], projected[3], center, &left_delta)) {
                tag_deltas[static_cast<size_t>(index)].push_back(left_delta);
            }
            cv::Point2f right_delta;
            if (estimateBlueTagEdgeOffset(blue_mask, projected[1], projected[2], center, &right_delta)) {
                tag_deltas[static_cast<size_t>(index)].push_back(right_delta);
            }
        }
    });

    std::vector<cv::Point2f> deltas;
    deltas.reserve(ref_tag_corners.size() * 2);
    for (const std::vector<cv::Point2f>& per_tag_deltas : tag_deltas) {
        deltas.insert(deltas.end(), per_tag_deltas.begin(), per_tag_deltas.end());
    }

    if (deltas.size() < 2) {
        return result;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(deltas.size());
    ys.reserve(deltas.size());
    for (const cv::Point2f& delta : deltas) {
        xs.push_back(delta.x);
        ys.push_back(delta.y);
    }
    cv::Point2f median_delta(
        static_cast<float>(medianValue(xs)),
        static_cast<float>(medianValue(ys)));

    int support = 0;
    for (const cv::Point2f& delta : deltas) {
        if (cv::norm(delta - median_delta) <= 7.0) {
            ++support;
        }
    }
    if (support < 2) {
        return result;
    }

    if (std::abs(median_delta.y) > 10.0f) {
        median_delta.y = 0.0f;
    } else if (std::abs(median_delta.y) < 3.0f) {
        median_delta.y = 0.0f;
    }
    if (std::abs(median_delta.x) < 3.0f && std::abs(median_delta.y) < 3.0f) {
        return result;
    }

    median_delta.x = std::clamp(-median_delta.x, -24.0f, 24.0f);
    median_delta.y = std::clamp(-median_delta.y, -8.0f, 8.0f);
    result.ok = true;
    result.delta = median_delta;
    result.support = support;
    return result;
}

SheetEdgeAlignment checkSheetEdgeAlignment(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& H_ref_to_frame,
    FrameMaskCache* frame_masks = nullptr,
    bool include_side_edges = false) {
    SheetEdgeAlignment result;
    if (frame_bgr.empty() || H_ref_to_frame.empty()) {
        return result;
    }

    std::vector<cv::Point2f> ref_sheet = referenceImageCorners(reference.reference_bgr.size());
    std::vector<cv::Point2f> frame_sheet = projectQuad(H_ref_to_frame, ref_sheet);
    if (frame_sheet.size() != 4) {
        return result;
    }

    cv::Mat green_mask = getCachedGreenMask(frame_bgr, frame_masks);
    cv::Point2f center = quadCenter(frame_sheet);

    auto checkEdge = [&](const cv::Point2f& a, const cv::Point2f& b) {
        double offset = 0.0;
        double support = 0.0;
        if (!estimateGreenEdgeOffset(green_mask, a, b, center, &offset, &support)) {
            return;
        }
        ++result.usable_edges;
        result.max_abs_offset = std::max(result.max_abs_offset, std::abs(offset));
    };

    checkEdge(frame_sheet[0], frame_sheet[1]); // top edge
    checkEdge(frame_sheet[3], frame_sheet[2]); // bottom edge
    if (include_side_edges) {
        checkEdge(frame_sheet[0], frame_sheet[3]); // left edge
        checkEdge(frame_sheet[1], frame_sheet[2]); // right edge
    }

    result.usable = result.usable_edges > 0;
    result.ok = !result.usable || result.max_abs_offset <= 18.0;
    return result;
}

struct TagCellAlignmentScore {
    double score = 0.0;
    int visible_cells = 0;
    int matched_cells = 0;
    int scored_tags = 0;
};

TagCellAlignmentScore scoreTagInnerCellAlignment(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& H_ref_to_frame,
    FrameMaskCache* frame_masks) {
    TagCellAlignmentScore result;
    if (frame_bgr.empty() || H_ref_to_frame.empty()) {
        return result;
    }
    const cv::Mat blue_mask = getCachedBlueMask(frame_bgr, frame_masks);
    const cv::Mat green_mask = getCachedGreenMask(frame_bgr, frame_masks);
    double tag_score_sum = 0.0;

    for (const auto& [id, pattern] : reference.reference_tag_patterns) {
        (void)id;
        if (pattern.inner_cell_count <= 0) {
            continue;
        }
        std::vector<cv::Point2f> ref_centers;
        ref_centers.reserve(static_cast<size_t>(pattern.inner_cell_count));
        for (int i = 0; i < pattern.inner_cell_count; ++i) {
            ref_centers.push_back(pattern.ref_inner_cell_centers[static_cast<size_t>(i)]);
        }
        std::vector<cv::Point2f> frame_centers;
        cv::perspectiveTransform(ref_centers, frame_centers, H_ref_to_frame);

        int visible = 0;
        int matched = 0;
        int visible_expected_blue = 0;
        int visible_expected_green = 0;
        for (int i = 0; i < pattern.inner_cell_count; ++i) {
            const unsigned char expected = pattern.inner_cell_labels[static_cast<size_t>(i)];
            const cv::Point2f& point = frame_centers[static_cast<size_t>(i)];
            if (expected == 0 || !std::isfinite(point.x) || !std::isfinite(point.y)) {
                continue;
            }
            const int x = cvRound(point.x);
            const int y = cvRound(point.y);
            const cv::Rect sample = cv::Rect(x - 1, y - 1, 3, 3) &
                cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
            if (sample.area() < 4) {
                continue;
            }
            const int blue = cv::countNonZero(blue_mask(sample));
            const int green = cv::countNonZero(green_mask(sample));
            const int classified = blue + green;
            if (classified < 3 || std::max(blue, green) * 3 < classified * 2) {
                continue;
            }
            const unsigned char observed = blue > green ? 1 : 2;
            ++visible;
            visible_expected_blue += expected == 1 ? 1 : 0;
            visible_expected_green += expected == 2 ? 1 : 0;
            matched += observed == expected ? 1 : 0;
        }

        // A tag contributes only when both payload colors are visible. This
        // prevents a generic patch of green sheet from masquerading as a tag.
        if (visible < 4 || visible_expected_blue < 1 || visible_expected_green < 1) {
            continue;
        }
        tag_score_sum += static_cast<double>(matched) / static_cast<double>(visible);
        result.visible_cells += visible;
        result.matched_cells += matched;
        ++result.scored_tags;
    }
    if (result.scored_tags > 0) {
        result.score = tag_score_sum / static_cast<double>(result.scored_tags);
    }
    return result;
}

VisualAlignmentScore scoreVisualAlignment(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& H_ref_to_frame,
    FrameMaskCache* frame_masks = nullptr,
    VisualAlignmentScratch* scratch = nullptr,
    bool use_cell_pattern = false) {
    VisualAlignmentScore result;
    if (frame_bgr.empty() || H_ref_to_frame.empty() ||
            reference.reference_blue_mask.empty() ||
            reference.reference_green_mask.empty()) {
        return result;
    }

    cv::Mat frame_blue = getCachedBlueMask(frame_bgr, frame_masks);
    cv::Mat frame_green = getCachedGreenMask(frame_bgr, frame_masks);
    cv::Mat frame_valid = getCachedValidMask(frame_bgr, frame_masks);

    cv::Mat local_warped_blue;
    cv::Mat local_warped_green;
    cv::Mat local_warped_label;
    cv::Mat local_label_blue;
    cv::Mat local_label_green;
    cv::Mat& warped_blue = scratch != nullptr ? scratch->warped_blue : local_warped_blue;
    cv::Mat& warped_green = scratch != nullptr ? scratch->warped_green : local_warped_green;
    cv::Mat& warped_label = scratch != nullptr ? scratch->warped_label : local_warped_label;
    cv::Mat& label_blue = scratch != nullptr ? scratch->label_blue : local_label_blue;
    cv::Mat& label_green = scratch != nullptr ? scratch->label_green : local_label_green;
    const bool use_label_warp =
        reference.reference_visual_masks_exclusive &&
        !reference.reference_visual_label_mask.empty();
    if (use_label_warp) {
        cv::warpPerspective(
            reference.reference_visual_label_mask,
            warped_label,
            H_ref_to_frame,
            frame_bgr.size(),
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0));
        cv::compare(warped_label, 1, label_blue, cv::CMP_EQ);
        warped_blue = label_blue;
    } else {
        cv::warpPerspective(
            reference.reference_blue_mask,
            warped_blue,
            H_ref_to_frame,
            frame_bgr.size(),
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0));
    }

    cv::Mat local_visible_blue;
    cv::Mat& visible_blue = scratch != nullptr ? scratch->visible_blue : local_visible_blue;
    cv::bitwise_and(warped_blue, frame_valid, visible_blue);
    result.visible_blue = cv::countNonZero(visible_blue);
    cv::Mat local_blue_hits;
    cv::Mat& blue_hits = scratch != nullptr ? scratch->blue_hits : local_blue_hits;
    cv::bitwise_and(warped_blue, frame_blue, blue_hits);
    int blue_hit_count = cv::countNonZero(blue_hits);
    result.blue_score = result.visible_blue > 0
        ? static_cast<double>(blue_hit_count) / static_cast<double>(result.visible_blue)
        : 0.0;

    if (result.visible_blue < 250) {
        result.score = -1000.0 + result.visible_blue * 0.001;
        return result;
    }

    if (!use_label_warp) {
        cv::warpPerspective(
            reference.reference_green_mask,
            warped_green,
            H_ref_to_frame,
            frame_bgr.size(),
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0));
    } else {
        cv::compare(warped_label, 2, label_green, cv::CMP_EQ);
        warped_green = label_green;
    }
    cv::Mat local_visible_green;
    cv::Mat& visible_green = scratch != nullptr ? scratch->visible_green : local_visible_green;
    cv::bitwise_and(warped_green, frame_valid, visible_green);
    result.visible_green = cv::countNonZero(visible_green);
    cv::Mat local_green_hits;
    cv::Mat& green_hits = scratch != nullptr ? scratch->green_hits : local_green_hits;
    cv::bitwise_and(warped_green, frame_green, green_hits);
    int green_hit_count = cv::countNonZero(green_hits);
    result.green_score = result.visible_green > 0
        ? static_cast<double>(green_hit_count) / static_cast<double>(result.visible_green)
        : 0.0;

    if (use_cell_pattern) {
        TagCellAlignmentScore cells = scoreTagInnerCellAlignment(
            frame_bgr, reference, H_ref_to_frame, frame_masks);
        result.cell_pattern_score = cells.score;
        result.visible_cells = cells.visible_cells;
        result.matched_cells = cells.matched_cells;
        result.scored_tags = cells.scored_tags;
    }

    SheetEdgeAlignment edges = checkSheetEdgeAlignment(frame_bgr, reference, H_ref_to_frame, frame_masks);
    result.edge_penalty = edges.usable ? edges.max_abs_offset : 0.0;

    double visible_bonus = std::min(1.0, static_cast<double>(result.visible_blue) / 3500.0);
    result.score =
        2.8 * result.blue_score +
        0.55 * result.green_score +
        (use_cell_pattern && result.scored_tags > 0 ? 1.8 * result.cell_pattern_score : 0.0) +
        0.15 * visible_bonus -
        0.018 * result.edge_penalty;
    return result;
}

cv::Point2f projectedSheetCenter(const ReferenceModel& reference, const cv::Mat& H_ref_to_frame) {
    std::vector<cv::Point2f> ref_sheet = referenceImageCorners(reference.reference_bgr.size());
    std::vector<cv::Point2f> frame_sheet = projectQuad(H_ref_to_frame, ref_sheet);
    return frame_sheet.size() == 4 ? quadCenter(frame_sheet) : cv::Point2f();
}

bool isFiniteQuad(const std::vector<cv::Point2f>& quad) {
    if (quad.size() != 4) {
        return false;
    }
    for (const cv::Point2f& point : quad) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
    }
    return std::abs(cv::contourArea(quad)) > 100.0;
}

double maxCornerMotion(
    const std::vector<cv::Point2f>& original,
    const std::vector<cv::Point2f>& candidate) {
    if (original.size() != 4 || candidate.size() != 4) {
        return std::numeric_limits<double>::max();
    }
    double max_motion = 0.0;
    for (int i = 0; i < 4; ++i) {
        max_motion = std::max(
            max_motion,
            static_cast<double>(cv::norm(candidate[static_cast<size_t>(i)] - original[static_cast<size_t>(i)])));
    }
    return max_motion;
}

bool sheetQuadShapePlausible(
    const std::vector<cv::Point2f>& original,
    const std::vector<cv::Point2f>& candidate) {
    if (!isFiniteQuad(original) || !isFiniteQuad(candidate)) {
        return false;
    }
    double original_area = std::abs(cv::contourArea(original));
    double candidate_area = std::abs(cv::contourArea(candidate));
    double area_ratio = candidate_area / std::max(1.0, original_area);
    if (area_ratio < 0.88 || area_ratio > 1.12) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& a0 = original[static_cast<size_t>(i)];
        const cv::Point2f& a1 = original[static_cast<size_t>((i + 1) % 4)];
        const cv::Point2f& b0 = candidate[static_cast<size_t>(i)];
        const cv::Point2f& b1 = candidate[static_cast<size_t>((i + 1) % 4)];
        double original_len = cv::norm(a1 - a0);
        double candidate_len = cv::norm(b1 - b0);
        double ratio = candidate_len / std::max(1.0, original_len);
        if (ratio < 0.86 || ratio > 1.16) {
            return false;
        }
    }
    return true;
}

bool matEqualExact(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() ||
            a.rows != b.rows ||
            a.cols != b.cols ||
            a.type() != b.type()) {
        return false;
    }
    if (a.isContinuous() && b.isContinuous()) {
        const size_t byte_count = a.total() * a.elemSize();
        return std::memcmp(a.ptr(), b.ptr(), byte_count) == 0;
    }
    for (int row = 0; row < a.rows; ++row) {
        const size_t byte_count = static_cast<size_t>(a.cols) * a.elemSize();
        if (std::memcmp(a.ptr(row), b.ptr(row), byte_count) != 0) {
            return false;
        }
    }
    return true;
}

cv::Mat homographyFromSheetQuad(
    const ReferenceModel& reference,
    const std::vector<cv::Point2f>& frame_sheet) {
    if (!isFiniteQuad(frame_sheet)) {
        return cv::Mat();
    }
    std::vector<cv::Point2f> ref_sheet = referenceImageCorners(reference.reference_bgr.size());
    return cv::getPerspectiveTransform(ref_sheet, frame_sheet);
}

cv::Mat refineFallbackHomographyByVisualSearch(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const std::vector<int>& hard_tag_ids,
    const cv::Mat& H_ref_to_frame,
    const std::vector<cv::Point2f>& guard_src_pts = {},
    const std::vector<cv::Point2f>& guard_dst_pts = {},
    FrameMaskCache* frame_masks = nullptr,
    bool use_cell_pattern = false,
    bool score_support_points = false) {
    ScopedStageTimer timer(&g_geometry_timing.visual_search);
    if (frame_bgr.empty() || H_ref_to_frame.empty()) {
        return H_ref_to_frame;
    }

    const std::vector<cv::Point2f> ref_sheet = referenceImageCorners(reference.reference_bgr.size());
    const std::vector<cv::Point2f> initial_sheet = projectQuad(H_ref_to_frame, ref_sheet);
    cv::Point2f center = initial_sheet.size() == 4 ? quadCenter(initial_sheet) : cv::Point2f();
    if (!std::isfinite(center.x) || !std::isfinite(center.y)) {
        return H_ref_to_frame;
    }

    const bool guard_hard_tags = !hard_tag_ids.empty();
    const double before_error = guard_hard_tags
        ? meanHardTagReprojectionError(reference, frame_tags, hard_tag_ids, H_ref_to_frame)
        : 0.0;
    const bool guard_support_points =
        guard_src_pts.size() >= 4 && guard_src_pts.size() == guard_dst_pts.size();
    const double before_support_error = guard_support_points
        ? meanCorrespondenceReprojectionError(guard_src_pts, guard_dst_pts, H_ref_to_frame)
        : 0.0;

    VisualAlignmentScratch score_scratch;
    std::vector<std::pair<cv::Mat, VisualAlignmentScore>> score_cache;
    auto scoreCandidate = [&](const cv::Mat& H) {
        for (const auto& [cached_H, cached_score] : score_cache) {
            if (matEqualExact(cached_H, H)) {
                return cached_score;
            }
        }
        VisualAlignmentScore score = scoreVisualAlignment(
            frame_bgr,
            reference,
            H,
            frame_masks,
            &score_scratch,
            use_cell_pattern);
        if (score_support_points && guard_support_points) {
            score.score -= 0.04 * meanCorrespondenceReprojectionError(
                guard_src_pts, guard_dst_pts, H);
        }
        score_cache.emplace_back(H.clone(), score);
        return score;
    };

    VisualAlignmentScore base = scoreCandidate(H_ref_to_frame);
    cv::Mat best_H = H_ref_to_frame;
    VisualAlignmentScore best = base;
    cv::Point2f best_delta(0.0f, 0.0f);
    double best_angle = 0.0;
    auto consider = [&](float dx, float dy, double angle) {
        cv::Mat candidate = rotateScaleTranslateHomography(
            H_ref_to_frame,
            center,
            angle,
            cv::Point2f(dx, dy));
        if (candidate.empty()) {
            return;
        }
        if (guard_hard_tags) {
            double after_error = meanHardTagReprojectionError(reference, frame_tags, hard_tag_ids, candidate);
            if (after_error > std::max(6.0, before_error + 3.0)) {
                return;
            }
        }
        if (guard_support_points) {
            double after_error = meanCorrespondenceReprojectionError(guard_src_pts, guard_dst_pts, candidate);
            if (after_error > std::max(6.0, before_support_error + 3.0)) {
                return;
            }
        }
        VisualAlignmentScore score = scoreCandidate(candidate);
        if (score.score > best.score) {
            best = score;
            best_H = candidate;
            best_delta = cv::Point2f(dx, dy);
            best_angle = angle;
        }
    };

    for (double angle : {-1.2, -0.6, 0.0, 0.6, 1.2}) {
        for (int dy = -10; dy <= 10; dy += 5) {
            for (int dx = -20; dx <= 20; dx += 5) {
                consider(static_cast<float>(dx), static_cast<float>(dy), angle);
            }
        }
    }
    cv::Point2f coarse_delta = best_delta;
    double coarse_angle = best_angle;
    for (double da : {-0.4, -0.2, 0.0, 0.2, 0.4}) {
        for (int ddy = -4; ddy <= 4; ddy += 2) {
            for (int ddx = -4; ddx <= 4; ddx += 2) {
                consider(
                    coarse_delta.x + static_cast<float>(ddx),
                    coarse_delta.y + static_cast<float>(ddy),
                    coarse_angle + da);
            }
        }
    }

    double improvement = best.score - base.score;
    bool affine_accepted =
        improvement >= 0.045 &&
        (best.blue_score >= base.blue_score + 0.015 ||
         (use_cell_pattern && best.scored_tags > 0 &&
          best.cell_pattern_score >= base.cell_pattern_score + 0.025)) &&
        best.visible_blue >= std::max(200, static_cast<int>(base.visible_blue * 0.70));
    if (!affine_accepted) {
        best_H = H_ref_to_frame;
        best = base;
    }

    // A whole-sheet translation/rotation cannot fix a small perspective twist,
    // which shows up as one corner or one bottom tag being several pixels off
    // while the rest of the sheet looks close. For fallback cases only, do a
    // tightly bounded coordinate search over the projected sheet corners and
    // keep it only when the same dense blue/green evidence improves.
    cv::Mat affine_best_H = best_H.clone();
    std::vector<cv::Point2f> best_sheet = projectQuad(
        best_H,
        ref_sheet);
    if (!isFiniteQuad(initial_sheet) || !isFiniteQuad(best_sheet)) {
        return best_H;
    }

    VisualAlignmentScore projective_base = best;
    auto considerSheet = [&](const std::vector<cv::Point2f>& candidate_sheet) {
        if (!sheetQuadShapePlausible(best_sheet, candidate_sheet) ||
                maxCornerMotion(initial_sheet, candidate_sheet) > 16.0) {
            return;
        }
        cv::Mat candidate = homographyFromSheetQuad(reference, candidate_sheet);
        if (candidate.empty()) {
            return;
        }
        if (guard_hard_tags) {
            double after_error = meanHardTagReprojectionError(reference, frame_tags, hard_tag_ids, candidate);
            if (after_error > std::max(6.0, before_error + 3.0)) {
                return;
            }
        }
        if (guard_support_points) {
            double after_error = meanCorrespondenceReprojectionError(guard_src_pts, guard_dst_pts, candidate);
            if (after_error > std::max(6.0, before_support_error + 3.0)) {
                return;
            }
        }
        VisualAlignmentScore score = scoreCandidate(candidate);
        if (score.score > best.score) {
            best = score;
            best_H = candidate;
            best_sheet = candidate_sheet;
        }
    };

    const std::vector<cv::Point2f> unit_directions = {
        {1.0f, 0.0f},
        {-1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.0f, -1.0f},
        {1.0f, 1.0f},
        {1.0f, -1.0f},
        {-1.0f, 1.0f},
        {-1.0f, -1.0f},
    };
    for (float step : {5.0f, 3.0f, 1.5f}) {
        bool improved = true;
        int pass_count = 0;
        while (improved && pass_count < 2) {
            improved = false;
            VisualAlignmentScore before_pass = best;
            for (int corner = 0; corner < 4; ++corner) {
                for (const cv::Point2f& direction : unit_directions) {
                    std::vector<cv::Point2f> candidate_sheet = best_sheet;
                    candidate_sheet[static_cast<size_t>(corner)] += direction * step;
                    considerSheet(candidate_sheet);
                }
            }
            improved = best.score > before_pass.score + 1e-6;
            ++pass_count;
        }
    }

    double projective_improvement = best.score - projective_base.score;
    if (projective_improvement < 0.025 ||
            (best.blue_score < projective_base.blue_score + 0.008 &&
             (!use_cell_pattern || best.scored_tags <= 0 ||
              best.cell_pattern_score < projective_base.cell_pattern_score + 0.015)) ||
            best.visible_blue < std::max(200, static_cast<int>(projective_base.visible_blue * 0.70))) {
        return affine_best_H.empty() ? H_ref_to_frame : affine_best_H;
    }
    return best_H;
}

std::vector<cv::Point2f> moveEdgeTowardGreenBoundary(
    const cv::Mat& green_mask,
    const std::vector<cv::Point2f>& frame_sheet,
    int edge_start,
    int edge_end,
    const cv::Point2f& sheet_center,
    bool* ok) {
    if (ok != nullptr) {
        *ok = false;
    }
    if (frame_sheet.size() != 4) {
        return {};
    }

    double offset = 0.0;
    double support = 0.0;
    if (!estimateGreenEdgeOffset(
            green_mask,
            frame_sheet[static_cast<size_t>(edge_start)],
            frame_sheet[static_cast<size_t>(edge_end)],
            sheet_center,
            &offset,
            &support)) {
        return {};
    }

    cv::Point2f p0 = frame_sheet[static_cast<size_t>(edge_start)];
    cv::Point2f p1 = frame_sheet[static_cast<size_t>(edge_end)];
    cv::Point2f edge = p1 - p0;
    double length = cv::norm(edge);
    if (length < 20.0) {
        return {};
    }

    cv::Point2f normal(-edge.y / static_cast<float>(length), edge.x / static_cast<float>(length));
    cv::Point2f mid = (p0 + p1) * 0.5f;
    if ((sheet_center - mid).dot(normal) < 0.0f) {
        normal *= -1.0f;
    }

    if (ok != nullptr) {
        *ok = true;
    }
    cv::Point2f delta = normal * static_cast<float>(offset);
    return {p0 + delta, p1 + delta};
}

std::vector<cv::Point2f> moveEdgeEndpointsTowardGreenBoundary(
    const cv::Mat& green_mask,
    const std::vector<cv::Point2f>& frame_sheet,
    int edge_start,
    int edge_end,
    const cv::Point2f& sheet_center,
    bool* ok) {
    if (ok != nullptr) {
        *ok = false;
    }
    if (frame_sheet.size() != 4) {
        return {};
    }

    double offset0 = 0.0;
    double offset1 = 0.0;
    double support = 0.0;
    cv::Point2f p0 = frame_sheet[static_cast<size_t>(edge_start)];
    cv::Point2f p1 = frame_sheet[static_cast<size_t>(edge_end)];
    if (!estimateGreenEdgeEndpointOffsets(
            green_mask,
            p0,
            p1,
            sheet_center,
            &offset0,
            &offset1,
            &support)) {
        return {};
    }
    if (support < 0.28) {
        return {};
    }

    cv::Point2f edge = p1 - p0;
    double length = cv::norm(edge);
    if (length < 20.0) {
        return {};
    }

    cv::Point2f normal(-edge.y / static_cast<float>(length), edge.x / static_cast<float>(length));
    cv::Point2f mid = (p0 + p1) * 0.5f;
    if ((sheet_center - mid).dot(normal) < 0.0f) {
        normal *= -1.0f;
    }

    if (ok != nullptr) {
        *ok = true;
    }
    return {
        p0 + normal * static_cast<float>(offset0),
        p1 + normal * static_cast<float>(offset1),
    };
}

cv::Mat refineHomographyWithSheetEdges(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& H,
    const std::vector<cv::Point2f>& src_pts,
    const std::vector<cv::Point2f>& dst_pts,
    FrameMaskCache* frame_masks = nullptr,
    bool include_side_edges = false) {
    ScopedStageTimer timer(&g_geometry_timing.sheet_edge_refinement);
    if (frame_bgr.empty() || H.empty() || src_pts.size() < 4 || dst_pts.size() < 4) {
        return H;
    }

    std::vector<cv::Point2f> ref_sheet = referenceImageCorners(reference.reference_bgr.size());
    std::vector<cv::Point2f> frame_sheet = projectQuad(H, ref_sheet);
    if (frame_sheet.size() != 4) {
        return H;
    }

    cv::Mat green_mask = getCachedGreenMask(frame_bgr, frame_masks);
    cv::Point2f center = quadCenter(frame_sheet);
    bool top_ok = false;
    bool bottom_ok = false;
    bool left_ok = false;
    bool right_ok = false;
    std::vector<cv::Point2f> top_edge = moveEdgeEndpointsTowardGreenBoundary(green_mask, frame_sheet, 0, 1, center, &top_ok);
    std::vector<cv::Point2f> bottom_edge = moveEdgeEndpointsTowardGreenBoundary(green_mask, frame_sheet, 3, 2, center, &bottom_ok);
    if (!top_ok) {
        top_edge = moveEdgeTowardGreenBoundary(green_mask, frame_sheet, 0, 1, center, &top_ok);
    }
    if (!bottom_ok) {
        bottom_edge = moveEdgeTowardGreenBoundary(green_mask, frame_sheet, 3, 2, center, &bottom_ok);
    }
    std::vector<cv::Point2f> left_edge;
    std::vector<cv::Point2f> right_edge;
    if (include_side_edges) {
        left_edge = moveEdgeEndpointsTowardGreenBoundary(
            green_mask, frame_sheet, 0, 3, center, &left_ok);
        right_edge = moveEdgeEndpointsTowardGreenBoundary(
            green_mask, frame_sheet, 1, 2, center, &right_ok);
        if (!left_ok) {
            left_edge = moveEdgeTowardGreenBoundary(
                green_mask, frame_sheet, 0, 3, center, &left_ok);
        }
        if (!right_ok) {
            right_edge = moveEdgeTowardGreenBoundary(
                green_mask, frame_sheet, 1, 2, center, &right_ok);
        }
    }
    if (!top_ok && !bottom_ok && !left_ok && !right_ok) {
        return H;
    }

    std::vector<cv::Point2f> refined_src = src_pts;
    std::vector<cv::Point2f> refined_dst = dst_pts;
    auto addEdgePrior = [&](const cv::Point2f& ref_a, const cv::Point2f& ref_b, const std::vector<cv::Point2f>& edge_points) {
        if (edge_points.size() != 2) {
            return;
        }
        cv::Point2f ref_mid = (ref_a + ref_b) * 0.5f;
        cv::Point2f dst_mid = (edge_points[0] + edge_points[1]) * 0.5f;
        // Sparse fallback can be locally correct at tags while shearing the
        // whole sheet. Give the visible sheet boundary enough weight to anchor
        // the global plane, but still keep hard/soft tag corners in the fit.
        for (int i = 0; i < 6; ++i) {
            refined_src.push_back(ref_a);
            refined_dst.push_back(edge_points[0]);
            refined_src.push_back(ref_mid);
            refined_dst.push_back(dst_mid);
            refined_src.push_back(ref_b);
            refined_dst.push_back(edge_points[1]);
        }
    };

    if (top_ok) {
        addEdgePrior(ref_sheet[0], ref_sheet[1], top_edge);
    }
    if (bottom_ok) {
        addEdgePrior(ref_sheet[3], ref_sheet[2], bottom_edge);
    }
    if (left_ok) {
        addEdgePrior(ref_sheet[0], ref_sheet[3], left_edge);
    }
    if (right_ok) {
        addEdgePrior(ref_sheet[1], ref_sheet[2], right_edge);
    }

    cv::Mat refined_H = cv::findHomography(refined_src, refined_dst, cv::RANSAC, 6.0);
    return refined_H.empty() ? H : refined_H;
}

struct CandidateScore {
    double score = 0.0;
    double visible_fraction = 0.0;
    double inner_cell_score = 0.0;
    int visible_inner_cells = 0;
    int matched_inner_cells = 0;
};

CandidateScore scoreCandidateAgainstReferenceTag(
    const cv::Mat& frame_bgr,
    const std::vector<cv::Point2f>& frame_quad,
    const ReferenceTagPattern& ref_pattern) {
    CandidateScore result;
    if (frame_quad.size() != 4 || ref_pattern.valid_pixels <= 0) {
        return result;
    }

    std::vector<cv::Point2f> canonical = {
        {0.0f, 0.0f},
        {static_cast<float>(kTagPatternPatchSize - 1), 0.0f},
        {static_cast<float>(kTagPatternPatchSize - 1), static_cast<float>(kTagPatternPatchSize - 1)},
        {0.0f, static_cast<float>(kTagPatternPatchSize - 1)},
    };

    cv::Mat H_frame = cv::getPerspectiveTransform(
        frame_quad,
        canonical);
    if (H_frame.empty()) {
        return result;
    }

    cv::Mat frame_patch;
    cv::warpPerspective(
        frame_bgr,
        frame_patch,
        H_frame,
        cv::Size(kTagPatternPatchSize, kTagPatternPatchSize),
        cv::INTER_LINEAR,
        cv::BORDER_REPLICATE);

    cv::Mat frame_hsv;
    cv::cvtColor(frame_patch, frame_hsv, cv::COLOR_BGR2HSV);
    cv::Mat frame_blue = makeBlueMaskFromHsv(frame_hsv);
    cv::Mat frame_green = makeGreenMaskFromHsv(frame_hsv);
    cv::Mat frame_valid;
    cv::bitwise_or(frame_blue, frame_green, frame_valid);

    constexpr int grid_size = 8;
    constexpr int payload_size = 6;
    constexpr int cell_size = kTagPatternPatchSize / grid_size;
    cv::Mat frame_gray;
    cv::cvtColor(frame_patch, frame_gray, cv::COLOR_BGR2GRAY);
    std::array<double, grid_size * grid_size> cell_values{};
    std::vector<double> border_values;
    std::vector<double> payload_values;
    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            const cv::Rect sample(
                col * cell_size + 2,
                row * cell_size + 2,
                cell_size - 4,
                cell_size - 4);
            std::vector<double> pixels;
            pixels.reserve(static_cast<size_t>(sample.area()));
            for (int y = sample.y; y < sample.y + sample.height; ++y) {
                const uchar* row_ptr = frame_gray.ptr<uchar>(y);
                for (int x = sample.x; x < sample.x + sample.width; ++x) {
                    pixels.push_back(row_ptr[x]);
                }
            }
            const double value = medianValue(std::move(pixels));
            cell_values[static_cast<size_t>(row * grid_size + col)] = value;
            if (row == 0 || row == grid_size - 1 || col == 0 || col == grid_size - 1) {
                border_values.push_back(value);
            } else {
                payload_values.push_back(value);
            }
        }
    }
    const double dark_level = medianValue(border_values);
    std::sort(payload_values.begin(), payload_values.end());
    std::vector<double> bright_values;
    if (!payload_values.empty()) {
        const size_t start = payload_values.size() * 2 / 3;
        bright_values.assign(payload_values.begin() + static_cast<std::ptrdiff_t>(start), payload_values.end());
    }
    const double bright_level = medianValue(bright_values);
    const double contrast = bright_level - dark_level;
    for (int row = 1; row <= payload_size; ++row) {
        for (int col = 1; col <= payload_size; ++col) {
            const int index = (row - 1) * payload_size + (col - 1);
            const unsigned char expected =
                ref_pattern.inner_cell_labels[static_cast<size_t>(index)];
            if (expected == 0) {
                continue;
            }
            if (contrast < 20.0) {
                continue;
            }
            const double normalized =
                (cell_values[static_cast<size_t>(row * grid_size + col)] - dark_level) /
                contrast;
            unsigned char observed = 0;
            if (normalized <= 0.35) {
                observed = 1;
            } else if (normalized >= 0.65) {
                observed = 2;
            } else {
                continue;
            }
            ++result.visible_inner_cells;
            result.matched_inner_cells += observed == expected ? 1 : 0;
        }
    }
    if (result.visible_inner_cells > 0) {
        result.inner_cell_score =
            static_cast<double>(result.matched_inner_cells) /
            static_cast<double>(result.visible_inner_cells);
    }

    cv::Mat visible;
    cv::bitwise_and(ref_pattern.valid_mask, frame_valid, visible);
    int visible_count = cv::countNonZero(visible);
    result.visible_fraction = static_cast<double>(visible_count) / static_cast<double>(ref_pattern.valid_pixels);
    if (result.visible_fraction < 0.12) {
        return result;
    }

    cv::Mat blue_match;
    cv::bitwise_and(ref_pattern.blue_mask, frame_blue, blue_match);
    cv::Mat green_match;
    cv::bitwise_and(ref_pattern.green_mask, frame_green, green_match);
    int matches = cv::countNonZero(blue_match) + cv::countNonZero(green_match);
    result.score = visible_count > 0
        ? static_cast<double>(matches) / static_cast<double>(visible_count)
        : 0.0;
    return result;
}

std::vector<cv::Point2f> translateQuad(const std::vector<cv::Point2f>& quad, float dx, float dy) {
    std::vector<cv::Point2f> translated = quad;
    for (cv::Point2f& point : translated) {
        point.x += dx;
        point.y += dy;
    }
    return translated;
}

std::vector<cv::Point2f> refineExpectedQuadByPattern(
    const cv::Mat& frame_bgr,
    const std::vector<cv::Point2f>& expected,
    const ReferenceTagPattern& ref_pattern,
    CandidateScore* best_pattern) {
    if (best_pattern != nullptr) {
        *best_pattern = CandidateScore();
    }
    if (expected.size() != 4 || ref_pattern.valid_pixels <= 0) {
        return {};
    }

    // Contours around partially occluded tags often describe only the visible
    // blue fragment. For color-mask soft tags, keep the full projected tag
    // shape from the single-tag seed and only search a small translation window
    // for the best blue/green reference-pattern agreement.
    double tag_size = projectedTagSize(expected);
    if (tag_size <= 1.0) {
        return expected;
    }

    int radius = std::max(4, static_cast<int>(std::lround(tag_size * 0.22)));
    int step = std::max(2, static_cast<int>(std::lround(tag_size * 0.035)));
    std::vector<cv::Point2f> best_quad = expected;
    double best_total = -std::numeric_limits<double>::max();
    CandidateScore best_score;

    for (int dy = -radius; dy <= radius; dy += step) {
        for (int dx = -radius; dx <= radius; dx += step) {
            std::vector<cv::Point2f> candidate = translateQuad(expected, static_cast<float>(dx), static_cast<float>(dy));
            if (!quadInFrame(candidate, frame_bgr.size())) {
                continue;
            }
            CandidateScore pattern = scoreCandidateAgainstReferenceTag(frame_bgr, candidate, ref_pattern);
            if (pattern.visible_fraction < 0.12) {
                continue;
            }
            double prior = 1.0 - (std::abs(dx) + std::abs(dy)) / std::max(1.0, 2.0 * static_cast<double>(radius));
            double total = 0.78 * pattern.score + 0.12 * pattern.visible_fraction + 0.10 * prior;
            if (total > best_total) {
                best_total = total;
                best_score = pattern;
                best_quad = std::move(candidate);
            }
        }
    }

    if (best_pattern != nullptr) {
        *best_pattern = best_score;
    }
    return best_quad;
}

TagPatternShift estimateDominantTagPatternShift(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& H_ref_to_frame) {
    TagPatternShift result;
    if (frame_bgr.empty() || H_ref_to_frame.empty()) {
        return result;
    }

    struct PatternShiftEntry {
        int id = -1;
        std::vector<cv::Point2f> ref_corners;
        const ReferenceTagPattern* pattern = nullptr;
    };
    std::vector<PatternShiftEntry> entries;
    entries.reserve(reference.reference_tags.detections.size());
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        auto pattern_it = reference.reference_tag_patterns.find(id);
        if (pattern_it == reference.reference_tag_patterns.end()) {
            continue;
        }
        entries.push_back(PatternShiftEntry{id, ref_corners, &pattern_it->second});
    }

    std::vector<cv::Point2f> indexed_deltas(entries.size());
    std::vector<unsigned char> delta_valid(entries.size(), 0);
    cv::parallel_for_(cv::Range(0, static_cast<int>(entries.size())), [&](const cv::Range& range) {
        for (int index = range.start; index < range.end; ++index) {
            const PatternShiftEntry& entry = entries[static_cast<size_t>(index)];
            std::vector<cv::Point2f> expected = projectQuad(H_ref_to_frame, entry.ref_corners);
            if (!quadInFrame(expected, frame_bgr.size())) {
                continue;
            }
            double tag_size = projectedTagSize(expected);
            if (tag_size < 12.0) {
                continue;
            }

            CandidateScore pattern;
            std::vector<cv::Point2f> refined = refineExpectedQuadByPattern(
                frame_bgr,
                expected,
                *entry.pattern,
                &pattern);
            if (refined.size() != 4) {
                continue;
            }

            cv::Point2f delta = quadCenter(refined) - quadCenter(expected);
            double delta_norm = cv::norm(delta);
            bool strong_partial = pattern.visible_fraction >= 0.18 && pattern.score >= 0.64;
            bool broad_moderate = pattern.visible_fraction >= 0.35 && pattern.score >= 0.56;
            if ((!strong_partial && !broad_moderate) ||
                    delta_norm < 1.0 ||
                    delta_norm > std::max(24.0, tag_size * 0.28)) {
                continue;
            }
            indexed_deltas[static_cast<size_t>(index)] = delta;
            delta_valid[static_cast<size_t>(index)] = 1;
        }
    });

    std::vector<cv::Point2f> deltas;
    deltas.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        if (delta_valid[i] != 0) {
            deltas.push_back(indexed_deltas[i]);
        }
    }

    if (deltas.size() < 2) {
        return result;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(deltas.size());
    ys.reserve(deltas.size());
    for (const cv::Point2f& delta : deltas) {
        xs.push_back(delta.x);
        ys.push_back(delta.y);
    }
    cv::Point2f median_delta(
        static_cast<float>(medianValue(xs)),
        static_cast<float>(medianValue(ys)));

    int support = 0;
    for (const cv::Point2f& delta : deltas) {
        if (cv::norm(delta - median_delta) <= 6.0) {
            ++support;
        }
    }
    if (support < 2) {
        return result;
    }

    // The sheet-edge check already handles vertical drift. This stage is meant
    // to catch the common fallback-only horizontal tag-edge slip, so keep any
    // vertical correction very small and reject large vertical votes.
    if (std::abs(median_delta.y) > 10.0f) {
        median_delta.y = 0.0f;
    } else if (std::abs(median_delta.y) < 3.0f) {
        median_delta.y = 0.0f;
    }
    if (std::abs(median_delta.x) < 3.0f && std::abs(median_delta.y) < 3.0f) {
        return result;
    }

    median_delta.x = std::clamp(median_delta.x, -24.0f, 24.0f);
    median_delta.y = std::clamp(median_delta.y, -8.0f, 8.0f);
    result.ok = true;
    result.delta = median_delta;
    result.support = support;
    return result;
}

cv::Mat refineFallbackHomographyWithTagPatternShift(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const std::vector<int>& hard_tag_ids,
    const cv::Mat& H_ref_to_frame,
    const std::vector<cv::Point2f>& guard_src_pts = {},
    const std::vector<cv::Point2f>& guard_dst_pts = {},
    bool use_cell_pattern = false,
    bool score_support_points = false) {
    cv::Mat candidate_H = H_ref_to_frame;
    FrameMaskCache frame_masks;
    const bool guard_support_points =
        guard_src_pts.size() >= 4 && guard_src_pts.size() == guard_dst_pts.size();

    // Fallback homographies can be a few pixels off even when hard/soft tag
    // correspondences look plausible. First apply the existing sparse tag-edge
    // shift, then let the dense blue/green visual matcher make a small local
    // correction. This function is only called by fallback paths; the normal
    // 3+ decoded-tag path never enters this refinement.
    TagPatternShift shift;
    {
        ScopedStageTimer timer(&g_geometry_timing.tag_pattern_shift);
        shift = estimateDominantBlueTagEdgeShift(frame_bgr, reference, H_ref_to_frame, &frame_masks);
        if (!shift.ok) {
            shift = estimateDominantTagPatternShift(frame_bgr, reference, H_ref_to_frame);
        }
    }
    if (shift.ok) {
        cv::Mat shifted_H = translateHomography(H_ref_to_frame, shift.delta);
        if (!shifted_H.empty()) {
            bool keep_shift = true;
            if (!hard_tag_ids.empty()) {
                double before_error = meanHardTagReprojectionError(reference, frame_tags, hard_tag_ids, H_ref_to_frame);
                double after_error = meanHardTagReprojectionError(reference, frame_tags, hard_tag_ids, shifted_H);
                if (after_error > std::max(6.0, before_error + 3.0)) {
                    keep_shift = false;
                }
            }
            if (guard_support_points) {
                double before_error = meanCorrespondenceReprojectionError(guard_src_pts, guard_dst_pts, H_ref_to_frame);
                double after_error = meanCorrespondenceReprojectionError(guard_src_pts, guard_dst_pts, shifted_H);
                if (after_error > std::max(6.0, before_error + 3.0)) {
                    keep_shift = false;
                }
            }
            if (keep_shift) {
                candidate_H = shifted_H;
            }
        }
    }

    return refineFallbackHomographyByVisualSearch(
        frame_bgr,
        reference,
        frame_tags,
        hard_tag_ids,
        candidate_H,
        guard_src_pts,
        guard_dst_pts,
        &frame_masks,
        use_cell_pattern,
        score_support_points);
}

bool hasSoftTagId(const std::vector<SoftTag>& tags, int id) {
    return std::any_of(tags.begin(), tags.end(), [id](const SoftTag& tag) {
        return tag.id == id;
    });
}

void appendSoftTagCorrespondences(
    const ReferenceModel& reference,
    const std::vector<SoftTag>& soft_tags,
    std::vector<cv::Point2f>* src_pts,
    std::vector<cv::Point2f>* dst_pts) {
    if (src_pts == nullptr || dst_pts == nullptr) {
        return;
    }
    for (const SoftTag& soft : soft_tags) {
        auto ref_it = reference.reference_tags.detections.find(soft.id);
        if (ref_it == reference.reference_tags.detections.end() || soft.frame_corners.size() != 4) {
            continue;
        }
        src_pts->insert(src_pts->end(), ref_it->second.begin(), ref_it->second.end());
        dst_pts->insert(dst_pts->end(), soft.frame_corners.begin(), soft.frame_corners.end());
    }
}

std::vector<SoftTag> recoverSoftTagsFromSeedPatternSupport(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const cv::Mat& H_seed,
    int hard_tag_id,
    const std::vector<SoftTag>& existing_tags) {
    ScopedStageTimer timer(&g_geometry_timing.seed_pattern_support);
    std::vector<SoftTag> recovered;
    if (H_seed.empty()) {
        return recovered;
    }

    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        if ((hard_tag_id >= 0 && id == hard_tag_id) ||
            frame_tags.detections.count(id) != 0 ||
            hasSoftTagId(existing_tags, id)) {
            continue;
        }

        std::vector<cv::Point2f> expected = projectQuad(H_seed, ref_corners);
        if (!quadInFrame(expected, frame_bgr.size())) {
            continue;
        }

        auto pattern_it = reference.reference_tag_patterns.find(id);
        if (pattern_it == reference.reference_tag_patterns.end()) {
            continue;
        }

        // This is a deliberately conservative "held seed" recovery source. It
        // does not decode an ID and does not invent a new shape from a contour;
        // it asks whether the previous/seed homography projects this known tag
        // onto pixels that still look like that tag's blue/green reference
        // pattern. A small translation search lets the quad snap to the visible
        // color pattern without letting it drift to an unrelated neighbor.
        CandidateScore pattern;
        std::vector<cv::Point2f> refined = refineExpectedQuadByPattern(
            frame_bgr,
            expected,
            pattern_it->second,
            &pattern);
        double tag_size = std::max(1.0, projectedTagSize(expected));
        double center_shift = cv::norm(quadCenter(refined) - quadCenter(expected));
        bool strong_partial = pattern.visible_fraction >= 0.18 && pattern.score >= 0.68;
        bool broad_moderate = pattern.visible_fraction >= 0.32 && pattern.score >= 0.58;
        if (!quadInFrame(refined, frame_bgr.size()) ||
            center_shift > tag_size * 0.25 ||
            (!strong_partial && !broad_moderate)) {
            continue;
        }

        SoftTag tag;
        tag.id = id;
        tag.frame_corners = std::move(refined);
        tag.score = pattern.score;
        tag.visible_fraction = pattern.visible_fraction;
        tag.source = "seed";
        recovered.push_back(std::move(tag));
    }

    return recovered;
}

std::vector<SoftTag> recoverSoftTagsFromSeedColorSupport(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const cv::Mat& H_seed,
    int hard_tag_id,
    const std::vector<SoftTag>& existing_tags,
    FrameMaskCache* frame_masks = nullptr) {
    ScopedStageTimer timer(&g_geometry_timing.seed_color_support);
    std::vector<SoftTag> recovered;
    if (H_seed.empty()) {
        return recovered;
    }

    cv::Mat blue_mask = getCachedBlueMask(frame_bgr, frame_masks);
    cv::Rect frame_bounds(0, 0, frame_bgr.cols, frame_bgr.rows);
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        if ((hard_tag_id >= 0 && id == hard_tag_id) ||
            frame_tags.detections.count(id) != 0 ||
            hasSoftTagId(existing_tags, id)) {
            continue;
        }

        std::vector<cv::Point2f> expected = projectQuad(H_seed, ref_corners);
        if (!quadInFrame(expected, frame_bgr.size())) {
            continue;
        }

        cv::Rect bounds = cv::boundingRect(expected) & frame_bounds;
        if (bounds.width < 6 || bounds.height < 6) {
            continue;
        }

        std::vector<cv::Point> local_quad;
        local_quad.reserve(expected.size());
        for (const cv::Point2f& point : expected) {
            local_quad.emplace_back(
                cvRound(point.x) - bounds.x,
                cvRound(point.y) - bounds.y);
        }

        cv::Mat quad_mask = cv::Mat::zeros(bounds.size(), CV_8U);
        cv::fillConvexPoly(quad_mask, local_quad, cv::Scalar(255), cv::LINE_AA);
        int quad_pixels = cv::countNonZero(quad_mask);
        if (quad_pixels < 50) {
            continue;
        }

        cv::Mat blue_in_quad;
        cv::bitwise_and(blue_mask(bounds), quad_mask, blue_in_quad);
        int blue_pixels = cv::countNonZero(blue_in_quad);
        double blue_fraction = static_cast<double>(blue_pixels) / static_cast<double>(quad_pixels);
        double tag_size = projectedTagSize(expected);

        // A projected tag that lands on the green sheet alone is not enough
        // evidence. Require a meaningful amount of blue tag material inside
        // the projected quad, scaled so small clipped tags still need a real
        // cluster of blue pixels.
        int min_blue_pixels = std::max(35, static_cast<int>(std::lround(quad_pixels * 0.055)));
        if (tag_size < 8.0 || blue_pixels < min_blue_pixels || blue_fraction < 0.055) {
            continue;
        }

        SoftTag tag;
        tag.id = id;
        tag.frame_corners = std::move(expected);
        tag.score = std::min(1.0, blue_fraction / 0.25);
        tag.visible_fraction = blue_fraction;
        tag.source = "seed-color";
        recovered.push_back(std::move(tag));
    }

    return recovered;
}

bool candidateGeometryPlausible(
    const std::vector<cv::Point2f>& candidate,
    const std::vector<cv::Point2f>& expected,
    double* geometry_score) {
    if (candidate.size() != 4 || expected.size() != 4) {
        return false;
    }

    double expected_area = quadArea(expected);
    double candidate_area = quadArea(candidate);
    if (expected_area <= 1.0 || candidate_area <= 1.0) {
        return false;
    }

    double area_ratio = candidate_area / expected_area;
    if (area_ratio < 0.4 || area_ratio > 2.5) {
        return false;
    }

    double expected_size = projectedTagSize(expected);
    if (expected_size <= 1.0) {
        return false;
    }
    double center_dist = cv::norm(quadCenter(candidate) - quadCenter(expected));
    if (center_dist > expected_size * 0.45) {
        return false;
    }

    if (geometry_score != nullptr) {
        double center_score = std::max(0.0, 1.0 - center_dist / (expected_size * 0.45));
        double area_score = std::max(0.0, 1.0 - std::abs(std::log(area_ratio)) / std::log(2.5));
        *geometry_score = 0.65 * center_score + 0.35 * area_score;
    }
    return true;
}

std::vector<SoftTag> recoverSoftTagsFromRejectedCandidates(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const cv::Mat& H_seed,
    int hard_tag_id) {
    ScopedStageTimer timer(&g_geometry_timing.soft_recovery_rejected);
    std::vector<SoftTag> recovered;
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        if ((hard_tag_id >= 0 && id == hard_tag_id) || frame_tags.detections.count(id) != 0) {
            continue;
        }

        std::vector<cv::Point2f> expected = projectQuad(H_seed, ref_corners);
        if (!quadInFrame(expected, frame_bgr.size())) {
            continue;
        }
        auto pattern_it = reference.reference_tag_patterns.find(id);
        if (pattern_it == reference.reference_tag_patterns.end()) {
            continue;
        }

        SoftTag best;
        double best_total_score = 0.0;
        for (const std::vector<cv::Point2f>& rejected : frame_tags.rejected) {
            if (rejected.size() != 4) {
                continue;
            }
            std::vector<cv::Point2f> candidate = orderCandidateLikeExpected(rejected, expected);
            double geometry_score = 0.0;
            if (!candidateGeometryPlausible(candidate, expected, &geometry_score)) {
                continue;
            }
            CandidateScore pattern = scoreCandidateAgainstReferenceTag(
                frame_bgr,
                candidate,
                pattern_it->second);
            if (pattern.visible_fraction < 0.15 || pattern.score < 0.55) {
                continue;
            }
            double lever_score = 0.5;
            auto hard_it = hard_tag_id >= 0 ? frame_tags.detections.find(hard_tag_id) : frame_tags.detections.end();
            if (hard_it != frame_tags.detections.end()) {
                double lever = cv::norm(quadCenter(candidate) - quadCenter(hard_it->second));
                lever_score = std::min(1.0, lever / std::max(1.0, static_cast<double>(frame_bgr.cols + frame_bgr.rows) * 0.25));
            }
            double total_score = 0.55 * pattern.score + 0.25 * geometry_score + 0.20 * lever_score;
            if (total_score > best_total_score) {
                best.id = id;
                best.frame_corners = candidate;
                best.score = pattern.score;
                best.visible_fraction = pattern.visible_fraction;
                best.source = "rejected";
                best_total_score = total_score;
            }
        }
        if (best.id >= 0) {
            recovered.push_back(std::move(best));
        }
    }
    return recovered;
}

std::vector<SoftTag> recoverSoftTagsFromColorMasks(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const cv::Mat& H_seed,
    int hard_tag_id,
    const std::vector<SoftTag>& already_recovered,
    FrameMaskCache* frame_masks = nullptr) {
    ScopedStageTimer timer(&g_geometry_timing.soft_recovery_color_masks);
    cv::Mat blue_mask = getCachedBlueMask(frame_bgr, frame_masks);
    std::vector<SoftTag> recovered;

    auto alreadyHas = [&](int id) {
        return std::any_of(already_recovered.begin(), already_recovered.end(), [id](const SoftTag& tag) {
            return tag.id == id;
        });
    };

    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        if ((hard_tag_id >= 0 && id == hard_tag_id) || frame_tags.detections.count(id) != 0 || alreadyHas(id)) {
            continue;
        }

        std::vector<cv::Point2f> expected = projectQuad(H_seed, ref_corners);
        if (!quadInFrame(expected, frame_bgr.size())) {
            continue;
        }
        auto pattern_it = reference.reference_tag_patterns.find(id);
        if (pattern_it == reference.reference_tag_patterns.end()) {
            continue;
        }

        cv::Rect bounds = cv::boundingRect(expected);
        int pad = static_cast<int>(std::lround(projectedTagSize(expected) * 0.65));
        bounds.x -= pad;
        bounds.y -= pad;
        bounds.width += 2 * pad;
        bounds.height += 2 * pad;
        bounds &= cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
        if (bounds.width < 8 || bounds.height < 8) {
            continue;
        }

        cv::Mat roi_mask = blue_mask(bounds).clone();
        cv::morphologyEx(roi_mask, roi_mask, cv::MORPH_CLOSE, morphKernel5());

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(roi_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        SoftTag best;
        double best_total_score = 0.0;
        for (const std::vector<cv::Point>& contour : contours) {
            if (cv::contourArea(contour) < 12.0) {
                continue;
            }
            cv::RotatedRect rect = cv::minAreaRect(contour);
            if (rect.size.width < 4.0f || rect.size.height < 4.0f) {
                continue;
            }
            cv::Point2f rect_points[4];
            rect.points(rect_points);
            std::vector<cv::Point2f> candidate(rect_points, rect_points + 4);
            for (cv::Point2f& point : candidate) {
                point.x += static_cast<float>(bounds.x);
                point.y += static_cast<float>(bounds.y);
            }
            candidate = orderCandidateLikeExpected(candidate, expected);

            double geometry_score = 0.0;
            if (!candidateGeometryPlausible(candidate, expected, &geometry_score)) {
                continue;
            }
            CandidateScore pattern;
            std::vector<cv::Point2f> refined_candidate = refineExpectedQuadByPattern(
                frame_bgr,
                expected,
                pattern_it->second,
                &pattern);
            if (pattern.visible_fraction < 0.18 || pattern.score < 0.60) {
                continue;
            }
            double refined_geometry_score = 0.0;
            if (candidateGeometryPlausible(refined_candidate, expected, &refined_geometry_score)) {
                geometry_score = std::max(geometry_score, refined_geometry_score);
            }
            double total_score = 0.65 * pattern.score + 0.35 * geometry_score;
            if (total_score > best_total_score) {
                best.id = id;
                best.frame_corners = refined_candidate;
                best.score = pattern.score;
                best.visible_fraction = pattern.visible_fraction;
                best.source = "color";
                best_total_score = total_score;
            }
        }

        if (best.id >= 0) {
            recovered.push_back(std::move(best));
        }
    }
    return recovered;
}

std::vector<cv::Point> toIntPoints(const std::vector<cv::Point2f>& points);
void drawPolyline(cv::Mat* image, const std::vector<cv::Point2f>& points, const cv::Scalar& color, int thickness);
cv::Point toIntPoint(const cv::Point2f& point);

void drawTagDiagnostics(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    const fs::path& outdir) {
    const std::string prefix = framePrefix(geometry.frame_index);

    cv::Mat hard_debug = frame_bgr.clone();
    for (const auto& [id, corners] : geometry.frame_tags.detections) {
        drawPolyline(&hard_debug, corners, cv::Scalar(0, 255, 255), 2);
        cv::putText(
            hard_debug,
            "hard " + std::to_string(id),
            toIntPoint(quadCenter(corners)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA);
    }
    cv::putText(
        hard_debug,
        "decoded hard tags: " + std::to_string(geometry.frame_tags.detections.size()),
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 255, 255),
        2,
        cv::LINE_AA);
    writePngFast(outdir / (prefix + "_v3_hard_tag_diagnostics.png"), hard_debug);

    cv::Mat rejected_debug = frame_bgr.clone();
    for (const std::vector<cv::Point2f>& corners : geometry.frame_tags.rejected) {
        drawPolyline(&rejected_debug, corners, cv::Scalar(96, 96, 255), 1);
    }
    for (const auto& [id, corners] : geometry.frame_tags.detections) {
        drawPolyline(&rejected_debug, corners, cv::Scalar(0, 255, 255), 2);
        cv::putText(
            rejected_debug,
            std::to_string(id),
            toIntPoint(quadCenter(corners)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.42,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA);
    }
    cv::putText(
        rejected_debug,
        "rejected quads: " + std::to_string(geometry.frame_tags.rejected.size()),
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(96, 96, 255),
        2,
        cv::LINE_AA);
    writePngFast(outdir / (prefix + "_v3_rejected_tag_candidates.png"), rejected_debug);

    cv::Mat debug = frame_bgr.clone();
    for (const auto& [id, corners] : geometry.frame_tags.detections) {
        drawPolyline(&debug, corners, cv::Scalar(0, 255, 255), 2);
        cv::putText(debug, "hard " + std::to_string(id), toIntPoint(quadCenter(corners)), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
    if (!geometry.H_seed_ref_to_frame.empty()) {
        for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
            if (geometry.frame_tags.detections.count(id) != 0) {
                continue;
            }
            std::vector<cv::Point2f> projected = projectQuad(geometry.H_seed_ref_to_frame, ref_corners);
            if (projected.size() == 4 && quadInFrame(projected, frame_bgr.size())) {
                drawPolyline(&debug, projected, cv::Scalar(160, 160, 160), 1);
            }
        }
    }
    for (const SoftTag& soft : geometry.soft_tags) {
        cv::Scalar color = cv::Scalar(255, 128, 0);
        if (soft.source == "rejected") {
            color = cv::Scalar(255, 0, 255);
        } else if (soft.source.rfind("seed", 0) == 0) {
            color = cv::Scalar(0, 255, 0);
        }
        drawPolyline(&debug, soft.frame_corners, color, 2);
        cv::putText(
            debug,
            "soft " + std::to_string(soft.id) + " " + soft.source,
            toIntPoint(quadCenter(soft.frame_corners)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv::LINE_AA);
    }

    if (geometry.board_quad.size() == 4) {
        drawPolyline(&debug, geometry.board_quad, cv::Scalar(0, 0, 255), 2);
    }
    const std::string geometry_status =
        "hard=" + std::to_string(geometry.common_tag_count) +
        " soft=" + std::to_string(geometry.soft_tag_count) +
        " seed=" + (geometry.H_seed_ref_to_frame.empty() ? "none" : "used") +
        " final=" + (geometry.ok ? "ok" : "failed");
    cv::putText(
        debug,
        geometry_status,
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.52,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA);

    writePngFast(outdir / (prefix + "_v3_soft_tag_diagnostics.png"), debug);
}

void printTagDiagnostics(
    const GeometryResult& geometry,
    const ReferenceModel& reference,
    const cv::Mat& frame_bgr) {
    auto print_corners = [](const std::vector<cv::Point2f>& corners) {
        for (size_t i = 0; i < corners.size(); ++i) {
            if (i != 0) {
                std::cout << ";";
            }
            std::cout << "(" << std::fixed << std::setprecision(2)
                      << corners[i].x << "," << corners[i].y << ")";
        }
    };

    for (const auto& [id, corners] : geometry.frame_tags.detections) {
        std::cout << "[TAG] frame=" << geometry.frame_index
                  << " kind=hard id=" << id << " corners=";
        print_corners(corners);
        auto pattern_it = reference.reference_tag_patterns.find(id);
        if (pattern_it != reference.reference_tag_patterns.end()) {
            CandidateScore identity = scoreCandidateAgainstReferenceTag(
                frame_bgr,
                corners,
                pattern_it->second);
            std::cout << " inner_cells=" << identity.matched_inner_cells
                      << "/" << identity.visible_inner_cells
                      << " inner_score=" << identity.inner_cell_score;
        }
        std::cout << "\n";
    }
    for (const SoftTag& tag : geometry.soft_tags) {
        std::cout << "[TAG] frame=" << geometry.frame_index
                  << " kind=partial id=" << tag.id
                  << " source=" << tag.source
                  << " score=" << std::fixed << std::setprecision(3) << tag.score
                  << " visible_fraction=" << tag.visible_fraction
                  << " corners=";
        print_corners(tag.frame_corners);
        auto pattern_it = reference.reference_tag_patterns.find(tag.id);
        if (pattern_it != reference.reference_tag_patterns.end()) {
            CandidateScore expected = scoreCandidateAgainstReferenceTag(
                frame_bgr,
                tag.frame_corners,
                pattern_it->second);
            int best_id = -1;
            double best_score = -1.0;
            double second_score = -1.0;
            for (const auto& [candidate_id, candidate_pattern] : reference.reference_tag_patterns) {
                CandidateScore identity = scoreCandidateAgainstReferenceTag(
                    frame_bgr,
                    tag.frame_corners,
                    candidate_pattern);
                if (identity.inner_cell_score > best_score) {
                    second_score = best_score;
                    best_score = identity.inner_cell_score;
                    best_id = candidate_id;
                } else if (identity.inner_cell_score > second_score) {
                    second_score = identity.inner_cell_score;
                }
            }
            std::cout << " inner_cells=" << expected.matched_inner_cells
                      << "/" << expected.visible_inner_cells
                      << " inner_score=" << expected.inner_cell_score
                      << " best_inner_id=" << best_id
                      << " best_inner_score=" << best_score
                      << " inner_margin=" << (best_score - second_score);
        }
        std::cout << "\n";
    }
    for (const SoftTag& candidate : geometry.soft_tag_candidates) {
        const bool used = std::any_of(
            geometry.soft_tags.begin(),
            geometry.soft_tags.end(),
            [&](const SoftTag& tag) {
                return tag.id == candidate.id && tag.source == candidate.source;
            });
        if (used) {
            continue;
        }
        std::cout << "[TAG] frame=" << geometry.frame_index
                  << " kind=partial-candidate-rejected id=" << candidate.id
                  << " source=" << candidate.source
                  << " score=" << std::fixed << std::setprecision(3) << candidate.score
                  << " visible_fraction=" << candidate.visible_fraction
                  << " corners=";
        print_corners(candidate.frame_corners);
        CandidateScore expected_pattern;
        auto expected_pattern_it = reference.reference_tag_patterns.find(candidate.id);
        if (expected_pattern_it != reference.reference_tag_patterns.end()) {
            expected_pattern = scoreCandidateAgainstReferenceTag(
                frame_bgr,
                candidate.frame_corners,
                expected_pattern_it->second);
        }
        int best_inner_id = -1;
        double best_inner_score = -1.0;
        double second_inner_score = -1.0;
        for (const auto& [pattern_id, pattern] : reference.reference_tag_patterns) {
            CandidateScore identity = scoreCandidateAgainstReferenceTag(
                frame_bgr,
                candidate.frame_corners,
                pattern);
            if (identity.inner_cell_score > best_inner_score) {
                second_inner_score = best_inner_score;
                best_inner_score = identity.inner_cell_score;
                best_inner_id = pattern_id;
            } else if (identity.inner_cell_score > second_inner_score) {
                second_inner_score = identity.inner_cell_score;
            }
        }
        std::cout << " inner_cells=" << expected_pattern.matched_inner_cells
                  << "/" << expected_pattern.visible_inner_cells
                  << " inner_score=" << expected_pattern.inner_cell_score
                  << " best_inner_id=" << best_inner_id
                  << " best_inner_score=" << best_inner_score
                  << " inner_margin=" << (best_inner_score - second_inner_score);
        auto ref_it = reference.reference_tags.detections.find(candidate.id);
        if (ref_it != reference.reference_tags.detections.end() &&
                !geometry.H_ref_to_frame.empty()) {
            std::vector<cv::Point2f> projected = projectQuad(
                geometry.H_ref_to_frame,
                ref_it->second);
            if (projected.size() == 4 && candidate.frame_corners.size() == 4) {
                double mean_error = 0.0;
                double max_error = 0.0;
                for (size_t i = 0; i < 4; ++i) {
                    const double error = cv::norm(projected[i] - candidate.frame_corners[i]);
                    mean_error += error;
                    max_error = std::max(max_error, error);
                }
                mean_error /= 4.0;
                const double tag_size = std::max(1.0, projectedTagSize(candidate.frame_corners));
                const double allowed_mean = std::max(8.0, tag_size * 0.10);
                const double allowed_max = std::max(18.0, tag_size * 0.22);
                std::cout << " reprojection_mean=" << mean_error
                          << " allowed_mean=" << allowed_mean
                          << " reprojection_max=" << max_error
                          << " allowed_max=" << allowed_max;
            }
        }
        std::cout << "\n";
    }
}

bool estimateFromSparseTagsAndSoftTags(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& seed_H_ref_to_frame,
    GeometryResult* result,
    bool fast_seed_hold = false,
    bool center_consensus = false,
    bool partial_cell_pattern = false) {
    if (result == nullptr || result->common_tag_count < 1 || result->common_tag_count >= 3) {
        return false;
    }
    ScopedStageTimer fallback_timer(&g_geometry_timing.sparse_soft_fallback);

    std::vector<int> hard_tag_ids;
    std::vector<cv::Point2f> hard_src_pts;
    std::vector<cv::Point2f> hard_dst_pts;
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        auto frame_it = result->frame_tags.detections.find(id);
        if (frame_it != result->frame_tags.detections.end()) {
            hard_tag_ids.push_back(id);
            hard_src_pts.insert(hard_src_pts.end(), ref_corners.begin(), ref_corners.end());
            hard_dst_pts.insert(hard_dst_pts.end(), frame_it->second.begin(), frame_it->second.end());
        }
    }
    if (hard_tag_ids.empty()) {
        return false;
    }

    bool used_external_seed = false;
    {
        ScopedStageTimer seed_timer(&g_geometry_timing.fallback_seed_setup);
        if (!seed_H_ref_to_frame.empty()) {
            double mean_seed_error = 0.0;
            int seed_points = 0;
            for (int id : hard_tag_ids) {
                std::vector<cv::Point2f> projected = projectQuad(seed_H_ref_to_frame, reference.reference_tags.detections.at(id));
                if (projected.size() != 4) {
                    continue;
                }
                const std::vector<cv::Point2f>& observed = result->frame_tags.detections.at(id);
                for (int i = 0; i < 4; ++i) {
                    mean_seed_error += cv::norm(projected[static_cast<size_t>(i)] - observed[static_cast<size_t>(i)]);
                    ++seed_points;
                }
            }
            if (seed_points > 0) {
                mean_seed_error /= static_cast<double>(seed_points);
                if (mean_seed_error <= 35.0) {
                    result->H_seed_ref_to_frame = seed_H_ref_to_frame.clone();
                    result->used_previous_homography_seed = true;
                    used_external_seed = true;
                }
            }
        }

        if (!used_external_seed && hard_tag_ids.size() == 1) {
            const int hard_tag_id = hard_tag_ids.front();
            const std::vector<cv::Point2f>& hard_ref = reference.reference_tags.detections.at(hard_tag_id);
            const std::vector<cv::Point2f>& hard_frame = result->frame_tags.detections.at(hard_tag_id);
            result->H_seed_ref_to_frame = cv::getPerspectiveTransform(
                orderCornersTLTRBRBL(hard_ref),
                orderCornersTLTRBRBL(hard_frame));
        } else if (!used_external_seed) {
            result->H_seed_ref_to_frame = cv::findHomography(hard_src_pts, hard_dst_pts, cv::RANSAC, 6.0);
        }
    }
    if (result->H_seed_ref_to_frame.empty()) {
        return false;
    }
    const double seed_hard_error = meanHardTagReprojectionError(
        reference,
        result->frame_tags,
        hard_tag_ids,
        result->H_seed_ref_to_frame);
    result->used_single_tag_fallback = true;
    if (fast_seed_hold && used_external_seed) {
        result->H_ref_to_frame = result->H_seed_ref_to_frame.clone();
        result->ok = true;
        return true;
    }
    FrameMaskCache frame_masks;

    // One or two decoded tags give us a sparse sheet pose. That pose is useful
    // as a hint for where known neighboring tags should be, but it can be less
    // stable than a homography supported by several spatially separated tags.
    const int anchor_hard_tag_id = hard_tag_ids.front();
    std::vector<SoftTag> soft_tags = recoverSoftTagsFromRejectedCandidates(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        anchor_hard_tag_id);
    std::vector<SoftTag> color_tags = recoverSoftTagsFromColorMasks(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        anchor_hard_tag_id,
        soft_tags,
        &frame_masks);
    soft_tags.insert(
        soft_tags.end(),
        std::make_move_iterator(color_tags.begin()),
        std::make_move_iterator(color_tags.end()));
    std::vector<SoftTag> seed_pattern_tags = recoverSoftTagsFromSeedPatternSupport(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        hard_tag_ids.size() == 1 ? hard_tag_ids.front() : -1,
        soft_tags);
    soft_tags.insert(
        soft_tags.end(),
        std::make_move_iterator(seed_pattern_tags.begin()),
        std::make_move_iterator(seed_pattern_tags.end()));
    std::vector<SoftTag> seed_color_tags = recoverSoftTagsFromSeedColorSupport(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        hard_tag_ids.size() == 1 ? hard_tag_ids.front() : -1,
        soft_tags,
        &frame_masks);
    soft_tags.insert(
        soft_tags.end(),
        std::make_move_iterator(seed_color_tags.begin()),
        std::make_move_iterator(seed_color_tags.end()));
    result->soft_tag_candidates = soft_tags;
    int rejected_soft_count = static_cast<int>(std::count_if(
        soft_tags.begin(),
        soft_tags.end(),
        [](const SoftTag& tag) { return tag.source == "rejected"; }));
    if (rejected_soft_count > 0) {
        std::vector<SoftTag> prioritized;
        for (const SoftTag& tag : soft_tags) {
            if (tag.source == "rejected" || tag.score >= 0.72 || tag.visible_fraction >= 0.45) {
                prioritized.push_back(tag);
            }
        }
        soft_tags = std::move(prioritized);
    }

    // Experimental path: only independently localized partials may influence
    // geometry. Seed/seed-color observations are useful confidence signals,
    // but their positions come from H_seed and therefore cannot validate it.
    if (center_consensus) {
        std::vector<SoftTag> independent_tags;
        for (const SoftTag& tag : soft_tags) {
            if (tag.source == "rejected" || tag.source == "color") {
                independent_tags.push_back(tag);
            }
        }
        soft_tags = std::move(independent_tags);
    }
    if (partial_cell_pattern) {
        std::vector<SoftTag> validated_tags;
        for (const SoftTag& tag : soft_tags) {
            if (tag.source != "rejected" && tag.source != "color") {
                continue;
            }
            auto expected_it = reference.reference_tag_patterns.find(tag.id);
            if (expected_it == reference.reference_tag_patterns.end()) {
                continue;
            }
            CandidateScore expected = scoreCandidateAgainstReferenceTag(
                frame_bgr, tag.frame_corners, expected_it->second);
            int best_id = -1;
            double best_score = -1.0;
            double second_score = -1.0;
            for (const auto& [candidate_id, pattern] : reference.reference_tag_patterns) {
                CandidateScore identity = scoreCandidateAgainstReferenceTag(
                    frame_bgr, tag.frame_corners, pattern);
                if (identity.inner_cell_score > best_score) {
                    second_score = best_score;
                    best_score = identity.inner_cell_score;
                    best_id = candidate_id;
                } else if (identity.inner_cell_score > second_score) {
                    second_score = identity.inner_cell_score;
                }
            }
            if (expected.visible_inner_cells >= 10 &&
                    expected.inner_cell_score >= 0.64 &&
                    best_id == tag.id &&
                    best_score - second_score >= 0.04) {
                validated_tags.push_back(tag);
            }
        }
        soft_tags = std::move(validated_tags);
    }
    const bool use_soft_centers = center_consensus || partial_cell_pattern;

    if (partial_cell_pattern && !soft_tags.empty()) {
        std::vector<cv::Point2f> support_src = hard_src_pts;
        std::vector<cv::Point2f> support_dst = hard_dst_pts;
        for (const SoftTag& tag : soft_tags) {
            support_src.push_back(quadCenter(reference.reference_tags.detections.at(tag.id)));
            support_dst.push_back(quadCenter(tag.frame_corners));
        }
        result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
            frame_bgr,
            reference,
            result->frame_tags,
            hard_tag_ids,
            result->H_seed_ref_to_frame,
            support_src,
            support_dst,
            false,
            true);
        result->soft_tags = soft_tags;
        result->soft_tag_count = static_cast<int>(result->soft_tags.size());
        result->ok = !result->H_ref_to_frame.empty();
        return result->ok;
    }

    auto sparseEvidenceHasSpread = [&]() {
        std::vector<cv::Point2f> centers;
        for (int id : hard_tag_ids) {
            centers.push_back(quadCenter(reference.reference_tags.detections.at(id)));
        }
        for (const SoftTag& soft : soft_tags) {
            if (soft.source == "rejected" || (use_soft_centers && soft.source == "color")) {
                centers.push_back(quadCenter(reference.reference_tags.detections.at(soft.id)));
            }
        }
        if (centers.size() < 2) {
            return false;
        }
        float min_x = centers.front().x;
        float max_x = centers.front().x;
        float min_y = centers.front().y;
        float max_y = centers.front().y;
        for (const cv::Point2f& center : centers) {
            min_x = std::min(min_x, center.x);
            max_x = std::max(max_x, center.x);
            min_y = std::min(min_y, center.y);
            max_y = std::max(max_y, center.y);
        }
        double x_fraction = (max_x - min_x) / std::max(1.0, static_cast<double>(reference.reference_bgr.cols));
        double y_fraction = (max_y - min_y) / std::max(1.0, static_cast<double>(reference.reference_bgr.rows));
        return x_fraction >= 0.22 && y_fraction >= 0.18;
    };

    if (used_external_seed && !sparseEvidenceHasSpread() &&
            !(partial_cell_pattern && !soft_tags.empty())) {
        std::vector<cv::Point2f> guard_src_pts = hard_src_pts;
        std::vector<cv::Point2f> guard_dst_pts = hard_dst_pts;
        if (!use_soft_centers) {
            appendSoftTagCorrespondences(reference, soft_tags, &guard_src_pts, &guard_dst_pts);
        }
        result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
            frame_bgr,
            reference,
            result->frame_tags,
            hard_tag_ids,
            result->H_seed_ref_to_frame,
            guard_src_pts,
            guard_dst_pts,
            false);
        result->soft_tags = std::move(soft_tags);
        result->soft_tag_count = static_cast<int>(result->soft_tags.size());
        result->ok = true;
        return true;
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    src_pts.insert(src_pts.end(), hard_src_pts.begin(), hard_src_pts.end());
    dst_pts.insert(dst_pts.end(), hard_dst_pts.begin(), hard_dst_pts.end());
    if (used_external_seed && !use_soft_centers) {
        std::vector<cv::Point2f> ref_sheet_corners = referenceImageCorners(reference.reference_bgr.size());
        std::vector<cv::Point2f> seed_sheet_corners = projectQuad(result->H_seed_ref_to_frame, ref_sheet_corners);
        if (seed_sheet_corners.size() == 4) {
            src_pts.insert(src_pts.end(), ref_sheet_corners.begin(), ref_sheet_corners.end());
            dst_pts.insert(dst_pts.end(), seed_sheet_corners.begin(), seed_sheet_corners.end());
        }
    }
    for (const SoftTag& soft : soft_tags) {
        const std::vector<cv::Point2f>& ref_corners = reference.reference_tags.detections.at(soft.id);
        if (use_soft_centers) {
            src_pts.push_back(quadCenter(ref_corners));
            dst_pts.push_back(quadCenter(soft.frame_corners));
        } else {
            src_pts.insert(src_pts.end(), ref_corners.begin(), ref_corners.end());
            dst_pts.insert(dst_pts.end(), soft.frame_corners.begin(), soft.frame_corners.end());
        }
    }

    cv::Mat inlier_mask;
    cv::Mat H;
    {
        ScopedStageTimer fit_timer(&g_geometry_timing.fallback_homography_fit);
        H = use_soft_centers
            ? cv::findHomography(src_pts, dst_pts, 0)
            : cv::findHomography(src_pts, dst_pts, cv::RANSAC, 6.0, inlier_mask);
    }
    int inlier_count = use_soft_centers
        ? static_cast<int>(src_pts.size())
        : (!inlier_mask.empty() ? cv::countNonZero(inlier_mask) : 0);
    int min_inliers = use_soft_centers ? 5 : (result->common_tag_count == 1 ? 8 : 6);
    if (H.empty() || inlier_count < min_inliers) {
        return false;
    }

    std::vector<SoftTag> inlier_soft_tags;
    for (const SoftTag& soft : soft_tags) {
        const std::vector<cv::Point2f>& ref_corners = reference.reference_tags.detections.at(soft.id);
        std::vector<cv::Point2f> projected = projectQuad(H, ref_corners);
        if (projected.size() != 4) {
            continue;
        }
        double mean_error = 0.0;
        double max_error = 0.0;
        for (int i = 0; i < 4; ++i) {
            double error = cv::norm(projected[static_cast<size_t>(i)] - soft.frame_corners[static_cast<size_t>(i)]);
            mean_error += error;
            max_error = std::max(max_error, error);
        }
        mean_error /= 4.0;
        double tag_size = std::max(1.0, projectedTagSize(soft.frame_corners));
        double allowed_mean = std::max(8.0, tag_size * 0.10);
        double allowed_max = std::max(18.0, tag_size * 0.22);
        const double center_error = cv::norm(
            quadCenter(projected) - quadCenter(soft.frame_corners));
        const double allowed_center = std::max(7.0, tag_size * 0.16);
        if ((use_soft_centers && center_error <= allowed_center) ||
                (!use_soft_centers && mean_error <= allowed_mean && max_error <= allowed_max)) {
            inlier_soft_tags.push_back(soft);
        }
    }

    int min_soft_tags = result->common_tag_count == 1 ? 1 : 0;
    if (static_cast<int>(inlier_soft_tags.size()) < min_soft_tags) {
        return false;
    }
    if (inlier_soft_tags.size() < soft_tags.size()) {
        src_pts.clear();
        dst_pts.clear();
        src_pts.insert(src_pts.end(), hard_src_pts.begin(), hard_src_pts.end());
        dst_pts.insert(dst_pts.end(), hard_dst_pts.begin(), hard_dst_pts.end());
        if (used_external_seed && !use_soft_centers) {
            std::vector<cv::Point2f> ref_sheet_corners = referenceImageCorners(reference.reference_bgr.size());
            std::vector<cv::Point2f> seed_sheet_corners = projectQuad(result->H_seed_ref_to_frame, ref_sheet_corners);
            if (seed_sheet_corners.size() == 4) {
                src_pts.insert(src_pts.end(), ref_sheet_corners.begin(), ref_sheet_corners.end());
                dst_pts.insert(dst_pts.end(), seed_sheet_corners.begin(), seed_sheet_corners.end());
            }
        }
        for (const SoftTag& soft : inlier_soft_tags) {
            const std::vector<cv::Point2f>& ref_corners = reference.reference_tags.detections.at(soft.id);
            if (use_soft_centers) {
                src_pts.push_back(quadCenter(ref_corners));
                dst_pts.push_back(quadCenter(soft.frame_corners));
            } else {
                src_pts.insert(src_pts.end(), ref_corners.begin(), ref_corners.end());
                dst_pts.insert(dst_pts.end(), soft.frame_corners.begin(), soft.frame_corners.end());
            }
        }
        cv::Mat refined_H;
        {
            ScopedStageTimer fit_timer(&g_geometry_timing.fallback_homography_fit);
            refined_H = use_soft_centers
                ? cv::findHomography(src_pts, dst_pts, 0)
                : cv::findHomography(src_pts, dst_pts, cv::RANSAC, 6.0);
        }
        if (!refined_H.empty()) {
            H = refined_H;
            soft_tags = std::move(inlier_soft_tags);
        }
    }

    std::vector<cv::Point2f> guard_src_pts = hard_src_pts;
    std::vector<cv::Point2f> guard_dst_pts = hard_dst_pts;
    if (use_soft_centers) {
        for (const SoftTag& soft : soft_tags) {
            guard_src_pts.push_back(quadCenter(reference.reference_tags.detections.at(soft.id)));
            guard_dst_pts.push_back(quadCenter(soft.frame_corners));
        }
    } else {
        appendSoftTagCorrespondences(reference, soft_tags, &guard_src_pts, &guard_dst_pts);
    }

    H = refineHomographyWithSheetEdges(
        frame_bgr,
        reference,
        H,
        src_pts,
        dst_pts,
        &frame_masks,
        use_soft_centers);

    SheetEdgeAlignment edge_alignment;
    {
        ScopedStageTimer edge_timer(&g_geometry_timing.sheet_edge_refinement);
        edge_alignment = checkSheetEdgeAlignment(
            frame_bgr,
            reference,
            H,
            &frame_masks,
            use_soft_centers);
    }
    if (edge_alignment.usable && !edge_alignment.ok) {
        if (used_external_seed && !use_soft_centers) {
            result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
                frame_bgr,
                reference,
                result->frame_tags,
                hard_tag_ids,
                result->H_seed_ref_to_frame,
                guard_src_pts,
                guard_dst_pts,
                false);
            result->soft_tags = std::move(soft_tags);
            result->soft_tag_count = static_cast<int>(result->soft_tags.size());
            result->ok = true;
            return true;
        }
        return false;
    }

    if (used_external_seed && !use_soft_centers) {
        double max_jump = projectedQuadMaxJump(H, result->H_seed_ref_to_frame, reference.reference_bgr.size());
        double allowed_jump = std::max(
            24.0,
            std::hypot(static_cast<double>(frame_bgr.cols), static_cast<double>(frame_bgr.rows)) * 0.035);
        if (max_jump > allowed_jump) {
            result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
                frame_bgr,
                reference,
                result->frame_tags,
                hard_tag_ids,
                result->H_seed_ref_to_frame,
                guard_src_pts,
                guard_dst_pts,
                false);
            result->soft_tags = std::move(soft_tags);
            result->soft_tag_count = static_cast<int>(result->soft_tags.size());
            result->ok = true;
            return true;
        }
    }

    double final_hard_error = meanHardTagReprojectionError(
        reference,
        result->frame_tags,
        hard_tag_ids,
        H);
    if (hard_tag_ids.size() >= 2 &&
            seed_hard_error <= 1.5 &&
            final_hard_error > std::max(3.0, seed_hard_error + 2.0)) {
        result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
            frame_bgr,
            reference,
            result->frame_tags,
            hard_tag_ids,
            result->H_seed_ref_to_frame,
            guard_src_pts,
            guard_dst_pts,
            false);
        result->soft_tags = std::move(soft_tags);
        result->soft_tag_count = static_cast<int>(result->soft_tags.size());
        result->ok = true;
        return true;
    }

    result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
        frame_bgr,
        reference,
        result->frame_tags,
        hard_tag_ids,
        H,
        guard_src_pts,
        guard_dst_pts,
        false);
    result->soft_tags = std::move(soft_tags);
    result->soft_tag_count = static_cast<int>(result->soft_tags.size());
    result->ok = true;
    return true;
}

bool estimateFromPreviousSeedAndSoftTags(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const cv::Mat& seed_H_ref_to_frame,
    GeometryResult* result) {
    if (result == nullptr || result->common_tag_count != 0 || seed_H_ref_to_frame.empty()) {
        return false;
    }
    ScopedStageTimer fallback_timer(&g_geometry_timing.previous_seed_fallback);

    {
        ScopedStageTimer seed_timer(&g_geometry_timing.fallback_seed_setup);
        result->H_seed_ref_to_frame = seed_H_ref_to_frame.clone();
        result->used_single_tag_fallback = true;
        result->used_previous_homography_seed = true;
    }
    FrameMaskCache frame_masks;

    // With no decoded tags, the previous frame's successful homography is only
    // a motion prior. Prefer fresh soft-tag evidence when available, but allow
    // a pure hold if the propagated sheet still agrees with the visible
    // top/bottom sheet boundaries.
    std::vector<SoftTag> soft_tags = recoverSoftTagsFromRejectedCandidates(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        -1);
    std::vector<SoftTag> color_tags = recoverSoftTagsFromColorMasks(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        -1,
        soft_tags,
        &frame_masks);
    soft_tags.insert(
        soft_tags.end(),
        std::make_move_iterator(color_tags.begin()),
        std::make_move_iterator(color_tags.end()));
    std::vector<SoftTag> seed_pattern_tags = recoverSoftTagsFromSeedPatternSupport(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        -1,
        soft_tags);
    soft_tags.insert(
        soft_tags.end(),
        std::make_move_iterator(seed_pattern_tags.begin()),
        std::make_move_iterator(seed_pattern_tags.end()));
    std::vector<SoftTag> seed_color_tags = recoverSoftTagsFromSeedColorSupport(
        frame_bgr,
        reference,
        result->frame_tags,
        result->H_seed_ref_to_frame,
        -1,
        soft_tags,
        &frame_masks);
    soft_tags.insert(
        soft_tags.end(),
        std::make_move_iterator(seed_color_tags.begin()),
        std::make_move_iterator(seed_color_tags.end()));
    result->soft_tag_candidates = soft_tags;
    int seed_supported_count = static_cast<int>(std::count_if(
        soft_tags.begin(),
        soft_tags.end(),
        [](const SoftTag& tag) { return tag.source.rfind("seed", 0) == 0; }));
    if (soft_tags.size() >= 2 && seed_supported_count == static_cast<int>(soft_tags.size())) {
        std::vector<cv::Point2f> guard_src_pts;
        std::vector<cv::Point2f> guard_dst_pts;
        appendSoftTagCorrespondences(reference, soft_tags, &guard_src_pts, &guard_dst_pts);
        result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
            frame_bgr,
            reference,
            result->frame_tags,
            {},
            result->H_seed_ref_to_frame,
            guard_src_pts,
            guard_dst_pts);
        result->soft_tags = std::move(soft_tags);
        result->soft_tag_count = static_cast<int>(result->soft_tags.size());
        result->ok = true;
        return true;
    }

    if (soft_tags.size() < 2) {
        SheetEdgeAlignment seed_edge_alignment;
        {
            ScopedStageTimer edge_timer(&g_geometry_timing.sheet_edge_refinement);
            seed_edge_alignment = checkSheetEdgeAlignment(
                frame_bgr,
                reference,
                result->H_seed_ref_to_frame,
                &frame_masks);
        }
        if (seed_edge_alignment.usable && seed_edge_alignment.ok) {
            std::vector<cv::Point2f> guard_src_pts;
            std::vector<cv::Point2f> guard_dst_pts;
            appendSoftTagCorrespondences(reference, soft_tags, &guard_src_pts, &guard_dst_pts);
            result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
                frame_bgr,
                reference,
                result->frame_tags,
                {},
                result->H_seed_ref_to_frame,
                guard_src_pts,
                guard_dst_pts);
            result->soft_tags = std::move(soft_tags);
            result->soft_tag_count = static_cast<int>(result->soft_tags.size());
            result->ok = true;
            return true;
        }
        return false;
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    std::vector<cv::Point2f> ref_sheet_corners = referenceImageCorners(reference.reference_bgr.size());
    std::vector<cv::Point2f> seed_sheet_corners = projectQuad(result->H_seed_ref_to_frame, ref_sheet_corners);
    if (seed_sheet_corners.size() == 4) {
        src_pts.insert(src_pts.end(), ref_sheet_corners.begin(), ref_sheet_corners.end());
        dst_pts.insert(dst_pts.end(), seed_sheet_corners.begin(), seed_sheet_corners.end());
    }
    for (const SoftTag& soft : soft_tags) {
        const std::vector<cv::Point2f>& ref_corners = reference.reference_tags.detections.at(soft.id);
        src_pts.insert(src_pts.end(), ref_corners.begin(), ref_corners.end());
        dst_pts.insert(dst_pts.end(), soft.frame_corners.begin(), soft.frame_corners.end());
    }

    cv::Mat inlier_mask;
    cv::Mat H;
    {
        ScopedStageTimer fit_timer(&g_geometry_timing.fallback_homography_fit);
        H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 6.0, inlier_mask);
    }
    int inlier_count = !inlier_mask.empty() ? cv::countNonZero(inlier_mask) : 0;
    int min_inlier_count = soft_tags.size() <= 2 ? 7 : 8;
    if (H.empty() || inlier_count < min_inlier_count) {
        return false;
    }

    std::vector<SoftTag> inlier_soft_tags;
    for (const SoftTag& soft : soft_tags) {
        const std::vector<cv::Point2f>& ref_corners = reference.reference_tags.detections.at(soft.id);
        std::vector<cv::Point2f> projected = projectQuad(H, ref_corners);
        if (projected.size() != 4) {
            continue;
        }
        double mean_error = 0.0;
        double max_error = 0.0;
        for (int i = 0; i < 4; ++i) {
            double error = cv::norm(projected[static_cast<size_t>(i)] - soft.frame_corners[static_cast<size_t>(i)]);
            mean_error += error;
            max_error = std::max(max_error, error);
        }
        mean_error /= 4.0;
        double tag_size = std::max(1.0, projectedTagSize(soft.frame_corners));
        double allowed_mean = std::max(8.0, tag_size * 0.10);
        double allowed_max = std::max(18.0, tag_size * 0.22);
        if (mean_error <= allowed_mean && max_error <= allowed_max) {
            inlier_soft_tags.push_back(soft);
        }
    }
    if (inlier_soft_tags.size() < 2) {
        return false;
    }
    if (inlier_soft_tags.size() < soft_tags.size()) {
        src_pts.clear();
        dst_pts.clear();
        if (seed_sheet_corners.size() == 4) {
            src_pts.insert(src_pts.end(), ref_sheet_corners.begin(), ref_sheet_corners.end());
            dst_pts.insert(dst_pts.end(), seed_sheet_corners.begin(), seed_sheet_corners.end());
        }
        for (const SoftTag& soft : inlier_soft_tags) {
            const std::vector<cv::Point2f>& ref_corners = reference.reference_tags.detections.at(soft.id);
            src_pts.insert(src_pts.end(), ref_corners.begin(), ref_corners.end());
            dst_pts.insert(dst_pts.end(), soft.frame_corners.begin(), soft.frame_corners.end());
        }
        cv::Mat refined_H;
        {
            ScopedStageTimer fit_timer(&g_geometry_timing.fallback_homography_fit);
            refined_H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 6.0);
        }
        if (!refined_H.empty()) {
            H = refined_H;
            soft_tags = std::move(inlier_soft_tags);
        }
    }

    std::vector<cv::Point2f> guard_src_pts;
    std::vector<cv::Point2f> guard_dst_pts;
    appendSoftTagCorrespondences(reference, soft_tags, &guard_src_pts, &guard_dst_pts);

    H = refineHomographyWithSheetEdges(frame_bgr, reference, H, src_pts, dst_pts, &frame_masks);

    SheetEdgeAlignment edge_alignment;
    {
        ScopedStageTimer edge_timer(&g_geometry_timing.sheet_edge_refinement);
        edge_alignment = checkSheetEdgeAlignment(frame_bgr, reference, H, &frame_masks);
    }
    if (edge_alignment.usable && !edge_alignment.ok) {
        result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
            frame_bgr,
            reference,
            result->frame_tags,
            {},
            result->H_seed_ref_to_frame,
            guard_src_pts,
            guard_dst_pts);
        result->soft_tags = std::move(soft_tags);
        result->soft_tag_count = static_cast<int>(result->soft_tags.size());
        result->ok = true;
        return true;
    }

    double max_jump = projectedQuadMaxJump(H, result->H_seed_ref_to_frame, reference.reference_bgr.size());
    double allowed_jump = std::max(
        24.0,
        std::hypot(static_cast<double>(frame_bgr.cols), static_cast<double>(frame_bgr.rows)) * 0.035);
    if (max_jump > allowed_jump) {
        result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
            frame_bgr,
            reference,
            result->frame_tags,
            {},
            result->H_seed_ref_to_frame,
            guard_src_pts,
            guard_dst_pts);
        result->soft_tags = std::move(soft_tags);
        result->soft_tag_count = static_cast<int>(result->soft_tags.size());
        result->ok = true;
        return true;
    }

    result->H_ref_to_frame = refineFallbackHomographyWithTagPatternShift(
        frame_bgr,
        reference,
        result->frame_tags,
        {},
        H,
        guard_src_pts,
        guard_dst_pts);
    result->soft_tags = std::move(soft_tags);
    result->soft_tag_count = static_cast<int>(result->soft_tags.size());
    result->ok = true;
    return true;
}

std::vector<cv::Point> toIntPoints(const std::vector<cv::Point2f>& points) {
    std::vector<cv::Point> out;
    out.reserve(points.size());
    for (const cv::Point2f& point : points) {
        out.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }
    return out;
}

void drawPolyline(cv::Mat* image, const std::vector<cv::Point2f>& points, const cv::Scalar& color, int thickness) {
    if (image == nullptr || image->empty() || points.size() != 4) {
        return;
    }
    std::vector<cv::Point> int_points = toIntPoints(points);
    cv::polylines(*image, std::vector<std::vector<cv::Point>>{int_points}, true, color, thickness);
}

cv::Point toIntPoint(const cv::Point2f& point) {
    return cv::Point(cvRound(point.x), cvRound(point.y));
}

void drawH5CornerOutline(
    cv::Mat* image,
    const H5CornerTrace* h5_corners,
    int frame_index) {
    if (image == nullptr || image->empty() || h5_corners == nullptr || !h5_corners->available ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->frame_corners.size() ||
            h5_corners->source_size.width <= 0.0f ||
            h5_corners->source_size.height <= 0.0f) {
        return;
    }

    const std::array<cv::Point2f, 4>& corners =
        h5_corners->frame_corners[static_cast<size_t>(frame_index)];
    double raw_area = std::abs(cv::contourArea(std::vector<cv::Point2f>(corners.begin(), corners.end())));
    if (raw_area <= 1.0) {
        return;
    }

    float scale_x = static_cast<float>(image->cols) / h5_corners->source_size.width;
    float scale_y = static_cast<float>(image->rows) / h5_corners->source_size.height;
    std::vector<cv::Point> scaled;
    scaled.reserve(4);
    for (const cv::Point2f& point : corners) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return;
        }
        scaled.emplace_back(cvRound(point.x * scale_x), cvRound(point.y * scale_y));
    }
    cv::polylines(*image, std::vector<std::vector<cv::Point>>{scaled}, true, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
}

bool h5StylusPointInFrameBySourceScale(
    const H5CornerTrace* h5_corners,
    int frame_index,
    cv::Size frame_size,
    cv::Point2f* frame_point,
    bool require_touch = true) {
    if (frame_point == nullptr ||
            h5_corners == nullptr ||
            !h5_corners->stylus_available ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_points.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_pressure.size() ||
            h5_corners->source_size.width <= 0.0f ||
            h5_corners->source_size.height <= 0.0f) {
        return false;
    }
    if (require_touch && h5_corners->stylus_pressure[static_cast<size_t>(frame_index)] <= 0.0f) {
        return false;
    }
    const cv::Point2f src = h5_corners->stylus_points[static_cast<size_t>(frame_index)];
    if (!std::isfinite(src.x) || !std::isfinite(src.y) || src.x < 0.0f || src.y < 0.0f) {
        return false;
    }
    *frame_point = cv::Point2f(
        src.x * static_cast<float>(frame_size.width) / h5_corners->source_size.width,
        src.y * static_cast<float>(frame_size.height) / h5_corners->source_size.height);
    return std::isfinite(frame_point->x) && std::isfinite(frame_point->y);
}

bool h5StylusPointOnV3Surface(
    const H5CornerTrace* h5_corners,
    int frame_index,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    cv::Size frame_size,
    cv::Point2f* frame_point,
    bool require_touch = true) {
    (void)frame_size;
    if (!geometry.ok || geometry.H_ref_to_frame.empty() || frame_point == nullptr) {
        return false;
    }
    cv::Point2f ref_point;
    if (!h5StylusPointInReferenceSurface(
            h5_corners,
            frame_index,
            reference.reference_bgr.size(),
            &ref_point,
            require_touch)) {
        return false;
    }
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(std::vector<cv::Point2f>{ref_point}, projected, geometry.H_ref_to_frame);
    if (projected.size() != 1 || !std::isfinite(projected[0].x) || !std::isfinite(projected[0].y)) {
        return false;
    }
    *frame_point = projected[0];
    return true;
}

bool h5StylusTiltRayOnV3Surface(
    const H5CornerTrace* h5_corners,
    int frame_index,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    cv::Size frame_size,
    cv::Point2f* ray_start,
    cv::Point2f* ray_end,
    bool require_touch = true);

bool refineStylusWedgeWidthsFromEdges(
    const cv::Mat& frame_bgr,
    const cv::Point2f& tip,
    const cv::Point2f& axis,
    float wedge_length,
    float* positive_width,
    float* negative_width) {
    if (frame_bgr.empty() ||
            positive_width == nullptr ||
            negative_width == nullptr ||
            wedge_length < 8.0f) {
        return false;
    }

    cv::Mat gray;
    if (frame_bgr.channels() == 1) {
        gray = frame_bgr;
    } else {
        cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.0);
    cv::Mat edges;
    cv::Canny(blurred, edges, 35.0, 105.0);

    const cv::Point2f normal(-axis.y, axis.x);
    std::vector<double> positive_offsets;
    std::vector<double> negative_offsets;
    const float min_along = std::max(4.0f, wedge_length * 0.25f);
    const float max_along = std::min(46.0f, wedge_length);
    for (float along = min_along; along <= max_along; along += 2.0f) {
        for (float offset = 3.0f; offset <= 24.0f; offset += 1.0f) {
            const cv::Point p = toIntPoint(tip + axis * along + normal * offset);
            if (p.x >= 0 && p.y >= 0 && p.x < edges.cols && p.y < edges.rows && edges.at<uchar>(p) != 0) {
                positive_offsets.push_back(static_cast<double>(offset));
                break;
            }
        }
        for (float offset = 3.0f; offset <= 24.0f; offset += 1.0f) {
            const cv::Point p = toIntPoint(tip + axis * along - normal * offset);
            if (p.x >= 0 && p.y >= 0 && p.x < edges.cols && p.y < edges.rows && edges.at<uchar>(p) != 0) {
                negative_offsets.push_back(static_cast<double>(offset));
                break;
            }
        }
    }

    bool refined = false;
    if (positive_offsets.size() >= 2) {
        *positive_width = static_cast<float>(std::clamp(medianValue(positive_offsets), 4.0, 18.0));
        refined = true;
    }
    if (negative_offsets.size() >= 2) {
        *negative_width = static_cast<float>(std::clamp(medianValue(negative_offsets), 4.0, 18.0));
        refined = true;
    }
    return refined;
}

bool fitStylusTipEdgeSlopesFromFrame(
    const cv::Mat& frame_bgr,
    const cv::Point2f& tip,
    const cv::Point2f& axis,
    float search_length,
    float* positive_slope,
    float* negative_slope) {
    if (frame_bgr.empty() ||
            positive_slope == nullptr ||
            negative_slope == nullptr ||
            search_length < 10.0f) {
        return false;
    }

    cv::Mat gray;
    if (frame_bgr.channels() == 1) {
        gray = frame_bgr;
    } else {
        cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.0);
    cv::Mat edges;
    cv::Canny(blurred, edges, 28.0, 84.0);

    const cv::Point2f normal(-axis.y, axis.x);
    std::vector<double> positive_slopes;
    std::vector<double> negative_slopes;
    const int max_along = static_cast<int>(std::round(std::min(42.0f, search_length)));
    for (int along = 5; along <= max_along; ++along) {
        int positive_edge = std::numeric_limits<int>::min();
        int negative_edge = std::numeric_limits<int>::max();
        for (int offset = -26; offset <= 26; ++offset) {
            const cv::Point p = toIntPoint(
                tip +
                axis * static_cast<float>(along) +
                normal * static_cast<float>(offset));
            if (p.x < 0 || p.y < 0 || p.x >= edges.cols || p.y >= edges.rows || edges.at<uchar>(p) == 0) {
                continue;
            }
            if (offset >= 2) {
                positive_edge = std::max(positive_edge, offset);
            } else if (offset <= -2) {
                negative_edge = std::min(negative_edge, offset);
            }
        }
        if (positive_edge != std::numeric_limits<int>::min()) {
            positive_slopes.push_back(static_cast<double>(positive_edge) / static_cast<double>(along));
        }
        if (negative_edge != std::numeric_limits<int>::max()) {
            negative_slopes.push_back(static_cast<double>(negative_edge) / static_cast<double>(along));
        }
    }

    if (positive_slopes.size() < 4 || negative_slopes.size() < 4) {
        return false;
    }

    const double pos = std::clamp(medianValue(positive_slopes), 0.08, 0.80);
    const double neg = std::clamp(medianValue(negative_slopes), -0.80, -0.08);
    if ((pos - neg) < 0.18 || (pos - neg) > 1.35) {
        return false;
    }
    *positive_slope = static_cast<float>(pos);
    *negative_slope = static_cast<float>(neg);
    return true;
}

void drawStylusTipWedge(
    cv::Mat& out,
    const cv::Mat& source_mask,
    const cv::Mat& frame_bgr,
    const cv::Point2f& adjusted_tip,
    const cv::Point2f& target,
    const cv::Point2f& fallback_direction,
    int tip_radius) {
    cv::Point2f axis = target - adjusted_tip;
    double axis_norm = cv::norm(axis);
    if (axis_norm < 1e-3 || !std::isfinite(axis_norm)) {
        axis = fallback_direction;
        axis_norm = cv::norm(axis);
    }
    if (axis_norm < 1e-3 || !std::isfinite(axis_norm)) {
        cv::circle(out, toIntPoint(adjusted_tip), tip_radius, cv::Scalar(255), cv::FILLED, cv::LINE_8);
        return;
    }
    axis *= static_cast<float>(1.0 / axis_norm);
    const cv::Point2f normal(-axis.y, axis.x);
    const float wedge_length = static_cast<float>(std::clamp(axis_norm + 14.0, 18.0, 46.0));
    float positive_width = static_cast<float>(std::clamp(wedge_length * 0.28f, 5.0f, 13.0f));
    float negative_width = positive_width;
    float connect_length = wedge_length;

    bool found_mask_cross_section = false;
    if (!source_mask.empty()) {
        for (float along = 5.0f; along <= wedge_length + 18.0f; along += 1.0f) {
            std::vector<int> hit_offsets;
            for (int offset = -30; offset <= 30; ++offset) {
                const cv::Point p = toIntPoint(adjusted_tip + axis * along + normal * static_cast<float>(offset));
                if (p.x >= 0 &&
                        p.y >= 0 &&
                        p.x < source_mask.cols &&
                        p.y < source_mask.rows &&
                        source_mask.at<uchar>(p) != 0) {
                    hit_offsets.push_back(offset);
                }
            }
            if (hit_offsets.empty()) {
                continue;
            }

            int best_start = hit_offsets.front();
            int best_end = hit_offsets.front();
            int run_start = hit_offsets.front();
            int prev = hit_offsets.front();
            auto consider_run = [&](int start, int end) {
                const int width = end - start + 1;
                const int best_width = best_end - best_start + 1;
                const double center = (static_cast<double>(start) + static_cast<double>(end)) * 0.5;
                const double best_center = (static_cast<double>(best_start) + static_cast<double>(best_end)) * 0.5;
                if (width > best_width || (width == best_width && std::abs(center) < std::abs(best_center))) {
                    best_start = start;
                    best_end = end;
                }
            };
            for (size_t i = 1; i < hit_offsets.size(); ++i) {
                if (hit_offsets[i] == prev + 1) {
                    prev = hit_offsets[i];
                    continue;
                }
                consider_run(run_start, prev);
                run_start = hit_offsets[i];
                prev = hit_offsets[i];
            }
            consider_run(run_start, prev);

            const int best_width = best_end - best_start + 1;
            const double best_center = (static_cast<double>(best_start) + static_cast<double>(best_end)) * 0.5;
            if (best_width >= 3 && std::abs(best_center) <= 18.0) {
                connect_length = along;
                positive_width = static_cast<float>(std::clamp(static_cast<double>(best_end), 3.0, 24.0));
                negative_width = static_cast<float>(std::clamp(static_cast<double>(-best_start), 3.0, 24.0));
                found_mask_cross_section = true;
                break;
            }
        }
    }
    if (!found_mask_cross_section) {
        refineStylusWedgeWidthsFromEdges(frame_bgr, adjusted_tip, axis, wedge_length, &positive_width, &negative_width);
    }

    float positive_slope = 0.0f;
    float negative_slope = 0.0f;
    if (fitStylusTipEdgeSlopesFromFrame(
            frame_bgr,
            adjusted_tip,
            axis,
            std::max(connect_length, wedge_length),
            &positive_slope,
            &negative_slope)) {
        const float ray_length = std::clamp(connect_length + 4.0f, 16.0f, 46.0f);
        std::vector<cv::Point> polygon = {
            toIntPoint(adjusted_tip),
            toIntPoint(adjusted_tip + axis * ray_length + normal * (positive_slope * ray_length)),
            toIntPoint(adjusted_tip + axis * ray_length + normal * (negative_slope * ray_length)),
        };
        cv::fillConvexPoly(out, polygon, cv::Scalar(255), cv::LINE_8);
        cv::circle(out, toIntPoint(adjusted_tip), tip_radius, cv::Scalar(255), cv::FILLED, cv::LINE_8);
        return;
    }

    for (float along = 0.0f; along <= connect_length; along += 1.0f) {
        const float t = connect_length > 1.0f ? along / connect_length : 1.0f;
        const float eased = t * t;
        const cv::Point2f center = adjusted_tip + axis * along;
        const cv::Point p0 = toIntPoint(center - normal * std::max(0.75f, negative_width * eased));
        const cv::Point p1 = toIntPoint(center + normal * std::max(0.75f, positive_width * eased));
        cv::line(out, p0, p1, cv::Scalar(255), 1, cv::LINE_8);
    }
    cv::circle(out, toIntPoint(adjusted_tip), tip_radius, cv::Scalar(255), cv::FILLED, cv::LINE_8);
}

cv::Mat makeStylusTipPriorMask(
    cv::Size size,
    const cv::Mat& surface_bounds_mask,
    const cv::Point2f& tip,
    cv::Point2f axis,
    std::vector<cv::Point>* prior_polygon = nullptr) {
    if (size.width <= 0 || size.height <= 0) {
        return cv::Mat();
    }
    double axis_norm = cv::norm(axis);
    if (axis_norm < 1e-3 || !std::isfinite(axis_norm)) {
        return cv::Mat();
    }
    axis *= static_cast<float>(1.0 / axis_norm);
    const cv::Point2f normal(-axis.y, axis.x);
    const float length = static_cast<float>(
        std::clamp(envInt("SHADOW_STYLUS_RGB_PRIOR_LENGTH", 52), 18, 100));
    const float backoff = static_cast<float>(
        std::clamp(envInt("SHADOW_STYLUS_RGB_PRIOR_BACKOFF", 6), 0, 24));
    const float near_half_width = static_cast<float>(
        std::clamp(envInt("SHADOW_STYLUS_RGB_PRIOR_NEAR_WIDTH", 5), 2, 16));
    const float far_half_width = static_cast<float>(
        std::clamp(envInt("SHADOW_STYLUS_RGB_PRIOR_FAR_WIDTH", 20), 6, 40));

    const cv::Point2f near_center = tip - axis * backoff;
    const cv::Point2f far_center = tip + axis * length;
    std::vector<cv::Point> polygon = {
        toIntPoint(near_center - normal * near_half_width),
        toIntPoint(near_center + normal * near_half_width),
        toIntPoint(far_center + normal * far_half_width),
        toIntPoint(far_center - normal * far_half_width),
    };
    cv::Mat prior(size, CV_8UC1, cv::Scalar(0));
    cv::fillConvexPoly(prior, polygon, cv::Scalar(255), cv::LINE_8);
    if (!surface_bounds_mask.empty() && surface_bounds_mask.size() == prior.size()) {
        cv::bitwise_and(prior, surface_bounds_mask, prior);
    }
    if (prior_polygon != nullptr) {
        *prior_polygon = std::move(polygon);
    }
    return prior;
}

cv::Mat makeStylusRgbCandidateMask(
    const cv::Mat& frame_bgr,
    const cv::Mat& prior_mask,
    cv::Mat* core_debug_mask = nullptr,
    cv::Mat* yellow_debug_mask = nullptr,
    cv::Mat* dark_debug_mask = nullptr) {
    if (frame_bgr.empty() || frame_bgr.channels() != 3 || prior_mask.empty() || prior_mask.size() != frame_bgr.size()) {
        return cv::Mat();
    }

    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat yellow_hsv;
    cv::inRange(
        hsv,
        cv::Scalar(
            std::clamp(envInt("SHADOW_STYLUS_RGB_YELLOW_HUE_LOW", 12), 0, 179),
            std::clamp(envInt("SHADOW_STYLUS_RGB_YELLOW_SAT_MIN", 42), 0, 255),
            std::clamp(envInt("SHADOW_STYLUS_RGB_YELLOW_VALUE_MIN", 70), 0, 255)),
        cv::Scalar(
            std::clamp(envInt("SHADOW_STYLUS_RGB_YELLOW_HUE_HIGH", 50), 0, 179),
            255,
            255),
        yellow_hsv);

    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat local_mean;
    cv::blur(gray, local_mean, cv::Size(15, 15));

    cv::Mat yellow_rgb(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat dark_rgb(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    const int dark_value_max = std::clamp(envInt("SHADOW_STYLUS_RGB_DARK_VALUE_MAX", 118), 30, 180);
    const int dark_contrast_min = std::clamp(envInt("SHADOW_STYLUS_RGB_DARK_CONTRAST_MIN", 18), 0, 80);
    cv::parallel_for_(cv::Range(0, frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* bgr_row = frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* hsv_row = hsv.ptr<cv::Vec3b>(y);
            const uchar* gray_row = gray.ptr<uchar>(y);
            const uchar* mean_row = local_mean.ptr<uchar>(y);
            uchar* yellow_row = yellow_rgb.ptr<uchar>(y);
            uchar* dark_row = dark_rgb.ptr<uchar>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                const int b = bgr_row[x][0];
                const int g = bgr_row[x][1];
                const int r = bgr_row[x][2];
                const int v = hsv_row[x][2];
                const int s = hsv_row[x][1];
                const int max_rg = std::max(r, g);
                const bool yellow_like =
                    g >= b + 14 &&
                    r >= b + 6 &&
                    max_rg >= 80 &&
                    std::abs(r - g) <= 100;
                if (yellow_like) {
                    yellow_row[x] = 255;
                }

                const bool dark_like =
                    v <= dark_value_max &&
                    gray_row[x] + dark_contrast_min <= mean_row[x] &&
                    !(s > 75 && b > r + 12 && b > g + 12) &&
                    !(s > 75 && g > r + 12 && g > b + 12);
                if (dark_like) {
                    dark_row[x] = 255;
                }
            }
        }
    });

    cv::bitwise_and(yellow_hsv, yellow_rgb, yellow_rgb);
    cv::bitwise_and(yellow_rgb, prior_mask, yellow_rgb);
    cv::bitwise_and(dark_rgb, prior_mask, dark_rgb);

    cv::Mat blue_mask = makeBlueMask(frame_bgr);
    cv::dilate(
        blue_mask,
        blue_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    cv::Mat non_blue;
    cv::bitwise_not(blue_mask, non_blue);
    cv::bitwise_and(yellow_rgb, non_blue, yellow_rgb);
    cv::bitwise_and(dark_rgb, non_blue, dark_rgb);

    cv::morphologyEx(yellow_rgb, yellow_rgb, cv::MORPH_OPEN, morphKernel3(), cv::Point(-1, -1), 1);
    cv::morphologyEx(dark_rgb, dark_rgb, cv::MORPH_OPEN, morphKernel3(), cv::Point(-1, -1), 1);

    cv::Mat core;
    cv::bitwise_or(yellow_rgb, dark_rgb, core);
    if (cv::countNonZero(core) == 0) {
        if (core_debug_mask != nullptr) {
            *core_debug_mask = core;
        }
        if (yellow_debug_mask != nullptr) {
            *yellow_debug_mask = yellow_rgb;
        }
        if (dark_debug_mask != nullptr) {
            *dark_debug_mask = dark_rgb;
        }
        return core;
    }

    cv::Mat highlight(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* bgr_row = frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* hsv_row = hsv.ptr<cv::Vec3b>(y);
            uchar* out_row = highlight.ptr<uchar>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                const int b = bgr_row[x][0];
                const int g = bgr_row[x][1];
                const int r = bgr_row[x][2];
                const int v = hsv_row[x][2];
                const int s = hsv_row[x][1];
                if (v >= 145 && s <= 95 && std::abs(r - g) <= 45 && std::abs(r - b) <= 55) {
                    out_row[x] = 255;
                }
            }
        }
    });
    cv::bitwise_and(highlight, prior_mask, highlight);
    cv::bitwise_and(highlight, non_blue, highlight);

    cv::Mat core_adjacent;
    const int core_dilate = std::clamp(envInt("SHADOW_STYLUS_RGB_CORE_DILATE", 3), 1, 9) | 1;
    cv::dilate(
        core,
        core_adjacent,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(core_dilate, core_dilate)),
        cv::Point(-1, -1),
        1);
    cv::bitwise_and(highlight, core_adjacent, highlight);

    cv::Mat candidate;
    cv::bitwise_or(core, highlight, candidate);
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE, morphKernel3(), cv::Point(-1, -1), 1);
    cv::bitwise_and(candidate, prior_mask, candidate);
    if (core_debug_mask != nullptr) {
        *core_debug_mask = core;
    }
    if (yellow_debug_mask != nullptr) {
        *yellow_debug_mask = yellow_rgb;
    }
    if (dark_debug_mask != nullptr) {
        *dark_debug_mask = dark_rgb;
    }
    return candidate;
}

bool selectStylusTipComponent(
    const cv::Mat& candidate_mask,
    const cv::Mat& core_mask,
    const cv::Mat& depth_difference_u8,
    const cv::Point2f& tip,
    cv::Point2f axis,
    cv::Mat* selected_mask,
    cv::Point2f* selected_centroid,
    int* selected_area,
    std::string* status) {
    if (selected_mask == nullptr || candidate_mask.empty()) {
        if (status != nullptr) {
            *status = "rgb_prior_no_candidate";
        }
        return false;
    }
    double axis_norm = cv::norm(axis);
    if (axis_norm < 1e-3 || !std::isfinite(axis_norm)) {
        if (status != nullptr) {
            *status = "rgb_prior_invalid_axis";
        }
        return false;
    }
    axis *= static_cast<float>(1.0 / axis_norm);
    const cv::Point2f normal(-axis.y, axis.x);

    cv::Mat binary;
    cv::threshold(candidate_mask, binary, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(binary) < std::clamp(envInt("SHADOW_STYLUS_RGB_MIN_PIXELS", 5), 1, 60)) {
        if (status != nullptr) {
            *status = "rgb_prior_no_candidate";
        }
        return false;
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    int best_label = -1;
    double best_score = -1e18;
    cv::Point2f best_centroid(0.0f, 0.0f);
    int best_area = 0;
    const int max_area = std::clamp(envInt("SHADOW_STYLUS_RGB_MAX_COMPONENT_AREA", 1800), 80, 3000);
    const double max_perp = static_cast<double>(std::clamp(envInt("SHADOW_STYLUS_RGB_MAX_AXIS_DISTANCE", 24), 6, 50));
    const double min_extent = static_cast<double>(std::clamp(envInt("SHADOW_STYLUS_RGB_MIN_AXIS_EXTENT", 7), 2, 35));
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < 4 || area > max_area) {
            continue;
        }
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (width > 70 || height > 70) {
            continue;
        }

        int seed_hits = 0;
        int core_hits = 0;
        double min_along = std::numeric_limits<double>::max();
        double max_along = -std::numeric_limits<double>::max();
        double max_abs_perp = 0.0;
        double sum_depth = 0.0;
        int depth_hits = 0;
        for (int y = top; y < top + height; ++y) {
            const int* label_row = labels.ptr<int>(y);
            const uchar* core_row = !core_mask.empty() && core_mask.size() == binary.size() ? core_mask.ptr<uchar>(y) : nullptr;
            const uchar* depth_row = !depth_difference_u8.empty() && depth_difference_u8.size() == binary.size()
                ? depth_difference_u8.ptr<uchar>(y)
                : nullptr;
            for (int x = left; x < left + width; ++x) {
                if (label_row[x] != label) {
                    continue;
                }
                const cv::Point2f delta(
                    static_cast<float>(x) - tip.x,
                    static_cast<float>(y) - tip.y);
                const double along = static_cast<double>(delta.dot(axis));
                const double perp = static_cast<double>(delta.dot(normal));
                min_along = std::min(min_along, along);
                max_along = std::max(max_along, along);
                max_abs_perp = std::max(max_abs_perp, std::abs(perp));
                if (along >= 1.0 && along <= 28.0 && std::abs(perp) <= 7.0) {
                    ++seed_hits;
                }
                if (core_row != nullptr && core_row[x] != 0) {
                    ++core_hits;
                }
                if (depth_row != nullptr) {
                    sum_depth += static_cast<double>(depth_row[x]);
                    ++depth_hits;
                }
            }
        }
        if (max_along < min_extent || min_along > 30.0 || max_abs_perp > max_perp) {
            continue;
        }
        if (seed_hits <= 0 && max_abs_perp > 10.0) {
            continue;
        }
        if (core_hits < std::max(3, area / 6)) {
            continue;
        }

        const double centroid_x = centroids.at<double>(label, 0);
        const double centroid_y = centroids.at<double>(label, 1);
        const cv::Point2f centroid(static_cast<float>(centroid_x), static_cast<float>(centroid_y));
        const cv::Point2f centroid_delta = centroid - tip;
        const double centroid_along = static_cast<double>(centroid_delta.dot(axis));
        const double centroid_perp = std::abs(static_cast<double>(centroid_delta.dot(normal)));
        const double mean_depth = depth_hits > 0 ? sum_depth / static_cast<double>(depth_hits) : 0.0;
        const double extent = max_along - min_along;
        const double compact_penalty = std::abs(static_cast<double>(width * height) - static_cast<double>(area)) * 0.015;
        const double score =
            (seed_hits * 8.0) +
            (core_hits * 1.5) +
            (std::min(area, 220) * 0.55) +
            (std::min(extent, 45.0) * 2.0) +
            (mean_depth * 0.08) -
            (std::abs(centroid_along - 18.0) * 1.1) -
            (centroid_perp * 4.5) -
            (max_abs_perp * 1.2) -
            compact_penalty;
        if (score > best_score) {
            best_score = score;
            best_label = label;
            best_centroid = centroid;
            best_area = area;
        }
    }

    if (best_label < 0) {
        if (status != nullptr) {
            *status = "rgb_prior_no_valid_component";
        }
        return false;
    }
    *selected_mask = labels == best_label;
    selected_mask->convertTo(*selected_mask, CV_8UC1);
    if (selected_centroid != nullptr) {
        *selected_centroid = best_centroid;
    }
    if (selected_area != nullptr) {
        *selected_area = best_area;
    }
    if (status != nullptr) {
        *status = "rgb_prior_success";
    }
    return true;
}

bool addStylusTipFromRgbPrior(
    cv::Mat& out,
    const cv::Mat& frame_bgr,
    const cv::Mat& depth_difference_u8,
    const cv::Mat& surface_bounds_mask,
    const cv::Point2f& tip,
    cv::Point2f axis,
    int tip_radius,
    int frame_index,
    StylusWedgeTemporalState* temporal_state,
    cv::Mat* debug_bgr = nullptr,
    cv::Mat* sobel_candidates_debug_bgr = nullptr,
    cv::Mat* sobel_raw_debug_bgr = nullptr,
    std::string* status = nullptr) {
    if (status != nullptr) {
        *status = "rgb_prior_started";
    }
    if (envInt("SHADOW_STYLUS_RGB_PRIOR_ENABLE", 1) == 0) {
        if (status != nullptr) {
            *status = "rgb_prior_disabled";
        }
        return false;
    }
    if (out.empty() || frame_bgr.empty() || frame_bgr.channels() != 3) {
        if (status != nullptr) {
            *status = "rgb_prior_frame_or_mask_missing";
        }
        return false;
    }
    double axis_norm = cv::norm(axis);
    if (axis_norm < 1e-3 || !std::isfinite(axis_norm)) {
        if (status != nullptr) {
            *status = "rgb_prior_invalid_axis";
        }
        return false;
    }
    axis *= static_cast<float>(1.0 / axis_norm);
    const cv::Point tip_i = toIntPoint(tip);
    std::vector<cv::Point> prior_polygon;
    cv::Mat prior = makeStylusTipPriorMask(out.size(), surface_bounds_mask, tip, axis, &prior_polygon);
    if (prior.empty() || cv::countNonZero(prior) < 12) {
        if (status != nullptr) {
            *status = "rgb_prior_empty";
        }
        return false;
    }

    auto ensureDebug = [&](cv::Mat* image) {
        if (image == nullptr || !image->empty()) {
            return;
        }
        *image = frame_bgr.clone();
    };
    ensureDebug(debug_bgr);
    ensureDebug(sobel_candidates_debug_bgr);
    ensureDebug(sobel_raw_debug_bgr);

    cv::Mat core_mask;
    cv::Mat yellow_mask;
    cv::Mat dark_mask;
    cv::Mat candidate = makeStylusRgbCandidateMask(frame_bgr, prior, &core_mask, &yellow_mask, &dark_mask);
    if (candidate.empty() || cv::countNonZero(candidate) < 4) {
        if (status != nullptr) {
            *status = "rgb_prior_no_candidate";
        }
        return false;
    }

    cv::Mat selected;
    cv::Point2f centroid;
    int area = 0;
    std::string select_status;
    if (!selectStylusTipComponent(
            candidate,
            core_mask,
            depth_difference_u8,
            tip,
            axis,
            &selected,
            &centroid,
            &area,
            &select_status)) {
        if (status != nullptr) {
            *status = select_status;
        }
        return false;
    }

    if (temporal_state != nullptr &&
            temporal_state->rgb_valid &&
            frame_index == temporal_state->rgb_frame_index + 1) {
        const cv::Point2f predicted_centroid = temporal_state->rgb_centroid + (tip - temporal_state->rgb_tip);
        const double max_jump = static_cast<double>(
            std::clamp(envInt("SHADOW_STYLUS_RGB_TEMPORAL_MAX_JUMP", 14), 4, 60));
        const double area_ratio = temporal_state->rgb_area > 0
            ? static_cast<double>(area) / static_cast<double>(temporal_state->rgb_area)
            : 1.0;
        if (cv::norm(centroid - predicted_centroid) > max_jump ||
                area_ratio < 0.35 ||
                area_ratio > 2.8) {
            temporal_state->rgb_valid = false;
        }
    }

    const int close_size = std::clamp(envInt("SHADOW_STYLUS_RGB_CLOSE_SIZE", 3), 1, 7) | 1;
    cv::morphologyEx(
        selected,
        selected,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close_size, close_size)),
        cv::Point(-1, -1),
        1);
    const int dilate_size = std::clamp(envInt("SHADOW_STYLUS_RGB_FINAL_DILATE", 3), 1, 7) | 1;
    cv::dilate(
        selected,
        selected,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_size, dilate_size)),
        cv::Point(-1, -1),
        1);
    cv::bitwise_and(selected, prior, selected);
    if (!surface_bounds_mask.empty() && surface_bounds_mask.size() == selected.size()) {
        cv::bitwise_and(selected, surface_bounds_mask, selected);
    }
    if (cv::countNonZero(selected) < 4) {
        if (status != nullptr) {
            *status = "rgb_prior_selected_empty_after_clip";
        }
        return false;
    }

    out.setTo(cv::Scalar(255), selected);
    if (temporal_state != nullptr) {
        temporal_state->rgb_valid = true;
        temporal_state->rgb_frame_index = frame_index;
        temporal_state->rgb_tip = tip;
        temporal_state->rgb_centroid = centroid;
        temporal_state->rgb_area = cv::countNonZero(selected);
    }

    auto drawRgbDebug = [&](cv::Mat* image, bool include_candidate) {
        if (image == nullptr || image->empty()) {
            return;
        }
        if (!prior_polygon.empty()) {
            cv::polylines(*image, std::vector<std::vector<cv::Point>>{prior_polygon}, true, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
        if (include_candidate) {
            cv::Mat magenta(image->size(), image->type(), cv::Scalar(255, 0, 255));
            magenta.copyTo(*image, candidate);
            if (!dark_mask.empty()) {
                cv::Mat blue(image->size(), image->type(), cv::Scalar(255, 0, 0));
                blue.copyTo(*image, dark_mask);
            }
            if (!yellow_mask.empty()) {
                cv::Mat yellow(image->size(), image->type(), cv::Scalar(0, 255, 255));
                yellow.copyTo(*image, yellow_mask);
            }
        }
        cv::line(*image, tip_i, toIntPoint(tip + axis * 34.0f), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        cv::circle(*image, tip_i, 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        cv::Mat red(image->size(), image->type(), cv::Scalar(0, 0, 255));
        red.copyTo(*image, selected);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(selected, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(*image, contours, -1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(
            *image,
            "rgb_prior_success",
            cv::Point(std::max(0, tip_i.x - 42), std::max(20, tip_i.y - 44)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 0, 255),
            1,
            cv::LINE_AA);
    };
    drawRgbDebug(debug_bgr, true);
    drawRgbDebug(sobel_candidates_debug_bgr, true);
    drawRgbDebug(sobel_raw_debug_bgr, false);

    if (status != nullptr) {
        *status = "rgb_prior_success";
    }
    return true;
}

bool addStylusTipFromAoiEdges(
    cv::Mat& out,
    const cv::Mat& frame_bgr,
    const cv::Mat& depth_difference_u8,
    const cv::Mat& surface_bounds_mask,
    const cv::Point2f& tip,
    cv::Point2f axis,
    int tip_radius,
    int frame_index,
    StylusWedgeTemporalState* temporal_state,
    cv::Mat* debug_bgr = nullptr,
    cv::Mat* sobel_candidates_debug_bgr = nullptr,
    cv::Mat* sobel_raw_debug_bgr = nullptr,
    std::string* status = nullptr) {
    if (status != nullptr) {
        *status = "aoi_started";
    }
    double axis_norm = cv::norm(axis);
    const int radius = std::clamp(envInt("SHADOW_STYLUS_AOI_RADIUS", 35), 20, 80);
    const cv::Point tip_i = toIntPoint(tip);
    auto putAoiStatus = [&](cv::Mat* image, const std::string& text) {
        if (status != nullptr) {
            *status = text;
        }
        if (image == nullptr || image->empty()) {
            return;
        }
        cv::putText(
            *image,
            text,
            cv::Point(std::max(0, tip_i.x - radius), std::max(20, tip_i.y - radius - 8)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 0, 255),
            1,
            cv::LINE_AA);
    };

    if (debug_bgr != nullptr) {
        if (!frame_bgr.empty() && frame_bgr.channels() == 1) {
            cv::cvtColor(frame_bgr, *debug_bgr, cv::COLOR_GRAY2BGR);
        } else if (!frame_bgr.empty()) {
            *debug_bgr = frame_bgr.clone();
        } else if (!out.empty()) {
            *debug_bgr = cv::Mat(out.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        }
        if (!debug_bgr->empty()) {
            cv::circle(*debug_bgr, tip_i, radius, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            cv::circle(*debug_bgr, tip_i, 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        }
    }
    if (sobel_candidates_debug_bgr != nullptr) {
        if (!frame_bgr.empty() && frame_bgr.channels() == 1) {
            cv::cvtColor(frame_bgr, *sobel_candidates_debug_bgr, cv::COLOR_GRAY2BGR);
        } else if (!frame_bgr.empty()) {
            *sobel_candidates_debug_bgr = frame_bgr.clone();
        } else if (!out.empty()) {
            *sobel_candidates_debug_bgr = cv::Mat(out.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        }
        if (!sobel_candidates_debug_bgr->empty()) {
            cv::circle(*sobel_candidates_debug_bgr, tip_i, radius, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            cv::circle(*sobel_candidates_debug_bgr, tip_i, 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        }
    }
    if (sobel_raw_debug_bgr != nullptr) {
        if (!frame_bgr.empty() && frame_bgr.channels() == 1) {
            cv::cvtColor(frame_bgr, *sobel_raw_debug_bgr, cv::COLOR_GRAY2BGR);
        } else if (!frame_bgr.empty()) {
            *sobel_raw_debug_bgr = frame_bgr.clone();
        } else if (!out.empty()) {
            *sobel_raw_debug_bgr = cv::Mat(out.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        }
        if (!sobel_raw_debug_bgr->empty()) {
            cv::circle(*sobel_raw_debug_bgr, tip_i, radius, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            cv::circle(*sobel_raw_debug_bgr, tip_i, 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        }
    }
    if (out.empty() || frame_bgr.empty()) {
        putAoiStatus(debug_bgr, "AOI: frame/mask missing");
        putAoiStatus(sobel_candidates_debug_bgr, "AOI: frame/mask missing");
        putAoiStatus(sobel_raw_debug_bgr, "AOI: frame/mask missing");
        return false;
    }
    if (depth_difference_u8.empty() || depth_difference_u8.size() != out.size()) {
        const std::string status = depth_difference_u8.empty()
            ? "AOI: depth difference missing"
            : "AOI: depth difference size mismatch";
        putAoiStatus(debug_bgr, status);
        putAoiStatus(sobel_candidates_debug_bgr, status);
        putAoiStatus(sobel_raw_debug_bgr, status);
        return false;
    }
    if (axis_norm < 1e-3 || !std::isfinite(axis_norm)) {
        putAoiStatus(debug_bgr, "AOI: invalid tilt axis");
        putAoiStatus(sobel_candidates_debug_bgr, "AOI: invalid tilt axis");
        putAoiStatus(sobel_raw_debug_bgr, "AOI: invalid tilt axis");
        return false;
    }
    axis *= static_cast<float>(1.0 / axis_norm);
    if (sobel_candidates_debug_bgr != nullptr && !sobel_candidates_debug_bgr->empty()) {
        cv::line(
            *sobel_candidates_debug_bgr,
            tip_i,
            toIntPoint(tip + axis * static_cast<float>(radius)),
            cv::Scalar(255, 0, 255),
            2,
            cv::LINE_AA);
    }
    const double alpha = static_cast<double>(std::clamp(envInt("SHADOW_STYLUS_AOI_DIFF_CONTRAST", 75), 1, 200)) / 35.0;
    const double beta = static_cast<double>(std::clamp(envInt("SHADOW_STYLUS_AOI_DIFF_BRIGHTNESS", 100), -255, 255));
    const int threshold = std::clamp(envInt("SHADOW_STYLUS_AOI_DIFF_THRESHOLD", 150), 0, 255);

    cv::Mat circle_mask(depth_difference_u8.size(), CV_8UC1, cv::Scalar(0));
    cv::circle(circle_mask, toIntPoint(tip), radius, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
    if (!surface_bounds_mask.empty() && surface_bounds_mask.size() == circle_mask.size()) {
        cv::bitwise_and(circle_mask, surface_bounds_mask, circle_mask);
        if (cv::countNonZero(circle_mask) == 0) {
            putAoiStatus(debug_bgr, "AOI: tip outside surface bounds");
            putAoiStatus(sobel_candidates_debug_bgr, "AOI: tip outside surface bounds");
            putAoiStatus(sobel_raw_debug_bgr, "AOI: tip outside surface bounds");
            return false;
        }
    }

    cv::Mat yellow_stylus_mask;
    cv::Mat yellow_edge_source;
    if (frame_bgr.channels() == 3 && envInt("SHADOW_STYLUS_AOI_YELLOW_TAPE", 1) != 0) {
        cv::Mat hsv;
        cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
        const int yellow_hue_low = std::clamp(envInt("SHADOW_STYLUS_AOI_YELLOW_HUE_LOW", 15), 0, 179);
        const int yellow_hue_high = std::clamp(envInt("SHADOW_STYLUS_AOI_YELLOW_HUE_HIGH", 45), 0, 179);
        const int yellow_sat_min = std::clamp(envInt("SHADOW_STYLUS_AOI_YELLOW_SAT_MIN", 45), 0, 255);
        const int yellow_value_min = std::clamp(envInt("SHADOW_STYLUS_AOI_YELLOW_VALUE_MIN", 80), 0, 255);
        cv::inRange(
            hsv,
            cv::Scalar(yellow_hue_low, yellow_sat_min, yellow_value_min),
            cv::Scalar(yellow_hue_high, 255, 255),
            yellow_stylus_mask);

        cv::Mat yellow_dominance(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
        cv::parallel_for_(cv::Range(0, frame_bgr.rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const cv::Vec3b* row = frame_bgr.ptr<cv::Vec3b>(y);
                uchar* out_row = yellow_dominance.ptr<uchar>(y);
                for (int x = 0; x < frame_bgr.cols; ++x) {
                    const int b = row[x][0];
                    const int g = row[x][1];
                    const int r = row[x][2];
                    const int max_rg = std::max(r, g);
                    const bool yellow_like =
                        g >= b + 18 &&
                        r >= b + 8 &&
                        max_rg >= 95 &&
                        std::abs(r - g) <= 90;
                    if (yellow_like) {
                        out_row[x] = 255;
                    }
                }
            }
        });
        cv::bitwise_and(yellow_stylus_mask, yellow_dominance, yellow_stylus_mask);
        cv::bitwise_and(yellow_stylus_mask, circle_mask, yellow_stylus_mask);
        cv::morphologyEx(
            yellow_stylus_mask,
            yellow_stylus_mask,
            cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)),
            cv::Point(-1, -1),
            1);
        cv::morphologyEx(
            yellow_stylus_mask,
            yellow_stylus_mask,
            cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)),
            cv::Point(-1, -1),
            1);

        cv::Mat lab;
        cv::cvtColor(frame_bgr, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        yellow_edge_source = lab_channels[2];
    }

    cv::Mat enhanced;
    depth_difference_u8.convertTo(enhanced, CV_8UC1, alpha, beta);
    cv::Mat bright;
    cv::threshold(enhanced, bright, threshold, 255, cv::THRESH_BINARY);
    cv::bitwise_and(bright, circle_mask, bright);
    cv::morphologyEx(
        bright,
        bright,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)),
        cv::Point(-1, -1),
        1);
    cv::morphologyEx(
        bright,
        bright,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    if (!yellow_stylus_mask.empty() && cv::countNonZero(yellow_stylus_mask) >= 12) {
        cv::bitwise_or(bright, yellow_stylus_mask, bright);
    }
    if (debug_bgr != nullptr && !debug_bgr->empty()) {
        cv::Mat magenta(debug_bgr->size(), debug_bgr->type(), cv::Scalar(255, 0, 255));
        magenta.copyTo(*debug_bgr, bright);
        if (!yellow_stylus_mask.empty()) {
            cv::Mat yellow_overlay(debug_bgr->size(), debug_bgr->type(), cv::Scalar(0, 255, 255));
            yellow_overlay.copyTo(*debug_bgr, yellow_stylus_mask);
        }
    }
    if (sobel_candidates_debug_bgr != nullptr && !sobel_candidates_debug_bgr->empty()) {
        cv::Mat magenta(sobel_candidates_debug_bgr->size(), sobel_candidates_debug_bgr->type(), cv::Scalar(255, 0, 255));
        magenta.copyTo(*sobel_candidates_debug_bgr, bright);
        if (!yellow_stylus_mask.empty()) {
            cv::Mat yellow_overlay(sobel_candidates_debug_bgr->size(), sobel_candidates_debug_bgr->type(), cv::Scalar(0, 255, 255));
            yellow_overlay.copyTo(*sobel_candidates_debug_bgr, yellow_stylus_mask);
        }
    }
    if (cv::countNonZero(bright) < 12) {
        putAoiStatus(debug_bgr, "AOI: no bright depth patch");
        putAoiStatus(sobel_candidates_debug_bgr, "AOI: no bright depth patch");
        putAoiStatus(sobel_raw_debug_bgr, "AOI: no bright depth patch");
        return false;
    }

    auto percentileFromMaskedU8 = [](const cv::Mat& image, const cv::Mat& mask, int percentile) {
        std::array<int, 256> hist = {};
        int count = 0;
        for (int y = 0; y < image.rows; ++y) {
            const uchar* image_row = image.ptr<uchar>(y);
            const uchar* mask_row = mask.ptr<uchar>(y);
            for (int x = 0; x < image.cols; ++x) {
                if (mask_row[x] == 0) {
                    continue;
                }
                ++hist[image_row[x]];
                ++count;
            }
        }
        if (count <= 0) {
            return 255;
        }
        const int target = std::max(1, (count * std::clamp(percentile, 0, 100) + 99) / 100);
        int running = 0;
        for (int value = 0; value < 256; ++value) {
            running += hist[static_cast<size_t>(value)];
            if (running >= target) {
                return value;
            }
        }
        return 255;
    };

    cv::Mat bright_neighborhood;
    const int bright_edge_dilate = std::clamp(envInt("SHADOW_STYLUS_AOI_BRIGHT_EDGE_DILATE", 23), 3, 61) | 1;
    cv::dilate(
        bright,
        bright_neighborhood,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(bright_edge_dilate, bright_edge_dilate)),
        cv::Point(-1, -1),
        1);

    std::vector<cv::Point> bright_points;
    cv::findNonZero(bright, bright_points);
    if (bright_points.size() < 12) {
        putAoiStatus(debug_bgr, "AOI: no bright depth points");
        putAoiStatus(sobel_candidates_debug_bgr, "AOI: no bright depth points");
        putAoiStatus(sobel_raw_debug_bgr, "AOI: no bright depth points");
        return false;
    }
    cv::Point2f bright_centroid(0.0f, 0.0f);
    for (const cv::Point& point : bright_points) {
        bright_centroid.x += static_cast<float>(point.x);
        bright_centroid.y += static_cast<float>(point.y);
    }
    bright_centroid *= 1.0f / static_cast<float>(bright_points.size());

    auto cross = [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x * b.y) - (a.y * b.x);
    };
    auto lineIntersection = [&](const cv::Point2f& p, const cv::Point2f& r,
                                const cv::Point2f& q, const cv::Point2f& s,
                                cv::Point2f* intersection) {
        const float denom = cross(r, s);
        if (std::abs(denom) < 1e-4f) {
            return false;
        }
        const float t = cross(q - p, s) / denom;
        *intersection = p + r * t;
        return std::isfinite(intersection->x) && std::isfinite(intersection->y);
    };

    struct CandidateLine {
        cv::Point2f p;
        cv::Point2f dir;
        double length = 0.0;
        double signed_angle_deg = 0.0;
    };
    cv::Mat gray;
    if (frame_bgr.channels() == 1) {
        gray = frame_bgr;
    } else {
        cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat raw_edge_roi_mask = circle_mask.clone();
    if (frame_bgr.channels() == 3 && envInt("SHADOW_STYLUS_AOI_SUPPRESS_BLUE_TAGS", 1) != 0) {
        cv::Mat blue_mask = makeBlueMask(frame_bgr);
        cv::dilate(
            blue_mask,
            blue_mask,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)),
            cv::Point(-1, -1),
            1);
        cv::Mat non_blue;
        cv::bitwise_not(blue_mask, non_blue);
        cv::bitwise_and(raw_edge_roi_mask, non_blue, raw_edge_roi_mask);
    }
    const int dark_edge_max = std::clamp(envInt("SHADOW_STYLUS_AOI_DARK_EDGE_MAX", 130), 0, 255);
    if (dark_edge_max > 0) {
        cv::Mat dark_mask;
        cv::threshold(gray, dark_mask, dark_edge_max, 255, cv::THRESH_BINARY_INV);
        cv::dilate(
            dark_mask,
            dark_mask,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)),
            cv::Point(-1, -1),
            1);
        cv::Mat allowed_edge_mask = dark_mask;
        if (!yellow_stylus_mask.empty() && cv::countNonZero(yellow_stylus_mask) >= 12) {
            cv::Mat yellow_edge_mask;
            cv::dilate(
                yellow_stylus_mask,
                yellow_edge_mask,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)),
                cv::Point(-1, -1),
                1);
            cv::bitwise_or(allowed_edge_mask, yellow_edge_mask, allowed_edge_mask);
        }
        cv::bitwise_and(raw_edge_roi_mask, allowed_edge_mask, raw_edge_roi_mask);
    }

    auto makeAoiEdges = [&](const cv::Mat& edge_gray,
                            const cv::Mat& edge_roi_mask,
                            bool depth_difference_pass) {
        cv::Mat sobel_x;
        cv::Mat sobel_y;
        cv::Sobel(edge_gray, sobel_x, CV_32F, 1, 0, 3);
        cv::Sobel(edge_gray, sobel_y, CV_32F, 0, 1, 3);
        cv::Mat sobel_mag;
        cv::magnitude(sobel_x, sobel_y, sobel_mag);
        cv::Mat sobel_u8;
        double min_sobel = 0.0;
        double max_sobel = 0.0;
        cv::minMaxLoc(sobel_mag, &min_sobel, &max_sobel, nullptr, nullptr, edge_roi_mask);
        if (max_sobel > min_sobel) {
            sobel_mag.convertTo(sobel_u8, CV_8UC1, 255.0 / (max_sobel - min_sobel), -min_sobel * 255.0 / (max_sobel - min_sobel));
        } else {
            sobel_u8 = cv::Mat::zeros(sobel_mag.size(), CV_8UC1);
        }
        cv::bitwise_and(sobel_u8, edge_roi_mask, sobel_u8);
        const int fixed_sobel_threshold = depth_difference_pass
            ? std::clamp(envInt("SHADOW_STYLUS_AOI_DEPTH_SOBEL_THRESHOLD", 70), 0, 255)
            : std::clamp(envInt("SHADOW_STYLUS_AOI_SOBEL_THRESHOLD", 92), 0, 255);
        const int percentile_sobel_threshold = percentileFromMaskedU8(
            sobel_u8,
            edge_roi_mask,
            depth_difference_pass
                ? std::clamp(envInt("SHADOW_STYLUS_AOI_DEPTH_SOBEL_PERCENTILE", 70), 0, 100)
                : std::clamp(envInt("SHADOW_STYLUS_AOI_SOBEL_PERCENTILE", 82), 0, 100));
        const int sobel_threshold = std::max(fixed_sobel_threshold, percentile_sobel_threshold);
        cv::Mat edges;
        cv::threshold(sobel_u8, edges, sobel_threshold, 255, cv::THRESH_BINARY);
        cv::bitwise_and(edges, edge_roi_mask, edges);
        cv::morphologyEx(
            edges,
            edges,
            cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
            cv::Point(-1, -1),
            1);
        cv::bitwise_and(edges, edge_roi_mask, edges);
        return edges;
    };

    cv::Mat raw_edges = makeAoiEdges(gray, raw_edge_roi_mask, false);
    if (!yellow_stylus_mask.empty() &&
            !yellow_edge_source.empty() &&
            cv::countNonZero(yellow_stylus_mask) >= 12) {
        cv::Mat yellow_edge_roi;
        cv::dilate(
            yellow_stylus_mask,
            yellow_edge_roi,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)),
            cv::Point(-1, -1),
            1);
        cv::bitwise_and(yellow_edge_roi, circle_mask, yellow_edge_roi);
        cv::bitwise_and(yellow_edge_roi, raw_edge_roi_mask, yellow_edge_roi);
        if (cv::countNonZero(yellow_edge_roi) >= 12) {
            cv::Mat yellow_edges = makeAoiEdges(yellow_edge_source, yellow_edge_roi, false);
            cv::bitwise_or(raw_edges, yellow_edges, raw_edges);
        }
    }
    if (sobel_raw_debug_bgr != nullptr && !sobel_raw_debug_bgr->empty()) {
        cv::Mat dimmed;
        sobel_raw_debug_bgr->convertTo(dimmed, -1, 0.35, 0.0);
        *sobel_raw_debug_bgr = dimmed;
        cv::Mat cyan_white(sobel_raw_debug_bgr->size(), sobel_raw_debug_bgr->type(), cv::Scalar(230, 255, 255));
        cyan_white.copyTo(*sobel_raw_debug_bgr, raw_edges);
        cv::circle(*sobel_raw_debug_bgr, tip_i, radius, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        cv::circle(*sobel_raw_debug_bgr, tip_i, 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
    }

    auto tryWedgeFromEdges = [&](const cv::Mat& edges, bool depth_difference_pass, std::string* failure) {
        cv::Mat relevant_edges;
        cv::bitwise_and(edges, bright_neighborhood, relevant_edges);
        cv::bitwise_and(relevant_edges, circle_mask, relevant_edges);
        if (debug_bgr != nullptr && !debug_bgr->empty()) {
            const cv::Scalar color = depth_difference_pass ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 255);
            cv::Mat overlay(debug_bgr->size(), debug_bgr->type(), color);
            overlay.copyTo(*debug_bgr, relevant_edges);
        }
        if (sobel_candidates_debug_bgr != nullptr && !sobel_candidates_debug_bgr->empty()) {
            const cv::Scalar color = depth_difference_pass ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 255);
            cv::Mat overlay(sobel_candidates_debug_bgr->size(), sobel_candidates_debug_bgr->type(), color);
            overlay.copyTo(*sobel_candidates_debug_bgr, relevant_edges);
        }

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(
            relevant_edges,
            lines,
            1.0,
            CV_PI / 180.0,
            depth_difference_pass
                ? std::clamp(envInt("SHADOW_STYLUS_AOI_DEPTH_HOUGH_THRESHOLD", 3), 1, 80)
                : std::clamp(envInt("SHADOW_STYLUS_AOI_HOUGH_THRESHOLD", 4), 1, 80),
            static_cast<double>(depth_difference_pass
                ? std::clamp(envInt("SHADOW_STYLUS_AOI_DEPTH_MIN_LINE", 4), 3, 60)
                : std::clamp(envInt("SHADOW_STYLUS_AOI_MIN_LINE", 5), 3, 60)),
            static_cast<double>(depth_difference_pass
                ? std::clamp(envInt("SHADOW_STYLUS_AOI_DEPTH_LINE_GAP", 14), 0, 30)
                : std::clamp(envInt("SHADOW_STYLUS_AOI_LINE_GAP", 12), 0, 30)));
        const cv::Scalar hough_color = depth_difference_pass ? cv::Scalar(255, 128, 0) : cv::Scalar(0, 165, 255);
        if (debug_bgr != nullptr && !debug_bgr->empty()) {
            for (const cv::Vec4i& line : lines) {
                cv::line(*debug_bgr, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), hough_color, 1, cv::LINE_AA);
            }
        }
        if (sobel_candidates_debug_bgr != nullptr && !sobel_candidates_debug_bgr->empty()) {
            for (const cv::Vec4i& line : lines) {
                cv::line(*sobel_candidates_debug_bgr, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), hough_color, 1, cv::LINE_AA);
            }
        }
        if (lines.size() < 2) {
            *failure = depth_difference_pass
                ? "AOI depth: fewer than 2 Hough lines"
                : "AOI: fewer than 2 Hough lines";
            return false;
        }

        std::vector<CandidateLine> candidates;
        const double min_tilt_angle = static_cast<double>(std::clamp(envInt("SHADOW_STYLUS_AOI_MIN_TILT_EDGE_DEG", 2), 0, 45));
        const double max_tilt_angle = static_cast<double>(std::clamp(envInt("SHADOW_STYLUS_AOI_MAX_TILT_EDGE_DEG", 82), 10, 89));
        for (const cv::Vec4i& line : lines) {
            cv::Point2f p0(static_cast<float>(line[0]), static_cast<float>(line[1]));
            cv::Point2f p1(static_cast<float>(line[2]), static_cast<float>(line[3]));
            cv::Point2f dir = p1 - p0;
            const double length = cv::norm(dir);
            if (length < 3.0) {
                continue;
            }
            dir *= static_cast<float>(1.0 / length);
            if (dir.dot(axis) < 0.0f) {
                dir *= -1.0f;
            }
            const double signed_angle = std::atan2(
                static_cast<double>(cross(axis, dir)),
                static_cast<double>(axis.dot(dir))) * 180.0 / CV_PI;
            if (std::abs(signed_angle) < min_tilt_angle || std::abs(signed_angle) > max_tilt_angle) {
                continue;
            }
            candidates.push_back({p0, dir, length, signed_angle});
            if (sobel_candidates_debug_bgr != nullptr && !sobel_candidates_debug_bgr->empty()) {
                const cv::Scalar color = signed_angle >= 0.0
                    ? cv::Scalar(0, 0, 255)
                    : cv::Scalar(255, 0, 0);
                cv::line(
                    *sobel_candidates_debug_bgr,
                    toIntPoint(p0),
                    toIntPoint(p0 + dir * static_cast<float>(std::max(12.0, length))),
                    color,
                    2,
                    cv::LINE_AA);
            }
        }
        if (candidates.size() < 2) {
            *failure = depth_difference_pass
                ? "AOI depth: fewer than 2 candidate lines"
                : "AOI: fewer than 2 candidate lines";
            return false;
        }

        double best_score = -1.0;
        cv::Point2f best_vertex;
        cv::Point2f best_d0;
        cv::Point2f best_d1;
        double best_length0 = 0.0;
        double best_length1 = 0.0;
        const double edge_support_px = static_cast<double>(
            std::clamp(envInt("SHADOW_STYLUS_AOI_EDGE_SUPPORT_PX", 6), 2, 20));
        const double edge_support_margin = static_cast<double>(
            std::clamp(envInt("SHADOW_STYLUS_AOI_EDGE_SUPPORT_MARGIN", 3), 0, 12));
        for (size_t i = 0; i < candidates.size(); ++i) {
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                if (candidates[i].signed_angle_deg * candidates[j].signed_angle_deg >= 0.0) {
                    continue;
                }
                const double angle_deg = std::abs(candidates[i].signed_angle_deg - candidates[j].signed_angle_deg);
                if (angle_deg < 8.0 || angle_deg > 95.0) {
                    continue;
                }

                cv::Point2f vertex;
                if (!lineIntersection(candidates[i].p, candidates[i].dir, candidates[j].p, candidates[j].dir, &vertex)) {
                    continue;
                }
                const double h5_distance = cv::norm(vertex - tip);
                if (h5_distance > static_cast<double>(radius) * 1.10 ||
                        vertex.x < 0.0f ||
                        vertex.y < 0.0f ||
                        vertex.x >= static_cast<float>(out.cols) ||
                        vertex.y >= static_cast<float>(out.rows)) {
                    continue;
                }

                cv::Point2f d0 = candidates[i].dir;
                cv::Point2f d1 = candidates[j].dir;
                const cv::Point2f to_bright = bright_centroid - vertex;
                if (d0.dot(to_bright) < 0.0f) {
                    d0 *= -1.0f;
                }
                if (d1.dot(to_bright) < 0.0f) {
                    d1 *= -1.0f;
                }
                if (std::abs(cross(d0, d1)) < 0.15f) {
                    continue;
                }

                int inside_count = 0;
                double max_distance = 0.0;
                double max_edge0_along = 0.0;
                double max_edge1_along = 0.0;
                const float orientation = cross(d0, d1);
                for (const cv::Point& point : bright_points) {
                    cv::Point2f rel(
                        static_cast<float>(point.x) - vertex.x,
                        static_cast<float>(point.y) - vertex.y);
                    const double dist = cv::norm(rel);
                    if (dist > static_cast<double>(radius) * 1.25 || d0.dot(rel) < -2.0f || d1.dot(rel) < -2.0f) {
                        continue;
                    }
                    const float c0 = cross(d0, rel);
                    const float c1 = cross(rel, d1);
                    const bool inside = orientation > 0.0f
                        ? (c0 >= -2.0f && c1 >= -2.0f)
                        : (c0 <= 2.0f && c1 <= 2.0f);
                    if (inside) {
                        ++inside_count;
                        max_distance = std::max(max_distance, dist);
                        const double along0 = static_cast<double>(d0.dot(rel));
                        const double along1 = static_cast<double>(d1.dot(rel));
                        const double edge0_distance = std::abs(static_cast<double>(cross(d0, rel)));
                        const double edge1_distance = std::abs(static_cast<double>(cross(d1, rel)));
                        if (along0 > 0.0 && edge0_distance <= edge_support_px) {
                            max_edge0_along = std::max(max_edge0_along, along0);
                        }
                        if (along1 > 0.0 && edge1_distance <= edge_support_px) {
                            max_edge1_along = std::max(max_edge1_along, along1);
                        }
                    }
                }
                const double score =
                    static_cast<double>(inside_count) +
                    (0.35 * (candidates[i].length + candidates[j].length)) -
                    (0.30 * h5_distance) -
                    (0.20 * std::abs(std::abs(candidates[i].signed_angle_deg) - std::abs(candidates[j].signed_angle_deg))) -
                    (0.06 * (std::abs(candidates[i].signed_angle_deg) + std::abs(candidates[j].signed_angle_deg)));
                if (score > best_score) {
                    best_score = score;
                    best_vertex = vertex;
                    best_d0 = d0;
                    best_d1 = d1;
                    const double centroid_distance = cv::norm(bright_centroid - vertex);
                    const double fallback_length = std::clamp(
                        std::max({max_distance + 4.0, centroid_distance + 6.0, 14.0}),
                        12.0,
                        static_cast<double>(radius));
                    best_length0 = std::clamp(
                        (max_edge0_along > 0.0 ? max_edge0_along + edge_support_margin : fallback_length * 0.75),
                        8.0,
                        fallback_length);
                    best_length1 = std::clamp(
                        (max_edge1_along > 0.0 ? max_edge1_along + edge_support_margin : fallback_length * 0.75),
                        8.0,
                        fallback_length);
                }
            }
        }
        if (best_score <= 0.0) {
            *failure = depth_difference_pass
                ? "AOI depth: no intersecting line pair"
                : "AOI: no intersecting line pair";
            return false;
        }

        const double wedge_length_scale = static_cast<double>(
            std::clamp(envInt("SHADOW_STYLUS_AOI_WEDGE_LENGTH_PERCENT", 100), 50, 120)) / 100.0;
        float ray_length0 = static_cast<float>(std::clamp(
            best_length0 * wedge_length_scale,
            8.0,
            static_cast<double>(radius)));
        float ray_length1 = static_cast<float>(std::clamp(
            best_length1 * wedge_length_scale,
            8.0,
            static_cast<double>(radius)));
        const double outlier_trim_scale = static_cast<double>(
            std::clamp(envInt("SHADOW_STYLUS_AOI_OUTLIER_SIDE_PERCENT", 72), 50, 100)) / 100.0;
        cv::Point2f endpoint0 = best_vertex + best_d0 * ray_length0;
        cv::Point2f endpoint1 = best_vertex + best_d1 * ray_length1;
        const double endpoint0_centroid_distance = cv::norm(endpoint0 - bright_centroid);
        const double endpoint1_centroid_distance = cv::norm(endpoint1 - bright_centroid);
        if (endpoint0_centroid_distance > endpoint1_centroid_distance * 1.12) {
            ray_length0 = static_cast<float>(std::max(8.0, static_cast<double>(ray_length0) * outlier_trim_scale));
        } else if (endpoint1_centroid_distance > endpoint0_centroid_distance * 1.12) {
            ray_length1 = static_cast<float>(std::max(8.0, static_cast<double>(ray_length1) * outlier_trim_scale));
        }
        endpoint0 = best_vertex + best_d0 * ray_length0;
        endpoint1 = best_vertex + best_d1 * ray_length1;
        cv::Point2f smoothed_vertex = best_vertex;
        cv::Point2f smoothed_endpoint0 = endpoint0;
        cv::Point2f smoothed_endpoint1 = endpoint1;
        const int temporal_percent = std::clamp(envInt("SHADOW_STYLUS_AOI_TEMPORAL_SMOOTH_PERCENT", 35), 0, 85);
        if (temporal_state != nullptr &&
                temporal_state->valid &&
                temporal_percent > 0 &&
                frame_index == temporal_state->frame_index + 1) {
            const cv::Point2f tip_delta = tip - temporal_state->tip;
            const cv::Point2f predicted_vertex = temporal_state->vertex + tip_delta;
            const cv::Point2f predicted_endpoint0 = temporal_state->endpoint0 + tip_delta;
            const cv::Point2f predicted_endpoint1 = temporal_state->endpoint1 + tip_delta;
            const double max_jump = static_cast<double>(
                std::clamp(envInt("SHADOW_STYLUS_AOI_TEMPORAL_MAX_JUMP", 18), 4, 80));
            const bool close_enough =
                cv::norm(smoothed_vertex - predicted_vertex) <= max_jump &&
                cv::norm(smoothed_endpoint0 - predicted_endpoint0) <= max_jump * 1.8 &&
                cv::norm(smoothed_endpoint1 - predicted_endpoint1) <= max_jump * 1.8;
            if (close_enough) {
                const float previous_weight = static_cast<float>(temporal_percent) / 100.0f;
                const float current_weight = 1.0f - previous_weight;
                smoothed_vertex = (smoothed_vertex * current_weight) + (predicted_vertex * previous_weight);
                smoothed_endpoint0 = (smoothed_endpoint0 * current_weight) + (predicted_endpoint0 * previous_weight);
                smoothed_endpoint1 = (smoothed_endpoint1 * current_weight) + (predicted_endpoint1 * previous_weight);
            }
        }
        std::vector<cv::Point> polygon = {
            toIntPoint(smoothed_vertex),
            toIntPoint(smoothed_endpoint0),
            toIntPoint(smoothed_endpoint1),
        };
        cv::Mat wedge_mask(out.size(), CV_8UC1, cv::Scalar(0));
        cv::fillConvexPoly(wedge_mask, polygon, cv::Scalar(255), cv::LINE_AA);
        if (!surface_bounds_mask.empty() && surface_bounds_mask.size() == wedge_mask.size()) {
            cv::bitwise_and(wedge_mask, surface_bounds_mask, wedge_mask);
            if (cv::countNonZero(wedge_mask) < 8) {
                if (failure != nullptr) {
                    *failure = "AOI: wedge outside surface bounds";
                }
                return false;
            }
        }
        const int wedge_support_dilate = std::clamp(envInt("SHADOW_STYLUS_AOI_WEDGE_SUPPORT_DILATE", 9), 1, 41) | 1;
        cv::Mat wedge_support;
        cv::dilate(
            bright,
            wedge_support,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(wedge_support_dilate, wedge_support_dilate)),
            cv::Point(-1, -1),
            1);
        if (envInt("SHADOW_STYLUS_AOI_CLIP_WEDGE_TO_BRIGHT", 0) != 0) {
            cv::Mat clipped_wedge;
            cv::bitwise_and(wedge_mask, wedge_support, clipped_wedge);
            const float vertex_keep_length = static_cast<float>(
                std::clamp(envInt("SHADOW_STYLUS_AOI_VERTEX_KEEP_LEN", 12), 0, 24));
            if (vertex_keep_length > 0.0f) {
                std::vector<cv::Point> vertex_keep_polygon = {
                    toIntPoint(smoothed_vertex),
                    toIntPoint(smoothed_vertex + (smoothed_endpoint0 - smoothed_vertex) * std::min(1.0f, vertex_keep_length / std::max(1.0f, static_cast<float>(cv::norm(smoothed_endpoint0 - smoothed_vertex))))),
                    toIntPoint(smoothed_vertex + (smoothed_endpoint1 - smoothed_vertex) * std::min(1.0f, vertex_keep_length / std::max(1.0f, static_cast<float>(cv::norm(smoothed_endpoint1 - smoothed_vertex))))),
                };
                cv::Mat vertex_keep_mask(out.size(), CV_8UC1, cv::Scalar(0));
                cv::fillConvexPoly(vertex_keep_mask, vertex_keep_polygon, cv::Scalar(255), cv::LINE_AA);
                cv::bitwise_or(clipped_wedge, vertex_keep_mask, clipped_wedge);
                cv::bitwise_and(clipped_wedge, wedge_mask, clipped_wedge);
            }
            if (cv::countNonZero(clipped_wedge) >= 8) {
                wedge_mask = clipped_wedge;
            }
        }
        out.setTo(cv::Scalar(255), wedge_mask);
        if (debug_bgr != nullptr && !debug_bgr->empty()) {
            cv::Mat red(debug_bgr->size(), debug_bgr->type(), cv::Scalar(0, 0, 255));
            red.copyTo(*debug_bgr, wedge_mask);
            std::vector<std::vector<cv::Point>> wedge_contours;
            cv::findContours(wedge_mask, wedge_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(*debug_bgr, wedge_contours, -1, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            const cv::Point vertex_i = toIntPoint(smoothed_vertex);
            cv::drawMarker(
                *debug_bgr,
                vertex_i,
                cv::Scalar(255, 0, 0),
                cv::MARKER_CROSS,
                9,
                1,
                cv::LINE_AA);
        }
        if (status != nullptr) {
            *status = depth_difference_pass ? "aoi_wedge_success_depth_difference" : "aoi_wedge_success_raw_frame";
        }
        if (temporal_state != nullptr) {
            temporal_state->valid = true;
            temporal_state->frame_index = frame_index;
            temporal_state->tip = tip;
            temporal_state->vertex = smoothed_vertex;
            temporal_state->endpoint0 = smoothed_endpoint0;
            temporal_state->endpoint1 = smoothed_endpoint1;
        }
        return true;
    };

    std::string raw_failure;
    if (tryWedgeFromEdges(raw_edges, false, &raw_failure)) {
        return true;
    }
    std::string depth_failure;
    if (envInt("SHADOW_STYLUS_AOI_DEPTH_EDGE_FALLBACK", 1) != 0) {
        cv::Mat depth_edges = makeAoiEdges(enhanced, circle_mask, true);
        if (sobel_raw_debug_bgr != nullptr && !sobel_raw_debug_bgr->empty()) {
            cv::Mat depth_debug_edges;
            cv::bitwise_and(depth_edges, bright_neighborhood, depth_debug_edges);
            cv::Mat orange(sobel_raw_debug_bgr->size(), sobel_raw_debug_bgr->type(), cv::Scalar(0, 180, 255));
            orange.copyTo(*sobel_raw_debug_bgr, depth_debug_edges);
        }
        if (tryWedgeFromEdges(depth_edges, true, &depth_failure)) {
            return true;
        }
    } else {
        depth_failure = "AOI depth: disabled";
    }

    const std::string final_failure = raw_failure + "; " + depth_failure;
    putAoiStatus(debug_bgr, final_failure);
    putAoiStatus(sobel_candidates_debug_bgr, final_failure);
    putAoiStatus(sobel_raw_debug_bgr, final_failure);
    return false;
}

cv::Mat makeSurfaceBoundsMask(const GeometryResult& geometry, cv::Size frame_size) {
    if (!geometry.ok || geometry.board_mask.empty()) {
        return cv::Mat();
    }
    cv::Mat surface_mask;
    if (geometry.board_mask.size() == frame_size) {
        surface_mask = geometry.board_mask.clone();
    } else {
        cv::resize(geometry.board_mask, surface_mask, frame_size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::threshold(surface_mask, surface_mask, 0, 255, cv::THRESH_BINARY);
    return surface_mask;
}

cv::Mat clipMaskToSurfaceBounds(const cv::Mat& mask_u8, const cv::Mat& surface_bounds_mask) {
    if (mask_u8.empty() || surface_bounds_mask.empty() || surface_bounds_mask.size() != mask_u8.size()) {
        return mask_u8;
    }
    cv::Mat binary;
    cv::threshold(mask_u8, binary, 0, 255, cv::THRESH_BINARY);
    cv::bitwise_and(binary, surface_bounds_mask, binary);
    return binary;
}

cv::Mat selectBestSurfaceDepthComponent(
    const cv::Mat& candidates_u8,
    const cv::Mat& depth_difference_u8,
    const cv::Point2f& tip,
    bool have_tip) {
    cv::Mat binary;
    cv::threshold(candidates_u8, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    if (component_count <= 1) {
        return cv::Mat::zeros(binary.size(), CV_8UC1);
    }

    int best_label = -1;
    double best_score = std::numeric_limits<double>::max();
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < 4) {
            continue;
        }
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const double centroid_x = centroids.at<double>(label, 0);
        const double centroid_y = centroids.at<double>(label, 1);

        double mean_value = 0.0;
        if (!depth_difference_u8.empty() && depth_difference_u8.size() == binary.size()) {
            cv::Mat component_mask = labels == label;
            mean_value = cv::mean(depth_difference_u8, component_mask)[0];
        }
        const double intensity_score = mean_value + (std::min(area, 20000) * 0.001);

        double score = -intensity_score;
        if (have_tip) {
            const double right = static_cast<double>(left + width - 1);
            const double bottom = static_cast<double>(top + height - 1);
            const double dx = std::max({static_cast<double>(left) - static_cast<double>(tip.x), 0.0, static_cast<double>(tip.x) - right});
            const double dy = std::max({static_cast<double>(top) - static_cast<double>(tip.y), 0.0, static_cast<double>(tip.y) - bottom});
            const double box_distance = std::hypot(dx, dy);
            const double centroid_distance = std::hypot(
                static_cast<double>(tip.x) - centroid_x,
                static_cast<double>(tip.y) - centroid_y);
            score = box_distance + (0.05 * centroid_distance) - (intensity_score * 0.15);
        }

        if (score < best_score) {
            best_score = score;
            best_label = label;
        }
    }

    if (best_label < 0) {
        return cv::Mat::zeros(binary.size(), CV_8UC1);
    }
    cv::Mat selected = labels == best_label;
    selected.convertTo(selected, CV_8UC1);
    return selected;
}

struct SheetGuidedDepthThresholdState {
    bool valid = false;
    int frame_index = -1;
    int threshold = 0;
};

struct SheetGuidedDepthThresholdResult {
    cv::Mat candidate;
    int base_threshold = 0;
    int selected_threshold = 0;
    int candidate_pixels = 0;
    int removed_sheet_pixels = 0;
    double mismatch_percent = 0.0;
    double retention_percent = 100.0;
    double visible_sheet_coverage_percent = 0.0;
    bool used_sheet_guidance = false;
    std::string status = "not_run";
};

SheetGuidedDepthThresholdResult selectSheetGuidedDepthCandidate(
    const cv::Mat& fallback_candidate_u8,
    const cv::Mat& depth_difference_u8,
    const cv::Mat& observed_bgr,
    int frame_index,
    const H5CornerTrace* h5_corners,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    const Options& options,
    SheetGuidedDepthThresholdState* temporal_state) {
    SheetGuidedDepthThresholdResult result;
    result.base_threshold = std::clamp(options.tilt_gradient_mask_threshold, 0, 255);
    result.selected_threshold = result.base_threshold;
    auto invalidateTemporalState = [&]() {
        if (temporal_state != nullptr) {
            temporal_state->valid = false;
        }
    };

    if (fallback_candidate_u8.empty()) {
        result.status = "empty_fallback_candidate";
        invalidateTemporalState();
        return result;
    }
    cv::threshold(fallback_candidate_u8, result.candidate, 0, 255, cv::THRESH_BINARY);
    result.candidate_pixels = cv::countNonZero(result.candidate);
    if (!options.sheet_guided_depth_threshold) {
        result.status = "disabled";
        invalidateTemporalState();
        return result;
    }
    if (depth_difference_u8.empty() ||
            depth_difference_u8.type() != CV_8UC1 ||
            depth_difference_u8.size() != result.candidate.size() ||
            observed_bgr.empty() ||
            observed_bgr.type() != CV_8UC3 ||
            observed_bgr.size() != result.candidate.size()) {
        result.status = "missing_or_mismatched_input";
        invalidateTemporalState();
        return result;
    }

    const cv::Mat surface_bounds = makeSurfaceBoundsMask(geometry, result.candidate.size());
    if (surface_bounds.empty()) {
        result.status = "surface_bounds_missing";
        invalidateTemporalState();
        return result;
    }
    cv::bitwise_and(result.candidate, surface_bounds, result.candidate);
    const int fallback_pixels = cv::countNonZero(result.candidate);
    result.candidate_pixels = fallback_pixels;
    if (fallback_pixels < 4) {
        result.status = "fallback_candidate_too_small";
        invalidateTemporalState();
        return result;
    }
    if (envInt("SHADOW_HAND_MASK_EXCLUDE_VISIBLE_SHEET", 1) == 0) {
        result.status = "visible_sheet_exclusion_disabled";
        invalidateTemporalState();
        return result;
    }

    cv::Mat visible_sheet = makeVisibleBoardColorProtectionMask(observed_bgr);
    cv::bitwise_and(visible_sheet, surface_bounds, visible_sheet);
    const int surface_pixels = cv::countNonZero(surface_bounds);
    const int visible_sheet_pixels = cv::countNonZero(visible_sheet);
    result.visible_sheet_coverage_percent = surface_pixels > 0
        ? 100.0 * static_cast<double>(visible_sheet_pixels) / static_cast<double>(surface_pixels)
        : 0.0;
    const int minimum_visible_sheet_pixels = std::max(
        500,
        static_cast<int>(std::ceil(static_cast<double>(surface_pixels) * 0.05)));
    if (visible_sheet_pixels < minimum_visible_sheet_pixels) {
        result.status = "insufficient_visible_sheet_evidence";
        invalidateTemporalState();
        return result;
    }

    cv::Point stylus_roi_center(-1, -1);
    cv::Point2f stylus_tip;
    if (h5StylusPointOnV3Surface(
            h5_corners,
            frame_index,
            reference,
            geometry,
            result.candidate.size(),
            &stylus_tip,
            false)) {
        stylus_roi_center = cv::Point(cvRound(stylus_tip.x), cvRound(stylus_tip.y));
    }

    cv::Mat scoring_sheet = visible_sheet.clone();
    if (envInt("SHADOW_HAND_MASK_EXCLUDE_VISIBLE_SHEET_KEEP_STYLUS_ROI", 1) != 0 &&
            stylus_roi_center.x >= 0 && stylus_roi_center.y >= 0 &&
            stylus_roi_center.x < scoring_sheet.cols && stylus_roi_center.y < scoring_sheet.rows) {
        const int radius = std::clamp(envInt("SHADOW_STYLUS_AOI_RADIUS", 35), 20, 80);
        cv::circle(scoring_sheet, stylus_roi_center, radius, cv::Scalar(0), cv::FILLED, cv::LINE_AA);
    }
    int sheet_dilate_size = std::clamp(
        envInt("SHADOW_HAND_MASK_EXCLUDE_VISIBLE_SHEET_DILATE", 3),
        1,
        15);
    if (sheet_dilate_size % 2 == 0) {
        ++sheet_dilate_size;
    }
    const cv::Mat sheet_dilate_kernel = sheet_dilate_size > 1
        ? cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(sheet_dilate_size, sheet_dilate_size))
        : cv::Mat();
    auto excludeUsingScoringSheet = [&](const cv::Mat& candidate) {
        cv::Mat refined;
        cv::threshold(candidate, refined, 0, 255, cv::THRESH_BINARY);
        cv::Mat candidate_sheet;
        cv::bitwise_and(scoring_sheet, refined, candidate_sheet);
        if (cv::countNonZero(candidate_sheet) == 0) {
            return refined;
        }
        if (!sheet_dilate_kernel.empty()) {
            cv::dilate(candidate_sheet, candidate_sheet, sheet_dilate_kernel);
            cv::bitwise_and(candidate_sheet, refined, candidate_sheet);
        }
        refined.setTo(0, candidate_sheet);
        return refined;
    };

    const cv::Mat cleaned_baseline = excludeUsingScoringSheet(result.candidate);
    const int cleaned_baseline_pixels = cv::countNonZero(cleaned_baseline);
    if (cleaned_baseline_pixels < 4) {
        result.status = "cleaned_baseline_too_small";
        invalidateTemporalState();
        return result;
    }

    struct CandidateScore {
        cv::Mat candidate;
        int threshold = 0;
        int candidate_pixels = 0;
        int removed_sheet_pixels = 0;
        double mismatch_percent = 100.0;
        double retention_percent = 0.0;
    };
    auto scoreThreshold = [&](int threshold) {
        CandidateScore score;
        score.threshold = threshold;
        cv::threshold(
            depth_difference_u8,
            score.candidate,
            std::clamp(threshold, 0, 255),
            255,
            cv::THRESH_BINARY);
        // The Python depth stage has already selected the hand-associated
        // component. Raising the threshold may split that component, so keep
        // every stronger pixel inside the original selection rather than
        // arbitrarily dropping all but one new fragment.
        cv::bitwise_and(score.candidate, result.candidate, score.candidate);
        cv::bitwise_and(score.candidate, surface_bounds, score.candidate);
        score.candidate_pixels = cv::countNonZero(score.candidate);
        if (score.candidate_pixels <= 0) {
            return score;
        }
        cv::Mat cleaned = excludeUsingScoringSheet(score.candidate);
        const int cleaned_pixels = cv::countNonZero(cleaned);
        score.removed_sheet_pixels = std::max(0, score.candidate_pixels - cleaned_pixels);
        score.mismatch_percent = 100.0 * static_cast<double>(score.removed_sheet_pixels) /
            static_cast<double>(score.candidate_pixels);
        cv::Mat retained_baseline;
        cv::bitwise_and(score.candidate, cleaned_baseline, retained_baseline);
        score.retention_percent = 100.0 * static_cast<double>(cv::countNonZero(retained_baseline)) /
            static_cast<double>(cleaned_baseline_pixels);
        return score;
    };

    const int max_threshold = std::clamp(
        std::max(result.base_threshold, options.sheet_guided_max_threshold),
        result.base_threshold,
        255);
    const double maximum_mismatch = std::clamp(options.sheet_guided_mismatch_percent, 0.0, 100.0);
    const double minimum_retention = std::clamp(options.sheet_guided_min_retention_percent, 0.0, 100.0);
    CandidateScore pending;
    bool have_pending = false;
    CandidateScore selected;
    bool have_selected = false;
    for (int threshold = result.base_threshold; threshold <= max_threshold; ++threshold) {
        CandidateScore score = scoreThreshold(threshold);
        const bool passes = score.candidate_pixels >= 4 &&
            score.mismatch_percent <= maximum_mismatch &&
            score.retention_percent >= minimum_retention;
        if (!passes) {
            have_pending = false;
            continue;
        }
        if (threshold == result.base_threshold) {
            selected = std::move(score);
            have_selected = true;
            break;
        }
        if (have_pending && pending.threshold + 1 == threshold) {
            selected = std::move(pending);
            have_selected = true;
            break;
        }
        pending = std::move(score);
        have_pending = true;
    }
    if (!have_selected) {
        result.status = "no_stable_threshold_met_constraints";
        invalidateTemporalState();
        return result;
    }

    // A lower threshold expands the mask. In sequential video, limit that
    // expansion to two threshold levels per frame; threshold increases remain
    // immediate so visible-sheet spill is never knowingly retained.
    if (temporal_state != nullptr && temporal_state->valid &&
            temporal_state->frame_index + 1 == frame_index &&
            selected.threshold < temporal_state->threshold - 2) {
        CandidateScore stabilized = scoreThreshold(temporal_state->threshold - 2);
        if (stabilized.candidate_pixels >= 4 && stabilized.retention_percent >= minimum_retention) {
            selected = std::move(stabilized);
        }
    }

    result.candidate = selected.candidate;
    result.selected_threshold = selected.threshold;
    result.candidate_pixels = selected.candidate_pixels;
    result.removed_sheet_pixels = selected.removed_sheet_pixels;
    result.mismatch_percent = selected.mismatch_percent;
    result.retention_percent = selected.retention_percent;
    result.used_sheet_guidance = true;
    result.status = selected.threshold == result.base_threshold
        ? "base_threshold_already_clean"
        : "raised_to_lowest_stable_clean_threshold";
    if (temporal_state != nullptr) {
        temporal_state->valid = true;
        temporal_state->frame_index = frame_index;
        temporal_state->threshold = selected.threshold;
    }
    return result;
}

cv::Mat connectV3StylusTipToMask(
    const cv::Mat& mask_u8,
    const cv::Mat& depth_difference_u8,
    const cv::Mat& frame_bgr,
    int frame_index,
    const H5CornerTrace* h5_corners,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    cv::Size frame_size,
    cv::Point* debug_wedge_target = nullptr,
    cv::Mat* debug_wedge_bgr = nullptr,
    cv::Mat* sobel_candidates_debug_bgr = nullptr,
    cv::Mat* sobel_raw_debug_bgr = nullptr,
    std::string* stylus_wedge_status = nullptr,
    StylusWedgeTemporalState* temporal_state = nullptr) {
    if (stylus_wedge_status != nullptr) {
        *stylus_wedge_status = "not_run";
    }
    if (debug_wedge_target != nullptr) {
        *debug_wedge_target = cv::Point(-1, -1);
    }
    const cv::Mat surface_bounds_mask = makeSurfaceBoundsMask(geometry, frame_size);
    auto finishWithinSurface = [&](const cv::Mat& mask) {
        return clipMaskToSurfaceBounds(mask, surface_bounds_mask);
    };
    if (mask_u8.empty()) {
        if (stylus_wedge_status != nullptr) {
            *stylus_wedge_status = "mask_empty";
        }
        return mask_u8;
    }
    cv::Point2f tip;
    if (!h5StylusPointOnV3Surface(
            h5_corners,
            frame_index,
            reference,
            geometry,
            frame_size,
            &tip,
            false)) {
        if (stylus_wedge_status != nullptr) {
            *stylus_wedge_status = "h5_tip_xy_missing";
        }
        return finishWithinSurface(mask_u8);
    }

    cv::Mat binary;
    cv::threshold(mask_u8, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat surface_depth_candidate;
    if (!surface_bounds_mask.empty() &&
            !depth_difference_u8.empty() &&
            depth_difference_u8.size() == frame_size) {
        cv::threshold(
            depth_difference_u8,
            surface_depth_candidate,
            std::clamp(envInt("SHADOW_DEPTH_GRADIENT_MASK_THRESHOLD", 16), 0, 255),
            255,
            cv::THRESH_BINARY);
        cv::bitwise_and(surface_depth_candidate, surface_bounds_mask, surface_depth_candidate);
        surface_depth_candidate = selectBestSurfaceDepthComponent(
            surface_depth_candidate,
            depth_difference_u8,
            tip,
            true);
    }
    if (envInt("SHADOW_STYLUS_AOI_RETHRESH_DEPTH_MASK", 0) != 0 &&
            !surface_depth_candidate.empty() &&
            cv::countNonZero(surface_depth_candidate) > 0) {
        binary = surface_depth_candidate;
    } else if (!surface_bounds_mask.empty()) {
        cv::bitwise_and(binary, surface_bounds_mask, binary);
    }
    cv::Mat out = binary.clone();
    const cv::Point tip_i(cvRound(tip.x), cvRound(tip.y));
    constexpr int tip_radius = 2;
    const float tip_backoff = static_cast<float>(std::clamp(envInt("SHADOW_STYLUS_TIP_BACKOFF", 14), 0, 40));
    if (tip_i.x < 0 || tip_i.y < 0 || tip_i.x >= frame_size.width || tip_i.y >= frame_size.height) {
        if (stylus_wedge_status != nullptr) {
            *stylus_wedge_status = "h5_tip_outside_frame";
        }
        return finishWithinSurface(out);
    }
    if (debug_wedge_bgr != nullptr) {
        if (frame_bgr.channels() == 1) {
            cv::cvtColor(frame_bgr, *debug_wedge_bgr, cv::COLOR_GRAY2BGR);
        } else {
            *debug_wedge_bgr = frame_bgr.clone();
        }
        const int radius = std::clamp(envInt("SHADOW_STYLUS_AOI_RADIUS", 35), 20, 80);
        cv::circle(*debug_wedge_bgr, tip_i, radius, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        cv::circle(*debug_wedge_bgr, tip_i, 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        if (depth_difference_u8.empty()) {
            cv::putText(
                *debug_wedge_bgr,
                "AOI: depth difference missing",
                cv::Point(std::max(0, tip_i.x - radius), std::max(20, tip_i.y - radius - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(0, 0, 255),
                1,
                cv::LINE_AA);
        } else if (depth_difference_u8.size() != frame_size) {
            cv::putText(
                *debug_wedge_bgr,
                "AOI: depth difference size mismatch",
                cv::Point(std::max(0, tip_i.x - radius), std::max(20, tip_i.y - radius - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(0, 0, 255),
                1,
                cv::LINE_AA);
        }
    }
    if (debug_wedge_target != nullptr) {
        *debug_wedge_target = tip_i;
    }

    cv::Point2f ray_start;
    cv::Point2f ray_end;
    cv::Point2f direction;
    bool have_ray = h5StylusTiltRayOnV3Surface(
        h5_corners,
        frame_index,
        reference,
        geometry,
        frame_size,
        &ray_start,
        &ray_end,
        false);
    if (!have_ray) {
        have_ray = h5StylusTiltRayFromCornerHomography(
            h5_corners,
            frame_index,
            frame_size,
            &ray_start,
            &ray_end,
            false);
    }
    if (have_ray) {
        direction = ray_end - ray_start;
        if (envInt("SHADOW_STYLUS_FLIP_RAY", 0) != 0) {
            direction *= -1.0f;
        }
    }
    double direction_norm = have_ray ? cv::norm(direction) : 0.0;
    std::string last_aoi_status = "aoi_not_attempted";
    std::string last_rgb_status = "rgb_prior_not_attempted";
    if (have_ray && direction_norm >= 1.0 && std::isfinite(direction_norm)) {
        cv::Point2f aoi_direction = direction * static_cast<float>(1.0 / direction_norm);
        if (debug_wedge_bgr != nullptr && !debug_wedge_bgr->empty()) {
            cv::line(
                *debug_wedge_bgr,
                tip_i,
                toIntPoint(tip + aoi_direction * 32.0f),
                cv::Scalar(255, 0, 255),
                2,
                cv::LINE_AA);
        }
        if (addStylusTipFromRgbPrior(out, frame_bgr, depth_difference_u8, surface_bounds_mask, tip, aoi_direction, tip_radius, frame_index, temporal_state, debug_wedge_bgr, sobel_candidates_debug_bgr, sobel_raw_debug_bgr, &last_rgb_status)) {
            if (stylus_wedge_status != nullptr) {
                *stylus_wedge_status = last_rgb_status;
            }
            return finishWithinSurface(out);
        }
        if (addStylusTipFromAoiEdges(out, frame_bgr, depth_difference_u8, surface_bounds_mask, tip, aoi_direction, tip_radius, frame_index, temporal_state, debug_wedge_bgr, sobel_candidates_debug_bgr, sobel_raw_debug_bgr, &last_aoi_status)) {
            if (stylus_wedge_status != nullptr) {
                *stylus_wedge_status = "fallback_aoi_after_rgb_prior_failed:" + last_rgb_status + ":" + last_aoi_status;
            }
            return finishWithinSurface(out);
        }
    } else if (h5_corners != nullptr &&
            h5_corners->tilt_available &&
            frame_index >= 0 &&
            static_cast<size_t>(frame_index) < h5_corners->stylus_tilt.size()) {
        const cv::Point2f tilt = h5_corners->stylus_tilt[static_cast<size_t>(frame_index)];
        if (std::isfinite(tilt.x) && std::isfinite(tilt.y)) {
            cv::Point2f raw_tilt_direction(tilt.x, tilt.y);
            const double raw_tilt_norm = cv::norm(raw_tilt_direction);
            if (raw_tilt_norm >= 1.0 && std::isfinite(raw_tilt_norm)) {
                raw_tilt_direction *= static_cast<float>(1.0 / raw_tilt_norm);
                if (envInt("SHADOW_STYLUS_FLIP_RAY", 0) != 0) {
                    raw_tilt_direction *= -1.0f;
                }
                if (debug_wedge_bgr != nullptr && !debug_wedge_bgr->empty()) {
                    cv::line(
                        *debug_wedge_bgr,
                        tip_i,
                        toIntPoint(tip + raw_tilt_direction * 32.0f),
                        cv::Scalar(255, 0, 255),
                        2,
                        cv::LINE_AA);
                }
                if (addStylusTipFromRgbPrior(out, frame_bgr, depth_difference_u8, surface_bounds_mask, tip, raw_tilt_direction, tip_radius, frame_index, temporal_state, debug_wedge_bgr, sobel_candidates_debug_bgr, sobel_raw_debug_bgr, &last_rgb_status)) {
                    if (stylus_wedge_status != nullptr) {
                        *stylus_wedge_status = last_rgb_status;
                    }
                    return finishWithinSurface(out);
                }
                if (addStylusTipFromAoiEdges(out, frame_bgr, depth_difference_u8, surface_bounds_mask, tip, raw_tilt_direction, tip_radius, frame_index, temporal_state, debug_wedge_bgr, sobel_candidates_debug_bgr, sobel_raw_debug_bgr, &last_aoi_status)) {
                    if (stylus_wedge_status != nullptr) {
                        *stylus_wedge_status = "fallback_aoi_after_rgb_prior_failed:" + last_rgb_status + ":" + last_aoi_status;
                    }
                    return finishWithinSurface(out);
                }
            }
        }
    }

    std::vector<cv::Point> points;
    cv::findNonZero(binary, points);
    if (points.empty()) {
        cv::circle(out, tip_i, tip_radius, cv::Scalar(255), cv::FILLED, cv::LINE_8);
        if (stylus_wedge_status != nullptr) {
            *stylus_wedge_status = "fallback_circle_empty_mask_after_" + last_aoi_status;
        }
        return finishWithinSurface(out);
    }

    if (!have_ray || direction_norm < 1.0 || !std::isfinite(direction_norm)) {
        double best_dist2 = std::numeric_limits<double>::max();
        cv::Point best_point = points.front();
        for (const cv::Point& point : points) {
            const double dx = static_cast<double>(point.x) - static_cast<double>(tip.x);
            const double dy = static_cast<double>(point.y) - static_cast<double>(tip.y);
            const double dist2 = dx * dx + dy * dy;
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best_point = point;
            }
        }

        const double nearest_distance = std::sqrt(best_dist2);
        if (nearest_distance > 90.0 || nearest_distance < 1.0) {
            cv::circle(out, tip_i, tip_radius, cv::Scalar(255), cv::FILLED, cv::LINE_8);
            if (stylus_wedge_status != nullptr) {
                *stylus_wedge_status = "fallback_circle_no_near_mask_after_" + last_aoi_status;
            }
            return finishWithinSurface(out);
        }
        direction = cv::Point2f(
            static_cast<float>(static_cast<double>(best_point.x) - static_cast<double>(tip.x)),
            static_cast<float>(static_cast<double>(best_point.y) - static_cast<double>(tip.y)));
        direction *= static_cast<float>(1.0 / nearest_distance);
        cv::Point2f nearest_f(static_cast<float>(best_point.x), static_cast<float>(best_point.y));
        cv::Point2f adjusted_tip = nearest_f - direction * tip_backoff;
        cv::Point adjusted_tip_i = toIntPoint(adjusted_tip);
        if (debug_wedge_target != nullptr) {
            *debug_wedge_target = adjusted_tip_i;
        }
        if (!addStylusTipFromAoiEdges(out, frame_bgr, depth_difference_u8, surface_bounds_mask, tip, direction, tip_radius, frame_index, temporal_state, debug_wedge_bgr, sobel_candidates_debug_bgr, sobel_raw_debug_bgr, &last_aoi_status)) {
            drawStylusTipWedge(out, binary, frame_bgr, adjusted_tip, nearest_f, direction, tip_radius);
            if (stylus_wedge_status != nullptr) {
                *stylus_wedge_status = "fallback_wedge_after_rgb_prior_failed:" + last_rgb_status + ":" + last_aoi_status;
            }
        } else if (stylus_wedge_status != nullptr) {
            *stylus_wedge_status = "fallback_aoi_after_rgb_prior_failed:" + last_rgb_status + ":" + last_aoi_status;
        }
        cv::morphologyEx(
            out,
            out,
            cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
            cv::Point(-1, -1),
            1);
        return finishWithinSurface(out);
    }

    direction *= static_cast<float>(1.0 / direction_norm);
    cv::Point2f normal(-direction.y, direction.x);
    constexpr double min_along = -20.0;
    constexpr double max_along = 140.0;
    constexpr double corridor_half_width = 24.0;
    double best_score = std::numeric_limits<double>::max();
    double best_along = 0.0;
    double best_perp = 0.0;
    cv::Point best_point(-1, -1);
    for (const cv::Point& point : points) {
        cv::Point2f delta(
            static_cast<float>(static_cast<double>(point.x) - static_cast<double>(tip.x)),
            static_cast<float>(static_cast<double>(point.y) - static_cast<double>(tip.y)));
        const double along = static_cast<double>(delta.dot(direction));
        const double perp = static_cast<double>(delta.dot(normal));
        if (along < min_along || along > max_along || std::abs(perp) > corridor_half_width) {
            continue;
        }
        const double score = (std::abs(perp) * 3.0) + (std::abs(along) * 0.20);
        if (score < best_score) {
            best_score = score;
            best_along = along;
            best_perp = perp;
            best_point = point;
        }
    }

    if (best_point.x < 0) {
        cv::circle(out, tip_i, tip_radius, cv::Scalar(255), cv::FILLED, cv::LINE_8);
        if (stylus_wedge_status != nullptr) {
            *stylus_wedge_status = "fallback_circle_no_corridor_point_after_" + last_aoi_status;
        }
        return finishWithinSurface(out);
    }
    if (debug_wedge_target != nullptr) {
        *debug_wedge_target = best_point;
    }

    cv::Point2f nearest_f(static_cast<float>(best_point.x), static_cast<float>(best_point.y));
    cv::Point2f adjusted_tip = nearest_f - direction * tip_backoff;
    cv::Point adjusted_tip_i = toIntPoint(adjusted_tip);
    if (debug_wedge_target != nullptr) {
        *debug_wedge_target = adjusted_tip_i;
    }
    if (!addStylusTipFromAoiEdges(out, frame_bgr, depth_difference_u8, surface_bounds_mask, tip, direction, tip_radius, frame_index, temporal_state, debug_wedge_bgr, sobel_candidates_debug_bgr, sobel_raw_debug_bgr, &last_aoi_status)) {
        drawStylusTipWedge(out, binary, frame_bgr, adjusted_tip, nearest_f, direction, tip_radius);
        if (stylus_wedge_status != nullptr) {
            *stylus_wedge_status = "fallback_wedge_after_rgb_prior_failed:" + last_rgb_status + ":" + last_aoi_status;
        }
    } else if (stylus_wedge_status != nullptr) {
        *stylus_wedge_status = "fallback_aoi_after_rgb_prior_failed:" + last_rgb_status + ":" + last_aoi_status;
    }
    cv::morphologyEx(
        out,
        out,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    return finishWithinSurface(out);
}

bool h5StylusTiltRayOnV3Surface(
    const H5CornerTrace* h5_corners,
    int frame_index,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    cv::Size frame_size,
    cv::Point2f* ray_start,
    cv::Point2f* ray_end,
    bool require_touch) {
    if (ray_start == nullptr ||
            ray_end == nullptr ||
            h5_corners == nullptr ||
            !h5_corners->stylus_available ||
            !h5_corners->tilt_available ||
            !geometry.ok ||
            geometry.H_ref_to_frame.empty() ||
            frame_index < 0 ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_pressure.size() ||
            static_cast<size_t>(frame_index) >= h5_corners->stylus_tilt.size()) {
        return false;
    }
    if (require_touch && h5_corners->stylus_pressure[static_cast<size_t>(frame_index)] <= 0.0f) {
        return false;
    }
    cv::Point2f ref_start;
    if (!h5StylusPointInReferenceSurface(
            h5_corners,
            frame_index,
            reference.reference_bgr.size(),
            &ref_start,
            require_touch)) {
        return false;
    }

    const cv::Point2f tilt = h5_corners->stylus_tilt[static_cast<size_t>(frame_index)];
    if (!std::isfinite(tilt.x) || !std::isfinite(tilt.y)) {
        return false;
    }
    const double tilt_x_rad = std::clamp(static_cast<double>(tilt.x), -60.0, 60.0) * CV_PI / 180.0;
    const double tilt_y_rad = std::clamp(static_cast<double>(tilt.y), -60.0, 60.0) * CV_PI / 180.0;

    std::vector<cv::Point2f> ref_sheet = referenceImageCorners(reference.reference_bgr.size());
    std::vector<cv::Point2f> frame_sheet = projectQuad(geometry.H_ref_to_frame, ref_sheet);
    if (!isFiniteQuad(frame_sheet)) {
        return false;
    }

    std::vector<cv::Point3f> object_points = {
        {0.0f, 0.0f, 0.0f},
        {static_cast<float>(reference.reference_bgr.cols - 1), 0.0f, 0.0f},
        {static_cast<float>(reference.reference_bgr.cols - 1), static_cast<float>(reference.reference_bgr.rows - 1), 0.0f},
        {0.0f, static_cast<float>(reference.reference_bgr.rows - 1), 0.0f},
    };

    const double focal_scale = static_cast<double>(std::max(25, envInt("SHADOW_H5_TILT_FOCAL_PERCENT", 120))) / 100.0;
    const double focal = focal_scale * static_cast<double>(std::max(frame_size.width, frame_size.height));
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        focal, 0.0, static_cast<double>(frame_size.width - 1) * 0.5,
        0.0, focal, static_cast<double>(frame_size.height - 1) * 0.5,
        0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    cv::Mat rvec;
    cv::Mat tvec;
    if (!cv::solvePnP(object_points, frame_sheet, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE)) {
        return false;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Vec3d normal_cam(
        R.at<double>(0, 2),
        R.at<double>(1, 2),
        R.at<double>(2, 2));
    const double normal_sign = normal_cam[2] > 0.0 ? -1.0 : 1.0;
    cv::Vec3d dir_sheet(std::tan(tilt_x_rad), std::tan(tilt_y_rad), normal_sign);
    const double dir_norm = cv::norm(dir_sheet);
    if (dir_norm <= 1e-9 || !std::isfinite(dir_norm)) {
        return false;
    }
    dir_sheet *= 1.0 / dir_norm;

    const double ray_length = static_cast<double>(std::clamp(envInt("SHADOW_H5_TILT_RAY_LENGTH", 140), 20, 600));
    std::vector<cv::Point3f> ray_object = {
        {ref_start.x, ref_start.y, 0.0f},
        {
            static_cast<float>(static_cast<double>(ref_start.x) + dir_sheet[0] * ray_length),
            static_cast<float>(static_cast<double>(ref_start.y) + dir_sheet[1] * ray_length),
            static_cast<float>(dir_sheet[2] * ray_length),
        },
    };
    std::vector<cv::Point2f> ray_projected;
    cv::projectPoints(ray_object, rvec, tvec, camera_matrix, dist_coeffs, ray_projected);
    if (ray_projected.size() != 2 ||
            !std::isfinite(ray_projected[0].x) || !std::isfinite(ray_projected[0].y) ||
            !std::isfinite(ray_projected[1].x) || !std::isfinite(ray_projected[1].y)) {
        return false;
    }

    std::vector<cv::Point2f> homography_start;
    cv::perspectiveTransform(std::vector<cv::Point2f>{ref_start}, homography_start, geometry.H_ref_to_frame);
    if (homography_start.size() != 1 ||
            !std::isfinite(homography_start[0].x) ||
            !std::isfinite(homography_start[0].y)) {
        return false;
    }
    *ray_start = homography_start[0];
    *ray_end = ray_projected[1];
    return true;
}

void drawH5StylusCursor(
    cv::Mat* image,
    const H5CornerTrace* h5_corners,
    int frame_index,
    const ReferenceModel* reference = nullptr,
    const GeometryResult* geometry = nullptr) {
    if (image == nullptr || image->empty()) {
        return;
    }
    cv::Point2f point;
    bool have_point = false;
    float pressure = 0.0f;
    bool have_pressure = false;
    if (h5_corners != nullptr &&
            frame_index >= 0 &&
            static_cast<size_t>(frame_index) < h5_corners->stylus_pressure.size()) {
        pressure = h5_corners->stylus_pressure[static_cast<size_t>(frame_index)];
        have_pressure = std::isfinite(pressure);
    }
    if (reference != nullptr && geometry != nullptr) {
        have_point = h5StylusPointOnV3Surface(
            h5_corners,
            frame_index,
            *reference,
            *geometry,
            image->size(),
            &point,
            false);
        if (!have_point && geometry->ok) {
            return;
        }
    }
    if (!have_point) {
        have_point = h5StylusPointFromCornerHomography(h5_corners, frame_index, image->size(), &point, false);
    }
    if (!have_point) {
        have_point = h5StylusPointInFrameBySourceScale(h5_corners, frame_index, image->size(), &point, false);
    }
    if (!have_point) {
        return;
    }
    if (geometry != nullptr && geometry->ok && !geometry->board_mask.empty()) {
        const int x = cvRound(point.x);
        const int y = cvRound(point.y);
        if (x < 0 || y < 0 || x >= geometry->board_mask.cols || y >= geometry->board_mask.rows ||
                geometry->board_mask.at<uchar>(y, x) == 0) {
            return;
        }
    }

    cv::Point2f ray_start;
    cv::Point2f ray_end;
    bool have_ray = false;
    if (reference != nullptr && geometry != nullptr) {
        have_ray = h5StylusTiltRayOnV3Surface(
            h5_corners,
            frame_index,
            *reference,
            *geometry,
            image->size(),
            &ray_start,
            &ray_end);
    }
    if (!have_ray && (geometry == nullptr || !geometry->ok)) {
        have_ray = h5StylusTiltRayFromCornerHomography(h5_corners, frame_index, image->size(), &ray_start, &ray_end);
    }
    if (have_ray) {
        cv::line(
            *image,
            toIntPoint(ray_start),
            toIntPoint(ray_end),
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA);
        cv::circle(*image, toIntPoint(ray_end), 3, cv::Scalar(0, 180, 255), cv::FILLED, cv::LINE_AA);
    }

    const cv::Point center(cvRound(point.x), cvRound(point.y));
    const bool touching = have_pressure && pressure > 0.0f;
    const cv::Scalar marker_color = touching ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 255);
    cv::circle(*image, center, 5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::circle(*image, center, 4, marker_color, 1, cv::LINE_AA);
    cv::line(*image, center + cv::Point(-7, 0), center + cv::Point(7, 0), marker_color, 1, cv::LINE_AA);
    cv::line(*image, center + cv::Point(0, -7), center + cv::Point(0, 7), marker_color, 1, cv::LINE_AA);
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

    model.reference_blue_mask = makeBlueMask(model.reference_bgr);
    model.reference_green_mask = makeGreenMask(model.reference_bgr);
    cv::Mat reference_mask_overlap;
    cv::bitwise_and(model.reference_blue_mask, model.reference_green_mask, reference_mask_overlap);
    model.reference_visual_masks_exclusive = cv::countNonZero(reference_mask_overlap) == 0;
    if (model.reference_visual_masks_exclusive) {
        model.reference_visual_label_mask = cv::Mat::zeros(model.reference_bgr.size(), CV_8UC1);
        model.reference_visual_label_mask.setTo(1, model.reference_blue_mask);
        model.reference_visual_label_mask.setTo(2, model.reference_green_mask);
    }

    model.reference_tags = detectAprilTags(model.reference_bgr);
    if (model.reference_tags.detections.size() < 2) {
        throw std::runtime_error("Reference sheet must contain at least 2 detectable AprilTags");
    }
    for (const auto& [id, ref_corners] : model.reference_tags.detections) {
        model.reference_tag_patterns[id] = buildReferenceTagPattern(model.reference_bgr, ref_corners);
    }
    return model;
}

GeometryResult estimateTagSheetGeometry(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options* options,
    const cv::Mat& seed_H_ref_to_frame,
    const DetectionResult* precomputed_frame_tags) {
    GeometryResult result;
    result.frame_index = frame_index;
    if (precomputed_frame_tags != nullptr) {
        result.frame_tags = *precomputed_frame_tags;
    } else {
        ScopedStageTimer timer(&g_geometry_timing.april_tag_detection);
        result.frame_tags = detectAprilTags(frame_bgr);
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        auto frame_it = result.frame_tags.detections.find(id);
        if (frame_it == result.frame_tags.detections.end()) {
            continue;
        }
        ++result.common_tag_count;
        const std::vector<cv::Point2f>& frame_corners = frame_it->second;
        src_pts.insert(src_pts.end(), ref_corners.begin(), ref_corners.end());
        dst_pts.insert(dst_pts.end(), frame_corners.begin(), frame_corners.end());
    }

    if (result.common_tag_count >= 3 && src_pts.size() >= 12) {
        {
            ScopedStageTimer timer(&g_geometry_timing.normal_homography);
            result.H_ref_to_frame = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 6.0);
        }
        result.ok = !result.H_ref_to_frame.empty();
        attachProjectedBoardMask(&result, reference.reference_bgr.size(), frame_bgr.size());
        return result;
    }

    if (result.common_tag_count >= 1 && result.common_tag_count < 3) {
        estimateFromSparseTagsAndSoftTags(
            frame_bgr,
            reference,
            seed_H_ref_to_frame,
            &result,
            options != nullptr && options->fast,
            options != nullptr && options->center_consensus,
            options != nullptr && options->partial_cell_pattern);
        if (options != nullptr && !options->frames.empty()) {
            drawTagDiagnostics(frame_bgr, reference, result, options->outdir);
        }
    } else if (result.common_tag_count == 0 && !seed_H_ref_to_frame.empty()) {
        estimateFromPreviousSeedAndSoftTags(frame_bgr, reference, seed_H_ref_to_frame, &result);
        if (options != nullptr && !options->frames.empty()) {
            drawTagDiagnostics(frame_bgr, reference, result, options->outdir);
        }
    }
    attachProjectedBoardMask(&result, reference.reference_bgr.size(), frame_bgr.size());
    return result;
}

FrameResult renderTagSheetOverlayWithGeometry(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    GeometryResult geometry,
    RenderScratch* scratch,
    const H5CornerTrace* h5_corners);

FrameResult renderTagSheetOverlayV3(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    const cv::Mat& seed_H_ref_to_frame,
    const DetectionResult* precomputed_frame_tags) {
    GeometryResult geometry = estimateTagSheetGeometry(
        frame_index,
        frame_bgr,
        reference,
        &options,
        seed_H_ref_to_frame,
        precomputed_frame_tags);
    return renderTagSheetOverlayWithGeometry(
        frame_index,
        frame_bgr,
        reference,
        options,
        std::move(geometry),
        nullptr,
        nullptr);
}

struct ColormatchArtifacts {
    cv::Mat raw_frame;
    cv::Mat aligned_ref;
    cv::Mat aligned_ref_colormatch;
    cv::Mat hand_stylus_mask;
    cv::Mat hand_mask_board_clipped;
    cv::Mat hand_mask_colormatch_fallback;
    cv::Mat hand_mask_visible_sheet_excluded;
    cv::Mat hand_mask_holes_filled;
    cv::Mat hand_mask_yellow_stylus_added;
    cv::Mat foreground_keep_mask_original;
    cv::Mat foreground_keep_mask_adjusted;
    cv::Mat aligned_ref_masked_difference;
    cv::Mat aligned_ref_masked_difference_clean;
    cv::Mat shadow_factor_bgr;
    cv::Mat surface_texture;
    cv::Mat surface_texture_texture_only;
    cv::Mat surface_texture_before_edge_cleanup;
    cv::Mat edge_chroma_cleanup_candidate;
    cv::Mat final_overlay;
    cv::Mat stylus_tip_wedge_debug;
    cv::Mat stylus_tip_wedge_debug_zoom;
    cv::Mat stylus_sobel_candidates_debug;
    cv::Mat stylus_sobel_candidates_debug_zoom;
    cv::Mat stylus_sobel_raw_debug;
    cv::Mat stylus_sobel_raw_debug_zoom;
    cv::Point stylus_wedge_debug_target = {-1, -1};
    bool ok = false;
};

ColormatchArtifacts makeColormatchArtifacts(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    const cv::Mat& depth_hand_mask,
    const cv::Mat& surface_texture_ref_bgr,
    const Options& options,
    const cv::Point& stylus_wedge_debug_target = cv::Point(-1, -1),
    const cv::Mat& stylus_tip_wedge_debug = cv::Mat(),
    const cv::Mat& stylus_sobel_candidates_debug = cv::Mat(),
    const cv::Mat& stylus_sobel_raw_debug = cv::Mat()) {
    ColormatchArtifacts artifacts;
    if (!geometry.ok || geometry.H_ref_to_frame.empty()) {
        return artifacts;
    }
    {
        ScopedStageTimer timer(&g_render_timing.raw_frame_clone);
        artifacts.raw_frame = frame_bgr.clone();
    }

    {
        ScopedStageTimer timer(&g_render_timing.aligned_ref_warp);
        cv::warpPerspective(
            reference.reference_bgr,
            artifacts.aligned_ref,
            geometry.H_ref_to_frame,
            frame_bgr.size(),
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0));
    }
    {
        ScopedStageTimer timer(&g_render_timing.aligned_ref_clone);
        artifacts.aligned_ref_colormatch = artifacts.aligned_ref.clone();
    }
    {
        ScopedStageTimer timer(&g_render_timing.tag_color_match);
        applyTopLeftTagColorMatch(
            reference.reference_bgr,
            frame_bgr,
            reference.reference_tags,
            geometry.frame_tags,
            geometry.board_mask,
            &artifacts.aligned_ref_colormatch);
    }
    {
        ScopedStageTimer timer(&g_render_timing.depth_mask_prepare);
        if (!depth_hand_mask.empty()) {
            artifacts.hand_stylus_mask = depth_hand_mask.clone();
        } else {
            artifacts.hand_stylus_mask = cv::Mat::zeros(frame_bgr.size(), CV_8UC1);
        }
        if (!geometry.board_mask.empty()) {
            cv::threshold(artifacts.hand_stylus_mask, artifacts.hand_stylus_mask, 0, 255, cv::THRESH_BINARY);
            cv::bitwise_and(artifacts.hand_stylus_mask, geometry.board_mask, artifacts.hand_stylus_mask);
        }
        artifacts.hand_mask_board_clipped = artifacts.hand_stylus_mask.clone();
    }
    const bool have_depth_hand_mask = cv::countNonZero(artifacts.hand_stylus_mask) > 0;
    {
        ScopedStageTimer timer(&g_render_timing.empty_mask_fallback);
        if (envInt("SHADOW_HAND_MASK_COLORMATCH_UNION", 0) != 0 ||
                cv::countNonZero(artifacts.hand_stylus_mask) == 0) {
            cv::Mat colormatch_difference = makeAlignedRefMaskedDifference(
                frame_bgr,
                artifacts.aligned_ref_colormatch,
                geometry.board_mask,
                cv::Mat());
            double fallback_otsu = 0.0;
            int fallback_threshold = 0;
            cv::Mat fallback_mask = makeHandPenMaskFromColormatchDifference(
                colormatch_difference,
                artifacts.aligned_ref_colormatch,
                frame_bgr,
                cv::Mat(),
                &fallback_otsu,
                &fallback_threshold);
            if (!geometry.board_mask.empty() && !fallback_mask.empty()) {
                cv::bitwise_and(fallback_mask, geometry.board_mask, fallback_mask);
            }
            if (!fallback_mask.empty() && cv::countNonZero(fallback_mask) > 0) {
                if (cv::countNonZero(artifacts.hand_stylus_mask) == 0) {
                    artifacts.hand_stylus_mask = std::move(fallback_mask);
                } else {
                    cv::bitwise_or(artifacts.hand_stylus_mask, fallback_mask, artifacts.hand_stylus_mask);
                }
            }
        }
        artifacts.hand_mask_colormatch_fallback = artifacts.hand_stylus_mask.clone();
        if (!have_depth_hand_mask || envInt("SHADOW_HAND_MASK_EXCLUDE_VISIBLE_SHEET_ON_DEPTH", 1) != 0) {
            artifacts.hand_stylus_mask = excludeVisibleSheetColorFromHandStylusMask(
                artifacts.hand_stylus_mask,
                frame_bgr,
                geometry.board_mask,
                stylus_wedge_debug_target);
        }
        artifacts.hand_mask_visible_sheet_excluded = artifacts.hand_stylus_mask.clone();
        if (envInt("SHADOW_HAND_MASK_FILL_HOLES", 1) != 0 &&
                (!have_depth_hand_mask || envInt("SHADOW_HAND_MASK_FILL_HOLES_ON_DEPTH", 0) != 0)) {
            artifacts.hand_stylus_mask = fillBinaryMaskHoles(artifacts.hand_stylus_mask);
            if (!geometry.board_mask.empty() && geometry.board_mask.size() == artifacts.hand_stylus_mask.size()) {
                cv::bitwise_and(artifacts.hand_stylus_mask, geometry.board_mask, artifacts.hand_stylus_mask);
            }
        }
        artifacts.hand_mask_holes_filled = artifacts.hand_stylus_mask.clone();
        if (!have_depth_hand_mask || envInt("SHADOW_HAND_MASK_INCLUDE_YELLOW_STYLUS_ON_DEPTH", 0) != 0) {
            artifacts.hand_stylus_mask = addYellowStylusBodyToHandMask(
                artifacts.hand_stylus_mask,
                frame_bgr,
                geometry.board_mask);
        }
        artifacts.hand_mask_yellow_stylus_added = artifacts.hand_stylus_mask.clone();
    }
    {
        ScopedStageTimer timer(&g_render_timing.foreground_keep_mask);
        artifacts.foreground_keep_mask_adjusted = makeForegroundKeepMaskForCompositing(
            artifacts.hand_stylus_mask,
            options,
            geometry.board_mask,
            &artifacts.foreground_keep_mask_original);
    }
    {
        ScopedStageTimer timer(&g_render_timing.masked_difference);
        artifacts.aligned_ref_masked_difference = makeAlignedRefMaskedDifference(
            frame_bgr,
            artifacts.aligned_ref_colormatch,
            geometry.board_mask,
            artifacts.hand_stylus_mask);
    }
    {
        ScopedStageTimer timer(&g_render_timing.clean_difference);
        artifacts.aligned_ref_masked_difference_clean = makeCleanAlignedRefMaskedDifference(
            artifacts.aligned_ref_masked_difference,
            artifacts.aligned_ref_colormatch,
            artifacts.hand_stylus_mask);
    }
    if (!surface_texture_ref_bgr.empty()) {
        {
            ScopedStageTimer timer(&g_render_timing.surface_texture);
            artifacts.surface_texture = makeSurfaceTextureOutput(
                frame_bgr,
                surface_texture_ref_bgr,
                geometry.H_ref_to_frame,
                geometry.board_mask,
                artifacts.foreground_keep_mask_adjusted,
                &artifacts.surface_texture_texture_only);
            artifacts.surface_texture_before_edge_cleanup = artifacts.surface_texture.clone();
            artifacts.surface_texture = suppressGreenBlueMaskEdgeFringe(
                artifacts.surface_texture,
                artifacts.surface_texture_texture_only,
                artifacts.foreground_keep_mask_adjusted,
                geometry.board_mask,
                &artifacts.edge_chroma_cleanup_candidate);
        }
        {
            ScopedStageTimer timer(&g_render_timing.shadow_factor);
            artifacts.shadow_factor_bgr = computeShadowFactorBgr(
                artifacts.aligned_ref_colormatch,
                frame_bgr,
                artifacts.aligned_ref_masked_difference_clean);
        }
        {
            ScopedStageTimer timer(&g_render_timing.apply_shadow_factor);
            artifacts.final_overlay = applyShadowFactorToBaseBgr(
                artifacts.surface_texture,
                artifacts.shadow_factor_bgr,
                options.shadow_strength);
        }
    }
    if (!options.fast) {
        ScopedStageTimer timer(&g_render_timing.debug_artifacts);
        artifacts.stylus_wedge_debug_target = stylus_wedge_debug_target;
        artifacts.stylus_tip_wedge_debug = stylus_tip_wedge_debug.clone();
        artifacts.stylus_sobel_candidates_debug = stylus_sobel_candidates_debug.clone();
        artifacts.stylus_sobel_raw_debug = stylus_sobel_raw_debug.clone();
        if (!stylus_tip_wedge_debug.empty() && stylus_wedge_debug_target.x >= 0 && stylus_wedge_debug_target.y >= 0) {
            const int radius = std::clamp(envInt("SHADOW_STYLUS_AOI_RADIUS", 35), 20, 80);
            const cv::Rect image_rect(0, 0, stylus_tip_wedge_debug.cols, stylus_tip_wedge_debug.rows);
            cv::Rect crop(
                stylus_wedge_debug_target.x - radius,
                stylus_wedge_debug_target.y - radius,
                radius * 2 + 1,
                radius * 2 + 1);
            crop &= image_rect;
            if (crop.width > 0 && crop.height > 0) {
                cv::resize(
                    stylus_tip_wedge_debug(crop),
                    artifacts.stylus_tip_wedge_debug_zoom,
                    cv::Size(crop.width * 5, crop.height * 5),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);
            }
        }
        if (!stylus_sobel_candidates_debug.empty() && stylus_wedge_debug_target.x >= 0 && stylus_wedge_debug_target.y >= 0) {
            const int radius = std::clamp(envInt("SHADOW_STYLUS_AOI_RADIUS", 35), 20, 80);
            const cv::Rect image_rect(0, 0, stylus_sobel_candidates_debug.cols, stylus_sobel_candidates_debug.rows);
            cv::Rect crop(
                stylus_wedge_debug_target.x - radius,
                stylus_wedge_debug_target.y - radius,
                radius * 2 + 1,
                radius * 2 + 1);
            crop &= image_rect;
            if (crop.width > 0 && crop.height > 0) {
                cv::resize(
                    stylus_sobel_candidates_debug(crop),
                    artifacts.stylus_sobel_candidates_debug_zoom,
                    cv::Size(crop.width * 5, crop.height * 5),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);
            }
        }
        if (!stylus_sobel_raw_debug.empty() && stylus_wedge_debug_target.x >= 0 && stylus_wedge_debug_target.y >= 0) {
            const int radius = std::clamp(envInt("SHADOW_STYLUS_AOI_RADIUS", 35), 20, 80);
            const cv::Rect image_rect(0, 0, stylus_sobel_raw_debug.cols, stylus_sobel_raw_debug.rows);
            cv::Rect crop(
                stylus_wedge_debug_target.x - radius,
                stylus_wedge_debug_target.y - radius,
                radius * 2 + 1,
                radius * 2 + 1);
            crop &= image_rect;
            if (crop.width > 0 && crop.height > 0) {
                cv::resize(
                    stylus_sobel_raw_debug(crop),
                    artifacts.stylus_sobel_raw_debug_zoom,
                    cv::Size(crop.width * 5, crop.height * 5),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);
            }
        }
    }
    artifacts.ok = true;
    return artifacts;
}

cv::Mat makeTextureFinalOverlay(
    const cv::Mat& frame_bgr,
    const GeometryResult& geometry,
    const ColormatchArtifacts& artifacts,
    const cv::Mat& surface_texture_ref_bgr,
    double shadow_strength = 1.0,
    cv::Mat* surface_texture_bgr = nullptr) {
    if (!artifacts.ok || surface_texture_ref_bgr.empty()) {
        return cv::Mat();
    }
    cv::Mat surface_texture;
    cv::Mat texture_only;
    cv::Mat cleanup_candidate;
    {
        ScopedStageTimer timer(&g_render_timing.texture_variant_surface);
        surface_texture = makeSurfaceTextureOutput(
            frame_bgr,
            surface_texture_ref_bgr,
            geometry.H_ref_to_frame,
            geometry.board_mask,
            artifacts.foreground_keep_mask_adjusted,
            &texture_only);
        surface_texture = suppressGreenBlueMaskEdgeFringe(
            surface_texture,
            texture_only,
            artifacts.foreground_keep_mask_adjusted,
            geometry.board_mask,
            &cleanup_candidate);
    }
    if (surface_texture_bgr != nullptr) {
        *surface_texture_bgr = surface_texture.clone();
    }
    if (!artifacts.shadow_factor_bgr.empty()) {
        ScopedStageTimer timer(&g_render_timing.texture_variant_apply_shadow);
        return applyShadowFactorToBaseBgr(surface_texture, artifacts.shadow_factor_bgr, shadow_strength);
    }
    ScopedStageTimer timer(&g_render_timing.texture_variant_apply_shadow);
    return transferableShadowOverlayBgr(
        artifacts.aligned_ref_colormatch,
        frame_bgr,
        artifacts.aligned_ref_masked_difference_clean,
        surface_texture);
}

void writeColormatchArtifacts(
    const fs::path& outdir,
    const std::string& prefix,
    const ColormatchArtifacts& artifacts,
    bool final_overlay_only,
    bool write_foreground_keep_mask_debug) {
    if (!artifacts.ok) {
        return;
    }
    if (final_overlay_only) {
        if (!artifacts.final_overlay.empty()) {
            writePngFast(outdir / (prefix + "_final_overlay.png"), artifacts.final_overlay);
        }
        if (write_foreground_keep_mask_debug) {
            if (!artifacts.foreground_keep_mask_original.empty()) {
                writePngFast(outdir / (prefix + "_foreground_keep_mask_original.png"), artifacts.foreground_keep_mask_original);
            }
            if (!artifacts.foreground_keep_mask_adjusted.empty()) {
                writePngFast(outdir / (prefix + "_foreground_keep_mask_adjusted.png"), artifacts.foreground_keep_mask_adjusted);
            }
        }
        return;
    }
    writePngFast(outdir / (prefix + "_raw_frame.png"), artifacts.raw_frame);
    writePngFast(outdir / (prefix + "_aligned_ref.png"), artifacts.aligned_ref);
    writePngFast(outdir / (prefix + "_aligned_ref_colormatch.png"), artifacts.aligned_ref_colormatch);
    writePngFast(outdir / (prefix + "_aligned_ref_colormatch_hand_stylus_mask.png"), artifacts.hand_stylus_mask);
    if (write_foreground_keep_mask_debug) {
        if (!artifacts.foreground_keep_mask_original.empty()) {
            writePngFast(outdir / (prefix + "_foreground_keep_mask_original.png"), artifacts.foreground_keep_mask_original);
        }
        if (!artifacts.foreground_keep_mask_adjusted.empty()) {
            writePngFast(outdir / (prefix + "_foreground_keep_mask_adjusted.png"), artifacts.foreground_keep_mask_adjusted);
        }
    }
    writePngFast(outdir / (prefix + "_aligned_ref_masked_difference.png"), artifacts.aligned_ref_masked_difference);
    writePngFast(outdir / (prefix + "_aligned_ref_masked_difference_clean.png"), artifacts.aligned_ref_masked_difference_clean);
    if (!artifacts.surface_texture.empty()) {
        writePngFast(outdir / (prefix + "_surface_texture.png"), artifacts.surface_texture);
    }
    if (!artifacts.surface_texture_texture_only.empty()) {
        writePngFast(
            outdir / (prefix + "_surface_texture_texture_only.png"),
            artifacts.surface_texture_texture_only);
    }
    if (!artifacts.surface_texture_before_edge_cleanup.empty()) {
        writePngFast(
            outdir / (prefix + "_surface_texture_before_edge_cleanup.png"),
            artifacts.surface_texture_before_edge_cleanup);
    }
    if (!artifacts.edge_chroma_cleanup_candidate.empty()) {
        writePngFast(outdir / (prefix + "_edge_chroma_cleanup_candidate.png"), artifacts.edge_chroma_cleanup_candidate);
    }
    if (!artifacts.shadow_factor_bgr.empty()) {
        cv::Mat shadow_factor_u8;
        artifacts.shadow_factor_bgr.convertTo(shadow_factor_u8, CV_8UC3, 255.0);
        writePngFast(
            outdir / (prefix + "_transferable_shadow_factor.png"),
            shadow_factor_u8);

        cv::Mat white_base(
            artifacts.shadow_factor_bgr.size(),
            CV_8UC3,
            cv::Scalar(255, 255, 255));
        cv::Mat shadow_on_white = applyShadowFactorToBaseBgr(
            white_base,
            artifacts.shadow_factor_bgr,
            1.0);
        writePngFast(
            outdir / (prefix + "_transferable_shadow_on_white.png"),
            shadow_on_white);
    }
    if (!artifacts.final_overlay.empty()) {
        writePngFast(outdir / (prefix + "_final_overlay.png"), artifacts.final_overlay);
    }
    if (!artifacts.stylus_tip_wedge_debug.empty()) {
        writePngFast(outdir / (prefix + "_stylus_tip_wedge_debug.png"), artifacts.stylus_tip_wedge_debug);
    }
    if (!artifacts.stylus_tip_wedge_debug_zoom.empty()) {
        writePngFast(outdir / (prefix + "_stylus_tip_wedge_debug_zoom.png"), artifacts.stylus_tip_wedge_debug_zoom);
    }
    if (!artifacts.stylus_sobel_candidates_debug.empty()) {
        writePngFast(outdir / (prefix + "_stylus_sobel_candidates_debug.png"), artifacts.stylus_sobel_candidates_debug);
    }
    if (!artifacts.stylus_sobel_candidates_debug_zoom.empty()) {
        writePngFast(outdir / (prefix + "_stylus_sobel_candidates_debug_zoom.png"), artifacts.stylus_sobel_candidates_debug_zoom);
    }
    if (!artifacts.stylus_sobel_raw_debug.empty()) {
        writePngFast(outdir / (prefix + "_stylus_sobel_raw_debug.png"), artifacts.stylus_sobel_raw_debug);
    }
    if (!artifacts.stylus_sobel_raw_debug_zoom.empty()) {
        writePngFast(outdir / (prefix + "_stylus_sobel_raw_debug_zoom.png"), artifacts.stylus_sobel_raw_debug_zoom);
    }
}

void writeHandMaskStagePngs(
    const fs::path& outdir,
    const std::string& prefix,
    const cv::Mat& depth_difference,
    const cv::Mat& depth_candidate,
    const cv::Mat& stylus_connected,
    const ColormatchArtifacts& artifacts) {
    auto writeStage = [&](const std::string& suffix, const cv::Mat& image) {
        if (!image.empty()) {
            writePngFast(outdir / (prefix + suffix + ".png"), image);
        }
    };
    writeStage("_hand_mask_stage_00_depth_difference", depth_difference);
    writeStage("_hand_mask_stage_01_depth_candidate", depth_candidate);
    writeStage("_hand_mask_stage_02_stylus_connected", stylus_connected);
    writeStage("_hand_mask_stage_03_board_clipped", artifacts.hand_mask_board_clipped);
    writeStage("_hand_mask_stage_04_colormatch_fallback", artifacts.hand_mask_colormatch_fallback);
    writeStage("_hand_mask_stage_05_visible_sheet_excluded", artifacts.hand_mask_visible_sheet_excluded);
    writeStage("_hand_mask_stage_06_holes_filled", artifacts.hand_mask_holes_filled);
    writeStage("_hand_mask_stage_07_yellow_stylus_added", artifacts.hand_mask_yellow_stylus_added);
}

FrameResult renderTagSheetOverlayWithGeometry(
    int frame_index,
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const Options& options,
    GeometryResult geometry,
    RenderScratch* scratch,
    const H5CornerTrace* h5_corners) {
    FrameResult result;
    result.frame_index = frame_index;
    result.geometry = std::move(geometry);

    if (!result.geometry.ok) {
        result.output_bgr = frame_bgr.clone();
        drawH5CornerOutline(&result.output_bgr, h5_corners, frame_index);
        drawH5StylusCursor(&result.output_bgr, h5_corners, frame_index);
        result.ok = true;
        return result;
    }

    RenderScratch local_scratch;
    RenderScratch& render_scratch = scratch != nullptr ? *scratch : local_scratch;
    cv::warpPerspective(
        reference.reference_bgr,
        render_scratch.warped_ref,
        result.geometry.H_ref_to_frame,
        frame_bgr.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));

    if (render_scratch.ref_mask.empty() ||
            render_scratch.ref_mask.size() != reference.reference_bgr.size() ||
            render_scratch.ref_mask.type() != CV_8U) {
        render_scratch.ref_mask.create(reference.reference_bgr.size(), CV_8U);
        render_scratch.ref_mask.setTo(cv::Scalar(255));
    }
    cv::warpPerspective(
        render_scratch.ref_mask,
        render_scratch.warped_mask,
        result.geometry.H_ref_to_frame,
        frame_bgr.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));

    const double alpha = std::clamp(options.overlay_alpha, 0.0, 1.0);
    cv::addWeighted(render_scratch.warped_ref, alpha, frame_bgr, 1.0 - alpha, 0.0, render_scratch.blended);

    result.output_bgr = frame_bgr.clone();
    render_scratch.blended.copyTo(result.output_bgr, render_scratch.warped_mask);
    drawH5CornerOutline(&result.output_bgr, h5_corners, frame_index);
    drawH5StylusCursor(&result.output_bgr, h5_corners, frame_index, &reference, &result.geometry);
    result.ok = true;
    return result;
}

double geometryChoiceScore(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const GeometryResult& geometry,
    FrameMaskCache* frame_masks = nullptr) {
    if (!geometry.ok || geometry.H_ref_to_frame.empty()) {
        return std::numeric_limits<double>::max();
    }

    double score = geometry.used_previous_homography_seed ? 3.0 : 0.0;
    SheetEdgeAlignment edge_alignment = checkSheetEdgeAlignment(
        frame_bgr,
        reference,
        geometry.H_ref_to_frame,
        frame_masks);
    if (edge_alignment.usable) {
        score += edge_alignment.max_abs_offset;
        if (!edge_alignment.ok) {
            score += 100.0;
        }
    }

    double hard_error = 0.0;
    int hard_points = 0;
    for (const auto& [id, frame_corners] : geometry.frame_tags.detections) {
        auto ref_it = reference.reference_tags.detections.find(id);
        if (ref_it == reference.reference_tags.detections.end()) {
            continue;
        }
        std::vector<cv::Point2f> projected = projectQuad(geometry.H_ref_to_frame, ref_it->second);
        if (projected.size() != 4 || frame_corners.size() != 4) {
            continue;
        }
        for (int i = 0; i < 4; ++i) {
            hard_error += cv::norm(projected[static_cast<size_t>(i)] - frame_corners[static_cast<size_t>(i)]);
            ++hard_points;
        }
    }
    if (hard_points > 0) {
        score += 0.35 * hard_error / static_cast<double>(hard_points);
    }

    return score;
}

double geometryReferenceXSpread(
    const ReferenceModel& reference,
    const GeometryResult& geometry) {
    std::vector<cv::Point2f> centers;
    for (const auto& [id, corners] : geometry.frame_tags.detections) {
        (void)corners;
        auto ref_it = reference.reference_tags.detections.find(id);
        if (ref_it != reference.reference_tags.detections.end()) {
            centers.push_back(quadCenter(ref_it->second));
        }
    }
    for (const SoftTag& soft : geometry.soft_tags) {
        auto ref_it = reference.reference_tags.detections.find(soft.id);
        if (ref_it != reference.reference_tags.detections.end()) {
            centers.push_back(quadCenter(ref_it->second));
        }
    }
    if (centers.size() < 2) {
        return 0.0;
    }
    float min_x = centers.front().x;
    float max_x = centers.front().x;
    for (const cv::Point2f& center : centers) {
        min_x = std::min(min_x, center.x);
        max_x = std::max(max_x, center.x);
    }
    return (max_x - min_x) / std::max(1.0, static_cast<double>(reference.reference_bgr.cols));
}

int countCommonReferenceTags(
    const ReferenceModel& reference,
    const DetectionResult& frame_tags) {
    int common = 0;
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        (void)ref_corners;
        if (frame_tags.detections.count(id) != 0) {
            ++common;
        }
    }
    return common;
}

GeometryResult geometryFromPreviousSeedIfConsistent(
    int frame_index,
    const ReferenceModel& reference,
    const DetectionResult& frame_tags,
    const cv::Mat& seed_H_ref_to_frame,
    cv::Size frame_size) {
    GeometryResult result;
    result.frame_index = frame_index;
    result.frame_tags = frame_tags;
    if (seed_H_ref_to_frame.empty()) {
        return result;
    }

    std::vector<int> hard_tag_ids;
    for (const auto& [id, ref_corners] : reference.reference_tags.detections) {
        (void)ref_corners;
        if (frame_tags.detections.count(id) != 0) {
            hard_tag_ids.push_back(id);
        }
    }
    result.common_tag_count = static_cast<int>(hard_tag_ids.size());
    if (hard_tag_ids.empty() || result.common_tag_count >= 3) {
        return result;
    }

    const double max_seed_error = static_cast<double>(envInt("SHADOW_FAST_SEED_HOLD_MAX_ERROR", 35));
    const double mean_seed_error = meanHardTagReprojectionError(
        reference,
        frame_tags,
        hard_tag_ids,
        seed_H_ref_to_frame);
    if (!std::isfinite(mean_seed_error) || mean_seed_error > max_seed_error) {
        return result;
    }

    result.H_seed_ref_to_frame = seed_H_ref_to_frame.clone();
    result.H_ref_to_frame = seed_H_ref_to_frame.clone();
    result.used_previous_homography_seed = true;
    result.ok = true;
    attachProjectedBoardMask(&result, reference.reference_bgr.size(), frame_size);
    return result;
}

bool seededFallbackStrongEnoughToSkipDirect(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    const GeometryResult& seeded_geometry,
    int direct_common_tag_count) {
    (void)frame_bgr;
    (void)reference;
    ++g_geometry_decisions.seeded_fast_accept_total;
    if (!seeded_geometry.ok ||
            seeded_geometry.H_ref_to_frame.empty()) {
        ++g_geometry_decisions.seeded_fast_reject_not_ok;
        return false;
    }
    if (!seeded_geometry.used_previous_homography_seed) {
        ++g_geometry_decisions.seeded_fast_reject_not_previous_seed;
        return false;
    }
    if (seeded_geometry.common_tag_count >= 3) {
        ++g_geometry_decisions.seeded_fast_accept_ok;
        ++g_geometry_decisions.seeded_fast_accept_ok_3plus;
        return true;
    }
    if (direct_common_tag_count < 3 && seeded_geometry.soft_tag_count >= 2) {
        ++g_geometry_decisions.seeded_fast_accept_ok;
        return true;
    }
    if (direct_common_tag_count < 2 && seeded_geometry.soft_tag_count >= direct_common_tag_count) {
        ++g_geometry_decisions.seeded_fast_accept_ok;
        return true;
    }
    ++g_geometry_decisions.seeded_fast_reject_soft_count;
    return false;
}

FrameResult chooseBetterGeometryResult(
    const cv::Mat& frame_bgr,
    const ReferenceModel& reference,
    FrameResult direct,
    FrameResult seeded) {
    ++g_geometry_decisions.chooser_total;
    if (!direct.geometry.ok) {
        ++g_geometry_decisions.chooser_seeded;
        ++g_geometry_decisions.chooser_seeded_direct_bad;
        return seeded;
    }
    if (!seeded.geometry.ok) {
        ++g_geometry_decisions.chooser_direct;
        ++g_geometry_decisions.chooser_direct_seeded_bad;
        return direct;
    }
    if (direct.geometry.common_tag_count < 3 &&
            direct.geometry.soft_tag_count < 2 &&
            seeded.geometry.used_previous_homography_seed) {
        ++g_geometry_decisions.chooser_seeded;
        ++g_geometry_decisions.chooser_seeded_weak_direct;
        return seeded;
    }
    if (direct.geometry.common_tag_count < 2 &&
            seeded.geometry.used_previous_homography_seed &&
            seeded.geometry.soft_tag_count >= direct.geometry.soft_tag_count) {
        ++g_geometry_decisions.chooser_seeded;
        ++g_geometry_decisions.chooser_seeded_soft_count;
        return seeded;
    }
    if (direct.geometry.common_tag_count < 3 &&
            geometryReferenceXSpread(reference, direct.geometry) < 0.22 &&
            seeded.geometry.used_previous_homography_seed) {
        ++g_geometry_decisions.chooser_seeded;
        ++g_geometry_decisions.chooser_seeded_spread;
        return seeded;
    }

    FrameMaskCache frame_masks;
    double direct_score = geometryChoiceScore(frame_bgr, reference, direct.geometry, &frame_masks);
    double seeded_score = geometryChoiceScore(frame_bgr, reference, seeded.geometry, &frame_masks);
    if (seeded_score + 1.0 < direct_score) {
        ++g_geometry_decisions.chooser_seeded;
        ++g_geometry_decisions.chooser_seeded_score;
        return seeded;
    }
    ++g_geometry_decisions.chooser_direct;
    return direct;
}

bool readFrameAt(cv::VideoCapture* cap, int frame_index, cv::Mat* frame) {
    if (cap == nullptr || frame == nullptr || frame_index < 0) {
        return false;
    }
    cap->set(cv::CAP_PROP_POS_FRAMES, frame_index);
    return cap->read(*frame) && !frame->empty();
}

struct DiagnosticPropagationCache {
    struct PropagatedSeed {
        int anchor_frame = -1;
        cv::Mat H;
    };

    cv::VideoCapture* cap = nullptr;
    const ReferenceModel* reference = nullptr;
    std::map<int, cv::Mat> frames;
    std::map<int, GeometryResult> direct_geometries;
    std::map<int, PropagatedSeed> propagated_after_frame;

    DiagnosticPropagationCache(cv::VideoCapture* cap_in, const ReferenceModel* reference_in)
        : cap(cap_in), reference(reference_in) {}

    bool getFrame(int frame_index, cv::Mat* frame) {
        if (frame == nullptr || frame_index < 0) {
            return false;
        }
        auto it = frames.find(frame_index);
        if (it != frames.end()) {
            *frame = it->second;
            return !frame->empty();
        }
        cv::Mat loaded;
        if (!readFrameAt(cap, frame_index, &loaded)) {
            return false;
        }
        auto inserted = frames.emplace(frame_index, loaded.clone());
        *frame = inserted.first->second;
        return true;
    }

    GeometryResult directGeometry(int frame_index) {
        auto it = direct_geometries.find(frame_index);
        if (it != direct_geometries.end()) {
            return it->second;
        }
        cv::Mat frame;
        GeometryResult geometry;
        geometry.frame_index = frame_index;
        if (reference != nullptr && getFrame(frame_index, &frame)) {
            geometry = estimateTagSheetGeometry(frame_index, frame, *reference);
        }
        direct_geometries.emplace(frame_index, geometry);
        return geometry;
    }

    cv::Mat propagatedSeedBefore(int target_frame) {
        if (reference == nullptr || target_frame <= 0) {
            return cv::Mat();
        }

        int anchor_frame = -1;
        cv::Mat seed_H;
        for (int frame_index = target_frame - 1; frame_index >= 0; --frame_index) {
            GeometryResult geometry = directGeometry(frame_index);
            if (geometry.ok && geometry.common_tag_count >= 2) {
                anchor_frame = frame_index;
                seed_H = geometry.H_ref_to_frame.clone();
                break;
            }
        }
        if (anchor_frame < 0 || seed_H.empty()) {
            return cv::Mat();
        }

        int start_frame = anchor_frame + 1;
        auto anchor_it = propagated_after_frame.find(anchor_frame);
        if (anchor_it == propagated_after_frame.end()) {
            propagated_after_frame[anchor_frame] = {anchor_frame, seed_H.clone()};
        }

        for (int frame_index = target_frame - 1; frame_index >= anchor_frame; --frame_index) {
            auto it = propagated_after_frame.find(frame_index);
            if (it != propagated_after_frame.end() &&
                    it->second.anchor_frame == anchor_frame &&
                    !it->second.H.empty()) {
                seed_H = it->second.H.clone();
                start_frame = frame_index + 1;
                break;
            }
        }

        // Keep the same propagation semantics as propagatePreviousHomographySeed:
        // every intermediate frame is estimated from the current seed, and only
        // successful intermediate homographies replace that seed.
        for (int frame_index = start_frame; frame_index < target_frame; ++frame_index) {
            cv::Mat frame;
            if (!getFrame(frame_index, &frame)) {
                continue;
            }
            const DetectionResult* precomputed_frame_tags = nullptr;
            auto direct_it = direct_geometries.find(frame_index);
            if (direct_it != direct_geometries.end()) {
                precomputed_frame_tags = &direct_it->second.frame_tags;
            }
            GeometryResult geometry = estimateTagSheetGeometry(
                frame_index,
                frame,
                *reference,
                nullptr,
                seed_H,
                precomputed_frame_tags);
            if (geometry.ok) {
                seed_H = geometry.H_ref_to_frame.clone();
            }
            propagated_after_frame[frame_index] = {anchor_frame, seed_H.clone()};
        }
        return seed_H;
    }
};

cv::Mat propagatePreviousHomographySeed(
    cv::VideoCapture* cap,
    int target_frame,
    const ReferenceModel& reference) {
    if (cap == nullptr || target_frame <= 0) {
        return cv::Mat();
    }

    int seed_frame = -1;
    cv::Mat seed_H;
    for (int frame_index = target_frame - 1; frame_index >= 0; --frame_index) {
        cv::Mat frame;
        if (!readFrameAt(cap, frame_index, &frame)) {
            continue;
        }
        GeometryResult geometry = estimateTagSheetGeometry(frame_index, frame, reference);
        if (geometry.ok && geometry.common_tag_count >= 2) {
            seed_frame = frame_index;
            seed_H = geometry.H_ref_to_frame.clone();
            break;
        }
    }
    if (seed_frame < 0 || seed_H.empty()) {
        return cv::Mat();
    }

    // Bring the seed forward one frame at a time. Each successful intermediate
    // geometry, including soft-only geometry, replaces the seed used by the next
    // frame. This avoids jumping a stale homography directly across an occlusion.
    for (int frame_index = seed_frame + 1; frame_index < target_frame; ++frame_index) {
        cv::Mat frame;
        if (!readFrameAt(cap, frame_index, &frame)) {
            continue;
        }
        GeometryResult geometry = estimateTagSheetGeometry(frame_index, frame, reference, nullptr, seed_H);
        if (geometry.ok) {
            seed_H = geometry.H_ref_to_frame.clone();
        }
    }
    return seed_H;
}

int runVideoV3(const Options& options) {
    configureOpenCvRuntime();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    fs::create_directories(options.outdir);
    g_geometry_timing = GeometryTimingStats();
    g_geometry_decisions = GeometryDecisionStats();
    g_render_timing = RenderTimingStats();
    g_geometry_timing_enabled = options.enable_timing;

    ReferenceModel reference = loadReferenceModel(options);
    cv::Mat surface_texture_ref_bgr;
    std::vector<SurfaceTextureEntry> surface_texture_entries;
    if (!options.surface_texture_path.empty()) {
        cv::Mat texture_bgr = cv::imread(options.surface_texture_path.string(), cv::IMREAD_COLOR);
        if (texture_bgr.empty()) {
            throw std::runtime_error("Could not read surface texture image: " + options.surface_texture_path.string());
        }
        surface_texture_ref_bgr = fitTextureToReference(texture_bgr, reference.reference_bgr.size());
        std::cout << "[INFO] Surface texture: " << options.surface_texture_path << "\n";
    } else if (!options.surface_texture_dir.empty()) {
        for (const fs::path& texture_path : listTexturePaths(options.surface_texture_dir)) {
            cv::Mat texture_bgr = cv::imread(texture_path.string(), cv::IMREAD_COLOR);
            if (texture_bgr.empty()) {
                throw std::runtime_error("Could not read surface texture image: " + texture_path.string());
            }
            surface_texture_entries.push_back(SurfaceTextureEntry{
                texture_path,
                fitTextureToReference(texture_bgr, reference.reference_bgr.size())
            });
        }
        if (surface_texture_entries.empty()) {
            throw std::runtime_error("No texture images found in: " + options.surface_texture_dir.string());
        }
        std::cout << "[INFO] Surface textures: " << surface_texture_entries.size()
                  << " from " << options.surface_texture_dir << "\n";
    }
    H5CornerTrace original_h5_corners = loadH5CornerTraceForVideo(options.video_path, false);
    if (options.surface_tracking_only && !original_h5_corners.available) {
        throw std::runtime_error(
            "Surface-tracking-only mode requires an original H5 sidecar with usable corners: " +
            h5SidecarPathForVideo(options.video_path).string());
    }
    H5CornerTrace fallback_h5_corners = loadH5CornerTraceForVideo(options.video_path, true);
    if (options.corrected_sidecar_geometry_only &&
            (!fallback_h5_corners.available || !fallback_h5_corners.corrected)) {
        throw std::runtime_error(
            "Corrected-sidecar geometry mode requires a usable verified sidecar: " +
            correctedH5SidecarPathForVideo(options.video_path).string());
    }
    const H5CornerTrace* h5_geometry_fallback_ptr =
        (options.h5_geometry_fallback && fallback_h5_corners.available) ? &fallback_h5_corners : nullptr;
    const H5CornerTrace* h5_stylus_ptr = fallback_h5_corners.stylus_available
        ? &fallback_h5_corners
        : (original_h5_corners.stylus_available ? &original_h5_corners : nullptr);
    const H5CornerTrace* h5_draw_ptr = fallback_h5_corners.available
        ? &fallback_h5_corners
        : (original_h5_corners.available ? &original_h5_corners : nullptr);
    std::vector<std::array<cv::Point2f, 4>> corrected_h5_corners = original_h5_corners.frame_corners;
    std::vector<unsigned char> corrected_h5_updated(corrected_h5_corners.size(), 0);
    int corrected_h5_update_count = 0;

    cv::VideoCapture cap(options.video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video: " + options.video_path);
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    auto updateCorrectedH5Corners = [&](int frame_index, const GeometryResult& geometry, cv::Size frame_size) -> bool {
        if (geometry.used_corrected_h5_corners ||
                !original_h5_corners.available ||
                frame_index < 0 ||
                static_cast<size_t>(frame_index) >= corrected_h5_corners.size()) {
            return false;
        }
        std::array<cv::Point2f, 4> corrected;
        if (!correctedH5CornersFromGeometry(
                geometry,
                reference,
                frame_size,
                original_h5_corners.source_size,
                &corrected)) {
            return false;
        }
        const size_t index = static_cast<size_t>(frame_index);
        corrected_h5_corners[index] = corrected;
        if (corrected_h5_updated[index] == 0) {
            corrected_h5_updated[index] = 1;
            ++corrected_h5_update_count;
        }
        return true;
    };

    auto writeCorrectedH5IfNeeded = [&]() {
        if (options.frames.empty() && corrected_h5_update_count > 0) {
            writeCorrectedH5Sidecar(options.video_path, corrected_h5_corners, corrected_h5_update_count);
        }
    };
    auto estimateGeometryForFrame = [&](int frame_index,
                                        const cv::Mat& frame,
                                        const cv::Mat& seed_H = cv::Mat(),
                                        const DetectionResult* precomputed_frame_tags = nullptr) {
        if (options.corrected_sidecar_geometry_only) {
            return geometryFromH5Corners(
                frame_index,
                reference,
                &fallback_h5_corners,
                frame.size());
        }
        return estimateTagSheetGeometry(
            frame_index,
            frame,
            reference,
            &options,
            seed_H,
            precomputed_frame_tags);
    };
    auto applyH5GeometryFallbackIfNeeded = [&](FrameResult* frame_result, const cv::Mat& frame) {
        if (frame_result == nullptr ||
                frame_result->geometry.ok ||
                h5_geometry_fallback_ptr == nullptr) {
            return;
        }
        GeometryResult h5_geometry = geometryFromH5Corners(
            frame_result->frame_index,
            reference,
            h5_geometry_fallback_ptr,
            frame.size());
        if (h5_geometry.ok) {
            frame_result->geometry = std::move(h5_geometry);
        }
    };

    std::vector<int> target_frames = options.frames;
    std::sort(target_frames.begin(), target_frames.end());
    target_frames.erase(std::unique(target_frames.begin(), target_frames.end()), target_frames.end());
    const bool use_raw_depth_mask_stream =
        ((options.write_colormatch && hasSurfaceTextureOutput(options)) ||
         options.write_hand_stylus_mask_video) &&
        !options.write_raw_depth;
    if (!use_raw_depth_mask_stream) {
        runRawDepthOutput(options, target_frames);
    }
    TimingStats timing;

    if (!target_frames.empty()) {
        RawDepthMaskStream diagnostic_depth_mask_stream;
        if (use_raw_depth_mask_stream) {
            diagnostic_depth_mask_stream.open(options, cv::Size(frame_width, frame_height), target_frames);
        }
        int processed_count = 0;
        RenderScratch render_scratch;
        DiagnosticPropagationCache diagnostic_cache(&cap, &reference);
        StylusWedgeTemporalState stylus_wedge_temporal;
        SheetGuidedDepthThresholdState sheet_guided_threshold_temporal;
        for (int target : target_frames) {
            cv::Mat frame;
            if (!diagnostic_cache.getFrame(target, &frame)) {
                continue;
            }
            const std::string prefix = framePrefix(target);
            if (!options.fast) {
                writePngFast(
                    options.outdir / (prefix + "_reference_model.png"),
                    reference.reference_bgr);
            }
            FrameResult selected;
            selected.frame_index = target;
            auto stage_start = std::chrono::steady_clock::now();
            selected.geometry = estimateGeometryForFrame(target, frame);
            diagnostic_cache.direct_geometries[target] = selected.geometry;
            if (options.enable_timing) {
                TimingStats::add(&timing.direct_geometry, TimingStats::elapsedMs(stage_start));
            }
            selected.ok = true;
            if (!selected.geometry.used_corrected_h5_corners &&
                    (!selected.geometry.ok || selected.geometry.common_tag_count < 3)) {
                stage_start = std::chrono::steady_clock::now();
                cv::Mat propagated_seed = diagnostic_cache.propagatedSeedBefore(target);
                if (options.enable_timing) {
                    TimingStats::add(&timing.propagation, TimingStats::elapsedMs(stage_start));
                }
                if (!propagated_seed.empty()) {
                    FrameResult propagated;
                    propagated.frame_index = target;
                    stage_start = std::chrono::steady_clock::now();
                    propagated.geometry = estimateGeometryForFrame(
                        target,
                        frame,
                        propagated_seed,
                        &selected.geometry.frame_tags);
                    if (options.enable_timing) {
                        TimingStats::add(&timing.seeded_geometry, TimingStats::elapsedMs(stage_start));
                    }
                    propagated.ok = true;
                    stage_start = std::chrono::steady_clock::now();
                    selected = chooseBetterGeometryResult(
                        frame,
                        reference,
                        std::move(selected),
                        std::move(propagated));
                    if (options.enable_timing) {
                        TimingStats::add(&timing.choose_geometry, TimingStats::elapsedMs(stage_start));
                    }
                }
            }
            applyH5GeometryFallbackIfNeeded(&selected, frame);
            FrameResult result;
            result.frame_index = target;
            if (options.write_overlay) {
                auto render_start = std::chrono::steady_clock::now();
                result = renderTagSheetOverlayWithGeometry(
                    target,
                    frame,
                    reference,
                    options,
                    std::move(selected.geometry),
                    &render_scratch,
                    h5_draw_ptr);
                if (options.enable_timing) {
                    const double render_ms = TimingStats::elapsedMs(render_start);
                    TimingStats::add(&timing.render, render_ms);
                    TimingStats::add(&timing.overlay_render, render_ms);
                }
                if (!result.ok) {
                    continue;
                }
                fs::path output = options.outdir / (framePrefix(target) + "_v3_tag_sheet_overlay.png");
                auto write_start = std::chrono::steady_clock::now();
                writePngFast(output, result.output_bgr);
                if (options.enable_timing) {
                    const double write_ms = TimingStats::elapsedMs(write_start);
                    TimingStats::add(&timing.write, write_ms);
                    TimingStats::add(&timing.png_write, write_ms);
                }
            } else {
                result.geometry = std::move(selected.geometry);
                result.ok = true;
            }
            updateCorrectedH5Corners(target, result.geometry, frame.size());
            if (options.write_board_mask) {
                fs::path output = options.outdir / (framePrefix(target) + "_v3_board_mask.png");
                auto write_start = std::chrono::steady_clock::now();
                writeBoardMaskPng(output, result.geometry.board_mask);
                if (options.enable_timing) {
                    const double write_ms = TimingStats::elapsedMs(write_start);
                    TimingStats::add(&timing.write, write_ms);
                    TimingStats::add(&timing.png_write, write_ms);
                }
            }
            if (options.write_colormatch || options.write_hand_stylus_mask_video) {
                auto artifact_start = std::chrono::steady_clock::now();
                cv::Mat depth_hand_mask;
                cv::Mat depth_difference;
                {
                    ScopedStageTimer timer(&g_render_timing.depth_mask_read);
                    if (use_raw_depth_mask_stream) {
                        depth_hand_mask = diagnostic_depth_mask_stream.readMask(frame.size());
                        depth_difference = diagnostic_depth_mask_stream.lastDifference();
                    } else {
                        depth_hand_mask = readRawDepthGradientMaskPng(options.outdir, framePrefix(target), frame.size());
                        depth_difference = readRawDepthGradientDifferencePng(options.outdir, framePrefix(target), frame.size());
                    }
                }
                cv::Point stylus_wedge_debug_target(-1, -1);
                cv::Mat stylus_tip_wedge_debug;
                cv::Mat stylus_sobel_candidates_debug;
                cv::Mat stylus_sobel_raw_debug;
                std::string stylus_wedge_status;
                SheetGuidedDepthThresholdResult threshold_result = selectSheetGuidedDepthCandidate(
                    depth_hand_mask,
                    depth_difference,
                    frame,
                    target,
                    h5_stylus_ptr,
                    reference,
                    result.geometry,
                    options,
                    &sheet_guided_threshold_temporal);
                depth_hand_mask = threshold_result.candidate;
                cv::Mat depth_candidate_mask = depth_hand_mask.clone();
                {
                    std::ostringstream threshold_message;
                    threshold_message << std::fixed << std::setprecision(2)
                        << "[DEPTH_THRESHOLD] frame=" << target
                        << " base=" << threshold_result.base_threshold
                        << " selected=" << threshold_result.selected_threshold
                        << " mismatch=" << threshold_result.mismatch_percent << "%"
                        << " retention=" << threshold_result.retention_percent << "%"
                        << " sheet_coverage=" << threshold_result.visible_sheet_coverage_percent << "%"
                        << " candidate_pixels=" << threshold_result.candidate_pixels
                        << " removed_sheet_pixels=" << threshold_result.removed_sheet_pixels
                        << " status=" << threshold_result.status;
                    std::cout << threshold_message.str() << "\n";
                }
                const bool collect_stylus_debug = !options.fast;
                {
                    ScopedStageTimer timer(&g_render_timing.stylus_mask_refinement);
                    depth_hand_mask = connectV3StylusTipToMask(
                        depth_hand_mask,
                        depth_difference,
                        frame,
                        target,
                        h5_stylus_ptr,
                        reference,
                        result.geometry,
                        frame.size(),
                        collect_stylus_debug ? &stylus_wedge_debug_target : nullptr,
                        collect_stylus_debug ? &stylus_tip_wedge_debug : nullptr,
                        collect_stylus_debug ? &stylus_sobel_candidates_debug : nullptr,
                        collect_stylus_debug ? &stylus_sobel_raw_debug : nullptr,
                        collect_stylus_debug ? &stylus_wedge_status : nullptr,
                        &stylus_wedge_temporal);
                }
                ColormatchArtifacts artifacts = makeColormatchArtifacts(
                    frame,
                    reference,
                    result.geometry,
                    depth_hand_mask,
                    surface_texture_ref_bgr,
                    options,
                    stylus_wedge_debug_target,
                    stylus_tip_wedge_debug,
                    stylus_sobel_candidates_debug,
                    stylus_sobel_raw_debug);
                if (options.enable_timing) {
                    const double artifact_ms = TimingStats::elapsedMs(artifact_start);
                    TimingStats::add(&timing.render, artifact_ms);
                    TimingStats::add(&timing.colormatch_artifacts, artifact_ms);
                }
                auto write_start = std::chrono::steady_clock::now();
                if (surface_texture_entries.empty()) {
                    writeColormatchArtifacts(
                        options.outdir,
                        prefix,
                        artifacts,
                        options.final_overlay_only,
                        options.write_foreground_keep_mask_debug);
                } else {
                    writeColormatchArtifacts(
                        options.outdir,
                        prefix,
                        artifacts,
                        options.final_overlay_only,
                        options.write_foreground_keep_mask_debug);
                    auto texture_start = std::chrono::steady_clock::now();
                    for (const SurfaceTextureEntry& entry : surface_texture_entries) {
                        cv::Mat texture_surface;
                        cv::Mat texture_final = makeTextureFinalOverlay(
                            frame,
                            result.geometry,
                            artifacts,
                            entry.texture_ref_bgr,
                            options.shadow_strength,
                            &texture_surface);
                        const std::string texture_prefix = prefix + "_" + entry.path.stem().string();
                        if (!options.final_overlay_only && !texture_surface.empty()) {
                            writePngFast(options.outdir / (texture_prefix + "_surface_texture.png"), texture_surface);
                        }
                        if (!texture_final.empty()) {
                            writePngFast(options.outdir / (texture_prefix + "_final_overlay.png"), texture_final);
                        }
                    }
                    if (options.enable_timing) {
                        TimingStats::add(&timing.texture_variants, TimingStats::elapsedMs(texture_start));
                    }
                }
                if (options.write_hand_stylus_mask_stage_videos) {
                    writeHandMaskStagePngs(
                        options.outdir,
                        prefix,
                        depth_difference,
                        depth_candidate_mask,
                        depth_hand_mask,
                        artifacts);
                }
                if (options.enable_timing) {
                    const double write_ms = TimingStats::elapsedMs(write_start);
                    TimingStats::add(&timing.write, write_ms);
                    TimingStats::add(&timing.png_write, write_ms);
                }
            }
            if (!options.fast) {
                drawTagDiagnostics(frame, reference, result.geometry, options.outdir);
            }
            printTagDiagnostics(result.geometry, reference, frame);
            std::cout << "[INFO] " << prefix
                      << " common_tags=" << result.geometry.common_tag_count
                      << " soft_tags=" << result.geometry.soft_tag_count
                      << " fallback=" << (result.geometry.used_single_tag_fallback ? "yes" : "no")
                      << " prev_seed=" << (result.geometry.used_previous_homography_seed ? "yes" : "no")
                      << " h5=" << (result.geometry.used_corrected_h5_corners ? "yes" : "no")
                      << " homography=" << (result.geometry.ok ? "ok" : "passthrough") << "\n";
            ++processed_count;
        }
        std::cout << "[DONE] v3 diagnostic frames processed: " << processed_count << "\n";
        if (use_raw_depth_mask_stream) {
            diagnostic_depth_mask_stream.close();
        }
        writeCorrectedH5IfNeeded();
        if (options.enable_timing) {
            timing.print("v3 diagnostic frames");
        }
        return 0;
    }

    FinalOverlayVideoWriter writer;
    if (options.write_overlay) {
        const std::string stem = fs::path(options.video_path).stem().string();
        writer.open(
            pipelineVideoOutputPath(options.outdir, stem, "_v3_tag_sheet_overlay", options),
            frame_width,
            frame_height,
            fps,
            pipelineVideoCodec(options));
    }
    FinalOverlayVideoWriter board_mask_writer;
    if (options.write_board_mask) {
        const std::string stem = fs::path(options.video_path).stem().string();
        board_mask_writer.open(
            pipelineVideoOutputPath(options.outdir, stem, "_v3_board_mask", options),
            frame_width,
            frame_height,
            fps,
            pipelineVideoCodec(options));
    }
    FinalOverlayVideoWriter raw_frame_writer;
    FinalOverlayVideoWriter aligned_ref_colormatch_writer;
    FinalOverlayVideoWriter surface_texture_writer;
    FinalOverlayVideoWriter final_overlay_writer;
    FinalOverlayVideoWriter preview_final_overlay_writer;
    std::vector<SurfaceTextureVideoOutput> texture_video_outputs;
    FinalOverlayVideoWriter hand_stylus_mask_writer;
    FinalOverlayVideoWriter hand_stage_depth_difference_writer;
    FinalOverlayVideoWriter hand_stage_depth_candidate_writer;
    FinalOverlayVideoWriter hand_stage_stylus_connected_writer;
    FinalOverlayVideoWriter hand_stage_board_clipped_writer;
    FinalOverlayVideoWriter hand_stage_colormatch_fallback_writer;
    FinalOverlayVideoWriter hand_stage_visible_sheet_excluded_writer;
    FinalOverlayVideoWriter hand_stage_holes_filled_writer;
    FinalOverlayVideoWriter hand_stage_yellow_stylus_added_writer;
    if (options.write_colormatch || options.write_hand_stylus_mask_video) {
        const std::string stem = fs::path(options.video_path).stem().string();
        if (options.write_colormatch && !options.final_overlay_only) {
            raw_frame_writer.open(
                pipelineVideoOutputPath(options.outdir, stem, "_raw_frame", options),
                frame_width,
                frame_height,
                fps,
                pipelineVideoCodec(options));
            aligned_ref_colormatch_writer.open(
                pipelineVideoOutputPath(options.outdir, stem, "_aligned_ref_colormatch", options),
                frame_width,
                frame_height,
                fps,
                pipelineVideoCodec(options));
        }
        if (options.write_hand_stylus_mask_video) {
            hand_stylus_mask_writer.open(
                pipelineVideoOutputPath(
                    options.outdir,
                    stem,
                    "_aligned_ref_colormatch_hand_stylus_mask",
                    options),
                frame_width,
                frame_height,
                fps,
                pipelineVideoCodec(options));
        }
        if (options.write_hand_stylus_mask_stage_videos) {
            auto open_stage = [&](FinalOverlayVideoWriter* stage_writer, const std::string& suffix) {
                stage_writer->open(
                    pipelineVideoOutputPath(options.outdir, stem, suffix, options),
                    frame_width, frame_height, fps, pipelineVideoCodec(options));
            };
            open_stage(&hand_stage_depth_difference_writer, "_hand_mask_stage_00_depth_difference");
            open_stage(&hand_stage_depth_candidate_writer, "_hand_mask_stage_01_depth_candidate");
            open_stage(&hand_stage_stylus_connected_writer, "_hand_mask_stage_02_stylus_connected");
            open_stage(&hand_stage_board_clipped_writer, "_hand_mask_stage_03_board_clipped");
            open_stage(&hand_stage_colormatch_fallback_writer, "_hand_mask_stage_04_colormatch_fallback");
            open_stage(&hand_stage_visible_sheet_excluded_writer, "_hand_mask_stage_05_visible_sheet_excluded");
            open_stage(&hand_stage_holes_filled_writer, "_hand_mask_stage_06_holes_filled");
            open_stage(&hand_stage_yellow_stylus_added_writer, "_hand_mask_stage_07_yellow_stylus_added");
        }
        if (!surface_texture_ref_bgr.empty() && !options.final_overlay_only) {
            surface_texture_writer.open(
                pipelineVideoOutputPath(options.outdir, stem, "_surface_texture", options),
                frame_width,
                frame_height,
                fps,
                pipelineVideoCodec(options));
        }
        if (!surface_texture_ref_bgr.empty()) {
            final_overlay_writer.open(
                finalOverlayVideoOutputPath(options.outdir, stem, "_final_overlay", options),
                frame_width,
                frame_height,
                fps,
                finalOverlayVideoCodec(options));
            if (options.write_preview_mp4) {
                preview_final_overlay_writer.open(
                    previewVideoPath(options.outdir, stem, "_final_overlay"),
                    frame_width,
                    frame_height,
                    fps,
                    "preview-mp4");
            }
        }
        if (!surface_texture_entries.empty()) {
            texture_video_outputs.reserve(surface_texture_entries.size());
            for (const SurfaceTextureEntry& entry : surface_texture_entries) {
                texture_video_outputs.emplace_back();
                SurfaceTextureVideoOutput& output = texture_video_outputs.back();
                output.texture_path = entry.path;
                output.output_path = finalOverlayVideoOutputPath(
                    options.outdir,
                    stem,
                    "_" + entry.path.stem().string(),
                    options);
                output.texture_ref_bgr = entry.texture_ref_bgr;
                output.final_overlay_writer.open(
                    output.output_path,
                    frame_width,
                    frame_height,
                    fps,
                    finalOverlayVideoCodec(options));
                if (options.write_preview_mp4) {
                    output.preview_writer.open(
                        previewVideoPath(options.outdir, stem, "_" + entry.path.stem().string()),
                        frame_width,
                        frame_height,
                        fps,
                        "preview-mp4");
                }
            }
            std::cout << "[INFO] Writing " << texture_video_outputs.size()
                      << " texture overlay videos\n";
            for (const SurfaceTextureVideoOutput& output : texture_video_outputs) {
                std::cout << "[INFO] Texture output: "
                          << output.texture_path.filename() << " -> "
                          << output.output_path << "\n";
            }
        }
    }

    cv::VideoCapture depth_mask_cap;
    const fs::path depth_mask_frame_dir = options.outdir / "_raw_depth_gradient_mask_frames";
    const bool use_depth_mask_frame_dir = fs::exists(depth_mask_frame_dir);
    RawDepthMaskStream depth_mask_stream;
    if (use_raw_depth_mask_stream) {
        depth_mask_stream.open(options, cv::Size(frame_width, frame_height));
    }
    if ((options.write_colormatch || options.write_hand_stylus_mask_video) &&
            !use_raw_depth_mask_stream &&
            !use_depth_mask_frame_dir) {
        const std::string stem = fs::path(options.video_path).stem().string();
        fs::path mask_video_path = options.outdir / (stem + "_raw_depth_gradient_mask.mp4");
        if (!fs::exists(mask_video_path)) {
            mask_video_path = options.outdir / (stem + "_raw_depth_gradient_mask.avi");
        }
        depth_mask_cap.open(mask_video_path.string());
        if (!depth_mask_cap.isOpened()) {
            std::cerr << "[WARN] Could not open raw depth gradient mask video: "
                      << mask_video_path << "\n";
        }
    }

    int written_count = 0;
    cv::Mat last_successful_H;
    RenderScratch render_scratch;
    StylusWedgeTemporalState stylus_wedge_temporal;
    SheetGuidedDepthThresholdState sheet_guided_threshold_temporal;
    bool have_previous_tracked_corners = false;
    std::array<cv::Point2f, 4> previous_tracked_corners{};
    int previous_tracked_frame = -1;
    int unchanged_corner_run_frames = 0;
    int stuck_corner_warning_count = 0;
    for (int frame_index = 0; ; ++frame_index) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        FrameResult selected;
        selected.frame_index = frame_index;
        bool selected_ready = false;
        bool seeded_choice_already_handled = false;
        if (options.corrected_sidecar_geometry_only) {
            selected.geometry = geometryFromH5Corners(
                frame_index,
                reference,
                &fallback_h5_corners,
                frame.size());
            selected.ok = true;
            selected_ready = true;
            seeded_choice_already_handled = true;
        }
        if (!selected_ready && options.prefer_seeded_fallback && !last_successful_H.empty()) {
            auto prefilter_start = std::chrono::steady_clock::now();
            DetectionResult precomputed_frame_tags;
            {
                ScopedStageTimer timer(&g_geometry_timing.april_tag_detection);
                precomputed_frame_tags = detectAprilTags(frame);
            }
            int hard_common_tags = countCommonReferenceTags(reference, precomputed_frame_tags);
            if (options.enable_timing) {
                TimingStats::add(&timing.seeded_prefilter, TimingStats::elapsedMs(prefilter_start));
            }

            if (hard_common_tags < 3) {
                seeded_choice_already_handled = true;
                if (options.fast) {
                    selected.geometry = geometryFromPreviousSeedIfConsistent(
                        frame_index,
                        reference,
                        precomputed_frame_tags,
                        last_successful_H,
                        frame.size());
                    if (selected.geometry.ok) {
                        selected.ok = true;
                        selected_ready = true;
                        ++g_geometry_decisions.seeded_fast_accept_total;
                        ++g_geometry_decisions.seeded_fast_accept_ok;
                    }
                }
            }

            if (!selected_ready && hard_common_tags < 3) {
                FrameResult seeded;
                seeded.frame_index = frame_index;
                auto stage_start = std::chrono::steady_clock::now();
                seeded.geometry = estimateGeometryForFrame(
                    frame_index,
                    frame,
                    last_successful_H,
                    &precomputed_frame_tags);
                double seeded_ms = TimingStats::elapsedMs(stage_start);
                if (options.enable_timing) {
                    TimingStats::add(&timing.seeded_geometry, seeded_ms);
                    TimingStats::add(&timing.seeded_fallback_geometry, seeded_ms);
                }
                seeded.ok = true;

                stage_start = std::chrono::steady_clock::now();
                bool accept_seeded = seededFallbackStrongEnoughToSkipDirect(
                    frame,
                    reference,
                    seeded.geometry,
                    hard_common_tags);
                if (options.enable_timing) {
                    TimingStats::add(&timing.seeded_fast_accept_check, TimingStats::elapsedMs(stage_start));
                }

                if (accept_seeded) {
                    selected = std::move(seeded);
                    selected_ready = true;
                } else {
                    stage_start = std::chrono::steady_clock::now();
                    selected.geometry = estimateGeometryForFrame(
                        frame_index,
                        frame,
                        cv::Mat(),
                        &precomputed_frame_tags);
                    double direct_ms = TimingStats::elapsedMs(stage_start);
                    if (options.enable_timing) {
                        TimingStats::add(&timing.direct_geometry, direct_ms);
                        TimingStats::add(&timing.direct_fallback_geometry, direct_ms);
                    }
                    selected.ok = true;
                    stage_start = std::chrono::steady_clock::now();
                    selected = chooseBetterGeometryResult(
                        frame,
                        reference,
                        std::move(selected),
                        std::move(seeded));
                    if (options.enable_timing) {
                        TimingStats::add(&timing.choose_geometry, TimingStats::elapsedMs(stage_start));
                    }
                    selected_ready = true;
                }
            } else if (!selected_ready) {
                auto stage_start = std::chrono::steady_clock::now();
                selected.geometry = estimateGeometryForFrame(
                    frame_index,
                    frame,
                    cv::Mat(),
                    &precomputed_frame_tags);
                if (options.enable_timing) {
                    TimingStats::add(&timing.direct_geometry, TimingStats::elapsedMs(stage_start));
                }
                selected.ok = true;
                selected_ready = true;
            }
        }

        if (!selected_ready) {
            auto stage_start = std::chrono::steady_clock::now();
            selected.geometry = estimateGeometryForFrame(frame_index, frame);
            double direct_ms = TimingStats::elapsedMs(stage_start);
            if (options.enable_timing) {
                TimingStats::add(&timing.direct_geometry, direct_ms);
                if (!selected.geometry.ok || selected.geometry.common_tag_count < 3) {
                    TimingStats::add(&timing.direct_fallback_geometry, direct_ms);
                }
            }
            selected.ok = true;
        }
        if (!seeded_choice_already_handled &&
                !selected.geometry.used_corrected_h5_corners &&
                (!selected.geometry.ok || selected.geometry.common_tag_count < 3) &&
                !last_successful_H.empty()) {
            FrameResult seeded;
            seeded.frame_index = frame_index;
            auto stage_start = std::chrono::steady_clock::now();
            seeded.geometry = estimateGeometryForFrame(
                frame_index,
                frame,
                last_successful_H,
                &selected.geometry.frame_tags);
            double seeded_ms = TimingStats::elapsedMs(stage_start);
            if (options.enable_timing) {
                TimingStats::add(&timing.seeded_geometry, seeded_ms);
                TimingStats::add(&timing.seeded_fallback_geometry, seeded_ms);
            }
            seeded.ok = true;
            stage_start = std::chrono::steady_clock::now();
            selected = chooseBetterGeometryResult(
                frame,
                reference,
                std::move(selected),
                std::move(seeded));
            if (options.enable_timing) {
                TimingStats::add(&timing.choose_geometry, TimingStats::elapsedMs(stage_start));
            }
        }
        applyH5GeometryFallbackIfNeeded(&selected, frame);
        if (selected.geometry.ok && !selected.geometry.used_corrected_h5_corners) {
            last_successful_H = selected.geometry.H_ref_to_frame.clone();
        }
        const bool corrected_corners_updated =
            updateCorrectedH5Corners(frame_index, selected.geometry, frame.size());
        if (options.surface_tracking_only) {
            if (corrected_corners_updated) {
                const auto& current = corrected_h5_corners[static_cast<size_t>(frame_index)];
                double max_move = 0.0;
                if (have_previous_tracked_corners && previous_tracked_frame + 1 == frame_index) {
                    for (size_t corner = 0; corner < current.size(); ++corner) {
                        max_move = std::max(
                            max_move,
                            static_cast<double>(cv::norm(current[corner] - previous_tracked_corners[corner])));
                    }
                    if (max_move <= options.stuck_corner_epsilon) {
                        ++unchanged_corner_run_frames;
                        if (unchanged_corner_run_frames == options.stuck_corner_frame_limit + 1) {
                            const int start_frame = frame_index - unchanged_corner_run_frames + 1;
                            std::cerr << "[WARN] Surface corners appear stuck for video "
                                      << fs::path(options.video_path).filename()
                                      << " frames=" << start_frame << "-" << frame_index
                                      << " max_move_px=" << max_move << "\n";
                            ++stuck_corner_warning_count;
                        }
                    } else {
                        unchanged_corner_run_frames = 1;
                    }
                } else {
                    unchanged_corner_run_frames = 1;
                }
                previous_tracked_corners = current;
                previous_tracked_frame = frame_index;
                have_previous_tracked_corners = true;
            } else {
                have_previous_tracked_corners = false;
                previous_tracked_frame = -1;
                unchanged_corner_run_frames = 0;
            }
        }
        if (options.write_board_mask) {
            cv::Mat mask_bgr(frame_height, frame_width, CV_8UC3, cv::Scalar(0, 0, 0));
            if (!selected.geometry.board_mask.empty()) {
                cv::cvtColor(selected.geometry.board_mask, mask_bgr, cv::COLOR_GRAY2BGR);
            }
            auto write_start = std::chrono::steady_clock::now();
            board_mask_writer.write(mask_bgr);
            if (options.enable_timing) {
                const double write_ms = TimingStats::elapsedMs(write_start);
                TimingStats::add(&timing.write, write_ms);
                TimingStats::add(&timing.video_write, write_ms);
            }
        }
        if (options.write_colormatch || options.write_hand_stylus_mask_video) {
            auto artifact_start = std::chrono::steady_clock::now();
            cv::Mat mask_frame;
            cv::Mat depth_hand_mask;
            cv::Mat depth_difference;
            {
                ScopedStageTimer timer(&g_render_timing.depth_mask_read);
                if (use_raw_depth_mask_stream) {
                    depth_hand_mask = depth_mask_stream.readMask(frame.size());
                    depth_difference = depth_mask_stream.lastDifference();
                } else if (use_depth_mask_frame_dir) {
                    depth_hand_mask = readRawDepthGradientMaskFramePng(
                        depth_mask_frame_dir,
                        frame_index,
                        frame.size());
                    depth_difference = readRawDepthGradientDifferencePng(
                        depth_mask_frame_dir,
                        framePrefix(frame_index),
                        frame.size());
                }
                if (depth_hand_mask.empty() && depth_mask_cap.isOpened() && depth_mask_cap.read(mask_frame)) {
                    depth_hand_mask = maskFromVideoFrame(mask_frame, frame.size());
                }
                if (depth_hand_mask.empty()) {
                    depth_hand_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                }
            }
            cv::Point stylus_wedge_debug_target(-1, -1);
            cv::Mat stylus_tip_wedge_debug;
            cv::Mat stylus_sobel_candidates_debug;
            cv::Mat stylus_sobel_raw_debug;
            std::string stylus_wedge_status;
            SheetGuidedDepthThresholdResult threshold_result = selectSheetGuidedDepthCandidate(
                depth_hand_mask,
                depth_difference,
                frame,
                frame_index,
                h5_stylus_ptr,
                reference,
                selected.geometry,
                options,
                &sheet_guided_threshold_temporal);
            depth_hand_mask = threshold_result.candidate;
            cv::Mat depth_candidate_mask = depth_hand_mask.clone();
            if (options.enable_timing && (written_count == 0 || (written_count + 1) % 25 == 0)) {
                std::ostringstream threshold_message;
                threshold_message << std::fixed << std::setprecision(2)
                    << "[DEPTH_THRESHOLD] frame=" << frame_index
                    << " base=" << threshold_result.base_threshold
                    << " selected=" << threshold_result.selected_threshold
                    << " mismatch=" << threshold_result.mismatch_percent << "%"
                    << " retention=" << threshold_result.retention_percent << "%"
                    << " status=" << threshold_result.status;
                std::cout << threshold_message.str() << "\n";
            }
            const bool collect_stylus_debug = !options.fast;
            {
                ScopedStageTimer timer(&g_render_timing.stylus_mask_refinement);
                depth_hand_mask = connectV3StylusTipToMask(
                    depth_hand_mask,
                    depth_difference,
                    frame,
                    frame_index,
                    h5_stylus_ptr,
                    reference,
                    selected.geometry,
                    frame.size(),
                    collect_stylus_debug ? &stylus_wedge_debug_target : nullptr,
                    collect_stylus_debug ? &stylus_tip_wedge_debug : nullptr,
                    collect_stylus_debug ? &stylus_sobel_candidates_debug : nullptr,
                    collect_stylus_debug ? &stylus_sobel_raw_debug : nullptr,
                    collect_stylus_debug ? &stylus_wedge_status : nullptr,
                    &stylus_wedge_temporal);
            }
            ColormatchArtifacts artifacts = makeColormatchArtifacts(
                frame,
                reference,
                selected.geometry,
                depth_hand_mask,
                surface_texture_ref_bgr,
                options,
                stylus_wedge_debug_target,
                stylus_tip_wedge_debug,
                stylus_sobel_candidates_debug,
                stylus_sobel_raw_debug);
            if (options.enable_timing) {
                const double artifact_ms = TimingStats::elapsedMs(artifact_start);
                TimingStats::add(&timing.render, artifact_ms);
                TimingStats::add(&timing.colormatch_artifacts, artifact_ms);
            }
            cv::Mat aligned_ref_colormatch_bgr(frame_height, frame_width, CV_8UC3, cv::Scalar(0, 0, 0));
            if (artifacts.ok) {
                aligned_ref_colormatch_bgr = artifacts.aligned_ref_colormatch;
            }
            auto write_start = std::chrono::steady_clock::now();
            if (options.write_colormatch && !options.final_overlay_only) {
                raw_frame_writer.write(frame);
                aligned_ref_colormatch_writer.write(aligned_ref_colormatch_bgr);
            }
            if (options.write_hand_stylus_mask_video) {
                cv::Mat hand_stylus_mask = artifacts.ok && !artifacts.hand_stylus_mask.empty()
                    ? artifacts.hand_stylus_mask
                    : cv::Mat::zeros(frame.size(), CV_8UC1);
                hand_stylus_mask_writer.write(hand_stylus_mask);
            }
            if (options.write_hand_stylus_mask_stage_videos) {
                const cv::Mat empty_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                auto stage_or_empty = [&](const cv::Mat& stage) -> const cv::Mat& {
                    return stage.empty() ? empty_mask : stage;
                };
                hand_stage_depth_difference_writer.write(stage_or_empty(depth_difference));
                hand_stage_depth_candidate_writer.write(stage_or_empty(depth_candidate_mask));
                hand_stage_stylus_connected_writer.write(stage_or_empty(depth_hand_mask));
                hand_stage_board_clipped_writer.write(stage_or_empty(artifacts.hand_mask_board_clipped));
                hand_stage_colormatch_fallback_writer.write(stage_or_empty(artifacts.hand_mask_colormatch_fallback));
                hand_stage_visible_sheet_excluded_writer.write(
                    stage_or_empty(artifacts.hand_mask_visible_sheet_excluded));
                hand_stage_holes_filled_writer.write(stage_or_empty(artifacts.hand_mask_holes_filled));
                hand_stage_yellow_stylus_added_writer.write(
                    stage_or_empty(artifacts.hand_mask_yellow_stylus_added));
            }
            if (!surface_texture_ref_bgr.empty()) {
                cv::Mat surface_texture_bgr = artifacts.ok && !artifacts.surface_texture.empty()
                    ? artifacts.surface_texture
                    : frame;
                cv::Mat final_overlay_bgr = artifacts.ok && !artifacts.final_overlay.empty()
                    ? artifacts.final_overlay
                    : frame;
                if (!options.final_overlay_only) {
                    surface_texture_writer.write(surface_texture_bgr);
                }
                final_overlay_writer.write(final_overlay_bgr);
                if (options.write_preview_mp4) {
                    preview_final_overlay_writer.write(final_overlay_bgr);
                }
            }
            auto texture_start = std::chrono::steady_clock::now();
            for (SurfaceTextureVideoOutput& output : texture_video_outputs) {
                cv::Mat final_overlay_bgr = makeTextureFinalOverlay(
                    frame,
                    selected.geometry,
                    artifacts,
                    output.texture_ref_bgr,
                    options.shadow_strength);
                if (final_overlay_bgr.empty()) {
                    final_overlay_bgr = frame;
                }
                output.final_overlay_writer.write(final_overlay_bgr);
                if (options.write_preview_mp4) {
                    output.preview_writer.write(final_overlay_bgr);
                }
            }
            if (options.enable_timing) {
                if (!texture_video_outputs.empty()) {
                    TimingStats::add(&timing.texture_variants, TimingStats::elapsedMs(texture_start));
                }
                const double write_ms = TimingStats::elapsedMs(write_start);
                TimingStats::add(&timing.write, write_ms);
                TimingStats::add(&timing.video_write, write_ms);
            }
        }
        if (options.write_overlay) {
            auto render_start = std::chrono::steady_clock::now();
            FrameResult result = renderTagSheetOverlayWithGeometry(
                frame_index,
                frame,
                reference,
                options,
                std::move(selected.geometry),
                &render_scratch,
                h5_draw_ptr);
            if (options.surface_tracking_only) {
                drawFrameNumberLabel(&result.output_bgr, frame_index);
            }
            if (options.enable_timing) {
                const double render_ms = TimingStats::elapsedMs(render_start);
                TimingStats::add(&timing.render, render_ms);
                TimingStats::add(&timing.overlay_render, render_ms);
            }
            auto write_start = std::chrono::steady_clock::now();
            writer.write(result.output_bgr);
            if (options.enable_timing) {
                const double write_ms = TimingStats::elapsedMs(write_start);
                TimingStats::add(&timing.write, write_ms);
                TimingStats::add(&timing.video_write, write_ms);
            }
        }
        ++written_count;
        if (written_count == 1 || written_count % 25 == 0) {
            if (options.surface_tracking_only) {
                std::cout << "[PROGRESS] surface tracking processed " << written_count << " frames\n" << std::flush;
            } else {
                std::cout << "[PROGRESS] v3 processed " << written_count << " frames\n" << std::flush;
            }
        }
    }

    if (options.write_overlay) {
        writer.release();
    }
    if (options.write_board_mask) {
        board_mask_writer.release();
    }
    if (options.write_colormatch || options.write_hand_stylus_mask_video) {
        if (options.write_colormatch && !options.final_overlay_only) {
            raw_frame_writer.release();
            aligned_ref_colormatch_writer.release();
        }
        if (options.write_hand_stylus_mask_video) {
            hand_stylus_mask_writer.release();
        }
        if (options.write_hand_stylus_mask_stage_videos) {
            hand_stage_depth_difference_writer.release();
            hand_stage_depth_candidate_writer.release();
            hand_stage_stylus_connected_writer.release();
            hand_stage_board_clipped_writer.release();
            hand_stage_colormatch_fallback_writer.release();
            hand_stage_visible_sheet_excluded_writer.release();
            hand_stage_holes_filled_writer.release();
            hand_stage_yellow_stylus_added_writer.release();
        }
        if (!surface_texture_ref_bgr.empty() && !options.final_overlay_only) {
            surface_texture_writer.release();
        }
        if (!surface_texture_ref_bgr.empty()) {
            final_overlay_writer.release();
            if (options.write_preview_mp4) {
                preview_final_overlay_writer.release();
            }
        }
        for (SurfaceTextureVideoOutput& output : texture_video_outputs) {
            output.final_overlay_writer.release();
            if (options.write_preview_mp4) {
                output.preview_writer.release();
            }
        }
    }
    if (use_raw_depth_mask_stream) {
        depth_mask_stream.close();
    }
    writeCorrectedH5IfNeeded();
    if (options.surface_tracking_only) {
        std::cout << "[TRACKING] video=" << fs::path(options.video_path).filename()
                  << " corrected_frames=" << corrected_h5_update_count
                  << " stuck_warnings=" << stuck_corner_warning_count << "\n";
    }
    std::cout << "[DONE] v3 video frames processed: " << written_count << "\n";
    if (options.enable_timing) {
        timing.print("v3 full video");
    }
    return 0;
}

int reportTagFailuresV3(const Options& options) {
    configureOpenCvRuntime();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    g_geometry_timing = GeometryTimingStats();
    g_geometry_decisions = GeometryDecisionStats();
    g_geometry_timing_enabled = options.enable_timing;
    ReferenceModel reference = loadReferenceModel(options);

    cv::VideoCapture cap(options.video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video: " + options.video_path);
    }

    std::ofstream report_file;
    std::ostream* report = &std::cout;
    if (!options.tag_failure_report_path.empty()) {
        fs::path parent = options.tag_failure_report_path.parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent);
        }
        report_file.open(options.tag_failure_report_path);
        if (!report_file.is_open()) {
            throw std::runtime_error("Could not open tag failure report: " + options.tag_failure_report_path.string());
        }
        report = &report_file;
    }

    *report << "frame,hard_common_tags,soft_tags,fallback_used,previous_seed_used,homography_ok,detected_tags\n";

    int scanned_count = 0;
    int report_row_count = 0;
    std::vector<int> target_frames = options.frames;
    std::sort(target_frames.begin(), target_frames.end());
    TimingStats timing;

    auto writeIfFailed = [&](int frame_index, const GeometryResult& geometry) {
        if (!geometry.ok) {
            *report << frame_index << ","
                    << geometry.common_tag_count << ","
                    << geometry.soft_tag_count << ","
                    << (geometry.used_single_tag_fallback ? 1 : 0) << ","
                    << (geometry.used_previous_homography_seed ? 1 : 0) << ","
                    << (geometry.ok ? 1 : 0) << ","
                    << geometry.frame_tags.detections.size() << "\n";
            ++report_row_count;
        }
    };

    auto inspectFrame = [&](int frame_index, const cv::Mat& frame, const cv::Mat& seed_H) {
        GeometryResult geometry = estimateTagSheetGeometry(frame_index, frame, reference, nullptr, seed_H);
        writeIfFailed(frame_index, geometry);
        ++scanned_count;
        return geometry;
    };

    if (!target_frames.empty()) {
        DiagnosticPropagationCache diagnostic_cache(&cap, &reference);
        for (int target : target_frames) {
            cv::Mat frame;
            if (!diagnostic_cache.getFrame(target, &frame)) {
                *report << target << ",unreadable,unreadable,unreadable,unreadable,unreadable,unreadable\n";
                ++report_row_count;
                continue;
            }
            auto stage_start = std::chrono::steady_clock::now();
            GeometryResult geometry = estimateTagSheetGeometry(target, frame, reference);
            if (options.enable_timing) {
                TimingStats::add(&timing.direct_geometry, TimingStats::elapsedMs(stage_start));
            }
            if (!geometry.ok || geometry.common_tag_count < 3) {
                stage_start = std::chrono::steady_clock::now();
                cv::Mat propagated_seed = diagnostic_cache.propagatedSeedBefore(target);
                if (options.enable_timing) {
                    TimingStats::add(&timing.propagation, TimingStats::elapsedMs(stage_start));
                }
                if (!propagated_seed.empty()) {
                    stage_start = std::chrono::steady_clock::now();
                    GeometryResult propagated_geometry = estimateTagSheetGeometry(
                        target,
                        frame,
                        reference,
                        nullptr,
                        propagated_seed,
                        &geometry.frame_tags);
                    if (options.enable_timing) {
                        TimingStats::add(&timing.seeded_geometry, TimingStats::elapsedMs(stage_start));
                    }
                    stage_start = std::chrono::steady_clock::now();
                    if (!geometry.ok ||
                            (geometry.common_tag_count < 3 &&
                             geometry.soft_tag_count < 2 &&
                             propagated_geometry.used_previous_homography_seed) ||
                            (geometry.common_tag_count < 3 &&
                             geometryReferenceXSpread(reference, geometry) < 0.22 &&
                             propagated_geometry.used_previous_homography_seed) ||
                            geometryChoiceScore(frame, reference, propagated_geometry) + 1.0 <
                            geometryChoiceScore(frame, reference, geometry)) {
                        geometry = std::move(propagated_geometry);
                    }
                    if (options.enable_timing) {
                        TimingStats::add(&timing.choose_geometry, TimingStats::elapsedMs(stage_start));
                    }
                }
            }
            writeIfFailed(target, geometry);
            ++scanned_count;
        }
    } else {
        cv::Mat last_successful_H;
        for (int frame_index = 0; ; ++frame_index) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                break;
            }
            GeometryResult geometry = estimateTagSheetGeometry(frame_index, frame, reference);
            if ((!geometry.ok || geometry.common_tag_count < 3) && !last_successful_H.empty()) {
                GeometryResult seeded_geometry = estimateTagSheetGeometry(
                    frame_index,
                    frame,
                    reference,
                    nullptr,
                    last_successful_H,
                    &geometry.frame_tags);
                if (!geometry.ok ||
                        (geometry.common_tag_count < 3 &&
                         geometry.soft_tag_count < 2 &&
                         seeded_geometry.used_previous_homography_seed) ||
                        (geometry.common_tag_count < 3 &&
                         geometryReferenceXSpread(reference, geometry) < 0.22 &&
                         seeded_geometry.used_previous_homography_seed) ||
                        geometryChoiceScore(frame, reference, seeded_geometry) + 1.0 <
                        geometryChoiceScore(frame, reference, geometry)) {
                    geometry = std::move(seeded_geometry);
                }
            }
            writeIfFailed(frame_index, geometry);
            ++scanned_count;
            if (geometry.ok) {
                last_successful_H = geometry.H_ref_to_frame.clone();
            }
            if (scanned_count == 1 || scanned_count % 100 == 0) {
                std::cerr << "[PROGRESS] v3 tag failure report scanned "
                          << scanned_count << " frames\n" << std::flush;
            }
        }
    }

    std::cerr << "[DONE] v3 tag failure report scanned " << scanned_count
              << " frames; rows=" << report_row_count << "\n";
    if (options.enable_timing) {
        timing.print("v3 tag failure report");
    }
    return 0;
}

}  // namespace shadow_v3
