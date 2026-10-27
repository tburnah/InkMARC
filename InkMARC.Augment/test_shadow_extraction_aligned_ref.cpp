#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kSurfaceWidthM = 0.224;
constexpr double kSurfaceHeightM = 0.148;
constexpr double kTagSizeM = 0.03;
constexpr double kTagCoordScale = 2.0 / kTagSizeM;
constexpr double kTagHalfSizeM = kTagSizeM / 2.0;

const std::map<int, cv::Point2f> kTagCenterLocations = {
    {8, {-0.0920f, 0.0540f}},  {9, {-0.0307f, 0.0540f}},
    {10, {0.0307f, 0.0540f}},  {11, {0.0920f, 0.0540f}},
    {4, {-0.0920f, 0.0000f}},  {5, {-0.0307f, 0.0000f}},
    {6, {0.0307f, 0.0000f}},   {7, {0.0920f, 0.0000f}},
    {0, {-0.0920f, -0.0540f}}, {1, {-0.0307f, -0.0540f}},
    {2, {0.0307f, -0.0540f}},  {3, {0.0920f, -0.0540f}},
};

const std::vector<cv::Point2f> kSurfaceCornersM = {
    {-static_cast<float>(kSurfaceWidthM / 2.0), static_cast<float>(kSurfaceHeightM / 2.0)},
    { static_cast<float>(kSurfaceWidthM / 2.0), static_cast<float>(kSurfaceHeightM / 2.0)},
    { static_cast<float>(kSurfaceWidthM / 2.0), -static_cast<float>(kSurfaceHeightM / 2.0)},
    {-static_cast<float>(kSurfaceWidthM / 2.0), -static_cast<float>(kSurfaceHeightM / 2.0)},
};

const std::vector<cv::Point2f> kTagLocalCorners = {
    {-1.0f, 1.0f},
    {1.0f, 1.0f},
    {1.0f, -1.0f},
    {-1.0f, -1.0f},
};

struct Args {
    std::string video;
    std::string reference;
    std::string outdir;
    std::string surface_texture;
    std::string surface_texture_dir;
    std::vector<int> frames;
    int num_samples = 2;
    double reference_scale = 1.0;
    bool write_raw_frame = false;
    bool write_aligned_ref = false;
    bool write_raw_frame_flat = false;
    bool write_aligned_ref_flat = false;
    bool write_combined_quality = false;
    bool write_colormatch = false;
};

struct SheetTrackerConfig {
    int maxFeatures = 1200;
    double qualityLevel = 0.01;
    double minDistance = 8.0;
    int lkWindowSize = 21;
    int lkMaxLevel = 3;
    double maxForwardBackwardError = 2.5;
    double maxLkError = 30.0;
    double ransacReprojThreshold = 3.0;
    int minTrackedPoints = 40;
    int minInliers = 25;
    double maxMedianReprojError = 4.0;
    double maxHomographyJumpPx = 80.0;
    int maxTrackedPoints = 160;
    int templateSearchRadius = 42;
    int predictedSearchRadius = 18;
    int maxHeldHomographyFrames = 40;
    bool directSheetRefineEnabled = true;
    double directSheetScoreDistancePx = 8.0;
    double directSheetMinScore = 0.18;
    double directSheetMinScoreGain = 0.003;
    double directSheetMaxCornerMovePx = 55.0;
    double directSheetMinProjectedCoverage = 0.05;
    bool directSheetRefineTwoTagSeeds = false;
    int tagDetectInterval = 1;
    bool writeDebugImages = true;
};

struct SheetTrackerState {
    cv::Mat H_ref_to_prev;
    cv::Mat H_ref_to_prevprev;
    cv::Mat prevFrameBgr;
    cv::Mat prevTagContrast;
    std::vector<cv::Point2f> referenceFeaturePoints;
    int heldHomographyFrames = 0;
    bool hasValidH = false;
};

struct DetectionResult {
    std::map<int, std::vector<cv::Point2f>> detections;
    std::vector<std::vector<cv::Point2f>> rejected;
};

struct ProfileTotals {
    std::map<std::string, double> ms_by_stage;
    int frames = 0;
};

struct ProcessedFrameArtifacts {
    int frame_index = -1;
    std::string prefix;
    cv::Mat frame_bgr;
    cv::Mat aligned_ref_bgr;
    cv::Mat H_ref_to_frame;
    cv::Mat board_mask;
    cv::Mat hand_stylus_mask;
    bool write_colormatch = false;
    bool write_aligned_ref_masked_difference = false;
    bool write_surface_texture = false;
};

struct FrameRenderData {
    cv::Mat frame_bgr;
    cv::Mat aligned_ref_bgr;
    cv::Mat H_ref_to_frame;
    cv::Mat board_mask;
    cv::Mat hand_stylus_mask;
    cv::Mat aligned_ref_masked_difference_clean;
    cv::Mat shadow_factor_bgr;
    cv::Mat board_render_mask;
    cv::Mat occlusion_render_mask;
};

struct FrameGeometry {
    int index = -1;
    cv::Mat frame_bgr;
    DetectionResult frame_tags;
    cv::Mat H_ref_to_frame;
    cv::Mat board_mask;
    bool ok = false;
};

struct TextureEntry {
    fs::path path;
    cv::Mat texture_ref_bgr;
};

struct TextureVideoWriter {
    fs::path texture_path;
    fs::path output_path;
    cv::Mat texture_ref_bgr;
    cv::Mat warped_texture;
    cv::Mat overlay;
    cv::VideoWriter writer;
};

ProfileTotals g_profile_totals;
std::mutex g_profile_mutex;
bool g_profile_enabled = true;
bool g_info_enabled = true;

class ScopedTimer {
public:
    ScopedTimer(std::map<std::string, double>* frame_profile, std::string name)
        : frame_profile_(frame_profile),
          name_(std::move(name)),
          start_(g_profile_enabled && frame_profile_ != nullptr
              ? std::chrono::steady_clock::now()
              : std::chrono::steady_clock::time_point()) {}

    ~ScopedTimer() {
        if (!g_profile_enabled || frame_profile_ == nullptr) {
            return;
        }
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        (*frame_profile_)[name_] += ms;
    }

private:
    std::map<std::string, double>* frame_profile_;
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

void printFrameProfile(const std::string& prefix, const std::map<std::string, double>& profile) {
    if (!g_profile_enabled) {
        return;
    }
    double total_ms = 0.0;
    for (const auto& [_, ms] : profile) {
        total_ms += ms;
    }

    std::vector<std::pair<std::string, double>> stages(profile.begin(), profile.end());
    std::sort(stages.begin(), stages.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::lock_guard<std::mutex> lock(g_profile_mutex);
    std::cout << "[PROFILE] " << prefix << " total=" << total_ms << " ms";
    for (const auto& [name, ms] : stages) {
        std::cout << " " << name << "=" << ms;
    }
    std::cout << "\n";
}

void addFrameProfile(const std::map<std::string, double>& profile) {
    if (!g_profile_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    ++g_profile_totals.frames;
    for (const auto& [name, ms] : profile) {
        g_profile_totals.ms_by_stage[name] += ms;
    }
}

void printProfileSummary() {
    if (!g_profile_enabled) {
        return;
    }
    if (g_profile_totals.frames == 0) {
        return;
    }

    double total_ms = 0.0;
    for (const auto& [_, ms] : g_profile_totals.ms_by_stage) {
        total_ms += ms;
    }

    std::vector<std::pair<std::string, double>> stages(
        g_profile_totals.ms_by_stage.begin(),
        g_profile_totals.ms_by_stage.end());
    std::sort(stages.begin(), stages.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::cout << "[PROFILE] summary frames=" << g_profile_totals.frames
              << " total=" << total_ms << " ms"
              << " avg=" << (total_ms / g_profile_totals.frames) << " ms/frame\n";
    for (const auto& [name, ms] : stages) {
        std::cout << "[PROFILE]   " << name
                  << " total=" << ms << " ms"
                  << " avg=" << (ms / g_profile_totals.frames) << " ms/frame\n";
    }
}

bool debugStagesEnabled() {
    const char* value = std::getenv("SHADOW_DEBUG_STAGES");
    return value != nullptr && std::string(value) != "0";
}

bool fastApproxEnabled() {
    const char* value = std::getenv("SHADOW_FAST_APPROX");
    return value == nullptr || std::string(value) != "0";
}

int envInt(const char* name, int default_value) {
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

void applyTrackerEnvOverrides(SheetTrackerConfig* config) {
    config->maxTrackedPoints = std::clamp(
        envInt("SHADOW_TRACKER_MAX_POINTS", config->maxTrackedPoints),
        config->minTrackedPoints,
        config->maxFeatures);
    config->templateSearchRadius = std::clamp(
        envInt("SHADOW_TRACKER_SEARCH_RADIUS", config->templateSearchRadius),
        8,
        80);
    config->predictedSearchRadius = std::clamp(
        envInt("SHADOW_TRACKER_PREDICTED_SEARCH_RADIUS", config->predictedSearchRadius),
        6,
        config->templateSearchRadius);
    config->maxHeldHomographyFrames = std::clamp(
        envInt("SHADOW_TRACKER_HOLD_FRAMES", config->maxHeldHomographyFrames),
        0,
        120);
    config->directSheetRefineEnabled = envInt("SHADOW_DIRECT_SHEET_REFINE", config->directSheetRefineEnabled ? 1 : 0) != 0;
    config->directSheetRefineTwoTagSeeds = envInt(
        "SHADOW_DIRECT_SHEET_REFINE_TWO_TAGS",
        config->directSheetRefineTwoTagSeeds ? 1 : 0) != 0;
    config->directSheetMaxCornerMovePx = static_cast<double>(std::clamp(
        envInt("SHADOW_DIRECT_SHEET_MAX_CORNER_MOVE", static_cast<int>(std::lround(config->directSheetMaxCornerMovePx))),
        5,
        160));
    config->tagDetectInterval = std::clamp(
        envInt("SHADOW_TAG_DETECT_INTERVAL", config->tagDetectInterval),
        1,
        120);
}

fs::path debugPath(const fs::path& prefix, const std::string& suffix) {
    return fs::path(prefix.string() + suffix);
}

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --video VIDEO --reference IMAGE --outdir DIR [options]\n"
        << "\n"
        << "Options:\n"
        << "  --frames N [N ...]        Specific frame indices to process.\n"
        << "  --num-samples N           Evenly sampled frames when --frames is omitted without --surface_texture. Default: 2\n"
        << "  --reference-scale SCALE   Scale after rotating reference. Default: 1.0\n"
        << "  --surface_texture IMAGE   Optional texture image to warp onto the surface. Without --frames, writes final_overlay.avi for every frame.\n"
        << "  --surface_texture_dir DIR Optional directory of texture images. Without --frames, writes one <video_stem>_<texture>.avi per texture.\n"
        << "  --all_debug               Write all intermediate debug PNGs for frame runs.\n"
        << "  --outputs LIST            Comma-separated outputs: all,raw,aligned,flat_raw,flat_aligned,quality,colormatch,none\n";
}

void parseOutputs(const std::string& value, Args* args) {
    args->write_raw_frame = false;
    args->write_aligned_ref = false;
    args->write_raw_frame_flat = false;
    args->write_aligned_ref_flat = false;
    args->write_combined_quality = false;
    args->write_colormatch = false;

    size_t start = 0;
    while (start <= value.size()) {
        size_t end = value.find(',', start);
        std::string token = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), token.end());

        if (token == "all") {
            args->write_raw_frame = true;
            args->write_aligned_ref = true;
            args->write_raw_frame_flat = true;
            args->write_aligned_ref_flat = true;
            args->write_combined_quality = true;
            args->write_colormatch = true;
        } else if (token == "none") {
            args->write_raw_frame = false;
            args->write_aligned_ref = false;
            args->write_raw_frame_flat = false;
            args->write_aligned_ref_flat = false;
            args->write_combined_quality = false;
            args->write_colormatch = false;
        } else if (token.empty()) {
        } else if (token == "raw") {
            args->write_raw_frame = true;
        } else if (token == "aligned") {
            args->write_aligned_ref = true;
        } else if (token == "flat_raw") {
            args->write_raw_frame_flat = true;
        } else if (token == "flat_aligned") {
            args->write_aligned_ref_flat = true;
        } else if (token == "quality") {
            args->write_combined_quality = true;
        } else if (token == "colormatch") {
            args->write_colormatch = true;
        } else {
            throw std::runtime_error("Unknown output token: " + token);
        }

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (key == "--video") {
            args.video = needValue(key);
        } else if (key == "--reference") {
            args.reference = needValue(key);
        } else if (key == "--outdir") {
            args.outdir = needValue(key);
        } else if (key == "--num-samples") {
            args.num_samples = std::stoi(needValue(key));
        } else if (key == "--reference-scale") {
            args.reference_scale = std::stod(needValue(key));
        } else if (key == "--surface_texture" || key == "--surface-texture") {
            args.surface_texture = needValue(key);
        } else if (key == "--surface_texture_dir" || key == "--surface-texture-dir" ||
                   key == "--surface_textures_dir" || key == "--surface-textures-dir") {
            args.surface_texture_dir = needValue(key);
        } else if (key == "--all_debug" || key == "--all-debug") {
            parseOutputs("all", &args);
        } else if (key == "--outputs") {
            parseOutputs(needValue(key), &args);
        } else if (key == "--frames") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                args.frames.push_back(std::stoi(argv[++i]));
            }
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.video.empty() || args.reference.empty() || args.outdir.empty()) {
        usage(argv[0]);
        throw std::runtime_error("--video, --reference, and --outdir are required");
    }
    if (args.reference_scale <= 0.0) {
        throw std::runtime_error("--reference-scale must be > 0");
    }
    if (args.num_samples < 1) {
        throw std::runtime_error("--num-samples must be >= 1");
    }
    return args;
}

std::vector<int> sampleFrameIndices(int total_frames, int count) {
    std::vector<int> out;
    if (total_frames <= 0) {
        return out;
    }
    if (count <= 1) {
        out.push_back(std::clamp(total_frames / 2, 0, total_frames - 1));
        return out;
    }

    std::set<int> unique;
    for (int i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(count - 1);
        int idx = static_cast<int>(std::llround(t * (total_frames - 1)));
        unique.insert(std::clamp(idx, 0, total_frames - 1));
    }
    out.assign(unique.begin(), unique.end());
    return out;
}

std::string framePrefix(int idx) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "frame_%06d", idx);
    return std::string(buf);
}

void clearPreviousOutputs(const fs::path& outdir, const std::string& prefix) {
    if (!fs::exists(outdir)) {
        return;
    }
    for (const auto& entry : fs::directory_iterator(outdir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix + "_", 0) == 0) {
            fs::remove(entry.path());
        }
    }
}

bool writePngFast(const fs::path& path, const cv::Mat& image) {
    return cv::imwrite(path.string(), image);
}

DetectionResult detectAprilTags(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    static const cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    static const cv::aruco::ArucoDetector detector = [] {
        cv::aruco::DetectorParameters params;
#ifdef CV_VERSION_EPOCH
        (void)params;
#endif
        params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
        return cv::aruco::ArucoDetector(dictionary, params);
    }();

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    DetectionResult result;
    detector.detectMarkers(gray, corners, ids, result.rejected);

    for (size_t i = 0; i < ids.size(); ++i) {
        result.detections[ids[i]] = corners[i];
    }

    if (result.detections.size() < 4) {
        constexpr double fallback_scale = 0.75;
        cv::Mat small_bgr;
        cv::resize(bgr, small_bgr, cv::Size(), fallback_scale, fallback_scale, cv::INTER_AREA);
        cv::Mat small_gray;
        cv::cvtColor(small_bgr, small_gray, cv::COLOR_BGR2GRAY);

        std::vector<std::vector<cv::Point2f>> small_corners;
        std::vector<int> small_ids;
        std::vector<std::vector<cv::Point2f>> small_rejected;
        detector.detectMarkers(small_gray, small_corners, small_ids, small_rejected);
        for (size_t i = 0; i < small_ids.size(); ++i) {
            int id = small_ids[i];
            if (result.detections.count(id)) {
                continue;
            }
            std::vector<cv::Point2f> scaled_corners = small_corners[i];
            for (auto& point : scaled_corners) {
                point.x = static_cast<float>(point.x / fallback_scale);
                point.y = static_cast<float>(point.y / fallback_scale);
            }
            result.detections[id] = std::move(scaled_corners);
        }
    }
    return result;
}

std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) {
        throw std::runtime_error("Expected 4 points");
    }

    std::vector<cv::Point2f> ordered(4);
    auto sum = [](const cv::Point2f& p) { return p.x + p.y; };
    auto diff = [](const cv::Point2f& p) { return p.x - p.y; };
    ordered[0] = *std::min_element(points.begin(), points.end(), [&](auto& a, auto& b) { return sum(a) < sum(b); });
    ordered[2] = *std::max_element(points.begin(), points.end(), [&](auto& a, auto& b) { return sum(a) < sum(b); });
    ordered[1] = *std::max_element(points.begin(), points.end(), [&](auto& a, auto& b) { return diff(a) < diff(b); });
    ordered[3] = *std::min_element(points.begin(), points.end(), [&](auto& a, auto& b) { return diff(a) < diff(b); });
    return ordered;
}

bool validateHomographyFromMatches(
    const cv::Mat& H,
    const std::vector<cv::Point2f>& src,
    const std::vector<cv::Point2f>& dst,
    std::string* error) {
    if (H.empty()) {
        if (error) *error = "homography is empty";
        return false;
    }

    cv::Mat H64;
    H.convertTo(H64, CV_64F);
    double det = cv::determinant(H64(cv::Rect(0, 0, 2, 2)));
    if (!std::isfinite(det) || std::abs(det) < 1e-8) {
        if (error) *error = "homography is near-singular";
        return false;
    }

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(src, projected, H);
    double total = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        total += cv::norm(projected[i] - dst[i]);
    }
    double mean = total / std::max<size_t>(1, projected.size());
    if (!std::isfinite(mean) || mean > 8.0) {
        if (error) *error = "match reprojection error too high: " + std::to_string(mean);
        return false;
    }
    return true;
}

cv::Mat makeTagContrastImage(const cv::Mat& bgr) {
    cv::Mat contrast(bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
            uchar* out = contrast.ptr<uchar>(y);
            for (int x = 0; x < bgr.cols; ++x) {
                float b = static_cast<float>(row[x][0]);
                float g = static_cast<float>(row[x][1]);
                float score = (b - g) / (b + g + 1.0f);
                out[x] = cv::saturate_cast<uchar>((score + 1.0f) * 127.5f);
            }
        }
    });
    cv::GaussianBlur(contrast, contrast, cv::Size(3, 3), 0.0);
    return contrast;
}

cv::Mat makeReferenceFeatureMask(const cv::Mat& reference_bgr) {
    cv::Mat hsv;
    cv::cvtColor(reference_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat saturated;
    cv::inRange(hsv, cv::Scalar(35, 25, 20), cv::Scalar(150, 255, 255), saturated);

    cv::Mat dominance(reference_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, reference_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* row = reference_bgr.ptr<cv::Vec3b>(y);
            uchar* out = dominance.ptr<uchar>(y);
            for (int x = 0; x < reference_bgr.cols; ++x) {
                int b = row[x][0];
                int g = row[x][1];
                int r = row[x][2];
                int maxc = std::max({b, g, r});
                int minc = std::min({b, g, r});
                int chroma = maxc - minc;
                bool blue_like = chroma >= 20 && b >= g + 10 && b >= r + 10;
                bool green_like = chroma >= 12 && g >= r + 5 && g >= b - 25;
                if (blue_like || green_like) {
                    out[x] = 255;
                }
            }
        }
    });

    cv::Mat mask;
    cv::bitwise_and(saturated, dominance, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8U), cv::Point(-1, -1), 1);
    return mask;
}

cv::Mat makeVisibleBoardColorProtectionMask(const cv::Mat& frame_bgr);

std::vector<cv::Point2f> buildReferenceFeaturePoints(
    const cv::Mat& reference_bgr,
    const cv::Mat& reference_tag_contrast,
    const SheetTrackerConfig& config) {
    cv::Mat feature_mask = makeReferenceFeatureMask(reference_bgr);
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(
        reference_tag_contrast,
        points,
        config.maxFeatures,
        config.qualityLevel,
        config.minDistance,
        feature_mask,
        3,
        false,
        0.04);
    return points;
}

std::vector<cv::Point2f> referenceImageCorners(cv::Size reference_size) {
    return {
        {0.0f, 0.0f},
        {static_cast<float>(reference_size.width - 1), 0.0f},
        {static_cast<float>(reference_size.width - 1), static_cast<float>(reference_size.height - 1)},
        {0.0f, static_cast<float>(reference_size.height - 1)},
    };
}

bool pointsInsideFrame(const std::vector<cv::Point2f>& points, cv::Size frame_size, float margin) {
    int inside = 0;
    for (const cv::Point2f& p : points) {
        if (p.x >= -margin && p.y >= -margin
                && p.x < frame_size.width + margin
                && p.y < frame_size.height + margin) {
            ++inside;
        }
    }
    return inside >= 3;
}

double medianPointError(
    const std::vector<cv::Point2f>& projected,
    const std::vector<cv::Point2f>& observed,
    const std::vector<uchar>& inliers) {
    std::vector<double> errors;
    for (size_t i = 0; i < projected.size() && i < observed.size() && i < inliers.size(); ++i) {
        if (inliers[i]) {
            errors.push_back(cv::norm(projected[i] - observed[i]));
        }
    }
    if (errors.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
    return errors[errors.size() / 2];
}

bool validateTrackedHomography(
    const cv::Mat& H_ref_to_curr,
    const std::vector<cv::Point2f>& ref_pts,
    const std::vector<cv::Point2f>& curr_pts,
    const std::vector<uchar>& inliers,
    const cv::Size& reference_size,
    const cv::Size& frame_size,
    const cv::Mat& H_ref_to_prev,
    const SheetTrackerConfig& config,
    std::string& failure_reason) {
    if (H_ref_to_curr.empty()) {
        failure_reason = "tracked homography is empty";
        return false;
    }

    cv::Mat H64;
    H_ref_to_curr.convertTo(H64, CV_64F);
    if (!std::isfinite(cv::determinant(H64(cv::Rect(0, 0, 2, 2))))) {
        failure_reason = "tracked homography determinant is not finite";
        return false;
    }

    int inlier_count = 0;
    for (uchar value : inliers) {
        if (value) {
            ++inlier_count;
        }
    }
    if (inlier_count < config.minInliers) {
        failure_reason = "too few fallback inliers: " + std::to_string(inlier_count);
        return false;
    }

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(ref_pts, projected, H_ref_to_curr);
    double median_error = medianPointError(projected, curr_pts, inliers);
    if (!std::isfinite(median_error) || median_error > config.maxMedianReprojError) {
        failure_reason = "fallback median reprojection error too high: " + std::to_string(median_error);
        return false;
    }

    std::vector<cv::Point2f> quad = referenceImageCorners(reference_size);
    std::vector<cv::Point2f> curr_quad;
    cv::perspectiveTransform(quad, curr_quad, H_ref_to_curr);
    float cropped_sheet_margin = std::max(frame_size.width, frame_size.height) * 1.35f;
    if (!pointsInsideFrame(curr_quad, frame_size, cropped_sheet_margin)) {
        failure_reason = "fallback projected sheet quad is outside frame";
        return false;
    }
    double quad_area = std::abs(cv::contourArea(curr_quad));
    double frame_area = static_cast<double>(frame_size.width) * static_cast<double>(frame_size.height);
    if (!std::isfinite(quad_area) || quad_area < frame_area * 0.05 || quad_area > frame_area * 3.0) {
        failure_reason = "fallback projected sheet quad area implausible: " + std::to_string(quad_area);
        return false;
    }

    if (!H_ref_to_prev.empty()) {
        std::vector<cv::Point2f> prev_quad;
        cv::perspectiveTransform(quad, prev_quad, H_ref_to_prev);
        double max_jump = 0.0;
        for (size_t i = 0; i < curr_quad.size(); ++i) {
            max_jump = std::max(max_jump, static_cast<double>(cv::norm(curr_quad[i] - prev_quad[i])));
        }
        if (max_jump > config.maxHomographyJumpPx) {
            failure_reason = "fallback homography jump too large: " + std::to_string(max_jump);
            return false;
        }
    }

    cv::Rect2f bounds;
    bool first = true;
    for (size_t i = 0; i < curr_pts.size() && i < inliers.size(); ++i) {
        if (!inliers[i]) {
            continue;
        }
        if (first) {
            bounds = cv::Rect2f(curr_pts[i].x, curr_pts[i].y, 1.0f, 1.0f);
            first = false;
        } else {
            bounds |= cv::Rect2f(curr_pts[i].x, curr_pts[i].y, 1.0f, 1.0f);
        }
    }
    if (bounds.width < frame_size.width * 0.08f || bounds.height < frame_size.height * 0.08f) {
        failure_reason = "fallback inliers are not spatially distributed";
        return false;
    }

    failure_reason = "ok";
    return true;
}

double projectedQuadMaxJump(
    const cv::Mat& H_a,
    const cv::Mat& H_b,
    cv::Size reference_size) {
    if (H_a.empty() || H_b.empty()) {
        return 0.0;
    }
    std::vector<cv::Point2f> quad = referenceImageCorners(reference_size);
    std::vector<cv::Point2f> a_quad;
    std::vector<cv::Point2f> b_quad;
    cv::perspectiveTransform(quad, a_quad, H_a);
    cv::perspectiveTransform(quad, b_quad, H_b);

    double max_jump = 0.0;
    for (size_t i = 0; i < a_quad.size() && i < b_quad.size(); ++i) {
        max_jump = std::max(max_jump, static_cast<double>(cv::norm(a_quad[i] - b_quad[i])));
    }
    return max_jump;
}

bool validateTagHomographyAgainstTracker(
    const cv::Mat& H_ref_to_frame,
    const cv::Size& reference_size,
    const SheetTrackerConfig& config,
    const SheetTrackerState& state,
    std::string& failure_reason) {
    if (!state.hasValidH || state.H_ref_to_prev.empty() || H_ref_to_frame.empty()) {
        failure_reason = "no previous homography for temporal tag check";
        return true;
    }

    double max_jump = projectedQuadMaxJump(H_ref_to_frame, state.H_ref_to_prev, reference_size);
    if (!std::isfinite(max_jump)) {
        failure_reason = "tag homography temporal jump is not finite";
        return false;
    }
    if (max_jump > config.maxHomographyJumpPx) {
        failure_reason = "tag homography jump too large: " + std::to_string(max_jump);
        return false;
    }

    failure_reason = "ok";
    return true;
}

int countCommonTags(const DetectionResult& ref_tags, const DetectionResult& frame_tags) {
    int count = 0;
    for (const auto& [id, _] : ref_tags.detections) {
        if (frame_tags.detections.count(id)) {
            ++count;
        }
    }
    return count;
}

void updateSheetTrackerState(
    const cv::Mat& frame_bgr,
    const cv::Mat& tag_contrast,
    const cv::Mat& H_ref_to_frame,
    SheetTrackerState* state) {
    if (state == nullptr || H_ref_to_frame.empty()) {
        return;
    }
    if (!state->H_ref_to_prev.empty()) {
        state->H_ref_to_prev.copyTo(state->H_ref_to_prevprev);
    }
    H_ref_to_frame.convertTo(state->H_ref_to_prev, CV_64F);
    frame_bgr.copyTo(state->prevFrameBgr);
    tag_contrast.copyTo(state->prevTagContrast);
    state->hasValidH = true;
}

std::pair<cv::Mat, std::vector<cv::Point2f>> makeProjectedBoardMask(
    const cv::Mat& H,
    cv::Size ref_size,
    cv::Size frame_size) {
    std::vector<cv::Point2f> ref_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(ref_size.width - 1), 0.0f},
        {static_cast<float>(ref_size.width - 1), static_cast<float>(ref_size.height - 1)},
        {0.0f, static_cast<float>(ref_size.height - 1)},
    };
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(ref_corners, projected, H);

    std::vector<cv::Point> poly;
    for (const auto& p : projected) {
        poly.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
    }

    cv::Mat mask(frame_size, CV_8UC1, cv::Scalar(0));
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return {mask, projected};
}

cv::Mat preprocessGreenMask(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat green_mask;
    cv::inRange(hsv, cv::Scalar(35, 45, 45), cv::Scalar(95, 255, 255), green_mask);
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    return green_mask;
}

std::vector<cv::Point2f> tagBoardCorners(const cv::Point2f& center_m) {
    return {
        {center_m.x - static_cast<float>(kTagHalfSizeM), center_m.y + static_cast<float>(kTagHalfSizeM)},
        {center_m.x + static_cast<float>(kTagHalfSizeM), center_m.y + static_cast<float>(kTagHalfSizeM)},
        {center_m.x + static_cast<float>(kTagHalfSizeM), center_m.y - static_cast<float>(kTagHalfSizeM)},
        {center_m.x - static_cast<float>(kTagHalfSizeM), center_m.y - static_cast<float>(kTagHalfSizeM)},
    };
}

cv::Point2f projectSurfaceCornerFromTag(
    const cv::Mat& H_tag,
    const cv::Point2f& surface_corner_m,
    const cv::Point2f& tag_center_m) {
    std::vector<cv::Point2f> local = {{
        static_cast<float>((surface_corner_m.x - tag_center_m.x) * kTagCoordScale),
        static_cast<float>((surface_corner_m.y - tag_center_m.y) * kTagCoordScale),
    }};
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(local, projected, H_tag);
    return projected[0];
}

bool estimateBoardQuadFromTags(
    const std::map<int, std::vector<cv::Point2f>>& detections,
    std::vector<cv::Point2f>* quad) {
    std::vector<int> visible_ids;
    for (const auto& [id, _] : detections) {
        if (kTagCenterLocations.count(id)) {
            visible_ids.push_back(id);
        }
    }
    if (visible_ids.empty()) {
        return false;
    }

    std::vector<int> sparse_ids = visible_ids;
    if (visible_ids.size() < 4) {
        std::map<int, int> column_counts;
        for (int id : visible_ids) {
            column_counts[id % 4]++;
        }
        int dominant_column = column_counts.begin()->first;
        int best_count = -1;
        for (const auto& [col, count] : column_counts) {
            if (count > best_count || (count == best_count && col < dominant_column)) {
                dominant_column = col;
                best_count = count;
            }
        }
        sparse_ids.clear();
        for (int id : visible_ids) {
            if (id % 4 == dominant_column) {
                sparse_ids.push_back(id);
            }
        }
    }

    std::vector<cv::Point2f> corner_src;
    std::vector<cv::Point2f> corner_dst;
    std::vector<std::vector<cv::Point2f>> per_corner_samples(4);
    std::vector<std::vector<double>> per_corner_weights(4);

    for (int id : sparse_ids) {
        const auto det_it = detections.find(id);
        const auto loc_it = kTagCenterLocations.find(id);
        if (det_it == detections.end() || loc_it == kTagCenterLocations.end() || det_it->second.size() != 4) {
            continue;
        }

        const auto board_corners = tagBoardCorners(loc_it->second);
        for (size_t i = 0; i < 4; ++i) {
            corner_src.push_back(board_corners[i]);
            corner_dst.push_back(det_it->second[i]);
        }

        cv::Mat H_tag = cv::getPerspectiveTransform(kTagLocalCorners, det_it->second);
        if (H_tag.empty()) {
            continue;
        }
        double weight = std::max(cv::contourArea(det_it->second), 1.0);
        for (size_t i = 0; i < kSurfaceCornersM.size(); ++i) {
            per_corner_samples[i].push_back(projectSurfaceCornerFromTag(H_tag, kSurfaceCornersM[i], loc_it->second));
            per_corner_weights[i].push_back(weight);
        }
    }

    if (corner_src.size() >= 4) {
        cv::Mat H_board;
        if (corner_src.size() == 4) {
            H_board = cv::getPerspectiveTransform(corner_src, corner_dst);
        } else {
            H_board = cv::findHomography(corner_src, corner_dst, cv::RANSAC, 6.0);
        }
        if (!H_board.empty()) {
            std::vector<cv::Point2f> projected;
            cv::perspectiveTransform(kSurfaceCornersM, projected, H_board);
            *quad = orderPoints(projected);
            return true;
        }
    }

    std::vector<cv::Point2f> out(4);
    for (size_t i = 0; i < per_corner_samples.size(); ++i) {
        if (per_corner_samples[i].empty()) {
            return false;
        }
        double sw = std::accumulate(per_corner_weights[i].begin(), per_corner_weights[i].end(), 0.0);
        cv::Point2f p(0.0f, 0.0f);
        for (size_t j = 0; j < per_corner_samples[i].size(); ++j) {
            p += per_corner_samples[i][j] * static_cast<float>(per_corner_weights[i][j] / sw);
        }
        out[i] = p;
    }
    *quad = orderPoints(out);
    return true;
}

bool findHomographyAndBoardMask(
    const cv::Mat& ref_bgr,
    const cv::Mat& frame_bgr,
    const DetectionResult& ref_tags,
    const DetectionResult& frame_tags,
    cv::Mat* H,
    cv::Mat* board_mask,
    std::string* error) {
    std::vector<int> common_ids;
    for (const auto& [id, _] : ref_tags.detections) {
        if (frame_tags.detections.count(id)) {
            common_ids.push_back(id);
        }
    }

    if (!common_ids.empty()) {
        std::vector<cv::Point2f> src_pts;
        std::vector<cv::Point2f> dst_pts;
        for (int id : common_ids) {
            const auto& ref_corners = ref_tags.detections.at(id);
            const auto& frame_corners = frame_tags.detections.at(id);
            src_pts.insert(src_pts.end(), ref_corners.begin(), ref_corners.end());
            dst_pts.insert(dst_pts.end(), frame_corners.begin(), frame_corners.end());
        }

        cv::Mat H_tags;
        if (src_pts.size() == 4) {
            H_tags = cv::getPerspectiveTransform(src_pts, dst_pts);
        } else {
            H_tags = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 6.0);
        }

        std::string validation_error;
        if (validateHomographyFromMatches(H_tags, src_pts, dst_pts, &validation_error)) {
            auto mask_and_quad = makeProjectedBoardMask(H_tags, ref_bgr.size(), frame_bgr.size());
            *H = H_tags;
            *board_mask = mask_and_quad.first;
            return true;
        }
        if (error) {
            *error = validation_error;
        }
    }

    cv::Mat green_mask = preprocessGreenMask(frame_bgr);
    cv::Mat fallback_mask = green_mask.clone();
    cv::Mat kernel = cv::Mat::ones(7, 7, CV_8U);
    cv::morphologyEx(fallback_mask, fallback_mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(fallback_mask, fallback_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    std::vector<cv::Point2f> quad;
    if (!estimateBoardQuadFromTags(frame_tags.detections, &quad)) {
        if (error && error->empty()) {
            *error = "no common reference/frame AprilTags and no usable frame tags for board estimate";
        }
        *board_mask = fallback_mask;
        return false;
    }

    std::vector<cv::Point2f> ref_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(ref_bgr.cols - 1), 0.0f},
        {static_cast<float>(ref_bgr.cols - 1), static_cast<float>(ref_bgr.rows - 1)},
        {0.0f, static_cast<float>(ref_bgr.rows - 1)},
    };
    cv::Mat H_quad = cv::getPerspectiveTransform(ref_corners, quad);
    auto mask_and_quad = makeProjectedBoardMask(H_quad, ref_bgr.size(), frame_bgr.size());
    *H = H_quad;
    *board_mask = mask_and_quad.first;
    return true;
}

void drawProjectedQuad(cv::Mat* image, const cv::Mat& H, cv::Size reference_size, const cv::Scalar& color) {
    if (image == nullptr || image->empty() || H.empty()) {
        return;
    }
    std::vector<cv::Point2f> quad = referenceImageCorners(reference_size);
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(quad, projected, H);
    for (size_t i = 0; i < projected.size(); ++i) {
        cv::Point a(
            static_cast<int>(std::lround(projected[i].x)),
            static_cast<int>(std::lround(projected[i].y)));
        cv::Point b(
            static_cast<int>(std::lround(projected[(i + 1) % projected.size()].x)),
            static_cast<int>(std::lround(projected[(i + 1) % projected.size()].y)));
        cv::line(*image, a, b, color, 2, cv::LINE_AA);
        cv::circle(*image, a, 4, color, cv::FILLED, cv::LINE_AA);
    }
}

bool trackPointByTemplate(
    const cv::Mat& prev_contrast,
    const cv::Mat& curr_contrast,
    const cv::Point2f& prev_point,
    const cv::Point2f& search_center,
    int window_size,
    int search_radius,
    cv::Point2f* curr_point,
    float* match_error) {
    int half = window_size / 2;
    if (prev_point.x < half || prev_point.y < half
            || prev_point.x >= prev_contrast.cols - half
            || prev_point.y >= prev_contrast.rows - half) {
        return false;
    }

    cv::Mat patch;
    cv::getRectSubPix(prev_contrast, cv::Size(window_size, window_size), prev_point, patch);

    int cx = static_cast<int>(std::lround(search_center.x));
    int cy = static_cast<int>(std::lround(search_center.y));
    int x0 = std::max(0, cx - search_radius - half);
    int y0 = std::max(0, cy - search_radius - half);
    int x1 = std::min(curr_contrast.cols, cx + search_radius + half + 1);
    int y1 = std::min(curr_contrast.rows, cy + search_radius + half + 1);
    cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
    if (roi.width < window_size || roi.height < window_size) {
        return false;
    }

    cv::Mat result;
    cv::matchTemplate(curr_contrast(roi), patch, result, cv::TM_SQDIFF_NORMED);
    double min_value = 0.0;
    cv::Point min_location;
    cv::minMaxLoc(result, &min_value, nullptr, &min_location, nullptr);
    if (!std::isfinite(min_value)) {
        return false;
    }

    *curr_point = cv::Point2f(
        static_cast<float>(roi.x + min_location.x + half),
        static_cast<float>(roi.y + min_location.y + half));
    if (match_error != nullptr) {
        *match_error = static_cast<float>(min_value * 100.0);
    }
    return true;
}

bool trackHomographyWithoutTags(
    const cv::Mat& curr_frame_bgr,
    const cv::Mat& curr_tag_contrast,
    const cv::Mat& reference_tag_contrast,
    const cv::Size& reference_size,
    const SheetTrackerConfig& config,
    SheetTrackerState& state,
    cv::Mat& H_ref_to_curr,
    std::string& failure_reason,
    const fs::path& debug_prefix = fs::path()) {
    (void)reference_tag_contrast;
    if (!state.hasValidH || state.H_ref_to_prev.empty() || state.prevTagContrast.empty()) {
        failure_reason = "tracker has no previous valid homography";
        return false;
    }
    if (state.referenceFeaturePoints.empty()) {
        failure_reason = "tracker has no reference feature points";
        return false;
    }

    std::vector<cv::Point2f> prev_projected;
    cv::perspectiveTransform(state.referenceFeaturePoints, prev_projected, state.H_ref_to_prev);

    std::vector<cv::Point2f> ref_pts;
    std::vector<cv::Point2f> prev_pts;
    constexpr float kMargin = 8.0f;
    for (size_t i = 0; i < prev_projected.size(); ++i) {
        const cv::Point2f& p = prev_projected[i];
        if (p.x >= kMargin && p.y >= kMargin
                && p.x < state.prevTagContrast.cols - kMargin
                && p.y < state.prevTagContrast.rows - kMargin) {
            ref_pts.push_back(state.referenceFeaturePoints[i]);
            prev_pts.push_back(p);
        }
    }
    if (static_cast<int>(prev_pts.size()) < config.minTrackedPoints) {
        failure_reason = "too few projected tracker points: " + std::to_string(prev_pts.size());
        return false;
    }
    if (static_cast<int>(prev_pts.size()) > config.maxTrackedPoints) {
        std::vector<cv::Point2f> limited_ref_pts;
        std::vector<cv::Point2f> limited_prev_pts;
        limited_ref_pts.reserve(static_cast<size_t>(config.maxTrackedPoints));
        limited_prev_pts.reserve(static_cast<size_t>(config.maxTrackedPoints));
        double step = static_cast<double>(prev_pts.size()) / static_cast<double>(config.maxTrackedPoints);
        for (int i = 0; i < config.maxTrackedPoints; ++i) {
            size_t index = std::min(
                prev_pts.size() - 1,
                static_cast<size_t>(std::lround(static_cast<double>(i) * step)));
            limited_ref_pts.push_back(ref_pts[index]);
            limited_prev_pts.push_back(prev_pts[index]);
        }
        ref_pts = std::move(limited_ref_pts);
        prev_pts = std::move(limited_prev_pts);
    }

    std::vector<cv::Point2f> predicted_pts = prev_pts;
    int search_radius = config.templateSearchRadius;
    if (!state.H_ref_to_prevprev.empty()) {
        std::vector<cv::Point2f> prevprev_projected;
        cv::perspectiveTransform(ref_pts, prevprev_projected, state.H_ref_to_prevprev);
        if (prevprev_projected.size() == prev_pts.size()) {
            for (size_t i = 0; i < prev_pts.size(); ++i) {
                predicted_pts[i] = prev_pts[i] + (prev_pts[i] - prevprev_projected[i]);
            }
            search_radius = config.predictedSearchRadius;
        }
    }

    std::vector<cv::Point2f> curr_pts(prev_pts.size());
    std::vector<uchar> lk_status;
    std::vector<float> lk_error(prev_pts.size(), std::numeric_limits<float>::infinity());
    lk_status.assign(prev_pts.size(), 0);
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        cv::Point2f tracked;
        float error_value = 0.0f;
        if (trackPointByTemplate(
                state.prevTagContrast,
                curr_tag_contrast,
                prev_pts[i],
                predicted_pts[i],
                config.lkWindowSize,
                search_radius,
                &tracked,
                &error_value)) {
            curr_pts[i] = tracked;
            lk_error[i] = error_value;
            lk_status[i] = 255;
        }
    }

    std::vector<cv::Point2f> back_pts(prev_pts.size());
    std::vector<uchar> back_status;
    std::vector<float> back_error(prev_pts.size(), std::numeric_limits<float>::infinity());
    back_status.assign(prev_pts.size(), 0);
    for (size_t i = 0; i < curr_pts.size(); ++i) {
        if (!lk_status[i]) {
            continue;
        }
        cv::Point2f tracked_back;
        float error_value = 0.0f;
        if (trackPointByTemplate(
                curr_tag_contrast,
                state.prevTagContrast,
                curr_pts[i],
                prev_pts[i],
                config.lkWindowSize,
                search_radius,
                &tracked_back,
                &error_value)) {
            back_pts[i] = tracked_back;
            back_error[i] = error_value;
            back_status[i] = 255;
        }
    }

    std::vector<cv::Point2f> good_ref_pts;
    std::vector<cv::Point2f> good_curr_pts;
    std::vector<uchar> track_accepted(prev_pts.size(), 0);
    cv::Mat visible_board_tracks;
    if (envInt("SHADOW_TRACKER_FILTER_VISIBLE_BOARD", 1) != 0) {
        visible_board_tracks = makeVisibleBoardColorProtectionMask(curr_frame_bgr);
        cv::dilate(
            visible_board_tracks,
            visible_board_tracks,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
            cv::Point(-1, -1),
            1);
    }
    int lk_valid_count = 0;
    int fb_valid_count = 0;
    int color_valid_count = 0;
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!lk_status[i]) {
            continue;
        }
        ++lk_valid_count;
        if (i >= back_status.size() || !back_status[i]) {
            continue;
        }
        if (lk_error[i] > config.maxLkError) {
            continue;
        }
        double fb_error = cv::norm(back_pts[i] - prev_pts[i]);
        if (fb_error > config.maxForwardBackwardError) {
            continue;
        }
        const cv::Point2f& p = curr_pts[i];
        if (p.x < 0.0f || p.y < 0.0f || p.x >= curr_frame_bgr.cols || p.y >= curr_frame_bgr.rows) {
            continue;
        }
        ++fb_valid_count;
        if (!visible_board_tracks.empty()) {
            int px = std::clamp(static_cast<int>(std::lround(p.x)), 0, visible_board_tracks.cols - 1);
            int py = std::clamp(static_cast<int>(std::lround(p.y)), 0, visible_board_tracks.rows - 1);
            if (visible_board_tracks.at<uchar>(py, px) == 0) {
                continue;
            }
        }
        ++color_valid_count;
        track_accepted[i] = 255;
        good_ref_pts.push_back(ref_pts[i]);
        good_curr_pts.push_back(p);
    }

    if (static_cast<int>(good_ref_pts.size()) < config.minTrackedPoints) {
        failure_reason = "too few forward-backward-valid tracks: " + std::to_string(good_ref_pts.size());
        return false;
    }

    cv::Mat inlier_mask_mat;
    cv::Mat H = cv::findHomography(
        good_ref_pts,
        good_curr_pts,
        cv::RANSAC,
        config.ransacReprojThreshold,
        inlier_mask_mat);
    if (!H.empty()) {
        H.convertTo(H, CV_64F);
    }

    std::vector<uchar> inliers(good_ref_pts.size(), 0);
    if (!inlier_mask_mat.empty()) {
        for (int i = 0; i < inlier_mask_mat.rows; ++i) {
            inliers[static_cast<size_t>(i)] = inlier_mask_mat.at<uchar>(i, 0) > 0 ? 255 : 0;
        }
    }

    bool valid = validateTrackedHomography(
        H,
        good_ref_pts,
        good_curr_pts,
        inliers,
        reference_size,
        curr_frame_bgr.size(),
        state.H_ref_to_prev,
        config,
        failure_reason);

    int inlier_count = 0;
    for (uchar value : inliers) {
        if (value) {
            ++inlier_count;
        }
    }
    if (g_info_enabled) {
        std::cout << "[INFO] tracker fallback projected=" << prev_pts.size()
                  << " lk_valid=" << lk_valid_count
                  << " fb_valid=" << fb_valid_count
                  << " color_valid=" << color_valid_count
                  << " inliers=" << inlier_count
                  << " result=" << (valid ? "ok" : failure_reason) << "\n";
    }

    if (config.writeDebugImages && !debug_prefix.empty()) {
        writePngFast(debugPath(debug_prefix, "_tag_contrast.png"), curr_tag_contrast);

        cv::Mat tracks = curr_frame_bgr.clone();
        for (size_t i = 0; i < prev_pts.size() && i < curr_pts.size(); ++i) {
            cv::Scalar color = track_accepted[i] ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);
            cv::Point p0(static_cast<int>(std::lround(prev_pts[i].x)), static_cast<int>(std::lround(prev_pts[i].y)));
            cv::Point p1(static_cast<int>(std::lround(curr_pts[i].x)), static_cast<int>(std::lround(curr_pts[i].y)));
            cv::line(tracks, p0, p1, color, 1, cv::LINE_AA);
            cv::circle(tracks, p1, 2, color, cv::FILLED, cv::LINE_AA);
        }
        writePngFast(debugPath(debug_prefix, "_tracked_points.png"), tracks);

        cv::Mat fallback_inliers = curr_frame_bgr.clone();
        for (size_t i = 0; i < good_curr_pts.size(); ++i) {
            cv::Scalar color = i < inliers.size() && inliers[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
            cv::circle(
                fallback_inliers,
                cv::Point(static_cast<int>(std::lround(good_curr_pts[i].x)), static_cast<int>(std::lround(good_curr_pts[i].y))),
                3,
                color,
                cv::FILLED,
                cv::LINE_AA);
        }
        writePngFast(debugPath(debug_prefix, "_fallback_inliers.png"), fallback_inliers);

        cv::Mat projected_quad = curr_frame_bgr.clone();
        if (!H.empty()) {
            drawProjectedQuad(&projected_quad, H, reference_size, valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255));
        }
        writePngFast(debugPath(debug_prefix, "_projected_sheet_quad.png"), projected_quad);
    }

    if (!valid) {
        return false;
    }

    H_ref_to_curr = H;
    return true;
}

double median(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];
    if (values.size() % 2 == 0) {
        auto max_low = *std::max_element(values.begin(), values.begin() + mid);
        med = (med + max_low) * 0.5;
    }
    return med;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    p = std::clamp(p, 0.0, 1.0);
    double pos = p * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    std::nth_element(values.begin(), values.begin() + lo, values.end());
    double lo_value = values[lo];
    if (hi == lo) {
        return lo_value;
    }
    std::nth_element(values.begin(), values.begin() + hi, values.end());
    double hi_value = values[hi];
    return lo_value + (hi_value - lo_value) * (pos - static_cast<double>(lo));
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

void buildReferenceColorMasks(const cv::Mat& image_bgr, const cv::Mat& valid_mask, cv::Mat* green_mask, cv::Mat* blue_mask) {
    cv::Mat hsv;
    cv::cvtColor(image_bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat valid;
    if (valid_mask.empty()) {
        valid = cv::Mat(image_bgr.size(), CV_8UC1, cv::Scalar(255));
    } else {
        cv::threshold(valid_mask, valid, 0, 255, cv::THRESH_BINARY);
    }

    cv::Mat sat_mask;
    cv::Mat val_mask;
    cv::inRange(hsv, cv::Scalar(0, 25, 25), cv::Scalar(179, 255, 255), sat_mask);
    cv::inRange(hsv, cv::Scalar(35, 25, 25), cv::Scalar(95, 255, 255), *green_mask);
    cv::inRange(hsv, cv::Scalar(96, 25, 25), cv::Scalar(150, 255, 255), *blue_mask);
    cv::bitwise_and(*green_mask, valid, *green_mask);
    cv::bitwise_and(*blue_mask, valid, *blue_mask);

    cv::Mat green_kernel = cv::Mat::ones(5, 5, CV_8U);
    cv::Mat blue_kernel = cv::Mat::ones(5, 5, CV_8U);
    cv::erode(*green_mask, *green_mask, green_kernel, cv::Point(-1, -1), 1);
    cv::dilate(*blue_mask, *blue_mask, blue_kernel, cv::Point(-1, -1), 1);
    cv::bitwise_and(*green_mask, valid, *green_mask);
    cv::bitwise_and(*blue_mask, valid, *blue_mask);
    (*green_mask).setTo(0, *blue_mask);
}

void buildStrictReferenceColorMasks(const cv::Mat& image_bgr, cv::Mat* green_mask, cv::Mat* blue_mask) {
    cv::Mat hsv;
    cv::cvtColor(image_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(35, 35, 35), cv::Scalar(95, 255, 255), *green_mask);
    cv::inRange(hsv, cv::Scalar(96, 35, 35), cv::Scalar(150, 255, 255), *blue_mask);

    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    cv::erode(*green_mask, *green_mask, kernel, cv::Point(-1, -1), 1);
    cv::erode(*blue_mask, *blue_mask, kernel, cv::Point(-1, -1), 1);
    (*green_mask).setTo(0, *blue_mask);
}

struct DirectSheetScore {
    double score = -std::numeric_limits<double>::infinity();
    double blue_score = 0.0;
    double green_score = 0.0;
    double coverage = 0.0;
    int projected_pixels = 0;
};

void buildCurrentSheetColorMasks(const cv::Mat& frame_bgr, cv::Mat* green_mask, cv::Mat* blue_mask) {
    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat green_hsv;
    cv::Mat blue_hsv;
    cv::inRange(hsv, cv::Scalar(35, 18, 35), cv::Scalar(115, 255, 255), green_hsv);
    cv::inRange(hsv, cv::Scalar(82, 28, 70), cv::Scalar(155, 255, 255), blue_hsv);

    cv::Mat green_dom(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat blue_dom(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* row = frame_bgr.ptr<cv::Vec3b>(y);
            uchar* green_row = green_dom.ptr<uchar>(y);
            uchar* blue_row = blue_dom.ptr<uchar>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                int b = row[x][0];
                int g = row[x][1];
                int r = row[x][2];
                int maxc = std::max({b, g, r});
                int minc = std::min({b, g, r});
                int chroma = maxc - minc;
                bool skin_like = r >= b + 8 && r >= g - 12 && maxc >= 70;
                bool dark_neutral = maxc <= 90 && chroma <= 28;
                if (!skin_like && !dark_neutral) {
                    if (chroma >= 10 && g >= r + 4 && g >= b - 35) {
                        green_row[x] = 255;
                    }
                    if (chroma >= 18 && b >= r + 14 && b >= g - 24) {
                        blue_row[x] = 255;
                    }
                }
            }
        }
    });

    cv::bitwise_and(green_hsv, green_dom, *green_mask);
    cv::bitwise_and(blue_hsv, blue_dom, *blue_mask);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    cv::morphologyEx(*green_mask, *green_mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(*blue_mask, *blue_mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    green_mask->setTo(0, *blue_mask);
}

cv::Mat makeMaskDistanceScoreImage(const cv::Mat& mask, double max_distance_px) {
    cv::Mat binary;
    cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat distance_src(binary.size(), CV_8UC1, cv::Scalar(255));
    distance_src.setTo(0, binary);
    cv::Mat distance;
    cv::distanceTransform(distance_src, distance, cv::DIST_L2, 3);
    cv::Mat score;
    distance.convertTo(score, CV_32F, -1.0 / std::max(1.0, max_distance_px), 1.0);
    cv::threshold(score, score, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(score, score, 1.0, 1.0, cv::THRESH_TRUNC);
    return score;
}

double maskedFloatMean(const cv::Mat& image_f, const cv::Mat& mask) {
    int count = cv::countNonZero(mask);
    if (count <= 0) {
        return 0.0;
    }
    return cv::mean(image_f, mask)[0];
}

DirectSheetScore scoreDirectSheetHomography(
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask,
    int ref_green_count,
    int ref_blue_count,
    const cv::Mat& frame_green_score,
    const cv::Mat& frame_blue_score,
    cv::Size frame_size,
    const SheetTrackerConfig& config) {
    DirectSheetScore out;
    if (H_ref_to_frame.empty()) {
        return out;
    }

    cv::Mat warped_green;
    cv::Mat warped_blue;
    cv::warpPerspective(ref_green_mask, warped_green, H_ref_to_frame, frame_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpPerspective(ref_blue_mask, warped_blue, H_ref_to_frame, frame_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

    int green_projected = cv::countNonZero(warped_green);
    int blue_projected = cv::countNonZero(warped_blue);
    int projected_total = green_projected + blue_projected;
    int ref_total = std::max(1, ref_green_count + ref_blue_count);
    out.projected_pixels = projected_total;
    out.coverage = static_cast<double>(projected_total) / static_cast<double>(ref_total);
    if (out.coverage < config.directSheetMinProjectedCoverage || blue_projected < 30) {
        return out;
    }

    out.green_score = maskedFloatMean(frame_green_score, warped_green);
    out.blue_score = maskedFloatMean(frame_blue_score, warped_blue);
    double blue_weight = std::max(1.0, static_cast<double>(blue_projected)) * 2.5;
    double green_weight = std::max(1.0, static_cast<double>(green_projected)) * 0.35;
    out.score = (out.blue_score * blue_weight + out.green_score * green_weight) / (blue_weight + green_weight);
    return out;
}

bool projectedQuadIsPlausible(
    const std::vector<cv::Point2f>& quad,
    cv::Size reference_size,
    cv::Size frame_size,
    const cv::Mat& H_seed,
    const SheetTrackerConfig& config,
    std::string* failure_reason) {
    if (quad.size() != 4) {
        if (failure_reason) *failure_reason = "direct sheet quad does not have four corners";
        return false;
    }
    for (const cv::Point2f& p : quad) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
                std::abs(p.x) > frame_size.width * 4.0f ||
                std::abs(p.y) > frame_size.height * 4.0f) {
            if (failure_reason) *failure_reason = "direct sheet quad contains implausible coordinates";
            return false;
        }
    }
    double area = std::abs(cv::contourArea(quad));
    double frame_area = static_cast<double>(frame_size.width) * static_cast<double>(frame_size.height);
    if (!std::isfinite(area) || area < frame_area * 0.05 || area > frame_area * 3.0) {
        if (failure_reason) *failure_reason = "direct sheet quad area implausible: " + std::to_string(area);
        return false;
    }
    cv::Mat H_candidate = cv::getPerspectiveTransform(referenceImageCorners(reference_size), quad);
    double jump = projectedQuadMaxJump(H_candidate, H_seed, reference_size);
    if (!std::isfinite(jump) || jump > config.directSheetMaxCornerMovePx) {
        if (failure_reason) *failure_reason = "direct sheet corner move too large: " + std::to_string(jump);
        return false;
    }
    return true;
}

bool refineHomographyByDirectSheetMatch(
    const cv::Mat& curr_frame_bgr,
    const cv::Mat& ref_bgr,
    const cv::Mat& H_seed,
    const SheetTrackerConfig& config,
    cv::Mat& H_refined,
    std::string& failure_reason,
    const fs::path& debug_prefix = fs::path(),
    DirectSheetScore* refined_score_out = nullptr,
    bool require_score_gain = true,
    bool log_success = true) {
    if (!config.directSheetRefineEnabled) {
        failure_reason = "direct sheet refinement disabled";
        return false;
    }
    if (H_seed.empty()) {
        failure_reason = "direct sheet refinement has no seed homography";
        return false;
    }

    cv::Mat ref_green;
    cv::Mat ref_blue;
    buildStrictReferenceColorMasks(ref_bgr, &ref_green, &ref_blue);
    int ref_green_count = cv::countNonZero(ref_green);
    int ref_blue_count = cv::countNonZero(ref_blue);
    if (ref_blue_count < 80 || ref_green_count < 80) {
        failure_reason = "direct sheet reference masks too sparse";
        return false;
    }

    cv::Mat frame_green;
    cv::Mat frame_blue;
    buildCurrentSheetColorMasks(curr_frame_bgr, &frame_green, &frame_blue);
    if (cv::countNonZero(frame_blue) < 80 || cv::countNonZero(frame_green) < 80) {
        failure_reason = "direct sheet current color masks too sparse";
        return false;
    }

    cv::Mat frame_green_score = makeMaskDistanceScoreImage(frame_green, config.directSheetScoreDistancePx);
    cv::Mat frame_blue_score = makeMaskDistanceScoreImage(frame_blue, config.directSheetScoreDistancePx);

    std::vector<cv::Point2f> seed_quad = referenceImageCorners(ref_bgr.size());
    cv::perspectiveTransform(seed_quad, seed_quad, H_seed);
    if (!projectedQuadIsPlausible(seed_quad, ref_bgr.size(), curr_frame_bgr.size(), H_seed, config, &failure_reason)) {
        return false;
    }

    auto scoreQuad = [&](const std::vector<cv::Point2f>& quad) -> DirectSheetScore {
        cv::Mat H = cv::getPerspectiveTransform(referenceImageCorners(ref_bgr.size()), quad);
        return scoreDirectSheetHomography(
            H,
            ref_green,
            ref_blue,
            ref_green_count,
            ref_blue_count,
            frame_green_score,
            frame_blue_score,
            curr_frame_bgr.size(),
            config);
    };

    std::vector<cv::Point2f> best_quad = seed_quad;
    DirectSheetScore seed_score = scoreQuad(seed_quad);
    DirectSheetScore best_score = seed_score;
    if (!std::isfinite(seed_score.score)) {
        failure_reason = "direct sheet seed score invalid";
        return false;
    }

    const std::array<double, 5> steps = {24.0, 12.0, 6.0, 3.0, 1.5};
    for (double step : steps) {
        bool improved = true;
        int passes = 0;
        while (improved && passes < 3) {
            improved = false;
            ++passes;
            for (int corner = 0; corner < 4; ++corner) {
                for (int axis = 0; axis < 2; ++axis) {
                    for (double direction : {-1.0, 1.0}) {
                        std::vector<cv::Point2f> candidate_quad = best_quad;
                        if (axis == 0) {
                            candidate_quad[corner].x += static_cast<float>(direction * step);
                        } else {
                            candidate_quad[corner].y += static_cast<float>(direction * step);
                        }
                        std::string candidate_error;
                        if (!projectedQuadIsPlausible(candidate_quad, ref_bgr.size(), curr_frame_bgr.size(), H_seed, config, &candidate_error)) {
                            continue;
                        }
                        DirectSheetScore candidate_score = scoreQuad(candidate_quad);
                        if (candidate_score.score > best_score.score + 1e-5) {
                            best_quad = std::move(candidate_quad);
                            best_score = candidate_score;
                            improved = true;
                        }
                    }
                }
            }
        }
    }

    double gain = best_score.score - seed_score.score;
    if (best_score.score < config.directSheetMinScore) {
        failure_reason = "direct sheet score too low: " + std::to_string(best_score.score);
        return false;
    }
    if (require_score_gain && gain < config.directSheetMinScoreGain) {
        failure_reason = "direct sheet score gain too small: " + std::to_string(gain);
        return false;
    }

    H_refined = cv::getPerspectiveTransform(referenceImageCorners(ref_bgr.size()), best_quad);
    H_refined.convertTo(H_refined, CV_64F);
    if (refined_score_out != nullptr) {
        *refined_score_out = best_score;
    }

    if (g_info_enabled && log_success) {
        std::cout << "[INFO] direct sheet refine seed_score=" << seed_score.score
                  << " best_score=" << best_score.score
                  << " gain=" << gain
                  << " blue=" << best_score.blue_score
                  << " green=" << best_score.green_score
                  << " coverage=" << best_score.coverage << "\n";
    }

    if (config.writeDebugImages && !debug_prefix.empty()) {
        cv::Mat debug = curr_frame_bgr.clone();
        drawProjectedQuad(&debug, H_seed, ref_bgr.size(), cv::Scalar(255, 255, 0));
        drawProjectedQuad(&debug, H_refined, ref_bgr.size(), cv::Scalar(255, 0, 255));
        writePngFast(debugPath(debug_prefix, "_direct_sheet_refine_quad.png"), debug);
        writePngFast(debugPath(debug_prefix, "_direct_sheet_frame_green_mask.png"), frame_green);
        writePngFast(debugPath(debug_prefix, "_direct_sheet_frame_blue_mask.png"), frame_blue);
    }

    failure_reason = "ok";
    return true;
}

void applyMaskedNormalizedBlur(cv::Mat* image_bgr, const cv::Mat& mask, double sigma) {
    if (mask.empty() || cv::countNonZero(mask) < 1) {
        return;
    }

    cv::Mat binary_mask;
    cv::threshold(mask, binary_mask, 0, 255, cv::THRESH_BINARY);

    cv::Mat image_f;
    image_bgr->convertTo(image_f, CV_32FC3);

    cv::Mat weight;
    binary_mask.convertTo(weight, CV_32F, 1.0 / 255.0);
    cv::Mat blurred_weight;
    cv::GaussianBlur(weight, blurred_weight, cv::Size(0, 0), sigma);

    std::vector<cv::Mat> channels;
    cv::split(image_f, channels);
    for (cv::Mat& channel : channels) {
        cv::Mat weighted;
        cv::multiply(channel, weight, weighted);
        cv::Mat blurred_weighted;
        cv::GaussianBlur(weighted, blurred_weighted, cv::Size(0, 0), sigma);
        cv::Mat safe_den;
        cv::max(blurred_weight, 1e-3, safe_den);
        cv::divide(blurred_weighted, safe_den, channel);
    }

    cv::Mat blurred_f;
    cv::merge(channels, blurred_f);
    cv::Mat blurred_u8;
    blurred_f.convertTo(blurred_u8, CV_8UC3);
    blurred_u8.copyTo(*image_bgr, binary_mask);
}

struct MaskedBlurPlan {
    cv::Mat mask;
    cv::Mat weight;
    cv::Mat blurred_weight;
};

MaskedBlurPlan makeMaskedBlurPlan(const cv::Mat& mask, double sigma) {
    MaskedBlurPlan plan;
    cv::threshold(mask, plan.mask, 0, 255, cv::THRESH_BINARY);
    plan.mask.convertTo(plan.weight, CV_32F, 1.0 / 255.0);
    cv::GaussianBlur(plan.weight, plan.blurred_weight, cv::Size(0, 0), sigma);
    return plan;
}

void applyMaskedNormalizedBlurPlan(
    const std::vector<cv::Mat>& image_channels_f,
    const MaskedBlurPlan& plan,
    double sigma,
    cv::Mat* image_bgr) {
    if (plan.mask.empty() || cv::countNonZero(plan.mask) < 1) {
        return;
    }

    std::vector<cv::Mat> blurred_channels;
    blurred_channels.reserve(image_channels_f.size());
    cv::Mat safe_den;
    cv::max(plan.blurred_weight, 1e-3, safe_den);
    for (const cv::Mat& channel : image_channels_f) {
        cv::Mat weighted;
        cv::multiply(channel, plan.weight, weighted);
        cv::Mat blurred_weighted;
        cv::GaussianBlur(weighted, blurred_weighted, cv::Size(0, 0), sigma);
        cv::Mat blurred_channel;
        cv::divide(blurred_weighted, safe_den, blurred_channel);
        blurred_channels.push_back(blurred_channel);
    }

    cv::Mat blurred_f;
    cv::merge(blurred_channels, blurred_f);
    cv::Mat blurred_u8;
    blurred_f.convertTo(blurred_u8, CV_8UC3);
    blurred_u8.copyTo(*image_bgr, plan.mask);
}

void applyMaskedNormalizedBlurPlans(
    cv::Mat* image_bgr,
    const MaskedBlurPlan& blue_plan,
    const MaskedBlurPlan& green_plan,
    double sigma) {
    cv::Mat image_f;
    image_bgr->convertTo(image_f, CV_32FC3);
    std::vector<cv::Mat> image_channels_f;
    cv::split(image_f, image_channels_f);
    applyMaskedNormalizedBlurPlan(image_channels_f, blue_plan, sigma, image_bgr);
    applyMaskedNormalizedBlurPlan(image_channels_f, green_plan, sigma, image_bgr);
}

cv::Mat keepScratchLikeComponents(const cv::Mat& mask) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (area < 3 || area > 2500 || width > 220 || height > 220) {
            continue;
        }

        int short_side = std::max(1, std::min(width, height));
        int long_side = std::max(width, height);
        double aspect = static_cast<double>(long_side) / static_cast<double>(short_side);
        bool thin = short_side <= 9;
        bool elongated = aspect >= 2.5 && area <= 1400;
        bool small_mark = area <= 120 && short_side <= 18;
        if (!thin && !elongated && !small_mark) {
            continue;
        }
        out.setTo(255, labels == label);
    }
    return out;
}

void removeFlatSurfaceScratches(cv::Mat* flat_raw_frame_bgr, const cv::Mat& green_mask, const cv::Mat& blue_mask) {
    cv::Mat color_mask;
    cv::bitwise_or(green_mask, blue_mask, color_mask);
    if (cv::countNonZero(color_mask) < 1) {
        return;
    }

    cv::Rect color_roi = cv::boundingRect(color_mask);
    if (color_roi.empty()) {
        return;
    }
    constexpr int margin = 3;
    color_roi.x = std::max(0, color_roi.x - margin);
    color_roi.y = std::max(0, color_roi.y - margin);
    color_roi.width = std::min(flat_raw_frame_bgr->cols - color_roi.x, color_roi.width + 2 * margin);
    color_roi.height = std::min(flat_raw_frame_bgr->rows - color_roi.y, color_roi.height + 2 * margin);

    cv::Mat local_color_mask = color_mask(color_roi).clone();
    cv::erode(local_color_mask, local_color_mask, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);

    cv::Mat repair_bgr;
    cv::Mat frame_roi = (*flat_raw_frame_bgr)(color_roi);
    cv::blur(frame_roi, repair_bgr, cv::Size(5, 5));

    cv::Mat frame_gray;
    cv::Mat repair_gray;
    cv::cvtColor(frame_roi, frame_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(repair_bgr, repair_gray, cv::COLOR_BGR2GRAY);
    cv::Mat max_diff;
    cv::absdiff(frame_gray, repair_gray, max_diff);

    cv::Mat scratch_mask;
    cv::threshold(max_diff, scratch_mask, 18, 255, cv::THRESH_BINARY);
    cv::bitwise_and(scratch_mask, local_color_mask, scratch_mask);
    cv::morphologyEx(scratch_mask, scratch_mask, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);
    if (cv::countNonZero(scratch_mask) < 1) {
        return;
    }

    cv::dilate(scratch_mask, scratch_mask, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);
    cv::bitwise_and(scratch_mask, local_color_mask, scratch_mask);

    repair_bgr.copyTo(frame_roi, scratch_mask);
}

cv::Point2f interpolateQuadPoint(const std::vector<cv::Point2f>& corners, float u, float v) {
    const cv::Point2f top = corners[0] * (1.0f - u) + corners[1] * u;
    const cv::Point2f bottom = corners[3] * (1.0f - u) + corners[2] * u;
    return top * (1.0f - v) + bottom * v;
}

double circularMeanHue(const cv::Mat& hsv_image, const cv::Mat& mask) {
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    int count = 0;
    for (int y = 0; y < hsv_image.rows; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        const cv::Vec3b* hsv_row = hsv_image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv_image.cols; ++x) {
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

double hueDistanceOpenCv(double a, double b) {
    double diff = std::abs(a - b);
    return std::min(diff, 180.0 - diff);
}

std::vector<cv::Point> makeCellPoly(const std::vector<cv::Point2f>& corners, int cells, int cx, int cy) {
    float u0 = static_cast<float>(cx) / static_cast<float>(cells);
    float u1 = static_cast<float>(cx + 1) / static_cast<float>(cells);
    float v0 = static_cast<float>(cy) / static_cast<float>(cells);
    float v1 = static_cast<float>(cy + 1) / static_cast<float>(cells);

    const cv::Point2f p00 = interpolateQuadPoint(corners, u0, v0);
    const cv::Point2f p10 = interpolateQuadPoint(corners, u1, v0);
    const cv::Point2f p11 = interpolateQuadPoint(corners, u1, v1);
    const cv::Point2f p01 = interpolateQuadPoint(corners, u0, v1);
    return {
        cv::Point(static_cast<int>(std::lround(p00.x)), static_cast<int>(std::lround(p00.y))),
        cv::Point(static_cast<int>(std::lround(p10.x)), static_cast<int>(std::lround(p10.y))),
        cv::Point(static_cast<int>(std::lround(p11.x)), static_cast<int>(std::lround(p11.y))),
        cv::Point(static_cast<int>(std::lround(p01.x)), static_cast<int>(std::lround(p01.y))),
    };
}

bool makeLocalCellMask(
    const std::vector<cv::Point>& poly,
    cv::Size image_size,
    cv::Rect* roi,
    cv::Mat* local_mask,
    std::vector<cv::Point>* local_poly = nullptr) {
    cv::Rect bounds = cv::boundingRect(poly) & cv::Rect(0, 0, image_size.width, image_size.height);
    if (bounds.empty()) {
        return false;
    }

    std::vector<cv::Point> shifted;
    shifted.reserve(poly.size());
    for (const auto& p : poly) {
        shifted.emplace_back(p.x - bounds.x, p.y - bounds.y);
    }

    *roi = bounds;
    *local_mask = cv::Mat(bounds.size(), CV_8UC1, cv::Scalar(0));
    cv::fillConvexPoly(*local_mask, shifted, cv::Scalar(255));
    if (local_poly != nullptr) {
        *local_poly = std::move(shifted);
    }
    return cv::countNonZero(*local_mask) > 0;
}

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

uchar homogeneityIntensityForMaskRoi(const cv::Mat& hsv_bgr_space, const cv::Mat& local_mask, const cv::Rect& roi) {
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
    const cv::Mat hsv_roi = hsv_bgr_space(roi);
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

struct QualityCellDefinition {
    cv::Rect roi;
    std::vector<cv::Point> local_poly;
    cv::Mat analysis_mask;
    cv::Mat expected_mask;
    double expected_hue = 0.0;
    bool valid = false;
};

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
                std::vector<cv::Point> poly = makeCellPoly(ref_corners, cells, cx, cy);
                cv::Rect roi;
                cv::Mat local_mask;
                std::vector<cv::Point> local_poly;
                if (!makeLocalCellMask(poly, image_size, &roi, &local_mask, &local_poly)) {
                    continue;
                }

                cv::Mat analysis_mask;
                cv::erode(local_mask, analysis_mask, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);
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

    auto add_cell = [&](const std::vector<cv::Point2f>& ref_corners, int cx, int cy) {
        std::vector<cv::Point> poly = makeCellPoly(ref_corners, cells, cx, cy);
        cv::Rect roi;
        cv::Mat local_mask;
        std::vector<cv::Point> local_poly;
        if (!makeLocalCellMask(poly, image_size, &roi, &local_mask, &local_poly)) {
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
                add_cell(ref_corners, cx, cy);
            }
        }
        for (int cy = -rings; cy < cells + rings; ++cy) {
            add_cell(ref_corners, -rings - 1, cy);
        }
    }

    return definitions;
}

uchar homogeneityIntensityForMask(const cv::Mat& hsv_bgr_space, const cv::Mat& mask) {
    std::vector<double> hues;
    std::vector<double> sats;
    std::vector<double> vals;
    int count = cv::countNonZero(mask);
    if (count < 12) {
        return 0;
    }
    hues.reserve(count);
    sats.reserve(count);
    vals.reserve(count);

    double sum_cos = 0.0;
    double sum_sin = 0.0;
    for (int y = 0; y < hsv_bgr_space.rows; ++y) {
        const uchar* mask_row = mask.ptr<uchar>(y);
        const cv::Vec3b* hsv_row = hsv_bgr_space.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv_bgr_space.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            const double h = hsv_row[x][0];
            const double angle = h * (CV_PI / 90.0);
            sum_cos += std::cos(angle);
            sum_sin += std::sin(angle);
            hues.push_back(h);
            sats.push_back(hsv_row[x][1]);
            vals.push_back(hsv_row[x][2]);
        }
    }

    double n = static_cast<double>(hues.size());
    double resultant = std::sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / std::max(1.0, n);
    resultant = std::clamp(resultant, 1e-6, 1.0);
    double hue_std = std::sqrt(-2.0 * std::log(resultant)) * (90.0 / CV_PI);
    double sat_range = percentile(sats, 0.90) - percentile(sats, 0.10);
    double val_range = percentile(vals, 0.90) - percentile(vals, 0.10);

    double score = hue_std * 5.0 + sat_range * 1.6 + val_range * 1.6;
    int intensity = 255 - static_cast<int>(std::lround(score));
    return cv::saturate_cast<uchar>(std::clamp(intensity, 0, 255));
}

cv::Mat makeTagCellHomogeneityImage(const cv::Mat& flat_raw_frame_bgr, const DetectionResult& ref_tags) {
    constexpr int cells = 8;
    cv::Mat hsv;
    cv::cvtColor(flat_raw_frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat homogeneity(flat_raw_frame_bgr.size(), CV_8UC1, cv::Scalar(0));

    for (const auto& [tag_id, ref_corners] : ref_tags.detections) {
        (void)tag_id;
        if (ref_corners.size() != 4) {
            continue;
        }
        for (int cy = 0; cy < cells; ++cy) {
            for (int cx = 0; cx < cells; ++cx) {
                std::vector<cv::Point> poly = makeCellPoly(ref_corners, cells, cx, cy);
                cv::Rect roi;
                cv::Mat local_mask;
                std::vector<cv::Point> local_poly;
                if (!makeLocalCellMask(poly, flat_raw_frame_bgr.size(), &roi, &local_mask, &local_poly)) {
                    continue;
                }
                uchar intensity = homogeneityIntensityForMaskRoi(hsv, local_mask, roi);
                cv::fillConvexPoly(homogeneity(roi), local_poly, cv::Scalar(intensity));
            }
        }
    }

    cv::Mat homogeneity_bgr;
    cv::cvtColor(homogeneity, homogeneity_bgr, cv::COLOR_GRAY2BGR);
    return homogeneity_bgr;
}

cv::Mat makeTagCellExpectedHueImage(
    const cv::Mat& flat_raw_frame_bgr,
    const cv::Mat& ref_hsv,
    const DetectionResult& ref_tags,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask) {
    constexpr int cells = 8;
    cv::Mat flat_hsv;
    cv::cvtColor(flat_raw_frame_bgr, flat_hsv, cv::COLOR_BGR2HSV);

    cv::Mat hue_match(flat_raw_frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    for (const auto& [tag_id, ref_corners] : ref_tags.detections) {
        (void)tag_id;
        if (ref_corners.size() != 4) {
            continue;
        }
        for (int cy = 0; cy < cells; ++cy) {
            for (int cx = 0; cx < cells; ++cx) {
                std::vector<cv::Point> poly = makeCellPoly(ref_corners, cells, cx, cy);
                cv::Rect roi;
                cv::Mat local_mask;
                std::vector<cv::Point> local_poly;
                if (!makeLocalCellMask(poly, flat_raw_frame_bgr.size(), &roi, &local_mask, &local_poly)) {
                    continue;
                }

                cv::Mat expected_green;
                cv::Mat expected_blue;
                cv::bitwise_and(local_mask, ref_green_mask(roi), expected_green);
                cv::bitwise_and(local_mask, ref_blue_mask(roi), expected_blue);
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

                double expected_hue = circularMeanHueRoi(ref_hsv, expected_mask, roi);
                double observed_hue = circularMeanHueRoi(flat_hsv, expected_mask, roi);
                double hue_distance = hueDistanceOpenCv(observed_hue, expected_hue);
                int intensity = 255 - static_cast<int>(std::lround(hue_distance * 5.0));
                cv::fillConvexPoly(hue_match(roi), local_poly, cv::Scalar(cv::saturate_cast<uchar>(std::clamp(intensity, 0, 255))));
            }
        }
    }

    cv::Mat hue_match_bgr;
    cv::cvtColor(hue_match, hue_match_bgr, cv::COLOR_GRAY2BGR);
    return hue_match_bgr;
}

cv::Mat combineTagCellQualityImages(const cv::Mat& homogeneity_bgr, const cv::Mat& expected_hue_bgr) {
    cv::Mat homogeneity_gray;
    cv::Mat expected_hue_gray;
    cv::cvtColor(homogeneity_bgr, homogeneity_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(expected_hue_bgr, expected_hue_gray, cv::COLOR_BGR2GRAY);

    cv::Mat combined(homogeneity_gray.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < combined.rows; ++y) {
        const uchar* h_row = homogeneity_gray.ptr<uchar>(y);
        const uchar* e_row = expected_hue_gray.ptr<uchar>(y);
        uchar* out_row = combined.ptr<uchar>(y);
        for (int x = 0; x < combined.cols; ++x) {
            double h = static_cast<double>(h_row[x]) / 255.0;
            double e = static_cast<double>(e_row[x]) / 255.0;
            out_row[x] = cv::saturate_cast<uchar>(std::lround(std::sqrt(h * e) * 255.0));
        }
    }

    cv::Mat combined_bgr;
    cv::cvtColor(combined, combined_bgr, cv::COLOR_GRAY2BGR);
    return combined_bgr;
}

cv::Mat makeTagCellCombinedQualityGrayImage(
    const cv::Mat& flat_hsv,
    const cv::Mat& ref_hsv,
    const DetectionResult& ref_tags,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask,
    const std::vector<QualityCellDefinition>* precomputed_cells = nullptr) {
    (void)ref_hsv;
    (void)ref_green_mask;
    (void)ref_blue_mask;
    cv::Mat combined(flat_hsv.size(), CV_8UC1, cv::Scalar(0));
    struct InnerCellResult {
        cv::Rect roi;
        std::vector<cv::Point> local_poly;
        uchar value = 0;
        bool valid = false;
    };

    std::vector<QualityCellDefinition> local_cells;
    const std::vector<QualityCellDefinition>* cells = precomputed_cells;
    if (cells == nullptr) {
        local_cells = buildInnerQualityCellDefinitions(
            ref_tags,
            flat_hsv.size(),
            ref_hsv,
            ref_green_mask,
            ref_blue_mask);
        cells = &local_cells;
    }

    std::vector<InnerCellResult> results(cells->size());
    cv::parallel_for_(cv::Range(0, static_cast<int>(cells->size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const QualityCellDefinition& cell = (*cells)[static_cast<size_t>(i)];
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
            InnerCellResult& result = results[static_cast<size_t>(i)];
            result.roi = cell.roi;
            result.local_poly = cell.local_poly;
            result.value = cv::saturate_cast<uchar>(
                std::lround(std::sqrt(homogeneity_score * hue_score) * brightness_score * 255.0));
            result.valid = true;
        }
    });

    for (const auto& result : results) {
        if (!result.valid) {
            continue;
        }
        cv::fillConvexPoly(combined(result.roi), result.local_poly, cv::Scalar(result.value));
    }

    return combined;
}

void addOuterGreenCellQuality(
    const cv::Mat& flat_hsv,
    const cv::Mat& ref_hsv,
    const DetectionResult& ref_tags,
    const cv::Mat& ref_green_mask,
    cv::Mat* combined_gray,
    const std::vector<QualityCellDefinition>* precomputed_cells = nullptr) {
    (void)ref_hsv;
    (void)ref_green_mask;
    struct OuterCellResult {
        cv::Rect roi;
        std::vector<cv::Point> local_poly;
        uchar value = 0;
        bool valid = false;
    };

    std::vector<QualityCellDefinition> local_cells;
    const std::vector<QualityCellDefinition>* cells = precomputed_cells;
    if (cells == nullptr) {
        local_cells = buildOuterGreenQualityCellDefinitions(
            ref_tags,
            flat_hsv.size(),
            ref_hsv,
            ref_green_mask);
        cells = &local_cells;
    }

    std::vector<OuterCellResult> results(cells->size());
    cv::parallel_for_(cv::Range(0, static_cast<int>(cells->size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const QualityCellDefinition& cell = (*cells)[static_cast<size_t>(i)];
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
            uchar combined = cv::saturate_cast<uchar>(
                std::lround(std::sqrt(homogeneity_score * hue_score) * brightness_score * 255.0));
            OuterCellResult& result = results[static_cast<size_t>(i)];
            result.roi = cell.roi;
            result.local_poly = cell.local_poly;
            result.value = combined;
            result.valid = true;
        }
    });

    for (const auto& result : results) {
        if (!result.valid) {
            continue;
        }
        cv::fillConvexPoly((*combined_gray)(result.roi), result.local_poly, cv::Scalar(result.value));
    }
}

cv::Vec3b meanBgrLocal(const cv::Mat& image_bgr, const cv::Mat& local_mask, const cv::Rect& roi) {
    cv::Scalar mean = cv::mean(image_bgr(roi), local_mask);
    return cv::Vec3b(
        cv::saturate_cast<uchar>(mean[0]),
        cv::saturate_cast<uchar>(mean[1]),
        cv::saturate_cast<uchar>(mean[2]));
}

struct FlatColorMatchCellRecord {
    cv::Rect roi;
    cv::Mat expected_mask;
    cv::Point2f center;
    bool green = false;
    int cells = 0;
    bool painted = false;
    int expected_count = 0;
    int accepted_count = 0;
    cv::Vec3b color = {0, 0, 0};
};

struct FlatColorMatchCellDefinition {
    cv::Rect roi;
    cv::Mat expected_mask;
    cv::Point2f center;
    bool green = false;
    int cells = 0;
    int expected_count = 0;
};

bool paintFlatColorMatchCell(
    const cv::Mat& flat_raw_frame_bgr,
    const cv::Mat& quality_gray,
    const cv::Mat& expected_mask,
    const cv::Rect& roi,
    cv::Mat* flat_overlay_bgr,
    cv::Mat* flat_overlay_mask,
    cv::Vec3b* sampled_color,
    int* expected_count_out,
    int* accepted_count_out) {
    constexpr int kColormatchQualityThreshold = 89;
    cv::Mat quality_ok;
    cv::threshold(quality_gray(roi), quality_ok, kColormatchQualityThreshold, 255, cv::THRESH_BINARY);

    cv::Mat accepted_mask;
    cv::bitwise_and(expected_mask, quality_ok, accepted_mask);
    int accepted_count = cv::countNonZero(accepted_mask);
    int expected_count = cv::countNonZero(expected_mask);
    if (expected_count_out != nullptr) {
        *expected_count_out = expected_count;
    }
    if (accepted_count_out != nullptr) {
        *accepted_count_out = accepted_count;
    }
    if (expected_count < 12 || accepted_count < std::max(6, static_cast<int>(std::lround(expected_count * 0.35)))) {
        return false;
    }

    cv::Vec3b color = meanBgrLocal(flat_raw_frame_bgr, accepted_mask, roi);
    if (sampled_color != nullptr) {
        *sampled_color = color;
    }
    (*flat_overlay_bgr)(roi).setTo(color, expected_mask);
    (*flat_overlay_mask)(roi).setTo(255, expected_mask);
    return true;
}

std::vector<FlatColorMatchCellDefinition> buildFlatColorMatchCellDefinitions(
    const DetectionResult& ref_tags,
    cv::Size image_size,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask) {
    std::vector<FlatColorMatchCellDefinition> definitions;

    auto addExpectedCell = [&](
        const cv::Mat& expected_mask,
        const cv::Rect& roi,
        const cv::Point2f& center,
        bool green,
        int cells) {
        int expected_count = cv::countNonZero(expected_mask);
        if (expected_count < 12) {
            return;
        }

        FlatColorMatchCellDefinition definition;
        definition.roi = roi;
        definition.expected_mask = expected_mask.clone();
        definition.center = center;
        definition.green = green;
        definition.cells = cells;
        definition.expected_count = expected_count;
        definitions.push_back(std::move(definition));
    };

    auto addCell = [&](const std::vector<cv::Point2f>& ref_corners, int cells, int cx, int cy, bool green_only) {
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

        cv::Mat expected_green;
        cv::bitwise_and(local_mask, ref_green_mask(roi), expected_green);
        if (green_only) {
            addExpectedCell(expected_green, roi, center, true, cells);
            return;
        }

        cv::Mat expected_blue;
        cv::bitwise_and(local_mask, ref_blue_mask(roi), expected_blue);
        int green_count = cv::countNonZero(expected_green);
        int blue_count = cv::countNonZero(expected_blue);
        if (blue_count >= 12) {
            addExpectedCell(expected_blue, roi, center, false, cells);
        }
        if (green_count >= 12) {
            addExpectedCell(expected_green, roi, center, true, cells);
        }
    };

    constexpr int inner_cells = 8;
    constexpr int outer_cells = 8;
    constexpr int rings = 4;
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

void fillNearestMaskedFlatColorMatchCells(
    const std::vector<FlatColorMatchCellRecord>& records,
    cv::Mat* flat_overlay_bgr,
    cv::Mat* flat_overlay_mask) {
    struct Group {
        bool green;
        int cells;
    };
    const std::vector<Group> groups = {
        {false, 8},
        {true, 8},
    };

    for (const Group& group : groups) {
        bool have_records = false;
        for (const FlatColorMatchCellRecord& record : records) {
            if (record.green == group.green && record.cells == group.cells && record.expected_count >= 12) {
                have_records = true;
                break;
            }
        }
        if (!have_records) {
            continue;
        }

        cv::Mat missing_union(flat_overlay_mask->size(), CV_8UC1, cv::Scalar(0));
        for (const FlatColorMatchCellRecord& record : records) {
            if (record.green != group.green || record.cells != group.cells || record.expected_count < 12) {
                continue;
            }
            cv::Mat already_filled;
            cv::bitwise_and((*flat_overlay_mask)(record.roi), record.expected_mask, already_filled);
            cv::Mat not_filled;
            cv::bitwise_not(already_filled, not_filled);
            cv::Mat missing_mask;
            cv::bitwise_and(record.expected_mask, not_filled, missing_mask);
            cv::bitwise_or(missing_union(record.roi), missing_mask, missing_union(record.roi));
        }

        if (cv::countNonZero(missing_union) < 1) {
            continue;
        }

        for (const FlatColorMatchCellRecord& target : records) {
            if (target.green != group.green || target.cells != group.cells || target.expected_count < 12) {
                continue;
            }

            cv::Mat already_filled;
            cv::bitwise_and((*flat_overlay_mask)(target.roi), target.expected_mask, already_filled);
            cv::Mat not_filled;
            cv::bitwise_not(already_filled, not_filled);
            cv::Mat missing_mask;
            cv::bitwise_and(target.expected_mask, not_filled, missing_mask);
            if (cv::countNonZero(missing_mask) < 1) {
                continue;
            }

            const FlatColorMatchCellRecord* best = nullptr;
            double best_dist2 = std::numeric_limits<double>::max();
            for (const FlatColorMatchCellRecord& candidate : records) {
                if (!candidate.painted || candidate.green != target.green || candidate.cells != target.cells) {
                    continue;
                }
                double dx = static_cast<double>(candidate.center.x - target.center.x);
                double dy = static_cast<double>(candidate.center.y - target.center.y);
                double dist2 = dx * dx + dy * dy;
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best = &candidate;
                }
            }

            if (best == nullptr) {
                continue;
            }
            (*flat_overlay_bgr)(target.roi).setTo(best->color, missing_mask);
            (*flat_overlay_mask)(target.roi).setTo(255, missing_mask);
        }
    }
}

cv::Mat makeAlignedRefColorMatchImage(
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& flat_aligned_ref_bgr,
    const cv::Mat& flat_raw_frame_bgr,
    const cv::Mat& flat_tag_combined_quality_gray,
    const DetectionResult& ref_tags,
    const cv::Mat& ref_green_mask,
    const cv::Mat& ref_blue_mask,
    const std::vector<FlatColorMatchCellDefinition>& cell_definitions,
    const cv::Mat& H_ref_to_frame,
    cv::Mat* flat_colormatch_bgr) {
    (void)ref_tags;
    (void)ref_green_mask;
    (void)ref_blue_mask;
    const cv::Mat& quality_gray = flat_tag_combined_quality_gray;

    cv::Mat flat_overlay_bgr(flat_raw_frame_bgr.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat flat_overlay_mask(flat_raw_frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    std::vector<FlatColorMatchCellRecord> cell_records;

    for (const FlatColorMatchCellDefinition& definition : cell_definitions) {
        FlatColorMatchCellRecord record;
        record.roi = definition.roi;
        record.expected_mask = definition.expected_mask;
        record.center = definition.center;
        record.green = definition.green;
        record.cells = definition.cells;
        record.expected_count = definition.expected_count;
        record.painted = paintFlatColorMatchCell(
            flat_raw_frame_bgr,
            quality_gray,
            definition.expected_mask,
            definition.roi,
            &flat_overlay_bgr,
            &flat_overlay_mask,
            &record.color,
            &record.expected_count,
            &record.accepted_count);
        cell_records.push_back(std::move(record));
    }
    fillNearestMaskedFlatColorMatchCells(cell_records, &flat_overlay_bgr, &flat_overlay_mask);

    if (flat_colormatch_bgr != nullptr) {
        *flat_colormatch_bgr = flat_aligned_ref_bgr.clone();
        flat_overlay_bgr.copyTo(*flat_colormatch_bgr, flat_overlay_mask);
    }

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

void buildTagColorMasks(const cv::Mat& ref_tag_patch_bgr, cv::Mat* green_mask, cv::Mat* blue_mask) {
    constexpr int margin = 6;
    cv::Mat inner(ref_tag_patch_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::rectangle(
        inner,
        cv::Rect(margin, margin, std::max(0, ref_tag_patch_bgr.cols - (2 * margin)), std::max(0, ref_tag_patch_bgr.rows - (2 * margin))),
        cv::Scalar(255),
        cv::FILLED);

    buildReferenceColorMasks(ref_tag_patch_bgr, inner, green_mask, blue_mask);

    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    cv::erode(*green_mask, *green_mask, kernel, cv::Point(-1, -1), 1);
    cv::erode(*blue_mask, *blue_mask, kernel, cv::Point(-1, -1), 1);
    cv::bitwise_and(*green_mask, inner, *green_mask);
    cv::bitwise_and(*blue_mask, inner, *blue_mask);
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
        auto frame_it = frame_tags.detections.find(id);
        if (frame_it == frame_tags.detections.end() || frame_it->second.empty()) {
            continue;
        }

        cv::Point2f center(0.0f, 0.0f);
        for (const auto& p : frame_it->second) {
            center += p;
        }
        center *= 1.0f / static_cast<float>(frame_it->second.size());

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
        cv::saturate_cast<uchar>(median(b)),
        cv::saturate_cast<uchar>(median(g)),
        cv::saturate_cast<uchar>(median(r)));
}

bool isReliableBlueBgr(const cv::Vec3b& color) {
    int b = color[0];
    int g = color[1];
    int r = color[2];
    int maxc = std::max({b, g, r});
    int minc = std::min({b, g, r});
    int chroma = maxc - minc;
    return maxc >= 155 && chroma >= 45 && b >= g + 30 && b >= r + 35;
}

bool isConsistentBlueCellColor(const cv::Vec3b& color, const cv::Vec3b& tag_blue_reference) {
    if (!isReliableBlueBgr(color)) {
        return false;
    }

    int b = color[0];
    int g = color[1];
    int r = color[2];
    int ref_b = tag_blue_reference[0];
    int ref_g = tag_blue_reference[1];
    int ref_r = tag_blue_reference[2];
    int maxc = std::max({b, g, r});
    int ref_maxc = std::max({ref_b, ref_g, ref_r});
    int sum = b + g + r;
    int ref_sum = ref_b + ref_g + ref_r;

    if (maxc < ref_maxc - 36 || sum < ref_sum - 86) {
        return false;
    }
    if (b < ref_b - 32 && g < ref_g - 26) {
        return false;
    }
    if (r < ref_r - 34 && sum < ref_sum - 60) {
        return false;
    }
    return true;
}

cv::Mat warpFrameMaskToTagPatch(
    const cv::Mat& frame_mask,
    const std::vector<cv::Point2f>& frame_corners,
    int size = 96) {
    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(size - 1), 0.0f},
        {static_cast<float>(size - 1), static_cast<float>(size - 1)},
        {0.0f, static_cast<float>(size - 1)},
    };
    cv::Mat H = cv::getPerspectiveTransform(frame_corners, dst);
    cv::Mat patch_mask;
    cv::warpPerspective(
        frame_mask,
        patch_mask,
        H,
        cv::Size(size, size),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));
    cv::threshold(patch_mask, patch_mask, 0, 255, cv::THRESH_BINARY);
    return patch_mask;
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

cv::Mat makeSurfaceGreenMask(const cv::Mat& frame_bgr, bool include_aqua) {
    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat hsv_mask;
    cv::inRange(
        hsv,
        cv::Scalar(35, 25, 70),
        cv::Scalar(include_aqua ? 115 : 95, 255, 255),
        hsv_mask);

    cv::Mat dominance(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < frame_bgr.rows; ++y) {
        const cv::Vec3b* row = frame_bgr.ptr<cv::Vec3b>(y);
        const cv::Vec3b* hsv_row = hsv.ptr<cv::Vec3b>(y);
        uchar* out = dominance.ptr<uchar>(y);
        for (int x = 0; x < frame_bgr.cols; ++x) {
            int b = row[x][0];
            int g = row[x][1];
            int r = row[x][2];
            int chroma = std::max({b, g, r}) - std::min({b, g, r});
            if (chroma >= 18 && g >= r + 6 && g >= b - 18) {
                out[x] = 255;
            }
        }
    }

    cv::Mat mask;
    cv::bitwise_and(hsv_mask, dominance, mask);
    return mask;
}

float hueDistance180(int a, int b) {
    int diff = std::abs(a - b);
    return static_cast<float>(std::min(diff, 180 - diff));
}

bool sampleTagClassColors(
    const cv::Mat& ref_bgr,
    const cv::Mat& frame_bgr,
    const std::vector<cv::Point2f>& ref_corners,
    const std::vector<cv::Point2f>& frame_corners,
    bool require_visible_color_check,
    const cv::Mat& sample_exclusion_mask,
    cv::Vec3b* sampled_green,
    cv::Vec3b* sampled_blue,
    bool* have_green,
    bool* have_blue) {
    const cv::Mat ref_patch = warpTagPatch(ref_bgr, ref_corners);
    const cv::Mat frame_patch = warpTagPatch(frame_bgr, frame_corners);

    cv::Mat tag_green_mask;
    cv::Mat tag_blue_mask;
    buildTagColorMasks(ref_patch, &tag_green_mask, &tag_blue_mask);

    cv::Mat green_sample_mask = tag_green_mask;
    cv::Mat blue_sample_mask = tag_blue_mask;
    if (require_visible_color_check) {
        green_sample_mask = makeVisibleColorSampleMask(frame_patch, tag_green_mask, true);
        blue_sample_mask = makeVisibleColorSampleMask(frame_patch, tag_blue_mask, false);
    }
    if (!sample_exclusion_mask.empty()) {
        cv::Mat exclusion_patch = warpFrameMaskToTagPatch(sample_exclusion_mask, frame_corners, frame_patch.cols);
        cv::dilate(exclusion_patch, exclusion_patch, cv::Mat::ones(5, 5, CV_8U), cv::Point(-1, -1), 1);
        green_sample_mask.setTo(0, exclusion_patch);
        blue_sample_mask.setTo(0, exclusion_patch);
    }

    const int green_count = cv::countNonZero(green_sample_mask);
    const int blue_count = cv::countNonZero(blue_sample_mask);
    const int min_green = require_visible_color_check ? 20 : 1;
    const int min_blue = require_visible_color_check ? 20 : 1;

    *have_green = green_count >= min_green;
    *have_blue = blue_count >= min_blue;
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

cv::Mat makeProjectedPatchCellMask(
    const std::vector<cv::Point2f>& ref_tag_corners,
    const cv::Mat& H_ref_to_frame,
    cv::Size frame_size,
    float x0,
    float y0,
    float x1,
    float y1);

cv::Mat makeProjectedTagAreaMask(
    const std::vector<cv::Point2f>& ref_tag_corners,
    const cv::Mat& H_ref_to_frame,
    cv::Size frame_size);

struct BlueCellCandidate {
    cv::Mat frame_cell_mask;
    cv::Vec3b color;
    double coverage = 0.0;
};

int fillBlueTagCells(
    const cv::Mat& ref_bgr,
    const cv::Mat& frame_bgr,
    const std::vector<cv::Point2f>& ref_corners,
    const std::vector<cv::Point2f>& frame_corners,
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& warped_blue_mask,
    const cv::Mat& sample_exclusion_mask,
    bool require_visible_color_check,
    cv::Mat* output_bgr) {
    (void)require_visible_color_check;
    constexpr int patch_size = 96;
    constexpr int cells = 8;
    constexpr int cell = patch_size / cells;

    const cv::Mat ref_patch = warpTagPatch(ref_bgr, ref_corners, patch_size);
    const cv::Mat frame_patch = warpTagPatch(frame_bgr, frame_corners, patch_size);

    cv::Mat tag_green_mask;
    cv::Mat tag_blue_mask;
    buildTagColorMasks(ref_patch, &tag_green_mask, &tag_blue_mask);
    cv::Mat visible_blue_mask = makeVisibleColorSampleMask(frame_patch, tag_blue_mask, false);
    if (!sample_exclusion_mask.empty()) {
        cv::Mat exclusion_patch = warpFrameMaskToTagPatch(sample_exclusion_mask, frame_corners, patch_size);
        cv::dilate(exclusion_patch, exclusion_patch, cv::Mat::ones(5, 5, CV_8U), cv::Point(-1, -1), 1);
        visible_blue_mask.setTo(0, exclusion_patch);
    }

    int filled_cells = 0;
    cv::Mat matched_blue_frame_mask(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat tag_blue_frame_mask = makeProjectedTagAreaMask(ref_corners, H_ref_to_frame, frame_bgr.size());
    cv::bitwise_and(tag_blue_frame_mask, warped_blue_mask, tag_blue_frame_mask);
    std::vector<BlueCellCandidate> candidates;
    for (int cy = 0; cy < cells; ++cy) {
        for (int cx = 0; cx < cells; ++cx) {
            cv::Mat cell_mask = cv::Mat::zeros(patch_size, patch_size, CV_8UC1);
            cv::Rect roi(cx * cell, cy * cell, cell, cell);
            cell_mask(roi).setTo(255);

            cv::Mat expected_cell_blue;
            cv::bitwise_and(tag_blue_mask, cell_mask, expected_cell_blue);
            int expected_count = cv::countNonZero(expected_cell_blue);
            if (expected_count < 4) {
                continue;
            }

            cv::Mat sample_mask;
            cv::bitwise_and(visible_blue_mask, cell_mask, sample_mask);
            int sample_count = cv::countNonZero(sample_mask);
            int min_count = std::max(12, static_cast<int>(std::lround(expected_count * 0.75)));
            if (sample_count < min_count) {
                continue;
            }

            cv::Vec3b cell_blue = medianColor(frame_patch, sample_mask);
            if (!isReliableBlueBgr(cell_blue)) {
                continue;
            }
            cv::Mat frame_cell_mask = makeProjectedPatchCellMask(
                ref_corners,
                H_ref_to_frame,
                frame_bgr.size(),
                static_cast<float>(roi.x),
                static_cast<float>(roi.y),
                static_cast<float>(roi.x + roi.width),
                static_cast<float>(roi.y + roi.height));
            cv::bitwise_and(frame_cell_mask, warped_blue_mask, frame_cell_mask);
            if (cv::countNonZero(frame_cell_mask) < 1) {
                continue;
            }

            BlueCellCandidate candidate;
            candidate.frame_cell_mask = frame_cell_mask;
            candidate.color = cell_blue;
            candidate.coverage = static_cast<double>(sample_count) / static_cast<double>(expected_count);
            candidates.push_back(candidate);
        }
    }

    if (!candidates.empty()) {
        std::vector<const BlueCellCandidate*> reference_candidates;
        for (const auto& candidate : candidates) {
            if (candidate.coverage >= 0.88) {
                reference_candidates.push_back(&candidate);
            }
        }
        if (reference_candidates.size() < 2) {
            std::vector<const BlueCellCandidate*> sorted_candidates;
            sorted_candidates.reserve(candidates.size());
            for (const auto& candidate : candidates) {
                sorted_candidates.push_back(&candidate);
            }
            std::sort(
                sorted_candidates.begin(),
                sorted_candidates.end(),
                [](const BlueCellCandidate* a, const BlueCellCandidate* b) {
                    return a->coverage > b->coverage;
                });
            const size_t keep_count = std::max<size_t>(1, (sorted_candidates.size() + 1) / 2);
            reference_candidates.assign(sorted_candidates.begin(), sorted_candidates.begin() + keep_count);
        }

        std::vector<double> ref_b;
        std::vector<double> ref_g;
        std::vector<double> ref_r;
        ref_b.reserve(reference_candidates.size());
        ref_g.reserve(reference_candidates.size());
        ref_r.reserve(reference_candidates.size());
        for (const BlueCellCandidate* candidate : reference_candidates) {
            ref_b.push_back(candidate->color[0]);
            ref_g.push_back(candidate->color[1]);
            ref_r.push_back(candidate->color[2]);
        }
        cv::Vec3b tag_blue_reference(
            cv::saturate_cast<uchar>(median(ref_b)),
            cv::saturate_cast<uchar>(median(ref_g)),
            cv::saturate_cast<uchar>(median(ref_r)));

        for (const auto& candidate : candidates) {
            if (!isConsistentBlueCellColor(candidate.color, tag_blue_reference)) {
                continue;
            }
            output_bgr->setTo(candidate.color, candidate.frame_cell_mask);
            cv::bitwise_or(matched_blue_frame_mask, candidate.frame_cell_mask, matched_blue_frame_mask);
            ++filled_cells;
        }
    }

    if (filled_cells > 0) {
        cv::Mat unmatched_blue_mask = tag_blue_frame_mask.clone();
        unmatched_blue_mask.setTo(0, matched_blue_frame_mask);
        if (cv::countNonZero(unmatched_blue_mask) > 0) {
            std::vector<cv::Point> source_points;
            source_points.reserve(cv::countNonZero(matched_blue_frame_mask));
            for (int y = 0; y < matched_blue_frame_mask.rows; ++y) {
                const uchar* row = matched_blue_frame_mask.ptr<uchar>(y);
                for (int x = 0; x < matched_blue_frame_mask.cols; ++x) {
                    if (row[x] > 0) {
                        source_points.emplace_back(x, y);
                    }
                }
            }

            cv::Mat distance_src(matched_blue_frame_mask.size(), CV_8UC1, cv::Scalar(255));
            distance_src.setTo(0, matched_blue_frame_mask);
            cv::Mat distances;
            cv::Mat labels;
            cv::distanceTransform(distance_src, distances, labels, cv::DIST_L2, 3, cv::DIST_LABEL_PIXEL);

            for (int y = 0; y < output_bgr->rows; ++y) {
                const uchar* target_row = unmatched_blue_mask.ptr<uchar>(y);
                const int* label_row = labels.ptr<int>(y);
                cv::Vec3b* out_row = output_bgr->ptr<cv::Vec3b>(y);
                for (int x = 0; x < output_bgr->cols; ++x) {
                    if (target_row[x] == 0) {
                        continue;
                    }
                    int label = label_row[x];
                    if (label <= 0 || label > static_cast<int>(source_points.size())) {
                        continue;
                    }
                    const cv::Point& src = source_points[static_cast<size_t>(label - 1)];
                    out_row[x] = output_bgr->at<cv::Vec3b>(src);
                }
            }
        }
    }

    return filled_cells;
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
            false,
            cv::Mat(),
            &sampled_green,
            &sampled_blue,
            &have_green,
            &have_blue)) {
        return false;
    }

    cv::Mat warped_green_mask;
    cv::Mat warped_blue_mask;
    buildReferenceColorMasks(*warped_ref_bgr, board_mask, &warped_green_mask, &warped_blue_mask);
    if (have_green) {
        warped_ref_bgr->setTo(sampled_green, warped_green_mask);
    }
    if (have_blue) {
        warped_ref_bgr->setTo(sampled_blue, warped_blue_mask);
    }

    if (g_info_enabled) {
        std::cout << "[INFO] Color match tag " << selected_id
                  << " green=" << static_cast<int>(sampled_green[0]) << ","
                  << static_cast<int>(sampled_green[1]) << ","
                  << static_cast<int>(sampled_green[2])
                  << " blue=" << static_cast<int>(sampled_blue[0]) << ","
                  << static_cast<int>(sampled_blue[1]) << ","
                  << static_cast<int>(sampled_blue[2]) << "\n";
    }
    return true;
}

cv::Mat warpPatchMaskToFrame(
    const cv::Mat& patch_mask,
    const std::vector<cv::Point2f>& ref_tag_corners,
    const cv::Mat& H_ref_to_frame,
    cv::Size frame_size) {
    constexpr int patch_size = 96;
    std::vector<cv::Point2f> patch_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(patch_size - 1), 0.0f},
        {static_cast<float>(patch_size - 1), static_cast<float>(patch_size - 1)},
        {0.0f, static_cast<float>(patch_size - 1)},
    };

    std::vector<cv::Point2f> projected_tag_corners;
    cv::perspectiveTransform(ref_tag_corners, projected_tag_corners, H_ref_to_frame);
    cv::Mat H_patch_to_frame = cv::getPerspectiveTransform(patch_corners, projected_tag_corners);

    cv::Mat frame_mask;
    cv::warpPerspective(
        patch_mask,
        frame_mask,
        H_patch_to_frame,
        frame_size,
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));
    return frame_mask;
}

cv::Mat makeProjectedTagAreaMask(
    const std::vector<cv::Point2f>& ref_tag_corners,
    const cv::Mat& H_ref_to_frame,
    cv::Size frame_size) {
    std::vector<cv::Point2f> projected_tag_corners;
    cv::perspectiveTransform(ref_tag_corners, projected_tag_corners, H_ref_to_frame);

    std::vector<cv::Point> poly;
    poly.reserve(projected_tag_corners.size());
    for (const auto& p : projected_tag_corners) {
        poly.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
    }

    cv::Mat mask(frame_size, CV_8UC1, cv::Scalar(0));
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return mask;
}

cv::Mat makeProjectedPatchCellMask(
    const std::vector<cv::Point2f>& ref_tag_corners,
    const cv::Mat& H_ref_to_frame,
    cv::Size frame_size,
    float x0,
    float y0,
    float x1,
    float y1) {
    constexpr int patch_size = 96;
    std::vector<cv::Point2f> patch_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(patch_size - 1), 0.0f},
        {static_cast<float>(patch_size - 1), static_cast<float>(patch_size - 1)},
        {0.0f, static_cast<float>(patch_size - 1)},
    };

    std::vector<cv::Point2f> projected_tag_corners;
    cv::perspectiveTransform(ref_tag_corners, projected_tag_corners, H_ref_to_frame);
    cv::Mat H_patch_to_frame = cv::getPerspectiveTransform(patch_corners, projected_tag_corners);

    std::vector<cv::Point2f> cell_patch = {
        {x0, y0},
        {x1, y0},
        {x1, y1},
        {x0, y1},
    };
    std::vector<cv::Point2f> cell_frame;
    cv::perspectiveTransform(cell_patch, cell_frame, H_patch_to_frame);

    std::vector<cv::Point> poly;
    poly.reserve(cell_frame.size());
    for (const auto& p : cell_frame) {
        poly.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
    }

    cv::Mat mask(frame_size, CV_8UC1, cv::Scalar(0));
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return mask;
}

cv::Mat makeShadowedTagClassSampleMask(
    const cv::Mat& frame_patch_bgr,
    const cv::Mat& expected_class_mask,
    bool green_class) {
    cv::Mat accepted(frame_patch_bgr.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < frame_patch_bgr.rows; ++y) {
        const cv::Vec3b* row = frame_patch_bgr.ptr<cv::Vec3b>(y);
        const uchar* expected_row = expected_class_mask.ptr<uchar>(y);
        uchar* accepted_row = accepted.ptr<uchar>(y);
        for (int x = 0; x < frame_patch_bgr.cols; ++x) {
            if (expected_row[x] == 0) {
                continue;
            }
            int b = row[x][0];
            int g = row[x][1];
            int r = row[x][2];
            int maxc = std::max({b, g, r});
            int minc = std::min({b, g, r});
            int chroma = maxc - minc;
            bool skin_like = r >= b + 10 && r >= g - 10 && maxc >= 65;
            bool too_dark_neutral = maxc <= 45 && chroma <= 8;
            if (skin_like || too_dark_neutral) {
                continue;
            }

            bool class_like = false;
            if (green_class) {
                class_like = g >= r + 3 && g >= b - 42 && (chroma >= 6 || maxc >= 55);
            } else {
                class_like = b >= r + 5 && b >= g - 42 && (chroma >= 6 || maxc >= 55);
            }
            if (class_like) {
                accepted_row[x] = 255;
            }
        }
    }
    cv::morphologyEx(accepted, accepted, cv::MORPH_OPEN, cv::Mat::ones(2, 2, CV_8U), cv::Point(-1, -1), 1);
    return accepted;
}

int applyProjectedTagShadowCellColorMatch(
    const cv::Mat& ref_bgr,
    const cv::Mat& frame_bgr,
    const DetectionResult& ref_tags,
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& board_mask,
    cv::Mat* output_bgr,
    cv::Mat* recolored_mask_out = nullptr) {
    if (H_ref_to_frame.empty() || output_bgr == nullptr || output_bgr->empty()) {
        return 0;
    }

    constexpr int patch_size = 96;
    constexpr int cells = 8;
    constexpr int cell = patch_size / cells;
    int recolored_cells = 0;
    if (recolored_mask_out != nullptr) {
        *recolored_mask_out = cv::Mat::zeros(frame_bgr.size(), CV_8UC1);
    }

    cv::Mat initial_difference;
    cv::absdiff(frame_bgr, *output_bgr, initial_difference);
    cv::Mat difference_value(initial_difference.size(), CV_8UC1);
    cv::parallel_for_(cv::Range(0, initial_difference.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* diff_row = initial_difference.ptr<cv::Vec3b>(y);
            uchar* value_row = difference_value.ptr<uchar>(y);
            for (int x = 0; x < initial_difference.cols; ++x) {
                value_row[x] = std::max({diff_row[x][0], diff_row[x][1], diff_row[x][2]});
            }
        }
    });
    cv::Mat high_difference;
    cv::threshold(difference_value, high_difference, 40, 255, cv::THRESH_BINARY);

    for (const auto& [tag_id, ref_corners] : ref_tags.detections) {
        (void)tag_id;
        std::vector<cv::Point2f> frame_corners;
        cv::perspectiveTransform(ref_corners, frame_corners, H_ref_to_frame);
        if (std::abs(cv::contourArea(frame_corners)) < 24.0) {
            continue;
        }

        cv::Mat tag_area_mask = makeProjectedTagAreaMask(ref_corners, H_ref_to_frame, frame_bgr.size());
        if (!board_mask.empty()) {
            cv::bitwise_and(tag_area_mask, board_mask, tag_area_mask);
        }
        int tag_area = cv::countNonZero(tag_area_mask);
        if (tag_area < 12) {
            continue;
        }
        double mean_difference = cv::mean(difference_value, tag_area_mask)[0];
        cv::Mat hot_tag_pixels;
        cv::bitwise_and(high_difference, tag_area_mask, hot_tag_pixels);
        int hot_count = cv::countNonZero(hot_tag_pixels);
        if (mean_difference < 16.0 && hot_count < static_cast<int>(std::lround(tag_area * 0.08))) {
            continue;
        }

        const cv::Mat ref_patch = warpTagPatch(ref_bgr, ref_corners, patch_size);
        const cv::Mat frame_patch = warpTagPatch(frame_bgr, frame_corners, patch_size);
        cv::Mat tag_green_mask;
        cv::Mat tag_blue_mask;
        buildTagColorMasks(ref_patch, &tag_green_mask, &tag_blue_mask);

        cv::Mat tag_overlay_patch(patch_size, patch_size, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat tag_overlay_mask(patch_size, patch_size, CV_8UC1, cv::Scalar(0));
        int tag_recolored_cells = 0;

        auto tryClassCell = [&](int cx, int cy, bool green_class) {
            cv::Rect roi(cx * cell, cy * cell, cell, cell);
            cv::Mat cell_mask = cv::Mat::zeros(patch_size, patch_size, CV_8UC1);
            cell_mask(roi).setTo(255);

            cv::Mat expected_cell;
            cv::bitwise_and(green_class ? tag_green_mask : tag_blue_mask, cell_mask, expected_cell);
            int expected_count = cv::countNonZero(expected_cell);
            if (expected_count < 4) {
                return;
            }

            cv::Mat accepted = makeShadowedTagClassSampleMask(frame_patch, expected_cell, green_class);
            int accepted_count = cv::countNonZero(accepted);
            if (accepted_count < std::max(4, static_cast<int>(std::lround(expected_count * 0.35)))) {
                return;
            }

            cv::Vec3b color = medianColor(frame_patch, accepted);
            cv::Mat write_patch_mask = accepted_count >= static_cast<int>(std::lround(expected_count * 0.65))
                ? expected_cell
                : accepted;
            tag_overlay_patch.setTo(color, write_patch_mask);
            tag_overlay_mask.setTo(255, write_patch_mask);
            ++tag_recolored_cells;
        };

        for (int cy = 0; cy < cells; ++cy) {
            for (int cx = 0; cx < cells; ++cx) {
                tryClassCell(cx, cy, true);
                tryClassCell(cx, cy, false);
            }
        }

        if (tag_recolored_cells > 0) {
            std::vector<cv::Point2f> patch_corners = {
                {0.0f, 0.0f},
                {static_cast<float>(patch_size - 1), 0.0f},
                {static_cast<float>(patch_size - 1), static_cast<float>(patch_size - 1)},
                {0.0f, static_cast<float>(patch_size - 1)},
            };
            cv::Mat H_patch_to_frame = cv::getPerspectiveTransform(patch_corners, frame_corners);
            cv::Mat frame_overlay;
            cv::Mat frame_overlay_mask;
            cv::warpPerspective(
                tag_overlay_patch,
                frame_overlay,
                H_patch_to_frame,
                frame_bgr.size(),
                cv::INTER_NEAREST,
                cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));
            cv::warpPerspective(
                tag_overlay_mask,
                frame_overlay_mask,
                H_patch_to_frame,
                frame_bgr.size(),
                cv::INTER_NEAREST,
                cv::BORDER_CONSTANT,
                cv::Scalar(0));
            if (!board_mask.empty()) {
                cv::bitwise_and(frame_overlay_mask, board_mask, frame_overlay_mask);
            }
            if (cv::countNonZero(frame_overlay_mask) > 0) {
                frame_overlay.copyTo(*output_bgr, frame_overlay_mask);
                if (recolored_mask_out != nullptr) {
                    cv::bitwise_or(*recolored_mask_out, frame_overlay_mask, *recolored_mask_out);
                }
                recolored_cells += tag_recolored_cells;
            }
        }
    }

    return recolored_cells;
}

cv::Mat makeStylusTipExclusionMask(const cv::Mat& frame_bgr, const cv::Mat& board_mask) {
    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat green_reflection;
    cv::inRange(hsv, cv::Scalar(30, 15, 20), cv::Scalar(115, 255, 230), green_reflection);

    cv::Mat dark_pen;
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(179, 170, 115), dark_pen);

    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 35, 110);
    cv::dilate(edges, edges, cv::Mat::ones(5, 5, CV_8U), cv::Point(-1, -1), 1);

    cv::Mat candidate;
    cv::bitwise_or(green_reflection, dark_pen, candidate);
    cv::bitwise_and(candidate, edges, candidate);
    if (!board_mask.empty()) {
        cv::bitwise_and(candidate, board_mask, candidate);
    }
    cv::dilate(candidate, candidate, cv::Mat::ones(9, 9, CV_8U), cv::Point(-1, -1), 1);
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE, cv::Mat::ones(15, 15, CV_8U), cv::Point(-1, -1), 1);
    cv::erode(candidate, candidate, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(candidate, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat exclusion(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    const double frame_area = static_cast<double>(frame_bgr.rows * frame_bgr.cols);
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 30.0 || area > frame_area * 0.02) {
            continue;
        }

        cv::RotatedRect rect = cv::minAreaRect(contour);
        double short_side = std::max(1.0f, std::min(rect.size.width, rect.size.height));
        double long_side = std::max(rect.size.width, rect.size.height);
        double aspect = long_side / short_side;

        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hull_area = std::max(cv::contourArea(hull), 1.0);
        double solidity = area / hull_area;

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.08 * cv::arcLength(contour, true), true);
        bool pointed_or_slender = aspect >= 2.2 || (approx.size() >= 3 && approx.size() <= 6 && solidity < 0.90);
        if (!pointed_or_slender) {
            continue;
        }

        cv::Mat contour_mask(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(contour_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
        cv::Mat dark_overlap;
        cv::Mat green_overlap;
        cv::bitwise_and(contour_mask, dark_pen, dark_overlap);
        cv::bitwise_and(contour_mask, green_reflection, green_overlap);
        if (cv::countNonZero(dark_overlap) < 25 || cv::countNonZero(green_overlap) < 3) {
            continue;
        }

        cv::drawContours(exclusion, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
    }

    if (cv::countNonZero(exclusion) > 0) {
        cv::dilate(exclusion, exclusion, cv::Mat::ones(7, 7, CV_8U), cv::Point(-1, -1), 1);
        if (!board_mask.empty()) {
            cv::bitwise_and(exclusion, board_mask, exclusion);
        }
    }

    return exclusion;
}

int fillGreenSurfaceCellsAroundTag(
    const cv::Mat& frame_bgr,
    const std::vector<cv::Point2f>& ref_tag_corners,
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& board_mask,
    const cv::Mat& pen_tip_exclusion_mask,
    cv::Mat* output_bgr,
    cv::Mat* surface_sample_mask) {
    constexpr float patch_size = 96.0f;
    constexpr float cell = patch_size / 8.0f;
    int filled_count = 0;

    cv::Mat greenish_mask = makeSurfaceGreenMask(frame_bgr, true);

    auto tryCell = [&](float x0, float y0, float x1, float y1) {
        cv::Mat cell_mask = makeProjectedPatchCellMask(
            ref_tag_corners,
            H_ref_to_frame,
            frame_bgr.size(),
            x0,
            y0,
            x1,
            y1);
        if (!board_mask.empty()) {
            cv::bitwise_and(cell_mask, board_mask, cell_mask);
        }
        if (!pen_tip_exclusion_mask.empty()) {
            cell_mask.setTo(0, pen_tip_exclusion_mask);
        }
        int cell_pixels = cv::countNonZero(cell_mask);
        if (cell_pixels < 8) {
            return;
        }

        cv::Mat green_pixels;
        cv::bitwise_and(cell_mask, greenish_mask, green_pixels);
        int green_count = cv::countNonZero(green_pixels);
        if (green_count < std::max(6, static_cast<int>(std::lround(cell_pixels * 0.35)))) {
            return;
        }

        cv::Vec3b green_color = medianColor(frame_bgr, green_pixels);
        output_bgr->setTo(green_color, green_pixels);
        if (surface_sample_mask != nullptr) {
            cv::bitwise_or(*surface_sample_mask, green_pixels, *surface_sample_mask);
        }
        ++filled_count;
    };

    for (int ring = 1; ring <= 4; ++ring) {
        float outside0 = -ring * cell;
        float outside1 = -(ring - 1) * cell;
        float lower0 = patch_size + (ring - 1) * cell;
        float lower1 = patch_size + ring * cell;
        for (int i = -ring; i < 8 + ring; ++i) {
            float a = i * cell;
            float b = (i + 1) * cell;
            tryCell(a, outside0, b, outside1);
            tryCell(a, lower0, b, lower1);
        }
        for (int i = 0; i < 8; ++i) {
            float a = i * cell;
            float b = (i + 1) * cell;
            tryCell(outside0, a, outside1, b);
            tryCell(lower0, a, lower1, b);
        }
    }

    return filled_count;
}

int nearestNeighborFillGreenSurface(
    const cv::Mat& board_mask,
    const cv::Mat& allowed_green_mask,
    const cv::Mat& pen_tip_exclusion_mask,
    const cv::Mat& surface_sample_mask,
    cv::Mat* output_bgr) {
    if (cv::countNonZero(surface_sample_mask) < 1) {
        return 0;
    }

    cv::Mat target_mask = allowed_green_mask.empty()
        ? cv::Mat(output_bgr->size(), CV_8UC1, cv::Scalar(255))
        : allowed_green_mask.clone();
    if (!board_mask.empty()) {
        cv::bitwise_and(target_mask, board_mask, target_mask);
    }
    if (!pen_tip_exclusion_mask.empty()) {
        target_mask.setTo(0, pen_tip_exclusion_mask);
    }
    target_mask.setTo(0, surface_sample_mask);

    int fill_count = cv::countNonZero(target_mask);
    if (fill_count < 1) {
        return 0;
    }

    std::vector<cv::Point> source_points;
    source_points.reserve(cv::countNonZero(surface_sample_mask));
    for (int y = 0; y < surface_sample_mask.rows; ++y) {
        const uchar* row = surface_sample_mask.ptr<uchar>(y);
        for (int x = 0; x < surface_sample_mask.cols; ++x) {
            if (row[x] > 0) {
                source_points.emplace_back(x, y);
            }
        }
    }

    cv::Mat distance_src(surface_sample_mask.size(), CV_8UC1, cv::Scalar(255));
    distance_src.setTo(0, surface_sample_mask);
    cv::Mat distances;
    cv::Mat labels;
    cv::distanceTransform(distance_src, distances, labels, cv::DIST_L2, 3, cv::DIST_LABEL_PIXEL);

    for (int y = 0; y < output_bgr->rows; ++y) {
        const uchar* target_row = target_mask.ptr<uchar>(y);
        const int* label_row = labels.ptr<int>(y);
        cv::Vec3b* out_row = output_bgr->ptr<cv::Vec3b>(y);
        for (int x = 0; x < output_bgr->cols; ++x) {
            if (target_row[x] == 0) {
                continue;
            }
            int label = label_row[x];
            if (label <= 0 || label > static_cast<int>(source_points.size())) {
                continue;
            }
            const cv::Point& src = source_points[static_cast<size_t>(label - 1)];
            out_row[x] = output_bgr->at<cv::Vec3b>(src);
        }
    }

    return fill_count;
}

int sampleConfidentGreenSurface(
    const cv::Mat& frame_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& board_mask,
    const cv::Mat& allowed_green_mask,
    const cv::Mat& pen_tip_exclusion_mask,
    cv::Mat* output_bgr,
    cv::Mat* surface_sample_mask) {
    cv::Mat confident_green = makeSurfaceGreenMask(frame_bgr, false);
    cv::Mat frame_hsv;
    cv::Mat aligned_hsv;
    cv::cvtColor(frame_bgr, frame_hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(aligned_ref_bgr, aligned_hsv, cv::COLOR_BGR2HSV);
    cv::Mat frame_lab;
    cv::Mat aligned_lab;
    cv::cvtColor(frame_bgr, frame_lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(aligned_ref_bgr, aligned_lab, cv::COLOR_BGR2Lab);

    for (int y = 0; y < confident_green.rows; ++y) {
        uchar* mask_row = confident_green.ptr<uchar>(y);
        const cv::Vec3b* frame_hsv_row = frame_hsv.ptr<cv::Vec3b>(y);
        const cv::Vec3b* ref_hsv_row = aligned_hsv.ptr<cv::Vec3b>(y);
        const cv::Vec3b* frame_lab_row = frame_lab.ptr<cv::Vec3b>(y);
        const cv::Vec3b* ref_lab_row = aligned_lab.ptr<cv::Vec3b>(y);
        const cv::Vec3b* frame_bgr_row = frame_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < confident_green.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            float hue_delta = hueDistance180(frame_hsv_row[x][0], ref_hsv_row[x][0]);
            int sat_delta = std::abs(static_cast<int>(frame_hsv_row[x][1]) - static_cast<int>(ref_hsv_row[x][1]));
            int val_delta = std::abs(static_cast<int>(frame_hsv_row[x][2]) - static_cast<int>(ref_hsv_row[x][2]));
            int dl = static_cast<int>(frame_lab_row[x][0]) - static_cast<int>(ref_lab_row[x][0]);
            int da = static_cast<int>(frame_lab_row[x][1]) - static_cast<int>(ref_lab_row[x][1]);
            int db = static_cast<int>(frame_lab_row[x][2]) - static_cast<int>(ref_lab_row[x][2]);
            float lab_distance = std::sqrt(static_cast<float>((dl * dl) + (da * da) + (db * db)));
            int b = frame_bgr_row[x][0];
            int g = frame_bgr_row[x][1];
            int r = frame_bgr_row[x][2];
            int maxc = std::max({b, g, r});
            int minc = std::min({b, g, r});
            int chroma = maxc - minc;
            bool close_enough = hue_delta <= 12.0f && sat_delta <= 70 && val_delta <= 95 && lab_distance <= 58.0f;
            bool not_gray = chroma >= 24;
            bool strong_green_surface = g >= r + 14 && g >= b - 12;
            bool plausible_ref_green = ref_hsv_row[x][0] >= 35 && ref_hsv_row[x][0] <= 105 && ref_hsv_row[x][1] >= 18 && ref_hsv_row[x][2] >= 45;
            if (!close_enough || !not_gray || !strong_green_surface || !plausible_ref_green) {
                mask_row[x] = 0;
            }
        }
    }

    if (!board_mask.empty()) {
        cv::bitwise_and(confident_green, board_mask, confident_green);
    }
    if (!allowed_green_mask.empty()) {
        cv::bitwise_and(confident_green, allowed_green_mask, confident_green);
    }
    if (!pen_tip_exclusion_mask.empty()) {
        confident_green.setTo(0, pen_tip_exclusion_mask);
    }
    if (surface_sample_mask != nullptr && !surface_sample_mask->empty()) {
        confident_green.setTo(0, *surface_sample_mask);
    }

    cv::morphologyEx(confident_green, confident_green, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);

    cv::Mat accepted_squares = cv::Mat::zeros(confident_green.size(), CV_8UC1);
    constexpr int square_size = 8;
    constexpr int stride = 8;
    for (int y = 0; y < confident_green.rows; y += stride) {
        for (int x = 0; x < confident_green.cols; x += stride) {
            cv::Rect roi(
                x,
                y,
                std::min(square_size, confident_green.cols - x),
                std::min(square_size, confident_green.rows - y));
            if (roi.width < 4 || roi.height < 4) {
                continue;
            }

            cv::Mat valid_roi = allowed_green_mask.empty()
                ? cv::Mat(roi.size(), CV_8UC1, cv::Scalar(255))
                : allowed_green_mask(roi).clone();
            if (!board_mask.empty()) {
                cv::bitwise_and(valid_roi, board_mask(roi), valid_roi);
            }
            if (!pen_tip_exclusion_mask.empty()) {
                valid_roi.setTo(0, pen_tip_exclusion_mask(roi));
            }
            if (surface_sample_mask != nullptr && !surface_sample_mask->empty()) {
                valid_roi.setTo(0, (*surface_sample_mask)(roi));
            }

            int valid_count = cv::countNonZero(valid_roi);
            if (valid_count < std::max(8, static_cast<int>(std::lround(roi.area() * 0.45)))) {
                continue;
            }

            cv::Mat confident_roi;
            cv::bitwise_and(confident_green(roi), valid_roi, confident_roi);
            int confident_count = cv::countNonZero(confident_roi);
            if (confident_count < valid_count) {
                continue;
            }

            accepted_squares(roi).setTo(255, valid_roi);
        }
    }

    confident_green = accepted_squares;
    int sample_count = cv::countNonZero(confident_green);
    if (sample_count < 1) {
        return 0;
    }

    frame_bgr.copyTo(*output_bgr, confident_green);
    if (surface_sample_mask != nullptr) {
        cv::bitwise_or(*surface_sample_mask, confident_green, *surface_sample_mask);
    }
    return sample_count;
}

void blurNonBlueRegions(cv::Mat* image_bgr, const cv::Mat& blue_lock_mask, const cv::Mat& board_mask) {
    cv::Mat blur_mask;
    if (board_mask.empty()) {
        blur_mask = cv::Mat(image_bgr->size(), CV_8UC1, cv::Scalar(255));
    } else {
        cv::threshold(board_mask, blur_mask, 0, 255, cv::THRESH_BINARY);
    }
    if (!blue_lock_mask.empty()) {
        blur_mask.setTo(0, blue_lock_mask);
    }

    if (cv::countNonZero(blur_mask) < 1) {
        return;
    }

    cv::Mat original = image_bgr->clone();
    cv::Mat image_f;
    image_bgr->convertTo(image_f, CV_32FC3);

    cv::Mat weight;
    blur_mask.convertTo(weight, CV_32F, 1.0 / 255.0);
    cv::Mat blurred_weight;
    cv::GaussianBlur(weight, blurred_weight, cv::Size(0, 0), 5.0);

    std::vector<cv::Mat> channels;
    cv::split(image_f, channels);
    for (cv::Mat& channel : channels) {
        cv::Mat weighted;
        cv::multiply(channel, weight, weighted);
        cv::Mat blurred_weighted;
        cv::GaussianBlur(weighted, blurred_weighted, cv::Size(0, 0), 5.0);
        cv::Mat safe_den;
        cv::max(blurred_weight, 1e-3, safe_den);
        cv::divide(blurred_weighted, safe_den, channel);
    }

    cv::Mat blurred_f;
    cv::merge(channels, blurred_f);
    cv::Mat blurred_u8;
    blurred_f.convertTo(blurred_u8, CV_8UC3);
    blurred_u8.copyTo(*image_bgr, blur_mask);
    if (!blue_lock_mask.empty()) {
        original.copyTo(*image_bgr, blue_lock_mask);
    }
}

struct BlueTagFillInfo {
    int id = -1;
    cv::Point2f center;
    cv::Mat mask;
    cv::Vec3b color;
};

cv::Point2f averagePoint(const std::vector<cv::Point2f>& points) {
    cv::Point2f center(0.0f, 0.0f);
    if (points.empty()) {
        return center;
    }
    for (const auto& point : points) {
        center += point;
    }
    center *= 1.0f / static_cast<float>(points.size());
    return center;
}

cv::Vec3b meanColorBgr(const cv::Mat& image_bgr, const cv::Mat& mask) {
    cv::Scalar mean = cv::mean(image_bgr, mask);
    return cv::Vec3b(
        cv::saturate_cast<uchar>(mean[0]),
        cv::saturate_cast<uchar>(mean[1]),
        cv::saturate_cast<uchar>(mean[2]));
}

int applyPerVisibleTagColorMatch(
    const cv::Mat& ref_bgr,
    const cv::Mat& frame_bgr,
    const DetectionResult& ref_tags,
    const DetectionResult& frame_tags,
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& fill_mask_source_bgr,
    const cv::Mat& board_mask,
    const fs::path& debug_prefix,
    cv::Mat* warped_ref_bgr) {
    int recolored_count = 0;
    cv::Mat base_aligned_ref_bgr = warped_ref_bgr->clone();
    cv::Mat warped_green_mask;
    cv::Mat warped_blue_mask;
    buildReferenceColorMasks(fill_mask_source_bgr, board_mask, &warped_green_mask, &warped_blue_mask);
    int projected_fallback_count = 0;
    int green_surface_cell_count = 0;
    std::vector<BlueTagFillInfo> touched_blue_tags;
    std::vector<BlueTagFillInfo> untouched_blue_tags;
    cv::Mat surface_sample_mask(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat pen_tip_exclusion_mask;
    cv::Mat nearest_fill_exclusion_mask;

    for (const auto& [tag_id, ref_corners] : ref_tags.detections) {
        auto frame_it = frame_tags.detections.find(tag_id);
        bool detected = frame_it != frame_tags.detections.end();
        std::vector<cv::Point2f> sample_frame_corners;
        if (detected) {
            sample_frame_corners = frame_it->second;
        } else {
            cv::perspectiveTransform(ref_corners, sample_frame_corners, H_ref_to_frame);
        }

        cv::Mat tag_area_mask = makeProjectedTagAreaMask(ref_corners, H_ref_to_frame, frame_bgr.size());
        cv::Mat frame_blue_mask;
        cv::bitwise_and(warped_blue_mask, tag_area_mask, frame_blue_mask);
        BlueTagFillInfo blue_tag_info;
        blue_tag_info.id = tag_id;
        blue_tag_info.center = averagePoint(sample_frame_corners);
        blue_tag_info.mask = frame_blue_mask;

        cv::Vec3b sampled_green(0, 0, 0);
        cv::Vec3b sampled_blue(0, 0, 0);
        bool have_green = false;
        bool have_blue = false;
        if (!sampleTagClassColors(
                ref_bgr,
                frame_bgr,
                ref_corners,
                sample_frame_corners,
                true,
                pen_tip_exclusion_mask,
                &sampled_green,
                &sampled_blue,
                &have_green,
                &have_blue)) {
            if (cv::countNonZero(frame_blue_mask) > 0) {
                untouched_blue_tags.push_back(blue_tag_info);
            }
            continue;
        }

        if (have_green) {
            cv::Mat frame_green_mask;
            cv::bitwise_and(warped_green_mask, tag_area_mask, frame_green_mask);
            warped_ref_bgr->setTo(sampled_green, frame_green_mask);
        }
        bool touched_blue = false;
        if (have_blue) {
            int blue_cells = fillBlueTagCells(
                ref_bgr,
                frame_bgr,
                ref_corners,
                sample_frame_corners,
                H_ref_to_frame,
                warped_blue_mask,
                pen_tip_exclusion_mask,
                true,
                warped_ref_bgr);
            if (blue_cells == 0) {
                if (isReliableBlueBgr(sampled_blue)) {
                    warped_ref_bgr->setTo(sampled_blue, frame_blue_mask);
                    touched_blue = cv::countNonZero(frame_blue_mask) > 0;
                }
            } else {
                touched_blue = true;
            }
        }
        if (cv::countNonZero(frame_blue_mask) > 0) {
            if (touched_blue) {
                blue_tag_info.color = meanColorBgr(*warped_ref_bgr, frame_blue_mask);
                touched_blue_tags.push_back(blue_tag_info);
            } else {
                untouched_blue_tags.push_back(blue_tag_info);
            }
        }
        green_surface_cell_count += fillGreenSurfaceCellsAroundTag(
            frame_bgr,
            ref_corners,
            H_ref_to_frame,
            board_mask,
            pen_tip_exclusion_mask,
            warped_ref_bgr,
            &surface_sample_mask);

        ++recolored_count;
        if (!detected) {
            ++projected_fallback_count;
        }
    }

    int nearest_blue_fill_count = 0;
    if (!touched_blue_tags.empty()) {
        for (const auto& untouched : untouched_blue_tags) {
            const BlueTagFillInfo* nearest = nullptr;
            float best_dist2 = std::numeric_limits<float>::max();
            for (const auto& touched : touched_blue_tags) {
                cv::Point2f delta = untouched.center - touched.center;
                float dist2 = delta.dot(delta);
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    nearest = &touched;
                }
            }
            if (nearest == nullptr || cv::countNonZero(untouched.mask) < 1) {
                continue;
            }
            warped_ref_bgr->setTo(nearest->color, untouched.mask);
            ++nearest_blue_fill_count;
        }
    }
    if (nearest_blue_fill_count > 0 && g_info_enabled) {
        std::cout << "[INFO] Untouched blue tag regions filled from nearest touched tag "
                  << nearest_blue_fill_count << " tags\n";
    }

    if (projected_fallback_count > 0 && g_info_enabled) {
        std::cout << "[INFO] Per-tag projected fallback recolored "
                  << projected_fallback_count << " non-detected tags\n";
    }
    if (green_surface_cell_count > 0 && g_info_enabled) {
        std::cout << "[INFO] Per-tag green surface cells filled "
                  << green_surface_cell_count << " cells\n";
    }
    if (!debug_prefix.empty()) {
        cv::imwrite(debugPath(debug_prefix, "_debug_after_tag_surface_cells.png").string(), *warped_ref_bgr);
        cv::imwrite(debugPath(debug_prefix, "_debug_surface_sample_mask_after_tag_cells.png").string(), surface_sample_mask);
    }
    int confident_green_count = sampleConfidentGreenSurface(
        frame_bgr,
        base_aligned_ref_bgr,
        board_mask,
        warped_green_mask,
        pen_tip_exclusion_mask,
        warped_ref_bgr,
        &surface_sample_mask);
    if (confident_green_count > 0 && g_info_enabled) {
        std::cout << "[INFO] Confident raw green surface pixels sampled "
                  << confident_green_count << " pixels\n";
    }
    if (!debug_prefix.empty()) {
        cv::imwrite(debugPath(debug_prefix, "_debug_after_confident_surface.png").string(), *warped_ref_bgr);
        cv::imwrite(debugPath(debug_prefix, "_debug_surface_sample_mask_after_confident.png").string(), surface_sample_mask);
    }
    int nearest_fill_count = nearestNeighborFillGreenSurface(
        board_mask,
        warped_green_mask,
        nearest_fill_exclusion_mask,
        surface_sample_mask,
        warped_ref_bgr);
    if (nearest_fill_count > 0 && g_info_enabled) {
        std::cout << "[INFO] Per-tag nearest-neighbor green surface pixels filled "
                  << nearest_fill_count << " pixels\n";
    }
    if (!debug_prefix.empty()) {
        cv::imwrite(debugPath(debug_prefix, "_debug_after_nearest_green_fill.png").string(), *warped_ref_bgr);
    }
    blurNonBlueRegions(warped_ref_bgr, warped_blue_mask, board_mask);
    if (!debug_prefix.empty()) {
        cv::imwrite(debugPath(debug_prefix, "_debug_after_non_blue_blur.png").string(), *warped_ref_bgr);
    }
    return recolored_count;
}

cv::Mat robustChannelGain(const cv::Mat& frame_bgr, const cv::Mat& warped_ref_bgr, const cv::Mat& mask) {
    cv::Mat aligned_f;
    warped_ref_bgr.convertTo(aligned_f, CV_32FC3);

    int valid_count = cv::countNonZero(mask);
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
            const uchar* mask_row = mask.ptr<uchar>(y);
            const cv::Vec3b* frame_row = frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* ref_row = warped_ref_bgr.ptr<cv::Vec3b>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                if (mask_row[x] == 0 || ref_row[x][c] <= 10) {
                    continue;
                }
                frame_vals.push_back(frame_row[x][c]);
                ref_vals.push_back(ref_row[x][c]);
            }
        }

        if (frame_vals.size() < 100 || ref_vals.size() < 100) {
            continue;
        }
        double frame_med = median(frame_vals);
        double ref_med = std::max(median(ref_vals), 1e-6);
        double gain = std::clamp(frame_med / ref_med, 0.5, 2.0);
        aligned_channels[c] *= gain;
    }

    cv::merge(aligned_channels, aligned_f);
    cv::Mat aligned_u8;
    aligned_f.convertTo(aligned_u8, CV_8UC3);
    return aligned_u8;
}

cv::Mat differenceMagnitudeMask(const cv::Mat& difference_bgr, int threshold) {
    std::vector<cv::Mat> channels;
    cv::split(difference_bgr, channels);
    cv::Mat magnitude;
    cv::max(channels[0], channels[1], magnitude);
    cv::max(magnitude, channels[2], magnitude);
    cv::GaussianBlur(magnitude, magnitude, cv::Size(3, 3), 0);
    cv::threshold(magnitude, magnitude, threshold, 255, cv::THRESH_BINARY);
    return magnitude;
}

cv::Mat makeSkinSeedMask(const cv::Mat& frame_bgr) {
    cv::Mat hsv;
    cv::Mat ycrcb;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame_bgr, ycrcb, cv::COLOR_BGR2YCrCb);

    cv::Mat ycrcb_skin;
    cv::inRange(ycrcb, cv::Scalar(35, 132, 72), cv::Scalar(255, 182, 142), ycrcb_skin);

    cv::Mat hsv_skin_low;
    cv::Mat hsv_skin_high;
    cv::inRange(hsv, cv::Scalar(0, 18, 55), cv::Scalar(25, 190, 255), hsv_skin_low);
    cv::inRange(hsv, cv::Scalar(165, 18, 55), cv::Scalar(179, 190, 255), hsv_skin_high);
    cv::Mat hsv_skin;
    cv::bitwise_or(hsv_skin_low, hsv_skin_high, hsv_skin);

    cv::Mat skin;
    cv::bitwise_and(ycrcb_skin, hsv_skin, skin);
    cv::morphologyEx(skin, skin, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);
    return skin;
}

cv::Mat makeDarkPenSeedMask(const cv::Mat& frame_bgr, const cv::Mat& surface_green_mask) {
    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 35, 110);
    cv::dilate(edges, edges, cv::Mat::ones(5, 5, CV_8U), cv::Point(-1, -1), 1);

    cv::Mat dark(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < frame_bgr.rows; ++y) {
        const cv::Vec3b* row = frame_bgr.ptr<cv::Vec3b>(y);
        const uchar* surface_row = surface_green_mask.empty() ? nullptr : surface_green_mask.ptr<uchar>(y);
        const uchar* edge_row = edges.ptr<uchar>(y);
        uchar* out = dark.ptr<uchar>(y);
        for (int x = 0; x < frame_bgr.cols; ++x) {
            int b = row[x][0];
            int g = row[x][1];
            int r = row[x][2];
            int maxc = std::max({b, g, r});
            int minc = std::min({b, g, r});
            int chroma = maxc - minc;
            bool surface_like = surface_row != nullptr && surface_row[x] > 0;
            bool dark_object = maxc <= 95 && (chroma <= 70 || maxc <= 70) && !surface_like && edge_row[x] > 0;
            if (dark_object) {
                out[x] = 255;
            }
        }
    }
    cv::morphologyEx(dark, dark, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);
    return dark;
}

cv::Mat growSeedIntoMask(const cv::Mat& seed, const cv::Mat& grow_region, int iterations) {
    cv::Mat grown;
    cv::bitwise_and(seed, grow_region, grown);
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
    for (int i = 0; i < iterations; ++i) {
        cv::Mat dilated;
        cv::dilate(grown, dilated, kernel, cv::Point(-1, -1), 1);
        cv::bitwise_and(dilated, grow_region, grown);
    }
    return grown;
}

cv::Mat keepLargeComponents(const cv::Mat& mask, int min_area) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_area) {
            continue;
        }
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        float aspect = static_cast<float>(std::max(width, height)) / static_cast<float>(std::max(1, std::min(width, height)));
        if (width < 18 || height < 24 || aspect > 10.0f) {
            continue;
        }
        out.setTo(255, labels == label);
    }
    return out;
}

cv::Mat fillInternalMaskHoles(const cv::Mat& mask) {
    cv::Mat inverted;
    cv::bitwise_not(mask, inverted);
    cv::Mat flood = inverted.clone();
    cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(0));
    cv::Mat holes;
    cv::bitwise_and(flood, inverted, holes);
    cv::Mat filled;
    cv::bitwise_or(mask, holes, filled);
    return filled;
}

void fillThinHorizontalMaskGaps(cv::Mat* mask, int max_gap_width) {
    for (int y = 0; y < mask->rows; ++y) {
        uchar* row = mask->ptr<uchar>(y);
        int x = 0;
        while (x < mask->cols) {
            while (x < mask->cols && row[x] > 0) {
                ++x;
            }
            int start = x;
            while (x < mask->cols && row[x] == 0) {
                ++x;
            }
            int end = x;
            int gap_width = end - start;
            if (start > 0 && end < mask->cols && gap_width > 0 && gap_width <= max_gap_width && row[start - 1] > 0 && row[end] > 0) {
                for (int i = start; i < end; ++i) {
                    row[i] = 255;
                }
            }
        }
    }
}

cv::Mat makeSurfaceMaskFromAlignedRef(const cv::Mat& aligned_ref_bgr) {
    cv::Mat gray;
    cv::cvtColor(aligned_ref_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat raw;
    cv::threshold(gray, raw, 5, 255, cv::THRESH_BINARY);
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(raw, raw, cv::MORPH_CLOSE, close_kernel, cv::Point(-1, -1), 2);

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
    if (count <= 1) {
        return cv::Mat::zeros(mask.size(), CV_8UC1);
    }

    int best_label = -1;
    int best_area = 0;
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > best_area) {
            best_area = area;
            best_label = label;
        }
    }
    if (best_label < 0 || best_area < min_area) {
        return cv::Mat::zeros(mask.size(), CV_8UC1);
    }
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8UC1);
    out.setTo(255, labels == best_label);
    return out;
}

cv::Mat makeTagMaskFromColoredAlignedRef(const cv::Mat& aligned_ref_bgr);

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
                bool dark_neutral_stylus_like = maxc <= 90 && chroma <= 28;
                bool green_or_aqua = chroma >= 10 && g >= r + 4 && g >= b - 35;
                bool blue_or_cyan = chroma >= 15 && b >= r + 12 && b >= g - 30;
                if ((green_or_aqua || blue_or_cyan) && !skin_like && !dark_neutral_stylus_like) {
                    out[x] = 255;
                }
            }
        }
    });

    cv::Mat visible_board;
    cv::bitwise_and(hsv_mask, dominance, visible_board);
    return visible_board;
}

cv::Mat makeVisibleReferenceTagProtectionMask(
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& observed_bgr,
    const cv::Mat& surface_mask) {
    cv::Mat tag_region = makeTagMaskFromColoredAlignedRef(aligned_ref_bgr);
    cv::bitwise_and(tag_region, surface_mask, tag_region);

    cv::Mat protected_tag(tag_region.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, observed_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* row = observed_bgr.ptr<cv::Vec3b>(y);
            const uchar* tag_row = tag_region.ptr<uchar>(y);
            uchar* out = protected_tag.ptr<uchar>(y);
            for (int x = 0; x < observed_bgr.cols; ++x) {
                if (tag_row[x] == 0) {
                    continue;
                }
                int b = row[x][0];
                int g = row[x][1];
                int r = row[x][2];
                int maxc = std::max({b, g, r});
                int minc = std::min({b, g, r});
                int chroma = maxc - minc;
                bool skin_like = r >= b + 8 && r >= g - 12 && maxc >= 70;
                bool dark_neutral_stylus_like = maxc <= 90 && chroma <= 28;
                bool blue_tag_like = chroma >= 18 && b >= r + 14 && b >= g - 18;
                bool green_tag_like = chroma >= 16 && g >= r + 6 && g >= b - 28;
                if ((blue_tag_like || green_tag_like) && !skin_like && !dark_neutral_stylus_like) {
                    out[x] = 255;
                }
            }
        }
    });

    cv::morphologyEx(
        protected_tag,
        protected_tag,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    return protected_tag;
}

cv::Mat makeReferenceColoredSurfaceProtectionMask(
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& observed_bgr,
    const cv::Mat& surface_mask) {
    cv::Mat protected_surface(surface_mask.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, observed_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* ref_row = aligned_ref_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* obs_row = observed_bgr.ptr<cv::Vec3b>(y);
            const uchar* surface_row = surface_mask.ptr<uchar>(y);
            uchar* out = protected_surface.ptr<uchar>(y);
            for (int x = 0; x < observed_bgr.cols; ++x) {
                if (surface_row[x] == 0) {
                    continue;
                }
                int ref_b = ref_row[x][0];
                int ref_g = ref_row[x][1];
                int ref_r = ref_row[x][2];
                int ref_max = std::max({ref_b, ref_g, ref_r});
                int ref_min = std::min({ref_b, ref_g, ref_r});
                int ref_chroma = ref_max - ref_min;
                bool ref_green = ref_chroma >= 12 && ref_g >= ref_r + 5 && ref_g >= ref_b - 35;
                bool ref_blue = ref_chroma >= 18 && ref_b >= ref_r + 15 && ref_b >= ref_g - 25;
                if (!ref_green && !ref_blue) {
                    continue;
                }

                int b = obs_row[x][0];
                int g = obs_row[x][1];
                int r = obs_row[x][2];
                int maxc = std::max({b, g, r});
                int minc = std::min({b, g, r});
                int chroma = maxc - minc;
                bool skin_like = r >= b + 8 && r >= g - 12 && maxc >= 70;
                bool observed_green = chroma >= 6 && g >= r + 3 && g >= b - 42;
                bool observed_blue = chroma >= 8 && b >= r + 8 && b >= g - 35;
                bool same_family = (ref_green && observed_green) || (ref_blue && observed_blue);
                if (same_family && !skin_like) {
                    out[x] = 255;
                }
            }
        }
    });
    cv::morphologyEx(
        protected_surface,
        protected_surface,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    return protected_surface;
}

bool isStylusTipLikeComponent(const cv::Mat& stats, int label, int min_area) {
    int area = stats.at<int>(label, cv::CC_STAT_AREA);
    int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    if (area < min_area || area > 2500 || width < 14 || height < 10 || width > 130 || height > 95) {
        return false;
    }

    int short_side = std::max(1, std::min(width, height));
    int long_side = std::max(width, height);
    double aspect = static_cast<double>(long_side) / static_cast<double>(short_side);
    double fill_ratio = static_cast<double>(area) / static_cast<double>(std::max(1, width * height));
    return aspect >= 1.25 && aspect <= 4.5 && fill_ratio >= 0.18 && fill_ratio <= 0.78;
}

cv::Mat makeDarkStylusTipPreserveMask(
    const cv::Mat& frame_bgr,
    const cv::Mat& hand_candidate,
    const cv::Mat& visible_board) {
    cv::Mat seed(frame_bgr.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* row = frame_bgr.ptr<cv::Vec3b>(y);
            const uchar* candidate_row = hand_candidate.ptr<uchar>(y);
            uchar* out = seed.ptr<uchar>(y);
            for (int x = 0; x < frame_bgr.cols; ++x) {
                if (candidate_row[x] == 0) {
                    continue;
                }
                int b = row[x][0];
                int g = row[x][1];
                int r = row[x][2];
                int maxc = std::max({b, g, r});
                int minc = std::min({b, g, r});
                int chroma = maxc - minc;
                bool blue_tag_like = b > g + 15 && b > r + 15;
                bool neutral_dark_tip = maxc <= 95 && chroma <= 25;
                bool colored_dark_tip = maxc <= 125 && minc <= 105 && !blue_tag_like;
                if ((neutral_dark_tip || colored_dark_tip) && !blue_tag_like) {
                    out[x] = 255;
                }
            }
        }
    });

    cv::morphologyEx(seed, seed, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 1);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int count = cv::connectedComponentsWithStats(seed, labels, stats, centroids, 8);
    cv::Mat selected(seed.size(), CV_8UC1, cv::Scalar(0));
    int large_min_area = std::max(120, static_cast<int>(std::lround(frame_bgr.total() * 0.0015)));
    int tip_like_min_area = std::clamp(
        envInt("SHADOW_HAND_TIP_SHAPE_MIN_AREA", static_cast<int>(std::lround(frame_bgr.total() * 0.00025))),
        80,
        large_min_area);
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= large_min_area || isStylusTipLikeComponent(stats, label, tip_like_min_area)) {
            selected.setTo(255, labels == label);
        }
    }
    if (cv::countNonZero(selected) == 0) {
        return selected;
    }

    cv::Mat protected_candidate;
    cv::bitwise_and(visible_board, hand_candidate, protected_candidate);
    cv::Mat grow_region;
    cv::bitwise_or(protected_candidate, selected, grow_region);
    cv::Mat grown = growSeedIntoMask(
        selected,
        grow_region,
        std::clamp(envInt("SHADOW_HAND_TIP_PRESERVE_ITERS", 6), 0, 12));
    cv::bitwise_and(grown, visible_board, grown);
    return grown;
}

cv::Mat makeHandPenMaskFromColormatchDifference(
    const cv::Mat& difference_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat* observed_bgr,
    const cv::Mat* precomputed_surface_mask,
    const cv::Mat* projected_tag_protection_mask,
    double* otsu_threshold,
    int* used_threshold) {
    cv::Mat surface = precomputed_surface_mask != nullptr && !precomputed_surface_mask->empty()
        ? *precomputed_surface_mask
        : makeSurfaceMaskFromAlignedRef(aligned_ref_bgr);
    cv::Mat value(difference_bgr.size(), CV_8UC1);
    cv::parallel_for_(cv::Range(0, difference_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* src_row = difference_bgr.ptr<cv::Vec3b>(y);
            uchar* value_row = value.ptr<uchar>(y);
            for (int x = 0; x < difference_bgr.cols; ++x) {
                value_row[x] = std::max({src_row[x][0], src_row[x][1], src_row[x][2]});
            }
        }
    });

    int surface_count = cv::countNonZero(surface);
    double otsu_t = 45.0;
    if (surface_count > 0) {
        cv::Mat values(surface_count, 1, CV_8UC1);
        int idx = 0;
        for (int y = 0; y < value.rows; ++y) {
            const uchar* surface_row = surface.ptr<uchar>(y);
            const uchar* value_row = value.ptr<uchar>(y);
            for (int x = 0; x < value.cols; ++x) {
                if (surface_row[x] == 0) {
                    continue;
                }
                values.at<uchar>(idx++, 0) = value_row[x];
            }
        }
        cv::Mat thresholded;
        otsu_t = cv::threshold(values, thresholded, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    const double min_otsu_for_mask = static_cast<double>(envInt("SHADOW_HAND_MIN_OTSU", 35));
    if (otsu_t < min_otsu_for_mask) {
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
    if (envInt("SHADOW_HAND_PROTECT_VISIBLE_BOARD", 1) != 0
            && observed_bgr != nullptr
            && !observed_bgr->empty()
            && observed_bgr->size() == difference_bgr.size()) {
        visible_board = makeVisibleBoardColorProtectionMask(*observed_bgr);
        cv::bitwise_and(visible_board, surface, visible_board);
        cv::Mat visible_reference_tags = makeVisibleReferenceTagProtectionMask(
            aligned_ref_bgr,
            *observed_bgr,
            surface);
        cv::bitwise_or(visible_board, visible_reference_tags, visible_board);
        cv::Mat reference_colored_surface = makeReferenceColoredSurfaceProtectionMask(
            aligned_ref_bgr,
            *observed_bgr,
            surface);
        cv::bitwise_or(visible_board, reference_colored_surface, visible_board);
        if (projected_tag_protection_mask != nullptr &&
                !projected_tag_protection_mask->empty() &&
                projected_tag_protection_mask->size() == difference_bgr.size()) {
            cv::Mat projected_tag_protection;
            cv::bitwise_and(*projected_tag_protection_mask, surface, projected_tag_protection);
            cv::bitwise_or(visible_board, projected_tag_protection, visible_board);
        }
        if (envInt("SHADOW_HAND_TIP_PRESERVE", 1) != 0) {
            cv::Mat tip_preserve = makeDarkStylusTipPreserveMask(*observed_bgr, candidate, visible_board);
            visible_board.setTo(0, tip_preserve);
        }
    }

    cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(candidate, candidate, cv::MORPH_OPEN, open_kernel, cv::Point(-1, -1), 1);
    if (!visible_board.empty()) {
        candidate.setTo(0, visible_board);
    }

    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE, close_kernel, cv::Point(-1, -1), 1);
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

cv::Mat makeHandPenMask(const cv::Mat& difference_bgr) {
    cv::Mat aligned_ref_like;
    cv::cvtColor(difference_bgr, aligned_ref_like, cv::COLOR_BGR2GRAY);
    cv::threshold(aligned_ref_like, aligned_ref_like, 0, 255, cv::THRESH_BINARY);
    cv::cvtColor(aligned_ref_like, aligned_ref_like, cv::COLOR_GRAY2BGR);
    return makeHandPenMaskFromColormatchDifference(difference_bgr, aligned_ref_like, nullptr, nullptr, nullptr, nullptr, nullptr);
}

cv::Mat fitTextureToReference(const cv::Mat& texture_bgr, cv::Size ref_size) {
    if (texture_bgr.empty() || texture_bgr.cols <= 0 || texture_bgr.rows <= 0) {
        throw std::runtime_error("Texture image has invalid dimensions");
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
    cv::Rect crop(x0, y0, std::min(ref_size.width, resized.cols - x0), std::min(ref_size.height, resized.rows - y0));
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
    std::string ext = lowerString(path.extension().string());
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

cv::Mat makeSurfaceTextureOutput(
    const cv::Mat& frame_bgr,
    const cv::Mat& surface_texture_ref_bgr,
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& board_mask,
    const cv::Mat& hand_stylus_mask) {
    cv::Mat warped_texture;
    cv::warpPerspective(
        surface_texture_ref_bgr,
        warped_texture,
        H_ref_to_frame,
        frame_bgr.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101);

    cv::Mat board_render_mask;
    cv::threshold(board_mask, board_render_mask, 0, 255, cv::THRESH_BINARY);
    cv::morphologyEx(
        board_render_mask,
        board_render_mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    cv::dilate(
        board_render_mask,
        board_render_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)),
        cv::Point(-1, -1),
        1);

    cv::Mat occlusion_render_mask;
    cv::threshold(hand_stylus_mask, occlusion_render_mask, 0, 255, cv::THRESH_BINARY);
    cv::dilate(
        occlusion_render_mask,
        occlusion_render_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)),
        cv::Point(-1, -1),
        1);
    cv::bitwise_and(occlusion_render_mask, board_render_mask, occlusion_render_mask);

    cv::Mat output = frame_bgr.clone();
    warped_texture.copyTo(output, board_render_mask);
    frame_bgr.copyTo(output, occlusion_render_mask);
    return output;
}

void makeSurfaceTextureRenderMasks(
    const cv::Mat& board_mask,
    const cv::Mat& hand_stylus_mask,
    cv::Mat* board_render_mask_out,
    cv::Mat* occlusion_render_mask_out) {
    cv::Mat board_render_mask;
    cv::threshold(board_mask, board_render_mask, 0, 255, cv::THRESH_BINARY);
    cv::morphologyEx(
        board_render_mask,
        board_render_mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    cv::dilate(
        board_render_mask,
        board_render_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)),
        cv::Point(-1, -1),
        1);

    cv::Mat occlusion_render_mask;
    cv::threshold(hand_stylus_mask, occlusion_render_mask, 0, 255, cv::THRESH_BINARY);
    cv::dilate(
        occlusion_render_mask,
        occlusion_render_mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)),
        cv::Point(-1, -1),
        1);
    cv::bitwise_and(occlusion_render_mask, board_render_mask, occlusion_render_mask);

    *board_render_mask_out = std::move(board_render_mask);
    *occlusion_render_mask_out = std::move(occlusion_render_mask);
}

cv::Mat makeSurfaceTextureOutputWithMasks(
    const cv::Mat& frame_bgr,
    const cv::Mat& surface_texture_ref_bgr,
    const cv::Mat& H_ref_to_frame,
    const cv::Mat& board_render_mask,
    const cv::Mat& occlusion_render_mask) {
    cv::Mat warped_texture;
    cv::warpPerspective(
        surface_texture_ref_bgr,
        warped_texture,
        H_ref_to_frame,
        frame_bgr.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT_101);

    cv::Mat output = frame_bgr.clone();
    warped_texture.copyTo(output, board_render_mask);
    frame_bgr.copyTo(output, occlusion_render_mask);
    return output;
}

cv::Mat makeAlignedRefMaskedDifference(
    const cv::Mat& frame_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat& board_mask,
    const cv::Mat& hand_stylus_mask) {
    cv::Mat difference;
    cv::absdiff(frame_bgr, aligned_ref_bgr, difference);
    if (!board_mask.empty()) {
        difference.setTo(cv::Scalar(0, 0, 0), board_mask == 0);
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
    for (int iter = 0; iter < max_fill_iterations && cv::countNonZero(unknown) > 0; ++iter) {
        cv::Mat expanded_valid;
        cv::dilate(valid_u8, expanded_valid, kernel, cv::Point(-1, -1), 1);
        cv::Mat candidates;
        cv::bitwise_and(expanded_valid, unknown, candidates);
        if (cv::countNonZero(candidates) == 0) {
            break;
        }

        cv::Mat weighted = cv::Mat::zeros(filled_f.size(), filled_f.type());
        filled_f.copyTo(weighted, valid_u8);
        cv::Mat sum_bgr;
        cv::boxFilter(weighted, sum_bgr, -1, cv::Size(3, 3), cv::Point(-1, -1), false, cv::BORDER_REPLICATE);

        cv::Mat weight_sum;
        cv::boxFilter(valid_f, weight_sum, -1, cv::Size(3, 3), cv::Point(-1, -1), false, cv::BORDER_REPLICATE);

        cv::parallel_for_(cv::Range(0, filled_f.rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const uchar* candidate_row = candidates.ptr<uchar>(y);
                const float* weight_row = weight_sum.ptr<float>(y);
                const cv::Vec3f* sum_row = sum_bgr.ptr<cv::Vec3f>(y);
                cv::Vec3f* out_row = filled_f.ptr<cv::Vec3f>(y);
                for (int x = 0; x < filled_f.cols; ++x) {
                    if (candidate_row[x] == 0 || weight_row[x] <= 1e-6f) {
                        continue;
                    }
                    out_row[x] = sum_row[x] * (1.0f / weight_row[x]);
                }
            }
        });

        valid_u8.setTo(255, candidates);
        valid_u8.convertTo(valid_f, CV_32F, 1.0 / 255.0);
        unknown.setTo(0, candidates);
    }

    cv::Mat filled_bgr;
    filled_f.convertTo(filled_bgr, CV_8UC3);
    cv::Mat out = image_bgr.clone();
    filled_bgr.copyTo(out(work_roi));
    return out;
}

cv::Mat makeCleanAlignedRefMaskedDifference(
    const cv::Mat& aligned_ref_masked_difference_bgr,
    const cv::Mat& aligned_ref_bgr,
    const cv::Mat* precomputed_surface_mask,
    const cv::Mat& hand_stylus_mask) {
    if (aligned_ref_masked_difference_bgr.empty() || aligned_ref_bgr.empty()) {
        return cv::Mat();
    }

    cv::Mat surface_mask = precomputed_surface_mask != nullptr && !precomputed_surface_mask->empty()
        ? *precomputed_surface_mask
        : makeSurfaceMaskFromAlignedRef(aligned_ref_bgr);
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

cv::Mat weightedBlurFloatWithDenominator(
    const cv::Mat& value,
    const cv::Mat& weight,
    const cv::Mat& denominator,
    double sigma) {
    cv::Mat weighted = value.mul(weight);
    cv::Mat numerator;
    cv::GaussianBlur(weighted, numerator, cv::Size(0, 0), sigma);
    return divideWeightedBlurNumerator(numerator, denominator);
}

cv::Mat weightedBlurFloat(const cv::Mat& value, const cv::Mat& weight, double sigma) {
    cv::Mat denominator;
    cv::GaussianBlur(weight, denominator, cv::Size(0, 0), sigma);
    return weightedBlurFloatWithDenominator(value, weight, denominator, sigma);
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
    cv::Mat working;
    cv::Mat raw;
    cv::Mat clean_diff;
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
        working = toFloat01Bgr(working_src_bgr);
        raw = toFloat01Bgr(raw_src_bgr);
        clean_diff = toFloat01Bgr(clean_difference_bgr);

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
                cv::Vec3f factor(
                    (1.0f - m) + (m * shadow_multiplier[0]),
                    (1.0f - m) + (m * shadow_multiplier[1]),
                    (1.0f - m) + (m * shadow_multiplier[2]));
                factor_row[x] = factor;
            }
        }
    });

    return factor;
}

cv::Mat applyShadowFactorToBaseBgr(const cv::Mat& new_base_bgr, const cv::Mat& shadow_factor_bgr, bool fast_approx) {
    cv::Mat base;
    cv::Mat base_lin;
    if (fast_approx && new_base_bgr.depth() == CV_8U) {
        base_lin = srgbU8ToLinearFastBgr(new_base_bgr);
    } else {
        base = toFloat01Bgr(new_base_bgr);
        base_lin = srgbToLinearBgr(base);
    }

    cv::Mat result_lin(base_lin.size(), CV_32FC3);
    cv::parallel_for_(cv::Range(0, base_lin.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3f* base_row = base_lin.ptr<cv::Vec3f>(y);
            const cv::Vec3f* factor_row = shadow_factor_bgr.ptr<cv::Vec3f>(y);
            cv::Vec3f* out_row = result_lin.ptr<cv::Vec3f>(y);
            for (int x = 0; x < base_lin.cols; ++x) {
                out_row[x] = cv::Vec3f(
                    base_row[x][0] * factor_row[x][0],
                    base_row[x][1] * factor_row[x][1],
                    base_row[x][2] * factor_row[x][2]);
            }
        }
    });

    return linearToSrgbBgrU8(result_lin);
}

void renderTextureOverlayFused(
    const FrameRenderData& render_data,
    const cv::Mat& surface_texture_ref_bgr,
    cv::Mat* warped_texture,
    cv::Mat* output) {
    cv::warpPerspective(
        surface_texture_ref_bgr,
        *warped_texture,
        render_data.H_ref_to_frame,
        render_data.frame_bgr.size(),
        cv::INTER_LINEAR,
        cv::BORDER_REPLICATE);

    output->create(render_data.frame_bgr.size(), CV_8UC3);
    const auto& lut = srgbU8ToLinearFastLut();
    cv::parallel_for_(cv::Range(0, render_data.frame_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const cv::Vec3b* frame_row = render_data.frame_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* texture_row = warped_texture->ptr<cv::Vec3b>(y);
            const uchar* board_row = render_data.board_render_mask.ptr<uchar>(y);
            const uchar* occlusion_row = render_data.occlusion_render_mask.ptr<uchar>(y);
            const cv::Vec3f* factor_row = render_data.shadow_factor_bgr.ptr<cv::Vec3f>(y);
            cv::Vec3b* out_row = output->ptr<cv::Vec3b>(y);
            for (int x = 0; x < render_data.frame_bgr.cols; ++x) {
                const cv::Vec3b& base = (board_row[x] != 0 && occlusion_row[x] == 0)
                    ? texture_row[x]
                    : frame_row[x];
                out_row[x] = cv::Vec3b(
                    cv::saturate_cast<uchar>(std::lround(linearToSrgbScalar(lut[base[0]] * factor_row[x][0]) * 255.0f)),
                    cv::saturate_cast<uchar>(std::lround(linearToSrgbScalar(lut[base[1]] * factor_row[x][1]) * 255.0f)),
                    cv::saturate_cast<uchar>(std::lround(linearToSrgbScalar(lut[base[2]] * factor_row[x][2]) * 255.0f)));
            }
        }
    });
}

cv::Mat transferableShadowOverlayBgr(
    const cv::Mat& working_src_bgr,
    const cv::Mat& raw_src_bgr,
    const cv::Mat& clean_difference_bgr,
    const cv::Mat& new_base_bgr,
    float mask_low = 0.02f,
    float mask_high = 0.35f,
    double blur_sigma_mask = 9.0,
    double blur_sigma_field = 15.0,
    float tint_strength = 0.12f,
    float darkness_min = 0.35f,
    float darkness_max = 1.0f) {
    const bool fast_approx = fastApproxEnabled() &&
        working_src_bgr.depth() == CV_8U &&
        raw_src_bgr.depth() == CV_8U &&
        clean_difference_bgr.depth() == CV_8U &&
        new_base_bgr.depth() == CV_8U;
    cv::Mat shadow_factor_bgr = computeShadowFactorBgr(
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
    return applyShadowFactorToBaseBgr(new_base_bgr, shadow_factor_bgr, fast_approx);
}

cv::Mat renderTextureOverlay(
    const FrameRenderData& render_data,
    const cv::Mat& surface_texture_ref_bgr) {
    cv::Mat surface_texture_output;
    if (!render_data.board_render_mask.empty() && !render_data.occlusion_render_mask.empty()) {
        surface_texture_output = makeSurfaceTextureOutputWithMasks(
            render_data.frame_bgr,
            surface_texture_ref_bgr,
            render_data.H_ref_to_frame,
            render_data.board_render_mask,
            render_data.occlusion_render_mask);
    } else {
        surface_texture_output = makeSurfaceTextureOutput(
            render_data.frame_bgr,
            surface_texture_ref_bgr,
            render_data.H_ref_to_frame,
            render_data.board_mask,
            render_data.hand_stylus_mask);
    }
    if (!render_data.shadow_factor_bgr.empty()) {
        return applyShadowFactorToBaseBgr(
            surface_texture_output,
            render_data.shadow_factor_bgr,
            fastApproxEnabled() && surface_texture_output.depth() == CV_8U);
    }
    return transferableShadowOverlayBgr(
        render_data.aligned_ref_bgr,
        render_data.frame_bgr,
        render_data.aligned_ref_masked_difference_clean,
        surface_texture_output,
        0.02f,
        0.35f,
        9.0,
        15.0,
        0.12f);
}

int temporalSupportRadius(int frame_delta) {
    int delta = std::max(1, std::abs(frame_delta));
    return std::clamp(18 + (delta * 2), 24, 90);
}

cv::Mat dilateMaskByRadius(const cv::Mat& mask, int radius) {
    cv::Mat binary;
    cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat dilated;
    cv::dilate(
        binary,
        dilated,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size((radius * 2) + 1, (radius * 2) + 1)),
        cv::Point(-1, -1),
        1);
    return dilated;
}

cv::Mat makeTemporalAllowedMask(
    const std::vector<ProcessedFrameArtifacts>& frames,
    size_t index,
    int* max_radius) {
    cv::Mat allowed(frames[index].hand_stylus_mask.size(), CV_8UC1, cv::Scalar(0));
    int radius_used = 0;

    auto addNeighbor = [&](size_t neighbor_index) {
        if (frames[neighbor_index].hand_stylus_mask.empty()) {
            return;
        }
        int radius = temporalSupportRadius(frames[index].frame_index - frames[neighbor_index].frame_index);
        radius_used = std::max(radius_used, radius);
        cv::Mat neighbor_allowed = dilateMaskByRadius(frames[neighbor_index].hand_stylus_mask, radius);
        cv::bitwise_or(allowed, neighbor_allowed, allowed);
    };

    if (index > 0) {
        addNeighbor(index - 1);
    }
    if (index + 1 < frames.size()) {
        addNeighbor(index + 1);
    }

    if (!frames[index].board_mask.empty()) {
        cv::bitwise_and(allowed, frames[index].board_mask, allowed);
    }
    if (max_radius != nullptr) {
        *max_radius = radius_used;
    }
    return allowed;
}

cv::Mat filterHandStylusMaskTemporally(
    const cv::Mat& current_mask,
    const cv::Mat& temporal_allowed_mask,
    const cv::Mat& board_mask,
    int temporal_radius) {
    if (current_mask.empty() || temporal_allowed_mask.empty() || cv::countNonZero(temporal_allowed_mask) == 0) {
        return current_mask.clone();
    }

    cv::Mat current;
    cv::threshold(current_mask, current, 0, 255, cv::THRESH_BINARY);
    if (!board_mask.empty()) {
        cv::bitwise_and(current, board_mask, current);
    }

    cv::Mat supported;
    cv::bitwise_and(current, temporal_allowed_mask, supported);

    cv::Mat not_supported(current.size(), CV_8UC1, cv::Scalar(255));
    not_supported.setTo(0, supported);
    cv::Mat distance_to_support;
    cv::distanceTransform(not_supported, distance_to_support, cv::DIST_L2, 3);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int label_count = cv::connectedComponentsWithStats(current, labels, stats, centroids, 8, CV_32S);
    cv::Mat filtered(current.size(), CV_8UC1, cv::Scalar(0));
    const int small_component_area = 2500;
    const int trim_distance = std::clamp(static_cast<int>(std::lround(temporal_radius * 0.40)), 12, 36);

    for (int label = 1; label < label_count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area <= 0) {
            continue;
        }

        cv::Mat component;
        cv::compare(labels, label, component, cv::CMP_EQ);
        cv::bitwise_and(component, current, component);

        cv::Mat component_supported;
        cv::bitwise_and(component, temporal_allowed_mask, component_supported);
        int support_count = cv::countNonZero(component_supported);
        if (support_count == 0) {
            if (area >= small_component_area) {
                filtered.setTo(255, component);
            }
            continue;
        }

        int required_support = std::max(15, static_cast<int>(std::lround(area * 0.02)));
        if (area < small_component_area && support_count < required_support) {
            continue;
        }

        cv::Mat keep_by_distance;
        cv::compare(distance_to_support, static_cast<double>(trim_distance), keep_by_distance, cv::CMP_LE);
        cv::Mat component_filtered;
        cv::bitwise_and(component, keep_by_distance, component_filtered);
        filtered.setTo(255, component_filtered);
    }

    cv::morphologyEx(
        filtered,
        filtered,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
        cv::Point(-1, -1),
        1);
    if (!board_mask.empty()) {
        cv::bitwise_and(filtered, board_mask, filtered);
    }
    return filtered;
}

void applyTemporalHandStylusMaskFiltering(
    std::vector<ProcessedFrameArtifacts>* frames,
    const fs::path& outdir,
    const cv::Mat& surface_texture_ref_bgr) {
    if (frames == nullptr || frames->size() < 2) {
        return;
    }

    std::vector<cv::Mat> filtered_masks(frames->size());
    int changed_frames = 0;
    for (size_t i = 0; i < frames->size(); ++i) {
        if ((*frames)[i].hand_stylus_mask.empty()) {
            continue;
        }
        int temporal_radius = 0;
        cv::Mat allowed = makeTemporalAllowedMask(*frames, i, &temporal_radius);
        filtered_masks[i] = filterHandStylusMaskTemporally(
            (*frames)[i].hand_stylus_mask,
            allowed,
            (*frames)[i].board_mask,
            temporal_radius);
    }

    for (size_t i = 0; i < frames->size(); ++i) {
        ProcessedFrameArtifacts& frame = (*frames)[i];
        if (filtered_masks[i].empty()) {
            continue;
        }

        cv::Mat changed;
        cv::bitwise_xor(frame.hand_stylus_mask, filtered_masks[i], changed);
        int changed_pixels = cv::countNonZero(changed);
        if (changed_pixels > 0) {
            ++changed_frames;
        }
        frame.hand_stylus_mask = filtered_masks[i];

        if (frame.write_colormatch) {
            writePngFast(
                outdir / (frame.prefix + "_aligned_ref_colormatch_hand_stylus_mask.png"),
                frame.hand_stylus_mask);
        }
        if (frame.write_aligned_ref_masked_difference && !frame.aligned_ref_bgr.empty()) {
            cv::Mat aligned_ref_masked_difference = makeAlignedRefMaskedDifference(
                frame.frame_bgr,
                frame.aligned_ref_bgr,
                frame.board_mask,
                frame.hand_stylus_mask);
            writePngFast(
                outdir / (frame.prefix + "_aligned_ref_masked_difference.png"),
                aligned_ref_masked_difference);
            cv::Mat clean_difference = makeCleanAlignedRefMaskedDifference(
                aligned_ref_masked_difference,
                frame.aligned_ref_bgr,
                nullptr,
                frame.hand_stylus_mask);
            writePngFast(
                outdir / (frame.prefix + "_aligned_ref_masked_difference_clean.png"),
                clean_difference);
        }
        if (frame.write_surface_texture && !surface_texture_ref_bgr.empty()) {
            cv::Mat surface_texture_output = makeSurfaceTextureOutput(
                frame.frame_bgr,
                surface_texture_ref_bgr,
                frame.H_ref_to_frame,
                frame.board_mask,
                frame.hand_stylus_mask);
            writePngFast(outdir / (frame.prefix + "_surface_texture.png"), surface_texture_output);

            cv::Mat aligned_ref_masked_difference = makeAlignedRefMaskedDifference(
                frame.frame_bgr,
                frame.aligned_ref_bgr,
                frame.board_mask,
                frame.hand_stylus_mask);
            cv::Mat clean_difference = makeCleanAlignedRefMaskedDifference(
                aligned_ref_masked_difference,
                frame.aligned_ref_bgr,
                nullptr,
                frame.hand_stylus_mask);
            cv::Mat final_overlay = transferableShadowOverlayBgr(
                frame.aligned_ref_bgr,
                frame.frame_bgr,
                clean_difference,
                surface_texture_output,
                0.02f,
                0.35f,
                9.0,
                15.0,
                0.12f);
            writePngFast(outdir / (frame.prefix + "_final_overlay.png"), final_overlay);
        }
        if (g_info_enabled) {
            std::cout << "[INFO] " << frame.prefix
                      << " temporal hand/stylus mask changed_pixels=" << changed_pixels << "\n";
        }
    }

    if (g_info_enabled) {
        std::cout << "[INFO] Temporal hand/stylus mask filtering updated "
                  << changed_frames << "/" << frames->size() << " frames\n";
    }
}

bool computeFrameGeometry(
    int frame_index,
    cv::Mat frame_bgr,
    const cv::Mat& ref_bgr,
    const DetectionResult& ref_tags,
    const cv::Mat& reference_tag_contrast,
    const SheetTrackerConfig* tracker_config,
    SheetTrackerState* tracker_state,
    const Args& args,
    const fs::path& outdir,
    const std::string& prefix,
    bool write_outputs,
    FrameGeometry* geometry,
    std::map<std::string, double>* profile) {
    geometry->index = frame_index;
    DetectionResult frame_tags;
    bool tracker_can_predict =
        tracker_config != nullptr &&
        tracker_state != nullptr &&
        tracker_state->hasValidH &&
        !tracker_state->H_ref_to_prev.empty();
    bool detect_tags_this_frame =
        tracker_config == nullptr ||
        tracker_config->tagDetectInterval <= 1 ||
        !tracker_can_predict ||
        frame_index % tracker_config->tagDetectInterval == 0;
    if (detect_tags_this_frame) {
        ScopedTimer timer(profile, "detect_frame_tags");
        frame_tags = detectAprilTags(frame_bgr);
    }
    cv::Mat curr_tag_contrast;
    if (tracker_config != nullptr && tracker_state != nullptr) {
        curr_tag_contrast = makeTagContrastImage(frame_bgr);
    }

    cv::Mat H;
    cv::Mat board_mask;
    std::string error;
    bool used_held_homography = false;
    {
        ScopedTimer timer(profile, "homography");
        bool homography_ok = findHomographyAndBoardMask(ref_bgr, frame_bgr, ref_tags, frame_tags, &H, &board_mask, &error);
        if (homography_ok && tracker_config != nullptr && tracker_state != nullptr && tracker_state->hasValidH) {
            std::string temporal_error;
            if (!validateTagHomographyAgainstTracker(H, ref_bgr.size(), *tracker_config, *tracker_state, temporal_error)) {
                if (g_info_enabled) {
                    std::cout << "[WARN] " << prefix
                              << ": AprilTag homography rejected by temporal tracker check: "
                              << temporal_error << "\n";
                }
                error = temporal_error;
                homography_ok = false;
            }
        }
        int common_tag_count = homography_ok ? countCommonTags(ref_tags, frame_tags) : 0;
        if (homography_ok &&
                common_tag_count == 1 &&
                tracker_config != nullptr &&
                tracker_state != nullptr &&
                tracker_state->hasValidH) {
            error = "one visible tag is insufficient for sheet homography; using temporal sheet tracker";
            homography_ok = false;
        }
        if (homography_ok &&
                tracker_config != nullptr &&
                tracker_state != nullptr &&
                tracker_config->directSheetRefineEnabled) {
            bool should_refine_sparse_tag_homography =
                common_tag_count == 2 && tracker_config->directSheetRefineTwoTagSeeds;
            if (should_refine_sparse_tag_homography) {
                cv::Mat best_direct_H;
                DirectSheetScore best_direct_score;
                bool best_direct_ok = false;
                std::string direct_error;
                fs::path debug_prefix;
                if (write_outputs && args.write_colormatch && tracker_config->writeDebugImages) {
                    debug_prefix = outdir / prefix;
                }

                cv::Mat H_direct_from_tags;
                DirectSheetScore tag_seed_score;
                std::string tag_seed_error;
                if (refineHomographyByDirectSheetMatch(
                        frame_bgr,
                        ref_bgr,
                        H,
                        *tracker_config,
                        H_direct_from_tags,
                        tag_seed_error,
                        fs::path(),
                        &tag_seed_score,
                        true,
                        false)) {
                    best_direct_H = H_direct_from_tags;
                    best_direct_score = tag_seed_score;
                    best_direct_ok = true;
                } else {
                    direct_error = "tag seed: " + tag_seed_error;
                }

                if (tracker_state->hasValidH && !tracker_state->H_ref_to_prev.empty()) {
                    cv::Mat H_direct_from_prev;
                    DirectSheetScore prev_seed_score;
                    std::string prev_seed_error;
                    if (refineHomographyByDirectSheetMatch(
                            frame_bgr,
                            ref_bgr,
                            tracker_state->H_ref_to_prev,
                            *tracker_config,
                            H_direct_from_prev,
                            prev_seed_error,
                            fs::path(),
                            &prev_seed_score,
                            false,
                            false)) {
                        if (!best_direct_ok || prev_seed_score.score > best_direct_score.score) {
                            best_direct_H = H_direct_from_prev;
                            best_direct_score = prev_seed_score;
                            best_direct_ok = true;
                        }
                    } else if (!best_direct_ok) {
                        direct_error += direct_error.empty() ? "" : "; ";
                        direct_error += "previous seed: " + prev_seed_error;
                    }
                }

                if (best_direct_ok) {
                    H = best_direct_H;
                    board_mask = makeProjectedBoardMask(H, ref_bgr.size(), frame_bgr.size()).first;
                    if (g_info_enabled && write_outputs) {
                        std::cout << "[INFO] " << prefix
                                  << " refined sparse " << common_tag_count
                                  << "-tag homography against expected sheet colors"
                                  << " score=" << best_direct_score.score
                                  << " coverage=" << best_direct_score.coverage << "\n";
                    }
                    if (!debug_prefix.empty()) {
                        cv::Mat debug = frame_bgr.clone();
                        drawProjectedQuad(&debug, H, ref_bgr.size(), cv::Scalar(255, 0, 255));
                        writePngFast(debugPath(debug_prefix, "_direct_sheet_refine_quad.png"), debug);
                    }
                } else if (g_info_enabled && write_outputs) {
                    std::cout << "[INFO] " << prefix
                              << " sparse " << common_tag_count
                              << "-tag direct sheet refinement skipped: "
                              << direct_error << "\n";
                }
            }
        }
        if (!homography_ok) {
            std::string fallback_error;
            bool fallback_ok = false;
            if (tracker_config != nullptr && tracker_state != nullptr && !curr_tag_contrast.empty()) {
                cv::Mat H_fallback;
                fs::path debug_prefix;
                if (write_outputs && args.write_colormatch && tracker_config->writeDebugImages) {
                    debug_prefix = outdir / prefix;
                }
                fallback_ok = trackHomographyWithoutTags(
                    frame_bgr,
                    curr_tag_contrast,
                    reference_tag_contrast,
                    ref_bgr.size(),
                    *tracker_config,
                    *tracker_state,
                    H_fallback,
                    fallback_error,
                    debug_prefix);
                if (fallback_ok) {
                    H = H_fallback;
                    board_mask = makeProjectedBoardMask(H, ref_bgr.size(), frame_bgr.size()).first;
                    if (g_info_enabled) {
                        std::cout << "[INFO] " << prefix << " using optical-flow sheet tracker fallback\n";
                    }
                }
            }
            if (!fallback_ok) {
                bool direct_refine_ok = false;
                if (tracker_config != nullptr &&
                        tracker_state != nullptr &&
                        tracker_config->directSheetRefineEnabled &&
                        tracker_state->hasValidH &&
                        !tracker_state->H_ref_to_prev.empty()) {
                    cv::Mat H_direct;
                    std::string direct_error;
                    fs::path debug_prefix;
                    if (write_outputs && args.write_colormatch && tracker_config->writeDebugImages) {
                        debug_prefix = outdir / prefix;
                    }
                    direct_refine_ok = refineHomographyByDirectSheetMatch(
                        frame_bgr,
                        ref_bgr,
                        tracker_state->H_ref_to_prev,
                        *tracker_config,
                        H_direct,
                        direct_error,
                        debug_prefix);
                    if (direct_refine_ok) {
                        H = H_direct;
                        board_mask = makeProjectedBoardMask(H, ref_bgr.size(), frame_bgr.size()).first;
                        fallback_ok = true;
                        if (g_info_enabled) {
                            std::cout << "[INFO] " << prefix << " using direct blue/green sheet alignment fallback\n";
                        }
                    } else if (g_info_enabled) {
                        std::cout << "[INFO] " << prefix << " direct sheet fallback failed: " << direct_error << "\n";
                    }
                }
            }
            if (!fallback_ok) {
                bool can_hold_previous_homography =
                    tracker_config != nullptr &&
                    tracker_state != nullptr &&
                    tracker_state->hasValidH &&
                    !tracker_state->H_ref_to_prev.empty() &&
                    tracker_config->maxHeldHomographyFrames > 0 &&
                    tracker_state->heldHomographyFrames < tracker_config->maxHeldHomographyFrames;
                if (can_hold_previous_homography) {
                    H = tracker_state->H_ref_to_prev.clone();
                    board_mask = makeProjectedBoardMask(H, ref_bgr.size(), frame_bgr.size()).first;
                    used_held_homography = true;
                    if (g_info_enabled) {
                        std::cout << "[INFO] " << prefix
                                  << " holding previous sheet homography through temporary tag occlusion";
                        if (!fallback_error.empty()) {
                            std::cout << " after tracker fallback failed: " << fallback_error;
                        }
                        std::cout << "\n";
                    }
                } else {
                    if (g_info_enabled) {
                        std::cerr << "[WARN] " << prefix << ": homography failed: " << error;
                        if (!fallback_error.empty()) {
                            std::cerr << "; tracker fallback failed: " << fallback_error;
                        }
                        std::cerr << "\n";
                    }
                    return false;
                }
            }
        }
    }
    if (tracker_config != nullptr && tracker_state != nullptr && !curr_tag_contrast.empty()) {
        if (used_held_homography) {
            ++tracker_state->heldHomographyFrames;
        } else {
            tracker_state->heldHomographyFrames = 0;
        }
        updateSheetTrackerState(frame_bgr, curr_tag_contrast, H, tracker_state);
    }

    if (!board_mask.empty() && envInt("SHADOW_BOARD_MASK_INCLUDE_VISIBLE_SURFACE", 1) != 0) {
        cv::Mat visible_surface = makeVisibleBoardColorProtectionMask(frame_bgr);
        int grow = std::clamp(envInt("SHADOW_BOARD_VISIBLE_SURFACE_GROW", 61), 0, 151);
        if (grow > 0) {
            if (grow % 2 == 0) {
                ++grow;
            }
            cv::Mat loose_projected_board;
            cv::dilate(
                board_mask,
                loose_projected_board,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(grow, grow)),
                cv::Point(-1, -1),
                1);
            cv::bitwise_and(visible_surface, loose_projected_board, visible_surface);
        }
        cv::bitwise_or(board_mask, visible_surface, board_mask);
    }

    geometry->index = frame_index;
    geometry->frame_bgr = std::move(frame_bgr);
    geometry->frame_tags = std::move(frame_tags);
    geometry->H_ref_to_frame = std::move(H);
    geometry->board_mask = std::move(board_mask);
    geometry->ok = true;
    return true;
}

bool processFrame(
    int frame_index,
    const cv::Mat& frame_bgr,
    const cv::Mat& ref_bgr,
    const cv::Mat& surface_texture_ref_bgr,
    const DetectionResult& ref_tags,
    const cv::Mat& reference_tag_contrast,
    const SheetTrackerConfig* tracker_config,
    SheetTrackerState* tracker_state,
    const Args& args,
    const fs::path& outdir,
    const std::string& prefix,
    bool write_outputs,
    ProcessedFrameArtifacts* artifacts,
    cv::Mat* final_overlay_out,
    FrameRenderData* render_data_out,
    const FrameGeometry* precomputed_geometry = nullptr) {
    std::map<std::string, double> profile;
    static cv::Size cached_ref_size;
    static cv::Mat cached_ref_hsv;
    static cv::Mat cached_flat_green_mask;
    static cv::Mat cached_flat_blue_mask;
    static std::shared_ptr<const std::vector<FlatColorMatchCellDefinition>> cached_flat_color_match_cells;
    static std::shared_ptr<const std::vector<QualityCellDefinition>> cached_inner_quality_cells;
    static std::shared_ptr<const std::vector<QualityCellDefinition>> cached_outer_quality_cells;
    static std::mutex cached_ref_mutex;

    {
        ScopedTimer timer(&profile, "clear_outputs");
        if (write_outputs) {
            clearPreviousOutputs(outdir, prefix);
        }
    }
    {
        ScopedTimer timer(&profile, "write_raw_frame");
        if (write_outputs && args.write_raw_frame) {
            writePngFast(outdir / (prefix + "_raw_frame.png"), frame_bgr);
        }
    }

    FrameGeometry computed_geometry;
    const FrameGeometry* geometry = precomputed_geometry;
    if (geometry == nullptr) {
        if (!computeFrameGeometry(
                frame_index,
                frame_bgr,
                ref_bgr,
                ref_tags,
                reference_tag_contrast,
                tracker_config,
                tracker_state,
                args,
                outdir,
                prefix,
                write_outputs,
                &computed_geometry,
                &profile)) {
            printFrameProfile(prefix, profile);
            addFrameProfile(profile);
            return false;
        }
        geometry = &computed_geometry;
    }

    const DetectionResult& frame_tags = geometry->frame_tags;
    const cv::Mat& H = geometry->H_ref_to_frame;
    const cv::Mat& board_mask = geometry->board_mask;

    bool need_surface_texture = !surface_texture_ref_bgr.empty();
    bool need_render_data = render_data_out != nullptr;
    bool need_colormatch_artifacts = args.write_colormatch || need_surface_texture || need_render_data;
    bool need_quality_map = args.write_combined_quality || need_colormatch_artifacts;
    bool need_aligned_ref = args.write_aligned_ref || args.write_aligned_ref_flat || need_colormatch_artifacts;
    cv::Mat warped_ref;
    if (need_aligned_ref) {
        ScopedTimer timer(&profile, "warp_ref_to_frame");
        cv::warpPerspective(ref_bgr, warped_ref, H, frame_bgr.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
    }
    cv::Mat aligned_ref;
    if (need_aligned_ref) {
        aligned_ref = warped_ref.clone();
        ScopedTimer timer(&profile, "aligned_ref_color");
        if (!applyTopLeftTagColorMatch(ref_bgr, frame_bgr, ref_tags, frame_tags, board_mask, &aligned_ref)) {
            if (g_info_enabled) {
                std::cerr << "[WARN] " << prefix << ": tag color sampling failed; using channel-gain fallback\n";
            }
            aligned_ref = robustChannelGain(frame_bgr, warped_ref, board_mask);
        }
    }
    {
        ScopedTimer timer(&profile, "write_aligned_ref");
        if (write_outputs && args.write_aligned_ref) {
            writePngFast(outdir / (prefix + "_aligned_ref.png"), aligned_ref);
        }
    }

    cv::Mat H_frame_to_ref = H.inv();
    cv::Mat flat_raw_frame;
    cv::Mat flat_aligned_ref;
    bool need_flat_aligned_ref = args.write_aligned_ref_flat || (write_outputs && args.write_colormatch);
    {
        ScopedTimer timer(&profile, "warp_flat_outputs");
        if (args.write_raw_frame_flat || need_quality_map) {
            cv::warpPerspective(
                frame_bgr,
                flat_raw_frame,
                H_frame_to_ref,
                ref_bgr.size(),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));
        }
        if (need_flat_aligned_ref) {
            cv::warpPerspective(
                aligned_ref,
                flat_aligned_ref,
                H_frame_to_ref,
                ref_bgr.size(),
                cv::INTER_NEAREST,
                cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));
        }
    }

    cv::Mat flat_green_mask;
    cv::Mat flat_blue_mask;
    cv::Mat ref_hsv;
    std::shared_ptr<const std::vector<FlatColorMatchCellDefinition>> flat_color_match_cells;
    std::shared_ptr<const std::vector<QualityCellDefinition>> inner_quality_cells;
    std::shared_ptr<const std::vector<QualityCellDefinition>> outer_quality_cells;
    {
        ScopedTimer timer(&profile, "flat_reference_cache");
        std::lock_guard<std::mutex> lock(cached_ref_mutex);
        if (cached_ref_size != ref_bgr.size() || cached_ref_hsv.empty()) {
            cached_ref_size = ref_bgr.size();
            cv::cvtColor(ref_bgr, cached_ref_hsv, cv::COLOR_BGR2HSV);
            buildStrictReferenceColorMasks(ref_bgr, &cached_flat_green_mask, &cached_flat_blue_mask);
            cached_flat_color_match_cells = std::make_shared<const std::vector<FlatColorMatchCellDefinition>>(
                buildFlatColorMatchCellDefinitions(
                    ref_tags,
                    ref_bgr.size(),
                    cached_flat_green_mask,
                    cached_flat_blue_mask));
            cached_inner_quality_cells = std::make_shared<const std::vector<QualityCellDefinition>>(
                buildInnerQualityCellDefinitions(
                    ref_tags,
                    ref_bgr.size(),
                    cached_ref_hsv,
                    cached_flat_green_mask,
                    cached_flat_blue_mask));
            cached_outer_quality_cells = std::make_shared<const std::vector<QualityCellDefinition>>(
                buildOuterGreenQualityCellDefinitions(
                    ref_tags,
                    ref_bgr.size(),
                    cached_ref_hsv,
                    cached_flat_green_mask));
        }
        ref_hsv = cached_ref_hsv;
        flat_green_mask = cached_flat_green_mask;
        flat_blue_mask = cached_flat_blue_mask;
        flat_color_match_cells = cached_flat_color_match_cells;
        inner_quality_cells = cached_inner_quality_cells;
        outer_quality_cells = cached_outer_quality_cells;
    }
    {
        ScopedTimer timer(&profile, "flat_scratch_cleanup");
        if (args.write_raw_frame_flat || need_quality_map) {
            removeFlatSurfaceScratches(&flat_raw_frame, flat_green_mask, flat_blue_mask);
        }
    }

    cv::Mat flat_tag_combined_quality_gray;
    cv::Mat aligned_ref_colormatch;
    cv::Mat aligned_ref_colormatch_difference;
    cv::Mat aligned_ref_colormatch_hand_stylus_mask;
    cv::Mat aligned_ref_masked_difference;
    cv::Mat aligned_ref_masked_difference_clean;
    cv::Mat flat_aligned_ref_colormatch;
    cv::Mat surface_texture_output;
    cv::Mat final_overlay;
    cv::Mat aligned_ref_surface_mask;
    cv::Mat projected_tag_cell_protection_mask;
    cv::Mat flat_hsv;
    if (need_quality_map) {
        ScopedTimer timer(&profile, "flat_hsv");
        cv::cvtColor(flat_raw_frame, flat_hsv, cv::COLOR_BGR2HSV);
    }
    {
        ScopedTimer timer(&profile, "quality_inner_combined");
        if (need_quality_map) {
            flat_tag_combined_quality_gray = makeTagCellCombinedQualityGrayImage(
                flat_hsv,
                ref_hsv,
                ref_tags,
                flat_green_mask,
                flat_blue_mask,
                inner_quality_cells.get());
        }
    }
    {
        ScopedTimer timer(&profile, "quality_outer_green");
        if (need_quality_map) {
            addOuterGreenCellQuality(
                flat_hsv,
                ref_hsv,
                ref_tags,
                flat_green_mask,
                &flat_tag_combined_quality_gray,
                outer_quality_cells.get());
        }
    }
    {
        ScopedTimer timer(&profile, "aligned_ref_colormatch");
        if (need_colormatch_artifacts) {
            aligned_ref_colormatch = makeAlignedRefColorMatchImage(
                aligned_ref,
                flat_aligned_ref,
                flat_raw_frame,
                flat_tag_combined_quality_gray,
                ref_tags,
                flat_green_mask,
                flat_blue_mask,
                *flat_color_match_cells,
                H,
                (write_outputs && args.write_colormatch) ? &flat_aligned_ref_colormatch : nullptr);
            if (envInt("SHADOW_PROJECTED_TAG_CELL_COLORMATCH", 1) != 0) {
                int recolored_tag_cells = applyProjectedTagShadowCellColorMatch(
                    ref_bgr,
                    frame_bgr,
                    ref_tags,
                    H,
                    board_mask,
                    &aligned_ref_colormatch,
                    &projected_tag_cell_protection_mask);
                if (g_info_enabled && recolored_tag_cells > 0) {
                    std::cout << "[INFO] " << prefix
                              << " projected shadowed tag color cells recolored "
                              << recolored_tag_cells << "\n";
                }
            }
            cv::absdiff(frame_bgr, aligned_ref_colormatch, aligned_ref_colormatch_difference);
            if (!board_mask.empty()) {
                aligned_ref_colormatch_difference.setTo(cv::Scalar(0, 0, 0), board_mask == 0);
            }
            double mask_otsu_threshold = 0.0;
            int mask_used_threshold = 0;
            aligned_ref_surface_mask = makeSurfaceMaskFromAlignedRef(aligned_ref);
            aligned_ref_colormatch_hand_stylus_mask = makeHandPenMaskFromColormatchDifference(
                aligned_ref_colormatch_difference,
                aligned_ref,
                &frame_bgr,
                &aligned_ref_surface_mask,
                &projected_tag_cell_protection_mask,
                &mask_otsu_threshold,
                &mask_used_threshold);
            if (g_info_enabled) {
                std::cout << "[INFO] " << prefix
                          << " hand/stylus mask otsu=" << mask_otsu_threshold
                          << " threshold=" << mask_used_threshold << "\n";
            }
        }
    }
    {
        ScopedTimer timer(&profile, "surface_texture");
        if (need_surface_texture) {
            surface_texture_output = makeSurfaceTextureOutput(
                frame_bgr,
                surface_texture_ref_bgr,
                H,
                board_mask,
                aligned_ref_colormatch_hand_stylus_mask);
        }
    }
    {
        ScopedTimer timer(&profile, "aligned_ref_masked_difference");
        if (args.write_colormatch || need_surface_texture || need_render_data) {
            aligned_ref_masked_difference = makeAlignedRefMaskedDifference(
                frame_bgr,
                aligned_ref,
                board_mask,
                aligned_ref_colormatch_hand_stylus_mask);
            aligned_ref_masked_difference_clean = makeCleanAlignedRefMaskedDifference(
                aligned_ref_masked_difference,
                aligned_ref,
                &aligned_ref_surface_mask,
                aligned_ref_colormatch_hand_stylus_mask);
        }
    }
    {
        ScopedTimer timer(&profile, "final_overlay");
        if (need_surface_texture) {
            final_overlay = transferableShadowOverlayBgr(
                aligned_ref,
                frame_bgr,
                aligned_ref_masked_difference_clean,
                surface_texture_output,
                0.02f,
                0.35f,
                9.0,
                15.0,
                0.12f);
        }
    }

    {
        ScopedTimer timer(&profile, "write_flat_outputs");
        if (write_outputs && args.write_raw_frame_flat) {
            writePngFast(outdir / (prefix + "_raw_frame_flat.png"), flat_raw_frame);
        }
        if (write_outputs && args.write_aligned_ref_flat) {
            writePngFast(outdir / (prefix + "_aligned_ref_flat.png"), flat_aligned_ref);
        }
        if (write_outputs && args.write_combined_quality) {
            cv::Mat flat_tag_combined_quality_bgr;
            cv::cvtColor(flat_tag_combined_quality_gray, flat_tag_combined_quality_bgr, cv::COLOR_GRAY2BGR);
            writePngFast(outdir / (prefix + "_flat_tag_combined_quality.png"), flat_tag_combined_quality_bgr);
        }
        if (write_outputs && args.write_colormatch) {
            writePngFast(outdir / (prefix + "_aligned_ref_colormatch.png"), aligned_ref_colormatch);
            writePngFast(outdir / (prefix + "_aligned_ref_colormatch_flat.png"), flat_aligned_ref_colormatch);
            writePngFast(outdir / (prefix + "_aligned_ref_colormatch_difference.png"), aligned_ref_colormatch_difference);
            writePngFast(outdir / (prefix + "_aligned_ref_colormatch_hand_stylus_mask.png"), aligned_ref_colormatch_hand_stylus_mask);
            writePngFast(outdir / (prefix + "_aligned_ref_masked_difference.png"), aligned_ref_masked_difference);
            writePngFast(outdir / (prefix + "_aligned_ref_masked_difference_clean.png"), aligned_ref_masked_difference_clean);
        }
        if (write_outputs && args.write_colormatch && need_surface_texture) {
            writePngFast(outdir / (prefix + "_surface_texture.png"), surface_texture_output);
        }
        if (write_outputs && need_surface_texture) {
            writePngFast(outdir / (prefix + "_final_overlay.png"), final_overlay);
        }
    }

    if (final_overlay_out != nullptr && !final_overlay.empty()) {
        *final_overlay_out = final_overlay;
    }

    if (render_data_out != nullptr && !aligned_ref_masked_difference_clean.empty()) {
        render_data_out->frame_bgr = frame_bgr.clone();
        render_data_out->aligned_ref_bgr = aligned_ref.clone();
        render_data_out->H_ref_to_frame = H.clone();
        render_data_out->board_mask = board_mask.clone();
        render_data_out->hand_stylus_mask = aligned_ref_colormatch_hand_stylus_mask.clone();
        render_data_out->aligned_ref_masked_difference_clean = aligned_ref_masked_difference_clean.clone();
        {
            ScopedTimer timer(&profile, "render_masks");
            makeSurfaceTextureRenderMasks(
                board_mask,
                aligned_ref_colormatch_hand_stylus_mask,
                &render_data_out->board_render_mask,
                &render_data_out->occlusion_render_mask);
        }
        {
            ScopedTimer timer(&profile, "shadow_factor");
            render_data_out->shadow_factor_bgr = computeShadowFactorBgr(
                aligned_ref,
                frame_bgr,
                aligned_ref_masked_difference_clean,
                0.02f,
                0.35f,
                9.0,
                15.0,
                0.12f,
                0.35f,
                1.0f,
                fastApproxEnabled() &&
                    aligned_ref.depth() == CV_8U &&
                    frame_bgr.depth() == CV_8U &&
                    aligned_ref_masked_difference_clean.depth() == CV_8U);
        }
    }

    if (artifacts != nullptr && write_outputs && need_colormatch_artifacts) {
        artifacts->frame_index = frame_index;
        artifacts->prefix = prefix;
        artifacts->frame_bgr = frame_bgr.clone();
        artifacts->aligned_ref_bgr = aligned_ref.clone();
        artifacts->H_ref_to_frame = H.clone();
        artifacts->board_mask = board_mask.clone();
        artifacts->hand_stylus_mask = aligned_ref_colormatch_hand_stylus_mask.clone();
        artifacts->write_colormatch = args.write_colormatch;
        artifacts->write_aligned_ref_masked_difference = args.write_colormatch;
        artifacts->write_surface_texture = need_surface_texture;
    }

#if 0
    cv::Mat per_tag_aligned_ref = aligned_ref.clone();
    fs::path debug_prefix;
    if (debugStagesEnabled()) {
        debug_prefix = outdir / prefix;
    }
    int per_tag_count = applyPerVisibleTagColorMatch(
        ref_bgr,
        frame_bgr,
        ref_tags,
        frame_tags,
        H,
        warped_ref,
        board_mask,
        debug_prefix,
        &per_tag_aligned_ref);
    if (per_tag_count == 0) {
        if (g_info_enabled) {
            std::cerr << "[WARN] " << prefix << ": no visible tags could be recolored for per-tag aligned reference\n";
        }
    } else {
        if (g_info_enabled) {
            std::cout << "[INFO] Per-tag color match recolored " << per_tag_count << " tags\n";
        }
    }
    cv::imwrite((outdir / (prefix + "_aligned_ref_per_tag.png")).string(), per_tag_aligned_ref);

    cv::Mat per_tag_color_difference;
    cv::absdiff(frame_bgr, per_tag_aligned_ref, per_tag_color_difference);
    if (!board_mask.empty()) {
        per_tag_color_difference.setTo(cv::Scalar(0, 0, 0), board_mask == 0);
    }
    cv::imwrite((outdir / (prefix + "_aligned_ref_per_tag_color_difference.png")).string(), per_tag_color_difference);

    cv::Mat hand_pen_mask = makeHandPenMask(per_tag_color_difference);
    cv::imwrite((outdir / (prefix + "_hand_pen_mask.png")).string(), hand_pen_mask);

    cv::Mat difference_no_hand_pen = per_tag_color_difference.clone();
    difference_no_hand_pen.setTo(cv::Scalar(0, 0, 0), hand_pen_mask);
    cv::imwrite((outdir / (prefix + "_aligned_ref_per_tag_color_difference_no_hand_pen.png")).string(), difference_no_hand_pen);
#endif
    printFrameProfile(prefix, profile);
    addFrameProfile(profile);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        cv::setUseOptimized(true);
        cv::setNumThreads(std::max(1, cv::getNumberOfCPUs()));

        Args args = parseArgs(argc, argv);
        bool saving_video = args.frames.empty() &&
            (!args.surface_texture.empty() || !args.surface_texture_dir.empty());
        if (saving_video) {
            g_info_enabled = false;
            g_profile_enabled = false;
            cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
        }
        if (std::getenv("SHADOW_INFO") != nullptr) {
            g_info_enabled = envInt("SHADOW_INFO", g_info_enabled ? 1 : 0) != 0;
        }
        if (std::getenv("SHADOW_PROFILE") != nullptr) {
            g_profile_enabled = envInt("SHADOW_PROFILE", g_profile_enabled ? 1 : 0) != 0;
        }
        fs::create_directories(args.outdir);

        cv::Mat ref_bgr = cv::imread(args.reference, cv::IMREAD_COLOR);
        if (ref_bgr.empty()) {
            throw std::runtime_error("Could not read reference image: " + args.reference);
        }

        cv::rotate(ref_bgr, ref_bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
        if (std::abs(args.reference_scale - 1.0) > 1e-6) {
            cv::Size old_size = ref_bgr.size();
            cv::Size new_size(
                std::max(1, static_cast<int>(std::lround(old_size.width * args.reference_scale))),
                std::max(1, static_cast<int>(std::lround(old_size.height * args.reference_scale))));
            int interpolation = args.reference_scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(ref_bgr, ref_bgr, new_size, 0.0, 0.0, interpolation);
            if (g_info_enabled) {
                std::cout << "[INFO] Reference scale: " << args.reference_scale
                          << " (" << old_size.width << "x" << old_size.height
                          << " -> " << new_size.width << "x" << new_size.height << ")\n";
            }
        }

        if (!args.surface_texture.empty() && !args.surface_texture_dir.empty()) {
            throw std::runtime_error("Use either --surface_texture or --surface_texture_dir, not both.");
        }

        cv::Mat surface_texture_ref_bgr;
        std::vector<TextureEntry> surface_texture_entries;
        if (!args.surface_texture.empty()) {
            cv::Mat texture_bgr = cv::imread(args.surface_texture, cv::IMREAD_COLOR);
            if (texture_bgr.empty()) {
                throw std::runtime_error("Could not read surface texture image: " + args.surface_texture);
            }
            surface_texture_ref_bgr = fitTextureToReference(texture_bgr, ref_bgr.size());
            if (g_info_enabled) {
                std::cout << "[INFO] Surface texture: " << args.surface_texture << "\n";
            }
        } else if (!args.surface_texture_dir.empty()) {
            for (const fs::path& texture_path : listTexturePaths(args.surface_texture_dir)) {
                cv::Mat texture_bgr = cv::imread(texture_path.string(), cv::IMREAD_COLOR);
                if (texture_bgr.empty()) {
                    throw std::runtime_error("Could not read surface texture image: " + texture_path.string());
                }
                surface_texture_entries.push_back(TextureEntry{
                    texture_path,
                    fitTextureToReference(texture_bgr, ref_bgr.size())
                });
            }
            if (surface_texture_entries.empty()) {
                throw std::runtime_error("No texture images found in: " + args.surface_texture_dir);
            }
            if (g_info_enabled) {
                std::cout << "[INFO] Surface textures: " << surface_texture_entries.size()
                          << " from " << args.surface_texture_dir << "\n";
            }
        }

        DetectionResult ref_tags = detectAprilTags(ref_bgr);
        if (g_info_enabled) {
            std::cout << "[INFO] Reference tags detected: " << ref_tags.detections.size() << "\n";
        }
        SheetTrackerConfig sheet_tracker_config;
        applyTrackerEnvOverrides(&sheet_tracker_config);
        bool sheet_tracker_enabled = envInt("SHADOW_SHEET_TRACKER", 1) != 0;
        cv::Mat reference_tag_contrast = makeTagContrastImage(ref_bgr);
        SheetTrackerState sheet_tracker_state;
        if (sheet_tracker_enabled) {
            sheet_tracker_state.referenceFeaturePoints = buildReferenceFeaturePoints(
                ref_bgr,
                reference_tag_contrast,
                sheet_tracker_config);
            if (g_info_enabled) {
                std::cout << "[INFO] Sheet tracker reference features: "
                          << sheet_tracker_state.referenceFeaturePoints.size() << "\n";
            }
        }

        cv::VideoCapture cap(args.video);
        if (!cap.isOpened()) {
            throw std::runtime_error("Could not open video: " + args.video);
        }

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        bool write_texture_dir_video = args.frames.empty() && !surface_texture_entries.empty();
        bool write_final_overlay_video = args.frames.empty() && (!surface_texture_ref_bgr.empty() || write_texture_dir_video);
        if (write_final_overlay_video) {
            int frame_width = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FRAME_WIDTH)));
            int frame_height = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
            double fps = cap.get(cv::CAP_PROP_FPS);
            if (frame_width <= 0 || frame_height <= 0) {
                throw std::runtime_error("Could not determine source video dimensions");
            }
            if (!(fps > 0.0) || !std::isfinite(fps)) {
                fps = 30.0;
            }

            auto openWriter = [&](const fs::path& output_path) -> cv::VideoWriter {
                cv::VideoWriter writer;
                int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                if (!writer.open(output_path.string(), fourcc, fps, cv::Size(frame_width, frame_height), true)) {
                    fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                    if (!writer.open(output_path.string(), fourcc, fps, cv::Size(frame_width, frame_height), true)) {
                        throw std::runtime_error("Could not open output video: " + output_path.string());
                    }
                }
                return writer;
            };

            fs::path video_stem = fs::path(args.video).stem();
            fs::path final_overlay_video_path = fs::path(args.outdir) / "final_overlay.avi";
            cv::VideoWriter writer;
            std::vector<TextureVideoWriter> texture_writers;
            if (write_texture_dir_video) {
                for (const TextureEntry& entry : surface_texture_entries) {
                    fs::path output_path = fs::path(args.outdir) /
                        (video_stem.string() + "_" + entry.path.stem().string() + ".avi");
                    TextureVideoWriter texture_writer;
                    texture_writer.texture_path = entry.path;
                    texture_writer.output_path = output_path;
                    texture_writer.texture_ref_bgr = entry.texture_ref_bgr;
                    texture_writer.writer = openWriter(output_path);
                    texture_writers.push_back(std::move(texture_writer));
                }
            } else {
                writer = openWriter(final_overlay_video_path);
            }

            if (g_info_enabled) {
                std::cout << "[INFO] Total frames: " << total_frames << "\n";
                if (write_texture_dir_video) {
                    std::cout << "[INFO] Writing " << texture_writers.size()
                              << " texture overlay videos"
                              << " (" << frame_width << "x" << frame_height << " @ " << fps << " fps)\n";
                    for (const TextureVideoWriter& texture_writer : texture_writers) {
                        std::cout << "[INFO] Texture output: "
                                  << texture_writer.texture_path.filename() << " -> "
                                  << texture_writer.output_path << "\n";
                    }
                } else {
                    std::cout << "[INFO] Writing final overlay video: " << final_overlay_video_path
                              << " (" << frame_width << "x" << frame_height << " @ " << fps << " fps)\n";
                }
            } else if (write_texture_dir_video) {
                std::cout << "[PROGRESS] Writing " << texture_writers.size()
                          << " texture videos for " << total_frames << " frames\n" << std::flush;
            } else {
                std::cout << "[PROGRESS] Writing final overlay video for "
                          << total_frames << " frames\n" << std::flush;
            }

            int ok_count = 0;
            int written_count = 0;
            int progress_interval = std::clamp(envInt("SHADOW_VIDEO_PROGRESS_INTERVAL", 100), 1, 1000000);
            int frame_workers = std::clamp(
                envInt("SHADOW_FRAME_WORKERS", std::min(12, std::max(1, cv::getNumberOfCPUs()))),
                1,
                std::max(1, cv::getNumberOfCPUs()));
            if (frame_workers > 1) {
                cv::setNumThreads(1);
            }
            if (g_info_enabled) {
                std::cout << "[INFO] Frame workers: " << frame_workers << "\n";
                if (sheet_tracker_enabled && frame_workers > 1) {
                    std::cout << "[INFO] Sheet tracker enabled; tracking remains ordered and frame rendering uses workers\n";
                }
            }
            int texture_workers = write_texture_dir_video
                ? std::clamp(
                    envInt("SHADOW_TEXTURE_WORKERS", 1),
                    1,
                    std::max(1, static_cast<int>(texture_writers.size())))
                : 1;
            if (write_texture_dir_video && g_info_enabled) {
                std::cout << "[INFO] Texture workers: " << texture_workers << "\n";
            }

            struct VideoFrameResult {
                int index = -1;
                bool ok = false;
                cv::Mat final_overlay;
                FrameRenderData render_data;
            };

            auto process_video_frame = [&](
                int idx,
                cv::Mat frame) -> VideoFrameResult {
                VideoFrameResult result;
                result.index = idx;
                result.ok = processFrame(
                    idx,
                    frame,
                    ref_bgr,
                    surface_texture_ref_bgr,
                    ref_tags,
                    reference_tag_contrast,
                    sheet_tracker_enabled ? &sheet_tracker_config : nullptr,
                    sheet_tracker_enabled ? &sheet_tracker_state : nullptr,
                    args,
                    args.outdir,
                    framePrefix(idx),
                    false,
                    nullptr,
                    write_texture_dir_video ? nullptr : &result.final_overlay,
                    write_texture_dir_video ? &result.render_data : nullptr);
                return result;
            };

            auto render_video_geometry = [&](
                FrameGeometry geometry) -> VideoFrameResult {
                VideoFrameResult result;
                result.index = geometry.index;
                if (!geometry.ok || geometry.frame_bgr.empty()) {
                    return result;
                }
                result.ok = processFrame(
                    geometry.index,
                    geometry.frame_bgr,
                    ref_bgr,
                    surface_texture_ref_bgr,
                    ref_tags,
                    reference_tag_contrast,
                    nullptr,
                    nullptr,
                    args,
                    args.outdir,
                    framePrefix(geometry.index),
                    false,
                    nullptr,
                    write_texture_dir_video ? nullptr : &result.final_overlay,
                    write_texture_dir_video ? &result.render_data : nullptr,
                    &geometry);
                return result;
            };

            auto render_texture_writer = [&](TextureVideoWriter& texture_writer, const FrameRenderData& render_data) {
                if (!render_data.shadow_factor_bgr.empty() &&
                    !render_data.board_render_mask.empty() &&
                    !render_data.occlusion_render_mask.empty()) {
                    renderTextureOverlayFused(
                        render_data,
                        texture_writer.texture_ref_bgr,
                        &texture_writer.warped_texture,
                        &texture_writer.overlay);
                } else {
                    texture_writer.overlay = renderTextureOverlay(
                        render_data,
                        texture_writer.texture_ref_bgr);
                }
                if (texture_writer.overlay.size() != cv::Size(frame_width, frame_height)) {
                    cv::resize(texture_writer.overlay, texture_writer.overlay, cv::Size(frame_width, frame_height), 0.0, 0.0, cv::INTER_LINEAR);
                }
            };

            auto write_result = [&](VideoFrameResult result) {
                if (write_texture_dir_video) {
                    if (result.ok && !result.render_data.frame_bgr.empty()) {
                        if (texture_workers <= 1 || texture_writers.size() <= 1) {
                            for (TextureVideoWriter& texture_writer : texture_writers) {
                                render_texture_writer(texture_writer, result.render_data);
                            }
                        } else {
                            std::vector<std::future<void>> render_futures;
                            size_t worker_count = static_cast<size_t>(texture_workers);
                            size_t chunk_size = (texture_writers.size() + worker_count - 1) / worker_count;
                            for (size_t start = 0; start < texture_writers.size(); start += chunk_size) {
                                size_t end = std::min(texture_writers.size(), start + chunk_size);
                                render_futures.push_back(std::async(
                                    std::launch::async,
                                    [&, start, end]() {
                                        for (size_t i = start; i < end; ++i) {
                                            render_texture_writer(texture_writers[i], result.render_data);
                                        }
                                    }));
                            }
                            for (std::future<void>& future : render_futures) {
                                future.get();
                            }
                        }

                        for (TextureVideoWriter& texture_writer : texture_writers) {
                            texture_writer.writer.write(texture_writer.overlay);
                        }
                        ++ok_count;
                    } else {
                        cv::Mat fallback(frame_height, frame_width, CV_8UC3, cv::Scalar(0, 0, 0));
                        for (TextureVideoWriter& texture_writer : texture_writers) {
                            texture_writer.writer.write(fallback);
                        }
                    }
                } else {
                    if (result.ok && !result.final_overlay.empty()) {
                        if (result.final_overlay.size() != cv::Size(frame_width, frame_height)) {
                            cv::resize(result.final_overlay, result.final_overlay, cv::Size(frame_width, frame_height), 0.0, 0.0, cv::INTER_LINEAR);
                        }
                        writer.write(result.final_overlay);
                        ++ok_count;
                    } else {
                        writer.write(cv::Mat(frame_height, frame_width, CV_8UC3, cv::Scalar(0, 0, 0)));
                    }
                }
                ++written_count;
                if (written_count == 1 || written_count % progress_interval == 0 ||
                    (total_frames > 0 && written_count >= total_frames)) {
                    std::cout << "[PROGRESS] wrote " << written_count;
                    if (total_frames > 0) {
                        std::cout << "/" << total_frames;
                    }
                    std::cout << " frames\n" << std::flush;
                }
            };

            if (frame_workers <= 1) {
                for (int idx = 0; ; ++idx) {
                    cv::Mat frame;
                    if (!cap.read(frame) || frame.empty()) {
                        break;
                    }
                    write_result(process_video_frame(idx, std::move(frame)));
                }
            } else if (sheet_tracker_enabled) {
                std::deque<std::future<VideoFrameResult>> pending;
                const size_t max_pending = static_cast<size_t>(frame_workers * 2);
                for (int idx = 0; ; ++idx) {
                    cv::Mat frame;
                    if (!cap.read(frame) || frame.empty()) {
                        break;
                    }
                    FrameGeometry geometry;
                    computeFrameGeometry(
                        idx,
                        std::move(frame),
                        ref_bgr,
                        ref_tags,
                        reference_tag_contrast,
                        &sheet_tracker_config,
                        &sheet_tracker_state,
                        args,
                        args.outdir,
                        framePrefix(idx),
                        false,
                        &geometry,
                        nullptr);
                    pending.push_back(std::async(
                        std::launch::async,
                        render_video_geometry,
                        std::move(geometry)));
                    if (pending.size() >= max_pending) {
                        write_result(pending.front().get());
                        pending.pop_front();
                    }
                }
                while (!pending.empty()) {
                    write_result(pending.front().get());
                    pending.pop_front();
                }
            } else {
                std::deque<std::future<VideoFrameResult>> pending;
                const size_t max_pending = static_cast<size_t>(frame_workers * 2);
                for (int idx = 0; ; ++idx) {
                    cv::Mat frame;
                    if (!cap.read(frame) || frame.empty()) {
                        break;
                    }
                    pending.push_back(std::async(
                        std::launch::async,
                        process_video_frame,
                        idx,
                        std::move(frame)));
                    if (pending.size() >= max_pending) {
                        write_result(pending.front().get());
                        pending.pop_front();
                    }
                }
                while (!pending.empty()) {
                    write_result(pending.front().get());
                    pending.pop_front();
                }
            }

            if (write_texture_dir_video) {
                for (TextureVideoWriter& texture_writer : texture_writers) {
                    texture_writer.writer.release();
                }
                std::cout << "[DONE] Finished. texture overlay videos written for "
                          << ok_count << "/" << written_count << " frames across "
                          << texture_writers.size() << " outputs\n";
            } else {
                writer.release();
                std::cout << "[DONE] Finished. final_overlay video written for "
                          << ok_count << "/" << written_count << " frames: "
                          << final_overlay_video_path << "\n";
            }
            printProfileSummary();
            return 0;
        }

        std::vector<int> frame_indices = args.frames.empty()
            ? sampleFrameIndices(total_frames, args.num_samples)
            : args.frames;

        std::cout << "[INFO] Total frames: " << total_frames << "\n";
        std::cout << "[INFO] Testing frames:";
        for (int idx : frame_indices) {
            std::cout << " " << idx;
        }
        std::cout << "\n";

        int ok_count = 0;
        int current_pos = -1;
        int tracker_preroll_frames = sheet_tracker_enabled
            ? std::clamp(envInt("SHADOW_TRACKER_PREROLL_FRAMES", 30), 0, 1000)
            : 0;
        std::vector<ProcessedFrameArtifacts> processed_artifacts;
        std::vector<int> processing_indices = frame_indices;
        std::sort(processing_indices.begin(), processing_indices.end());
        for (int idx : processing_indices) {
            if (idx < current_pos || current_pos < 0 || idx - current_pos > 30) {
                int seek_idx = idx;
                if (tracker_preroll_frames > 0) {
                    seek_idx = std::max(0, idx - tracker_preroll_frames);
                }
                cap.set(cv::CAP_PROP_POS_FRAMES, seek_idx);
                current_pos = seek_idx;
            } else {
                while (current_pos < idx) {
                    if (sheet_tracker_enabled) {
                        cv::Mat skipped_frame;
                        if (!cap.read(skipped_frame) || skipped_frame.empty()) {
                            std::cerr << "[WARN] Could not advance to frame " << idx << "\n";
                            break;
                        }
                        FrameGeometry skipped_geometry;
                        computeFrameGeometry(
                            current_pos,
                            std::move(skipped_frame),
                            ref_bgr,
                            ref_tags,
                            reference_tag_contrast,
                            &sheet_tracker_config,
                            &sheet_tracker_state,
                            args,
                            args.outdir,
                            framePrefix(current_pos),
                            false,
                            &skipped_geometry,
                            nullptr);
                    } else {
                        if (!cap.grab()) {
                            std::cerr << "[WARN] Could not advance to frame " << idx << "\n";
                            break;
                        }
                    }
                    ++current_pos;
                }
            }
            while (current_pos < idx) {
                cv::Mat preroll_frame;
                if (!cap.read(preroll_frame) || preroll_frame.empty()) {
                    std::cerr << "[WARN] Could not pre-roll to frame " << idx << "\n";
                    break;
                }
                FrameGeometry preroll_geometry;
                computeFrameGeometry(
                    current_pos,
                    std::move(preroll_frame),
                    ref_bgr,
                    ref_tags,
                    reference_tag_contrast,
                    &sheet_tracker_config,
                    &sheet_tracker_state,
                    args,
                    args.outdir,
                    framePrefix(current_pos),
                    false,
                    &preroll_geometry,
                    nullptr);
                ++current_pos;
            }
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "[WARN] Could not read frame " << idx << "\n";
                continue;
            }
            ++current_pos;
            ProcessedFrameArtifacts artifacts;
            if (processFrame(
                    idx,
                    frame,
                    ref_bgr,
                    surface_texture_ref_bgr,
                    ref_tags,
                    reference_tag_contrast,
                    sheet_tracker_enabled ? &sheet_tracker_config : nullptr,
                    sheet_tracker_enabled ? &sheet_tracker_state : nullptr,
                    args,
                    args.outdir,
                    framePrefix(idx),
                    true,
                    &artifacts,
                    nullptr,
                    nullptr)) {
                ++ok_count;
                if (!artifacts.hand_stylus_mask.empty()) {
                    processed_artifacts.push_back(std::move(artifacts));
                }
            }
        }

        std::cout << "[DONE] Finished. processed "
                  << ok_count << "/" << frame_indices.size() << " frames.\n";
        printProfileSummary();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[ERROR] " << exc.what() << "\n";
        return 1;
    }
}
