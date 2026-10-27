#include "shadow_pipeline_v2.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --video VIDEO --reference IMAGE --outdir DIR [options]\n\n"
        << "Options:\n"
        << "  --surface_texture IMAGE      One texture image.\n"
        << "  --surface_texture_dir DIR    Directory of texture images.\n"
        << "  --frames N [N ...]           Specific frames for diagnostics.\n"
        << "  --outputs LIST               Passed through to the draft-compatible backend.\n"
        << "  --reference-scale SCALE      Scale after rotating reference. Default: 1.0\n"
        << "  --frame-workers N            Frame/render workers. Default: 8\n"
        << "  --tag-detect-interval N      AprilTag cadence after tracker lock. Default: 5\n";
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

shadow_v2::Options parseArgs(int argc, char** argv) {
    shadow_v2::Options options;
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
        } else if (arg == "--surface_texture") {
            options.texture_path = needValue("--surface_texture");
        } else if (arg == "--surface_texture_dir") {
            options.texture_dir = needValue("--surface_texture_dir");
        } else if (arg == "--frames") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                options.frames.push_back(intArg(argv[++i], 0));
            }
        } else if (arg == "--outputs") {
            options.outputs = needValue("--outputs");
        } else if (arg == "--reference-scale") {
            options.reference_scale = doubleArg(needValue("--reference-scale"), options.reference_scale);
        } else if (arg == "--frame-workers") {
            options.frame_workers = intArg(needValue("--frame-workers"), options.frame_workers);
        } else if (arg == "--tag-detect-interval") {
            options.tag_detect_interval = intArg(needValue("--tag-detect-interval"), options.tag_detect_interval);
        } else if (arg == "--debug") {
            options.write_debug_frames = true;
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
    if (options.texture_path.empty() && options.texture_dir.empty()) {
        throw std::runtime_error("One of --surface_texture or --surface_texture_dir is required");
    }
    if (!options.texture_path.empty() && !options.texture_dir.empty()) {
        throw std::runtime_error("Use either --surface_texture or --surface_texture_dir, not both");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        shadow_v2::Options options = parseArgs(argc, argv);
        int exit_code = shadow_v2::runVideoV2(options);
        if (exit_code == 0) {
            std::cout << "[DONE] v2 run completed\n";
        }
        return exit_code;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
