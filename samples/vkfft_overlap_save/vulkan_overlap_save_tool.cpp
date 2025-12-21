#include "vulkan_overlap_save.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void printUsage() {
    std::cout << "Usage: vulkan_overlap_save_tool --input <in.wav> --output <out.wav> "
                 "[--filter <path.bin>] [--filter-json <path.json>] [--fft-size <pow2>] "
                 "[--chunk-frames <frames>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    VulkanOverlapSaveOptions opts{};
    opts.filterPath = "data/coefficients/filter_44k_8x_2m_min_phase.bin";
    opts.filterMetadataPath = "data/coefficients/filter_44k_8x_2m_min_phase.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            opts.inputPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opts.outputPath = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            opts.filterPath = argv[++i];
        } else if (arg == "--filter-json" && i + 1 < argc) {
            opts.filterMetadataPath = argv[++i];
        } else if (arg == "--fft-size" && i + 1 < argc) {
            opts.fftSizeOverride = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--chunk-frames" && i + 1 < argc) {
            opts.chunkFrames = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--help") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    if (opts.filterMetadataPath.empty() && !opts.filterPath.empty()) {
        auto pos = opts.filterPath.rfind('.');
        if (pos != std::string::npos) {
            opts.filterMetadataPath = opts.filterPath.substr(0, pos) + ".json";
        }
    }

    if (opts.inputPath.empty() || opts.outputPath.empty()) {
        printUsage();
        return 1;
    }

    return runVulkanOverlapSave(opts);
}
