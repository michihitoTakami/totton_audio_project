#include "core/partition_runtime_utils.h"

#include <iostream>
#include <string>

namespace PartitionRuntime {

void applyPartitionPolicy(const RuntimeRequest& request, ConvolutionEngine::GPUUpsampler& upsampler,
                          AppConfig& config, const char* daemonTag) {
    const bool partitionActive = upsampler.isPartitionedConvolutionEnabled();
    const std::string tag = daemonTag ? daemonTag : "Daemon";

    if (!request.partitionRequested) {
        if (partitionActive) {
            std::cout << "[Partition][" << tag
                      << "] Warning: GPU partition plan active although config disabled. "
                      << "Reverting to legacy mode." << std::endl;
        }
        config.partitionedConvolution.enabled = false;
        config.eqEnabled = request.eqEnabled;
        config.crossfeed.enabled = request.crossfeedEnabled;
        return;
    }

    if (!partitionActive) {
        std::cout << "[Partition][" << tag
                  << "] Disabled: falling back to legacy streaming (see previous log for reason)"
                  << std::endl;
        config.partitionedConvolution.enabled = false;
        config.eqEnabled = request.eqEnabled;
        config.crossfeed.enabled = request.crossfeedEnabled;
        return;
    }

    config.partitionedConvolution.enabled = true;
    config.eqEnabled = request.eqEnabled;
    config.crossfeed.enabled = false;

    const auto& plan = upsampler.getPartitionPlan();
    const int outputRate = upsampler.getOutputSampleRate();
    std::cout << "[Partition][" << tag << "] Active: " << plan.describe(outputRate) << std::endl;
    std::cout << "[Partition][" << tag
              << "] Stream block: " << upsampler.getStreamValidInputPerBlock() << " input samples ("
              << static_cast<size_t>(upsampler.getStreamValidInputPerBlock()) *
                     static_cast<size_t>(upsampler.getUpsampleRatio())
              << " output)" << std::endl;

    if (request.eqEnabled) {
        std::cout << "[Partition][" << tag << "] EQ enabled for low-latency streaming" << std::endl;
    }
    if (request.crossfeedEnabled) {
        std::cout << "[Partition][" << tag
                  << "] Crossfeed disabled (unsupported in low-latency mode)" << std::endl;
    }
}

}  // namespace PartitionRuntime
