#ifndef PARTITION_RUNTIME_UTILS_H
#define PARTITION_RUNTIME_UTILS_H

#include "convolution_engine.h"
#include "core/config_loader.h"

namespace PartitionRuntime {

struct RuntimeRequest {
    bool partitionRequested = false;
    bool eqEnabled = false;
    bool crossfeedEnabled = false;
};

void applyPartitionPolicy(const RuntimeRequest& request, ConvolutionEngine::GPUUpsampler& upsampler,
                          AppConfig& config, const char* daemonTag);

}  // namespace PartitionRuntime

#endif  // PARTITION_RUNTIME_UTILS_H
