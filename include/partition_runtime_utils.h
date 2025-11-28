#ifndef PARTITION_RUNTIME_UTILS_H
#define PARTITION_RUNTIME_UTILS_H

#include "config_loader.h"
#include "convolution_engine.h"

namespace PartitionRuntime {

struct RuntimeRequest {
    bool partitionRequested = false;
    bool eqEnabled = false;
    bool crossfeedEnabled = false;
};

void applyPartitionPolicy(const RuntimeRequest& request,
                          ConvolutionEngine::GPUUpsampler& upsampler, AppConfig& config,
                          const char* daemonTag);

}  // namespace PartitionRuntime

#endif  // PARTITION_RUNTIME_UTILS_H


