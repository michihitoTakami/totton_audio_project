#ifndef GPU_PARTITION_PLAN_H
#define GPU_PARTITION_PLAN_H

#include "config_loader.h"

#include <string>
#include <vector>

namespace ConvolutionEngine {

struct PartitionDescriptor {
    int taps = 0;
    int fftSize = 0;
    int validOutput = 0;
    bool realtime = false;
};

struct PartitionPlan {
    bool enabled = false;
    int totalTaps = 0;
    int realtimeTaps = 0;
    std::vector<PartitionDescriptor> partitions;

    std::string describe(int outputSampleRate) const;
};

PartitionPlan buildPartitionPlan(int totalTaps, int upsampleRatio,
                                 const AppConfig::PartitionedConvolutionConfig& config);

}  // namespace ConvolutionEngine

#endif  // GPU_PARTITION_PLAN_H


