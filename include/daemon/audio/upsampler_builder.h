#pragma once

#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/partition_runtime_utils.h"

#include <atomic>
#include <memory>

namespace daemon_audio {

enum class UpsamplerBuildStatus { Success, Failure, Interrupted };

struct UpsamplerBuildResult {
    UpsamplerBuildStatus status = UpsamplerBuildStatus::Failure;
    std::unique_ptr<ConvolutionEngine::IAudioUpsampler> upsampler;
    ConvolutionEngine::RateFamily initialRateFamily = ConvolutionEngine::RateFamily::RATE_44K;
    int currentInputRate = 0;
    int currentOutputRate = 0;
};

UpsamplerBuildResult buildUpsampler(AppConfig& config, int inputSampleRate,
                                    const PartitionRuntime::RuntimeRequest& partitionRequest,
                                    const std::atomic<bool>& runningFlag);

}  // namespace daemon_audio
