#pragma once

#include "config_loader.h"
#include "daemon/dac_manager.h"
#include "fallback_manager.h"
#include "filter_headroom.h"

#include <atomic>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>

namespace ConvolutionEngine {
class GPUUpsampler;
}

namespace runtime_stats {

struct Dependencies {
    const AppConfig* config = nullptr;
    const ConvolutionEngine::GPUUpsampler* upsampler = nullptr;
    const FilterHeadroomCache* headroomCache = nullptr;
    dac::DacManager* dacManager = nullptr;
    const FallbackManager::Manager* fallbackManager = nullptr;
    const std::atomic<bool>* fallbackActive = nullptr;
    const int* inputSampleRate = nullptr;
    const std::atomic<float>* headroomGain = nullptr;
    const std::atomic<float>* outputGain = nullptr;
    const std::atomic<float>* limiterGain = nullptr;
    const std::atomic<float>* effectiveGain = nullptr;
};

void reset();

void recordClip();
void addSamples(std::size_t count);
void addDroppedFrames(std::size_t count);

void updateInputPeak(float peak);
void updateUpsamplerPeak(float peak);
void updatePostCrossfeedPeak(float peak);
void updatePostGainPeak(float peak);

std::size_t clipCount();
std::size_t totalSamples();
std::size_t droppedFrames();

nlohmann::json collect(const Dependencies& deps, std::size_t bufferCapacityFrames);
void writeStatsFile(const Dependencies& deps, std::size_t bufferCapacityFrames,
                    const std::string& path);

}  // namespace runtime_stats
