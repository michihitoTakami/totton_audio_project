#pragma once

#include "audio/fallback_manager.h"
#include "audio/filter_headroom.h"
#include "core/config_loader.h"
#include "daemon/pcm/dac_manager.h"

#include <atomic>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>

namespace ConvolutionEngine {
class IAudioUpsampler;
}

namespace daemon_output {
class PlaybackBufferManager;
}

namespace runtime_stats {

struct Dependencies {
    const AppConfig* config = nullptr;
    const ConvolutionEngine::IAudioUpsampler* upsampler = nullptr;
    const FilterHeadroomCache* headroomCache = nullptr;
    dac::DacManager* dacManager = nullptr;
    const FallbackManager::Manager* fallbackManager = nullptr;
    const std::atomic<bool>* fallbackActive = nullptr;
    const int* inputSampleRate = nullptr;
    const std::atomic<float>* headroomGain = nullptr;
    const std::atomic<float>* outputGain = nullptr;
    const std::atomic<float>* limiterGain = nullptr;
    const std::atomic<float>* effectiveGain = nullptr;
    const std::atomic<int>* delimiterMode = nullptr;
    const std::atomic<int>* delimiterFallbackReason = nullptr;
    const std::atomic<bool>* delimiterBypassLocked = nullptr;
    const std::atomic<bool>* delimiterEnabled = nullptr;
    const std::atomic<bool>* delimiterWarmup = nullptr;
    const std::atomic<std::size_t>* delimiterQueueSamples = nullptr;
    const std::atomic<double>* delimiterQueueSeconds = nullptr;
    const std::atomic<double>* delimiterLastInferenceMs = nullptr;
    const std::atomic<bool>* delimiterBackendAvailable = nullptr;
    const std::atomic<bool>* delimiterBackendValid = nullptr;
    const std::atomic<int>* delimiterTargetMode = nullptr;
    daemon_output::PlaybackBufferManager* playbackBuffer = nullptr;
    const std::atomic<bool>* outputReady = nullptr;
};

void reset();

void recordClip();
void addSamples(std::size_t count);
void addDroppedFrames(std::size_t count);

// Output starvation / silence rendering diagnostics
void recordRenderedSilenceBlock();
void addRenderedSilenceFrames(std::size_t frames);

// Upsampler diagnostics
// "Need more input" (streaming accumulator not yet full) vs hard error
void recordUpsamplerNeedMoreBlock(bool leftChannel);
void recordUpsamplerErrorBlock(bool leftChannel);

void updateInputPeak(float peak);
void updateUpsamplerPeak(float peak);
void updatePostCrossfeedPeak(float peak);
void updatePostGainPeak(float peak);

std::size_t clipCount();
std::size_t totalSamples();
std::size_t droppedFrames();

std::size_t renderedSilenceBlocks();
std::size_t renderedSilenceFrames();
std::size_t upsamplerNeedMoreBlocksLeft();
std::size_t upsamplerNeedMoreBlocksRight();
std::size_t upsamplerErrorBlocksLeft();
std::size_t upsamplerErrorBlocksRight();

nlohmann::json collect(const Dependencies& deps, std::size_t bufferCapacityFrames);
void writeStatsFile(const Dependencies& deps, std::size_t bufferCapacityFrames,
                    const std::string& path, const nlohmann::json* precomputedStats = nullptr);

}  // namespace runtime_stats
