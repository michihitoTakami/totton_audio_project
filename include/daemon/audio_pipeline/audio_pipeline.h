#pragma once

#include "audio/soft_mute.h"
#include "convolution_engine.h"
#include "core/config_loader.h"
#include "core/daemon_constants.h"
#include "crossfeed_engine.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/metrics/runtime_stats.h"
#include "daemon/output/playback_buffer_manager.h"
#include "logging/logger.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace audio_pipeline {

struct BufferResources {
    daemon_output::PlaybackBufferManager* playbackBuffer = nullptr;
};

struct Upsampler {
    using ProcessFn = std::function<bool(
        const float* inputData, size_t inputFrames,
        ConvolutionEngine::StreamFloatVector& outputData, cudaStream_t stream,
        ConvolutionEngine::StreamFloatVector& streamInputBuffer, size_t& streamInputAccumulated)>;

    ProcessFn process;
    cudaStream_t streamLeft = nullptr;
    cudaStream_t streamRight = nullptr;
    bool available = false;
};

struct OutputState {
    std::atomic<float>* outputGain = nullptr;
    std::atomic<float>* limiterGain = nullptr;
    std::atomic<float>* effectiveGain = nullptr;
};

struct Dependencies {
    const AppConfig* config = nullptr;
    Upsampler upsampler;
    OutputState output;
    std::atomic<bool>* fallbackActive = nullptr;
    std::atomic<bool>* outputReady = nullptr;
    std::atomic<bool>* crossfeedEnabled = nullptr;
    CrossfeedEngine::HRTFProcessor* crossfeedProcessor = nullptr;
    std::mutex* crossfeedMutex = nullptr;
    std::vector<float>* cfStreamInputLeft = nullptr;
    std::vector<float>* cfStreamInputRight = nullptr;
    size_t* cfStreamAccumulatedLeft = nullptr;
    size_t* cfStreamAccumulatedRight = nullptr;
    std::vector<float>* cfOutputLeft = nullptr;
    std::vector<float>* cfOutputRight = nullptr;

    ConvolutionEngine::StreamFloatVector* streamInputLeft = nullptr;
    ConvolutionEngine::StreamFloatVector* streamInputRight = nullptr;
    size_t* streamAccumulatedLeft = nullptr;
    size_t* streamAccumulatedRight = nullptr;
    ConvolutionEngine::StreamFloatVector* upsamplerOutputLeft = nullptr;
    ConvolutionEngine::StreamFloatVector* upsamplerOutputRight = nullptr;

    streaming_cache::StreamingCacheManager* streamingCacheManager = nullptr;
    BufferResources buffer;
    std::function<size_t()> maxOutputBufferFrames;
    std::function<int()> currentOutputRate;
    std::mutex* inputMutex = nullptr;
};

struct RenderResult {
    size_t framesRequested = 0;
    size_t framesRendered = 0;
    bool wroteSilence = false;
};

class AudioPipeline {
   public:
    explicit AudioPipeline(Dependencies deps);
    bool process(const float* inputSamples, uint32_t nFrames);
    RenderResult renderOutput(size_t frames, std::vector<int32_t>& interleavedOut,
                              std::vector<float>& floatScratch, SoftMute::Controller* softMute);
    void trimOutputBuffer(size_t minFramesToRemove);
    const BufferResources& bufferResources() const;

   private:
    bool hasBufferState() const;
    bool isUpsamplerAvailable() const;
    bool isOutputReady() const;
    void logDroppingInput();
    void trimInternal(size_t minFramesToRemove);
    float computeStereoPeak(const float* left, const float* right, size_t frames) const;
    float applyOutputLimiter(float* interleaved, size_t frames);

    template <typename Container>
    size_t enqueueOutputFramesLocked(const Container& left, const Container& right);

    Dependencies deps_;
    std::chrono::steady_clock::time_point lastDropWarn_{std::chrono::steady_clock::now() -
                                                        std::chrono::seconds(6)};
    std::chrono::steady_clock::time_point lastInputLockWarn_{std::chrono::steady_clock::now() -
                                                             std::chrono::seconds(6)};
    std::chrono::steady_clock::time_point lastCrossfeedLockWarn_{std::chrono::steady_clock::now() -
                                                                 std::chrono::seconds(6)};

    // RT パスで毎回 std::vector を生成しないためのワークバッファ (Issue #894)
    std::vector<float> workLeft_;
    std::vector<float> workRight_;
};

template <typename Container>
size_t AudioPipeline::enqueueOutputFramesLocked(const Container& left, const Container& right) {
    if (!hasBufferState() || !deps_.buffer.playbackBuffer) {
        return 0;
    }

    size_t framesAvailable = std::min(left.size(), right.size());
    if (framesAvailable == 0) {
        return 0;
    }

    int outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
    if (deps_.currentOutputRate) {
        outputRate = deps_.currentOutputRate();
        if (outputRate <= 0) {
            outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
        }
    }

    size_t stored = 0;
    size_t dropped = 0;
    if (!deps_.buffer.playbackBuffer->enqueue(left.data(), right.data(), framesAvailable,
                                              outputRate, stored, dropped)) {
        LOG_ERROR("Failed to enqueue output frames into playback buffer");
        return 0;
    }
    if (dropped > 0) {
        runtime_stats::addDroppedFrames(dropped);
    }
    return stored;
}

}  // namespace audio_pipeline
