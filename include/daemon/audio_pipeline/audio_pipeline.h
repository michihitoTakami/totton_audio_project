#pragma once

#include "config_loader.h"
#include "convolution_engine.h"
#include "crossfeed_engine.h"
#include "daemon/audio_pipeline/streaming_cache_manager.h"
#include "daemon/metrics/runtime_stats.h"
#include "daemon_constants.h"
#include "logging/logger.h"
#include "playback_buffer.h"

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
    std::vector<float>* outputBufferLeft = nullptr;
    std::vector<float>* outputBufferRight = nullptr;
    size_t* outputReadPos = nullptr;
    std::mutex* bufferMutex = nullptr;
    std::condition_variable* bufferCv = nullptr;
};

struct Upsampler {
    using ProcessFn = std::function<bool(
        const float* inputData, size_t inputFrames,
        ConvolutionEngine::StreamFloatVector& outputData, cudaStream_t stream,
        ConvolutionEngine::StreamFloatVector& streamInputBuffer, size_t& streamInputAccumulated)>;

    ProcessFn process;
    cudaStream_t streamLeft = 0;
    cudaStream_t streamRight = 0;
    bool available = false;
};

struct Dependencies {
    const AppConfig* config = nullptr;
    Upsampler upsampler;
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

class AudioPipeline {
   public:
    explicit AudioPipeline(Dependencies deps);
    bool process(const float* inputSamples, uint32_t nFrames);
    void trimOutputBuffer(size_t minFramesToRemove);
    const BufferResources& bufferResources() const;

   private:
    bool hasBufferState() const;
    bool isUpsamplerAvailable() const;
    bool isOutputReady() const;
    void logDroppingInput();
    void trimInternal(size_t minFramesToRemove);
    float computeStereoPeak(const float* left, const float* right, size_t frames) const;

    template <typename Container>
    size_t enqueueOutputFramesLocked(const Container& left, const Container& right);

    Dependencies deps_;
    std::chrono::steady_clock::time_point lastDropWarn_{std::chrono::steady_clock::now() -
                                                        std::chrono::seconds(6)};
};

template <typename Container>
size_t AudioPipeline::enqueueOutputFramesLocked(const Container& left, const Container& right) {
    if (!hasBufferState() || !deps_.buffer.bufferMutex || !deps_.buffer.outputBufferLeft ||
        !deps_.buffer.outputBufferRight || !deps_.buffer.outputReadPos) {
        return 0;
    }

    size_t framesAvailable = std::min(left.size(), right.size());
    if (framesAvailable == 0) {
        return 0;
    }

    size_t capacityFrames = 1;
    if (deps_.maxOutputBufferFrames) {
        capacityFrames = std::max<size_t>(1, deps_.maxOutputBufferFrames());
    }

    size_t bufferSize = deps_.buffer.outputBufferLeft->size();
    size_t currentFrames = (bufferSize >= *deps_.buffer.outputReadPos)
                               ? (bufferSize - *deps_.buffer.outputReadPos)
                               : 0;
    auto decision =
        PlaybackBuffer::planCapacityEnforcement(currentFrames, framesAvailable, capacityFrames);

    size_t totalDropped = decision.dropFromExisting + decision.newDataOffset;
    if (totalDropped > 0) {
        int outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
        if (deps_.currentOutputRate) {
            outputRate = deps_.currentOutputRate();
            if (outputRate <= 0) {
                outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
            }
        }
        float seconds = static_cast<float>(totalDropped) / static_cast<float>(outputRate);
        LOG_WARN(
            "Output buffer overflow: dropping {} frames ({:.3f}s) [queued={}, incoming={}, max={}]",
            totalDropped, seconds, currentFrames, framesAvailable, capacityFrames);
        runtime_stats::addDroppedFrames(totalDropped);
    }

    if (decision.dropFromExisting > 0) {
        *deps_.buffer.outputReadPos += decision.dropFromExisting;
    }

    size_t minFramesToRemove = capacityFrames;
    trimInternal(minFramesToRemove);

    if (decision.framesToStore == 0) {
        return 0;
    }

    size_t startIndex = framesAvailable - decision.framesToStore;
    auto startOffset = static_cast<std::ptrdiff_t>(startIndex);
    auto endOffset = static_cast<std::ptrdiff_t>(framesAvailable);
    deps_.buffer.outputBufferLeft->insert(deps_.buffer.outputBufferLeft->end(),
                                          left.begin() + startOffset, left.begin() + endOffset);
    deps_.buffer.outputBufferRight->insert(deps_.buffer.outputBufferRight->end(),
                                           right.begin() + startOffset, right.begin() + endOffset);
    return decision.framesToStore;
}

}  // namespace audio_pipeline
