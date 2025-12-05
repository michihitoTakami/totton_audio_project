#ifndef STREAMING_CACHE_MANAGER_H
#define STREAMING_CACHE_MANAGER_H

#include "audio/input_stall_detector.h"
#include "convolution_engine.h"
#include "logging/logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace streaming_cache {

struct StreamingCacheDependencies {
    std::vector<float>* outputBufferLeft = nullptr;
    std::vector<float>* outputBufferRight = nullptr;
    std::mutex* bufferMutex = nullptr;
    size_t* outputReadPos = nullptr;

    ConvolutionEngine::StreamFloatVector* streamInputLeft = nullptr;
    ConvolutionEngine::StreamFloatVector* streamInputRight = nullptr;
    size_t* streamAccumulatedLeft = nullptr;
    size_t* streamAccumulatedRight = nullptr;
    std::mutex* streamingMutex = nullptr;

    ConvolutionEngine::GPUUpsampler** upsamplerPtr = nullptr;

    std::function<void()> onCrossfeedReset;
};

class StreamingCacheManager {
   public:
    explicit StreamingCacheManager(const StreamingCacheDependencies& deps) : deps_(deps) {}

    void handleInputBlock() {
        auto now = std::chrono::steady_clock::now();
        std::int64_t nowNs = AudioInput::toNanoseconds(now);
        std::int64_t previousNs = lastInputTimestampNs_.exchange(nowNs, std::memory_order_acq_rel);
        if (AudioInput::shouldResetAfterStall(previousNs, nowNs)) {
            std::chrono::nanoseconds gap(nowNs - previousNs);
            flushCachesInternal(gap);
        }
    }

    void flushCaches(std::chrono::nanoseconds gap = std::chrono::nanoseconds::zero()) {
        flushCachesInternal(gap);
    }

   private:
    void flushCachesInternal(std::chrono::nanoseconds gap) {
        auto resetPlaybackBuffers = [&]() {
            if (deps_.outputBufferLeft && deps_.outputBufferRight) {
                deps_.outputBufferLeft->clear();
                deps_.outputBufferRight->clear();
            }
            if (deps_.outputReadPos) {
                *deps_.outputReadPos = 0;
            }
        };

        auto resetStreamingBuffers = [&]() {
            if (deps_.streamInputLeft) {
                std::fill(deps_.streamInputLeft->begin(), deps_.streamInputLeft->end(), 0.0f);
            }
            if (deps_.streamInputRight) {
                std::fill(deps_.streamInputRight->begin(), deps_.streamInputRight->end(), 0.0f);
            }
            if (deps_.streamAccumulatedLeft) {
                *deps_.streamAccumulatedLeft = 0;
            }
            if (deps_.streamAccumulatedRight) {
                *deps_.streamAccumulatedRight = 0;
            }
        };

        auto resetStreamingEngine = [&]() {
            if (deps_.upsamplerPtr && *deps_.upsamplerPtr) {
                (*deps_.upsamplerPtr)->resetStreaming();
            }
        };

        bool hasStreamingLock = deps_.streamingMutex != nullptr;
        bool hasBufferLock = deps_.bufferMutex != nullptr;

        if (hasStreamingLock && hasBufferLock) {
            std::scoped_lock<std::mutex, std::mutex> lock(*deps_.streamingMutex,
                                                          *deps_.bufferMutex);
            resetPlaybackBuffers();
            resetStreamingBuffers();
            resetStreamingEngine();
        } else {
            if (hasBufferLock) {
                std::lock_guard<std::mutex> lock(*deps_.bufferMutex);
                resetPlaybackBuffers();
            }
            if (hasStreamingLock) {
                std::lock_guard<std::mutex> streamLock(*deps_.streamingMutex);
                resetStreamingBuffers();
                resetStreamingEngine();
            }
        }

        if (deps_.onCrossfeedReset) {
            deps_.onCrossfeedReset();
        }

        double gapMs = static_cast<double>(gap.count()) / 1'000'000.0;
        LOG_INFO("[Stream] Input gap {:.3f}ms detected → flushing streaming cache", gapMs);
    }

    StreamingCacheDependencies deps_;
    std::atomic<std::int64_t> lastInputTimestampNs_{0};
};

}  // namespace streaming_cache

#endif  // STREAMING_CACHE_MANAGER_H
