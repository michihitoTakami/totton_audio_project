// Buffer manager for coordinating producer/consumer threads on the playback path.
#pragma once

#include "io/audio_ring_buffer.h"
#include "io/playback_buffer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

namespace daemon_output {

class PlaybackBufferManager {
   public:
    using CapacityProvider = std::function<size_t()>;

    explicit PlaybackBufferManager(CapacityProvider capacityProvider);

    // Queue stats (frames, per-channel).
    size_t queuedFramesLocked() const;

    // Producer API
    bool enqueue(const float* left, const float* right, size_t frames, int outputRate,
                 size_t& storedFrames, size_t& droppedFrames);

    // Consumer API
    bool readPlanar(float* dstLeft, float* dstRight, size_t frames);

    // Notify/wait primitives
    std::mutex& mutex();
    std::condition_variable& cv();

    void reset();

    void throttleProducerIfFull(const std::atomic<bool>& running,
                                const std::function<int()>& currentOutputRate);

   private:
    void ensureCapacityLocked();
    void dropFramesLocked(size_t frames);

    CapacityProvider capacityProvider_;
    size_t capacityFrames_{0};
    AudioRingBuffer outputLeft_;
    AudioRingBuffer outputRight_;
    std::vector<float> dropScratch_;
    mutable std::mutex bufferMutex_;
    std::condition_variable bufferCv_;
    std::chrono::steady_clock::time_point lastThrottleWarn_;
};

}  // namespace daemon_output
