// Buffer manager for coordinating producer/consumer threads on the playback path.
#pragma once

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

    std::vector<float>& left();
    std::vector<float>& right();
    size_t& readPos();
    std::mutex& mutex();
    std::condition_variable& cv();

    // Caller must hold buffer mutex when invoking.
    size_t queuedFramesLocked() const;

    void reset();

    void throttleProducerIfFull(const std::atomic<bool>& running,
                                const std::function<int()>& currentOutputRate);

   private:
    CapacityProvider capacityProvider_;
    std::vector<float> outputLeft_;
    std::vector<float> outputRight_;
    size_t readPos_{0};
    mutable std::mutex bufferMutex_;
    std::condition_variable bufferCv_;
    std::chrono::steady_clock::time_point lastThrottleWarn_;
};

}  // namespace daemon_output
