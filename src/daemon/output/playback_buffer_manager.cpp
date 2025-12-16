// SPDX-License-Identifier: MIT
#include "daemon/output/playback_buffer_manager.h"

#include "core/daemon_constants.h"
#include "logging/logger.h"

#include <algorithm>

namespace daemon_output {

PlaybackBufferManager::PlaybackBufferManager(CapacityProvider capacityProvider)
    : capacityProvider_(std::move(capacityProvider)),
      lastThrottleWarn_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

std::vector<float>& PlaybackBufferManager::left() {
    return outputLeft_;
}

std::vector<float>& PlaybackBufferManager::right() {
    return outputRight_;
}

size_t& PlaybackBufferManager::readPos() {
    return readPos_;
}

std::mutex& PlaybackBufferManager::mutex() {
    return bufferMutex_;
}

std::condition_variable& PlaybackBufferManager::cv() {
    return bufferCv_;
}

size_t PlaybackBufferManager::queuedFramesLocked() const {
    if (outputLeft_.size() <= readPos_) {
        return 0;
    }
    return outputLeft_.size() - readPos_;
}

void PlaybackBufferManager::reset() {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    outputLeft_.clear();
    outputRight_.clear();
    readPos_ = 0;
}

void PlaybackBufferManager::throttleProducerIfFull(const std::atomic<bool>& running,
                                                   const std::function<int()>& currentOutputRate) {
    size_t capacity = capacityProvider_ ? capacityProvider_() : 0;
    if (capacity == 0) {
        return;
    }

    // Hysteresis to avoid wake/sleep oscillation.
    size_t high = std::max<size_t>(1, (capacity * 9) / 10);
    size_t low = std::min(high, std::max<size_t>(1, (capacity * 7) / 10));

    std::unique_lock<std::mutex> lock(bufferMutex_);
    size_t queued = queuedFramesLocked();
    if (queued < high) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    if (now - lastThrottleWarn_ > std::chrono::seconds(5)) {
        int outputRate = currentOutputRate ? currentOutputRate() : 0;
        if (outputRate <= 0) {
            outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
        }
        double queuedSec = static_cast<double>(queued) / static_cast<double>(outputRate);
        double capSec = static_cast<double>(capacity) / static_cast<double>(outputRate);
        LOG_WARN(
            "Throttling input: output queue near full (queued={} frames / {:.3f}s, cap={} frames / "
            "{:.3f}s)",
            queued, queuedSec, capacity, capSec);
        lastThrottleWarn_ = now;
    }

    bufferCv_.wait_for(lock, std::chrono::milliseconds(200), [&]() {
        return !running.load(std::memory_order_acquire) || queuedFramesLocked() <= low;
    });
}

}  // namespace daemon_output
