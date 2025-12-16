// SPDX-License-Identifier: MIT
#include "daemon/output/playback_buffer_manager.h"

#include "core/daemon_constants.h"
#include "logging/logger.h"

#include <algorithm>
#include <vector>

namespace daemon_output {

PlaybackBufferManager::PlaybackBufferManager(CapacityProvider capacityProvider)
    : capacityProvider_(std::move(capacityProvider)),
      lastThrottleWarn_(std::chrono::steady_clock::now() - std::chrono::seconds(10)) {}

std::mutex& PlaybackBufferManager::mutex() {
    return bufferMutex_;
}

std::condition_variable& PlaybackBufferManager::cv() {
    return bufferCv_;
}

size_t PlaybackBufferManager::queuedFramesLocked() const {
    size_t left = outputLeft_.availableToRead();
    size_t right = outputRight_.availableToRead();
    return std::min(left, right);
}

void PlaybackBufferManager::reset() {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    ensureCapacityLocked();
    outputLeft_.clear();
    outputRight_.clear();
}

void PlaybackBufferManager::ensureCapacityLocked() {
    size_t desired = capacityProvider_ ? capacityProvider_() : 0;
    if (desired == 0 || desired == capacityFrames_) {
        return;
    }
    capacityFrames_ = desired;
    outputLeft_.init(capacityFrames_);
    outputRight_.init(capacityFrames_);
}

void PlaybackBufferManager::dropFramesLocked(size_t frames) {
    if (frames == 0) {
        return;
    }
    std::vector<float> scratch(std::min<size_t>(frames, capacityFrames_), 0.0f);
    size_t remaining = frames;
    while (remaining > 0) {
        size_t chunk = std::min(remaining, scratch.size());
        outputLeft_.read(scratch.data(), chunk);
        outputRight_.read(scratch.data(), chunk);
        remaining -= chunk;
    }
}

bool PlaybackBufferManager::enqueue(const float* left, const float* right, size_t frames,
                                    int outputRate, size_t& storedFrames, size_t& droppedFrames) {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    storedFrames = 0;
    droppedFrames = 0;

    if (!left || !right || frames == 0) {
        return false;
    }

    ensureCapacityLocked();
    if (capacityFrames_ == 0) {
        return false;
    }

    size_t current = queuedFramesLocked();
    auto decision = PlaybackBuffer::planCapacityEnforcement(current, frames, capacityFrames_);
    droppedFrames = decision.dropFromExisting + decision.newDataOffset;

    if (decision.dropFromExisting > 0) {
        dropFramesLocked(decision.dropFromExisting);
    }

    if (decision.framesToStore == 0) {
        if (droppedFrames > 0 && outputRate > 0) {
            double seconds = static_cast<double>(droppedFrames) / static_cast<double>(outputRate);
            LOG_WARN(
                "Output buffer overflow: dropping {} frames ({:.3f}s) [queued={}, incoming={}, "
                "max={}]",
                droppedFrames, seconds, current, frames, capacityFrames_);
        }
        return true;
    }

    size_t startIndex = frames - decision.framesToStore;
    const float* leftPtr = left + startIndex;
    const float* rightPtr = right + startIndex;

    // AudioRingBuffer::write returns false if insufficient space (should not happen after drop).
    bool leftOk = outputLeft_.write(leftPtr, decision.framesToStore);
    bool rightOk = outputRight_.write(rightPtr, decision.framesToStore);
    if (!leftOk || !rightOk) {
        return false;
    }

    storedFrames = decision.framesToStore;

    if (droppedFrames > 0 && outputRate > 0) {
        double seconds = static_cast<double>(droppedFrames) / static_cast<double>(outputRate);
        LOG_WARN(
            "Output buffer overflow: dropping {} frames ({:.3f}s) [queued={}, incoming={}, max={}]",
            droppedFrames, seconds, current, frames, capacityFrames_);
    }

    return true;
}

bool PlaybackBufferManager::readInterleaved(float* dstInterleaved, size_t frames) {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    if (!dstInterleaved || frames == 0) {
        return false;
    }
    if (queuedFramesLocked() < frames) {
        return false;
    }

    std::vector<float> left(frames);
    std::vector<float> right(frames);
    if (!outputLeft_.read(left.data(), frames) || !outputRight_.read(right.data(), frames)) {
        return false;
    }

    for (size_t i = 0; i < frames; ++i) {
        dstInterleaved[i * 2] = left[i];
        dstInterleaved[i * 2 + 1] = right[i];
    }
    return true;
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
