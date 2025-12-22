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
    size_t chunkSize = std::min<size_t>(frames, capacityFrames_);
    if (dropScratch_.size() < chunkSize) {
        dropScratch_.assign(chunkSize, 0.0f);
    }
    float* scratch = dropScratch_.data();
    size_t remaining = frames;
    while (remaining > 0) {
        size_t chunk = std::min(remaining, chunkSize);
        outputLeft_.read(scratch, chunk);
        outputRight_.read(scratch, chunk);
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

    // Ensure both channels have room; avoid L/R mismatch.
    if (outputLeft_.availableToWrite() < decision.framesToStore ||
        outputRight_.availableToWrite() < decision.framesToStore) {
        LOG_ERROR("Output buffer write failed: insufficient space after drop");
        return false;
    }

    bool leftOk = outputLeft_.write(leftPtr, decision.framesToStore);
    bool rightOk = outputRight_.write(rightPtr, decision.framesToStore);
    if (!leftOk || !rightOk) {
        LOG_ERROR("Output buffer write failed: write() returned false");
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

bool PlaybackBufferManager::readPlanar(float* dstLeft, float* dstRight, size_t frames) {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    if (!dstLeft || !dstRight || frames == 0) {
        return false;
    }
    if (queuedFramesLocked() < frames) {
        return false;
    }

    if (!outputLeft_.read(dstLeft, frames) || !outputRight_.read(dstRight, frames)) {
        return false;
    }
    // Notify waiting producers that space became available.
    bufferCv_.notify_all();
    return true;
}

void PlaybackBufferManager::throttleProducerIfFull(const std::atomic<bool>& running,
                                                   const std::function<int()>& currentOutputRate,
                                                   size_t incomingFramesHint) {
    size_t capacity = capacityProvider_ ? capacityProvider_() : 0;
    if (capacity == 0) {
        return;
    }

    // Hysteresis to avoid wake/sleep oscillation.
    size_t high = std::max<size_t>(1, (capacity * 9) / 10);
    size_t low = std::min(high, std::max<size_t>(1, (capacity * 7) / 10));

    // Ensure enough headroom for large incoming blocks (e.g., delimiter hop size).
    if (incomingFramesHint > 0) {
        size_t requiredHeadroom = std::min(incomingFramesHint, capacity);
        size_t targetMaxQueued = (capacity > requiredHeadroom) ? (capacity - requiredHeadroom) : 0;
        high = std::max<size_t>(1, std::min(high, targetMaxQueued));
        low = std::min(low, targetMaxQueued);
        if (low > high) {
            low = high;
        }
    }

    std::unique_lock<std::mutex> lock(bufferMutex_);
    while (running.load(std::memory_order_acquire)) {
        size_t queued = queuedFramesLocked();
        if (queued < high) {
            break;
        }

        auto now = std::chrono::steady_clock::now();
        if (now - lastThrottleWarn_ > std::chrono::seconds(5)) {
            int outputRate = currentOutputRate ? currentOutputRate() : 0;
            if (outputRate <= 0) {
                outputRate = DaemonConstants::DEFAULT_OUTPUT_SAMPLE_RATE;
            }
            double queuedSec = static_cast<double>(queued) / static_cast<double>(outputRate);
            double capSec = static_cast<double>(capacity) / static_cast<double>(outputRate);
            if (incomingFramesHint > 0) {
                LOG_WARN(
                    "Throttling input: output queue near full (queued={} frames / {:.3f}s, "
                    "cap={} frames / {:.3f}s, incoming_hint={} frames)",
                    queued, queuedSec, capacity, capSec, incomingFramesHint);
            } else {
                LOG_WARN(
                    "Throttling input: output queue near full (queued={} frames / {:.3f}s, "
                    "cap={} frames / {:.3f}s)",
                    queued, queuedSec, capacity, capSec);
            }
            lastThrottleWarn_ = now;
        }

        bufferCv_.wait_for(lock, std::chrono::milliseconds(200), [&]() {
            return !running.load(std::memory_order_acquire) || queuedFramesLocked() <= low;
        });

        if (queuedFramesLocked() <= low) {
            break;
        }
    }
}

}  // namespace daemon_output
