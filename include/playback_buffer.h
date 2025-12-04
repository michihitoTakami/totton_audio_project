#ifndef PLAYBACK_BUFFER_H
#define PLAYBACK_BUFFER_H

#include <algorithm>
#include <cstddef>

namespace PlaybackBuffer {

/**
 * @brief Compute the number of samples required in the output ring buffer
 *        before the ALSA thread wakes up to consume audio.
 *
 * @param periodSize          ALSA period size in samples (per channel).
 * @param crossfeedActive     Whether crossfeed processing is currently enabled.
 * @param crossfeedBlockSize  Crossfeed block size reported by the HRTF engine.
 * @param producerBlockSize   Producer (upsampler) output block size in samples. When crossfeed is
 *                            disabled this value is clamped to [periodSize, periodSize *
 *                            defaultMultiplier] and used as the threshold. Pass 0 to use only the
 *                            default multiplier logic.
 * @param defaultMultiplier   Multiplier applied to periodSize when crossfeed is inactive.
 *
 * @return Samples required before playback should resume.
 */
inline size_t computeReadyThreshold(size_t periodSize, bool crossfeedActive,
                                    size_t crossfeedBlockSize,
                                    size_t producerBlockSize = 0,
                                    size_t defaultMultiplier = 3) {
    size_t safePeriod = std::max<size_t>(periodSize, 1);
    size_t defaultReady = safePeriod * defaultMultiplier;

    if (!crossfeedActive || crossfeedBlockSize == 0) {
        if (producerBlockSize > 0) {
            size_t clampedProducer =
                std::clamp(producerBlockSize, safePeriod, defaultReady);
            return clampedProducer;
        }
        return defaultReady;
    }

    size_t clampedCrossfeed = std::max(safePeriod, crossfeedBlockSize);
    if (clampedCrossfeed > defaultReady) {
        return defaultReady;
    }
    return clampedCrossfeed;
}

struct CapacityDecision {
    size_t dropFromExisting = 0;
    size_t newDataOffset = 0;
    size_t framesToStore = 0;
};

/**
 * @brief Determine how many frames must be dropped when applying a hard
 *        capacity limit to the playback buffer.
 *
 * @param currentFrames Currently queued frames (per channel).
 * @param producedFrames Frames produced by the upsampler for this block.
 * @param maxFrames Maximum number of frames allowed in the buffer.
 *
 * @return Decision describing how many existing frames to drop, how many new
 *         frames need to be skipped, and how many frames should be stored.
 */
inline CapacityDecision planCapacityEnforcement(size_t currentFrames, size_t producedFrames,
                                                size_t maxFrames) {
    CapacityDecision decision{};

    if (maxFrames == 0) {
        decision.dropFromExisting = currentFrames;
        decision.newDataOffset = producedFrames;
        decision.framesToStore = 0;
        return decision;
    }

    size_t framesToStore = producedFrames;
    size_t newDataOffset = 0;
    if (framesToStore > maxFrames) {
        newDataOffset = framesToStore - maxFrames;
        framesToStore = maxFrames;
    }

    size_t clampedCurrent = std::min(currentFrames, maxFrames);
    size_t combined = clampedCurrent + framesToStore;
    size_t dropExisting = (combined > maxFrames) ? (combined - maxFrames) : 0;

    decision.dropFromExisting = std::min(dropExisting, clampedCurrent);
    decision.newDataOffset = newDataOffset;
    decision.framesToStore = framesToStore;
    return decision;
}

}  // namespace PlaybackBuffer

#endif  // PLAYBACK_BUFFER_H


