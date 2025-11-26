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
 * @param defaultMultiplier   Multiplier applied to periodSize when crossfeed is inactive.
 *
 * @return Samples required before playback should resume.
 */
inline size_t computeReadyThreshold(size_t periodSize, bool crossfeedActive,
                                    size_t crossfeedBlockSize, size_t defaultMultiplier = 3) {
    size_t safePeriod = std::max<size_t>(periodSize, 1);
    size_t defaultReady = safePeriod * defaultMultiplier;

    if (!crossfeedActive || crossfeedBlockSize == 0) {
        return defaultReady;
    }

    size_t clampedCrossfeed = std::max(safePeriod, crossfeedBlockSize);
    if (clampedCrossfeed > defaultReady) {
        return defaultReady;
    }
    return clampedCrossfeed;
}

}  // namespace PlaybackBuffer

#endif  // PLAYBACK_BUFFER_H


