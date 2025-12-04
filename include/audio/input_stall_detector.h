#ifndef AUDIO_INPUT_STALL_DETECTOR_H
#define AUDIO_INPUT_STALL_DETECTOR_H

#include <chrono>
#include <cstdint>

namespace AudioInput {

// Stall detection threshold for streaming input gaps (ns) - 200ms
constexpr std::int64_t kStreamStallThresholdNs = 200000000;

inline bool shouldResetAfterStall(std::int64_t previousTimestampNs, std::int64_t currentTimestampNs,
                                  std::int64_t thresholdNs = kStreamStallThresholdNs) {
    if (previousTimestampNs <= 0 || currentTimestampNs <= previousTimestampNs) {
        return false;
    }
    return (currentTimestampNs - previousTimestampNs) > thresholdNs;
}

inline std::int64_t toNanoseconds(std::chrono::steady_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

}  // namespace AudioInput

#endif  // AUDIO_INPUT_STALL_DETECTOR_H
