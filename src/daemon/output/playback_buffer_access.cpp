#include "daemon/output/playback_buffer_access.h"

#include "core/daemon_constants.h"

#include <algorithm>
#include <atomic>
#include <memory>

namespace daemon_output {

size_t maxOutputBufferFrames(const daemon_app::RuntimeState& state) {
    using namespace DaemonConstants;
    double seconds = static_cast<double>(MAX_OUTPUT_BUFFER_SECONDS);
    if (seconds <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }

    int outputRate = state.rates.currentOutputRate.load(std::memory_order_acquire);
    if (outputRate <= 0) {
        outputRate = DEFAULT_OUTPUT_SAMPLE_RATE;
    }

    double frames = seconds * static_cast<double>(outputRate);
    if (frames <= 0.0) {
        return DEFAULT_MAX_OUTPUT_BUFFER_FRAMES;
    }
    return static_cast<size_t>(frames);
}

PlaybackBufferManager& playbackBuffer(daemon_app::RuntimeState& state) {
    if (!state.playback.buffer) {
        state.playback.buffer = std::make_unique<PlaybackBufferManager>(
            [&state]() { return maxOutputBufferFrames(state); });
    }
    return *state.playback.buffer;
}

}  // namespace daemon_output
