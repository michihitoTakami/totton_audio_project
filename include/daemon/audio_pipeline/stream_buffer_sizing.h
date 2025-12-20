#pragma once

#include "core/daemon_constants.h"
#include "daemon/app/runtime_state.h"

#include <algorithm>
#include <cstddef>

namespace audio_pipeline {

// Compute a safe capacity for streaming input buffers.
//
// Goal: avoid reallocations in RT path while allowing bursty upstream.
inline size_t computeStreamBufferCapacity(const daemon_app::RuntimeState& state,
                                          size_t streamValidInputPerBlock) {
    using namespace DaemonConstants;
    size_t frames = static_cast<size_t>(DEFAULT_BLOCK_SIZE);
    if (state.config.blockSize > 0) {
        frames = std::max(frames, static_cast<size_t>(state.config.blockSize));
    }
    if (state.config.periodSize > 0) {
        frames = std::max(frames, static_cast<size_t>(state.config.periodSize));
    }
    if (state.config.loopback.periodFrames > 0) {
        frames = std::max(frames, static_cast<size_t>(state.config.loopback.periodFrames));
    }
    frames = std::max(frames, streamValidInputPerBlock);
    // 2x safety margin for bursty upstream (no reallocation in RT path)
    return frames * 2;
}

}  // namespace audio_pipeline
