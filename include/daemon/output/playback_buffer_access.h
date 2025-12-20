#pragma once

#include "daemon/app/runtime_state.h"
#include "daemon/output/playback_buffer_manager.h"

#include <cstddef>

namespace daemon_output {

size_t maxOutputBufferFrames(const daemon_app::RuntimeState& state);

// Lazily constructs the playback buffer manager.
PlaybackBufferManager& playbackBuffer(daemon_app::RuntimeState& state);

}  // namespace daemon_output
