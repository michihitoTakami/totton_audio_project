#pragma once

#include "core/config_loader.h"
#include "daemon/app/runtime_state.h"

namespace daemon_audio {

struct CrossfeedInitResult {
    bool initialized = false;
    bool skipped = false;
};

enum class CrossfeedSwitchStatus { NotInitialized, Switched, Failed };

CrossfeedInitResult initializeCrossfeed(daemon_app::RuntimeState& state,
                                        bool partitionedConvolutionEnabled);

void resetCrossfeedStreamStateLocked(daemon_app::CrossfeedState& state);

void clearCrossfeedRuntimeBuffers(daemon_app::CrossfeedState& state);

CrossfeedSwitchStatus switchCrossfeedRateFamilyLocked(daemon_app::CrossfeedState& state,
                                                      const AppConfig& config,
                                                      ConvolutionEngine::RateFamily targetFamily);

void shutdownCrossfeed(daemon_app::CrossfeedState& state);

}  // namespace daemon_audio
