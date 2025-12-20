#pragma once

#include "daemon/app/runtime_state.h"

#include <functional>

namespace audio_pipeline {

void applySoftMuteForFilterSwitch(daemon_app::RuntimeState& state,
                                  std::function<bool()> filterSwitchFunc);

bool resetStreamingCachesForSwitch(daemon_app::RuntimeState& state);

bool reinitializeStreamingForLegacyMode(daemon_app::RuntimeState& state);

bool handleRateSwitch(daemon_app::RuntimeState& state, int newInputRate);

}  // namespace audio_pipeline
