#pragma once

#include "core/config_loader.h"
#include "daemon/app/runtime_state.h"

#include <alsa/asoundlib.h>
#include <string>

namespace daemon_input {

snd_pcm_format_t parseI2sFormat(const std::string& formatStr);
bool validateI2sConfig(const AppConfig& cfg);

// Dedicated I2S capture loop.
//
// - Uses non-blocking ALSA handle so shutdown does not hang.
// - Performs XRUN recovery.
// - When ALSA negotiates a different hardware rate, schedules a follow-up by setting
//   state.rates.pendingRateChange.
void i2sCaptureThread(daemon_app::RuntimeState& state, const std::string& device,
                      snd_pcm_format_t format, unsigned int requestedRate, unsigned int channels,
                      snd_pcm_uframes_t periodFrames);

}  // namespace daemon_input
