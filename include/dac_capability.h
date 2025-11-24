#ifndef DAC_CAPABILITY_H
#define DAC_CAPABILITY_H

#include "error_codes.h"

#include <string>
#include <vector>

namespace DacCapability {

// DAC capability information
struct Capability {
    std::string deviceName;           // ALSA device name (e.g., "hw:0")
    int maxSampleRate;                // Maximum supported sample rate
    int minSampleRate;                // Minimum supported sample rate
    std::vector<int> supportedRates;  // List of supported rates (if discrete)
    int maxChannels;                  // Maximum channels
    bool isValid;                     // Whether scan was successful
    std::string errorMessage;         // Error message if scan failed

    // Error code for structured error handling (Issue #44)
    AudioEngine::ErrorCode errorCode = AudioEngine::ErrorCode::OK;
    int alsaErrno = 0;  // ALSA error number if applicable
};

// Scan DAC capabilities via ALSA
Capability scan(const std::string& device);

// Get list of available ALSA playback devices
std::vector<std::string> listPlaybackDevices();

// Check if a specific sample rate is supported by the DAC
bool isRateSupported(const Capability& cap, int sampleRate);

// Get the maximum supported rate that is <= the requested rate
int getBestSupportedRate(const Capability& cap, int requestedRate);

// Common target rates for Magic Box
constexpr int TARGET_RATE_44K_FAMILY = 705600;  // 44.1k × 16
constexpr int TARGET_RATE_48K_FAMILY = 768000;  // 48k × 16

}  // namespace DacCapability

#endif  // DAC_CAPABILITY_H
