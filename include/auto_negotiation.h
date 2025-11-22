#ifndef AUTO_NEGOTIATION_H
#define AUTO_NEGOTIATION_H

#include "convolution_engine.h"
#include "dac_capability.h"

#include <string>

namespace AutoNegotiation {

// Result of auto-negotiation
struct NegotiatedConfig {
    int inputRate;                              // Input sample rate
    ConvolutionEngine::RateFamily inputFamily;  // RATE_44K or RATE_48K
    int outputRate;                             // Negotiated output rate
    int upsampleRatio;                          // Upsampling ratio (outputRate / inputRate)
    bool isValid;                               // Whether negotiation succeeded
    bool requiresReconfiguration;  // True if ALSA needs reconfiguration (family change)
    std::string errorMessage;      // Error message if failed
};

// Negotiate optimal output rate based on input and DAC capabilities
// Parameters:
//   - inputRate: Current input sample rate (e.g., 44100, 48000, 96000)
//   - dacCap: DAC capability information from DacCapability::scan()
//   - currentOutputRate: Currently configured output rate (0 if not configured)
// Returns:
//   - NegotiatedConfig with optimal settings
//
// Design Decision (Issue #41):
//   Cross-family switching (44.1k <-> 48k) sets requiresReconfiguration=true.
//   This causes ~1 second soft mute during ALSA reconfiguration.
//   Same-family switching is instant and glitch-free.
//   No libsoxr resampling is used to preserve ultimate audio quality.
NegotiatedConfig negotiate(int inputRate, const DacCapability::Capability& dacCap,
                           int currentOutputRate = 0);

// Get the target output rate for a given family
// Returns TARGET_RATE_44K_FAMILY (705600) or TARGET_RATE_48K_FAMILY (768000)
int getTargetRateForFamily(ConvolutionEngine::RateFamily family);

// Get the best supported rate for a family on a specific DAC
// Falls back to lower multiples if the ideal rate is not supported
int getBestRateForFamily(ConvolutionEngine::RateFamily family,
                         const DacCapability::Capability& dacCap);

// Calculate the upsampling ratio
int calculateUpsampleRatio(int inputRate, int outputRate);

// Check if two rates belong to the same family
bool isSameFamily(int rate1, int rate2);

// Determine the rate family for a given sample rate
ConvolutionEngine::RateFamily getRateFamily(int sampleRate);

}  // namespace AutoNegotiation

#endif  // AUTO_NEGOTIATION_H
