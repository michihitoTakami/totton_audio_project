#include "auto_negotiation.h"

#include <algorithm>

namespace AutoNegotiation {

// 44.1kHz family rates (base × 1, 2, 4, 8, 16)
static const int RATES_44K_FAMILY[] = {44100, 88200, 176400, 352800, 705600};
static const int RATES_44K_COUNT = 5;

// 48kHz family rates (base × 1, 2, 4, 8, 16)
static const int RATES_48K_FAMILY[] = {48000, 96000, 192000, 384000, 768000};
static const int RATES_48K_COUNT = 5;

ConvolutionEngine::RateFamily getRateFamily(int sampleRate) {
    // Check if it's a 44.1kHz family rate
    for (int i = 0; i < RATES_44K_COUNT; ++i) {
        if (sampleRate == RATES_44K_FAMILY[i]) {
            return ConvolutionEngine::RateFamily::RATE_44K;
        }
    }

    // Check if it's a 48kHz family rate
    for (int i = 0; i < RATES_48K_COUNT; ++i) {
        if (sampleRate == RATES_48K_FAMILY[i]) {
            return ConvolutionEngine::RateFamily::RATE_48K;
        }
    }

    // For non-standard rates, determine family by divisibility
    if (sampleRate % 44100 == 0 || sampleRate % 11025 == 0) {
        return ConvolutionEngine::RateFamily::RATE_44K;
    }

    // Default to 48kHz family
    return ConvolutionEngine::RateFamily::RATE_48K;
}

bool isSameFamily(int rate1, int rate2) {
    return getRateFamily(rate1) == getRateFamily(rate2);
}

int getTargetRateForFamily(ConvolutionEngine::RateFamily family) {
    if (family == ConvolutionEngine::RateFamily::RATE_44K) {
        return DacCapability::TARGET_RATE_44K_FAMILY;  // 705600
    }
    return DacCapability::TARGET_RATE_48K_FAMILY;  // 768000
}

int getBestRateForFamily(ConvolutionEngine::RateFamily family,
                         const DacCapability::Capability& dacCap) {
    if (!dacCap.isValid) {
        return 0;
    }

    const int* rates;
    int count;

    if (family == ConvolutionEngine::RateFamily::RATE_44K) {
        rates = RATES_44K_FAMILY;
        count = RATES_44K_COUNT;
    } else {
        rates = RATES_48K_FAMILY;
        count = RATES_48K_COUNT;
    }

    // Find the highest supported rate in descending order
    for (int i = count - 1; i >= 0; --i) {
        if (DacCapability::isRateSupported(dacCap, rates[i])) {
            return rates[i];
        }
    }

    // No rate supported
    return 0;
}

int calculateUpsampleRatio(int inputRate, int outputRate) {
    if (inputRate <= 0 || outputRate <= 0) {
        return 0;
    }
    if (outputRate % inputRate != 0) {
        return 0;  // Not an integer ratio
    }
    return outputRate / inputRate;
}

NegotiatedConfig negotiate(int inputRate, const DacCapability::Capability& dacCap,
                           int currentOutputRate) {
    NegotiatedConfig config;
    config.inputRate = inputRate;
    config.inputFamily = getRateFamily(inputRate);
    config.outputRate = 0;
    config.upsampleRatio = 0;
    config.isValid = false;
    config.requiresReconfiguration = false;
    config.errorMessage.clear();

    // Validate DAC capability
    if (!dacCap.isValid) {
        config.errorMessage = "Invalid DAC capability: " + dacCap.errorMessage;
        return config;
    }

    // Validate input rate
    if (inputRate <= 0) {
        config.errorMessage = "Invalid input rate: " + std::to_string(inputRate);
        return config;
    }

    // Get the best supported output rate for this input family
    int targetOutputRate = getBestRateForFamily(config.inputFamily, dacCap);

    if (targetOutputRate == 0) {
        config.errorMessage = "No supported output rate for input family";
        return config;
    }

    // Ensure output rate is >= input rate
    if (targetOutputRate < inputRate) {
        config.errorMessage = "Target output rate (" + std::to_string(targetOutputRate) +
                              ") is less than input rate (" + std::to_string(inputRate) + ")";
        return config;
    }

    // Calculate upsampling ratio
    int ratio = calculateUpsampleRatio(inputRate, targetOutputRate);
    if (ratio == 0) {
        config.errorMessage = "Cannot calculate integer upsampling ratio";
        return config;
    }

    config.outputRate = targetOutputRate;
    config.upsampleRatio = ratio;
    config.isValid = true;

    // Determine if reconfiguration is needed
    // Reconfiguration is required when:
    // 1. First time configuration (currentOutputRate == 0)
    // 2. Output rate changes (which implies family change since same-family
    //    always uses the same max output rate)
    if (currentOutputRate == 0) {
        // First time - always need to configure
        config.requiresReconfiguration = true;
    } else if (currentOutputRate != targetOutputRate) {
        // Output rate changed - this means family changed
        // (Same family always targets the same max rate)
        config.requiresReconfiguration = true;
    } else {
        // Same output rate - no reconfiguration needed
        // This is the fast path for same-family input rate changes
        config.requiresReconfiguration = false;
    }

    return config;
}

}  // namespace AutoNegotiation
