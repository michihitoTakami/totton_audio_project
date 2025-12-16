#include "audio/auto_negotiation.h"

#include "audio/pcm_format_set.h"

#include <algorithm>

namespace AutoNegotiation {

ConvolutionEngine::RateFamily getRateFamily(int sampleRate) {
    if (PcmFormatSet::is44kFamilyRate(static_cast<uint32_t>(sampleRate))) {
        return ConvolutionEngine::RateFamily::RATE_44K;
    }
    if (PcmFormatSet::is48kFamilyRate(static_cast<uint32_t>(sampleRate))) {
        return ConvolutionEngine::RateFamily::RATE_48K;
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

    const auto& rates = (family == ConvolutionEngine::RateFamily::RATE_44K)
                            ? PcmFormatSet::kRates44k
                            : PcmFormatSet::kRates48k;

    // Find the highest supported rate in descending order
    for (auto it = rates.rbegin(); it != rates.rend(); ++it) {
        if (DacCapability::isRateSupported(dacCap, static_cast<int>(*it))) {
            return static_cast<int>(*it);
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
    config.inputFamily = ConvolutionEngine::RateFamily::RATE_UNKNOWN;
    config.outputRate = 0;
    config.upsampleRatio = 0;
    config.isValid = false;
    config.requiresReconfiguration = false;
    config.errorMessage.clear();

    // Validate input rate first (before calling getRateFamily to avoid UB with 0)
    if (inputRate <= 0) {
        config.errorMessage = "Invalid input rate: " + std::to_string(inputRate);
        return config;
    }

    // Validate DAC capability
    if (!dacCap.isValid) {
        config.errorMessage = "Invalid DAC capability: " + dacCap.errorMessage;
        return config;
    }

    // Now safe to determine rate family
    config.inputFamily = getRateFamily(inputRate);

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

    // Validate that the ratio is supported by the GPU engine
    // Valid ratios: {1, 2, 4, 8, 16} (corresponding to MULTI_RATE_CONFIGS)
    // ratio 1 = bypass mode (input already at output rate, no upsampling needed)
    if (ratio != 1 && ratio != 2 && ratio != 4 && ratio != 8 && ratio != 16) {
        config.errorMessage = "Unsupported input rate: " + std::to_string(inputRate) +
                              " Hz (ratio " + std::to_string(ratio) + " not in {1, 2, 4, 8, 16})";
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
