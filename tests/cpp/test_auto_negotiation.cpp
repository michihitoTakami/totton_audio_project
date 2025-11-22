/**
 * @file test_auto_negotiation.cpp
 * @brief Unit tests for Auto-Negotiation logic (CPU-only, no GPU required)
 *
 * Tests the automatic negotiation of sample rates between input source
 * and DAC capabilities. Covers:
 * - Rate family detection
 * - Output rate negotiation
 * - Cross-family switching detection (requiresReconfiguration flag)
 * - DAC capability validation
 */

#include "auto_negotiation.h"
#include "dac_capability.h"

#include <gtest/gtest.h>

using namespace AutoNegotiation;
using namespace ConvolutionEngine;

class AutoNegotiationTest : public ::testing::Test {
   protected:
    // Create a mock DAC capability that supports all rates
    DacCapability::Capability createFullCapabilityDac() {
        DacCapability::Capability cap;
        cap.deviceName = "test:full";
        cap.minSampleRate = 44100;
        cap.maxSampleRate = 768000;
        cap.supportedRates = {44100,  48000,  88200,  96000,  176400,
                              192000, 352800, 384000, 705600, 768000};
        cap.maxChannels = 2;
        cap.isValid = true;
        cap.errorMessage.clear();
        return cap;
    }

    // Create a mock DAC capability limited to 192kHz
    DacCapability::Capability createLimitedDac() {
        DacCapability::Capability cap;
        cap.deviceName = "test:limited";
        cap.minSampleRate = 44100;
        cap.maxSampleRate = 192000;
        cap.supportedRates = {44100, 48000, 88200, 96000, 176400, 192000};
        cap.maxChannels = 2;
        cap.isValid = true;
        cap.errorMessage.clear();
        return cap;
    }

    // Create an invalid DAC capability
    DacCapability::Capability createInvalidDac() {
        DacCapability::Capability cap;
        cap.deviceName = "test:invalid";
        cap.isValid = false;
        cap.errorMessage = "Device not found";
        return cap;
    }
};

// ============================================================================
// Rate Family Detection Tests
// ============================================================================

TEST_F(AutoNegotiationTest, DetectRateFamily_44kFamily) {
    EXPECT_EQ(getRateFamily(44100), RateFamily::RATE_44K);
    EXPECT_EQ(getRateFamily(88200), RateFamily::RATE_44K);
    EXPECT_EQ(getRateFamily(176400), RateFamily::RATE_44K);
    EXPECT_EQ(getRateFamily(352800), RateFamily::RATE_44K);
    EXPECT_EQ(getRateFamily(705600), RateFamily::RATE_44K);
}

TEST_F(AutoNegotiationTest, DetectRateFamily_48kFamily) {
    EXPECT_EQ(getRateFamily(48000), RateFamily::RATE_48K);
    EXPECT_EQ(getRateFamily(96000), RateFamily::RATE_48K);
    EXPECT_EQ(getRateFamily(192000), RateFamily::RATE_48K);
    EXPECT_EQ(getRateFamily(384000), RateFamily::RATE_48K);
    EXPECT_EQ(getRateFamily(768000), RateFamily::RATE_48K);
}

TEST_F(AutoNegotiationTest, IsSameFamily_SameFamily) {
    // 44.1kHz family members
    EXPECT_TRUE(isSameFamily(44100, 88200));
    EXPECT_TRUE(isSameFamily(44100, 176400));
    EXPECT_TRUE(isSameFamily(88200, 352800));

    // 48kHz family members
    EXPECT_TRUE(isSameFamily(48000, 96000));
    EXPECT_TRUE(isSameFamily(48000, 192000));
    EXPECT_TRUE(isSameFamily(96000, 384000));
}

TEST_F(AutoNegotiationTest, IsSameFamily_DifferentFamily) {
    EXPECT_FALSE(isSameFamily(44100, 48000));
    EXPECT_FALSE(isSameFamily(88200, 96000));
    EXPECT_FALSE(isSameFamily(176400, 192000));
    EXPECT_FALSE(isSameFamily(705600, 768000));
}

TEST_F(AutoNegotiationTest, DetectRateFamily_NonStandardRates) {
    // 22050 is 44100/2, should be detected as 44k family (11025 divisible)
    EXPECT_EQ(getRateFamily(22050), RateFamily::RATE_44K);

    // 32000 is not in either standard family, defaults to 48k
    EXPECT_EQ(getRateFamily(32000), RateFamily::RATE_48K);

    // DSD rates (multiples of 44100)
    EXPECT_EQ(getRateFamily(2822400), RateFamily::RATE_44K);  // DSD64

    // Non-standard rate defaults to 48k
    EXPECT_EQ(getRateFamily(12345), RateFamily::RATE_48K);
}

// ============================================================================
// Target Rate Tests
// ============================================================================

TEST_F(AutoNegotiationTest, GetTargetRateForFamily) {
    EXPECT_EQ(getTargetRateForFamily(RateFamily::RATE_44K), 705600);
    EXPECT_EQ(getTargetRateForFamily(RateFamily::RATE_48K), 768000);
}

TEST_F(AutoNegotiationTest, GetBestRateForFamily_FullDac) {
    auto dac = createFullCapabilityDac();
    EXPECT_EQ(getBestRateForFamily(RateFamily::RATE_44K, dac), 705600);
    EXPECT_EQ(getBestRateForFamily(RateFamily::RATE_48K, dac), 768000);
}

TEST_F(AutoNegotiationTest, GetBestRateForFamily_LimitedDac) {
    auto dac = createLimitedDac();
    // Limited to 192kHz, so should return 176400 for 44k family and 192000 for 48k
    EXPECT_EQ(getBestRateForFamily(RateFamily::RATE_44K, dac), 176400);
    EXPECT_EQ(getBestRateForFamily(RateFamily::RATE_48K, dac), 192000);
}

// ============================================================================
// Upsample Ratio Tests
// ============================================================================

TEST_F(AutoNegotiationTest, CalculateUpsampleRatio_Valid) {
    EXPECT_EQ(calculateUpsampleRatio(44100, 705600), 16);
    EXPECT_EQ(calculateUpsampleRatio(88200, 705600), 8);
    EXPECT_EQ(calculateUpsampleRatio(176400, 705600), 4);
    EXPECT_EQ(calculateUpsampleRatio(352800, 705600), 2);

    EXPECT_EQ(calculateUpsampleRatio(48000, 768000), 16);
    EXPECT_EQ(calculateUpsampleRatio(96000, 768000), 8);
    EXPECT_EQ(calculateUpsampleRatio(192000, 768000), 4);
    EXPECT_EQ(calculateUpsampleRatio(384000, 768000), 2);
}

TEST_F(AutoNegotiationTest, CalculateUpsampleRatio_Invalid) {
    EXPECT_EQ(calculateUpsampleRatio(0, 705600), 0);
    EXPECT_EQ(calculateUpsampleRatio(44100, 0), 0);
    EXPECT_EQ(calculateUpsampleRatio(44100, 100000), 0);  // Not integer ratio
}

// ============================================================================
// Negotiation Tests - Full DAC
// ============================================================================

TEST_F(AutoNegotiationTest, Negotiate_44kFamily_FullDac) {
    auto dac = createFullCapabilityDac();

    auto config = negotiate(44100, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.inputRate, 44100);
    EXPECT_EQ(config.inputFamily, RateFamily::RATE_44K);
    EXPECT_EQ(config.outputRate, 705600);
    EXPECT_EQ(config.upsampleRatio, 16);
    EXPECT_TRUE(config.requiresReconfiguration);  // First time config

    config = negotiate(88200, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 705600);
    EXPECT_EQ(config.upsampleRatio, 8);

    config = negotiate(176400, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 705600);
    EXPECT_EQ(config.upsampleRatio, 4);
}

TEST_F(AutoNegotiationTest, Negotiate_48kFamily_FullDac) {
    auto dac = createFullCapabilityDac();

    auto config = negotiate(48000, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.inputRate, 48000);
    EXPECT_EQ(config.inputFamily, RateFamily::RATE_48K);
    EXPECT_EQ(config.outputRate, 768000);
    EXPECT_EQ(config.upsampleRatio, 16);

    config = negotiate(96000, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 768000);
    EXPECT_EQ(config.upsampleRatio, 8);

    config = negotiate(192000, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 768000);
    EXPECT_EQ(config.upsampleRatio, 4);
}

// ============================================================================
// Reconfiguration Detection Tests (Critical for Issue #41)
// ============================================================================

TEST_F(AutoNegotiationTest, Negotiate_SameFamily_NoReconfiguration) {
    auto dac = createFullCapabilityDac();

    // Start with 44100Hz, output is 705600Hz
    auto config1 = negotiate(44100, dac, 0);
    EXPECT_TRUE(config1.requiresReconfiguration);  // First time

    // Switch to 88200Hz (same family), output stays 705600Hz
    auto config2 = negotiate(88200, dac, 705600);
    EXPECT_FALSE(config2.requiresReconfiguration);  // Same output rate!
    EXPECT_EQ(config2.outputRate, 705600);

    // Switch to 176400Hz (same family), output stays 705600Hz
    auto config3 = negotiate(176400, dac, 705600);
    EXPECT_FALSE(config3.requiresReconfiguration);
    EXPECT_EQ(config3.outputRate, 705600);
}

TEST_F(AutoNegotiationTest, Negotiate_DifferentFamily_RequiresReconfiguration) {
    auto dac = createFullCapabilityDac();

    // Start with 44100Hz, output is 705600Hz
    auto config1 = negotiate(44100, dac, 0);
    EXPECT_TRUE(config1.requiresReconfiguration);
    EXPECT_EQ(config1.outputRate, 705600);

    // Switch to 48000Hz (different family!), output changes to 768000Hz
    auto config2 = negotiate(48000, dac, 705600);
    EXPECT_TRUE(config2.requiresReconfiguration);  // MUST be true!
    EXPECT_EQ(config2.outputRate, 768000);
    EXPECT_EQ(config2.inputFamily, RateFamily::RATE_48K);

    // Switch back to 44100Hz (different family again!)
    auto config3 = negotiate(44100, dac, 768000);
    EXPECT_TRUE(config3.requiresReconfiguration);  // MUST be true!
    EXPECT_EQ(config3.outputRate, 705600);
}

TEST_F(AutoNegotiationTest, Negotiate_48kTo44k_RequiresReconfiguration) {
    auto dac = createFullCapabilityDac();

    // Start with 96000Hz (48k family), output is 768000Hz
    auto config1 = negotiate(96000, dac, 0);
    EXPECT_TRUE(config1.requiresReconfiguration);
    EXPECT_EQ(config1.outputRate, 768000);

    // Switch to 88200Hz (44k family!), output changes to 705600Hz
    auto config2 = negotiate(88200, dac, 768000);
    EXPECT_TRUE(config2.requiresReconfiguration);  // Family change!
    EXPECT_EQ(config2.outputRate, 705600);
    EXPECT_EQ(config2.inputFamily, RateFamily::RATE_44K);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(AutoNegotiationTest, Negotiate_InvalidDac) {
    auto dac = createInvalidDac();

    auto config = negotiate(44100, dac);
    EXPECT_FALSE(config.isValid);
    EXPECT_FALSE(config.errorMessage.empty());
}

TEST_F(AutoNegotiationTest, Negotiate_InvalidInputRate) {
    auto dac = createFullCapabilityDac();

    auto config = negotiate(0, dac);
    EXPECT_FALSE(config.isValid);

    config = negotiate(-1, dac);
    EXPECT_FALSE(config.isValid);
}

// ============================================================================
// Limited DAC Tests
// ============================================================================

TEST_F(AutoNegotiationTest, Negotiate_LimitedDac_44kFamily) {
    auto dac = createLimitedDac();  // Max 192kHz

    auto config = negotiate(44100, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 176400);  // Fallback to 176400
    EXPECT_EQ(config.upsampleRatio, 4);    // 44100 × 4 = 176400
}

TEST_F(AutoNegotiationTest, Negotiate_LimitedDac_48kFamily) {
    auto dac = createLimitedDac();  // Max 192kHz

    auto config = negotiate(48000, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 192000);  // Fallback to 192000
    EXPECT_EQ(config.upsampleRatio, 4);    // 48000 × 4 = 192000
}
