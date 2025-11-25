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

// ============================================================================
// Unsupported Input Rate Tests (Issue #134)
// ============================================================================

TEST_F(AutoNegotiationTest, Negotiate_UnsupportedRate_11025Hz) {
    auto dac = createFullCapabilityDac();

    // 11025Hz would require 64x upsampling to reach 705600Hz
    // Only {2, 4, 8, 16} are valid ratios
    auto config = negotiate(11025, dac);
    EXPECT_FALSE(config.isValid);
    EXPECT_FALSE(config.errorMessage.empty());
    EXPECT_NE(config.errorMessage.find("Unsupported"), std::string::npos);
}

TEST_F(AutoNegotiationTest, Negotiate_UnsupportedRate_32000Hz) {
    auto dac = createFullCapabilityDac();

    // 32000Hz is in 48k family but would need 24x upsampling (768000/32000)
    auto config = negotiate(32000, dac);
    EXPECT_FALSE(config.isValid);
    EXPECT_FALSE(config.errorMessage.empty());
}

TEST_F(AutoNegotiationTest, Negotiate_UnsupportedRate_22050Hz) {
    auto dac = createFullCapabilityDac();

    // 22050Hz is in 44k family but would need 32x upsampling (705600/22050)
    auto config = negotiate(22050, dac);
    EXPECT_FALSE(config.isValid);
}

// ============================================================================
// Empty supportedRates Tests
// ============================================================================

TEST_F(AutoNegotiationTest, Negotiate_EmptySupportedRates_RejectsHighRates) {
    // DAC where supportedRates is empty (couldn't determine actual support)
    DacCapability::Capability dac;
    dac.deviceName = "test:empty";
    dac.minSampleRate = 44100;
    dac.maxSampleRate = 768000;
    dac.supportedRates = {};  // Empty - unknown actual support
    dac.maxChannels = 2;
    dac.isValid = true;

    // With empty supportedRates, isRateSupported returns false
    // So getBestRateForFamily should return 0
    auto config = negotiate(44100, dac);
    EXPECT_FALSE(config.isValid);
    EXPECT_NE(config.errorMessage.find("No supported output rate"), std::string::npos);
}

TEST_F(AutoNegotiationTest, IsRateSupported_EmptySupportedRates) {
    DacCapability::Capability dac;
    dac.deviceName = "test:empty";
    dac.minSampleRate = 44100;
    dac.maxSampleRate = 768000;
    dac.supportedRates = {};  // Empty
    dac.isValid = true;

    // Should return false for any rate when supportedRates is empty
    EXPECT_FALSE(DacCapability::isRateSupported(dac, 44100));
    EXPECT_FALSE(DacCapability::isRateSupported(dac, 705600));
    EXPECT_FALSE(DacCapability::isRateSupported(dac, 768000));
}

// =============================================================================
// Issue #231: Multi-rate support - all 8 input rate negotiations
// =============================================================================

TEST_F(AutoNegotiationTest, Issue231_Negotiate_AllEightRates_44kFamily) {
    auto dac = createFullCapabilityDac();

    // 44.1kHz -> 705.6kHz (16x)
    auto config1 = negotiate(44100, dac);
    EXPECT_TRUE(config1.isValid);
    EXPECT_EQ(config1.outputRate, 705600);
    EXPECT_EQ(config1.upsampleRatio, 16);

    // 88.2kHz -> 705.6kHz (8x)
    auto config2 = negotiate(88200, dac);
    EXPECT_TRUE(config2.isValid);
    EXPECT_EQ(config2.outputRate, 705600);
    EXPECT_EQ(config2.upsampleRatio, 8);

    // 176.4kHz -> 705.6kHz (4x)
    auto config3 = negotiate(176400, dac);
    EXPECT_TRUE(config3.isValid);
    EXPECT_EQ(config3.outputRate, 705600);
    EXPECT_EQ(config3.upsampleRatio, 4);

    // 352.8kHz -> 705.6kHz (2x)
    auto config4 = negotiate(352800, dac);
    EXPECT_TRUE(config4.isValid);
    EXPECT_EQ(config4.outputRate, 705600);
    EXPECT_EQ(config4.upsampleRatio, 2);
}

TEST_F(AutoNegotiationTest, Issue231_Negotiate_AllEightRates_48kFamily) {
    auto dac = createFullCapabilityDac();

    // 48kHz -> 768kHz (16x)
    auto config1 = negotiate(48000, dac);
    EXPECT_TRUE(config1.isValid);
    EXPECT_EQ(config1.outputRate, 768000);
    EXPECT_EQ(config1.upsampleRatio, 16);

    // 96kHz -> 768kHz (8x)
    auto config2 = negotiate(96000, dac);
    EXPECT_TRUE(config2.isValid);
    EXPECT_EQ(config2.outputRate, 768000);
    EXPECT_EQ(config2.upsampleRatio, 8);

    // 192kHz -> 768kHz (4x)
    auto config3 = negotiate(192000, dac);
    EXPECT_TRUE(config3.isValid);
    EXPECT_EQ(config3.outputRate, 768000);
    EXPECT_EQ(config3.upsampleRatio, 4);

    // 384kHz -> 768kHz (2x)
    auto config4 = negotiate(384000, dac);
    EXPECT_TRUE(config4.isValid);
    EXPECT_EQ(config4.outputRate, 768000);
    EXPECT_EQ(config4.upsampleRatio, 2);
}

TEST_F(AutoNegotiationTest, Issue231_SameFamilyDifferentRate_NoReconfiguration) {
    auto dac = createFullCapabilityDac();

    // Start from 44.1kHz (16x), switch to 88.2kHz (8x) - same family
    auto config1 = negotiate(44100, dac);
    EXPECT_TRUE(config1.isValid);
    EXPECT_EQ(config1.inputFamily, RateFamily::RATE_44K);

    // 88.2kHz with current output at 705.6kHz (same family)
    auto config2 = negotiate(88200, dac, 705600);
    EXPECT_TRUE(config2.isValid);
    EXPECT_EQ(config2.inputFamily, RateFamily::RATE_44K);
    EXPECT_FALSE(config2.requiresReconfiguration);  // Same output rate

    // 176.4kHz with current output at 705.6kHz (same family)
    auto config3 = negotiate(176400, dac, 705600);
    EXPECT_TRUE(config3.isValid);
    EXPECT_EQ(config3.inputFamily, RateFamily::RATE_44K);
    EXPECT_FALSE(config3.requiresReconfiguration);
}

TEST_F(AutoNegotiationTest, Issue231_CrossFamilySwitch_RequiresReconfiguration) {
    auto dac = createFullCapabilityDac();

    // Start from 96kHz (48k family, output 768kHz)
    auto config1 = negotiate(96000, dac);
    EXPECT_TRUE(config1.isValid);
    EXPECT_EQ(config1.outputRate, 768000);

    // Switch to 88.2kHz (44k family) - requires reconfiguration
    auto config2 = negotiate(88200, dac, 768000);
    EXPECT_TRUE(config2.isValid);
    EXPECT_EQ(config2.outputRate, 705600);
    EXPECT_TRUE(config2.requiresReconfiguration);  // Different output rate
}

// ============================================================================
// Issue #238: 1x Bypass Mode Tests
// ============================================================================

TEST_F(AutoNegotiationTest, Issue238_Negotiate_BypassMode_705_6kHz) {
    auto dac = createFullCapabilityDac();

    // 705.6kHz input should result in 1x ratio (bypass)
    auto config = negotiate(705600, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 705600);
    EXPECT_EQ(config.upsampleRatio, 1);  // Bypass mode
}

TEST_F(AutoNegotiationTest, Issue238_Negotiate_BypassMode_768kHz) {
    auto dac = createFullCapabilityDac();

    // 768kHz input should result in 1x ratio (bypass)
    auto config = negotiate(768000, dac);
    EXPECT_TRUE(config.isValid);
    EXPECT_EQ(config.outputRate, 768000);
    EXPECT_EQ(config.upsampleRatio, 1);  // Bypass mode
}

TEST_F(AutoNegotiationTest, Issue238_CalculateUpsampleRatio_BypassRates) {
    // Bypass: output rate equals input rate
    EXPECT_EQ(calculateUpsampleRatio(705600, 705600), 1);
    EXPECT_EQ(calculateUpsampleRatio(768000, 768000), 1);
}

// ============================================================================
// Issue #218: Rate Negotiation Handshake Tests
// See: docs/architecture/rate-negotiation-handshake.md
// ============================================================================

/**
 * Test cases from Issue #218 design document (Section 7.1):
 * - Input rate detection -> candidate rate determination
 * - Expected output rates for various input rates
 * - Reconfiguration flag behavior
 */

TEST_F(AutoNegotiationTest, Issue218_InputRateDetection_44kFamily) {
    auto dac = createFullCapabilityDac();

    // Test all 44.1kHz family input rates
    struct TestCase {
        int inputRate;
        int expectedOutputRate;
        int expectedRatio;
    };
    std::vector<TestCase> testCases = {
        {44100, 705600, 16},
        {88200, 705600, 8},
        {176400, 705600, 4},
        {352800, 705600, 2},
    };

    for (const auto& tc : testCases) {
        auto config = negotiate(tc.inputRate, dac);
        EXPECT_TRUE(config.isValid) << "Input rate: " << tc.inputRate;
        EXPECT_EQ(config.outputRate, tc.expectedOutputRate) << "Input rate: " << tc.inputRate;
        EXPECT_EQ(config.upsampleRatio, tc.expectedRatio) << "Input rate: " << tc.inputRate;
        EXPECT_EQ(config.inputFamily, RateFamily::RATE_44K) << "Input rate: " << tc.inputRate;
    }
}

TEST_F(AutoNegotiationTest, Issue218_InputRateDetection_48kFamily) {
    auto dac = createFullCapabilityDac();

    // Test all 48kHz family input rates
    struct TestCase {
        int inputRate;
        int expectedOutputRate;
        int expectedRatio;
    };
    std::vector<TestCase> testCases = {
        {48000, 768000, 16},
        {96000, 768000, 8},
        {192000, 768000, 4},
        {384000, 768000, 2},
    };

    for (const auto& tc : testCases) {
        auto config = negotiate(tc.inputRate, dac);
        EXPECT_TRUE(config.isValid) << "Input rate: " << tc.inputRate;
        EXPECT_EQ(config.outputRate, tc.expectedOutputRate) << "Input rate: " << tc.inputRate;
        EXPECT_EQ(config.upsampleRatio, tc.expectedRatio) << "Input rate: " << tc.inputRate;
        EXPECT_EQ(config.inputFamily, RateFamily::RATE_48K) << "Input rate: " << tc.inputRate;
    }
}

TEST_F(AutoNegotiationTest, Issue218_FamilySwitch_44kTo48k) {
    auto dac = createFullCapabilityDac();

    // Start with 44.1kHz family
    auto config1 = negotiate(44100, dac, 0);
    EXPECT_TRUE(config1.isValid);
    EXPECT_EQ(config1.outputRate, 705600);
    EXPECT_TRUE(config1.requiresReconfiguration);

    // Switch to 48kHz family - MUST require reconfiguration
    auto config2 = negotiate(48000, dac, 705600);
    EXPECT_TRUE(config2.isValid);
    EXPECT_EQ(config2.outputRate, 768000);
    EXPECT_TRUE(config2.requiresReconfiguration)
        << "Cross-family switch (44k->48k) must require reconfiguration";
    EXPECT_EQ(config2.inputFamily, RateFamily::RATE_48K);
}

TEST_F(AutoNegotiationTest, Issue218_FamilySwitch_48kTo44k) {
    auto dac = createFullCapabilityDac();

    // Start with 48kHz family
    auto config1 = negotiate(96000, dac, 0);
    EXPECT_TRUE(config1.isValid);
    EXPECT_EQ(config1.outputRate, 768000);

    // Switch to 44.1kHz family - MUST require reconfiguration
    auto config2 = negotiate(88200, dac, 768000);
    EXPECT_TRUE(config2.isValid);
    EXPECT_EQ(config2.outputRate, 705600);
    EXPECT_TRUE(config2.requiresReconfiguration)
        << "Cross-family switch (48k->44k) must require reconfiguration";
    EXPECT_EQ(config2.inputFamily, RateFamily::RATE_44K);
}

TEST_F(AutoNegotiationTest, Issue218_SameFamily_NoReconfiguration) {
    auto dac = createFullCapabilityDac();

    // Start with 44.1kHz
    auto config1 = negotiate(44100, dac, 0);
    EXPECT_TRUE(config1.requiresReconfiguration);

    // Switch within same family (44100 -> 88200)
    auto config2 = negotiate(88200, dac, 705600);
    EXPECT_TRUE(config2.isValid);
    EXPECT_FALSE(config2.requiresReconfiguration)
        << "Same-family switch should NOT require reconfiguration";
    EXPECT_EQ(config2.outputRate, 705600);

    // Switch within same family again (88200 -> 176400)
    auto config3 = negotiate(176400, dac, 705600);
    EXPECT_TRUE(config3.isValid);
    EXPECT_FALSE(config3.requiresReconfiguration)
        << "Same-family switch should NOT require reconfiguration";
}

TEST_F(AutoNegotiationTest, Issue218_DacLimitation_384kHz) {
    // Create DAC limited to 384kHz
    DacCapability::Capability dac;
    dac.deviceName = "test:384k";
    dac.minSampleRate = 44100;
    dac.maxSampleRate = 384000;
    dac.supportedRates = {44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000};
    dac.maxChannels = 2;
    dac.isValid = true;

    // 44.1kHz input should negotiate to 352800 (8x)
    auto config44k = negotiate(44100, dac);
    EXPECT_TRUE(config44k.isValid);
    EXPECT_EQ(config44k.outputRate, 352800);
    EXPECT_EQ(config44k.upsampleRatio, 8);

    // 48kHz input should negotiate to 384000 (8x)
    auto config48k = negotiate(48000, dac, 352800);
    EXPECT_TRUE(config48k.isValid);
    EXPECT_EQ(config48k.outputRate, 384000);
    EXPECT_EQ(config48k.upsampleRatio, 8);
}

TEST_F(AutoNegotiationTest, Issue218_UnsupportedInputRate) {
    auto dac = createFullCapabilityDac();

    // Rates that are not standard multiples
    // 22050Hz - would need 32x upsampling (not supported)
    auto config1 = negotiate(22050, dac);
    EXPECT_FALSE(config1.isValid);

    // 11025Hz - would need 64x upsampling (not supported)
    auto config2 = negotiate(11025, dac);
    EXPECT_FALSE(config2.isValid);

    // 0Hz - invalid input
    auto config3 = negotiate(0, dac);
    EXPECT_FALSE(config3.isValid);
}
