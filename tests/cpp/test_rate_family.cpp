/**
 * @file test_rate_family.cpp
 * @brief Unit tests for Rate Family detection logic (CPU-only, no GPU required)
 */

#include "convolution_engine.h"

#include <gtest/gtest.h>

using namespace ConvolutionEngine;

class RateFamilyTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup code if needed
    }
};

// Test 44.1kHz family detection
TEST_F(RateFamilyTest, Detect44kFamily) {
    EXPECT_EQ(detectRateFamily(44100), RateFamily::RATE_44K);
    EXPECT_EQ(detectRateFamily(88200), RateFamily::RATE_44K);
    EXPECT_EQ(detectRateFamily(176400), RateFamily::RATE_44K);
}

// Test 48kHz family detection
TEST_F(RateFamilyTest, Detect48kFamily) {
    EXPECT_EQ(detectRateFamily(48000), RateFamily::RATE_48K);
    EXPECT_EQ(detectRateFamily(96000), RateFamily::RATE_48K);
    EXPECT_EQ(detectRateFamily(192000), RateFamily::RATE_48K);
}

// Test unknown rate detection
TEST_F(RateFamilyTest, DetectUnknownRate) {
    EXPECT_EQ(detectRateFamily(12345), RateFamily::RATE_UNKNOWN);
    EXPECT_EQ(detectRateFamily(22050), RateFamily::RATE_UNKNOWN);  // Half of 44100
    // Note: 0 % 44100 == 0, so detectRateFamily(0) returns RATE_44K (by design)
}

// Test output sample rate calculation
TEST_F(RateFamilyTest, OutputSampleRate44k) {
    EXPECT_EQ(getOutputSampleRate(RateFamily::RATE_44K), 705600);
}

TEST_F(RateFamilyTest, OutputSampleRate48k) {
    EXPECT_EQ(getOutputSampleRate(RateFamily::RATE_48K), 768000);
}

TEST_F(RateFamilyTest, OutputSampleRateUnknown) {
    EXPECT_EQ(getOutputSampleRate(RateFamily::RATE_UNKNOWN), 0);
}

// Test base sample rate
TEST_F(RateFamilyTest, BaseSampleRate44k) {
    EXPECT_EQ(getBaseSampleRate(RateFamily::RATE_44K), 44100);
}

TEST_F(RateFamilyTest, BaseSampleRate48k) {
    EXPECT_EQ(getBaseSampleRate(RateFamily::RATE_48K), 48000);
}

TEST_F(RateFamilyTest, BaseSampleRateUnknown) {
    EXPECT_EQ(getBaseSampleRate(RateFamily::RATE_UNKNOWN), 0);
}

// =============================================================================
// Issue #238: 1x Bypass support tests
// =============================================================================

// Test bypass rate (705.6kHz, 768kHz) detection - should return same family
TEST_F(RateFamilyTest, Issue238_Detect705_6kHz) {
    EXPECT_EQ(detectRateFamily(705600), RateFamily::RATE_44K);
}

TEST_F(RateFamilyTest, Issue238_Detect768kHz) {
    EXPECT_EQ(detectRateFamily(768000), RateFamily::RATE_48K);
}

// Test getUpsampleRatioForInputRate returns 1 for bypass rates
TEST_F(RateFamilyTest, Issue238_UpsampleRatio_BypassRates) {
    EXPECT_EQ(getUpsampleRatioForInputRate(705600), 1);  // 44k family bypass
    EXPECT_EQ(getUpsampleRatioForInputRate(768000), 1);  // 48k family bypass
}

// Test findMultiRateConfigIndex for bypass rates
TEST_F(RateFamilyTest, Issue238_FindConfigIndex_BypassRates) {
    // 44k family bypass is at index 4 (after 16x,8x,4x,2x)
    EXPECT_EQ(findMultiRateConfigIndex(705600), 4);
    // 48k family bypass is at index 9 (after all 44k configs and 48k 16x,8x,4x,2x)
    EXPECT_EQ(findMultiRateConfigIndex(768000), 9);
}

// Test MULTI_RATE_CONFIGS structure for bypass entries
TEST_F(RateFamilyTest, Issue238_MultiRateConfigs_BypassEntries) {
    // 44k family bypass config (index 4)
    EXPECT_EQ(MULTI_RATE_CONFIGS[4].inputRate, 705600);
    EXPECT_EQ(MULTI_RATE_CONFIGS[4].outputRate, 705600);
    EXPECT_EQ(MULTI_RATE_CONFIGS[4].ratio, 1);
    EXPECT_EQ(MULTI_RATE_CONFIGS[4].family, RateFamily::RATE_44K);

    // 48k family bypass config (index 9)
    EXPECT_EQ(MULTI_RATE_CONFIGS[9].inputRate, 768000);
    EXPECT_EQ(MULTI_RATE_CONFIGS[9].outputRate, 768000);
    EXPECT_EQ(MULTI_RATE_CONFIGS[9].ratio, 1);
    EXPECT_EQ(MULTI_RATE_CONFIGS[9].family, RateFamily::RATE_48K);
}

// Test total config count is now 10
TEST_F(RateFamilyTest, Issue238_MultiRateConfigCount) {
    EXPECT_EQ(MULTI_RATE_CONFIG_COUNT, 10);
}
