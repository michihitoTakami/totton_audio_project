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
// Issue #231: Multi-rate support tests
// =============================================================================

// Test high-res rate detection (352.8kHz, 384kHz - 2x upsampling)
TEST_F(RateFamilyTest, Issue231_Detect352_8kHz) {
    EXPECT_EQ(detectRateFamily(352800), RateFamily::RATE_44K);
}

TEST_F(RateFamilyTest, Issue231_Detect384kHz) {
    EXPECT_EQ(detectRateFamily(384000), RateFamily::RATE_48K);
}

// Test getUpsampleRatioForInputRate for all 8 supported rates
TEST_F(RateFamilyTest, Issue231_UpsampleRatio_44kFamily) {
    EXPECT_EQ(getUpsampleRatioForInputRate(44100), 16);  // 44.1k → 705.6k
    EXPECT_EQ(getUpsampleRatioForInputRate(88200), 8);   // 88.2k → 705.6k
    EXPECT_EQ(getUpsampleRatioForInputRate(176400), 4);  // 176.4k → 705.6k
    EXPECT_EQ(getUpsampleRatioForInputRate(352800), 2);  // 352.8k → 705.6k
}

TEST_F(RateFamilyTest, Issue231_UpsampleRatio_48kFamily) {
    EXPECT_EQ(getUpsampleRatioForInputRate(48000), 16);  // 48k → 768k
    EXPECT_EQ(getUpsampleRatioForInputRate(96000), 8);   // 96k → 768k
    EXPECT_EQ(getUpsampleRatioForInputRate(192000), 4);  // 192k → 768k
    EXPECT_EQ(getUpsampleRatioForInputRate(384000), 2);  // 384k → 768k
}

TEST_F(RateFamilyTest, Issue231_UpsampleRatio_Unsupported) {
    EXPECT_EQ(getUpsampleRatioForInputRate(22050), 0);   // Unsupported
    EXPECT_EQ(getUpsampleRatioForInputRate(32000), 0);   // Unsupported
    EXPECT_EQ(getUpsampleRatioForInputRate(705600), 0);  // Output rate, not input
}

// Test findMultiRateConfigIndex
TEST_F(RateFamilyTest, Issue231_FindConfigIndex_Valid) {
    // 44.1k family indices 0-3
    EXPECT_EQ(findMultiRateConfigIndex(44100), 0);
    EXPECT_EQ(findMultiRateConfigIndex(88200), 1);
    EXPECT_EQ(findMultiRateConfigIndex(176400), 2);
    EXPECT_EQ(findMultiRateConfigIndex(352800), 3);
    // 48k family indices 4-7
    EXPECT_EQ(findMultiRateConfigIndex(48000), 4);
    EXPECT_EQ(findMultiRateConfigIndex(96000), 5);
    EXPECT_EQ(findMultiRateConfigIndex(192000), 6);
    EXPECT_EQ(findMultiRateConfigIndex(384000), 7);
}

TEST_F(RateFamilyTest, Issue231_FindConfigIndex_Invalid) {
    EXPECT_EQ(findMultiRateConfigIndex(22050), -1);
    EXPECT_EQ(findMultiRateConfigIndex(32000), -1);
    EXPECT_EQ(findMultiRateConfigIndex(705600), -1);
}

// Test MULTI_RATE_CONFIGS structure
TEST_F(RateFamilyTest, Issue231_MultiRateConfigs_OutputRate) {
    // All 44.1k family rates should output 705.6kHz
    EXPECT_EQ(MULTI_RATE_CONFIGS[0].outputRate, 705600);
    EXPECT_EQ(MULTI_RATE_CONFIGS[1].outputRate, 705600);
    EXPECT_EQ(MULTI_RATE_CONFIGS[2].outputRate, 705600);
    EXPECT_EQ(MULTI_RATE_CONFIGS[3].outputRate, 705600);
    // All 48k family rates should output 768kHz
    EXPECT_EQ(MULTI_RATE_CONFIGS[4].outputRate, 768000);
    EXPECT_EQ(MULTI_RATE_CONFIGS[5].outputRate, 768000);
    EXPECT_EQ(MULTI_RATE_CONFIGS[6].outputRate, 768000);
    EXPECT_EQ(MULTI_RATE_CONFIGS[7].outputRate, 768000);
}
