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
