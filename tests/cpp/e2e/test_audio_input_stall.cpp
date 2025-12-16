#include "audio/input_stall_detector.h"

#include <gtest/gtest.h>

using namespace AudioInput;

TEST(AudioInputStallDetector, ReturnsFalseWithoutPreviousTimestamp) {
    EXPECT_FALSE(shouldResetAfterStall(0, kStreamStallThresholdNs + 1));
}

TEST(AudioInputStallDetector, RejectsNonPositiveGap) {
    EXPECT_FALSE(shouldResetAfterStall(1000, 999));
    EXPECT_FALSE(shouldResetAfterStall(1000, 1000));
}

TEST(AudioInputStallDetector, HonoursThreshold) {
    EXPECT_FALSE(shouldResetAfterStall(100, 100 + kStreamStallThresholdNs));
    EXPECT_TRUE(shouldResetAfterStall(100, 100 + kStreamStallThresholdNs + 1));
}

TEST(AudioInputStallDetector, AcceptsCustomThreshold) {
    constexpr std::int64_t customThreshold = 50;
    EXPECT_FALSE(shouldResetAfterStall(0, customThreshold, customThreshold));
    EXPECT_FALSE(shouldResetAfterStall(10, 10 + customThreshold));
    EXPECT_TRUE(shouldResetAfterStall(10, 10 + customThreshold + 1, customThreshold));
}
